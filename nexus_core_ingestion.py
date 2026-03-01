"""
The Nexus Core - Document Ingestion & Retrieval

Provides zero-dependency document ingestion and keyword retrieval for the
knowledge base. If sentence-transformers or LlamaIndex are available, it
upgrades automatically to vector-based search.

Usage:
    from nexus_core_ingestion import ingest_file, search_knowledge_base

    ingest_file("/path/to/file.txt")  # process and store
    chunks = search_knowledge_base("my query", top_k=5)
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from nexus_core_config import config as _config

# ---------------------------------------------------------------------------
# Optional: vector embeddings via sentence-transformers
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer as _ST
    import numpy as _np
    _EMBEDDING_MODEL = _ST("all-MiniLM-L6-v2")
    _VECTOR_SEARCH_AVAILABLE = True
except Exception:
    _EMBEDDING_MODEL = None
    _np = None
    _VECTOR_SEARCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".log", ".csv", ".json"}
# PDF / DOCX stubs — add parsers here when those deps are installed
_SUPPORTED_ALL_EXTENSIONS = _SUPPORTED_TEXT_EXTENSIONS | {".pdf", ".docx"}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text into overlapping chunks.

    Uses sentence-boundary splitting where possible so chunks don't cut
    mid-sentence. Falls back to word-boundary splitting.

    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Number of characters shared between adjacent chunks

    Returns:
        List of text chunks
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Tokenise into sentences using simple punctuation rules
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # If one sentence alone exceeds chunk_size, hard-split it
        if len(sentence) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            words = sentence.split()
            buf = ""
            for word in words:
                if len(buf) + len(word) + 1 > chunk_size:
                    if buf:
                        chunks.append(buf.strip())
                    buf = word
                else:
                    buf = (buf + " " + word).strip()
            if buf:
                current = buf
            continue

        if len(current) + len(sentence) + 1 > chunk_size:
            if current:
                chunks.append(current.strip())
            # Start next chunk with overlap from previous
            if chunks and overlap > 0:
                prev_words = chunks[-1].split()
                overlap_text = " ".join(prev_words[-max(1, overlap // 6):])
                current = overlap_text + " " + sentence
            else:
                current = sentence
        else:
            current = (current + " " + sentence).strip()

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

def read_file(file_path: Path) -> Optional[str]:
    """
    Read a file and return its text content.

    Supports .txt, .md, .rst, .log, .csv, .json natively.
    Returns None for unsupported or unreadable files.
    """
    ext = file_path.suffix.lower()

    try:
        if ext in _SUPPORTED_TEXT_EXTENSIONS:
            return file_path.read_text(encoding="utf-8", errors="replace")

        if ext == ".pdf":
            try:
                import pdfplumber
                text_parts = []
                with pdfplumber.open(str(file_path)) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                return "\n\n".join(text_parts) if text_parts else None
            except ImportError:
                pass
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(str(file_path))
                return "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except ImportError:
                return None

        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(file_path))
                return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return None

    except Exception as exc:
        print(f"[ingestion] Could not read {file_path}: {exc}")
        return None

    return None


# ---------------------------------------------------------------------------
# Chunk store  (persisted as nexus_data/knowledge_base/chunks.jsonl)
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    Simple append-only JSONL store for text chunks.

    Each line in chunks.jsonl is one JSON object:
    {
        "chunk_id":    "<sha256 prefix>",
        "source_file": "document.txt",
        "chunk_index": 0,
        "total_chunks": 5,
        "text":        "...",
        "word_count":  83,
        "created_at":  "2026-02-28T..."
    }

    Ingested files are tracked in ingested_files.json by content hash
    so re-uploading an unchanged file is a no-op.
    """

    def __init__(self, kb_dir: Path):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_file = self.kb_dir / "chunks.jsonl"
        self.registry_file = self.kb_dir / "ingested_files.json"
        self._registry: Dict[str, Any] = self._load_registry()

    # -- registry (which files have been ingested) ---------------------------

    def _load_registry(self) -> Dict[str, Any]:
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_registry(self):
        self.registry_file.write_text(
            json.dumps(self._registry, indent=2), encoding="utf-8"
        )

    def is_ingested(self, content_hash: str) -> bool:
        return content_hash in self._registry

    def mark_ingested(self, filename: str, content_hash: str, chunk_count: int):
        self._registry[content_hash] = {
            "filename": filename,
            "chunk_count": chunk_count,
            "ingested_at": datetime.now().isoformat(),
        }
        self._save_registry()

    def get_ingested_files(self) -> List[Dict[str, Any]]:
        return [
            {"content_hash": h, **v}
            for h, v in self._registry.items()
        ]

    # -- chunk storage -------------------------------------------------------

    def append_chunks(self, chunks: List[Dict[str, Any]]):
        """Append chunks to the JSONL file."""
        with self.chunks_file.open("a", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load_all_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunks from disk."""
        if not self.chunks_file.exists():
            return []
        chunks = []
        with self.chunks_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return chunks

    def chunk_count(self) -> int:
        if not self.chunks_file.exists():
            return 0
        return sum(1 for _ in self.chunks_file.open(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Scoring / retrieval
# ---------------------------------------------------------------------------

def _keyword_score(query: str, chunk_text: str) -> float:
    """
    Simple term-overlap score: fraction of unique query tokens present
    in the chunk, weighted by term frequency.
    """
    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens:
        return 0.0
    c_text_lower = chunk_text.lower()
    c_words = re.findall(r"\w+", c_text_lower)
    if not c_words:
        return 0.0

    hits = sum(1 for t in c_words if t in q_tokens)
    presence = sum(1 for t in q_tokens if t in c_text_lower)
    # Blend: token presence (coverage) + density (hits per word)
    coverage = presence / len(q_tokens)
    density = hits / len(c_words)
    return 0.7 * coverage + 0.3 * density


def _vector_score(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    """Vector cosine similarity scores (requires sentence-transformers)."""
    if not _VECTOR_SEARCH_AVAILABLE or not chunks:
        return [0.0] * len(chunks)
    texts = [c["text"] for c in chunks]
    q_emb = _EMBEDDING_MODEL.encode([query], normalize_embeddings=True)
    c_emb = _EMBEDDING_MODEL.encode(texts, normalize_embeddings=True)
    scores = (q_emb @ c_emb.T)[0].tolist()
    return scores


def search_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Score and rank chunks against a query.

    Uses vector search if sentence-transformers is available, otherwise
    falls back to keyword overlap scoring.
    """
    if not chunks:
        return []

    if _VECTOR_SEARCH_AVAILABLE:
        scores = _vector_score(query, chunks)
    else:
        scores = [_keyword_score(query, c["text"]) for c in chunks]

    ranked = sorted(
        zip(scores, chunks), key=lambda x: x[0], reverse=True
    )
    results = []
    seen_sources: Dict[str, int] = {}

    for score, chunk in ranked[:top_k * 3]:  # over-fetch for diversity
        if score <= 0.0:
            continue
        src = chunk["source_file"]
        seen_sources[src] = seen_sources.get(src, 0) + 1
        # Allow max 2 chunks per source to ensure diversity
        if seen_sources[src] > 2:
            continue
        results.append({**chunk, "score": round(score, 4)})
        if len(results) >= top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_file(
    file_path: str | Path,
    data_dir: Optional[str | Path] = None,
    chunk_size: int = 512,
    overlap: int = 64,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Ingest a file into the knowledge base.

    Args:
        file_path: Path to the file to ingest
        data_dir: Root data directory (defaults to config.nexus_data_path)
        chunk_size: Target chunk size in characters
        overlap: Overlap between adjacent chunks in characters
        force: Re-ingest even if file content hasn't changed

    Returns:
        Dict with ingestion stats:
        {
            "status": "ok" | "skipped" | "error",
            "filename": "...",
            "chunks_created": N,
            "message": "..."
        }
    """
    file_path = Path(file_path)
    if data_dir is None:
        data_dir = _config.nexus_data_path
    kb_dir = Path(data_dir) / "knowledge_base"

    store = ChunkStore(kb_dir)

    # --- Read the file -------------------------------------------------------
    if not file_path.exists():
        return {"status": "error", "filename": file_path.name, "chunks_created": 0,
                "message": "File not found"}

    if file_path.suffix.lower() not in _SUPPORTED_ALL_EXTENSIONS:
        return {"status": "skipped", "filename": file_path.name, "chunks_created": 0,
                "message": f"Unsupported extension: {file_path.suffix}"}

    text = read_file(file_path)
    if text is None or not text.strip():
        return {"status": "skipped", "filename": file_path.name, "chunks_created": 0,
                "message": "No extractable text"}

    # --- Deduplication -------------------------------------------------------
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    if not force and store.is_ingested(content_hash):
        reg = store._registry[content_hash]
        return {
            "status": "skipped",
            "filename": file_path.name,
            "chunks_created": 0,
            "message": f"Already ingested ({reg['chunk_count']} chunks, {reg['ingested_at'][:10]})",
        }

    # --- Chunk ---------------------------------------------------------------
    raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    now = datetime.now().isoformat()
    chunk_records = []

    for i, chunk_str in enumerate(raw_chunks):
        chunk_id = hashlib.sha256(
            f"{content_hash}:{i}".encode()
        ).hexdigest()[:12]
        chunk_records.append({
            "chunk_id": chunk_id,
            "source_file": file_path.name,
            "source_path": str(file_path),
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "text": chunk_str,
            "word_count": len(chunk_str.split()),
            "created_at": now,
        })

    # --- Persist -------------------------------------------------------------
    store.append_chunks(chunk_records)
    store.mark_ingested(file_path.name, content_hash, len(chunk_records))

    return {
        "status": "ok",
        "filename": file_path.name,
        "chunks_created": len(chunk_records),
        "message": f"Ingested {len(chunk_records)} chunks from {file_path.name}",
    }


def search_knowledge_base(
    query: str,
    data_dir: Optional[str | Path] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Search the knowledge base for chunks relevant to a query.

    Args:
        query: The search query
        data_dir: Root data directory (defaults to config.nexus_data_path)
        top_k: Number of results (defaults to config.search_top_k)

    Returns:
        List of chunk dicts ordered by relevance, each containing:
        {
            "chunk_id": "...",
            "source_file": "document.txt",
            "chunk_index": 0,
            "text": "...",
            "score": 0.82,
            ...
        }
    """
    if data_dir is None:
        data_dir = _config.nexus_data_path
    if top_k is None:
        top_k = _config.search_top_k

    kb_dir = Path(data_dir) / "knowledge_base"
    store = ChunkStore(kb_dir)
    chunks = store.load_all_chunks()

    if not chunks:
        return []

    return search_chunks(query, chunks, top_k=top_k)


def get_knowledge_base_stats(
    data_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Return stats about the current knowledge base."""
    if data_dir is None:
        data_dir = _config.nexus_data_path
    kb_dir = Path(data_dir) / "knowledge_base"
    store = ChunkStore(kb_dir)

    ingested = store.get_ingested_files()
    return {
        "total_chunks": store.chunk_count(),
        "ingested_files": len(ingested),
        "files": ingested,
        "vector_search_available": _VECTOR_SEARCH_AVAILABLE,
        "search_mode": "vector" if _VECTOR_SEARCH_AVAILABLE else "keyword",
    }

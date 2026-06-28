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

import csv
import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nexus_core_config import config as _config

# ---------------------------------------------------------------------------
# Optional: vector embeddings via sentence-transformers
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_DEVICE = "cpu"
_VECTOR_SEARCH_AVAILABLE = False
_EMBEDDING_MODEL = None
_np = None

try:
    import torch as _torch
    import numpy as _np
    from sentence_transformers import SentenceTransformer as _ST

    _DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    _EMBEDDING_MODEL = _ST(_EMBEDDING_MODEL_NAME, device=_DEVICE)
    _VECTOR_SEARCH_AVAILABLE = True
    _gpu_name = (
        _torch.cuda.get_device_name(0) if _DEVICE == "cuda" else "CPU"
    )
    print(f"[ingestion] Vector search enabled — model: {_EMBEDDING_MODEL_NAME} | device: {_DEVICE} ({_gpu_name})")
except Exception as _e:
    print(f"[ingestion] Vector search unavailable, using keyword fallback ({_e})")


# ---------------------------------------------------------------------------
# Embedding cache — built once at ingestion time, reused on every search.
#
# Without this, search_knowledge_base() re-encodes every chunk on every call.
# For 44,500 chunks that is ~68 MB of float32 embeddings computed from scratch
# on each of the 5 swarm persona searches — enough to hang or crash the server.
#
# The fix: persist the embedding matrix to disk at ingest time and load it
# into memory on first search.  Subsequent searches only embed the query
# (1 vector) and multiply against the cached matrix with numpy.
# ---------------------------------------------------------------------------
_emb_lock: threading.Lock = threading.Lock()
_emb_matrix: Optional[Any] = None          # numpy float32 (n, 384) or None
_emb_chunk_ids: List[str] = []             # chunk_id[i] matches row i of matrix
_emb_kb_dir: Optional[Path] = None        # tracks which kb_dir is cached


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".log", ".csv", ".json"}
# PDF / DOCX / HTML / EPUB stubs — add parsers here when those deps are installed
_SUPPORTED_ALL_EXTENSIONS = _SUPPORTED_TEXT_EXTENSIONS | {".pdf", ".docx", ".html", ".htm", ".epub"}


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
    Supports .html/.htm via html.parser (stdlib — always available).
    Supports .pdf via pdfplumber or PyPDF2 (install either).
    Supports .docx via python-docx (install python-docx).
    Supports .epub via ebooklib + html.parser (install ebooklib).
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
                print("[ingestion] .pdf support requires pdfplumber or PyPDF2: pip install pdfplumber")
                return None

        if ext == ".docx":
            try:
                import docx
                doc = docx.Document(str(file_path))
                return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                print("[ingestion] .docx support requires python-docx: pip install python-docx")
                return None

        if ext in (".html", ".htm"):
            from html.parser import HTMLParser

            class _TextExtractor(HTMLParser):
                def __init__(self) -> None:
                    super().__init__()
                    self._parts: list[str] = []
                    self._skip = False

                def handle_starttag(self, tag: str, attrs: object) -> None:
                    if tag in ("script", "style", "head"):
                        self._skip = True

                def handle_endtag(self, tag: str) -> None:
                    if tag in ("script", "style", "head"):
                        self._skip = False

                def handle_data(self, data: str) -> None:
                    if not self._skip:
                        stripped = data.strip()
                        if stripped:
                            self._parts.append(stripped)

            raw_html = file_path.read_text(encoding="utf-8", errors="replace")
            parser = _TextExtractor()
            parser.feed(raw_html)
            return "\n".join(parser._parts) or None

        if ext == ".epub":
            try:
                import ebooklib
                from ebooklib import epub as _epub
                from html.parser import HTMLParser

                class _EpubTextExtractor(HTMLParser):
                    def __init__(self) -> None:
                        super().__init__()
                        self._parts: list[str] = []
                        self._skip = False

                    def handle_starttag(self, tag: str, attrs: object) -> None:
                        if tag in ("script", "style"):
                            self._skip = True

                    def handle_endtag(self, tag: str) -> None:
                        if tag in ("script", "style"):
                            self._skip = False

                    def handle_data(self, data: str) -> None:
                        if not self._skip:
                            stripped = data.strip()
                            if stripped:
                                self._parts.append(stripped)

                book = _epub.read_epub(str(file_path))
                parts: list[str] = []
                for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                    content = item.get_content().decode("utf-8", errors="replace")
                    p = _EpubTextExtractor()
                    p.feed(content)
                    parts.extend(p._parts)
                return "\n".join(parts) or None
            except ImportError:
                print("[ingestion] .epub support requires ebooklib: pip install ebooklib")
                return None

    except Exception as exc:
        print(f"[ingestion] Could not read {file_path}: {exc}")
        return None

    return None


def _read_csv_rows(file_path: Path) -> Optional[List[str]]:
    """Parse a CSV file and return one text string per data row.

    Each row becomes: "Column1: value1. Column2: value2. Column3: value3."
    Empty values and the header row are skipped.  This keeps all facts
    about one record together in a single chunk so the LLM can answer
    questions like "what is the common name of X" without needing to
    cross-reference separate chunks.

    Returns None if the file is not a valid tabular CSV (e.g. a flat
    log export with no headers), in which case callers should fall back
    to read_file() + chunk_text().
    """
    try:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(raw.splitlines())
        headers = reader.fieldnames
        if not headers or len(headers) < 2:
            return None   # Not a tabular CSV — use plain text fallback

        rows: List[str] = []
        for row in reader:
            # Build "Field: value. Field: value." string for the row
            parts = []
            for col in headers:
                val = (row.get(col) or "").strip()
                col_clean = col.strip()
                if val and col_clean:
                    parts.append(f"{col_clean}: {val}")
            if parts:
                rows.append(". ".join(parts) + ".")
        return rows if rows else None
    except Exception as exc:
        print(f"[ingestion] CSV row parse failed for {file_path.name}: {exc}")
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
        self.embeddings_file = self.kb_dir / "embeddings.npy"
        self.embedding_ids_file = self.kb_dir / "embedding_ids.txt"
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

    def load_chunks_by_ids(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load only the chunks matching a set of chunk_ids (single JSONL scan)."""
        if not self.chunks_file.exists() or not ids:
            return {}
        wanted = set(ids)
        result: Dict[str, Dict[str, Any]] = {}
        with self.chunks_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    cid = chunk.get("chunk_id", "")
                    if cid in wanted:
                        result[cid] = chunk
                        if len(result) == len(wanted):
                            break  # found all — stop early
                except json.JSONDecodeError:
                    pass
        return result

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


# Weights for combining vector and keyword scores in hybrid search.
# Vector dominates (semantic meaning) but keyword provides an exact-match
# signal that saves cases like subspecies names that are near-identical in
# embedding space to their parent species.
_HYBRID_VECTOR_WEIGHT = 0.7
_HYBRID_KEYWORD_WEIGHT = 0.3


def _keyword_candidate_scores(
    query: str, store: "ChunkStore", limit: int
) -> Dict[str, float]:
    """Scan the JSONL once and return {chunk_id: keyword_score} for the top
    `limit` chunks by keyword overlap.

    Used by the hybrid search path to find exact-term matches that vector
    search may miss (e.g. subspecies names with near-identical embeddings to
    the parent species).  Pure text scanning — no embeddings computed.
    """
    q_tokens = set(re.findall(r"\w+", query.lower()))
    if not q_tokens or not store.chunks_file.exists():
        return {}

    scored: List[tuple] = []   # (score, chunk_id)
    with store.chunks_file.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                chunk = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = chunk.get("text", "")
            score = _keyword_score(query, text)
            if score > 0.0:
                scored.append((score, chunk["chunk_id"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return {cid: s for s, cid in scored[:limit]}


def _get_embedding_cache(store: "ChunkStore") -> Tuple[Any, List[str]]:
    """Return (matrix, chunk_ids) from the in-memory cache, loading from disk
    or rebuilding from scratch if the cache is cold or stale.

    Thread-safe: uses _emb_lock so concurrent swarm searches don't race.
    Returns (None, []) if vector search is unavailable.
    """
    global _emb_matrix, _emb_chunk_ids, _emb_kb_dir

    if not _VECTOR_SEARCH_AVAILABLE or _np is None:
        return None, []

    with _emb_lock:
        # Cache hit: same kb_dir and non-empty
        if (
            _emb_matrix is not None
            and _emb_kb_dir == store.kb_dir
            and len(_emb_chunk_ids) > 0
        ):
            return _emb_matrix, _emb_chunk_ids

        # Try loading from persisted files
        if store.embeddings_file.exists() and store.embedding_ids_file.exists():
            try:
                # mmap_mode='r': the array lives on disk, OS pages it in/out as
                # needed.  Python heap never allocates the full matrix — peak RAM
                # per search is determined by the batch size in _batched_dot, not
                # the total corpus size.
                matrix = _np.load(str(store.embeddings_file), mmap_mode="r")
                ids = store.embedding_ids_file.read_text(encoding="utf-8").splitlines()
                ids = [i for i in ids if i]   # strip blank lines
                if len(ids) == matrix.shape[0]:
                    _emb_matrix = matrix
                    _emb_chunk_ids = ids
                    _emb_kb_dir = store.kb_dir
                    print(f"[ingestion] Loaded embedding cache: {len(ids)} chunks (mmap, not in RAM)")
                    return _emb_matrix, _emb_chunk_ids

                # IDs file and matrix are out of sync — repair without re-encoding.
                # Trust the matrix (it was written atomically by numpy.save); rebuild
                # the IDs file from chunks.jsonl to match the matrix row count.
                n_rows = matrix.shape[0]
                all_chunks = store.load_all_chunks()
                all_ids = [c["chunk_id"] for c in all_chunks]
                if len(all_ids) >= n_rows:
                    repaired_ids = all_ids[:n_rows]
                    store.embedding_ids_file.write_text(
                        "\n".join(repaired_ids), encoding="utf-8"
                    )
                    _emb_matrix = matrix
                    _emb_chunk_ids = repaired_ids
                    _emb_kb_dir = store.kb_dir
                    print(
                        f"[ingestion] Repaired embedding IDs file "
                        f"({len(repaired_ids)} / {len(all_ids)} chunks) — no re-encode needed"
                    )
                    return _emb_matrix, _emb_chunk_ids
                # Matrix has more rows than chunks on disk — stale; delete and fall
                # through to cold rebuild (handled below with the size guard).
                print(
                    f"[ingestion] Embedding cache stale "
                    f"(matrix={n_rows}, chunks={len(all_ids)}); will rebuild"
                )
                store.embeddings_file.unlink(missing_ok=True)
                store.embedding_ids_file.unlink(missing_ok=True)
            except Exception as exc:
                print(f"[ingestion] Embedding cache load failed ({exc}), rebuilding …")

        # Cold start or corrupted: build from all chunks.
        # Guard against OOM on large corpora — if the chunk count exceeds the
        # threshold, skip vector search for this session rather than attempting
        # to encode tens of thousands of chunks while the LLM is already loaded.
        _COLD_REBUILD_LIMIT = 50_000
        chunks = store.load_all_chunks()
        if not chunks:
            _emb_matrix = _np.zeros((0, 384), dtype="float32")
            _emb_chunk_ids = []
            _emb_kb_dir = store.kb_dir
            return _emb_matrix, _emb_chunk_ids

        if len(chunks) > _COLD_REBUILD_LIMIT:
            print(
                f"[ingestion] Corpus too large for in-request cold rebuild "
                f"({len(chunks)} chunks > {_COLD_REBUILD_LIMIT} limit); "
                f"using keyword search this session. "
                f"Call rebuild_embedding_cache(data_dir) once to index the full corpus."
            )
            _emb_kb_dir = store.kb_dir   # mark dir so we don't retry every search
            return None, []

        texts = [c["text"] for c in chunks]
        ids = [c["chunk_id"] for c in chunks]
        print(f"[ingestion] Building embedding cache for {len(chunks)} chunks …")
        try:
            matrix = _EMBEDDING_MODEL.encode(
                texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
            )
        except Exception as exc:  # includes CUDA OOM
            print(f"[ingestion] Embedding encode failed ({exc}); using keyword search this session.")
            _emb_kb_dir = store.kb_dir
            return None, []
        _emb_matrix = _np.array(matrix, dtype="float32")
        _emb_chunk_ids = ids
        _emb_kb_dir = store.kb_dir
        # Persist so the next server restart doesn't rebuild
        _np.save(str(store.embeddings_file), _emb_matrix)
        store.embedding_ids_file.write_text("\n".join(ids), encoding="utf-8")
        print(f"[ingestion] Embedding cache built and saved ({len(ids)} chunks)")
        return _emb_matrix, _emb_chunk_ids


def _append_embedding_cache(
    store: "ChunkStore", chunk_ids: List[str], texts: List[str]
) -> None:
    """Extend the in-memory and on-disk embedding cache with newly ingested chunks.

    Called by ingest_file() immediately after chunk_records are persisted so
    the cache stays current without a full rebuild.
    """
    global _emb_matrix, _emb_chunk_ids, _emb_kb_dir

    if not _VECTOR_SEARCH_AVAILABLE or _np is None or not chunk_ids:
        return

    with _emb_lock:
        new_embs = _EMBEDDING_MODEL.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        new_embs = _np.array(new_embs, dtype="float32")

        # If the in-memory cache is cold or points at a different kb_dir, load
        # the persisted matrix from disk before appending — otherwise we'd
        # overwrite the full matrix with just the new file's chunks.
        if _emb_matrix is None or _emb_kb_dir != store.kb_dir or _emb_matrix.shape[0] == 0:
            if store.embeddings_file.exists() and store.embedding_ids_file.exists():
                try:
                    disk_matrix = _np.load(str(store.embeddings_file))
                    disk_ids = [
                        i for i in store.embedding_ids_file.read_text(
                            encoding="utf-8"
                        ).splitlines() if i
                    ]
                    if len(disk_ids) == disk_matrix.shape[0]:
                        _emb_matrix = disk_matrix
                        _emb_chunk_ids = disk_ids
                        _emb_kb_dir = store.kb_dir
                except Exception:
                    pass  # fall through to replace-mode below

        # Note: the disk_matrix loaded above (if any) is a plain ndarray copy
        # since mmap of a file we're about to overwrite would be unsafe during
        # the vstack+save below.  After saving we re-open the new file as mmap.

        if _emb_matrix is None or _emb_kb_dir != store.kb_dir or _emb_matrix.shape[0] == 0:
            _emb_matrix = new_embs
            _emb_chunk_ids = list(chunk_ids)
        else:
            _emb_matrix = _np.vstack([_emb_matrix, new_embs])
            _emb_chunk_ids.extend(chunk_ids)
        _emb_kb_dir = store.kb_dir

        # Persist updated matrix and id list, then swap to mmap reference so
        # the full array is released from Python heap.
        _np.save(str(store.embeddings_file), _emb_matrix)
        with store.embedding_ids_file.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(chunk_ids) + "\n")
        try:
            _emb_matrix = _np.load(str(store.embeddings_file), mmap_mode="r")
        except Exception:
            pass  # keep the in-RAM copy if mmap re-open fails


def _batched_dot(
    matrix: Any,
    q_vec: Any,
    fetch_k: int,
    batch_size: int = 10_000,
) -> Tuple[Any, Any]:
    """Score every row in *matrix* against *q_vec* using sequential batches.

    This bounds peak RAM regardless of corpus size:
    - `matrix` should be a memory-mapped numpy array (``mmap_mode='r'``).
    - Only ``batch_size`` rows are paged into physical memory at a time
      (~15 MB for batch_size=10_000, dim=384).
    - The returned ``scores`` array is only ``n × 4`` bytes (400 KB for 100k
      chunks, 4 MB for 1M chunks) — not proportional to the matrix.

    Returns
    -------
    top_idx : ndarray, shape (fetch_k,)
        Row indices of the top-fetch_k highest-scoring rows, sorted descending.
    scores : ndarray, shape (n,)
        Full score vector (needed for hybrid re-ranking of keyword candidates).
    """
    n = matrix.shape[0]
    scores = _np.empty(n, dtype="float32")
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        scores[start:end] = matrix[start:end] @ q_vec   # pages in one slice

    actual_fetch = min(fetch_k, n)
    top_idx = _np.argpartition(scores, -actual_fetch)[-actual_fetch:]
    top_idx = top_idx[_np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores


def rebuild_embedding_cache(
    data_dir: Optional[str | Path] = None,
    batch_size: int = 64,
) -> int:
    """Re-index any chunks in the knowledge base that are not yet embedded.

    Safe to call at any time — it only encodes chunks whose IDs are absent
    from the current embedding matrix.  Already-indexed chunks are skipped.
    Run this once after a bulk ingest to bring the vector cache up to date.

    Returns the number of newly embedded chunks.
    """
    if not _VECTOR_SEARCH_AVAILABLE or _np is None:
        print("[ingestion] rebuild_embedding_cache: sentence-transformers not available.")
        return 0

    if data_dir is None:
        data_dir = _config.nexus_data_path
    kb_dir = Path(data_dir) / "knowledge_base"
    store = ChunkStore(kb_dir)

    global _emb_matrix, _emb_chunk_ids, _emb_kb_dir

    all_chunks = store.load_all_chunks()
    if not all_chunks:
        print("[ingestion] rebuild_embedding_cache: no chunks found.")
        return 0

    # Load the current persisted set of embedded IDs (not the in-memory cache)
    existing_ids: set[str] = set()
    if store.embedding_ids_file.exists():
        existing_ids = set(
            i for i in store.embedding_ids_file.read_text(encoding="utf-8").splitlines() if i
        )

    missing = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
    if not missing:
        print(f"[ingestion] rebuild_embedding_cache: all {len(all_chunks)} chunks already indexed.")
        return 0

    print(f"[ingestion] rebuild_embedding_cache: encoding {len(missing)} missing chunks "
          f"(out of {len(all_chunks)} total) …")

    # Encode in batches so progress is visible and GPU memory peaks are bounded
    total_encoded = 0
    REPORT_EVERY = 5_000
    batch_ids: list[str] = []
    batch_texts: list[str] = []

    def _flush() -> None:
        nonlocal total_encoded
        if not batch_ids:
            return
        _append_embedding_cache(store, chunk_ids=batch_ids, texts=batch_texts)
        total_encoded += len(batch_ids)
        batch_ids.clear()
        batch_texts.clear()

    for chunk in missing:
        batch_ids.append(chunk["chunk_id"])
        batch_texts.append(chunk["text"])
        if len(batch_ids) >= batch_size * 8:   # flush every ~512 chunks
            _flush()
            if total_encoded % REPORT_EVERY < batch_size * 8:
                print(f"[ingestion] rebuild_embedding_cache: {total_encoded}/{len(missing)} …")

    _flush()   # flush remainder

    # Invalidate the in-memory cache so the next search reloads from disk
    with _emb_lock:
        _emb_matrix = None
        _emb_chunk_ids = []
        _emb_kb_dir = None

    print(f"[ingestion] rebuild_embedding_cache: done — {total_encoded} new chunks indexed, "
          f"{len(all_chunks)} total in corpus.")
    return total_encoded


    """Vector cosine similarity scores (requires sentence-transformers).
    Kept for backward compatibility with search_chunks(); not used by the
    optimised search_knowledge_base() path.
    """
    if not _VECTOR_SEARCH_AVAILABLE or not chunks:
        return [0.0] * len(chunks)
    texts = [c["text"] for c in chunks]
    q_emb = _EMBEDDING_MODEL.encode([query], normalize_embeddings=True, batch_size=1)
    c_emb = _EMBEDDING_MODEL.encode(texts, normalize_embeddings=True, batch_size=32)
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

    # --- Read and chunk the file ---------------------------------------------
    # CSV files get row-based chunking: one self-contained chunk per data row
    # (Field: value. Field: value. …) so the LLM can answer record-level
    # questions without needing to cross-reference split chunks.
    # All other formats use the generic prose chunker.
    raw_chunks: List[str] = []
    text_for_hash: str = ""

    if file_path.suffix.lower() == ".csv":
        csv_rows = _read_csv_rows(file_path)
        if csv_rows:
            raw_chunks = csv_rows
            text_for_hash = "\n".join(csv_rows[:100])  # hash on first 100 rows
        # If _read_csv_rows returns None, fall through to plain text below

    if not raw_chunks:
        text = read_file(file_path)
        if text is None or not text.strip():
            return {"status": "skipped", "filename": file_path.name, "chunks_created": 0,
                    "message": "No extractable text"}
        text_for_hash = text
        raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not raw_chunks:
        return {"status": "skipped", "filename": file_path.name, "chunks_created": 0,
                "message": "No chunks produced"}

    # --- Deduplication -------------------------------------------------------
    content_hash = hashlib.sha256(text_for_hash.encode("utf-8")).hexdigest()[:16]
    if not force and store.is_ingested(content_hash):
        reg = store._registry[content_hash]
        return {
            "status": "skipped",
            "filename": file_path.name,
            "chunks_created": 0,
            "message": f"Already ingested ({reg['chunk_count']} chunks, {reg['ingested_at'][:10]})",
        }

    # --- Build chunk records -------------------------------------------------
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

    # Pre-compute and cache embeddings so search never has to re-encode chunks
    _append_embedding_cache(
        store,
        chunk_ids=[c["chunk_id"] for c in chunk_records],
        texts=[c["text"] for c in chunk_records],
    )

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

    Uses a pre-computed embedding cache so only the query needs to be encoded
    at search time — not all chunks.  For large KBs (tens of thousands of
    chunks) this is orders of magnitude faster than the naïve approach of
    re-encoding all chunks on every call.

    Falls back to keyword scoring when sentence-transformers is unavailable.
    """
    if data_dir is None:
        data_dir = _config.nexus_data_path
    if top_k is None:
        top_k = _config.search_top_k

    kb_dir = Path(data_dir) / "knowledge_base"
    store = ChunkStore(kb_dir)

    # --- Fast vector path (sentence-transformers available) ------------------
    if _VECTOR_SEARCH_AVAILABLE and _np is not None:
        matrix, cache_ids = _get_embedding_cache(store)

        if matrix is not None and len(cache_ids) > 0:
            # Encode only the query (1 vector, not all chunks)
            q_emb = _EMBEDDING_MODEL.encode(
                [query], normalize_embeddings=True, batch_size=1
            )
            q_vec = _np.array(q_emb[0], dtype="float32")

            # Batched dot product — processes matrix in 10k-row slices so
            # peak RAM is ~15 MB regardless of total corpus size.
            fetch_k = min(top_k * 3, len(cache_ids))
            top_idx, scores = _batched_dot(matrix, q_vec, fetch_k)

            vec_scores: Dict[str, float] = {cache_ids[i]: float(scores[i]) for i in top_idx}

            # Hybrid: merge with keyword candidates so exact term matches
            # (e.g. subspecies names) aren't buried by near-identical embeddings.
            kw_scores: Dict[str, float] = _keyword_candidate_scores(
                query, store, limit=top_k * 3
            )

            # Union of both candidate pools
            all_candidate_ids: List[str] = list(
                dict.fromkeys(list(vec_scores) + [cid for cid in kw_scores if cid not in vec_scores])
            )

            # Reverse index so keyword-only candidates can get their real
            # vector score (scores[i]) rather than defaulting to 0.0.
            # Without this a keyword hit ranked #160 in vector space gets
            # 0.7×0.0 + 0.3×0.57 = 0.17 — worse than a pure-vector top-15
            # entry with 0.7×0.46 + 0.3×0.0 = 0.32.
            cache_id_to_idx: Dict[str, int] = {cid: i for i, cid in enumerate(cache_ids)}

            # Compute hybrid score for every candidate
            hybrid_scores: Dict[str, float] = {}
            for cid in all_candidate_ids:
                if cid in vec_scores:
                    v = vec_scores[cid]
                else:
                    idx = cache_id_to_idx.get(cid, -1)
                    v = float(scores[idx]) if idx >= 0 else 0.0
                k = kw_scores.get(cid, 0.0)
                hybrid_scores[cid] = _HYBRID_VECTOR_WEIGHT * v + _HYBRID_KEYWORD_WEIGHT * k

            # Sort combined pool by hybrid score descending
            all_candidate_ids.sort(key=lambda c: hybrid_scores[c], reverse=True)

            # Load only the candidate chunks from disk (avoids full 44k+ load)
            id_to_chunk = store.load_chunks_by_ids(all_candidate_ids)

            results: List[Dict[str, Any]] = []
            seen_sources: Dict[str, int] = {}
            for cid in all_candidate_ids:
                score = hybrid_scores[cid]
                if score <= 0.0:
                    continue
                chunk = id_to_chunk.get(cid)
                if not chunk:
                    continue
                src = chunk["source_file"]
                seen_sources[src] = seen_sources.get(src, 0) + 1
                # Per-source cap prevents one long prose document from filling
                # all results.  But for a single-source KB (e.g. one database
                # CSV where every chunk is an independent record) the cap of 2
                # would mean the LLM only ever sees 2 records per query.
                # When there is only one distinct source seen so far, allow up
                # to top_k results from it; cap at 2 per source once multiple
                # sources are present.
                per_source_cap = top_k if len(seen_sources) <= 1 else 2
                if seen_sources[src] > per_source_cap:
                    continue
                results.append({**chunk, "score": round(score, 4)})
                if len(results) >= top_k:
                    break
            return results

    # --- Keyword fallback ----------------------------------------------------
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
        "embedding_model": _EMBEDDING_MODEL_NAME if _VECTOR_SEARCH_AVAILABLE else None,
        "embedding_device": _DEVICE if _VECTOR_SEARCH_AVAILABLE else None,
    }

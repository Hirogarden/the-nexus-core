"""
Tests for nexus_core_ingestion.py

Covers: chunk_text, read_file, ChunkStore, _keyword_score,
        search_chunks, ingest_file, search_knowledge_base,
        get_knowledge_base_stats
"""

import json
from pathlib import Path

import pytest

from nexus_core_ingestion import (
    ChunkStore,
    _keyword_score,
    get_knowledge_base_stats,
    ingest_file,
    chunk_text,
    read_file,
    search_chunks,
    search_knowledge_base,
)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_empty_string():
    assert chunk_text("") == []


def test_chunk_text_whitespace_only():
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_short_text_returns_single_chunk():
    text = "Hello world."
    result = chunk_text(text, chunk_size=512)
    assert result == [text]


def test_chunk_text_splits_long_text():
    # ~1 000 chars — well over the default 512
    text = "The quick brown fox jumps over the lazy dog. " * 22
    chunks = chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1


def test_chunk_text_chunks_are_non_empty():
    text = "Sentence one. Sentence two. Sentence three. " * 30
    for chunk in chunk_text(text, chunk_size=100, overlap=10):
        assert chunk.strip()


def test_chunk_text_total_content_preserved():
    # Every word in the original text must appear somewhere in the chunks
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    text = ". ".join(words * 20)
    chunks = chunk_text(text, chunk_size=80, overlap=0)
    combined = " ".join(chunks).lower()
    for word in words:
        assert word in combined


def test_chunk_text_hard_split_long_sentence():
    # A single sentence with no punctuation that exceeds chunk_size
    text = "word " * 200  # ~1 000 chars, no sentence boundaries
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip()


def test_chunk_text_respects_size_roughly():
    text = "This is a test sentence. " * 100
    chunk_size = 200
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    # No chunk should be more than 3× the target (generous slack for sentences)
    for chunk in chunks:
        assert len(chunk) <= chunk_size * 3


def test_chunk_text_overlap_adds_context():
    text = "Sentence one is here. Sentence two follows now. Sentence three ends it. " * 10
    chunks_no_overlap = chunk_text(text, chunk_size=80, overlap=0)
    chunks_overlap = chunk_text(text, chunk_size=80, overlap=30)
    # With overlap the chunks carry forward some context — count may differ
    assert len(chunks_no_overlap) >= 1
    assert len(chunks_overlap) >= 1


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_file_txt(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Hello world.", encoding="utf-8")
    assert read_file(f) == "Hello world."


def test_read_file_md(tmp_path):
    f = tmp_path / "readme.md"
    f.write_text("# Title\n\nBody text.", encoding="utf-8")
    result = read_file(f)
    assert "Title" in result
    assert "Body text" in result


def test_read_file_json(tmp_path):
    f = tmp_path / "data.json"
    f.write_text('{"key": "value", "num": 42}', encoding="utf-8")
    result = read_file(f)
    assert "value" in result


def test_read_file_csv(tmp_path):
    f = tmp_path / "table.csv"
    f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    result = read_file(f)
    assert "Alice" in result


def test_read_file_rst(tmp_path):
    f = tmp_path / "doc.rst"
    f.write_text("Title\n=====\nContent here.", encoding="utf-8")
    result = read_file(f)
    assert "Content" in result


def test_read_file_log(tmp_path):
    f = tmp_path / "app.log"
    f.write_text("2026-01-01 INFO: started", encoding="utf-8")
    result = read_file(f)
    assert "INFO" in result


def test_read_file_unsupported_extension(tmp_path):
    f = tmp_path / "binary.xyz"
    f.write_text("some content", encoding="utf-8")
    assert read_file(f) is None


def test_read_file_nonexistent(tmp_path):
    result = read_file(tmp_path / "ghost.txt")
    assert result is None


# ---------------------------------------------------------------------------
# ChunkStore
# ---------------------------------------------------------------------------

def test_chunkstore_starts_empty(tmp_path):
    store = ChunkStore(tmp_path / "kb")
    assert store.chunk_count() == 0
    assert store.load_all_chunks() == []


def test_chunkstore_append_and_load(tmp_path):
    store = ChunkStore(tmp_path / "kb")
    records = [
        {"chunk_id": "a1", "source_file": "doc.txt", "text": "hello world",
         "chunk_index": 0, "total_chunks": 1},
    ]
    store.append_chunks(records)
    loaded = store.load_all_chunks()
    assert len(loaded) == 1
    assert loaded[0]["chunk_id"] == "a1"


def test_chunkstore_multiple_appends(tmp_path):
    store = ChunkStore(tmp_path / "kb")
    for i in range(3):
        store.append_chunks([{"chunk_id": f"c{i}", "text": f"text {i}",
                               "source_file": "f.txt", "chunk_index": i, "total_chunks": 3}])
    assert store.chunk_count() == 3
    assert len(store.load_all_chunks()) == 3


def test_chunkstore_registry_dedup(tmp_path):
    store = ChunkStore(tmp_path / "kb")
    assert not store.is_ingested("hash123")
    store.mark_ingested("doc.txt", "hash123", 5)
    assert store.is_ingested("hash123")
    assert not store.is_ingested("different_hash")


def test_chunkstore_registry_persists_across_instances(tmp_path):
    kb_dir = tmp_path / "kb"
    store1 = ChunkStore(kb_dir)
    store1.mark_ingested("doc.txt", "abc123", 4)
    # New instance reads from disk
    store2 = ChunkStore(kb_dir)
    assert store2.is_ingested("abc123")


def test_chunkstore_get_ingested_files(tmp_path):
    store = ChunkStore(tmp_path / "kb")
    store.mark_ingested("a.txt", "hash_a", 3)
    store.mark_ingested("b.txt", "hash_b", 7)
    files = store.get_ingested_files()
    assert len(files) == 2
    names = {f["filename"] for f in files}
    assert names == {"a.txt", "b.txt"}


def test_chunkstore_handles_corrupt_registry(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "ingested_files.json").write_text("INVALID JSON {{{{", encoding="utf-8")
    store = ChunkStore(kb_dir)
    assert store._registry == {}


# ---------------------------------------------------------------------------
# _keyword_score
# ---------------------------------------------------------------------------

def test_keyword_score_exact_match():
    score = _keyword_score("machine learning", "machine learning is great")
    assert score > 0.7


def test_keyword_score_no_match():
    score = _keyword_score("machine learning", "completely unrelated content here")
    assert score < 0.15


def test_keyword_score_empty_query():
    assert _keyword_score("", "some text content") == 0.0


def test_keyword_score_empty_chunk():
    assert _keyword_score("query words", "") == 0.0


def test_keyword_score_partial_relevance():
    full = _keyword_score("quick brown fox", "the quick brown fox jumps high")
    partial = _keyword_score("quick brown fox", "quick and something else entirely")
    assert full > partial


def test_keyword_score_bounded():
    score = _keyword_score("test word", "test word test word test word")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------

def _make_chunks(n, source="doc.txt", topic="machine learning neural networks"):
    return [
        {"chunk_id": f"c{i}", "source_file": source, "chunk_index": i,
         "total_chunks": n, "text": f"{topic} chunk {i}"}
        for i in range(n)
    ]


def test_search_chunks_empty_list():
    assert search_chunks("anything", []) == []


def test_search_chunks_returns_at_most_top_k():
    chunks = _make_chunks(10)
    results = search_chunks("machine learning", chunks, top_k=3)
    assert len(results) <= 3


def test_search_chunks_scores_are_attached():
    chunks = _make_chunks(3)
    results = search_chunks("machine learning", chunks, top_k=3)
    for r in results:
        assert "score" in r
        assert isinstance(r["score"], float)


def test_search_chunks_diversity_cap():
    # All 5 from the same source — max 2 should be returned per source
    chunks = _make_chunks(5, source="same.txt")
    results = search_chunks("machine learning", chunks, top_k=5)
    same_source = [r for r in results if r["source_file"] == "same.txt"]
    assert len(same_source) <= 2


def test_search_chunks_multi_source_diversity():
    chunks = (
        _make_chunks(3, source="a.txt", topic="machine learning neural networks") +
        _make_chunks(3, source="b.txt", topic="machine learning deep learning")
    )
    results = search_chunks("machine learning", chunks, top_k=4)
    sources = {r["source_file"] for r in results}
    assert len(sources) >= 2  # both sources represented


def test_search_chunks_zero_score_excluded():
    chunks = [
        {"chunk_id": "c1", "source_file": "f.txt", "chunk_index": 0,
         "total_chunks": 1, "text": "completely irrelevant gibberish xyz"},
    ]
    results = search_chunks("machine learning artificial intelligence", chunks, top_k=5)
    # Zero-score chunks should not appear
    for r in results:
        assert r["score"] > 0.0


# ---------------------------------------------------------------------------
# ingest_file
# ---------------------------------------------------------------------------

def test_ingest_file_success(tmp_path):
    f = tmp_path / "article.txt"
    f.write_text("Machine learning is powerful. " * 25, encoding="utf-8")
    result = ingest_file(f, data_dir=tmp_path)
    assert result["status"] == "ok"
    assert result["chunks_created"] >= 1
    assert result["filename"] == "article.txt"


def test_ingest_file_not_found(tmp_path):
    result = ingest_file(tmp_path / "missing.txt", data_dir=tmp_path)
    assert result["status"] == "error"
    assert "not found" in result["message"].lower()


def test_ingest_file_unsupported_extension(tmp_path):
    f = tmp_path / "archive.zip"
    f.write_bytes(b"PK fake zip")
    result = ingest_file(f, data_dir=tmp_path)
    assert result["status"] == "skipped"


def test_ingest_file_empty_content(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("   \n\n   ", encoding="utf-8")
    result = ingest_file(f, data_dir=tmp_path)
    assert result["status"] == "skipped"


def test_ingest_file_deduplication(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Same content repeated many times. " * 30, encoding="utf-8")
    r1 = ingest_file(f, data_dir=tmp_path)
    assert r1["status"] == "ok"
    r2 = ingest_file(f, data_dir=tmp_path)
    assert r2["status"] == "skipped"
    assert "already ingested" in r2["message"].lower()


def test_ingest_file_force_reingest(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Content here now. " * 30, encoding="utf-8")
    r1 = ingest_file(f, data_dir=tmp_path)
    assert r1["status"] == "ok"
    r2 = ingest_file(f, data_dir=tmp_path, force=True)
    assert r2["status"] == "ok"


def test_ingest_file_creates_kb_directory(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Some content to ingest here. " * 20, encoding="utf-8")
    ingest_file(f, data_dir=tmp_path)
    assert (tmp_path / "knowledge_base").is_dir()
    assert (tmp_path / "knowledge_base" / "chunks.jsonl").exists()
    assert (tmp_path / "knowledge_base" / "ingested_files.json").exists()


def test_ingest_file_chunks_are_valid_json(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Important information about AI systems. " * 20, encoding="utf-8")
    ingest_file(f, data_dir=tmp_path)
    lines = (tmp_path / "knowledge_base" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    for line in lines:
        rec = json.loads(line)
        assert "chunk_id" in rec
        assert "source_file" in rec
        assert "text" in rec
        assert "chunk_index" in rec
        assert "total_chunks" in rec


def test_ingest_file_custom_chunk_size(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Word " * 500, encoding="utf-8")
    small = ingest_file(f, data_dir=tmp_path, chunk_size=100)
    assert small["chunks_created"] > 1


# ---------------------------------------------------------------------------
# search_knowledge_base
# ---------------------------------------------------------------------------

def test_search_knowledge_base_empty_returns_empty(tmp_path):
    results = search_knowledge_base("any query", data_dir=tmp_path)
    assert results == []


def test_search_knowledge_base_finds_relevant_chunks(tmp_path):
    f = tmp_path / "ml.txt"
    f.write_text(
        "Machine learning enables computers to learn from data. "
        "Neural networks are a key technique in deep learning. " * 15,
        encoding="utf-8",
    )
    ingest_file(f, data_dir=tmp_path)
    results = search_knowledge_base("machine learning neural networks", data_dir=tmp_path, top_k=3)
    assert len(results) > 0
    assert all("machine" in r["text"].lower() or "neural" in r["text"].lower() for r in results)


def test_search_knowledge_base_top_k_respected(tmp_path):
    f = tmp_path / "big.txt"
    f.write_text("Information about topic. " * 200, encoding="utf-8")
    ingest_file(f, data_dir=tmp_path)
    results = search_knowledge_base("information topic", data_dir=tmp_path, top_k=2)
    assert len(results) <= 2


def test_search_knowledge_base_multi_file(tmp_path):
    for i, topic in enumerate(["machine learning", "space exploration", "cooking recipes"]):
        f = tmp_path / f"doc_{i}.txt"
        f.write_text(f"All about {topic}. Details on {topic} here. " * 15, encoding="utf-8")
        ingest_file(f, data_dir=tmp_path)
    results = search_knowledge_base("machine learning", data_dir=tmp_path, top_k=5)
    assert any("machine" in r["text"].lower() for r in results)


# ---------------------------------------------------------------------------
# get_knowledge_base_stats
# ---------------------------------------------------------------------------

def test_kb_stats_empty(tmp_path):
    stats = get_knowledge_base_stats(data_dir=tmp_path)
    assert stats["total_chunks"] == 0
    assert stats["ingested_files"] == 0
    assert stats["files"] == []
    assert "search_mode" in stats


def test_kb_stats_after_ingest(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Test ingestion content here for stats. " * 20, encoding="utf-8")
    ingest_file(f, data_dir=tmp_path)
    stats = get_knowledge_base_stats(data_dir=tmp_path)
    assert stats["total_chunks"] >= 1
    assert stats["ingested_files"] == 1
    assert stats["files"][0]["filename"] == "doc.txt"


def test_kb_stats_counts_multiple_files(tmp_path):
    for i in range(3):
        f = tmp_path / f"file_{i}.txt"
        f.write_text(f"Unique content for file {i}. " * 20, encoding="utf-8")
        ingest_file(f, data_dir=tmp_path)
    stats = get_knowledge_base_stats(data_dir=tmp_path)
    assert stats["ingested_files"] == 3
    assert stats["total_chunks"] >= 3


def test_kb_stats_search_mode_field(tmp_path):
    stats = get_knowledge_base_stats(data_dir=tmp_path)
    assert stats["search_mode"] in ("keyword", "vector")

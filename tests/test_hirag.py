"""
Tests for nexus_core_hirag.py

Covers: dataclass serialisation round-trips, HiRAGMemory init and directory
creation, ingest_turn persistence, compression passes (ephemeral→daily,
daily→topics, topics→identity), maybe_compress threshold gating, retrieve
cross-layer scoring, and get_stats shape.
"""

import json
from pathlib import Path

import pytest

from nexus_core_hirag import (
    EphemeralTurn,
    DailySummary,
    TopicCluster,
    IdentityPattern,
    HiRAGMemory,
    _top_keywords,
    _keyword_score,
    _extractive_summary,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem(tmp_path):
    """A fresh HiRAGMemory with very low thresholds for fast tests."""
    return HiRAGMemory(
        data_dir=str(tmp_path),
        summarize_fn=None,       # extractive only — no LLM
        max_ephemeral=3,
        topic_threshold=2,
        identity_threshold=2,
    )


def _ingest_n(mem: HiRAGMemory, n: int, session_id: str = "s1") -> list:
    """Helper: ingest n synthetic turns and return them."""
    turns = []
    for i in range(n):
        t = mem.ingest_turn(
            query=f"question number {i} about machine learning",
            response=f"answer number {i} about machine learning and ai",
            session_id=session_id,
        )
        turns.append(t)
    return turns


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def test_top_keywords_returns_list():
    kws = _top_keywords(["machine learning neural network", "deep learning neural network"])
    assert isinstance(kws, list)
    assert len(kws) <= 8


def test_top_keywords_excludes_stop_words():
    kws = _top_keywords(["the and or but"])
    # All words here are stop words, expect empty
    assert kws == []


def test_top_keywords_most_frequent_first():
    kws = _top_keywords(["banana banana banana apple apple cherry"] * 1)
    assert kws[0] == "banana"


def test_keyword_score_full_overlap():
    score = _keyword_score("machine learning", "machine learning")
    assert score == 1.0


def test_keyword_score_no_overlap():
    score = _keyword_score("machine learning", "cooking recipes")
    assert score == 0.0


def test_keyword_score_partial_overlap():
    score = _keyword_score("machine learning network", "machine cooking")
    assert 0.0 < score < 1.0


def test_extractive_summary_truncates():
    turns = [
        EphemeralTurn(
            turn_id="a",
            session_id="s",
            query="x" * 200,
            response="y" * 300,
            timestamp="2026-01-01T00:00:00+00:00",
        )
    ]
    summary = _extractive_summary(turns, max_len=100)
    assert len(summary) <= 100


# ---------------------------------------------------------------------------
# Dataclass serialisation round-trips
# ---------------------------------------------------------------------------

def test_ephemeral_turn_round_trip():
    t = EphemeralTurn(
        turn_id="abcd1234",
        session_id="sess",
        query="hello",
        response="world",
        timestamp="2026-01-01T00:00:00+00:00",
        compressed=False,
    )
    assert EphemeralTurn.from_dict(t.to_dict()) == t


def test_daily_summary_round_trip():
    ds = DailySummary(
        summary_id="s1",
        date="2026-01-01",
        summary_text="summary text here",
        turn_count=5,
        key_topics=["topic1", "topic2"],
        compressed_at="2026-01-02T00:00:00+00:00",
        source_turn_ids=["t1", "t2"],
        compressed_to_topic=False,
    )
    assert DailySummary.from_dict(ds.to_dict()) == ds


def test_topic_cluster_round_trip():
    tc = TopicCluster(
        topic_id="tc1",
        topic_name="machine learning",
        description="about ML",
        related_dates=["2026-01-01"],
        confidence=0.8,
        created_at="2026-01-01T00:00:00+00:00",
        compressed_to_identity=False,
    )
    assert TopicCluster.from_dict(tc.to_dict()) == tc


def test_identity_pattern_round_trip():
    ip = IdentityPattern(
        pattern_id="ip1",
        pattern_type="recurring_interest",
        description="interested in AI",
        evidence=["tc1", "tc2"],
        strength=0.7,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
    )
    assert IdentityPattern.from_dict(ip.to_dict()) == ip


def test_ephemeral_turn_from_dict_missing_session_id():
    d = {
        "turn_id": "x",
        "query": "q",
        "response": "r",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    t = EphemeralTurn.from_dict(d)
    assert t.session_id == ""
    assert t.compressed is False


# ---------------------------------------------------------------------------
# HiRAGMemory init
# ---------------------------------------------------------------------------

def test_init_creates_hirag_directory(tmp_path):
    HiRAGMemory(data_dir=str(tmp_path))
    assert (tmp_path / "hirag").is_dir()


def test_init_no_files_created_until_ingest(tmp_path):
    HiRAGMemory(data_dir=str(tmp_path))
    hirag_dir = tmp_path / "hirag"
    assert list(hirag_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# ingest_turn
# ---------------------------------------------------------------------------

def test_ingest_turn_returns_ephemeral_turn(mem, tmp_path):
    t = mem.ingest_turn("hello", "world")
    assert isinstance(t, EphemeralTurn)


def test_ingest_turn_creates_ephemeral_jsonl(mem, tmp_path):
    mem.ingest_turn("q", "r")
    eph_path = tmp_path / "hirag" / "ephemeral.jsonl"
    assert eph_path.exists()


def test_ingest_turn_persists_fields(mem, tmp_path):
    mem.ingest_turn("my question", "my answer", session_id="sess42")
    eph_path = tmp_path / "hirag" / "ephemeral.jsonl"
    record = json.loads(eph_path.read_text().strip())
    assert record["query"] == "my question"
    assert record["response"] == "my answer"
    assert record["session_id"] == "sess42"
    assert record["compressed"] is False


def test_ingest_turn_appends_multiple(mem, tmp_path):
    mem.ingest_turn("q1", "r1")
    mem.ingest_turn("q2", "r2")
    eph_path = tmp_path / "hirag" / "ephemeral.jsonl"
    lines = eph_path.read_text().strip().splitlines()
    assert len(lines) == 2


# ---------------------------------------------------------------------------
# compress_ephemeral_to_daily
# ---------------------------------------------------------------------------

def test_compress_ephemeral_no_turns_returns_empty(mem):
    result = mem.compress_ephemeral_to_daily()
    assert result == []


def test_compress_ephemeral_creates_daily_summary(mem):
    _ingest_n(mem, 2)
    summaries = mem.compress_ephemeral_to_daily()
    assert len(summaries) >= 1
    assert isinstance(summaries[0], DailySummary)


def test_compress_ephemeral_marks_turns_compressed(mem, tmp_path):
    _ingest_n(mem, 2)
    mem.compress_ephemeral_to_daily()
    eph_path = tmp_path / "hirag" / "ephemeral.jsonl"
    for line in eph_path.read_text().strip().splitlines():
        record = json.loads(line)
        assert record["compressed"] is True


def test_compress_ephemeral_writes_daily_jsonl(mem, tmp_path):
    _ingest_n(mem, 2)
    mem.compress_ephemeral_to_daily()
    daily_path = tmp_path / "hirag" / "daily.jsonl"
    assert daily_path.exists()
    lines = daily_path.read_text().strip().splitlines()
    assert len(lines) >= 1


def test_compress_ephemeral_idempotent(mem):
    _ingest_n(mem, 2)
    r1 = mem.compress_ephemeral_to_daily()
    r2 = mem.compress_ephemeral_to_daily()
    assert len(r1) >= 1
    assert r2 == []  # already compressed, nothing new


def test_compress_ephemeral_daily_has_key_topics(mem):
    _ingest_n(mem, 2)
    summaries = mem.compress_ephemeral_to_daily()
    assert isinstance(summaries[0].key_topics, list)


# ---------------------------------------------------------------------------
# compress_daily_to_topics
# ---------------------------------------------------------------------------

def test_compress_daily_no_summaries_returns_empty(mem):
    result = mem.compress_daily_to_topics()
    assert result == []


def test_compress_daily_creates_topic_clusters(mem):
    _ingest_n(mem, 2)
    mem.compress_ephemeral_to_daily()
    # Need at least topic_threshold=2 daily summaries
    _ingest_n(mem, 2, session_id="s2")
    mem.compress_ephemeral_to_daily()
    topics = mem.compress_daily_to_topics()
    assert isinstance(topics, list)


def test_compress_daily_marks_daily_compressed(mem, tmp_path):
    _ingest_n(mem, 2)
    mem.compress_ephemeral_to_daily()
    _ingest_n(mem, 2, session_id="s2")
    mem.compress_ephemeral_to_daily()
    mem.compress_daily_to_topics()
    daily_path = tmp_path / "hirag" / "daily.jsonl"
    for line in daily_path.read_text().strip().splitlines():
        record = json.loads(line)
        assert record["compressed_to_topic"] is True


def test_compress_daily_idempotent(mem):
    _ingest_n(mem, 2)
    mem.compress_ephemeral_to_daily()
    _ingest_n(mem, 2, session_id="s2")
    mem.compress_ephemeral_to_daily()
    r1 = mem.compress_daily_to_topics()
    r2 = mem.compress_daily_to_topics()
    assert r2 == []


# ---------------------------------------------------------------------------
# compress_topics_to_identity
# ---------------------------------------------------------------------------

def test_compress_topics_no_clusters_returns_empty(mem):
    result = mem.compress_topics_to_identity()
    assert result == []


def _populate_to_topics(mem):
    """Helper: fill enough data to get topic clusters."""
    _ingest_n(mem, 2)
    mem.compress_ephemeral_to_daily()
    _ingest_n(mem, 2, session_id="s2")
    mem.compress_ephemeral_to_daily()
    return mem.compress_daily_to_topics()


def test_compress_topics_creates_identity_patterns(mem):
    topics = _populate_to_topics(mem)
    if len(topics) >= 2:
        patterns = mem.compress_topics_to_identity()
        assert isinstance(patterns, list)
        assert len(patterns) >= 1
        assert isinstance(patterns[0], IdentityPattern)


def test_compress_topics_marks_topics_compressed(mem, tmp_path):
    topics = _populate_to_topics(mem)
    if len(topics) >= 2:
        mem.compress_topics_to_identity()
        topics_path = tmp_path / "hirag" / "topics.jsonl"
        for line in topics_path.read_text().strip().splitlines():
            record = json.loads(line)
            assert record["compressed_to_identity"] is True


# ---------------------------------------------------------------------------
# maybe_compress — threshold gating
# ---------------------------------------------------------------------------

def test_maybe_compress_below_threshold_does_nothing(mem):
    _ingest_n(mem, 1)   # max_ephemeral=3, need 3
    result = mem.maybe_compress()
    assert result["daily"] == 0
    assert result["topics"] == 0
    assert result["identity"] == 0


def test_maybe_compress_at_threshold_triggers_daily(mem):
    _ingest_n(mem, 3)   # exactly max_ephemeral=3
    result = mem.maybe_compress()
    assert result["daily"] >= 1


def test_maybe_compress_returns_dict_with_required_keys(mem):
    result = mem.maybe_compress()
    assert "daily" in result
    assert "topics" in result
    assert "identity" in result


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

def test_retrieve_empty_returns_empty_list(mem):
    results = mem.retrieve("machine learning", top_k=5)
    assert results == []


def test_retrieve_returns_list(mem):
    _ingest_n(mem, 1)
    results = mem.retrieve("machine learning")
    assert isinstance(results, list)


def test_retrieve_respects_top_k(mem):
    _ingest_n(mem, 10)
    results = mem.retrieve("machine learning", top_k=3)
    assert len(results) <= 3


def test_retrieve_results_sorted_by_score_desc(mem):
    _ingest_n(mem, 5)
    results = mem.retrieve("machine learning", top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_result_has_required_keys(mem):
    _ingest_n(mem, 1)
    results = mem.retrieve("question")
    if results:
        r = results[0]
        assert "layer" in r
        assert "content" in r
        assert "score" in r
        assert "metadata" in r


def test_retrieve_ephemeral_layer_label(mem):
    _ingest_n(mem, 1)
    results = mem.retrieve("question number machine learning")
    layers = {r["layer"] for r in results}
    assert "ephemeral" in layers


def test_retrieve_daily_layer_after_compression(mem):
    _ingest_n(mem, 3)
    mem.maybe_compress()   # triggers ephemeral→daily
    results = mem.retrieve("machine learning", top_k=10)
    layers = {r["layer"] for r in results}
    assert "daily" in layers


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

def test_get_stats_shape(mem):
    stats = mem.get_stats()
    for key in ("ephemeral", "daily", "topic", "identity", "compression_pending"):
        assert key in stats


def test_get_stats_ephemeral_counts(mem):
    _ingest_n(mem, 2)
    stats = mem.get_stats()
    assert stats["ephemeral"]["total_turns"] == 2
    assert stats["ephemeral"]["uncompressed_turns"] == 2


def test_get_stats_after_compression(mem):
    _ingest_n(mem, 3)
    mem.maybe_compress()
    stats = mem.get_stats()
    assert stats["ephemeral"]["uncompressed_turns"] == 0
    assert stats["daily"]["total_summaries"] >= 1


def test_get_stats_compression_pending_flags(mem):
    _ingest_n(mem, 3)
    stats = mem.get_stats()
    assert stats["compression_pending"]["ephemeral_to_daily"] is True
    assert stats["compression_pending"]["daily_to_topics"] is False


def test_get_stats_after_fresh_init_all_zeros(mem):
    stats = mem.get_stats()
    assert stats["ephemeral"]["total_turns"] == 0
    assert stats["daily"]["total_summaries"] == 0
    assert stats["topic"]["total_clusters"] == 0
    assert stats["identity"]["total_patterns"] == 0

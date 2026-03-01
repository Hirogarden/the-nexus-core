"""
Tests for the anti-specialisation query blending in _generate_warmup_queries().

Verifies:
  - _GENERAL_WARMUP_QUERIES constant integrity (size, uniqueness, type)
  - With an empty KB the returned list still contains general queries
  - With a rich KB the returned list still contains general queries (~25 %)
  - Returned list length always equals target_count
  - No duplicate queries in the blended result
  - general_share scales correctly (max(5, target_count // 4))
"""

import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from brain_like_ai import BrainLikeAI, _GENERAL_WARMUP_QUERIES


# ---------------------------------------------------------------------------
# Helper: create a minimal BrainLikeAI-like object without running __init__
# ---------------------------------------------------------------------------

def _bare_instance(tmp_path: Path) -> BrainLikeAI:
    """
    Bypass BrainLikeAI.__init__ (which requires many live subsystems) and
    return an object with just the attributes _generate_warmup_queries needs.
    """
    obj = object.__new__(BrainLikeAI)
    obj.base_path = tmp_path
    return obj


# ---------------------------------------------------------------------------
# _GENERAL_WARMUP_QUERIES constant — integrity checks
# ---------------------------------------------------------------------------

def test_general_query_pool_minimum_size():
    """Pool must have enough entries to cover at least 25 % of the default target."""
    assert len(_GENERAL_WARMUP_QUERIES) >= 20


def test_general_query_pool_all_strings():
    for q in _GENERAL_WARMUP_QUERIES:
        assert isinstance(q, str) and q.strip(), (
            f"All general queries must be non-empty strings; got: {q!r}"
        )


def test_general_query_pool_no_duplicates():
    assert len(_GENERAL_WARMUP_QUERIES) == len(set(_GENERAL_WARMUP_QUERIES)), (
        "_GENERAL_WARMUP_QUERIES must not contain duplicate entries"
    )


# ---------------------------------------------------------------------------
# _generate_warmup_queries — empty KB
# ---------------------------------------------------------------------------

def _no_kb(q, data_dir, top_k):
    """Mock search_knowledge_base that always returns nothing."""
    return []


def _no_keywords(texts, n=25):
    """Mock _top_keywords that always returns an empty list."""
    return []


@patch("brain_like_ai._search_kb", _no_kb)
@patch("nexus_core_hirag._top_keywords", _no_keywords)
def test_empty_kb_returns_target_count(tmp_path):
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=20)
    assert len(result) == 20


@patch("brain_like_ai._search_kb", _no_kb)
@patch("nexus_core_hirag._top_keywords", _no_keywords)
def test_empty_kb_includes_general_queries(tmp_path):
    """Even with an empty KB the result must include queries from the general pool."""
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=20)
    general_set = set(_GENERAL_WARMUP_QUERIES)
    general_hits = [q for q in result if q in general_set]
    assert len(general_hits) >= 5, (
        f"Expected >=5 general queries in result, got {len(general_hits)}: {general_hits}"
    )


@patch("brain_like_ai._search_kb", _no_kb)
@patch("nexus_core_hirag._top_keywords", _no_keywords)
def test_empty_kb_no_duplicate_queries(tmp_path):
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=20)
    assert len(result) == len(set(result)), "Returned query list must not have duplicates"


# ---------------------------------------------------------------------------
# _generate_warmup_queries — rich KB
# ---------------------------------------------------------------------------

_KB_CHUNKS = [
    {
        "chunk_id": f"c{i}",
        "source_file": "doc.txt",
        "chunk_index": i,
        "total_chunks": 10,
        "score": 0.9,
        "text": f"Machine learning algorithm neural network training data epoch {i}",
    }
    for i in range(10)
]


def _rich_kb(q, data_dir, top_k):
    return _KB_CHUNKS[:top_k]


def _rich_keywords(texts, n=25):
    return [
        "machine learning", "neural network", "training data",
        "epoch", "algorithm", "gradient descent", "loss function",
        "backpropagation", "weights", "bias", "activation",
        "convolutional", "recurrent", "transformer", "attention",
        "embedding", "tokenization", "fine-tuning", "inference",
        "hyperparameter", "regularisation", "overfitting", "underfitting",
        "validation", "benchmark",
    ][:n]


@patch("brain_like_ai._search_kb", _rich_kb)
@patch("nexus_core_hirag._top_keywords", _rich_keywords)
def test_rich_kb_returns_target_count(tmp_path):
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=30)
    assert len(result) == 30


@patch("brain_like_ai._search_kb", _rich_kb)
@patch("nexus_core_hirag._top_keywords", _rich_keywords)
def test_rich_kb_still_includes_general_queries(tmp_path):
    """Even with a full KB the result must include at least 5 general queries."""
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=30)
    general_set = set(_GENERAL_WARMUP_QUERIES)
    general_hits = [q for q in result if q in general_set]
    assert len(general_hits) >= 5, (
        f"Expected >=5 general queries in rich-KB result, got {len(general_hits)}"
    )


@patch("brain_like_ai._search_kb", _rich_kb)
@patch("nexus_core_hirag._top_keywords", _rich_keywords)
def test_rich_kb_kb_queries_also_present(tmp_path):
    """KB-derived queries must still appear alongside general queries."""
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=30)
    general_set = set(_GENERAL_WARMUP_QUERIES)
    kb_hits = [q for q in result if q not in general_set]
    assert len(kb_hits) >= 5, (
        f"Expected >=5 KB-derived queries in result, got {len(kb_hits)}"
    )


@patch("brain_like_ai._search_kb", _rich_kb)
@patch("nexus_core_hirag._top_keywords", _rich_keywords)
def test_rich_kb_no_duplicate_queries(tmp_path):
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=30)
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# _generate_warmup_queries — general_share scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("target_count,expected_min_general", [
    (20,  5),   # max(5, 20//4) = 5
    (30,  7),   # max(5, 30//4) = 7
    (40, 10),   # max(5, 40//4) = 10
    (8,   5),   # max(5,  8//4) = 5 (floor keeps it at minimum)
])
@patch("brain_like_ai._search_kb", _no_kb)
@patch("nexus_core_hirag._top_keywords", _no_keywords)
def test_general_share_scales_with_target(tmp_path, target_count, expected_min_general):
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=target_count)
    general_set = set(_GENERAL_WARMUP_QUERIES)
    general_hits = [q for q in result if q in general_set]
    assert len(general_hits) >= expected_min_general, (
        f"target_count={target_count}: expected >={expected_min_general} general "
        f"queries, got {len(general_hits)}"
    )


# ---------------------------------------------------------------------------
# _generate_warmup_queries — edge: target_count smaller than general pool
# ---------------------------------------------------------------------------

@patch("brain_like_ai._search_kb", _no_kb)
@patch("nexus_core_hirag._top_keywords", _no_keywords)
def test_small_target_count_does_not_exceed(tmp_path):
    """When target_count is very small, the result must not be longer than target_count."""
    inst = _bare_instance(tmp_path)
    result = inst._generate_warmup_queries(target_count=6)
    assert len(result) == 6

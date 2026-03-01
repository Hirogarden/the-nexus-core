"""
Tests for layered_memory_system.py — ShortTermMemory, LongTermMemory,
and LayeredMemorySystem.

Run with:
    pytest "c:/Users/hirog/The Nexus Core/tests/test_memory.py" -v
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from layered_memory_system import (
    ShortTermMemory,
    LongTermMemory,
    LayeredMemorySystem,
    MemoryItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_memory_item(
    content: str = "test memory",
    memory_type: str = "episodic",
    importance: float = 0.5,
    tags=None,
    override_id: str = None,
    age_days: int = 0,
) -> MemoryItem:
    """Construct a MemoryItem directly for LTM tests."""
    import hashlib

    memory_id = override_id or hashlib.sha256(
        f"{content}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:16]

    created = (datetime.now() - timedelta(days=age_days)).isoformat()

    return MemoryItem(
        memory_id=memory_id,
        content=content,
        memory_type=memory_type,
        importance=importance,
        access_count=1,
        created_at=created,
        last_accessed=datetime.now().isoformat(),
        decay_rate=0.1,
        tags=tags or [],
        context={},
    )


# ===========================================================================
# 1 – ShortTermMemory
# ===========================================================================

def test_stm_store_returns_memory_item_with_correct_fields():
    stm = ShortTermMemory()
    item = stm.store("Hello world", memory_type="semantic", importance=0.6)

    assert isinstance(item, MemoryItem)
    assert item.content == "Hello world"
    assert item.memory_type == "semantic"
    assert pytest.approx(item.importance) == 0.6
    assert len(item.memory_id) == 16
    # access_count starts at 1 on first store
    assert item.access_count == 1


def test_stm_store_at_max_capacity_evicts_oldest():
    stm = ShortTermMemory(max_capacity=3)

    first = stm.store("first item")
    stm.store("second item")
    stm.store("third item")
    # Storing a fourth item should evict the first from the deque
    stm.store("fourth item")

    all_items = stm.get_all()
    assert len(all_items) == 3
    assert first not in all_items


def test_stm_retrieve_increments_access_count_and_boosts_importance():
    stm = ShortTermMemory()
    item = stm.store("Retrieve me", importance=0.5)
    initial_importance = item.importance  # 0.5
    initial_count = item.access_count     # 1

    retrieved = stm.retrieve(item.memory_id)

    assert retrieved is not None
    assert retrieved.access_count == initial_count + 1
    assert retrieved.importance == pytest.approx(initial_importance + 0.05)


def test_stm_retrieve_importance_never_exceeds_one():
    stm = ShortTermMemory()
    item = stm.store("High importance", importance=0.98)
    # Retrieve multiple times to push importance above 1.0
    for _ in range(5):
        stm.retrieve(item.memory_id)
    assert item.importance <= 1.0


def test_stm_retrieve_returns_none_for_missing_id():
    stm = ShortTermMemory()
    assert stm.retrieve("deadbeef00000000") is None


def test_stm_search_returns_matching_items():
    stm = ShortTermMemory()
    stm.store("Python is a programming language")
    stm.store("Java is also a programming language")
    stm.store("Cats are independent animals")

    results = stm.search("programming")
    contents = [item.content for item in results]
    assert any("Python" in c for c in contents)
    assert any("Java" in c for c in contents)
    assert not any("Cats" in c for c in contents)


def test_stm_get_all_returns_all_stored_items():
    stm = ShortTermMemory()
    stm.store("alpha")
    stm.store("beta")
    stm.store("gamma")

    all_items = stm.get_all()
    contents = {item.content for item in all_items}
    assert contents == {"alpha", "beta", "gamma"}


def test_stm_get_important_items_filters_by_threshold():
    stm = ShortTermMemory()
    stm.store("low", importance=0.3)
    stm.store("medium", importance=0.65)
    stm.store("high", importance=0.9)

    important = stm.get_important_items(threshold=0.7)
    contents = [item.content for item in important]
    assert "high" in contents
    assert "low" not in contents
    assert "medium" not in contents


def test_stm_clear_wipes_all():
    stm = ShortTermMemory()
    stm.store("first")
    stm.store("second")
    assert len(stm.get_all()) == 2

    stm.clear()
    assert len(stm.get_all()) == 0
    assert len(stm.item_map) == 0


# ===========================================================================
# 2 – LongTermMemory
# ===========================================================================

def test_ltm_store_creates_json_file(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    item = make_memory_item(content="Persistent fact")
    ltm.store(item)

    expected_file = tmp_path / "ltm" / f"{item.memory_id}.json"
    assert expected_file.is_file()
    data = json.loads(expected_file.read_text())
    assert data["memory_id"] == item.memory_id
    assert data["content"] == "Persistent fact"


def test_ltm_store_returns_true(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    item = make_memory_item()
    assert ltm.store(item) is True


def test_ltm_retrieve_increments_access_count(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    item = make_memory_item(content="Remember this")
    ltm.store(item)
    initial_count = item.access_count

    retrieved = ltm.retrieve(item.memory_id)
    assert retrieved is not None
    assert retrieved.access_count == initial_count + 1


def test_ltm_retrieve_returns_none_for_missing_id(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    assert ltm.retrieve("0000000000000000") is None


def test_ltm_search_by_query(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    ltm.store(make_memory_item(content="neural networks are fascinating"))
    ltm.store(make_memory_item(content="Python is a great language"))

    results = ltm.search(query="neural")
    assert len(results) >= 1
    assert any("neural" in item.content for item in results)


def test_ltm_search_by_tags(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    ltm.store(make_memory_item(content="tagged item", tags=["science", "data"]))
    ltm.store(make_memory_item(content="untagged item", tags=[]))

    results = ltm.search(tags=["science"])
    assert len(results) == 1
    assert results[0].content == "tagged item"


def test_ltm_search_by_memory_type(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    ltm.store(make_memory_item(content="episodic event", memory_type="episodic"))
    ltm.store(make_memory_item(content="how to ride a bike", memory_type="procedural"))

    results = ltm.search(memory_type="procedural")
    assert len(results) == 1
    assert results[0].content == "how to ride a bike"


def test_ltm_search_by_min_importance(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    ltm.store(make_memory_item(content="trivial", importance=0.2))
    ltm.store(make_memory_item(content="critical", importance=0.9))

    results = ltm.search(min_importance=0.5)
    contents = [item.content for item in results]
    assert "critical" in contents
    assert "trivial" not in contents


def test_ltm_prune_removes_low_importance_items(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    # importance 0.05 < default min_importance threshold 0.1
    item = make_memory_item(
        content="barely remembered",
        importance=0.05,
        override_id="aabbccdd00000001",
    )
    ltm.store(item)
    assert "aabbccdd00000001" in ltm.index

    removed = ltm.prune(min_importance=0.1)
    assert removed >= 1
    assert "aabbccdd00000001" not in ltm.index


def test_ltm_get_statistics_structure(tmp_path):
    ltm = LongTermMemory(storage_path=str(tmp_path / "ltm"))
    ltm.store(make_memory_item(content="one"))
    ltm.store(make_memory_item(content="two"))

    stats = ltm.get_statistics()
    assert "total_memories" in stats
    assert stats["total_memories"] == 2
    assert "avg_importance" in stats
    assert "avg_access_count" in stats
    assert "memory_types" in stats


# ===========================================================================
# 3 – LayeredMemorySystem
# ===========================================================================

def test_lms_store_high_importance_goes_to_ltm(tmp_path):
    lms = LayeredMemorySystem(ltm_storage_path=str(tmp_path / "ltm"))
    item = lms.store("Very important fact", importance=0.9)

    # Must appear in short-term
    assert any(i.memory_id == item.memory_id for i in lms.short_term.get_all())
    # Must also appear in long-term
    assert item.memory_id in lms.long_term.index


def test_lms_store_low_importance_stays_in_stm_only(tmp_path):
    lms = LayeredMemorySystem(ltm_storage_path=str(tmp_path / "ltm"))
    item = lms.store("Casual remark", importance=0.3)

    assert any(i.memory_id == item.memory_id for i in lms.short_term.get_all())
    assert item.memory_id not in lms.long_term.index


def test_lms_store_force_long_term(tmp_path):
    lms = LayeredMemorySystem(ltm_storage_path=str(tmp_path / "ltm"))
    item = lms.store("Forced to disk", importance=0.1, force_long_term=True)

    assert item.memory_id in lms.long_term.index


def test_lms_retrieve_searches_both_layers(tmp_path):
    lms = LayeredMemorySystem(ltm_storage_path=str(tmp_path / "ltm"))
    # Store one item in STM only
    stm_item = lms.store("short term only", importance=0.3)
    # Store one directly in LTM only (simulate a previously consolidated item)
    ltm_item = make_memory_item(content="long term only", importance=0.7)
    lms.long_term.store(ltm_item)

    results = lms.retrieve("only", search_both=True)
    result_ids = {i.memory_id for i in results}

    assert stm_item.memory_id in result_ids
    assert ltm_item.memory_id in result_ids


def test_lms_consolidate_memories_returns_stats_dict(tmp_path):
    lms = LayeredMemorySystem(ltm_storage_path=str(tmp_path / "ltm"))
    lms.store("alpha", importance=0.4)
    lms.store("beta", importance=0.6)

    result = lms.consolidate_memories()
    assert "consolidated" in result
    assert "stm_count" in result
    assert "ltm_count" in result
    assert "timestamp" in result
    assert isinstance(result["consolidated"], int)


def test_lms_get_memory_status_structure(tmp_path):
    lms = LayeredMemorySystem(
        stm_capacity=10,
        stm_retention_minutes=30,
        ltm_storage_path=str(tmp_path / "ltm"),
    )
    lms.store("hello", importance=0.5)

    status = lms.get_memory_status()
    assert "short_term" in status
    assert "long_term" in status
    assert "consolidation_rules" in status

    st = status["short_term"]
    assert "count" in st
    assert "capacity" in st
    assert "utilization" in st
    assert st["capacity"] == 10
    assert 0.0 <= st["utilization"] <= 1.0

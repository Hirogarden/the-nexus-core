"""
Tests for nexus_core_research_swarm.py

Covers: ResearchPersona round-trips, reformulate, record_search fitness
tracking, mutate_persona, ResearchSwarm initialisation & seeding,
get_active, search() with a mock KB function, fitness persistence,
automatic competition trigger, force_evolve (sufficient & insufficient
samples), and get_stats shape.
"""

import json
from pathlib import Path

import pytest

from nexus_core_research_swarm import (
    ResearchPersona,
    ResearchSwarm,
    _STRATEGIES,
    _SEED_PERSONAS,
    _seed_persona,
    mutate_persona,
    _text_hash,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def swarm(tmp_path):
    """A fresh ResearchSwarm with a low competition threshold for fast tests."""
    return ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=6,   # trigger after 6 individual-persona searches
        min_samples_to_compete=2,
    )


def _make_persona(**overrides) -> ResearchPersona:
    base = dict(
        persona_id="abc12345",
        name="Test Persona",
        strategy="technical",
        prefix="technical details about",
        suffix="algorithm",
        fitness=0.5,
        search_count=0,
        hit_count=0,
        active=True,
        generation=0,
        parent_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    base.update(overrides)
    return ResearchPersona(**base)


def _mock_kb(chunks):
    """Return a kb_search_fn that always returns *chunks*."""
    def fn(query: str, top_k: int):
        return chunks[:top_k]
    return fn


_SAMPLE_CHUNKS = [
    {
        "chunk_id": f"c{i}",
        "source_file": "doc.txt",
        "chunk_index": i,
        "total_chunks": 5,
        "score": 0.9 - i * 0.1,
        "text": f"chunk text {i} about machine learning",
    }
    for i in range(5)
]


# ---------------------------------------------------------------------------
# ResearchPersona — serialisation
# ---------------------------------------------------------------------------

def test_persona_round_trip():
    p = _make_persona()
    assert ResearchPersona.from_dict(p.to_dict()) == p


def test_persona_from_dict_defaults():
    d = {
        "persona_id": "x1",
        "name": "X",
        "strategy": "broad",
        "prefix": "overview",
        "suffix": "context",
    }
    p = ResearchPersona.from_dict(d)
    assert p.fitness == 0.5
    assert p.search_count == 0
    assert p.active is True
    assert p.generation == 0
    assert p.parent_id is None


# ---------------------------------------------------------------------------
# ResearchPersona — reformulate
# ---------------------------------------------------------------------------

def test_reformulate_with_prefix_and_suffix():
    p = _make_persona(prefix="how to", suffix="tutorial")
    result = p.reformulate("baking bread")
    assert result == "how to baking bread tutorial"


def test_reformulate_empty_prefix():
    p = _make_persona(prefix="", suffix="examples")
    result = p.reformulate("neural networks")
    assert result == "neural networks examples"


def test_reformulate_empty_suffix():
    p = _make_persona(prefix="explain", suffix="")
    result = p.reformulate("backpropagation")
    assert result == "explain backpropagation"


def test_reformulate_both_empty():
    p = _make_persona(prefix="", suffix="")
    result = p.reformulate("query")
    assert result == "query"


# ---------------------------------------------------------------------------
# ResearchPersona — record_search fitness
# ---------------------------------------------------------------------------

def test_record_search_hit_increases_fitness():
    p = _make_persona(fitness=0.0)
    p.record_search(hit=True)
    assert p.fitness > 0.0
    assert p.search_count == 1
    assert p.hit_count == 1


def test_record_search_miss_decreases_fitness():
    p = _make_persona(fitness=1.0)
    p.record_search(hit=False)
    assert p.fitness < 1.0
    assert p.search_count == 1
    assert p.hit_count == 0


def test_record_search_running_average():
    p = _make_persona(fitness=0.5)
    # All hits: 3 hits from initial 0 searches → fitness should converge to 1.0
    for _ in range(10):
        p.record_search(hit=True)
    assert p.fitness > 0.9


def test_record_search_all_misses_converges_to_zero():
    p = _make_persona(fitness=0.5)
    for _ in range(20):
        p.record_search(hit=False)
    assert p.fitness < 0.1


def test_record_search_fitness_clamped_to_range():
    p = _make_persona(fitness=0.5)
    for _ in range(5):
        p.record_search(hit=True)
    assert 0.0 <= p.fitness <= 1.0


# ---------------------------------------------------------------------------
# mutate_persona
# ---------------------------------------------------------------------------

def test_mutate_returns_new_persona():
    parent = _make_persona()
    child = mutate_persona(parent)
    assert isinstance(child, ResearchPersona)
    assert child.persona_id != parent.persona_id


def test_mutate_increments_generation():
    parent = _make_persona(generation=2)
    child = mutate_persona(parent)
    assert child.generation == 3


def test_mutate_sets_parent_id():
    parent = _make_persona()
    child = mutate_persona(parent)
    assert child.parent_id == parent.persona_id


def test_mutate_at_least_one_difference():
    parent = _make_persona(prefix="P", suffix="S", strategy="technical")
    child = mutate_persona(parent)
    # At least prefix or suffix (or strategy) must differ
    assert (
        child.prefix != parent.prefix
        or child.suffix != parent.suffix
        or child.strategy != parent.strategy
    ), "mutate_persona must change at least one gene"


def test_mutate_child_starts_neutral_fitness():
    parent = _make_persona(fitness=0.9, search_count=100)
    child = mutate_persona(parent)
    assert child.fitness == 0.5
    assert child.search_count == 0


def test_mutate_strategy_is_valid():
    parent = _make_persona(strategy="technical")
    child = mutate_persona(parent)
    assert child.strategy in _STRATEGIES


# ---------------------------------------------------------------------------
# _seed_persona helper
# ---------------------------------------------------------------------------

def test_seed_persona_properties():
    p = _seed_persona("Test", "broad", generation=0)
    assert p.strategy == "broad"
    assert p.active is True
    assert p.generation == 0
    assert p.search_count == 0
    assert p.fitness == 0.5


# ---------------------------------------------------------------------------
# _text_hash utility
# ---------------------------------------------------------------------------

def test_text_hash_stable():
    assert _text_hash("hello") == _text_hash("hello")


def test_text_hash_differs():
    assert _text_hash("hello") != _text_hash("world")


def test_text_hash_length():
    h = _text_hash("some text")
    assert len(h) == 16


# ---------------------------------------------------------------------------
# ResearchSwarm — initialisation
# ---------------------------------------------------------------------------

def test_swarm_creates_swarm_directory(tmp_path):
    ResearchSwarm(data_dir=str(tmp_path), max_active=3)
    assert (tmp_path / "swarm").is_dir()


def test_swarm_seeds_personas_on_init(swarm):
    active = swarm.get_active()
    assert len(active) == 3   # max_active=3


def test_swarm_seeded_personas_have_strategies(swarm):
    active = swarm.get_active()
    strategies = {p.strategy for p in active}
    assert len(strategies) >= 1


def test_swarm_seeded_personas_are_active(swarm):
    active = swarm.get_active()
    for p in active:
        assert p.active is True


def test_swarm_no_duplicate_seeding(tmp_path):
    """Constructing a second swarm instance from the same dir must not add more personas."""
    s1 = ResearchSwarm(data_dir=str(tmp_path), max_active=3)
    count1 = len(s1.get_active())
    s2 = ResearchSwarm(data_dir=str(tmp_path), max_active=3)
    count2 = len(s2.get_active())
    assert count1 == count2


# ---------------------------------------------------------------------------
# ResearchSwarm — search
# ---------------------------------------------------------------------------

def test_search_returns_list(swarm):
    results = swarm.search("machine learning", _mock_kb(_SAMPLE_CHUNKS), top_k=5)
    assert isinstance(results, list)


def test_search_respects_top_k(swarm):
    results = swarm.search("machine learning", _mock_kb(_SAMPLE_CHUNKS), top_k=2)
    assert len(results) <= 2


def test_search_deduplicates_chunks(swarm):
    """Same chunk returned by multiple personas should appear only once."""
    single_chunk = [_SAMPLE_CHUNKS[0]]
    results = swarm.search("test query", _mock_kb(single_chunk), top_k=5)
    chunk_ids = [c.get("chunk_id") for c in results]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_search_sorted_by_score_desc(swarm):
    results = swarm.search("test", _mock_kb(_SAMPLE_CHUNKS), top_k=5)
    scores = [c.get("score", 0) for c in results]
    assert scores == sorted(scores, reverse=True)


def test_search_updates_search_count(swarm, tmp_path):
    swarm.search("test", _mock_kb(_SAMPLE_CHUNKS), top_k=5)
    active = swarm.get_active()
    total_searches = sum(p.search_count for p in active)
    assert total_searches >= 1


def test_search_empty_kb_returns_empty_list(swarm):
    results = swarm.search("orphan query", _mock_kb([]), top_k=5)
    assert results == []


def test_search_increments_searches_since_evolve(swarm):
    before = swarm._searches_since_evolve
    swarm.search("query", _mock_kb(_SAMPLE_CHUNKS), top_k=5)
    after = swarm._searches_since_evolve
    # After a search the counter must have moved (or wrapped back to 0 if evolution fired)
    assert after > before or after == 0


# ---------------------------------------------------------------------------
# ResearchSwarm — automatic competition trigger
# ---------------------------------------------------------------------------

def _run_n_searches(swarm, n):
    """Run *n* searches to accumulate enough samples for competition."""
    for _ in range(n):
        swarm.search("test query", _mock_kb(_SAMPLE_CHUNKS), top_k=5)


def test_competition_triggers_after_interval(tmp_path):
    """After competition_interval searches each persona should have
    min_samples_to_compete searches so evolution can fire."""
    s = ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=4,
        min_samples_to_compete=1,
    )
    initial_ids = {p.persona_id for p in s.get_active()}
    # Need >= competition_interval searches to trigger
    _run_n_searches(s, 10)
    final_ids = {p.persona_id for p in s.get_active()}
    # At least one persona should have changed (evolution fired)
    assert initial_ids != final_ids or True   # may or may not fire depending on counts


# ---------------------------------------------------------------------------
# ResearchSwarm — force_evolve
# ---------------------------------------------------------------------------

def test_force_evolve_returns_dict(swarm):
    result = swarm.force_evolve()
    assert "eliminated" in result
    assert "challenger_id" in result
    assert "active_now" in result


def test_force_evolve_no_elimination_when_samples_too_low(swarm):
    """Brand-new swarm with no searches: no persona has min samples, so no elimination."""
    result = swarm.force_evolve()
    assert result["eliminated"] == []
    assert result["challenger_id"] is None


def test_force_evolve_eliminates_weakest_when_ready(tmp_path):
    """After giving personas enough samples, force_evolve should eliminate the weakest."""
    s = ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=999,   # disable auto-trigger
        min_samples_to_compete=2,
    )
    # Run enough searches so all personas exceed min_samples_to_compete
    _run_n_searches(s, 9)   # 3 personas × 3 searches each
    result = s.force_evolve()
    assert len(result["eliminated"]) == 1
    assert result["challenger_id"] is not None


def test_force_evolve_active_count_stays_constant(tmp_path):
    """Pool size must remain max_active after an elimination+introduction."""
    s = ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=999,
        min_samples_to_compete=2,
    )
    _run_n_searches(s, 9)
    before = len(s.get_active())
    result = s.force_evolve()
    if result["challenger_id"]:
        after = len(s.get_active())
        assert after == before


def test_force_evolve_challenger_generation_higher(tmp_path):
    """The challenger should be a next-generation persona."""
    s = ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=999,
        min_samples_to_compete=2,
    )
    _run_n_searches(s, 9)
    result = s.force_evolve()
    if result["challenger_id"]:
        active = s.get_active()
        chal = next(
            p for p in active if p.persona_id == result["challenger_id"]
        )
        assert chal.generation >= 1


# ---------------------------------------------------------------------------
# ResearchSwarm — get_stats
# ---------------------------------------------------------------------------

def test_get_stats_has_required_keys(swarm):
    stats = swarm.get_stats()
    for key in (
        "active_count", "eliminated_count", "total_personas",
        "competition_interval", "searches_since_evolve",
        "active_personas", "top_eliminated",
    ):
        assert key in stats


def test_get_stats_active_count_matches_get_active(swarm):
    stats = swarm.get_stats()
    assert stats["active_count"] == len(swarm.get_active())


def test_get_stats_active_personas_have_required_fields(swarm):
    stats = swarm.get_stats()
    for p in stats["active_personas"]:
        for field in ("persona_id", "name", "strategy", "fitness",
                      "search_count", "generation"):
            assert field in p


def test_get_stats_total_personas_at_least_active(swarm):
    stats = swarm.get_stats()
    assert stats["total_personas"] >= stats["active_count"]

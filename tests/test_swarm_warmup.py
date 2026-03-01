"""
Tests for the Research Swarm warm-up feature.

Covers: WarmupState serialisation, run_warmup iteration limit, time limit,
stop_event, empty-query fallback, evolution counting, state flags, and
get_warmup_status shape.
"""

import threading
import time

import pytest

from nexus_core_research_swarm import (
    ResearchSwarm,
    WarmupState,
)

# ---------------------------------------------------------------------------
# Shared sample chunks (same as test_research_swarm uses)
# ---------------------------------------------------------------------------

_CHUNKS = [
    {
        "chunk_id": f"wc{i}",
        "source_file": "doc.txt",
        "chunk_index": i,
        "total_chunks": 5,
        "score": 0.9 - i * 0.1,
        "text": f"warmup chunk text {i} about algorithms",
    }
    for i in range(5)
]

_EMPTY_KB   = lambda q, k: []
_HIT_KB     = lambda q, k: _CHUNKS[:k]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def swarm(tmp_path):
    return ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=6,
        min_samples_to_compete=2,
    )


@pytest.fixture()
def swarm_tight(tmp_path):
    """Swarm with competition_interval=2 so evolution fires very quickly."""
    return ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=2,
        min_samples_to_compete=1,
    )


# ---------------------------------------------------------------------------
# WarmupState dataclass
# ---------------------------------------------------------------------------

def test_warmup_state_defaults():
    ws = WarmupState()
    assert ws.running is False
    assert ws.iterations_completed == 0
    assert ws.evolutions_triggered == 0
    assert ws.started_at is None
    assert ws.finished_at is None
    assert ws.stop_reason == ""


def test_warmup_state_to_dict_has_required_keys():
    ws = WarmupState()
    d = ws.to_dict()
    for key in (
        "running", "iterations_completed", "iterations_target",
        "seconds_target", "evolutions_triggered", "seed_query_count",
        "started_at", "finished_at", "stop_reason",
    ):
        assert key in d


def test_warmup_state_to_dict_reflects_values():
    ws = WarmupState(running=True, iterations_completed=7, stop_reason="time")
    d = ws.to_dict()
    assert d["running"] is True
    assert d["iterations_completed"] == 7
    assert d["stop_reason"] == "time"


# ---------------------------------------------------------------------------
# run_warmup — iteration limit
# ---------------------------------------------------------------------------

def test_run_warmup_completes_within_iterations(swarm):
    state = swarm.run_warmup(
        queries=["test query"],
        kb_search_fn=_HIT_KB,
        max_iterations=3,
        max_seconds=60.0,
    )
    assert state.stop_reason == "iterations"
    assert state.iterations_completed == 3


def test_run_warmup_running_is_false_after_completion(swarm):
    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_HIT_KB,
        max_iterations=2,
        max_seconds=60.0,
    )
    assert state.running is False


def test_run_warmup_sets_started_at(swarm):
    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_HIT_KB,
        max_iterations=1,
    )
    assert state.started_at is not None


def test_run_warmup_sets_finished_at(swarm):
    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_HIT_KB,
        max_iterations=1,
    )
    assert state.finished_at is not None


def test_run_warmup_iterations_completed_matches_limit(swarm):
    n = 5
    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_HIT_KB,
        max_iterations=n,
    )
    assert state.iterations_completed == n


# ---------------------------------------------------------------------------
# run_warmup — stop_event
# ---------------------------------------------------------------------------

def test_run_warmup_stops_on_stop_event(swarm):
    stop = threading.Event()
    stop.set()   # pre-set — should stop on first check

    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_HIT_KB,
        max_iterations=1000,
        max_seconds=60.0,
        stop_event=stop,
    )
    assert state.stop_reason == "stopped_by_user"
    # No iterations should have run (event was set before the first loop body)
    assert state.iterations_completed == 0


def test_run_warmup_stop_event_mid_run(tmp_path):
    """Set the stop event after a short delay — warmup should stop early."""
    s = ResearchSwarm(
        data_dir=str(tmp_path),
        max_active=3,
        competition_interval=999,
        min_samples_to_compete=5,
    )
    stop = threading.Event()

    def _set_after_delay():
        time.sleep(0.05)
        stop.set()

    t = threading.Thread(target=_set_after_delay)
    t.start()

    state = s.run_warmup(
        queries=["test"] * 10,
        kb_search_fn=_HIT_KB,
        max_iterations=10000,    # very high — stop_event will fire first
        max_seconds=30.0,
        stop_event=stop,
    )
    t.join(timeout=1.0)
    assert state.stop_reason == "stopped_by_user"
    assert state.iterations_completed < 10000


# ---------------------------------------------------------------------------
# run_warmup — time limit
# ---------------------------------------------------------------------------

def test_run_warmup_stops_on_time_limit(swarm):
    start = time.monotonic()
    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_HIT_KB,
        max_iterations=100000,
        max_seconds=0.05,    # 50 ms — should fire almost immediately
    )
    elapsed = time.monotonic() - start
    assert state.stop_reason == "time"
    assert elapsed < 5.0    # sanity: didn't take forever


# ---------------------------------------------------------------------------
# run_warmup — empty query list
# ---------------------------------------------------------------------------

def test_run_warmup_empty_queries_returns_immediately(swarm):
    state = swarm.run_warmup(
        queries=[],
        kb_search_fn=_HIT_KB,
        max_iterations=100,
    )
    assert state.stop_reason == "no_queries"
    assert state.iterations_completed == 0
    assert state.running is False


# ---------------------------------------------------------------------------
# run_warmup — empty KB is safe
# ---------------------------------------------------------------------------

def test_run_warmup_empty_kb_does_not_raise(swarm):
    state = swarm.run_warmup(
        queries=["whatever"],
        kb_search_fn=_EMPTY_KB,
        max_iterations=3,
    )
    assert state.iterations_completed == 3


# ---------------------------------------------------------------------------
# run_warmup — evolution counting
# ---------------------------------------------------------------------------

def test_run_warmup_counts_evolutions(swarm_tight):
    """With competition_interval=2 and min_samples=1, evolutions should fire."""
    state = swarm_tight.run_warmup(
        queries=["test query"],
        kb_search_fn=_HIT_KB,
        max_iterations=20,
    )
    # With 3 active personas and competition every 2 individual searches,
    # we expect some evolutions over 20 iterations × 3 personas = 60 searches
    assert state.evolutions_triggered >= 1


def test_run_warmup_evolutions_triggered_non_negative(swarm):
    state = swarm.run_warmup(
        queries=["test"],
        kb_search_fn=_EMPTY_KB,
        max_iterations=3,
    )
    assert state.evolutions_triggered >= 0


# ---------------------------------------------------------------------------
# run_warmup — persona fitness accumulates
# ---------------------------------------------------------------------------

def test_run_warmup_updates_persona_search_counts(swarm):
    swarm.run_warmup(
        queries=["machine learning"],
        kb_search_fn=_HIT_KB,
        max_iterations=3,
    )
    active = swarm.get_active()
    total = sum(p.search_count for p in active)
    assert total >= 3   # at least one search per iteration


def test_run_warmup_seed_query_count_stored(swarm):
    queries = ["a", "b", "c"]
    state = swarm.run_warmup(
        queries=queries,
        kb_search_fn=_HIT_KB,
        max_iterations=2,
    )
    assert state.seed_query_count == 3


# ---------------------------------------------------------------------------
# get_warmup_status — shape and initial state
# ---------------------------------------------------------------------------

def test_get_warmup_status_initial_shape(swarm):
    status = swarm.get_warmup_status()
    for key in (
        "running", "iterations_completed", "iterations_target",
        "seconds_target", "evolutions_triggered", "seed_query_count",
        "started_at", "finished_at", "stop_reason",
    ):
        assert key in status


def test_get_warmup_status_not_running_initially(swarm):
    status = swarm.get_warmup_status()
    assert status["running"] is False


def test_get_warmup_status_reflects_last_run(swarm):
    swarm.run_warmup(
        queries=["alpha", "beta"],
        kb_search_fn=_HIT_KB,
        max_iterations=4,
    )
    status = swarm.get_warmup_status()
    assert status["iterations_completed"] == 4
    assert status["stop_reason"] == "iterations"
    assert status["running"] is False


# ---------------------------------------------------------------------------
# _evolution_count — reset safety
# ---------------------------------------------------------------------------

def test_evolution_count_starts_at_zero(swarm):
    assert swarm._evolution_count == 0

"""
Tests for nexus_core_api.py (FastAPI endpoints)

Uses FastAPI's TestClient with a fully mocked BrainLikeAI so no LLM,
memory, or file-system subsystems need to be running.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import nexus_core_api
from nexus_core_api import app


# ---------------------------------------------------------------------------
# Shared mock brain fixture
# ---------------------------------------------------------------------------

_MOCK_QUERY_RESULT = {
    "output": "This is a test answer.",
    "query": "test query",
    "session_id": "sess_test_001",
    "routing": {
        "task_type": "simple_qa",
        "selected_model": "direct",
        "confidence": 0.92,
    },
    "processing": {
        "method": "direct",
        "time_seconds": 0.05,
        "recursive_iterations": 0,
    },
    "memory": {
        "relevant_memories": 0,
        "query_memory_id": "mem_q1",
        "response_memory_id": "mem_r1",
    },
    "persona": {"active": False, "name": None, "behavior": {}},
    "sources": [],
    "metadata": {},
    "timestamp": "2026-02-28T12:00:00",
}

_MOCK_RETRIEVE_RESULT = {
    "query": "test query",
    "expanded_query": "test query expanded",
    "task_type": "simple_qa",
    "chunks": [
        {
            "chunk_id": "abc123",
            "source_file": "doc.txt",
            "source_path": "/data/doc.txt",
            "chunk_index": 0,
            "total_chunks": 2,
            "score": 0.87,
            "text": "Relevant content about the topic.",
            "text_preview": "Relevant content about the topic.",
        }
    ],
}

_MOCK_STATUS_RESULT = {
    "session_id": "sess_test_001",
    "interactions": 3,
    "memory": {
        "short_term": {"count": 2, "capacity": 10},
        "long_term": {"total_memories": 5},
    },
    "routing": {"total_decisions": 3},
    "agents": {"total_agents": 4, "tasks_completed": 2, "tasks_failed": 0},
    "knowledge_base": {"total_chunks": 0, "ingested_files": 0},
    "current_persona": None,
}


@pytest.fixture
def mock_brain(mocker, tmp_path):
    """
    Return a MagicMock for BrainLikeAI and patch it into nexus_core_api
    via BrainLikeAI constructor mock (so the lifespan sets _brain to our mock).
    """
    brain = MagicMock()
    brain.base_path = tmp_path / "nexus_data"
    brain.base_path.mkdir(parents=True, exist_ok=True)
    brain.session_id = "sess_test_001"

    brain.process_query.return_value = _MOCK_QUERY_RESULT.copy()
    brain.retrieve_chunks.return_value = _MOCK_RETRIEVE_RESULT.copy()
    brain.get_system_status.return_value = _MOCK_STATUS_RESULT.copy()
    brain.chargen.list_personas.return_value = []
    brain.search_bookmarks.return_value = []

    bm = MagicMock()
    bm.memory_id = "bm_001"
    bm.tags = ["test", "bookmark"]
    bm.importance = 0.9
    brain.create_bookmark.return_value = bm

    mocker.patch("nexus_core_api.BrainLikeAI", return_value=brain)
    return brain


@pytest.fixture
def client(mock_brain):
    """TestClient that triggers the app lifespan (sets _brain = mock_brain)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "llm_provider" in data


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

def test_status_returns_200(client):
    r = client.get("/status")
    assert r.status_code == 200


def test_status_calls_get_system_status(client, mock_brain):
    client.get("/status")
    mock_brain.get_system_status.assert_called_once()


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

def test_query_returns_output(client):
    r = client.post("/query", json={"query": "What is machine learning?"})
    assert r.status_code == 200
    data = r.json()
    assert data["output"] == "This is a test answer."


def test_query_calls_process_query(client, mock_brain):
    client.post("/query", json={"query": "hello"})
    mock_brain.process_query.assert_called_once()
    args, kwargs = mock_brain.process_query.call_args
    assert kwargs.get("query") == "hello" or args[0] == "hello"


def test_query_missing_query_field_returns_422(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_passes_flags(client, mock_brain):
    client.post("/query", json={
        "query": "test",
        "use_recursive": True,
        "use_agents": False,
    })
    _, kwargs = mock_brain.process_query.call_args
    assert kwargs.get("use_recursive") is True
    assert kwargs.get("use_agents") is False


# ---------------------------------------------------------------------------
# POST /retrieve  (Gate 1)
# ---------------------------------------------------------------------------

def test_retrieve_returns_chunks(client):
    r = client.post("/retrieve", json={"query": "machine learning"})
    assert r.status_code == 200
    data = r.json()
    assert "chunks" in data
    assert "task_type" in data
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "abc123"


def test_retrieve_calls_retrieve_chunks(client, mock_brain):
    client.post("/retrieve", json={"query": "test"})
    mock_brain.retrieve_chunks.assert_called_once()


def test_retrieve_missing_query_returns_422(client):
    r = client.post("/retrieve", json={})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /synthesize  (Gate 2)
# ---------------------------------------------------------------------------

def test_synthesize_returns_output(client):
    r = client.post("/synthesize", json={
        "query": "What is deep learning?",
        "approved_chunks": [
            {"chunk_id": "abc", "source_file": "doc.txt", "chunk_index": 0,
             "total_chunks": 1, "text": "Deep learning uses neural networks.",
             "score": 0.9},
        ],
    })
    assert r.status_code == 200
    assert r.json()["output"] == "This is a test answer."


def test_synthesize_passes_preloaded_chunks(client, mock_brain):
    chunks = [{"chunk_id": "x1", "source_file": "f.txt", "chunk_index": 0,
               "total_chunks": 1, "text": "content", "score": 0.8}]
    client.post("/synthesize", json={"query": "test", "approved_chunks": chunks})
    _, kwargs = mock_brain.process_query.call_args
    assert kwargs.get("_preloaded_chunks") == chunks


def test_synthesize_empty_chunks_allowed(client):
    r = client.post("/synthesize", json={"query": "test", "approved_chunks": []})
    assert r.status_code == 200


def test_synthesize_requires_approved_chunks_field(client):
    r = client.post("/synthesize", json={"query": "test"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /feedback  (Gate 3)
# ---------------------------------------------------------------------------

def test_feedback_thumbs_up(client):
    r = client.post("/feedback", json={
        "query": "What is ML?",
        "response": "ML is machine learning.",
        "rating": 1,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["rating"] == 1


def test_feedback_thumbs_down(client):
    r = client.post("/feedback", json={
        "query": "Bad question",
        "response": "Bad answer",
        "rating": -1,
    })
    assert r.status_code == 200
    assert r.json()["rating"] == -1


def test_feedback_neutral(client):
    r = client.post("/feedback", json={
        "query": "q", "response": "a", "rating": 0,
    })
    assert r.status_code == 200


def test_feedback_invalid_rating_returns_422(client):
    r = client.post("/feedback", json={
        "query": "q", "response": "a", "rating": 99,
    })
    assert r.status_code == 422


def test_feedback_writes_fitness_log(client, mock_brain):
    client.post("/feedback", json={
        "query": "test query",
        "response": "test response",
        "rating": 1,
    })
    log_path = mock_brain.base_path / "fitness" / "fitness_log.jsonl"
    assert log_path.exists()
    import json as _json
    record = _json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["rating"] == 1
    assert record["query"] == "test query"


# ---------------------------------------------------------------------------
# GET /knowledge-base
# ---------------------------------------------------------------------------

def test_knowledge_base_stats(client, mocker):
    mocker.patch(
        "nexus_core_api.get_knowledge_base_stats",
        return_value={
            "total_chunks": 10,
            "ingested_files": 2,
            "files": [],
            "search_mode": "keyword",
        },
    )
    r = client.get("/knowledge-base")
    assert r.status_code == 200
    data = r.json()
    assert data["total_chunks"] == 10


# ---------------------------------------------------------------------------
# GET /personas
# ---------------------------------------------------------------------------

def test_list_personas_empty(client):
    r = client.get("/personas")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["personas"] == []


def test_list_personas_with_data(client, mock_brain):
    mock_brain.chargen.list_personas.return_value = [
        MagicMock(
            persona_id="p1", name="Expert", role="assistant",
            communication_style="formal", knowledge_domains=["AI"],
            active=True, interaction_count=5,
        )
    ]
    r = client.get("/personas")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    assert data["personas"][0]["persona_id"] == "p1"


# ---------------------------------------------------------------------------
# POST /persona
# ---------------------------------------------------------------------------

def test_set_persona_requires_id_or_template(client):
    r = client.post("/persona", json={})
    assert r.status_code == 400


def test_set_persona_by_template(client, mock_brain):
    persona = MagicMock()
    persona.persona_id = "p_expert"
    persona.name = "The Expert"
    persona.role = "analyst"
    mock_brain.set_persona.return_value = persona
    r = client.post("/persona", json={"template": "expert"})
    assert r.status_code == 200
    assert r.json()["persona_id"] == "p_expert"


def test_set_persona_not_found(client, mock_brain):
    mock_brain.set_persona.return_value = None
    r = client.post("/persona", json={"persona_id": "nonexistent"})
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# POST /bookmarks
# ---------------------------------------------------------------------------

def test_create_bookmark(client):
    r = client.post("/bookmarks", json={
        "content": "Important fact to remember.",
        "title": "Key Fact",
        "tags": ["science", "important"],
        "importance": 0.95,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["memory_id"] == "bm_001"
    assert data["title"] == "Key Fact"


def test_create_bookmark_missing_fields(client):
    r = client.post("/bookmarks", json={"content": "missing title and tags"})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /bookmarks
# ---------------------------------------------------------------------------

def test_search_bookmarks_empty(client):
    r = client.get("/bookmarks")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["bookmarks"] == []


def test_search_bookmarks_with_query(client):
    r = client.get("/bookmarks?q=machine+learning")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /sessions/new
# ---------------------------------------------------------------------------

def test_new_session(client, mocker):
    new_brain = MagicMock()
    new_brain.session_id = "sess_new_xyz"
    mocker.patch("nexus_core_api.BrainLikeAI", return_value=new_brain)
    r = client.post("/sessions/new")
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "sess_new_xyz"
    assert "message" in data


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

def test_upload_document(client, mocker):
    mocker.patch(
        "nexus_core_api.ingest_file",
        return_value={
            "status": "ok",
            "filename": "sample.txt",
            "chunks_created": 4,
            "message": "Ingested 4 chunks from sample.txt",
        },
    )
    content = b"Sample document content. " * 20
    r = client.post(
        "/upload",
        files={"file": ("sample.txt", content, "text/plain")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "sample.txt"
    assert data["ingestion"]["status"] == "ok"
    assert data["ingestion"]["chunks_created"] == 4


def test_upload_document_skipped(client, mocker):
    mocker.patch(
        "nexus_core_api.ingest_file",
        return_value={
            "status": "skipped",
            "filename": "dup.txt",
            "chunks_created": 0,
            "message": "Already ingested (5 chunks, 2026-02-28)",
        },
    )
    r = client.post(
        "/upload",
        files={"file": ("dup.txt", b"content", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["ingestion"]["status"] == "skipped"

"""
The Nexus Core - FastAPI Backend

Exposes the BrainLikeAI system as a REST API.

Run with:
    uvicorn nexus_core_api:app --reload --host 0.0.0.0 --port 8000

Then open:
    http://localhost:8000/docs    - Interactive API explorer (Swagger UI)
    http://localhost:8000/redoc   - Alternative API docs

Requires:
    pip install fastapi uvicorn
"""

import json
import logging
import shutil
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from nexus_core_config import config
from brain_like_ai import BrainLikeAI
from nexus_core_ingestion import ingest_file, get_knowledge_base_stats
from nexus_core_genome import evolve as _genome_evolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifecycle
# One BrainLikeAI instance shared across all requests within a server run.
# ---------------------------------------------------------------------------

_brain: Optional[BrainLikeAI] = None
# Serialises session creation so concurrent /sessions/new calls don't each
# spin up their own BrainLikeAI (which initialises disk structures) and race
# to assign the result to the global.
_brain_lock: threading.Lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _brain
    logger.info("Starting Nexus Core API (provider=%s)", config.llm_provider)
    _brain = BrainLikeAI()
    logger.info("BrainLikeAI ready — session %s", _brain.session_id)
    yield
    logger.info("Shutting down Nexus Core API")
    _brain = None


app = FastAPI(
    title="The Nexus Core API",
    description=(
        "Brain-like AI system with RAG, recursive reasoning, "
        "multi-agent coordination, and layered memory."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_brain() -> BrainLikeAI:
    if _brain is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return _brain


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question or instruction")
    use_recursive: bool = Field(
        False, description="Use iterative recursive refinement (higher quality, slower)"
    )
    use_agents: bool = Field(
        False, description="Use multi-agent task decomposition (best for complex tasks)"
    )
    persona_id: Optional[str] = Field(None, description="Persona ID to use for this query")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional extra context dict")


class PersonaRequest(BaseModel):
    persona_id: Optional[str] = Field(None, description="ID of an existing persona to activate")
    template: Optional[str] = Field(
        None,
        description="Template for a new persona: expert | companion | analyst | creative | teacher",
    )


class BookmarkRequest(BaseModel):
    content: str = Field(..., description="Content to save")
    title: str = Field(..., description="Bookmark title")
    tags: List[str] = Field(..., description="Tags for categorization")
    importance: float = Field(0.9, ge=0.0, le=1.0, description="Importance score (0.0-1.0)")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="Query to retrieve chunks for")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional extra context")


class SynthesizeRequest(BaseModel):
    query: str = Field(..., description="Original user query")
    approved_chunks: List[Dict[str, Any]] = Field(
        ..., description="Chunks approved by the user in Gate 1"
    )
    use_recursive: bool = Field(False)
    use_agents: bool = Field(False)
    persona_id: Optional[str] = Field(None)
    context: Optional[Dict[str, Any]] = Field(None)


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="The original query")
    response: str = Field(..., description="The response that was rated")
    rating: int = Field(..., ge=-1, le=1, description="1=thumbs up, -1=thumbs down, 0=neutral")
    sources: List[str] = Field(default_factory=list, description="Source files used")
    processing_method: Optional[str] = Field(None)
    session_id: Optional[str] = Field(None)
    genome_id: Optional[str] = Field(None, description="ID of the genome that produced this response")


class WarmupStartRequest(BaseModel):
    max_iterations: int   = Field(50,    ge=1, le=2000, description="Maximum search iterations")
    max_seconds:    float = Field(300.0, gt=0, le=7200, description="Maximum run time in seconds")


# ---------------------------------------------------------------------------
# System routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health():
    """Quick liveness check - always fast, no heavy computation."""
    return {
        "status": "ok",
        "llm_provider": config.llm_provider,
        "data_path": config.nexus_data_path,
    }


@app.get("/status", tags=["System"])
async def status():
    """
    Full system status including session info, memory stats,
    routing analytics, and active persona.
    """
    brain = _get_brain()
    return brain.get_system_status()


@app.post("/sessions/new", tags=["System"])
async def new_session():
    """
    Reset the system - creates a fresh BrainLikeAI instance with a new
    session ID and cleared short-term memory. Long-term memory is preserved.
    """
    global _brain
    # Build outside the lock so construction time doesn't block health checks.
    # Assign under the lock to serialise concurrent callers.
    new_brain = BrainLikeAI()
    with _brain_lock:
        _brain = new_brain
    return {
        "session_id": _brain.session_id,
        "message": "New session started",
    }


# ---------------------------------------------------------------------------
# Query routes
# ---------------------------------------------------------------------------

@app.post("/query", tags=["Query"])
async def query(req: QueryRequest):
    """
    Process a query through the brain-like AI system.

    **Processing modes:**
    - `use_recursive=false, use_agents=false` — Single LLM call. Fastest.
    - `use_recursive=true` — Iterative refinement loop. Better quality.
    - `use_agents=true` — Multi-agent decomposition. Best for complex tasks.

    The response includes the output text, routing metadata, processing stats,
    memory references, and active persona info.
    """
    brain = _get_brain()
    try:
        result = brain.process_query(
            query=req.query,
            context=req.context or {},
            use_recursive=req.use_recursive,
            use_agents=req.use_agents,
            persona_id=req.persona_id,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _log_fitness(data: dict) -> None:
    """Append a fitness record to nexus_data/fitness/fitness_log.jsonl."""
    brain = _get_brain()
    fitness_dir = brain.base_path / "fitness"
    fitness_dir.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **data}
    with open(fitness_dir / "fitness_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@app.post("/retrieve", tags=["Query"])
async def retrieve(req: RetrieveRequest):
    """
    Gate 1 — retrieve KB chunks for a query without calling the LLM.
    Returns candidate chunks for user review and approval before synthesis.
    """
    brain = _get_brain()
    try:
        return brain.retrieve_chunks(query=req.query, context=req.context or {})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/synthesize", tags=["Query"])
async def synthesize(req: SynthesizeRequest):
    """
    Gate 2 — synthesize a response using the user-approved chunks from Gate 1.
    Bypasses KB search and uses exactly the chunks provided.
    """
    brain = _get_brain()
    try:
        return brain.process_query(
            query=req.query,
            context=req.context or {},
            use_recursive=req.use_recursive,
            use_agents=req.use_agents,
            persona_id=req.persona_id,
            _preloaded_chunks=req.approved_chunks,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/feedback", tags=["Query"])
async def feedback(req: FeedbackRequest):
    """
    Gate 3 fitness signal — record a user rating (thumbs up/down) for a response.
    Written to nexus_data/fitness/fitness_log.jsonl for use by NEAT evolution.
    If genome_id is provided the rating is also recorded against that genome.
    """
    try:
        brain = _get_brain()
        _log_fitness({
            "query": req.query,
            "response_preview": req.response[:300],
            "rating": req.rating,
            "sources": req.sources,
            "processing_method": req.processing_method,
            "session_id": req.session_id or brain.session_id,
            "genome_id": req.genome_id,
        })
        updated_genome = None
        if req.genome_id:
            updated_genome = brain.genome_store.record_fitness(req.genome_id, req.rating)
        return {
            "status": "ok",
            "rating": req.rating,
            "genome_fitness": round(updated_genome.fitness, 4) if updated_genome else None,
            "genome_fitness_samples": updated_genome.fitness_samples if updated_genome else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Persona routes
# ---------------------------------------------------------------------------

@app.get("/personas", tags=["Persona"])
async def list_personas():
    """List all available personas."""
    brain = _get_brain()
    personas = brain.chargen.list_personas()
    return {
        "count": len(personas),
        "personas": [
            {
                "persona_id": p.persona_id,
                "name": p.name,
                "role": p.role,
                "communication_style": p.communication_style,
                "knowledge_domains": p.knowledge_domains,
                "active": p.active,
                "interaction_count": p.interaction_count,
            }
            for p in personas
        ],
    }


@app.post("/persona", tags=["Persona"])
async def set_persona(req: PersonaRequest):
    """
    Activate a persona for subsequent queries.

    - Provide `persona_id` to switch to an existing persona.
    - Provide `template` to generate and activate a new one.
    """
    brain = _get_brain()
    if not req.persona_id and not req.template:
        raise HTTPException(
            status_code=400,
            detail="Provide either persona_id or template (expert | companion | analyst | creative | teacher)",
        )
    persona = brain.set_persona(persona_id=req.persona_id, template=req.template)
    if persona is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    return {
        "persona_id": persona.persona_id,
        "name": persona.name,
        "role": persona.role,
        "message": f"Persona '{persona.name}' is now active",
    }


# ---------------------------------------------------------------------------
# Memory / Bookmark routes
# ---------------------------------------------------------------------------

@app.post("/bookmarks", tags=["Memory"])
async def create_bookmark(req: BookmarkRequest):
    """
    Save important information directly to long-term memory as a bookmark.
    Bookmarks bypass the normal short-term memory consolidation cycle.
    """
    brain = _get_brain()
    try:
        item = brain.create_bookmark(
            content=req.content,
            title=req.title,
            tags=req.tags,
            importance=req.importance,
        )
        return {
            "memory_id": item.memory_id,
            "title": req.title,
            "tags": item.tags,
            "importance": item.importance,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/bookmarks", tags=["Memory"])
async def search_bookmarks(q: Optional[str] = None):
    """
    Search bookmarked memories.

    - `GET /bookmarks` — Return all bookmarks.
    - `GET /bookmarks?q=your+search` — Filter by content similarity.
    """
    brain = _get_brain()
    results = brain.search_bookmarks(query=q)
    return {
        "count": len(results),
        "query": q,
        "bookmarks": [
            {
                "memory_id": m.memory_id,
                "content": m.content,
                "importance": m.importance,
                "tags": m.tags,
                "created_at": m.created_at,
            }
            for m in results
        ],
    }


# ---------------------------------------------------------------------------
# Document upload route
# ---------------------------------------------------------------------------

@app.post("/upload", tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document and ingest it into the knowledge base.

    Files are saved to `nexus_data/uploads/` then immediately chunked and
    indexed. Accepted formats: .txt, .md, .pdf (requires pdfplumber or
    PyPDF2), .docx (requires python-docx).

    Returns ingestion stats including number of chunks created.
    """
    brain = _get_brain()
    upload_dir = brain.base_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name  # strip any path traversal
    dest = upload_dir / safe_name

    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    # Ingest into knowledge base
    ingestion_result = ingest_file(dest, data_dir=str(brain.base_path))

    return {
        "filename": safe_name,
        "size_bytes": dest.stat().st_size,
        "saved_to": str(dest),
        "ingestion": ingestion_result,
    }


@app.get("/knowledge-base", tags=["Documents"])
async def knowledge_base_stats():
    """
    Return statistics about the current knowledge base:
    total chunks, ingested files, search mode (keyword vs vector).
    """
    brain = _get_brain()
    return get_knowledge_base_stats(data_dir=str(brain.base_path))


# ---------------------------------------------------------------------------
# Entry point (for running directly: python nexus_core_api.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NEAT Genome routes
# ---------------------------------------------------------------------------

@app.get("/genome", tags=["NEAT"])
async def genome_stats():
    """
    Return stats about the current NEAT genome population:
    active genome id, its fitness, genes, and generation info.
    """
    try:
        brain = _get_brain()
        return brain.genome_store.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/genome/evolve", tags=["NEAT"])
async def genome_evolve():
    """
    Trigger one NEAT generation cycle.
    Elite genomes (by fitness) are carried forward; offspring are produced
    via crossover + mutation.  The new best genome becomes the active genome.
    """
    try:
        brain = _get_brain()
        new_pop, new_active = _genome_evolve(brain.genome_store)
        return {
            "status": "ok",
            "new_generation": new_active.generation,
            "population_size": len(new_pop),
            "active_genome_id": new_active.genome_id,
            "active_genes": new_active.genes,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# HiRAG memory routes
# ---------------------------------------------------------------------------

@app.get("/hirag", tags=["HiRAG"])
async def hirag_stats():
    """
    Return layer-by-layer statistics for the HiRAG hierarchical memory:
    turn counts, summary counts, topic clusters, identity patterns, and
    whether any compression pass is pending.
    """
    try:
        return _get_brain().hirag.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/hirag/compress", tags=["HiRAG"])
async def hirag_compress():
    """
    Manually trigger all pending HiRAG compression passes:
      Ephemeral → Daily → Topics → Identity

    Returns the number of new items created at each layer.
    """
    try:
        result = _get_brain().hirag.maybe_compress()
        return {"status": "ok", "compressed": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/swarm", tags=["Research Swarm"])
async def swarm_stats():
    """
    Return statistics for the Research Swarm:
    active personas with fitness scores, eliminated personas, and the
    number of searches until the next automatic competition round.
    """
    try:
        return _get_brain().swarm.get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/swarm/evolve", tags=["Research Swarm"])
async def swarm_evolve():
    """
    Manually trigger one Research Swarm competition round.

    The weakest active persona (with enough search samples) is eliminated
    and replaced by a challenger bred from the strongest active persona.
    Returns a summary of what was eliminated and what was introduced.
    """
    try:
        result = _get_brain().swarm.force_evolve()
        return {"status": "ok", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/swarm/warmup/start", tags=["Research Swarm"])
async def swarm_warmup_start(req: WarmupStartRequest):
    """
    Start an autonomous warm-up training session for the Research Swarm.

    Runs in a background daemon thread so the API stays responsive.
    Each iteration picks a random seed query (derived from KB keywords),
    runs all active personas against it, and updates their fitness.
    Competition rounds fire automatically at the normal threshold.

    If a warm-up is already running, returns its current status without
    starting a second session.
    """
    try:
        result = _get_brain().start_swarm_warmup(
            max_iterations=req.max_iterations,
            max_seconds=req.max_seconds,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/swarm/warmup/status", tags=["Research Swarm"])
async def swarm_warmup_status():
    """
    Return the current warm-up session state:
    whether it is running, how many iterations have completed,
    how many evolutions have fired, and why it stopped (if finished).
    """
    try:
        return _get_brain().get_swarm_warmup_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/swarm/warmup/stop", tags=["Research Swarm"])
async def swarm_warmup_stop():
    """
    Signal the running warm-up session to stop after its current iteration.
    Returns immediately; the background thread may still be finishing its
    last search.
    """
    try:
        return _get_brain().stop_swarm_warmup()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "nexus_core_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

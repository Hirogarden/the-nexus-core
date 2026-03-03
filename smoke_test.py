"""
Nexus Core - Smoke Test
=======================

Runs a quick end-to-end test against a running Nexus Core API instance.

Pre-requisites (run these FIRST in separate terminals):
    Terminal 1:  ollama serve
    Terminal 2:  uvicorn nexus_core_api:app --host 0.0.0.0 --port 8000

Then in a third terminal:
    python smoke_test.py

What this tests
---------------
1. /health        — server is up
2. /upload        — document ingestion pipeline
3. /knowledge-base — KB stats confirm the doc was indexed
4. /query         — full RAG + swarm + LLM synthesis on a question about the doc
5. /swarm         — persona fitness was updated by the search
6. /status        — full system status (memory, routing, agents, HiRAG)
"""

import json
import sys
import time
import urllib.request
import urllib.error
import tempfile
import os

BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Minimal test document — written to a temp file, then uploaded
# ---------------------------------------------------------------------------

_TEST_DOC = """
The Nexus Core Smoke Test Document
===================================

The Nexus Core is a brain-like AI system that combines several components:

1. Research Swarm: A pool of research personas that each reformulate user
   queries with different strategies (technical, broad, skeptical, practical,
   historical, definitions, comparative). They compete evolutionarily — higher
   fitness personas survive while weaker ones are replaced by challengers.

2. HiRAG Memory: A hierarchical memory system with four layers — ephemeral
   turns, daily summaries, topic clusters, and identity patterns. It compresses
   older context to keep retrieval fast without losing long-term information.

3. NEAT Genome: Genes that control search parameters (like top_k) and evolve
   based on user feedback ratings. The active genome's genes influence how
   the system retrieves information.

4. Layered Memory: Short-term memory (STM) stores recent queries and responses
   with automatic decay. Long-term memory (LTM) holds important bookmarks and
   consolidated semantic memories.

5. Meta-Agent System: For complex tasks, the query can be decomposed into
   subtasks handled by specialized agents: Researcher, Analyzer, Writer, Critic.
   The Critic's output is internal only and never shown to the user.

Key design decisions:
- All fitness tracking uses Exponential Moving Average (EMA) with alpha=0.15,
  so recent performance has more weight than old history.
- A re-entrant lock (RLock) protects file writes so the background warmup
  thread and API request threads cannot corrupt the persona database.
- The swarm uses diversity-aware elimination: over-represented strategies face
  a fitness penalty so no single approach monopolises the pool.
"""

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str) -> dict:
    url = BASE + path
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}", "detail": body}
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, payload: dict) -> dict:
    url = BASE + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}", "detail": body}
    except Exception as e:
        return {"error": str(e)}


def _upload(path: str, file_path: str) -> dict:
    """Multipart file upload using only stdlib."""
    url = BASE + path
    boundary = "----NexusSmokeTestBoundary"
    filename = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        file_data = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/plain\r\n\r\n"
    ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body_err = e.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {e.code}", "detail": body_err}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run() -> None:
    pass_count = 0
    fail_count = 0

    def ok(label: str, detail: str = "") -> None:
        nonlocal pass_count
        pass_count += 1
        print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))

    def fail(label: str, detail: str = "") -> None:
        nonlocal fail_count
        fail_count += 1
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))

    print("\n" + "="*60)
    print("  NEXUS CORE SMOKE TEST")
    print("="*60)

    # ------------------------------------------------------------------
    # 1. Health check
    # ------------------------------------------------------------------
    print("\n[1/6] Health check ...")
    r = _get("/health")
    if r.get("status") == "ok":
        ok("Server responded", f"provider={r.get('llm_provider')}")
    else:
        fail("Server not responding", str(r))
        print("\nThe API does not appear to be running.")
        print("Start it with:  uvicorn nexus_core_api:app --host 0.0.0.0 --port 8000")
        print("(from the Nexus Core directory, in a separate terminal)")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Upload test document
    # ------------------------------------------------------------------
    print("\n[2/6] Uploading test document ...")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(_TEST_DOC)
        tmp_path = tf.name

    try:
        r = _upload("/upload", tmp_path)
    finally:
        os.unlink(tmp_path)

    if r.get("error"):
        fail("Upload failed", str(r))
    else:
        chunks = r.get("ingestion", {}).get("chunks_created", "?")
        ok("Document uploaded and indexed", f"{chunks} chunks created")

    # ------------------------------------------------------------------
    # 3. Knowledge-base stats
    # ------------------------------------------------------------------
    print("\n[3/6] Checking knowledge base ...")
    r = _get("/knowledge-base")
    if r.get("error"):
        fail("KB stats failed", str(r))
    else:
        total = r.get("total_chunks", 0)
        if total > 0:
            ok("KB has content", f"{total} chunks total, {r.get('ingested_files', '?')} files")
        else:
            fail("KB appears empty after upload", str(r))

    # ------------------------------------------------------------------
    # 4. Query — full pipeline
    # ------------------------------------------------------------------
    print("\n[4/6] Running a query (this calls Ollama — may take ~30s) ...")
    r = _post("/query", {
        "query": "What is the Research Swarm and how does it evolve?",
        "use_recursive": False,
        "use_agents": False,
    })

    if r.get("error"):
        fail("Query failed", str(r))
    else:
        output = r.get("output", "")
        sources = r.get("sources", [])
        genome_id = r.get("genome_id", "")
        method = r.get("processing", {}).get("method", "?")
        elapsed = r.get("processing", {}).get("time_seconds", 0)

        if output and not output.startswith("[Stub"):
            ok("Got real LLM response",
               f"{len(output)} chars, {len(sources)} sources, {elapsed:.1f}s")
        elif output.startswith("[Stub"):
            fail("Got stub response — LLM not active",
                 "Check .env LLM_PROVIDER and that Ollama is running")
        else:
            fail("Empty response")

        if sources:
            ok("Sources retrieved from KB",
               f"top score={sources[0].get('score', 0):.3f}")
        else:
            fail("No KB sources in response — swarm search may have failed")

        print(f"\n  Response preview:\n  {output[:300].replace(chr(10), chr(10)+'  ')}")

    # ------------------------------------------------------------------
    # 5. Swarm stats — confirm personas ran
    # ------------------------------------------------------------------
    print("\n[5/6] Checking swarm stats ...")
    r = _get("/swarm")
    if r.get("error"):
        fail("Swarm stats failed", str(r))
    else:
        active = r.get("active_personas", [])
        total_searches = sum(p.get("search_count", 0) for p in active)
        if total_searches > 0:
            ok("Personas accumulated search counts",
               f"{total_searches} total searches across {len(active)} active personas")
            for p in active[:3]:
                print(f"    {p['name']:<28} fitness={p['fitness']:.4f}  searches={p['search_count']}")
        else:
            fail("No search counts on personas — swarm may not have run")

    # ------------------------------------------------------------------
    # 6. Full system status
    # ------------------------------------------------------------------
    print("\n[6/6] System status ...")
    r = _get("/status")
    if r.get("error"):
        fail("Status endpoint failed", str(r))
    else:
        mem = r.get("memory", {})
        stm = mem.get("short_term", {}).get("count", "?")
        ok("System status returned",
           f"STM={stm}, interactions={r.get('interactions', '?')}, "
           f"KB_chunks={r.get('knowledge_base', {}).get('total_chunks', '?')}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    total = pass_count + fail_count
    print(f"  {pass_count}/{total} checks passed  |  {fail_count} failed")
    print("="*60 + "\n")

    if fail_count:
        sys.exit(1)


if __name__ == "__main__":
    run()

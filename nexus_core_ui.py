"""
The Nexus Core - Streamlit UI

Run with:
    streamlit run nexus_core_ui.py

Requires the FastAPI server to be running:
    uvicorn nexus_core_api:app --reload --host 0.0.0.0 --port 8000
"""

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="The Nexus Core",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar - API connection
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Nexus Core")
    st.markdown("---")

    api_url = st.text_input(
        "API URL",
        value="http://localhost:8000",
        help="URL of the running Nexus Core API server",
    ).rstrip("/")

    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            st.success("Connected")
            st.caption(f"Provider: {data.get('llm_provider', 'unknown')}")
        else:
            st.error(f"API error {r.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API")
        st.caption("Start the server:\n    uvicorn nexus_core_api:app --reload")
    except Exception as exc:
        st.warning(f"Connection issue: {exc}")

    st.markdown("---")
    gate_mode = st.toggle(
        "3-Stage Gate Mode",
        value=False,
        help=(
            "Gate 1: review retrieved chunks before synthesis. "
            "Gate 2: review draft before acceptance. "
            "Gate 3: rate the response (fitness signal for NEAT)."
        ),
    )
    st.markdown("---")
    st.caption("Tabs: Chat - Upload - Personas - Bookmarks - Status")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def api(method: str, path: str, **kwargs):
    url = f"{api_url}{path}"
    try:
        r = getattr(requests, method)(url, timeout=60, **kwargs)
        if r.status_code < 300:
            return True, r.json()
        return False, {"error": r.json().get("detail", r.text)}
    except requests.exceptions.ConnectionError:
        return False, {"error": "Cannot reach API server"}
    except Exception as exc:
        return False, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_upload, tab_personas, tab_bookmarks, tab_status = st.tabs(
    ["Chat", "Upload", "Personas", "Bookmarks", "Status"]
)


# ============================================================
# TAB 1: CHAT
# ============================================================

with tab_chat:
    st.header("Chat")

    col_opts1, col_opts2, col_opts3 = st.columns(3)
    with col_opts1:
        use_recursive = st.toggle(
            "Recursive refinement", value=False,
            help="Iterative self-reflection loop. Slower but higher quality.",
        )
    with col_opts2:
        use_agents = st.toggle(
            "Multi-agent mode", value=False,
            help="Decomposes complex tasks into sub-agents.",
        )
    with col_opts3:
        ok, personas_data = api("get", "/personas")
        persona_options = {"None": None}
        if ok:
            for p in personas_data.get("personas", []):
                persona_options[f"{p['name']} ({p['role']})"] = p["persona_id"]
        persona_label = st.selectbox("Active persona", list(persona_options.keys()))
        selected_persona_id = persona_options[persona_label]

    st.markdown("---")

    # ---- Session state initialisation ----------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for key, default in [
        ("gate_stage", "idle"),
        ("gate_query", ""),
        ("gate_chunks", []),
        ("gate_task_type", ""),
        ("gate_use_recursive", False),
        ("gate_use_agents", False),
        ("gate_persona_id", None),
        ("gate_response", {}),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # If gate mode is turned OFF mid-session, reset any in-flight gate
    if not gate_mode and st.session_state.gate_stage != "idle":
        st.session_state.gate_stage = "idle"

    # ---- Message history -------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"Sources ({len(msg['sources'])} chunks)", expanded=False):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['source_file']}** — chunk {s['chunk_index'] + 1}/{s['total_chunks']} "
                            f"(score: {s['score']:.2f})"
                        )
                        st.caption(s["text_preview"])
                        st.divider()
            if "meta" in msg:
                with st.expander("Metadata", expanded=False):
                    m = msg["meta"]
                    cols = st.columns(4)
                    cols[0].metric("Method", m.get("processing", {}).get("method", "-"))
                    cols[1].metric("Time (s)", f"{m.get('processing', {}).get('time_seconds', 0):.2f}")
                    cols[2].metric("Task type", m.get("routing", {}).get("task_type", "-"))
                    cols[3].metric("Memories used", m.get("memory", {}).get("relevant_memories", 0))

    # ---- Gate 1: Review Retrieved Chunks ---------------------------------------
    if st.session_state.gate_stage == "gate1":
        st.divider()
        st.subheader("Gate 1 — Review Retrieved Chunks")
        st.caption(
            f"Query: *{st.session_state.gate_query}* | Task type: `{st.session_state.gate_task_type}`"
        )
        chunks = st.session_state.gate_chunks
        if not chunks:
            st.info(
                "No matching chunks found in the knowledge base. "
                "The response will draw on general knowledge only."
            )
        else:
            for i, c in enumerate(chunks):
                with st.expander(
                    f"Chunk {i + 1}: **{c['source_file']}** — "
                    f"{c['chunk_index'] + 1}/{c['total_chunks']} (score: {c['score']:.2f})",
                    expanded=(i == 0),
                ):
                    st.markdown(c["text"])

        st.markdown("")
        col_approve, col_discard = st.columns(2)
        with col_approve:
            if st.button("Approve & Synthesize", type="primary", use_container_width=True):
                with st.spinner("Synthesizing response..."):
                    ok2, result = api("post", "/synthesize", json={
                        "query": st.session_state.gate_query,
                        "approved_chunks": st.session_state.gate_chunks,
                        "use_recursive": st.session_state.gate_use_recursive,
                        "use_agents": st.session_state.gate_use_agents,
                        "persona_id": st.session_state.gate_persona_id,
                    })
                if ok2:
                    st.session_state.gate_response = result
                    st.session_state.gate_stage = "gate2"
                    st.rerun()
                else:
                    st.error(result.get("error", "Synthesis failed"))
        with col_discard:
            if st.button("Discard & Start Over", use_container_width=True):
                st.session_state.gate_stage = "idle"
                st.rerun()

    # ---- Gate 2: Review Draft Response -----------------------------------------
    elif st.session_state.gate_stage == "gate2":
        st.divider()
        st.subheader("Gate 2 — Review Draft Response")
        draft = st.session_state.gate_response
        draft_text = draft.get("output", "")

        with st.container(border=True):
            st.markdown(draft_text)
            sources = draft.get("sources", [])
            if sources:
                with st.expander(f"Sources used ({len(sources)} chunks)", expanded=False):
                    for s in sources:
                        st.markdown(
                            f"**{s['source_file']}** — chunk {s['chunk_index'] + 1}/{s['total_chunks']} "
                            f"(score: {s['score']:.2f})"
                        )
                        st.caption(s["text_preview"])
                        st.divider()

        st.markdown("")
        col_accept, col_regen, col_discard2 = st.columns(3)
        with col_accept:
            if st.button("Accept Response", type="primary", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": draft_text,
                    "sources": draft.get("sources", []),
                    "meta": draft,
                })
                st.session_state.gate_stage = "rating"
                st.rerun()
        with col_regen:
            if st.button("Regenerate", use_container_width=True):
                with st.spinner("Regenerating..."):
                    ok3, result2 = api("post", "/synthesize", json={
                        "query": st.session_state.gate_query,
                        "approved_chunks": st.session_state.gate_chunks,
                        "use_recursive": st.session_state.gate_use_recursive,
                        "use_agents": st.session_state.gate_use_agents,
                        "persona_id": st.session_state.gate_persona_id,
                    })
                if ok3:
                    st.session_state.gate_response = result2
                    st.rerun()
                else:
                    st.error(result2.get("error", "Regeneration failed"))
        with col_discard2:
            if st.button("Discard", use_container_width=True):
                st.session_state.messages.pop()  # remove the user message added at gate1
                st.session_state.gate_stage = "idle"
                st.rerun()

    # ---- Gate 3: Rate the Response ---------------------------------------------
    elif st.session_state.gate_stage == "rating":
        st.divider()
        st.subheader("Gate 3 — Rate this response")
        st.caption("Your rating is stored as a fitness signal for NEAT evolution.")
        draft = st.session_state.gate_response
        col_up, col_down, col_skip = st.columns(3)
        with col_up:
            if st.button("Thumbs Up", type="primary", use_container_width=True):
                api("post", "/feedback", json={
                    "query": st.session_state.gate_query,
                    "response": draft.get("output", ""),
                    "rating": 1,
                    "sources": [s["source_file"] for s in draft.get("sources", [])],
                    "processing_method": draft.get("processing", {}).get("method"),
                    "session_id": draft.get("session_id"),
                })
                st.session_state.gate_stage = "idle"
                st.rerun()
        with col_down:
            if st.button("Thumbs Down", use_container_width=True):
                api("post", "/feedback", json={
                    "query": st.session_state.gate_query,
                    "response": draft.get("output", ""),
                    "rating": -1,
                    "sources": [s["source_file"] for s in draft.get("sources", [])],
                    "processing_method": draft.get("processing", {}).get("method"),
                    "session_id": draft.get("session_id"),
                })
                st.session_state.gate_stage = "idle"
                st.rerun()
        with col_skip:
            if st.button("Skip Rating", use_container_width=True):
                st.session_state.gate_stage = "idle"
                st.rerun()

    # ---- Idle: Normal or Gate-Mode Query Input ---------------------------------
    else:
        if gate_mode:
            # Gate mode input — visible form instead of floating chat input
            with st.form("gate_query_form", clear_on_submit=True):
                gate_q = st.text_area(
                    "Ask anything (Gate Mode):",
                    height=80,
                    placeholder="Type your question here...",
                )
                col_btn, col_rec, col_ag = st.columns([3, 1, 1])
                submitted = col_btn.form_submit_button("Retrieve Chunks", type="primary")
                gate_use_recursive_input = col_rec.checkbox("Recursive", value=use_recursive)
                gate_use_agents_input = col_ag.checkbox("Agents", value=use_agents)

            if submitted and gate_q.strip():
                with st.spinner("Retrieving chunks..."):
                    ok_r, r_data = api("post", "/retrieve", json={"query": gate_q.strip()})
                if ok_r:
                    st.session_state.gate_query = gate_q.strip()
                    st.session_state.gate_chunks = r_data.get("chunks", [])
                    st.session_state.gate_task_type = r_data.get("task_type", "")
                    st.session_state.gate_use_recursive = gate_use_recursive_input
                    st.session_state.gate_use_agents = gate_use_agents_input
                    st.session_state.gate_persona_id = selected_persona_id
                    st.session_state.messages.append({"role": "user", "content": gate_q.strip()})
                    st.session_state.gate_stage = "gate1"
                    st.rerun()
                else:
                    st.error(r_data.get("error", "Retrieval failed"))
        else:
            # Normal chat input (unchanged from original behaviour)
            query = st.chat_input("Ask anything...")
            if query:
                st.session_state.messages.append({"role": "user", "content": query})
                with st.chat_message("user"):
                    st.markdown(query)

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        ok, result = api("post", "/query", json={
                            "query": query,
                            "use_recursive": use_recursive,
                            "use_agents": use_agents,
                            "persona_id": selected_persona_id,
                        })

                    if ok:
                        output = result.get("output", "")
                        st.markdown(output)

                        sources = result.get("sources", [])
                        if sources:
                            with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
                                for s in sources:
                                    st.markdown(
                                        f"**{s['source_file']}** — chunk {s['chunk_index'] + 1}/{s['total_chunks']} "
                                        f"(score: {s['score']:.2f})"
                                    )
                                    st.caption(s["text_preview"])
                                    st.divider()

                        with st.expander("Metadata", expanded=False):
                            cols = st.columns(4)
                            cols[0].metric("Method", result.get("processing", {}).get("method", "-"))
                            cols[1].metric("Time (s)", f"{result.get('processing', {}).get('time_seconds', 0):.2f}")
                            cols[2].metric("Task type", result.get("routing", {}).get("task_type", "-"))
                            cols[3].metric("Memories used", result.get("memory", {}).get("relevant_memories", 0))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": output,
                            "sources": result.get("sources", []),
                            "meta": result,
                        })
                    else:
                        err = result.get("error", "Unknown error")
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": f"Error: {err}"})

    if st.session_state.messages and st.session_state.gate_stage == "idle":
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# TAB 2: UPLOAD
# ============================================================

with tab_upload:
    st.header("Upload Documents")
    st.markdown(
        "Upload files to the knowledge base. "
        "Accepted formats: `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.json`, and more."
    )
    st.info(
        "Files are currently **staged** in `nexus_data/uploads/`. "
        "Full ingestion pipeline is in development."
    )

    uploaded = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        help="Files are saved to nexus_data/uploads/ on the server.",
    )

    if uploaded:
        if st.button("Upload to server", type="primary"):
            for f in uploaded:
                ok, data = api(
                    "post", "/upload",
                    files={"file": (f.name, f.getvalue(), f.type or "application/octet-stream")},
                )
                if ok:
                    ing = data.get("ingestion", {})
                    status = ing.get("status", "?")
                    chunks = ing.get("chunks_created", 0)
                    msg = ing.get("message", "")
                    if status == "ok":
                        st.success(f"**{f.name}** — {chunks} chunks ingested. {msg}")
                    elif status == "skipped":
                        st.info(f"**{f.name}** — Skipped: {msg}")
                    else:
                        st.warning(f"**{f.name}** — {msg}")
                else:
                    st.error(f"**{f.name}** failed: {data.get('error', 'Upload failed')}")


# ============================================================
# TAB 3: PERSONAS
# ============================================================

with tab_personas:
    st.header("Personas")

    col_list, col_set = st.columns([2, 1])

    with col_list:
        st.subheader("Available personas")
        ok, data = api("get", "/personas")
        if ok:
            personas = data.get("personas", [])
            if not personas:
                st.caption("No personas yet. Create one using the panel on the right.")
            for p in personas:
                active_badge = " **[ACTIVE]**" if p.get("active") else ""
                st.markdown(
                    f"**{p['name']}**{active_badge}  \n"
                    f"`{p['persona_id']}` - {p['role']}  \n"
                    f"Style: {p.get('communication_style', '-')} - "
                    f"Interactions: {p.get('interaction_count', 0)}"
                )
                st.markdown("---")
        else:
            st.error(data.get("error"))

    with col_set:
        st.subheader("Set / Create persona")
        mode = st.radio("Mode", ["By ID", "From template"], horizontal=True)

        if mode == "By ID":
            pid = st.text_input("Persona ID")
            if st.button("Activate", type="primary", key="activate_by_id"):
                if pid:
                    ok, result = api("post", "/persona", json={"persona_id": pid})
                    if ok:
                        st.success(f"Activated **{result['name']}**")
                        st.rerun()
                    else:
                        st.error(result.get("error"))
                else:
                    st.warning("Enter a persona ID.")
        else:
            template = st.selectbox(
                "Template",
                ["expert", "companion", "analyst", "creative", "teacher"],
            )
            if st.button("Generate and activate", type="primary", key="activate_template"):
                ok, result = api("post", "/persona", json={"template": template})
                if ok:
                    st.success(f"Created and activated **{result['name']}**")
                    st.rerun()
                else:
                    st.error(result.get("error"))


# ============================================================
# TAB 4: BOOKMARKS
# ============================================================

with tab_bookmarks:
    st.header("Bookmarks")

    col_create, col_search = st.columns([1, 1])

    with col_create:
        st.subheader("Create bookmark")
        bm_content = st.text_area("Content", height=120, placeholder="Information to remember...")
        bm_title = st.text_input("Title")
        bm_tags = st.text_input("Tags (comma-separated)", placeholder="AI, research, notes")
        bm_importance = st.slider("Importance", 0.0, 1.0, 0.9, 0.05)

        if st.button("Save bookmark", type="primary"):
            if bm_content and bm_title:
                tags = [t.strip() for t in bm_tags.split(",") if t.strip()]
                ok, result = api("post", "/bookmarks", json={
                    "content": bm_content,
                    "title": bm_title,
                    "tags": tags,
                    "importance": bm_importance,
                })
                if ok:
                    st.success(f"Saved - memory ID: `{result['memory_id']}`")
                else:
                    st.error(result.get("error"))
            else:
                st.warning("Content and title are required.")

    with col_search:
        st.subheader("Search bookmarks")
        search_q = st.text_input("Search query (leave blank for all)")
        if st.button("Search", key="bm_search"):
            path = "/bookmarks"
            if search_q:
                path += f"?q={requests.utils.quote(search_q)}"
            ok, data = api("get", path)
            if ok:
                bms = data.get("bookmarks", [])
                if not bms:
                    st.caption("No bookmarks found.")
                for bm in bms:
                    with st.expander(
                        f"{bm.get('memory_id', '?')} (importance {bm.get('importance', 0):.2f})"
                    ):
                        st.markdown(bm.get("content", ""))
                        st.caption(
                            f"Tags: {', '.join(bm.get('tags', []))}  |  "
                            f"Created: {bm.get('created_at', '-')}"
                        )
            else:
                st.error(data.get("error"))


# ============================================================
# TAB 5: STATUS
# ============================================================

with tab_status:
    st.header("System Status")

    if st.button("Refresh", type="primary"):
        st.rerun()

    ok, data = api("get", "/status")
    if ok:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Session ID", data.get("session_id", "-")[-8:])
        col2.metric("Interactions", data.get("interactions", 0))
        col3.metric("Active persona", data.get("current_persona") or "None")
        col4.metric("Total personas", data.get("personas", 0))

        st.markdown("---")

        mem = data.get("memory", {})
        st.subheader("Memory")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            stm = mem.get("short_term", {})
            st.markdown("**Short-term memory**")
            st.metric("Items", stm.get("count", 0))
            st.metric("Capacity", stm.get("capacity", 0))
        with col_m2:
            ltm = mem.get("long_term", {})
            st.markdown("**Long-term memory**")
            st.metric("Total memories", ltm.get("total_memories", 0))
            st.metric("Bookmarks", ltm.get("bookmark_count", ltm.get("total_memories", 0)))

        st.markdown("---")

        routing = data.get("routing", {})
        st.subheader("Query routing")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Total decisions", routing.get("total_decisions", 0))
        col_r2.metric("Avg confidence", f"{routing.get('average_confidence', 0):.0%}")
        task_dist = routing.get("task_type_distribution", {})
        if task_dist:
            col_r3.dataframe(
                {"Task type": list(task_dist.keys()), "Count": list(task_dist.values())},
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("---")

        agents = data.get("agents", {})
        st.subheader("Agent system")
        col_a1, col_a2, col_a3 = st.columns(3)
        col_a1.metric("Total agents", agents.get("total_agents", 0))
        col_a2.metric("Tasks completed", agents.get("tasks_completed", 0))
        col_a3.metric("Tasks failed", agents.get("tasks_failed", 0))

        st.markdown("---")

        kb = data.get("knowledge_base", {})
        st.subheader("Knowledge base")
        col_k1, col_k2, col_k3 = st.columns(3)
        col_k1.metric("Total chunks", kb.get("total_chunks", 0))
        col_k2.metric("Ingested files", kb.get("ingested_files", 0))
        col_k3.metric("Search mode", kb.get("search_mode", "keyword"))

        st.markdown("---")

        st.subheader("Session management")
        if st.button(
            "Start new session",
            help="Clears short-term memory and resets session ID. Long-term memory is preserved.",
        ):
            ok2, result = api("post", "/sessions/new")
            if ok2:
                st.success(f"New session: {result.get('session_id')}")
                st.rerun()
            else:
                st.error(result.get("error"))
    else:
        st.error(f"Could not fetch status: {data.get('error')}")

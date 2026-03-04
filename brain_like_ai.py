"""
The Nexus Core - Brain-Like AI Integration
Unified system combining recursive processing, meta-agents, personas, routing, and layered memory.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json
import threading

# Import all the subsystems
from recursive_language_model import RecursiveLanguageModel, ReasoningChainBuilder
from meta_agent_system import MetaAgentCoordinator, AgentRole, AgentTask
from chargen_system import CharGenSystem, Persona
from llm_router import LLMRouter, TaskType
from layered_memory_system import LayeredMemorySystem, MemoryItem

# Config and LLM adapter
from nexus_core_config import config as _config
from nexus_core_llm_adapters import llm as _llm
from nexus_core_ingestion import search_knowledge_base as _search_kb, get_knowledge_base_stats as _kb_stats
from nexus_core_genome import GenomeStore as _GenomeStore
from nexus_core_hirag import HiRAGMemory as _HiRAGMemory
from nexus_core_research_swarm import ResearchSwarm as _ResearchSwarm

# System prompts for each agent role
_AGENT_SYSTEM_PROMPTS = {
    AgentRole.RESEARCHER: (
        "You are a specialized research agent. Your job is to gather and organize "
        "all relevant information about the given task. Be thorough, factual, and "
        "cite key details clearly."
    ),
    AgentRole.ANALYZER: (
        "You are a specialized analysis agent. Analyze the provided information, "
        "identify patterns, extract key insights, and draw well-reasoned conclusions."
    ),
    AgentRole.WRITER: (
        "You are a specialized writing agent. Generate a clear, well-structured, "
        "and comprehensive response based on the task description."
    ),
    AgentRole.CRITIC: (
        "You are a specialized critic agent. Review the output critically. "
        "Identify weaknesses, errors, or gaps and suggest concrete improvements."
    ),
}

# ---------------------------------------------------------------------------
# General-purpose warmup queries (domain-agnostic)
#
# Mixed into the warmup seed pool alongside KB-derived queries to provide
# query variety.  Note: because all queries are evaluated against the same
# KB, general queries that have no matching KB content result in zero hits
# for ALL personas equally — so they do not directly select for generalist
# strategies.  Anti-specialisation pressure is instead enforced by the
# evolution mechanism in ResearchSwarm._run_competition(), which applies a
# fitness penalty to over-represented strategy types and periodically
# re-injects missing strategies regardless of fitness.  These general queries
# still serve a useful signal when the KB *does* contain relevant content
# (e.g. a methodology or concept document), giving personas that reformulate
# broadly a chance to surface chunks they might otherwise miss.
# ---------------------------------------------------------------------------
_GENERAL_WARMUP_QUERIES: List[str] = [
    # Conceptual / definitional
    "what is the difference between correlation and causation",
    "explain the concept of emergence in complex systems",
    "what is entropy and why does it matter",
    "define heuristic and give an example",
    "what is the Pareto principle",
    # How-to / practical
    "how do you debug a problem you do not fully understand",
    "how to structure an argument clearly",
    "how does peer review work",
    "how to evaluate the credibility of a source",
    "how do you break a large problem into smaller parts",
    # Comparative
    "compare deductive and inductive reasoning",
    "difference between efficiency and effectiveness",
    "analogy versus metaphor what is the distinction",
    "what are the trade-offs between speed and accuracy",
    "compare short-term and long-term thinking",
    # Broad knowledge
    "what causes economic inflation",
    "how does the scientific method work",
    "what is the role of feedback in learning",
    "why do systems fail unexpectedly",
    "what are the stages of a typical project lifecycle",
    # Critical / skeptical
    "what are common logical fallacies",
    "how can data be misleading",
    "what are the limits of expert opinion",
    "when is simplicity better than complexity",
    "what can go wrong with prediction models",
    # Creative / open-ended
    "describe an unexpected use for a common tool",
    "what would an ideal knowledge management system look like",
    "how might this topic look different in twenty years",
    "what questions are still unanswered in this domain",
    "what does success look like and how would you measure it",
]

# Import existing Nexus Core components
try:
    from nexus_core_engine import NexusCoreEngine
    from nexus_core_indexing import HierarchicalIndexManager
    from nexus_core_enhancements import (
        CitationManager,
        RelevanceRanker,
        QueryExpander
    )
    NEXUS_CORE_AVAILABLE = True
except ImportError:
    NEXUS_CORE_AVAILABLE = False


class BrainLikeAI:
    """
    Brain-like AI system that integrates:
    - Recursive Language Model (iterative refinement)
    - Meta-Agent System (task decomposition and coordination)
    - CharGen System (persona-based interactions)
    - LLM Router (intelligent query routing)
    - Layered Memory (short-term and long-term memory)
    - Existing RAG components (indexing, search, enhancements)
    """
    
    def __init__(self, base_path: str = None):
        """
        Initialize the brain-like AI system.

        Args:
            base_path: Base directory for all data storage.
                       Defaults to the path set in config (NEXUS_DATA_PATH).
        """
        if base_path is None:
            base_path = _config.nexus_data_path
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize all subsystems
        print("[BrainLikeAI] Initializing Brain-Like AI System...")

        # Core cognitive components
        self.recursive_model = RecursiveLanguageModel(
            max_depth=_config.recursive_max_depth,
            reflection_threshold=_config.recursive_reflection_threshold
        )
        self.reasoning_chain = ReasoningChainBuilder()

        # Agent coordination
        self.meta_agents = MetaAgentCoordinator(llm_fn=lambda p: _llm.complete(p))

        # Wire LLM processors to each agent role
        for agent in self.meta_agents.agents.values():
            if agent.role in _AGENT_SYSTEM_PROMPTS:
                _sys_prompt = _AGENT_SYSTEM_PROMPTS[agent.role]
                _role_name = agent.role.value

                def _make_processor(sp: str, role: str):
                    def processor(input_data: dict) -> dict:
                        task_text = input_data.get("main_task", str(input_data))
                        response = _llm.complete(task_text, system_prompt=sp)
                        return {"output": response, "processed": True, "role": role}
                    return processor

                agent.processor = _make_processor(_sys_prompt, _role_name)

        # Persona management
        self.chargen = CharGenSystem(str(self.base_path / "personas"))

        # Query routing
        self.router = LLMRouter()

        # Memory systems
        self.memory = LayeredMemorySystem(
            stm_capacity=_config.memory_stm_capacity,
            stm_retention_minutes=_config.memory_stm_retention_minutes,
            ltm_storage_path=str(self.base_path / "long_term_memory")
        )
        
        # RAG components (if available)
        if NEXUS_CORE_AVAILABLE:
            self.rag_engine = NexusCoreEngine(str(self.base_path))
            self.indexing = HierarchicalIndexManager(str(self.base_path / "indices"))
            self.citations = CitationManager()
            self.ranker = RelevanceRanker()
            self.query_expander = QueryExpander(llm_fn=lambda p: _llm.complete(p))
        else:
            self.rag_engine = None
            self.indexing = None
            self.citations = None
            self.ranker = None
            self.query_expander = None
        
        # System state
        self.current_persona: Optional[Persona] = None
        self.session_id: str = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.interaction_count: int = 0

        # NEAT genome store
        self.genome_store = _GenomeStore(data_dir=str(self.base_path))

        # HiRAG hierarchical memory
        self.hirag = _HiRAGMemory(
            data_dir=str(self.base_path),
            summarize_fn=lambda p: _llm.complete(p),
        )

        # Research swarm — multi-perspective KB search with continuous evolution
        self.swarm = _ResearchSwarm(data_dir=str(self.base_path))

        # Warmup threading controls
        self._warmup_thread:     Optional[threading.Thread] = None
        self._warmup_stop_event: threading.Event            = threading.Event()

        print("[BrainLikeAI] Brain-Like AI System initialized successfully")
    
    def retrieve_chunks(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Gate 1 — retrieve KB chunks for a query without calling the LLM.
        Returns candidate chunks so the user can approve/reject before synthesis.
        """
        if context is None:
            context = {}
        routing_decision = self.router.route_query(query, context)
        expanded_query = query
        if self.query_expander:
            expanded_query, _ = self.query_expander.expand_query(query)
        try:
            chunks = self.swarm.search(
                expanded_query,
                kb_search_fn=lambda q, k: _search_kb(
                    q, data_dir=str(self.base_path), top_k=k
                ),
                top_k=_config.search_top_k,
            )
        except Exception:
            chunks = []
        return {
            "query": query,
            "expanded_query": expanded_query,
            "task_type": routing_decision.detected_task_type.value,
            "chunks": [
                {
                    "chunk_id": c.get("chunk_id", ""),
                    "source_file": c["source_file"],
                    "source_path": c.get("source_path", ""),
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "score": c.get("score", 0.0),
                    "text": c["text"],
                    "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                }
                for c in chunks
            ],
        }

    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_recursive: bool = True,
        use_agents: bool = False,
        persona_id: Optional[str] = None,
        _preloaded_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Process a query through the brain-like system.
        
        Args:
            query: User query
            context: Optional context information
            use_recursive: Whether to use recursive processing
            use_agents: Whether to use meta-agent decomposition
            persona_id: Optional persona to use for response
        
        Returns:
            Comprehensive response with all metadata
        """
        if context is None:
            context = {}

        start_time = datetime.now()
        self.interaction_count += 1

        # Load the active NEAT genome — genes may override config defaults
        _active_genome = self.genome_store.get_active_genome()

        # Retrieve HiRAG context from all 4 memory layers
        context["hirag_context"] = self.hirag.retrieve(query, top_k=3)

        # Step 1: Route the query to determine optimal processing
        routing_decision = self.router.route_query(query, context)
        
        # Step 2: Store query in short-term memory
        query_memory = self.memory.store(
            content=query,
            memory_type="episodic",
            importance=0.6,
            tags=["query", routing_decision.detected_task_type.value],
            context={"session_id": self.session_id, "routing": routing_decision.selected_model}
        )
        
        # Step 3: Retrieve relevant memories
        relevant_memories = self.memory.retrieve(query, search_both=True)
        context["relevant_memories"] = [
            {"content": m.content, "importance": m.importance}
            for m in relevant_memories[:3]
        ]
        
        # Step 4: Expand query if RAG available
        expanded_query = query
        if self.query_expander:
            expanded_query, expansion_terms = self.query_expander.expand_query(query)
            context["expansion_terms"] = expansion_terms

        # Step 4.5: Search knowledge base for relevant document chunks
        if _preloaded_chunks is not None:
            # User-approved chunks from Gate 1 — skip KB search
            context["retrieved_chunks"] = _preloaded_chunks
        else:
            try:
                retrieved_chunks = self.swarm.search(
                    expanded_query,
                    kb_search_fn=lambda q, k: _search_kb(
                        q, data_dir=str(self.base_path), top_k=k
                    ),
                    # Clamp genome-evolved top_k to a safe range: at least 1,
                    # at most 50, to guard against a mutated gene of 0 or a
                    # runaway value that would saturate memory.
                    top_k=max(1, min(50, int(
                        _active_genome.genes.get("search_top_k", _config.search_top_k)
                    ))),
                )
                context["retrieved_chunks"] = retrieved_chunks
            except Exception:
                context["retrieved_chunks"] = []

        # Step 5: Select or use persona
        if persona_id:
            persona = self.chargen.get_persona(persona_id)
            if persona:
                self.current_persona = persona
                self.chargen.record_interaction(persona_id)
        
        # Adapt persona to context
        persona_behavior = {}
        if self.current_persona:
            persona_behavior = self.chargen.adapt_persona_to_context(
                self.current_persona.persona_id,
                context
            )
            context["persona"] = {
                "name": self.current_persona.name,
                "role": self.current_persona.role,
                "behavior": persona_behavior
            }
        
        # Step 6: Process query based on complexity
        response = None
        processing_method = "direct"
        
        if use_agents and routing_decision.detected_task_type in [
            TaskType.COMPLEX_ANALYSIS,
            TaskType.CODE_GENERATION
        ]:
            # Use meta-agent decomposition for complex tasks
            processing_method = "meta_agents"
            response = self._process_with_agents(query, context)
        
        elif use_recursive:
            # Use recursive processing for refinement
            processing_method = "recursive"
            response = self._process_recursive(query, context)
        
        else:
            # Direct processing
            processing_method = "direct"
            response = self._process_direct(query, context)
        
        # Step 7: Store response in memory
        response_memory = self.memory.store(
            content=response.get("output", ""),
            memory_type="episodic",
            importance=0.7,
            tags=["response", routing_decision.detected_task_type.value],
            context={
                "session_id": self.session_id,
                "query": query[:100],
                "processing_method": processing_method
            }
        )
        
        # Step 8: Log to RAG system if available
        if self.rag_engine:
            self.rag_engine.log_conversation_turn(
                session_id=self.session_id,
                user_message=query,
                assistant_response=response.get("output", ""),
                metadata={
                    "routing": routing_decision.selected_model,
                    "processing_method": processing_method,
                    "persona": self.current_persona.name if self.current_persona else None
                }
            )
        
        # Step 9: Perform memory consolidation periodically (background thread
        # — avoids blocking the user response on every 5th query)
        if self.interaction_count % 5 == 0:
            threading.Thread(
                target=self.memory.consolidate_memories,
                daemon=True,
                name="memory-consolidate",
            ).start()

        # Step 9b: Ingest turn into HiRAG synchronously (order matters for
        # retrieval accuracy), but run the potentially LLM-heavy compression
        # pass in a background thread so it doesn't add latency.
        _output_text = response.get("output", "")
        self.hirag.ingest_turn(query, _output_text, session_id=self.session_id)
        threading.Thread(
            target=self.hirag.maybe_compress,
            daemon=True,
            name="hirag-compress",
        ).start()

        # Step 10: Compile comprehensive response
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "output": response.get("output", ""),
            "query": query,
            "session_id": self.session_id,
            "routing": {
                "task_type": routing_decision.detected_task_type.value,
                "selected_model": routing_decision.selected_model,
                "confidence": routing_decision.confidence
            },
            "processing": {
                "method": processing_method,
                "time_seconds": processing_time,
                "recursive_iterations": response.get("iterations", 0) if use_recursive else 0
            },
            "memory": {
                "relevant_memories": len(relevant_memories),
                "query_memory_id": query_memory.memory_id,
                "response_memory_id": response_memory.memory_id
            },
            "persona": {
                "active": self.current_persona is not None,
                "name": self.current_persona.name if self.current_persona else None,
                "behavior": persona_behavior
            },
            "sources": [
                {
                    "source_file": c["source_file"],
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c["total_chunks"],
                    "score": c.get("score", 0.0),
                    "text_preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                }
                for c in context.get("retrieved_chunks", [])
            ],
            "metadata": response.get("metadata", {}),
            "genome_id": _active_genome.genome_id,
            "hirag": self.hirag.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build a system prompt from the active persona (if any).

        If context contains a 'persona.behavior' dict produced by
        adapt_persona_to_context(), the verbosity/formality/empathy scores
        are translated into concrete style instructions appended to the prompt.
        """
        if not self.current_persona:
            return ""
        p = self.current_persona
        parts = [f"You are {p.name}, {p.role}."]
        if p.backstory:
            parts.append(p.backstory)
        if p.goals:
            parts.append(f"Your goals are: {', '.join(p.goals)}.")
        if p.communication_style:
            parts.append(f"Communication style: {p.communication_style}.")

        # Apply context-adapted behaviour if available
        behavior = (context or {}).get("persona", {}).get("behavior", {})
        if behavior:
            hints: List[str] = []
            verbosity = behavior.get("verbosity")
            formality  = behavior.get("formality")
            empathy    = behavior.get("empathy")
            if verbosity is not None:
                if verbosity < 0.3:
                    hints.append("Be concise.")
                elif verbosity > 0.7:
                    hints.append("Be thorough and detailed.")
            if formality is not None:
                if formality > 0.7:
                    hints.append("Maintain a formal tone.")
                elif formality < 0.3:
                    hints.append("Use a casual, conversational tone.")
            if empathy is not None and empathy > 0.7:
                hints.append("Be empathetic and supportive.")
            if hints:
                parts.append(" ".join(hints))

        return " ".join(parts)

    def _build_rag_query(self, query: str, context: Dict[str, Any]) -> str:
        """
        Prepend HiRAG memory context and KB chunks to the query so the LLM
        can ground its answer in both the user's documents and prior conversation
        history.  Returns the original query unchanged when both are empty.
        """
        sections: List[str] = []

        # HiRAG memory context (all 4 layers)
        hirag_ctx = context.get("hirag_context", [])
        if hirag_ctx:
            hirag_lines = "\n".join(f"  {r['content']}" for r in hirag_ctx)
            sections.append(f"MEMORY CONTEXT:\n{hirag_lines}")

        # Knowledge base chunks
        chunks = context.get("retrieved_chunks", [])
        if chunks:
            kb_parts = []
            for c in chunks:
                header = (
                    f"[Source: {c['source_file']} | "
                    f"chunk {c['chunk_index'] + 1}/{c['total_chunks']}]"
                )
                kb_parts.append(f"{header}\n{c['text']}")
            sections.append(
                "RELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
                + "\n\n".join(kb_parts)
            )

        if not sections:
            return query

        # Ground the LLM in the retrieved sources.  Without this, small models
        # (e.g. llama3.2) often ignore context, hallucinate, or cross-reference
        # data from different records when multiple similar entries are present.
        grounding = (
            "Instructions: Answer using ONLY the sources above. "
            "Identify the record whose subject exactly matches the entity in the question. "
            "Quote its field values directly — do not paraphrase, invent, or borrow values "
            "from other records."
        )
        return "\n\n".join(sections) + "\n\n---\n\n" + grounding + "\n\nQUESTION: " + query

    def _process_direct(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Direct processing - single LLM call."""
        augmented_query = self._build_rag_query(query, context)
        output = _llm.complete(augmented_query, system_prompt=self._build_system_prompt(context), context=context)
        return {"output": output, "metadata": {"method": "direct"}}

    def _process_recursive(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process with recursive refinement through the LLM."""
        augmented_query = self._build_rag_query(query, context)
        result = self.recursive_model.recursive_process(
            augmented_query,
            _llm.as_processor(system_prompt=self._build_system_prompt(context)),
            context
        )

        chain_id = f"chain_{self.session_id}_{self.interaction_count}"
        self.reasoning_chain.add_chain(chain_id, result["reasoning_chain"])

        return {
            "output": result["output"],
            "iterations": result["total_iterations"],
            "termination_reason": result["termination_reason"],
            "reasoning_chain_id": chain_id,
            "metadata": {"method": "recursive"}
        }
    
    def _process_with_agents(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process using meta-agent decomposition."""
        # Decompose task
        subtasks = self.meta_agents.decompose_task(query, context)

        # Execute workflow
        workflow_result = self.meta_agents.execute_workflow(subtasks)

        # Compile user-facing output.
        # The CRITIC agent produces internal quality feedback meant to drive
        # refinement — it must not appear in the response shown to the user.
        # Each successful processor result is a dict with an "output" key;
        # fall back to str() for any non-standard processor shape.
        output_parts = []
        for result in workflow_result["results"]:
            if not result["success"]:
                continue
            agent_result = result.get("result")
            if not agent_result:
                continue
            # Skip critic — internal feedback only
            if isinstance(agent_result, dict):
                if agent_result.get("role") == AgentRole.CRITIC.value:
                    continue
                output_parts.append(agent_result.get("output", str(agent_result)))
            else:
                output_parts.append(str(agent_result))

        output = "\n\n".join(output_parts) if output_parts else "Task processing completed"

        return {
            "output": output,
            "subtasks_completed": workflow_result["completed"],
            "subtasks_failed": workflow_result["failed"],
            "metadata": {"method": "meta_agents"}
        }
    
    def set_persona(self, persona_id: Optional[str] = None, template: Optional[str] = None) -> Optional[Persona]:
        """
        Set the active persona.
        
        Args:
            persona_id: Existing persona ID
            template: Template to create new persona
        
        Returns:
            Active persona
        """
        if persona_id:
            persona = self.chargen.get_persona(persona_id)
            if persona:
                self.current_persona = persona
                return persona
        
        if template:
            persona = self.chargen.generate_persona(
                name=f"{template.title()} Assistant",
                template=template
            )
            self.current_persona = persona
            return persona
        
        return None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        memory_status = self.memory.get_memory_status()
        router_analytics = self.router.get_routing_analytics()
        agent_status = self.meta_agents.get_system_status()

        try:
            kb_stats = _kb_stats(data_dir=str(self.base_path))
        except Exception:
            kb_stats = {"total_chunks": 0, "ingested_files": 0}

        return {
            "session_id": self.session_id,
            "interactions": self.interaction_count,
            "current_persona": self.current_persona.name if self.current_persona else None,
            "memory": memory_status,
            "routing": router_analytics,
            "agents": agent_status,
            "personas": len(self.chargen.list_personas()),
            "knowledge_base": kb_stats,
            "hirag": self.hirag.get_stats(),
            "swarm": self.swarm.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    
    def create_bookmark(
        self,
        content: str,
        title: str,
        tags: List[str],
        importance: float = 0.9
    ) -> MemoryItem:
        """
        Create a bookmarked memory for important information.
        
        Args:
            content: Content to bookmark
            title: Bookmark title
            tags: Tags for categorization
            importance: Importance score (default high for bookmarks)
        
        Returns:
            Created memory item
        """
        # Add bookmark tag
        tags = tags + ["bookmark", title]
        
        # Force to long-term memory
        memory = self.memory.store(
            content=content,
            memory_type="semantic",
            importance=importance,
            tags=tags,
            context={"title": title, "bookmarked": True},
            force_long_term=True
        )
        
        return memory
    
    def search_bookmarks(self, query: Optional[str] = None, tags: Optional[List[str]] = None) -> List[MemoryItem]:
        """Search bookmarked memories."""
        search_tags = ["bookmark"]
        if tags:
            search_tags.extend(tags)
        
        return self.memory.long_term.search(
            query=query,
            tags=search_tags,
            min_importance=0.8,
            top_k=20
        )
    
    def _generate_warmup_queries(self, target_count: int = 30) -> List[str]:
        """
        Build a list of seed queries for the swarm warm-up.

        The returned list blends two pools:

        * **KB-derived queries** (~75 % of target_count) — short phrases built
          from top keywords extracted from KB chunks.  These keep personas
          relevant for the user's actual knowledge domain.

        * **General queries** (~25 % of target_count, minimum 5) — drawn from
          ``_GENERAL_WARMUP_QUERIES``.  These add query variety and can surface
          KB content that topic-specific keywords miss.  They do not directly
          provide anti-specialisation selection pressure (all personas score zero
          on queries with no KB match); that is handled in the evolution layer
          via diversity penalties and strategy revival in _run_competition().

        Falls back to a built-in list of generic KB probes when the KB is empty.
        """
        import random as _random
        from nexus_core_hirag import _top_keywords

        # How many slots to reserve for general queries (at least 5, ~25 %)
        general_share = max(5, target_count // 4)
        kb_share      = target_count - general_share

        # --- KB-derived queries -------------------------------------------
        keywords: List[str] = []
        try:
            sample = _search_kb(
                "information",
                data_dir=str(self.base_path),
                top_k=40,
            )
            texts = [c.get("text", "") for c in sample]
            if texts:
                keywords = _top_keywords(texts, n=25)
        except Exception:
            pass

        kb_queries: List[str] = []
        templates = [
            "{kw}",
            "what is {kw}",
            "{kw} overview",
            "how does {kw} work",
            "examples of {kw}",
        ]
        for kw in keywords:
            for tmpl in templates:
                kb_queries.append(tmpl.format(kw=kw))
                if len(kb_queries) >= kb_share:
                    break
            if len(kb_queries) >= kb_share:
                break

        # Fallback KB probes when the knowledge base is empty / sparse
        if len(kb_queries) < 5:
            kb_queries += [
                "overview of main topics",
                "key concepts and definitions",
                "how does this work",
                "important information summary",
                "technical details and specifications",
                "practical examples and use cases",
                "history and background",
                "current status and updates",
                "analysis and insights",
                "step by step guide",
                "what are the main components",
                "common problems and solutions",
            ]

        kb_queries = kb_queries[:kb_share]

        # --- General queries (anti-specialisation) ------------------------
        general_pool  = list(_GENERAL_WARMUP_QUERIES)
        _random.shuffle(general_pool)
        general_queries = general_pool[:general_share]

        # Pad with extra general queries if the KB pool came up short
        # (happens when KB is sparse and the fallback list < kb_share)
        shortfall = target_count - len(kb_queries) - len(general_queries)
        if shortfall > 0:
            already_chosen = set(general_queries)
            extras = [q for q in general_pool if q not in already_chosen]
            general_queries = general_queries + extras[:shortfall]

        # --- Blend and shuffle -------------------------------------------
        # Deduplicate across both pools (KB templates can overlap with general
        # queries) while preserving KB-first ordering before the shuffle.
        seen: set = set()
        combined: List[str] = []
        for q in kb_queries + general_queries:
            if q not in seen:
                seen.add(q)
                combined.append(q)
        _random.shuffle(combined)
        return combined[:target_count]

    def start_swarm_warmup(
        self,
        max_iterations: int = 50,
        max_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Start the swarm warm-up in a background daemon thread.

        If a warmup session is already running, returns its current status
        without starting a second one.

        Parameters
        ----------
        max_iterations
            Maximum number of search iterations to run (default 50).
        max_seconds
            Maximum wall-clock seconds to run (default 300 = 5 minutes).
        """
        if self.swarm._warmup_state.running:
            return {
                "status": "already_running",
                **self.swarm.get_warmup_status(),
            }

        self._warmup_stop_event.clear()
        queries = self._generate_warmup_queries()

        def _run() -> None:
            self.swarm.run_warmup(
                queries=queries,
                kb_search_fn=lambda q, k: _search_kb(
                    q, data_dir=str(self.base_path), top_k=k
                ),
                max_iterations=max_iterations,
                max_seconds=max_seconds,
                top_k=_config.search_top_k,
                stop_event=self._warmup_stop_event,
            )

        self._warmup_thread = threading.Thread(
            target=_run, daemon=True, name="swarm-warmup"
        )
        self._warmup_thread.start()

        return {"status": "started", **self.swarm.get_warmup_status()}

    def stop_swarm_warmup(self) -> Dict[str, Any]:
        """
        Signal the running warmup to stop after its current iteration.
        Returns immediately; the background thread may not have stopped yet.
        """
        self._warmup_stop_event.set()
        return {"status": "stop_signaled", **self.swarm.get_warmup_status()}

    def get_swarm_warmup_status(self) -> Dict[str, Any]:
        """Return the current warmup session state."""
        return self.swarm.get_warmup_status()

    def export_session(self) -> str:
        status = self.get_system_status()
        
        return json.dumps({
            "session_export": status,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


def demo_brain_like_ai():
    """Demonstration of the brain-like AI system."""
    print("="*60)
    print("BRAIN-LIKE AI SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize system
    brain = BrainLikeAI("./demo_brain_ai")
    
    # Set a persona
    print("\nSetting expert persona...")
    brain.set_persona(template="expert")
    
    # Process some queries
    queries = [
        "What is machine learning?",
        "Explain the concept of neural networks in detail",
        "How do I implement a basic neural network?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        result = brain.process_query(
            query,
            use_recursive=(i == 2),  # Use recursive for query 2
            use_agents=(i == 3)  # Use agents for query 3
        )
        
        print(f"\nResponse:")
        print(f"{result['output'][:200]}...")
        
        print(f"\nMetadata:")
        print(f"   Task Type: {result['routing']['task_type']}")
        print(f"   Processing: {result['processing']['method']}")
        print(f"   Time: {result['processing']['time_seconds']:.3f}s")
        print(f"   Relevant Memories: {result['memory']['relevant_memories']}")
    
    # Create a bookmark
    print(f"\n{'='*60}")
    print("Creating bookmark...")
    bookmark = brain.create_bookmark(
        content="Neural networks are computational models inspired by biological neural networks",
        title="Neural Networks Definition",
        tags=["AI", "machine learning", "neural networks"]
    )
    print(f"   Bookmark created: {bookmark.memory_id}")
    
    # Show system status
    print(f"\n{'='*60}")
    print("System Status:")
    status = brain.get_system_status()
    print(f"   Interactions: {status['interactions']}")
    print(f"   STM Count: {status['memory']['short_term']['count']}")
    print(f"   LTM Count: {status['memory']['long_term']['total_memories']}")
    print(f"   Active Persona: {status['current_persona']}")
    print(f"   Routing Decisions: {status['routing']['total_decisions']}")
    
    print(f"\n{'='*60}")
    print("Demonstration complete!")
    print("="*60)


if __name__ == "__main__":
    demo_brain_like_ai()

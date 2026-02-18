"""
The Nexus Core - Brain-Like AI Integration
Unified system combining recursive processing, meta-agents, personas, routing, and layered memory.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import json

# Import all the subsystems
from recursive_language_model import RecursiveLanguageModel, ReasoningChainBuilder
from meta_agent_system import MetaAgentCoordinator, AgentRole, AgentTask
from chargen_system import CharGenSystem, Persona
from llm_router import LLMRouter, TaskType
from layered_memory_system import LayeredMemorySystem, MemoryItem

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
    
    def __init__(self, base_path: str = "./nexus_data"):
        """
        Initialize the brain-like AI system.
        
        Args:
            base_path: Base directory for all data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize all subsystems
        print("🧠 Initializing Brain-Like AI System...")
        
        # Core cognitive components
        self.recursive_model = RecursiveLanguageModel(
            max_depth=3,
            reflection_threshold=0.8
        )
        self.reasoning_chain = ReasoningChainBuilder()
        
        # Agent coordination
        self.meta_agents = MetaAgentCoordinator()
        
        # Persona management
        self.chargen = CharGenSystem(str(self.base_path / "personas"))
        
        # Query routing
        self.router = LLMRouter()
        
        # Memory systems
        self.memory = LayeredMemorySystem(
            stm_capacity=20,
            stm_retention_minutes=30,
            ltm_storage_path=str(self.base_path / "long_term_memory")
        )
        
        # RAG components (if available)
        if NEXUS_CORE_AVAILABLE:
            self.rag_engine = NexusCoreEngine(str(self.base_path))
            self.indexing = HierarchicalIndexManager(str(self.base_path / "indices"))
            self.citations = CitationManager()
            self.ranker = RelevanceRanker()
            self.query_expander = QueryExpander()
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
        
        print("✅ Brain-Like AI System initialized successfully")
    
    def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        use_recursive: bool = True,
        use_agents: bool = False,
        persona_id: Optional[str] = None
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
        
        # Step 9: Perform memory consolidation periodically
        if self.interaction_count % 5 == 0:
            self.memory.consolidate_memories()
        
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
            "metadata": response.get("metadata", {}),
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_direct(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Direct processing without recursion or agents."""
        # Simple processing for demonstration
        output = f"Processing query: {query}"
        
        # Add persona influence if available
        if self.current_persona:
            output += f"\n\n[Response styled as {self.current_persona.name}, {self.current_persona.role}]"
        
        return {
            "output": output,
            "metadata": {"method": "direct"}
        }
    
    def _process_recursive(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process with recursive refinement."""
        def processor(text: str, ctx: Dict[str, Any]) -> str:
            # Simulate processing with context
            depth = ctx.get("depth", 0)
            if depth == 0:
                return f"Initial response to: {text}"
            else:
                prev = ctx.get("previous_output", "")
                return f"{prev} [refined at depth {depth}]"
        
        result = self.recursive_model.recursive_process(
            query,
            processor,
            context
        )
        
        # Store reasoning chain
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
        
        # Compile results
        output_parts = []
        for result in workflow_result["results"]:
            if result["success"] and result.get("result"):
                output_parts.append(str(result["result"]))
        
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
        
        return {
            "session_id": self.session_id,
            "interactions": self.interaction_count,
            "current_persona": self.current_persona.name if self.current_persona else None,
            "memory": memory_status,
            "routing": router_analytics,
            "agents": agent_status,
            "personas": len(self.chargen.list_personas()),
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
    
    def export_session(self) -> str:
        """Export current session data."""
        status = self.get_system_status()
        
        return json.dumps({
            "session_export": status,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


def demo_brain_like_ai():
    """Demonstration of the brain-like AI system."""
    print("="*60)
    print("🧠 BRAIN-LIKE AI SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize system
    brain = BrainLikeAI("./demo_brain_ai")
    
    # Set a persona
    print("\n👤 Setting expert persona...")
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
        
        print(f"\n📤 Response:")
        print(f"{result['output'][:200]}...")
        
        print(f"\n📊 Metadata:")
        print(f"   Task Type: {result['routing']['task_type']}")
        print(f"   Processing: {result['processing']['method']}")
        print(f"   Time: {result['processing']['time_seconds']:.3f}s")
        print(f"   Relevant Memories: {result['memory']['relevant_memories']}")
    
    # Create a bookmark
    print(f"\n{'='*60}")
    print("🔖 Creating bookmark...")
    bookmark = brain.create_bookmark(
        content="Neural networks are computational models inspired by biological neural networks",
        title="Neural Networks Definition",
        tags=["AI", "machine learning", "neural networks"]
    )
    print(f"   Bookmark created: {bookmark.memory_id}")
    
    # Show system status
    print(f"\n{'='*60}")
    print("📊 System Status:")
    status = brain.get_system_status()
    print(f"   Interactions: {status['interactions']}")
    print(f"   STM Count: {status['memory']['short_term']['count']}")
    print(f"   LTM Count: {status['memory']['long_term']['total_memories']}")
    print(f"   Active Persona: {status['current_persona']}")
    print(f"   Routing Decisions: {status['routing']['total_decisions']}")
    
    print(f"\n{'='*60}")
    print("✅ Demonstration complete!")
    print("="*60)


if __name__ == "__main__":
    demo_brain_like_ai()

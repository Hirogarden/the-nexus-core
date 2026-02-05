#!/usr/bin/env python3
"""
Complete Example: Using the Brain-Like AI System

This example demonstrates all major features of the brain-like AI system:
- Recursive processing
- Multi-agent coordination
- Dynamic personas
- Layered memory
- Intelligent routing
- Bookmarks
"""

from brain_like_ai import BrainLikeAI

def main():
    print("="*70)
    print("🧠 BRAIN-LIKE AI SYSTEM - COMPLETE EXAMPLE")
    print("="*70)
    
    # Initialize the brain-like AI system
    print("\n1️⃣  Initializing Brain-Like AI...")
    brain = BrainLikeAI("./example_brain_data")
    print("   ✅ System initialized")
    
    # Set a persona
    print("\n2️⃣  Setting up an Expert persona...")
    persona = brain.set_persona(template="expert")
    print(f"   ✅ Persona active: {persona.name}")
    print(f"   Role: {persona.role}")
    
    # Example 1: Simple query with direct processing
    print("\n" + "="*70)
    print("📝 EXAMPLE 1: Simple Query (Direct Processing)")
    print("="*70)
    
    result1 = brain.process_query(
        query="What is machine learning?",
        use_recursive=False,
        use_agents=False
    )
    
    print(f"\n✅ Response:")
    print(f"   {result1['output'][:200]}...")
    print(f"\n📊 Metadata:")
    print(f"   - Task Type: {result1['routing']['task_type']}")
    print(f"   - Processing Method: {result1['processing']['method']}")
    print(f"   - Processing Time: {result1['processing']['time_seconds']:.3f}s")
    print(f"   - Relevant Memories: {result1['memory']['relevant_memories']}")
    
    # Example 2: Complex query with recursive processing
    print("\n" + "="*70)
    print("📝 EXAMPLE 2: Complex Analysis (Recursive Processing)")
    print("="*70)
    
    result2 = brain.process_query(
        query="Explain the architecture of transformer models in detail",
        use_recursive=True,  # Enable iterative refinement
        use_agents=False
    )
    
    print(f"\n✅ Response:")
    print(f"   {result2['output'][:200]}...")
    print(f"\n📊 Metadata:")
    print(f"   - Task Type: {result2['routing']['task_type']}")
    print(f"   - Processing Method: {result2['processing']['method']}")
    print(f"   - Recursive Iterations: {result2['processing']['recursive_iterations']}")
    print(f"   - Processing Time: {result2['processing']['time_seconds']:.3f}s")
    
    # Example 3: Very complex task with multi-agent coordination
    print("\n" + "="*70)
    print("📝 EXAMPLE 3: Multi-Step Task (Agent Coordination)")
    print("="*70)
    
    result3 = brain.process_query(
        query="Research recent advances in AI, analyze their impact, and write a summary",
        use_recursive=False,
        use_agents=True  # Enable multi-agent decomposition
    )
    
    print(f"\n✅ Response:")
    print(f"   {result3['output'][:200]}...")
    print(f"\n📊 Metadata:")
    print(f"   - Task Type: {result3['routing']['task_type']}")
    print(f"   - Processing Method: {result3['processing']['method']}")
    print(f"   - Processing Time: {result3['processing']['time_seconds']:.3f}s")
    
    # Example 4: Create bookmarks for important information
    print("\n" + "="*70)
    print("🔖 EXAMPLE 4: Creating Bookmarks")
    print("="*70)
    
    bookmark1 = brain.create_bookmark(
        content="Transformer models use self-attention mechanisms to process sequences in parallel, achieving state-of-the-art results in NLP tasks.",
        title="Transformer Architecture",
        tags=["AI", "transformers", "NLP", "architecture"],
        importance=0.95
    )
    
    bookmark2 = brain.create_bookmark(
        content="BERT uses bidirectional training, GPT uses autoregressive training. Both are based on transformers but have different training objectives.",
        title="BERT vs GPT",
        tags=["AI", "transformers", "BERT", "GPT"],
        importance=0.9
    )
    
    print(f"   ✅ Created bookmark 1: {bookmark1.memory_id}")
    print(f"   ✅ Created bookmark 2: {bookmark2.memory_id}")
    
    # Example 5: Search bookmarks
    print("\n" + "="*70)
    print("🔍 EXAMPLE 5: Searching Bookmarks")
    print("="*70)
    
    bookmarks = brain.search_bookmarks(query="transformer", tags=["AI"])
    print(f"\n   Found {len(bookmarks)} bookmarks:")
    for i, bm in enumerate(bookmarks, 1):
        print(f"\n   {i}. {bm.content[:80]}...")
        print(f"      Importance: {bm.importance:.2f}")
        print(f"      Tags: {', '.join(bm.tags[:5])}")
    
    # Example 6: Switch persona for different context
    print("\n" + "="*70)
    print("👤 EXAMPLE 6: Switching Personas")
    print("="*70)
    
    print("\n   Switching to 'teacher' persona...")
    teacher = brain.set_persona(template="teacher")
    print(f"   ✅ New persona: {teacher.name}")
    
    result4 = brain.process_query(
        query="Explain machine learning to a beginner",
        use_recursive=False,
        use_agents=False
    )
    
    print(f"\n   Response with teacher persona:")
    print(f"   {result4['output'][:200]}...")
    
    # Example 7: Memory system status
    print("\n" + "="*70)
    print("🧠 EXAMPLE 7: Memory System Status")
    print("="*70)
    
    memory_status = brain.memory.get_memory_status()
    
    print(f"\n   Short-Term Memory:")
    print(f"   - Capacity: {memory_status['short_term']['count']}/{int(memory_status['short_term']['capacity'])}")
    print(f"   - Utilization: {memory_status['short_term']['utilization']:.1%}")
    print(f"   - Avg Importance: {memory_status['short_term']['avg_importance']:.2f}")
    
    print(f"\n   Long-Term Memory:")
    print(f"   - Total Memories: {memory_status['long_term']['total_memories']}")
    if memory_status['long_term']['total_memories'] > 0:
        print(f"   - Avg Importance: {memory_status['long_term']['avg_importance']:.2f}")
        print(f"   - Memory Types: {memory_status['long_term']['memory_types']}")
    
    # Example 8: System status
    print("\n" + "="*70)
    print("📊 EXAMPLE 8: Complete System Status")
    print("="*70)
    
    status = brain.get_system_status()
    
    print(f"\n   Session: {status['session_id']}")
    print(f"   Total Interactions: {status['interactions']}")
    print(f"   Current Persona: {status['current_persona']}")
    print(f"   Total Personas: {status['personas']}")
    
    print(f"\n   Routing Analytics:")
    if status['routing']['total_decisions'] > 0:
        print(f"   - Total Decisions: {status['routing']['total_decisions']}")
        print(f"   - Avg Confidence: {status['routing']['avg_confidence']:.2f}")
        print(f"   - Most Used Model: {status['routing'].get('most_used_model', 'N/A')}")
    
    # Example 9: Memory consolidation
    print("\n" + "="*70)
    print("🔄 EXAMPLE 9: Memory Consolidation")
    print("="*70)
    
    print("\n   Consolidating memories from short-term to long-term...")
    consolidation = brain.memory.consolidate_memories()
    print(f"   ✅ Consolidated: {consolidation['consolidated']} memories")
    print(f"   Current STM count: {consolidation['stm_count']}")
    print(f"   Current LTM count: {consolidation['ltm_count']}")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ COMPLETE EXAMPLE FINISHED")
    print("="*70)
    
    print("\n📚 Key Takeaways:")
    print("   1. Direct processing for simple queries")
    print("   2. Recursive processing for quality improvement")
    print("   3. Agent coordination for complex multi-step tasks")
    print("   4. Bookmarks for quick access to important info")
    print("   5. Dynamic personas for context-appropriate responses")
    print("   6. Layered memory with automatic consolidation")
    print("   7. Intelligent routing to optimal processing strategies")
    print("   8. Comprehensive system monitoring")
    
    print("\n🎯 Next Steps:")
    print("   - Explore BRAIN_LIKE_AI.md for detailed documentation")
    print("   - Try different persona templates")
    print("   - Experiment with routing constraints")
    print("   - Configure memory consolidation rules")
    print("   - Integrate with your own applications")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

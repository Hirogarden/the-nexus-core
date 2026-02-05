"""
The Nexus Core - Recursive Language Model Integration
Implements MIT-style recursive reasoning and iterative refinement.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json


class RecursiveLanguageModel:
    """
    Recursive processing system that iteratively refines outputs.
    Inspired by MIT's recursive language model research.
    
    Key features:
    - Self-reflection on generated outputs
    - Iterative refinement through recursive calls
    - Reasoning chain construction
    - Depth control and termination conditions
    """
    
    def __init__(
        self,
        max_depth: int = 3,
        reflection_threshold: float = 0.8,
        enable_reasoning_chains: bool = True
    ):
        """
        Initialize the recursive language model.
        
        Args:
            max_depth: Maximum recursion depth
            reflection_threshold: Quality threshold for termination (0.0-1.0)
            enable_reasoning_chains: Whether to build reasoning chains
        """
        self.max_depth = max_depth
        self.reflection_threshold = reflection_threshold
        self.enable_reasoning_chains = enable_reasoning_chains
        self.processing_history: List[Dict[str, Any]] = []
    
    def recursive_process(
        self,
        input_text: str,
        processor: Callable[[str, Dict[str, Any]], str],
        context: Optional[Dict[str, Any]] = None,
        current_depth: int = 0
    ) -> Dict[str, Any]:
        """
        Recursively process input through reflection and refinement.
        
        Args:
            input_text: Input to process
            processor: Function that takes (text, context) and returns processed text
            context: Additional context for processing
            current_depth: Current recursion depth
        
        Returns:
            Dictionary with final output, reasoning chain, and metadata
        """
        if context is None:
            context = {}
        
        # Check termination conditions
        if current_depth >= self.max_depth:
            return self._finalize_result(input_text, "max_depth_reached", current_depth)
        
        # Process the input
        timestamp = datetime.now().isoformat()
        output = processor(input_text, context)
        
        # Record processing step
        step = {
            "depth": current_depth,
            "timestamp": timestamp,
            "input": input_text[:200],  # Truncate for storage
            "output": output[:200],
            "context": context
        }
        self.processing_history.append(step)
        
        # Self-reflection: evaluate the output quality
        quality_score = self._evaluate_output_quality(output, input_text, context)
        
        # Check if quality threshold met
        if quality_score >= self.reflection_threshold:
            return self._finalize_result(
                output, 
                "quality_threshold_met",
                current_depth,
                quality_score
            )
        
        # Build reflection context for next iteration
        reflection_context = {
            **context,
            "previous_output": output,
            "quality_score": quality_score,
            "depth": current_depth,
            "improvement_needed": self._identify_improvements(output, input_text)
        }
        
        # Recursive call with refined understanding
        refined_input = self._create_refinement_prompt(output, reflection_context)
        return self.recursive_process(
            refined_input,
            processor,
            reflection_context,
            current_depth + 1
        )
    
    def _evaluate_output_quality(
        self,
        output: str,
        input_text: str,
        context: Dict[str, Any]
    ) -> float:
        """
        Evaluate the quality of generated output.
        
        Returns score between 0.0 and 1.0
        """
        score = 1.0
        
        # Length check
        if len(output) < 10:
            score -= 0.3
        
        # Coherence check (basic heuristics)
        sentences = output.split('.')
        if len(sentences) < 2:
            score -= 0.2
        
        # Repetition check - minimum unique word ratio for quality content
        MIN_UNIQUE_WORD_RATIO = 0.6
        words = output.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < MIN_UNIQUE_WORD_RATIO:
            score -= 0.2
        
        # Relevance to input (keyword overlap)
        input_words = set(input_text.lower().split())
        output_words = set(output.lower().split())
        overlap = len(input_words & output_words) / max(len(input_words), 1)
        if overlap < 0.3:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _identify_improvements(self, output: str, input_text: str) -> List[str]:
        """Identify areas for improvement in the output."""
        improvements = []
        
        if len(output) < 20:
            improvements.append("Output too brief, needs more detail")
        
        if "." not in output:
            improvements.append("Add proper sentence structure")
        
        words = output.lower().split()
        if len(set(words)) < len(words) * 0.6:
            improvements.append("Reduce repetition")
        
        return improvements
    
    def _create_refinement_prompt(
        self,
        previous_output: str,
        context: Dict[str, Any]
    ) -> str:
        """Create a prompt for the next refinement iteration."""
        improvements = context.get("improvement_needed", [])
        
        refinement = f"Previous output: {previous_output}\n\n"
        refinement += f"Quality score: {context.get('quality_score', 0):.2f}\n"
        
        if improvements:
            refinement += "Areas to improve:\n"
            for imp in improvements:
                refinement += f"- {imp}\n"
        
        refinement += "\nPlease refine the output addressing these concerns."
        
        return refinement
    
    def _finalize_result(
        self,
        output: str,
        termination_reason: str,
        depth: int,
        quality_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """Finalize the recursive processing result."""
        return {
            "output": output,
            "termination_reason": termination_reason,
            "depth_reached": depth,
            "quality_score": quality_score,
            "reasoning_chain": self.processing_history if self.enable_reasoning_chains else [],
            "total_iterations": len(self.processing_history),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_reasoning_chain(self) -> List[Dict[str, Any]]:
        """Get the complete reasoning chain from last processing."""
        return self.processing_history.copy()
    
    def clear_history(self):
        """Clear processing history."""
        self.processing_history = []
    
    def analyze_reasoning_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in the reasoning chain."""
        if not self.processing_history:
            return {"error": "No processing history available"}
        
        depths = [step["depth"] for step in self.processing_history]
        
        return {
            "total_steps": len(self.processing_history),
            "max_depth": max(depths),
            "avg_depth": sum(depths) / len(depths),
            "refinement_count": len([s for s in self.processing_history if s["depth"] > 0]),
            "timeline": [
                {
                    "depth": step["depth"],
                    "timestamp": step["timestamp"],
                    "input_preview": step["input"][:50] + "..."
                }
                for step in self.processing_history
            ]
        }


class ReasoningChainBuilder:
    """
    Build and visualize reasoning chains from recursive processing.
    """
    
    def __init__(self):
        self.chains: Dict[str, List[Dict[str, Any]]] = {}
    
    def add_chain(self, chain_id: str, reasoning_steps: List[Dict[str, Any]]):
        """Store a reasoning chain."""
        self.chains[chain_id] = reasoning_steps
    
    def get_chain(self, chain_id: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve a reasoning chain."""
        return self.chains.get(chain_id)
    
    def visualize_chain(self, chain_id: str) -> str:
        """Create a text visualization of a reasoning chain."""
        chain = self.chains.get(chain_id)
        if not chain:
            return f"No chain found with id: {chain_id}"
        
        lines = [f"Reasoning Chain: {chain_id}", "=" * 50]
        
        for i, step in enumerate(chain, 1):
            depth = step.get("depth", 0)
            indent = "  " * depth
            lines.append(f"\n{indent}Step {i} (Depth {depth}):")
            lines.append(f"{indent}Input: {step.get('input', 'N/A')[:80]}...")
            lines.append(f"{indent}Output: {step.get('output', 'N/A')[:80]}...")
        
        return "\n".join(lines)
    
    def compare_chains(self, chain_id1: str, chain_id2: str) -> Dict[str, Any]:
        """Compare two reasoning chains."""
        chain1 = self.chains.get(chain_id1)
        chain2 = self.chains.get(chain_id2)
        
        if not chain1 or not chain2:
            return {"error": "One or both chains not found"}
        
        return {
            "chain1_length": len(chain1),
            "chain2_length": len(chain2),
            "depth_diff": max(s["depth"] for s in chain1) - max(s["depth"] for s in chain2),
            "complexity_ratio": len(chain1) / len(chain2) if chain2 else 0
        }


if __name__ == "__main__":
    # Demo usage
    print("🔄 Recursive Language Model - Demo\n")
    
    # Create a simple processor function for demonstration
    def demo_processor(text: str, context: Dict[str, Any]) -> str:
        """Simple demo processor that adds detail based on context."""
        depth = context.get("depth", 0)
        
        if depth == 0:
            return f"Processing: {text}"
        else:
            prev = context.get("previous_output", "")
            return f"{prev} [refined at depth {depth}]"
    
    # Initialize recursive model
    rlm = RecursiveLanguageModel(
        max_depth=3,
        reflection_threshold=0.85,
        enable_reasoning_chains=True
    )
    
    # Process with recursion
    result = rlm.recursive_process(
        "Explain machine learning in simple terms",
        demo_processor,
        {"mode": "educational"}
    )
    
    print(f"✅ Processing complete!")
    print(f"   Termination: {result['termination_reason']}")
    print(f"   Depth reached: {result['depth_reached']}")
    print(f"   Total iterations: {result['total_iterations']}")
    print(f"   Output: {result['output'][:100]}...")
    
    # Analyze reasoning patterns
    analysis = rlm.analyze_reasoning_patterns()
    print(f"\n📊 Reasoning Analysis:")
    print(f"   Total steps: {analysis['total_steps']}")
    print(f"   Max depth: {analysis['max_depth']}")
    print(f"   Refinements: {analysis['refinement_count']}")
    
    # Demonstrate reasoning chain builder
    print("\n🔗 Building reasoning chain...")
    chain_builder = ReasoningChainBuilder()
    chain_builder.add_chain("demo_chain", rlm.get_reasoning_chain())
    
    print("\n" + chain_builder.visualize_chain("demo_chain"))

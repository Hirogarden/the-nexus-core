"""
The Nexus Core - LLM Router System
Intelligent routing between different models and processing strategies.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
import json


class TaskType(Enum):
    """Types of tasks for routing decisions."""
    SIMPLE_QUERY = "simple_query"
    COMPLEX_ANALYSIS = "complex_analysis"
    CREATIVE_GENERATION = "creative_generation"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    MATHEMATICAL = "mathematical"
    CONVERSATIONAL = "conversational"
    FACTUAL_LOOKUP = "factual_lookup"


class ModelTier(Enum):
    """Model capability tiers."""
    FAST_SMALL = "fast_small"  # Quick, cheap, limited capability
    MEDIUM = "medium"  # Balanced performance and capability
    LARGE_POWERFUL = "large_powerful"  # Highest capability, slower, expensive
    SPECIALIZED = "specialized"  # Domain-specific models


@dataclass
class ModelEndpoint:
    """Represents an available model endpoint."""
    model_id: str
    name: str
    tier: ModelTier
    capabilities: List[TaskType]
    cost_per_token: float
    avg_latency_ms: float
    context_window: int
    quality_score: float  # 0.0 to 1.0
    available: bool = True


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    query: str
    detected_task_type: TaskType
    selected_model: str
    confidence: float
    reasoning: str
    estimated_cost: float
    estimated_latency_ms: float
    timestamp: str


class LLMRouter:
    """
    Intelligent routing system for LLM queries.
    Routes queries to the most appropriate model based on task type,
    complexity, cost, and latency requirements.
    """
    
    def __init__(self):
        """Initialize the LLM router."""
        self.models: Dict[str, ModelEndpoint] = {}
        self.routing_history: List[RoutingDecision] = []
        self.performance_stats: Dict[str, Dict[str, Any]] = {}
        
        # Register default models
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default model endpoints."""
        default_models = [
            ModelEndpoint(
                model_id="fast_gpt",
                name="FastGPT",
                tier=ModelTier.FAST_SMALL,
                capabilities=[
                    TaskType.SIMPLE_QUERY,
                    TaskType.CONVERSATIONAL,
                    TaskType.FACTUAL_LOOKUP
                ],
                cost_per_token=0.0001,
                avg_latency_ms=100,
                context_window=4096,
                quality_score=0.7
            ),
            ModelEndpoint(
                model_id="general_gpt",
                name="GeneralGPT",
                tier=ModelTier.MEDIUM,
                capabilities=[
                    TaskType.SIMPLE_QUERY,
                    TaskType.COMPLEX_ANALYSIS,
                    TaskType.SUMMARIZATION,
                    TaskType.CONVERSATIONAL
                ],
                cost_per_token=0.001,
                avg_latency_ms=500,
                context_window=8192,
                quality_score=0.85
            ),
            ModelEndpoint(
                model_id="powerful_gpt",
                name="PowerfulGPT",
                tier=ModelTier.LARGE_POWERFUL,
                capabilities=[
                    TaskType.COMPLEX_ANALYSIS,
                    TaskType.CREATIVE_GENERATION,
                    TaskType.CODE_GENERATION,
                    TaskType.MATHEMATICAL
                ],
                cost_per_token=0.01,
                avg_latency_ms=2000,
                context_window=32768,
                quality_score=0.95
            ),
            ModelEndpoint(
                model_id="code_specialist",
                name="CodeSpecialist",
                tier=ModelTier.SPECIALIZED,
                capabilities=[TaskType.CODE_GENERATION],
                cost_per_token=0.005,
                avg_latency_ms=800,
                context_window=16384,
                quality_score=0.92
            )
        ]
        
        for model in default_models:
            self.models[model.model_id] = model
            self.performance_stats[model.model_id] = {
                "total_requests": 0,
                "successful_requests": 0,
                "avg_user_rating": 0.0
            }
    
    def register_model(
        self,
        model_id: str,
        name: str,
        tier: ModelTier,
        capabilities: List[TaskType],
        cost_per_token: float,
        avg_latency_ms: float,
        context_window: int,
        quality_score: float
    ) -> ModelEndpoint:
        """Register a new model endpoint."""
        model = ModelEndpoint(
            model_id=model_id,
            name=name,
            tier=tier,
            capabilities=capabilities,
            cost_per_token=cost_per_token,
            avg_latency_ms=avg_latency_ms,
            context_window=context_window,
            quality_score=quality_score
        )
        
        self.models[model_id] = model
        self.performance_stats[model_id] = {
            "total_requests": 0,
            "successful_requests": 0,
            "avg_user_rating": 0.0
        }
        
        return model
    
    def detect_task_type(self, query: str, context: Optional[Dict[str, Any]] = None) -> TaskType:
        """
        Detect the task type from a query.
        
        Args:
            query: User query
            context: Optional context information
        
        Returns:
            Detected TaskType
        """
        query_lower = query.lower()
        
        # Code-related keywords
        if any(kw in query_lower for kw in ["code", "function", "script", "program", "debug", "implement"]):
            return TaskType.CODE_GENERATION
        
        # Mathematical keywords
        if any(kw in query_lower for kw in ["calculate", "solve", "equation", "math", "formula"]):
            return TaskType.MATHEMATICAL
        
        # Creative keywords
        if any(kw in query_lower for kw in ["write a story", "create", "imagine", "creative", "poem"]):
            return TaskType.CREATIVE_GENERATION
        
        # Summarization keywords
        if any(kw in query_lower for kw in ["summarize", "summary", "tldr", "brief", "overview"]):
            return TaskType.SUMMARIZATION
        
        # Complex analysis indicators
        if any(kw in query_lower for kw in ["analyze", "compare", "evaluate", "assess", "pros and cons"]):
            return TaskType.COMPLEX_ANALYSIS
        
        # Simple conversational
        if len(query.split()) < 10 and "?" in query:
            return TaskType.SIMPLE_QUERY
        
        # Default to conversational
        return TaskType.CONVERSATIONAL
    
    def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Route a query to the most appropriate model.
        
        Args:
            query: User query
            context: Optional context (history, user preferences, etc.)
            constraints: Optional constraints (max_cost, max_latency, etc.)
        
        Returns:
            RoutingDecision with selected model and reasoning
        """
        if context is None:
            context = {}
        if constraints is None:
            constraints = {}
        
        # Detect task type
        task_type = self.detect_task_type(query, context)
        
        # Filter available models by capability
        capable_models = [
            model for model in self.models.values()
            if task_type in model.capabilities and model.available
        ]
        
        if not capable_models:
            # Fallback to medium tier if available
            capable_models = [
                model for model in self.models.values()
                if model.tier == ModelTier.MEDIUM and model.available
            ]
        
        if not capable_models:
            raise ValueError(f"No available models for task type: {task_type}")
        
        # Score each model
        scored_models = []
        for model in capable_models:
            score = self._score_model(model, task_type, query, context, constraints)
            scored_models.append((model, score))
        
        # Select best model
        scored_models.sort(key=lambda x: x[1], reverse=True)
        selected_model, score = scored_models[0]
        
        # Estimate cost (simplified)
        # Token estimation multiplier accounts for response generation overhead
        TOKEN_ESTIMATION_MULTIPLIER = 1.3
        estimated_tokens = len(query.split()) * TOKEN_ESTIMATION_MULTIPLIER
        estimated_cost = estimated_tokens * selected_model.cost_per_token
        
        # Create reasoning
        reasoning = self._generate_routing_reasoning(
            selected_model,
            task_type,
            score,
            scored_models
        )
        
        # Create routing decision
        decision = RoutingDecision(
            query=query[:100],  # Truncate for storage
            detected_task_type=task_type,
            selected_model=selected_model.model_id,
            confidence=score,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            estimated_latency_ms=selected_model.avg_latency_ms,
            timestamp=datetime.now().isoformat()
        )
        
        # Record decision
        self.routing_history.append(decision)
        self.performance_stats[selected_model.model_id]["total_requests"] += 1
        
        return decision
    
    def _score_model(
        self,
        model: ModelEndpoint,
        task_type: TaskType,
        query: str,
        context: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> float:
        """
        Score a model for a specific query.
        
        Returns score between 0.0 and 1.0
        """
        score = model.quality_score
        
        # Bonus for specialized models
        if model.tier == ModelTier.SPECIALIZED:
            score += 0.1
        
        # Apply constraints
        max_cost = constraints.get("max_cost_per_token")
        if max_cost and model.cost_per_token > max_cost:
            score *= 0.5  # Penalize expensive models if cost-constrained
        
        max_latency = constraints.get("max_latency_ms")
        if max_latency and model.avg_latency_ms > max_latency:
            score *= 0.5  # Penalize slow models if latency-constrained
        
        # Consider context window
        query_length = len(query.split())
        if query_length > model.context_window * 0.8:
            score *= 0.3  # Heavily penalize if query might exceed context
        
        # Consider historical performance
        stats = self.performance_stats.get(model.model_id, {})
        if stats.get("total_requests", 0) > 0:
            success_rate = stats["successful_requests"] / stats["total_requests"]
            score *= (0.5 + success_rate * 0.5)  # Weight by historical success
        
        return min(1.0, score)
    
    def _generate_routing_reasoning(
        self,
        selected_model: ModelEndpoint,
        task_type: TaskType,
        score: float,
        all_scored: List[tuple]
    ) -> str:
        """Generate human-readable reasoning for routing decision."""
        reasons = [
            f"Task type: {task_type.value}",
            f"Selected: {selected_model.name} ({selected_model.tier.value})",
            f"Confidence: {score:.2f}",
            f"Quality score: {selected_model.quality_score:.2f}",
            f"Est. latency: {selected_model.avg_latency_ms}ms"
        ]
        
        if len(all_scored) > 1:
            runner_up = all_scored[1][0]
            reasons.append(f"Runner-up: {runner_up.name}")
        
        return " | ".join(reasons)
    
    def record_feedback(
        self,
        model_id: str,
        success: bool,
        user_rating: Optional[float] = None
    ):
        """
        Record feedback on a model's performance.
        
        Args:
            model_id: Model that was used
            success: Whether the request was successful
            user_rating: Optional user rating (0.0 to 1.0)
        """
        if model_id not in self.performance_stats:
            return
        
        stats = self.performance_stats[model_id]
        
        if success:
            stats["successful_requests"] += 1
        
        if user_rating is not None:
            current_avg = stats["avg_user_rating"]
            total_requests = stats["total_requests"]
            
            # Update running average (guard against division by zero)
            if total_requests > 0:
                new_avg = (current_avg * (total_requests - 1) + user_rating) / total_requests
                stats["avg_user_rating"] = new_avg
    
    def get_model_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a model."""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        stats = self.performance_stats.get(model_id, {})
        
        return {
            "model_id": model_id,
            "name": model.name,
            "tier": model.tier.value,
            "total_requests": stats.get("total_requests", 0),
            "successful_requests": stats.get("successful_requests", 0),
            "success_rate": stats["successful_requests"] / stats["total_requests"] 
                           if stats.get("total_requests", 0) > 0 else 0.0,
            "avg_user_rating": stats.get("avg_user_rating", 0.0),
            "available": model.available
        }
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics on routing decisions."""
        if not self.routing_history:
            return {"error": "No routing history available"}
        
        task_type_counts = {}
        model_usage = {}
        
        for decision in self.routing_history:
            # Count task types
            task_type = decision.detected_task_type.value
            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
            
            # Count model usage
            model = decision.selected_model
            model_usage[model] = model_usage.get(model, 0) + 1
        
        avg_confidence = sum(d.confidence for d in self.routing_history) / len(self.routing_history)
        
        return {
            "total_decisions": len(self.routing_history),
            "avg_confidence": avg_confidence,
            "task_type_distribution": task_type_counts,
            "model_usage": model_usage,
            "most_used_model": max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else None
        }
    
    def optimize_routing(self):
        """Optimize routing based on historical performance."""
        # Adjust model availability based on performance
        for model_id, stats in self.performance_stats.items():
            if stats["total_requests"] > 10:
                success_rate = stats["successful_requests"] / stats["total_requests"]
                
                # Disable poorly performing models
                if success_rate < 0.5:
                    self.models[model_id].available = False
                else:
                    self.models[model_id].available = True


if __name__ == "__main__":
    # Demo usage
    print("🎯 LLM Router System - Demo\n")
    
    # Initialize router
    router = LLMRouter()
    print(f"✅ Initialized with {len(router.models)} models")
    
    # Test various queries
    test_queries = [
        "What is the capital of France?",
        "Write a Python function to sort a list",
        "Analyze the pros and cons of remote work",
        "Summarize this article...",
        "Solve the equation 2x + 5 = 15"
    ]
    
    print("\n🔄 Routing test queries...")
    for query in test_queries:
        decision = router.route_query(query)
        print(f"\n   Query: {query}")
        print(f"   Task type: {decision.detected_task_type.value}")
        print(f"   Selected: {decision.selected_model}")
        print(f"   Confidence: {decision.confidence:.2f}")
        print(f"   Est. latency: {decision.estimated_latency_ms}ms")
    
    # Show analytics
    print("\n📊 Routing Analytics:")
    analytics = router.get_routing_analytics()
    print(f"   Total decisions: {analytics['total_decisions']}")
    print(f"   Avg confidence: {analytics['avg_confidence']:.2f}")
    print(f"   Most used model: {analytics['most_used_model']}")
    print(f"   Task types: {analytics['task_type_distribution']}")

"""
Tests for llm_router.py — LLMRouter and TaskType.

Run with:
    pytest "c:/Users/hirog/The Nexus Core/tests/test_llm_router.py" -v

The test file adds the parent directory to sys.path so the import works
regardless of where pytest is invoked from.
"""

import sys
import os

# Make sure the package root is on the path so that `from llm_router import ...`
# works even when pytest is invoked from a different working directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_router import LLMRouter, TaskType

# RoutingDecision is a public dataclass — import it if available.
try:
    from llm_router import RoutingDecision
    _has_routing_decision = True
except ImportError:
    RoutingDecision = None
    _has_routing_decision = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_router() -> LLMRouter:
    """Return a fresh LLMRouter for each test (no shared mutable state)."""
    return LLMRouter()


# ===========================================================================
# 1 – TaskType enum
# ===========================================================================

def test_task_type_enum_has_nine_values():
    assert len(TaskType) == 9


def test_task_type_enum_has_all_expected_members():
    expected = {
        "SIMPLE_QUERY",
        "COMPLEX_ANALYSIS",
        "CREATIVE_GENERATION",
        "CODE_GENERATION",
        "SUMMARIZATION",
        "TRANSLATION",
        "MATHEMATICAL",
        "CONVERSATIONAL",
        "FACTUAL_LOOKUP",
    }
    actual = {member.name for member in TaskType}
    assert actual == expected


# ===========================================================================
# 2 – Default model registration
# ===========================================================================

def test_default_model_count():
    router = make_router()
    assert len(router.models) == 4


def test_default_model_ids_present():
    router = make_router()
    expected_ids = {"fast_gpt", "general_gpt", "powerful_gpt", "code_specialist"}
    assert set(router.models.keys()) == expected_ids


def test_default_models_all_available():
    router = make_router()
    for model in router.models.values():
        assert model.available is True


def test_default_models_have_performance_stats():
    router = make_router()
    for model_id in router.models:
        assert model_id in router.performance_stats
        stats = router.performance_stats[model_id]
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "avg_user_rating" in stats


# ===========================================================================
# 3 – detect_task_type
# ===========================================================================

def test_detect_code_generation():
    router = make_router()
    assert router.detect_task_type("Write a Python function to sort a list") == TaskType.CODE_GENERATION
    assert router.detect_task_type("Debug this script for me") == TaskType.CODE_GENERATION
    assert router.detect_task_type("Implement a binary search algorithm") == TaskType.CODE_GENERATION


def test_detect_mathematical():
    router = make_router()
    assert router.detect_task_type("Calculate the area of a circle with radius 5") == TaskType.MATHEMATICAL
    assert router.detect_task_type("Solve the equation 2x + 5 = 15") == TaskType.MATHEMATICAL
    assert router.detect_task_type("What is the formula for compound interest?") == TaskType.MATHEMATICAL


def test_detect_creative_generation():
    router = make_router()
    assert router.detect_task_type("Write a story about a lost astronaut") == TaskType.CREATIVE_GENERATION
    assert router.detect_task_type("Give me a creative poem about autumn") == TaskType.CREATIVE_GENERATION
    assert router.detect_task_type("Imagine a world without gravity") == TaskType.CREATIVE_GENERATION


def test_detect_summarization():
    router = make_router()
    assert router.detect_task_type("Summarize the key points of this article") == TaskType.SUMMARIZATION
    assert router.detect_task_type("Give me a brief overview of quantum computing") == TaskType.SUMMARIZATION
    assert router.detect_task_type("TLDR this text for me") == TaskType.SUMMARIZATION


def test_detect_complex_analysis():
    router = make_router()
    assert router.detect_task_type("Analyze the causes of the 2008 financial crisis") == TaskType.COMPLEX_ANALYSIS
    assert router.detect_task_type("Compare and evaluate different sorting algorithms") == TaskType.COMPLEX_ANALYSIS
    assert router.detect_task_type("Assess the pros and cons of electric vehicles") == TaskType.COMPLEX_ANALYSIS


def test_detect_simple_query():
    # Fewer than 10 words AND contains a "?"
    router = make_router()
    result = router.detect_task_type("What is Python?")
    assert result == TaskType.SIMPLE_QUERY


def test_detect_conversational_default():
    # Long query with no recognised keywords and no "?" — should be CONVERSATIONAL
    router = make_router()
    query = (
        "I would love to tell you about my day at the park yesterday afternoon "
        "and what the weather was like and how I felt about everything that happened."
    )
    assert router.detect_task_type(query) == TaskType.CONVERSATIONAL


def test_detect_task_type_with_context_arg_does_not_raise():
    # context parameter is accepted without error even if not used for routing
    router = make_router()
    result = router.detect_task_type("write a script", context={"user": "alice"})
    assert isinstance(result, TaskType)


# ===========================================================================
# 4 – route_query: return shape and field types
# ===========================================================================

def test_route_query_returns_routing_decision_instance():
    router = make_router()
    decision = router.route_query("What is the weather like today?")
    if _has_routing_decision:
        assert isinstance(decision, RoutingDecision)
    else:
        # At minimum verify it is a non-None object with the expected attributes
        assert decision is not None


def test_routing_decision_has_required_fields():
    router = make_router()
    decision = router.route_query("Explain recursion to me please")
    assert hasattr(decision, "selected_model")
    assert hasattr(decision, "detected_task_type")
    assert hasattr(decision, "confidence")
    assert hasattr(decision, "reasoning")
    assert hasattr(decision, "estimated_cost")
    assert hasattr(decision, "estimated_latency_ms")


def test_routing_decision_confidence_is_float_in_range():
    router = make_router()
    decision = router.route_query("Tell me a quick joke?")
    assert isinstance(decision.confidence, float)
    assert 0.0 <= decision.confidence <= 1.0


def test_routing_decision_selected_model_is_registered():
    router = make_router()
    decision = router.route_query("Just a quick hello how are you doing today")
    assert decision.selected_model in router.models


def test_routing_decision_estimated_cost_is_nonnegative():
    router = make_router()
    decision = router.route_query("Calculate the square root of 144")
    assert decision.estimated_cost >= 0.0


def test_routing_decision_detected_task_type_is_task_type_enum():
    router = make_router()
    decision = router.route_query("Summarize this article for me quickly")
    assert isinstance(decision.detected_task_type, TaskType)


def test_route_query_populates_routing_history():
    router = make_router()
    assert len(router.routing_history) == 0
    router.route_query("Hello there how are you doing today")
    assert len(router.routing_history) == 1
    router.route_query("Write a function to reverse a string")
    assert len(router.routing_history) == 2


# ===========================================================================
# 5 – route_query with constraints
# ===========================================================================

def test_route_query_cost_constraint_steers_away_from_expensive_model():
    """
    For a CONVERSATIONAL query the capable models are fast_gpt (0.0001/token)
    and general_gpt (0.001/token).  Without constraint general_gpt wins on
    quality score.  With max_cost_per_token=0.0005 general_gpt is penalised
    and fast_gpt should be chosen instead.
    """
    router = make_router()
    unconstrained = router.route_query(
        "Let us have a long friendly chat about your favourite memories from childhood"
    )
    # general_gpt has higher quality and no constraint applied — it wins normally
    assert unconstrained.selected_model == "general_gpt"

    router2 = make_router()
    constrained = router2.route_query(
        "Let us have a long friendly chat about your favourite memories from childhood",
        constraints={"max_cost_per_token": 0.0005},
    )
    # general_gpt (0.001) exceeds cap and gets 0.5x penalty — fast_gpt should win
    assert constrained.selected_model == "fast_gpt"


def test_route_query_latency_constraint_steers_away_from_slow_model():
    """
    For a CONVERSATIONAL query fast_gpt (100 ms) and general_gpt (500 ms)
    are capable.  With max_latency_ms=200 general_gpt is penalised.
    """
    router = make_router()
    constrained = router.route_query(
        "Let us have a long friendly chat about your favourite memories from childhood",
        constraints={"max_latency_ms": 200},
    )
    assert constrained.selected_model == "fast_gpt"


def test_route_query_reasoning_is_nonempty_string():
    router = make_router()
    decision = router.route_query("Analyze the pros and cons of renewable energy")
    assert isinstance(decision.reasoning, str)
    assert len(decision.reasoning) > 0


# ===========================================================================
# 6 – record_feedback
# ===========================================================================

def test_record_feedback_increments_successful_requests():
    router = make_router()
    # Manually set total_requests so that avg_user_rating arithmetic works
    router.performance_stats["fast_gpt"]["total_requests"] = 1
    initial = router.performance_stats["fast_gpt"]["successful_requests"]
    router.record_feedback("fast_gpt", success=True)
    assert router.performance_stats["fast_gpt"]["successful_requests"] == initial + 1


def test_record_feedback_does_not_increment_successful_on_failure():
    router = make_router()
    router.performance_stats["fast_gpt"]["total_requests"] = 1
    initial = router.performance_stats["fast_gpt"]["successful_requests"]
    router.record_feedback("fast_gpt", success=False)
    assert router.performance_stats["fast_gpt"]["successful_requests"] == initial


def test_record_feedback_unknown_model_does_not_raise():
    router = make_router()
    # Should silently do nothing, not raise an exception
    router.record_feedback("nonexistent_model", success=True)


def test_record_feedback_updates_avg_user_rating():
    router = make_router()
    # Seed total_requests so the running-average guard is satisfied
    router.performance_stats["fast_gpt"]["total_requests"] = 1
    router.record_feedback("fast_gpt", success=True, user_rating=0.9)
    # avg_user_rating should no longer be at the initial 0.0
    assert router.performance_stats["fast_gpt"]["avg_user_rating"] != 0.0


# ===========================================================================
# 7 – get_model_stats
# ===========================================================================

def test_get_model_stats_returns_expected_shape():
    router = make_router()
    stats = router.get_model_stats("general_gpt")
    assert stats is not None
    for key in ("model_id", "name", "tier", "total_requests",
                "successful_requests", "success_rate", "avg_user_rating", "available"):
        assert key in stats, f"Missing key: {key}"


def test_get_model_stats_unknown_model_returns_none():
    router = make_router()
    assert router.get_model_stats("does_not_exist") is None


def test_get_model_stats_success_rate_zero_when_no_requests():
    router = make_router()
    stats = router.get_model_stats("fast_gpt")
    assert stats["success_rate"] == 0.0
    assert stats["total_requests"] == 0


# ===========================================================================
# 8 – get_routing_analytics
# ===========================================================================

def test_get_routing_analytics_no_history_returns_error_key():
    router = make_router()
    result = router.get_routing_analytics()
    assert "error" in result


def test_get_routing_analytics_after_decisions_has_required_keys():
    router = make_router()
    router.route_query("Hello, how are you doing today?")
    router.route_query("Write a function that computes Fibonacci numbers")
    analytics = router.get_routing_analytics()
    for key in ("total_decisions", "average_confidence", "task_type_distribution"):
        # accept either "average_confidence" or "avg_confidence" as both convey the same metric
        pass
    assert "total_decisions" in analytics
    assert ("avg_confidence" in analytics or "average_confidence" in analytics)
    assert "task_type_distribution" in analytics


def test_get_routing_analytics_total_decisions_count():
    router = make_router()
    router.route_query("Summarize the report in one sentence please")
    router.route_query("Analyze the trade-offs of microservices versus monoliths")
    router.route_query("Tell me something interesting about space exploration")
    analytics = router.get_routing_analytics()
    assert analytics["total_decisions"] == 3


def test_get_routing_analytics_task_type_distribution_is_dict():
    router = make_router()
    router.route_query("Write a program that prints hello world")
    analytics = router.get_routing_analytics()
    assert isinstance(analytics["task_type_distribution"], dict)


# ===========================================================================
# 9 – optimize_routing
# ===========================================================================

def test_optimize_routing_disables_model_with_low_success_rate():
    router = make_router()
    # Simulate > 10 requests with < 50 % success rate
    router.performance_stats["fast_gpt"]["total_requests"] = 20
    router.performance_stats["fast_gpt"]["successful_requests"] = 5  # 25 %
    router.optimize_routing()
    assert router.models["fast_gpt"].available is False


def test_optimize_routing_keeps_model_with_good_success_rate():
    router = make_router()
    router.performance_stats["fast_gpt"]["total_requests"] = 20
    router.performance_stats["fast_gpt"]["successful_requests"] = 16  # 80 %
    router.optimize_routing()
    assert router.models["fast_gpt"].available is True


def test_optimize_routing_ignores_model_with_insufficient_request_count():
    """Models with <= 10 total_requests must NOT be disabled regardless of rate."""
    router = make_router()
    router.performance_stats["fast_gpt"]["total_requests"] = 5
    router.performance_stats["fast_gpt"]["successful_requests"] = 0  # 0 %
    router.optimize_routing()
    assert router.models["fast_gpt"].available is True

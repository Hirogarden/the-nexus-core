"""
Tests for meta_agent_system.py

Covers: AgentRole enum, AgentTask dataclass, Agent dataclass,
MetaAgentCoordinator (heuristic decomposition, LLM decomposition,
task assignment, workflow execution, system status).
"""

import json
import pytest

from meta_agent_system import (
    AgentRole,
    AgentTask,
    Agent,
    MetaAgentCoordinator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coordinator_with_processors() -> MetaAgentCoordinator:
    """Return a no-LLM coordinator whose agents all have simple processors."""
    coord = MetaAgentCoordinator()
    for agent in coord.agents.values():
        agent.processor = lambda data: {"processed": True, "data": data}
    return coord


# ---------------------------------------------------------------------------
# AgentRole enum
# ---------------------------------------------------------------------------

class TestAgentRole:
    def test_has_all_required_members(self):
        """All seven documented roles must be present."""
        required = {
            "COORDINATOR", "RESEARCHER", "ANALYZER",
            "WRITER", "CRITIC", "EXECUTOR", "PLANNER",
        }
        actual = {member.name for member in AgentRole}
        assert required == actual

    def test_member_string_values(self):
        """Each role's .value must be its lowercase name."""
        for member in AgentRole:
            assert member.value == member.name.lower()

    def test_enum_members_are_distinct(self):
        """No two roles share the same value."""
        values = [m.value for m in AgentRole]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Default agent registration
# ---------------------------------------------------------------------------

class TestDefaultAgents:
    def test_coordinator_has_five_agents(self):
        coord = MetaAgentCoordinator()
        assert len(coord.agents) == 5

    def test_default_agent_ids_present(self):
        coord = MetaAgentCoordinator()
        expected_ids = {"coord_001", "research_001", "analyze_001", "write_001", "critic_001"}
        assert set(coord.agents.keys()) == expected_ids

    def test_default_agent_roles(self):
        coord = MetaAgentCoordinator()
        role_map = {a.agent_id: a.role for a in coord.agents.values()}
        assert role_map["coord_001"] == AgentRole.COORDINATOR
        assert role_map["research_001"] == AgentRole.RESEARCHER
        assert role_map["analyze_001"] == AgentRole.ANALYZER
        assert role_map["write_001"] == AgentRole.WRITER
        assert role_map["critic_001"] == AgentRole.CRITIC

    def test_all_default_agents_are_active(self):
        coord = MetaAgentCoordinator()
        for agent in coord.agents.values():
            assert agent.active is True

    def test_default_agents_have_empty_task_history(self):
        coord = MetaAgentCoordinator()
        for agent in coord.agents.values():
            assert agent.task_history == []


# ---------------------------------------------------------------------------
# Heuristic decomposition
# ---------------------------------------------------------------------------

class TestHeuristicDecompose:
    @pytest.fixture
    def tasks(self):
        coord = MetaAgentCoordinator()  # no llm_fn
        return coord.decompose_task("build something cool")

    def test_returns_exactly_four_tasks(self, tasks):
        assert len(tasks) == 4

    def test_role_order(self, tasks):
        expected_roles = [
            AgentRole.RESEARCHER,
            AgentRole.ANALYZER,
            AgentRole.WRITER,
            AgentRole.CRITIC,
        ]
        assert [t.role for t in tasks] == expected_roles

    def test_priorities_are_one_through_four(self, tasks):
        assert [t.priority for t in tasks] == [1, 2, 3, 4]

    def test_task_ids_are_unique_strings(self, tasks):
        ids = [t.task_id for t in tasks]
        assert all(isinstance(tid, str) for tid in ids)
        assert len(ids) == len(set(ids))

    def test_first_task_has_no_dependencies(self, tasks):
        assert tasks[0].dependencies == []

    def test_dependency_chain(self, tasks):
        """Each task (index 1-3) must depend on exactly the preceding task."""
        for i in range(1, 4):
            assert tasks[i].dependencies == [tasks[i - 1].task_id]

    def test_all_tasks_have_main_task_in_input_data(self, tasks):
        for task in tasks:
            assert "main_task" in task.input_data
            assert task.input_data["main_task"] == "build something cool"

    def test_context_propagated_to_first_task(self):
        coord = MetaAgentCoordinator()
        tasks = coord.decompose_task("demo", context={"env": "test"})
        assert tasks[0].input_data.get("env") == "test"

    def test_tasks_registered_in_coordinator(self):
        coord = MetaAgentCoordinator()
        tasks = coord.decompose_task("register test")
        for task in tasks:
            assert task.task_id in coord.tasks

    def test_task_counter_is_monotonically_increasing(self):
        """Counters across two separate decompose calls must never repeat."""
        coord = MetaAgentCoordinator()
        first = coord.decompose_task("task A")
        second = coord.decompose_task("task B")
        all_ids = [t.task_id for t in first + second]
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------

class TestExecuteWorkflow:
    @pytest.fixture
    def workflow_result(self):
        coord = _make_coordinator_with_processors()
        tasks = coord.decompose_task("end-to-end test")
        return coord.execute_workflow(tasks)

    def test_all_four_tasks_complete(self, workflow_result):
        assert workflow_result["completed"] == 4

    def test_no_failed_tasks(self, workflow_result):
        assert workflow_result["failed"] == 0
        assert workflow_result["failed_task_ids"] == []

    def test_workflow_result_has_required_keys(self, workflow_result):
        required_keys = {
            "total_tasks", "completed", "failed", "failed_task_ids",
            "results", "timestamp",
        }
        assert required_keys.issubset(workflow_result.keys())

    def test_total_tasks_count(self, workflow_result):
        assert workflow_result["total_tasks"] == 4

    def test_results_list_length(self, workflow_result):
        assert len(workflow_result["results"]) == 4

    def test_all_results_have_success_flag(self, workflow_result):
        for result in workflow_result["results"]:
            assert "success" in result
            assert result["success"] is True


# ---------------------------------------------------------------------------
# System status
# ---------------------------------------------------------------------------

class TestGetSystemStatus:
    def test_required_keys_present(self):
        coord = MetaAgentCoordinator()
        status = coord.get_system_status()
        required = {
            "total_agents", "active_agents", "total_tasks",
            "pending_tasks", "in_progress_tasks", "completed_tasks",
            "failed_tasks", "execution_log_size",
        }
        assert required == set(status.keys())

    def test_initial_agent_counts(self):
        coord = MetaAgentCoordinator()
        status = coord.get_system_status()
        assert status["total_agents"] == 5
        assert status["active_agents"] == 5

    def test_task_counts_after_workflow(self):
        coord = _make_coordinator_with_processors()
        tasks = coord.decompose_task("status check")
        coord.execute_workflow(tasks)
        status = coord.get_system_status()
        assert status["total_tasks"] == 4
        assert status["completed_tasks"] == 4
        assert status["pending_tasks"] == 0
        assert status["failed_tasks"] == 0

    def test_execution_log_grows_after_workflow(self):
        coord = _make_coordinator_with_processors()
        tasks = coord.decompose_task("log test")
        coord.execute_workflow(tasks)
        status = coord.get_system_status()
        assert status["execution_log_size"] == 4


# ---------------------------------------------------------------------------
# LLM-backed decomposition
# ---------------------------------------------------------------------------

class TestLLMDecompose:
    _VALID_JSON = json.dumps([
        {"role": "researcher", "description": "gather relevant data", "depends_on": None},
        {"role": "writer", "description": "write the report", "depends_on": 0},
    ])

    def _llm_fn_valid(self, _prompt: str) -> str:
        return self._VALID_JSON

    def test_llm_path_returns_at_least_two_tasks(self):
        coord = MetaAgentCoordinator(llm_fn=self._llm_fn_valid)
        tasks = coord.decompose_task("llm driven task")
        assert len(tasks) >= 2

    def test_llm_path_correct_roles(self):
        coord = MetaAgentCoordinator(llm_fn=self._llm_fn_valid)
        tasks = coord.decompose_task("llm driven task")
        assert tasks[0].role == AgentRole.RESEARCHER
        assert tasks[1].role == AgentRole.WRITER

    def test_llm_path_depends_on_wiring(self):
        """tasks[1].dependencies must contain tasks[0].task_id."""
        coord = MetaAgentCoordinator(llm_fn=self._llm_fn_valid)
        tasks = coord.decompose_task("dependency wiring test")
        assert tasks[0].task_id in tasks[1].dependencies

    def test_llm_path_first_task_no_dependencies(self):
        coord = MetaAgentCoordinator(llm_fn=self._llm_fn_valid)
        tasks = coord.decompose_task("first task deps")
        assert tasks[0].dependencies == []

    def test_llm_path_unknown_role_defaults_to_writer(self):
        garbage_role_json = json.dumps([
            {"role": "unicorn", "description": "do magic", "depends_on": None},
        ])
        coord = MetaAgentCoordinator(llm_fn=lambda _p: garbage_role_json)
        tasks = coord.decompose_task("unknown role test")
        assert tasks[0].role == AgentRole.WRITER

    def test_llm_garbage_response_falls_back_to_heuristic(self):
        """If llm_fn returns non-JSON, decompose must return the 4-step heuristic."""
        coord = MetaAgentCoordinator(llm_fn=lambda _p: "this is not json at all!!")
        tasks = coord.decompose_task("fallback test")
        assert len(tasks) == 4
        assert tasks[0].role == AgentRole.RESEARCHER

    def test_llm_empty_array_falls_back_to_heuristic(self):
        """An empty JSON array from the LLM should trigger the fallback."""
        coord = MetaAgentCoordinator(llm_fn=lambda _p: "[]")
        tasks = coord.decompose_task("empty array fallback")
        assert len(tasks) == 4

    def test_llm_input_data_contains_main_task(self):
        coord = MetaAgentCoordinator(llm_fn=self._llm_fn_valid)
        tasks = coord.decompose_task("llm input check")
        for task in tasks:
            assert "main_task" in task.input_data
            assert task.input_data["main_task"] == "llm input check"


# ---------------------------------------------------------------------------
# Task assignment
# ---------------------------------------------------------------------------

class TestAssignTask:
    def test_returns_agent_with_matching_role(self):
        coord = MetaAgentCoordinator()
        task = AgentTask(
            task_id="t_test",
            role=AgentRole.RESEARCHER,
            description="find stuff",
            input_data={"main_task": "test"},
        )
        agent = coord.assign_task(task)
        assert agent is not None
        assert agent.role == AgentRole.RESEARCHER

    def test_returns_none_when_no_matching_role(self):
        coord = MetaAgentCoordinator()
        # No EXECUTOR or PLANNER registered by default
        task = AgentTask(
            task_id="t_exec",
            role=AgentRole.EXECUTOR,
            description="execute something",
            input_data={"main_task": "test"},
        )
        agent = coord.assign_task(task)
        assert agent is None

    def test_prefers_agent_with_shorter_task_history(self):
        coord = MetaAgentCoordinator()
        # Register two researcher agents and give one a head start
        coord.register_agent("research_002", AgentRole.RESEARCHER, ["extra"])
        coord.agents["research_001"].task_history = ["old_task_1", "old_task_2"]

        task = AgentTask(
            task_id="t_new",
            role=AgentRole.RESEARCHER,
            description="new research",
            input_data={"main_task": "test"},
        )
        selected = coord.assign_task(task)
        # research_002 has an empty history so it should be chosen
        assert selected is not None
        assert selected.agent_id == "research_002"

    def test_assign_task_adds_task_to_agent_history(self):
        coord = MetaAgentCoordinator()
        task = AgentTask(
            task_id="t_hist",
            role=AgentRole.ANALYZER,
            description="analyze",
            input_data={"main_task": "test"},
        )
        agent = coord.assign_task(task)
        assert agent is not None
        assert "t_hist" in agent.task_history


# ---------------------------------------------------------------------------
# execute_task dependency enforcement
# ---------------------------------------------------------------------------

class TestExecuteTaskDependencies:
    def test_fails_when_dependency_not_completed(self):
        coord = _make_coordinator_with_processors()
        tasks = coord.decompose_task("dep enforcement test")
        # Attempt to execute the second task without completing the first
        result = coord.execute_task(tasks[1])
        assert result["success"] is False
        assert "Dependency" in result["error"]

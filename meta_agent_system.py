"""
The Nexus Core - MetaGPT-Inspired Meta-Agent System
Multi-agent coordination and task decomposition framework.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from enum import Enum
import json
from dataclasses import dataclass, asdict


class AgentRole(Enum):
    """Predefined agent roles for different tasks."""
    COORDINATOR = "coordinator"  # Orchestrates other agents
    RESEARCHER = "researcher"  # Gathers information
    ANALYZER = "analyzer"  # Analyzes data
    WRITER = "writer"  # Generates content
    CRITIC = "critic"  # Reviews and critiques outputs
    EXECUTOR = "executor"  # Executes specific tasks
    PLANNER = "planner"  # Creates plans and strategies


@dataclass
class AgentTask:
    """Represents a task assigned to an agent."""
    task_id: str
    role: AgentRole
    description: str
    input_data: Dict[str, Any]
    priority: int = 1
    dependencies: List[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class Agent:
    """Represents an agent with a specific role and capabilities."""
    agent_id: str
    role: AgentRole
    capabilities: List[str]
    processor: Optional[Callable] = None
    active: bool = True
    task_history: List[str] = None
    
    def __post_init__(self):
        if self.task_history is None:
            self.task_history = []


class MetaAgentCoordinator:
    """
    Meta-agent system inspired by MetaGPT.
    Coordinates multiple specialized agents for complex tasks.
    """
    
    def __init__(self):
        """Initialize the meta-agent coordinator."""
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, AgentTask] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # Register default agents
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register default set of agents."""
        default_agents = [
            Agent(
                agent_id="coord_001",
                role=AgentRole.COORDINATOR,
                capabilities=["task_decomposition", "agent_assignment", "workflow_management"]
            ),
            Agent(
                agent_id="research_001",
                role=AgentRole.RESEARCHER,
                capabilities=["information_gathering", "data_collection", "source_verification"]
            ),
            Agent(
                agent_id="analyze_001",
                role=AgentRole.ANALYZER,
                capabilities=["data_analysis", "pattern_recognition", "insight_generation"]
            ),
            Agent(
                agent_id="write_001",
                role=AgentRole.WRITER,
                capabilities=["content_generation", "summarization", "documentation"]
            ),
            Agent(
                agent_id="critic_001",
                role=AgentRole.CRITIC,
                capabilities=["quality_assessment", "error_detection", "improvement_suggestions"]
            )
        ]
        
        for agent in default_agents:
            self.agents[agent.agent_id] = agent
    
    def register_agent(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: List[str],
        processor: Optional[Callable] = None
    ) -> Agent:
        """Register a new agent with the coordinator."""
        agent = Agent(
            agent_id=agent_id,
            role=role,
            capabilities=capabilities,
            processor=processor
        )
        self.agents[agent_id] = agent
        return agent
    
    def decompose_task(
        self,
        main_task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[AgentTask]:
        """
        Decompose a complex task into subtasks for different agents.
        
        Args:
            main_task: High-level task description
            context: Additional context for task decomposition
        
        Returns:
            List of decomposed subtasks
        """
        if context is None:
            context = {}
        
        # Simple heuristic-based decomposition
        # In production, this would use an LLM
        subtasks = []
        
        # Always start with research
        subtasks.append(AgentTask(
            task_id=f"task_{len(self.tasks) + 1}",
            role=AgentRole.RESEARCHER,
            description=f"Research and gather information for: {main_task}",
            input_data={"main_task": main_task, **context},
            priority=1
        ))
        
        # Then analysis
        subtasks.append(AgentTask(
            task_id=f"task_{len(self.tasks) + 2}",
            role=AgentRole.ANALYZER,
            description=f"Analyze gathered information for: {main_task}",
            input_data={"main_task": main_task},
            priority=2,
            dependencies=[subtasks[0].task_id]
        ))
        
        # Writing/generation
        subtasks.append(AgentTask(
            task_id=f"task_{len(self.tasks) + 3}",
            role=AgentRole.WRITER,
            description=f"Generate output for: {main_task}",
            input_data={"main_task": main_task},
            priority=3,
            dependencies=[subtasks[1].task_id]
        ))
        
        # Critical review
        subtasks.append(AgentTask(
            task_id=f"task_{len(self.tasks) + 4}",
            role=AgentRole.CRITIC,
            description=f"Review and validate output for: {main_task}",
            input_data={"main_task": main_task},
            priority=4,
            dependencies=[subtasks[2].task_id]
        ))
        
        # Store tasks
        for task in subtasks:
            self.tasks[task.task_id] = task
        
        return subtasks
    
    def assign_task(self, task: AgentTask) -> Optional[Agent]:
        """
        Assign a task to the most suitable agent.
        
        Args:
            task: The task to assign
        
        Returns:
            The assigned agent, or None if no suitable agent found
        """
        # Find agents with matching role
        suitable_agents = [
            agent for agent in self.agents.values()
            if agent.role == task.role and agent.active
        ]
        
        if not suitable_agents:
            return None
        
        # Select agent with least workload
        selected_agent = min(suitable_agents, key=lambda a: len(a.task_history))
        selected_agent.task_history.append(task.task_id)
        
        return selected_agent
    
    def execute_task(
        self,
        task: AgentTask,
        agent: Optional[Agent] = None
    ) -> Dict[str, Any]:
        """
        Execute a specific task.
        
        Args:
            task: The task to execute
            agent: Optional specific agent to use
        
        Returns:
            Task execution result
        """
        # Check dependencies
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                return {
                    "success": False,
                    "error": f"Dependency {dep_id} not completed",
                    "task_id": task.task_id
                }
        
        # Assign agent if not provided
        if agent is None:
            agent = self.assign_task(task)
        
        if agent is None:
            return {
                "success": False,
                "error": "No suitable agent available",
                "task_id": task.task_id
            }
        
        # Update task status
        task.status = "in_progress"
        
        # Execute task
        try:
            if agent.processor:
                result = agent.processor(task.input_data)
            else:
                # Default processing
                result = self._default_process(task, agent)
            
            task.status = "completed"
            task.result = result
            
            # Log execution
            self.execution_log.append({
                "task_id": task.task_id,
                "agent_id": agent.agent_id,
                "role": agent.role.value,
                "timestamp": datetime.now().isoformat(),
                "success": True
            })
            
            return {
                "success": True,
                "task_id": task.task_id,
                "agent_id": agent.agent_id,
                "result": result
            }
            
        except Exception as e:
            task.status = "failed"
            
            self.execution_log.append({
                "task_id": task.task_id,
                "agent_id": agent.agent_id,
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": str(e)
            })
            
            return {
                "success": False,
                "task_id": task.task_id,
                "error": str(e)
            }
    
    def _default_process(self, task: AgentTask, agent: Agent) -> Dict[str, Any]:
        """Default processing for tasks without custom processor."""
        return {
            "role": agent.role.value,
            "task_description": task.description,
            "processed": True,
            "timestamp": datetime.now().isoformat(),
            "note": "Default processing applied"
        }
    
    def execute_workflow(
        self,
        tasks: List[AgentTask]
    ) -> Dict[str, Any]:
        """
        Execute a complete workflow of tasks.
        
        Args:
            tasks: List of tasks to execute in order
        
        Returns:
            Workflow execution summary
        """
        results = []
        failed_tasks = []
        
        # Sort tasks by priority and dependencies
        sorted_tasks = self._topological_sort(tasks)
        
        for task in sorted_tasks:
            result = self.execute_task(task)
            results.append(result)
            
            if not result["success"]:
                failed_tasks.append(task.task_id)
        
        return {
            "total_tasks": len(tasks),
            "completed": len([r for r in results if r["success"]]),
            "failed": len(failed_tasks),
            "failed_task_ids": failed_tasks,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _topological_sort(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks based on dependencies and priority."""
        # Simple priority-based sort (in production, use proper topological sort)
        return sorted(tasks, key=lambda t: (t.priority, len(t.dependencies)))
    
    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a specific agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return None
        
        return {
            "agent_id": agent.agent_id,
            "role": agent.role.value,
            "capabilities": agent.capabilities,
            "active": agent.active,
            "tasks_completed": len(agent.task_history),
            "current_tasks": [
                task_id for task_id in agent.task_history
                if self.tasks.get(task_id) and self.tasks[task_id].status == "in_progress"
            ]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            "total_agents": len(self.agents),
            "active_agents": len([a for a in self.agents.values() if a.active]),
            "total_tasks": len(self.tasks),
            "pending_tasks": len([t for t in self.tasks.values() if t.status == "pending"]),
            "in_progress_tasks": len([t for t in self.tasks.values() if t.status == "in_progress"]),
            "completed_tasks": len([t for t in self.tasks.values() if t.status == "completed"]),
            "failed_tasks": len([t for t in self.tasks.values() if t.status == "failed"]),
            "execution_log_size": len(self.execution_log)
        }
    
    def export_workflow(self, task_ids: List[str]) -> str:
        """Export a workflow as JSON."""
        workflow_tasks = [
            asdict(self.tasks[tid]) for tid in task_ids
            if tid in self.tasks
        ]
        
        # Convert Enum to string for JSON serialization
        for task in workflow_tasks:
            task["role"] = task["role"].value if isinstance(task["role"], AgentRole) else task["role"]
        
        return json.dumps({
            "workflow": workflow_tasks,
            "timestamp": datetime.now().isoformat()
        }, indent=2)


if __name__ == "__main__":
    # Demo usage
    print("🤖 MetaGPT-Inspired Meta-Agent System - Demo\n")
    
    # Initialize coordinator
    coordinator = MetaAgentCoordinator()
    
    print(f"✅ Initialized with {len(coordinator.agents)} default agents")
    
    # Decompose a complex task
    main_task = "Analyze recent trends in AI and write a comprehensive report"
    print(f"\n📋 Decomposing task: '{main_task}'")
    
    subtasks = coordinator.decompose_task(main_task, {"domain": "artificial_intelligence"})
    print(f"   Created {len(subtasks)} subtasks:")
    for task in subtasks:
        print(f"   - {task.role.value}: {task.description[:60]}...")
    
    # Execute the workflow
    print("\n⚙️  Executing workflow...")
    workflow_result = coordinator.execute_workflow(subtasks)
    
    print(f"\n✅ Workflow complete!")
    print(f"   Total tasks: {workflow_result['total_tasks']}")
    print(f"   Completed: {workflow_result['completed']}")
    print(f"   Failed: {workflow_result['failed']}")
    
    # Show system status
    status = coordinator.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Active agents: {status['active_agents']}/{status['total_agents']}")
    print(f"   Completed tasks: {status['completed_tasks']}")
    print(f"   Execution log entries: {status['execution_log_size']}")

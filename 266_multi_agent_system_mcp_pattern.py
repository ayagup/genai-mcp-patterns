"""
Pattern 266: Multi-Agent System MCP Pattern

This pattern demonstrates a comprehensive multi-agent system with
agent communication, coordination, and task distribution.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class MultiAgentSystemState(TypedDict):
    """State for multi-agent system workflow"""
    messages: Annotated[List[str], add]
    system_config: Dict[str, Any]
    agents: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    task_assignments: List[Dict[str, Any]]
    execution_results: List[Dict[str, Any]]


class Agent:
    """Generic agent in the multi-agent system"""
    
    def __init__(self, agent_id: str, capabilities: List[str], capacity: int):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.capacity = capacity
        self.current_load = 0
        self.assigned_tasks = []
        self.completed_tasks = []
    
    def can_handle(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle the task"""
        required_capability = task.get("required_capability", "")
        has_capability = required_capability in self.capabilities
        has_capacity = self.current_load < self.capacity
        return has_capability and has_capacity
    
    def assign_task(self, task: Dict[str, Any]):
        """Assign task to agent"""
        self.assigned_tasks.append(task)
        self.current_load += 1
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute assigned task"""
        task_complexity = task.get("complexity", 1)
        execution_time = task_complexity * (1.0 / len(self.capabilities))
        
        result = {
            "task_id": task.get("id", "unknown"),
            "agent_id": self.agent_id,
            "status": "completed",
            "execution_time": execution_time,
            "quality": 0.9 if task.get("required_capability") in self.capabilities else 0.6
        }
        
        self.completed_tasks.append(task)
        self.current_load = max(0, self.current_load - 1)
        
        return result


class MultiAgentCoordinator:
    """Coordinates multiple agents"""
    
    def __init__(self):
        self.agents = []
        self.task_queue = []
    
    def register_agent(self, agent: Agent):
        """Register agent in the system"""
        self.agents.append(agent)
    
    def assign_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assign tasks to appropriate agents"""
        assignments = []
        
        for task in tasks:
            assigned = False
            
            # Try to find best agent for task
            for agent in self.agents:
                if agent.can_handle(task):
                    agent.assign_task(task)
                    assignments.append({
                        "task_id": task.get("id"),
                        "task_name": task.get("name"),
                        "assigned_to": agent.agent_id,
                        "required_capability": task.get("required_capability"),
                        "complexity": task.get("complexity", 1)
                    })
                    assigned = True
                    break
            
            if not assigned:
                assignments.append({
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "assigned_to": "UNASSIGNED",
                    "required_capability": task.get("required_capability"),
                    "reason": "No available agent with required capability"
                })
        
        return assignments
    
    def execute_all_tasks(self) -> List[Dict[str, Any]]:
        """Execute all assigned tasks"""
        results = []
        
        for agent in self.agents:
            for task in agent.assigned_tasks:
                result = agent.execute_task(task)
                results.append(result)
        
        return results


def configure_system_agent(state: MultiAgentSystemState) -> MultiAgentSystemState:
    """Configure multi-agent system"""
    print("\n⚙️ Configuring Multi-Agent System...")
    
    system_config = {
        "name": "Distributed Task Processing System",
        "num_agents": 5,
        "task_types": ["data_processing", "analysis", "reporting", "optimization"],
        "load_balancing": "capability-based"
    }
    
    agents_config = [
        {"id": "Agent_Data_1", "capabilities": ["data_processing", "analysis"], "capacity": 3},
        {"id": "Agent_Data_2", "capabilities": ["data_processing"], "capacity": 5},
        {"id": "Agent_Analytics", "capabilities": ["analysis", "reporting"], "capacity": 4},
        {"id": "Agent_Optimizer", "capabilities": ["optimization", "analysis"], "capacity": 2},
        {"id": "Agent_Reporter", "capabilities": ["reporting"], "capacity": 6}
    ]
    
    print(f"\n  System: {system_config['name']}")
    print(f"  Agents: {len(agents_config)}")
    print(f"  Supported Task Types: {system_config['task_types']}")
    
    print(f"\n  Agent Configuration:")
    for agent_conf in agents_config:
        print(f"    • {agent_conf['id']}")
        print(f"      Capabilities: {agent_conf['capabilities']}")
        print(f"      Capacity: {agent_conf['capacity']} concurrent tasks")
    
    return {
        **state,
        "system_config": system_config,
        "agents": agents_config,
        "messages": [f"✓ System configured with {len(agents_config)} agents"]
    }


def assign_tasks_agent(state: MultiAgentSystemState) -> MultiAgentSystemState:
    """Assign tasks to agents"""
    print("\n📋 Assigning Tasks...")
    
    tasks = [
        {"id": "T1", "name": "Process customer data", "required_capability": "data_processing", "complexity": 2},
        {"id": "T2", "name": "Analyze sales trends", "required_capability": "analysis", "complexity": 3},
        {"id": "T3", "name": "Generate monthly report", "required_capability": "reporting", "complexity": 1},
        {"id": "T4", "name": "Optimize delivery routes", "required_capability": "optimization", "complexity": 4},
        {"id": "T5", "name": "Process inventory data", "required_capability": "data_processing", "complexity": 2},
        {"id": "T6", "name": "Analyze customer feedback", "required_capability": "analysis", "complexity": 2},
        {"id": "T7", "name": "Generate quarterly report", "required_capability": "reporting", "complexity": 2},
        {"id": "T8", "name": "Process transactions", "required_capability": "data_processing", "complexity": 1}
    ]
    
    # Create coordinator and agents
    coordinator = MultiAgentCoordinator()
    for agent_config in state["agents"]:
        agent = Agent(
            agent_config["id"],
            agent_config["capabilities"],
            agent_config["capacity"]
        )
        coordinator.register_agent(agent)
    
    # Assign tasks
    assignments = coordinator.assign_tasks(tasks)
    
    print(f"\n  Total Tasks: {len(tasks)}")
    print(f"\n  Task Assignments:")
    for assignment in assignments:
        print(f"    • {assignment['task_name']} ({assignment['task_id']})")
        print(f"      Assigned to: {assignment['assigned_to']}")
        print(f"      Capability: {assignment['required_capability']}")
        print(f"      Complexity: {assignment.get('complexity', 'N/A')}")
    
    # Execute tasks
    results = coordinator.execute_all_tasks()
    
    print(f"\n  Execution Results: {len(results)} tasks completed")
    
    return {
        **state,
        "tasks": tasks,
        "task_assignments": assignments,
        "execution_results": results,
        "messages": [f"✓ {len(assignments)} tasks assigned, {len(results)} executed"]
    }


def generate_multiagent_report_agent(state: MultiAgentSystemState) -> MultiAgentSystemState:
    """Generate multi-agent system report"""
    print("\n" + "="*70)
    print("MULTI-AGENT SYSTEM REPORT")
    print("="*70)
    
    print(f"\n⚙️ System Configuration:")
    print(f"  Name: {state['system_config']['name']}")
    print(f"  Total Agents: {len(state['agents'])}")
    print(f"  Load Balancing: {state['system_config']['load_balancing']}")
    
    print(f"\n🤖 Registered Agents:")
    for agent in state["agents"]:
        print(f"\n  {agent['id']}:")
        print(f"    Capabilities: {', '.join(agent['capabilities'])}")
        print(f"    Capacity: {agent['capacity']} concurrent tasks")
    
    print(f"\n📋 Task Distribution:")
    print(f"  Total Tasks: {len(state['tasks'])}")
    
    # Group by assigned agent
    by_agent = {}
    for assignment in state["task_assignments"]:
        agent_id = assignment["assigned_to"]
        by_agent[agent_id] = by_agent.get(agent_id, []) + [assignment]
    
    for agent_id, agent_tasks in by_agent.items():
        print(f"\n  {agent_id}: {len(agent_tasks)} task(s)")
        for task in agent_tasks:
            print(f"    • {task['task_name']} (Complexity: {task.get('complexity', 'N/A')})")
    
    print(f"\n✅ Execution Results:")
    successful = [r for r in state["execution_results"] if r["status"] == "completed"]
    print(f"  Completed: {len(successful)}/{len(state['tasks'])}")
    
    for result in state["execution_results"]:
        print(f"\n  Task {result['task_id']}:")
        print(f"    Agent: {result['agent_id']}")
        print(f"    Status: {result['status']}")
        print(f"    Execution Time: {result['execution_time']:.2f}s")
        print(f"    Quality: {result['quality']:.0%}")
    
    print(f"\n📊 Performance Metrics:")
    if state["execution_results"]:
        avg_time = sum(r["execution_time"] for r in state["execution_results"]) / len(state["execution_results"])
        avg_quality = sum(r["quality"] for r in state["execution_results"]) / len(state["execution_results"])
        print(f"  Average Execution Time: {avg_time:.2f}s")
        print(f"  Average Quality: {avg_quality:.0%}")
        print(f"  Success Rate: {len(successful)/len(state['tasks']):.0%}")
    
    # Load analysis
    print(f"\n📈 Load Distribution:")
    for agent in state["agents"]:
        agent_results = [r for r in state["execution_results"] if r["agent_id"] == agent["id"]]
        load_pct = (len(agent_results) / agent["capacity"] * 100) if agent["capacity"] > 0 else 0
        print(f"  {agent['id']}: {len(agent_results)}/{agent['capacity']} tasks ({load_pct:.0f}% utilization)")
    
    print(f"\n💡 Multi-Agent System Benefits:")
    print("  • Distributed task processing")
    print("  • Capability-based routing")
    print("  • Load balancing")
    print("  • Fault tolerance")
    print("  • Scalable architecture")
    print("  • Parallel execution")
    print("  • Resource optimization")
    
    print("\n="*70)
    print("✅ Multi-Agent System Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_multiagent_system_graph():
    workflow = StateGraph(MultiAgentSystemState)
    workflow.add_node("configure", configure_system_agent)
    workflow.add_node("assign", assign_tasks_agent)
    workflow.add_node("report", generate_multiagent_report_agent)
    workflow.add_edge(START, "configure")
    workflow.add_edge("configure", "assign")
    workflow.add_edge("assign", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 266: Multi-Agent System MCP Pattern")
    print("="*70)
    
    app = create_multiagent_system_graph()
    final_state = app.invoke({
        "messages": [],
        "system_config": {},
        "agents": [],
        "tasks": [],
        "task_assignments": [],
        "execution_results": []
    })
    print("\n✅ Multi-Agent System Pattern Complete!")


if __name__ == "__main__":
    main()

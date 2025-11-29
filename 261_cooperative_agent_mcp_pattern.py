"""
Pattern 261: Cooperative Agent MCP Pattern

This pattern demonstrates cooperative agent behavior - multiple agents working
together towards common goals through collaboration and coordination.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CooperativeAgentState(TypedDict):
    """State for cooperative agent workflow"""
    messages: Annotated[List[str], add]
    shared_goal: str
    agents: List[Dict[str, Any]]
    collaboration_results: List[Dict[str, Any]]
    final_outcome: Dict[str, Any]


class CooperativeAgent:
    """Represents a cooperative agent"""
    
    def __init__(self, agent_id: str, specialty: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.specialty = specialty
        self.capabilities = capabilities
        self.contributions = []
    
    def can_contribute(self, task: str) -> bool:
        """Check if agent can contribute to task"""
        return any(cap.lower() in task.lower() for cap in self.capabilities)
    
    def contribute(self, task: str) -> Dict[str, Any]:
        """Make contribution to shared goal"""
        contribution = {
            "agent_id": self.agent_id,
            "specialty": self.specialty,
            "task": task,
            "status": "completed",
            "quality": 0.85 + (len(self.contributions) * 0.02)  # Improves with collaboration
        }
        self.contributions.append(contribution)
        return contribution


class CooperativeSystem:
    """Manages cooperative agent system"""
    
    def __init__(self):
        self.agents = [
            CooperativeAgent("A1", "Data Analysis", ["analyze", "data", "statistics"]),
            CooperativeAgent("A2", "Code Development", ["code", "implement", "develop"]),
            CooperativeAgent("A3", "Testing & QA", ["test", "validate", "quality"]),
            CooperativeAgent("A4", "Documentation", ["document", "write", "explain"])
        ]
    
    def assign_tasks(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Cooperatively assign and complete tasks"""
        results = []
        
        for task in tasks:
            # Find capable agents
            capable_agents = [a for a in self.agents if a.can_contribute(task)]
            
            if capable_agents:
                # Primary agent contributes
                primary = capable_agents[0]
                contribution = primary.contribute(task)
                
                # Supporting agents assist
                supporters = capable_agents[1:2]  # Max 1 supporter
                for supporter in supporters:
                    support = supporter.contribute(f"Support: {task}")
                    contribution["support"] = support
                
                results.append(contribution)
        
        return results


def initialize_cooperative_system_agent(state: CooperativeAgentState) -> CooperativeAgentState:
    """Initialize cooperative agent system"""
    print("\n🤝 Initializing Cooperative Agent System...")
    
    shared_goal = "Build and deploy ML model for customer churn prediction"
    
    system = CooperativeSystem()
    agents = [
        {
            "id": agent.agent_id,
            "specialty": agent.specialty,
            "capabilities": agent.capabilities
        }
        for agent in system.agents
    ]
    
    print(f"\n  Shared Goal: {shared_goal}")
    print(f"\n  Cooperative Agents: {len(agents)}")
    for agent in agents:
        print(f"    • {agent['id']}: {agent['specialty']}")
        print(f"      Capabilities: {', '.join(agent['capabilities'])}")
    
    return {
        **state,
        "shared_goal": shared_goal,
        "agents": agents,
        "messages": [f"✓ Initialized {len(agents)} cooperative agents"]
    }


def collaborate_on_tasks_agent(state: CooperativeAgentState) -> CooperativeAgentState:
    """Agents collaborate on tasks"""
    print("\n👥 Agents Collaborating on Tasks...")
    
    tasks = [
        "analyze customer data patterns",
        "develop ML model code",
        "test model accuracy",
        "document model usage"
    ]
    
    system = CooperativeSystem()
    collaboration_results = system.assign_tasks(tasks)
    
    print(f"\n  Tasks Completed: {len(collaboration_results)}")
    for result in collaboration_results:
        print(f"\n    Task: {result['task']}")
        print(f"    Primary Agent: {result['agent_id']} ({result['specialty']})")
        print(f"    Quality: {result['quality']:.1%}")
        if "support" in result:
            print(f"    Support: {result['support']['agent_id']}")
    
    return {
        **state,
        "collaboration_results": collaboration_results,
        "messages": [f"✓ Completed {len(collaboration_results)} collaborative tasks"]
    }


def synthesize_results_agent(state: CooperativeAgentState) -> CooperativeAgentState:
    """Synthesize collaborative results"""
    print("\n🔄 Synthesizing Collaborative Results...")
    
    total_tasks = len(state["collaboration_results"])
    avg_quality = sum(r["quality"] for r in state["collaboration_results"]) / total_tasks if total_tasks > 0 else 0
    
    # Calculate collaboration bonus (working together improves outcomes)
    collaboration_bonus = 0.10  # 10% improvement from cooperation
    final_quality = min(0.99, avg_quality + collaboration_bonus)
    
    final_outcome = {
        "goal_achieved": state["shared_goal"],
        "tasks_completed": total_tasks,
        "average_quality": avg_quality,
        "collaboration_bonus": collaboration_bonus,
        "final_quality": final_quality,
        "success": final_quality >= 0.85
    }
    
    print(f"\n  Goal: {final_outcome['goal_achieved']}")
    print(f"  Tasks Completed: {final_outcome['tasks_completed']}")
    print(f"  Base Quality: {avg_quality:.1%}")
    print(f"  Collaboration Bonus: +{collaboration_bonus:.1%}")
    print(f"  Final Quality: {final_quality:.1%}")
    print(f"  Success: {final_outcome['success']}")
    
    return {
        **state,
        "final_outcome": final_outcome,
        "messages": ["✓ Results synthesized"]
    }


def generate_cooperation_report_agent(state: CooperativeAgentState) -> CooperativeAgentState:
    """Generate cooperation report"""
    print("\n" + "="*70)
    print("COOPERATIVE AGENT REPORT")
    print("="*70)
    
    print(f"\n🎯 Shared Goal:")
    print(f"  {state['shared_goal']}")
    
    print(f"\n🤝 Participating Agents ({len(state['agents'])}):")
    for agent in state["agents"]:
        print(f"  • {agent['id']}: {agent['specialty']}")
    
    print(f"\n👥 Collaboration Results:")
    for i, result in enumerate(state["collaboration_results"], 1):
        print(f"\n  Task {i}: {result['task']}")
        print(f"    Agent: {result['agent_id']} ({result['specialty']})")
        print(f"    Quality: {result['quality']:.1%}")
        if "support" in result:
            print(f"    Supported by: {result['support']['agent_id']}")
    
    print(f"\n📊 Final Outcome:")
    outcome = state["final_outcome"]
    print(f"  Tasks Completed: {outcome['tasks_completed']}")
    print(f"  Base Quality: {outcome['average_quality']:.1%}")
    print(f"  Cooperation Bonus: +{outcome['collaboration_bonus']:.1%}")
    print(f"  Final Quality: {outcome['final_quality']:.1%}")
    print(f"  Goal Achieved: {'✅ YES' if outcome['success'] else '❌ NO'}")
    
    print("\n💡 Cooperative Agent Benefits:")
    print("  • Shared workload distribution")
    print("  • Complementary skills utilized")
    print("  • Quality improvement through collaboration")
    print("  • Faster goal achievement")
    print("  • Mutual support and assistance")
    print("  • Collective problem-solving")
    
    print("\n="*70)
    print("✅ Cooperative Agent Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_cooperative_agent_graph():
    workflow = StateGraph(CooperativeAgentState)
    workflow.add_node("initialize", initialize_cooperative_system_agent)
    workflow.add_node("collaborate", collaborate_on_tasks_agent)
    workflow.add_node("synthesize", synthesize_results_agent)
    workflow.add_node("report", generate_cooperation_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "collaborate")
    workflow.add_edge("collaborate", "synthesize")
    workflow.add_edge("synthesize", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 261: Cooperative Agent MCP Pattern")
    print("="*70)
    
    app = create_cooperative_agent_graph()
    final_state = app.invoke({
        "messages": [],
        "shared_goal": "",
        "agents": [],
        "collaboration_results": [],
        "final_outcome": {}
    })
    print("\n✅ Cooperative Agent Pattern Complete!")


if __name__ == "__main__":
    main()

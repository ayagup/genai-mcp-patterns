"""
Pattern 262: Competitive Agent MCP Pattern

This pattern demonstrates competitive agent dynamics where agents compete
for resources, tasks, or performance metrics, driving optimization through
competition.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CompetitiveAgentState(TypedDict):
    """State for competitive agent workflow"""
    messages: Annotated[List[str], add]
    task: Dict[str, Any]
    agents: List[Dict[str, Any]]
    competition_results: List[Dict[str, Any]]
    winner: Dict[str, Any]


class CompetitiveAgent:
    """Agent that competes with others"""
    
    def __init__(self, agent_id: str, strategy: str, capabilities: Dict[str, float]):
        self.agent_id = agent_id
        self.strategy = strategy
        self.capabilities = capabilities
        self.score = 0
        self.completed_tasks = []
    
    def compete(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Compete on a task"""
        task_requirements = task.get("requirements", {})
        
        # Calculate performance based on capability match
        performance_score = 0
        capability_match = {}
        
        for req, importance in task_requirements.items():
            capability = self.capabilities.get(req, 0)
            contribution = capability * importance
            performance_score += contribution
            capability_match[req] = {
                "capability": capability,
                "importance": importance,
                "contribution": contribution
            }
        
        # Strategy bonus
        strategy_bonus = self._apply_strategy(task)
        final_score = performance_score + strategy_bonus
        
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy,
            "performance_score": performance_score,
            "strategy_bonus": strategy_bonus,
            "final_score": final_score,
            "capability_match": capability_match,
            "time_taken": self._estimate_time(task)
        }
    
    def _apply_strategy(self, task: Dict[str, Any]) -> float:
        """Apply competitive strategy"""
        if self.strategy == "aggressive":
            return 0.15  # Risk more for higher reward
        elif self.strategy == "balanced":
            return 0.10  # Moderate approach
        elif self.strategy == "conservative":
            return 0.05  # Safe approach
        return 0.0
    
    def _estimate_time(self, task: Dict[str, Any]) -> float:
        """Estimate completion time"""
        base_time = task.get("complexity", 5)
        efficiency = sum(self.capabilities.values()) / max(len(self.capabilities), 1)
        return base_time / (efficiency + 0.1)


class CompetitionManager:
    """Manages agent competition"""
    
    def __init__(self):
        self.agents = []
        self.leaderboard = []
    
    def add_agent(self, agent: CompetitiveAgent):
        """Add agent to competition"""
        self.agents.append(agent)
    
    def run_competition(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run competition among agents"""
        results = []
        
        for agent in self.agents:
            result = agent.compete(task)
            results.append(result)
        
        # Sort by final score (descending)
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return results
    
    def declare_winner(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Declare competition winner"""
        if not results:
            return {}
        
        winner = results[0]
        margin = winner["final_score"] - results[1]["final_score"] if len(results) > 1 else 0
        
        return {
            "winner_id": winner["agent_id"],
            "strategy": winner["strategy"],
            "score": winner["final_score"],
            "margin": margin,
            "ranking": results
        }


def initialize_agents_agent(state: CompetitiveAgentState) -> CompetitiveAgentState:
    """Initialize competitive agents"""
    print("\n🤖 Initializing Competitive Agents...")
    
    agents_data = [
        {
            "id": "Agent Alpha",
            "strategy": "aggressive",
            "capabilities": {
                "speed": 0.95,
                "accuracy": 0.80,
                "efficiency": 0.85,
                "innovation": 0.90
            }
        },
        {
            "id": "Agent Beta",
            "strategy": "balanced",
            "capabilities": {
                "speed": 0.85,
                "accuracy": 0.90,
                "efficiency": 0.88,
                "innovation": 0.82
            }
        },
        {
            "id": "Agent Gamma",
            "strategy": "conservative",
            "capabilities": {
                "speed": 0.75,
                "accuracy": 0.95,
                "efficiency": 0.92,
                "innovation": 0.78
            }
        },
        {
            "id": "Agent Delta",
            "strategy": "aggressive",
            "capabilities": {
                "speed": 0.92,
                "accuracy": 0.78,
                "efficiency": 0.80,
                "innovation": 0.95
            }
        }
    ]
    
    print(f"\n  Registered Agents: {len(agents_data)}")
    for agent_data in agents_data:
        print(f"\n    • {agent_data['id']}")
        print(f"      Strategy: {agent_data['strategy']}")
        print(f"      Capabilities: {agent_data['capabilities']}")
    
    return {
        **state,
        "agents": agents_data,
        "messages": [f"✓ Initialized {len(agents_data)} competitive agents"]
    }


def run_competition_agent(state: CompetitiveAgentState) -> CompetitiveAgentState:
    """Run agent competition"""
    print("\n🏆 Running Competition...")
    
    task = {
        "name": "Data Processing Challenge",
        "requirements": {
            "speed": 0.30,
            "accuracy": 0.35,
            "efficiency": 0.20,
            "innovation": 0.15
        },
        "complexity": 8
    }
    
    print(f"\n  Task: {task['name']}")
    print(f"  Requirements:")
    for req, weight in task["requirements"].items():
        print(f"    • {req}: {weight:.0%}")
    
    # Create agents and manager
    manager = CompetitionManager()
    for agent_data in state["agents"]:
        agent = CompetitiveAgent(
            agent_data["id"],
            agent_data["strategy"],
            agent_data["capabilities"]
        )
        manager.add_agent(agent)
    
    # Run competition
    results = manager.run_competition(task)
    winner_info = manager.declare_winner(results)
    
    print(f"\n  Competition Results:")
    for i, result in enumerate(results, 1):
        print(f"\n    Rank {i}: {result['agent_id']}")
        print(f"      Strategy: {result['strategy']}")
        print(f"      Performance Score: {result['performance_score']:.3f}")
        print(f"      Strategy Bonus: {result['strategy_bonus']:.3f}")
        print(f"      Final Score: {result['final_score']:.3f}")
        print(f"      Est. Time: {result['time_taken']:.1f} hours")
    
    print(f"\n  🏆 Winner: {winner_info['winner_id']}")
    print(f"     Margin: {winner_info['margin']:.3f}")
    
    return {
        **state,
        "task": task,
        "competition_results": results,
        "winner": winner_info,
        "messages": [f"✓ Competition complete: {winner_info['winner_id']} wins"]
    }


def generate_competition_report_agent(state: CompetitiveAgentState) -> CompetitiveAgentState:
    """Generate competition report"""
    print("\n" + "="*70)
    print("COMPETITIVE AGENT REPORT")
    print("="*70)
    
    print(f"\n📋 Competition Task:")
    print(f"  Name: {state['task']['name']}")
    print(f"  Complexity: {state['task']['complexity']}")
    print(f"\n  Requirements:")
    for req, weight in state['task']['requirements'].items():
        print(f"    • {req}: {weight:.0%}")
    
    print(f"\n🤖 Participating Agents: {len(state['agents'])}")
    for agent in state["agents"]:
        print(f"\n  {agent['id']}:")
        print(f"    Strategy: {agent['strategy']}")
        for cap, value in agent["capabilities"].items():
            print(f"    {cap}: {value:.0%}")
    
    print(f"\n🏆 Competition Results:")
    for i, result in enumerate(state["competition_results"], 1):
        print(f"\n  Rank {i}: {result['agent_id']}")
        print(f"    Strategy: {result['strategy']}")
        print(f"    Performance: {result['performance_score']:.3f}")
        print(f"    Strategy Bonus: {result['strategy_bonus']:.3f}")
        print(f"    Final Score: {result['final_score']:.3f}")
        print(f"    Time Estimate: {result['time_taken']:.1f} hours")
        
        print(f"\n    Capability Breakdown:")
        for req, match in result["capability_match"].items():
            print(f"      {req}: capability={match['capability']:.2f}, "
                  f"importance={match['importance']:.2f}, "
                  f"contribution={match['contribution']:.3f}")
    
    winner = state["winner"]
    print(f"\n✅ Winner Announcement:")
    print(f"  🥇 Champion: {winner['winner_id']}")
    print(f"  Strategy: {winner['strategy']}")
    print(f"  Winning Score: {winner['score']:.3f}")
    print(f"  Victory Margin: {winner['margin']:.3f}")
    
    print(f"\n💡 Competitive Agent Benefits:")
    print("  • Drives performance optimization")
    print("  • Encourages innovation")
    print("  • Resource efficiency through selection")
    print("  • Natural quality improvement")
    print("  • Identifies best strategies")
    print("  • Motivates continuous improvement")
    
    print(f"\n📊 Competition Statistics:")
    scores = [r["final_score"] for r in state["competition_results"]]
    print(f"  Highest Score: {max(scores):.3f}")
    print(f"  Lowest Score: {min(scores):.3f}")
    print(f"  Average Score: {sum(scores)/len(scores):.3f}")
    print(f"  Score Range: {max(scores) - min(scores):.3f}")
    
    # Strategy analysis
    strategies = {}
    for result in state["competition_results"]:
        strategy = result["strategy"]
        strategies[strategy] = strategies.get(strategy, []) + [result["final_score"]]
    
    print(f"\n📈 Strategy Performance:")
    for strategy, scores in strategies.items():
        avg_score = sum(scores) / len(scores)
        print(f"  {strategy.title()}: Avg Score = {avg_score:.3f}")
    
    print("\n="*70)
    print("✅ Competitive Agent Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_competitive_agent_graph():
    workflow = StateGraph(CompetitiveAgentState)
    workflow.add_node("initialize", initialize_agents_agent)
    workflow.add_node("compete", run_competition_agent)
    workflow.add_node("report", generate_competition_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "compete")
    workflow.add_edge("compete", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 262: Competitive Agent MCP Pattern")
    print("="*70)
    
    app = create_competitive_agent_graph()
    final_state = app.invoke({
        "messages": [],
        "task": {},
        "agents": [],
        "competition_results": [],
        "winner": {}
    })
    print("\n✅ Competitive Agent Pattern Complete!")


if __name__ == "__main__":
    main()

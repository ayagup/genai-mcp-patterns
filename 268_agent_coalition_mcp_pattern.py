"""
Pattern 268: Agent Coalition MCP Pattern

This pattern demonstrates agent coalition formation where agents form
temporary alliances to achieve common goals.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class AgentCoalitionState(TypedDict):
    """State for agent coalition workflow"""
    messages: Annotated[List[str], add]
    opportunity: Dict[str, Any]
    available_agents: List[Dict[str, Any]]
    coalitions: List[Dict[str, Any]]
    coalition_performance: Dict[str, Any]


class CoalitionAgent:
    """Agent that can join coalitions"""
    
    def __init__(self, agent_id: str, resources: Dict[str, float], preferences: Dict[str, float]):
        self.agent_id = agent_id
        self.resources = resources
        self.preferences = preferences
        self.utility = 0.0
    
    def evaluate_coalition(self, coalition_members: List, opportunity: Dict[str, Any]) -> float:
        """Evaluate utility of joining a coalition"""
        required_resources = opportunity.get("required_resources", {})
        reward = opportunity.get("reward", 0)
        
        # Calculate coalition's total resources
        total_resources = {}
        for member in coalition_members:
            for resource, amount in member.resources.items():
                total_resources[resource] = total_resources.get(resource, 0) + amount
        
        # Check if coalition can fulfill opportunity
        can_fulfill = all(
            total_resources.get(res, 0) >= amt
            for res, amt in required_resources.items()
        )
        
        if not can_fulfill:
            return 0.0
        
        # Calculate utility: share of reward weighted by contribution
        my_contribution = sum(self.resources.values())
        total_contribution = sum(
            sum(m.resources.values()) for m in coalition_members
        )
        
        my_share = (my_contribution / total_contribution) if total_contribution > 0 else 0
        utility = my_share * reward
        
        # Apply preferences
        coalition_size = len(coalition_members)
        size_preference = self.preferences.get("coalition_size_preference", 0.5)
        utility *= (1.0 + size_preference * (1.0 / coalition_size))
        
        return utility


class CoalitionFormation:
    """Manages coalition formation"""
    
    def __init__(self):
        self.agents = []
    
    def register_agent(self, agent: CoalitionAgent):
        """Register agent"""
        self.agents.append(agent)
    
    def form_coalitions(self, opportunity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Form coalitions for opportunity"""
        required_resources = opportunity.get("required_resources", {})
        
        # Try to form minimal coalition
        coalitions = []
        
        # Greedy coalition formation
        remaining_agents = self.agents.copy()
        
        while remaining_agents:
            # Start new coalition with highest resource agent
            coalition = [remaining_agents.pop(0)]
            
            # Add agents until requirements met
            while not self._can_fulfill(coalition, required_resources):
                if not remaining_agents:
                    break
                
                # Find best agent to add
                best_agent = None
                best_utility = 0
                
                for agent in remaining_agents:
                    test_coalition = coalition + [agent]
                    utility = agent.evaluate_coalition(test_coalition, opportunity)
                    if utility > best_utility:
                        best_utility = utility
                        best_agent = agent
                
                if best_agent:
                    coalition.append(best_agent)
                    remaining_agents.remove(best_agent)
                else:
                    break
            
            # Check if coalition can fulfill opportunity
            if self._can_fulfill(coalition, required_resources):
                coalition_data = self._create_coalition_data(coalition, opportunity)
                coalitions.append(coalition_data)
                break
        
        return coalitions
    
    def _can_fulfill(self, coalition: List[CoalitionAgent], required_resources: Dict[str, float]) -> bool:
        """Check if coalition can fulfill requirements"""
        total_resources = {}
        for agent in coalition:
            for resource, amount in agent.resources.items():
                total_resources[resource] = total_resources.get(resource, 0) + amount
        
        return all(
            total_resources.get(res, 0) >= amt
            for res, amt in required_resources.items()
        )
    
    def _create_coalition_data(self, coalition: List[CoalitionAgent], opportunity: Dict[str, Any]) -> Dict[str, Any]:
        """Create coalition data structure"""
        members = []
        total_contribution = sum(sum(agent.resources.values()) for agent in coalition)
        reward = opportunity.get("reward", 0)
        
        for agent in coalition:
            contribution = sum(agent.resources.values())
            share = (contribution / total_contribution) if total_contribution > 0 else 0
            utility = share * reward
            
            members.append({
                "agent_id": agent.agent_id,
                "resources": agent.resources,
                "contribution": contribution,
                "share": share,
                "utility": utility
            })
        
        return {
            "coalition_id": f"Coalition_{len(coalition)}_agents",
            "members": members,
            "size": len(coalition),
            "total_resources": self._calculate_total_resources(coalition),
            "total_reward": reward,
            "formation_success": True
        }
    
    def _calculate_total_resources(self, coalition: List[CoalitionAgent]) -> Dict[str, float]:
        """Calculate total coalition resources"""
        total = {}
        for agent in coalition:
            for resource, amount in agent.resources.items():
                total[resource] = total.get(resource, 0) + amount
        return total


def identify_opportunity_agent(state: AgentCoalitionState) -> AgentCoalitionState:
    """Identify opportunity requiring coalition"""
    print("\n🎯 Identifying Opportunity...")
    
    opportunity = {
        "name": "Large Infrastructure Project",
        "description": "Deploy global cloud infrastructure",
        "required_resources": {
            "compute": 100,
            "storage": 80,
            "network": 60,
            "expertise": 40
        },
        "reward": 1000,
        "duration": "6 months"
    }
    
    print(f"\n  Opportunity: {opportunity['name']}")
    print(f"  Description: {opportunity['description']}")
    print(f"  Duration: {opportunity['duration']}")
    print(f"  Reward: ${opportunity['reward']}K")
    print(f"\n  Required Resources:")
    for resource, amount in opportunity["required_resources"].items():
        print(f"    • {resource}: {amount} units")
    
    return {
        **state,
        "opportunity": opportunity,
        "messages": ["✓ Opportunity identified"]
    }


def form_coalition_agent(state: AgentCoalitionState) -> AgentCoalitionState:
    """Form agent coalition"""
    print("\n🤝 Forming Coalition...")
    
    # Create agents
    agents_data = [
        {
            "id": "Agent_Cloud_Provider",
            "resources": {"compute": 60, "storage": 40, "network": 30},
            "preferences": {"coalition_size_preference": 0.2}
        },
        {
            "id": "Agent_Network_Specialist",
            "resources": {"network": 40, "expertise": 20},
            "preferences": {"coalition_size_preference": 0.3}
        },
        {
            "id": "Agent_Storage_Expert",
            "resources": {"storage": 50, "expertise": 15},
            "preferences": {"coalition_size_preference": 0.1}
        },
        {
            "id": "Agent_Consultant",
            "resources": {"expertise": 30, "compute": 20},
            "preferences": {"coalition_size_preference": 0.4}
        }
    ]
    
    print(f"\n  Available Agents: {len(agents_data)}")
    for agent_data in agents_data:
        print(f"\n    {agent_data['id']}:")
        print(f"      Resources: {agent_data['resources']}")
    
    # Form coalition
    formation = CoalitionFormation()
    for agent_data in agents_data:
        agent = CoalitionAgent(
            agent_data["id"],
            agent_data["resources"],
            agent_data["preferences"]
        )
        formation.register_agent(agent)
    
    coalitions = formation.form_coalitions(state["opportunity"])
    
    if coalitions:
        coalition = coalitions[0]
        print(f"\n  Coalition Formed:")
        print(f"    Coalition ID: {coalition['coalition_id']}")
        print(f"    Members: {coalition['size']}")
        print(f"    Total Reward: ${coalition['total_reward']}K")
        
        print(f"\n    Member Details:")
        for member in coalition["members"]:
            print(f"\n      {member['agent_id']}:")
            print(f"        Contribution: {member['contribution']:.1f} units")
            print(f"        Share: {member['share']:.0%}")
            print(f"        Utility: ${member['utility']:.1f}K")
    
    # Performance metrics
    performance = {
        "coalitions_formed": len(coalitions),
        "success_rate": 1.0 if coalitions and coalitions[0]["formation_success"] else 0.0,
        "average_coalition_size": sum(c["size"] for c in coalitions) / len(coalitions) if coalitions else 0
    }
    
    return {
        **state,
        "available_agents": agents_data,
        "coalitions": coalitions,
        "coalition_performance": performance,
        "messages": [f"✓ {len(coalitions)} coalition(s) formed"]
    }


def generate_coalition_report_agent(state: AgentCoalitionState) -> AgentCoalitionState:
    """Generate coalition report"""
    print("\n" + "="*70)
    print("AGENT COALITION REPORT")
    print("="*70)
    
    print(f"\n🎯 Opportunity:")
    opp = state["opportunity"]
    print(f"  Name: {opp['name']}")
    print(f"  Reward: ${opp['reward']}K")
    print(f"  Duration: {opp['duration']}")
    print(f"\n  Required Resources:")
    for resource, amount in opp["required_resources"].items():
        print(f"    • {resource}: {amount} units")
    
    print(f"\n🤖 Available Agents:")
    for agent in state["available_agents"]:
        print(f"\n  {agent['id']}:")
        print(f"    Resources:")
        for resource, amount in agent["resources"].items():
            print(f"      • {resource}: {amount} units")
    
    print(f"\n🤝 Formed Coalitions:")
    for coalition in state["coalitions"]:
        print(f"\n  {coalition['coalition_id']}:")
        print(f"    Members: {coalition['size']}")
        print(f"    Total Reward: ${coalition['total_reward']}K")
        
        print(f"\n    Total Coalition Resources:")
        for resource, amount in coalition["total_resources"].items():
            required = opp["required_resources"].get(resource, 0)
            status = "✓" if amount >= required else "✗"
            print(f"      {status} {resource}: {amount}/{required} units")
        
        print(f"\n    Member Contributions:")
        for member in coalition["members"]:
            print(f"\n      {member['agent_id']}:")
            print(f"        Resources: {member['resources']}")
            print(f"        Contribution: {member['contribution']:.1f} units")
            print(f"        Reward Share: {member['share']:.0%} (${member['utility']:.1f}K)")
    
    print(f"\n📊 Coalition Performance:")
    perf = state["coalition_performance"]
    print(f"  Coalitions Formed: {perf['coalitions_formed']}")
    print(f"  Success Rate: {perf['success_rate']:.0%}")
    print(f"  Average Size: {perf['average_coalition_size']:.1f} agents")
    
    print(f"\n💡 Agent Coalition Benefits:")
    print("  • Resource pooling")
    print("  • Risk sharing")
    print("  • Capability combination")
    print("  • Economies of scale")
    print("  • Flexible partnerships")
    print("  • Goal alignment")
    print("  • Fair reward distribution")
    
    print("\n="*70)
    print("✅ Agent Coalition Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_agent_coalition_graph():
    workflow = StateGraph(AgentCoalitionState)
    workflow.add_node("identify", identify_opportunity_agent)
    workflow.add_node("form", form_coalition_agent)
    workflow.add_node("report", generate_coalition_report_agent)
    workflow.add_edge(START, "identify")
    workflow.add_edge("identify", "form")
    workflow.add_edge("form", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 268: Agent Coalition MCP Pattern")
    print("="*70)
    
    app = create_agent_coalition_graph()
    final_state = app.invoke({
        "messages": [],
        "opportunity": {},
        "available_agents": [],
        "coalitions": [],
        "coalition_performance": {}
    })
    print("\n✅ Agent Coalition Pattern Complete!")


if __name__ == "__main__":
    main()

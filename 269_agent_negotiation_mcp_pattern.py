"""
Pattern 269: Agent Negotiation MCP Pattern

This pattern demonstrates agent negotiation where agents negotiate
terms, resources, and agreements to reach mutually beneficial outcomes.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class AgentNegotiationState(TypedDict):
    """State for agent negotiation workflow"""
    messages: Annotated[List[str], add]
    negotiation_context: Dict[str, Any]
    agents: List[Dict[str, Any]]
    negotiation_rounds: List[Dict[str, Any]]
    final_agreement: Dict[str, Any]


class NegotiatingAgent:
    """Agent that can negotiate"""
    
    def __init__(self, agent_id: str, initial_position: Dict[str, float], constraints: Dict[str, Any]):
        self.agent_id = agent_id
        self.initial_position = initial_position
        self.current_position = initial_position.copy()
        self.constraints = constraints
        self.concession_rate = 0.1
    
    def make_offer(self, round_num: int) -> Dict[str, Any]:
        """Make an offer in negotiation"""
        return {
            "agent_id": self.agent_id,
            "round": round_num,
            "offer": self.current_position.copy(),
            "flexibility": self._calculate_flexibility()
        }
    
    def evaluate_offer(self, offer: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate counter-offer"""
        utility = self._calculate_utility(offer)
        acceptable = utility >= self.constraints.get("min_utility", 0.5)
        
        return {
            "utility": utility,
            "acceptable": acceptable,
            "gap": self._calculate_gap(offer)
        }
    
    def make_concession(self, counter_offer: Dict[str, float]):
        """Make concession toward counter-offer"""
        for key in self.current_position:
            if key in counter_offer:
                current = self.current_position[key]
                target = counter_offer[key]
                # Move toward counter-offer
                self.current_position[key] = current + (target - current) * self.concession_rate
    
    def _calculate_utility(self, offer: Dict[str, float]) -> float:
        """Calculate utility of an offer"""
        preferences = self.constraints.get("preferences", {})
        total_utility = 0
        
        for key, value in offer.items():
            preference = preferences.get(key, 0.5)
            # Higher preference = want higher value
            total_utility += value * preference
        
        return total_utility / max(len(offer), 1)
    
    def _calculate_flexibility(self) -> float:
        """Calculate how flexible agent is"""
        total_movement = 0
        for key in self.current_position:
            initial = self.initial_position[key]
            current = self.current_position[key]
            total_movement += abs(current - initial)
        
        return total_movement / max(len(self.current_position), 1)
    
    def _calculate_gap(self, offer: Dict[str, float]) -> float:
        """Calculate gap between positions"""
        total_gap = 0
        for key in self.current_position:
            if key in offer:
                total_gap += abs(self.current_position[key] - offer[key])
        
        return total_gap / max(len(self.current_position), 1)


class NegotiationProtocol:
    """Manages negotiation protocol"""
    
    def __init__(self, max_rounds: int = 10):
        self.max_rounds = max_rounds
        self.agents = []
        self.history = []
    
    def add_agent(self, agent: NegotiatingAgent):
        """Add negotiating agent"""
        self.agents.append(agent)
    
    def conduct_negotiation(self) -> List[Dict[str, Any]]:
        """Conduct negotiation rounds"""
        rounds = []
        
        for round_num in range(1, self.max_rounds + 1):
            # Each agent makes offer
            offers = []
            for agent in self.agents:
                offer = agent.make_offer(round_num)
                offers.append(offer)
            
            # Calculate average position (compromise)
            compromise = self._calculate_compromise(offers)
            
            # Agents evaluate compromise
            evaluations = []
            all_accept = True
            
            for agent in self.agents:
                eval_result = agent.evaluate_offer(compromise)
                evaluations.append({
                    "agent_id": agent.agent_id,
                    "utility": eval_result["utility"],
                    "acceptable": eval_result["acceptable"],
                    "gap": eval_result["gap"]
                })
                
                if not eval_result["acceptable"]:
                    all_accept = False
                    # Make concession
                    agent.make_concession(compromise)
            
            round_data = {
                "round": round_num,
                "offers": offers,
                "compromise": compromise,
                "evaluations": evaluations,
                "agreement_reached": all_accept
            }
            rounds.append(round_data)
            
            # Stop if agreement reached
            if all_accept:
                break
        
        return rounds
    
    def _calculate_compromise(self, offers: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate compromise position"""
        if not offers:
            return {}
        
        # Average all offers
        compromise = {}
        all_keys = set()
        for offer in offers:
            all_keys.update(offer["offer"].keys())
        
        for key in all_keys:
            values = [offer["offer"][key] for offer in offers if key in offer["offer"]]
            compromise[key] = sum(values) / len(values) if values else 0
        
        return compromise


def setup_negotiation_agent(state: AgentNegotiationState) -> AgentNegotiationState:
    """Setup negotiation context"""
    print("\n🤝 Setting Up Negotiation...")
    
    negotiation_context = {
        "topic": "Service Agreement",
        "parameters": ["price", "quality", "delivery_time", "support_level"],
        "duration": "Multi-round negotiation"
    }
    
    agents_config = [
        {
            "id": "Buyer_Agent",
            "role": "buyer",
            "initial_position": {
                "price": 0.3,  # Low price preferred (0-1 scale)
                "quality": 0.9,  # High quality wanted
                "delivery_time": 0.2,  # Fast delivery wanted
                "support_level": 0.8  # Good support wanted
            },
            "constraints": {
                "min_utility": 0.6,
                "preferences": {
                    "price": -1.0,  # Negative = want lower
                    "quality": 1.0,  # Positive = want higher
                    "delivery_time": -0.5,
                    "support_level": 0.8
                }
            }
        },
        {
            "id": "Seller_Agent",
            "role": "seller",
            "initial_position": {
                "price": 0.8,  # High price preferred
                "quality": 0.7,  # Moderate quality offered
                "delivery_time": 0.6,  # Moderate delivery time
                "support_level": 0.6  # Standard support
            },
            "constraints": {
                "min_utility": 0.55,
                "preferences": {
                    "price": 1.0,  # Want higher price
                    "quality": -0.5,  # Prefer not to over-deliver
                    "delivery_time": 0.3,
                    "support_level": -0.4  # Prefer less support commitment
                }
            }
        }
    ]
    
    print(f"\n  Topic: {negotiation_context['topic']}")
    print(f"  Parameters: {', '.join(negotiation_context['parameters'])}")
    print(f"\n  Negotiating Parties:")
    for agent in agents_config:
        print(f"\n    {agent['id']} ({agent['role']}):")
        print(f"      Initial Position: {agent['initial_position']}")
    
    return {
        **state,
        "negotiation_context": negotiation_context,
        "agents": agents_config,
        "messages": ["✓ Negotiation setup complete"]
    }


def negotiate_agent(state: AgentNegotiationState) -> AgentNegotiationState:
    """Conduct negotiation"""
    print("\n🗣️ Conducting Negotiation...")
    
    # Create negotiation protocol
    protocol = NegotiationProtocol(max_rounds=10)
    
    # Create agents
    for agent_config in state["agents"]:
        agent = NegotiatingAgent(
            agent_config["id"],
            agent_config["initial_position"],
            agent_config["constraints"]
        )
        protocol.add_agent(agent)
    
    # Conduct negotiation
    rounds = protocol.conduct_negotiation()
    
    print(f"\n  Negotiation Rounds: {len(rounds)}")
    
    # Show key rounds
    for round_data in rounds:
        print(f"\n  Round {round_data['round']}:")
        print(f"    Compromise: {round_data['compromise']}")
        print(f"    Agreement: {'✓ Reached' if round_data['agreement_reached'] else '✗ Not yet'}")
        
        for eval_item in round_data["evaluations"]:
            print(f"      {eval_item['agent_id']}: "
                  f"Utility={eval_item['utility']:.2f}, "
                  f"Gap={eval_item['gap']:.2f}, "
                  f"Accept={'Yes' if eval_item['acceptable'] else 'No'}")
    
    # Final agreement
    final_round = rounds[-1] if rounds else {}
    agreement = {
        "reached": final_round.get("agreement_reached", False),
        "rounds_taken": len(rounds),
        "final_terms": final_round.get("compromise", {}),
        "all_parties_satisfied": final_round.get("agreement_reached", False)
    }
    
    if agreement["reached"]:
        print(f"\n  ✅ Agreement Reached in {agreement['rounds_taken']} rounds!")
    else:
        print(f"\n  ❌ No agreement after {agreement['rounds_taken']} rounds")
    
    return {
        **state,
        "negotiation_rounds": rounds,
        "final_agreement": agreement,
        "messages": [f"✓ Negotiation completed in {len(rounds)} rounds"]
    }


def generate_negotiation_report_agent(state: AgentNegotiationState) -> AgentNegotiationState:
    """Generate negotiation report"""
    print("\n" + "="*70)
    print("AGENT NEGOTIATION REPORT")
    print("="*70)
    
    print(f"\n📋 Negotiation Context:")
    ctx = state["negotiation_context"]
    print(f"  Topic: {ctx['topic']}")
    print(f"  Parameters: {', '.join(ctx['parameters'])}")
    
    print(f"\n👥 Negotiating Parties:")
    for agent in state["agents"]:
        print(f"\n  {agent['id']} ({agent['role']}):")
        print(f"    Initial Position:")
        for param, value in agent["initial_position"].items():
            print(f"      • {param}: {value:.2f}")
        print(f"    Minimum Acceptable Utility: {agent['constraints']['min_utility']:.2f}")
    
    print(f"\n🗣️ Negotiation Process:")
    for round_data in state["negotiation_rounds"]:
        print(f"\n  Round {round_data['round']}:")
        print(f"    Proposed Compromise:")
        for param, value in round_data["compromise"].items():
            print(f"      • {param}: {value:.2f}")
        
        print(f"\n    Agent Evaluations:")
        for eval_item in round_data["evaluations"]:
            status = "✓ Acceptable" if eval_item["acceptable"] else "✗ Not acceptable"
            print(f"      {eval_item['agent_id']}: {status}")
            print(f"        Utility: {eval_item['utility']:.2f}")
            print(f"        Position Gap: {eval_item['gap']:.2f}")
        
        if round_data["agreement_reached"]:
            print(f"\n    🎉 Agreement Reached!")
            break
    
    print(f"\n✅ Final Agreement:")
    agreement = state["final_agreement"]
    print(f"  Status: {'Reached' if agreement['reached'] else 'Not Reached'}")
    print(f"  Rounds Required: {agreement['rounds_taken']}")
    
    if agreement["reached"]:
        print(f"\n  Agreed Terms:")
        for param, value in agreement["final_terms"].items():
            print(f"    • {param}: {value:.2f}")
        
        # Show how final terms compare to initial positions
        print(f"\n  Position Changes:")
        for agent in state["agents"]:
            print(f"\n    {agent['id']}:")
            for param in agent["initial_position"]:
                initial = agent["initial_position"][param]
                final_val = agreement["final_terms"].get(param, initial)
                change = final_val - initial
                direction = "→" if abs(change) < 0.05 else ("↑" if change > 0 else "↓")
                print(f"      {param}: {initial:.2f} {direction} {final_val:.2f} (Δ{change:+.2f})")
    
    print(f"\n💡 Agent Negotiation Benefits:")
    print("  • Automated conflict resolution")
    print("  • Fair outcome through compromise")
    print("  • Efficient agreement process")
    print("  • Transparent decision-making")
    print("  • Mutually beneficial outcomes")
    print("  • Reduced human intervention")
    
    print(f"\n📊 Negotiation Metrics:")
    if state["negotiation_rounds"]:
        total_rounds = len(state["negotiation_rounds"])
        print(f"  Total Rounds: {total_rounds}")
        print(f"  Efficiency: {'High' if total_rounds <= 5 else 'Moderate' if total_rounds <= 8 else 'Low'}")
        print(f"  Outcome: {'Success' if agreement['reached'] else 'Stalemate'}")
        
        # Calculate convergence
        if len(state["negotiation_rounds"]) > 1:
            first_gap = state["negotiation_rounds"][0]["evaluations"][0]["gap"]
            last_gap = state["negotiation_rounds"][-1]["evaluations"][0]["gap"]
            convergence = (first_gap - last_gap) / first_gap if first_gap > 0 else 0
            print(f"  Convergence Rate: {convergence:.0%}")
    
    print("\n="*70)
    print("✅ Agent Negotiation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_agent_negotiation_graph():
    workflow = StateGraph(AgentNegotiationState)
    workflow.add_node("setup", setup_negotiation_agent)
    workflow.add_node("negotiate", negotiate_agent)
    workflow.add_node("report", generate_negotiation_report_agent)
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "negotiate")
    workflow.add_edge("negotiate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 269: Agent Negotiation MCP Pattern")
    print("="*70)
    
    app = create_agent_negotiation_graph()
    final_state = app.invoke({
        "messages": [],
        "negotiation_context": {},
        "agents": [],
        "negotiation_rounds": [],
        "final_agreement": {}
    })
    print("\n✅ Agent Negotiation Pattern Complete!")


if __name__ == "__main__":
    main()

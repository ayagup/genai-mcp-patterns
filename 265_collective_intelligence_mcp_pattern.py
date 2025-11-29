"""
Pattern 265: Collective Intelligence MCP Pattern

This pattern demonstrates collective intelligence - aggregating knowledge
and decisions from multiple agents to achieve superior outcomes.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CollectiveIntelligenceState(TypedDict):
    """State for collective intelligence workflow"""
    messages: Annotated[List[str], add]
    problem: Dict[str, Any]
    agent_contributions: List[Dict[str, Any]]
    aggregated_knowledge: Dict[str, Any]
    collective_decision: Dict[str, Any]


class IntelligentAgent:
    """Agent contributing to collective intelligence"""
    
    def __init__(self, agent_id: str, expertise: str, knowledge_base: Dict[str, Any]):
        self.agent_id = agent_id
        self.expertise = expertise
        self.knowledge_base = knowledge_base
    
    def contribute(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Contribute knowledge and analysis"""
        problem_domain = problem.get("domain", "")
        
        # Relevance based on expertise match
        relevance = 0.9 if problem_domain == self.expertise else 0.5
        
        # Generate insights
        insights = self._generate_insights(problem)
        
        # Provide recommendation
        recommendation = self._make_recommendation(problem)
        
        # Confidence based on relevance and knowledge
        confidence = relevance * (len(insights) / 5)
        
        return {
            "agent_id": self.agent_id,
            "expertise": self.expertise,
            "relevance": relevance,
            "insights": insights,
            "recommendation": recommendation,
            "confidence": min(confidence, 1.0)
        }
    
    def _generate_insights(self, problem: Dict[str, Any]) -> List[str]:
        """Generate domain-specific insights"""
        insights = []
        
        # Based on knowledge base
        for key, value in self.knowledge_base.items():
            if key in str(problem):
                insights.append(f"{self.expertise} perspective: {value}")
        
        # Generic insights based on expertise
        if self.expertise == "Technical":
            insights.extend([
                "System scalability needs consideration",
                "Performance optimization is critical"
            ])
        elif self.expertise == "Business":
            insights.extend([
                "ROI should be primary metric",
                "Market timing is crucial"
            ])
        elif self.expertise == "User Experience":
            insights.extend([
                "User adoption drives success",
                "Simplicity beats complexity"
            ])
        elif self.expertise == "Security":
            insights.extend([
                "Data protection is non-negotiable",
                "Compliance requirements must be met"
            ])
        
        return insights[:5]
    
    def _make_recommendation(self, problem: Dict[str, Any]) -> str:
        """Make domain-specific recommendation"""
        recommendations = {
            "Technical": "Implement scalable microservices architecture",
            "Business": "Prioritize features with highest ROI",
            "User Experience": "Focus on intuitive user interface",
            "Security": "Implement zero-trust security model"
        }
        return recommendations.get(self.expertise, "Analyze from multiple angles")


class CollectiveIntelligenceSystem:
    """Aggregates intelligence from multiple agents"""
    
    def __init__(self):
        self.agents = []
    
    def add_agent(self, agent: IntelligentAgent):
        """Add agent to collective"""
        self.agents.append(agent)
    
    def gather_contributions(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gather contributions from all agents"""
        contributions = []
        for agent in self.agents:
            contribution = agent.contribute(problem)
            contributions.append(contribution)
        return contributions
    
    def aggregate_knowledge(self, contributions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate knowledge from contributions"""
        all_insights = []
        all_recommendations = []
        total_confidence = 0
        
        for contrib in contributions:
            all_insights.extend(contrib["insights"])
            all_recommendations.append({
                "source": contrib["agent_id"],
                "expertise": contrib["expertise"],
                "recommendation": contrib["recommendation"],
                "confidence": contrib["confidence"]
            })
            total_confidence += contrib["confidence"]
        
        # Remove duplicate insights
        unique_insights = list(set(all_insights))
        
        return {
            "total_insights": len(unique_insights),
            "insights": unique_insights,
            "recommendations": all_recommendations,
            "collective_confidence": total_confidence / len(contributions) if contributions else 0,
            "consensus_level": self._calculate_consensus(contributions)
        }
    
    def _calculate_consensus(self, contributions: List[Dict[str, Any]]) -> float:
        """Calculate level of consensus among agents"""
        if len(contributions) < 2:
            return 1.0
        
        # Simple consensus based on confidence variance
        confidences = [c["confidence"] for c in contributions]
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence)**2 for c in confidences) / len(confidences)
        
        # Lower variance = higher consensus
        consensus = 1.0 / (1.0 + variance)
        return consensus
    
    def make_collective_decision(self, aggregated_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Make decision based on collective intelligence"""
        recommendations = aggregated_knowledge["recommendations"]
        
        # Weight by confidence
        weighted_scores = {}
        for rec in recommendations:
            expertise = rec["expertise"]
            confidence = rec["confidence"]
            weighted_scores[expertise] = weighted_scores.get(expertise, 0) + confidence
        
        # Find highest weighted recommendation
        best_expertise = max(weighted_scores, key=weighted_scores.get) if weighted_scores else "Unknown"
        best_recommendation = next(
            (r["recommendation"] for r in recommendations if r["expertise"] == best_expertise),
            "No consensus reached"
        )
        
        return {
            "decision": best_recommendation,
            "primary_expertise": best_expertise,
            "confidence": aggregated_knowledge["collective_confidence"],
            "consensus_level": aggregated_knowledge["consensus_level"],
            "supporting_insights": len(aggregated_knowledge["insights"]),
            "contributing_agents": len(recommendations)
        }


def initialize_collective_agent(state: CollectiveIntelligenceState) -> CollectiveIntelligenceState:
    """Initialize collective intelligence system"""
    print("\n🧠 Initializing Collective Intelligence...")
    
    problem = {
        "domain": "Technical",
        "description": "Design new e-commerce platform",
        "requirements": [
            "Scalable architecture",
            "Secure payment processing",
            "Excellent user experience",
            "Cost-effective solution"
        ]
    }
    
    print(f"\n  Problem Domain: {problem['domain']}")
    print(f"  Description: {problem['description']}")
    print(f"  Requirements:")
    for req in problem["requirements"]:
        print(f"    • {req}")
    
    return {
        **state,
        "problem": problem,
        "messages": ["✓ Problem initialized"]
    }


def gather_agent_contributions_agent(state: CollectiveIntelligenceState) -> CollectiveIntelligenceState:
    """Gather contributions from agents"""
    print("\n👥 Gathering Agent Contributions...")
    
    # Create collective intelligence system
    collective = CollectiveIntelligenceSystem()
    
    # Add agents with different expertise
    agents_data = [
        ("Agent_Tech", "Technical", {
            "architecture": "Microservices provide flexibility",
            "scalability": "Horizontal scaling is key"
        }),
        ("Agent_Business", "Business", {
            "roi": "Focus on user acquisition cost",
            "market": "Time-to-market is competitive advantage"
        }),
        ("Agent_UX", "User Experience", {
            "design": "Mobile-first approach essential",
            "usability": "Reduce friction in checkout flow"
        }),
        ("Agent_Security", "Security", {
            "payment": "PCI DSS compliance required",
            "data": "Encrypt all sensitive data"
        })
    ]
    
    for agent_id, expertise, knowledge in agents_data:
        agent = IntelligentAgent(agent_id, expertise, knowledge)
        collective.add_agent(agent)
    
    # Gather contributions
    contributions = collective.gather_contributions(state["problem"])
    
    print(f"\n  Contributions Received: {len(contributions)}")
    for contrib in contributions:
        print(f"\n    {contrib['agent_id']} ({contrib['expertise']}):")
        print(f"      Relevance: {contrib['relevance']:.0%}")
        print(f"      Confidence: {contrib['confidence']:.0%}")
        print(f"      Insights: {len(contrib['insights'])}")
        print(f"      Recommendation: {contrib['recommendation']}")
    
    return {
        **state,
        "agent_contributions": contributions,
        "messages": [f"✓ Gathered {len(contributions)} contributions"]
    }


def aggregate_collective_knowledge_agent(state: CollectiveIntelligenceState) -> CollectiveIntelligenceState:
    """Aggregate collective knowledge and make decision"""
    print("\n🔄 Aggregating Collective Knowledge...")
    
    collective = CollectiveIntelligenceSystem()
    
    # Aggregate knowledge
    aggregated = collective.aggregate_knowledge(state["agent_contributions"])
    
    print(f"\n  Knowledge Aggregation:")
    print(f"    Total Unique Insights: {aggregated['total_insights']}")
    print(f"    Collective Confidence: {aggregated['collective_confidence']:.0%}")
    print(f"    Consensus Level: {aggregated['consensus_level']:.0%}")
    
    # Make collective decision
    decision = collective.make_collective_decision(aggregated)
    
    print(f"\n  Collective Decision:")
    print(f"    Decision: {decision['decision']}")
    print(f"    Primary Expertise: {decision['primary_expertise']}")
    print(f"    Confidence: {decision['confidence']:.0%}")
    print(f"    Consensus: {decision['consensus_level']:.0%}")
    print(f"    Supporting Insights: {decision['supporting_insights']}")
    
    return {
        **state,
        "aggregated_knowledge": aggregated,
        "collective_decision": decision,
        "messages": ["✓ Collective decision made"]
    }


def generate_collective_report_agent(state: CollectiveIntelligenceState) -> CollectiveIntelligenceState:
    """Generate collective intelligence report"""
    print("\n" + "="*70)
    print("COLLECTIVE INTELLIGENCE REPORT")
    print("="*70)
    
    print(f"\n📋 Problem:")
    print(f"  Domain: {state['problem']['domain']}")
    print(f"  Description: {state['problem']['description']}")
    print(f"  Requirements:")
    for req in state["problem"]["requirements"]:
        print(f"    • {req}")
    
    print(f"\n👥 Agent Contributions:")
    for contrib in state["agent_contributions"]:
        print(f"\n  {contrib['agent_id']} - {contrib['expertise']} Expert:")
        print(f"    Relevance: {contrib['relevance']:.0%}")
        print(f"    Confidence: {contrib['confidence']:.0%}")
        print(f"    Recommendation: {contrib['recommendation']}")
        print(f"    Key Insights:")
        for insight in contrib["insights"][:3]:
            print(f"      • {insight}")
    
    print(f"\n🧠 Aggregated Collective Knowledge:")
    agg = state["aggregated_knowledge"]
    print(f"  Total Unique Insights: {agg['total_insights']}")
    print(f"  Collective Confidence: {agg['collective_confidence']:.0%}")
    print(f"  Consensus Level: {agg['consensus_level']:.0%}")
    
    print(f"\n  All Insights:")
    for i, insight in enumerate(agg["insights"][:10], 1):
        print(f"    {i}. {insight}")
    if len(agg["insights"]) > 10:
        print(f"    ... and {len(agg['insights']) - 10} more")
    
    print(f"\n✅ Collective Decision:")
    decision = state["collective_decision"]
    print(f"  Decision: {decision['decision']}")
    print(f"  Led by: {decision['primary_expertise']} expertise")
    print(f"  Confidence: {decision['confidence']:.0%}")
    print(f"  Consensus Level: {decision['consensus_level']:.0%}")
    print(f"  Based on {decision['supporting_insights']} insights")
    print(f"  From {decision['contributing_agents']} agents")
    
    print(f"\n💡 Collective Intelligence Benefits:")
    print("  • Diverse perspectives")
    print("  • Comprehensive analysis")
    print("  • Reduced individual bias")
    print("  • Higher quality decisions")
    print("  • Knowledge synthesis")
    print("  • Emergent insights")
    print("  • Robust conclusions")
    
    print(f"\n📊 Decision Quality Indicators:")
    if decision['consensus_level'] >= 0.8:
        print("  ✅ High Consensus - Strong agreement among agents")
    elif decision['consensus_level'] >= 0.6:
        print("  ⚠️ Moderate Consensus - Some disagreement exists")
    else:
        print("  ❌ Low Consensus - Significant disagreement")
    
    if decision['confidence'] >= 0.8:
        print("  ✅ High Confidence - Agents are very confident")
    elif decision['confidence'] >= 0.6:
        print("  ⚠️ Moderate Confidence - Reasonable certainty")
    else:
        print("  ❌ Low Confidence - Uncertain recommendation")
    
    print("\n="*70)
    print("✅ Collective Intelligence Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_collective_intelligence_graph():
    workflow = StateGraph(CollectiveIntelligenceState)
    workflow.add_node("initialize", initialize_collective_agent)
    workflow.add_node("gather", gather_agent_contributions_agent)
    workflow.add_node("aggregate", aggregate_collective_knowledge_agent)
    workflow.add_node("report", generate_collective_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "gather")
    workflow.add_edge("gather", "aggregate")
    workflow.add_edge("aggregate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 265: Collective Intelligence MCP Pattern")
    print("="*70)
    
    app = create_collective_intelligence_graph()
    final_state = app.invoke({
        "messages": [],
        "problem": {},
        "agent_contributions": [],
        "aggregated_knowledge": {},
        "collective_decision": {}
    })
    print("\n✅ Collective Intelligence Pattern Complete!")


if __name__ == "__main__":
    main()

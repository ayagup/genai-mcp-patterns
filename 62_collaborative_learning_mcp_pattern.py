"""
Collaborative Learning MCP Pattern

This pattern demonstrates multiple agents learning together, sharing knowledge
and insights to collectively improve their performance.

Key Features:
- Multi-agent learning
- Knowledge sharing
- Collaborative insights
- Collective improvement
- Shared experience pool
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class CollaborativeLearningState(TypedDict):
    """State for collaborative learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    shared_experiences: list[dict[str, str]]
    shared_knowledge: dict[str, list[str]]
    agent_insights: dict[str, list[str]]
    collaborative_patterns: list[str]
    team_performance: float
    learning_round: int


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Agent A - Data Specialist
def agent_a_learn(state: CollaborativeLearningState) -> CollaborativeLearningState:
    """Agent A learns and shares data-related insights"""
    task = state.get("task", "")
    shared_knowledge = state.get("shared_knowledge", {})
    learning_round = state.get("learning_round", 0)
    
    system_message = SystemMessage(content="""You are Agent A, a data specialist. 
    Learn from tasks and share data-related insights with the team.""")
    
    # Review shared knowledge from other agents
    team_knowledge = "\n".join([
        f"- {item}" for items in shared_knowledge.values() for item in items[:2]
    ]) if shared_knowledge else "No shared knowledge yet"
    
    user_message = HumanMessage(content=f"""Task: {task}

Team Knowledge:
{team_knowledge}

As a data specialist, provide insights about data handling, analysis, and quality.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate insights
    insights = [
        f"Round {learning_round + 1}: Data validation improves accuracy",
        f"Round {learning_round + 1}: Structured data formats enhance processing",
        f"Round {learning_round + 1}: Data profiling reveals patterns"
    ]
    
    # Create experience record
    experience = {
        "agent": "Agent A (Data)",
        "round": str(learning_round + 1),
        "insight": insights[0],
        "contribution": "Data handling expertise"
    }
    
    return {
        "messages": [AIMessage(content=f"👤 Agent A (Data Specialist):\n{response.content}\n\n💡 Insights: {len(insights)}")],
        "shared_experiences": [experience],
        "agent_insights": {"agent_a": insights}
    }


# Agent B - Algorithm Specialist
def agent_b_learn(state: CollaborativeLearningState) -> CollaborativeLearningState:
    """Agent B learns and shares algorithm-related insights"""
    task = state.get("task", "")
    shared_knowledge = state.get("shared_knowledge", {})
    agent_insights = state.get("agent_insights", {})
    learning_round = state.get("learning_round", 0)
    
    system_message = SystemMessage(content="""You are Agent B, an algorithm specialist. 
    Learn from tasks and share algorithm-related insights with the team.""")
    
    # Review Agent A's insights
    agent_a_insights = agent_insights.get("agent_a", [])
    previous_insights = "\n".join([f"- {insight}" for insight in agent_a_insights[:2]]) if agent_a_insights else "None yet"
    
    user_message = HumanMessage(content=f"""Task: {task}

Agent A's Insights:
{previous_insights}

As an algorithm specialist, provide insights about algorithms, optimization, and efficiency.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate insights
    insights = [
        f"Round {learning_round + 1}: Algorithm selection impacts performance",
        f"Round {learning_round + 1}: Optimization reduces complexity",
        f"Round {learning_round + 1}: Parallel processing increases throughput"
    ]
    
    # Create experience record
    experience = {
        "agent": "Agent B (Algorithm)",
        "round": str(learning_round + 1),
        "insight": insights[0],
        "contribution": "Algorithm optimization"
    }
    
    return {
        "messages": [AIMessage(content=f"👤 Agent B (Algorithm Specialist):\n{response.content}\n\n💡 Insights: {len(insights)}")],
        "shared_experiences": [experience],
        "agent_insights": {"agent_b": insights}
    }


# Agent C - Performance Specialist
def agent_c_learn(state: CollaborativeLearningState) -> CollaborativeLearningState:
    """Agent C learns and shares performance-related insights"""
    task = state.get("task", "")
    shared_knowledge = state.get("shared_knowledge", {})
    agent_insights = state.get("agent_insights", {})
    learning_round = state.get("learning_round", 0)
    
    system_message = SystemMessage(content="""You are Agent C, a performance specialist. 
    Learn from tasks and share performance-related insights with the team.""")
    
    # Review previous agents' insights
    agent_a_insights = agent_insights.get("agent_a", [])
    agent_b_insights = agent_insights.get("agent_b", [])
    previous_insights = "\n".join([
        f"- Agent A: {agent_a_insights[0] if agent_a_insights else 'None'}",
        f"- Agent B: {agent_b_insights[0] if agent_b_insights else 'None'}"
    ])
    
    user_message = HumanMessage(content=f"""Task: {task}

Team Insights:
{previous_insights}

As a performance specialist, provide insights about monitoring, metrics, and optimization.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate insights
    insights = [
        f"Round {learning_round + 1}: Metrics tracking enables improvement",
        f"Round {learning_round + 1}: Bottleneck identification is critical",
        f"Round {learning_round + 1}: Caching strategies boost performance"
    ]
    
    # Create experience record
    experience = {
        "agent": "Agent C (Performance)",
        "round": str(learning_round + 1),
        "insight": insights[0],
        "contribution": "Performance optimization"
    }
    
    return {
        "messages": [AIMessage(content=f"👤 Agent C (Performance Specialist):\n{response.content}\n\n💡 Insights: {len(insights)}")],
        "shared_experiences": [experience],
        "agent_insights": {"agent_c": insights}
    }


# Knowledge Synthesizer
def knowledge_synthesizer(state: CollaborativeLearningState) -> CollaborativeLearningState:
    """Synthesizes insights from all agents into shared knowledge"""
    agent_insights = state.get("agent_insights", {})
    shared_knowledge = state.get("shared_knowledge", {})
    learning_round = state.get("learning_round", 0)
    
    system_message = SystemMessage(content="""You are a knowledge synthesizer. 
    Combine insights from multiple agents into cohesive shared knowledge.""")
    
    # Collect all insights
    all_insights = []
    for agent_name, insights in agent_insights.items():
        all_insights.extend(insights[:1])  # Take top insight from each
    
    insights_text = "\n".join([f"  • {insight}" for insight in all_insights])
    
    user_message = HumanMessage(content=f"""Synthesize team insights:

Individual Insights:
{insights_text}

Create unified knowledge base.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Update shared knowledge
    if "data_practices" not in shared_knowledge:
        shared_knowledge["data_practices"] = []
    if "algorithm_strategies" not in shared_knowledge:
        shared_knowledge["algorithm_strategies"] = []
    if "performance_tips" not in shared_knowledge:
        shared_knowledge["performance_tips"] = []
    
    # Categorize insights
    for agent_name, insights in agent_insights.items():
        if agent_name == "agent_a":
            shared_knowledge["data_practices"].extend(insights[:1])
        elif agent_name == "agent_b":
            shared_knowledge["algorithm_strategies"].extend(insights[:1])
        elif agent_name == "agent_c":
            shared_knowledge["performance_tips"].extend(insights[:1])
    
    total_knowledge = sum(len(items) for items in shared_knowledge.values())
    
    return {
        "messages": [AIMessage(content=f"🔄 Knowledge Synthesizer:\n{response.content}\n\n📚 Shared Knowledge: {total_knowledge} items")],
        "shared_knowledge": shared_knowledge
    }


# Pattern Identifier
def pattern_identifier(state: CollaborativeLearningState) -> CollaborativeLearningState:
    """Identifies collaborative learning patterns"""
    shared_experiences = state.get("shared_experiences", [])
    shared_knowledge = state.get("shared_knowledge", {})
    learning_round = state.get("learning_round", 0)
    
    system_message = SystemMessage(content="""You are a pattern identifier. 
    Analyze collaborative learning to identify successful team patterns.""")
    
    experiences_summary = "\n".join([
        f"  • {exp['agent']}: {exp['insight']}"
        for exp in shared_experiences[:3]
    ])
    
    user_message = HumanMessage(content=f"""Identify collaborative patterns:

Recent Team Experiences:
{experiences_summary}

Find patterns in how agents collaborate and learn.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Identify patterns
    patterns = [
        f"Pattern {learning_round + 1}A: Specialists complement each other's strengths",
        f"Pattern {learning_round + 1}B: Sequential learning builds on previous insights",
        f"Pattern {learning_round + 1}C: Knowledge synthesis creates team intelligence"
    ]
    
    return {
        "messages": [AIMessage(content=f"🔍 Pattern Identifier:\n{response.content}\n\n✅ Patterns: {len(patterns)}")],
        "collaborative_patterns": patterns
    }


# Team Performance Evaluator
def team_evaluator(state: CollaborativeLearningState) -> CollaborativeLearningState:
    """Evaluates team learning performance"""
    task = state.get("task", "")
    learning_round = state.get("learning_round", 0)
    shared_experiences = state.get("shared_experiences", [])
    shared_knowledge = state.get("shared_knowledge", {})
    agent_insights = state.get("agent_insights", {})
    collaborative_patterns = state.get("collaborative_patterns", [])
    
    # Calculate team performance (improves with rounds)
    base_performance = 0.65
    collaboration_bonus = len(agent_insights) * 0.05
    knowledge_bonus = sum(len(items) for items in shared_knowledge.values()) * 0.01
    team_performance = min(0.95, base_performance + collaboration_bonus + knowledge_bonus)
    
    kb_summary = "\n".join([
        f"    • {category.replace('_', ' ').title()}: {len(items)} items"
        for category, items in shared_knowledge.items()
    ])
    
    summary = f"""
    ✅ COLLABORATIVE LEARNING PATTERN - Round {learning_round + 1}
    
    Team Learning Summary:
    • Task: {task[:80]}...
    • Learning Round: {learning_round + 1}
    • Active Agents: {len(agent_insights)}
    • Shared Experiences: {len(shared_experiences)}
    • Collaborative Patterns: {len(collaborative_patterns)}
    • Team Performance: {team_performance:.1%}
    
    Shared Knowledge Base:
{kb_summary if kb_summary else "    • Empty"}
    
    Agent Contributions:
    • Agent A (Data): {len(agent_insights.get('agent_a', []))} insights
    • Agent B (Algorithm): {len(agent_insights.get('agent_b', []))} insights
    • Agent C (Performance): {len(agent_insights.get('agent_c', []))} insights
    
    Collaboration Cycle:
    1. Each Agent Learns → 2. Share Insights → 3. Synthesize Knowledge → 
    4. Identify Patterns → 5. Evaluate Team Performance → 6. Repeat
    
    Collaborative Learning Benefits:
    • Diverse perspectives
    • Knowledge multiplication
    • Complementary expertise
    • Accelerated learning
    • Collective intelligence
    • Synergistic improvement
    
    Latest Patterns:
{chr(10).join([f"    • {p}" for p in collaborative_patterns[:3]]) if collaborative_patterns else "    • None yet"}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Team Evaluator:\n{summary}")],
        "team_performance": team_performance,
        "learning_round": learning_round + 1
    }


# Build the graph
def build_collaborative_learning_graph():
    """Build the collaborative learning pattern graph"""
    workflow = StateGraph(CollaborativeLearningState)
    
    workflow.add_node("agent_a", agent_a_learn)
    workflow.add_node("agent_b", agent_b_learn)
    workflow.add_node("agent_c", agent_c_learn)
    workflow.add_node("synthesizer", knowledge_synthesizer)
    workflow.add_node("identifier", pattern_identifier)
    workflow.add_node("evaluator", team_evaluator)
    
    workflow.add_edge(START, "agent_a")
    workflow.add_edge("agent_a", "agent_b")
    workflow.add_edge("agent_b", "agent_c")
    workflow.add_edge("agent_c", "synthesizer")
    workflow.add_edge("synthesizer", "identifier")
    workflow.add_edge("identifier", "evaluator")
    workflow.add_edge("evaluator", END)
    
    return workflow.compile()


# Example usage - Multiple collaborative learning rounds
if __name__ == "__main__":
    graph = build_collaborative_learning_graph()
    
    print("=== Collaborative Learning MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "task": "Build a high-performance recommendation system",
        "shared_experiences": [],
        "shared_knowledge": {},
        "agent_insights": {},
        "collaborative_patterns": [],
        "team_performance": 0.0,
        "learning_round": 0
    }
    
    # Run multiple learning rounds
    for i in range(3):
        print(f"\n{'=' * 70}")
        print(f"COLLABORATIVE LEARNING ROUND {i + 1}")
        print('=' * 70)
        
        result = graph.invoke(state)
        
        # Show messages for this round
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next round
        state = {
            "messages": [],
            "task": state["task"],
            "shared_experiences": result.get("shared_experiences", []),
            "shared_knowledge": result.get("shared_knowledge", {}),
            "agent_insights": {},  # Reset for next round
            "collaborative_patterns": result.get("collaborative_patterns", []),
            "team_performance": result.get("team_performance", 0.0),
            "learning_round": result.get("learning_round", i + 1)
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL COLLABORATIVE LEARNING RESULTS")
    print('=' * 70)
    print(f"\nTotal Learning Rounds: {state['learning_round']}")
    print(f"Patterns Identified: {len(state['collaborative_patterns'])}")
    print(f"Shared Knowledge Items: {sum(len(items) for items in state['shared_knowledge'].values())}")
    print(f"Final Team Performance: {state['team_performance']:.1%}")

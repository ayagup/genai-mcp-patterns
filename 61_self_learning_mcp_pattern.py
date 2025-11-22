"""
Self-Learning MCP Pattern

This pattern demonstrates an agent that learns and improves from its own
experiences, building knowledge over time.

Key Features:
- Experience collection
- Pattern recognition
- Performance tracking
- Knowledge base updating
- Continuous improvement
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SelfLearningState(TypedDict):
    """State for self-learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    iteration: int
    experiences: list[dict[str, str]]
    learned_patterns: list[str]
    performance_scores: list[float]
    knowledge_base: dict[str, list[str]]
    current_performance: float
    improvement_rate: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Executor
def task_executor(state: SelfLearningState) -> SelfLearningState:
    """Executes task and collects experience"""
    task = state.get("task", "")
    iteration = state.get("iteration", 0)
    knowledge_base = state.get("knowledge_base", {})
    
    system_message = SystemMessage(content="""You are a self-learning agent. Execute 
    tasks while learning from your experiences to improve over time.""")
    
    # Use learned knowledge if available
    relevant_knowledge = "\n".join([
        f"- {item}" for items in knowledge_base.values() for item in items[:2]
    ]) if knowledge_base else "No prior knowledge"
    
    user_message = HumanMessage(content=f"""Execute task (iteration {iteration + 1}): {task}

Relevant learned knowledge:
{relevant_knowledge}

Execute task and document your approach.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate performance that improves over iterations
    base_performance = 0.6
    improvement = min(0.3, iteration * 0.05)
    current_performance = base_performance + improvement
    
    # Collect experience
    experience = {
        "iteration": str(iteration + 1),
        "approach": response.content[:100],
        "performance": f"{current_performance:.2f}",
        "timestamp": f"iter_{iteration + 1}"
    }
    
    return {
        "messages": [AIMessage(content=f"🎯 Task Executor (Iteration {iteration + 1}): {response.content}\n\n📊 Performance: {current_performance:.2%}")],
        "iteration": iteration + 1,
        "experiences": [experience],
        "current_performance": current_performance
    }


# Pattern Recognizer
def pattern_recognizer(state: SelfLearningState) -> SelfLearningState:
    """Identifies patterns from experiences"""
    experiences = state.get("experiences", [])
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are a pattern recognizer. Analyze 
    experiences to identify successful strategies and patterns.""")
    
    experiences_text = "\n".join([
        f"Iteration {exp['iteration']}: Performance {exp['performance']}"
        for exp in experiences[-3:]  # Last 3 experiences
    ])
    
    user_message = HumanMessage(content=f"""Analyze experiences to find patterns:

Recent Experiences:
{experiences_text}

Identify what works well and what to improve.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Identify patterns based on iteration
    patterns = []
    if iteration >= 2:
        patterns.append(f"Pattern {len(patterns) + 1}: Breaking down complex tasks improves results")
    if iteration >= 3:
        patterns.append(f"Pattern {len(patterns) + 1}: Validation after each step reduces errors")
    if iteration >= 4:
        patterns.append(f"Pattern {len(patterns) + 1}: Using structured approach increases consistency")
    
    return {
        "messages": [AIMessage(content=f"🔍 Pattern Recognizer: {response.content}\n\n✅ Identified {len(patterns)} patterns")],
        "learned_patterns": patterns
    }


# Knowledge Updater
def knowledge_updater(state: SelfLearningState) -> SelfLearningState:
    """Updates knowledge base with learned patterns"""
    learned_patterns = state.get("learned_patterns", [])
    knowledge_base = state.get("knowledge_base", {})
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are a knowledge updater. Store 
    learned patterns in the knowledge base for future use.""")
    
    patterns_text = "\n".join([f"  • {pattern}" for pattern in learned_patterns])
    
    user_message = HumanMessage(content=f"""Update knowledge base:

New Patterns:
{patterns_text}

Organize and store in knowledge base.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Update knowledge base
    if "strategies" not in knowledge_base:
        knowledge_base["strategies"] = []
    if "best_practices" not in knowledge_base:
        knowledge_base["best_practices"] = []
    
    for pattern in learned_patterns:
        if "strategy" in pattern.lower() or "approach" in pattern.lower():
            knowledge_base["strategies"].append(pattern)
        else:
            knowledge_base["best_practices"].append(pattern)
    
    total_knowledge = sum(len(items) for items in knowledge_base.values())
    
    return {
        "messages": [AIMessage(content=f"💾 Knowledge Updater: {response.content}\n\n✅ Knowledge base: {total_knowledge} items")],
        "knowledge_base": knowledge_base
    }


# Performance Analyzer
def performance_analyzer(state: SelfLearningState) -> SelfLearningState:
    """Analyzes performance trends"""
    performance_scores = state.get("performance_scores", [])
    current_performance = state.get("current_performance", 0.0)
    iteration = state.get("iteration", 0)
    
    system_message = SystemMessage(content="""You are a performance analyzer. Track 
    performance trends to measure learning progress.""")
    
    # Add current score to history
    performance_scores = performance_scores + [current_performance]
    
    # Calculate improvement
    if len(performance_scores) > 1:
        improvement_rate = ((performance_scores[-1] - performance_scores[0]) / 
                           performance_scores[0] * 100)
    else:
        improvement_rate = 0.0
    
    scores_text = ", ".join([f"{score:.2%}" for score in performance_scores])
    
    user_message = HumanMessage(content=f"""Analyze performance trend:

Performance History: {scores_text}
Current: {current_performance:.2%}
Improvement: {improvement_rate:+.1f}%

Assess learning progress.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📈 Performance Analyzer: {response.content}\n\n✅ Improvement: {improvement_rate:+.1f}%")],
        "performance_scores": performance_scores,
        "improvement_rate": improvement_rate
    }


# Learning Monitor
def learning_monitor(state: SelfLearningState) -> SelfLearningState:
    """Monitors overall learning progress"""
    task = state.get("task", "")
    iteration = state.get("iteration", 0)
    experiences = state.get("experiences", [])
    learned_patterns = state.get("learned_patterns", [])
    knowledge_base = state.get("knowledge_base", {})
    performance_scores = state.get("performance_scores", [])
    improvement_rate = state.get("improvement_rate", 0.0)
    
    kb_summary = "\n".join([
        f"  • {category.title()}: {len(items)} items"
        for category, items in knowledge_base.items()
    ])
    
    performance_trend = " → ".join([f"{score:.0%}" for score in performance_scores])
    
    summary = f"""
    ✅ SELF-LEARNING PATTERN COMPLETE
    
    Learning Summary:
    • Task: {task[:80]}...
    • Iterations: {iteration}
    • Experiences Collected: {len(experiences)}
    • Patterns Learned: {len(learned_patterns)}
    • Improvement Rate: {improvement_rate:+.1f}%
    
    Knowledge Base:
{kb_summary if kb_summary else "  • Empty"}
    
    Performance Trend:
    {performance_trend}
    
    Learning Cycle:
    1. Execute Task → 2. Collect Experience → 3. Recognize Patterns → 
    4. Update Knowledge → 5. Analyze Performance → 6. Improve
    
    Self-Learning Benefits:
    • Autonomous improvement
    • Experience-based learning
    • Pattern recognition
    • Knowledge accumulation
    • Continuous adaptation
    • Performance optimization
    
    Latest Learned Patterns:
{chr(10).join([f"  • {p}" for p in learned_patterns[:3]]) if learned_patterns else "  • None yet"}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Learning Monitor:\n{summary}")]
    }


# Build the graph
def build_self_learning_graph():
    """Build the self-learning pattern graph"""
    workflow = StateGraph(SelfLearningState)
    
    workflow.add_node("executor", task_executor)
    workflow.add_node("recognizer", pattern_recognizer)
    workflow.add_node("updater", knowledge_updater)
    workflow.add_node("analyzer", performance_analyzer)
    workflow.add_node("monitor", learning_monitor)
    
    workflow.add_edge(START, "executor")
    workflow.add_edge("executor", "recognizer")
    workflow.add_edge("recognizer", "updater")
    workflow.add_edge("updater", "analyzer")
    workflow.add_edge("analyzer", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage - Multiple learning iterations
if __name__ == "__main__":
    graph = build_self_learning_graph()
    
    print("=== Self-Learning MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "task": "Optimize database query performance",
        "iteration": 0,
        "experiences": [],
        "learned_patterns": [],
        "performance_scores": [],
        "knowledge_base": {},
        "current_performance": 0.0,
        "improvement_rate": 0.0
    }
    
    # Run multiple learning iterations
    for i in range(5):
        print(f"\n{'=' * 70}")
        print(f"LEARNING ITERATION {i + 1}")
        print('=' * 70)
        
        result = graph.invoke(state)
        
        # Show messages for this iteration
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next iteration
        state = {
            "messages": [],
            "task": state["task"],
            "iteration": result.get("iteration", i + 1),
            "experiences": result.get("experiences", []),
            "learned_patterns": result.get("learned_patterns", []),
            "performance_scores": result.get("performance_scores", []),
            "knowledge_base": result.get("knowledge_base", {}),
            "current_performance": 0.0,
            "improvement_rate": result.get("improvement_rate", 0.0)
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL LEARNING RESULTS")
    print('=' * 70)
    print(f"\nTotal Iterations: {state['iteration']}")
    print(f"Patterns Learned: {len(state['learned_patterns'])}")
    print(f"Knowledge Items: {sum(len(items) for items in state['knowledge_base'].values())}")
    print(f"Final Improvement: {state['improvement_rate']:+.1f}%")

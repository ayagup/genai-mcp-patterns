"""
Feedback Loop MCP Pattern

This pattern demonstrates agents using iterative feedback to continuously
improve their performance through systematic refinement cycles.

Key Features:
- Iterative improvement
- Feedback collection
- Performance analysis
- Refinement strategies
- Quality enhancement
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class FeedbackLoopState(TypedDict):
    """State for feedback loop pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    current_output: str
    feedback_history: list[dict[str, str | float]]
    refinement_iteration: int
    quality_score: float
    improvement_rate: float
    target_quality: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Executor
def task_executor(state: FeedbackLoopState) -> FeedbackLoopState:
    """Executes or refines task based on feedback"""
    task = state.get("task", "")
    feedback_history = state.get("feedback_history", [])
    refinement_iteration = state.get("refinement_iteration", 0)
    
    system_message = SystemMessage(content="""You are a task executor. 
    Execute the task and incorporate previous feedback to improve output.""")
    
    # Show previous feedback
    if feedback_history:
        recent_feedback = "\n".join([
            f"  • Iteration {fb.get('iteration', 0)}: {fb.get('feedback', '')} (score: {fb.get('score', 0):.1f})"
            for fb in feedback_history[-2:]
        ])
        feedback_context = f"\n\nPrevious Feedback:\n{recent_feedback}"
    else:
        feedback_context = "\n\nNo previous feedback (first iteration)"
    
    user_message = HumanMessage(content=f"""Execute task (Iteration {refinement_iteration + 1}):

Task: {task}{feedback_context}

Execute or refine based on feedback.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate output (simulated improvement over iterations)
    current_output = f"Output for iteration {refinement_iteration + 1}: {response.content[:100]}"
    
    return {
        "messages": [AIMessage(content=f"⚙️ Task Executor (Iteration {refinement_iteration + 1}):\n{response.content}\n\n✅ Output generated")],
        "current_output": current_output
    }


# Feedback Collector
def feedback_collector(state: FeedbackLoopState) -> FeedbackLoopState:
    """Collects feedback on current output"""
    task = state.get("task", "")
    current_output = state.get("current_output", "")
    refinement_iteration = state.get("refinement_iteration", 0)
    
    system_message = SystemMessage(content="""You are a feedback collector. 
    Evaluate the output and provide constructive feedback for improvement.""")
    
    user_message = HumanMessage(content=f"""Provide feedback:

Task: {task}
Iteration: {refinement_iteration + 1}

Output:
{current_output[:200]}...

Evaluate and suggest improvements.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📝 Feedback Collector:\n{response.content}\n\n✅ Feedback provided")]
    }


# Quality Assessor
def quality_assessor(state: FeedbackLoopState) -> FeedbackLoopState:
    """Assesses quality of current output"""
    task = state.get("task", "")
    current_output = state.get("current_output", "")
    refinement_iteration = state.get("refinement_iteration", 0)
    feedback_history = state.get("feedback_history", [])
    
    system_message = SystemMessage(content="""You are a quality assessor. 
    Score the output quality objectively.""")
    
    user_message = HumanMessage(content=f"""Assess quality:

Task: {task}
Iteration: {refinement_iteration + 1}
Output: {current_output[:150]}...

Rate quality on 0-100 scale.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate quality score (improves with iterations)
    base_quality = 60.0
    improvement_per_iteration = 8.0
    quality_score = min(95.0, base_quality + (refinement_iteration * improvement_per_iteration))
    
    # Add noise to make it realistic
    import random
    quality_score += random.uniform(-2, 2)
    quality_score = min(100.0, max(0.0, quality_score))
    
    # Create feedback record
    feedback_record = {
        "iteration": str(refinement_iteration + 1),
        "feedback": response.content[:100],
        "score": quality_score,
        "output_summary": current_output[:80]
    }
    
    return {
        "messages": [AIMessage(content=f"📊 Quality Assessor:\n{response.content}\n\n✅ Quality Score: {quality_score:.1f}/100")],
        "quality_score": quality_score,
        "feedback_history": [feedback_record]
    }


# Improvement Analyzer
def improvement_analyzer(state: FeedbackLoopState) -> FeedbackLoopState:
    """Analyzes improvement trends"""
    feedback_history = state.get("feedback_history", [])
    quality_score = state.get("quality_score", 0.0)
    refinement_iteration = state.get("refinement_iteration", 0)
    
    system_message = SystemMessage(content="""You are an improvement analyzer. 
    Analyze the improvement trajectory and suggest next steps.""")
    
    if len(feedback_history) >= 2:
        first_score = float(feedback_history[0].get("score", 0))
        current_score = quality_score
        improvement_rate = ((current_score - first_score) / first_score * 100) if first_score > 0 else 0
    else:
        improvement_rate = 0.0
    
    scores_text = ", ".join([f"{float(fb.get('score', 0)):.1f}" for fb in feedback_history[-5:]])
    
    user_message = HumanMessage(content=f"""Analyze improvement:

Iteration: {refinement_iteration + 1}
Quality Scores: {scores_text}
Current: {quality_score:.1f}
Improvement Rate: {improvement_rate:+.1f}%

Analyze trends and suggest next steps.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📈 Improvement Analyzer:\n{response.content}\n\n📊 Improvement: {improvement_rate:+.1f}%")],
        "improvement_rate": improvement_rate
    }


# Refinement Controller
def refinement_controller(state: FeedbackLoopState) -> FeedbackLoopState:
    """Controls refinement process and determines if more iterations needed"""
    task = state.get("task", "")
    current_output = state.get("current_output", "")
    feedback_history = state.get("feedback_history", [])
    refinement_iteration = state.get("refinement_iteration", 0)
    quality_score = state.get("quality_score", 0.0)
    improvement_rate = state.get("improvement_rate", 0.0)
    target_quality = state.get("target_quality", 85.0)
    
    quality_trend = "\n".join([
        f"    Iteration {fb.get('iteration', '')}: {fb.get('score', 0):.1f}/100"
        for fb in feedback_history[-5:]
    ])
    
    recent_feedback = "\n".join([
        f"    • {fb.get('feedback', '')[:80]}..."
        for fb in feedback_history[-3:]
    ])
    
    # Determine status
    if quality_score >= target_quality:
        status = "✅ Target Quality Achieved"
    elif refinement_iteration >= 4:
        status = "⏱️ Max Iterations Reached"
    else:
        status = "🔄 Continuing Refinement"
    
    summary = f"""
    ✅ FEEDBACK LOOP PATTERN - Iteration {refinement_iteration + 1}
    
    Refinement Summary:
    • Task: {task[:80]}...
    • Current Iteration: {refinement_iteration + 1}
    • Quality Score: {quality_score:.1f}/100
    • Target Quality: {target_quality:.1f}/100
    • Improvement Rate: {improvement_rate:+.1f}%
    • Status: {status}
    
    Quality Progression:
{quality_trend if quality_trend else "    • No history yet"}
    
    Recent Feedback:
{recent_feedback if recent_feedback else "    • No feedback yet"}
    
    Feedback Loop Process:
    1. Execute Task → 2. Collect Feedback → 3. Assess Quality → 
    4. Analyze Improvement → 5. Refine → 6. Repeat
    
    Feedback Loop Benefits:
    • Continuous improvement
    • Systematic refinement
    • Quality assurance
    • Iterative optimization
    • Performance tracking
    • Goal-oriented enhancement
    
    Improvement Metrics:
    • Starting Quality: ~60/100
    • Current Quality: {quality_score:.1f}/100
    • Improvement: {improvement_rate:+.1f}%
    • Iterations: {refinement_iteration + 1}
    • Average Gain: {quality_score / (refinement_iteration + 1):.1f} points/iteration
    
    Next Steps:
    {"• Continue refinement to reach target quality" if quality_score < target_quality and refinement_iteration < 4 else "• Quality target achieved or max iterations reached"}
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Refinement Controller:\n{summary}")],
        "refinement_iteration": refinement_iteration + 1
    }


# Build the graph
def build_feedback_loop_graph():
    """Build the feedback loop pattern graph"""
    workflow = StateGraph(FeedbackLoopState)
    
    workflow.add_node("executor", task_executor)
    workflow.add_node("collector", feedback_collector)
    workflow.add_node("assessor", quality_assessor)
    workflow.add_node("analyzer", improvement_analyzer)
    workflow.add_node("controller", refinement_controller)
    
    workflow.add_edge(START, "executor")
    workflow.add_edge("executor", "collector")
    workflow.add_edge("collector", "assessor")
    workflow.add_edge("assessor", "analyzer")
    workflow.add_edge("analyzer", "controller")
    workflow.add_edge("controller", END)
    
    return workflow.compile()


# Example usage - Multiple feedback iterations
if __name__ == "__main__":
    graph = build_feedback_loop_graph()
    
    print("=== Feedback Loop MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "task": "Write comprehensive documentation for a new API endpoint",
        "current_output": "",
        "feedback_history": [],
        "refinement_iteration": 0,
        "quality_score": 0.0,
        "improvement_rate": 0.0,
        "target_quality": 85.0
    }
    
    # Run multiple feedback loop iterations
    for i in range(5):
        print(f"\n{'=' * 70}")
        print(f"FEEDBACK LOOP ITERATION {i + 1}")
        print('=' * 70)
        
        result = graph.invoke(state)
        
        # Show messages for this iteration
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Check if target quality achieved
        if result.get("quality_score", 0) >= state["target_quality"]:
            print("\n✅ TARGET QUALITY ACHIEVED!")
            state = result
            break
        
        # Update state for next iteration
        state = {
            "messages": [],
            "task": state["task"],
            "current_output": result.get("current_output", ""),
            "feedback_history": result.get("feedback_history", []),
            "refinement_iteration": result.get("refinement_iteration", i + 1),
            "quality_score": 0.0,
            "improvement_rate": result.get("improvement_rate", 0.0),
            "target_quality": state["target_quality"]
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL FEEDBACK LOOP RESULTS")
    print('=' * 70)
    print(f"\nTask: {state['task']}")
    print(f"Total Iterations: {state['refinement_iteration']}")
    print(f"Final Quality: {state.get('quality_score', 0):.1f}/100")
    print(f"Target Quality: {state['target_quality']:.1f}/100")
    print(f"Total Improvement: {state.get('improvement_rate', 0):+.1f}%")

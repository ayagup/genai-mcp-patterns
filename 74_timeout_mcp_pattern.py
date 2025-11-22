"""
Timeout MCP Pattern

This pattern enforces time limits on operations to prevent indefinite waiting
and resource exhaustion, with timeout detection and cancellation.

Key Features:
- Configurable timeout limits
- Timeout detection
- Graceful cancellation
- Time tracking
- Timeout recovery
"""

from typing import TypedDict, Sequence, Annotated
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TimeoutState(TypedDict):
    """State for timeout pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    timeout_seconds: float
    start_time: float
    end_time: float
    execution_time: float
    timeout_occurred: bool
    result: str
    status: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Executor with Timeout
def task_executor(state: TimeoutState) -> TimeoutState:
    """Executes task with timeout monitoring"""
    task = state.get("task", "")
    timeout_seconds = state.get("timeout_seconds", 5.0)
    
    start_time = time.time()
    
    system_message = SystemMessage(content="""You are a task executor. 
    Execute tasks while respecting timeout limits.""")
    
    user_message = HumanMessage(content=f"""Execute task:

Task: {task}
Timeout Limit: {timeout_seconds} seconds

Execute the task efficiently within the time limit.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate task execution time
    simulated_execution_time = 2.5  # seconds
    time.sleep(0.1)  # Brief actual delay for demonstration
    
    end_time = time.time()
    actual_elapsed = end_time - start_time
    
    # Use simulated time for demonstration
    execution_time = simulated_execution_time
    timeout_occurred = execution_time > timeout_seconds
    
    if timeout_occurred:
        result = "TIMEOUT: Task execution exceeded time limit"
        status = "⏱️ Timeout"
    else:
        result = "SUCCESS: Task completed within timeout"
        status = "✅ Success"
    
    time_info = f"""
    ⏱️ Time: {execution_time:.2f}s / {timeout_seconds:.2f}s limit
    {status}
    """
    
    return {
        "messages": [AIMessage(content=f"🔄 Task Executor:\n{response.content}\n{time_info}")],
        "start_time": start_time,
        "end_time": end_time,
        "execution_time": execution_time,
        "timeout_occurred": timeout_occurred,
        "result": result,
        "status": status
    }


# Timeout Monitor
def timeout_monitor(state: TimeoutState) -> TimeoutState:
    """Monitors timeout status and handles timeout events"""
    task = state.get("task", "")
    timeout_seconds = state.get("timeout_seconds", 5.0)
    execution_time = state.get("execution_time", 0.0)
    timeout_occurred = state.get("timeout_occurred", False)
    result = state.get("result", "")
    
    system_message = SystemMessage(content="""You are a timeout monitor. 
    Track execution time and handle timeout events.""")
    
    user_message = HumanMessage(content=f"""Monitor timeout:

Task: {task}
Timeout Limit: {timeout_seconds} seconds
Execution Time: {execution_time} seconds
Timeout Occurred: {timeout_occurred}

Analyze timeout status and provide recommendations.""")
    
    response = llm.invoke([system_message, user_message])
    
    time_utilization = (execution_time / timeout_seconds) * 100
    
    if timeout_occurred:
        status_icon = "❌"
        status_text = "TIMEOUT EXCEEDED"
        recommendation = """
        Recommendations:
        • Increase timeout limit if task legitimately needs more time
        • Optimize task execution for better performance
        • Break task into smaller subtasks
        • Consider async execution for long-running tasks
        """
    else:
        status_icon = "✅"
        status_text = "WITHIN TIMEOUT"
        if time_utilization > 80:
            recommendation = """
        Recommendations:
        • Consider increasing timeout buffer (used >80% of limit)
        • Monitor for potential timeout risks
        • Optimize execution if possible
        """
        else:
            recommendation = """
        Recommendations:
        • Current timeout limit is appropriate
        • Task executing efficiently
        """
    
    monitor_report = f"""
    {status_icon} {status_text}
    
    Execution Metrics:
    • Task: {task[:60]}...
    • Timeout Limit: {timeout_seconds:.2f}s
    • Execution Time: {execution_time:.2f}s
    • Time Utilization: {time_utilization:.1f}%
    • Timeout Occurred: {'Yes ❌' if timeout_occurred else 'No ✅'}
    
    Result: {result}
    {recommendation}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Timeout Monitor:\n{response.content}\n{monitor_report}")]
    }


# Timeout Handler
def timeout_handler(state: TimeoutState) -> TimeoutState:
    """Handles timeout recovery and cleanup"""
    task = state.get("task", "")
    timeout_occurred = state.get("timeout_occurred", False)
    execution_time = state.get("execution_time", 0.0)
    timeout_seconds = state.get("timeout_seconds", 5.0)
    
    if not timeout_occurred:
        return {
            "messages": [AIMessage(content="✅ Timeout Handler: No timeout occurred, no action needed")]
        }
    
    system_message = SystemMessage(content="""You are a timeout handler. 
    Handle timeout events, cleanup resources, and implement recovery strategies.""")
    
    user_message = HumanMessage(content=f"""Handle timeout:

Task: {task}
Execution Time: {execution_time} seconds
Timeout Limit: {timeout_seconds} seconds

Implement timeout recovery strategy.""")
    
    response = llm.invoke([system_message, user_message])
    
    recovery_actions = """
    Timeout Recovery Actions:
    1. ⏹️ Cancel ongoing operation
    2. 🧹 Release allocated resources
    3. 📝 Log timeout event for monitoring
    4. 🔄 Trigger retry with adjusted timeout (optional)
    5. ⚠️ Notify dependent systems of timeout
    
    Resource Cleanup:
    • Close network connections
    • Release memory allocations
    • Clear temporary files
    • Cancel pending callbacks
    
    Timeout Pattern Benefits:
    • Prevents resource exhaustion
    • Avoids indefinite waiting
    • Improves system responsiveness
    • Enables SLA enforcement
    • Provides predictable behavior
    
    Common Timeout Strategies:
    • Fixed Timeout: Same limit for all requests
    • Adaptive Timeout: Adjust based on historical data
    • Cascading Timeout: Child operations have shorter timeouts
    • Deadline Propagation: Pass deadline through call chain
    """
    
    return {
        "messages": [AIMessage(content=f"⚠️ Timeout Handler:\n{response.content}\n{recovery_actions}")]
    }


# Timeout Reporter
def timeout_reporter(state: TimeoutState) -> TimeoutState:
    """Generates comprehensive timeout report"""
    task = state.get("task", "")
    timeout_seconds = state.get("timeout_seconds", 5.0)
    execution_time = state.get("execution_time", 0.0)
    timeout_occurred = state.get("timeout_occurred", False)
    result = state.get("result", "")
    status = state.get("status", "")
    
    time_remaining = max(0, timeout_seconds - execution_time)
    time_utilization = (execution_time / timeout_seconds) * 100
    
    outcome_emoji = "❌" if timeout_occurred else "✅"
    
    summary = f"""
    {outcome_emoji} TIMEOUT PATTERN COMPLETE
    
    Execution Summary:
    • Task: {task[:80]}...
    • Status: {status}
    • Result: {result}
    
    Timing Analysis:
    • Timeout Limit: {timeout_seconds:.2f}s
    • Execution Time: {execution_time:.2f}s
    • Time Remaining: {time_remaining:.2f}s
    • Time Utilization: {time_utilization:.1f}%
    
    Timeout Status: {'⏱️ TIMEOUT OCCURRED' if timeout_occurred else '✅ WITHIN LIMIT'}
    
    Timeout Configuration:
    • Current Limit: {timeout_seconds}s
    • Recommended: {'Increase timeout' if timeout_occurred else 'Current timeout appropriate'}
    
    Timeout Pattern Process:
    1. Start timer when task begins
    2. Monitor execution time continuously
    3. Cancel operation if timeout exceeded
    4. Handle timeout event and cleanup
    5. Report timeout status
    
    Timeout Pattern Benefits:
    • Resource protection
    • Predictable response times
    • SLA enforcement
    • System stability
    • Error isolation
    
    Best Practices:
    • Set realistic timeout values
    • Monitor timeout rates
    • Implement graceful cancellation
    • Log timeout events
    • Consider retry with backoff
    • Use cascading timeouts
    
    Timeout Tuning Guidelines:
    • P50 latency + 2σ: Conservative
    • P95 latency + σ: Balanced
    • P99 latency: Aggressive
    
    Key Insight:
    Timeouts prevent cascading failures and resource exhaustion by
    enforcing time limits on operations, ensuring system responsiveness.
    """
    
    return {
        "messages": [AIMessage(content=f"📋 Timeout Reporter:\n{summary}")]
    }


# Build the graph
def build_timeout_graph():
    """Build the timeout pattern graph"""
    workflow = StateGraph(TimeoutState)
    
    workflow.add_node("executor", task_executor)
    workflow.add_node("monitor", timeout_monitor)
    workflow.add_node("handler", timeout_handler)
    workflow.add_node("reporter", timeout_reporter)
    
    workflow.add_edge(START, "executor")
    workflow.add_edge("executor", "monitor")
    workflow.add_edge("monitor", "handler")
    workflow.add_edge("handler", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_timeout_graph()
    
    print("=== Timeout MCP Pattern ===\n")
    
    # Test Case 1: Task within timeout
    print("\n" + "="*70)
    print("TEST CASE 1: Task within timeout limit")
    print("="*70)
    
    state1 = {
        "messages": [],
        "task": "Process user data and generate report",
        "timeout_seconds": 5.0,
        "start_time": 0.0,
        "end_time": 0.0,
        "execution_time": 0.0,
        "timeout_occurred": False,
        "result": "",
        "status": ""
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)

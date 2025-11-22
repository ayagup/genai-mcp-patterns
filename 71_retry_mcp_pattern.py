"""
Retry MCP Pattern

This pattern demonstrates automatic retry logic with exponential backoff
to handle transient failures gracefully.

Key Features:
- Automatic retry on failure
- Exponential backoff
- Maximum retry attempts
- Retry condition evaluation
- Success tracking
"""

from typing import TypedDict, Sequence, Annotated
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class RetryState(TypedDict):
    """State for retry pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    attempt_number: int
    max_retries: int
    success: bool
    error_message: str
    backoff_seconds: float
    retry_history: list[dict[str, str | int | float]]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Executor with Simulated Failures
def task_executor(state: RetryState) -> RetryState:
    """Executes task with potential for transient failures"""
    task = state.get("task", "")
    attempt_number = state.get("attempt_number", 0)
    max_retries = state.get("max_retries", 3)
    
    system_message = SystemMessage(content="""You are a task executor. 
    Execute tasks that may experience transient failures.""")
    
    user_message = HumanMessage(content=f"""Execute task (Attempt {attempt_number + 1}/{max_retries + 1}):

Task: {task}

Attempt execution.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate transient failures (fail first 2 attempts, succeed on 3rd)
    if attempt_number < 2:
        success = False
        error_message = f"Transient error: Network timeout on attempt {attempt_number + 1}"
    else:
        success = True
        error_message = ""
    
    status_emoji = "✅" if success else "❌"
    
    return {
        "messages": [AIMessage(content=f"⚙️ Task Executor (Attempt {attempt_number + 1}):\n{response.content}\n\n{status_emoji} Status: {'Success' if success else f'Failed - {error_message}'}")],
        "success": success,
        "error_message": error_message
    }


# Retry Controller
def retry_controller(state: RetryState) -> RetryState:
    """Controls retry logic and backoff"""
    task = state.get("task", "")
    attempt_number = state.get("attempt_number", 0)
    max_retries = state.get("max_retries", 3)
    success = state.get("success", False)
    error_message = state.get("error_message", "")
    retry_history = state.get("retry_history", [])
    
    system_message = SystemMessage(content="""You are a retry controller. 
    Manage retry attempts with exponential backoff.""")
    
    user_message = HumanMessage(content=f"""Control retry logic:

Task: {task}
Attempt: {attempt_number + 1}/{max_retries + 1}
Success: {success}
Error: {error_message}

Determine retry strategy.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate exponential backoff
    base_backoff = 1.0  # 1 second
    backoff_seconds = base_backoff * (2 ** attempt_number)
    
    # Record attempt
    retry_record = {
        "attempt": attempt_number + 1,
        "success": success,
        "error": error_message,
        "backoff": backoff_seconds
    }
    retry_history = retry_history + [retry_record]
    
    return {
        "messages": [AIMessage(content=f"🔄 Retry Controller:\n{response.content}\n\n✅ Backoff: {backoff_seconds}s | Next attempt in {backoff_seconds}s")],
        "attempt_number": attempt_number + 1,
        "backoff_seconds": backoff_seconds,
        "retry_history": retry_history
    }


# Retry Monitor
def retry_monitor(state: RetryState) -> RetryState:
    """Monitors retry attempts and final outcome"""
    task = state.get("task", "")
    attempt_number = state.get("attempt_number", 0)
    max_retries = state.get("max_retries", 3)
    success = state.get("success", False)
    retry_history = state.get("retry_history", [])
    
    retry_summary = "\n".join([
        f"    Attempt {r['attempt']}: {'✅ Success' if r['success'] else f'❌ Failed ({r['error']})'} - Backoff: {r['backoff']}s"
        for r in retry_history
    ])
    
    total_backoff = sum(r["backoff"] for r in retry_history if not r["success"])
    
    if success:
        outcome = "✅ TASK SUCCEEDED"
        details = f"Succeeded after {attempt_number} attempt(s)"
    elif attempt_number > max_retries:
        outcome = "❌ MAX RETRIES EXCEEDED"
        details = f"Failed after {attempt_number} attempts"
    else:
        outcome = "🔄 RETRY IN PROGRESS"
        details = f"Attempt {attempt_number}/{max_retries + 1}"
    
    summary = f"""
    ✅ RETRY PATTERN COMPLETE
    
    Retry Summary:
    • Task: {task[:80]}...
    • Outcome: {outcome}
    • Total Attempts: {attempt_number}
    • Max Retries: {max_retries}
    • Total Backoff Time: {total_backoff:.1f}s
    
    Retry History:
{retry_summary if retry_summary else "    • No attempts yet"}
    
    Retry Strategy:
    • Exponential Backoff: 1s, 2s, 4s, 8s, ...
    • Max Retries: {max_retries}
    • Retry on: Transient errors
    • Skip retry on: Permanent errors
    
    Retry Pattern Process:
    1. Execute Task → 2. Check Success → 3. Calculate Backoff → 
    4. Wait → 5. Retry → 6. Repeat until success or max retries
    
    Retry Pattern Benefits:
    • Handle transient failures
    • Automatic recovery
    • Exponential backoff prevents overload
    • Configurable retry limits
    • Success tracking
    • Error resilience
    
    Details: {details}
    
    Key Insight:
    Retry pattern with exponential backoff handles temporary failures
    gracefully while preventing system overload through increasing delays.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Retry Monitor:\n{summary}")]
    }


# Build the graph with conditional retry logic
def build_retry_graph():
    """Build the retry pattern graph"""
    workflow = StateGraph(RetryState)
    
    workflow.add_node("executor", task_executor)
    workflow.add_node("controller", retry_controller)
    workflow.add_node("monitor", retry_monitor)
    
    # Start with executor
    workflow.add_edge(START, "executor")
    workflow.add_edge("executor", "controller")
    workflow.add_edge("controller", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage - Multiple retry attempts
if __name__ == "__main__":
    graph = build_retry_graph()
    
    print("=== Retry MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "task": "Process payment transaction with external payment gateway",
        "attempt_number": 0,
        "max_retries": 3,
        "success": False,
        "error_message": "",
        "backoff_seconds": 0.0,
        "retry_history": []
    }
    
    # Execute with retries
    for attempt in range(state["max_retries"] + 1):
        print(f"\n{'=' * 70}")
        print(f"RETRY ATTEMPT {attempt + 1}")
        print('=' * 70)
        
        result = graph.invoke(state)
        
        # Show messages
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Check if succeeded
        if result.get("success", False):
            print("\n✅ TASK SUCCEEDED - NO MORE RETRIES NEEDED")
            break
        
        # Check if max retries exceeded
        if result.get("attempt_number", 0) > state["max_retries"]:
            print("\n❌ MAX RETRIES EXCEEDED - GIVING UP")
            break
        
        # Update state for next retry
        backoff = result.get("backoff_seconds", 1.0)
        print(f"\n⏳ Waiting {backoff}s before retry...")
        time.sleep(min(backoff, 0.1))  # Shortened for demo
        
        state = {
            "messages": [],
            "task": state["task"],
            "attempt_number": result.get("attempt_number", attempt + 1),
            "max_retries": state["max_retries"],
            "success": False,
            "error_message": "",
            "backoff_seconds": 0.0,
            "retry_history": result.get("retry_history", [])
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL RETRY RESULTS")
    print('=' * 70)
    print(f"\nTask: {state['task']}")
    print(f"Final Attempt: {result.get('attempt_number', 0)}")
    print(f"Success: {result.get('success', False)}")
    print(f"Total Retries: {len(result.get('retry_history', []))}")

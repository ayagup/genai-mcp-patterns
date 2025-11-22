"""
Debugging MCP Pattern

This pattern implements interactive debugging capabilities with
breakpoint management, variable inspection, and execution flow analysis.

Key Features:
- Breakpoint management
- Variable inspection
- Stack trace analysis
- Step-through debugging
- Remote debugging support
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class DebuggingState(TypedDict):
    """State for debugging pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    breakpoints: List[Dict]  # [{file, line, condition, hit_count}]
    stack_frames: List[Dict]  # [{function, file, line, locals}]
    variables: Dict[str, any]
    debug_session_active: bool
    execution_step: str  # "run", "step_over", "step_into", "step_out"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Debug Controller
def debug_controller(state: DebuggingState) -> DebuggingState:
    """Controls debug session and breakpoints"""
    breakpoints = state.get("breakpoints", [])
    
    system_message = SystemMessage(content="""You are a debug controller.
    Manage debug sessions, breakpoints, and execution flow.""")
    
    user_message = HumanMessage(content=f"""Start debug session:

Breakpoints: {len(breakpoints) if breakpoints else 'None set'}

Initialize debugging.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Set up breakpoints if not provided
    if not breakpoints:
        breakpoints = [
            {
                "file": "api/users.py",
                "line": 45,
                "condition": None,
                "hit_count": 0,
                "enabled": True
            },
            {
                "file": "db/queries.py",
                "line": 123,
                "condition": "user_id == 123",
                "hit_count": 0,
                "enabled": True
            },
            {
                "file": "auth/validator.py",
                "line": 78,
                "condition": None,
                "hit_count": 0,
                "enabled": True
            }
        ]
    
    # Simulate stack frames at breakpoint
    stack_frames = [
        {
            "function": "get_user",
            "file": "api/users.py",
            "line": 45,
            "locals": {
                "user_id": 123,
                "request": "<Request object>",
                "session": "<SQLAlchemy Session>"
            }
        },
        {
            "function": "handle_request",
            "file": "api/router.py",
            "line": 234,
            "locals": {
                "path": "/api/users/123",
                "method": "GET",
                "headers": {"Authorization": "Bearer xxx"}
            }
        },
        {
            "function": "main",
            "file": "app.py",
            "line": 12,
            "locals": {
                "app": "<Flask app>",
                "config": {"DEBUG": True, "PORT": 8080}
            }
        }
    ]
    
    variables = {
        "user_id": 123,
        "user_data": {
            "id": 123,
            "name": "John Doe",
            "email": "john@example.com",
            "created_at": "2024-01-15"
        },
        "is_authenticated": True,
        "permissions": ["read", "write"]
    }
    
    report = f"""
    🐛 Debug Controller:
    
    Debug Session:
    • Active: Yes
    • Breakpoints: {len(breakpoints)}
    • Stack Depth: {len(stack_frames)}
    • Current Frame: {stack_frames[0]['function']}() at line {stack_frames[0]['line']}
    
    Debugging Concepts:
    
    Breakpoint Types:
    
    Line Breakpoint:
    • Stop at specific line
    • Most common type
    • File + line number
    • Simple to set
    
    Conditional Breakpoint:
    • Stop when condition true
    • user_id == 123
    • Reduces noise
    • Performance impact
    
    Function Breakpoint:
    • Stop on function entry
    • Name-based
    • Works with dynamic code
    • Language-specific
    
    Exception Breakpoint:
    • Stop on exception
    • Caught/uncaught
    • Exception type filter
    • Stack trace available
    
    Logpoint:
    • Log without stopping
    • Message interpolation
    • Minimal overhead
    • Production-safe
    
    Debug Commands:
    
    Continue (c):
    • Resume execution
    • Run until next breakpoint
    • Or program completion
    
    Step Over (n):
    • Execute current line
    • Don't enter functions
    • Stay at same level
    • Fast debugging
    
    Step Into (s):
    • Enter function calls
    • Deep debugging
    • Inspect implementation
    • Full visibility
    
    Step Out (fin):
    • Exit current function
    • Return to caller
    • Skip details
    • Jump up stack
    
    Debugger Tools:
    
    pdb (Python):
    ```python
    import pdb
    
    def get_user(user_id):
        # Set breakpoint
        pdb.set_trace()
        
        user = db.query(User).get(user_id)
        return user
    
    # Commands in pdb:
    # (Pdb) p user_id       # Print variable
    # (Pdb) pp user_data    # Pretty print
    # (Pdb) l               # List source code
    # (Pdb) w               # Where (stack trace)
    # (Pdb) n               # Next line
    # (Pdb) s               # Step into
    # (Pdb) c               # Continue
    # (Pdb) b 45            # Set breakpoint at line 45
    # (Pdb) cl 1            # Clear breakpoint 1
    # (Pdb) disable 1       # Disable breakpoint 1
    ```
    
    ipdb (Enhanced pdb):
    ```python
    import ipdb
    
    ipdb.set_trace()
    
    # Features:
    # - Tab completion
    # - Syntax highlighting
    # - Better introspection
    # - IPython integration
    ```
    
    VS Code Debugger:
    ```json
    {{
        "version": "0.2.0",
        "configurations": [
            {{
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${{file}}",
                "console": "integratedTerminal",
                "justMyCode": false,
                "env": {{"DEBUG": "True"}},
                "args": ["--config", "dev.yaml"]
            }},
            {{
                "name": "Python: Remote Attach",
                "type": "python",
                "request": "attach",
                "connect": {{
                    "host": "localhost",
                    "port": 5678
                }},
                "pathMappings": [
                    {{
                        "localRoot": "${{workspaceFolder}}",
                        "remoteRoot": "/app"
                    }}
                ]
            }}
        ]
    }}
    ```
    
    Remote Debugging (debugpy):
    ```python
    import debugpy
    
    # Enable remote debugging
    debugpy.listen(("0.0.0.0", 5678))
    print("Waiting for debugger attach...")
    debugpy.wait_for_client()
    
    # Your code here
    app.run()
    ```
    
    Variable Inspection:
    
    Simple Variables:
    • Print values
    • Type checking
    • String representation
    • Truthiness
    
    Collections:
    • Length
    • Contents
    • Iteration
    • Slicing
    
    Objects:
    • Attributes
    • Methods
    • __dict__ inspection
    • dir() listing
    
    Complex Types:
    • JSON serialization
    • Pretty printing
    • Recursive inspection
    • Memory addresses
    
    Watch Expressions:
    • Monitor variables
    • Evaluate expressions
    • Update on changes
    • Conditional display
    
    Post-Mortem Debugging:
    ```python
    import pdb
    import sys
    
    def main():
        try:
            risky_operation()
        except Exception:
            # Drop into debugger on exception
            pdb.post_mortem(sys.exc_info()[2])
    ```
    
    Debugging Production:
    
    Logging:
    • Strategic log points
    • Correlation IDs
    • Structured logging
    • Log levels
    
    Profiling:
    • Performance data
    • Hot paths
    • Resource usage
    • Bottlenecks
    
    Distributed Tracing:
    • Request flow
    • Service interactions
    • Latency breakdown
    • Error propagation
    
    Core Dumps:
    • Process snapshots
    • Post-mortem analysis
    • Memory inspection
    • Stack traces
    """
    
    return {
        "messages": [AIMessage(content=f"🐛 Debug Controller:\n{response.content}\n{report}")],
        "breakpoints": breakpoints,
        "stack_frames": stack_frames,
        "variables": variables,
        "debug_session_active": True,
        "execution_step": "paused"
    }


# Variable Inspector
def variable_inspector(state: DebuggingState) -> DebuggingState:
    """Inspects and analyzes variables"""
    variables = state.get("variables", {})
    stack_frames = state.get("stack_frames", [])
    
    system_message = SystemMessage(content="""You are a variable inspector.
    Analyze variable values and identify potential issues.""")
    
    user_message = HumanMessage(content=f"""Inspect variables:

Variables: {len(variables)}
Stack Frames: {len(stack_frames)}

Analyze current state.""")
    
    response = llm.invoke([system_message, user_message])
    
    summary = f"""
    📊 DEBUGGING COMPLETE
    
    Debug Session Summary:
    • Breakpoints Set: {len(state.get('breakpoints', []))}
    • Stack Depth: {len(stack_frames)}
    • Variables Inspected: {len(variables)}
    • Session Active: {state.get('debug_session_active', False)}
    
    Current Stack Trace:
    {chr(10).join(f"  {i}. {frame['function']}() at {frame['file']}:{frame['line']}" for i, frame in enumerate(stack_frames))}
    
    Debugging Pattern Process:
    1. Debug Controller → Manage breakpoints and execution
    2. Variable Inspector → Analyze variable state
    
    Advanced Debugging Techniques:
    
    Time-Travel Debugging:
    • Record execution
    • Replay backwards
    • Inspect any state
    • Tools: rr, UndoDB
    
    Reverse Debugging:
    • Step backwards
    • Reverse continue
    • Find bug origin
    • GDB reverse mode
    
    Memory Debugging:
    • Valgrind
    • AddressSanitizer
    • Memory leaks
    • Buffer overflows
    
    Concurrency Debugging:
    • Thread inspection
    • Deadlock detection
    • Race conditions
    • Helgrind, TSan
    
    Debugging Best Practices:
    
    Reproduce Consistently:
    • Minimal test case
    • Isolated environment
    • Fixed inputs
    • Deterministic
    
    Hypothesis-Driven:
    • Form hypothesis
    • Design test
    • Verify result
    • Iterate
    
    Binary Search:
    • Divide problem space
    • Bisect history (git bisect)
    • Narrow down cause
    • Efficient approach
    
    Document Findings:
    • What you tried
    • What worked
    • Root cause
    • Fix applied
    
    Debugging Checklist:
    
    Before Debugging:
    □ Can you reproduce it?
    □ What changed recently?
    □ Check logs and metrics
    □ Review recent commits
    □ Check environment
    
    During Debugging:
    □ Set strategic breakpoints
    □ Inspect variable values
    □ Trace execution flow
    □ Check assumptions
    □ Test hypotheses
    
    After Debugging:
    □ Write regression test
    □ Document root cause
    □ Update runbook
    □ Share learnings
    □ Prevent recurrence
    
    Key Insight:
    Effective debugging combines systematic approaches
    with powerful tools to quickly identify and resolve
    issues in complex systems.
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Variable Inspector:\n{response.content}\n{summary}")]
    }


# Build the graph
def build_debugging_graph():
    """Build the debugging pattern graph"""
    workflow = StateGraph(DebuggingState)
    
    workflow.add_node("debug_controller", debug_controller)
    workflow.add_node("variable_inspector", variable_inspector)
    
    workflow.add_edge(START, "debug_controller")
    workflow.add_edge("debug_controller", "variable_inspector")
    workflow.add_edge("variable_inspector", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_debugging_graph()
    
    print("=== Debugging MCP Pattern ===\n")
    
    # Test Case: Interactive debugging session
    print("\n" + "="*70)
    print("TEST CASE: Debug Session with Breakpoints")
    print("="*70)
    
    state = {
        "messages": [],
        "breakpoints": [],
        "stack_frames": [],
        "variables": {},
        "debug_session_active": False,
        "execution_step": "run"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nDebug Session:")
    print(f"Breakpoints: {len(result.get('breakpoints', []))}")
    print(f"Stack Depth: {len(result.get('stack_frames', []))}")
    print(f"Variables: {len(result.get('variables', {}))}")

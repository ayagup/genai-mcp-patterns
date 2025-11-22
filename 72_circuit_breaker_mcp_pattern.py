"""
Circuit Breaker MCP Pattern

This pattern prevents cascading failures by monitoring error rates and
temporarily blocking requests to failing services.

Key Features:
- State management (Closed, Open, Half-Open)
- Failure threshold monitoring
- Automatic recovery attempts
- Fast-fail mechanism
- Circuit state transitions
"""

from typing import TypedDict, Sequence, Annotated
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class CircuitBreakerState(TypedDict):
    """State for circuit breaker pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_name: str
    circuit_state: str  # CLOSED, OPEN, HALF_OPEN
    failure_count: int
    success_count: int
    failure_threshold: int
    recovery_timeout: float
    last_failure_time: float
    request_count: int
    request_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Service Caller
def service_caller(state: CircuitBreakerState) -> CircuitBreakerState:
    """Calls service through circuit breaker"""
    service_name = state.get("service_name", "")
    circuit_state = state.get("circuit_state", "CLOSED")
    request_count = state.get("request_count", 0)
    
    system_message = SystemMessage(content="""You are a service caller. 
    Attempt to call the service based on circuit breaker state.""")
    
    user_message = HumanMessage(content=f"""Call service:

Service: {service_name}
Circuit State: {circuit_state}
Request: {request_count + 1}

Attempt service call.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate service behavior
    # Requests 1-3: fail, Requests 4-5: succeed (for demo)
    if request_count < 3:
        request_result = "FAILURE"
    else:
        request_result = "SUCCESS"
    
    return {
        "messages": [AIMessage(content=f"📞 Service Caller (Request {request_count + 1}):\n{response.content}\n\n⚡ Circuit State: {circuit_state}")],
        "request_count": request_count + 1,
        "request_result": request_result
    }


# Circuit Breaker Controller
def circuit_breaker_controller(state: CircuitBreakerState) -> CircuitBreakerState:
    """Controls circuit breaker state transitions"""
    service_name = state.get("service_name", "")
    circuit_state = state.get("circuit_state", "CLOSED")
    request_result = state.get("request_result", "")
    failure_count = state.get("failure_count", 0)
    success_count = state.get("success_count", 0)
    failure_threshold = state.get("failure_threshold", 3)
    recovery_timeout = state.get("recovery_timeout", 5.0)
    last_failure_time = state.get("last_failure_time", 0.0)
    
    system_message = SystemMessage(content="""You are a circuit breaker controller. 
    Manage circuit state based on success/failure patterns.""")
    
    current_time = time.time()
    
    # Update counters based on result
    if request_result == "SUCCESS":
        success_count += 1
        new_failure_count = 0  # Reset on success
    else:
        new_failure_count = failure_count + 1
        last_failure_time = current_time
    
    # Determine state transitions
    old_state = circuit_state
    
    if circuit_state == "CLOSED":
        # CLOSED -> OPEN if failures exceed threshold
        if new_failure_count >= failure_threshold:
            new_circuit_state = "OPEN"
            transition = "CLOSED → OPEN (threshold exceeded)"
        else:
            new_circuit_state = "CLOSED"
            transition = "CLOSED (monitoring)"
    
    elif circuit_state == "OPEN":
        # OPEN -> HALF_OPEN after recovery timeout
        if current_time - last_failure_time >= recovery_timeout:
            new_circuit_state = "HALF_OPEN"
            transition = "OPEN → HALF_OPEN (recovery timeout elapsed)"
        else:
            new_circuit_state = "OPEN"
            remaining = recovery_timeout - (current_time - last_failure_time)
            transition = f"OPEN (wait {remaining:.1f}s)"
    
    else:  # HALF_OPEN
        # HALF_OPEN -> CLOSED on success, or OPEN on failure
        if request_result == "SUCCESS":
            new_circuit_state = "CLOSED"
            transition = "HALF_OPEN → CLOSED (service recovered)"
        else:
            new_circuit_state = "OPEN"
            transition = "HALF_OPEN → OPEN (still failing)"
    
    user_message = HumanMessage(content=f"""Update circuit breaker:

Service: {service_name}
Old State: {old_state}
Request Result: {request_result}
Failures: {new_failure_count}/{failure_threshold}
Successes: {success_count}

Transition: {transition}""")
    
    response = llm.invoke([system_message, user_message])
    
    state_emoji = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}
    
    return {
        "messages": [AIMessage(content=f"🔧 Circuit Breaker Controller:\n{response.content}\n\n{state_emoji.get(new_circuit_state, '⚪')} State: {new_circuit_state} | {transition}")],
        "circuit_state": new_circuit_state,
        "failure_count": new_failure_count,
        "success_count": success_count,
        "last_failure_time": last_failure_time
    }


# Circuit Breaker Monitor
def circuit_breaker_monitor(state: CircuitBreakerState) -> CircuitBreakerState:
    """Monitors circuit breaker metrics"""
    service_name = state.get("service_name", "")
    circuit_state = state.get("circuit_state", "CLOSED")
    failure_count = state.get("failure_count", 0)
    success_count = state.get("success_count", 0)
    failure_threshold = state.get("failure_threshold", 3)
    request_count = state.get("request_count", 0)
    request_result = state.get("request_result", "")
    
    total_requests = request_count
    failure_rate = (failure_count / total_requests * 100) if total_requests > 0 else 0
    success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
    
    state_descriptions = {
        "CLOSED": "Circuit is CLOSED - requests flowing normally",
        "OPEN": "Circuit is OPEN - requests blocked (fast-fail)",
        "HALF_OPEN": "Circuit is HALF_OPEN - testing service recovery"
    }
    
    state_emoji = {"CLOSED": "🟢", "OPEN": "🔴", "HALF_OPEN": "🟡"}
    
    summary = f"""
    ✅ CIRCUIT BREAKER PATTERN - Request {request_count}
    
    Circuit Status:
    • Service: {service_name}
    • State: {state_emoji.get(circuit_state, '⚪')} {circuit_state}
    • Description: {state_descriptions.get(circuit_state, 'Unknown')}
    • Last Request: {request_result}
    
    Metrics:
    • Total Requests: {total_requests}
    • Failures: {failure_count} (threshold: {failure_threshold})
    • Successes: {success_count}
    • Failure Rate: {failure_rate:.1f}%
    • Success Rate: {success_rate:.1f}%
    
    Circuit States:
    • 🟢 CLOSED: Normal operation, monitoring failures
    • 🔴 OPEN: Service down, blocking requests (fast-fail)
    • 🟡 HALF_OPEN: Testing recovery, allowing limited requests
    
    State Transitions:
    • CLOSED → OPEN: When failures ≥ {failure_threshold}
    • OPEN → HALF_OPEN: After recovery timeout
    • HALF_OPEN → CLOSED: On successful request
    • HALF_OPEN → OPEN: On failed request
    
    Circuit Breaker Process:
    1. Monitor Requests → 2. Count Failures → 3. Trip Circuit (if threshold) → 
    4. Block Requests → 5. Test Recovery → 6. Resume if Healthy
    
    Circuit Breaker Benefits:
    • Prevents cascading failures
    • Fast-fail for failing services
    • Automatic recovery detection
    • Protects system resources
    • Improves resilience
    • Reduces latency during outages
    
    Key Insight:
    Circuit breaker acts like an electrical circuit breaker - it "trips" when
    too many failures occur, preventing further damage until service recovers.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Circuit Breaker Monitor:\n{summary}")]
    }


# Build the graph
def build_circuit_breaker_graph():
    """Build the circuit breaker pattern graph"""
    workflow = StateGraph(CircuitBreakerState)
    
    workflow.add_node("caller", service_caller)
    workflow.add_node("controller", circuit_breaker_controller)
    workflow.add_node("monitor", circuit_breaker_monitor)
    
    workflow.add_edge(START, "caller")
    workflow.add_edge("caller", "controller")
    workflow.add_edge("controller", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage - Multiple requests showing circuit breaker behavior
if __name__ == "__main__":
    graph = build_circuit_breaker_graph()
    
    print("=== Circuit Breaker MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "service_name": "Payment Processing Service",
        "circuit_state": "CLOSED",
        "failure_count": 0,
        "success_count": 0,
        "failure_threshold": 3,
        "recovery_timeout": 5.0,
        "last_failure_time": 0.0,
        "request_count": 0,
        "request_result": ""
    }
    
    # Simulate multiple requests
    num_requests = 6
    
    for i in range(num_requests):
        print(f"\n{'=' * 70}")
        print(f"REQUEST {i + 1}")
        print('=' * 70)
        
        # Check if circuit is OPEN and should block request
        if state.get("circuit_state") == "OPEN":
            current_time = time.time()
            last_failure = state.get("last_failure_time", 0)
            recovery_timeout = state.get("recovery_timeout", 5.0)
            
            if current_time - last_failure < recovery_timeout:
                print("\n🔴 CIRCUIT OPEN - REQUEST BLOCKED (Fast-fail)")
                print("Waiting for recovery timeout...")
                time.sleep(0.1)  # Short delay for demo
                continue
        
        result = graph.invoke(state)
        
        # Show messages
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next request
        state = {
            "messages": [],
            "service_name": state["service_name"],
            "circuit_state": result.get("circuit_state", "CLOSED"),
            "failure_count": result.get("failure_count", 0),
            "success_count": result.get("success_count", 0),
            "failure_threshold": state["failure_threshold"],
            "recovery_timeout": state["recovery_timeout"],
            "last_failure_time": result.get("last_failure_time", 0.0),
            "request_count": result.get("request_count", 0),
            "request_result": ""
        }
        
        # Small delay between requests
        time.sleep(0.1)
    
    print(f"\n\n{'=' * 70}")
    print("FINAL CIRCUIT BREAKER RESULTS")
    print('=' * 70)
    print(f"\nService: {state['service_name']}")
    print(f"Final State: {state['circuit_state']}")
    print(f"Total Requests: {state['request_count']}")
    print(f"Failures: {state['failure_count']}")
    print(f"Successes: {state['success_count']}")

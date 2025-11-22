"""
Redundancy MCP Pattern

This pattern uses duplicate resources and parallel execution to improve
reliability and availability through redundant components.

Key Features:
- Multiple redundant instances
- Parallel execution
- Result comparison
- Consensus mechanisms
- Fault tolerance
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class RedundancyState(TypedDict):
    """State for redundancy pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    num_redundant_instances: int
    instance_results: dict[str, dict[str, any]]  # instance_id -> {result, success, execution_time}
    consensus_result: str
    consensus_method: str
    success_count: int
    failure_count: int
    final_result: str
    reliability_score: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Redundant Instance 1
def redundant_instance_1(state: RedundancyState) -> RedundancyState:
    """First redundant instance executing the task"""
    task = state.get("task", "")
    
    system_message = SystemMessage(content="""You are redundant instance #1. 
    Execute tasks independently for reliability.""")
    
    user_message = HumanMessage(content=f"""Execute task (Instance 1):

Task: {task}

Process independently.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate execution
    instance_success = True
    instance_result = "Result from Instance 1: SUCCESS - Data processed correctly"
    execution_time = 1.2
    
    instance_results = state.get("instance_results", {})
    instance_results["instance_1"] = {
        "result": instance_result,
        "success": instance_success,
        "execution_time": execution_time
    }
    
    status = "✅ Success" if instance_success else "❌ Failed"
    
    return {
        "messages": [AIMessage(content=f"🔵 Instance 1:\n{response.content}\n\n{status} ({execution_time}s)")],
        "instance_results": instance_results
    }


# Redundant Instance 2
def redundant_instance_2(state: RedundancyState) -> RedundancyState:
    """Second redundant instance executing the task"""
    task = state.get("task", "")
    
    system_message = SystemMessage(content="""You are redundant instance #2. 
    Execute tasks independently for reliability.""")
    
    user_message = HumanMessage(content=f"""Execute task (Instance 2):

Task: {task}

Process independently.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate execution with similar result
    instance_success = True
    instance_result = "Result from Instance 2: SUCCESS - Data processed correctly"
    execution_time = 1.5
    
    instance_results = state.get("instance_results", {})
    instance_results["instance_2"] = {
        "result": instance_result,
        "success": instance_success,
        "execution_time": execution_time
    }
    
    status = "✅ Success" if instance_success else "❌ Failed"
    
    return {
        "messages": [AIMessage(content=f"🟢 Instance 2:\n{response.content}\n\n{status} ({execution_time}s)")],
        "instance_results": instance_results
    }


# Redundant Instance 3
def redundant_instance_3(state: RedundancyState) -> RedundancyState:
    """Third redundant instance executing the task"""
    task = state.get("task", "")
    
    system_message = SystemMessage(content="""You are redundant instance #3. 
    Execute tasks independently for reliability.""")
    
    user_message = HumanMessage(content=f"""Execute task (Instance 3):

Task: {task}

Process independently.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate execution - this one fails
    instance_success = False
    instance_result = "Result from Instance 3: FAILED - Network timeout"
    execution_time = 0.8
    
    instance_results = state.get("instance_results", {})
    instance_results["instance_3"] = {
        "result": instance_result,
        "success": instance_success,
        "execution_time": execution_time
    }
    
    status = "✅ Success" if instance_success else "❌ Failed"
    
    return {
        "messages": [AIMessage(content=f"🟡 Instance 3:\n{response.content}\n\n{status} ({execution_time}s)")],
        "instance_results": instance_results
    }


# Consensus Builder
def consensus_builder(state: RedundancyState) -> RedundancyState:
    """Builds consensus from redundant instance results"""
    task = state.get("task", "")
    instance_results = state.get("instance_results", {})
    num_instances = len(instance_results)
    
    system_message = SystemMessage(content="""You are a consensus builder. 
    Analyze results from redundant instances and build consensus.""")
    
    # Count successes and failures
    success_count = sum(1 for r in instance_results.values() if r.get("success", False))
    failure_count = num_instances - success_count
    
    results_summary = "\n".join([
        f"{iid}: {'✅ Success' if r.get('success') else '❌ Failed'} - {r.get('result', 'N/A')[:50]}..."
        for iid, r in instance_results.items()
    ])
    
    user_message = HumanMessage(content=f"""Build consensus:

Task: {task}
Total Instances: {num_instances}
Successful: {success_count}
Failed: {failure_count}

Instance Results:
{results_summary}

Determine consensus result using majority voting.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Use majority voting
    consensus_method = "Majority Voting"
    
    if success_count > failure_count:
        # Majority succeeded - use successful result
        successful_results = [r for r in instance_results.values() if r.get("success", False)]
        consensus_result = successful_results[0].get("result", "")
        final_result = "SUCCESS - Consensus from majority of instances"
    elif success_count == failure_count:
        # Tie - be conservative, consider as success if any succeeded
        if success_count > 0:
            successful_results = [r for r in instance_results.values() if r.get("success", False)]
            consensus_result = successful_results[0].get("result", "")
            final_result = "SUCCESS - Partial consensus (tie-breaker: success)"
        else:
            consensus_result = "All instances failed"
            final_result = "FAILED - No consensus available"
    else:
        # Majority failed
        consensus_result = "Majority failed"
        final_result = "FAILED - Majority of instances failed"
    
    reliability_score = (success_count / num_instances) * 100 if num_instances > 0 else 0
    
    consensus_summary = f"""
    📊 Consensus Analysis:
    
    • Method: {consensus_method}
    • Instances Polled: {num_instances}
    • Successful: {success_count} ({success_count/num_instances*100:.0f}%)
    • Failed: {failure_count} ({failure_count/num_instances*100:.0f}%)
    • Reliability Score: {reliability_score:.1f}%
    
    Consensus Result: {final_result}
    
    Voting Breakdown:
{chr(10).join(f'    {iid}: {"✅" if r.get("success") else "❌"}' for iid, r in instance_results.items())}
    """
    
    return {
        "messages": [AIMessage(content=f"🗳️ Consensus Builder:\n{response.content}\n{consensus_summary}")],
        "consensus_result": consensus_result,
        "consensus_method": consensus_method,
        "success_count": success_count,
        "failure_count": failure_count,
        "final_result": final_result,
        "reliability_score": reliability_score
    }


# Redundancy Monitor
def redundancy_monitor(state: RedundancyState) -> RedundancyState:
    """Monitors redundancy performance and provides insights"""
    task = state.get("task", "")
    num_redundant_instances = state.get("num_redundant_instances", 0)
    instance_results = state.get("instance_results", {})
    consensus_result = state.get("consensus_result", "")
    consensus_method = state.get("consensus_method", "")
    success_count = state.get("success_count", 0)
    failure_count = state.get("failure_count", 0)
    final_result = state.get("final_result", "")
    reliability_score = state.get("reliability_score", 0.0)
    
    # Calculate execution times
    exec_times = [r.get("execution_time", 0) for r in instance_results.values()]
    avg_exec_time = sum(exec_times) / len(exec_times) if exec_times else 0
    min_exec_time = min(exec_times) if exec_times else 0
    max_exec_time = max(exec_times) if exec_times else 0
    
    outcome_emoji = "✅" if "SUCCESS" in final_result else "❌"
    
    summary = f"""
    {outcome_emoji} REDUNDANCY PATTERN COMPLETE
    
    Task: {task}
    Final Result: {final_result}
    Reliability Score: {reliability_score:.1f}%
    
    Redundancy Configuration:
    • Total Instances: {num_redundant_instances}
    • Successful: {success_count}
    • Failed: {failure_count}
    • Consensus Method: {consensus_method}
    
    Performance Metrics:
    • Average Execution Time: {avg_exec_time:.2f}s
    • Fastest Instance: {min_exec_time:.2f}s
    • Slowest Instance: {max_exec_time:.2f}s
    
    Instance Details:
{chr(10).join(f'    {iid}: {"✅" if r.get("success") else "❌"} - {r.get("execution_time", 0):.2f}s' for iid, r in instance_results.items())}
    
    Redundancy Pattern Process:
    1. Spawn Instances → Create N redundant instances
    2. Execute Parallel → All instances process task independently
    3. Collect Results → Gather outputs from all instances
    4. Build Consensus → Use voting/comparison to determine result
    5. Return Result → Provide consensus result to caller
    
    Consensus Methods:
    • Majority Voting: Use result from majority of instances
    • First Success: Return first successful result (fastest)
    • All Success: Require all instances to succeed
    • Best Result: Compare results and choose best quality
    • Quorum: Require minimum N of M instances to agree
    
    Redundancy Pattern Benefits:
    • High availability (failure tolerance)
    • Increased reliability
    • Fault detection through comparison
    • No single point of failure
    • Continued operation despite failures
    • Error detection and correction
    
    Trade-offs:
    ✅ Pros:
    • Better reliability and availability
    • Fault tolerance
    • Can detect and correct errors
    • No downtime from single failure
    
    ❌ Cons:
    • Increased resource consumption (N× cost)
    • Higher latency (wait for consensus)
    • Complexity in result reconciliation
    • Network overhead
    
    Redundancy Strategies:
    • Active-Active: All instances always running
    • Active-Passive: Standby instances activated on failure
    • N+1: N instances + 1 spare
    • N+M: N instances + M spares
    • Geographic: Instances in different regions
    
    Best Practices:
    • Use odd number of instances (3, 5, 7) for voting
    • Monitor instance health continuously
    • Auto-replace failed instances
    • Use independent failure domains
    • Balance redundancy vs cost
    
    Use Cases:
    • Critical financial transactions
    • Safety-critical systems
    • High-availability services
    • Distributed databases (replication)
    • Load balancing with fault tolerance
    
    Key Insight:
    Redundancy pattern provides fault tolerance through duplicate resources,
    enabling systems to continue operating correctly even when components fail.
    Essential for high-availability and mission-critical systems.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Redundancy Monitor:\n{summary}")]
    }


# Build the graph
def build_redundancy_graph():
    """Build the redundancy pattern graph"""
    workflow = StateGraph(RedundancyState)
    
    # Add redundant instances (execute in parallel)
    workflow.add_node("instance1", redundant_instance_1)
    workflow.add_node("instance2", redundant_instance_2)
    workflow.add_node("instance3", redundant_instance_3)
    workflow.add_node("consensus", consensus_builder)
    workflow.add_node("monitor", redundancy_monitor)
    
    # Execute instances in parallel
    workflow.add_edge(START, "instance1")
    workflow.add_edge(START, "instance2")
    workflow.add_edge(START, "instance3")
    
    # All instances feed into consensus
    workflow.add_edge("instance1", "consensus")
    workflow.add_edge("instance2", "consensus")
    workflow.add_edge("instance3", "consensus")
    
    workflow.add_edge("consensus", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_redundancy_graph()
    
    print("=== Redundancy MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "task": "Process critical payment transaction",
        "num_redundant_instances": 3,
        "instance_results": {},
        "consensus_result": "",
        "consensus_method": "",
        "success_count": 0,
        "failure_count": 0,
        "final_result": "",
        "reliability_score": 0.0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("REDUNDANCY PATTERN DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nTask: {state['task']}")
    print(f"Redundant Instances: {state['num_redundant_instances']}")
    print(f"Final Result: {result.get('final_result', 'N/A')}")
    print(f"Reliability: {result.get('reliability_score', 0):.1f}%")
    print(f"Successful Instances: {result.get('success_count', 0)}/{state['num_redundant_instances']}")

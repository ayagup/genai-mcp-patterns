"""
Partitioning MCP Pattern

This pattern demonstrates partitioning workload based on specific criteria
such as data type, business domain, or processing requirements.

Key Features:
- Criteria-based partitioning
- Domain-driven distribution
- Specialized processing lanes
- Logical workload separation
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class PartitioningState(TypedDict):
    """State for partitioning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict]  # includes partition_type field
    partition_assignments: dict[str, list[str]]  # partition -> task_ids
    completed_tasks: list[str]
    results: dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Partition Router
def partition_router(state: PartitioningState) -> PartitioningState:
    """Routes tasks to appropriate partitions"""
    tasks = state.get("tasks", [])
    
    system_message = SystemMessage(content="""You are a partition router. Categorize tasks 
    into logical partitions (OLTP, OLAP, batch, streaming) based on their processing requirements.""")
    
    tasks_info = "\n".join([
        f"Task {t['id']}: {t['description']} (Type: {t.get('partition_type', 'oltp')})"
        for t in tasks
    ])
    
    user_message = HumanMessage(content=f"""Partition these tasks by processing type:
    
{tasks_info}

Partitions:
- OLTP: Online transaction processing (real-time, low latency)
- OLAP: Online analytical processing (complex queries, aggregations)
- Batch: Batch processing (scheduled, bulk operations)
- Streaming: Stream processing (continuous, event-driven)

Route tasks to appropriate partitions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Partition tasks by type
    partition_assignments = {
        "oltp": [],
        "olap": [],
        "batch": [],
        "streaming": []
    }
    
    for task in tasks:
        partition_type = task.get("partition_type", "oltp").lower()
        if partition_type in partition_assignments:
            partition_assignments[partition_type].append(task['id'])
    
    routing_summary = "\n".join([
        f"  {partition.upper()}: {len(tasks)} tasks"
        for partition, tasks in partition_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"🔀 Partition Router: {response.content}\n\nPartitions:\n{routing_summary}")],
        "partition_assignments": partition_assignments
    }


# OLTP Processor
def oltp_processor(state: PartitioningState) -> PartitioningState:
    """Handles online transaction processing"""
    assignments = state.get("partition_assignments", {}).get("oltp", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="⚡ OLTP Processor: No tasks in partition")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the OLTP processor. Handle real-time 
    transactional workloads with low latency. Focus on ACID compliance and quick responses.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these OLTP transactions:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[OLTP-Real-Time] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"⚡ OLTP Processor: {response.content}\n\nProcessed {len(my_tasks)} transactions (avg: 5ms)")],
        "completed_tasks": completed,
        "results": results
    }


# OLAP Processor
def olap_processor(state: PartitioningState) -> PartitioningState:
    """Handles online analytical processing"""
    assignments = state.get("partition_assignments", {}).get("olap", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="📊 OLAP Processor: No tasks in partition")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the OLAP processor. Handle complex 
    analytical queries, aggregations, and business intelligence workloads.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these OLAP analytics:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[OLAP-Analytics] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"📊 OLAP Processor: {response.content}\n\nProcessed {len(my_tasks)} analytics (avg: 500ms)")],
        "completed_tasks": completed,
        "results": results
    }


# Batch Processor
def batch_processor(state: PartitioningState) -> PartitioningState:
    """Handles batch processing"""
    assignments = state.get("partition_assignments", {}).get("batch", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="📦 Batch Processor: No tasks in partition")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the batch processor. Handle scheduled 
    bulk operations, ETL jobs, and large-scale data processing.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these batch jobs:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[Batch-Scheduled] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"📦 Batch Processor: {response.content}\n\nProcessed {len(my_tasks)} batch jobs")],
        "completed_tasks": completed,
        "results": results
    }


# Streaming Processor
def streaming_processor(state: PartitioningState) -> PartitioningState:
    """Handles stream processing"""
    assignments = state.get("partition_assignments", {}).get("streaming", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🌊 Streaming Processor: No tasks in partition")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are the streaming processor. Handle continuous 
    event-driven workloads, real-time data pipelines, and stream analytics.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these streaming events:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"[Streaming-Continuous] {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"🌊 Streaming Processor: {response.content}\n\nProcessed {len(my_tasks)} streams")],
        "completed_tasks": completed,
        "results": results
    }


# Partition Monitor
def partition_monitor(state: PartitioningState) -> PartitioningState:
    """Monitors partition distribution"""
    partition_assignments = state.get("partition_assignments", {})
    completed = state.get("completed_tasks", [])
    
    distribution = "\n".join([
        f"  {partition.upper()}: {len(tasks)} tasks"
        for partition, tasks in partition_assignments.items() if tasks
    ])
    
    summary = f"""
    ✅ PARTITIONING COMPLETE
    
    Total Tasks: {len(completed)}
    
    Partition Distribution:
{distribution}
    
    Partitioning Benefits:
    • Specialized processing for different workload types
    • Optimal resource allocation per partition
    • Clear separation of concerns
    • Independent scaling per partition
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Partition Monitor:\n{summary}")]
    }


# Build the graph
def build_partitioning_graph():
    """Build the partitioning MCP pattern graph"""
    workflow = StateGraph(PartitioningState)
    
    workflow.add_node("router", partition_router)
    workflow.add_node("oltp", oltp_processor)
    workflow.add_node("olap", olap_processor)
    workflow.add_node("batch", batch_processor)
    workflow.add_node("streaming", streaming_processor)
    workflow.add_node("monitor", partition_monitor)
    
    workflow.add_edge(START, "router")
    workflow.add_edge("router", "oltp")
    workflow.add_edge("router", "olap")
    workflow.add_edge("router", "batch")
    workflow.add_edge("router", "streaming")
    workflow.add_edge("oltp", "monitor")
    workflow.add_edge("olap", "monitor")
    workflow.add_edge("batch", "monitor")
    workflow.add_edge("streaming", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_partitioning_graph()
    
    print("=== Partitioning MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Process customer payment", "partition_type": "oltp"},
            {"id": "task_2", "description": "Update inventory count", "partition_type": "oltp"},
            {"id": "task_3", "description": "Generate sales report", "partition_type": "olap"},
            {"id": "task_4", "description": "Analyze customer behavior", "partition_type": "olap"},
            {"id": "task_5", "description": "Daily ETL pipeline", "partition_type": "batch"},
            {"id": "task_6", "description": "Monthly data migration", "partition_type": "batch"},
            {"id": "task_7", "description": "Process click stream", "partition_type": "streaming"},
            {"id": "task_8", "description": "Real-time event analytics", "partition_type": "streaming"},
        ],
        "partition_assignments": {"oltp": [], "olap": [], "batch": [], "streaming": []},
        "completed_tasks": [],
        "results": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Partitioning Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)

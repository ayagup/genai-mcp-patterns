"""
Map-Reduce MCP Pattern

This pattern demonstrates the classic Map-Reduce paradigm where tasks are
mapped to multiple workers for parallel processing, then results are reduced
to a final aggregated output.

Key Features:
- Map phase: Distribute work to mappers
- Parallel processing
- Reduce phase: Aggregate results
- Scalable data processing
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class MapReduceState(TypedDict):
    """State for map-reduce pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    input_data: list[dict]
    map_results: dict[str, list[dict]]  # mapper -> results
    reduce_result: dict


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Map Coordinator
def map_coordinator(state: MapReduceState) -> MapReduceState:
    """Coordinates the map phase"""
    input_data = state.get("input_data", [])
    
    system_message = SystemMessage(content="""You are a map coordinator. Distribute data 
    across mappers for parallel processing. Each mapper will process a chunk of data.""")
    
    data_summary = f"{len(input_data)} data items to process"
    
    user_message = HumanMessage(content=f"""Coordinate map phase for: {data_summary}

Distribute data evenly across 3 mappers for parallel processing.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Distribute data across mappers
    chunk_size = (len(input_data) + 2) // 3  # Divide into 3 chunks
    
    map_assignments = {
        "mapper_1": input_data[0:chunk_size],
        "mapper_2": input_data[chunk_size:2*chunk_size],
        "mapper_3": input_data[2*chunk_size:]
    }
    
    distribution = "\n".join([
        f"  {mapper}: {len(data)} items"
        for mapper, data in map_assignments.items()
    ])
    
    return {
        "messages": [AIMessage(content=f"🗺️ Map Coordinator: {response.content}\n\nData Distribution:\n{distribution}")]
    }


# Mapper 1
def mapper_1(state: MapReduceState) -> MapReduceState:
    """First mapper processes its data chunk"""
    input_data = state.get("input_data", [])
    chunk_size = (len(input_data) + 2) // 3
    my_data = input_data[0:chunk_size]
    
    if not my_data:
        return {"messages": [AIMessage(content="📊 Mapper 1: No data to process")]}
    
    system_message = SystemMessage(content="""You are Mapper 1. Process your data chunk 
    and extract key-value pairs for the reduce phase.""")
    
    data_desc = f"{len(my_data)} items: " + ", ".join([d.get("item", "unknown") for d in my_data[:3]])
    if len(my_data) > 3:
        data_desc += "..."
    
    user_message = HumanMessage(content=f"""Map phase - process: {data_desc}

Extract and transform data for reduction.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Process data (example: count, sum, group)
    map_results = state.get("map_results", {"mapper_1": [], "mapper_2": [], "mapper_3": []})
    map_results["mapper_1"] = [{"processed": d.get("item", ""), "value": d.get("value", 0)} for d in my_data]
    
    return {
        "messages": [AIMessage(content=f"📊 Mapper 1: {response.content}\n\nProcessed {len(my_data)} items")],
        "map_results": map_results
    }


# Mapper 2
def mapper_2(state: MapReduceState) -> MapReduceState:
    """Second mapper processes its data chunk"""
    input_data = state.get("input_data", [])
    chunk_size = (len(input_data) + 2) // 3
    my_data = input_data[chunk_size:2*chunk_size]
    
    if not my_data:
        return {"messages": [AIMessage(content="📊 Mapper 2: No data to process")]}
    
    system_message = SystemMessage(content="""You are Mapper 2. Process your data chunk 
    and extract key-value pairs for the reduce phase.""")
    
    data_desc = f"{len(my_data)} items: " + ", ".join([d.get("item", "unknown") for d in my_data[:3]])
    if len(my_data) > 3:
        data_desc += "..."
    
    user_message = HumanMessage(content=f"""Map phase - process: {data_desc}

Extract and transform data for reduction.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Process data
    map_results = state.get("map_results", {"mapper_1": [], "mapper_2": [], "mapper_3": []})
    map_results["mapper_2"] = [{"processed": d.get("item", ""), "value": d.get("value", 0)} for d in my_data]
    
    return {
        "messages": [AIMessage(content=f"📊 Mapper 2: {response.content}\n\nProcessed {len(my_data)} items")],
        "map_results": map_results
    }


# Mapper 3
def mapper_3(state: MapReduceState) -> MapReduceState:
    """Third mapper processes its data chunk"""
    input_data = state.get("input_data", [])
    chunk_size = (len(input_data) + 2) // 3
    my_data = input_data[2*chunk_size:]
    
    if not my_data:
        return {"messages": [AIMessage(content="📊 Mapper 3: No data to process")]}
    
    system_message = SystemMessage(content="""You are Mapper 3. Process your data chunk 
    and extract key-value pairs for the reduce phase.""")
    
    data_desc = f"{len(my_data)} items: " + ", ".join([d.get("item", "unknown") for d in my_data[:3]])
    if len(my_data) > 3:
        data_desc += "..."
    
    user_message = HumanMessage(content=f"""Map phase - process: {data_desc}

Extract and transform data for reduction.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Process data
    map_results = state.get("map_results", {"mapper_1": [], "mapper_2": [], "mapper_3": []})
    map_results["mapper_3"] = [{"processed": d.get("item", ""), "value": d.get("value", 0)} for d in my_data]
    
    return {
        "messages": [AIMessage(content=f"📊 Mapper 3: {response.content}\n\nProcessed {len(my_data)} items")],
        "map_results": map_results
    }


# Reducer
def reducer(state: MapReduceState) -> MapReduceState:
    """Reduces mapped results into final output"""
    map_results = state.get("map_results", {})
    
    system_message = SystemMessage(content="""You are a reducer. Aggregate results from 
    all mappers into a final combined output. Perform aggregation, summarization, or combination.""")
    
    # Collect all mapped results
    all_results = []
    for mapper, results in map_results.items():
        all_results.extend(results)
    
    results_summary = f"Total mapped items: {len(all_results)}"
    
    user_message = HumanMessage(content=f"""Reduce phase - aggregate results:

{results_summary}

Combine all mapper outputs into final result.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Perform reduction (example: sum values, count items)
    total_value = sum(item.get("value", 0) for item in all_results)
    item_count = len(all_results)
    
    reduce_result = {
        "total_items": item_count,
        "total_value": total_value,
        "average_value": total_value / item_count if item_count > 0 else 0,
        "mappers_used": len(map_results)
    }
    
    return {
        "messages": [AIMessage(content=f"🔄 Reducer: {response.content}\n\nReduction complete")],
        "reduce_result": reduce_result
    }


# Final Reporter
def final_reporter(state: MapReduceState) -> MapReduceState:
    """Reports final map-reduce results"""
    reduce_result = state.get("reduce_result", {})
    input_data = state.get("input_data", [])
    
    summary = f"""
    ✅ MAP-REDUCE COMPLETE
    
    Input Data: {len(input_data)} items
    
    Map Phase:
    • Data split across 3 mappers
    • Parallel processing completed
    
    Reduce Phase:
    • Results aggregated successfully
    
    Final Results:
    • Total Items Processed: {reduce_result.get("total_items", 0)}
    • Total Value: {reduce_result.get("total_value", 0)}
    • Average Value: {reduce_result.get("average_value", 0):.2f}
    • Mappers Used: {reduce_result.get("mappers_used", 0)}
    
    Map-Reduce pattern successfully executed!
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Final Reporter:\n{summary}")]
    }


# Build the graph
def build_map_reduce_graph():
    """Build the map-reduce MCP pattern graph"""
    workflow = StateGraph(MapReduceState)
    
    # Add nodes
    workflow.add_node("coordinator", map_coordinator)
    workflow.add_node("mapper_1", mapper_1)
    workflow.add_node("mapper_2", mapper_2)
    workflow.add_node("mapper_3", mapper_3)
    workflow.add_node("reducer", reducer)
    workflow.add_node("reporter", final_reporter)
    
    # Map phase
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "mapper_1")
    workflow.add_edge("coordinator", "mapper_2")
    workflow.add_edge("coordinator", "mapper_3")
    
    # Reduce phase
    workflow.add_edge("mapper_1", "reducer")
    workflow.add_edge("mapper_2", "reducer")
    workflow.add_edge("mapper_3", "reducer")
    workflow.add_edge("reducer", "reporter")
    workflow.add_edge("reporter", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_map_reduce_graph()
    
    print("=== Map-Reduce MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "input_data": [
            {"item": "product_A", "value": 100},
            {"item": "product_B", "value": 150},
            {"item": "product_C", "value": 200},
            {"item": "product_D", "value": 120},
            {"item": "product_E", "value": 180},
            {"item": "product_F", "value": 90},
            {"item": "product_G", "value": 160},
            {"item": "product_H", "value": 140},
            {"item": "product_I", "value": 110},
            {"item": "product_J", "value": 130},
        ],
        "map_results": {"mapper_1": [], "mapper_2": [], "mapper_3": []},
        "reduce_result": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Map-Reduce Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Reduce Result ===")
    print(result.get("reduce_result", {}))

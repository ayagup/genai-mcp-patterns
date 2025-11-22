"""
Streaming MCP Pattern
======================
Continuous data streaming where data flows from source to consumers
in real-time without waiting for complete dataset.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator


# Define the state
class StreamingState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    stream_data: List[str]  # Streaming data chunks
    processed_chunks: List[str]  # Processed stream chunks
    stream_metadata: Dict[str, any]  # Metadata about the stream


# Stream Source Agent
def stream_source_agent(state: StreamingState):
    """Agent that generates streaming data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Stream Source. Generate continuous streaming data chunks. "
                "Create 5 sequential data chunks that represent a data stream."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Generate stream chunks
    stream_data = []
    content = response.content
    chunk_size = len(content) // 5
    
    for i in range(5):
        start = i * chunk_size
        end = start + chunk_size if i < 4 else len(content)
        chunk = content[start:end]
        stream_data.append(f"Chunk {i+1}: {chunk}")
    
    metadata = {
        "total_chunks": len(stream_data),
        "source": "stream_source_agent",
        "stream_active": True
    }
    
    return {
        "messages": [AIMessage(content=f"Stream Source: Streaming {len(stream_data)} chunks")],
        "stream_data": stream_data,
        "stream_metadata": metadata
    }


# Stream Processor 1 - Filter
def stream_filter_agent(state: StreamingState):
    """Agent that filters streaming data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    stream_data = state.get("stream_data", [])
    
    if not stream_data:
        return {"messages": [], "processed_chunks": []}
    
    # Process first 2 chunks
    chunks_to_process = stream_data[:2]
    
    system_msg = SystemMessage(
        content=f"You are a Stream Filter. Filter these streaming data chunks:\n"
                f"Chunks: {chunks_to_process}\n"
                "Extract and filter relevant information from the stream."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    processed = state.get("processed_chunks", [])
    processed.extend([f"Filtered: {chunk}" for chunk in chunks_to_process])
    
    return {
        "messages": [AIMessage(content=f"Stream Filter: Filtered {len(chunks_to_process)} chunks - {response.content}")],
        "processed_chunks": processed
    }


# Stream Processor 2 - Transform
def stream_transform_agent(state: StreamingState):
    """Agent that transforms streaming data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    stream_data = state.get("stream_data", [])
    
    if len(stream_data) < 3:
        return {"messages": [], "processed_chunks": state.get("processed_chunks", [])}
    
    # Process next 2 chunks
    chunks_to_process = stream_data[2:4]
    
    system_msg = SystemMessage(
        content=f"You are a Stream Transformer. Transform these streaming data chunks:\n"
                f"Chunks: {chunks_to_process}\n"
                "Transform and enrich the streaming data."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    processed = state.get("processed_chunks", [])
    processed.extend([f"Transformed: {chunk}" for chunk in chunks_to_process])
    
    return {
        "messages": [AIMessage(content=f"Stream Transform: Transformed {len(chunks_to_process)} chunks - {response.content}")],
        "processed_chunks": processed
    }


# Stream Processor 3 - Aggregate
def stream_aggregate_agent(state: StreamingState):
    """Agent that aggregates streaming data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    stream_data = state.get("stream_data", [])
    
    if len(stream_data) < 5:
        return {"messages": [], "processed_chunks": state.get("processed_chunks", [])}
    
    # Process final chunk
    chunks_to_process = stream_data[4:]
    
    system_msg = SystemMessage(
        content=f"You are a Stream Aggregator. Aggregate these streaming data chunks:\n"
                f"Chunks: {chunks_to_process}\n"
                "Aggregate and summarize the streaming data."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    processed = state.get("processed_chunks", [])
    processed.extend([f"Aggregated: {chunk}" for chunk in chunks_to_process])
    
    return {
        "messages": [AIMessage(content=f"Stream Aggregate: Aggregated {len(chunks_to_process)} chunks - {response.content}")],
        "processed_chunks": processed
    }


# Stream Monitor
def stream_monitor_agent(state: StreamingState):
    """Monitor streaming performance and status."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    metadata = state.get("stream_metadata", {})
    stream_data = state.get("stream_data", [])
    processed = state.get("processed_chunks", [])
    
    system_msg = SystemMessage(
        content=f"You are a Stream Monitor. Report streaming statistics:\n"
                f"Total Chunks: {len(stream_data)}\n"
                f"Processed Chunks: {len(processed)}\n"
                f"Stream Active: {metadata.get('stream_active', False)}\n"
                "Provide streaming performance summary."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Stream Monitor: {response.content}")]
    }


# Build the streaming graph
def create_streaming_graph():
    """Create a streaming workflow graph."""
    workflow = StateGraph(StreamingState)
    
    # Add nodes
    workflow.add_node("source", stream_source_agent)
    workflow.add_node("filter", stream_filter_agent)
    workflow.add_node("transform", stream_transform_agent)
    workflow.add_node("aggregate", stream_aggregate_agent)
    workflow.add_node("monitor", stream_monitor_agent)
    
    # Streaming flow: Source -> Filter -> Transform -> Aggregate -> Monitor
    workflow.add_edge(START, "source")
    workflow.add_edge("source", "filter")
    workflow.add_edge("filter", "transform")
    workflow.add_edge("transform", "aggregate")
    workflow.add_edge("aggregate", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_streaming_graph()
    
    print("=" * 60)
    print("STREAMING MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Real-time data streaming
    print("\n[Scenario: Real-time Log Processing]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Stream and process real-time system logs for analysis")],
        "stream_data": [],
        "processed_chunks": [],
        "stream_metadata": {}
    })
    
    print("\n--- Streaming Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Stream Chunks ---")
    for i, chunk in enumerate(result.get("stream_data", []), 1):
        print(f"{i}. {chunk[:80]}...")
    
    print(f"\n--- Processed Stream ---")
    for i, chunk in enumerate(result.get("processed_chunks", []), 1):
        print(f"{i}. {chunk[:80]}...")
    
    print(f"\n--- Stream Metadata ---")
    metadata = result.get("stream_metadata", {})
    for key, value in metadata.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 60)

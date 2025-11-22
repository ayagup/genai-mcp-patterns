"""
Bidirectional Streaming MCP Pattern
====================================
Two-way streaming communication where both client and server send
streams of messages to each other simultaneously.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator


# Define the state
class BidirectionalStreamState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    client_stream: List[str]  # Stream from client
    server_stream: List[str]  # Stream from server
    bidirectional_exchange: List[Dict[str, str]]  # Exchange log
    stream_status: str


# Client Streaming Agent
def client_stream_agent(state: BidirectionalStreamState):
    """Client that sends streaming data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Client in bidirectional streaming. "
                "Generate a stream of 3 request chunks to send to the server."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Generate client stream
    client_stream = []
    content = response.content
    chunk_size = len(content) // 3
    
    for i in range(3):
        start = i * chunk_size
        end = start + chunk_size if i < 2 else len(content)
        chunk = f"Client-Chunk-{i+1}: {content[start:end]}"
        client_stream.append(chunk)
    
    # Log exchange
    exchange = state.get("bidirectional_exchange", [])
    exchange.append({
        "direction": "client_to_server",
        "chunks": len(client_stream),
        "timestamp": f"T{len(exchange)}"
    })
    
    return {
        "messages": [AIMessage(content=f"Client: Streaming {len(client_stream)} chunks to server")],
        "client_stream": client_stream,
        "bidirectional_exchange": exchange,
        "stream_status": "client_streaming"
    }


# Server Streaming Agent
def server_stream_agent(state: BidirectionalStreamState):
    """Server that sends streaming data while receiving."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    client_stream = state.get("client_stream", [])
    
    system_msg = SystemMessage(
        content=f"You are a Server in bidirectional streaming. "
                f"You received client stream: {client_stream}\n"
                "Generate a stream of 3 response chunks to send back to client."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Generate server stream
    server_stream = []
    content = response.content
    chunk_size = len(content) // 3
    
    for i in range(3):
        start = i * chunk_size
        end = start + chunk_size if i < 2 else len(content)
        chunk = f"Server-Chunk-{i+1}: {content[start:end]}"
        server_stream.append(chunk)
    
    # Log exchange
    exchange = state.get("bidirectional_exchange", [])
    exchange.append({
        "direction": "server_to_client",
        "chunks": len(server_stream),
        "timestamp": f"T{len(exchange)}"
    })
    
    return {
        "messages": [AIMessage(content=f"Server: Streaming {len(server_stream)} chunks to client")],
        "server_stream": server_stream,
        "bidirectional_exchange": exchange,
        "stream_status": "server_streaming"
    }


# Client Response Processor
def client_response_processor(state: BidirectionalStreamState):
    """Client processes server's streaming response."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    server_stream = state.get("server_stream", [])
    
    system_msg = SystemMessage(
        content=f"You are the Client processing server's streaming response:\n"
                f"Server Stream: {server_stream}\n"
                "Process the streamed responses and send additional requests if needed."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Generate follow-up client stream
    client_stream = state.get("client_stream", [])
    client_stream.append(f"Client-Follow-up: {response.content[:100]}")
    
    # Log exchange
    exchange = state.get("bidirectional_exchange", [])
    exchange.append({
        "direction": "client_follow_up",
        "chunks": 1,
        "timestamp": f"T{len(exchange)}"
    })
    
    return {
        "messages": [AIMessage(content=f"Client: Processed server stream and sent follow-up - {response.content}")],
        "client_stream": client_stream,
        "bidirectional_exchange": exchange,
        "stream_status": "bidirectional_active"
    }


# Server Final Response
def server_final_response(state: BidirectionalStreamState):
    """Server sends final streaming response."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    client_stream = state.get("client_stream", [])
    
    system_msg = SystemMessage(
        content=f"You are the Server sending final streaming response.\n"
                f"All client messages: {client_stream}\n"
                "Send final stream chunks to complete the bidirectional exchange."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    server_stream = state.get("server_stream", [])
    server_stream.append(f"Server-Final: {response.content[:100]}")
    
    # Log exchange
    exchange = state.get("bidirectional_exchange", [])
    exchange.append({
        "direction": "server_final",
        "chunks": 1,
        "timestamp": f"T{len(exchange)}"
    })
    
    return {
        "messages": [AIMessage(content=f"Server: Final stream complete - {response.content}")],
        "server_stream": server_stream,
        "bidirectional_exchange": exchange,
        "stream_status": "streams_complete"
    }


# Build the bidirectional streaming graph
def create_bidirectional_stream_graph():
    """Create a bidirectional streaming workflow graph."""
    workflow = StateGraph(BidirectionalStreamState)
    
    # Add nodes
    workflow.add_node("client_stream", client_stream_agent)
    workflow.add_node("server_stream", server_stream_agent)
    workflow.add_node("client_response", client_response_processor)
    workflow.add_node("server_final", server_final_response)
    
    # Bidirectional flow: Client <-> Server
    workflow.add_edge(START, "client_stream")
    workflow.add_edge("client_stream", "server_stream")
    workflow.add_edge("server_stream", "client_response")
    workflow.add_edge("client_response", "server_final")
    workflow.add_edge("server_final", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_bidirectional_stream_graph()
    
    print("=" * 60)
    print("BIDIRECTIONAL STREAMING MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Two-way streaming communication
    print("\n[Scenario: Real-time Chat Conversation]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Establish bidirectional streaming for real-time conversation")],
        "client_stream": [],
        "server_stream": [],
        "bidirectional_exchange": [],
        "stream_status": "initializing"
    })
    
    print("\n--- Bidirectional Streaming Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Client Stream ({len(result.get('client_stream', []))} chunks) ---")
    for i, chunk in enumerate(result.get("client_stream", []), 1):
        print(f"{i}. {chunk[:80]}...")
    
    print(f"\n--- Server Stream ({len(result.get('server_stream', []))} chunks) ---")
    for i, chunk in enumerate(result.get("server_stream", []), 1):
        print(f"{i}. {chunk[:80]}...")
    
    print(f"\n--- Bidirectional Exchange Log ---")
    for exchange in result.get("bidirectional_exchange", []):
        print(f"  {exchange['timestamp']}: {exchange['direction']} ({exchange['chunks']} chunks)")
    
    print(f"\n--- Stream Status: {result.get('stream_status')} ---")
    
    print("\n" + "=" * 60)

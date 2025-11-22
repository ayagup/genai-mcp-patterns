"""
Asynchronous Messaging MCP Pattern
===================================
Non-blocking asynchronous communication where senders don't wait
for receivers to process messages. Messages are sent and processed independently.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator


# Define the state
class AsyncMessagingState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    async_queue: List[Dict[str, str]]  # Async message queue
    pending_messages: List[str]  # Messages waiting to be processed
    processed_messages: List[str]  # Completed messages
    message_status: Dict[str, str]  # Message ID -> status


# Async Sender 1
def async_sender_1(state: AsyncMessagingState):
    """Sender that sends messages asynchronously without waiting."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are Async Sender 1. Send messages asynchronously without waiting for response. "
                "Generate 2 messages to send."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    async_queue = state.get("async_queue", [])
    pending = state.get("pending_messages", [])
    status = state.get("message_status", {})
    
    # Send messages asynchronously (non-blocking)
    msg1_id = "async_msg_1"
    msg2_id = "async_msg_2"
    
    async_queue.append({"id": msg1_id, "sender": "sender_1", "data": response.content[:100]})
    async_queue.append({"id": msg2_id, "sender": "sender_1", "data": response.content[100:200]})
    
    pending.extend([msg1_id, msg2_id])
    status[msg1_id] = "pending"
    status[msg2_id] = "pending"
    
    return {
        "messages": [AIMessage(content=f"Async Sender 1: Sent 2 messages (non-blocking) - {response.content}")],
        "async_queue": async_queue,
        "pending_messages": pending,
        "message_status": status
    }


# Async Sender 2
def async_sender_2(state: AsyncMessagingState):
    """Another sender sending asynchronously."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are Async Sender 2. Send messages asynchronously without waiting for response. "
                "Generate 2 different messages to send."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    async_queue = state.get("async_queue", [])
    pending = state.get("pending_messages", [])
    status = state.get("message_status", {})
    
    # Send messages asynchronously (non-blocking)
    msg3_id = "async_msg_3"
    msg4_id = "async_msg_4"
    
    async_queue.append({"id": msg3_id, "sender": "sender_2", "data": response.content[:100]})
    async_queue.append({"id": msg4_id, "sender": "sender_2", "data": response.content[100:200]})
    
    pending.extend([msg3_id, msg4_id])
    status[msg3_id] = "pending"
    status[msg4_id] = "pending"
    
    return {
        "messages": [AIMessage(content=f"Async Sender 2: Sent 2 messages (non-blocking) - {response.content}")],
        "async_queue": async_queue,
        "pending_messages": pending,
        "message_status": status
    }


# Async Processor 1
def async_processor_1(state: AsyncMessagingState):
    """Processor that handles messages asynchronously."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    async_queue = state.get("async_queue", [])
    pending = state.get("pending_messages", [])
    processed = state.get("processed_messages", [])
    status = state.get("message_status", {})
    
    # Process first 2 pending messages asynchronously
    messages_to_process = async_queue[:2] if len(async_queue) >= 2 else async_queue
    
    if not messages_to_process:
        return {"messages": [], "processed_messages": [], "message_status": {}}
    
    system_msg = SystemMessage(
        content=f"You are Async Processor 1. Process these messages asynchronously:\n"
                f"Messages: {messages_to_process}\n"
                "Process independently without blocking."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update status asynchronously
    for msg in messages_to_process:
        msg_id = msg["id"]
        if msg_id in pending:
            processed.append(msg_id)
            status[msg_id] = "processed"
    
    return {
        "messages": [AIMessage(content=f"Async Processor 1: Processed {len(messages_to_process)} messages - {response.content}")],
        "processed_messages": processed,
        "message_status": status
    }


# Async Processor 2
def async_processor_2(state: AsyncMessagingState):
    """Another processor handling messages asynchronously."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    async_queue = state.get("async_queue", [])
    pending = state.get("pending_messages", [])
    processed = state.get("processed_messages", [])
    status = state.get("message_status", {})
    
    # Process remaining messages asynchronously
    processed_ids = set(processed)
    messages_to_process = [msg for msg in async_queue if msg["id"] not in processed_ids]
    
    if not messages_to_process:
        return {"messages": [], "processed_messages": processed, "message_status": status}
    
    system_msg = SystemMessage(
        content=f"You are Async Processor 2. Process these messages asynchronously:\n"
                f"Messages: {messages_to_process}\n"
                "Process independently without blocking."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update status asynchronously
    for msg in messages_to_process:
        msg_id = msg["id"]
        if msg_id not in processed_ids:
            processed.append(msg_id)
            status[msg_id] = "processed"
    
    return {
        "messages": [AIMessage(content=f"Async Processor 2: Processed {len(messages_to_process)} messages - {response.content}")],
        "processed_messages": processed,
        "message_status": status
    }


# Async Monitor
def async_monitor(state: AsyncMessagingState):
    """Monitor asynchronous message processing status."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    async_queue = state.get("async_queue", [])
    pending = state.get("pending_messages", [])
    processed = state.get("processed_messages", [])
    status = state.get("message_status", {})
    
    pending_count = sum(1 for s in status.values() if s == "pending")
    processed_count = sum(1 for s in status.values() if s == "processed")
    
    system_msg = SystemMessage(
        content=f"You are an Async Monitor. Report asynchronous messaging status:\n"
                f"Total Messages in Queue: {len(async_queue)}\n"
                f"Pending: {pending_count}\n"
                f"Processed: {processed_count}\n"
                f"Processing Rate: {(processed_count/len(async_queue)*100) if async_queue else 0:.1f}%\n"
                "Provide async messaging summary."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Async Monitor: {response.content}")]
    }


# Build the async messaging graph
def create_async_messaging_graph():
    """Create an asynchronous messaging workflow graph."""
    workflow = StateGraph(AsyncMessagingState)
    
    # Add nodes
    workflow.add_node("sender_1", async_sender_1)
    workflow.add_node("sender_2", async_sender_2)
    workflow.add_node("processor_1", async_processor_1)
    workflow.add_node("processor_2", async_processor_2)
    workflow.add_node("monitor", async_monitor)
    
    # Async flow: Senders (parallel) -> Processors (parallel) -> Monitor
    workflow.add_edge(START, "sender_1")
    workflow.add_edge(START, "sender_2")
    workflow.add_edge("sender_1", "processor_1")
    workflow.add_edge("sender_2", "processor_1")
    workflow.add_edge("processor_1", "processor_2")
    workflow.add_edge("processor_2", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_async_messaging_graph()
    
    print("=" * 60)
    print("ASYNCHRONOUS MESSAGING MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Non-blocking message processing
    print("\n[Scenario: Async Task Processing System]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Process multiple tasks asynchronously without blocking")],
        "async_queue": [],
        "pending_messages": [],
        "processed_messages": [],
        "message_status": {}
    })
    
    print("\n--- Async Messaging Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Async Message Queue ({len(result.get('async_queue', []))}) ---")
    for msg in result.get("async_queue", []):
        print(f"  ID: {msg['id']}, Sender: {msg['sender']}, Status: {result.get('message_status', {}).get(msg['id'], 'unknown')}")
    
    print(f"\n--- Message Status ---")
    for msg_id, status in result.get("message_status", {}).items():
        print(f"  {msg_id}: {status}")
    
    print(f"\n--- Processing Summary ---")
    print(f"Total Pending: {len(result.get('pending_messages', []))}")
    print(f"Total Processed: {len(result.get('processed_messages', []))}")
    
    print("\n" + "=" * 60)

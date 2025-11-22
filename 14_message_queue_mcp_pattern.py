"""
Message Queue MCP Pattern
==========================
FIFO (First-In-First-Out) message queue where producers add messages
and consumers process them sequentially in order.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator
from collections import deque


# Define the state
class MessageQueueState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    message_queue: List[Dict[str, str]]  # FIFO queue of messages
    processed_messages: List[Dict[str, str]]  # Successfully processed messages
    queue_stats: Dict[str, int]  # Queue statistics


# Producer Agent 1
def producer_1_agent(state: MessageQueueState):
    """Producer that adds messages to the queue."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are Producer 1. Generate messages to add to the message queue. "
                "Create 2-3 task messages for processing."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    queue = state.get("message_queue", [])
    
    # Add messages to queue (FIFO)
    queue.append({"id": "msg_1", "producer": "producer_1", "data": response.content[:100]})
    queue.append({"id": "msg_2", "producer": "producer_1", "data": response.content[100:200]})
    
    return {
        "messages": [AIMessage(content=f"Producer 1: Added messages to queue - {response.content}")],
        "message_queue": queue
    }


# Producer Agent 2
def producer_2_agent(state: MessageQueueState):
    """Another producer that adds messages to the queue."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are Producer 2. Generate messages to add to the message queue. "
                "Create 2-3 different task messages for processing."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    queue = state.get("message_queue", [])
    
    # Add messages to queue (FIFO)
    queue.append({"id": "msg_3", "producer": "producer_2", "data": response.content[:100]})
    queue.append({"id": "msg_4", "producer": "producer_2", "data": response.content[100:200]})
    
    return {
        "messages": [AIMessage(content=f"Producer 2: Added messages to queue - {response.content}")],
        "message_queue": queue
    }


# Consumer Agent 1
def consumer_1_agent(state: MessageQueueState):
    """Consumer that processes messages from the queue in FIFO order."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    queue = state.get("message_queue", [])
    processed = state.get("processed_messages", [])
    
    # Process first 2 messages from queue (FIFO)
    messages_to_process = queue[:2] if len(queue) >= 2 else queue
    
    if not messages_to_process:
        return {"messages": [], "processed_messages": []}
    
    system_msg = SystemMessage(
        content=f"You are Consumer 1. Process these messages from the queue (FIFO order):\n"
                f"Messages: {messages_to_process}\n"
                "Process each message in order."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Mark messages as processed
    processed.extend(messages_to_process)
    
    return {
        "messages": [AIMessage(content=f"Consumer 1: Processed {len(messages_to_process)} messages - {response.content}")],
        "processed_messages": processed
    }


# Consumer Agent 2
def consumer_2_agent(state: MessageQueueState):
    """Another consumer that processes remaining messages."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    queue = state.get("message_queue", [])
    processed = state.get("processed_messages", [])
    
    # Process remaining messages from queue
    processed_ids = {msg["id"] for msg in processed}
    messages_to_process = [msg for msg in queue if msg["id"] not in processed_ids]
    
    if not messages_to_process:
        return {"messages": [], "processed_messages": processed}
    
    system_msg = SystemMessage(
        content=f"You are Consumer 2. Process these messages from the queue (FIFO order):\n"
                f"Messages: {messages_to_process}\n"
                "Process each message in order."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Mark messages as processed
    processed.extend(messages_to_process)
    
    return {
        "messages": [AIMessage(content=f"Consumer 2: Processed {len(messages_to_process)} messages - {response.content}")],
        "processed_messages": processed
    }


# Queue Monitor
def queue_monitor(state: MessageQueueState):
    """Monitor queue statistics and status."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    queue = state.get("message_queue", [])
    processed = state.get("processed_messages", [])
    
    stats = {
        "total_enqueued": len(queue),
        "total_processed": len(processed),
        "remaining": len(queue) - len(processed)
    }
    
    system_msg = SystemMessage(
        content=f"You are a Queue Monitor. Report queue statistics:\n"
                f"Total Enqueued: {stats['total_enqueued']}\n"
                f"Total Processed: {stats['total_processed']}\n"
                f"Remaining: {stats['remaining']}\n"
                "Provide a status summary."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Queue Monitor: {response.content}")],
        "queue_stats": stats
    }


# Build the message queue graph
def create_message_queue_graph():
    """Create a message queue workflow graph."""
    workflow = StateGraph(MessageQueueState)
    
    # Add nodes
    workflow.add_node("producer_1", producer_1_agent)
    workflow.add_node("producer_2", producer_2_agent)
    workflow.add_node("consumer_1", consumer_1_agent)
    workflow.add_node("consumer_2", consumer_2_agent)
    workflow.add_node("monitor", queue_monitor)
    
    # Message Queue flow: Producers -> Consumers -> Monitor
    workflow.add_edge(START, "producer_1")
    workflow.add_edge(START, "producer_2")
    workflow.add_edge("producer_1", "consumer_1")
    workflow.add_edge("producer_2", "consumer_1")
    workflow.add_edge("consumer_1", "consumer_2")
    workflow.add_edge("consumer_2", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_message_queue_graph()
    
    print("=" * 60)
    print("MESSAGE QUEUE MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: FIFO message processing
    print("\n[Scenario: Task Queue Processing]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Process incoming data analysis tasks and report generation requests")],
        "message_queue": [],
        "processed_messages": [],
        "queue_stats": {}
    })
    
    print("\n--- Message Queue Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Queue Contents (FIFO) ---")
    for i, msg in enumerate(result.get("message_queue", []), 1):
        print(f"{i}. ID: {msg['id']}, Producer: {msg['producer']}")
    
    print(f"\n--- Processed Messages ---")
    for msg in result.get("processed_messages", []):
        print(f"  - {msg['id']} from {msg['producer']}")
    
    print(f"\n--- Queue Statistics ---")
    stats = result.get("queue_stats", {})
    print(f"Total Enqueued: {stats.get('total_enqueued', 0)}")
    print(f"Total Processed: {stats.get('total_processed', 0)}")
    print(f"Remaining: {stats.get('remaining', 0)}")
    
    print("\n" + "=" * 60)

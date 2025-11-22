"""
Broadcast MCP Pattern
======================
One-to-many communication where a broadcaster sends the same message
to all registered receivers simultaneously.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator


# Define the state
class BroadcastState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    broadcast_message: str  # Message to broadcast
    receivers: List[str]  # List of receiver IDs
    receiver_responses: Dict[str, str]  # Responses from each receiver
    broadcast_id: str


# Broadcaster Agent
def broadcaster_agent(state: BroadcastState):
    """Agent that broadcasts messages to all receivers."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Broadcaster. Create an important message to broadcast "
                "to all receivers in the network. This message will reach everyone simultaneously."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Identify all receivers
    receivers = ["receiver_1", "receiver_2", "receiver_3", "receiver_4"]
    
    return {
        "messages": [AIMessage(content=f"Broadcaster: Broadcasting to all receivers - {response.content}")],
        "broadcast_message": response.content,
        "receivers": receivers,
        "broadcast_id": "BROADCAST_001"
    }


# Receiver 1
def receiver_1_agent(state: BroadcastState):
    """Receiver that processes broadcast message."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    broadcast_msg = state.get("broadcast_message", "No broadcast")
    broadcast_id = state.get("broadcast_id", "")
    
    system_msg = SystemMessage(
        content=f"You are Receiver 1. You received this broadcast message:\n"
                f"Broadcast ID: {broadcast_id}\n"
                f"Message: {broadcast_msg}\n"
                "Process and respond to the broadcast."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    responses = state.get("receiver_responses", {})
    responses["receiver_1"] = response.content
    
    return {
        "messages": [AIMessage(content=f"Receiver 1: Received broadcast - {response.content}")],
        "receiver_responses": responses
    }


# Receiver 2
def receiver_2_agent(state: BroadcastState):
    """Receiver that processes broadcast message."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    broadcast_msg = state.get("broadcast_message", "No broadcast")
    broadcast_id = state.get("broadcast_id", "")
    
    system_msg = SystemMessage(
        content=f"You are Receiver 2. You received this broadcast message:\n"
                f"Broadcast ID: {broadcast_id}\n"
                f"Message: {broadcast_msg}\n"
                "Process and respond to the broadcast."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    responses = state.get("receiver_responses", {})
    responses["receiver_2"] = response.content
    
    return {
        "messages": [AIMessage(content=f"Receiver 2: Received broadcast - {response.content}")],
        "receiver_responses": responses
    }


# Receiver 3
def receiver_3_agent(state: BroadcastState):
    """Receiver that processes broadcast message."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    broadcast_msg = state.get("broadcast_message", "No broadcast")
    broadcast_id = state.get("broadcast_id", "")
    
    system_msg = SystemMessage(
        content=f"You are Receiver 3. You received this broadcast message:\n"
                f"Broadcast ID: {broadcast_id}\n"
                f"Message: {broadcast_msg}\n"
                "Process and respond to the broadcast."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    responses = state.get("receiver_responses", {})
    responses["receiver_3"] = response.content
    
    return {
        "messages": [AIMessage(content=f"Receiver 3: Received broadcast - {response.content}")],
        "receiver_responses": responses
    }


# Receiver 4
def receiver_4_agent(state: BroadcastState):
    """Receiver that processes broadcast message."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    broadcast_msg = state.get("broadcast_message", "No broadcast")
    broadcast_id = state.get("broadcast_id", "")
    
    system_msg = SystemMessage(
        content=f"You are Receiver 4. You received this broadcast message:\n"
                f"Broadcast ID: {broadcast_id}\n"
                f"Message: {broadcast_msg}\n"
                "Process and respond to the broadcast."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    responses = state.get("receiver_responses", {})
    responses["receiver_4"] = response.content
    
    return {
        "messages": [AIMessage(content=f"Receiver 4: Received broadcast - {response.content}")],
        "receiver_responses": responses
    }


# Broadcast Monitor
def broadcast_monitor(state: BroadcastState):
    """Monitor broadcast delivery and responses."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    receivers = state.get("receivers", [])
    responses = state.get("receiver_responses", {})
    broadcast_id = state.get("broadcast_id", "")
    
    system_msg = SystemMessage(
        content=f"You are a Broadcast Monitor. Report broadcast statistics:\n"
                f"Broadcast ID: {broadcast_id}\n"
                f"Total Receivers: {len(receivers)}\n"
                f"Responses Received: {len(responses)}\n"
                f"Response Rate: {len(responses)/len(receivers)*100:.1f}%\n"
                "Summarize broadcast delivery status."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Broadcast Monitor: {response.content}")]
    }


# Build the broadcast graph
def create_broadcast_graph():
    """Create a broadcast workflow graph."""
    workflow = StateGraph(BroadcastState)
    
    # Add nodes
    workflow.add_node("broadcaster", broadcaster_agent)
    workflow.add_node("receiver_1", receiver_1_agent)
    workflow.add_node("receiver_2", receiver_2_agent)
    workflow.add_node("receiver_3", receiver_3_agent)
    workflow.add_node("receiver_4", receiver_4_agent)
    workflow.add_node("monitor", broadcast_monitor)
    
    # Broadcast flow: Broadcaster -> All Receivers (parallel) -> Monitor
    workflow.add_edge(START, "broadcaster")
    workflow.add_edge("broadcaster", "receiver_1")
    workflow.add_edge("broadcaster", "receiver_2")
    workflow.add_edge("broadcaster", "receiver_3")
    workflow.add_edge("broadcaster", "receiver_4")
    workflow.add_edge("receiver_1", "monitor")
    workflow.add_edge("receiver_2", "monitor")
    workflow.add_edge("receiver_3", "monitor")
    workflow.add_edge("receiver_4", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_broadcast_graph()
    
    print("=" * 60)
    print("BROADCAST MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: System-wide announcement
    print("\n[Scenario: Emergency Alert Broadcast]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Broadcast system maintenance alert to all agents")],
        "broadcast_message": "",
        "receivers": [],
        "receiver_responses": {},
        "broadcast_id": ""
    })
    
    print("\n--- Broadcast Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Broadcast Message ---")
    print(f"{result.get('broadcast_message', 'N/A')[:200]}...")
    
    print(f"\n--- Receivers ({len(result.get('receivers', []))}) ---")
    for receiver in result.get("receivers", []):
        print(f"  - {receiver}")
    
    print(f"\n--- Receiver Responses ---")
    for receiver, response in result.get("receiver_responses", {}).items():
        print(f"\n{receiver}: {response[:100]}...")
    
    print(f"\n--- Broadcast ID: {result.get('broadcast_id')} ---")
    
    print("\n" + "=" * 60)

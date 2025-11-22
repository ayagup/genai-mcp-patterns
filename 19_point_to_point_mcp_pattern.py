"""
Point-to-Point MCP Pattern
===========================
Direct one-to-one communication where a sender sends a message
directly to a specific receiver without intermediaries.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict
import operator


# Define the state
class PointToPointState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    direct_messages: Dict[str, str]  # sender_receiver -> message
    acknowledgments: Dict[str, bool]  # message_id -> acknowledged
    conversation_log: list


# Agent A (Sender/Receiver)
def agent_a(state: PointToPointState):
    """Agent A sends direct message to Agent B."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are Agent A. Send a direct point-to-point message to Agent B. "
                "Create a specific request or information for Agent B only."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    direct_messages = state.get("direct_messages", {})
    direct_messages["A_to_B"] = response.content
    
    conversation_log = state.get("conversation_log", [])
    conversation_log.append({"from": "A", "to": "B", "message": response.content[:50]})
    
    return {
        "messages": [AIMessage(content=f"Agent A -> Agent B: {response.content}")],
        "direct_messages": direct_messages,
        "conversation_log": conversation_log
    }


# Agent B (Receiver/Sender)
def agent_b(state: PointToPointState):
    """Agent B receives from A and sends to C."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    msg_from_a = state.get("direct_messages", {}).get("A_to_B", "No message")
    
    system_msg = SystemMessage(
        content=f"You are Agent B. You received a direct message from Agent A:\n"
                f"Message: {msg_from_a}\n"
                "1. Acknowledge receipt to Agent A\n"
                "2. Send a new direct message to Agent C based on this information."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    direct_messages = state.get("direct_messages", {})
    direct_messages["B_to_C"] = response.content
    
    acknowledgments = state.get("acknowledgments", {})
    acknowledgments["A_to_B"] = True
    
    conversation_log = state.get("conversation_log", [])
    conversation_log.append({"from": "B", "to": "C", "message": response.content[:50]})
    
    return {
        "messages": [AIMessage(content=f"Agent B: Acknowledged A's message. B -> Agent C: {response.content}")],
        "direct_messages": direct_messages,
        "acknowledgments": acknowledgments,
        "conversation_log": conversation_log
    }


# Agent C (Receiver/Sender)
def agent_c(state: PointToPointState):
    """Agent C receives from B and sends to D."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    msg_from_b = state.get("direct_messages", {}).get("B_to_C", "No message")
    
    system_msg = SystemMessage(
        content=f"You are Agent C. You received a direct message from Agent B:\n"
                f"Message: {msg_from_b}\n"
                "1. Acknowledge receipt to Agent B\n"
                "2. Send a new direct message to Agent D."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    direct_messages = state.get("direct_messages", {})
    direct_messages["C_to_D"] = response.content
    
    acknowledgments = state.get("acknowledgments", {})
    acknowledgments["B_to_C"] = True
    
    conversation_log = state.get("conversation_log", [])
    conversation_log.append({"from": "C", "to": "D", "message": response.content[:50]})
    
    return {
        "messages": [AIMessage(content=f"Agent C: Acknowledged B's message. C -> Agent D: {response.content}")],
        "direct_messages": direct_messages,
        "acknowledgments": acknowledgments,
        "conversation_log": conversation_log
    }


# Agent D (Receiver)
def agent_d(state: PointToPointState):
    """Agent D receives from C and responds back to A."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    msg_from_c = state.get("direct_messages", {}).get("C_to_D", "No message")
    
    system_msg = SystemMessage(
        content=f"You are Agent D. You received a direct message from Agent C:\n"
                f"Message: {msg_from_c}\n"
                "1. Acknowledge receipt to Agent C\n"
                "2. Send a final response back to Agent A."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    direct_messages = state.get("direct_messages", {})
    direct_messages["D_to_A"] = response.content
    
    acknowledgments = state.get("acknowledgments", {})
    acknowledgments["C_to_D"] = True
    
    conversation_log = state.get("conversation_log", [])
    conversation_log.append({"from": "D", "to": "A", "message": response.content[:50]})
    
    return {
        "messages": [AIMessage(content=f"Agent D: Acknowledged C's message. D -> Agent A: {response.content}")],
        "direct_messages": direct_messages,
        "acknowledgments": acknowledgments,
        "conversation_log": conversation_log
    }


# Message Tracker
def message_tracker(state: PointToPointState):
    """Track all point-to-point communications."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    conversation_log = state.get("conversation_log", [])
    acknowledgments = state.get("acknowledgments", {})
    
    chain = " -> ".join([log['from'] for log in conversation_log])
    
    system_msg = SystemMessage(
        content=f"You are a Message Tracker. Summarize point-to-point communications:\n"
                f"Total Messages: {len(conversation_log)}\n"
                f"Acknowledgments: {acknowledgments}\n"
                f"Conversation Chain: {chain}\n"
                "Provide communication summary."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Message Tracker: {response.content}")]
    }


# Build the point-to-point graph
def create_point_to_point_graph():
    """Create a point-to-point workflow graph."""
    workflow = StateGraph(PointToPointState)
    
    # Add nodes
    workflow.add_node("agent_a", agent_a)
    workflow.add_node("agent_b", agent_b)
    workflow.add_node("agent_c", agent_c)
    workflow.add_node("agent_d", agent_d)
    workflow.add_node("tracker", message_tracker)
    
    # Point-to-point chain: A -> B -> C -> D -> Tracker
    workflow.add_edge(START, "agent_a")
    workflow.add_edge("agent_a", "agent_b")
    workflow.add_edge("agent_b", "agent_c")
    workflow.add_edge("agent_c", "agent_d")
    workflow.add_edge("agent_d", "tracker")
    workflow.add_edge("tracker", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_point_to_point_graph()
    
    print("=" * 60)
    print("POINT-TO-POINT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Direct agent-to-agent communication
    print("\n[Scenario: Secure Information Chain]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Initiate secure point-to-point communication chain for confidential data transfer")],
        "direct_messages": {},
        "acknowledgments": {},
        "conversation_log": []
    })
    
    print("\n--- Point-to-Point Communication Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Direct Messages ---")
    for route, message in result.get("direct_messages", {}).items():
        print(f"\n{route}:")
        print(f"  {message[:150]}...")
    
    print(f"\n--- Acknowledgments ---")
    for msg_id, ack in result.get("acknowledgments", {}).items():
        print(f"  {msg_id}: {'✓ Acknowledged' if ack else '✗ Not Acknowledged'}")
    
    print(f"\n--- Conversation Log ---")
    for log in result.get("conversation_log", []):
        print(f"  {log['from']} -> {log['to']}: {log['message']}...")
    
    print("\n" + "=" * 60)

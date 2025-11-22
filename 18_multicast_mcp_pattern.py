"""
Multicast MCP Pattern
======================
Selective group communication where a sender multicasts messages
to specific groups of receivers based on group membership.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List, Dict
import operator


# Define the state
class MulticastState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    groups: Dict[str, List[str]]  # Group name -> member IDs
    multicast_messages: Dict[str, str]  # Group -> message
    group_responses: Dict[str, List[str]]  # Group -> responses


# Multicast Sender Agent
def multicast_sender_agent(state: MulticastState):
    """Agent that sends messages to specific groups."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Multicast Sender. Create targeted messages for different groups:\n"
                "Groups: 'engineering', 'sales', 'management'\n"
                "Generate appropriate messages for each group."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Define group memberships
    groups = {
        "engineering": ["eng_1", "eng_2", "eng_3"],
        "sales": ["sales_1", "sales_2"],
        "management": ["mgr_1", "mgr_2"]
    }
    
    # Create multicast messages for each group
    content = response.content
    multicast_messages = {
        "engineering": f"Engineering Team: {content[:150]}",
        "sales": f"Sales Team: {content[150:300]}",
        "management": f"Management Team: {content[300:450]}"
    }
    
    return {
        "messages": [AIMessage(content=f"Multicast Sender: Sent messages to {len(groups)} groups - {response.content}")],
        "groups": groups,
        "multicast_messages": multicast_messages
    }


# Engineering Group Receivers
def engineering_group_agent(state: MulticastState):
    """Engineering group receives and processes multicast."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    group_msg = state.get("multicast_messages", {}).get("engineering", "No message")
    members = state.get("groups", {}).get("engineering", [])
    
    system_msg = SystemMessage(
        content=f"You are the Engineering Group ({len(members)} members).\n"
                f"Multicast message: {group_msg}\n"
                "Process this message as an engineering team and provide collective response."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    group_responses = state.get("group_responses", {})
    group_responses["engineering"] = [f"Member response: {response.content}"]
    
    return {
        "messages": [AIMessage(content=f"Engineering Group: Received multicast - {response.content}")],
        "group_responses": group_responses
    }


# Sales Group Receivers
def sales_group_agent(state: MulticastState):
    """Sales group receives and processes multicast."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    group_msg = state.get("multicast_messages", {}).get("sales", "No message")
    members = state.get("groups", {}).get("sales", [])
    
    system_msg = SystemMessage(
        content=f"You are the Sales Group ({len(members)} members).\n"
                f"Multicast message: {group_msg}\n"
                "Process this message as a sales team and provide collective response."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    group_responses = state.get("group_responses", {})
    group_responses["sales"] = [f"Member response: {response.content}"]
    
    return {
        "messages": [AIMessage(content=f"Sales Group: Received multicast - {response.content}")],
        "group_responses": group_responses
    }


# Management Group Receivers
def management_group_agent(state: MulticastState):
    """Management group receives and processes multicast."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    group_msg = state.get("multicast_messages", {}).get("management", "No message")
    members = state.get("groups", {}).get("management", [])
    
    system_msg = SystemMessage(
        content=f"You are the Management Group ({len(members)} members).\n"
                f"Multicast message: {group_msg}\n"
                "Process this message as a management team and provide collective response."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    group_responses = state.get("group_responses", {})
    group_responses["management"] = [f"Member response: {response.content}"]
    
    return {
        "messages": [AIMessage(content=f"Management Group: Received multicast - {response.content}")],
        "group_responses": group_responses
    }


# Multicast Aggregator
def multicast_aggregator(state: MulticastState):
    """Aggregate responses from all multicast groups."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    groups = state.get("groups", {})
    group_responses = state.get("group_responses", {})
    
    system_msg = SystemMessage(
        content=f"You are a Multicast Aggregator. Summarize group responses:\n"
                f"Total Groups: {len(groups)}\n"
                f"Engineering Response: {group_responses.get('engineering', ['N/A'])[0]}\n"
                f"Sales Response: {group_responses.get('sales', ['N/A'])[0]}\n"
                f"Management Response: {group_responses.get('management', ['N/A'])[0]}\n"
                "Provide an aggregated summary of all group feedback."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Multicast Aggregator: {response.content}")]
    }


# Build the multicast graph
def create_multicast_graph():
    """Create a multicast workflow graph."""
    workflow = StateGraph(MulticastState)
    
    # Add nodes
    workflow.add_node("sender", multicast_sender_agent)
    workflow.add_node("engineering", engineering_group_agent)
    workflow.add_node("sales", sales_group_agent)
    workflow.add_node("management", management_group_agent)
    workflow.add_node("aggregator", multicast_aggregator)
    
    # Multicast flow: Sender -> Groups (parallel) -> Aggregator
    workflow.add_edge(START, "sender")
    workflow.add_edge("sender", "engineering")
    workflow.add_edge("sender", "sales")
    workflow.add_edge("sender", "management")
    workflow.add_edge("engineering", "aggregator")
    workflow.add_edge("sales", "aggregator")
    workflow.add_edge("management", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_multicast_graph()
    
    print("=" * 60)
    print("MULTICAST MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Group-targeted communication
    print("\n[Scenario: Department-Specific Announcements]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Send department-specific updates about Q4 objectives and initiatives")],
        "groups": {},
        "multicast_messages": {},
        "group_responses": {}
    })
    
    print("\n--- Multicast Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Group Memberships ---")
    for group, members in result.get("groups", {}).items():
        print(f"{group}: {members}")
    
    print(f"\n--- Multicast Messages ---")
    for group, msg in result.get("multicast_messages", {}).items():
        print(f"\n{group}:")
        print(f"  {msg[:150]}...")
    
    print(f"\n--- Group Responses ---")
    for group, responses in result.get("group_responses", {}).items():
        print(f"\n{group}:")
        for response in responses:
            print(f"  {response[:100]}...")
    
    print("\n" + "=" * 60)

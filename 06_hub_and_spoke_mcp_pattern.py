"""
Hub-and-Spoke MCP Pattern
==========================
A central hub agent coordinates with multiple spoke agents radiating from the center.
All communication flows through the hub, which manages distribution and aggregation.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator


# Define the state
class HubSpokeState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    hub_instructions: Dict[str, str]  # Instructions from hub to each spoke
    spoke_results: Dict[str, str]  # Results from each spoke
    aggregated_result: str
    active_spokes: List[str]


# Hub Agent - Central Coordinator
def hub_agent(state: HubSpokeState):
    """Central hub that coordinates all spoke agents."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are the central Hub agent. Analyze the task and create specific "
                "instructions for each spoke agent:\n"
                "- Spoke 1: Data gathering specialist\n"
                "- Spoke 2: Analysis specialist\n"
                "- Spoke 3: Recommendation specialist\n"
                "Provide clear, specific instructions for each."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Create instructions for each spoke
    instructions = {
        "spoke_1": f"Gather data: {response.content[:100]}",
        "spoke_2": f"Analyze data: {response.content[:100]}",
        "spoke_3": f"Generate recommendations: {response.content[:100]}"
    }
    
    return {
        "messages": [AIMessage(content=f"Hub: Distributing tasks to spokes - {response.content}")],
        "hub_instructions": instructions,
        "active_spokes": ["spoke_1", "spoke_2", "spoke_3"]
    }


# Spoke 1 - Data Gatherer
def spoke_1_agent(state: HubSpokeState):
    """Spoke agent specialized in data gathering."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    instruction = state.get("hub_instructions", {}).get("spoke_1", "Gather data")
    
    system_msg = SystemMessage(
        content=f"You are Spoke 1 - Data Gatherer. Hub instruction: {instruction}\n"
                "Focus on collecting relevant data and report back to hub."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Spoke 1 (Data Gatherer): {response.content}")],
        "spoke_results": {"spoke_1": response.content}
    }


# Spoke 2 - Analyzer
def spoke_2_agent(state: HubSpokeState):
    """Spoke agent specialized in analysis."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    instruction = state.get("hub_instructions", {}).get("spoke_2", "Analyze data")
    spoke_1_data = state.get("spoke_results", {}).get("spoke_1", "No data available")
    
    system_msg = SystemMessage(
        content=f"You are Spoke 2 - Analyzer. Hub instruction: {instruction}\n"
                f"Data from Spoke 1: {spoke_1_data}\n"
                "Analyze the data and report back to hub."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Spoke 2 (Analyzer): {response.content}")],
        "spoke_results": {"spoke_2": response.content}
    }


# Spoke 3 - Recommender
def spoke_3_agent(state: HubSpokeState):
    """Spoke agent specialized in recommendations."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    instruction = state.get("hub_instructions", {}).get("spoke_3", "Generate recommendations")
    analysis = state.get("spoke_results", {}).get("spoke_2", "No analysis available")
    
    system_msg = SystemMessage(
        content=f"You are Spoke 3 - Recommender. Hub instruction: {instruction}\n"
                f"Analysis from Spoke 2: {analysis}\n"
                "Generate recommendations and report back to hub."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Spoke 3 (Recommender): {response.content}")],
        "spoke_results": {"spoke_3": response.content}
    }


# Hub Aggregator
def hub_aggregator(state: HubSpokeState):
    """Hub aggregates results from all spokes."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    spoke_results = state.get("spoke_results", {})
    
    system_msg = SystemMessage(
        content=f"You are the Hub aggregator. Synthesize results from all spokes:\n"
                f"Spoke 1 (Data): {spoke_results.get('spoke_1', 'N/A')}\n"
                f"Spoke 2 (Analysis): {spoke_results.get('spoke_2', 'N/A')}\n"
                f"Spoke 3 (Recommendations): {spoke_results.get('spoke_3', 'N/A')}\n"
                "Provide a comprehensive final result."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Hub Aggregator: {response.content}")],
        "aggregated_result": response.content
    }


# Build the hub-and-spoke graph
def create_hub_spoke_graph():
    """Create a hub-and-spoke workflow graph."""
    workflow = StateGraph(HubSpokeState)
    
    # Add hub and spoke nodes
    workflow.add_node("hub", hub_agent)
    workflow.add_node("spoke_1", spoke_1_agent)
    workflow.add_node("spoke_2", spoke_2_agent)
    workflow.add_node("spoke_3", spoke_3_agent)
    workflow.add_node("hub_aggregator", hub_aggregator)
    
    # Hub-and-spoke topology: START -> HUB -> SPOKES -> HUB_AGGREGATOR -> END
    workflow.add_edge(START, "hub")
    
    # Hub distributes to spokes
    workflow.add_edge("hub", "spoke_1")
    workflow.add_edge("spoke_1", "spoke_2")
    workflow.add_edge("spoke_2", "spoke_3")
    
    # Spokes report back to hub aggregator
    workflow.add_edge("spoke_3", "hub_aggregator")
    workflow.add_edge("hub_aggregator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the hub-and-spoke agent system
    graph = create_hub_spoke_graph()
    
    print("=" * 60)
    print("HUB-AND-SPOKE MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Centralized coordination with specialized spokes
    print("\n[Task: Product launch strategy]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Develop a comprehensive product launch strategy for our new AI assistant")],
        "hub_instructions": {},
        "spoke_results": {},
        "aggregated_result": "",
        "active_spokes": []
    })
    
    print("\n--- Hub-and-Spoke Workflow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Final Aggregated Result ---")
    print(f"{result.get('aggregated_result', 'N/A')[:300]}...")
    
    print(f"\n--- Active Spokes ---")
    print(result.get("active_spokes", []))
    
    print("\n" + "=" * 60)

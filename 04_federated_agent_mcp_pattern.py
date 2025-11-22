"""
Federated Agent MCP Pattern
============================
Multiple autonomous agents with local decision-making capabilities that coordinate
through a federation. Each agent maintains local autonomy while participating in
a federated system with shared protocols and standards.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, List
import operator


# Define the state
class FederatedState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    federation_registry: dict  # Shared registry of agent capabilities
    local_results: dict  # Results from each federated agent
    consensus_result: str
    active_agents: List[str]


# Region Agent 1 (North America)
def north_america_agent(state: FederatedState):
    """Federated agent for North America region with local autonomy."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a North America regional agent with autonomy to make local decisions. "
                "Analyze the request from a North American perspective."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update local registry
    registry = state.get("federation_registry", {})
    registry["north_america"] = {
        "capabilities": ["market_analysis", "regulations", "consumer_trends"],
        "status": "active"
    }
    
    return {
        "messages": [AIMessage(content=f"North America Agent: {response.content}")],
        "federation_registry": registry,
        "local_results": {"north_america": response.content}
    }


# Region Agent 2 (Europe)
def europe_agent(state: FederatedState):
    """Federated agent for Europe region with local autonomy."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Europe regional agent with autonomy to make local decisions. "
                "Analyze the request from a European perspective."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update local registry
    registry = state.get("federation_registry", {})
    registry["europe"] = {
        "capabilities": ["gdpr_compliance", "market_analysis", "regulations"],
        "status": "active"
    }
    
    return {
        "messages": [AIMessage(content=f"Europe Agent: {response.content}")],
        "federation_registry": registry,
        "local_results": {"europe": response.content}
    }


# Region Agent 3 (Asia Pacific)
def asia_pacific_agent(state: FederatedState):
    """Federated agent for Asia Pacific region with local autonomy."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are an Asia Pacific regional agent with autonomy to make local decisions. "
                "Analyze the request from an Asia Pacific perspective."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update local registry
    registry = state.get("federation_registry", {})
    registry["asia_pacific"] = {
        "capabilities": ["market_growth", "manufacturing", "innovation"],
        "status": "active"
    }
    
    return {
        "messages": [AIMessage(content=f"Asia Pacific Agent: {response.content}")],
        "federation_registry": registry,
        "local_results": {"asia_pacific": response.content}
    }


# Federation Coordinator
def federation_coordinator(state: FederatedState):
    """Coordinates federated agents and builds consensus."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    local_results = state.get("local_results", {})
    registry = state.get("federation_registry", {})
    
    # Build consensus from federated results
    system_msg = SystemMessage(
        content=f"You are a federation coordinator. Synthesize insights from autonomous regional agents:\n"
                f"Registry: {registry}\n"
                f"North America: {local_results.get('north_america', 'N/A')}\n"
                f"Europe: {local_results.get('europe', 'N/A')}\n"
                f"Asia Pacific: {local_results.get('asia_pacific', 'N/A')}\n"
                "Provide a federated consensus while respecting local autonomy."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Federation Coordinator: {response.content}")],
        "consensus_result": response.content,
        "active_agents": list(registry.keys())
    }


# Build the federated graph
def create_federated_agent_graph():
    """Create a federated agent workflow graph."""
    workflow = StateGraph(FederatedState)
    
    # Add federated agent nodes
    workflow.add_node("north_america", north_america_agent)
    workflow.add_node("europe", europe_agent)
    workflow.add_node("asia_pacific", asia_pacific_agent)
    workflow.add_node("coordinator", federation_coordinator)
    
    # Add edges - agents work in parallel, then coordinate
    workflow.add_edge(START, "north_america")
    workflow.add_edge(START, "europe")
    workflow.add_edge(START, "asia_pacific")
    
    workflow.add_edge("north_america", "coordinator")
    workflow.add_edge("europe", "coordinator")
    workflow.add_edge("asia_pacific", "coordinator")
    
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the federated agent system
    graph = create_federated_agent_graph()
    
    print("=" * 60)
    print("FEDERATED AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Global market analysis with federated agents
    print("\n[Task: Global market expansion analysis]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Analyze opportunities for expanding our SaaS product globally")],
        "federation_registry": {},
        "local_results": {},
        "consensus_result": "",
        "active_agents": []
    })
    
    print("\n--- Federated Agent Results ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Federation Registry ---")
    for agent, info in result.get("federation_registry", {}).items():
        print(f"{agent}: {info}")
    
    print(f"\n--- Active Agents ---")
    print(result.get("active_agents", []))
    
    print("\n" + "=" * 60)

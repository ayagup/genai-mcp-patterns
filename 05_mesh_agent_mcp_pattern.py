"""
Mesh Agent MCP Pattern
======================
A fully connected mesh network where every agent can communicate with every other agent.
This pattern enables decentralized collaboration with direct peer-to-peer communication.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator


# Define the state
class MeshState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    mesh_network: Dict[str, List[str]]  # Adjacency list of agent connections
    agent_messages: Dict[str, List[str]]  # Messages between agents
    agent_results: Dict[str, str]
    processed_agents: List[str]


# Agent A - Data Collector
def agent_a(state: MeshState):
    """Agent A in mesh network - collects data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Check messages from other agents
    incoming = state.get("agent_messages", {}).get("agent_a", [])
    
    system_msg = SystemMessage(
        content=f"You are Agent A in a mesh network. Your role: data collection.\n"
                f"Messages from peers: {incoming}\n"
                "Collect relevant data and share with peers."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Send messages to all connected agents
    agent_messages = state.get("agent_messages", {})
    for peer in ["agent_b", "agent_c", "agent_d"]:
        if peer not in agent_messages:
            agent_messages[peer] = []
        agent_messages[peer].append(f"Agent A: {response.content[:100]}")
    
    return {
        "messages": [AIMessage(content=f"Agent A (Data Collector): {response.content}")],
        "agent_messages": agent_messages,
        "agent_results": {"agent_a": response.content}
    }


# Agent B - Analyzer
def agent_b(state: MeshState):
    """Agent B in mesh network - analyzes data."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    incoming = state.get("agent_messages", {}).get("agent_b", [])
    
    system_msg = SystemMessage(
        content=f"You are Agent B in a mesh network. Your role: data analysis.\n"
                f"Messages from peers: {incoming}\n"
                "Analyze data and share insights with peers."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Send messages to all connected agents
    agent_messages = state.get("agent_messages", {})
    for peer in ["agent_a", "agent_c", "agent_d"]:
        if peer not in agent_messages:
            agent_messages[peer] = []
        agent_messages[peer].append(f"Agent B: {response.content[:100]}")
    
    return {
        "messages": [AIMessage(content=f"Agent B (Analyzer): {response.content}")],
        "agent_messages": agent_messages,
        "agent_results": {"agent_b": response.content}
    }


# Agent C - Validator
def agent_c(state: MeshState):
    """Agent C in mesh network - validates results."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    incoming = state.get("agent_messages", {}).get("agent_c", [])
    
    system_msg = SystemMessage(
        content=f"You are Agent C in a mesh network. Your role: validation.\n"
                f"Messages from peers: {incoming}\n"
                "Validate peer results and provide feedback."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Send messages to all connected agents
    agent_messages = state.get("agent_messages", {})
    for peer in ["agent_a", "agent_b", "agent_d"]:
        if peer not in agent_messages:
            agent_messages[peer] = []
        agent_messages[peer].append(f"Agent C: {response.content[:100]}")
    
    return {
        "messages": [AIMessage(content=f"Agent C (Validator): {response.content}")],
        "agent_messages": agent_messages,
        "agent_results": {"agent_c": response.content}
    }


# Agent D - Synthesizer
def agent_d(state: MeshState):
    """Agent D in mesh network - synthesizes all inputs."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    incoming = state.get("agent_messages", {}).get("agent_d", [])
    results = state.get("agent_results", {})
    
    system_msg = SystemMessage(
        content=f"You are Agent D in a mesh network. Your role: synthesis.\n"
                f"Messages from peers: {incoming}\n"
                f"Agent Results: {results}\n"
                "Synthesize all peer inputs into final output."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Agent D (Synthesizer): {response.content}")],
        "agent_results": {"agent_d": response.content}
    }


# Build the mesh network graph
def create_mesh_agent_graph():
    """Create a mesh agent workflow graph with full connectivity."""
    workflow = StateGraph(MeshState)
    
    # Add all agents as nodes
    workflow.add_node("agent_a", agent_a)
    workflow.add_node("agent_b", agent_b)
    workflow.add_node("agent_c", agent_c)
    workflow.add_node("agent_d", agent_d)
    
    # Create mesh topology - all agents can reach all others
    # Layer 1: Initial data collection and processing
    workflow.add_edge(START, "agent_a")
    workflow.add_edge(START, "agent_b")
    workflow.add_edge(START, "agent_c")
    
    # Layer 2: Synthesis after initial processing
    workflow.add_edge("agent_a", "agent_d")
    workflow.add_edge("agent_b", "agent_d")
    workflow.add_edge("agent_c", "agent_d")
    
    workflow.add_edge("agent_d", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the mesh agent system
    graph = create_mesh_agent_graph()
    
    print("=" * 60)
    print("MESH AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Initialize mesh network topology
    mesh_network = {
        "agent_a": ["agent_b", "agent_c", "agent_d"],
        "agent_b": ["agent_a", "agent_c", "agent_d"],
        "agent_c": ["agent_a", "agent_b", "agent_d"],
        "agent_d": ["agent_a", "agent_b", "agent_c"]
    }
    
    # Example: Collaborative analysis in mesh network
    print("\n[Task: Collaborative market research]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Conduct comprehensive market research on renewable energy")],
        "mesh_network": mesh_network,
        "agent_messages": {},
        "agent_results": {},
        "processed_agents": []
    })
    
    print("\n--- Mesh Network Agent Outputs ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Mesh Network Topology ---")
    for agent, connections in mesh_network.items():
        print(f"{agent} -> {connections}")
    
    print("\n" + "=" * 60)

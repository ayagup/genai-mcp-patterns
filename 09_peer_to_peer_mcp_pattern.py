"""
Peer-to-Peer Agent MCP Pattern
===============================
Decentralized agents that communicate directly without a central coordinator.
Each peer has equal status and can initiate communication with any other peer.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator


# Define the state
class P2PState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    peer_network: Dict[str, List[str]]  # Peer connections
    peer_messages: Dict[str, List[str]]  # Direct peer-to-peer messages
    peer_contributions: Dict[str, str]  # Each peer's contribution
    consensus: str
    active_peers: List[str]


# Peer A
def peer_a_agent(state: P2PState):
    """Peer A - Equal participant in P2P network."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    # Check direct messages from other peers
    incoming = state.get("peer_messages", {}).get("peer_a", [])
    
    system_msg = SystemMessage(
        content=f"You are Peer A in a decentralized P2P network. All peers have equal status.\n"
                f"Direct messages from peers: {incoming}\n"
                "Contribute your perspective and communicate with other peers directly."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Send direct messages to connected peers
    peer_messages = state.get("peer_messages", {})
    for peer in ["peer_b", "peer_c", "peer_d"]:
        if peer not in peer_messages:
            peer_messages[peer] = []
        peer_messages[peer].append(f"Peer A: {response.content[:80]}")
    
    return {
        "messages": [AIMessage(content=f"Peer A: {response.content}")],
        "peer_messages": peer_messages,
        "peer_contributions": {"peer_a": response.content}
    }


# Peer B
def peer_b_agent(state: P2PState):
    """Peer B - Equal participant in P2P network."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    incoming = state.get("peer_messages", {}).get("peer_b", [])
    
    system_msg = SystemMessage(
        content=f"You are Peer B in a decentralized P2P network. All peers have equal status.\n"
                f"Direct messages from peers: {incoming}\n"
                "Contribute your perspective and communicate with other peers directly."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Send direct messages to connected peers
    peer_messages = state.get("peer_messages", {})
    for peer in ["peer_a", "peer_c", "peer_d"]:
        if peer not in peer_messages:
            peer_messages[peer] = []
        peer_messages[peer].append(f"Peer B: {response.content[:80]}")
    
    return {
        "messages": [AIMessage(content=f"Peer B: {response.content}")],
        "peer_messages": peer_messages,
        "peer_contributions": {"peer_b": response.content}
    }


# Peer C
def peer_c_agent(state: P2PState):
    """Peer C - Equal participant in P2P network."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    incoming = state.get("peer_messages", {}).get("peer_c", [])
    
    system_msg = SystemMessage(
        content=f"You are Peer C in a decentralized P2P network. All peers have equal status.\n"
                f"Direct messages from peers: {incoming}\n"
                "Contribute your perspective and communicate with other peers directly."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Send direct messages to connected peers
    peer_messages = state.get("peer_messages", {})
    for peer in ["peer_a", "peer_b", "peer_d"]:
        if peer not in peer_messages:
            peer_messages[peer] = []
        peer_messages[peer].append(f"Peer C: {response.content[:80]}")
    
    return {
        "messages": [AIMessage(content=f"Peer C: {response.content}")],
        "peer_messages": peer_messages,
        "peer_contributions": {"peer_c": response.content}
    }


# Peer D
def peer_d_agent(state: P2PState):
    """Peer D - Equal participant in P2P network."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    incoming = state.get("peer_messages", {}).get("peer_d", [])
    
    system_msg = SystemMessage(
        content=f"You are Peer D in a decentralized P2P network. All peers have equal status.\n"
                f"Direct messages from peers: {incoming}\n"
                "Contribute your perspective and communicate with other peers directly."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Peer D: {response.content}")],
        "peer_contributions": {"peer_d": response.content}
    }


# Consensus Builder
def consensus_builder(state: P2PState):
    """Build consensus from all peer contributions (can be done by any peer)."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    contributions = state.get("peer_contributions", {})
    
    system_msg = SystemMessage(
        content=f"Build consensus from peer contributions (all peers have equal weight):\n"
                f"Peer A: {contributions.get('peer_a', 'N/A')}\n"
                f"Peer B: {contributions.get('peer_b', 'N/A')}\n"
                f"Peer C: {contributions.get('peer_c', 'N/A')}\n"
                f"Peer D: {contributions.get('peer_d', 'N/A')}\n"
                "Create a decentralized consensus that represents all peer views."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"P2P Consensus: {response.content}")],
        "consensus": response.content,
        "active_peers": list(contributions.keys())
    }


# Build the peer-to-peer graph
def create_p2p_agent_graph():
    """Create a peer-to-peer workflow graph."""
    workflow = StateGraph(P2PState)
    
    # Add peer nodes (all equal status)
    workflow.add_node("peer_a", peer_a_agent)
    workflow.add_node("peer_b", peer_b_agent)
    workflow.add_node("peer_c", peer_c_agent)
    workflow.add_node("peer_d", peer_d_agent)
    workflow.add_node("consensus", consensus_builder)
    
    # P2P topology: all peers process in parallel, then build consensus
    workflow.add_edge(START, "peer_a")
    workflow.add_edge(START, "peer_b")
    workflow.add_edge(START, "peer_c")
    workflow.add_edge(START, "peer_d")
    
    # All peers contribute to consensus
    workflow.add_edge("peer_a", "consensus")
    workflow.add_edge("peer_b", "consensus")
    workflow.add_edge("peer_c", "consensus")
    workflow.add_edge("peer_d", "consensus")
    
    workflow.add_edge("consensus", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the P2P agent system
    graph = create_p2p_agent_graph()
    
    print("=" * 60)
    print("PEER-TO-PEER AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Initialize P2P network
    peer_network = {
        "peer_a": ["peer_b", "peer_c", "peer_d"],
        "peer_b": ["peer_a", "peer_c", "peer_d"],
        "peer_c": ["peer_a", "peer_b", "peer_d"],
        "peer_d": ["peer_a", "peer_b", "peer_c"]
    }
    
    # Example: Decentralized decision making
    print("\n[Task: Decentralized project planning]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Plan our next quarter's product roadmap collaboratively")],
        "peer_network": peer_network,
        "peer_messages": {},
        "peer_contributions": {},
        "consensus": "",
        "active_peers": []
    })
    
    print("\n--- Peer Contributions (Decentralized) ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- P2P Network Topology ---")
    for peer, connections in peer_network.items():
        print(f"{peer} <-> {connections}")
    
    print(f"\n--- Decentralized Consensus ---")
    print(f"{result.get('consensus', 'N/A')[:300]}...")
    
    print(f"\n--- Active Peers ---")
    print(result.get("active_peers", []))
    
    print("\n" + "=" * 60)

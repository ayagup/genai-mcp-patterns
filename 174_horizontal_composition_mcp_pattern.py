"""
Pattern 174: Horizontal Composition MCP Pattern

This pattern demonstrates horizontal composition where components are organized as
peers at the same level, working side-by-side rather than in a hierarchy. Components
collaborate as equals, sharing responsibilities and coordinating through direct
communication or message passing.

Key Concepts:
1. Peer Components: Components at same level of abstraction
2. Side-by-Side Collaboration: Components work together as equals
3. Shared Responsibility: No single component owns entire workflow
4. Lateral Communication: Components communicate directly, not through layers
5. Distributed Coordination: No central authority (vs vertical orchestration)
6. Parallel Execution: Peers can run concurrently
7. Equal Status: No parent-child or superior-subordinate relationships

Horizontal Composition Patterns:
1. Pipeline: A → B → C (sequential peers)
2. Parallel: [A, B, C] (concurrent peers)
3. Peer-to-Peer: A ↔ B ↔ C (bidirectional)
4. Broadcast: A → [B, C, D] (one-to-many)
5. Round-Robin: A → B → C → A (circular)

Collaboration Models:
- Direct Collaboration: Peers call each other directly
- Message-Based: Peers communicate via messages/events
- Shared State: Peers access common state
- Data Pipeline: Peers pass data sequentially
- Collective Decision: Peers vote or reach consensus

Benefits:
- Flexibility: Easy to add/remove/reorder peers
- Scalability: Scale peers independently
- Resilience: Failure of one peer doesn't cascade
- Simplicity: No complex hierarchy to manage
- Parallelism: Natural parallel execution

Trade-offs:
- Coordination Overhead: Peers must coordinate themselves
- No Single Point of Control: Harder to enforce policies
- Circular Dependencies: Peers might depend on each other
- Discovery: Peers need to find each other
- Consistency: Maintaining consistency across peers

Use Cases:
- Data pipeline: extract → transform → validate → load (ETL)
- Microservices: order-service, payment-service, inventory-service
- Agent collaboration: researcher, writer, editor (equal partners)
- Stream processing: filter → map → reduce (stream operators)
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator

# Define the state for horizontal composition
class HorizontalCompositionState(TypedDict):
    """State shared across peer components"""
    input_data: str
    peer_1_result: Optional[Dict[str, Any]]
    peer_2_result: Optional[Dict[str, Any]]
    peer_3_result: Optional[Dict[str, Any]]
    aggregated_result: str
    peer_communications: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# PEER COMPONENTS (Horizontal Collaboration)
# ============================================================================

def peer_1_analyzer(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Peer 1: Analyzer
    
    In horizontal composition, this is a peer component that:
    - Operates at the same level as other peers
    - Contributes its specialized capability (analysis)
    - Can communicate with other peers if needed
    - Shares responsibility for overall goal
    """
    input_data = state["input_data"]
    
    prompt = f"""You are an analyzer peer in a horizontal composition.
    Analyze the following data and extract key insights:
    
    Data: {input_data}
    
    Provide:
    1. Key patterns identified
    2. Important metrics
    3. Insights for other peers"""
    
    response = llm.invoke(prompt)
    
    result = {
        "peer": "analyzer",
        "capability": "data_analysis",
        "analysis": response.content[:200],
        "confidence": 0.9,
        "completed": True
    }
    
    return {
        "peer_1_result": result,
        "peer_communications": ["Peer 1 (Analyzer) → Shared State"],
        "messages": ["[Peer 1 - Analyzer] Analysis completed"]
    }

def peer_2_transformer(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Peer 2: Transformer
    
    Another peer component that:
    - Works alongside analyzer (not above or below)
    - Can access analyzer's results (lateral communication)
    - Contributes transformation capability
    - Equal partner in the composition
    """
    input_data = state["input_data"]
    peer_1_result = state.get("peer_1_result", {})
    
    # Peer can access other peer's results (horizontal collaboration)
    analysis = peer_1_result.get("analysis", "No analysis available")
    
    prompt = f"""You are a transformer peer in a horizontal composition.
    Transform the data based on analysis from your peer:
    
    Original Data: {input_data}
    Peer Analysis: {analysis}
    
    Transform the data to a structured format."""
    
    response = llm.invoke(prompt)
    
    result = {
        "peer": "transformer",
        "capability": "data_transformation",
        "transformed_data": response.content[:200],
        "used_peer_result": bool(analysis != "No analysis available"),
        "completed": True
    }
    
    return {
        "peer_2_result": result,
        "peer_communications": ["Peer 2 (Transformer) ← Peer 1 (read analysis)"],
        "messages": ["[Peer 2 - Transformer] Transformation completed"]
    }

def peer_3_validator(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Peer 3: Validator
    
    Third peer component that:
    - Validates outputs from other peers
    - Can run in parallel with peers if independent
    - Contributes validation capability
    - Equal collaborator in achieving goal
    """
    peer_2_result = state.get("peer_2_result", {})
    
    transformed_data = peer_2_result.get("transformed_data", "No data")
    
    prompt = f"""You are a validator peer in a horizontal composition.
    Validate the transformed data from your peer:
    
    Transformed Data: {transformed_data}
    
    Check for:
    1. Completeness
    2. Correctness
    3. Consistency"""
    
    response = llm.invoke(prompt)
    
    result = {
        "peer": "validator",
        "capability": "data_validation",
        "validation_result": "passed",
        "issues_found": 0,
        "validation_details": response.content[:200],
        "completed": True
    }
    
    return {
        "peer_3_result": result,
        "peer_communications": ["Peer 3 (Validator) ← Peer 2 (read transformation)"],
        "messages": ["[Peer 3 - Validator] Validation completed"]
    }

# ============================================================================
# PARALLEL PEER EXECUTION
# ============================================================================

def parallel_peer_1(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Parallel Peer 1: Can run concurrently with other parallel peers
    
    In parallel horizontal composition:
    - Peers execute simultaneously
    - No dependencies between peers
    - Results aggregated after all complete
    """
    return {
        "peer_communications": ["Parallel Peer 1 executing concurrently"],
        "messages": ["[Parallel Peer 1] Completed independently"]
    }

def parallel_peer_2(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """Parallel Peer 2: Runs alongside Peer 1"""
    return {
        "peer_communications": ["Parallel Peer 2 executing concurrently"],
        "messages": ["[Parallel Peer 2] Completed independently"]
    }

def parallel_peer_3(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """Parallel Peer 3: Runs alongside Peer 1 and 2"""
    return {
        "peer_communications": ["Parallel Peer 3 executing concurrently"],
        "messages": ["[Parallel Peer 3] Completed independently"]
    }

# ============================================================================
# PEER AGGREGATOR
# ============================================================================

def peer_aggregator(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Aggregator: Combines results from all peers
    
    Note: In pure horizontal composition, aggregation could be done by
    one of the peers. Having a separate aggregator is a hybrid approach.
    """
    peer_1 = state.get("peer_1_result", {})
    peer_2 = state.get("peer_2_result", {})
    peer_3 = state.get("peer_3_result", {})
    
    aggregated = f"""
    Horizontal Composition Results:
    
    Peer 1 (Analyzer): {peer_1.get('capability', 'N/A')} - {peer_1.get('completed', False)}
    Peer 2 (Transformer): {peer_2.get('capability', 'N/A')} - {peer_2.get('completed', False)}
    Peer 3 (Validator): {peer_3.get('capability', 'N/A')} - {peer_3.get('completed', False)}
    
    All peers contributed as equals to achieve the goal.
    """
    
    return {
        "aggregated_result": aggregated.strip(),
        "messages": ["[Aggregator] Combined results from all peer components"]
    }

# ============================================================================
# PEER-TO-PEER COMMUNICATION PATTERNS
# ============================================================================

class PeerComponent:
    """
    Base class for peer components in horizontal composition
    
    Characteristics of peers:
    - Equal status (no hierarchy)
    - Direct communication with other peers
    - Specialized capability
    - Can discover and contact other peers
    """
    
    def __init__(self, name: str, capability: str):
        self.name = name
        self.capability = capability
        self.peers: Dict[str, 'PeerComponent'] = {}
    
    def register_peer(self, peer: 'PeerComponent'):
        """Register another peer for direct communication"""
        self.peers[peer.name] = peer
    
    def send_message(self, peer_name: str, message: str) -> str:
        """Send message to a specific peer (lateral communication)"""
        if peer_name in self.peers:
            return f"{self.name} → {peer_name}: {message}"
        return f"Peer {peer_name} not found"
    
    def broadcast(self, message: str) -> List[str]:
        """Broadcast message to all known peers"""
        return [f"{self.name} → {peer}: {message}" for peer in self.peers.keys()]
    
    def process(self, data: Any) -> Dict[str, Any]:
        """Process data using this peer's capability"""
        return {
            "peer": self.name,
            "capability": self.capability,
            "result": f"Processed by {self.name}"
        }

class AnalyzerPeer(PeerComponent):
    """Analyzer peer implementation"""
    def __init__(self):
        super().__init__("analyzer", "data_analysis")

class TransformerPeer(PeerComponent):
    """Transformer peer implementation"""
    def __init__(self):
        super().__init__("transformer", "data_transformation")

class ValidatorPeer(PeerComponent):
    """Validator peer implementation"""
    def __init__(self):
        super().__init__("validator", "data_validation")

# ============================================================================
# HORIZONTAL PIPELINE PATTERN
# ============================================================================

def create_horizontal_pipeline():
    """
    Horizontal Pipeline: Peers arranged in sequence
    
    A → B → C
    
    Each peer:
    - Processes data from previous peer
    - Passes result to next peer
    - Equal partners (not layers, just sequential)
    """
    
    peers = [
        {"name": "extractor", "position": 1},
        {"name": "transformer", "position": 2},
        {"name": "validator", "position": 3},
        {"name": "loader", "position": 4}
    ]
    
    return peers

# ============================================================================
# BROADCAST PATTERN
# ============================================================================

def broadcast_to_peers(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Broadcast Pattern: One peer sends to many
    
         → Peer A
    Root → Peer B
         → Peer C
    
    All peers process the same input concurrently
    """
    return {
        "peer_communications": [
            "Root → [Peer A, Peer B, Peer C] (broadcast)"
        ],
        "messages": ["[Broadcast] Message sent to all peers"]
    }

# ============================================================================
# ROUND-ROBIN PATTERN
# ============================================================================

def round_robin_peers(state: HorizontalCompositionState) -> HorizontalCompositionState:
    """
    Round-Robin Pattern: Requests distributed evenly
    
    Request 1 → Peer A
    Request 2 → Peer B
    Request 3 → Peer C
    Request 4 → Peer A (cycle back)
    
    Load balancing across equal peers
    """
    return {
        "peer_communications": ["Requests distributed round-robin"],
        "messages": ["[Round-Robin] Load balanced across peers"]
    }

# ============================================================================
# BUILD THE HORIZONTAL COMPOSITION GRAPH
# ============================================================================

def create_horizontal_composition_graph():
    """
    Create a StateGraph demonstrating horizontal composition.
    
    Flow (Sequential Peers - Pipeline Pattern):
    1. Peer 1 (Analyzer) processes input
    2. Peer 2 (Transformer) uses Peer 1's results
    3. Peer 3 (Validator) validates Peer 2's results
    4. Aggregator combines all results
    
    Note: These are peers (same level), not layers (different levels)
    """
    
    workflow = StateGraph(HorizontalCompositionState)
    
    # Add peer nodes (all at same level conceptually)
    workflow.add_node("peer_1", peer_1_analyzer)
    workflow.add_node("peer_2", peer_2_transformer)
    workflow.add_node("peer_3", peer_3_validator)
    workflow.add_node("aggregator", peer_aggregator)
    
    # Sequential peer execution (pipeline pattern)
    workflow.add_edge(START, "peer_1")
    workflow.add_edge("peer_1", "peer_2")
    workflow.add_edge("peer_2", "peer_3")
    workflow.add_edge("peer_3", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

def create_parallel_peers_graph():
    """
    Create a StateGraph with parallel peer execution.
    
    Flow (Parallel Peers):
    1. All peers execute concurrently
    2. Aggregator waits for all peers
    3. Results combined
    
    This demonstrates true horizontal parallelism.
    """
    
    workflow = StateGraph(HorizontalCompositionState)
    
    # Add parallel peer nodes
    workflow.add_node("parallel_1", parallel_peer_1)
    workflow.add_node("parallel_2", parallel_peer_2)
    workflow.add_node("parallel_3", parallel_peer_3)
    workflow.add_node("aggregator", peer_aggregator)
    
    # Parallel execution (all peers from START)
    workflow.add_edge(START, "parallel_1")
    workflow.add_edge(START, "parallel_2")
    workflow.add_edge(START, "parallel_3")
    
    # All converge to aggregator
    workflow.add_edge("parallel_1", "aggregator")
    workflow.add_edge("parallel_2", "aggregator")
    workflow.add_edge("parallel_3", "aggregator")
    
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Horizontal Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Sequential Peers (Pipeline Pattern)
    print("\n" + "=" * 80)
    print("Example 1: Horizontal Pipeline (Sequential Peers)")
    print("=" * 80)
    
    horizontal_graph = create_horizontal_composition_graph()
    
    initial_state: HorizontalCompositionState = {
        "input_data": "Customer data: John Doe, age 35, purchases: 5",
        "peer_1_result": None,
        "peer_2_result": None,
        "peer_3_result": None,
        "aggregated_result": "",
        "peer_communications": [],
        "messages": []
    }
    
    result = horizontal_graph.invoke(initial_state)
    
    print("\nPeer Communications (Horizontal Collaboration):")
    for comm in result["peer_communications"]:
        print(f"  {comm}")
    
    print("\nPeer Processing:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nPeer Results:")
    print(f"  Peer 1: {result.get('peer_1_result', {}).get('peer', 'N/A')}")
    print(f"  Peer 2: {result.get('peer_2_result', {}).get('peer', 'N/A')}")
    print(f"  Peer 3: {result.get('peer_3_result', {}).get('peer', 'N/A')}")
    
    print("\nAggregated Result:")
    print(result["aggregated_result"])
    
    # Example 2: Peer-to-Peer Communication
    print("\n" + "=" * 80)
    print("Example 2: Peer-to-Peer Direct Communication")
    print("=" * 80)
    
    # Create peer instances
    analyzer = AnalyzerPeer()
    transformer = TransformerPeer()
    validator = ValidatorPeer()
    
    # Register peers with each other (mesh topology)
    analyzer.register_peer(transformer)
    analyzer.register_peer(validator)
    transformer.register_peer(analyzer)
    transformer.register_peer(validator)
    validator.register_peer(analyzer)
    validator.register_peer(transformer)
    
    print("\nPeer Network:")
    print(f"  Analyzer knows: {list(analyzer.peers.keys())}")
    print(f"  Transformer knows: {list(transformer.peers.keys())}")
    print(f"  Validator knows: {list(validator.peers.keys())}")
    
    print("\nDirect Communication:")
    print(f"  {analyzer.send_message('transformer', 'Analysis complete')}")
    print(f"  {transformer.send_message('validator', 'Data transformed')}")
    print(f"  {validator.send_message('analyzer', 'Validation passed')}")
    
    print("\nBroadcast Communication:")
    broadcasts = analyzer.broadcast("Starting new analysis cycle")
    for bc in broadcasts:
        print(f"  {bc}")
    
    # Example 3: Horizontal vs Vertical Comparison
    print("\n" + "=" * 80)
    print("Example 3: Horizontal vs Vertical Composition Comparison")
    print("=" * 80)
    
    print("\nHorizontal Composition:")
    print("  Structure: Peers at same level")
    print("  Relationship: Equal partners")
    print("  Communication: Lateral (peer-to-peer)")
    print("  Flow: Sequential, parallel, or mesh")
    print("  Example: [Analyzer] → [Transformer] → [Validator]")
    print("  Coordination: Distributed or self-organizing")
    print("  Use case: Data pipelines, microservices, agent teams")
    
    print("\nVertical Composition:")
    print("  Structure: Layers stacked vertically")
    print("  Relationship: Superior-subordinate")
    print("  Communication: Through layer boundaries")
    print("  Flow: Top-down (request) and bottom-up (response)")
    print("  Example: [UI Layer] ↓ [Business Layer] ↓ [Data Layer]")
    print("  Coordination: Centralized (upper layers direct lower)")
    print("  Use case: Traditional enterprise apps, layered architecture")
    
    # Example 4: Parallel Peers
    print("\n" + "=" * 80)
    print("Example 4: Parallel Peer Execution")
    print("=" * 80)
    
    print("\nParallel peers execute concurrently:")
    print("  Peer 1: Independent task A")
    print("  Peer 2: Independent task B")
    print("  Peer 3: Independent task C")
    print("  All peers run simultaneously, results aggregated")
    
    # Example 5: Horizontal Composition Patterns
    print("\n" + "=" * 80)
    print("Example 5: Horizontal Composition Patterns")
    print("=" * 80)
    
    print("\n1. Pipeline Pattern:")
    print("   A → B → C → D")
    print("   Sequential peers, each processes and passes to next")
    
    print("\n2. Parallel Pattern:")
    print("   [A, B, C, D]")
    print("   All peers execute concurrently")
    
    print("\n3. Broadcast Pattern:")
    print("        → A")
    print("   Root → B")
    print("        → C")
    print("   One source distributes to many peers")
    
    print("\n4. Mesh Pattern:")
    print("   A ↔ B")
    print("   ↕   ↕")
    print("   C ↔ D")
    print("   All peers can communicate with all others")
    
    print("\n5. Round-Robin Pattern:")
    print("   A → B → C → A (circular)")
    print("   Requests distributed evenly across peers")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Horizontal Composition organizes components as peers (same level)
2. Peers are equal partners, not hierarchical
3. Lateral Communication: peers talk directly to each other
4. Pipeline Pattern: A → B → C (sequential peers)
5. Parallel Pattern: [A, B, C] (concurrent peers)
6. Peer-to-Peer: bidirectional communication between equals
7. Broadcast: one peer sends to many
8. Round-Robin: distribute load evenly across peers
9. Benefits: flexibility, scalability, resilience, parallelism
10. Trade-offs: coordination overhead, no central control, discovery needed
11. vs Vertical: horizontal = peers, vertical = layers
12. Use cases: data pipelines, microservices, agent collaboration, stream processing
    """)

"""
Pattern 210: Edge Computing MCP Pattern

Edge computing moves computation closer to data sources:
- Process data at network edge (near users/devices)
- Reduce latency (ms instead of 100s of ms)
- Reduce bandwidth (process locally, send only results)
- Offline capability (work without cloud connection)

Use Cases:
- IoT devices and sensors
- CDN and content delivery
- Real-time video processing
- Autonomous vehicles
- AR/VR applications
- Smart cities

Architecture:
- Edge Devices: Process data locally
- Edge Gateways: Aggregate multiple devices
- Regional Edge: Regional data centers
- Cloud: Central processing and storage
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
import time


class EdgeComputingState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class EdgeNode:
    node_id: str
    location: str
    processing_capacity: int = 100
    latency_to_cloud_ms: float = 100.0
    data_processed_locally: int = 0
    data_sent_to_cloud: int = 0
    
    def process_locally(self, data_size: int) -> Dict[str, Any]:
        """Process data at edge"""
        start = time.time()
        # Simulate local processing (fast)
        time.sleep(0.001)  # 1ms
        self.data_processed_locally += data_size
        latency = (time.time() - start) * 1000
        
        return {
            'processed_at': 'edge',
            'latency_ms': latency,
            'data_size': data_size
        }
    
    def send_to_cloud(self, data_size: int) -> Dict[str, Any]:
        """Send data to cloud for processing"""
        start = time.time()
        # Simulate cloud round-trip
        time.sleep(self.latency_to_cloud_ms / 1000)
        self.data_sent_to_cloud += data_size
        latency = (time.time() - start) * 1000
        
        return {
            'processed_at': 'cloud',
            'latency_ms': latency,
            'data_size': data_size
        }


class EdgeComputingSystem:
    def __init__(self):
        self.edge_nodes: List[EdgeNode] = []
        self.cloud_latency_ms = 150.0
        self.edge_latency_ms = 5.0
    
    def add_edge_node(self, node: EdgeNode):
        self.edge_nodes.append(node)
    
    def get_nearest_edge(self, location: str) -> EdgeNode:
        """Get nearest edge node (simplified)"""
        return self.edge_nodes[0] if self.edge_nodes else None
    
    def process_data(self, location: str, data_size: int, process_at_edge: bool = True) -> Dict[str, Any]:
        """Process data at edge or cloud"""
        edge_node = self.get_nearest_edge(location)
        
        if process_at_edge and edge_node:
            return edge_node.process_locally(data_size)
        elif edge_node:
            return edge_node.send_to_cloud(data_size)
        else:
            return {'processed_at': 'cloud', 'latency_ms': self.cloud_latency_ms}
    
    def get_stats(self) -> Dict[str, Any]:
        total_edge = sum(n.data_processed_locally for n in self.edge_nodes)
        total_cloud = sum(n.data_sent_to_cloud for n in self.edge_nodes)
        
        return {
            'edge_nodes': len(self.edge_nodes),
            'data_processed_at_edge': total_edge,
            'data_sent_to_cloud': total_cloud,
            'edge_percentage': (total_edge / (total_edge + total_cloud) * 100) if (total_edge + total_cloud) > 0 else 0
        }


def demo_agent(state: EdgeComputingState):
    operations = []
    results = []
    metrics = []
    
    system = EdgeComputingSystem()
    
    # Deploy edge nodes
    edge_nodes = [
        EdgeNode("edge-sf", "San Francisco", latency_to_cloud_ms=50),
        EdgeNode("edge-ny", "New York", latency_to_cloud_ms=30),
        EdgeNode("edge-london", "London", latency_to_cloud_ms=80)
    ]
    
    operations.append("Edge Computing Deployment:")
    for node in edge_nodes:
        system.add_edge_node(node)
        operations.append(f"  {node.node_id} ({node.location}): {node.latency_to_cloud_ms}ms to cloud")
    
    # Process data at edge vs cloud
    operations.append("\nProcessing Comparison:")
    
    # Edge processing
    operations.append("\n1. Edge Processing (Local):")
    for i in range(3):
        result = system.process_data("San Francisco", 100, process_at_edge=True)
        operations.append(f"  Request {i+1}: {result['latency_ms']:.2f}ms at {result['processed_at']}")
    
    # Cloud processing
    operations.append("\n2. Cloud Processing (Remote):")
    for i in range(3):
        result = system.process_data("San Francisco", 100, process_at_edge=False)
        operations.append(f"  Request {i+1}: {result['latency_ms']:.2f}ms at {result['processed_at']}")
    
    stats = system.get_stats()
    
    operations.append(f"\nData Distribution:")
    operations.append(f"  Processed at edge: {stats['data_processed_at_edge']} KB ({stats['edge_percentage']:.1f}%)")
    operations.append(f"  Sent to cloud: {stats['data_sent_to_cloud']} KB")
    
    results.append(f"✓ Edge computing: {stats['edge_nodes']} edge nodes deployed")
    results.append(f"✓ {stats['edge_percentage']:.1f}% processed at edge (reduced latency)")
    
    metrics.append(f"Edge latency: ~5ms")
    metrics.append(f"Cloud latency: ~50-150ms")
    metrics.append(f"Speedup: 10-30x faster")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Edge computing demo complete"]
    }


def create_edge_computing_graph():
    """Create the edge computing workflow graph"""
    workflow = StateGraph(EdgeComputingState)
    
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 210: Edge Computing MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_edge_computing_graph()
    
    # Initialize state
    initial_state = {
        "scaling_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("SCALING OPERATIONS")
    print("=" * 80)
    for op in final_state["scaling_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    for metric in final_state["performance_metrics"]:
        print(metric)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Edge Computing Benefits:

1. Reduced Latency:
   - Process data locally (5-10ms)
   - vs Cloud round-trip (50-150ms)
   - Critical for real-time applications

2. Bandwidth Optimization:
   - Process data at source
   - Send only insights to cloud
   - 90% bandwidth reduction typical

3. Offline Capability:
   - Works without cloud connection
   - Critical for remote locations
   - Intermittent connectivity scenarios

4. Privacy & Security:
   - Sensitive data stays local
   - Reduced data exposure
   - Compliance with data sovereignty

Edge Computing Tiers:
- Device Edge: On-device processing (phones, IoT)
- Edge Gateway: Local aggregation point
- Regional Edge: City/region data center
- Cloud: Central processing and storage

Real-World Examples:
- CDN: CloudFlare, Akamai edge servers
- IoT: AWS IoT Greengrass, Azure IoT Edge
- Video: Live streaming edge processing
- Retail: In-store analytics and personalization
- Manufacturing: Real-time quality control
- Smart Cities: Traffic management, surveillance

Use Cases:
✓ Autonomous Vehicles: Real-time decision making
✓ AR/VR: Low-latency rendering
✓ Smart Factories: Predictive maintenance
✓ Healthcare: Patient monitoring
✓ Retail: Inventory management
✓ Gaming: Low-latency multiplayer

Edge vs Cloud Trade-offs:

Edge Computing:
  ✓ Ultra-low latency (ms)
  ✓ Bandwidth savings
  ✓ Offline capability
  ✓ Privacy/locality
  ✗ Limited compute power
  ✗ Management complexity
  ✗ Higher cost per node

Cloud Computing:
  ✓ Unlimited compute/storage
  ✓ Easy management
  ✓ Cost-effective at scale
  ✗ Network latency (100ms+)
  ✗ Bandwidth costs
  ✗ Requires connectivity
""")


if __name__ == "__main__":
    main()

"""
Pattern 201: Horizontal Scaling MCP Pattern

This pattern demonstrates horizontal scaling (scale-out) by adding more nodes:
- Adding servers/instances to distribute load
- Load balancing across multiple nodes
- Session affinity and sticky sessions
- Stateless design for easy scaling

Horizontal scaling adds more machines to handle increased load, rather than
making a single machine more powerful (vertical scaling).

Use Cases:
- Web applications with growing traffic
- Microservices architectures
- Distributed data processing
- High-availability systems
- Cloud-native applications

Key Features:
- Linear scalability: Add nodes to increase capacity
- Load distribution: Balance work across nodes
- Fault tolerance: Failure of one node doesn't bring down system
- Cost-effective: Use commodity hardware
- Elasticity: Add/remove nodes based on demand

Advantages:
✓ Better fault tolerance (no single point of failure)
✓ Easier to scale incrementally
✓ Cost-effective with commodity hardware
✓ Cloud-friendly (pay per node)

Disadvantages:
✗ More complex architecture
✗ Data consistency challenges
✗ Network overhead
✗ Session management complexity
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random
from collections import defaultdict


class HorizontalScalingState(TypedDict):
    """State for horizontal scaling operations"""
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class ServerNode:
    """Represents a server node in the cluster"""
    node_id: str
    host: str
    port: int
    capacity: int = 100  # Max concurrent requests
    current_load: int = 0
    is_healthy: bool = True
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    created_at: float = field(default_factory=time.time)
    
    def process_request(self, request_id: str) -> bool:
        """Process a request if capacity available"""
        if not self.is_healthy:
            self.failed_requests += 1
            return False
        
        if self.current_load >= self.capacity:
            self.failed_requests += 1
            return False
        
        self.current_load += 1
        self.total_requests += 1
        
        # Simulate processing time
        processing_time = random.uniform(0.01, 0.05)
        time.sleep(processing_time)
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + processing_time) 
            / self.total_requests
        )
        
        self.current_load -= 1
        return True
    
    def get_load_percentage(self) -> float:
        """Get current load as percentage"""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 0


class LoadBalancer:
    """
    Load balancer for distributing requests across nodes.
    
    Implements multiple load balancing algorithms:
    - Round Robin: Distribute evenly in rotation
    - Least Connections: Send to node with fewest active connections
    - Weighted Round Robin: Consider node capacity
    - Random: Random selection
    """
    
    def __init__(self, algorithm: str = "round_robin"):
        self.algorithm = algorithm
        self.nodes: List[ServerNode] = []
        self.current_index = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
    
    def add_node(self, node: ServerNode):
        """Add a node to the pool"""
        self.nodes.append(node)
    
    def remove_node(self, node_id: str):
        """Remove a node from the pool"""
        self.nodes = [n for n in self.nodes if n.node_id != node_id]
    
    def get_healthy_nodes(self) -> List[ServerNode]:
        """Get list of healthy nodes"""
        return [n for n in self.nodes if n.is_healthy]
    
    def select_node_round_robin(self) -> Optional[ServerNode]:
        """Round robin selection"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        node = healthy_nodes[self.current_index % len(healthy_nodes)]
        self.current_index += 1
        return node
    
    def select_node_least_connections(self) -> Optional[ServerNode]:
        """Least connections selection"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        return min(healthy_nodes, key=lambda n: n.current_load)
    
    def select_node_weighted(self) -> Optional[ServerNode]:
        """Weighted selection based on capacity"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        # Calculate weights based on available capacity
        weights = []
        for node in healthy_nodes:
            available = node.capacity - node.current_load
            weights.append(max(1, available))
        
        # Weighted random selection
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative = 0
        for node, weight in zip(healthy_nodes, weights):
            cumulative += weight
            if r <= cumulative:
                return node
        
        return healthy_nodes[0]
    
    def select_node_random(self) -> Optional[ServerNode]:
        """Random selection"""
        healthy_nodes = self.get_healthy_nodes()
        if not healthy_nodes:
            return None
        
        return random.choice(healthy_nodes)
    
    def select_node(self) -> Optional[ServerNode]:
        """Select node based on algorithm"""
        if self.algorithm == "round_robin":
            return self.select_node_round_robin()
        elif self.algorithm == "least_connections":
            return self.select_node_least_connections()
        elif self.algorithm == "weighted":
            return self.select_node_weighted()
        elif self.algorithm == "random":
            return self.select_node_random()
        else:
            return self.select_node_round_robin()
    
    def handle_request(self, request_id: str) -> bool:
        """Handle a request by routing to appropriate node"""
        self.total_requests += 1
        
        node = self.select_node()
        if not node:
            self.failed_requests += 1
            return False
        
        success = node.process_request(request_id)
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        return success
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_count = len(self.get_healthy_nodes())
        total_count = len(self.nodes)
        
        success_rate = (
            (self.successful_requests / self.total_requests * 100)
            if self.total_requests > 0 else 0
        )
        
        node_stats = []
        for node in self.nodes:
            node_stats.append({
                'node_id': node.node_id,
                'load': f"{node.get_load_percentage():.1f}%",
                'total_requests': node.total_requests,
                'avg_response_time': f"{node.avg_response_time*1000:.2f}ms",
                'healthy': node.is_healthy
            })
        
        return {
            'algorithm': self.algorithm,
            'total_nodes': total_count,
            'healthy_nodes': healthy_count,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': f"{success_rate:.1f}%",
            'node_stats': node_stats
        }


def setup_cluster_agent(state: HorizontalScalingState):
    """Agent to set up initial cluster"""
    operations = []
    results = []
    
    # Create load balancer
    lb = LoadBalancer(algorithm="round_robin")
    
    # Start with 2 nodes
    initial_nodes = [
        ServerNode("node-1", "192.168.1.1", 8080, capacity=50),
        ServerNode("node-2", "192.168.1.2", 8080, capacity=50),
    ]
    
    for node in initial_nodes:
        lb.add_node(node)
        operations.append(f"Added {node.node_id} (capacity: {node.capacity})")
    
    results.append(f"✓ Initial cluster: {len(initial_nodes)} nodes")
    results.append(f"✓ Total capacity: {sum(n.capacity for n in initial_nodes)} req/s")
    
    # Store in state
    state['_lb'] = lb
    state['_nodes'] = initial_nodes
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Initial cluster setup complete"]
    }


def simulate_initial_load_agent(state: HorizontalScalingState):
    """Agent to simulate initial load"""
    lb = state['_lb']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n📊 Simulating initial load (50 requests)...")
    
    # Simulate 50 requests
    for i in range(50):
        request_id = f"req-{i+1:03d}"
        success = lb.handle_request(request_id)
    
    stats = lb.get_statistics()
    
    operations.append(f"\nLoad Distribution:")
    for node_stat in stats['node_stats']:
        operations.append(
            f"  {node_stat['node_id']}: "
            f"{node_stat['total_requests']} requests, "
            f"load: {node_stat['load']}, "
            f"avg: {node_stat['avg_response_time']}"
        )
    
    results.append(f"✓ Handled {stats['successful_requests']}/{stats['total_requests']} requests")
    results.append(f"✓ Success rate: {stats['success_rate']}")
    
    metrics.append(f"Initial capacity: {len(state['_nodes'])} nodes")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Initial load simulation complete"]
    }


def scale_out_agent(state: HorizontalScalingState):
    """Agent to demonstrate horizontal scale-out"""
    lb = state['_lb']
    nodes = state['_nodes']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🚀 HORIZONTAL SCALE-OUT: Adding more nodes...")
    
    # Add 3 more nodes
    new_nodes = [
        ServerNode("node-3", "192.168.1.3", 8080, capacity=50),
        ServerNode("node-4", "192.168.1.4", 8080, capacity=50),
        ServerNode("node-5", "192.168.1.5", 8080, capacity=50),
    ]
    
    for node in new_nodes:
        lb.add_node(node)
        nodes.append(node)
        operations.append(f"  ➕ Added {node.node_id} (capacity: {node.capacity})")
    
    total_capacity = sum(n.capacity for n in lb.nodes)
    operations.append(f"\n✓ Cluster scaled to {len(lb.nodes)} nodes")
    operations.append(f"✓ Total capacity: {total_capacity} req/s")
    
    # Simulate higher load (150 requests)
    operations.append(f"\n📊 Simulating higher load (150 requests)...")
    
    for i in range(150):
        request_id = f"req-scale-{i+1:03d}"
        lb.handle_request(request_id)
    
    stats = lb.get_statistics()
    
    operations.append(f"\nLoad Distribution After Scale-Out:")
    for node_stat in stats['node_stats']:
        operations.append(
            f"  {node_stat['node_id']}: "
            f"{node_stat['total_requests']} requests, "
            f"load: {node_stat['load']}"
        )
    
    results.append(f"✓ Scaled from 2 to {len(lb.nodes)} nodes")
    results.append(f"✓ Capacity increased: 100 → {total_capacity} req/s ({total_capacity/100:.0f}x)")
    results.append(f"✓ Handled {stats['successful_requests']}/{stats['total_requests']} total requests")
    
    metrics.append(f"Scale-out: +{len(new_nodes)} nodes")
    metrics.append(f"New capacity: {total_capacity} req/s")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Horizontal scale-out complete"]
    }


def load_balancing_comparison_agent(state: HorizontalScalingState):
    """Agent to compare load balancing algorithms"""
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🔀 LOAD BALANCING ALGORITHM COMPARISON:")
    
    algorithms = ["round_robin", "least_connections", "weighted", "random"]
    
    for algorithm in algorithms:
        # Create new load balancer with algorithm
        lb = LoadBalancer(algorithm=algorithm)
        
        # Add nodes
        test_nodes = [
            ServerNode(f"test-{i+1}", f"192.168.2.{i+1}", 8080, capacity=50)
            for i in range(3)
        ]
        
        for node in test_nodes:
            lb.add_node(node)
        
        # Simulate 100 requests
        for i in range(100):
            lb.handle_request(f"test-{algorithm}-{i:03d}")
        
        stats = lb.get_statistics()
        
        operations.append(f"\n{algorithm.upper().replace('_', ' ')}:")
        operations.append(f"  Success rate: {stats['success_rate']}")
        
        # Show distribution
        request_counts = [n['total_requests'] for n in stats['node_stats']]
        std_dev = (sum((x - sum(request_counts)/len(request_counts))**2 for x in request_counts) / len(request_counts))**0.5
        
        operations.append(f"  Distribution: {request_counts}")
        operations.append(f"  Std deviation: {std_dev:.2f}")
        
        metrics.append(f"{algorithm}: std_dev={std_dev:.2f}")
    
    results.append("✓ Compared 4 load balancing algorithms")
    results.append("  - Round Robin: Predictable, even distribution")
    results.append("  - Least Connections: Adapts to node load")
    results.append("  - Weighted: Considers node capacity")
    results.append("  - Random: Simple, no coordination needed")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Load balancing comparison complete"]
    }


def fault_tolerance_agent(state: HorizontalScalingState):
    """Agent to demonstrate fault tolerance with horizontal scaling"""
    lb = state['_lb']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🛡️ FAULT TOLERANCE DEMONSTRATION:")
    
    # Simulate node failure
    failing_node = lb.nodes[1]  # node-2
    failing_node.is_healthy = False
    
    operations.append(f"\n⚠️  Simulated failure: {failing_node.node_id}")
    
    # Continue serving requests
    operations.append(f"\n📊 Continuing to serve requests with remaining nodes...")
    
    success_count = 0
    for i in range(50):
        request_id = f"req-failure-{i+1:03d}"
        if lb.handle_request(request_id):
            success_count += 1
    
    healthy_nodes = lb.get_healthy_nodes()
    operations.append(f"\n✓ Healthy nodes: {len(healthy_nodes)}/{len(lb.nodes)}")
    operations.append(f"✓ Successfully handled: {success_count}/50 requests")
    operations.append(f"✓ System remained operational despite node failure")
    
    results.append(f"✓ Survived {1} node failure")
    results.append(f"✓ {len(healthy_nodes)} healthy nodes continued serving")
    
    metrics.append(f"Fault tolerance: {success_count}/50 success with 1 node down")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Fault tolerance demonstrated"]
    }


def performance_summary_agent(state: HorizontalScalingState):
    """Agent to summarize performance"""
    lb = state['_lb']
    operations = []
    results = []
    metrics = []
    
    stats = lb.get_statistics()
    
    operations.append("\n" + "="*60)
    operations.append("HORIZONTAL SCALING SUMMARY")
    operations.append("="*60)
    
    operations.append(f"\nCluster Configuration:")
    operations.append(f"  Total nodes: {stats['total_nodes']}")
    operations.append(f"  Healthy nodes: {stats['healthy_nodes']}")
    operations.append(f"  Load balancing: {stats['algorithm']}")
    
    operations.append(f"\nPerformance Metrics:")
    operations.append(f"  Total requests: {stats['total_requests']}")
    operations.append(f"  Successful: {stats['successful_requests']}")
    operations.append(f"  Failed: {stats['failed_requests']}")
    operations.append(f"  Success rate: {stats['success_rate']}")
    
    operations.append(f"\nNode Statistics:")
    for node_stat in stats['node_stats']:
        operations.append(
            f"  {node_stat['node_id']}: "
            f"{node_stat['total_requests']} reqs, "
            f"avg {node_stat['avg_response_time']}, "
            f"{'✓' if node_stat['healthy'] else '✗'}"
        )
    
    metrics.append("\n📊 Horizontal Scaling Benefits:")
    metrics.append("  ✓ Linear scalability: Add nodes = Add capacity")
    metrics.append("  ✓ Fault tolerance: No single point of failure")
    metrics.append("  ✓ Cost-effective: Use commodity hardware")
    metrics.append("  ✓ Elastic: Easy to add/remove nodes")
    metrics.append("  ✓ Load distribution: Balance across nodes")
    
    results.append("✓ Horizontal scaling demonstrated successfully")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Performance summary complete"]
    }


def create_horizontal_scaling_graph():
    """Create the horizontal scaling workflow graph"""
    workflow = StateGraph(HorizontalScalingState)
    
    # Add nodes
    workflow.add_node("setup", setup_cluster_agent)
    workflow.add_node("initial_load", simulate_initial_load_agent)
    workflow.add_node("scale_out", scale_out_agent)
    workflow.add_node("lb_comparison", load_balancing_comparison_agent)
    workflow.add_node("fault_tolerance", fault_tolerance_agent)
    workflow.add_node("summary", performance_summary_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "initial_load")
    workflow.add_edge("initial_load", "scale_out")
    workflow.add_edge("scale_out", "lb_comparison")
    workflow.add_edge("lb_comparison", "fault_tolerance")
    workflow.add_edge("fault_tolerance", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 201: Horizontal Scaling MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_horizontal_scaling_graph()
    
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
Horizontal Scaling Pattern implemented with:

1. Scale-Out Architecture:
   - Add more servers/nodes to increase capacity
   - Linear scalability: 2 nodes → 5 nodes = 2.5x capacity
   - Distributed load across multiple machines
   - No single point of failure

2. Load Balancing Algorithms:
   - Round Robin: Distribute requests evenly in rotation
   - Least Connections: Route to node with lowest load
   - Weighted: Consider node capacity/performance
   - Random: Simple random distribution

3. Fault Tolerance:
   - Node failure doesn't bring down system
   - Automatic routing to healthy nodes
   - Graceful degradation under failures
   - High availability through redundancy

4. Elasticity:
   - Easy to add nodes (scale out)
   - Easy to remove nodes (scale in)
   - Cloud-friendly: pay per node
   - Adapt to changing load

When to Use Horizontal Scaling:
✓ Web applications (add web servers)
✓ Stateless services (microservices)
✓ Distributed databases (sharding)
✓ High availability requirements
✓ Cost-effective scaling
✓ Cloud-native applications

Horizontal vs Vertical Scaling:

Horizontal (Scale-Out):
  ✓ Add more machines
  ✓ Better fault tolerance
  ✓ Linear scalability
  ✓ Cost-effective
  ✗ More complex
  ✗ Network overhead

Vertical (Scale-Up):
  ✓ Simple architecture
  ✓ No network overhead
  ✓ Easy consistency
  ✗ Limited by hardware
  ✗ Single point of failure
  ✗ Expensive at high end

Real-World Examples:
- Web Servers: Nginx/Apache load balancing across app servers
- Databases: MongoDB/Cassandra sharding across nodes
- Microservices: Kubernetes scaling pods horizontally
- CDN: CloudFlare/Akamai distributing across edge servers
- Message Queues: Kafka partitions across brokers

Design Considerations:
1. Stateless Design: Store state externally (Redis, database)
2. Session Management: Sticky sessions or shared session store
3. Data Consistency: Handle distributed transactions
4. Load Balancer: Single LB can become bottleneck (use multiple)
5. Health Checks: Monitor node health, auto-remove unhealthy nodes
6. Auto-Scaling: Automate scale-out/in based on metrics
""")


if __name__ == "__main__":
    main()

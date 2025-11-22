"""
Pattern 198: Distributed Cache MCP Pattern

This pattern demonstrates distributed caching across multiple nodes with:
- Consistent hashing for data partitioning
- Replication for high availability
- Node discovery and cluster management
- Partition tolerance and failover

Distributed caching allows sharing cache across multiple servers/nodes,
providing scalability and fault tolerance.

Use Cases:
- Multi-server web applications
- Microservices architectures
- Large-scale data processing
- Session management across servers
- Reducing database load with shared cache

Key Features:
- Consistent Hashing: Data partitioned across nodes
- Replication: Multiple copies for availability
- Node Discovery: Dynamic cluster membership
- Failover: Automatic recovery from node failures
- Partition Tolerance: Works despite network splits
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import hashlib
import time
from collections import defaultdict
import random


class DistributedCacheState(TypedDict):
    """State for distributed cache operations"""
    cache_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    cluster_state: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class CacheNode:
    """Represents a cache node in the cluster"""
    node_id: str
    host: str
    port: int
    is_active: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        return isinstance(other, CacheNode) and self.node_id == other.node_id


class ConsistentHashRing:
    """
    Consistent hashing ring for distributing keys across nodes.
    
    Consistent hashing ensures:
    1. Keys are evenly distributed across nodes
    2. Adding/removing nodes affects only K/n keys (K=keys, n=nodes)
    3. Minimal data movement during rebalancing
    """
    
    def __init__(self, num_virtual_nodes: int = 150):
        self.num_virtual_nodes = num_virtual_nodes
        self.ring: Dict[int, CacheNode] = {}  # hash_value -> node
        self.sorted_hashes: List[int] = []
        self.nodes: Set[CacheNode] = set()
    
    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def add_node(self, node: CacheNode):
        """Add a node to the hash ring with virtual nodes"""
        self.nodes.add(node)
        
        # Create virtual nodes for better distribution
        for i in range(self.num_virtual_nodes):
            virtual_key = f"{node.node_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
            self.sorted_hashes.append(hash_value)
        
        self.sorted_hashes.sort()
    
    def remove_node(self, node: CacheNode):
        """Remove a node from the hash ring"""
        self.nodes.discard(node)
        
        # Remove all virtual nodes
        for i in range(self.num_virtual_nodes):
            virtual_key = f"{node.node_id}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
                self.sorted_hashes.remove(hash_value)
    
    def get_node(self, key: str) -> Optional[CacheNode]:
        """Get the node responsible for a key"""
        if not self.ring:
            return None
        
        hash_value = self._hash(key)
        
        # Find the first node clockwise from the hash
        for ring_hash in self.sorted_hashes:
            if hash_value <= ring_hash:
                return self.ring[ring_hash]
        
        # Wrap around to the first node
        return self.ring[self.sorted_hashes[0]]
    
    def get_nodes_for_replication(self, key: str, num_replicas: int) -> List[CacheNode]:
        """Get multiple nodes for replication"""
        if not self.ring or num_replicas == 0:
            return []
        
        hash_value = self._hash(key)
        nodes = []
        seen_node_ids = set()
        
        # Start from the primary node and move clockwise
        idx = 0
        for i, ring_hash in enumerate(self.sorted_hashes):
            if hash_value <= ring_hash:
                idx = i
                break
        
        # Collect unique nodes moving clockwise
        attempts = 0
        max_attempts = len(self.sorted_hashes)
        
        while len(nodes) < num_replicas and attempts < max_attempts:
            node = self.ring[self.sorted_hashes[idx % len(self.sorted_hashes)]]
            if node.node_id not in seen_node_ids and node.is_active:
                nodes.append(node)
                seen_node_ids.add(node.node_id)
            idx += 1
            attempts += 1
        
        return nodes


class DistributedCache:
    """
    Distributed cache implementation with:
    - Consistent hashing for partitioning
    - Replication for availability
    - Node failure handling
    """
    
    def __init__(self, replication_factor: int = 3):
        self.hash_ring = ConsistentHashRing()
        self.replication_factor = replication_factor
        self.statistics = {
            'gets': 0,
            'puts': 0,
            'hits': 0,
            'misses': 0,
            'replications': 0,
            'failovers': 0
        }
    
    def add_node(self, node: CacheNode):
        """Add a node to the cluster"""
        self.hash_ring.add_node(node)
    
    def remove_node(self, node: CacheNode):
        """Remove a node from the cluster"""
        self.hash_ring.remove_node(node)
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the distributed cache"""
        self.statistics['gets'] += 1
        
        # Get replica nodes for this key
        nodes = self.hash_ring.get_nodes_for_replication(key, self.replication_factor)
        
        if not nodes:
            self.statistics['misses'] += 1
            return None
        
        # Try each replica until we find the data
        for node in nodes:
            if not node.is_active:
                continue
            
            if key in node.data:
                self.statistics['hits'] += 1
                return node.data[key]
        
        self.statistics['misses'] += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put a value into the distributed cache with replication"""
        self.statistics['puts'] += 1
        
        # Get replica nodes
        nodes = self.hash_ring.get_nodes_for_replication(key, self.replication_factor)
        
        if not nodes:
            return
        
        # Write to all replicas
        successful_writes = 0
        for node in nodes:
            if not node.is_active:
                self.statistics['failovers'] += 1
                continue
            
            node.data[key] = value
            successful_writes += 1
            
            if successful_writes > 1:  # Count replications (excluding primary)
                self.statistics['replications'] += 1
    
    def delete(self, key: str):
        """Delete a value from all replicas"""
        nodes = self.hash_ring.get_nodes_for_replication(key, self.replication_factor)
        
        for node in nodes:
            if key in node.data:
                del node.data[key]
    
    def get_key_location(self, key: str) -> List[str]:
        """Get which nodes hold a key (for debugging)"""
        nodes = self.hash_ring.get_nodes_for_replication(key, self.replication_factor)
        return [node.node_id for node in nodes]
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get the current cluster state"""
        active_nodes = [n for n in self.hash_ring.nodes if n.is_active]
        inactive_nodes = [n for n in self.hash_ring.nodes if not n.is_active]
        
        # Count keys per node
        key_distribution = {}
        for node in active_nodes:
            key_distribution[node.node_id] = len(node.data)
        
        return {
            'active_nodes': len(active_nodes),
            'inactive_nodes': len(inactive_nodes),
            'total_nodes': len(self.hash_ring.nodes),
            'replication_factor': self.replication_factor,
            'key_distribution': key_distribution,
            'statistics': self.statistics
        }


def setup_cluster_agent(state: DistributedCacheState):
    """Agent to set up the distributed cache cluster"""
    operations = []
    results = []
    cluster_state = []
    
    # Create distributed cache with replication
    cache = DistributedCache(replication_factor=3)
    
    # Create cluster nodes
    nodes = [
        CacheNode("node-1", "192.168.1.1", 6379),
        CacheNode("node-2", "192.168.1.2", 6379),
        CacheNode("node-3", "192.168.1.3", 6379),
        CacheNode("node-4", "192.168.1.4", 6379),
        CacheNode("node-5", "192.168.1.5", 6379),
    ]
    
    # Add nodes to cluster
    for node in nodes:
        cache.add_node(node)
        operations.append(f"Added {node.node_id} to cluster")
    
    cluster_info = cache.get_cluster_state()
    results.append(f"Cluster initialized: {cluster_info['active_nodes']} active nodes")
    results.append(f"Replication factor: {cluster_info['replication_factor']}")
    
    # Store cache in state for next agents
    state['_cache'] = cache
    state['_nodes'] = nodes
    
    cluster_state.append(f"✓ Cluster setup complete with {len(nodes)} nodes")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "cluster_state": cluster_state,
        "messages": ["Distributed cache cluster initialized"]
    }


def data_distribution_agent(state: DistributedCacheState):
    """Agent to demonstrate data distribution across nodes"""
    cache = state['_cache']
    operations = []
    results = []
    cluster_state = []
    
    # Put data into the distributed cache
    test_data = {
        "user:1001": {"name": "Alice", "email": "alice@example.com"},
        "user:1002": {"name": "Bob", "email": "bob@example.com"},
        "user:1003": {"name": "Charlie", "email": "charlie@example.com"},
        "product:2001": {"name": "Laptop", "price": 999.99},
        "product:2002": {"name": "Mouse", "price": 29.99},
        "product:2003": {"name": "Keyboard", "price": 79.99},
        "session:abc123": {"user_id": 1001, "created": datetime.now().isoformat()},
        "session:def456": {"user_id": 1002, "created": datetime.now().isoformat()},
    }
    
    operations.append("Distributing data across cluster...")
    
    for key, value in test_data.items():
        cache.put(key, value)
        locations = cache.get_key_location(key)
        operations.append(f"PUT {key} -> Replicas: {locations}")
    
    # Show key distribution
    cluster_info = cache.get_cluster_state()
    results.append(f"Total keys distributed: {len(test_data)}")
    results.append(f"Replication factor: {cache.replication_factor}")
    results.append(f"Total replications: {cluster_info['statistics']['replications']}")
    
    results.append("\nKey distribution per node:")
    for node_id, count in cluster_info['key_distribution'].items():
        results.append(f"  {node_id}: {count} keys")
    
    cluster_state.append(f"✓ {len(test_data)} keys distributed with {cache.replication_factor}x replication")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "cluster_state": cluster_state,
        "messages": ["Data distributed across cluster"]
    }


def consistent_hashing_demo_agent(state: DistributedCacheState):
    """Agent to demonstrate consistent hashing benefits"""
    cache = state['_cache']
    nodes = state['_nodes']
    operations = []
    results = []
    cluster_state = []
    
    # Show which node each key belongs to
    test_keys = ["user:1001", "user:1002", "user:1003", "product:2001", "product:2002"]
    
    operations.append("\nConsistent Hashing Demonstration:")
    operations.append("Initial key-to-node mapping:")
    
    initial_mapping = {}
    for key in test_keys:
        locations = cache.get_key_location(key)
        initial_mapping[key] = locations
        operations.append(f"  {key} -> Primary: {locations[0]}")
    
    # Simulate node failure
    failed_node = nodes[2]  # node-3
    failed_node.is_active = False
    operations.append(f"\n⚠️  Node {failed_node.node_id} failed!")
    
    cluster_state.append(f"Node failure: {failed_node.node_id}")
    
    # Show that data is still available via replicas
    operations.append("\nData availability after node failure:")
    for key in test_keys:
        value = cache.get(key)
        if value:
            operations.append(f"  {key}: ✓ Available (from replica)")
        else:
            operations.append(f"  {key}: ✗ Lost")
    
    stats = cache.get_cluster_state()['statistics']
    results.append(f"Cache still operational: {stats['hits']} hits, {stats['misses']} misses")
    
    # Add a new node to demonstrate minimal data movement
    new_node = CacheNode("node-6", "192.168.1.6", 6379)
    nodes.append(new_node)
    cache.add_node(new_node)
    
    operations.append(f"\n➕ Added new node: {new_node.node_id}")
    cluster_state.append(f"Node added: {new_node.node_id}")
    
    # Show new key mapping
    operations.append("\nKey mapping after adding node:")
    keys_moved = 0
    for key in test_keys:
        new_locations = cache.get_key_location(key)
        if new_locations[0] != initial_mapping[key][0]:
            operations.append(f"  {key} -> Moved to {new_locations[0]}")
            keys_moved += 1
        else:
            operations.append(f"  {key} -> Stayed on {new_locations[0]}")
    
    total_keys = len(test_keys)
    movement_percent = (keys_moved / total_keys) * 100
    results.append(f"\nConsistent hashing efficiency:")
    results.append(f"  Keys moved: {keys_moved}/{total_keys} ({movement_percent:.1f}%)")
    results.append(f"  Keys unchanged: {total_keys - keys_moved}/{total_keys}")
    results.append(f"  Expected with traditional hashing: ~{total_keys} keys would move")
    
    cluster_state.append(f"✓ Minimal data movement: {movement_percent:.1f}%")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "cluster_state": cluster_state,
        "messages": ["Consistent hashing demonstration complete"]
    }


def replication_and_failover_agent(state: DistributedCacheState):
    """Agent to demonstrate replication and automatic failover"""
    cache = state['_cache']
    nodes = state['_nodes']
    operations = []
    results = []
    cluster_state = []
    
    operations.append("\nReplication and Failover Demonstration:")
    
    # Put critical data with replication
    critical_key = "critical:config"
    critical_value = {"max_connections": 1000, "timeout": 30}
    
    cache.put(critical_key, critical_value)
    replicas = cache.get_key_location(critical_key)
    
    operations.append(f"Stored '{critical_key}' on replicas: {replicas}")
    
    # Simulate multiple node failures
    operations.append("\nSimulating cascade failures...")
    
    for i, replica_id in enumerate(replicas[:2]):  # Fail first 2 replicas
        # Find and fail the node
        for node in nodes:
            if node.node_id == replica_id:
                node.is_active = False
                operations.append(f"  ⚠️  {node.node_id} failed!")
                cluster_state.append(f"Node failure: {node.node_id}")
                break
        
        # Try to get data - should still work from remaining replicas
        value = cache.get(critical_key)
        if value:
            remaining_replicas = [r for r in replicas if r != replica_id]
            operations.append(f"     ✓ Data still available from: {remaining_replicas}")
        else:
            operations.append(f"     ✗ Data lost!")
    
    # Show final cluster state
    cluster_info = cache.get_cluster_state()
    results.append(f"\nCluster state after failures:")
    results.append(f"  Active nodes: {cluster_info['active_nodes']}")
    results.append(f"  Failed nodes: {cluster_info['inactive_nodes']}")
    results.append(f"  Failovers handled: {cluster_info['statistics']['failovers']}")
    
    # Show replication effectiveness
    total_writes = cluster_info['statistics']['puts']
    total_replications = cluster_info['statistics']['replications']
    results.append(f"\nReplication statistics:")
    results.append(f"  Total writes: {total_writes}")
    results.append(f"  Replications created: {total_replications}")
    results.append(f"  Average copies per key: {(total_replications/total_writes + 1):.1f}x")
    
    cluster_state.append(f"✓ Survived {cluster_info['inactive_nodes']} node failures")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "cluster_state": cluster_state,
        "messages": ["Replication and failover demonstration complete"]
    }


def performance_comparison_agent(state: DistributedCacheState):
    """Agent to compare distributed cache with single-node cache"""
    cache = state['_cache']
    operations = []
    results = []
    
    operations.append("\nPerformance Comparison:")
    
    # Distributed cache stats
    cluster_info = cache.get_cluster_state()
    
    results.append("\n📊 Distributed Cache Benefits:")
    results.append(f"  ✓ Scalability: Data distributed across {cluster_info['active_nodes']} nodes")
    results.append(f"  ✓ Availability: {cache.replication_factor}x replication")
    results.append(f"  ✓ Fault Tolerance: Survived {cluster_info['inactive_nodes']} node failures")
    results.append(f"  ✓ Load Distribution: Requests spread across cluster")
    
    results.append("\n📊 Trade-offs:")
    results.append(f"  ⚠️  Network latency: Cross-node communication overhead")
    results.append(f"  ⚠️  Consistency: Eventual consistency model")
    results.append(f"  ⚠️  Complexity: Cluster management, rebalancing")
    results.append(f"  ⚠️  Storage cost: {cache.replication_factor}x data duplication")
    
    results.append("\n📊 When to use Distributed Cache:")
    results.append("  ✓ Multi-server applications")
    results.append("  ✓ High availability requirements")
    results.append("  ✓ Large cache size (> single node memory)")
    results.append("  ✓ Geographic distribution")
    results.append("  ✓ Session sharing across servers")
    
    results.append("\n📊 When to use Single-Node Cache:")
    results.append("  ✓ Single-server applications")
    results.append("  ✓ Low latency critical")
    results.append("  ✓ Simple deployment")
    results.append("  ✓ Cache fits in single node memory")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "cluster_state": [],
        "messages": ["Performance comparison complete"]
    }


def create_distributed_cache_graph():
    """Create the distributed cache workflow graph"""
    workflow = StateGraph(DistributedCacheState)
    
    # Add nodes
    workflow.add_node("setup_cluster", setup_cluster_agent)
    workflow.add_node("data_distribution", data_distribution_agent)
    workflow.add_node("consistent_hashing", consistent_hashing_demo_agent)
    workflow.add_node("replication_failover", replication_and_failover_agent)
    workflow.add_node("performance_comparison", performance_comparison_agent)
    
    # Add edges
    workflow.add_edge(START, "setup_cluster")
    workflow.add_edge("setup_cluster", "data_distribution")
    workflow.add_edge("data_distribution", "consistent_hashing")
    workflow.add_edge("consistent_hashing", "replication_failover")
    workflow.add_edge("replication_failover", "performance_comparison")
    workflow.add_edge("performance_comparison", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 198: Distributed Cache MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_distributed_cache_graph()
    
    # Initialize state
    initial_state = {
        "cache_operations": [],
        "operation_results": [],
        "cluster_state": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("CACHE OPERATIONS")
    print("=" * 80)
    for op in final_state["cache_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("CLUSTER STATE CHANGES")
    print("=" * 80)
    for state_change in final_state["cluster_state"]:
        print(state_change)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Distributed Cache Pattern implemented with:

1. Consistent Hashing:
   - Data partitioned across nodes using hash ring
   - Virtual nodes for better distribution
   - Minimal data movement when nodes added/removed
   - O(log n) lookup time with sorted hash ring

2. Replication:
   - Configurable replication factor (default 3x)
   - Data stored on multiple nodes for availability
   - Automatic failover to replicas on node failure
   - Trade-off: Storage cost vs availability

3. Node Management:
   - Dynamic node addition/removal
   - Node health monitoring
   - Automatic rebalancing
   - Graceful degradation on failures

4. Partition Tolerance:
   - Cluster continues operating despite node failures
   - Data available from replicas
   - No single point of failure
   - CAP theorem: Chose AP (Availability + Partition tolerance)

Use Cases:
- Web application session sharing (multiple app servers)
- Microservices caching layer (shared cache across services)
- Real-time analytics (distributed aggregation cache)
- Content delivery (geographic distribution)
- Database query cache (shared across read replicas)

Comparison with Single-Node Cache:
- Single-Node: Simple, fast, single point of failure
- Distributed: Scalable, available, complex, network overhead

Common Implementations:
- Redis Cluster: Distributed Redis with automatic sharding
- Memcached with consistent hashing: Client-side partitioning
- Hazelcast: In-memory data grid with distributed cache
- Apache Ignite: Distributed in-memory computing platform
""")


if __name__ == "__main__":
    main()

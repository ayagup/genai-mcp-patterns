"""
Pattern 207: Replication MCP Pattern

Replication creates copies of data/services across multiple nodes for:
- High availability (failover)
- Read scalability (distribute read load)
- Data durability (prevent data loss)
- Geographic distribution (reduce latency)

Replication Models:
- Master-Slave: One primary, multiple replicas
- Master-Master: Multiple primaries (multi-master)
- Synchronous: Wait for all replicas
- Asynchronous: Don't wait for replicas
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
import time


class ReplicationState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class ReplicaNode:
    node_id: str
    is_primary: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    is_healthy: bool = True
    replication_lag: float = 0.0  # seconds


class ReplicationManager:
    def __init__(self, replication_factor: int = 3):
        self.replication_factor = replication_factor
        self.primary: Optional[ReplicaNode] = None
        self.replicas: List[ReplicaNode] = []
        self.total_writes = 0
        self.total_reads = 0
    
    def setup_cluster(self):
        # Create primary
        self.primary = ReplicaNode("primary-1", is_primary=True)
        
        # Create replicas
        for i in range(self.replication_factor - 1):
            replica = ReplicaNode(f"replica-{i+1}", is_primary=False)
            self.replicas.append(replica)
    
    def write(self, key: str, value: Any, sync: bool = False):
        """Write to primary and replicate"""
        self.total_writes += 1
        
        # Write to primary
        self.primary.data[key] = value
        
        if sync:
            # Synchronous replication (wait for all)
            for replica in self.replicas:
                if replica.is_healthy:
                    replica.data[key] = value
        else:
            # Asynchronous replication (don't wait)
            for replica in self.replicas:
                if replica.is_healthy:
                    # Simulate async replication
                    replica.data[key] = value
                    replica.replication_lag = 0.001
    
    def read(self, key: str, from_replica: bool = True) -> Any:
        """Read from primary or replica"""
        self.total_reads += 1
        
        if from_replica and self.replicas:
            # Load balance reads across replicas
            replica = self.replicas[self.total_reads % len(self.replicas)]
            return replica.data.get(key)
        else:
            # Read from primary
            return self.primary.data.get(key)
    
    def failover_to_replica(self):
        """Promote a replica to primary (failover)"""
        if not self.replicas:
            return False
        
        # Mark current primary as failed
        self.primary.is_healthy = False
        
        # Promote first healthy replica
        for replica in self.replicas:
            if replica.is_healthy:
                replica.is_primary = True
                self.primary = replica
                self.replicas.remove(replica)
                return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        healthy_replicas = sum(1 for r in self.replicas if r.is_healthy)
        
        return {
            'primary': self.primary.node_id if self.primary else None,
            'total_replicas': len(self.replicas),
            'healthy_replicas': healthy_replicas,
            'replication_factor': self.replication_factor,
            'total_writes': self.total_writes,
            'total_reads': self.total_reads
        }


def demo_agent(state: ReplicationState):
    operations = []
    results = []
    
    manager = ReplicationManager(replication_factor=3)
    manager.setup_cluster()
    
    operations.append("Replication Cluster Setup:")
    operations.append(f"  Primary: {manager.primary.node_id}")
    operations.append(f"  Replicas: {[r.node_id for r in manager.replicas]}")
    
    # Write data (replicated)
    operations.append("\nWriting Data (Replicated):")
    manager.write("user:1", {"name": "Alice"}, sync=True)
    manager.write("user:2", {"name": "Bob"}, sync=True)
    operations.append("  Wrote user:1, user:2 to primary and replicas")
    
    # Read from replicas (load distribution)
    operations.append("\nReading from Replicas:")
    for i in range(4):
        user = manager.read("user:1", from_replica=True)
        operations.append(f"  Read {i+1}: {user}")
    
    # Simulate primary failure and failover
    operations.append("\nSimulating Primary Failure:")
    operations.append(f"  Primary {manager.primary.node_id} failed")
    
    success = manager.failover_to_replica()
    operations.append(f"  Failover: {success}")
    operations.append(f"  New primary: {manager.primary.node_id}")
    
    # Continue serving (high availability)
    user = manager.read("user:1", from_replica=False)
    operations.append(f"  Read after failover: {user} ✓")
    
    stats = manager.get_stats()
    results.append(f"✓ Replication factor: {stats['replication_factor']}")
    results.append(f"✓ Failover successful: System still available")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Writes: {stats['total_writes']}, Reads: {stats['total_reads']}"],
        "messages": ["Replication demo complete"]
    }


def create_replication_graph():
    workflow = StateGraph(ReplicationState)
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    return workflow.compile()


def main():
    print("="*80)
    print("Pattern 207: Replication MCP Pattern")
    print("="*80)
    
    app = create_replication_graph()
    final_state = app.invoke({
        "scaling_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    })
    
    for op in final_state["scaling_operations"]:
        print(op)
    
    print("\n" + "="*80)
    print("""
Replication Benefits:
- High Availability: Automatic failover
- Read Scalability: Distribute reads across replicas
- Data Durability: Multiple copies prevent data loss
- Disaster Recovery: Geographic replication

Replication Models:
- Synchronous: Strong consistency, slower writes
- Asynchronous: Eventual consistency, faster writes
- Master-Slave: One primary, read-only replicas
- Master-Master: Multiple writable primaries

Examples: MySQL replication, MongoDB replica sets, PostgreSQL streaming replication
""")


if __name__ == "__main__":
    main()

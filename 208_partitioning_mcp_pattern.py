"""
Pattern 208: Partitioning MCP Pattern

Partitioning (Sharding) divides data across multiple nodes:
- Horizontal partitioning: Split rows across nodes
- Vertical partitioning: Split columns across nodes
- Hash-based: Partition by hash(key)
- Range-based: Partition by key ranges
- Geographic: Partition by location

Benefits:
- Scalability: Distribute load across nodes
- Performance: Parallel query processing
- Manageability: Smaller data sets per node
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
import hashlib


class PartitioningState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class Partition:
    partition_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def size(self) -> int:
        return len(self.data)


class PartitionManager:
    def __init__(self, num_partitions: int = 4):
        self.num_partitions = num_partitions
        self.partitions = [Partition(f"partition-{i}") for i in range(num_partitions)]
    
    def _hash_key(self, key: str) -> int:
        """Hash-based partitioning"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16) % self.num_partitions
    
    def write(self, key: str, value: Any):
        """Write to appropriate partition"""
        partition_idx = self._hash_key(key)
        self.partitions[partition_idx].data[key] = value
    
    def read(self, key: str) -> Optional[Any]:
        """Read from appropriate partition"""
        partition_idx = self._hash_key(key)
        return self.partitions[partition_idx].data.get(key)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'num_partitions': self.num_partitions,
            'partition_sizes': [p.size() for p in self.partitions],
            'total_keys': sum(p.size() for p in self.partitions)
        }


def demo_agent(state: PartitioningState):
    operations = []
    results = []
    
    manager = PartitionManager(num_partitions=4)
    
    operations.append("Partitioning Setup:")
    operations.append(f"  Number of partitions: {manager.num_partitions}")
    
    # Write data (automatically partitioned)
    keys = ["user:1", "user:2", "user:3", "user:4", "user:5", "user:6", "user:7", "user:8"]
    
    operations.append("\nWriting Data (Hash Partitioned):")
    for key in keys:
        value = {"name": f"User {key.split(':')[1]}"}
        manager.write(key, value)
        partition_idx = manager._hash_key(key)
        operations.append(f"  {key} → partition-{partition_idx}")
    
    stats = manager.get_stats()
    operations.append(f"\nPartition Distribution:")
    for i, size in enumerate(stats['partition_sizes']):
        operations.append(f"  partition-{i}: {size} keys")
    
    results.append(f"✓ Data partitioned across {stats['num_partitions']} partitions")
    results.append(f"✓ Total keys: {stats['total_keys']}")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Partitions: {stats['num_partitions']}"],
        "messages": ["Partitioning demo complete"]
    }


def create_partitioning_graph():
    workflow = StateGraph(PartitioningState)
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    return workflow.compile()


def main():
    print("="*80)
    print("Pattern 208: Partitioning MCP Pattern")
    print("="*80)
    
    app = create_partitioning_graph()
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
Partitioning Strategies:
- Hash-based: Even distribution, simple
- Range-based: Sequential access, hotspots possible
- List-based: Explicit mapping
- Composite: Combination of above

Benefits:
- Scalability: Add more partitions
- Performance: Parallel processing
- Manageability: Smaller datasets

Examples: MongoDB sharding, Cassandra partitions, PostgreSQL table partitioning
""")


if __name__ == "__main__":
    main()

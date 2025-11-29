"""
Pattern 288: Distributed State MCP Pattern

This pattern demonstrates distributed state management across multiple nodes
with synchronization, replication, and consistency mechanisms.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import hashlib


class DistributedStatePattern(TypedDict):
    """State for distributed state management"""
    messages: Annotated[List[str], add]
    nodes: Dict[str, Dict[str, Any]]
    replicated_data: Dict[str, Any]
    sync_log: List[Dict[str, Any]]
    consistency_metrics: Dict[str, Any]


class DistributedNode:
    """Represents a node in distributed system"""
    
    def __init__(self, node_id: str, region: str):
        self.node_id = node_id
        self.region = region
        self.local_state = {}
        self.version_vector = {}
        self.last_sync = time.time()
    
    def update_local(self, key: str, value: Any):
        """Update local state"""
        self.local_state[key] = {
            "value": value,
            "version": self.version_vector.get(key, 0) + 1,
            "timestamp": time.time(),
            "node_id": self.node_id
        }
        self.version_vector[key] = self.local_state[key]["version"]
    
    def get_local(self, key: str):
        """Get value from local state"""
        if key in self.local_state:
            return self.local_state[key]["value"]
        return None
    
    def get_state_snapshot(self):
        """Get snapshot of current state"""
        return {
            "node_id": self.node_id,
            "region": self.region,
            "data": {k: v["value"] for k, v in self.local_state.items()},
            "versions": self.version_vector.copy(),
            "last_sync": self.last_sync
        }


class DistributedStateManager:
    """Manages distributed state across nodes"""
    
    def __init__(self):
        self.nodes = {}
        self.sync_log = []
        self.replication_factor = 3
    
    def add_node(self, node_id: str, region: str):
        """Add node to cluster"""
        self.nodes[node_id] = DistributedNode(node_id, region)
    
    def replicate(self, key: str, value: Any, source_node: str):
        """Replicate data across nodes"""
        if source_node not in self.nodes:
            return
        
        # Update source node
        self.nodes[source_node].update_local(key, value)
        
        # Replicate to other nodes
        sync_operations = []
        for node_id, node in self.nodes.items():
            if node_id != source_node:
                node.update_local(key, value)
                sync_operations.append({
                    "from": source_node,
                    "to": node_id,
                    "key": key,
                    "timestamp": time.time()
                })
        
        self.sync_log.extend(sync_operations)
        return sync_operations
    
    def sync_nodes(self):
        """Synchronize state across nodes"""
        sync_operations = []
        
        # Collect all unique keys
        all_keys = set()
        for node in self.nodes.values():
            all_keys.update(node.local_state.keys())
        
        # For each key, find latest version
        for key in all_keys:
            latest_version = 0
            latest_node = None
            latest_value = None
            
            for node in self.nodes.values():
                if key in node.version_vector:
                    if node.version_vector[key] > latest_version:
                        latest_version = node.version_vector[key]
                        latest_node = node.node_id
                        latest_value = node.get_local(key)
            
            # Propagate latest version to all nodes
            if latest_node:
                for node in self.nodes.values():
                    current_version = node.version_vector.get(key, 0)
                    if current_version < latest_version:
                        node.update_local(key, latest_value)
                        node.last_sync = time.time()
                        sync_operations.append({
                            "key": key,
                            "from": latest_node,
                            "to": node.node_id,
                            "version": latest_version
                        })
        
        self.sync_log.extend(sync_operations)
        return sync_operations
    
    def check_consistency(self):
        """Check consistency across nodes"""
        all_keys = set()
        for node in self.nodes.values():
            all_keys.update(node.local_state.keys())
        
        consistency_report = {
            "total_keys": len(all_keys),
            "consistent_keys": 0,
            "inconsistent_keys": 0,
            "replication_coverage": {}
        }
        
        for key in all_keys:
            values = []
            for node in self.nodes.values():
                value = node.get_local(key)
                if value is not None:
                    values.append(value)
            
            # Check if all values are the same
            if len(set(str(v) for v in values)) == 1:
                consistency_report["consistent_keys"] += 1
            else:
                consistency_report["inconsistent_keys"] += 1
            
            consistency_report["replication_coverage"][key] = len(values)
        
        return consistency_report


def initialize_distributed_state_agent(state: DistributedStatePattern) -> DistributedStatePattern:
    """Initialize distributed state"""
    print("\n🌐 Initializing Distributed State...")
    
    manager = DistributedStateManager()
    
    # Add nodes in different regions
    nodes_config = [
        ("node_us_east", "us-east-1"),
        ("node_us_west", "us-west-2"),
        ("node_eu_central", "eu-central-1"),
        ("node_ap_southeast", "ap-southeast-1")
    ]
    
    print(f"  Cluster Configuration:")
    for node_id, region in nodes_config:
        manager.add_node(node_id, region)
        print(f"    • {node_id} ({region})")
    
    print(f"\n  Total Nodes: {len(manager.nodes)}")
    print(f"  Replication Factor: {manager.replication_factor}")
    print(f"  Features:")
    print(f"    • Multi-region deployment")
    print(f"    • Automatic replication")
    print(f"    • Version vectors")
    print(f"    • Consistency checks")
    
    nodes_state = {node_id: node.get_state_snapshot() 
                   for node_id, node in manager.nodes.items()}
    
    return {
        **state,
        "nodes": nodes_state,
        "replicated_data": {},
        "sync_log": [],
        "consistency_metrics": {},
        "messages": [f"✓ Initialized {len(manager.nodes)} nodes"]
    }


def replicate_data_agent(state: DistributedStatePattern) -> DistributedStatePattern:
    """Replicate data across nodes"""
    print("\n🔄 Replicating Data Across Nodes...")
    
    manager = DistributedStateManager()
    
    # Recreate nodes from state
    for node_id, node_data in state["nodes"].items():
        manager.add_node(node_id, node_data["region"])
    
    # Replicate various data items
    replications = [
        ("user_profile_123", {"name": "Alice", "email": "alice@example.com"}, "node_us_east"),
        ("product_catalog", {"total": 1000, "categories": 50}, "node_us_west"),
        ("session_state", {"active": 250, "peak": 500}, "node_eu_central"),
        ("cache_config", {"ttl": 300, "max_size": 10000}, "node_ap_southeast")
    ]
    
    all_replicated = {}
    
    for key, value, source_node in replications:
        sync_ops = manager.replicate(key, value, source_node)
        all_replicated[key] = value
        
        print(f"\n  Replicated: {key}")
        print(f"    Source: {source_node}")
        print(f"    Target Nodes: {len(sync_ops) if sync_ops else 0}")
        print(f"    Data: {value}")
    
    print(f"\n  Total Replications: {len(manager.sync_log)}")
    
    return {
        **state,
        "replicated_data": all_replicated,
        "sync_log": manager.sync_log.copy(),
        "messages": [f"✓ Replicated {len(replications)} items"]
    }


def synchronize_nodes_agent(state: DistributedStatePattern) -> DistributedStatePattern:
    """Synchronize state across nodes"""
    print("\n⚡ Synchronizing Nodes...")
    
    manager = DistributedStateManager()
    
    # Recreate nodes
    for node_id, node_data in state["nodes"].items():
        manager.add_node(node_id, node_data["region"])
    
    # Restore replicated data
    for key, value in state["replicated_data"].items():
        manager.nodes["node_us_east"].update_local(key, value)
    
    # Perform synchronization
    sync_operations = manager.sync_nodes()
    
    print(f"  Synchronization Operations: {len(sync_operations)}")
    
    # Show sync details
    if sync_operations:
        print(f"\n  Recent Sync Operations:")
        for op in sync_operations[:5]:
            print(f"    {op['from']} → {op['to']}: {op['key']} (v{op['version']})")
    
    # Update nodes state
    nodes_state = {node_id: node.get_state_snapshot() 
                   for node_id, node in manager.nodes.items()}
    
    return {
        **state,
        "nodes": nodes_state,
        "sync_log": state["sync_log"] + sync_operations,
        "messages": [f"✓ Synchronized {len(sync_operations)} operations"]
    }


def check_consistency_agent(state: DistributedStatePattern) -> DistributedStatePattern:
    """Check consistency across distributed nodes"""
    print("\n✅ Checking Consistency...")
    
    manager = DistributedStateManager()
    
    # Recreate nodes with their data
    for node_id, node_data in state["nodes"].items():
        manager.add_node(node_id, node_data["region"])
        for key, value in node_data.get("data", {}).items():
            manager.nodes[node_id].update_local(key, value)
    
    # Check consistency
    consistency = manager.check_consistency()
    
    print(f"  Total Keys: {consistency['total_keys']}")
    print(f"  Consistent Keys: {consistency['consistent_keys']}")
    print(f"  Inconsistent Keys: {consistency['inconsistent_keys']}")
    
    if consistency['total_keys'] > 0:
        consistency_rate = consistency['consistent_keys'] / consistency['total_keys']
        print(f"  Consistency Rate: {consistency_rate:.1%}")
    
    print(f"\n  Replication Coverage:")
    for key, count in consistency['replication_coverage'].items():
        coverage = (count / len(manager.nodes)) * 100
        print(f"    {key}: {count}/{len(manager.nodes)} nodes ({coverage:.0f}%)")
    
    return {
        **state,
        "consistency_metrics": consistency,
        "messages": ["✓ Consistency check complete"]
    }


def generate_distributed_state_report_agent(state: DistributedStatePattern) -> DistributedStatePattern:
    """Generate distributed state report"""
    print("\n" + "="*70)
    print("DISTRIBUTED STATE REPORT")
    print("="*70)
    
    print(f"\n🌐 Cluster Overview:")
    print(f"  Total Nodes: {len(state['nodes'])}")
    print(f"  Replicated Keys: {len(state['replicated_data'])}")
    print(f"  Sync Operations: {len(state['sync_log'])}")
    
    print(f"\n🖥️ Node Status:")
    for node_id, node_data in state['nodes'].items():
        print(f"  • {node_id}:")
        print(f"      Region: {node_data['region']}")
        print(f"      Data Items: {len(node_data.get('data', {}))}")
        print(f"      Version Vector: {node_data.get('versions', {})}")
    
    print(f"\n📊 Replicated Data:")
    for key, value in state['replicated_data'].items():
        print(f"  • {key}: {value}")
    
    print(f"\n🔄 Synchronization Log:")
    recent_syncs = state['sync_log'][-10:] if len(state['sync_log']) > 10 else state['sync_log']
    print(f"  Recent Operations: {len(recent_syncs)} (of {len(state['sync_log'])} total)")
    for op in recent_syncs[:5]:
        print(f"    {op.get('from', 'N/A')} → {op.get('to', 'N/A')}: {op.get('key', 'N/A')}")
    
    print(f"\n✅ Consistency Metrics:")
    metrics = state['consistency_metrics']
    if metrics:
        print(f"  Total Keys: {metrics.get('total_keys', 0)}")
        print(f"  Consistent: {metrics.get('consistent_keys', 0)}")
        print(f"  Inconsistent: {metrics.get('inconsistent_keys', 0)}")
        
        if metrics.get('total_keys', 0) > 0:
            rate = metrics['consistent_keys'] / metrics['total_keys']
            print(f"  Consistency Rate: {rate:.1%}")
    
    print(f"\n💡 Distributed State Benefits:")
    print("  ✓ High availability")
    print("  ✓ Geographic distribution")
    print("  ✓ Fault tolerance")
    print("  ✓ Load distribution")
    print("  ✓ Disaster recovery")
    print("  ✓ Low latency (regional)")
    
    print(f"\n🔧 Replication Strategy:")
    print("  • Multi-master replication")
    print("  • Version vectors")
    print("  • Eventual consistency")
    print("  • Conflict resolution")
    print("  • Automatic sync")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Global applications")
    print("  • CDN services")
    print("  • Distributed databases")
    print("  • Multi-region deployments")
    print("  • High-availability systems")
    print("  • Geo-replicated caches")
    
    print(f"\n🎯 Consistency Models:")
    print("  • Strong consistency")
    print("  • Eventual consistency")
    print("  • Causal consistency")
    print("  • Session consistency")
    print("  • Read-your-writes")
    
    print(f"\n⚠️ Challenges:")
    print("  • Network partitions")
    print("  • Conflict resolution")
    print("  • Synchronization overhead")
    print("  • Latency variations")
    print("  • Data consistency")
    
    print("\n" + "="*70)
    print("✅ Distributed State Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_distributed_state_graph():
    """Create distributed state workflow"""
    workflow = StateGraph(DistributedStatePattern)
    
    workflow.add_node("initialize", initialize_distributed_state_agent)
    workflow.add_node("replicate", replicate_data_agent)
    workflow.add_node("synchronize", synchronize_nodes_agent)
    workflow.add_node("consistency", check_consistency_agent)
    workflow.add_node("report", generate_distributed_state_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "replicate")
    workflow.add_edge("replicate", "synchronize")
    workflow.add_edge("synchronize", "consistency")
    workflow.add_edge("consistency", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 288: Distributed State MCP Pattern")
    print("="*70)
    
    app = create_distributed_state_graph()
    final_state = app.invoke({
        "messages": [],
        "nodes": {},
        "replicated_data": {},
        "sync_log": [],
        "consistency_metrics": {}
    })
    
    print("\n✅ Distributed State Pattern Complete!")


if __name__ == "__main__":
    main()

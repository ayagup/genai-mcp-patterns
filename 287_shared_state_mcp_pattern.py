"""
Pattern 287: Shared State MCP Pattern

This pattern demonstrates shared state management across multiple agents,
processes, or services with synchronization mechanisms.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import threading


class SharedStatePattern(TypedDict):
    """State for shared state management"""
    messages: Annotated[List[str], add]
    shared_data: Dict[str, Any]
    access_log: List[Dict[str, Any]]
    lock_info: Dict[str, Any]
    concurrent_updates: List[Dict[str, Any]]


class SharedStateManager:
    """Manages shared state with synchronization"""
    
    def __init__(self):
        self.shared_data = {}
        self.access_log = []
        self.locks = {}
        self.version = 0
        self._lock = threading.Lock()
    
    def read(self, key: str, agent_id: str):
        """Thread-safe read operation"""
        with self._lock:
            value = self.shared_data.get(key)
            self.access_log.append({
                "operation": "read",
                "key": key,
                "agent_id": agent_id,
                "timestamp": time.time(),
                "version": self.version
            })
            return value
    
    def write(self, key: str, value: Any, agent_id: str):
        """Thread-safe write operation"""
        with self._lock:
            old_value = self.shared_data.get(key)
            self.shared_data[key] = value
            self.version += 1
            
            self.access_log.append({
                "operation": "write",
                "key": key,
                "agent_id": agent_id,
                "old_value": old_value,
                "new_value": value,
                "timestamp": time.time(),
                "version": self.version
            })
    
    def acquire_lock(self, resource: str, agent_id: str) -> bool:
        """Acquire lock on shared resource"""
        with self._lock:
            if resource not in self.locks or not self.locks[resource]["locked"]:
                self.locks[resource] = {
                    "locked": True,
                    "owner": agent_id,
                    "acquired_at": time.time()
                }
                return True
            return False
    
    def release_lock(self, resource: str, agent_id: str) -> bool:
        """Release lock on shared resource"""
        with self._lock:
            if resource in self.locks and self.locks[resource]["owner"] == agent_id:
                self.locks[resource]["locked"] = False
                self.locks[resource]["released_at"] = time.time()
                return True
            return False
    
    def get_concurrent_state(self):
        """Get current state for concurrent access"""
        with self._lock:
            return {
                "data": self.shared_data.copy(),
                "version": self.version,
                "active_locks": {k: v for k, v in self.locks.items() if v.get("locked", False)}
            }


def initialize_shared_state_agent(state: SharedStatePattern) -> SharedStatePattern:
    """Initialize shared state"""
    print("\n🔗 Initializing Shared State...")
    
    manager = SharedStateManager()
    
    # Initialize shared data
    initial_data = {
        "counter": 0,
        "inventory": {"product_a": 100, "product_b": 50},
        "user_sessions": [],
        "system_config": {"max_connections": 100, "timeout": 30}
    }
    
    for key, value in initial_data.items():
        manager.write(key, value, "system")
    
    print(f"  Shared State Initialized")
    print(f"  Initial Version: {manager.version}")
    print(f"  Data Items: {len(manager.shared_data)}")
    print(f"  Features:")
    print(f"    • Thread-safe operations")
    print(f"    • Lock mechanisms")
    print(f"    • Version tracking")
    print(f"    • Access logging")
    
    return {
        **state,
        "shared_data": manager.shared_data.copy(),
        "access_log": manager.access_log.copy(),
        "lock_info": {},
        "concurrent_updates": [],
        "messages": ["✓ Shared state initialized"]
    }


def simulate_concurrent_access_agent(state: SharedStatePattern) -> SharedStatePattern:
    """Simulate concurrent access from multiple agents"""
    print("\n👥 Simulating Concurrent Access...")
    
    manager = SharedStateManager()
    manager.shared_data = state["shared_data"].copy()
    manager.version = len(state["access_log"])
    
    # Simulate multiple agents accessing shared state
    agents = ["agent_1", "agent_2", "agent_3"]
    
    print(f"  Active Agents: {len(agents)}")
    
    for agent_id in agents:
        # Read current counter
        counter = manager.read("counter", agent_id)
        print(f"\n  {agent_id}:")
        print(f"    Read counter: {counter}")
        
        # Simulate processing
        time.sleep(0.01)
        
        # Update counter
        new_counter = (counter or 0) + 1
        manager.write("counter", new_counter, agent_id)
        print(f"    Updated counter: {new_counter}")
        
        # Update inventory
        inventory = manager.read("inventory", agent_id)
        if inventory and "product_a" in inventory:
            inventory["product_a"] -= 5
            manager.write("inventory", inventory, agent_id)
            print(f"    Updated inventory: product_a = {inventory['product_a']}")
    
    print(f"\n  Total Operations: {len(manager.access_log)}")
    print(f"  Current Version: {manager.version}")
    
    return {
        **state,
        "shared_data": manager.shared_data.copy(),
        "access_log": manager.access_log.copy(),
        "messages": [f"✓ Simulated {len(agents)} concurrent agents"]
    }


def demonstrate_locking_agent(state: SharedStatePattern) -> SharedStatePattern:
    """Demonstrate resource locking"""
    print("\n🔒 Demonstrating Resource Locking...")
    
    manager = SharedStateManager()
    manager.shared_data = state["shared_data"].copy()
    
    # Demonstrate lock acquisition and release
    lock_scenarios = [
        ("database_connection", "agent_1"),
        ("file_writer", "agent_2"),
        ("cache_update", "agent_3")
    ]
    
    for resource, agent_id in lock_scenarios:
        # Acquire lock
        acquired = manager.acquire_lock(resource, agent_id)
        print(f"\n  {agent_id} acquiring lock on {resource}:")
        print(f"    Status: {'✓ Acquired' if acquired else '✗ Failed'}")
        
        if acquired:
            # Simulate work with locked resource
            time.sleep(0.01)
            
            # Release lock
            released = manager.release_lock(resource, agent_id)
            print(f"    Released: {'✓ Yes' if released else '✗ No'}")
    
    # Show active locks
    state_snapshot = manager.get_concurrent_state()
    print(f"\n  Active Locks: {len(state_snapshot['active_locks'])}")
    
    return {
        **state,
        "lock_info": manager.locks.copy(),
        "messages": [f"✓ Demonstrated {len(lock_scenarios)} lock scenarios"]
    }


def track_concurrent_updates_agent(state: SharedStatePattern) -> SharedStatePattern:
    """Track concurrent updates"""
    print("\n📝 Tracking Concurrent Updates...")
    
    access_log = state["access_log"]
    
    # Analyze access patterns
    reads = [log for log in access_log if log["operation"] == "read"]
    writes = [log for log in access_log if log["operation"] == "write"]
    
    print(f"  Total Read Operations: {len(reads)}")
    print(f"  Total Write Operations: {len(writes)}")
    
    # Group by agent
    agent_stats = {}
    for log in access_log:
        agent_id = log["agent_id"]
        if agent_id not in agent_stats:
            agent_stats[agent_id] = {"reads": 0, "writes": 0}
        
        if log["operation"] == "read":
            agent_stats[agent_id]["reads"] += 1
        elif log["operation"] == "write":
            agent_stats[agent_id]["writes"] += 1
    
    print(f"\n  Agent Statistics:")
    for agent_id, stats in agent_stats.items():
        print(f"    {agent_id}:")
        print(f"      Reads: {stats['reads']}")
        print(f"      Writes: {stats['writes']}")
        print(f"      Total: {stats['reads'] + stats['writes']}")
    
    # Identify concurrent updates (writes within 100ms)
    concurrent_updates = []
    for i, write in enumerate(writes):
        for other_write in writes[i+1:]:
            time_diff = abs(write["timestamp"] - other_write["timestamp"])
            if time_diff < 0.1:  # Within 100ms
                concurrent_updates.append({
                    "agent1": write["agent_id"],
                    "agent2": other_write["agent_id"],
                    "time_diff_ms": time_diff * 1000,
                    "key": write["key"]
                })
    
    if concurrent_updates:
        print(f"\n  Concurrent Updates Detected: {len(concurrent_updates)}")
        for update in concurrent_updates[:3]:  # Show first 3
            print(f"    {update['agent1']} & {update['agent2']}:")
            print(f"      Time diff: {update['time_diff_ms']:.2f}ms")
            print(f"      Resource: {update['key']}")
    
    return {
        **state,
        "concurrent_updates": concurrent_updates,
        "messages": ["✓ Concurrent updates tracked"]
    }


def generate_shared_state_report_agent(state: SharedStatePattern) -> SharedStatePattern:
    """Generate shared state report"""
    print("\n" + "="*70)
    print("SHARED STATE REPORT")
    print("="*70)
    
    print(f"\n🔗 Shared State Overview:")
    print(f"  Data Items: {len(state['shared_data'])}")
    print(f"  Total Operations: {len(state['access_log'])}")
    print(f"  Synchronization: Enabled")
    
    print(f"\n📊 Shared Data:")
    for key, value in state['shared_data'].items():
        print(f"  • {key}: {value}")
    
    print(f"\n📝 Access Log Summary:")
    reads = [log for log in state['access_log'] if log['operation'] == 'read']
    writes = [log for log in state['access_log'] if log['operation'] == 'write']
    print(f"  Read Operations: {len(reads)}")
    print(f"  Write Operations: {len(writes)}")
    print(f"  Total Operations: {len(state['access_log'])}")
    
    print(f"\n🔒 Lock Information:")
    if state['lock_info']:
        for resource, lock_data in state['lock_info'].items():
            status = "🔒 Locked" if lock_data.get('locked') else "🔓 Unlocked"
            print(f"  {resource}: {status}")
            if lock_data.get('owner'):
                print(f"    Owner: {lock_data['owner']}")
    else:
        print("  No locks active")
    
    print(f"\n⚡ Concurrent Updates:")
    if state['concurrent_updates']:
        print(f"  Detected: {len(state['concurrent_updates'])}")
        for update in state['concurrent_updates'][:5]:
            print(f"  • Agents: {update['agent1']} & {update['agent2']}")
            print(f"    Resource: {update['key']}")
            print(f"    Time diff: {update['time_diff_ms']:.2f}ms")
    else:
        print("  No concurrent updates detected")
    
    print(f"\n💡 Shared State Benefits:")
    print("  ✓ Multi-agent coordination")
    print("  ✓ Thread-safe operations")
    print("  ✓ Resource synchronization")
    print("  ✓ Consistency guarantees")
    print("  ✓ Conflict prevention")
    print("  ✓ Access tracking")
    
    print(f"\n🔧 Synchronization Mechanisms:")
    print("  • Thread locks")
    print("  • Version control")
    print("  • Access logging")
    print("  • Resource locking")
    print("  • Atomic operations")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Multi-agent systems")
    print("  • Distributed processing")
    print("  • Shared resources")
    print("  • Concurrent workflows")
    print("  • Collaborative editing")
    print("  • Database connections")
    
    print(f"\n🎯 Concurrency Patterns:")
    print("  • Read-write locks")
    print("  • Optimistic locking")
    print("  • Pessimistic locking")
    print("  • Version-based concurrency")
    print("  • Lock-free algorithms")
    
    print("\n" + "="*70)
    print("✅ Shared State Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_shared_state_graph():
    """Create shared state workflow"""
    workflow = StateGraph(SharedStatePattern)
    
    workflow.add_node("initialize", initialize_shared_state_agent)
    workflow.add_node("concurrent", simulate_concurrent_access_agent)
    workflow.add_node("locking", demonstrate_locking_agent)
    workflow.add_node("tracking", track_concurrent_updates_agent)
    workflow.add_node("report", generate_shared_state_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "concurrent")
    workflow.add_edge("concurrent", "locking")
    workflow.add_edge("locking", "tracking")
    workflow.add_edge("tracking", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 287: Shared State MCP Pattern")
    print("="*70)
    
    app = create_shared_state_graph()
    final_state = app.invoke({
        "messages": [],
        "shared_data": {},
        "access_log": [],
        "lock_info": {},
        "concurrent_updates": []
    })
    
    print("\n✅ Shared State Pattern Complete!")


if __name__ == "__main__":
    main()

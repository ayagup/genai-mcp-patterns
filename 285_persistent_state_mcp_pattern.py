"""
Pattern 285: Persistent State MCP Pattern

This pattern demonstrates persistent state management using storage
backends to maintain state across restarts and failures.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json
import time


class PersistentStatePattern(TypedDict):
    """State for persistent storage management"""
    messages: Annotated[List[str], add]
    storage_backend: str
    persisted_data: Dict[str, Any]
    snapshots: List[Dict[str, Any]]
    recovery_points: List[Dict[str, Any]]


class PersistentStorage:
    """Manages persistent state storage"""
    
    def __init__(self, storage_type: str = "file"):
        self.storage_type = storage_type
        self.data = {}
        self.snapshots = []
        self.transaction_log = []
    
    def save(self, key: str, value: Any):
        """Save data persistently"""
        timestamp = time.time()
        self.data[key] = {
            "value": value,
            "saved_at": timestamp,
            "version": len(self.transaction_log) + 1
        }
        
        # Log transaction
        self.transaction_log.append({
            "operation": "save",
            "key": key,
            "timestamp": timestamp,
            "version": len(self.transaction_log) + 1
        })
        
        # Simulate persistence
        if self.storage_type == "file":
            self._persist_to_file(key, value)
        elif self.storage_type == "database":
            self._persist_to_database(key, value)
    
    def load(self, key: str):
        """Load data from persistent storage"""
        if key in self.data:
            return self.data[key]["value"]
        return None
    
    def create_snapshot(self):
        """Create snapshot of current state"""
        snapshot = {
            "timestamp": time.time(),
            "data": {k: v["value"] for k, v in self.data.items()},
            "version": len(self.snapshots) + 1
        }
        self.snapshots.append(snapshot)
        return snapshot
    
    def restore_snapshot(self, version: int):
        """Restore from snapshot"""
        if 0 < version <= len(self.snapshots):
            snapshot = self.snapshots[version - 1]
            for key, value in snapshot["data"].items():
                self.save(key, value)
            return snapshot
        return None
    
    def _persist_to_file(self, key: str, value: Any):
        """Simulate file persistence"""
        # In production: write to actual file
        pass
    
    def _persist_to_database(self, key: str, value: Any):
        """Simulate database persistence"""
        # In production: write to actual database
        pass
    
    def get_transaction_log(self):
        """Get transaction history"""
        return self.transaction_log


def initialize_persistent_storage_agent(state: PersistentStatePattern) -> PersistentStatePattern:
    """Initialize persistent storage"""
    print("\n💾 Initializing Persistent Storage...")
    
    storage = PersistentStorage(storage_type="file")
    
    print(f"  Storage Type: {storage.storage_type}")
    print(f"  Status: Initialized")
    print(f"  Features:")
    print(f"    • Data persistence")
    print(f"    • Transaction logging")
    print(f"    • Snapshot support")
    print(f"    • Version control")
    
    return {
        **state,
        "storage_backend": storage.storage_type,
        "persisted_data": {},
        "snapshots": [],
        "recovery_points": [],
        "messages": ["✓ Persistent storage initialized"]
    }


def persist_data_agent(state: PersistentStatePattern) -> PersistentStatePattern:
    """Persist data to storage"""
    print("\n📝 Persisting Data...")
    
    storage = PersistentStorage(storage_type=state["storage_backend"])
    
    # Save various types of data
    data_items = [
        ("user_preferences", {
            "theme": "dark",
            "language": "en",
            "notifications": True
        }),
        ("application_state", {
            "version": "1.2.3",
            "last_update": "2024-01-15",
            "features_enabled": ["ai", "search", "analytics"]
        }),
        ("session_data", {
            "active_sessions": 156,
            "peak_sessions": 200,
            "avg_duration": 1800
        }),
        ("cache_metadata", {
            "total_items": 1000,
            "hit_rate": 0.85,
            "evictions": 50
        })
    ]
    
    for key, value in data_items:
        storage.save(key, value)
        print(f"  ✓ Saved: {key}")
        print(f"    Version: {storage.data[key]['version']}")
    
    print(f"\n  Total Items Persisted: {len(storage.data)}")
    print(f"  Transaction Log Entries: {len(storage.transaction_log)}")
    
    persisted_data = {k: v["value"] for k, v in storage.data.items()}
    
    return {
        **state,
        "persisted_data": persisted_data,
        "messages": [f"✓ Persisted {len(data_items)} items"]
    }


def create_snapshots_agent(state: PersistentStatePattern) -> PersistentStatePattern:
    """Create recovery snapshots"""
    print("\n📸 Creating Snapshots...")
    
    storage = PersistentStorage()
    
    # Restore persisted data
    for key, value in state["persisted_data"].items():
        storage.save(key, value)
    
    # Create multiple snapshots
    snapshots = []
    snapshot_count = 3
    
    for i in range(snapshot_count):
        # Modify data between snapshots
        storage.save(f"snapshot_marker_{i}", f"checkpoint_{i}")
        snapshot = storage.create_snapshot()
        snapshots.append(snapshot)
        
        print(f"  Snapshot #{snapshot['version']} created")
        print(f"    Timestamp: {snapshot['timestamp']:.2f}")
        print(f"    Data items: {len(snapshot['data'])}")
        
        time.sleep(0.1)  # Simulate time passing
    
    print(f"\n  Total Snapshots: {len(snapshots)}")
    
    return {
        **state,
        "snapshots": snapshots,
        "messages": [f"✓ Created {snapshot_count} snapshots"]
    }


def demonstrate_recovery_agent(state: PersistentStatePattern) -> PersistentStatePattern:
    """Demonstrate recovery from snapshots"""
    print("\n🔄 Demonstrating Recovery...")
    
    storage = PersistentStorage()
    
    # Restore data and snapshots
    for key, value in state["persisted_data"].items():
        storage.save(key, value)
    
    storage.snapshots = state["snapshots"]
    
    print(f"  Available snapshots: {len(storage.snapshots)}")
    
    # Simulate recovery scenario
    recovery_points = []
    
    if len(storage.snapshots) > 0:
        # Restore from snapshot 1
        snapshot_version = 1
        restored = storage.restore_snapshot(snapshot_version)
        
        if restored:
            print(f"\n  ✓ Restored from snapshot #{snapshot_version}")
            print(f"    Restored {len(restored['data'])} items")
            print(f"    Recovery timestamp: {restored['timestamp']:.2f}")
            
            recovery_points.append({
                "version": snapshot_version,
                "timestamp": restored["timestamp"],
                "items": len(restored["data"])
            })
    
    # Show transaction log
    print(f"\n  Transaction Log:")
    for i, txn in enumerate(storage.transaction_log[-5:], 1):
        print(f"    {i}. {txn['operation']} - {txn.get('key', 'N/A')} (v{txn['version']})")
    
    return {
        **state,
        "recovery_points": recovery_points,
        "messages": ["✓ Recovery demonstration complete"]
    }


def generate_persistent_state_report_agent(state: PersistentStatePattern) -> PersistentStatePattern:
    """Generate persistent state report"""
    print("\n" + "="*70)
    print("PERSISTENT STATE REPORT")
    print("="*70)
    
    print(f"\n💾 Storage Configuration:")
    print(f"  Backend: {state['storage_backend']}")
    print(f"  Status: Active")
    print(f"  Mode: Persistent")
    
    print(f"\n📊 Persisted Data:")
    print(f"  Total Items: {len(state['persisted_data'])}")
    for key, value in state['persisted_data'].items():
        print(f"  • {key}:")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"      {value}")
    
    print(f"\n📸 Snapshots:")
    print(f"  Total Snapshots: {len(state['snapshots'])}")
    for snapshot in state['snapshots']:
        print(f"  • Version {snapshot['version']}:")
        print(f"      Timestamp: {snapshot['timestamp']:.2f}")
        print(f"      Items: {len(snapshot['data'])}")
    
    print(f"\n🔄 Recovery Points:")
    if state['recovery_points']:
        for rp in state['recovery_points']:
            print(f"  • Version {rp['version']}:")
            print(f"      Timestamp: {rp['timestamp']:.2f}")
            print(f"      Items Restored: {rp['items']}")
    else:
        print("  No recovery performed")
    
    print(f"\n💡 Persistent State Benefits:")
    print("  ✓ Survives restarts")
    print("  ✓ Disaster recovery")
    print("  ✓ Data durability")
    print("  ✓ Version control")
    print("  ✓ Point-in-time recovery")
    print("  ✓ Transaction logging")
    
    print(f"\n🔧 Storage Features:")
    print("  • Automatic persistence")
    print("  • Snapshot creation")
    print("  • Transaction logging")
    print("  • Version tracking")
    print("  • Data recovery")
    print("  • Multiple backends")
    
    print(f"\n⚙️ Use Cases:")
    print("  • User data storage")
    print("  • Application state")
    print("  • Configuration management")
    print("  • Session persistence")
    print("  • Audit trails")
    print("  • Disaster recovery")
    
    print(f"\n🎯 Storage Backends:")
    print("  • File system")
    print("  • Database (SQL/NoSQL)")
    print("  • Object storage (S3)")
    print("  • Key-value stores (Redis)")
    print("  • Document stores (MongoDB)")
    
    print("\n" + "="*70)
    print("✅ Persistent State Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_persistent_state_graph():
    """Create persistent state workflow"""
    workflow = StateGraph(PersistentStatePattern)
    
    workflow.add_node("initialize", initialize_persistent_storage_agent)
    workflow.add_node("persist", persist_data_agent)
    workflow.add_node("snapshot", create_snapshots_agent)
    workflow.add_node("recover", demonstrate_recovery_agent)
    workflow.add_node("report", generate_persistent_state_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "persist")
    workflow.add_edge("persist", "snapshot")
    workflow.add_edge("snapshot", "recover")
    workflow.add_edge("recover", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 285: Persistent State MCP Pattern")
    print("="*70)
    
    app = create_persistent_state_graph()
    final_state = app.invoke({
        "messages": [],
        "storage_backend": "",
        "persisted_data": {},
        "snapshots": [],
        "recovery_points": []
    })
    
    print("\n✅ Persistent State Pattern Complete!")


if __name__ == "__main__":
    main()

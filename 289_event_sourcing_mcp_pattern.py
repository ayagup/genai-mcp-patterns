"""
Pattern 289: Event Sourcing MCP Pattern

This pattern demonstrates event sourcing where state changes are stored
as a sequence of events that can be replayed to reconstruct state.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
from datetime import datetime


class EventSourcingPattern(TypedDict):
    """State for event sourcing"""
    messages: Annotated[List[str], add]
    event_store: List[Dict[str, Any]]
    current_state: Dict[str, Any]
    snapshots: List[Dict[str, Any]]
    replay_results: Dict[str, Any]


class Event:
    """Represents a domain event"""
    
    def __init__(self, event_type: str, data: Dict[str, Any], aggregate_id: str):
        self.event_id = f"evt_{int(time.time() * 1000)}"
        self.event_type = event_type
        self.data = data
        self.aggregate_id = aggregate_id
        self.timestamp = time.time()
        self.version = 0
    
    def to_dict(self):
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp,
            "version": self.version
        }


class EventStore:
    """Stores and manages events"""
    
    def __init__(self):
        self.events = []
        self.version = 0
    
    def append(self, event: Event):
        """Append event to store"""
        event.version = self.version + 1
        self.version = event.version
        self.events.append(event)
    
    def get_events(self, aggregate_id: str = None, from_version: int = 0):
        """Retrieve events"""
        filtered = self.events
        
        if aggregate_id:
            filtered = [e for e in filtered if e.aggregate_id == aggregate_id]
        
        if from_version > 0:
            filtered = [e for e in filtered if e.version > from_version]
        
        return filtered
    
    def get_all_events(self):
        """Get all events"""
        return [e.to_dict() for e in self.events]


class AccountAggregate:
    """Bank account aggregate rebuilt from events"""
    
    def __init__(self, account_id: str):
        self.account_id = account_id
        self.balance = 0
        self.status = "pending"
        self.transactions = []
        self.version = 0
    
    def apply_event(self, event: Event):
        """Apply event to rebuild state"""
        self.version = event.version
        
        if event.event_type == "AccountCreated":
            self.balance = event.data.get("initial_balance", 0)
            self.status = "active"
        
        elif event.event_type == "MoneyDeposited":
            amount = event.data.get("amount", 0)
            self.balance += amount
            self.transactions.append({
                "type": "deposit",
                "amount": amount,
                "timestamp": event.timestamp
            })
        
        elif event.event_type == "MoneyWithdrawn":
            amount = event.data.get("amount", 0)
            self.balance -= amount
            self.transactions.append({
                "type": "withdrawal",
                "amount": amount,
                "timestamp": event.timestamp
            })
        
        elif event.event_type == "AccountClosed":
            self.status = "closed"
    
    def get_state(self):
        """Get current state"""
        return {
            "account_id": self.account_id,
            "balance": self.balance,
            "status": self.status,
            "transaction_count": len(self.transactions),
            "version": self.version
        }


def initialize_event_store_agent(state: EventSourcingPattern) -> EventSourcingPattern:
    """Initialize event store"""
    print("\n📚 Initializing Event Store...")
    
    store = EventStore()
    
    print(f"  Event Store: Ready")
    print(f"  Features:")
    print(f"    • Immutable event log")
    print(f"    • Event versioning")
    print(f"    • State reconstruction")
    print(f"    • Audit trail")
    print(f"    • Time travel")
    
    return {
        **state,
        "event_store": [],
        "current_state": {},
        "snapshots": [],
        "replay_results": {},
        "messages": ["✓ Event store initialized"]
    }


def record_events_agent(state: EventSourcingPattern) -> EventSourcingPattern:
    """Record domain events"""
    print("\n✍️ Recording Events...")
    
    store = EventStore()
    account_id = "acc_12345"
    
    # Simulate series of events
    events_to_record = [
        Event("AccountCreated", {"initial_balance": 1000, "owner": "Alice"}, account_id),
        Event("MoneyDeposited", {"amount": 500, "source": "paycheck"}, account_id),
        Event("MoneyDeposited", {"amount": 200, "source": "bonus"}, account_id),
        Event("MoneyWithdrawn", {"amount": 150, "purpose": "groceries"}, account_id),
        Event("MoneyWithdrawn", {"amount": 300, "purpose": "rent"}, account_id),
        Event("MoneyDeposited", {"amount": 1000, "source": "client payment"}, account_id)
    ]
    
    for event in events_to_record:
        time.sleep(0.01)  # Small delay for unique timestamps
        store.append(event)
        print(f"  ✓ Event #{event.version}: {event.event_type}")
        print(f"    Data: {event.data}")
    
    print(f"\n  Total Events Recorded: {len(store.events)}")
    print(f"  Current Version: {store.version}")
    
    event_dicts = store.get_all_events()
    
    return {
        **state,
        "event_store": event_dicts,
        "messages": [f"✓ Recorded {len(events_to_record)} events"]
    }


def replay_events_agent(state: EventSourcingPattern) -> EventSourcingPattern:
    """Replay events to rebuild state"""
    print("\n🔄 Replaying Events to Rebuild State...")
    
    # Reconstruct event store
    store = EventStore()
    for event_dict in state["event_store"]:
        event = Event(
            event_dict["event_type"],
            event_dict["data"],
            event_dict["aggregate_id"]
        )
        event.version = event_dict["version"]
        event.timestamp = event_dict["timestamp"]
        store.events.append(event)
    
    # Replay events to rebuild aggregate
    account_id = state["event_store"][0]["aggregate_id"] if state["event_store"] else "acc_12345"
    account = AccountAggregate(account_id)
    
    events = store.get_events(account_id)
    
    print(f"  Replaying {len(events)} events...")
    
    for event in events:
        account.apply_event(event)
        print(f"    Event #{event.version}: {event.event_type}")
    
    final_state = account.get_state()
    
    print(f"\n  Reconstructed State:")
    print(f"    Account ID: {final_state['account_id']}")
    print(f"    Balance: ${final_state['balance']}")
    print(f"    Status: {final_state['status']}")
    print(f"    Transactions: {final_state['transaction_count']}")
    print(f"    Version: {final_state['version']}")
    
    return {
        **state,
        "current_state": final_state,
        "messages": [f"✓ Replayed {len(events)} events"]
    }


def create_snapshots_agent(state: EventSourcingPattern) -> EventSourcingPattern:
    """Create state snapshots for performance"""
    print("\n📸 Creating Snapshots...")
    
    # Reconstruct events
    store = EventStore()
    for event_dict in state["event_store"]:
        event = Event(
            event_dict["event_type"],
            event_dict["data"],
            event_dict["aggregate_id"]
        )
        event.version = event_dict["version"]
        store.events.append(event)
    
    snapshots = []
    account_id = state["event_store"][0]["aggregate_id"] if state["event_store"] else "acc_12345"
    account = AccountAggregate(account_id)
    
    # Create snapshots at intervals
    snapshot_intervals = [2, 4, 6]
    
    for event in store.events:
        account.apply_event(event)
        
        if event.version in snapshot_intervals:
            snapshot = {
                "version": event.version,
                "timestamp": time.time(),
                "state": account.get_state(),
                "event_count": event.version
            }
            snapshots.append(snapshot)
            print(f"  Snapshot at version {event.version}:")
            print(f"    Balance: ${account.balance}")
            print(f"    Transactions: {len(account.transactions)}")
    
    print(f"\n  Total Snapshots: {len(snapshots)}")
    
    return {
        **state,
        "snapshots": snapshots,
        "messages": [f"✓ Created {len(snapshots)} snapshots"]
    }


def demonstrate_time_travel_agent(state: EventSourcingPattern) -> EventSourcingPattern:
    """Demonstrate time travel capabilities"""
    print("\n⏰ Demonstrating Time Travel...")
    
    # Reconstruct events
    store = EventStore()
    for event_dict in state["event_store"]:
        event = Event(
            event_dict["event_type"],
            event_dict["data"],
            event_dict["aggregate_id"]
        )
        event.version = event_dict["version"]
        event.timestamp = event_dict["timestamp"]
        store.events.append(event)
    
    account_id = state["event_store"][0]["aggregate_id"] if state["event_store"] else "acc_12345"
    
    # Show state at different points in time
    time_points = [2, 4, len(store.events)]
    replay_results = {}
    
    for version in time_points:
        account = AccountAggregate(account_id)
        events = store.get_events(account_id)[:version]
        
        for event in events:
            account.apply_event(event)
        
        state_at_version = account.get_state()
        replay_results[f"version_{version}"] = state_at_version
        
        print(f"\n  State at Version {version}:")
        print(f"    Balance: ${state_at_version['balance']}")
        print(f"    Transactions: {state_at_version['transaction_count']}")
        print(f"    Status: {state_at_version['status']}")
    
    return {
        **state,
        "replay_results": replay_results,
        "messages": [f"✓ Time traveled to {len(time_points)} versions"]
    }


def generate_event_sourcing_report_agent(state: EventSourcingPattern) -> EventSourcingPattern:
    """Generate event sourcing report"""
    print("\n" + "="*70)
    print("EVENT SOURCING REPORT")
    print("="*70)
    
    print(f"\n📚 Event Store:")
    print(f"  Total Events: {len(state['event_store'])}")
    print(f"  Current Version: {state['event_store'][-1]['version'] if state['event_store'] else 0}")
    
    print(f"\n📋 Event Log:")
    for event in state['event_store'][:10]:  # Show first 10
        print(f"  • v{event['version']}: {event['event_type']}")
        print(f"      Data: {event['data']}")
        print(f"      Timestamp: {datetime.fromtimestamp(event['timestamp']).strftime('%H:%M:%S')}")
    
    print(f"\n🎯 Current State (Rebuilt from Events):")
    for key, value in state['current_state'].items():
        print(f"  {key}: {value}")
    
    print(f"\n📸 Snapshots:")
    print(f"  Total Snapshots: {len(state['snapshots'])}")
    for snapshot in state['snapshots']:
        print(f"  • Version {snapshot['version']}:")
        print(f"      Balance: ${snapshot['state']['balance']}")
        print(f"      Events: {snapshot['event_count']}")
    
    print(f"\n⏰ Time Travel Results:")
    for version_key, version_state in state['replay_results'].items():
        print(f"  {version_key}:")
        print(f"    Balance: ${version_state['balance']}")
        print(f"    Transactions: {version_state['transaction_count']}")
    
    print(f"\n💡 Event Sourcing Benefits:")
    print("  ✓ Complete audit trail")
    print("  ✓ State reconstruction")
    print("  ✓ Time travel queries")
    print("  ✓ Event replay")
    print("  ✓ Debugging capabilities")
    print("  ✓ Historical analysis")
    
    print(f"\n🔧 Key Concepts:")
    print("  • Immutable event log")
    print("  • Event versioning")
    print("  • Aggregate reconstruction")
    print("  • Snapshot optimization")
    print("  • Event replay")
    print("  • Point-in-time queries")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Financial systems")
    print("  • Order management")
    print("  • Inventory tracking")
    print("  • Audit requirements")
    print("  • Version control")
    print("  • Compliance systems")
    
    print(f"\n🎯 Event Patterns:")
    print("  • Domain events")
    print("  • Command events")
    print("  • Integration events")
    print("  • System events")
    
    print("\n" + "="*70)
    print("✅ Event Sourcing Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_event_sourcing_graph():
    """Create event sourcing workflow"""
    workflow = StateGraph(EventSourcingPattern)
    
    workflow.add_node("initialize", initialize_event_store_agent)
    workflow.add_node("record", record_events_agent)
    workflow.add_node("replay", replay_events_agent)
    workflow.add_node("snapshot", create_snapshots_agent)
    workflow.add_node("time_travel", demonstrate_time_travel_agent)
    workflow.add_node("report", generate_event_sourcing_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "record")
    workflow.add_edge("record", "replay")
    workflow.add_edge("replay", "snapshot")
    workflow.add_edge("snapshot", "time_travel")
    workflow.add_edge("time_travel", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 289: Event Sourcing MCP Pattern")
    print("="*70)
    
    app = create_event_sourcing_graph()
    final_state = app.invoke({
        "messages": [],
        "event_store": [],
        "current_state": {},
        "snapshots": [],
        "replay_results": {}
    })
    
    print("\n✅ Event Sourcing Pattern Complete!")


if __name__ == "__main__":
    main()

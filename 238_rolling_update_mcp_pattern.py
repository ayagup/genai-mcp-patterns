"""
Pattern 238: Rolling Update MCP Pattern

This pattern demonstrates rolling updates - gradually updating instances
one at a time to maintain service availability.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class RollingUpdateState(TypedDict):
    """State for rolling update workflow"""
    messages: Annotated[List[str], add]
    instances: List[dict]
    update_progress: List[dict]


class RollingUpdateManager:
    """Manages rolling updates"""
    
    def __init__(self, num_instances: int = 5):
        self.instances = [
            {"id": i+1, "version": "v1.0.0", "status": "running"}
            for i in range(num_instances)
        ]
    
    def update_instance(self, instance_id: int, new_version: str):
        """Update a single instance"""
        for instance in self.instances:
            if instance["id"] == instance_id:
                instance["version"] = new_version
                instance["status"] = "updated"
                break


def initialize_instances_agent(state: RollingUpdateState) -> RollingUpdateState:
    """Initialize instances"""
    print("\n🖥️ Initializing Instances...")
    
    manager = RollingUpdateManager(5)
    instances = manager.instances
    
    print(f"\n  Total Instances: {len(instances)}")
    for inst in instances:
        print(f"    Instance {inst['id']}: {inst['version']} ({inst['status']})")
    
    return {
        **state,
        "instances": instances,
        "messages": [f"✓ Initialized {len(instances)} instances"]
    }


def rolling_update_agent(state: RollingUpdateState) -> RollingUpdateState:
    """Perform rolling update"""
    print("\n🔄 Performing Rolling Update...")
    
    manager = RollingUpdateManager(5)
    manager.instances = state["instances"]
    new_version = "v2.0.0"
    update_progress = []
    
    for i in range(1, 6):
        print(f"\n  Updating Instance {i}...")
        manager.update_instance(i, new_version)
        
        # Count updated vs remaining
        updated_count = sum(1 for inst in manager.instances if inst["version"] == new_version)
        remaining_count = len(manager.instances) - updated_count
        
        update_progress.append({
            "instance_id": i,
            "version": new_version,
            "updated": updated_count,
            "remaining": remaining_count
        })
        
        print(f"    ✓ Instance {i} updated to {new_version}")
        print(f"    Progress: {updated_count}/{len(manager.instances)} instances updated")
    
    return {
        **state,
        "instances": manager.instances,
        "update_progress": update_progress,
        "messages": ["✓ Rolling update complete"]
    }


def generate_rolling_update_report_agent(state: RollingUpdateState) -> RollingUpdateState:
    """Generate rolling update report"""
    print("\n" + "="*70)
    print("ROLLING UPDATE REPORT")
    print("="*70)
    
    print(f"\n📊 Update Progress:")
    for progress in state["update_progress"]:
        print(f"  Step {progress['instance_id']}: {progress['updated']}/{progress['updated'] + progress['remaining']} instances updated")
    
    print(f"\n🖥️ Final Instance Status:")
    for inst in state["instances"]:
        print(f"  Instance {inst['id']}: {inst['version']} ({inst['status']})")
    
    print("\n💡 Rolling Update Benefits:")
    print("  • Zero downtime")
    print("  • Gradual deployment")
    print("  • Easy rollback")
    print("  • Service always available")
    
    print("\n="*70)
    print("✅ Rolling Update Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_rolling_update_graph():
    workflow = StateGraph(RollingUpdateState)
    workflow.add_node("initialize", initialize_instances_agent)
    workflow.add_node("update", rolling_update_agent)
    workflow.add_node("report", generate_rolling_update_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "update")
    workflow.add_edge("update", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 238: Rolling Update MCP Pattern")
    print("="*70)
    
    app = create_rolling_update_graph()
    final_state = app.invoke({"messages": [], "instances": [], "update_progress": []})
    print("\n✅ Rolling Update Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Pattern 236: Blue-Green Deployment MCP Pattern

This pattern demonstrates blue-green deployment - maintaining two identical
production environments and switching between them for zero-downtime deployments.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class BlueGreenDeploymentState(TypedDict):
    """State for blue-green deployment workflow"""
    messages: Annotated[List[str], add]
    blue_environment: dict
    green_environment: dict
    active_environment: str
    deployment_results: List[dict]


# Environment representation
class Environment:
    """Represents a deployment environment"""
    
    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.is_healthy = True
        self.traffic_percentage = 0
    
    def deploy(self, new_version: str):
        """Deploy new version to this environment"""
        self.version = new_version
        self.is_healthy = True
    
    def health_check(self) -> bool:
        """Check environment health"""
        return self.is_healthy
    
    def get_status(self) -> dict:
        """Get environment status"""
        return {
            "name": self.name,
            "version": self.version,
            "healthy": self.is_healthy,
            "traffic": f"{self.traffic_percentage}%"
        }


# Blue-Green Deployment Manager
class BlueGreenDeploymentManager:
    """Manages blue-green deployment"""
    
    def __init__(self):
        self.blue = Environment("Blue", "v1.0.0")
        self.green = Environment("Green", "v1.0.0")
        self.active = "blue"
        self.blue.traffic_percentage = 100
        self.green.traffic_percentage = 0
    
    def deploy_to_inactive(self, new_version: str):
        """Deploy new version to inactive environment"""
        inactive = self.green if self.active == "blue" else self.blue
        inactive.deploy(new_version)
        return inactive
    
    def switch_active(self):
        """Switch active environment (instant cutover)"""
        if self.active == "blue":
            self.active = "green"
            self.blue.traffic_percentage = 0
            self.green.traffic_percentage = 100
        else:
            self.active = "blue"
            self.blue.traffic_percentage = 100
            self.green.traffic_percentage = 0
    
    def get_active_environment(self) -> Environment:
        """Get currently active environment"""
        return self.blue if self.active == "blue" else self.green
    
    def get_inactive_environment(self) -> Environment:
        """Get currently inactive environment"""
        return self.green if self.active == "blue" else self.blue
    
    def rollback(self):
        """Rollback to previous environment"""
        self.switch_active()


# Agent functions
def setup_environments_agent(state: BlueGreenDeploymentState) -> BlueGreenDeploymentState:
    """Setup blue and green environments"""
    print("\n🔵🟢 Setting Up Blue-Green Environments...")
    
    manager = BlueGreenDeploymentManager()
    
    blue_status = manager.blue.get_status()
    green_status = manager.green.get_status()
    
    print(f"\n  Blue Environment: v{manager.blue.version} (Active: {manager.active == 'blue'})")
    print(f"    Traffic: {manager.blue.traffic_percentage}%")
    
    print(f"\n  Green Environment: v{manager.green.version} (Active: {manager.active == 'green'})")
    print(f"    Traffic: {manager.green.traffic_percentage}%")
    
    return {
        **state,
        "blue_environment": blue_status,
        "green_environment": green_status,
        "active_environment": "blue",
        "messages": ["✓ Blue-Green environments initialized"]
    }


def deploy_to_inactive_agent(state: BlueGreenDeploymentState) -> BlueGreenDeploymentState:
    """Deploy new version to inactive environment"""
    print("\n📦 Deploying to Inactive Environment...")
    
    manager = BlueGreenDeploymentManager()
    new_version = "v2.0.0"
    
    inactive = manager.deploy_to_inactive(new_version)
    
    print(f"\n  Deployed v{new_version} to {inactive.name} environment")
    print(f"  Active environment ({manager.active.capitalize()}): Still running v{manager.get_active_environment().version}")
    print(f"  Zero downtime - users unaffected")
    
    deployment_results = [{
        "action": "Deploy to Inactive",
        "version": new_version,
        "target_environment": inactive.name,
        "status": "SUCCESS"
    }]
    
    return {
        **state,
        "deployment_results": deployment_results,
        "messages": [f"✓ Deployed v{new_version} to inactive environment"]
    }


def test_inactive_environment_agent(state: BlueGreenDeploymentState) -> BlueGreenDeploymentState:
    """Test inactive environment before switching"""
    print("\n🧪 Testing Inactive Environment...")
    
    manager = BlueGreenDeploymentManager()
    inactive = manager.get_inactive_environment()
    
    # Run health checks
    health_ok = inactive.health_check()
    
    print(f"\n  Running tests on {inactive.name} environment (v{inactive.version})...")
    print(f"    Health Check: {'✓ PASS' if health_ok else '✗ FAIL'}")
    print(f"    Integration Tests: ✓ PASS")
    print(f"    Smoke Tests: ✓ PASS")
    
    test_result = {
        "action": "Test Inactive Environment",
        "environment": inactive.name,
        "version": inactive.version,
        "health_check": health_ok,
        "status": "READY" if health_ok else "NOT_READY"
    }
    
    deployment_results = state.get("deployment_results", []) + [test_result]
    
    return {
        **state,
        "deployment_results": deployment_results,
        "messages": ["✓ Inactive environment tested and ready"]
    }


def switch_traffic_agent(state: BlueGreenDeploymentState) -> BlueGreenDeploymentState:
    """Switch traffic to new environment (instant cutover)"""
    print("\n🔀 Switching Traffic...")
    
    manager = BlueGreenDeploymentManager()
    old_active = manager.active
    old_version = manager.get_active_environment().version
    
    # Instant switch
    manager.switch_active()
    
    new_active = manager.active
    new_version = manager.get_active_environment().version
    
    print(f"\n  Traffic Switch: {old_active.capitalize()} → {new_active.capitalize()}")
    print(f"  Version Change: v{old_version} → v{new_version}")
    print(f"  Downtime: 0 milliseconds (instant switch)")
    print(f"\n  Current Traffic Distribution:")
    print(f"    Blue: {manager.blue.traffic_percentage}%")
    print(f"    Green: {manager.green.traffic_percentage}%")
    
    switch_result = {
        "action": "Switch Traffic",
        "from": old_active,
        "to": new_active,
        "old_version": old_version,
        "new_version": new_version,
        "downtime_ms": 0,
        "status": "SUCCESS"
    }
    
    deployment_results = state.get("deployment_results", []) + [switch_result]
    
    return {
        **state,
        "active_environment": new_active,
        "deployment_results": deployment_results,
        "messages": [f"✓ Traffic switched to {new_active} (v{new_version})"]
    }


def monitor_and_rollback_agent(state: BlueGreenDeploymentState) -> BlueGreenDeploymentState:
    """Monitor new environment and demonstrate rollback capability"""
    print("\n📊 Monitoring New Environment...")
    
    manager = BlueGreenDeploymentManager()
    active = manager.get_active_environment()
    
    print(f"\n  Monitoring {active.name} environment...")
    print(f"    Error Rate: 0.01% ✓")
    print(f"    Response Time: 45ms ✓")
    print(f"    CPU Usage: 35% ✓")
    print(f"    Memory Usage: 62% ✓")
    print(f"\n  ✓ All metrics healthy!")
    
    print(f"\n💡 Rollback Capability:")
    print(f"  Inactive environment ({manager.get_inactive_environment().name}) still available")
    print(f"  Can instantly rollback if issues detected")
    print(f"  Just switch traffic back to previous environment")
    
    monitor_result = {
        "action": "Monitor Active Environment",
        "environment": active.name,
        "version": active.version,
        "metrics": {
            "error_rate": "0.01%",
            "response_time": "45ms",
            "cpu": "35%",
            "memory": "62%"
        },
        "status": "HEALTHY",
        "rollback_ready": True
    }
    
    deployment_results = state.get("deployment_results", []) + [monitor_result]
    
    return {
        **state,
        "deployment_results": deployment_results,
        "messages": ["✓ Deployment successful - monitoring healthy"]
    }


def generate_deployment_report_agent(state: BlueGreenDeploymentState) -> BlueGreenDeploymentState:
    """Generate blue-green deployment report"""
    print("\n" + "="*70)
    print("BLUE-GREEN DEPLOYMENT REPORT")
    print("="*70)
    
    print(f"\n🎯 Deployment Strategy: Blue-Green")
    print(f"   Active Environment: {state['active_environment'].capitalize()}")
    
    print(f"\n📋 Deployment Steps:")
    for i, result in enumerate(state["deployment_results"], 1):
        status_icon = "✓" if result["status"] in ["SUCCESS", "READY", "HEALTHY"] else "✗"
        print(f"\n  {status_icon} Step {i}: {result['action']}")
        print(f"      Status: {result['status']}")
        if "version" in result:
            print(f"      Version: {result['version']}")
        if "downtime_ms" in result:
            print(f"      Downtime: {result['downtime_ms']}ms")
    
    print("\n💡 Blue-Green Deployment Benefits:")
    print("  • Zero downtime deployments")
    print("  • Instant rollback capability")
    print("  • Full testing before traffic switch")
    print("  • Reduced risk - easy to revert")
    print("  • Simple and reliable process")
    
    print("\n📚 Key Concepts:")
    print("  • Two identical production environments")
    print("  • Deploy to inactive, test thoroughly")
    print("  • Instant traffic switch when ready")
    print("  • Keep previous version for quick rollback")
    
    print("\n" + "="*70)
    print("✅ Blue-Green Deployment Complete - Zero Downtime!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Blue-green deployment report generated"]
    }


# Create the graph
def create_blue_green_deployment_graph():
    """Create the blue-green deployment workflow graph"""
    workflow = StateGraph(BlueGreenDeploymentState)
    
    # Add nodes
    workflow.add_node("setup", setup_environments_agent)
    workflow.add_node("deploy", deploy_to_inactive_agent)
    workflow.add_node("test", test_inactive_environment_agent)
    workflow.add_node("switch", switch_traffic_agent)
    workflow.add_node("monitor", monitor_and_rollback_agent)
    workflow.add_node("report", generate_deployment_report_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "deploy")
    workflow.add_edge("deploy", "test")
    workflow.add_edge("test", "switch")
    workflow.add_edge("switch", "monitor")
    workflow.add_edge("monitor", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 236: Blue-Green Deployment MCP Pattern")
    print("="*70)
    print("\nBlue-Green: Zero-downtime deployments with instant rollback")
    
    # Create and run the workflow
    app = create_blue_green_deployment_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "blue_environment": {},
        "green_environment": {},
        "active_environment": "",
        "deployment_results": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Blue-Green Deployment Pattern Complete!")


if __name__ == "__main__":
    main()

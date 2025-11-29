"""
Pattern 237: Canary Deployment MCP Pattern

This pattern demonstrates canary deployment - gradually rolling out changes
to a small subset of users before full deployment.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class CanaryDeploymentState(TypedDict):
    """State for canary deployment workflow"""
    messages: Annotated[List[str], add]
    canary_percentage: int
    deployment_stages: List[dict]


class CanaryDeploymentManager:
    """Manages canary deployment"""
    
    def __init__(self):
        self.stable_version = "v1.0.0"
        self.canary_version = "v2.0.0"
        self.canary_percentage = 0
    
    def increase_canary(self, percentage: int):
        """Increase canary traffic percentage"""
        self.canary_percentage = min(100, self.canary_percentage + percentage)
    
    def get_traffic_distribution(self) -> dict:
        """Get current traffic distribution"""
        return {
            "stable": f"{100 - self.canary_percentage}%",
            "canary": f"{self.canary_percentage}%"
        }


def deploy_canary_agent(state: CanaryDeploymentState) -> CanaryDeploymentState:
    """Deploy canary version"""
    print("\n🐤 Deploying Canary Version...")
    
    manager = CanaryDeploymentManager()
    
    stages = [{
        "stage": "Initial Deployment",
        "canary_percentage": 0,
        "action": "Deploy v2.0.0 to canary servers",
        "status": "SUCCESS"
    }]
    
    print(f"  ✓ Deployed {manager.canary_version} to canary servers")
    print(f"  Traffic: Stable {100}%, Canary {0}%")
    
    return {
        **state,
        "canary_percentage": 0,
        "deployment_stages": stages,
        "messages": ["✓ Canary deployed"]
    }


def gradual_rollout_agent(state: CanaryDeploymentState) -> CanaryDeploymentState:
    """Gradually increase canary traffic"""
    print("\n📈 Gradual Rollout...")
    
    manager = CanaryDeploymentManager()
    stages = state.get("deployment_stages", [])
    
    # Stage 1: 10%
    manager.increase_canary(10)
    print(f"\n  Stage 1: {manager.canary_percentage}% canary traffic")
    stages.append({"stage": "10% Rollout", "canary_percentage": 10, "status": "MONITORING"})
    
    # Stage 2: 25%
    manager.increase_canary(15)
    print(f"  Stage 2: {manager.canary_percentage}% canary traffic")
    stages.append({"stage": "25% Rollout", "canary_percentage": 25, "status": "MONITORING"})
    
    # Stage 3: 50%
    manager.increase_canary(25)
    print(f"  Stage 3: {manager.canary_percentage}% canary traffic")
    stages.append({"stage": "50% Rollout", "canary_percentage": 50, "status": "MONITORING"})
    
    # Stage 4: 100%
    manager.increase_canary(50)
    print(f"  Stage 4: {manager.canary_percentage}% canary traffic")
    stages.append({"stage": "100% Rollout", "canary_percentage": 100, "status": "COMPLETE"})
    
    return {
        **state,
        "canary_percentage": 100,
        "deployment_stages": stages,
        "messages": ["✓ Gradual rollout complete"]
    }


def generate_canary_report_agent(state: CanaryDeploymentState) -> CanaryDeploymentState:
    """Generate canary deployment report"""
    print("\n" + "="*70)
    print("CANARY DEPLOYMENT REPORT")
    print("="*70)
    
    print(f"\n📊 Deployment Stages:")
    for stage in state["deployment_stages"]:
        print(f"  • {stage['stage']}: {stage.get('canary_percentage', 0)}% canary")
    
    print("\n💡 Canary Benefits:")
    print("  • Gradual risk mitigation")
    print("  • Early issue detection")
    print("  • Limited blast radius")
    print("  • Confidence building")
    
    print("\n="*70)
    print("✅ Canary Deployment Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_canary_deployment_graph():
    workflow = StateGraph(CanaryDeploymentState)
    workflow.add_node("deploy", deploy_canary_agent)
    workflow.add_node("rollout", gradual_rollout_agent)
    workflow.add_node("report", generate_canary_report_agent)
    workflow.add_edge(START, "deploy")
    workflow.add_edge("deploy", "rollout")
    workflow.add_edge("rollout", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 237: Canary Deployment MCP Pattern")
    print("="*70)
    
    app = create_canary_deployment_graph()
    final_state = app.invoke({"messages": [], "canary_percentage": 0, "deployment_stages": []})
    print("\n✅ Canary Deployment Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Pattern 239: Feature Toggle MCP Pattern

This pattern demonstrates feature toggles (feature flags) - enabling or disabling
features at runtime without deploying new code.
"""

from typing import TypedDict, Annotated, List, Dict
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class FeatureToggleState(TypedDict):
    """State for feature toggle workflow"""
    messages: Annotated[List[str], add]
    features: Dict[str, bool]
    toggle_history: List[dict]


class FeatureToggleManager:
    """Manages feature toggles"""
    
    def __init__(self):
        self.features = {
            "new_ui": False,
            "beta_search": False,
            "premium_features": True,
            "dark_mode": True,
            "experimental_ai": False
        }
    
    def is_enabled(self, feature_name: str) -> bool:
        """Check if feature is enabled"""
        return self.features.get(feature_name, False)
    
    def toggle(self, feature_name: str, enabled: bool):
        """Toggle feature on/off"""
        if feature_name in self.features:
            self.features[feature_name] = enabled


def initialize_features_agent(state: FeatureToggleState) -> FeatureToggleState:
    """Initialize feature toggles"""
    print("\n🎚️ Initializing Feature Toggles...")
    
    manager = FeatureToggleManager()
    
    print(f"\n  Feature States:")
    for feature, enabled in manager.features.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"    {feature}: {status}")
    
    return {
        **state,
        "features": manager.features.copy(),
        "messages": [f"✓ Initialized {len(manager.features)} feature toggles"]
    }


def toggle_features_agent(state: FeatureToggleState) -> FeatureToggleState:
    """Toggle features dynamically"""
    print("\n🔀 Toggling Features...")
    
    manager = FeatureToggleManager()
    manager.features = state["features"].copy()
    toggle_history = []
    
    # Enable new_ui
    print("\n  Enabling new_ui...")
    manager.toggle("new_ui", True)
    toggle_history.append({"feature": "new_ui", "action": "enabled", "reason": "Launch new interface"})
    
    # Enable beta_search
    print("  Enabling beta_search...")
    manager.toggle("beta_search", True)
    toggle_history.append({"feature": "beta_search", "action": "enabled", "reason": "Beta testing"})
    
    # Disable experimental_ai (keep it off)
    print("  Keeping experimental_ai disabled...")
    toggle_history.append({"feature": "experimental_ai", "action": "kept_disabled", "reason": "Still in development"})
    
    print(f"\n  Updated Feature States:")
    for feature, enabled in manager.features.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"    {feature}: {status}")
    
    return {
        **state,
        "features": manager.features,
        "toggle_history": toggle_history,
        "messages": ["✓ Features toggled successfully"]
    }


def generate_feature_toggle_report_agent(state: FeatureToggleState) -> FeatureToggleState:
    """Generate feature toggle report"""
    print("\n" + "="*70)
    print("FEATURE TOGGLE REPORT")
    print("="*70)
    
    print(f"\n🎚️ Current Feature States:")
    for feature, enabled in state["features"].items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {feature}: {status}")
    
    print(f"\n📋 Toggle History:")
    for toggle in state["toggle_history"]:
        print(f"  • {toggle['feature']}: {toggle['action']}")
        print(f"    Reason: {toggle['reason']}")
    
    print("\n💡 Feature Toggle Benefits:")
    print("  • Deploy code without enabling features")
    print("  • A/B testing and gradual rollout")
    print("  • Quick rollback without redeployment")
    print("  • Different features for different users")
    print("  • Reduce deployment risk")
    
    print("\n="*70)
    print("✅ Feature Toggle Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_feature_toggle_graph():
    workflow = StateGraph(FeatureToggleState)
    workflow.add_node("initialize", initialize_features_agent)
    workflow.add_node("toggle", toggle_features_agent)
    workflow.add_node("report", generate_feature_toggle_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "toggle")
    workflow.add_edge("toggle", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 239: Feature Toggle MCP Pattern")
    print("="*70)
    
    app = create_feature_toggle_graph()
    final_state = app.invoke({"messages": [], "features": {}, "toggle_history": []})
    print("\n✅ Feature Toggle Pattern Complete!")


if __name__ == "__main__":
    main()

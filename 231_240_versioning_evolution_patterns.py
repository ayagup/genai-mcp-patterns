"""
Patterns 231-240: Versioning and Evolution Patterns

231. Semantic Versioning - MAJOR.MINOR.PATCH
232. Backward Compatibility - Support older clients
233. Forward Compatibility - Support newer versions
234. API Versioning - Multiple API versions
235. Schema Evolution - Evolve data schemas
236. Blue-Green Deployment - Two environments
237. Canary Deployment - Gradual rollout
238. Rolling Update - Sequential updates
239. Feature Toggle - Enable/disable features
240. Deprecation - Gracefully retire features

Combined Implementation
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dataclasses import dataclass
from datetime import datetime, timedelta


class VersioningState(TypedDict):
    versioning_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


# Pattern 231: Semantic Versioning
@dataclass
class SemanticVersion:
    major: int
    minor: int
    patch: int
    
    def __str__(self):
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def bump_major(self):
        """Breaking changes"""
        return SemanticVersion(self.major + 1, 0, 0)
    
    def bump_minor(self):
        """New features, backward compatible"""
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def bump_patch(self):
        """Bug fixes"""
        return SemanticVersion(self.major, self.minor, self.patch + 1)


# Pattern 232 & 233: Compatibility
class APICompatibility:
    """Handle backward and forward compatibility"""
    def __init__(self):
        self.supported_versions = ["1.0", "1.1", "2.0"]
    
    def handle_request(self, version: str, data: Dict) -> Dict:
        """Backward compatible: support old versions"""
        if version == "1.0":
            # Old format
            return {"status": "ok", "result": data.get("field")}
        elif version in ["1.1", "2.0"]:
            # New format
            return {"status": "success", "data": data}
        else:
            return {"status": "error", "message": "Unsupported version"}


# Pattern 234: API Versioning
class APIVersionManager:
    """Manage multiple API versions"""
    def __init__(self):
        self.versions = {
            "v1": self._handle_v1,
            "v2": self._handle_v2,
            "v3": self._handle_v3
        }
    
    def _handle_v1(self, data):
        return {"version": "v1", "result": "legacy"}
    
    def _handle_v2(self, data):
        return {"version": "v2", "result": "current", "features": ["feature_a"]}
    
    def _handle_v3(self, data):
        return {"version": "v3", "result": "latest", "features": ["feature_a", "feature_b"]}
    
    def route(self, version: str, data: Dict) -> Dict:
        handler = self.versions.get(version, self._handle_v2)  # Default to v2
        return handler(data)


# Pattern 235: Schema Evolution
class SchemaVersion:
    """Evolve schemas over time"""
    def __init__(self):
        self.schema_v1 = {"name": "string", "email": "string"}
        self.schema_v2 = {"name": "string", "email": "string", "phone": "string"}  # Added field
    
    def migrate_v1_to_v2(self, data_v1: Dict) -> Dict:
        """Migrate old schema to new"""
        return {
            **data_v1,
            "phone": ""  # Default value for new field
        }


# Pattern 236: Blue-Green Deployment
class BlueGreenDeployment:
    """Two identical environments"""
    def __init__(self):
        self.blue_version = "1.0.0"
        self.green_version = "1.1.0"
        self.active = "blue"  # Currently serving traffic
        self.traffic = {"blue": 100, "green": 0}
    
    def switch_to_green(self):
        """Instant switch to new version"""
        self.active = "green"
        self.traffic = {"blue": 0, "green": 100}
    
    def rollback_to_blue(self):
        """Instant rollback"""
        self.active = "blue"
        self.traffic = {"blue": 100, "green": 0}


# Pattern 237: Canary Deployment
class CanaryDeployment:
    """Gradual rollout"""
    def __init__(self):
        self.stable_version = "1.0.0"
        self.canary_version = "1.1.0"
        self.canary_percentage = 0
    
    def increase_canary(self, percentage: int):
        """Gradually increase canary traffic"""
        self.canary_percentage = min(100, self.canary_percentage + percentage)
    
    def get_traffic_split(self) -> Dict[str, int]:
        return {
            "stable": 100 - self.canary_percentage,
            "canary": self.canary_percentage
        }


# Pattern 238: Rolling Update
class RollingUpdate:
    """Update instances sequentially"""
    def __init__(self, total_instances: int = 5):
        self.instances = [{"id": i, "version": "1.0.0"} for i in range(total_instances)]
        self.updated_count = 0
    
    def update_next_instance(self):
        """Update one instance at a time"""
        if self.updated_count < len(self.instances):
            self.instances[self.updated_count]["version"] = "1.1.0"
            self.updated_count += 1
    
    def get_status(self) -> Dict:
        return {
            "total": len(self.instances),
            "updated": self.updated_count,
            "pending": len(self.instances) - self.updated_count
        }


# Pattern 239: Feature Toggle
class FeatureToggle:
    """Enable/disable features dynamically"""
    def __init__(self):
        self.features = {
            "new_ui": False,
            "beta_feature": False,
            "premium_feature": True
        }
    
    def is_enabled(self, feature: str) -> bool:
        return self.features.get(feature, False)
    
    def toggle(self, feature: str, enabled: bool):
        self.features[feature] = enabled


# Pattern 240: Deprecation
class DeprecationManager:
    """Gracefully retire features"""
    def __init__(self):
        self.deprecated_features = {}
    
    def deprecate(self, feature: str, sunset_date: datetime):
        """Mark feature as deprecated"""
        self.deprecated_features[feature] = {
            "sunset_date": sunset_date,
            "alternative": f"{feature}_v2"
        }
    
    def is_deprecated(self, feature: str) -> bool:
        return feature in self.deprecated_features
    
    def get_deprecation_warning(self, feature: str) -> str:
        if feature in self.deprecated_features:
            info = self.deprecated_features[feature]
            return f"⚠️ {feature} is deprecated. Use {info['alternative']}. Sunset: {info['sunset_date']}"
        return ""


def setup_agent(state: VersioningState):
    operations = []
    operations.append("Versioning & Evolution Patterns (231-240):")
    operations.append("\n231. Semantic Versioning: MAJOR.MINOR.PATCH")
    operations.append("232. Backward Compatibility: Support old clients")
    operations.append("233. Forward Compatibility: Support new versions")
    operations.append("234. API Versioning: Multiple API versions")
    operations.append("235. Schema Evolution: Migrate schemas")
    operations.append("236. Blue-Green: Two environments")
    operations.append("237. Canary: Gradual rollout")
    operations.append("238. Rolling Update: Sequential updates")
    operations.append("239. Feature Toggle: Dynamic features")
    operations.append("240. Deprecation: Retire gracefully")
    
    return {
        "versioning_operations": operations,
        "operation_results": ["✓ All 10 patterns initialized"],
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def demo_versioning_agent(state: VersioningState):
    operations = []
    operations.append("\n📦 Versioning Patterns Demo:")
    
    # 231: Semantic Versioning
    version = SemanticVersion(1, 0, 0)
    v_minor = version.bump_minor()
    operations.append(f"\n231. Semantic: {version} → {v_minor} (new feature)")
    
    # 232-233: Compatibility
    compat = APICompatibility()
    result = compat.handle_request("1.0", {"field": "value"})
    operations.append(f"232-233. Compatibility: Supports v1.0, v1.1, v2.0")
    
    # 234: API Versioning
    api = APIVersionManager()
    v1_result = api.route("v1", {})
    v3_result = api.route("v3", {})
    operations.append(f"234. API Versioning: v1 → {v1_result['result']}, v3 → {v3_result['result']}")
    
    # 235: Schema Evolution
    schema = SchemaVersion()
    old_data = {"name": "John", "email": "john@example.com"}
    new_data = schema.migrate_v1_to_v2(old_data)
    operations.append(f"235. Schema Evolution: Migrated {len(old_data)} → {len(new_data)} fields")
    
    # 236: Blue-Green
    blue_green = BlueGreenDeployment()
    blue_green.switch_to_green()
    operations.append(f"236. Blue-Green: Switched to green ({blue_green.green_version})")
    
    # 237: Canary
    canary = CanaryDeployment()
    canary.increase_canary(10)
    canary.increase_canary(20)
    split = canary.get_traffic_split()
    operations.append(f"237. Canary: {split['stable']}% stable, {split['canary']}% canary")
    
    # 238: Rolling Update
    rolling = RollingUpdate(5)
    rolling.update_next_instance()
    rolling.update_next_instance()
    status = rolling.get_status()
    operations.append(f"238. Rolling Update: {status['updated']}/{status['total']} instances updated")
    
    # 239: Feature Toggle
    toggle = FeatureToggle()
    toggle.toggle("new_ui", True)
    operations.append(f"239. Feature Toggle: new_ui = {toggle.is_enabled('new_ui')}")
    
    # 240: Deprecation
    deprecation = DeprecationManager()
    sunset = datetime.now() + timedelta(days=90)
    deprecation.deprecate("old_api", sunset)
    warning = deprecation.get_deprecation_warning("old_api")
    operations.append(f"240. Deprecation: {warning[:50]}...")
    
    return {
        "versioning_operations": operations,
        "operation_results": ["✓ All 10 patterns demonstrated"],
        "performance_metrics": ["Patterns cover full version lifecycle"],
        "messages": ["Demo complete"]
    }


def statistics_agent(state: VersioningState):
    operations = []
    operations.append("\n" + "="*60)
    operations.append("VERSIONING & EVOLUTION PATTERNS COMPLETE")
    operations.append("="*60)
    
    operations.append("\n✅ Pattern 231: Semantic Versioning")
    operations.append("✅ Pattern 232: Backward Compatibility")
    operations.append("✅ Pattern 233: Forward Compatibility")
    operations.append("✅ Pattern 234: API Versioning")
    operations.append("✅ Pattern 235: Schema Evolution")
    operations.append("✅ Pattern 236: Blue-Green Deployment")
    operations.append("✅ Pattern 237: Canary Deployment")
    operations.append("✅ Pattern 238: Rolling Update")
    operations.append("✅ Pattern 239: Feature Toggle")
    operations.append("✅ Pattern 240: Deprecation")
    
    operations.append("\n🎉 Testing Patterns (221-230): COMPLETE")
    operations.append("🎉 Versioning Patterns (231-240): COMPLETE")
    operations.append("\n📊 Total: 20 patterns (221-240)")
    operations.append("📊 Progress: 240/400 (60% complete!)")
    
    return {
        "versioning_operations": operations,
        "operation_results": ["✓ Patterns 231-240 complete"],
        "performance_metrics": ["Milestone: 60% of all patterns!"],
        "messages": ["All patterns demonstrated"]
    }


def create_versioning_graph():
    workflow = StateGraph(VersioningState)
    workflow.add_node("setup", setup_agent)
    workflow.add_node("demo", demo_versioning_agent)
    workflow.add_node("stats", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "demo")
    workflow.add_edge("demo", "stats")
    workflow.add_edge("stats", END)
    
    return workflow.compile()


def main():
    print("=" * 80)
    print("Patterns 231-240: Versioning and Evolution Patterns")
    print("=" * 80)
    
    app = create_versioning_graph()
    final_state = app.invoke({
        "versioning_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    })
    
    for op in final_state["versioning_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("MILESTONE: 240/400 Patterns Complete (60%)! 🎉")
    print("=" * 80)


if __name__ == "__main__":
    main()

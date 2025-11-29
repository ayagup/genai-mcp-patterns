"""
Pattern 284: Application State MCP Pattern

This pattern demonstrates application-wide state management shared across
all users and sessions for global application data.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class ApplicationStatePattern(TypedDict):
    """State for application-wide management"""
    messages: Annotated[List[str], add]
    app_config: Dict[str, Any]
    global_cache: Dict[str, Any]
    system_metrics: Dict[str, Any]
    feature_flags: Dict[str, bool]


class ApplicationStateManager:
    """Manages global application state"""
    
    def __init__(self):
        self.config = {
            "app_name": "MCP Demo App",
            "version": "1.2.3",
            "environment": "production",
            "maintenance_mode": False,
            "max_users": 10000,
            "rate_limit": 100,
            "features": {
                "ai_assistant": True,
                "advanced_search": True,
                "beta_features": False
            }
        }
        
        self.cache = {}
        self.metrics = {
            "total_requests": 0,
            "active_users": 0,
            "uptime_seconds": 3600,
            "error_count": 0
        }
        
        self.feature_flags = {
            "new_ui": True,
            "experimental_ai": False,
            "dark_mode": True,
            "analytics": True,
            "notifications": True
        }
    
    def get_config(self, key: str = None):
        """Get application configuration"""
        if key:
            return self.config.get(key)
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update application configuration"""
        self.config.update(updates)
    
    def get_feature_flag(self, feature: str) -> bool:
        """Check if feature is enabled"""
        return self.feature_flags.get(feature, False)
    
    def update_metrics(self, metric: str, value: Any):
        """Update system metrics"""
        if metric in self.metrics:
            if isinstance(value, (int, float)):
                self.metrics[metric] += value
            else:
                self.metrics[metric] = value
    
    def cache_data(self, key: str, value: Any, ttl: int = 300):
        """Cache data globally"""
        import time
        self.cache[key] = {
            "value": value,
            "cached_at": time.time(),
            "ttl": ttl
        }
    
    def get_cached(self, key: str):
        """Retrieve cached data"""
        import time
        if key in self.cache:
            cached = self.cache[key]
            if time.time() - cached["cached_at"] < cached["ttl"]:
                return cached["value"]
            else:
                del self.cache[key]
        return None


def initialize_app_state_agent(state: ApplicationStatePattern) -> ApplicationStatePattern:
    """Initialize application state"""
    print("\n🚀 Initializing Application State...")
    
    manager = ApplicationStateManager()
    
    print(f"  App Name: {manager.config['app_name']}")
    print(f"  Version: {manager.config['version']}")
    print(f"  Environment: {manager.config['environment']}")
    print(f"  Maintenance Mode: {manager.config['maintenance_mode']}")
    
    print(f"\n  Feature Flags:")
    for feature, enabled in manager.feature_flags.items():
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"    {feature}: {status}")
    
    return {
        **state,
        "app_config": manager.config,
        "global_cache": manager.cache,
        "feature_flags": manager.feature_flags,
        "messages": ["✓ Application state initialized"]
    }


def manage_global_cache_agent(state: ApplicationStatePattern) -> ApplicationStatePattern:
    """Manage global cache"""
    print("\n💾 Managing Global Cache...")
    
    manager = ApplicationStateManager()
    manager.config = state["app_config"]
    
    # Cache frequently accessed data
    cache_items = [
        ("popular_products", ["P001", "P002", "P003"], 600),
        ("api_config", {"endpoint": "https://api.example.com", "version": "v2"}, 3600),
        ("system_status", {"healthy": True, "latency": 45}, 60),
        ("user_count", 1523, 300)
    ]
    
    for key, value, ttl in cache_items:
        manager.cache_data(key, value, ttl)
        print(f"  Cached: {key} (TTL: {ttl}s)")
    
    # Demonstrate cache retrieval
    cached_products = manager.get_cached("popular_products")
    print(f"\n  Retrieved from cache:")
    print(f"    popular_products: {cached_products}")
    
    return {
        **state,
        "global_cache": manager.cache,
        "messages": [f"✓ Cached {len(cache_items)} items"]
    }


def track_system_metrics_agent(state: ApplicationStatePattern) -> ApplicationStatePattern:
    """Track system-wide metrics"""
    print("\n📊 Tracking System Metrics...")
    
    manager = ApplicationStateManager()
    
    # Simulate metric updates
    manager.update_metrics("total_requests", 150)
    manager.update_metrics("active_users", 45)
    manager.metrics["uptime_seconds"] = 7200
    manager.metrics["error_count"] = 3
    
    # Calculate derived metrics
    metrics = {
        **manager.metrics,
        "avg_requests_per_user": manager.metrics["total_requests"] / max(manager.metrics["active_users"], 1),
        "error_rate": manager.metrics["error_count"] / max(manager.metrics["total_requests"], 1),
        "uptime_hours": manager.metrics["uptime_seconds"] / 3600
    }
    
    print(f"  Total Requests: {metrics['total_requests']}")
    print(f"  Active Users: {metrics['active_users']}")
    print(f"  Uptime: {metrics['uptime_hours']:.1f} hours")
    print(f"  Error Rate: {metrics['error_rate']:.2%}")
    print(f"  Avg Requests/User: {metrics['avg_requests_per_user']:.1f}")
    
    return {
        **state,
        "system_metrics": metrics,
        "messages": ["✓ System metrics updated"]
    }


def generate_app_state_report_agent(state: ApplicationStatePattern) -> ApplicationStatePattern:
    """Generate application state report"""
    print("\n" + "="*70)
    print("APPLICATION STATE REPORT")
    print("="*70)
    
    config = state["app_config"]
    cache = state["global_cache"]
    metrics = state["system_metrics"]
    flags = state["feature_flags"]
    
    print(f"\n⚙️ Application Configuration:")
    print(f"  Name: {config['app_name']}")
    print(f"  Version: {config['version']}")
    print(f"  Environment: {config['environment']}")
    print(f"  Maintenance Mode: {config['maintenance_mode']}")
    print(f"  Max Users: {config['max_users']:,}")
    print(f"  Rate Limit: {config['rate_limit']} req/min")
    
    print(f"\n🎯 Feature Configuration:")
    for feature, enabled in config["features"].items():
        print(f"  {feature}: {enabled}")
    
    print(f"\n🚩 Feature Flags (Runtime Toggle):")
    for feature, enabled in flags.items():
        status = "🟢 ON" if enabled else "🔴 OFF"
        print(f"  {status} {feature}")
    
    print(f"\n💾 Global Cache Status:")
    print(f"  Cached Items: {len(cache)}")
    for key, cached_data in cache.items():
        import time
        age = time.time() - cached_data["cached_at"]
        ttl_remaining = cached_data["ttl"] - age
        print(f"  • {key}: {ttl_remaining:.0f}s remaining")
    
    print(f"\n📊 System Metrics:")
    print(f"  Total Requests: {metrics['total_requests']:,}")
    print(f"  Active Users: {metrics['active_users']:,}")
    print(f"  Uptime: {metrics['uptime_hours']:.1f} hours")
    print(f"  Error Count: {metrics['error_count']}")
    print(f"  Error Rate: {metrics['error_rate']:.2%}")
    print(f"  Avg Requests/User: {metrics['avg_requests_per_user']:.1f}")
    
    print(f"\n💡 Application State Benefits:")
    print("  ✓ Global configuration management")
    print("  ✓ Shared cache across all users")
    print("  ✓ System-wide metrics")
    print("  ✓ Feature flag control")
    print("  ✓ Centralized state")
    print("  ✓ Dynamic configuration")
    
    print(f"\n🔄 State Scope:")
    print("  • Application-wide (all users)")
    print("  • Persists across sessions")
    print("  • Shared resources")
    print("  • Global cache")
    print("  • System configuration")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Feature toggles")
    print("  • Global caching")
    print("  • System configuration")
    print("  • Metrics collection")
    print("  • Maintenance mode")
    print("  • Rate limiting")
    
    print("\n" + "="*70)
    print("✅ Application State Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_application_state_graph():
    """Create application state workflow"""
    workflow = StateGraph(ApplicationStatePattern)
    
    workflow.add_node("initialize", initialize_app_state_agent)
    workflow.add_node("cache", manage_global_cache_agent)
    workflow.add_node("metrics", track_system_metrics_agent)
    workflow.add_node("report", generate_app_state_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "cache")
    workflow.add_edge("cache", "metrics")
    workflow.add_edge("metrics", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 284: Application State MCP Pattern")
    print("="*70)
    
    app = create_application_state_graph()
    final_state = app.invoke({
        "messages": [],
        "app_config": {},
        "global_cache": {},
        "system_metrics": {},
        "feature_flags": {}
    })
    
    print("\n✅ Application State Pattern Complete!")


if __name__ == "__main__":
    main()

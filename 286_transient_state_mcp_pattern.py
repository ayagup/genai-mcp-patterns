"""
Pattern 286: Transient State MCP Pattern

This pattern demonstrates transient state management for temporary data
that exists only during processing and is not persisted.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class TransientStatePattern(TypedDict):
    """State for transient data management"""
    messages: Annotated[List[str], add]
    in_memory_data: Dict[str, Any]
    temp_cache: Dict[str, Any]
    processing_context: Dict[str, Any]
    volatile_metrics: Dict[str, Any]


class TransientStateManager:
    """Manages transient state"""
    
    def __init__(self):
        self.memory_cache = {}
        self.processing_data = {}
        self.temp_results = {}
        self.lifecycle_start = time.time()
    
    def store_temp(self, key: str, value: Any, ttl: int = 60):
        """Store temporary data with TTL"""
        self.memory_cache[key] = {
            "value": value,
            "created_at": time.time(),
            "ttl": ttl,
            "access_count": 0
        }
    
    def get_temp(self, key: str):
        """Retrieve temporary data"""
        if key in self.memory_cache:
            item = self.memory_cache[key]
            age = time.time() - item["created_at"]
            
            if age < item["ttl"]:
                item["access_count"] += 1
                return item["value"]
            else:
                # Expired - remove
                del self.memory_cache[key]
        return None
    
    def store_processing_context(self, context: Dict[str, Any]):
        """Store processing context"""
        self.processing_data.update(context)
    
    def get_processing_context(self):
        """Get current processing context"""
        return self.processing_data
    
    def cache_intermediate_result(self, step: str, result: Any):
        """Cache intermediate processing results"""
        self.temp_results[step] = {
            "result": result,
            "timestamp": time.time()
        }
    
    def clear_expired(self):
        """Clear expired temporary data"""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.memory_cache.items():
            if current_time - item["created_at"] >= item["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        return len(expired_keys)
    
    def get_lifecycle_duration(self):
        """Get duration since lifecycle started"""
        return time.time() - self.lifecycle_start
    
    def cleanup(self):
        """Cleanup all transient state"""
        self.memory_cache.clear()
        self.processing_data.clear()
        self.temp_results.clear()


def initialize_transient_state_agent(state: TransientStatePattern) -> TransientStatePattern:
    """Initialize transient state"""
    print("\n⚡ Initializing Transient State...")
    
    manager = TransientStateManager()
    
    print(f"  State Type: Transient (In-Memory)")
    print(f"  Lifecycle: Request-scoped")
    print(f"  Persistence: None")
    print(f"  Features:")
    print(f"    • Fast access")
    print(f"    • Automatic cleanup")
    print(f"    • TTL-based expiration")
    print(f"    • No disk I/O")
    
    return {
        **state,
        "in_memory_data": {},
        "temp_cache": {},
        "processing_context": {},
        "volatile_metrics": {},
        "messages": ["✓ Transient state initialized"]
    }


def store_temporary_data_agent(state: TransientStatePattern) -> TransientStatePattern:
    """Store temporary data"""
    print("\n💨 Storing Temporary Data...")
    
    manager = TransientStateManager()
    
    # Store various temporary data with different TTLs
    temp_items = [
        ("api_response", {"status": 200, "data": [1, 2, 3]}, 30),
        ("user_input", {"query": "search term", "filters": ["active"]}, 60),
        ("calculation_cache", {"result": 42, "formula": "6*7"}, 15),
        ("validation_result", {"valid": True, "errors": []}, 45),
        ("session_token", "temp_token_12345", 300)
    ]
    
    for key, value, ttl in temp_items:
        manager.store_temp(key, value, ttl)
        print(f"  ✓ Cached: {key}")
        print(f"    TTL: {ttl}s")
        print(f"    Type: {type(value).__name__}")
    
    print(f"\n  Total Temporary Items: {len(manager.memory_cache)}")
    
    in_memory_data = {k: v["value"] for k, v in manager.memory_cache.items()}
    
    return {
        **state,
        "in_memory_data": in_memory_data,
        "temp_cache": manager.memory_cache,
        "messages": [f"✓ Stored {len(temp_items)} temporary items"]
    }


def process_with_context_agent(state: TransientStatePattern) -> TransientStatePattern:
    """Process data with transient context"""
    print("\n🔄 Processing with Transient Context...")
    
    manager = TransientStateManager()
    manager.memory_cache = state["temp_cache"]
    
    # Store processing context
    context = {
        "request_id": "req_789",
        "user_id": "user_456",
        "operation": "data_transformation",
        "start_time": time.time(),
        "parameters": {
            "format": "json",
            "compress": True,
            "validate": True
        }
    }
    
    manager.store_processing_context(context)
    
    print(f"  Request ID: {context['request_id']}")
    print(f"  Operation: {context['operation']}")
    print(f"  Parameters: {context['parameters']}")
    
    # Simulate multi-step processing with intermediate results
    processing_steps = [
        ("data_fetch", {"records": 100, "source": "database"}),
        ("data_transform", {"transformed": 100, "skipped": 0}),
        ("data_validate", {"valid": 98, "invalid": 2}),
        ("data_output", {"written": 98, "format": "json"})
    ]
    
    print(f"\n  Processing Steps:")
    for step, result in processing_steps:
        manager.cache_intermediate_result(step, result)
        print(f"    {step}: {result}")
        time.sleep(0.05)  # Simulate processing time
    
    return {
        **state,
        "processing_context": manager.processing_data,
        "messages": [f"✓ Processed {len(processing_steps)} steps"]
    }


def track_volatile_metrics_agent(state: TransientStatePattern) -> TransientStatePattern:
    """Track volatile metrics"""
    print("\n📊 Tracking Volatile Metrics...")
    
    manager = TransientStateManager()
    manager.memory_cache = state["temp_cache"]
    
    # Calculate metrics from transient state
    metrics = {
        "cache_size": len(manager.memory_cache),
        "total_accesses": sum(item["access_count"] for item in manager.memory_cache.values()),
        "lifecycle_duration": manager.get_lifecycle_duration(),
        "avg_ttl": sum(item["ttl"] for item in manager.memory_cache.values()) / max(len(manager.memory_cache), 1),
        "memory_usage_mb": sum(len(str(item["value"])) for item in manager.memory_cache.values()) / (1024 * 1024)
    }
    
    print(f"  Cache Size: {metrics['cache_size']} items")
    print(f"  Total Accesses: {metrics['total_accesses']}")
    print(f"  Lifecycle Duration: {metrics['lifecycle_duration']:.2f}s")
    print(f"  Average TTL: {metrics['avg_ttl']:.0f}s")
    print(f"  Memory Usage: {metrics['memory_usage_mb']:.4f} MB")
    
    # Demonstrate expiration
    time.sleep(0.1)
    expired_count = manager.clear_expired()
    
    print(f"\n  Cleanup:")
    print(f"    Expired Items: {expired_count}")
    print(f"    Remaining Items: {len(manager.memory_cache)}")
    
    return {
        **state,
        "volatile_metrics": metrics,
        "messages": ["✓ Volatile metrics tracked"]
    }


def generate_transient_state_report_agent(state: TransientStatePattern) -> TransientStatePattern:
    """Generate transient state report"""
    print("\n" + "="*70)
    print("TRANSIENT STATE REPORT")
    print("="*70)
    
    print(f"\n⚡ State Characteristics:")
    print(f"  Type: Transient (In-Memory)")
    print(f"  Persistence: None")
    print(f"  Lifetime: Request/Session scope")
    print(f"  Storage: RAM only")
    
    print(f"\n💨 In-Memory Data:")
    print(f"  Active Items: {len(state['in_memory_data'])}")
    for key, value in state['in_memory_data'].items():
        print(f"  • {key}: {value}")
    
    print(f"\n📦 Temporary Cache:")
    print(f"  Cached Items: {len(state['temp_cache'])}")
    for key, item in state['temp_cache'].items():
        print(f"  • {key}:")
        print(f"      TTL: {item['ttl']}s")
        print(f"      Access Count: {item['access_count']}")
        print(f"      Age: {time.time() - item['created_at']:.2f}s")
    
    print(f"\n🔄 Processing Context:")
    if state['processing_context']:
        for key, value in state['processing_context'].items():
            print(f"  • {key}: {value}")
    else:
        print("  No active processing context")
    
    print(f"\n📊 Volatile Metrics:")
    for key, value in state['volatile_metrics'].items():
        print(f"  • {key}: {value}")
    
    print(f"\n💡 Transient State Benefits:")
    print("  ✓ Ultra-fast access")
    print("  ✓ No persistence overhead")
    print("  ✓ Automatic cleanup")
    print("  ✓ Low latency")
    print("  ✓ No disk I/O")
    print("  ✓ Simple lifecycle")
    
    print(f"\n⚙️ Lifecycle:")
    print("  1. Created: On request start")
    print("  2. Used: During processing")
    print("  3. Expired: After TTL")
    print("  4. Destroyed: On request end")
    print("  5. No persistence: Data lost on restart")
    
    print(f"\n🎯 Use Cases:")
    print("  • Request-scoped data")
    print("  • Processing intermediate results")
    print("  • Temporary calculations")
    print("  • API response caching")
    print("  • Session tokens")
    print("  • Validation state")
    
    print(f"\n⚠️ Limitations:")
    print("  • Lost on restart")
    print("  • Not shared across instances")
    print("  • Memory constrained")
    print("  • No durability guarantees")
    
    print(f"\n✨ Best For:")
    print("  • Short-lived data")
    print("  • High-performance needs")
    print("  • Non-critical state")
    print("  • Temporary caching")
    print("  • Processing context")
    
    print("\n" + "="*70)
    print("✅ Transient State Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_transient_state_graph():
    """Create transient state workflow"""
    workflow = StateGraph(TransientStatePattern)
    
    workflow.add_node("initialize", initialize_transient_state_agent)
    workflow.add_node("store", store_temporary_data_agent)
    workflow.add_node("process", process_with_context_agent)
    workflow.add_node("metrics", track_volatile_metrics_agent)
    workflow.add_node("report", generate_transient_state_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "store")
    workflow.add_edge("store", "process")
    workflow.add_edge("process", "metrics")
    workflow.add_edge("metrics", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 286: Transient State MCP Pattern")
    print("="*70)
    
    app = create_transient_state_graph()
    final_state = app.invoke({
        "messages": [],
        "in_memory_data": {},
        "temp_cache": {},
        "processing_context": {},
        "volatile_metrics": {}
    })
    
    print("\n✅ Transient State Pattern Complete!")


if __name__ == "__main__":
    main()

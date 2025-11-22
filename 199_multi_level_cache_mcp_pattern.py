"""
Pattern 199: Multi-Level Cache MCP Pattern

This pattern demonstrates hierarchical caching with multiple levels:
- L1 Cache: Fast in-memory cache (local to process)
- L2 Cache: Shared remote cache (e.g., Redis)
- L3 Cache: Persistent data store (database)

Multi-level caching provides:
- Performance: Fast local cache for hot data
- Scalability: Shared cache across instances
- Persistence: Durable storage as fallback

Use Cases:
- Web applications with multiple servers
- Microservices architectures
- CDN edge caching (Edge -> Regional -> Origin)
- CPU cache hierarchy (L1 -> L2 -> L3 -> RAM)
- Browser caching (Memory -> Disk -> Network)

Key Features:
- Cache Promotion: Pull data up from lower levels
- Cache Demotion: Push data down to lower levels
- Write-Through: Propagate updates through all levels
- Invalidation: Clear data from all levels
- Hit Rate per Level: Track cache effectiveness
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
from collections import OrderedDict


class MultiLevelCacheState(TypedDict):
    """State for multi-level cache operations"""
    cache_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class CacheEntry:
    """Entry stored in cache with metadata"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0


class L1Cache:
    """
    L1 Cache: Fast in-memory cache local to process
    
    Characteristics:
    - Fastest access (nanoseconds)
    - Smallest capacity (typically MBs)
    - LRU eviction policy
    - Process-local (not shared)
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get from L1 cache"""
        if key in self.cache:
            self.hits += 1
            entry = self.cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            # Move to end (most recent)
            self.cache.move_to_end(key)
            return entry.value
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put into L1 cache with LRU eviction"""
        if key in self.cache:
            # Update existing
            entry = self.cache[key]
            entry.value = value
            entry.last_accessed = time.time()
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict LRU (first item)
                self.cache.popitem(last=False)
                self.evictions += 1
            
            entry = CacheEntry(key=key, value=value)
            self.cache[key] = entry
    
    def invalidate(self, key: str):
        """Remove from L1 cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'level': 'L1',
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }


class L2Cache:
    """
    L2 Cache: Shared remote cache (e.g., Redis, Memcached)
    
    Characteristics:
    - Fast access (microseconds to milliseconds)
    - Medium capacity (typically GBs)
    - Shared across processes/servers
    - Network latency
    """
    
    def __init__(self, capacity: int = 1000, latency_ms: float = 1.0):
        self.capacity = capacity
        self.latency_ms = latency_ms  # Simulated network latency
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get from L2 cache with simulated latency"""
        # Simulate network latency
        time.sleep(self.latency_ms / 1000)
        
        if key in self.cache:
            self.hits += 1
            entry = self.cache[key]
            entry.last_accessed = time.time()
            entry.access_count += 1
            return entry.value
        else:
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put into L2 cache"""
        # Simulate network latency
        time.sleep(self.latency_ms / 1000)
        
        if key in self.cache:
            entry = self.cache[key]
            entry.value = value
            entry.last_accessed = time.time()
        else:
            entry = CacheEntry(key=key, value=value)
            self.cache[key] = entry
    
    def invalidate(self, key: str):
        """Remove from L2 cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear all entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'level': 'L2',
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'latency_ms': self.latency_ms
        }


class L3DataStore:
    """
    L3 Cache: Persistent data store (database)
    
    Characteristics:
    - Slow access (milliseconds to seconds)
    - Large capacity (typically TBs)
    - Persistent storage
    - High latency
    """
    
    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms  # Simulated database latency
        self.data: Dict[str, Any] = {}
        self.reads = 0
        self.writes = 0
    
    def read(self, key: str) -> Optional[Any]:
        """Read from data store with simulated latency"""
        # Simulate database query latency
        time.sleep(self.latency_ms / 1000)
        
        self.reads += 1
        return self.data.get(key)
    
    def write(self, key: str, value: Any):
        """Write to data store"""
        # Simulate database write latency
        time.sleep(self.latency_ms / 1000)
        
        self.writes += 1
        self.data[key] = value
    
    def delete(self, key: str):
        """Delete from data store"""
        if key in self.data:
            del self.data[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data store statistics"""
        return {
            'level': 'L3 (DataStore)',
            'size': len(self.data),
            'reads': self.reads,
            'writes': self.writes,
            'latency_ms': self.latency_ms
        }


class MultiLevelCache:
    """
    Multi-level cache coordinating L1, L2, and L3
    
    Read path: L1 -> L2 -> L3
    - Check L1 first (fastest)
    - On L1 miss, check L2 and promote to L1
    - On L2 miss, check L3 and promote to L1+L2
    
    Write path: All levels
    - Write to L3 (persistence)
    - Write to L2 (shared cache)
    - Write to L1 (local cache)
    """
    
    def __init__(self):
        self.l1 = L1Cache(capacity=10)  # Small, fast
        self.l2 = L2Cache(capacity=100, latency_ms=1.0)  # Medium, shared
        self.l3 = L3DataStore(latency_ms=50.0)  # Large, slow
        
        self.total_gets = 0
        self.l1_hits = 0
        self.l2_hits = 0
        self.l3_hits = 0
        self.promotions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from multi-level cache.
        Tries L1 -> L2 -> L3, promoting upwards on cache miss.
        """
        self.total_gets += 1
        start_time = time.time()
        
        # Try L1 first
        value = self.l1.get(key)
        if value is not None:
            self.l1_hits += 1
            latency = (time.time() - start_time) * 1000
            return value, 'L1', latency
        
        # Try L2
        value = self.l2.get(key)
        if value is not None:
            self.l2_hits += 1
            # Promote to L1
            self.l1.put(key, value)
            self.promotions += 1
            latency = (time.time() - start_time) * 1000
            return value, 'L2', latency
        
        # Try L3
        value = self.l3.read(key)
        if value is not None:
            self.l3_hits += 1
            # Promote to L2 and L1
            self.l2.put(key, value)
            self.l1.put(key, value)
            self.promotions += 2
            latency = (time.time() - start_time) * 1000
            return value, 'L3', latency
        
        # Not found
        latency = (time.time() - start_time) * 1000
        return None, 'MISS', latency
    
    def put(self, key: str, value: Any):
        """
        Put value into all cache levels (write-through).
        Writes propagate through all levels for consistency.
        """
        # Write to L3 first (persistence)
        self.l3.write(key, value)
        
        # Write to L2 (shared cache)
        self.l2.put(key, value)
        
        # Write to L1 (local cache)
        self.l1.put(key, value)
    
    def invalidate(self, key: str):
        """Invalidate key from all cache levels"""
        self.l1.invalidate(key)
        self.l2.invalidate(key)
        self.l3.delete(key)
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache levels"""
        return {
            'total_gets': self.total_gets,
            'l1_hit_rate': (self.l1_hits / self.total_gets * 100) if self.total_gets > 0 else 0,
            'l2_hit_rate': (self.l2_hits / self.total_gets * 100) if self.total_gets > 0 else 0,
            'l3_hit_rate': (self.l3_hits / self.total_gets * 100) if self.total_gets > 0 else 0,
            'promotions': self.promotions,
            'l1_stats': self.l1.get_stats(),
            'l2_stats': self.l2.get_stats(),
            'l3_stats': self.l3.get_stats()
        }


def setup_cache_agent(state: MultiLevelCacheState):
    """Agent to set up multi-level cache"""
    operations = []
    results = []
    
    # Create multi-level cache
    cache = MultiLevelCache()
    
    operations.append("Initialized Multi-Level Cache:")
    operations.append(f"  L1 (Local Memory): capacity={cache.l1.capacity}, latency=~nanoseconds")
    operations.append(f"  L2 (Remote Cache): capacity={cache.l2.capacity}, latency={cache.l2.latency_ms}ms")
    operations.append(f"  L3 (Data Store): latency={cache.l3.latency_ms}ms")
    
    results.append("✓ Multi-level cache hierarchy initialized")
    
    # Store cache in state
    state['_cache'] = cache
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Multi-level cache setup complete"]
    }


def populate_data_agent(state: MultiLevelCacheState):
    """Agent to populate data through all cache levels"""
    cache = state['_cache']
    operations = []
    results = []
    
    # Populate data
    test_data = {
        "user:1": {"name": "Alice", "email": "alice@example.com"},
        "user:2": {"name": "Bob", "email": "bob@example.com"},
        "user:3": {"name": "Charlie", "email": "charlie@example.com"},
        "product:1": {"name": "Laptop", "price": 999.99},
        "product:2": {"name": "Mouse", "price": 29.99},
    }
    
    operations.append("\nPopulating data through all cache levels:")
    
    for key, value in test_data.items():
        cache.put(key, value)
        operations.append(f"  PUT {key} -> L1, L2, L3")
    
    results.append(f"✓ Populated {len(test_data)} items through all levels")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Data populated"]
    }


def cache_hit_demonstration_agent(state: MultiLevelCacheState):
    """Agent to demonstrate cache hits at different levels"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    operations.append("\nDemonstrating cache hit at different levels:")
    
    # L1 hit (hot data)
    operations.append("\n1. L1 Cache Hit (Hot Data):")
    value, level, latency = cache.get("user:1")
    operations.append(f"   GET user:1 -> Found in {level}, latency: {latency:.3f}ms")
    
    # Clear L1 to force L2 hit
    cache.l1.clear()
    operations.append("\n2. L2 Cache Hit (Clear L1):")
    value, level, latency = cache.get("user:2")
    operations.append(f"   GET user:2 -> Found in {level}, latency: {latency:.3f}ms")
    operations.append(f"   ↑ Promoted to L1")
    
    # Clear L1 and L2 to force L3 hit
    cache.l1.clear()
    cache.l2.clear()
    operations.append("\n3. L3 Cache Hit (Clear L1 & L2):")
    value, level, latency = cache.get("user:3")
    operations.append(f"   GET user:3 -> Found in {level}, latency: {latency:.3f}ms")
    operations.append(f"   ↑ Promoted to L1 & L2")
    
    # Cache miss
    operations.append("\n4. Cache Miss:")
    value, level, latency = cache.get("user:999")
    operations.append(f"   GET user:999 -> {level}, latency: {latency:.3f}ms")
    
    results.append("✓ Demonstrated cache hits at all levels")
    
    metrics.append("\nLatency by cache level:")
    metrics.append("  L1 hit: ~0.001ms (nanoseconds)")
    metrics.append(f"  L2 hit: ~{cache.l2.latency_ms}ms")
    metrics.append(f"  L3 hit: ~{cache.l3.latency_ms}ms")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Cache hit demonstration complete"]
    }


def hot_data_pattern_agent(state: MultiLevelCacheState):
    """Agent to demonstrate hot data staying in L1"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    operations.append("\nHot Data Access Pattern:")
    
    # Reset cache
    cache.l1.clear()
    cache.l2.clear()
    
    # Simulate hot data access (repeatedly access same key)
    hot_key = "product:1"
    
    operations.append(f"\nAccessing '{hot_key}' 5 times:")
    total_latency = 0
    
    for i in range(5):
        value, level, latency = cache.get(hot_key)
        operations.append(f"  Access {i+1}: {level} hit, {latency:.3f}ms")
        total_latency += latency
    
    avg_latency = total_latency / 5
    operations.append(f"\nAverage latency: {avg_latency:.3f}ms")
    
    # First access: L3 hit (~50ms)
    # Subsequent: L1 hit (~0.001ms)
    speedup = 50.0 / avg_latency
    results.append(f"✓ Hot data caching: {speedup:.1f}x speedup over L3-only")
    
    metrics.append(f"\nFirst access: L3 (~50ms)")
    metrics.append(f"Subsequent: L1 (~0.001ms)")
    metrics.append(f"Average: {avg_latency:.3f}ms")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Hot data pattern demonstrated"]
    }


def cache_invalidation_agent(state: MultiLevelCacheState):
    """Agent to demonstrate cache invalidation"""
    cache = state['_cache']
    operations = []
    results = []
    
    operations.append("\nCache Invalidation Demonstration:")
    
    # Put data
    key = "user:100"
    value = {"name": "Test User", "version": 1}
    cache.put(key, value)
    operations.append(f"\nPUT {key} (v1) -> All levels")
    
    # Read to warm up L1
    val, level, _ = cache.get(key)
    operations.append(f"GET {key} -> {level}")
    
    # Update data (invalidate and rewrite)
    operations.append(f"\nUpdate {key} (v2):")
    cache.invalidate(key)
    operations.append("  1. Invalidate from all levels")
    
    new_value = {"name": "Test User", "version": 2}
    cache.put(key, new_value)
    operations.append("  2. Write new value to all levels")
    
    # Verify update
    val, level, _ = cache.get(key)
    operations.append(f"  3. GET {key} -> version={val['version']}")
    
    results.append("✓ Cache invalidation maintains consistency across all levels")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Cache invalidation demonstrated"]
    }


def performance_statistics_agent(state: MultiLevelCacheState):
    """Agent to show performance statistics"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    stats = cache.get_all_stats()
    
    operations.append("\n" + "="*60)
    operations.append("PERFORMANCE STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nOverall Cache Performance:")
    operations.append(f"  Total requests: {stats['total_gets']}")
    operations.append(f"  L1 hit rate: {stats['l1_hit_rate']:.1f}%")
    operations.append(f"  L2 hit rate: {stats['l2_hit_rate']:.1f}%")
    operations.append(f"  L3 hit rate: {stats['l3_hit_rate']:.1f}%")
    operations.append(f"  Cache promotions: {stats['promotions']}")
    
    operations.append(f"\nL1 Cache (Local Memory):")
    l1 = stats['l1_stats']
    operations.append(f"  Size: {l1['size']}/{l1['capacity']}")
    operations.append(f"  Hits: {l1['hits']}, Misses: {l1['misses']}")
    operations.append(f"  Hit rate: {l1['hit_rate']:.1f}%")
    operations.append(f"  Evictions: {l1['evictions']}")
    
    operations.append(f"\nL2 Cache (Remote Shared):")
    l2 = stats['l2_stats']
    operations.append(f"  Size: {l2['size']}/{l2['capacity']}")
    operations.append(f"  Hits: {l2['hits']}, Misses: {l2['misses']}")
    operations.append(f"  Hit rate: {l2['hit_rate']:.1f}%")
    operations.append(f"  Latency: {l2['latency_ms']}ms")
    
    operations.append(f"\nL3 Data Store (Persistent):")
    l3 = stats['l3_stats']
    operations.append(f"  Size: {l3['size']}")
    operations.append(f"  Reads: {l3['reads']}, Writes: {l3['writes']}")
    operations.append(f"  Latency: {l3['latency_ms']}ms")
    
    metrics.append("\n📊 Multi-Level Cache Benefits:")
    metrics.append("  ✓ Reduced latency for hot data")
    metrics.append("  ✓ Load reduction on data store")
    metrics.append("  ✓ Scalable across multiple servers (L2 shared)")
    metrics.append("  ✓ Automatic promotion of popular data")
    
    results.append("✓ Multi-level cache demonstrated successfully")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Performance statistics generated"]
    }


def create_multi_level_cache_graph():
    """Create the multi-level cache workflow graph"""
    workflow = StateGraph(MultiLevelCacheState)
    
    # Add nodes
    workflow.add_node("setup", setup_cache_agent)
    workflow.add_node("populate", populate_data_agent)
    workflow.add_node("cache_hits", cache_hit_demonstration_agent)
    workflow.add_node("hot_data", hot_data_pattern_agent)
    workflow.add_node("invalidation", cache_invalidation_agent)
    workflow.add_node("statistics", performance_statistics_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "populate")
    workflow.add_edge("populate", "cache_hits")
    workflow.add_edge("cache_hits", "hot_data")
    workflow.add_edge("hot_data", "invalidation")
    workflow.add_edge("invalidation", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 199: Multi-Level Cache MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_multi_level_cache_graph()
    
    # Initialize state
    initial_state = {
        "cache_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("CACHE OPERATIONS")
    print("=" * 80)
    for op in final_state["cache_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    for metric in final_state["performance_metrics"]:
        print(metric)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Multi-Level Cache Pattern implemented with:

1. Cache Hierarchy:
   L1: Fast local memory cache (nanoseconds, MBs)
   L2: Shared remote cache (milliseconds, GBs)
   L3: Persistent data store (seconds, TBs)

2. Cache Promotion:
   - Data moves up hierarchy on cache miss
   - Hot data automatically promoted to L1
   - Frequently accessed data stays in faster levels

3. Write Strategy:
   - Write-through: Updates propagate to all levels
   - Ensures consistency across cache hierarchy
   - L3 written first for persistence

4. Performance Optimization:
   - L1 serves hot data with minimal latency
   - L2 reduces load on data store
   - L3 provides persistence and large capacity

Use Cases:
- Web Applications: Browser → CDN → Origin
- CPU Architecture: L1 → L2 → L3 → RAM
- Microservices: Local → Redis → Database
- CDN: Edge → Regional → Origin
- DNS: Resolver → ISP → Root servers

Real-World Examples:
- CPU Cache: L1 (32KB, 1ns) → L2 (256KB, 4ns) → L3 (8MB, 40ns) → RAM (GB, 100ns)
- Web Browser: Memory → Disk → HTTP Cache
- CDN: Edge (ms) → Regional (10ms) → Origin (100ms)
- Database: Buffer Pool → Query Cache → Disk

Performance Characteristics:
- L1 hit: ~1000x faster than L3
- L2 hit: ~50x faster than L3
- Automatic promotion reduces L3 load by 80-90%
- Hot data stays in L1 for maximum performance
""")


if __name__ == "__main__":
    main()

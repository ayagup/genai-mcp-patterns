"""
Pattern 191: Read-Through Cache MCP Pattern

This pattern demonstrates a caching strategy where the cache automatically loads
data from the backing store on a cache miss, making the cache transparent to the
application. The application always reads from cache, and the cache handles
loading missing data.

Key Concepts:
1. Transparent Loading: Cache loads data automatically on miss
2. Single Interface: Application only interacts with cache
3. Lazy Loading: Data loaded only when needed
4. Cache Population: Automatically fills cache on demand
5. Consistency: Cache and store stay synchronized on reads

Cache Operations:
- Cache Hit: Data found in cache, return immediately
- Cache Miss: Data not in cache, load from store, cache it, then return
- Automatic Loading: Cache handles loading logic internally
- Transparent: Application unaware of cache miss handling

Benefits:
- Simplicity: Application doesn't manage cache misses
- Lazy Loading: Only loads needed data
- Consistent Interface: Single API for all reads
- Automatic Population: Cache fills itself
- Reduced Complexity: No explicit cache management needed

Trade-offs:
- Initial Latency: First access slower (cache miss)
- Single Failure Point: Cache failure affects all reads
- Stale Data: Cache may not reflect latest updates
- Memory Usage: Cache grows with accessed data
- Complexity: Cache logic more complex

Use Cases:
- Read-Heavy Workloads: Frequently accessed data
- Database Caching: Reduce DB load
- API Responses: Cache external API calls
- Computed Results: Cache expensive calculations
- Session Data: User session information
- Configuration: App settings, feature flags

Comparison with Cache-Aside:
- Read-Through: Cache loads data automatically
- Cache-Aside: Application loads data manually
- Read-Through: Transparent to application
- Cache-Aside: Application aware of cache
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

# Define the state for read-through cache
class ReadThroughCacheState(TypedDict):
    """State for read-through cache operations"""
    cache_requests: List[Dict[str, Any]]
    cache_responses: List[Dict[str, Any]]
    cache_statistics: Dict[str, Any]
    backing_store_calls: int
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# READ-THROUGH CACHE IMPLEMENTATION
# ============================================================================

class BackingStore:
    """
    Simulates a backing data store (database, API, file system)
    
    This is the source of truth for data. The cache loads from here
    on cache misses.
    """
    
    def __init__(self):
        """Initialize with some data"""
        self.data = {
            "user:1": {"id": 1, "name": "Alice", "email": "alice@example.com"},
            "user:2": {"id": 2, "name": "Bob", "email": "bob@example.com"},
            "user:3": {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
            "product:101": {"id": 101, "name": "Laptop", "price": 999},
            "product:102": {"id": 102, "name": "Mouse", "price": 29},
            "product:103": {"id": 103, "name": "Keyboard", "price": 79},
        }
        self.read_count = 0
        self.latency_ms = 100  # Simulate 100ms latency
    
    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Read data from backing store
        
        Simulates database query or API call with latency
        """
        self.read_count += 1
        
        # Simulate latency
        time.sleep(self.latency_ms / 1000.0)
        
        return self.data.get(key)
    
    def get_read_count(self) -> int:
        """Get number of reads from backing store"""
        return self.read_count

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int

class ReadThroughCache:
    """
    Read-Through Cache Implementation
    
    On cache miss, automatically loads data from backing store,
    caches it, and returns to caller.
    
    Features:
    - Automatic loading on miss
    - TTL-based expiration
    - Cache statistics
    - Transparent to application
    """
    
    def __init__(self, backing_store: BackingStore, ttl_seconds: int = 300):
        """
        Initialize read-through cache
        
        Args:
            backing_store: Source of truth for data
            ttl_seconds: Time-to-live for cache entries (default 5 minutes)
        """
        self.backing_store = backing_store
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get data from cache (read-through)
        
        Flow:
        1. Check cache for key
        2. If found and not expired, return (cache hit)
        3. If not found or expired, load from backing store (cache miss)
        4. Store in cache
        5. Return data
        
        Application only calls this method - cache handles everything
        """
        # Check if key exists in cache
        if key in self.cache:
            entry = self.cache[key]
            
            # Check if entry is expired
            age = datetime.now() - entry.created_at
            if age.total_seconds() < self.ttl_seconds:
                # Cache hit!
                self.hits += 1
                
                # Update access metadata
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                return entry.value
            else:
                # Expired - treat as cache miss
                del self.cache[key]
        
        # Cache miss - load from backing store
        self.misses += 1
        
        # Load data from backing store
        data = self.backing_store.read(key)
        
        if data is not None:
            # Store in cache
            self.cache[key] = CacheEntry(
                key=key,
                value=data,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1
            )
        
        return data
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.get_hit_rate(), 2),
            "backing_store_reads": self.backing_store.get_read_count()
        }
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

# Global instances
backing_store = BackingStore()
read_through_cache = ReadThroughCache(backing_store, ttl_seconds=300)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_cache_requests(state: ReadThroughCacheState) -> ReadThroughCacheState:
    """Process cache read requests"""
    requests = state["cache_requests"]
    responses = []
    
    for request in requests:
        key = request.get("key")
        request_time = datetime.now()
        
        # Use read-through cache
        # Application just calls get() - cache handles loading automatically
        start_time = time.time()
        data = read_through_cache.get(key)
        elapsed_ms = (time.time() - start_time) * 1000
        
        responses.append({
            "key": key,
            "data": data,
            "found": data is not None,
            "latency_ms": round(elapsed_ms, 2),
            "request_time": request_time.isoformat()
        })
    
    return {
        "cache_responses": responses,
        "messages": [f"[Read-Through Cache] Processed {len(requests)} requests"]
    }

def collect_statistics(state: ReadThroughCacheState) -> ReadThroughCacheState:
    """Collect cache statistics"""
    stats = read_through_cache.get_statistics()
    
    return {
        "cache_statistics": stats,
        "backing_store_calls": backing_store.get_read_count(),
        "messages": [
            f"[Statistics] Hit Rate: {stats['hit_rate']}%, "
            f"Cache Size: {stats['cache_size']}, "
            f"Backing Store Reads: {stats['backing_store_reads']}"
        ]
    }

# ============================================================================
# BUILD THE CACHE GRAPH
# ============================================================================

def create_read_through_cache_graph():
    """
    Create a StateGraph demonstrating read-through cache pattern.
    
    Flow:
    1. Process cache requests (automatic loading on miss)
    2. Collect statistics
    """
    
    workflow = StateGraph(ReadThroughCacheState)
    
    # Add nodes
    workflow.add_node("process_requests", process_cache_requests)
    workflow.add_node("statistics", collect_statistics)
    
    # Define flow
    workflow.add_edge(START, "process_requests")
    workflow.add_edge("process_requests", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Read-Through Cache MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic Read-Through Cache
    print("\n" + "=" * 80)
    print("Example 1: Read-Through Cache with Automatic Loading")
    print("=" * 80)
    
    cache_graph = create_read_through_cache_graph()
    
    # First round: All cache misses (load from backing store)
    print("\n--- Round 1: Initial Requests (Cache Misses) ---")
    
    initial_state: ReadThroughCacheState = {
        "cache_requests": [
            {"key": "user:1"},
            {"key": "product:101"},
            {"key": "user:2"},
        ],
        "cache_responses": [],
        "cache_statistics": {},
        "backing_store_calls": 0,
        "messages": []
    }
    
    result = cache_graph.invoke(initial_state)
    
    print("\nCache Operations:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nResponses:")
    for response in result["cache_responses"]:
        print(f"  Key: {response['key']}")
        print(f"    Found: {response['found']}")
        print(f"    Latency: {response['latency_ms']}ms (includes backing store load)")
        print(f"    Data: {response['data']}")
    
    # Second round: Same requests (cache hits)
    print("\n--- Round 2: Same Requests (Cache Hits) ---")
    
    second_state: ReadThroughCacheState = {
        "cache_requests": [
            {"key": "user:1"},
            {"key": "product:101"},
            {"key": "user:2"},
        ],
        "cache_responses": [],
        "cache_statistics": {},
        "backing_store_calls": 0,
        "messages": []
    }
    
    result2 = cache_graph.invoke(second_state)
    
    print("\nCache Operations:")
    for msg in result2["messages"]:
        print(f"  {msg}")
    
    print("\nResponses:")
    for response in result2["cache_responses"]:
        print(f"  Key: {response['key']}")
        print(f"    Found: {response['found']}")
        print(f"    Latency: {response['latency_ms']}ms (from cache - much faster!)")
    
    print(f"\nNotice: Cache hits are ~100ms faster (no backing store access)")
    
    # Example 2: Mixed Hits and Misses
    print("\n" + "=" * 80)
    print("Example 2: Mixed Cache Hits and Misses")
    print("=" * 80)
    
    mixed_state: ReadThroughCacheState = {
        "cache_requests": [
            {"key": "user:1"},      # Hit (already cached)
            {"key": "user:3"},      # Miss (new key)
            {"key": "product:101"}, # Hit (already cached)
            {"key": "product:102"}, # Miss (new key)
        ],
        "cache_responses": [],
        "cache_statistics": {},
        "backing_store_calls": 0,
        "messages": []
    }
    
    result3 = cache_graph.invoke(mixed_state)
    
    print("\nCache Performance:")
    stats = result3["cache_statistics"]
    print(f"  Total Requests: {stats['hits'] + stats['misses']}")
    print(f"  Cache Hits: {stats['hits']}")
    print(f"  Cache Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']}%")
    print(f"  Backing Store Reads: {stats['backing_store_reads']}")
    
    # Example 3: Cache Statistics
    print("\n" + "=" * 80)
    print("Example 3: Cache Statistics and Efficiency")
    print("=" * 80)
    
    stats = read_through_cache.get_statistics()
    
    print("\nOverall Cache Performance:")
    print(f"  Cache Size: {stats['cache_size']} entries")
    print(f"  Total Hits: {stats['hits']}")
    print(f"  Total Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']}%")
    print(f"  Backing Store Reads: {stats['backing_store_reads']}")
    
    print("\nEfficiency Calculation:")
    print(f"  Without Cache: {stats['hits'] + stats['misses']} backing store reads")
    print(f"  With Cache: {stats['backing_store_reads']} backing store reads")
    print(f"  Reduction: {stats['hits']} reads saved ({stats['hit_rate']}%)")
    
    # Example 4: Read-Through vs Cache-Aside
    print("\n" + "=" * 80)
    print("Example 4: Read-Through vs Cache-Aside Comparison")
    print("=" * 80)
    
    print("\nRead-Through Cache:")
    print("  Application code:")
    print("    data = cache.get(key)  # That's it!")
    print("    # Cache automatically loads from backing store on miss")
    print()
    print("  Pros:")
    print("    ✓ Simpler application code")
    print("    ✓ Transparent caching")
    print("    ✓ Consistent interface")
    print("    ✓ Automatic cache population")
    
    print("\nCache-Aside Pattern:")
    print("  Application code:")
    print("    data = cache.get(key)")
    print("    if data is None:")
    print("        data = backing_store.read(key)  # Manual load")
    print("        cache.put(key, data)            # Manual cache")
    print()
    print("  Pros:")
    print("    ✓ More control over loading")
    print("    ✓ Can handle errors differently")
    print("    ✓ Application aware of cache")
    
    # Example 5: TTL Expiration Demo
    print("\n" + "=" * 80)
    print("Example 5: TTL-Based Expiration")
    print("=" * 80)
    
    # Create cache with short TTL for demo
    short_ttl_cache = ReadThroughCache(backing_store, ttl_seconds=2)
    
    print("\nCache with 2-second TTL:")
    print("  1. First access: Cache miss (load from store)")
    data1 = short_ttl_cache.get("user:1")
    print(f"     Result: {data1}")
    
    print("\n  2. Immediate second access: Cache hit (from cache)")
    data2 = short_ttl_cache.get("user:1")
    print(f"     Result: {data2}")
    
    print("\n  3. Wait 3 seconds...")
    time.sleep(3)
    
    print("  4. Access after TTL: Cache miss (expired, reload)")
    data3 = short_ttl_cache.get("user:1")
    print(f"     Result: {data3}")
    
    print(f"\n  Statistics: {short_ttl_cache.get_statistics()}")
    
    print("\n" + "=" * 80)
    print("Read-Through Cache Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Read-through cache loads data automatically on cache miss
  • Application code is simple - just call cache.get(key)
  • Cache transparently handles backing store access
  • First access is slow (cache miss + load)
  • Subsequent accesses are fast (cache hit)
  • TTL-based expiration prevents stale data
  • Reduces backing store load significantly
  • Ideal for read-heavy workloads
  • Simpler than cache-aside but less flexible
  • Use when: Application doesn't need to control loading logic
  • Tools: Redis, Memcached, Caffeine, Ehcache
  • Best practices: Set appropriate TTL, monitor hit rate
    """)

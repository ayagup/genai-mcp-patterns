"""
Pattern 192: Write-Through Cache MCP Pattern

This pattern demonstrates a caching strategy where writes are made to both the
cache and the backing store synchronously. Data is written to cache first, then
immediately to the backing store before returning success.

Key Concepts:
1. Synchronous Writes: Cache and store updated together
2. Strong Consistency: Cache always matches backing store
3. Write Latency: Slower writes (both cache + store)
4. Data Safety: No data loss on cache failure
5. Transparent: Application unaware of dual writes

Write Operations:
- Write to cache first (fast)
- Write to backing store second (slower)
- Return success only after both complete
- If backing store write fails, invalidate cache entry

Benefits:
- Strong Consistency: Cache and store always in sync
- Data Safety: No data loss on cache failure
- Simple Reads: Always return latest data
- No Stale Data: Cache reflects all writes immediately
- Durability: Data persisted to backing store

Trade-offs:
- Write Latency: Slower writes (cache + store)
- Write Amplification: Every write hits both systems
- Reduced Write Throughput: Limited by backing store speed
- Complexity: Must handle partial failures
- No Write Coalescing: Can't batch writes

Use Cases:
- Strong Consistency Required: Financial transactions
- Data Durability Critical: User preferences, settings
- Read-Heavy, Write-Moderate: Product catalogs
- Audit Requirements: Must persist all writes
- Session Management: User sessions with persistence
- Configuration Updates: System settings

Comparison with Write-Back:
- Write-Through: Synchronous, slower, consistent
- Write-Back: Asynchronous, faster, eventual consistency
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
import time

# Define the state for write-through cache
class WriteThroughCacheState(TypedDict):
    """State for write-through cache operations"""
    write_operations: List[Dict[str, Any]]
    write_results: List[Dict[str, Any]]
    read_operations: List[Dict[str, Any]]
    read_results: List[Dict[str, Any]]
    cache_statistics: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# WRITE-THROUGH CACHE IMPLEMENTATION
# ============================================================================

class PersistentStore:
    """
    Simulates a persistent backing store (database, disk)
    
    This is the durable storage - data here survives cache failures
    """
    
    def __init__(self):
        """Initialize persistent store"""
        self.data: Dict[str, Dict[str, Any]] = {}
        self.write_count = 0
        self.read_count = 0
        self.write_latency_ms = 50  # Simulate 50ms write latency
        self.read_latency_ms = 100  # Simulate 100ms read latency
    
    def write(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Write data to persistent store
        
        Simulates database write with latency
        """
        self.write_count += 1
        
        # Simulate write latency
        time.sleep(self.write_latency_ms / 1000.0)
        
        self.data[key] = value.copy()
        return True
    
    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """Read data from persistent store"""
        self.read_count += 1
        
        # Simulate read latency
        time.sleep(self.read_latency_ms / 1000.0)
        
        return self.data.get(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics"""
        return {
            "write_count": self.write_count,
            "read_count": self.read_count,
            "total_records": len(self.data)
        }

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    updated_at: datetime
    version: int

class WriteThroughCache:
    """
    Write-Through Cache Implementation
    
    Writes go to both cache and backing store synchronously.
    Cache is always consistent with backing store.
    
    Features:
    - Synchronous dual writes
    - Strong consistency
    - Automatic cache invalidation on write failure
    - Read-through on cache miss
    """
    
    def __init__(self, backing_store: PersistentStore):
        """
        Initialize write-through cache
        
        Args:
            backing_store: Persistent storage backend
        """
        self.backing_store = backing_store
        self.cache: Dict[str, CacheEntry] = {}
        self.write_count = 0
        self.cache_read_hits = 0
        self.cache_read_misses = 0
    
    def write(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Write data (write-through)
        
        Flow:
        1. Write to cache (fast, in-memory)
        2. Write to backing store (slower, persistent)
        3. If backing store write succeeds, keep cache entry
        4. If backing store write fails, remove from cache (maintain consistency)
        5. Return success/failure
        
        Both writes complete before returning - SYNCHRONOUS
        """
        self.write_count += 1
        start_time = time.time()
        
        # Step 1: Write to cache first (optimistic)
        existing_entry = self.cache.get(key)
        version = existing_entry.version + 1 if existing_entry else 1
        
        cache_entry = CacheEntry(
            key=key,
            value=value.copy(),
            created_at=existing_entry.created_at if existing_entry else datetime.now(),
            updated_at=datetime.now(),
            version=version
        )
        
        self.cache[key] = cache_entry
        
        # Step 2: Write to backing store (synchronous)
        try:
            success = self.backing_store.write(key, value)
            
            if not success:
                # Backing store write failed - remove from cache
                # to maintain consistency
                if key in self.cache:
                    del self.cache[key]
                return False
            
            # Both writes succeeded
            return True
            
        except Exception as e:
            # Backing store write failed - invalidate cache entry
            if key in self.cache:
                del self.cache[key]
            raise e
    
    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Read data (read-through)
        
        Flow:
        1. Check cache
        2. If found, return (cache hit)
        3. If not found, load from backing store (cache miss)
        4. Store in cache
        5. Return data
        """
        # Check cache first
        if key in self.cache:
            self.cache_read_hits += 1
            return self.cache[key].value.copy()
        
        # Cache miss - load from backing store
        self.cache_read_misses += 1
        data = self.backing_store.read(key)
        
        if data is not None:
            # Populate cache
            self.cache[key] = CacheEntry(
                key=key,
                value=data.copy(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version=1
            )
        
        return data
    
    def delete(self, key: str) -> bool:
        """
        Delete data (write-through)
        
        Remove from both cache and backing store
        """
        # Remove from cache
        if key in self.cache:
            del self.cache[key]
        
        # Remove from backing store
        if key in self.backing_store.data:
            del self.backing_store.data[key]
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_reads = self.cache_read_hits + self.cache_read_misses
        hit_rate = (self.cache_read_hits / total_reads * 100) if total_reads > 0 else 0.0
        
        return {
            "cache_size": len(self.cache),
            "write_count": self.write_count,
            "cache_read_hits": self.cache_read_hits,
            "cache_read_misses": self.cache_read_misses,
            "read_hit_rate": round(hit_rate, 2),
            "backing_store": self.backing_store.get_statistics()
        }

# Global instances
persistent_store = PersistentStore()
write_through_cache = WriteThroughCache(persistent_store)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_writes(state: WriteThroughCacheState) -> WriteThroughCacheState:
    """Process write operations"""
    write_ops = state["write_operations"]
    results = []
    
    for op in write_ops:
        key = op.get("key")
        value = op.get("value")
        
        start_time = time.time()
        success = write_through_cache.write(key, value)
        elapsed_ms = (time.time() - start_time) * 1000
        
        results.append({
            "key": key,
            "success": success,
            "latency_ms": round(elapsed_ms, 2),
            "operation": "write"
        })
    
    return {
        "write_results": results,
        "messages": [f"[Write-Through] Processed {len(write_ops)} writes (cache + store)"]
    }

def process_reads(state: WriteThroughCacheState) -> WriteThroughCacheState:
    """Process read operations"""
    read_ops = state["read_operations"]
    results = []
    
    for op in read_ops:
        key = op.get("key")
        
        start_time = time.time()
        data = write_through_cache.read(key)
        elapsed_ms = (time.time() - start_time) * 1000
        
        results.append({
            "key": key,
            "data": data,
            "found": data is not None,
            "latency_ms": round(elapsed_ms, 2),
            "operation": "read"
        })
    
    return {
        "read_results": results,
        "messages": [f"[Read] Processed {len(read_ops)} reads"]
    }

def collect_statistics(state: WriteThroughCacheState) -> WriteThroughCacheState:
    """Collect cache statistics"""
    stats = write_through_cache.get_statistics()
    
    return {
        "cache_statistics": stats,
        "messages": [
            f"[Statistics] Cache Size: {stats['cache_size']}, "
            f"Writes: {stats['write_count']}, "
            f"Read Hit Rate: {stats['read_hit_rate']}%"
        ]
    }

# ============================================================================
# BUILD THE CACHE GRAPH
# ============================================================================

def create_write_through_cache_graph():
    """
    Create a StateGraph demonstrating write-through cache pattern.
    
    Flow:
    1. Process writes (synchronous to cache + store)
    2. Process reads (from cache or store)
    3. Collect statistics
    """
    
    workflow = StateGraph(WriteThroughCacheState)
    
    # Add nodes
    workflow.add_node("writes", process_writes)
    workflow.add_node("reads", process_reads)
    workflow.add_node("statistics", collect_statistics)
    
    # Define flow
    workflow.add_edge(START, "writes")
    workflow.add_edge("writes", "reads")
    workflow.add_edge("reads", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Write-Through Cache MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic Write-Through Cache
    print("\n" + "=" * 80)
    print("Example 1: Write-Through Cache - Synchronous Writes")
    print("=" * 80)
    
    cache_graph = create_write_through_cache_graph()
    
    initial_state: WriteThroughCacheState = {
        "write_operations": [
            {"key": "user:1", "value": {"name": "Alice", "email": "alice@example.com"}},
            {"key": "user:2", "value": {"name": "Bob", "email": "bob@example.com"}},
            {"key": "product:101", "value": {"name": "Laptop", "price": 999}},
        ],
        "write_results": [],
        "read_operations": [
            {"key": "user:1"},
            {"key": "user:2"},
            {"key": "product:101"},
        ],
        "read_results": [],
        "cache_statistics": {},
        "messages": []
    }
    
    result = cache_graph.invoke(initial_state)
    
    print("\nOperations Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nWrite Results (synchronous to cache + backing store):")
    for write_result in result["write_results"]:
        print(f"  Key: {write_result['key']}")
        print(f"    Success: {write_result['success']}")
        print(f"    Latency: {write_result['latency_ms']}ms (includes backing store write)")
    
    print("\nRead Results (from cache - fast!):")
    for read_result in result["read_results"]:
        print(f"  Key: {read_result['key']}")
        print(f"    Latency: {read_result['latency_ms']}ms")
        print(f"    Data: {read_result['data']}")
    
    # Example 2: Consistency Guarantee
    print("\n" + "=" * 80)
    print("Example 2: Strong Consistency - Cache Matches Store")
    print("=" * 80)
    
    # Write to cache
    print("\n1. Write user:3 to cache (write-through):")
    start = time.time()
    success = write_through_cache.write("user:3", {"name": "Charlie", "age": 30})
    write_time = (time.time() - start) * 1000
    print(f"   Success: {success}, Latency: {write_time:.2f}ms")
    
    # Read from cache
    print("\n2. Read user:3 from cache:")
    cache_data = write_through_cache.read("user:3")
    print(f"   Cache data: {cache_data}")
    
    # Read directly from backing store
    print("\n3. Read user:3 from backing store:")
    store_data = persistent_store.read("user:3")
    print(f"   Store data: {store_data}")
    
    print("\n4. Verify consistency:")
    print(f"   Cache == Store: {cache_data == store_data}")
    print("   ✓ Cache and backing store are always in sync!")
    
    # Example 3: Write Latency Analysis
    print("\n" + "=" * 80)
    print("Example 3: Write Latency Breakdown")
    print("=" * 80)
    
    print("\nWrite-Through Cache Latency:")
    print("  Total Write Time = Cache Write + Backing Store Write")
    print(f"  = ~0ms (in-memory) + ~50ms (persistent store)")
    print(f"  = ~50ms total")
    print()
    print("  Trade-off: Slower writes for strong consistency")
    
    # Example 4: Update Operations
    print("\n" + "=" * 80)
    print("Example 4: Update Operations with Versioning")
    print("=" * 80)
    
    print("\n1. Initial write:")
    write_through_cache.write("config:app", {"theme": "light", "lang": "en"})
    data1 = write_through_cache.read("config:app")
    print(f"   Config: {data1}")
    
    print("\n2. Update config (write-through):")
    write_through_cache.write("config:app", {"theme": "dark", "lang": "en"})
    data2 = write_through_cache.read("config:app")
    print(f"   Updated Config: {data2}")
    
    print("\n3. Verify backing store has latest:")
    store_data = persistent_store.read("config:app")
    print(f"   Store has: {store_data}")
    print(f"   ✓ Backing store immediately reflects updates")
    
    # Example 5: Statistics
    print("\n" + "=" * 80)
    print("Example 5: Cache Statistics")
    print("=" * 80)
    
    stats = write_through_cache.get_statistics()
    
    print("\nCache Performance:")
    print(f"  Cache Size: {stats['cache_size']} entries")
    print(f"  Total Writes: {stats['write_count']}")
    print(f"  Read Hits: {stats['cache_read_hits']}")
    print(f"  Read Misses: {stats['cache_read_misses']}")
    print(f"  Read Hit Rate: {stats['read_hit_rate']}%")
    
    print("\nBacking Store:")
    backing_stats = stats['backing_store']
    print(f"  Total Writes: {backing_stats['write_count']}")
    print(f"  Total Reads: {backing_stats['read_count']}")
    print(f"  Total Records: {backing_stats['total_records']}")
    
    print("\nObservation:")
    print(f"  Cache writes ({stats['write_count']}) == Store writes ({backing_stats['write_count']})")
    print("  Every cache write triggers a store write (write-through)")
    
    # Example 6: Comparison with Other Patterns
    print("\n" + "=" * 80)
    print("Example 6: Write-Through vs Write-Back vs Cache-Aside")
    print("=" * 80)
    
    print("\nWrite-Through (this pattern):")
    print("  ✓ Synchronous writes to cache + store")
    print("  ✓ Strong consistency guarantee")
    print("  ✓ No data loss on cache failure")
    print("  ✗ Slower writes (limited by store latency)")
    print("  ✗ Write amplification")
    print("  Use when: Consistency > Write performance")
    
    print("\nWrite-Back (write-behind):")
    print("  ✓ Fast writes (cache only, async to store)")
    print("  ✓ Write coalescing possible")
    print("  ✓ Better write throughput")
    print("  ✗ Eventual consistency")
    print("  ✗ Potential data loss on cache failure")
    print("  Use when: Write performance > Consistency")
    
    print("\nCache-Aside:")
    print("  ✓ Application controls caching")
    print("  ✓ Flexible invalidation strategies")
    print("  ✗ Application must manage cache")
    print("  ✗ More complex application code")
    print("  Use when: Need fine-grained control")
    
    print("\n" + "=" * 80)
    print("Write-Through Cache Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Write-through cache writes to cache AND backing store synchronously
  • Strong consistency: Cache always matches backing store
  • Writes are slower (cache + store latency)
  • Reads are fast (from cache after first write)
  • No data loss on cache failure (persisted to store)
  • Ideal for data durability requirements
  • Trade-off: Write latency for consistency
  • Use cases: User settings, financial data, audit trails
  • Tools: Redis (can be configured for write-through), Ehcache
  • Best practices: Monitor write latency, use for critical data
  • Comparison: Slower writes than write-back, stronger consistency
    """)

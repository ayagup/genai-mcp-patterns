"""
Pattern 193: Write-Back (Write-Behind) Cache MCP Pattern

This pattern demonstrates a caching strategy where writes are made to the cache
immediately and to the backing store asynchronously. This provides fast writes
with eventual consistency between cache and backing store.

Key Concepts:
1. Asynchronous Writes: Write to cache immediately, store later
2. Write Coalescing: Multiple writes can be batched
3. Fast Writes: Return immediately after cache write
4. Eventual Consistency: Cache ahead of backing store temporarily
5. Write Buffering: Queue writes for batch processing

Write Operations:
- Write to cache immediately (fast)
- Queue write for later (async)
- Return success immediately
- Background process flushes to backing store
- Multiple writes to same key can be coalesced

Benefits:
- Fast Writes: No waiting for backing store
- Higher Write Throughput: Cache limited, not store limited
- Write Coalescing: Batch multiple writes
- Reduced Store Load: Fewer write operations
- Better User Experience: Faster response times

Trade-offs:
- Eventual Consistency: Cache ahead of store
- Data Loss Risk: Cache failure before flush loses data
- Complexity: Need background flush process
- Memory Usage: Buffered writes consume memory
- Ordering: May need to maintain write order

Use Cases:
- Write-Heavy Workloads: High write throughput needed
- Gaming: Player state, leaderboards
- Analytics: Event tracking, metrics
- Logging: Application logs, audit trails
- Social Media: Likes, comments, views
- IoT: Sensor data collection

Comparison:
- Write-Through: Synchronous, slower, consistent
- Write-Back: Asynchronous, faster, eventual consistency
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass, field
from datetime import datetime
import time
import threading
from queue import Queue

# Define the state for write-back cache
class WriteBackCacheState(TypedDict):
    """State for write-back cache operations"""
    write_operations: List[Dict[str, Any]]
    write_results: List[Dict[str, Any]]
    read_operations: List[Dict[str, Any]]
    read_results: List[Dict[str, Any]]
    cache_statistics: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# WRITE-BACK CACHE IMPLEMENTATION
# ============================================================================

class BackingStore:
    """Simulates a persistent backing store"""
    
    def __init__(self):
        self.data: Dict[str, Dict[str, Any]] = {}
        self.write_count = 0
        self.write_latency_ms = 50
    
    def write(self, key: str, value: Dict[str, Any]) -> bool:
        """Write data to backing store"""
        self.write_count += 1
        time.sleep(self.write_latency_ms / 1000.0)
        self.data[key] = value.copy()
        return True
    
    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """Read data from backing store"""
        return self.data.get(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "write_count": self.write_count,
            "total_records": len(self.data)
        }

@dataclass
class DirtyEntry:
    """Entry that needs to be flushed to backing store"""
    key: str
    value: Dict[str, Any]
    timestamp: datetime
    write_count: int = 1  # Number of writes to this key since last flush

@dataclass
class CacheEntry:
    """Cache entry with dirty flag"""
    key: str
    value: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    is_dirty: bool = False  # True if cache ahead of backing store
    version: int = 1

class WriteBackCache:
    """
    Write-Back (Write-Behind) Cache Implementation
    
    Writes go to cache immediately, then asynchronously to backing store.
    Provides fast writes with eventual consistency.
    
    Features:
    - Immediate cache writes
    - Asynchronous backing store writes
    - Write coalescing
    - Periodic flush
    - Dirty tracking
    """
    
    def __init__(self, backing_store: BackingStore, flush_interval_seconds: int = 5):
        """
        Initialize write-back cache
        
        Args:
            backing_store: Persistent storage backend
            flush_interval_seconds: How often to flush dirty entries
        """
        self.backing_store = backing_store
        self.flush_interval = flush_interval_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.dirty_queue: Dict[str, DirtyEntry] = {}  # Entries to flush
        self.write_count = 0
        self.flush_count = 0
        self.coalesced_writes = 0
        self.lock = threading.Lock()
        
        # Background flush thread
        self.flush_thread = None
        self.running = False
    
    def start_flush_thread(self):
        """Start background thread to flush dirty entries"""
        self.running = True
        self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self.flush_thread.start()
    
    def stop_flush_thread(self):
        """Stop background flush thread"""
        self.running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=2)
    
    def _flush_worker(self):
        """Background worker that periodically flushes dirty entries"""
        while self.running:
            time.sleep(self.flush_interval)
            self.flush_dirty_entries()
    
    def write(self, key: str, value: Dict[str, Any]) -> bool:
        """
        Write data (write-back)
        
        Flow:
        1. Write to cache immediately (fast)
        2. Mark entry as dirty
        3. Add to dirty queue for async flush
        4. Return success immediately (don't wait for backing store)
        
        Write-back: ASYNCHRONOUS to backing store
        """
        with self.lock:
            self.write_count += 1
            
            # Write to cache immediately
            existing_entry = self.cache.get(key)
            version = existing_entry.version + 1 if existing_entry else 1
            
            cache_entry = CacheEntry(
                key=key,
                value=value.copy(),
                created_at=existing_entry.created_at if existing_entry else datetime.now(),
                updated_at=datetime.now(),
                is_dirty=True,  # Mark as dirty (cache ahead of store)
                version=version
            )
            
            self.cache[key] = cache_entry
            
            # Add to dirty queue for async flush
            if key in self.dirty_queue:
                # Key already in queue - coalesce writes
                self.dirty_queue[key].value = value.copy()
                self.dirty_queue[key].write_count += 1
                self.coalesced_writes += 1
            else:
                # New dirty entry
                self.dirty_queue[key] = DirtyEntry(
                    key=key,
                    value=value.copy(),
                    timestamp=datetime.now()
                )
            
            # Return immediately (don't wait for backing store)
            return True
    
    def flush_dirty_entries(self) -> int:
        """
        Flush dirty entries to backing store
        
        Called periodically by background thread or manually
        Returns number of entries flushed
        """
        with self.lock:
            if not self.dirty_queue:
                return 0
            
            # Get all dirty entries
            entries_to_flush = list(self.dirty_queue.values())
            
            # Clear dirty queue
            self.dirty_queue.clear()
        
        # Flush to backing store (outside lock to avoid blocking writes)
        flushed_count = 0
        for entry in entries_to_flush:
            try:
                success = self.backing_store.write(entry.key, entry.value)
                if success:
                    # Mark cache entry as clean
                    with self.lock:
                        if entry.key in self.cache:
                            self.cache[entry.key].is_dirty = False
                    flushed_count += 1
            except Exception as e:
                # Re-queue failed writes
                with self.lock:
                    self.dirty_queue[entry.key] = entry
        
        self.flush_count += flushed_count
        return flushed_count
    
    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Read data
        
        Always read from cache if available (cache is most up-to-date)
        """
        with self.lock:
            if key in self.cache:
                return self.cache[key].value.copy()
        
        # Cache miss - load from backing store
        data = self.backing_store.read(key)
        
        if data is not None:
            with self.lock:
                self.cache[key] = CacheEntry(
                    key=key,
                    value=data.copy(),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                    is_dirty=False,
                    version=1
                )
        
        return data
    
    def get_dirty_count(self) -> int:
        """Get number of dirty entries waiting to be flushed"""
        with self.lock:
            return len(self.dirty_queue)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            dirty_count = len(self.dirty_queue)
            cache_size = len(self.cache)
        
        return {
            "cache_size": cache_size,
            "dirty_entries": dirty_count,
            "write_count": self.write_count,
            "flush_count": self.flush_count,
            "coalesced_writes": self.coalesced_writes,
            "backing_store": self.backing_store.get_statistics()
        }

# Global instances
backing_store = BackingStore()
write_back_cache = WriteBackCache(backing_store, flush_interval_seconds=2)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_writes(state: WriteBackCacheState) -> WriteBackCacheState:
    """Process write operations"""
    write_ops = state["write_operations"]
    results = []
    
    for op in write_ops:
        key = op.get("key")
        value = op.get("value")
        
        start_time = time.time()
        success = write_back_cache.write(key, value)
        elapsed_ms = (time.time() - start_time) * 1000
        
        results.append({
            "key": key,
            "success": success,
            "latency_ms": round(elapsed_ms, 2),
            "operation": "write"
        })
    
    return {
        "write_results": results,
        "messages": [f"[Write-Back] Processed {len(write_ops)} writes (async to store)"]
    }

def process_reads(state: WriteBackCacheState) -> WriteBackCacheState:
    """Process read operations"""
    read_ops = state["read_operations"]
    results = []
    
    for op in read_ops:
        key = op.get("key")
        
        start_time = time.time()
        data = write_back_cache.read(key)
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

def collect_statistics(state: WriteBackCacheState) -> WriteBackCacheState:
    """Collect cache statistics"""
    stats = write_back_cache.get_statistics()
    
    return {
        "cache_statistics": stats,
        "messages": [
            f"[Statistics] Writes: {stats['write_count']}, "
            f"Flushed: {stats['flush_count']}, "
            f"Dirty: {stats['dirty_entries']}, "
            f"Coalesced: {stats['coalesced_writes']}"
        ]
    }

# ============================================================================
# BUILD THE CACHE GRAPH
# ============================================================================

def create_write_back_cache_graph():
    """
    Create a StateGraph demonstrating write-back cache pattern.
    
    Flow:
    1. Process writes (immediate to cache, async to store)
    2. Process reads
    3. Collect statistics
    """
    
    workflow = StateGraph(WriteBackCacheState)
    
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
    print("Write-Back (Write-Behind) Cache MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Fast Writes with Write-Back
    print("\n" + "=" * 80)
    print("Example 1: Write-Back Cache - Fast Asynchronous Writes")
    print("=" * 80)
    
    cache_graph = create_write_back_cache_graph()
    
    initial_state: WriteBackCacheState = {
        "write_operations": [
            {"key": "score:player1", "value": {"score": 100, "level": 5}},
            {"key": "score:player2", "value": {"score": 200, "level": 7}},
            {"key": "score:player3", "value": {"score": 150, "level": 6}},
        ],
        "write_results": [],
        "read_operations": [
            {"key": "score:player1"},
            {"key": "score:player2"},
        ],
        "read_results": [],
        "cache_statistics": {},
        "messages": []
    }
    
    result = cache_graph.invoke(initial_state)
    
    print("\nOperations Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nWrite Results (immediate - no wait for backing store!):")
    for write_result in result["write_results"]:
        print(f"  Key: {write_result['key']}")
        print(f"    Latency: {write_result['latency_ms']}ms (cache only - fast!)")
    
    print("\nRead Results (from cache):")
    for read_result in result["read_results"]:
        print(f"  Key: {read_result['key']}: {read_result['data']}")
    
    print("\nCache State:")
    stats = result["cache_statistics"]
    print(f"  Dirty Entries (not yet flushed): {stats['dirty_entries']}")
    print(f"  Backing Store Writes: {stats['backing_store']['write_count']}")
    print("  → Writes are in cache but not yet persisted!")
    
    # Example 2: Write Coalescing
    print("\n" + "=" * 80)
    print("Example 2: Write Coalescing - Multiple Writes to Same Key")
    print("=" * 80)
    
    print("\nMultiple updates to same key:")
    write_back_cache.write("counter:pageviews", {"count": 1})
    print("  Write 1: count=1")
    write_back_cache.write("counter:pageviews", {"count": 2})
    print("  Write 2: count=2")
    write_back_cache.write("counter:pageviews", {"count": 3})
    print("  Write 3: count=3")
    
    print("\nBefore flush:")
    print(f"  Cache value: {write_back_cache.read('counter:pageviews')}")
    print(f"  Backing store value: {backing_store.read('counter:pageviews')}")
    print(f"  Dirty entries: {write_back_cache.get_dirty_count()}")
    
    print("\nFlush to backing store:")
    flushed = write_back_cache.flush_dirty_entries()
    print(f"  Flushed {flushed} entries")
    
    print("\nAfter flush:")
    print(f"  Cache value: {write_back_cache.read('counter:pageviews')}")
    print(f"  Backing store value: {backing_store.read('counter:pageviews')}")
    print(f"  ✓ Only 1 write to backing store instead of 3 (coalescing!)")
    
    stats = write_back_cache.get_statistics()
    print(f"\n  Coalesced writes: {stats['coalesced_writes']}")
    
    # Example 3: Latency Comparison
    print("\n" + "=" * 80)
    print("Example 3: Write Latency Comparison")
    print("=" * 80)
    
    print("\nWrite-Back (this pattern):")
    start = time.time()
    write_back_cache.write("test:wb", {"value": "fast"})
    wb_latency = (time.time() - start) * 1000
    print(f"  Write latency: {wb_latency:.2f}ms")
    print("  (Immediate return - cache only)")
    
    print("\nWrite-Through (synchronous):")
    print(f"  Write latency: ~50ms")
    print("  (Must wait for cache + backing store)")
    
    print(f"\nSpeedup: {50 / wb_latency:.1f}x faster!")
    print("  Trade-off: Eventual consistency vs strong consistency")
    
    # Example 4: Periodic Flush
    print("\n" + "=" * 80)
    print("Example 4: Periodic Flush Demonstration")
    print("=" * 80)
    
    print("\n1. Write multiple entries:")
    for i in range(5):
        write_back_cache.write(f"event:{i}", {"type": "click", "time": i})
    
    print(f"   Wrote 5 entries")
    print(f"   Dirty entries: {write_back_cache.get_dirty_count()}")
    
    print("\n2. Wait for automatic flush (2 seconds)...")
    write_back_cache.start_flush_thread()
    time.sleep(3)
    
    print(f"   Dirty entries after flush: {write_back_cache.get_dirty_count()}")
    print("   ✓ Background thread flushed dirty entries automatically")
    
    write_back_cache.stop_flush_thread()
    
    # Example 5: Statistics
    print("\n" + "=" * 80)
    print("Example 5: Cache Statistics and Efficiency")
    print("=" * 80)
    
    stats = write_back_cache.get_statistics()
    
    print("\nCache Performance:")
    print(f"  Cache Size: {stats['cache_size']} entries")
    print(f"  Total Writes: {stats['write_count']}")
    print(f"  Flushed to Store: {stats['flush_count']}")
    print(f"  Coalesced Writes: {stats['coalesced_writes']}")
    print(f"  Dirty Entries: {stats['dirty_entries']}")
    
    print("\nBacking Store:")
    backing_stats = stats['backing_store']
    print(f"  Total Writes: {backing_stats['write_count']}")
    print(f"  Total Records: {backing_stats['total_records']}")
    
    print("\nEfficiency:")
    if stats['write_count'] > 0:
        reduction = ((stats['write_count'] - backing_stats['write_count']) / 
                    stats['write_count'] * 100)
        print(f"  Cache writes: {stats['write_count']}")
        print(f"  Store writes: {backing_stats['write_count']}")
        print(f"  Reduction: {reduction:.1f}% fewer backing store writes!")
    
    # Example 6: Comparison
    print("\n" + "=" * 80)
    print("Example 6: Pattern Comparison")
    print("=" * 80)
    
    print("\nWrite-Back (Write-Behind) - This Pattern:")
    print("  ✓ Fast writes (~0-1ms, cache only)")
    print("  ✓ High write throughput")
    print("  ✓ Write coalescing reduces store load")
    print("  ✓ Better user experience (faster)")
    print("  ✗ Eventual consistency")
    print("  ✗ Risk of data loss on cache failure")
    print("  Use when: Write performance > Immediate consistency")
    
    print("\nWrite-Through:")
    print("  ✓ Strong consistency")
    print("  ✓ No data loss risk")
    print("  ✗ Slower writes (~50ms+, cache + store)")
    print("  ✗ Lower write throughput")
    print("  Use when: Consistency > Write performance")
    
    print("\n" + "=" * 80)
    print("Write-Back Cache Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Write-back cache writes to cache immediately, store asynchronously
  • Fast writes: ~0-1ms (cache only) vs ~50ms+ (write-through)
  • Eventual consistency: Cache temporarily ahead of backing store
  • Write coalescing: Multiple writes to same key batched
  • Periodic flush: Background thread syncs to backing store
  • Higher throughput: Not limited by backing store speed
  • Data loss risk: Cache failure before flush loses data
  • Use cases: Gaming, analytics, logging, social media
  • Tools: Redis, Memcached with write-behind, Ehcache
  • Best practices: Regular flushes, monitor dirty queue size
  • Trade-off: Write speed for eventual consistency
    """)

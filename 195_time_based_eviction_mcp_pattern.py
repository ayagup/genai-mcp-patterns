"""
Pattern 195: Time-Based Eviction MCP Pattern

This pattern demonstrates cache eviction based on time-to-live (TTL) or expiration.
Entries are automatically removed after a specified time period, ensuring data
freshness and preventing stale data from being served.

Key Concepts:
1. TTL (Time-To-Live): Maximum time entry stays in cache
2. Absolute Expiration: Expires at specific time
3. Sliding Expiration: Resets on access
4. Automatic Cleanup: Background thread removes expired entries
5. Lazy Eviction: Remove on access if expired

Eviction Strategies:
- Fixed TTL: All entries same expiration
- Per-Entry TTL: Different TTL per entry
- Absolute: Expires at fixed time regardless of access
- Sliding: Extends expiration on each access
- Idle Timeout: Expires after period of no access

Benefits:
- Data Freshness: Prevents serving stale data
- Automatic: No manual invalidation needed
- Memory Management: Frees memory automatically
- Predictable: Known maximum staleness
- Simple: Easy to implement and understand

Trade-offs:
- May evict hot data: Active items still expire
- Clock-dependent: Requires accurate time
- Background overhead: Cleanup thread uses resources
- Granularity: May not match data update frequency
- Cold start: Cache empty after evictions

Use Cases:
- API Responses: External API with changing data
- Session Data: User sessions with timeout
- Token/JWT: Security tokens with expiration
- DNS Cache: DNS records with TTL
- News/Feeds: Time-sensitive content
- Weather Data: Periodic updates
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import threading

# Define the state
class TimeBasedEvictionState(TypedDict):
    """State for time-based eviction cache"""
    cache_operations: List[Dict[str, Any]]
    operation_results: List[Dict[str, Any]]
    eviction_stats: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# TIME-BASED EVICTION CACHE
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with expiration"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    last_accessed: datetime
    sliding: bool = False  # Sliding expiration
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.sliding:
            # Sliding expiration - based on last access
            age = datetime.now() - self.last_accessed
        else:
            # Absolute expiration - based on creation
            age = datetime.now() - self.created_at
        
        return age.total_seconds() > self.ttl_seconds
    
    def get_remaining_seconds(self) -> float:
        """Get remaining seconds until expiration"""
        if self.sliding:
            elapsed = (datetime.now() - self.last_accessed).total_seconds()
        else:
            elapsed = (datetime.now() - self.created_at).total_seconds()
        
        return max(0, self.ttl_seconds - elapsed)
    
    def touch(self):
        """Update last accessed time (for sliding expiration)"""
        self.last_accessed = datetime.now()

class TimeBasedCache:
    """
    Cache with time-based eviction
    
    Features:
    - Per-entry TTL
    - Absolute or sliding expiration
    - Lazy eviction on access
    - Background cleanup thread
    - Eviction statistics
    """
    
    def __init__(self, default_ttl_seconds: int = 300, cleanup_interval: int = 60):
        """
        Initialize time-based cache
        
        Args:
            default_ttl_seconds: Default TTL for entries (5 minutes)
            cleanup_interval: How often to run cleanup (60 seconds)
        """
        self.default_ttl = default_ttl_seconds
        self.cleanup_interval = cleanup_interval
        self.cache: Dict[str, CacheEntry] = {}
        self.evicted_count = 0
        self.expired_on_access = 0
        self.expired_by_cleanup = 0
        self.lock = threading.Lock()
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.running = False
    
    def start_cleanup_thread(self):
        """Start background cleanup thread"""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def stop_cleanup_thread(self):
        """Stop background cleanup thread"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=2)
    
    def _cleanup_worker(self):
        """Background worker that periodically removes expired entries"""
        while self.running:
            time.sleep(self.cleanup_interval)
            self.cleanup_expired()
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            sliding: bool = False):
        """
        Put entry in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds (None = use default)
            sliding: If True, expiration resets on access
        """
        with self.lock:
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            now = datetime.now()
            
            self.cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                ttl_seconds=ttl,
                last_accessed=now,
                sliding=sliding
            )
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get entry from cache
        
        Lazy eviction: If expired, remove and return None
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check expiration (lazy eviction)
            if entry.is_expired():
                # Expired - remove from cache
                del self.cache[key]
                self.evicted_count += 1
                self.expired_on_access += 1
                return None
            
            # Update access time (for sliding expiration)
            entry.touch()
            
            return entry.value
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries (background cleanup)
        
        Returns number of entries removed
        """
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self.cache[key]
                self.evicted_count += 1
                self.expired_by_cleanup += 1
            
            return len(expired_keys)
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get information about cache entry"""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            return {
                "key": key,
                "created_at": entry.created_at.isoformat(),
                "ttl_seconds": entry.ttl_seconds,
                "remaining_seconds": round(entry.get_remaining_seconds(), 2),
                "sliding": entry.sliding,
                "is_expired": entry.is_expired()
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                "cache_size": len(self.cache),
                "total_evicted": self.evicted_count,
                "expired_on_access": self.expired_on_access,
                "expired_by_cleanup": self.expired_by_cleanup
            }

# Global cache instance
ttl_cache = TimeBasedCache(default_ttl_seconds=10, cleanup_interval=2)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_cache_operations(state: TimeBasedEvictionState) -> TimeBasedEvictionState:
    """Process cache operations"""
    operations = state["cache_operations"]
    results = []
    
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "put":
            key = op.get("key")
            value = op.get("value")
            ttl = op.get("ttl_seconds")
            sliding = op.get("sliding", False)
            
            ttl_cache.put(key, value, ttl, sliding)
            
            results.append({
                "operation": "put",
                "key": key,
                "ttl": ttl,
                "sliding": sliding
            })
        
        elif op_type == "get":
            key = op.get("key")
            value = ttl_cache.get(key)
            info = ttl_cache.get_entry_info(key) if value is not None else None
            
            results.append({
                "operation": "get",
                "key": key,
                "found": value is not None,
                "value": value,
                "info": info
            })
        
        elif op_type == "wait":
            seconds = op.get("seconds", 1)
            time.sleep(seconds)
            results.append({
                "operation": "wait",
                "seconds": seconds
            })
    
    return {
        "operation_results": results,
        "messages": [f"[Time-Based Cache] Processed {len(operations)} operations"]
    }

def collect_eviction_stats(state: TimeBasedEvictionState) -> TimeBasedEvictionState:
    """Collect eviction statistics"""
    stats = ttl_cache.get_statistics()
    
    return {
        "eviction_stats": stats,
        "messages": [
            f"[Eviction Stats] Size: {stats['cache_size']}, "
            f"Evicted: {stats['total_evicted']}"
        ]
    }

# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_time_based_eviction_graph():
    """Create graph for time-based eviction"""
    
    workflow = StateGraph(TimeBasedEvictionState)
    
    workflow.add_node("operations", process_cache_operations)
    workflow.add_node("stats", collect_eviction_stats)
    
    workflow.add_edge(START, "operations")
    workflow.add_edge("operations", "stats")
    workflow.add_edge("stats", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Time-Based Eviction MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic TTL Expiration
    print("\n" + "=" * 80)
    print("Example 1: Absolute TTL Expiration")
    print("=" * 80)
    
    graph = create_time_based_eviction_graph()
    
    # Put with 3-second TTL
    state1: TimeBasedEvictionState = {
        "cache_operations": [
            {"type": "put", "key": "session:1", "value": {"user_id": 123}, "ttl_seconds": 3},
            {"type": "get", "key": "session:1"},  # Immediate access
            {"type": "wait", "seconds": 2},
            {"type": "get", "key": "session:1"},  # After 2 seconds (still valid)
            {"type": "wait", "seconds": 2},
            {"type": "get", "key": "session:1"},  # After 4 seconds total (expired!)
        ],
        "operation_results": [],
        "eviction_stats": {},
        "messages": []
    }
    
    result1 = graph.invoke(state1)
    
    print("\nOperation Results:")
    for res in result1["operation_results"]:
        if res["operation"] == "get":
            if res["found"]:
                remaining = res["info"]["remaining_seconds"]
                print(f"  {res['key']}: Found (expires in {remaining}s)")
            else:
                print(f"  {res['key']}: Not found (expired!)")
    
    # Example 2: Sliding Expiration
    print("\n" + "=" * 80)
    print("Example 2: Sliding Expiration (Reset on Access)")
    print("=" * 80)
    
    cache2 = TimeBasedCache(default_ttl_seconds=5)
    
    print("\n1. Put with 5-second sliding TTL:")
    cache2.put("token:abc", {"user": "alice"}, ttl_seconds=5, sliding=True)
    info = cache2.get_entry_info("token:abc")
    print(f"   Expires in: {info['remaining_seconds']}s")
    
    print("\n2. Wait 3 seconds...")
    time.sleep(3)
    info = cache2.get_entry_info("token:abc")
    print(f"   Expires in: {info['remaining_seconds']}s")
    
    print("\n3. Access token (resets expiration):")
    cache2.get("token:abc")
    info = cache2.get_entry_info("token:abc")
    print(f"   Expires in: {info['remaining_seconds']}s (reset!)")
    
    # Example 3: Background Cleanup
    print("\n" + "=" * 80)
    print("Example 3: Background Cleanup Thread")
    print("=" * 80)
    
    cache3 = TimeBasedCache(default_ttl_seconds=2, cleanup_interval=1)
    
    print("\n1. Start cleanup thread:")
    cache3.start_cleanup_thread()
    
    print("2. Add entries with 2-second TTL:")
    for i in range(5):
        cache3.put(f"temp:{i}", {"data": i}, ttl_seconds=2)
    
    print(f"   Cache size: {len(cache3.cache)}")
    
    print("\n3. Wait 3 seconds for expiration...")
    time.sleep(3)
    
    print(f"   Cache size: {len(cache3.cache)} (cleanup removed expired)")
    
    stats = cache3.get_statistics()
    print(f"   Evicted by cleanup: {stats['expired_by_cleanup']}")
    
    cache3.stop_cleanup_thread()
    
    # Example 4: Per-Entry TTL
    print("\n" + "=" * 80)
    print("Example 4: Different TTL Per Entry")
    print("=" * 80)
    
    cache4 = TimeBasedCache()
    
    cache4.put("short_lived", "data1", ttl_seconds=2)
    cache4.put("medium_lived", "data2", ttl_seconds=5)
    cache4.put("long_lived", "data3", ttl_seconds=10)
    
    print("\nEntries with different TTLs:")
    for key in ["short_lived", "medium_lived", "long_lived"]:
        info = cache4.get_entry_info(key)
        print(f"  {key}: expires in {info['remaining_seconds']}s")
    
    # Example 5: Use Cases
    print("\n" + "=" * 80)
    print("Example 5: Common TTL Values by Use Case")
    print("=" * 80)
    
    print("\nTypical TTL Values:")
    print("  Session tokens: 30 minutes (1800s)")
    print("  API responses: 5 minutes (300s)")
    print("  DNS records: 5 minutes to 24 hours")
    print("  JWT tokens: 15 minutes to 1 hour")
    print("  Static content: 1 hour to 1 day")
    print("  Database queries: 1-5 minutes")
    print("  Weather data: 15-30 minutes")
    print("  Stock prices: 1-5 seconds (real-time)")
    
    # Example 6: Statistics
    print("\n" + "=" * 80)
    print("Example 6: Eviction Statistics")
    print("=" * 80)
    
    stats = ttl_cache.get_statistics()
    
    print("\nCache Statistics:")
    print(f"  Current Size: {stats['cache_size']}")
    print(f"  Total Evicted: {stats['total_evicted']}")
    print(f"  Evicted on Access (lazy): {stats['expired_on_access']}")
    print(f"  Evicted by Cleanup: {stats['expired_by_cleanup']}")
    
    print("\n" + "=" * 80)
    print("Time-Based Eviction Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Time-based eviction removes entries after TTL expires
  • Absolute expiration: Fixed time from creation
  • Sliding expiration: Resets on each access
  • Lazy eviction: Remove on access if expired
  • Background cleanup: Periodic removal of expired entries
  • Per-entry TTL: Different expiration per entry
  • Prevents stale data from being served
  • Automatic memory management
  • Use cases: Sessions, tokens, API responses, DNS
  • Tools: Redis (EXPIRE), Memcached (TTL), Ehcache
  • Best practices: Set appropriate TTL for data freshness needs
    """)

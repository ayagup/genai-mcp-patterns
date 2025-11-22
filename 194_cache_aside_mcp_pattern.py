"""
Pattern 194: Cache-Aside (Lazy Loading) MCP Pattern

This pattern demonstrates a caching strategy where the application code explicitly
manages the cache. The application checks the cache, and if the data is not found,
loads it from the backing store and populates the cache.

Key Concepts:
1. Explicit Management: Application controls caching logic
2. Lazy Loading: Load data only when needed
3. Cache Check: Application checks cache first
4. Manual Population: Application loads and caches data
5. Flexible Invalidation: Application controls cache updates

Cache Operations:
- Read: App checks cache → if miss, load from store → cache it
- Write: App updates store → invalidate or update cache
- Delete: App deletes from store → invalidate cache

Benefits:
- Control: Application has full control over caching
- Flexibility: Custom caching logic per use case
- Resilience: Cache failure doesn't break application
- Selective Caching: Cache only what's needed
- Custom TTL: Different expiration per entry

Trade-offs:
- Complexity: More application code
- Consistency: Application must manage invalidation
- Duplication: Caching logic in multiple places
- Error-Prone: Easy to miss cache updates
- Cache Penetration: Multiple requests can

 miss same key

Use Cases:
- Read-Heavy Workloads: Database query results
- API Responses: External API call caching
- Computed Results: Expensive calculations
- User Sessions: Session data
- Configuration: Application settings
- Reference Data: Product catalogs, lookups

Most Common Pattern: Used in majority of caching implementations
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
import time

# Define the state
class CacheAsideState(TypedDict):
    """State for cache-aside operations"""
    operations: List[Dict[str, Any]]
    operation_results: List[Dict[str, Any]]
    cache_statistics: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# CACHE-ASIDE IMPLEMENTATION
# ============================================================================

class Database:
    """Simulates a database (backing store)"""
    
    def __init__(self):
        self.data = {
            "user:1": {"id": 1, "name": "Alice", "email": "alice@example.com"},
            "user:2": {"id": 2, "name": "Bob", "email": "bob@example.com"},
            "product:101": {"id": 101, "name": "Laptop", "price": 999},
            "product:102": {"id": 102, "name": "Mouse", "price": 29},
        }
        self.read_count = 0
        self.write_count = 0
        self.latency_ms = 100
    
    def read(self, key: str) -> Optional[Dict[str, Any]]:
        """Read from database"""
        self.read_count += 1
        time.sleep(self.latency_ms / 1000.0)
        return self.data.get(key)
    
    def write(self, key: str, value: Dict[str, Any]):
        """Write to database"""
        self.write_count += 1
        time.sleep(self.latency_ms / 1000.0)
        self.data[key] = value.copy()
    
    def delete(self, key: str):
        """Delete from database"""
        if key in self.data:
            del self.data[key]

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    ttl_seconds: int
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        age = datetime.now() - self.created_at
        return age.total_seconds() > self.ttl_seconds

class SimpleCache:
    """
    Simple in-memory cache
    
    Application uses this explicitly with manual management
    """
    
    def __init__(self):
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get from cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            if not entry.is_expired():
                self.hits += 1
                return entry.value.copy()
            else:
                # Expired - remove and treat as miss
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Dict[str, Any], ttl_seconds: int = 300):
        """Put into cache"""
        self.cache[key] = CacheEntry(
            key=key,
            value=value.copy(),
            created_at=datetime.now(),
            ttl_seconds=ttl_seconds
        )
    
    def invalidate(self, key: str):
        """Invalidate cache entry"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 2)
        }

# Global instances
database = Database()
cache = SimpleCache()

# ============================================================================
# APPLICATION CODE (Cache-Aside Pattern)
# ============================================================================

def get_user_cache_aside(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Get user with cache-aside pattern
    
    Application explicitly manages cache:
    1. Check cache
    2. If miss, load from DB
    3. Populate cache
    4. Return data
    """
    key = f"user:{user_id}"
    
    # Step 1: Try cache first
    data = cache.get(key)
    
    if data is not None:
        # Cache hit!
        return data
    
    # Step 2: Cache miss - load from database
    data = database.read(key)
    
    if data is not None:
        # Step 3: Populate cache for future requests
        cache.put(key, data, ttl_seconds=300)
    
    # Step 4: Return data
    return data

def update_user_cache_aside(user_id: int, user_data: Dict[str, Any]):
    """
    Update user with cache-aside pattern
    
    Application manages cache invalidation:
    1. Update database
    2. Invalidate cache (or update it)
    """
    key = f"user:{user_id}"
    
    # Step 1: Update database (source of truth)
    database.write(key, user_data)
    
    # Step 2: Invalidate cache
    # Option A: Invalidate (next read will reload)
    cache.invalidate(key)
    
    # Option B: Update cache immediately (write-through style)
    # cache.put(key, user_data, ttl_seconds=300)

def delete_user_cache_aside(user_id: int):
    """
    Delete user with cache-aside pattern
    
    1. Delete from database
    2. Invalidate cache
    """
    key = f"user:{user_id}"
    
    # Delete from database
    database.delete(key)
    
    # Invalidate cache
    cache.invalidate(key)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_operations(state: CacheAsideState) -> CacheAsideState:
    """Process cache-aside operations"""
    operations = state["operations"]
    results = []
    
    for op in operations:
        op_type = op.get("type")
        
        start_time = time.time()
        
        if op_type == "read":
            user_id = op.get("user_id")
            data = get_user_cache_aside(user_id)
            
            results.append({
                "type": "read",
                "user_id": user_id,
                "data": data,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })
        
        elif op_type == "write":
            user_id = op.get("user_id")
            user_data = op.get("data")
            update_user_cache_aside(user_id, user_data)
            
            results.append({
                "type": "write",
                "user_id": user_id,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })
        
        elif op_type == "delete":
            user_id = op.get("user_id")
            delete_user_cache_aside(user_id)
            
            results.append({
                "type": "delete",
                "user_id": user_id,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })
    
    return {
        "operation_results": results,
        "messages": [f"[Cache-Aside] Processed {len(operations)} operations"]
    }

def collect_statistics(state: CacheAsideState) -> CacheAsideState:
    """Collect statistics"""
    cache_stats = cache.get_statistics()
    
    return {
        "cache_statistics": {
            "cache": cache_stats,
            "database_reads": database.read_count,
            "database_writes": database.write_count
        },
        "messages": [
            f"[Statistics] Cache Hit Rate: {cache_stats['hit_rate']}%, "
            f"DB Reads: {database.read_count}"
        ]
    }

# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_cache_aside_graph():
    """Create graph for cache-aside pattern"""
    
    workflow = StateGraph(CacheAsideState)
    
    workflow.add_node("operations", process_operations)
    workflow.add_node("statistics", collect_statistics)
    
    workflow.add_edge(START, "operations")
    workflow.add_edge("operations", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Cache-Aside (Lazy Loading) MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic Cache-Aside
    print("\n" + "=" * 80)
    print("Example 1: Cache-Aside Pattern - Explicit Management")
    print("=" * 80)
    
    graph = create_cache_aside_graph()
    
    # First read - cache miss
    print("\n--- Round 1: Initial Reads (Cache Misses) ---")
    
    state1: CacheAsideState = {
        "operations": [
            {"type": "read", "user_id": 1},
            {"type": "read", "user_id": 2},
        ],
        "operation_results": [],
        "cache_statistics": {},
        "messages": []
    }
    
    result1 = graph.invoke(state1)
    
    for res in result1["operation_results"]:
        print(f"  Read user:{res['user_id']}: {res['latency_ms']}ms (cache miss)")
    
    # Second read - cache hit
    print("\n--- Round 2: Repeated Reads (Cache Hits) ---")
    
    state2: CacheAsideState = {
        "operations": [
            {"type": "read", "user_id": 1},
            {"type": "read", "user_id": 2},
        ],
        "operation_results": [],
        "cache_statistics": {},
        "messages": []
    }
    
    result2 = graph.invoke(state2)
    
    for res in result2["operation_results"]:
        print(f"  Read user:{res['user_id']}: {res['latency_ms']}ms (cache hit - fast!)")
    
    # Example 2: Application Code Pattern
    print("\n" + "=" * 80)
    print("Example 2: Typical Application Code")
    print("=" * 80)
    
    print("""
def get_user(user_id):
    # Cache-Aside Pattern
    key = f"user:{user_id}"
    
    # 1. Check cache
    data = cache.get(key)
    if data:
        return data  # Cache hit
    
    # 2. Cache miss - load from database
    data = database.read(key)
    
    # 3. Populate cache
    if data:
        cache.put(key, data, ttl=300)
    
    return data
    """)
    
    # Example 3: Write Pattern
    print("\n" + "=" * 80)
    print("Example 3: Write Pattern with Cache Invalidation")
    print("=" * 80)
    
    print("\n1. Update user in database and invalidate cache:")
    state3: CacheAsideState = {
        "operations": [
            {"type": "write", "user_id": 1, "data": {"id": 1, "name": "Alice Updated", "email": "alice.new@example.com"}},
        ],
        "operation_results": [],
        "cache_statistics": {},
        "messages": []
    }
    
    result3 = graph.invoke(state3)
    print(f"   Write completed in {result3['operation_results'][0]['latency_ms']}ms")
    
    print("\n2. Next read will reload from database:")
    data = get_user_cache_aside(1)
    print(f"   Data: {data}")
    print("   (Cache was invalidated, so this loaded from DB)")
    
    # Example 4: Pattern Comparison
    print("\n" + "=" * 80)
    print("Example 4: Cache-Aside vs Other Patterns")
    print("=" * 80)
    
    print("\nCache-Aside (Lazy Loading) - This Pattern:")
    print("  ✓ Application has full control")
    print("  ✓ Flexible caching logic")
    print("  ✓ Cache failure doesn't break app")
    print("  ✓ Most common pattern")
    print("  ✗ More application code")
    print("  ✗ Must manage invalidation")
    print("  ✗ Can have stale data")
    
    print("\nRead-Through:")
    print("  ✓ Transparent to application")
    print("  ✓ Simpler application code")
    print("  ✗ Less control")
    print("  ✗ Cache must handle loading")
    
    print("\nWrite-Through:")
    print("  ✓ Strong consistency")
    print("  ✗ Slower writes")
    
    # Example 5: Statistics
    print("\n" + "=" * 80)
    print("Example 5: Cache Statistics")
    print("=" * 80)
    
    stats = cache.get_statistics()
    
    print(f"\nCache Performance:")
    print(f"  Cache Size: {stats['size']}")
    print(f"  Cache Hits: {stats['hits']}")
    print(f"  Cache Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']}%")
    
    print(f"\nDatabase Load:")
    print(f"  Database Reads: {database.read_count}")
    print(f"  Database Writes: {database.write_count}")
    
    print("\n" + "=" * 80)
    print("Cache-Aside Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Cache-aside gives application full control over caching
  • Application explicitly checks cache, loads data, and populates cache
  • Lazy loading: Only cache what's accessed
  • Write: Update DB, then invalidate cache (or update it)
  • Most common caching pattern in practice
  • Flexible but requires more application code
  • Use cases: Database queries, API responses, computed results
  • Tools: Redis, Memcached with application code
  • Best practices: Consistent TTL, handle cache failures gracefully
    """)

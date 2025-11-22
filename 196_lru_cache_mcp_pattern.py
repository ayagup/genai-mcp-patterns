"""
Pattern 196: LRU (Least Recently Used) Cache MCP Pattern

This pattern demonstrates a cache eviction policy that removes the least recently
used entries when the cache reaches capacity. Items accessed most recently are
kept, while older items are evicted.

Key Concepts:
1. Recency-Based: Track when each entry was last accessed
2. Capacity Limit: Fixed maximum cache size
3. Automatic Eviction: Remove LRU item when full
4. Access Ordering: Maintain access order efficiently
5. O(1) Operations: Fast get/put using hash map + doubly-linked list

Data Structure:
- Hash Map: Fast O(1) key lookup
- Doubly-Linked List: Maintain access order
- Head: Most recently used
- Tail: Least recently used

Operations:
- Get: Move accessed item to head (most recent)
- Put: Add to head, evict tail if capacity exceeded
- Both O(1) time complexity

Benefits:
- Locality: Keeps frequently accessed data
- Simple: Easy to understand and implement
- Predictable: Fixed memory usage
- Efficient: O(1) get and put operations
- Widely Used: Standard caching algorithm

Trade-offs:
- May evict useful data: One-time access marks as recent
- No frequency consideration: Ignores access count
- Fixed capacity: Must set size limit
- Memory overhead: Linked list pointers
- Not optimal for all access patterns

Use Cases:
- Page Caching: OS virtual memory, browser cache
- CPU Cache: L1/L2/L3 processor caches
- Database Buffer Pool: Keep hot pages in memory
- CDN: Content delivery caching
- Application Cache: General-purpose caching
- DNS Cache: Domain name resolution
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass

# Define the state
class LRUCacheState(TypedDict):
    """State for LRU cache operations"""
    cache_operations: List[Dict[str, Any]]
    operation_results: List[Dict[str, Any]]
    cache_state: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# LRU CACHE IMPLEMENTATION
# ============================================================================

@dataclass
class Node:
    """Doubly-linked list node"""
    key: str
    value: Any
    prev: Optional['Node'] = None
    next: Optional['Node'] = None

class LRUCache:
    """
    LRU (Least Recently Used) Cache Implementation
    
    Uses hash map + doubly-linked list for O(1) operations.
    
    Structure:
    - Head (dummy): Most recently used
    - Tail (dummy): Least recently used
    - Hash map: key -> node mapping
    
    Access order: Head -> ... -> Tail
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache
        
        Args:
            capacity: Maximum number of entries
        """
        self.capacity = capacity
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail nodes
        self.head = Node("HEAD", None)
        self.tail = Node("TAIL", None)
        self.head.next = self.tail
        self.tail.prev = self.head
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _remove_node(self, node: Node):
        """Remove node from linked list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def _add_to_head(self, node: Node):
        """Add node right after head (most recent position)"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def _move_to_head(self, node: Node):
        """Move existing node to head (mark as recently used)"""
        self._remove_node(node)
        self._add_to_head(node)
    
    def _remove_tail(self) -> Node:
        """Remove and return least recently used node"""
        lru_node = self.tail.prev
        self._remove_node(lru_node)
        return lru_node
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        If found, move to head (mark as recently used)
        Returns None if not found
        """
        if key in self.cache:
            node = self.cache[key]
            self._move_to_head(node)  # Mark as recently used
            self.hits += 1
            return node.value
        
        self.misses += 1
        return None
    
    def put(self, key: str, value: Any):
        """
        Put key-value pair in cache
        
        If cache is full, evict LRU entry
        """
        if key in self.cache:
            # Update existing entry
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)  # Mark as recently used
        else:
            # New entry
            if len(self.cache) >= self.capacity:
                # Cache full - evict LRU
                lru_node = self._remove_tail()
                del self.cache[lru_node.key]
                self.evictions += 1
            
            # Add new node
            new_node = Node(key, value)
            self.cache[key] = new_node
            self._add_to_head(new_node)
    
    def get_access_order(self) -> List[str]:
        """Get keys in access order (most recent first)"""
        keys = []
        current = self.head.next
        
        while current != self.tail:
            keys.append(current.key)
            current = current.next
        
        return keys
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 2),
            "access_order": self.get_access_order()
        }

# Global LRU cache
lru_cache = LRUCache(capacity=3)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_operations(state: LRUCacheState) -> LRUCacheState:
    """Process cache operations"""
    operations = state["cache_operations"]
    results = []
    
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "put":
            key = op.get("key")
            value = op.get("value")
            lru_cache.put(key, value)
            
            results.append({
                "operation": "put",
                "key": key,
                "access_order": lru_cache.get_access_order()
            })
        
        elif op_type == "get":
            key = op.get("key")
            value = lru_cache.get(key)
            
            results.append({
                "operation": "get",
                "key": key,
                "found": value is not None,
                "value": value,
                "access_order": lru_cache.get_access_order()
            })
    
    return {
        "operation_results": results,
        "messages": [f"[LRU Cache] Processed {len(operations)} operations"]
    }

def collect_state(state: LRUCacheState) -> LRUCacheState:
    """Collect cache state"""
    stats = lru_cache.get_statistics()
    
    return {
        "cache_state": stats,
        "messages": [
            f"[State] Size: {stats['size']}/{stats['capacity']}, "
            f"Evictions: {stats['evictions']}, "
            f"Hit Rate: {stats['hit_rate']}%"
        ]
    }

# ============================================================================
# BUILD THE GRAPH
# ============================================================================

def create_lru_cache_graph():
    """Create graph for LRU cache"""
    
    workflow = StateGraph(LRUCacheState)
    
    workflow.add_node("operations", process_operations)
    workflow.add_node("state", collect_state)
    
    workflow.add_edge(START, "operations")
    workflow.add_edge("operations", "state")
    workflow.add_edge("state", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LRU (Least Recently Used) Cache MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic LRU Eviction
    print("\n" + "=" * 80)
    print("Example 1: LRU Eviction on Capacity Limit")
    print("=" * 80)
    
    graph = create_lru_cache_graph()
    
    state1: LRUCacheState = {
        "cache_operations": [
            {"type": "put", "key": "A", "value": 1},
            {"type": "put", "key": "B", "value": 2},
            {"type": "put", "key": "C", "value": 3},
            {"type": "put", "key": "D", "value": 4},  # Evicts A (LRU)
        ],
        "operation_results": [],
        "cache_state": {},
        "messages": []
    }
    
    result1 = graph.invoke(state1)
    
    print("\nOperation Sequence:")
    for res in result1["operation_results"]:
        print(f"  {res['operation'].upper()} {res['key']}: {res['access_order']}")
    
    print(f"\nCache full (capacity=3), added D → evicted A (least recently used)")
    
    # Example 2: Access Pattern Updates Order
    print("\n" + "=" * 80)
    print("Example 2: Access Updates Recency Order")
    print("=" * 80)
    
    cache2 = LRUCache(capacity=3)
    
    print("\n1. Put A, B, C:")
    cache2.put("A", 1)
    cache2.put("B", 2)
    cache2.put("C", 3)
    print(f"   Order (MRU→LRU): {cache2.get_access_order()}")
    
    print("\n2. Access A (move to front):")
    cache2.get("A")
    print(f"   Order: {cache2.get_access_order()}")
    print("   A moved to front (most recent)")
    
    print("\n3. Add D (evicts B, not A!):")
    cache2.put("D", 4)
    print(f"   Order: {cache2.get_access_order()}")
    print("   B evicted (was LRU after A was accessed)")
    
    # Example 3: Hit Rate
    print("\n" + "=" * 80)
    print("Example 3: Cache Hit Rate Demonstration")
    print("=" * 80)
    
    cache3 = LRUCache(capacity=2)
    
    # Access pattern
    keys = ["A", "B", "A", "C", "A", "B"]
    
    print("\nAccess pattern: A, B, A, C, A, B")
    print("Cache capacity: 2\n")
    
    for key in keys:
        value = cache3.get(key)
        if value is None:
            # Cache miss - load and cache
            cache3.put(key, ord(key))
            print(f"  GET {key}: MISS → cached")
        else:
            print(f"  GET {key}: HIT")
        
        print(f"     Order: {cache3.get_access_order()}")
    
    stats = cache3.get_statistics()
    print(f"\nHit Rate: {stats['hit_rate']}%")
    print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
    
    # Example 4: Comparison with LFU
    print("\n" + "=" * 80)
    print("Example 4: LRU vs LFU Comparison")
    print("=" * 80)
    
    print("\nAccess Pattern: A, B, C, A, A, D")
    print("Cache Capacity: 3\n")
    
    print("LRU Eviction:")
    print("  1. Put A, B, C → Cache: [C, B, A]")
    print("  2. Access A twice → Cache: [A, C, B]")
    print("  3. Put D → Evicts B (LRU) → Cache: [D, A, C]")
    print("  → B evicted even though A was accessed more")
    
    print("\nLFU Eviction:")
    print("  1. Put A, B, C → Cache: [A:1, B:1, C:1]")
    print("  2. Access A twice → Cache: [A:3, B:1, C:1]")
    print("  3. Put D → Evicts B or C (LFU) → Cache: [D:1, A:3, C:1]")
    print("  → Evicts least frequently used (B or C)")
    
    print("\nWhen to use:")
    print("  LRU: When recent access predicts future access")
    print("  LFU: When frequency predicts future access")
    
    # Example 5: O(1) Complexity
    print("\n" + "=" * 80)
    print("Example 5: O(1) Time Complexity")
    print("=" * 80)
    
    print("\nLRU Cache Implementation:")
    print("  Data structures:")
    print("    • Hash Map: key → node (O(1) lookup)")
    print("    • Doubly-Linked List: maintain order (O(1) add/remove)")
    
    print("\n  Operations:")
    print("    • GET: Hash lookup + move to head = O(1)")
    print("    • PUT: Hash lookup + add to head + evict tail = O(1)")
    
    print("\n  Space complexity: O(capacity)")
    
    # Example 6: Use Cases
    print("\n" + "=" * 80)
    print("Example 6: Real-World Use Cases")
    print("=" * 80)
    
    print("\n1. Operating System:")
    print("   • Page cache: Recently accessed pages")
    print("   • Buffer cache: File system blocks")
    
    print("\n2. CPU:")
    print("   • L1/L2/L3 cache: Recently accessed memory")
    
    print("\n3. Web Browser:")
    print("   • Page cache: Recently visited pages")
    print("   • Image cache: Recently loaded images")
    
    print("\n4. Database:")
    print("   • Buffer pool: Recently queried pages")
    print("   • Query result cache")
    
    print("\n5. CDN:")
    print("   • Content cache: Recently requested files")
    
    print("\n" + "=" * 80)
    print("LRU Cache Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • LRU evicts least recently used entries when cache is full
  • Maintains access order: Most recent → Least recent
  • O(1) get and put operations (hash map + doubly-linked list)
  • Simple and widely used caching algorithm
  • Works well when recent access predicts future access
  • Fixed capacity prevents unbounded memory growth
  • Recency-based (ignores access frequency)
  • Use cases: Page cache, CPU cache, browser cache, database buffer pool
  • Tools: Built into Python (functools.lru_cache), Redis (MAXMEMORY allkeys-lru)
  • Best practices: Set appropriate capacity, monitor hit rate
  • Trade-off: May evict items accessed once recently over frequently accessed old items
    """)

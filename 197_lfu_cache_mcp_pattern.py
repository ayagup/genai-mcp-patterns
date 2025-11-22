"""
Pattern 197: LFU (Least Frequently Used) Cache MCP Pattern

This pattern demonstrates a cache eviction policy that removes the least frequently
used entries when the cache reaches capacity. Items accessed most often are kept,
while rarely accessed items are evicted.

Key Concepts:
1. Frequency-Based: Track how many times each entry is accessed
2. Capacity Limit: Fixed maximum cache size
3. Automatic Eviction: Remove LFU item when full
4. Access Count: Maintain frequency counter per entry
5. Tie-Breaking: Use recency when frequencies are equal

Eviction Strategy:
- Track access frequency for each entry
- Evict entry with lowest frequency count
- If tied, evict least recently used among them
- Good for skewed access patterns

Benefits:
- Keeps Popular Data: Frequently accessed items stay cached
- Better Than LRU: For workloads with hot items
- Resistant to Scans: One-time accesses don't pollute cache
- Long-term Optimization: Considers entire access history

Trade-offs:
- Complexity: More complex than LRU
- Cold Start: New items have low frequency
- Stale Popular Items: Old popular items may linger
- Memory Overhead: Need to track frequencies
- Not Always Better: Depends on access pattern

Use Cases:
- Content Delivery: Popular content caching
- Database Query Cache: Frequently run queries
- Recommendation Systems: Popular recommendations
- Video Streaming: Popular videos
- API Rate Limiting: Track API usage frequency
- Search Results: Common queries

Comparison with LRU:
- LFU: Evicts least frequently used (considers all history)
- LRU: Evicts least recently used (considers only recency)
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from collections import defaultdict
import time

# Define the state
class LFUCacheState(TypedDict):
    """State for LFU cache operations"""
    cache_operations: List[Dict[str, Any]]
    operation_results: List[Dict[str, Any]]
    cache_state: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# LFU CACHE IMPLEMENTATION
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with frequency and recency"""
    key: str
    value: Any
    frequency: int
    last_accessed: float  # Timestamp for tie-breaking

class LFUCache:
    """
    LFU (Least Frequently Used) Cache Implementation
    
    Evicts entries with lowest access frequency.
    Uses recency as tie-breaker.
    
    Data structures:
    - cache: key -> CacheEntry
    - freq_map: frequency -> set of keys
    - min_freq: Track minimum frequency for O(1) eviction
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LFU cache
        
        Args:
            capacity: Maximum number of entries
        """
        self.capacity = capacity
        self.cache: Dict[str, CacheEntry] = {}
        self.freq_map: Dict[int, set] = defaultdict(set)  # frequency -> keys
        self.min_freq = 0  # Track minimum frequency
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _update_frequency(self, entry: CacheEntry):
        """Update frequency of entry"""
        old_freq = entry.frequency
        new_freq = old_freq + 1
        
        # Remove from old frequency set
        self.freq_map[old_freq].remove(entry.key)
        if not self.freq_map[old_freq] and old_freq == self.min_freq:
            # If min frequency set is empty, increment min_freq
            self.min_freq += 1
        
        # Add to new frequency set
        entry.frequency = new_freq
        entry.last_accessed = time.time()
        self.freq_map[new_freq].add(entry.key)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        If found, increment frequency
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        self._update_frequency(entry)
        self.hits += 1
        
        return entry.value
    
    def put(self, key: str, value: Any):
        """
        Put key-value pair in cache
        
        If cache is full, evict LFU entry
        """
        if self.capacity <= 0:
            return
        
        if key in self.cache:
            # Update existing entry
            entry = self.cache[key]
            entry.value = value
            self._update_frequency(entry)
        else:
            # New entry
            if len(self.cache) >= self.capacity:
                # Cache full - evict LFU entry
                self._evict_lfu()
            
            # Add new entry with frequency 1
            new_entry = CacheEntry(
                key=key,
                value=value,
                frequency=1,
                last_accessed=time.time()
            )
            
            self.cache[key] = new_entry
            self.freq_map[1].add(key)
            self.min_freq = 1
    
    def _evict_lfu(self):
        """Evict least frequently used entry"""
        # Get keys with minimum frequency
        lfu_keys = self.freq_map[self.min_freq]
        
        # Among LFU keys, evict the least recently used (tie-breaker)
        lru_key = min(lfu_keys, key=lambda k: self.cache[k].last_accessed)
        
        # Remove from cache and frequency map
        lfu_keys.remove(lru_key)
        del self.cache[lru_key]
        self.evictions += 1
    
    def get_frequency_distribution(self) -> Dict[int, int]:
        """Get distribution of frequencies"""
        dist = {}
        for freq, keys in self.freq_map.items():
            if keys:
                dist[freq] = len(keys)
        return dist
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0.0
        
        # Get entries sorted by frequency
        entries_by_freq = sorted(
            self.cache.items(),
            key=lambda x: (-x[1].frequency, -x[1].last_accessed)
        )
        
        return {
            "capacity": self.capacity,
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(hit_rate, 2),
            "min_frequency": self.min_freq,
            "frequency_distribution": self.get_frequency_distribution(),
            "top_entries": [
                {"key": k, "frequency": v.frequency}
                for k, v in entries_by_freq[:5]
            ]
        }

# Global LFU cache
lfu_cache = LFUCache(capacity=3)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def process_operations(state: LFUCacheState) -> LFUCacheState:
    """Process cache operations"""
    operations = state["cache_operations"]
    results = []
    
    for op in operations:
        op_type = op.get("type")
        
        if op_type == "put":
            key = op.get("key")
            value = op.get("value")
            lfu_cache.put(key, value)
            
            results.append({
                "operation": "put",
                "key": key
            })
        
        elif op_type == "get":
            key = op.get("key")
            value = lfu_cache.get(key)
            
            entry_info = None
            if key in lfu_cache.cache:
                entry = lfu_cache.cache[key]
                entry_info = {
                    "frequency": entry.frequency,
                    "value": entry.value
                }
            
            results.append({
                "operation": "get",
                "key": key,
                "found": value is not None,
                "info": entry_info
            })
    
    return {
        "operation_results": results,
        "messages": [f"[LFU Cache] Processed {len(operations)} operations"]
    }

def collect_state(state: LFUCacheState) -> LFUCacheState:
    """Collect cache state"""
    stats = lfu_cache.get_statistics()
    
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

def create_lfu_cache_graph():
    """Create graph for LFU cache"""
    
    workflow = StateGraph(LFUCacheState)
    
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
    print("LFU (Least Frequently Used) Cache MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic LFU Eviction
    print("\n" + "=" * 80)
    print("Example 1: LFU Eviction Based on Frequency")
    print("=" * 80)
    
    graph = create_lfu_cache_graph()
    
    state1: LFUCacheState = {
        "cache_operations": [
            {"type": "put", "key": "A", "value": 1},
            {"type": "put", "key": "B", "value": 2},
            {"type": "put", "key": "C", "value": 3},
            {"type": "get", "key": "A"},  # A: freq=2
            {"type": "get", "key": "A"},  # A: freq=3
            {"type": "get", "key": "B"},  # B: freq=2
            {"type": "put", "key": "D", "value": 4},  # Evicts C (freq=1, LFU)
        ],
        "operation_results": [],
        "cache_state": {},
        "messages": []
    }
    
    result1 = graph.invoke(state1)
    
    print("\nFrequency Build-up:")
    print("  A accessed 3 times → frequency=3")
    print("  B accessed 2 times → frequency=2")
    print("  C accessed 1 time → frequency=1")
    print("  D added → evicts C (least frequently used)")
    
    stats = result1["cache_state"]
    print(f"\nTop Entries:")
    for entry in stats["top_entries"]:
        print(f"  {entry['key']}: frequency={entry['frequency']}")
    
    # Example 2: LFU vs LRU Comparison
    print("\n" + "=" * 80)
    print("Example 2: LFU vs LRU Behavior Comparison")
    print("=" * 80)
    
    print("\nAccess Pattern: A, B, C, A, A, D")
    print("Cache Capacity: 3\n")
    
    # LRU behavior
    print("LRU Cache:")
    print("  Put A, B, C → [C, B, A]")
    print("  Get A twice → [A, C, B] (A moved to front)")
    print("  Put D → Evicts B → [D, A, C]")
    print("  Result: B evicted (least recently used)")
    
    # LFU behavior
    print("\nLFU Cache:")
    print("  Put A, B, C → [A:1, B:1, C:1]")
    print("  Get A twice → [A:3, B:1, C:1]")
    print("  Put D → Evicts B or C (freq=1) → [D:1, A:3, C:1 or B:1]")
    print("  Result: B or C evicted (least frequently used)")
    
    print("\nKey Difference:")
    print("  LRU: Only cares about WHEN accessed (recency)")
    print("  LFU: Only cares about HOW OFTEN accessed (frequency)")
    
    # Example 3: Frequency Distribution
    print("\n" + "=" * 80)
    print("Example 3: Frequency Distribution")
    print("=" * 80)
    
    cache3 = LFUCache(capacity=5)
    
    # Access pattern with varying frequencies
    accesses = {
        "hot": 10,    # Very popular
        "warm": 5,    # Moderately popular
        "cool": 2,    # Less popular
        "cold": 1     # Rarely accessed
    }
    
    print("\nAccess Pattern:")
    for key, count in accesses.items():
        for _ in range(count):
            cache3.put(key, f"data_{key}")
            cache3.get(key)
        print(f"  {key}: accessed {count} times")
    
    # Add one more item to trigger eviction
    print("\nAdd new item 'new' to full cache:")
    cache3.put("new", "data_new")
    
    stats3 = cache3.get_statistics()
    print(f"\nFrequency Distribution: {stats3['frequency_distribution']}")
    print(f"Evicted: 'cold' (frequency=1, least frequently used)")
    
    # Example 4: Use Cases
    print("\n" + "=" * 80)
    print("Example 4: When to Use LFU vs LRU")
    print("=" * 80)
    
    print("\nUse LFU when:")
    print("  ✓ Access pattern is skewed (some items very popular)")
    print("  ✓ Popular items remain popular over time")
    print("  ✓ Want to resist one-time access pollution")
    print("  ✓ Examples: Video streaming, CDN, popular content")
    
    print("\nUse LRU when:")
    print("  ✓ Recent access predicts future access")
    print("  ✓ Working set changes over time")
    print("  ✓ Temporal locality is strong")
    print("  ✓ Examples: Page cache, general-purpose caching")
    
    # Example 5: Statistics
    print("\n" + "=" * 80)
    print("Example 5: Cache Statistics")
    print("=" * 80)
    
    stats = lfu_cache.get_statistics()
    
    print(f"\nCache Performance:")
    print(f"  Capacity: {stats['capacity']}")
    print(f"  Current Size: {stats['size']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']}%")
    print(f"  Evictions: {stats['evictions']}")
    print(f"  Min Frequency: {stats['min_frequency']}")
    
    print(f"\nFrequency Distribution:")
    for freq, count in sorted(stats['frequency_distribution'].items()):
        print(f"  Frequency {freq}: {count} entries")
    
    print("\n" + "=" * 80)
    print("LFU Cache Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • LFU evicts least frequently used entries when cache is full
  • Tracks access frequency (count) for each entry
  • Keeps popular/hot data in cache longer
  • Better than LRU for skewed access patterns
  • Resistant to one-time access pollution
  • Uses recency as tie-breaker when frequencies equal
  • More complex than LRU (frequency tracking overhead)
  • Use cases: CDN, video streaming, popular content caching
  • Tools: Redis (MAXMEMORY allkeys-lfu), custom implementations
  • Best practices: Monitor frequency distribution, adjust capacity
  • Trade-off: Complexity vs better hit rate for hot items
    """)

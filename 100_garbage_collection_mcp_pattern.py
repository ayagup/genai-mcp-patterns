"""
Garbage Collection MCP Pattern

This pattern automatically identifies and reclaims unused resources,
preventing memory leaks and optimizing resource utilization.

Key Features:
- Automatic resource detection
- Reference tracking
- Memory reclamation
- Leak detection and prevention
- Performance optimization
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class GarbageCollectionState(TypedDict):
    """State for garbage collection pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    gc_strategy: str  # "reference_counting", "mark_and_sweep", "generational", "concurrent"
    total_objects: int
    live_objects: int
    garbage_objects: int
    memory_before_mb: float
    memory_after_mb: float
    gc_cycles: int
    collection_time_ms: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Garbage Collector
def garbage_collector(state: GarbageCollectionState) -> GarbageCollectionState:
    """Performs garbage collection"""
    gc_strategy = state.get("gc_strategy", "mark_and_sweep")
    total_objects = state.get("total_objects", 10000)
    
    system_message = SystemMessage(content="""You are a garbage collector.
    Identify and reclaim unused resources automatically.""")
    
    user_message = HumanMessage(content=f"""Perform garbage collection:

Strategy: {gc_strategy}
Total Objects: {total_objects}

Run garbage collection cycle.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate garbage collection
    import random
    garbage_objects = int(total_objects * 0.25)  # 25% garbage
    live_objects = total_objects - garbage_objects
    memory_before_mb = total_objects * 0.01  # 10KB per object
    memory_after_mb = live_objects * 0.01
    collection_time_ms = random.uniform(10, 50)
    gc_cycles = state.get("gc_cycles", 0) + 1
    
    report = f"""
    🗑️ Garbage Collection:
    
    Collection Results:
    • Strategy: {gc_strategy.upper()}
    • Objects Scanned: {total_objects:,}
    • Live Objects: {live_objects:,}
    • Garbage Collected: {garbage_objects:,}
    • Memory Before: {memory_before_mb:.2f} MB
    • Memory After: {memory_after_mb:.2f} MB
    • Memory Freed: {memory_before_mb - memory_after_mb:.2f} MB
    • Collection Time: {collection_time_ms:.2f} ms
    • GC Cycles: {gc_cycles}
    
    Garbage Collection Strategies:
    
    1. Reference Counting:
       • Track reference count per object
       • Deallocate when count = 0
       • Immediate collection
       • Cannot handle cycles
       • Example: Python (partial), Swift
    
    2. Mark and Sweep:
       • Mark: Trace from roots
       • Sweep: Collect unmarked
       • Handles cycles
       • Stop-the-world pauses
       • Example: Java, Go
    
    3. Generational:
       • Young generation (frequent)
       • Old generation (infrequent)
       • Most objects die young
       • Optimized for common case
       • Example: Java G1GC, Python
    
    4. Concurrent:
       • Run alongside application
       • Reduce pause times
       • More complex
       • Higher throughput
       • Example: Go, Java CMS
    
    5. Incremental:
       • Break collection into steps
       • Interleave with application
       • Bounded pause times
       • More overhead
       • Example: V8 (JavaScript)
    
    GC Algorithms in Detail:
    
    Mark-and-Sweep:
    ```python
    def mark_and_sweep():
        # Mark phase
        marked = set()
        stack = [root for root in gc_roots]
        
        while stack:
            obj = stack.pop()
            if obj not in marked:
                marked.add(obj)
                stack.extend(obj.references)
        
        # Sweep phase
        for obj in all_objects:
            if obj not in marked:
                deallocate(obj)
    ```
    
    Generational:
    ```python
    class GenerationalGC:
        def __init__(self):
            self.young = []  # Recently allocated
            self.old = []    # Survived multiple GCs
        
        def collect_young(self):
            # Minor GC - frequent, fast
            survivors = []
            for obj in self.young:
                if is_reachable(obj):
                    obj.age += 1
                    if obj.age > THRESHOLD:
                        self.old.append(obj)
                    else:
                        survivors.append(obj)
            self.young = survivors
        
        def collect_old(self):
            # Major GC - infrequent, slow
            # Full mark-and-sweep
            pass
    ```
    
    Reference Counting:
    ```python
    class RefCounted:
        def __init__(self):
            self.ref_count = 0
            self.data = None
        
        def incref(self):
            self.ref_count += 1
        
        def decref(self):
            self.ref_count -= 1
            if self.ref_count == 0:
                self.deallocate()
        
        def deallocate(self):
            # Free resources
            del self.data
    ```
    
    Language-Specific GC:
    
    Java:
    • Serial GC: Single-threaded
    • Parallel GC: Multi-threaded
    • G1GC: Generational, region-based
    • ZGC: Low-latency, concurrent
    • Shenandoah: Ultra-low pause
    
    Python:
    • Reference counting (primary)
    • Cycle detector (backup)
    • Generational (3 generations)
    • Manual control: gc.collect()
    
    Go:
    • Concurrent mark-sweep
    • Tri-color marking
    • Write barriers
    • Sub-millisecond pauses
    • Automatic tuning
    
    JavaScript (V8):
    • Generational (Scavenge, Mark-Compact)
    • Incremental marking
    • Concurrent sweeping
    • Parallel compaction
    • Orinoco optimizer
    
    .NET:
    • Generational (Gen 0, 1, 2)
    • Server vs Workstation GC
    • Background GC
    • Large Object Heap
    • Pinned objects
    
    GC Roots (Starting Points):
    
    Common Roots:
    • Stack variables
    • Static fields
    • CPU registers
    • JNI references
    • Thread locals
    • Finalizer queue
    
    GC Tuning:
    
    JVM Flags:
    ```bash
    # Use G1GC
    -XX:+UseG1GC
    
    # Set heap size
    -Xms2g -Xmx4g
    
    # GC logging
    -Xlog:gc*:file=gc.log
    
    # Pause time goal
    -XX:MaxGCPauseMillis=200
    
    # Young generation size
    -XX:NewRatio=2
    ```
    
    Go GC Tuning:
    ```bash
    # Set GC target percentage
    GOGC=100  # Default: GC when heap doubles
    
    # Debug GC
    GODEBUG=gctrace=1
    
    # Soft memory limit
    GOMEMLIMIT=4GiB
    ```
    
    Python GC Control:
    ```python
    import gc
    
    # Disable automatic GC
    gc.disable()
    
    # Manual collection
    gc.collect()
    
    # Tune thresholds
    gc.set_threshold(700, 10, 10)
    
    # Get stats
    print(gc.get_stats())
    ```
    """
    
    return {
        "messages": [AIMessage(content=f"🗑️ Garbage Collector:\n{response.content}\n{report}")],
        "live_objects": live_objects,
        "garbage_objects": garbage_objects,
        "memory_after_mb": memory_after_mb,
        "collection_time_ms": collection_time_ms,
        "gc_cycles": gc_cycles
    }


# Leak Detector
def leak_detector(state: GarbageCollectionState) -> GarbageCollectionState:
    """Detects memory leaks and resource issues"""
    gc_cycles = state.get("gc_cycles", 0)
    memory_after_mb = state.get("memory_after_mb", 0.0)
    
    system_message = SystemMessage(content="""You are a memory leak detector.
    Identify potential memory leaks and resource issues.""")
    
    user_message = HumanMessage(content=f"""Detect memory leaks:

GC Cycles: {gc_cycles}
Current Memory: {memory_after_mb:.2f} MB

Analyze for leaks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate leak detection
    leak_detected = memory_after_mb > 90.0  # Threshold
    
    report = f"""
    🔍 Leak Detection:
    
    Analysis Results:
    • GC Cycles Run: {gc_cycles}
    • Current Memory: {memory_after_mb:.2f} MB
    • Leak Detected: {'Yes ⚠️' if leak_detected else 'No ✅'}
    
    Common Memory Leak Causes:
    
    1. Forgotten References:
       • Event listeners not removed
       • Callbacks not cleared
       • Cache entries never expire
       • Global variables accumulate
    
    2. Circular References:
       • Objects reference each other
       • Parent-child cycles
       • Closure captures
       • Not handled by ref counting
    
    3. Resource Leaks:
       • Unclosed files
       • Unreleased connections
       • Unfreed memory allocations
       • Thread leaks
    
    4. Framework-Specific:
       • React: Unmounted components
       • Android: Context references
       • iOS: Retain cycles
       • Node.js: Event emitters
    
    Leak Detection Tools:
    
    Java:
    • VisualVM
    • Java Mission Control
    • Eclipse MAT (Memory Analyzer)
    • YourKit Java Profiler
    • JProfiler
    
    Python:
    • tracemalloc
    • memory_profiler
    • objgraph
    • pympler
    • guppy3
    
    JavaScript:
    • Chrome DevTools Heap Profiler
    • Memory snapshots
    • Allocation timeline
    • Retaining path analysis
    
    .NET:
    • dotMemory
    • PerfView
    • Debug Diagnostic Tool
    • CLR Profiler
    
    Detection Techniques:
    
    Heap Dump Analysis:
    ```python
    # Python
    import tracemalloc
    
    tracemalloc.start()
    
    # ... run code ...
    
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    for stat in top_stats[:10]:
        print(stat)
    ```
    
    Object Tracking:
    ```python
    import gc
    import sys
    
    def find_leaks(obj_type):
        objects = gc.get_objects()
        return [obj for obj in objects 
                if isinstance(obj, obj_type)]
    
    # Find all dict objects
    dicts = find_leaks(dict)
    print(f"Dict count: {{len(dicts)}}")
    ```
    
    Reference Counting:
    ```python
    import sys
    
    obj = {{'data': 'value'}}
    print(sys.getrefcount(obj))  # 2
    
    ref = obj
    print(sys.getrefcount(obj))  # 3
    
    del ref
    print(sys.getrefcount(obj))  # 2
    ```
    
    Memory Profiling:
    ```python
    from memory_profiler import profile
    
    @profile
    def my_function():
        large_list = [i for i in range(1000000)]
        return sum(large_list)
    ```
    
    Prevention Strategies:
    
    1. RAII Pattern (C++):
       • Resource Acquisition Is Initialization
       • Automatic cleanup in destructor
       • Smart pointers (unique_ptr, shared_ptr)
    
    2. Context Managers (Python):
       ```python
       with open('file.txt') as f:
           data = f.read()
       # File automatically closed
       ```
    
    3. Try-Finally:
       ```python
       resource = acquire()
       try:
           use(resource)
       finally:
           release(resource)
       ```
    
    4. Weak References:
       ```python
       import weakref
       
       obj = MyClass()
       weak_ref = weakref.ref(obj)
       
       # Won't prevent GC
       obj = None  # Object can be collected
       ```
    
    5. Event Cleanup:
       ```javascript
       // Remove event listeners
       element.removeEventListener('click', handler);
       
       // React useEffect cleanup
       useEffect(() => {{
         const timer = setInterval(...);
         return () => clearInterval(timer);
       }}, []);
       ```
    
    Monitoring Metrics:
    
    Key Indicators:
    • Memory growth over time
    • GC frequency increasing
    • GC pause times growing
    • Heap size not shrinking
    • Object count increasing
    
    Alerting Thresholds:
    • Memory > 80% for 10+ minutes
    • GC pause > 1 second
    • GC every < 10 seconds
    • Heap growth > 10% per hour
    • Old gen usage > 90%
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Leak Detector:\n{response.content}\n{report}")]
    }


# GC Monitor
def gc_monitor(state: GarbageCollectionState) -> GarbageCollectionState:
    """Monitors garbage collection performance"""
    gc_strategy = state.get("gc_strategy", "")
    total_objects = state.get("total_objects", 0)
    live_objects = state.get("live_objects", 0)
    garbage_objects = state.get("garbage_objects", 0)
    memory_before_mb = state.get("memory_before_mb", 0.0)
    memory_after_mb = state.get("memory_after_mb", 0.0)
    collection_time_ms = state.get("collection_time_ms", 0.0)
    gc_cycles = state.get("gc_cycles", 0)
    
    collection_efficiency = (garbage_objects / total_objects * 100) if total_objects > 0 else 0
    memory_freed = memory_before_mb - memory_after_mb
    
    summary = f"""
    📊 GARBAGE COLLECTION COMPLETE
    
    GC Summary:
    • Strategy: {gc_strategy.upper()}
    • Total Objects: {total_objects:,}
    • Live Objects: {live_objects:,}
    • Garbage Collected: {garbage_objects:,}
    • Collection Efficiency: {collection_efficiency:.1f}%
    • Memory Freed: {memory_freed:.2f} MB
    • Collection Time: {collection_time_ms:.2f} ms
    • GC Cycles: {gc_cycles}
    
    Garbage Collection Pattern Process:
    1. Garbage Collector → Reclaim unused resources
    2. Leak Detector → Identify memory leaks
    3. Monitor → Track GC performance
    
    GC Performance Metrics:
    
    Throughput:
    • Application time / Total time
    • Higher is better
    • 95%+ is good
    • Trade-off with latency
    
    Latency:
    • GC pause duration
    • Lower is better
    • < 100ms for interactive apps
    • < 10ms for real-time systems
    
    Memory Overhead:
    • Heap size / Live data
    • Lower is better
    • 1.5-2x is typical
    • Depends on GC algorithm
    
    Best Practices:
    
    Application Design:
    • Minimize object allocation
    • Reuse objects (pools)
    • Use primitives when possible
    • Avoid finalizers
    • Clear references explicitly
    
    GC Configuration:
    • Choose appropriate GC
    • Set heap size correctly
    • Tune generation sizes
    • Monitor and adjust
    • Profile before optimizing
    
    Code Patterns:
    • Use try-with-resources
    • Implement AutoCloseable
    • Avoid large objects
    • Stream large datasets
    • Batch operations
    
    Monitoring:
    • Track GC frequency
    • Monitor pause times
    • Watch heap usage
    • Alert on anomalies
    • Analyze GC logs
    
    Real-World Examples:
    
    Twitter:
    • Custom GC tuning for JVM
    • Reduced GC pause times
    • Improved throughput
    • Better user experience
    
    Netflix:
    • Microservices with tuned GC
    • G1GC for most services
    • ZGC for ultra-low latency
    • Continuous monitoring
    
    Google:
    • Custom GC in Go
    • Concurrent mark-sweep
    • Sub-millisecond pauses
    • Billions of objects/sec
    
    GC Trade-offs:
    
    Throughput vs Latency:
    • Throughput GC: Longer pauses, higher throughput
    • Low-latency GC: Shorter pauses, lower throughput
    • Choose based on requirements
    
    Memory vs CPU:
    • More memory → Less GC
    • More CPU → Faster GC
    • Balance based on resources
    
    Simplicity vs Performance:
    • Simple GC: Easy to tune
    • Complex GC: Better performance
    • Consider maintenance cost
    
    When to Optimize GC:
    
    Symptoms:
    • Frequent out-of-memory errors
    • Long GC pause times
    • High GC overhead (> 10%)
    • Memory leaks detected
    • Poor application performance
    
    Actions:
    • Profile heap usage
    • Analyze GC logs
    • Tune GC parameters
    • Fix memory leaks
    • Optimize code
    • Consider GC-less regions
    
    Alternative Approaches:
    
    Manual Memory Management:
    • C/C++: malloc/free
    • Rust: Ownership system
    • Full control
    • No GC pauses
    • Higher complexity
    
    Arena Allocation:
    • Allocate in arena
    • Free entire arena at once
    • Fast allocation
    • No per-object tracking
    • Good for temporary data
    
    Stack Allocation:
    • Automatic cleanup
    • No GC needed
    • Very fast
    • Limited lifetime
    
    Key Insight:
    Garbage collection automates memory management but
    requires understanding and tuning for optimal
    performance. Monitor GC metrics and optimize when needed.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 GC Monitor:\n{summary}")]
    }


# Build the graph
def build_gc_graph():
    """Build the garbage collection pattern graph"""
    workflow = StateGraph(GarbageCollectionState)
    
    workflow.add_node("collector", garbage_collector)
    workflow.add_node("leak_detector", leak_detector)
    workflow.add_node("monitor", gc_monitor)
    
    workflow.add_edge(START, "collector")
    workflow.add_edge("collector", "leak_detector")
    workflow.add_edge("leak_detector", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_gc_graph()
    
    print("=== Garbage Collection MCP Pattern ===\n")
    
    # Test Case: Mark-and-sweep garbage collection
    print("\n" + "="*70)
    print("TEST CASE: Automatic Garbage Collection")
    print("="*70)
    
    state = {
        "messages": [],
        "gc_strategy": "mark_and_sweep",
        "total_objects": 10000,
        "live_objects": 0,
        "garbage_objects": 0,
        "memory_before_mb": 100.0,
        "memory_after_mb": 0.0,
        "gc_cycles": 0,
        "collection_time_ms": 0.0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nGarbage Collected: {result.get('garbage_objects', 0):,} objects")
    print(f"Memory Freed: {result.get('memory_before_mb', 0) - result.get('memory_after_mb', 0):.2f} MB")
    print(f"Collection Time: {result.get('collection_time_ms', 0):.2f} ms")
    print(f"\n🎉 RESOURCE MANAGEMENT PATTERNS COMPLETE! (Patterns 91-100)")

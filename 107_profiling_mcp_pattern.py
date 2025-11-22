"""
Profiling MCP Pattern

This pattern implements performance profiling to identify bottlenecks,
optimize resource usage, and improve application performance.

Key Features:
- CPU profiling
- Memory profiling
- I/O profiling
- Hotspot identification
- Performance recommendations
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ProfilingState(TypedDict):
    """State for profiling pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    profiling_data: Dict  # {cpu_time, memory_usage, io_operations, function_calls}
    hotspots: List[Dict]  # [{function, time_ms, percentage, calls}]
    recommendations: List[str]
    profiler_type: str  # "cpu", "memory", "io", "full"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Performance Profiler
def performance_profiler(state: ProfilingState) -> ProfilingState:
    """Profiles application performance"""
    profiler_type = state.get("profiler_type", "cpu")
    
    system_message = SystemMessage(content="""You are a performance profiler.
    Analyze application performance and identify bottlenecks.""")
    
    user_message = HumanMessage(content=f"""Profile application:

Type: {profiler_type}

Collect performance data.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate profiling data
    profiling_data = {
        "total_execution_time_ms": 5234.5,
        "cpu_time_ms": 4123.2,
        "io_wait_time_ms": 987.3,
        "memory_peak_mb": 256.8,
        "function_calls": 15234,
        "profiler": profiler_type
    }
    
    # Simulate hotspots (functions consuming most time)
    hotspots = [
        {
            "function": "process_data",
            "file": "data/processor.py",
            "line": 45,
            "time_ms": 2145.6,
            "percentage": 41.0,
            "calls": 1250,
            "time_per_call_ms": 1.72
        },
        {
            "function": "database_query",
            "file": "db/queries.py",
            "line": 123,
            "time_ms": 876.3,
            "percentage": 16.7,
            "calls": 45,
            "time_per_call_ms": 19.47
        },
        {
            "function": "serialize_json",
            "file": "api/serializers.py",
            "line": 234,
            "time_ms": 543.2,
            "percentage": 10.4,
            "calls": 3420,
            "time_per_call_ms": 0.16
        },
        {
            "function": "validate_input",
            "file": "validators.py",
            "line": 67,
            "time_ms": 432.1,
            "percentage": 8.3,
            "calls": 1250,
            "time_per_call_ms": 0.35
        }
    ]
    
    report = f"""
    ⚡ Performance Profiler:
    
    Profiling Results:
    • Total Time: {profiling_data['total_execution_time_ms']:.2f}ms
    • CPU Time: {profiling_data['cpu_time_ms']:.2f}ms
    • I/O Wait: {profiling_data['io_wait_time_ms']:.2f}ms
    • Peak Memory: {profiling_data['memory_peak_mb']:.2f}MB
    • Function Calls: {profiling_data['function_calls']:,}
    
    Profiling Concepts:
    
    Profiler Types:
    
    CPU Profiler:
    • Time spent in functions
    • Call counts
    • Call graph
    • Hot paths
    • Tools: cProfile, profile
    
    Memory Profiler:
    • Memory allocation
    • Peak usage
    • Memory leaks
    • Object lifetime
    • Tools: memory_profiler, tracemalloc
    
    I/O Profiler:
    • Disk operations
    • Network calls
    • Wait times
    • Throughput
    • Tools: iotop, iostat
    
    Line Profiler:
    • Line-by-line timing
    • Precise hotspots
    • Kernel-level detail
    • Tools: line_profiler
    
    Python Profiling:
    
    cProfile (Standard):
    ```python
    import cProfile
    import pstats
    
    # Profile a function
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = expensive_function()
    
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    # Save to file
    stats.dump_stats('profile.stats')
    ```
    
    Context Manager:
    ```python
    import cProfile
    
    with cProfile.Profile() as pr:
        expensive_function()
    
    pr.print_stats(sort='cumulative')
    ```
    
    Decorator:
    ```python
    import functools
    import cProfile
    import pstats
    
    def profile(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile()
            pr.enable()
            result = func(*args, **kwargs)
            pr.disable()
            
            stats = pstats.Stats(pr)
            stats.sort_stats('cumulative')
            stats.print_stats()
            
            return result
        return wrapper
    
    @profile
    def slow_function():
        # Your code
        pass
    ```
    
    Memory Profiling:
    ```python
    from memory_profiler import profile
    
    @profile
    def memory_intensive():
        large_list = [i for i in range(1000000)]
        large_dict = {{i: i**2 for i in range(100000)}}
        return len(large_list) + len(large_dict)
    
    # Output:
    # Line #    Mem usage    Increment  Occurrences   Line Contents
    # ======================================================
    #      2     38.5 MiB     38.5 MiB           1   @profile
    #      3                                         def memory_intensive():
    #      4     46.1 MiB      7.6 MiB           1       large_list = [...]
    #      5     53.7 MiB      7.6 MiB           1       large_dict = {{...}}
    #      6     53.7 MiB      0.0 MiB           1       return len(...)
    ```
    
    tracemalloc (Standard Library):
    ```python
    import tracemalloc
    
    tracemalloc.start()
    
    # Your code
    result = process_data()
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current: {{current / 10**6:.1f}}MB")
    print(f"Peak: {{peak / 10**6:.1f}}MB")
    
    # Top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    
    for stat in top_stats[:10]:
        print(stat)
    
    tracemalloc.stop()
    ```
    
    Line Profiler:
    ```python
    from line_profiler import LineProfiler
    
    def slow_function():
        total = 0
        for i in range(1000000):
            total += i ** 2
        return total
    
    lp = LineProfiler()
    lp.add_function(slow_function)
    lp.enable()
    slow_function()
    lp.disable()
    lp.print_stats()
    ```
    
    Visualization Tools:
    
    SnakeViz:
    ```bash
    # Install
    pip install snakeviz
    
    # Profile and visualize
    python -m cProfile -o program.prof myscript.py
    snakeviz program.prof
    ```
    
    gprof2dot:
    ```bash
    # Install
    pip install gprof2dot
    
    # Generate graph
    python -m cProfile -o profile.pstats script.py
    gprof2dot -f pstats profile.pstats | dot -Tpng -o profile.png
    ```
    
    py-spy (Sampling Profiler):
    ```bash
    # Install
    pip install py-spy
    
    # Profile running process
    py-spy top --pid 12345
    
    # Record flame graph
    py-spy record -o profile.svg --pid 12345
    
    # Profile script
    py-spy record -o profile.svg -- python script.py
    ```
    
    Profiling Best Practices:
    
    Representative Workload:
    • Real-world data
    • Production-like scale
    • Typical usage patterns
    • Edge cases
    
    Isolate Performance:
    • Profile in isolation
    • Consistent environment
    • Minimize external factors
    • Reproducible results
    
    Focus on Hotspots:
    • 80/20 rule applies
    • Optimize critical paths
    • Measure improvements
    • Don't premature optimize
    
    Profile in Production:
    • Low-overhead profilers
    • Sampling profilers
    • Continuous profiling
    • Real traffic patterns
    """
    
    return {
        "messages": [AIMessage(content=f"⚡ Performance Profiler:\n{response.content}\n{report}")],
        "profiling_data": profiling_data,
        "hotspots": hotspots
    }


# Optimizer
def optimizer(state: ProfilingState) -> ProfilingState:
    """Analyzes profiling data and provides recommendations"""
    hotspots = state.get("hotspots", [])
    profiling_data = state.get("profiling_data", {})
    
    system_message = SystemMessage(content="""You are a performance optimizer.
    Analyze profiling results and recommend optimizations.""")
    
    user_message = HumanMessage(content=f"""Analyze performance:

Hotspots: {len(hotspots)}
Total Time: {profiling_data.get('total_execution_time_ms', 0):.2f}ms

Provide optimization recommendations.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate recommendations based on hotspots
    recommendations = []
    
    for hotspot in hotspots:
        if hotspot["percentage"] > 15:
            if "database" in hotspot["function"].lower():
                recommendations.append(
                    f"Optimize {hotspot['function']}: Consider query optimization, "
                    f"indexing, or caching (currently {hotspot['percentage']:.1f}% of execution time)"
                )
            elif "serialize" in hotspot["function"].lower():
                recommendations.append(
                    f"Optimize {hotspot['function']}: Use faster serialization library "
                    f"like orjson or msgpack ({hotspot['calls']} calls)"
                )
            else:
                recommendations.append(
                    f"Optimize {hotspot['function']}: Profile deeper to identify "
                    f"specific bottleneck ({hotspot['time_ms']:.2f}ms total)"
                )
    
    # Add general recommendations
    if profiling_data.get("io_wait_time_ms", 0) > profiling_data.get("cpu_time_ms", 0) * 0.2:
        recommendations.append(
            "High I/O wait time detected. Consider async I/O, connection pooling, "
            "or caching to reduce blocking operations."
        )
    
    summary = f"""
    📊 PROFILING COMPLETE
    
    Performance Summary:
    • Total Execution: {profiling_data.get('total_execution_time_ms', 0):.2f}ms
    • Hotspots Found: {len(hotspots)}
    • Top Bottleneck: {hotspots[0]['function']} ({hotspots[0]['percentage']:.1f}%) if hotspots else 'None'
    • Recommendations: {len(recommendations)}
    
    Top Hotspots:
    {chr(10).join(f"  {i+1}. {h['function']}() - {h['time_ms']:.2f}ms ({h['percentage']:.1f}%) - {h['calls']} calls" for i, h in enumerate(hotspots[:5]))}
    
    Profiling Pattern Process:
    1. Performance Profiler → Collect timing data
    2. Optimizer → Generate recommendations
    
    Optimization Strategies:
    
    Algorithmic:
    • Better data structures
    • O(n²) → O(n log n)
    • Reduce complexity
    • Smarter algorithms
    
    Caching:
    • Memoization
    • Query caching
    • CDN for static assets
    • Redis for hot data
    
    Parallelization:
    • Multiprocessing
    • Threading (I/O-bound)
    • Async/await
    • GPU acceleration
    
    Database:
    • Add indexes
    • Query optimization
    • Connection pooling
    • Read replicas
    • Denormalization
    
    Code-Level:
    • Avoid premature optimization
    • Use built-ins (faster)
    • List comprehensions
    • Generator expressions
    • Avoid global lookups
    
    Performance Metrics:
    
    Latency (Response Time):
    • P50: Median
    • P95: 95th percentile
    • P99: 99th percentile
    • P99.9: Tail latency
    
    Throughput:
    • Requests/second
    • Transactions/minute
    • Data processed/hour
    • Concurrent users
    
    Resource Usage:
    • CPU utilization
    • Memory consumption
    • Network bandwidth
    • Disk I/O
    
    Continuous Profiling:
    
    Tools:
    • Pyroscope
    • Google Cloud Profiler
    • AWS CodeGuru Profiler
    • Datadog Continuous Profiler
    
    Benefits:
    • Always-on profiling
    • Production insights
    • Historical data
    • Regression detection
    • Low overhead (<1%)
    
    Key Recommendations:
    {chr(10).join(f"  • {rec}" for rec in recommendations) if recommendations else "  • No specific recommendations - performance looks good!"}
    
    Key Insight:
    Performance profiling identifies bottlenecks and
    guides optimization efforts to maximize impact
    with minimal code changes.
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Optimizer:\n{response.content}\n{summary}")],
        "recommendations": recommendations
    }


# Build the graph
def build_profiling_graph():
    """Build the profiling pattern graph"""
    workflow = StateGraph(ProfilingState)
    
    workflow.add_node("performance_profiler", performance_profiler)
    workflow.add_node("optimizer", optimizer)
    
    workflow.add_edge(START, "performance_profiler")
    workflow.add_edge("performance_profiler", "optimizer")
    workflow.add_edge("optimizer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_profiling_graph()
    
    print("=== Profiling MCP Pattern ===\n")
    
    # Test Case: CPU profiling with optimization
    print("\n" + "="*70)
    print("TEST CASE: Application Performance Profiling")
    print("="*70)
    
    state = {
        "messages": [],
        "profiling_data": {},
        "hotspots": [],
        "recommendations": [],
        "profiler_type": "cpu"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nProfiling Results:")
    print(f"Total Time: {result.get('profiling_data', {}).get('total_execution_time_ms', 0):.2f}ms")
    print(f"Hotspots: {len(result.get('hotspots', []))}")
    print(f"Recommendations: {len(result.get('recommendations', []))}")

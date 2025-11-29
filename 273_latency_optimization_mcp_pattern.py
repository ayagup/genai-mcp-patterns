"""
Pattern 273: Latency Optimization MCP Pattern

This pattern demonstrates latency optimization strategies including response time
reduction, caching, pre-computation, and performance tuning.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class LatencyOptimizationState(TypedDict):
    """State for latency optimization workflow"""
    messages: Annotated[List[str], add]
    latency_metrics: Dict[str, Any]
    bottlenecks: List[Dict[str, Any]]
    optimization_plan: Dict[str, Any]
    performance_results: Dict[str, Any]


class LatencyProfiler:
    """Profiles request latencies and identifies slow points"""
    
    def __init__(self):
        self.percentiles = [50, 90, 95, 99]
    
    def profile_latencies(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Profile latency metrics"""
        latencies = [r["latency_ms"] for r in requests]
        latencies.sort()
        
        n = len(latencies)
        percentile_values = {}
        for p in self.percentiles:
            idx = int(n * p / 100)
            percentile_values[f"p{p}"] = latencies[min(idx, n-1)]
        
        # Group by operation
        by_operation = {}
        for req in requests:
            op = req["operation"]
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(req["latency_ms"])
        
        operation_stats = {}
        for op, lats in by_operation.items():
            operation_stats[op] = {
                "count": len(lats),
                "avg": sum(lats) / len(lats),
                "min": min(lats),
                "max": max(lats),
                "p95": sorted(lats)[int(len(lats) * 0.95)]
            }
        
        return {
            "total_requests": n,
            "avg_latency": sum(latencies) / n if n > 0 else 0,
            "min_latency": min(latencies) if latencies else 0,
            "max_latency": max(latencies) if latencies else 0,
            "percentiles": percentile_values,
            "by_operation": operation_stats
        }
    
    def identify_slow_operations(self, requests: List[Dict[str, Any]], 
                                 metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify slow operations"""
        slow_ops = []
        threshold = metrics["percentiles"]["p90"]
        
        for req in requests:
            if req["latency_ms"] > threshold:
                causes = []
                
                # Identify causes
                if req.get("cache_hit", True) == False:
                    causes.append("Cache miss")
                
                if req.get("db_queries", 0) > 3:
                    causes.append(f"Multiple DB queries ({req['db_queries']})")
                
                if req.get("external_calls", 0) > 0:
                    causes.append(f"External API calls ({req['external_calls']})")
                
                if req.get("computation_ms", 0) > 100:
                    causes.append("Heavy computation")
                
                if req.get("network_latency_ms", 0) > 50:
                    causes.append("High network latency")
                
                slow_ops.append({
                    "operation": req["operation"],
                    "latency_ms": req["latency_ms"],
                    "causes": causes if causes else ["Unknown"],
                    "severity": "critical" if req["latency_ms"] > threshold * 2 else "high"
                })
        
        return sorted(slow_ops, key=lambda x: x["latency_ms"], reverse=True)


class LatencyOptimizer:
    """Generates latency optimization strategies"""
    
    def __init__(self):
        self.optimization_strategies = {
            "Cache miss": {
                "strategy": "Implement caching",
                "techniques": ["Add cache layer", "Pre-warm cache", "Increase cache TTL"],
                "expected_improvement": 0.70
            },
            "Multiple DB queries": {
                "strategy": "Optimize database access",
                "techniques": ["Use query batching", "Add database indexes", "Implement connection pooling"],
                "expected_improvement": 0.60
            },
            "External API calls": {
                "strategy": "Reduce external dependencies",
                "techniques": ["Cache API responses", "Use async calls", "Implement circuit breakers"],
                "expected_improvement": 0.50
            },
            "Heavy computation": {
                "strategy": "Optimize computation",
                "techniques": ["Use pre-computation", "Implement lazy evaluation", "Optimize algorithms"],
                "expected_improvement": 0.65
            },
            "High network latency": {
                "strategy": "Reduce network overhead",
                "techniques": ["Use CDN", "Enable compression", "Minimize payload size"],
                "expected_improvement": 0.40
            }
        }
    
    def create_optimization_plan(self, bottlenecks: List[Dict[str, Any]], 
                                metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization plan"""
        optimizations = []
        
        # Group by operation
        operation_bottlenecks = {}
        for bottleneck in bottlenecks:
            op = bottleneck["operation"]
            if op not in operation_bottlenecks:
                operation_bottlenecks[op] = []
            operation_bottlenecks[op].append(bottleneck)
        
        # Create optimization for each operation
        for op, op_bottlenecks in operation_bottlenecks.items():
            all_causes = set()
            max_latency = 0
            
            for bottleneck in op_bottlenecks:
                all_causes.update(bottleneck["causes"])
                max_latency = max(max_latency, bottleneck["latency_ms"])
            
            # Select best optimization strategies
            strategies = []
            total_improvement = 1.0
            
            for cause in all_causes:
                if cause in self.optimization_strategies:
                    opt = self.optimization_strategies[cause]
                    strategies.append({
                        "target": cause,
                        "strategy": opt["strategy"],
                        "techniques": opt["techniques"],
                        "improvement_factor": opt["expected_improvement"]
                    })
                    total_improvement *= (1 - opt["expected_improvement"])
            
            if strategies:
                current_latency = metrics["by_operation"][op]["avg"]
                expected_latency = current_latency * total_improvement
                improvement = (current_latency - expected_latency) / current_latency if current_latency > 0 else 0
                
                optimizations.append({
                    "operation": op,
                    "current_latency_ms": current_latency,
                    "expected_latency_ms": expected_latency,
                    "improvement_percentage": improvement * 100,
                    "strategies": strategies,
                    "priority": 1 if max_latency > metrics["percentiles"]["p95"] * 2 else 2
                })
        
        return {
            "optimizations": sorted(optimizations, key=lambda x: x["improvement_percentage"], reverse=True),
            "total_operations": len(optimizations),
            "high_priority": sum(1 for o in optimizations if o["priority"] == 1)
        }


def collect_latency_metrics_agent(state: LatencyOptimizationState) -> LatencyOptimizationState:
    """Collect latency metrics"""
    print("\n⏱️ Collecting Latency Metrics...")
    
    # Sample requests with varying latencies
    requests = [
        {"operation": "search", "latency_ms": 45, "cache_hit": True, "db_queries": 1, "external_calls": 0, "computation_ms": 20, "network_latency_ms": 10},
        {"operation": "search", "latency_ms": 320, "cache_hit": False, "db_queries": 5, "external_calls": 1, "computation_ms": 150, "network_latency_ms": 80},
        {"operation": "search", "latency_ms": 55, "cache_hit": True, "db_queries": 1, "external_calls": 0, "computation_ms": 25, "network_latency_ms": 15},
        {"operation": "recommendation", "latency_ms": 890, "cache_hit": False, "db_queries": 8, "external_calls": 2, "computation_ms": 450, "network_latency_ms": 120},
        {"operation": "recommendation", "latency_ms": 120, "cache_hit": True, "db_queries": 2, "external_calls": 0, "computation_ms": 80, "network_latency_ms": 20},
        {"operation": "user_profile", "latency_ms": 25, "cache_hit": True, "db_queries": 1, "external_calls": 0, "computation_ms": 10, "network_latency_ms": 5},
        {"operation": "user_profile", "latency_ms": 450, "cache_hit": False, "db_queries": 6, "external_calls": 3, "computation_ms": 200, "network_latency_ms": 100},
        {"operation": "analytics", "latency_ms": 1200, "cache_hit": False, "db_queries": 12, "external_calls": 1, "computation_ms": 800, "network_latency_ms": 50},
        {"operation": "analytics", "latency_ms": 180, "cache_hit": True, "db_queries": 2, "external_calls": 0, "computation_ms": 120, "network_latency_ms": 30},
        {"operation": "search", "latency_ms": 280, "cache_hit": False, "db_queries": 4, "external_calls": 1, "computation_ms": 130, "network_latency_ms": 70}
    ]
    
    profiler = LatencyProfiler()
    metrics = profiler.profile_latencies(requests)
    
    print(f"\n  Total Requests: {metrics['total_requests']}")
    print(f"  Average Latency: {metrics['avg_latency']:.1f}ms")
    print(f"  Latency Range: {metrics['min_latency']}ms - {metrics['max_latency']}ms")
    print(f"\n  Percentiles:")
    for p, value in metrics["percentiles"].items():
        print(f"    {p.upper()}: {value}ms")
    
    print(f"\n  By Operation:")
    for op, stats in list(metrics["by_operation"].items())[:3]:
        print(f"    • {op}: avg={stats['avg']:.1f}ms, p95={stats['p95']}ms")
    
    return {
        **state,
        "latency_metrics": {**metrics, "requests": requests},
        "messages": [f"✓ Collected metrics for {len(requests)} requests"]
    }


def identify_bottlenecks_agent(state: LatencyOptimizationState) -> LatencyOptimizationState:
    """Identify latency bottlenecks"""
    print("\n🔍 Identifying Latency Bottlenecks...")
    
    profiler = LatencyProfiler()
    bottlenecks = profiler.identify_slow_operations(
        state["latency_metrics"]["requests"],
        state["latency_metrics"]
    )
    
    print(f"\n  Slow Operations Found: {len(bottlenecks)}")
    print(f"  Critical: {sum(1 for b in bottlenecks if b['severity'] == 'critical')}")
    print(f"  High: {sum(1 for b in bottlenecks if b['severity'] == 'high')}")
    
    print(f"\n  Top Bottlenecks:")
    for bottleneck in bottlenecks[:3]:
        print(f"\n    {bottleneck['operation']} ({bottleneck['severity'].upper()}):")
        print(f"      Latency: {bottleneck['latency_ms']}ms")
        print(f"      Causes: {', '.join(bottleneck['causes'])}")
    
    return {
        **state,
        "bottlenecks": bottlenecks,
        "messages": [f"✓ Identified {len(bottlenecks)} latency bottlenecks"]
    }


def create_optimization_plan_agent(state: LatencyOptimizationState) -> LatencyOptimizationState:
    """Create latency optimization plan"""
    print("\n💡 Creating Optimization Plan...")
    
    optimizer = LatencyOptimizer()
    plan = optimizer.create_optimization_plan(
        state["bottlenecks"],
        state["latency_metrics"]
    )
    
    print(f"\n  Operations to Optimize: {plan['total_operations']}")
    print(f"  High Priority: {plan['high_priority']}")
    
    print(f"\n  Top Optimizations:")
    for opt in plan["optimizations"][:3]:
        print(f"\n    {opt['operation']}:")
        print(f"      Current: {opt['current_latency_ms']:.1f}ms")
        print(f"      Expected: {opt['expected_latency_ms']:.1f}ms")
        print(f"      Improvement: {opt['improvement_percentage']:.1f}%")
        print(f"      Strategies: {len(opt['strategies'])}")
    
    return {
        **state,
        "optimization_plan": plan,
        "messages": [f"✓ Created plan for {plan['total_operations']} operations"]
    }


def generate_latency_report_agent(state: LatencyOptimizationState) -> LatencyOptimizationState:
    """Generate latency optimization report"""
    print("\n" + "="*70)
    print("LATENCY OPTIMIZATION REPORT")
    print("="*70)
    
    metrics = state["latency_metrics"]
    print(f"\n⏱️ Current Latency Profile:")
    print(f"  Total Requests Analyzed: {metrics['total_requests']}")
    print(f"  Average Latency: {metrics['avg_latency']:.1f}ms")
    print(f"  Latency Range: {metrics['min_latency']}ms - {metrics['max_latency']}ms")
    
    print(f"\n  Latency Percentiles:")
    for p, value in metrics["percentiles"].items():
        print(f"    {p.upper()}: {value}ms")
    
    print(f"\n  Latency by Operation:")
    for op, stats in metrics["by_operation"].items():
        print(f"\n    {op}:")
        print(f"      Requests: {stats['count']}")
        print(f"      Average: {stats['avg']:.1f}ms")
        print(f"      P95: {stats['p95']}ms")
        print(f"      Range: {stats['min']}ms - {stats['max']}ms")
    
    print(f"\n🔍 Latency Bottlenecks:")
    print(f"  Total Bottlenecks: {len(state['bottlenecks'])}")
    print(f"  Critical: {sum(1 for b in state['bottlenecks'] if b['severity'] == 'critical')}")
    print(f"  High: {sum(1 for b in state['bottlenecks'] if b['severity'] == 'high')}")
    
    for i, bottleneck in enumerate(state["bottlenecks"][:5], 1):
        severity_label = "🔴 CRITICAL" if bottleneck["severity"] == "critical" else "🟡 HIGH"
        print(f"\n  {i}. {severity_label}: {bottleneck['operation']}")
        print(f"      Latency: {bottleneck['latency_ms']}ms")
        print(f"      Root Causes: {', '.join(bottleneck['causes'])}")
    
    print(f"\n💡 Optimization Plan:")
    plan = state["optimization_plan"]
    print(f"  Operations to Optimize: {plan['total_operations']}")
    print(f"  High Priority: {plan['high_priority']}")
    
    for i, opt in enumerate(plan["optimizations"], 1):
        priority_label = "🔴 HIGH" if opt["priority"] == 1 else "🟡 MEDIUM"
        print(f"\n  {i}. {priority_label}: {opt['operation']}")
        print(f"      Current Latency: {opt['current_latency_ms']:.1f}ms")
        print(f"      Expected Latency: {opt['expected_latency_ms']:.1f}ms")
        print(f"      Improvement: {opt['improvement_percentage']:.1f}%")
        print(f"\n      Optimization Strategies:")
        for strategy in opt["strategies"]:
            print(f"        • {strategy['strategy']} (targeting: {strategy['target']})")
            print(f"          Techniques: {', '.join(strategy['techniques'][:2])}")
    
    print(f"\n📈 Expected Performance Improvements:")
    total_improvement = sum(opt["improvement_percentage"] for opt in plan["optimizations"]) / len(plan["optimizations"]) if plan["optimizations"] else 0
    current_avg = metrics["avg_latency"]
    expected_avg = current_avg * (1 - total_improvement / 100)
    
    print(f"  Current Average Latency: {current_avg:.1f}ms")
    print(f"  Expected Average Latency: {expected_avg:.1f}ms")
    print(f"  Average Improvement: {total_improvement:.1f}%")
    print(f"  Speedup Factor: {current_avg / expected_avg:.2f}x")
    
    print(f"\n💡 Latency Optimization Benefits:")
    print("  • Improved user experience")
    print("  • Higher request throughput")
    print("  • Reduced resource consumption")
    print("  • Better system scalability")
    print("  • Lower infrastructure costs")
    print("  • Increased customer satisfaction")
    
    print("\n="*70)
    print("✅ Latency Optimization Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_latency_optimization_graph():
    workflow = StateGraph(LatencyOptimizationState)
    workflow.add_node("collect_metrics", collect_latency_metrics_agent)
    workflow.add_node("identify_bottlenecks", identify_bottlenecks_agent)
    workflow.add_node("create_plan", create_optimization_plan_agent)
    workflow.add_node("generate_report", generate_latency_report_agent)
    workflow.add_edge(START, "collect_metrics")
    workflow.add_edge("collect_metrics", "identify_bottlenecks")
    workflow.add_edge("identify_bottlenecks", "create_plan")
    workflow.add_edge("create_plan", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 273: Latency Optimization MCP Pattern")
    print("="*70)
    
    app = create_latency_optimization_graph()
    final_state = app.invoke({
        "messages": [],
        "latency_metrics": {},
        "bottlenecks": [],
        "optimization_plan": {},
        "performance_results": {}
    })
    print("\n✅ Latency Optimization Pattern Complete!")


if __name__ == "__main__":
    main()

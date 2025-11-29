"""
Pattern 276: Query Optimization MCP Pattern

This pattern demonstrates database query optimization including query analysis,
index recommendations, and performance tuning.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class QueryOptimizationState(TypedDict):
    """State for query optimization workflow"""
    messages: Annotated[List[str], add]
    query_analysis: Dict[str, Any]
    slow_queries: List[Dict[str, Any]]
    optimization_recommendations: List[Dict[str, Any]]
    performance_projection: Dict[str, Any]


class QueryAnalyzer:
    """Analyzes database queries for performance issues"""
    
    def __init__(self):
        self.slow_threshold_ms = 100
        self.very_slow_threshold_ms = 1000
    
    def analyze_queries(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze query performance"""
        total_queries = len(queries)
        total_execution_time = sum(q["execution_time_ms"] for q in queries)
        avg_execution_time = total_execution_time / total_queries if total_queries > 0 else 0
        
        slow_queries = [q for q in queries if q["execution_time_ms"] > self.slow_threshold_ms]
        very_slow_queries = [q for q in queries if q["execution_time_ms"] > self.very_slow_threshold_ms]
        
        # Group by query type
        by_type = {}
        for query in queries:
            qtype = query["type"]
            if qtype not in by_type:
                by_type[qtype] = {"count": 0, "total_time": 0}
            by_type[qtype]["count"] += 1
            by_type[qtype]["total_time"] += query["execution_time_ms"]
        
        return {
            "total_queries": total_queries,
            "total_execution_time": total_execution_time,
            "avg_execution_time": avg_execution_time,
            "slow_queries_count": len(slow_queries),
            "very_slow_queries_count": len(very_slow_queries),
            "by_type": by_type
        }
    
    def identify_slow_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify and categorize slow queries"""
        slow_queries = []
        
        for query in queries:
            if query["execution_time_ms"] > self.slow_threshold_ms:
                issues = []
                
                # Check for missing indexes
                if not query.get("uses_index", True):
                    issues.append("Table scan - missing index")
                
                # Check for inefficient joins
                if query.get("join_count", 0) > 3:
                    issues.append(f"Multiple joins ({query['join_count']})")
                
                # Check for large result sets
                if query.get("rows_examined", 0) > 10000:
                    issues.append(f"Large dataset scan ({query['rows_examined']} rows)")
                
                # Check for subqueries
                if query.get("has_subquery", False):
                    issues.append("Contains subqueries")
                
                # Check for wildcard searches
                if query.get("has_wildcard", False):
                    issues.append("Wildcard search")
                
                # Check for OR conditions
                if query.get("has_or_conditions", False):
                    issues.append("OR conditions prevent index usage")
                
                severity = "critical" if query["execution_time_ms"] > self.very_slow_threshold_ms else "high"
                
                slow_queries.append({
                    "query_id": query["id"],
                    "query_type": query["type"],
                    "execution_time_ms": query["execution_time_ms"],
                    "rows_examined": query.get("rows_examined", 0),
                    "rows_returned": query.get("rows_returned", 0),
                    "issues": issues if issues else ["General inefficiency"],
                    "severity": severity
                })
        
        return sorted(slow_queries, key=lambda x: x["execution_time_ms"], reverse=True)


class QueryOptimizer:
    """Generates query optimization recommendations"""
    
    def __init__(self):
        self.optimization_strategies = {
            "Table scan - missing index": {
                "recommendation": "Add appropriate indexes",
                "techniques": ["Create index on WHERE clause columns", "Add composite indexes", "Analyze query execution plan"],
                "expected_improvement": 0.80
            },
            "Multiple joins": {
                "recommendation": "Optimize join strategy",
                "techniques": ["Reduce number of joins", "Use covering indexes", "Denormalize data"],
                "expected_improvement": 0.50
            },
            "Large dataset scan": {
                "recommendation": "Limit data scanned",
                "techniques": ["Add WHERE clause filters", "Use pagination", "Partition tables"],
                "expected_improvement": 0.70
            },
            "Contains subqueries": {
                "recommendation": "Rewrite subqueries",
                "techniques": ["Convert to joins", "Use CTEs", "Materialize intermediate results"],
                "expected_improvement": 0.60
            },
            "Wildcard search": {
                "recommendation": "Optimize search strategy",
                "techniques": ["Use full-text search", "Limit wildcard position", "Add search indexes"],
                "expected_improvement": 0.75
            },
            "OR conditions prevent index usage": {
                "recommendation": "Rewrite OR conditions",
                "techniques": ["Use UNION instead of OR", "Split into multiple queries", "Use IN clause"],
                "expected_improvement": 0.65
            }
        }
    
    def generate_recommendations(self, slow_queries: List[Dict[str, Any]], 
                                analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for query in slow_queries:
            query_recommendations = []
            total_improvement = 1.0
            
            for issue in query["issues"]:
                if issue in self.optimization_strategies:
                    strategy = self.optimization_strategies[issue]
                    query_recommendations.append({
                        "issue": issue,
                        "recommendation": strategy["recommendation"],
                        "techniques": strategy["techniques"],
                        "improvement_factor": strategy["expected_improvement"]
                    })
                    total_improvement *= (1 - strategy["expected_improvement"])
            
            if query_recommendations:
                current_time = query["execution_time_ms"]
                expected_time = current_time * total_improvement
                improvement_pct = ((current_time - expected_time) / current_time * 100) if current_time > 0 else 0
                
                recommendations.append({
                    "query_id": query["query_id"],
                    "query_type": query["query_type"],
                    "current_time_ms": current_time,
                    "expected_time_ms": expected_time,
                    "improvement_percentage": improvement_pct,
                    "time_saved_ms": current_time - expected_time,
                    "recommendations": query_recommendations,
                    "priority": 1 if query["severity"] == "critical" else 2
                })
        
        return sorted(recommendations, key=lambda x: x["time_saved_ms"], reverse=True)


def collect_query_data_agent(state: QueryOptimizationState) -> QueryOptimizationState:
    """Collect query performance data"""
    print("\n🔍 Collecting Query Performance Data...")
    
    queries = [
        {"id": "Q1", "type": "SELECT", "execution_time_ms": 45, "uses_index": True, "join_count": 1, 
         "rows_examined": 500, "rows_returned": 10, "has_subquery": False, "has_wildcard": False, "has_or_conditions": False},
        {"id": "Q2", "type": "SELECT", "execution_time_ms": 2500, "uses_index": False, "join_count": 5, 
         "rows_examined": 50000, "rows_returned": 100, "has_subquery": True, "has_wildcard": True, "has_or_conditions": False},
        {"id": "Q3", "type": "UPDATE", "execution_time_ms": 180, "uses_index": True, "join_count": 2, 
         "rows_examined": 2000, "rows_returned": 50, "has_subquery": False, "has_wildcard": False, "has_or_conditions": True},
        {"id": "Q4", "type": "SELECT", "execution_time_ms": 1200, "uses_index": False, "join_count": 4, 
         "rows_examined": 30000, "rows_returned": 200, "has_subquery": True, "has_wildcard": False, "has_or_conditions": False},
        {"id": "Q5", "type": "SELECT", "execution_time_ms": 80, "uses_index": True, "join_count": 2, 
         "rows_examined": 1000, "rows_returned": 20, "has_subquery": False, "has_wildcard": False, "has_or_conditions": False},
        {"id": "Q6", "type": "INSERT", "execution_time_ms": 30, "uses_index": True, "join_count": 0, 
         "rows_examined": 1, "rows_returned": 1, "has_subquery": False, "has_wildcard": False, "has_or_conditions": False},
        {"id": "Q7", "type": "SELECT", "execution_time_ms": 850, "uses_index": True, "join_count": 3, 
         "rows_examined": 15000, "rows_returned": 150, "has_subquery": False, "has_wildcard": True, "has_or_conditions": False},
        {"id": "Q8", "type": "DELETE", "execution_time_ms": 120, "uses_index": True, "join_count": 1, 
         "rows_examined": 500, "rows_returned": 25, "has_subquery": False, "has_wildcard": False, "has_or_conditions": False}
    ]
    
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze_queries(queries)
    
    print(f"\n  Total Queries Analyzed: {analysis['total_queries']}")
    print(f"  Average Execution Time: {analysis['avg_execution_time']:.1f}ms")
    print(f"  Slow Queries: {analysis['slow_queries_count']}")
    print(f"  Very Slow Queries: {analysis['very_slow_queries_count']}")
    
    print(f"\n  Queries by Type:")
    for qtype, stats in list(analysis["by_type"].items())[:3]:
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        print(f"    • {qtype}: {stats['count']} queries, avg {avg_time:.1f}ms")
    
    return {
        **state,
        "query_analysis": {**analysis, "queries": queries},
        "messages": [f"✓ Analyzed {len(queries)} queries"]
    }


def identify_slow_queries_agent(state: QueryOptimizationState) -> QueryOptimizationState:
    """Identify slow queries"""
    print("\n🐌 Identifying Slow Queries...")
    
    analyzer = QueryAnalyzer()
    slow_queries = analyzer.identify_slow_queries(state["query_analysis"]["queries"])
    
    print(f"\n  Slow Queries Found: {len(slow_queries)}")
    print(f"  Critical: {sum(1 for q in slow_queries if q['severity'] == 'critical')}")
    print(f"  High: {sum(1 for q in slow_queries if q['severity'] == 'high')}")
    
    print(f"\n  Top Slow Queries:")
    for query in slow_queries[:3]:
        print(f"\n    {query['query_id']} ({query['severity'].upper()}):")
        print(f"      Type: {query['query_type']}")
        print(f"      Execution Time: {query['execution_time_ms']}ms")
        print(f"      Rows Examined: {query['rows_examined']}")
        print(f"      Issues: {', '.join(query['issues'])}")
    
    return {
        **state,
        "slow_queries": slow_queries,
        "messages": [f"✓ Identified {len(slow_queries)} slow queries"]
    }


def generate_optimization_recommendations_agent(state: QueryOptimizationState) -> QueryOptimizationState:
    """Generate optimization recommendations"""
    print("\n💡 Generating Optimization Recommendations...")
    
    optimizer = QueryOptimizer()
    recommendations = optimizer.generate_recommendations(
        state["slow_queries"],
        state["query_analysis"]
    )
    
    # Calculate overall impact
    total_time_saved = sum(r["time_saved_ms"] for r in recommendations)
    current_slow_query_time = sum(q["execution_time_ms"] for q in state["slow_queries"])
    improvement_pct = (total_time_saved / current_slow_query_time * 100) if current_slow_query_time > 0 else 0
    
    projection = {
        "current_slow_query_time": current_slow_query_time,
        "expected_time": current_slow_query_time - total_time_saved,
        "time_saved": total_time_saved,
        "improvement_percentage": improvement_pct,
        "queries_optimized": len(recommendations)
    }
    
    print(f"\n  Recommendations Generated: {len(recommendations)}")
    print(f"  Total Time Savings: {total_time_saved:.1f}ms")
    print(f"  Overall Improvement: {improvement_pct:.1f}%")
    
    return {
        **state,
        "optimization_recommendations": recommendations,
        "performance_projection": projection,
        "messages": [f"✓ Generated {len(recommendations)} recommendations"]
    }


def generate_query_report_agent(state: QueryOptimizationState) -> QueryOptimizationState:
    """Generate query optimization report"""
    print("\n" + "="*70)
    print("QUERY OPTIMIZATION REPORT")
    print("="*70)
    
    analysis = state["query_analysis"]
    print(f"\n📊 Query Performance Summary:")
    print(f"  Total Queries: {analysis['total_queries']}")
    print(f"  Total Execution Time: {analysis['total_execution_time']}ms")
    print(f"  Average Execution Time: {analysis['avg_execution_time']:.1f}ms")
    print(f"  Slow Queries: {analysis['slow_queries_count']}")
    print(f"  Very Slow Queries: {analysis['very_slow_queries_count']}")
    
    print(f"\n  Performance by Query Type:")
    for qtype, stats in analysis["by_type"].items():
        avg_time = stats["total_time"] / stats["count"] if stats["count"] > 0 else 0
        print(f"    • {qtype}: {stats['count']} queries, avg {avg_time:.1f}ms, total {stats['total_time']:.1f}ms")
    
    print(f"\n🐌 Slow Query Analysis:")
    print(f"  Total Slow Queries: {len(state['slow_queries'])}")
    
    for i, query in enumerate(state["slow_queries"], 1):
        severity_label = "🔴 CRITICAL" if query["severity"] == "critical" else "🟡 HIGH"
        print(f"\n  {i}. {severity_label}: {query['query_id']} ({query['query_type']})")
        print(f"      Execution Time: {query['execution_time_ms']}ms")
        print(f"      Rows Examined: {query['rows_examined']:,}")
        print(f"      Rows Returned: {query['rows_returned']}")
        print(f"      Issues:")
        for issue in query["issues"]:
            print(f"        • {issue}")
    
    print(f"\n💡 Optimization Recommendations:")
    for i, rec in enumerate(state["optimization_recommendations"], 1):
        priority_label = "🔴 HIGH" if rec["priority"] == 1 else "🟡 MEDIUM"
        print(f"\n  {i}. {priority_label}: {rec['query_id']} ({rec['query_type']})")
        print(f"      Current Time: {rec['current_time_ms']}ms")
        print(f"      Expected Time: {rec['expected_time_ms']:.1f}ms")
        print(f"      Improvement: {rec['improvement_percentage']:.1f}%")
        print(f"      Time Saved: {rec['time_saved_ms']:.1f}ms")
        print(f"\n      Recommendations:")
        for recommendation in rec["recommendations"]:
            print(f"        • {recommendation['recommendation']}")
            print(f"          Issue: {recommendation['issue']}")
            print(f"          Techniques: {', '.join(recommendation['techniques'][:2])}")
    
    print(f"\n📈 Performance Projection:")
    proj = state["performance_projection"]
    print(f"  Current Slow Query Time: {proj['current_slow_query_time']}ms")
    print(f"  Expected Time After Optimization: {proj['expected_time']:.1f}ms")
    print(f"  Total Time Saved: {proj['time_saved']:.1f}ms")
    print(f"  Overall Improvement: {proj['improvement_percentage']:.1f}%")
    print(f"  Queries Optimized: {proj['queries_optimized']}")
    
    print(f"\n💡 Query Optimization Benefits:")
    print("  • Faster response times")
    print("  • Reduced database load")
    print("  • Lower infrastructure costs")
    print("  • Better scalability")
    print("  • Improved user experience")
    print("  • More efficient resource usage")
    
    print("\n="*70)
    print("✅ Query Optimization Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_query_optimization_graph():
    workflow = StateGraph(QueryOptimizationState)
    workflow.add_node("collect_data", collect_query_data_agent)
    workflow.add_node("identify_slow", identify_slow_queries_agent)
    workflow.add_node("generate_recommendations", generate_optimization_recommendations_agent)
    workflow.add_node("generate_report", generate_query_report_agent)
    workflow.add_edge(START, "collect_data")
    workflow.add_edge("collect_data", "identify_slow")
    workflow.add_edge("identify_slow", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 276: Query Optimization MCP Pattern")
    print("="*70)
    
    app = create_query_optimization_graph()
    final_state = app.invoke({
        "messages": [],
        "query_analysis": {},
        "slow_queries": [],
        "optimization_recommendations": [],
        "performance_projection": {}
    })
    print("\n✅ Query Optimization Pattern Complete!")


if __name__ == "__main__":
    main()

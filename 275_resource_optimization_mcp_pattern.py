"""
Pattern 275: Resource Optimization MCP Pattern

This pattern demonstrates resource optimization including CPU, memory,
storage, and network resource management and allocation.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class ResourceOptimizationState(TypedDict):
    """State for resource optimization workflow"""
    messages: Annotated[List[str], add]
    resource_usage: Dict[str, Any]
    inefficiencies: List[Dict[str, Any]]
    allocation_plan: Dict[str, Any]
    optimization_results: Dict[str, Any]


class ResourceAnalyzer:
    """Analyzes resource usage patterns"""
    
    def __init__(self):
        self.thresholds = {
            "cpu": {"warning": 70, "critical": 90},
            "memory": {"warning": 75, "critical": 90},
            "storage": {"warning": 80, "critical": 95},
            "network": {"warning": 70, "critical": 85}
        }
    
    def analyze_resources(self, resources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource usage"""
        by_type = {}
        total_allocated = {}
        total_used = {}
        
        for resource in resources:
            rtype = resource["type"]
            if rtype not in by_type:
                by_type[rtype] = []
                total_allocated[rtype] = 0
                total_used[rtype] = 0
            
            by_type[rtype].append(resource)
            total_allocated[rtype] += resource["allocated"]
            total_used[rtype] += resource["used"]
        
        # Calculate utilization by type
        utilization = {}
        for rtype in by_type:
            util = (total_used[rtype] / total_allocated[rtype] * 100) if total_allocated[rtype] > 0 else 0
            utilization[rtype] = {
                "allocated": total_allocated[rtype],
                "used": total_used[rtype],
                "utilization": util,
                "wasted": total_allocated[rtype] - total_used[rtype]
            }
        
        return {
            "by_type": by_type,
            "utilization": utilization,
            "total_resources": len(resources)
        }
    
    def identify_inefficiencies(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify resource inefficiencies"""
        inefficiencies = []
        
        for resource in resources:
            rtype = resource["type"]
            allocated = resource["allocated"]
            used = resource["used"]
            util = (used / allocated * 100) if allocated > 0 else 0
            
            issues = []
            severity = "low"
            
            # Check for over-allocation
            if util < 30:
                issues.append("Severe over-allocation (<30% utilization)")
                severity = "high"
            elif util < 50:
                issues.append("Over-allocation (<50% utilization)")
                severity = "medium"
            
            # Check for under-allocation (approaching limits)
            if util > self.thresholds[rtype]["critical"]:
                issues.append(f"Critical resource pressure (>{self.thresholds[rtype]['critical']}%)")
                severity = "critical"
            elif util > self.thresholds[rtype]["warning"]:
                issues.append(f"Resource pressure (>{self.thresholds[rtype]['warning']}%)")
                severity = "high" if severity == "low" else severity
            
            # Check for fragmentation
            if resource.get("fragmented", False):
                issues.append("Resource fragmentation")
                severity = "medium" if severity == "low" else severity
            
            # Check for idle resources
            if resource.get("idle_time_pct", 0) > 80:
                issues.append("Mostly idle resource (>80% idle time)")
                severity = "medium" if severity == "low" else severity
            
            if issues:
                waste_amount = allocated - used
                inefficiencies.append({
                    "service": resource["service"],
                    "type": rtype,
                    "allocated": allocated,
                    "used": used,
                    "utilization": util,
                    "waste_amount": waste_amount,
                    "issues": issues,
                    "severity": severity
                })
        
        return sorted(inefficiencies, key=lambda x: x["waste_amount"], reverse=True)


class ResourceOptimizer:
    """Generates resource optimization recommendations"""
    
    def __init__(self):
        self.optimization_strategies = {
            "cpu": {
                "over_allocated": ["Right-size CPU allocation", "Use auto-scaling", "Enable CPU throttling"],
                "under_allocated": ["Increase CPU allocation", "Add CPU cores", "Optimize workload distribution"],
                "fragmented": ["Consolidate workloads", "Use CPU pinning", "Optimize thread pooling"]
            },
            "memory": {
                "over_allocated": ["Reduce memory limits", "Implement memory pooling", "Enable memory compression"],
                "under_allocated": ["Increase memory allocation", "Add RAM", "Optimize memory usage"],
                "fragmented": ["Defragment memory", "Use memory compaction", "Optimize allocation patterns"]
            },
            "storage": {
                "over_allocated": ["Reduce storage allocation", "Implement storage tiering", "Enable compression"],
                "under_allocated": ["Increase storage capacity", "Add storage volumes", "Optimize data placement"],
                "fragmented": ["Defragment storage", "Optimize file placement", "Use storage pools"]
            },
            "network": {
                "over_allocated": ["Optimize bandwidth allocation", "Implement traffic shaping", "Use QoS"],
                "under_allocated": ["Increase bandwidth", "Add network interfaces", "Optimize routing"],
                "fragmented": ["Optimize network topology", "Reduce network hops", "Use connection pooling"]
            }
        }
    
    def create_allocation_plan(self, inefficiencies: List[Dict[str, Any]], 
                               analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create resource allocation plan"""
        recommendations = []
        
        for inefficiency in inefficiencies:
            rtype = inefficiency["type"]
            util = inefficiency["utilization"]
            
            # Determine optimization category
            if util < 50:
                category = "over_allocated"
                target_allocation = inefficiency["used"] * 1.3  # 30% buffer
            elif util > 80:
                category = "under_allocated"
                target_allocation = inefficiency["allocated"] * 1.5  # 50% increase
            else:
                category = "fragmented" if "fragmentation" in str(inefficiency["issues"]) else "over_allocated"
                target_allocation = inefficiency["used"] * 1.2  # 20% buffer
            
            # Get strategies
            strategies = self.optimization_strategies.get(rtype, {}).get(category, [])
            
            resource_saved = inefficiency["allocated"] - target_allocation
            
            recommendations.append({
                "service": inefficiency["service"],
                "resource_type": rtype,
                "current_allocation": inefficiency["allocated"],
                "current_usage": inefficiency["used"],
                "current_utilization": util,
                "recommended_allocation": max(target_allocation, inefficiency["used"]),
                "resource_saved": max(resource_saved, 0),
                "savings_percentage": (resource_saved / inefficiency["allocated"] * 100) if inefficiency["allocated"] > 0 and resource_saved > 0 else 0,
                "strategies": strategies[:3],
                "priority": 1 if inefficiency["severity"] in ["critical", "high"] else 2,
                "implementation_effort": "low" if util < 50 else "medium"
            })
        
        return {
            "recommendations": sorted(recommendations, key=lambda x: x["resource_saved"], reverse=True),
            "total_recommendations": len(recommendations),
            "high_priority": sum(1 for r in recommendations if r["priority"] == 1)
        }


def collect_resource_usage_agent(state: ResourceOptimizationState) -> ResourceOptimizationState:
    """Collect resource usage data"""
    print("\n📊 Collecting Resource Usage Data...")
    
    resources = [
        {"service": "Web Server", "type": "cpu", "allocated": 8, "used": 2.1, "fragmented": False, "idle_time_pct": 60},
        {"service": "Web Server", "type": "memory", "allocated": 16384, "used": 12800, "fragmented": False, "idle_time_pct": 20},
        {"service": "Database", "type": "cpu", "allocated": 16, "used": 14.2, "fragmented": False, "idle_time_pct": 10},
        {"service": "Database", "type": "memory", "allocated": 32768, "used": 28000, "fragmented": True, "idle_time_pct": 5},
        {"service": "Database", "type": "storage", "allocated": 500000, "used": 420000, "fragmented": False, "idle_time_pct": 0},
        {"service": "Cache", "type": "memory", "allocated": 8192, "used": 2400, "fragmented": False, "idle_time_pct": 70},
        {"service": "API Gateway", "type": "cpu", "allocated": 4, "used": 1.2, "fragmented": False, "idle_time_pct": 65},
        {"service": "API Gateway", "type": "network", "allocated": 10000, "used": 3200, "fragmented": False, "idle_time_pct": 50},
        {"service": "Worker", "type": "cpu", "allocated": 12, "used": 3.5, "fragmented": False, "idle_time_pct": 75},
        {"service": "Worker", "type": "memory", "allocated": 16384, "used": 4800, "fragmented": False, "idle_time_pct": 70},
        {"service": "File Server", "type": "storage", "allocated": 1000000, "used": 890000, "fragmented": True, "idle_time_pct": 0},
        {"service": "Load Balancer", "type": "network", "allocated": 20000, "used": 18500, "fragmented": False, "idle_time_pct": 5}
    ]
    
    print(f"\n  Services Monitored: {len(set(r['service'] for r in resources))}")
    print(f"  Resource Types: {len(set(r['type'] for r in resources))}")
    print(f"  Total Resources: {len(resources)}")
    
    print(f"\n  Sample Resources:")
    for resource in resources[:3]:
        util = (resource["used"] / resource["allocated"] * 100) if resource["allocated"] > 0 else 0
        print(f"    • {resource['service']} ({resource['type']}): {util:.1f}% utilized")
    
    return {
        **state,
        "resource_usage": {"resources": resources},
        "messages": [f"✓ Collected data for {len(resources)} resources"]
    }


def analyze_resources_agent(state: ResourceOptimizationState) -> ResourceOptimizationState:
    """Analyze resource usage"""
    print("\n🔍 Analyzing Resource Usage...")
    
    analyzer = ResourceAnalyzer()
    analysis = analyzer.analyze_resources(state["resource_usage"]["resources"])
    inefficiencies = analyzer.identify_inefficiencies(state["resource_usage"]["resources"])
    
    print(f"\n  Resource Types Analyzed: {len(analysis['utilization'])}")
    print(f"\n  Utilization by Type:")
    for rtype, util_data in analysis["utilization"].items():
        print(f"    • {rtype.upper()}: {util_data['utilization']:.1f}%")
        print(f"      Allocated: {util_data['allocated']}, Used: {util_data['used']}, Wasted: {util_data['wasted']}")
    
    print(f"\n  Inefficiencies Found: {len(inefficiencies)}")
    print(f"  Critical: {sum(1 for i in inefficiencies if i['severity'] == 'critical')}")
    print(f"  High: {sum(1 for i in inefficiencies if i['severity'] == 'high')}")
    
    analysis["inefficiencies"] = inefficiencies
    
    return {
        **state,
        "resource_usage": {**state["resource_usage"], "analysis": analysis},
        "inefficiencies": inefficiencies,
        "messages": [f"✓ Identified {len(inefficiencies)} resource inefficiencies"]
    }


def create_allocation_plan_agent(state: ResourceOptimizationState) -> ResourceOptimizationState:
    """Create resource allocation plan"""
    print("\n💡 Creating Resource Allocation Plan...")
    
    optimizer = ResourceOptimizer()
    plan = optimizer.create_allocation_plan(
        state["inefficiencies"],
        state["resource_usage"]["analysis"]
    )
    
    # Calculate total savings
    total_savings = {}
    for rec in plan["recommendations"]:
        rtype = rec["resource_type"]
        if rtype not in total_savings:
            total_savings[rtype] = 0
        total_savings[rtype] += rec["resource_saved"]
    
    print(f"\n  Recommendations: {plan['total_recommendations']}")
    print(f"  High Priority: {plan['high_priority']}")
    
    print(f"\n  Potential Resource Savings:")
    for rtype, saved in total_savings.items():
        print(f"    • {rtype.upper()}: {saved:.1f} units")
    
    return {
        **state,
        "allocation_plan": plan,
        "optimization_results": {"total_savings": total_savings},
        "messages": [f"✓ Created plan with {plan['total_recommendations']} recommendations"]
    }


def generate_resource_report_agent(state: ResourceOptimizationState) -> ResourceOptimizationState:
    """Generate resource optimization report"""
    print("\n" + "="*70)
    print("RESOURCE OPTIMIZATION REPORT")
    print("="*70)
    
    analysis = state["resource_usage"]["analysis"]
    print(f"\n📊 Resource Utilization Summary:")
    print(f"  Total Resources: {analysis['total_resources']}")
    
    print(f"\n  By Resource Type:")
    for rtype, util_data in analysis["utilization"].items():
        print(f"\n    {rtype.upper()}:")
        print(f"      Allocated: {util_data['allocated']:.1f} units")
        print(f"      Used: {util_data['used']:.1f} units")
        print(f"      Utilization: {util_data['utilization']:.1f}%")
        print(f"      Wasted: {util_data['wasted']:.1f} units")
    
    print(f"\n🔍 Resource Inefficiencies:")
    print(f"  Total Inefficiencies: {len(state['inefficiencies'])}")
    print(f"  Critical: {sum(1 for i in state['inefficiencies'] if i['severity'] == 'critical')}")
    print(f"  High: {sum(1 for i in state['inefficiencies'] if i['severity'] == 'high')}")
    
    for i, ineff in enumerate(state["inefficiencies"][:5], 1):
        severity_map = {"critical": "🔴 CRITICAL", "high": "🟡 HIGH", "medium": "🟢 MEDIUM", "low": "⚪ LOW"}
        severity_label = severity_map.get(ineff["severity"], "⚪ LOW")
        
        print(f"\n  {i}. {severity_label}: {ineff['service']} - {ineff['type'].upper()}")
        print(f"      Allocated: {ineff['allocated']:.1f}, Used: {ineff['used']:.1f}")
        print(f"      Utilization: {ineff['utilization']:.1f}%")
        print(f"      Wasted: {ineff['waste_amount']:.1f} units")
        print(f"      Issues: {', '.join(ineff['issues'])}")
    
    print(f"\n💡 Resource Allocation Plan:")
    plan = state["allocation_plan"]
    print(f"  Total Recommendations: {plan['total_recommendations']}")
    print(f"  High Priority: {plan['high_priority']}")
    
    for i, rec in enumerate(plan["recommendations"][:5], 1):
        priority_label = "🔴 HIGH" if rec["priority"] == 1 else "🟡 MEDIUM"
        print(f"\n  {i}. {priority_label}: {rec['service']} - {rec['resource_type'].upper()}")
        print(f"      Current Allocation: {rec['current_allocation']:.1f}")
        print(f"      Current Usage: {rec['current_usage']:.1f}")
        print(f"      Current Utilization: {rec['current_utilization']:.1f}%")
        print(f"      Recommended Allocation: {rec['recommended_allocation']:.1f}")
        print(f"      Resource Saved: {rec['resource_saved']:.1f} ({rec['savings_percentage']:.1f}%)")
        print(f"      Implementation: {rec['implementation_effort'].upper()} effort")
        print(f"      Strategies:")
        for strategy in rec["strategies"]:
            print(f"        • {strategy}")
    
    print(f"\n📈 Optimization Impact:")
    savings = state["optimization_results"]["total_savings"]
    print(f"  Total Resource Savings by Type:")
    for rtype, saved in savings.items():
        allocated = analysis["utilization"][rtype]["allocated"]
        savings_pct = (saved / allocated * 100) if allocated > 0 else 0
        print(f"    • {rtype.upper()}: {saved:.1f} units ({savings_pct:.1f}% reduction)")
    
    print(f"\n💡 Resource Optimization Benefits:")
    print("  • Reduced infrastructure costs")
    print("  • Better resource utilization")
    print("  • Improved system performance")
    print("  • Reduced waste and inefficiency")
    print("  • More sustainable operations")
    print("  • Predictable resource usage")
    
    print("\n="*70)
    print("✅ Resource Optimization Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_resource_optimization_graph():
    workflow = StateGraph(ResourceOptimizationState)
    workflow.add_node("collect_usage", collect_resource_usage_agent)
    workflow.add_node("analyze", analyze_resources_agent)
    workflow.add_node("create_plan", create_allocation_plan_agent)
    workflow.add_node("generate_report", generate_resource_report_agent)
    workflow.add_edge(START, "collect_usage")
    workflow.add_edge("collect_usage", "analyze")
    workflow.add_edge("analyze", "create_plan")
    workflow.add_edge("create_plan", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 275: Resource Optimization MCP Pattern")
    print("="*70)
    
    app = create_resource_optimization_graph()
    final_state = app.invoke({
        "messages": [],
        "resource_usage": {},
        "inefficiencies": [],
        "allocation_plan": {},
        "optimization_results": {}
    })
    print("\n✅ Resource Optimization Pattern Complete!")


if __name__ == "__main__":
    main()

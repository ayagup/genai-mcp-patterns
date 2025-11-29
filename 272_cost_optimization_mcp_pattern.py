"""
Pattern 272: Cost Optimization MCP Pattern

This pattern demonstrates cost optimization strategies including cost
tracking, analysis, and reduction recommendations for cloud and AI systems.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CostOptimizationState(TypedDict):
    """State for cost optimization workflow"""
    messages: Annotated[List[str], add]
    cost_data: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    optimization_strategies: List[Dict[str, Any]]
    savings_projection: Dict[str, Any]


class CostAnalyzer:
    """Analyzes cost patterns and inefficiencies"""
    
    def __init__(self):
        self.cost_categories = {}
    
    def analyze_costs(self, cost_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze cost breakdown"""
        total_cost = sum(item["cost"] for item in cost_items)
        
        # Group by category
        by_category = {}
        by_resource = {}
        
        for item in cost_items:
            category = item.get("category", "other")
            resource = item.get("resource", "unknown")
            
            by_category[category] = by_category.get(category, 0) + item["cost"]
            by_resource[resource] = by_resource.get(resource, 0) + item["cost"]
        
        # Find highest costs
        top_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)[:5]
        top_resources = sorted(by_resource.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_cost": total_cost,
            "by_category": by_category,
            "by_resource": by_resource,
            "top_categories": top_categories,
            "top_resources": top_resources,
            "average_cost_per_item": total_cost / len(cost_items) if cost_items else 0
        }
    
    def identify_waste(self, cost_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify wasteful spending"""
        waste = []
        
        for item in cost_items:
            utilization = item.get("utilization", 1.0)
            cost = item["cost"]
            
            issues = []
            
            # Check for low utilization
            if utilization < 0.3:
                issues.append("Very low utilization (<30%)")
                waste_amount = cost * (1 - utilization)
            elif utilization < 0.5:
                issues.append("Low utilization (<50%)")
                waste_amount = cost * (1 - utilization) * 0.5
            else:
                waste_amount = 0
            
            # Check for overprovisioning
            if item.get("provisioned", 100) > item.get("actual_usage", 50) * 1.5:
                issues.append("Overprovisioned resources")
                waste_amount += cost * 0.2
            
            # Check for expensive tier
            if item.get("tier") == "premium" and item.get("required_tier") == "standard":
                issues.append("Using expensive tier unnecessarily")
                waste_amount += cost * 0.3
            
            if issues:
                waste.append({
                    "resource": item.get("resource", "unknown"),
                    "category": item.get("category", "other"),
                    "cost": cost,
                    "waste_amount": waste_amount,
                    "issues": issues,
                    "severity": "high" if waste_amount > cost * 0.5 else "medium"
                })
        
        return sorted(waste, key=lambda x: x["waste_amount"], reverse=True)


class CostOptimizer:
    """Generates cost optimization recommendations"""
    
    def __init__(self):
        self.strategies = {
            "Very low utilization (<30%)": [
                "Right-size resources to match actual usage",
                "Consider using spot/preemptible instances",
                "Implement auto-scaling"
            ],
            "Low utilization (<50%)": [
                "Optimize resource allocation",
                "Enable auto-shutdown during idle periods"
            ],
            "Overprovisioned resources": [
                "Reduce resource allocation",
                "Use dynamic scaling",
                "Monitor and adjust capacity"
            ],
            "Using expensive tier unnecessarily": [
                "Downgrade to appropriate tier",
                "Review SLA requirements",
                "Use reserved instances for predictable workloads"
            ]
        }
    
    def generate_recommendations(self, waste_items: List[Dict[str, Any]], 
                                analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for waste in waste_items:
            strategies = []
            for issue in waste["issues"]:
                strategies.extend(self.strategies.get(issue, ["Review and optimize"]))
            
            potential_savings = waste["waste_amount"]
            
            recommendations.append({
                "resource": waste["resource"],
                "category": waste["category"],
                "current_cost": waste["cost"],
                "potential_savings": potential_savings,
                "savings_percentage": (potential_savings / waste["cost"] * 100) if waste["cost"] > 0 else 0,
                "strategies": list(set(strategies)),
                "priority": 1 if waste["severity"] == "high" else 2,
                "implementation_effort": "low" if potential_savings > 100 else "medium"
            })
        
        # Add category-level recommendations
        for category, cost in analysis["top_categories"]:
            if cost > analysis["total_cost"] * 0.3:  # Category is >30% of total
                recommendations.append({
                    "resource": f"All {category} resources",
                    "category": category,
                    "current_cost": cost,
                    "potential_savings": cost * 0.15,  # Conservative 15% savings
                    "savings_percentage": 15.0,
                    "strategies": [
                        "Conduct category-wide optimization review",
                        "Negotiate volume discounts",
                        "Consider alternative providers"
                    ],
                    "priority": 2,
                    "implementation_effort": "high"
                })
        
        return sorted(recommendations, key=lambda x: x["potential_savings"], reverse=True)


def collect_cost_data_agent(state: CostOptimizationState) -> CostOptimizationState:
    """Collect cost data"""
    print("\n💰 Collecting Cost Data...")
    
    cost_items = [
        {"resource": "Compute-Prod", "category": "compute", "cost": 1200, "utilization": 0.25, 
         "provisioned": 100, "actual_usage": 25, "tier": "premium", "required_tier": "standard"},
        {"resource": "Storage-Data", "category": "storage", "cost": 800, "utilization": 0.65,
         "provisioned": 1000, "actual_usage": 650, "tier": "standard", "required_tier": "standard"},
        {"resource": "Database-Main", "category": "database", "cost": 1500, "utilization": 0.45,
         "provisioned": 200, "actual_usage": 90, "tier": "premium", "required_tier": "premium"},
        {"resource": "AI-API-Calls", "category": "ai_services", "cost": 2000, "utilization": 0.80,
         "provisioned": 50000, "actual_usage": 40000, "tier": "standard", "required_tier": "standard"},
        {"resource": "Network-Transfer", "category": "network", "cost": 400, "utilization": 0.70,
         "provisioned": 500, "actual_usage": 350, "tier": "standard", "required_tier": "standard"},
        {"resource": "Compute-Dev", "category": "compute", "cost": 600, "utilization": 0.15,
         "provisioned": 50, "actual_usage": 7, "tier": "standard", "required_tier": "standard"},
        {"resource": "Backup-Storage", "category": "storage", "cost": 300, "utilization": 0.90,
         "provisioned": 500, "actual_usage": 450, "tier": "standard", "required_tier": "standard"}
    ]
    
    cost_data = {
        "items": cost_items,
        "total_items": len(cost_items),
        "period": "Monthly"
    }
    
    print(f"\n  Cost Items: {len(cost_items)}")
    print(f"  Period: {cost_data['period']}")
    print(f"\n  Sample Items:")
    for item in cost_items[:3]:
        print(f"    • {item['resource']}: ${item['cost']} ({item['category']})")
        print(f"      Utilization: {item['utilization']:.0%}")
    
    return {
        **state,
        "cost_data": cost_data,
        "messages": [f"✓ Collected data for {len(cost_items)} cost items"]
    }


def analyze_costs_agent(state: CostOptimizationState) -> CostOptimizationState:
    """Analyze cost patterns"""
    print("\n📊 Analyzing Costs...")
    
    analyzer = CostAnalyzer()
    analysis = analyzer.analyze_costs(state["cost_data"]["items"])
    waste = analyzer.identify_waste(state["cost_data"]["items"])
    
    print(f"\n  Total Cost: ${analysis['total_cost']}")
    print(f"\n  Top Cost Categories:")
    for category, cost in analysis["top_categories"]:
        percentage = (cost / analysis["total_cost"] * 100) if analysis["total_cost"] > 0 else 0
        print(f"    • {category}: ${cost:.2f} ({percentage:.1f}%)")
    
    print(f"\n  Waste Identified: {len(waste)} items")
    total_waste = sum(w["waste_amount"] for w in waste)
    print(f"  Total Waste: ${total_waste:.2f}")
    
    analysis["waste_items"] = waste
    analysis["total_waste"] = total_waste
    
    return {
        **state,
        "cost_analysis": analysis,
        "messages": [f"✓ Analysis complete: ${total_waste:.2f} in potential savings identified"]
    }


def generate_optimization_strategies_agent(state: CostOptimizationState) -> CostOptimizationState:
    """Generate optimization strategies"""
    print("\n💡 Generating Optimization Strategies...")
    
    optimizer = CostOptimizer()
    strategies = optimizer.generate_recommendations(
        state["cost_analysis"]["waste_items"],
        state["cost_analysis"]
    )
    
    total_savings = sum(s["potential_savings"] for s in strategies)
    current_cost = state["cost_analysis"]["total_cost"]
    savings_percentage = (total_savings / current_cost * 100) if current_cost > 0 else 0
    
    savings_projection = {
        "current_monthly_cost": current_cost,
        "potential_monthly_savings": total_savings,
        "optimized_monthly_cost": current_cost - total_savings,
        "savings_percentage": savings_percentage,
        "annual_savings": total_savings * 12,
        "total_strategies": len(strategies),
        "high_priority": sum(1 for s in strategies if s["priority"] == 1)
    }
    
    print(f"\n  Optimization Strategies: {len(strategies)}")
    print(f"  Potential Monthly Savings: ${total_savings:.2f}")
    print(f"  Savings Percentage: {savings_percentage:.1f}%")
    print(f"  Annual Savings: ${savings_projection['annual_savings']:.2f}")
    
    print(f"\n  Top 3 Opportunities:")
    for strategy in strategies[:3]:
        print(f"\n    {strategy['resource']}:")
        print(f"      Current Cost: ${strategy['current_cost']}")
        print(f"      Potential Savings: ${strategy['potential_savings']:.2f} ({strategy['savings_percentage']:.1f}%)")
    
    return {
        **state,
        "optimization_strategies": strategies,
        "savings_projection": savings_projection,
        "messages": [f"✓ Generated {len(strategies)} optimization strategies"]
    }


def generate_cost_report_agent(state: CostOptimizationState) -> CostOptimizationState:
    """Generate cost optimization report"""
    print("\n" + "="*70)
    print("COST OPTIMIZATION REPORT")
    print("="*70)
    
    print(f"\n💰 Current Cost Structure:")
    analysis = state["cost_analysis"]
    print(f"  Total Monthly Cost: ${analysis['total_cost']:.2f}")
    print(f"  Total Items: {len(state['cost_data']['items'])}")
    
    print(f"\n  Cost Breakdown by Category:")
    for category, cost in analysis["top_categories"]:
        percentage = (cost / analysis["total_cost"] * 100) if analysis["total_cost"] > 0 else 0
        print(f"    • {category}: ${cost:.2f} ({percentage:.1f}%)")
    
    print(f"\n  Top Cost Resources:")
    for resource, cost in analysis["top_resources"]:
        percentage = (cost / analysis["total_cost"] * 100) if analysis["total_cost"] > 0 else 0
        print(f"    • {resource}: ${cost:.2f} ({percentage:.1f}%)")
    
    print(f"\n🔍 Waste Analysis:")
    print(f"  Wasteful Items: {len(analysis['waste_items'])}")
    print(f"  Total Waste: ${analysis['total_waste']:.2f}")
    
    for waste in analysis["waste_items"][:5]:
        print(f"\n    {waste['resource']} ({waste['severity'].upper()}):")
        print(f"      Current Cost: ${waste['cost']}")
        print(f"      Waste Amount: ${waste['waste_amount']:.2f}")
        print(f"      Issues: {', '.join(waste['issues'])}")
    
    print(f"\n💡 Optimization Strategies:")
    for i, strategy in enumerate(state["optimization_strategies"][:5], 1):
        priority_label = "🔴 HIGH" if strategy["priority"] == 1 else "🟡 MEDIUM"
        print(f"\n  {i}. {priority_label}: {strategy['resource']}")
        print(f"      Current Cost: ${strategy['current_cost']}")
        print(f"      Potential Savings: ${strategy['potential_savings']:.2f} ({strategy['savings_percentage']:.1f}%)")
        print(f"      Implementation: {strategy['implementation_effort'].upper()} effort")
        print(f"      Strategies:")
        for strat in strategy["strategies"][:2]:
            print(f"        • {strat}")
    
    print(f"\n📈 Savings Projection:")
    proj = state["savings_projection"]
    print(f"  Current Monthly Cost: ${proj['current_monthly_cost']:.2f}")
    print(f"  Potential Monthly Savings: ${proj['potential_monthly_savings']:.2f}")
    print(f"  Optimized Monthly Cost: ${proj['optimized_monthly_cost']:.2f}")
    print(f"  Savings Percentage: {proj['savings_percentage']:.1f}%")
    print(f"  Annual Savings: ${proj['annual_savings']:.2f}")
    print(f"  High Priority Items: {proj['high_priority']}")
    
    print(f"\n💡 Cost Optimization Benefits:")
    print("  • Reduced operational expenses")
    print("  • Better resource utilization")
    print("  • Improved cost visibility")
    print("  • Predictable spending")
    print("  • Sustainable operations")
    print("  • Competitive advantage")
    
    print("\n="*70)
    print("✅ Cost Optimization Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_cost_optimization_graph():
    workflow = StateGraph(CostOptimizationState)
    workflow.add_node("collect", collect_cost_data_agent)
    workflow.add_node("analyze", analyze_costs_agent)
    workflow.add_node("optimize", generate_optimization_strategies_agent)
    workflow.add_node("report", generate_cost_report_agent)
    workflow.add_edge(START, "collect")
    workflow.add_edge("collect", "analyze")
    workflow.add_edge("analyze", "optimize")
    workflow.add_edge("optimize", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 272: Cost Optimization MCP Pattern")
    print("="*70)
    
    app = create_cost_optimization_graph()
    final_state = app.invoke({
        "messages": [],
        "cost_data": {},
        "cost_analysis": {},
        "optimization_strategies": [],
        "savings_projection": {}
    })
    print("\n✅ Cost Optimization Pattern Complete!")


if __name__ == "__main__":
    main()

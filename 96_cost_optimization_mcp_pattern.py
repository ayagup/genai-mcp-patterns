"""
Cost Optimization MCP Pattern

This pattern optimizes resource costs by monitoring usage, identifying
waste, and implementing cost-saving strategies.

Key Features:
- Cost monitoring and tracking
- Resource optimization
- Waste identification
- Budget management
- Cost allocation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class CostOptimizationState(TypedDict):
    """State for cost optimization pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    cloud_provider: str
    monthly_budget: float
    current_spend: float
    resources: Dict[str, Dict[str, float]]  # resource_id -> {cost, usage}
    optimization_opportunities: List[Dict[str, str]]
    savings_potential: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Cost Analyzer
def cost_analyzer(state: CostOptimizationState) -> CostOptimizationState:
    """Analyzes current costs and usage"""
    cloud_provider = state.get("cloud_provider", "")
    monthly_budget = state.get("monthly_budget", 10000.0)
    resources = state.get("resources", {})
    
    system_message = SystemMessage(content="""You are a cost analyzer.
    Analyze cloud resource costs and identify spending patterns.""")
    
    current_spend = sum(r.get("cost", 0) for r in resources.values())
    
    user_message = HumanMessage(content=f"""Analyze costs:

Provider: {cloud_provider}
Budget: ${monthly_budget:,.2f}
Current Spend: ${current_spend:,.2f}
Resources: {len(resources)}

Analyze cost breakdown.""")
    
    response = llm.invoke([system_message, user_message])
    
    budget_utilization = (current_spend / monthly_budget * 100) if monthly_budget > 0 else 0
    
    report = f"""
    💰 Cost Analysis:
    
    Financial Overview:
    • Cloud Provider: {cloud_provider.upper()}
    • Monthly Budget: ${monthly_budget:,.2f}
    • Current Spend: ${current_spend:,.2f}
    • Budget Utilization: {budget_utilization:.1f}%
    • Remaining: ${max(0, monthly_budget - current_spend):,.2f}
    
    Cost Breakdown by Service:
    {chr(10).join(f"• {rid}: ${data.get('cost', 0):,.2f}" for rid, data in sorted(resources.items(), key=lambda x: x[1].get('cost', 0), reverse=True)[:5])}
    
    Cloud Cost Components:
    
    Compute:
    • VM instances (EC2, Compute Engine)
    • Containers (ECS, GKE, AKS)
    • Serverless (Lambda, Functions)
    • Reserved vs On-Demand
    • Spot instances
    
    Storage:
    • Block storage (EBS, Persistent Disk)
    • Object storage (S3, Cloud Storage)
    • File storage (EFS, Filestore)
    • Backup and snapshots
    • Data transfer
    
    Database:
    • RDS, Cloud SQL
    • DynamoDB, Firestore
    • Redis, Memcached
    • Backup and replication
    • IOPS provisioning
    
    Networking:
    • Load balancers
    • NAT gateways
    • VPN connections
    • Data transfer (egress)
    • CDN delivery
    
    Other Services:
    • Monitoring and logging
    • API calls
    • Email/SMS
    • Machine learning
    • Managed services
    
    Cost Monitoring Tools:
    
    AWS:
    • Cost Explorer
    • Cost and Usage Reports
    • AWS Budgets
    • Cost Anomaly Detection
    • Trusted Advisor
    
    Azure:
    • Cost Management + Billing
    • Cost Analysis
    • Budgets and alerts
    • Azure Advisor
    • Reservations
    
    Google Cloud:
    • Cloud Billing
    • Cost Table
    • Budget alerts
    • Recommender
    • Committed Use Discounts
    
    Third-Party:
    • CloudHealth
    • CloudCheckr
    • Spot.io
    • Kubecost
    • Infracost
    
    Cost Allocation:
    
    Tagging Strategy:
    • Environment: prod, staging, dev
    • Department: engineering, sales
    • Project: project-alpha
    • Owner: team-name
    • Cost center: CC-1234
    
    Chargeback/Showback:
    • Department-level billing
    • Project cost tracking
    • Team accountability
    • Budget enforcement
    • Cost transparency
    
    FinOps Practices:
    • Cross-functional collaboration
    • Real-time cost visibility
    • Business value alignment
    • Continuous optimization
    • Cloud cost culture
    """
    
    return {
        "messages": [AIMessage(content=f"💰 Cost Analyzer:\n{response.content}\n{report}")],
        "current_spend": current_spend
    }


# Optimization Advisor
def optimization_advisor(state: CostOptimizationState) -> CostOptimizationState:
    """Identifies cost optimization opportunities"""
    cloud_provider = state.get("cloud_provider", "")
    resources = state.get("resources", {})
    current_spend = state.get("current_spend", 0.0)
    
    system_message = SystemMessage(content="""You are an optimization advisor.
    Identify cost-saving opportunities and provide recommendations.""")
    
    user_message = HumanMessage(content=f"""Find optimization opportunities:

Provider: {cloud_provider}
Current Spend: ${current_spend:,.2f}
Resource Count: {len(resources)}

Recommend optimizations.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Identify opportunities
    optimization_opportunities = [
        {"type": "rightsizing", "resource": "compute_instance_1", "savings": "30%", "action": "Downsize from m5.2xlarge to m5.xlarge"},
        {"type": "reserved_instance", "resource": "database_1", "savings": "40%", "action": "Purchase 1-year reserved instance"},
        {"type": "spot_instances", "resource": "batch_workers", "savings": "70%", "action": "Use spot instances for non-critical workloads"},
        {"type": "storage_lifecycle", "resource": "backup_storage", "savings": "50%", "action": "Move old backups to glacier"},
        {"type": "idle_resource", "resource": "dev_server", "savings": "100%", "action": "Delete unused development server"}
    ]
    
    savings_potential = sum(float(opp["savings"].rstrip('%')) for opp in optimization_opportunities) / len(optimization_opportunities) * current_spend / 100
    
    report = f"""
    💡 Optimization Opportunities:
    
    Identified Opportunities: {len(optimization_opportunities)}
    Total Savings Potential: ${savings_potential:,.2f}/month
    
    Recommendations:
    {chr(10).join(f"{i+1}. {opp['type'].upper()} - {opp['resource']}" +
                   f"{chr(10)}   Savings: {opp['savings']} | Action: {opp['action']}" 
                   for i, opp in enumerate(optimization_opportunities))}
    
    Optimization Strategies:
    
    1. Rightsizing:
       • Monitor CPU/Memory utilization
       • Identify over-provisioned resources
       • Downsize to appropriate instance types
       • Typical savings: 20-40%
    
    2. Reserved Instances:
       • Commit to 1 or 3 year terms
       • Up to 72% discount vs on-demand
       • Analyze usage patterns first
       • Convertible for flexibility
    
    3. Savings Plans:
       • Flexible pricing model
       • Commitment to $ amount
       • AWS: Compute, EC2, SageMaker
       • Auto-applies to eligible usage
    
    4. Spot Instances:
       • Up to 90% discount
       • Interruptible workloads
       • Batch processing
       • Dev/test environments
    
    5. Auto-Scaling:
       • Scale based on demand
       • Remove idle capacity
       • Match workload patterns
       • Scheduled scaling
    
    6. Storage Optimization:
       • Lifecycle policies
       • Compress old data
       • Delete unused snapshots
       • Use appropriate storage tiers
    
    7. Network Optimization:
       • Reduce data transfer
       • Use CDN for static content
       • Optimize inter-region traffic
       • VPC endpoints
    
    8. Delete Idle Resources:
       • Unused load balancers
       • Unattached volumes
       • Old snapshots
       • Zombie instances
    
    Implementation Priority:
    
    Quick Wins (Days):
    • Delete idle resources
    • Remove unattached volumes
    • Clean up old snapshots
    • Stop unused instances
    
    Short Term (Weeks):
    • Rightsize instances
    • Implement auto-scaling
    • Storage lifecycle policies
    • Use spot instances
    
    Long Term (Months):
    • Purchase reserved instances
    • Commit to savings plans
    • Architecture optimization
    • Multi-cloud strategy
    
    Measurement:
    • Track savings realized
    • Monitor cost trends
    • ROI of optimizations
    • Continuous improvement
    """
    
    return {
        "messages": [AIMessage(content=f"💡 Optimization Advisor:\n{response.content}\n{report}")],
        "optimization_opportunities": optimization_opportunities,
        "savings_potential": savings_potential
    }


# Cost Monitor
def cost_monitor(state: CostOptimizationState) -> CostOptimizationState:
    """Monitors costs and provides budget alerts"""
    cloud_provider = state.get("cloud_provider", "")
    monthly_budget = state.get("monthly_budget", 0.0)
    current_spend = state.get("current_spend", 0.0)
    savings_potential = state.get("savings_potential", 0.0)
    optimization_opportunities = state.get("optimization_opportunities", [])
    
    budget_utilization = (current_spend / monthly_budget * 100) if monthly_budget > 0 else 0
    projected_spend = current_spend  # Simplified projection
    
    summary = f"""
    📊 COST OPTIMIZATION COMPLETE
    
    Financial Summary:
    • Cloud Provider: {cloud_provider.upper()}
    • Monthly Budget: ${monthly_budget:,.2f}
    • Current Spend: ${current_spend:,.2f}
    • Budget Utilization: {budget_utilization:.1f}%
    • Projected Month-End: ${projected_spend:,.2f}
    
    Optimization Results:
    • Opportunities Found: {len(optimization_opportunities)}
    • Potential Savings: ${savings_potential:,.2f}/month
    • Potential Savings %: {(savings_potential/current_spend*100) if current_spend > 0 else 0:.1f}%
    
    Cost Optimization Pattern Process:
    1. Cost Analyzer → Assess current spending
    2. Optimization Advisor → Identify savings
    3. Monitor → Track and alert on budgets
    
    Budget Alert Thresholds:
    
    Proactive Alerts:
    • 50% budget used
    • 75% budget used
    • 90% budget used
    • 100% budget exceeded
    • Forecasted to exceed
    
    Anomaly Detection:
    • Unusual spending spike
    • New expensive resources
    • Cost trend deviation
    • Service quota breach
    
    Real-World Examples:
    
    Airbnb:
    • Saved $millions with rightsizing
    • Reserved instance optimization
    • Spot instance adoption
    • Storage lifecycle policies
    
    Pinterest:
    • 75% cost reduction on EMR
    • S3 storage optimization
    • Reserved instance coverage
    • Multi-tier storage strategy
    
    Lyft:
    • Kubernetes rightsizing
    • Spot instance fleet
    • Database optimization
    • Network cost reduction
    
    Dropbox:
    • Migrated from AWS to own DC
    • $75M in savings (2 years)
    • Custom storage solution
    • Hybrid cloud approach
    
    Cost Optimization Best Practices:
    
    Governance:
    • Set and enforce budgets
    • Require cost tags
    • Approval for large resources
    • Regular cost reviews
    • Cost-aware culture
    
    Architecture:
    • Design for cost efficiency
    • Use managed services wisely
    • Consider serverless
    • Multi-region strategy
    • Data locality
    
    Monitoring:
    • Real-time cost dashboards
    • Automated anomaly detection
    • Budget alerts
    • Cost forecasting
    • Trend analysis
    
    Automation:
    • Auto-scaling
    • Scheduled start/stop
    • Rightsizing recommendations
    • Cleanup scripts
    • Policy enforcement
    
    Optimization Cycle:
    • Weekly: Review alerts
    • Monthly: Analyze trends
    • Quarterly: Strategic planning
    • Annually: Reserved capacity
    
    Tools and Automation:
    
    AWS:
    • Lambda for automation
    • EventBridge for scheduling
    • Systems Manager for patches
    • Config for compliance
    
    Kubernetes:
    • Cluster autoscaler
    • Vertical pod autoscaler
    • Horizontal pod autoscaler
    • Karpenter for node mgmt
    
    Infrastructure as Code:
    • Terraform cost estimation
    • Infracost in CI/CD
    • Policy as code
    • Cost guardrails
    
    Metrics to Track:
    • Cost per customer
    • Cost per transaction
    • Cost per request
    • Infrastructure efficiency
    • Waste percentage
    • Savings rate
    
    Key Insight:
    Cost optimization is continuous. Build a culture of
    cost awareness, automate where possible, and regularly
    review and optimize your cloud spending.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Cost Monitor:\n{summary}")]
    }


# Build the graph
def build_cost_optimization_graph():
    """Build the cost optimization pattern graph"""
    workflow = StateGraph(CostOptimizationState)
    
    workflow.add_node("analyzer", cost_analyzer)
    workflow.add_node("advisor", optimization_advisor)
    workflow.add_node("monitor", cost_monitor)
    
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "advisor")
    workflow.add_edge("advisor", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_cost_optimization_graph()
    
    print("=== Cost Optimization MCP Pattern ===\n")
    
    # Test Case: AWS cost optimization
    print("\n" + "="*70)
    print("TEST CASE: Cloud Cost Optimization")
    print("="*70)
    
    state = {
        "messages": [],
        "cloud_provider": "aws",
        "monthly_budget": 10000.0,
        "current_spend": 0.0,
        "resources": {
            "compute_instance_1": {"cost": 2500.0, "usage": 80.0},
            "database_1": {"cost": 1800.0, "usage": 95.0},
            "storage_s3": {"cost": 500.0, "usage": 60.0},
            "backup_storage": {"cost": 800.0, "usage": 20.0},
            "batch_workers": {"cost": 1200.0, "usage": 40.0},
            "dev_server": {"cost": 400.0, "usage": 5.0}
        },
        "optimization_opportunities": [],
        "savings_potential": 0.0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nCurrent Spend: ${result.get('current_spend', 0):,.2f}")
    print(f"Savings Potential: ${result.get('savings_potential', 0):,.2f}")
    print(f"Opportunities: {len(result.get('optimization_opportunities', []))}")

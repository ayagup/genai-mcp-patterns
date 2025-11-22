"""
Pattern 203: Auto-Scaling MCP Pattern

This pattern demonstrates automatic scaling based on metrics:
- Monitor system metrics (CPU, memory, requests/sec)
- Define scaling policies (scale out/in thresholds)
- Automatic scale-out when load increases
- Automatic scale-in when load decreases
- Cost optimization through right-sizing

Auto-scaling automatically adjusts capacity based on actual demand,
eliminating manual intervention and optimizing costs.

Use Cases:
- Web applications with variable traffic
- Batch processing workloads
- Microservices in Kubernetes
- Cloud-native applications
- Cost-sensitive applications

Key Features:
- Metric-Based: Scale based on CPU, memory, requests, custom metrics
- Policy-Driven: Define when to scale out/in
- Automatic: No manual intervention required
- Cost-Effective: Scale down during low traffic
- Responsive: React to load changes quickly

Scaling Policies:
- Target Tracking: Maintain target metric (e.g., 70% CPU)
- Step Scaling: Scale by amounts based on threshold breaches
- Scheduled Scaling: Scale at specific times (e.g., business hours)
- Predictive Scaling: ML-based forecasting
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random


class AutoScalingState(TypedDict):
    """State for auto-scaling operations"""
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    metric_name: str
    scale_out_threshold: float  # Scale out when metric > threshold
    scale_in_threshold: float   # Scale in when metric < threshold
    min_instances: int
    max_instances: int
    cooldown_seconds: int = 60  # Wait before next scaling action
    
    def should_scale_out(self, metric_value: float, current_instances: int) -> bool:
        """Check if should scale out"""
        return (metric_value > self.scale_out_threshold and 
                current_instances < self.max_instances)
    
    def should_scale_in(self, metric_value: float, current_instances: int) -> bool:
        """Check if should scale in"""
        return (metric_value < self.scale_in_threshold and 
                current_instances > self.min_instances)


@dataclass
class Instance:
    """Server instance"""
    instance_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    requests_per_sec: float = 0.0
    is_healthy: bool = True
    created_at: float = field(default_factory=time.time)


class AutoScalingGroup:
    """
    Auto-scaling group that manages instances.
    
    Monitors metrics and automatically scales out/in based on policies.
    """
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.instances: List[Instance] = []
        self.instance_counter = 0
        self.last_scaling_time = 0
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Initialize with min instances
        for _ in range(policy.min_instances):
            self.add_instance()
    
    def add_instance(self) -> Instance:
        """Add a new instance"""
        self.instance_counter += 1
        instance = Instance(f"instance-{self.instance_counter}")
        self.instances.append(instance)
        
        self.scaling_history.append({
            'action': 'scale_out',
            'instance_id': instance.instance_id,
            'count': len(self.instances),
            'timestamp': time.time()
        })
        
        return instance
    
    def remove_instance(self) -> Optional[Instance]:
        """Remove an instance"""
        if len(self.instances) <= self.policy.min_instances:
            return None
        
        instance = self.instances.pop()
        
        self.scaling_history.append({
            'action': 'scale_in',
            'instance_id': instance.instance_id,
            'count': len(self.instances),
            'timestamp': time.time()
        })
        
        return instance
    
    def get_average_metric(self, metric_name: str) -> float:
        """Get average metric value across all instances"""
        if not self.instances:
            return 0.0
        
        if metric_name == "cpu":
            return sum(i.cpu_usage for i in self.instances) / len(self.instances)
        elif metric_name == "memory":
            return sum(i.memory_usage for i in self.instances) / len(self.instances)
        elif metric_name == "requests":
            return sum(i.requests_per_sec for i in self.instances) / len(self.instances)
        
        return 0.0
    
    def simulate_load(self, load_level: float):
        """Simulate load on instances"""
        # Distribute load across instances
        base_cpu = load_level
        base_memory = load_level * 0.8
        base_requests = load_level * 100
        
        for instance in self.instances:
            # Add some randomness
            instance.cpu_usage = min(100, base_cpu + random.uniform(-10, 10))
            instance.memory_usage = min(100, base_memory + random.uniform(-10, 10))
            instance.requests_per_sec = max(0, base_requests + random.uniform(-20, 20))
    
    def check_and_scale(self) -> Optional[str]:
        """Check metrics and scale if needed"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_scaling_time < self.policy.cooldown_seconds:
            return None
        
        metric_value = self.get_average_metric(self.policy.metric_name)
        current_count = len(self.instances)
        
        # Check if should scale out
        if self.policy.should_scale_out(metric_value, current_count):
            self.add_instance()
            self.last_scaling_time = current_time
            return f"Scaled OUT: {metric_value:.1f}% > {self.policy.scale_out_threshold}% → +1 instance (now {len(self.instances)})"
        
        # Check if should scale in
        if self.policy.should_scale_in(metric_value, current_count):
            removed = self.remove_instance()
            if removed:
                self.last_scaling_time = current_time
                return f"Scaled IN: {metric_value:.1f}% < {self.policy.scale_in_threshold}% → -1 instance (now {len(self.instances)})"
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        scale_outs = sum(1 for h in self.scaling_history if h['action'] == 'scale_out')
        scale_ins = sum(1 for h in self.scaling_history if h['action'] == 'scale_in')
        
        return {
            'current_instances': len(self.instances),
            'min_instances': self.policy.min_instances,
            'max_instances': self.policy.max_instances,
            'total_scale_outs': scale_outs - self.policy.min_instances,  # Exclude initial
            'total_scale_ins': scale_ins,
            'avg_cpu': self.get_average_metric('cpu'),
            'avg_memory': self.get_average_metric('memory'),
            'avg_requests': self.get_average_metric('requests')
        }


def setup_autoscaling_agent(state: AutoScalingState):
    """Agent to set up auto-scaling group"""
    operations = []
    results = []
    
    # Create auto-scaling policy
    policy = ScalingPolicy(
        metric_name="cpu",
        scale_out_threshold=70.0,  # Scale out when CPU > 70%
        scale_in_threshold=30.0,   # Scale in when CPU < 30%
        min_instances=2,
        max_instances=10,
        cooldown_seconds=5  # Short cooldown for demo
    )
    
    # Create auto-scaling group
    asg = AutoScalingGroup(policy)
    
    operations.append("Auto-Scaling Configuration:")
    operations.append(f"  Metric: {policy.metric_name.upper()}")
    operations.append(f"  Scale-out threshold: {policy.scale_out_threshold}%")
    operations.append(f"  Scale-in threshold: {policy.scale_in_threshold}%")
    operations.append(f"  Min instances: {policy.min_instances}")
    operations.append(f"  Max instances: {policy.max_instances}")
    operations.append(f"  Cooldown: {policy.cooldown_seconds}s")
    
    results.append(f"✓ Auto-scaling group created with {len(asg.instances)} instances")
    
    # Store in state
    state['_asg'] = asg
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Auto-scaling setup complete"]
    }


def simulate_traffic_pattern_agent(state: AutoScalingState):
    """Agent to simulate realistic traffic pattern"""
    asg = state['_asg']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n📊 Simulating Traffic Pattern (10 time periods):")
    
    # Simulate traffic pattern: low → high → low
    traffic_pattern = [
        (1, 20),   # Period 1: Low traffic (20% load)
        (2, 25),
        (3, 50),   # Starting to increase
        (4, 75),   # High traffic - should scale out
        (5, 85),   # Peak traffic - more scale out
        (6, 90),   # Maximum traffic
        (7, 70),   # Declining
        (8, 50),   # Back to medium
        (9, 30),   # Low again - should scale in
        (10, 20),  # Minimal traffic
    ]
    
    for period, load in traffic_pattern:
        # Simulate load
        asg.simulate_load(load)
        
        # Check and scale
        scaling_action = asg.check_and_scale()
        
        stats = asg.get_statistics()
        
        operations.append(f"\nPeriod {period}: Load={load}%")
        operations.append(f"  Instances: {stats['current_instances']}")
        operations.append(f"  Avg CPU: {stats['avg_cpu']:.1f}%")
        
        if scaling_action:
            operations.append(f"  🔄 {scaling_action}")
        else:
            operations.append(f"  ⏸️  No scaling action (cooldown or within thresholds)")
        
        # Short delay to simulate time passing
        time.sleep(0.1)
    
    final_stats = asg.get_statistics()
    
    results.append(f"✓ Handled traffic pattern")
    results.append(f"✓ Scale-out events: {final_stats['total_scale_outs']}")
    results.append(f"✓ Scale-in events: {final_stats['total_scale_ins']}")
    
    metrics.append(f"Final instances: {final_stats['current_instances']}")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Traffic pattern simulation complete"]
    }


def cost_analysis_agent(state: AutoScalingState):
    """Agent to analyze cost savings"""
    asg = state['_asg']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n💰 Cost Analysis:")
    
    stats = asg.get_statistics()
    
    # Calculate costs
    cost_per_instance_hour = 0.10  # $0.10/hour
    simulation_hours = len(asg.scaling_history)  # Simplified
    
    # With auto-scaling (actual usage)
    total_instance_hours = sum(h['count'] for h in asg.scaling_history)
    auto_scaling_cost = total_instance_hours * cost_per_instance_hour
    
    # Without auto-scaling (always max capacity)
    max_instances = stats['max_instances']
    static_cost = max_instances * simulation_hours * cost_per_instance_hour
    
    savings = static_cost - auto_scaling_cost
    savings_percent = (savings / static_cost * 100) if static_cost > 0 else 0
    
    operations.append(f"  Simulation periods: {len(asg.scaling_history)}")
    operations.append(f"  Cost per instance-hour: ${cost_per_instance_hour}")
    operations.append(f"\nWithout Auto-Scaling (Static {max_instances} instances):")
    operations.append(f"  Cost: ${static_cost:.2f}")
    operations.append(f"\nWith Auto-Scaling:")
    operations.append(f"  Average instances: {total_instance_hours/len(asg.scaling_history):.1f}")
    operations.append(f"  Cost: ${auto_scaling_cost:.2f}")
    operations.append(f"\nSavings: ${savings:.2f} ({savings_percent:.1f}%)")
    
    results.append(f"✓ Cost savings: {savings_percent:.1f}%")
    
    metrics.append(f"Auto-scaling saved ${savings:.2f}")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Cost analysis complete"]
    }


def summary_agent(state: AutoScalingState):
    """Agent to provide summary"""
    asg = state['_asg']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("AUTO-SCALING SUMMARY")
    operations.append("="*60)
    
    stats = asg.get_statistics()
    
    operations.append(f"\nFinal State:")
    operations.append(f"  Current instances: {stats['current_instances']}")
    operations.append(f"  Min/Max configured: {stats['min_instances']}-{stats['max_instances']}")
    operations.append(f"  Total scale-outs: {stats['total_scale_outs']}")
    operations.append(f"  Total scale-ins: {stats['total_scale_ins']}")
    
    metrics.append("\n📊 Auto-Scaling Benefits:")
    metrics.append("  ✓ Cost Optimization: Pay only for needed capacity")
    metrics.append("  ✓ Performance: Scale out during peak load")
    metrics.append("  ✓ Automation: No manual intervention")
    metrics.append("  ✓ Responsiveness: React to load changes")
    metrics.append("  ✓ Efficiency: Scale in during low traffic")
    
    results.append("✓ Auto-scaling demonstrated successfully")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Summary complete"]
    }


def create_auto_scaling_graph():
    """Create the auto-scaling workflow graph"""
    workflow = StateGraph(AutoScalingState)
    
    # Add nodes
    workflow.add_node("setup", setup_autoscaling_agent)
    workflow.add_node("simulate_traffic", simulate_traffic_pattern_agent)
    workflow.add_node("cost_analysis", cost_analysis_agent)
    workflow.add_node("summary", summary_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "simulate_traffic")
    workflow.add_edge("simulate_traffic", "cost_analysis")
    workflow.add_edge("cost_analysis", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 203: Auto-Scaling MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_auto_scaling_graph()
    
    # Initialize state
    initial_state = {
        "scaling_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("SCALING OPERATIONS")
    print("=" * 80)
    for op in final_state["scaling_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    for metric in final_state["performance_metrics"]:
        print(metric)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Auto-Scaling Pattern implemented with:

1. Metric-Based Scaling:
   - Monitor CPU utilization
   - Scale out when > 70%
   - Scale in when < 30%
   - Configurable thresholds

2. Scaling Policies:
   - Min instances: 2 (always available)
   - Max instances: 10 (cost control)
   - Cooldown: Prevent rapid scaling
   - Target tracking: Maintain target metric

3. Cost Optimization:
   - 30-60% cost savings vs static capacity
   - Pay only for needed resources
   - Automatic scale-in during low traffic

4. Real-World Implementations:
   - AWS Auto Scaling Groups
   - Kubernetes Horizontal Pod Autoscaler (HPA)
   - Azure Scale Sets
   - Google Cloud Instance Groups

Scaling Metrics:
- CPU Utilization: Most common
- Memory Usage: For memory-intensive apps
- Request Count: For web applications
- Queue Depth: For async processing
- Custom Metrics: Business-specific (orders/sec, etc.)

Best Practices:
1. Set Appropriate Thresholds: Not too sensitive
2. Use Cooldown Periods: Prevent thrashing
3. Monitor Scaling Events: Ensure policies work
4. Set Min/Max Limits: Control costs
5. Test Scaling Policies: Simulate load
6. Combine with Load Balancing: Distribute traffic
7. Use Multiple Metrics: CPU + Memory + Custom
""")


if __name__ == "__main__":
    main()

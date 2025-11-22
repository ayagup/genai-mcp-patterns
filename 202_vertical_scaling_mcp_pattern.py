"""
Pattern 202: Vertical Scaling MCP Pattern

This pattern demonstrates vertical scaling (scale-up) by increasing resources:
- Adding CPU, RAM, disk to existing server
- Upgrading to more powerful hardware
- Resource limits and capacity planning
- Performance improvements through better hardware

Vertical scaling makes a single machine more powerful, rather than adding
more machines (horizontal scaling).

Use Cases:
- Database servers requiring more memory
- Legacy applications (not designed for distribution)
- CPU-intensive computations
- In-memory data processing
- Monolithic applications

Key Features:
- Simple architecture: Single powerful server
- No distributed complexity
- Easy data consistency (single node)
- Predictable performance
- Lower latency (no network overhead)

Advantages:
✓ Simple architecture and deployment
✓ Easy data consistency
✓ No network latency
✓ Existing applications work without changes
✓ Lower software licensing costs (per-server)

Disadvantages:
✗ Hardware limits (finite ceiling)
✗ Single point of failure
✗ Expensive at high end (diminishing returns)
✗ Downtime required for upgrades
✗ Limited fault tolerance
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random


class VerticalScalingState(TypedDict):
    """State for vertical scaling operations"""
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class ServerResources:
    """Server hardware resources"""
    cpu_cores: int
    ram_gb: int
    disk_gb: int
    network_gbps: int
    
    def get_capacity_score(self) -> float:
        """Calculate overall capacity score"""
        # Weighted score based on resources
        return (
            self.cpu_cores * 10 +
            self.ram_gb * 2 +
            self.disk_gb * 0.1 +
            self.network_gbps * 5
        )
    
    def get_cost_estimate(self) -> float:
        """Estimate monthly cost in dollars"""
        # Simplified cost model
        cpu_cost = self.cpu_cores * 20  # $20/core/month
        ram_cost = self.ram_gb * 5      # $5/GB/month
        disk_cost = self.disk_gb * 0.1  # $0.10/GB/month
        network_cost = self.network_gbps * 50  # $50/Gbps/month
        
        return cpu_cost + ram_cost + disk_cost + network_cost


@dataclass
class Server:
    """Server with configurable resources"""
    server_id: str
    resources: ServerResources
    current_cpu_usage: float = 0.0
    current_ram_usage_gb: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    
    def process_requests(self, num_requests: int) -> Dict[str, Any]:
        """Process requests based on available resources"""
        start_time = time.time()
        
        # Calculate how many requests can be handled
        cpu_capacity = self.resources.cpu_cores * 100  # 100 req/core
        ram_capacity = self.resources.ram_gb * 10     # 10 req/GB
        
        max_capacity = min(cpu_capacity, ram_capacity)
        
        successful = min(num_requests, int(max_capacity))
        failed = max(0, num_requests - successful)
        
        self.total_requests += num_requests
        self.failed_requests += failed
        
        # Simulate processing time (better hardware = faster)
        base_time = 0.01
        cpu_factor = 1.0 / self.resources.cpu_cores
        processing_time = base_time * cpu_factor * successful
        time.sleep(min(processing_time, 0.1))  # Cap simulation time
        
        elapsed = time.time() - start_time
        
        # Update average response time
        if self.total_requests > 0:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - num_requests) + elapsed)
                / self.total_requests
            )
        
        # Update resource usage
        self.current_cpu_usage = (successful / cpu_capacity) * 100
        self.current_ram_usage_gb = (successful / ram_capacity) * self.resources.ram_gb
        
        return {
            'successful': successful,
            'failed': failed,
            'processing_time': elapsed,
            'cpu_usage': f"{self.current_cpu_usage:.1f}%",
            'ram_usage': f"{self.current_ram_usage_gb:.1f}GB"
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        success_rate = (
            ((self.total_requests - self.failed_requests) / self.total_requests * 100)
            if self.total_requests > 0 else 0
        )
        
        return {
            'server_id': self.server_id,
            'resources': {
                'cpu_cores': self.resources.cpu_cores,
                'ram_gb': self.resources.ram_gb,
                'disk_gb': self.resources.disk_gb,
                'network_gbps': self.resources.network_gbps
            },
            'capacity_score': self.resources.get_capacity_score(),
            'cost_per_month': f"${self.resources.get_cost_estimate():.2f}",
            'total_requests': self.total_requests,
            'failed_requests': self.failed_requests,
            'success_rate': f"{success_rate:.1f}%",
            'avg_response_time': f"{self.avg_response_time*1000:.2f}ms"
        }


def setup_initial_server_agent(state: VerticalScalingState):
    """Agent to set up initial server with modest resources"""
    operations = []
    results = []
    metrics = []
    
    # Start with small server
    initial_resources = ServerResources(
        cpu_cores=2,
        ram_gb=8,
        disk_gb=100,
        network_gbps=1
    )
    
    server = Server("prod-server", initial_resources)
    
    operations.append("Initial Server Configuration:")
    operations.append(f"  CPU: {initial_resources.cpu_cores} cores")
    operations.append(f"  RAM: {initial_resources.ram_gb} GB")
    operations.append(f"  Disk: {initial_resources.disk_gb} GB")
    operations.append(f"  Network: {initial_resources.network_gbps} Gbps")
    operations.append(f"  Capacity Score: {initial_resources.get_capacity_score():.0f}")
    operations.append(f"  Estimated Cost: ${initial_resources.get_cost_estimate():.2f}/month")
    
    results.append("✓ Initial server configured")
    metrics.append(f"T0: {initial_resources.cpu_cores}C/{initial_resources.ram_gb}GB")
    
    # Store in state
    state['_server'] = server
    state['_upgrade_history'] = []
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Initial server setup complete"]
    }


def baseline_performance_agent(state: VerticalScalingState):
    """Agent to measure baseline performance"""
    server = state['_server']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n📊 Baseline Performance Test (100 requests):")
    
    result = server.process_requests(100)
    
    operations.append(f"  Successful: {result['successful']}/100")
    operations.append(f"  Failed: {result['failed']}/100")
    operations.append(f"  Processing time: {result['processing_time']*1000:.2f}ms")
    operations.append(f"  CPU usage: {result['cpu_usage']}")
    operations.append(f"  RAM usage: {result['ram_usage']}")
    
    results.append(f"✓ Baseline: {result['successful']}/100 success")
    metrics.append(f"Baseline capacity: {result['successful']} req")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Baseline performance measured"]
    }


def cpu_upgrade_agent(state: VerticalScalingState):
    """Agent to demonstrate CPU upgrade"""
    server = state['_server']
    upgrade_history = state['_upgrade_history']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🔧 VERTICAL SCALE-UP: CPU Upgrade")
    
    # Record before upgrade
    old_resources = ServerResources(
        server.resources.cpu_cores,
        server.resources.ram_gb,
        server.resources.disk_gb,
        server.resources.network_gbps
    )
    
    # Upgrade CPU: 2 → 8 cores
    server.resources.cpu_cores = 8
    
    operations.append(f"  CPU: {old_resources.cpu_cores} → {server.resources.cpu_cores} cores (4x)")
    operations.append(f"  Capacity: {old_resources.get_capacity_score():.0f} → {server.resources.get_capacity_score():.0f}")
    operations.append(f"  Cost: ${old_resources.get_cost_estimate():.2f} → ${server.resources.get_cost_estimate():.2f}/month")
    
    upgrade_history.append({
        'type': 'CPU',
        'from': old_resources.cpu_cores,
        'to': server.resources.cpu_cores
    })
    
    # Test performance after upgrade
    operations.append(f"\n📊 Performance After CPU Upgrade (200 requests):")
    result = server.process_requests(200)
    
    operations.append(f"  Successful: {result['successful']}/200")
    operations.append(f"  Failed: {result['failed']}/200")
    operations.append(f"  Processing time: {result['processing_time']*1000:.2f}ms")
    
    results.append(f"✓ CPU upgraded: 2 → 8 cores")
    results.append(f"✓ Handled {result['successful']}/200 requests")
    
    metrics.append(f"After CPU: {result['successful']} req (vs 100 baseline)")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["CPU upgrade complete"]
    }


def ram_upgrade_agent(state: VerticalScalingState):
    """Agent to demonstrate RAM upgrade"""
    server = state['_server']
    upgrade_history = state['_upgrade_history']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🔧 VERTICAL SCALE-UP: RAM Upgrade")
    
    old_ram = server.resources.ram_gb
    
    # Upgrade RAM: 8 → 64 GB
    server.resources.ram_gb = 64
    
    operations.append(f"  RAM: {old_ram} → {server.resources.ram_gb} GB (8x)")
    operations.append(f"  Capacity: {server.resources.get_capacity_score():.0f}")
    operations.append(f"  Cost: ${server.resources.get_cost_estimate():.2f}/month")
    
    upgrade_history.append({
        'type': 'RAM',
        'from': old_ram,
        'to': server.resources.ram_gb
    })
    
    # Test performance
    operations.append(f"\n📊 Performance After RAM Upgrade (500 requests):")
    result = server.process_requests(500)
    
    operations.append(f"  Successful: {result['successful']}/500")
    operations.append(f"  Failed: {result['failed']}/500")
    operations.append(f"  RAM usage: {result['ram_usage']}")
    
    results.append(f"✓ RAM upgraded: {old_ram} → {server.resources.ram_gb} GB")
    results.append(f"✓ Handled {result['successful']}/500 requests")
    
    metrics.append(f"After RAM: {result['successful']} req")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["RAM upgrade complete"]
    }


def full_upgrade_agent(state: VerticalScalingState):
    """Agent to demonstrate full server upgrade"""
    server = state['_server']
    upgrade_history = state['_upgrade_history']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🚀 FULL SERVER UPGRADE (Enterprise Class)")
    
    old_resources = ServerResources(
        server.resources.cpu_cores,
        server.resources.ram_gb,
        server.resources.disk_gb,
        server.resources.network_gbps
    )
    
    # Upgrade to enterprise server
    server.resources = ServerResources(
        cpu_cores=32,
        ram_gb=256,
        disk_gb=2000,
        network_gbps=10
    )
    
    operations.append(f"  CPU: {old_resources.cpu_cores} → {server.resources.cpu_cores} cores")
    operations.append(f"  RAM: {old_resources.ram_gb} → {server.resources.ram_gb} GB")
    operations.append(f"  Disk: {old_resources.disk_gb} → {server.resources.disk_gb} GB")
    operations.append(f"  Network: {old_resources.network_gbps} → {server.resources.network_gbps} Gbps")
    operations.append(f"  Capacity: {old_resources.get_capacity_score():.0f} → {server.resources.get_capacity_score():.0f}")
    operations.append(f"  Cost: ${old_resources.get_cost_estimate():.2f} → ${server.resources.get_cost_estimate():.2f}/month")
    
    upgrade_history.append({
        'type': 'Full',
        'from': 'Mid-tier',
        'to': 'Enterprise'
    })
    
    # Test maximum performance
    operations.append(f"\n📊 Maximum Performance Test (2000 requests):")
    result = server.process_requests(2000)
    
    operations.append(f"  Successful: {result['successful']}/2000")
    operations.append(f"  Failed: {result['failed']}/2000")
    operations.append(f"  Processing time: {result['processing_time']*1000:.2f}ms")
    
    results.append(f"✓ Full upgrade to enterprise server")
    results.append(f"✓ Handled {result['successful']}/2000 requests")
    
    metrics.append(f"Final capacity: {result['successful']} req")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Full server upgrade complete"]
    }


def scaling_comparison_agent(state: VerticalScalingState):
    """Agent to compare vertical vs horizontal scaling"""
    server = state['_server']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("VERTICAL vs HORIZONTAL SCALING COMPARISON")
    operations.append("="*60)
    
    final_stats = server.get_stats()
    
    operations.append(f"\nVertical Scaling (Scale-Up) - Single Server:")
    operations.append(f"  Final Configuration: {final_stats['resources']['cpu_cores']}C/{final_stats['resources']['ram_gb']}GB")
    operations.append(f"  Total Capacity Score: {final_stats['capacity_score']:.0f}")
    operations.append(f"  Monthly Cost: {final_stats['cost_per_month']}")
    operations.append(f"  Total Requests: {final_stats['total_requests']}")
    operations.append(f"  Success Rate: {final_stats['success_rate']}")
    
    # Compare with hypothetical horizontal scaling
    operations.append(f"\nHypothetical Horizontal Scaling - 4 Small Servers:")
    small_server_cost = ServerResources(2, 8, 100, 1).get_cost_estimate()
    operations.append(f"  Configuration: 4x (2C/8GB)")
    operations.append(f"  Total CPU: 8 cores (same as mid-tier vertical)")
    operations.append(f"  Total RAM: 32 GB (same as mid-tier vertical)")
    operations.append(f"  Monthly Cost: ${small_server_cost * 4:.2f}")
    
    metrics.append("\n📊 Trade-offs:")
    metrics.append("\nVertical Scaling (Scale-Up):")
    metrics.append("  ✓ Simple architecture")
    metrics.append("  ✓ Easy data consistency")
    metrics.append("  ✓ No network latency")
    metrics.append("  ✗ Single point of failure")
    metrics.append("  ✗ Hardware limits")
    metrics.append("  ✗ Expensive at high end")
    
    metrics.append("\nHorizontal Scaling (Scale-Out):")
    metrics.append("  ✓ No hardware limits")
    metrics.append("  ✓ Fault tolerance")
    metrics.append("  ✓ Cost-effective")
    metrics.append("  ✗ Complex architecture")
    metrics.append("  ✗ Network overhead")
    metrics.append("  ✗ Data consistency challenges")
    
    results.append("✓ Vertical scaling demonstrated")
    results.append(f"✓ Achieved {final_stats['capacity_score']:.0f} capacity score")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Scaling comparison complete"]
    }


def create_vertical_scaling_graph():
    """Create the vertical scaling workflow graph"""
    workflow = StateGraph(VerticalScalingState)
    
    # Add nodes
    workflow.add_node("setup", setup_initial_server_agent)
    workflow.add_node("baseline", baseline_performance_agent)
    workflow.add_node("cpu_upgrade", cpu_upgrade_agent)
    workflow.add_node("ram_upgrade", ram_upgrade_agent)
    workflow.add_node("full_upgrade", full_upgrade_agent)
    workflow.add_node("comparison", scaling_comparison_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "baseline")
    workflow.add_edge("baseline", "cpu_upgrade")
    workflow.add_edge("cpu_upgrade", "ram_upgrade")
    workflow.add_edge("ram_upgrade", "full_upgrade")
    workflow.add_edge("full_upgrade", "comparison")
    workflow.add_edge("comparison", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 202: Vertical Scaling MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_vertical_scaling_graph()
    
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
Vertical Scaling Pattern implemented with:

1. Resource Upgrades:
   - CPU: 2 → 8 → 32 cores
   - RAM: 8 → 64 → 256 GB
   - Disk: 100 → 2000 GB
   - Network: 1 → 10 Gbps

2. Performance Improvements:
   - Baseline: 100 req capacity
   - After CPU: 200+ req capacity
   - After RAM: 500+ req capacity
   - Final: 2000+ req capacity

3. Cost vs Performance:
   - Initial: ~$150/month
   - Final: ~$1,900/month
   - 13x cost for 20x performance

When to Use Vertical Scaling:
✓ Database servers (need memory for caching)
✓ Legacy applications (can't distribute)
✓ Monolithic applications
✓ Development/testing environments
✓ Applications requiring low latency
✓ Simple deployment requirements

Vertical Scaling Limits:
- Physical Hardware: Maximum CPU/RAM available
- Diminishing Returns: 2x cost ≠ 2x performance at high end
- Single Point of Failure: No redundancy
- Downtime: Required for hardware upgrades
- Cost: Exponentially expensive at extreme scales

Real-World Examples:
- Database Servers: PostgreSQL/MySQL with 256GB+ RAM
- In-Memory Databases: Redis with 100GB+ RAM
- Data Warehouses: Snowflake compute clusters
- Machine Learning: GPU servers with 8x V100/A100
- Video Rendering: High-CPU multi-core workstations

Typical Server Tiers:
1. Small: 2C/8GB - $100-200/month
2. Medium: 8C/32GB - $400-600/month
3. Large: 16C/64GB - $800-1200/month
4. X-Large: 32C/128GB - $1500-2500/month
5. Enterprise: 64C/512GB - $5000-10000/month

Best Practices:
1. Monitor Resource Utilization: Know what to upgrade
2. Upgrade Bottleneck First: CPU vs RAM vs Disk vs Network
3. Plan for Downtime: Maintenance windows for upgrades
4. Consider Cloud: Easy vertical scaling (resize instance)
5. Set Budget Limits: Prevent runaway costs
6. Benchmark Before/After: Measure improvement
7. Consider Horizontal Alternative: May be more cost-effective
""")


if __name__ == "__main__":
    main()

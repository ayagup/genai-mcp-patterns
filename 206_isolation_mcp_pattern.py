"""
Pattern 206: Isolation MCP Pattern

Isolation ensures workloads/tenants don't interfere with each other through:
- Resource limits (CPU, memory quotas)
- Network isolation (VPCs, namespaces)
- Process isolation (containers, VMs)
- Data isolation (separate databases/schemas)

Use Cases:
- Multi-tenant systems
- Security-sensitive applications
- Compliance requirements (PCI-DSS, HIPAA)
- Preventing noisy neighbor problems
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END


class IsolationState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class IsolatedWorkload:
    workload_id: str
    cpu_limit: int
    memory_limit: int
    network_namespace: str
    cpu_usage: int = 0
    memory_usage: int = 0
    
    def can_allocate(self, cpu: int, memory: int) -> bool:
        return (self.cpu_usage + cpu <= self.cpu_limit and
                self.memory_usage + memory <= self.memory_limit)
    
    def allocate(self, cpu: int, memory: int) -> bool:
        if self.can_allocate(cpu, memory):
            self.cpu_usage += cpu
            self.memory_usage += memory
            return True
        return False


class IsolationManager:
    def __init__(self):
        self.workloads: Dict[str, IsolatedWorkload] = {}
    
    def create_workload(self, workload_id: str, cpu_limit: int, memory_limit: int) -> IsolatedWorkload:
        workload = IsolatedWorkload(
            workload_id=workload_id,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            network_namespace=f"ns-{workload_id}"
        )
        self.workloads[workload_id] = workload
        return workload
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_workloads': len(self.workloads),
            'workloads': [
                {
                    'id': w.workload_id,
                    'cpu': f"{w.cpu_usage}/{w.cpu_limit}",
                    'memory': f"{w.memory_usage}/{w.memory_limit}",
                    'namespace': w.network_namespace
                }
                for w in self.workloads.values()
            ]
        }


def demo_agent(state: IsolationState):
    operations = []
    results = []
    
    manager = IsolationManager()
    
    # Create isolated workloads
    workloads_config = [
        ("workload-A", 100, 1000),  # CPU, Memory limits
        ("workload-B", 50, 500),
        ("workload-C", 200, 2000)
    ]
    
    operations.append("Creating Isolated Workloads:")
    for wid, cpu, mem in workloads_config:
        workload = manager.create_workload(wid, cpu, mem)
        operations.append(f"  {wid}: CPU limit={cpu}, Memory limit={mem}, Namespace={workload.network_namespace}")
    
    # Allocate resources (isolated)
    operations.append("\nResource Allocation (Isolated):")
    
    # Workload A tries to use resources
    w_a = manager.workloads["workload-A"]
    success = w_a.allocate(50, 300)
    operations.append(f"  workload-A: Allocate 50CPU/300Mem → {success}")
    operations.append(f"    Usage: {w_a.cpu_usage}/{w_a.cpu_limit} CPU, {w_a.memory_usage}/{w_a.memory_limit} Mem")
    
    # Workload A tries to exceed limit (should fail)
    success = w_a.allocate(100, 1000)
    operations.append(f"  workload-A: Allocate 100CPU/1000Mem → {success} (exceeds limit)")
    
    stats = manager.get_stats()
    results.append(f"✓ Isolation enforced for {stats['total_workloads']} workloads")
    results.append("✓ Resource limits prevent interference")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Isolated workloads: {stats['total_workloads']}"],
        "messages": ["Isolation demo complete"]
    }


def create_isolation_graph():
    workflow = StateGraph(IsolationState)
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    return workflow.compile()


def main():
    print("="*80)
    print("Pattern 206: Isolation MCP Pattern")
    print("="*80)
    
    app = create_isolation_graph()
    final_state = app.invoke({
        "scaling_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    })
    
    for op in final_state["scaling_operations"]:
        print(op)
    
    print("\n" + "="*80)
    print("""
Isolation Techniques:
- Process Isolation: Containers (Docker), VMs
- Resource Isolation: cgroups, resource quotas
- Network Isolation: VPCs, namespaces
- Data Isolation: Separate databases

Benefits:
- Security: Prevent unauthorized access
- Performance: Prevent noisy neighbors
- Reliability: Fault isolation
- Compliance: Meet regulatory requirements

Examples: Kubernetes namespaces, AWS VPC, Docker containers
""")


if __name__ == "__main__":
    main()

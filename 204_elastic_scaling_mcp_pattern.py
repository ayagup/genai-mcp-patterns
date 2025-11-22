"""
Pattern 204: Elastic Scaling MCP Pattern

Elastic scaling combines auto-scaling with rapid provisioning/deprovisioning.
True elasticity means:
- Scale out in seconds/minutes (not hours)
- Scale in immediately when load decreases
- Pay-per-use pricing model
- Seamless capacity adjustments

Use Cases:
- Cloud-native applications
- Serverless functions
- Container orchestration
- Burst workloads
- Variable traffic patterns
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
import time
import random


class ElasticScalingState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class ElasticResource:
    resource_id: str
    capacity: int
    current_load: int = 0
    provisioning_time_sec: float = 30.0  # Time to provision
    
    def can_handle(self, load: int) -> bool:
        return (self.current_load + load) <= self.capacity


class ElasticPool:
    def __init__(self, initial_size: int = 0, provision_time: float = 30.0):
        self.resources: List[ElasticResource] = []
        self.provision_time = provision_time
        self.resource_counter = 0
        self.total_provisioned = 0
        self.total_deprovisioned = 0
        
        for _ in range(initial_size):
            self.provision_resource()
    
    def provision_resource(self) -> ElasticResource:
        """Provision a new resource"""
        self.resource_counter += 1
        resource = ElasticResource(f"resource-{self.resource_counter}", capacity=100, provisioning_time_sec=self.provision_time)
        
        # Simulate provisioning time
        time.sleep(min(self.provision_time / 10, 0.1))  # Scaled for demo
        
        self.resources.append(resource)
        self.total_provisioned += 1
        return resource
    
    def deprovision_resource(self) -> bool:
        """Deprovision a resource"""
        if not self.resources:
            return False
        
        # Remove resource with lowest load
        resource = min(self.resources, key=lambda r: r.current_load)
        if resource.current_load == 0:
            self.resources.remove(resource)
            self.total_deprovisioned += 1
            return True
        
        return False
    
    def handle_load(self, load: int) -> Dict[str, Any]:
        """Handle incoming load elastically"""
        start_time = time.time()
        
        # Try to handle with existing resources
        for resource in self.resources:
            if resource.can_handle(load):
                resource.current_load += load
                return {
                    'handled': True,
                    'scaled': False,
                    'resources': len(self.resources),
                    'time': time.time() - start_time
                }
        
        # Need to scale out
        new_resource = self.provision_resource()
        new_resource.current_load = load
        
        return {
            'handled': True,
            'scaled': True,
            'resources': len(self.resources),
            'time': time.time() - start_time
        }
    
    def release_load(self, load: int):
        """Release load and potentially scale in"""
        for resource in self.resources:
            if resource.current_load >= load:
                resource.current_load -= load
                break
        
        # Try to consolidate
        self.deprovision_resource()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'active_resources': len(self.resources),
            'total_provisioned': self.total_provisioned,
            'total_deprovisioned': self.total_deprovisioned,
            'total_capacity': sum(r.capacity for r in self.resources),
            'current_load': sum(r.current_load for r in self.resources)
        }


def demo_agent(state: ElasticScalingState):
    operations = []
    results = []
    
    pool = ElasticPool(initial_size=1, provision_time=1.0)
    
    operations.append("Elastic Scaling Demo:")
    operations.append(f"Initial resources: {len(pool.resources)}")
    
    # Simulate burst load
    loads = [50, 150, 300, 100, 50]
    for i, load in enumerate(loads):
        result = pool.handle_load(load)
        operations.append(f"\nLoad {i+1}: {load} units")
        operations.append(f"  Resources: {result['resources']}")
        operations.append(f"  Scaled: {result['scaled']}")
        
        time.sleep(0.1)
        pool.release_load(load)
    
    stats = pool.get_stats()
    results.append(f"✓ Elastically handled burst: {stats['total_provisioned']} provisions, {stats['total_deprovisioned']} deprovisions")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Elasticity: {stats['total_provisioned']} scale-outs"],
        "messages": ["Elastic scaling demo complete"]
    }


def create_elastic_scaling_graph():
    workflow = StateGraph(ElasticScalingState)
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    return workflow.compile()


def main():
    print("="*80)
    print("Pattern 204: Elastic Scaling MCP Pattern")
    print("="*80)
    
    app = create_elastic_scaling_graph()
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
Elastic Scaling Benefits:
- Rapid scale-out (seconds/minutes)
- Immediate scale-in
- Pay-per-use pricing
- Cloud-native

Examples: AWS Lambda, Kubernetes, Azure Functions
""")


if __name__ == "__main__":
    main()

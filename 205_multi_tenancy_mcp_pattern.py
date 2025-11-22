"""
Pattern 205: Multi-Tenancy MCP Pattern

Multi-tenancy allows multiple customers (tenants) to share the same infrastructure
while keeping their data and configurations isolated.

Key Features:
- Shared infrastructure, isolated data
- Resource pooling and efficiency
- Per-tenant configuration
- Cost optimization
- Tenant isolation

Models:
- Shared Database, Shared Schema
- Shared Database, Separate Schemas
- Separate Databases per Tenant
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END


class MultiTenancyState(TypedDict):
    scaling_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class Tenant:
    tenant_id: str
    name: str
    tier: str  # free, basic, premium
    data: Dict[str, Any] = field(default_factory=dict)
    resource_quota: int = 100
    current_usage: int = 0


class MultiTenantSystem:
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.shared_resources = {"cpu": 1000, "memory": 10000}
    
    def add_tenant(self, tenant: Tenant):
        self.tenants[tenant.tenant_id] = tenant
    
    def get_tenant_data(self, tenant_id: str, key: str) -> Any:
        """Get tenant-specific data"""
        if tenant_id in self.tenants:
            return self.tenants[tenant_id].data.get(key)
        return None
    
    def set_tenant_data(self, tenant_id: str, key: str, value: Any):
        """Set tenant-specific data (isolated)"""
        if tenant_id in self.tenants:
            self.tenants[tenant_id].data[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_tenants': len(self.tenants),
            'by_tier': {
                'free': sum(1 for t in self.tenants.values() if t.tier == 'free'),
                'basic': sum(1 for t in self.tenants.values() if t.tier == 'basic'),
                'premium': sum(1 for t in self.tenants.values() if t.tier == 'premium')
            }
        }


def demo_agent(state: MultiTenancyState):
    operations = []
    results = []
    
    system = MultiTenantSystem()
    
    # Add tenants
    tenants = [
        Tenant("tenant-1", "Acme Corp", "premium"),
        Tenant("tenant-2", "Beta LLC", "basic"),
        Tenant("tenant-3", "Gamma Inc", "free")
    ]
    
    for tenant in tenants:
        system.add_tenant(tenant)
        operations.append(f"Added tenant: {tenant.name} ({tenant.tier})")
    
    # Isolated data operations
    system.set_tenant_data("tenant-1", "users_count", 1000)
    system.set_tenant_data("tenant-2", "users_count", 50)
    system.set_tenant_data("tenant-3", "users_count", 5)
    
    operations.append("\nTenant Data (Isolated):")
    for tid in ["tenant-1", "tenant-2", "tenant-3"]:
        users = system.get_tenant_data(tid, "users_count")
        operations.append(f"  {tid}: {users} users")
    
    stats = system.get_stats()
    results.append(f"✓ Multi-tenancy: {stats['total_tenants']} tenants sharing infrastructure")
    
    return {
        "scaling_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Tenants: {stats['total_tenants']}"],
        "messages": ["Multi-tenancy demo complete"]
    }


def create_multi_tenancy_graph():
    workflow = StateGraph(MultiTenancyState)
    workflow.add_node("demo", demo_agent)
    workflow.add_edge(START, "demo")
    workflow.add_edge("demo", END)
    return workflow.compile()


def main():
    print("="*80)
    print("Pattern 205: Multi-Tenancy MCP Pattern")
    print("="*80)
    
    app = create_multi_tenancy_graph()
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
Multi-Tenancy Benefits:
- Cost efficiency (shared resources)
- Easier maintenance
- Tenant isolation
- Scalable architecture

Examples: Salesforce, Slack, GitHub (SaaS platforms)
""")


if __name__ == "__main__":
    main()

"""
Pattern 244: Context Isolation MCP Pattern

This pattern demonstrates context isolation - keeping different contexts
separate to prevent interference and maintain security.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextIsolationState(TypedDict):
    """State for context isolation workflow"""
    messages: Annotated[List[str], add]
    isolated_contexts: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]


class IsolatedContext:
    """Represents an isolated context"""
    
    def __init__(self, context_id: str, permissions: List[str]):
        self.context_id = context_id
        self.permissions = permissions
        self.data = {}
    
    def can_access(self, resource: str) -> bool:
        """Check if context can access resource"""
        return resource in self.permissions
    
    def set_data(self, key: str, value: Any):
        """Set data in isolated context"""
        self.data[key] = value
    
    def get_data(self, key: str) -> Any:
        """Get data from isolated context"""
        return self.data.get(key)


def create_isolated_contexts_agent(state: ContextIsolationState) -> ContextIsolationState:
    """Create isolated contexts"""
    print("\n🔒 Creating Isolated Contexts...")
    
    contexts = [
        {"id": "user_context", "permissions": ["read_profile", "update_profile"]},
        {"id": "admin_context", "permissions": ["read_profile", "update_profile", "delete_user", "manage_roles"]},
        {"id": "guest_context", "permissions": ["read_profile"]},
        {"id": "service_context", "permissions": ["read_data", "write_data", "execute_tasks"]}
    ]
    
    isolated_contexts = []
    for ctx_config in contexts:
        ctx = IsolatedContext(ctx_config["id"], ctx_config["permissions"])
        isolated_contexts.append({
            "id": ctx.context_id,
            "permissions": ctx.permissions,
            "data_count": len(ctx.data)
        })
        
        print(f"\n  ✓ Created {ctx.context_id}")
        print(f"    Permissions: {', '.join(ctx.permissions)}")
    
    return {
        **state,
        "isolated_contexts": isolated_contexts,
        "messages": [f"✓ Created {len(isolated_contexts)} isolated contexts"]
    }


def validate_isolation_agent(state: ContextIsolationState) -> ContextIsolationState:
    """Validate context isolation"""
    print("\n✅ Validating Context Isolation...")
    
    validation_results = []
    
    # Test access control
    test_cases = [
        {"context": "user_context", "resource": "read_profile", "expected": True},
        {"context": "user_context", "resource": "delete_user", "expected": False},
        {"context": "admin_context", "resource": "manage_roles", "expected": True},
        {"context": "guest_context", "resource": "update_profile", "expected": False},
    ]
    
    for test in test_cases:
        ctx_info = next((c for c in state["isolated_contexts"] if c["id"] == test["context"]), None)
        if ctx_info:
            has_permission = test["resource"] in ctx_info["permissions"]
            passed = has_permission == test["expected"]
            
            validation_results.append({
                "context": test["context"],
                "resource": test["resource"],
                "has_access": has_permission,
                "expected": test["expected"],
                "passed": passed
            })
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n  {status}: {test['context']} → {test['resource']}")
            print(f"    Expected: {test['expected']}, Got: {has_permission}")
    
    passed_count = sum(1 for r in validation_results if r["passed"])
    print(f"\n  Validation Results: {passed_count}/{len(validation_results)} passed")
    
    return {
        **state,
        "validation_results": validation_results,
        "messages": [f"✓ Validated {len(validation_results)} isolation tests"]
    }


def generate_context_isolation_report_agent(state: ContextIsolationState) -> ContextIsolationState:
    """Generate context isolation report"""
    print("\n" + "="*70)
    print("CONTEXT ISOLATION REPORT")
    print("="*70)
    
    print(f"\n🔒 Isolated Contexts ({len(state['isolated_contexts'])}):")
    for ctx in state["isolated_contexts"]:
        print(f"\n  • {ctx['id']}")
        print(f"    Permissions: {len(ctx['permissions'])} total")
        for perm in ctx["permissions"]:
            print(f"      - {perm}")
    
    print(f"\n✅ Validation Results:")
    passed = sum(1 for r in state["validation_results"] if r["passed"])
    total = len(state["validation_results"])
    print(f"  Passed: {passed}/{total}")
    
    for result in state["validation_results"]:
        status = "✓" if result["passed"] else "✗"
        print(f"  {status} {result['context']} accessing {result['resource']}")
    
    print("\n💡 Context Isolation Benefits:")
    print("  • Security through separation")
    print("  • Prevent unauthorized access")
    print("  • Data integrity protection")
    print("  • Reduced attack surface")
    print("  • Clear permission boundaries")
    print("  • Auditability")
    
    print("\n="*70)
    print("✅ Context Isolation Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_isolation_graph():
    workflow = StateGraph(ContextIsolationState)
    workflow.add_node("create", create_isolated_contexts_agent)
    workflow.add_node("validate", validate_isolation_agent)
    workflow.add_node("report", generate_context_isolation_report_agent)
    workflow.add_edge(START, "create")
    workflow.add_edge("create", "validate")
    workflow.add_edge("validate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 244: Context Isolation MCP Pattern")
    print("="*70)
    
    app = create_context_isolation_graph()
    final_state = app.invoke({
        "messages": [],
        "isolated_contexts": [],
        "validation_results": []
    })
    print("\n✅ Context Isolation Pattern Complete!")


if __name__ == "__main__":
    main()

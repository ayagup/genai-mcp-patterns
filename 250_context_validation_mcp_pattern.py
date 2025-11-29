"""
Pattern 250: Context Validation MCP Pattern

This pattern demonstrates context validation - verifying context completeness,
consistency, and correctness before use.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextValidationState(TypedDict):
    """State for context validation workflow"""
    messages: Annotated[List[str], add]
    context: Dict[str, Any]
    validation_results: List[Dict[str, Any]]
    is_valid: bool


class ContextValidator:
    """Validates context"""
    
    def __init__(self):
        self.required_fields = ["user_id", "action", "timestamp"]
        self.field_types = {
            "user_id": str,
            "action": str,
            "timestamp": str,
            "priority": int,
            "metadata": dict
        }
    
    def validate_completeness(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all required fields are present"""
        missing = [field for field in self.required_fields if field not in context]
        
        return {
            "check": "completeness",
            "passed": len(missing) == 0,
            "details": f"Missing fields: {missing}" if missing else "All required fields present"
        }
    
    def validate_types(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate field types"""
        type_errors = []
        
        for field, expected_type in self.field_types.items():
            if field in context:
                if not isinstance(context[field], expected_type):
                    type_errors.append(f"{field}: expected {expected_type.__name__}, got {type(context[field]).__name__}")
        
        return {
            "check": "type_validation",
            "passed": len(type_errors) == 0,
            "details": f"Type errors: {type_errors}" if type_errors else "All types valid"
        }
    
    def validate_consistency(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate logical consistency"""
        errors = []
        
        # Example: priority should be 1-10
        if "priority" in context:
            if not (1 <= context["priority"] <= 10):
                errors.append("priority must be between 1 and 10")
        
        # Example: timestamp format (simplified check)
        if "timestamp" in context:
            if not isinstance(context["timestamp"], str) or len(context["timestamp"]) < 10:
                errors.append("timestamp format invalid")
        
        return {
            "check": "consistency",
            "passed": len(errors) == 0,
            "details": f"Errors: {errors}" if errors else "All consistency checks passed"
        }
    
    def validate_security(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate security constraints"""
        issues = []
        
        # Check for sensitive data exposure
        if "password" in str(context).lower():
            issues.append("Sensitive data (password) detected")
        
        # Check for SQL injection patterns
        dangerous_patterns = ["DROP TABLE", "DELETE FROM", "INSERT INTO"]
        for pattern in dangerous_patterns:
            if pattern in str(context).upper():
                issues.append(f"Potential SQL injection: {pattern}")
        
        return {
            "check": "security",
            "passed": len(issues) == 0,
            "details": f"Issues: {issues}" if issues else "No security issues detected"
        }


def load_context_for_validation_agent(state: ContextValidationState) -> ContextValidationState:
    """Load context to validate"""
    print("\n📥 Loading Context for Validation...")
    
    context = {
        "user_id": "user_12345",
        "action": "update_profile",
        "timestamp": "2024-11-29T10:30:00Z",
        "priority": 5,
        "metadata": {
            "source": "web_app",
            "version": "2.0.0"
        }
    }
    
    print(f"\n  Context Fields: {len(context)}")
    for key, value in context.items():
        print(f"    {key}: {value}")
    
    return {
        **state,
        "context": context,
        "messages": [f"✓ Loaded context with {len(context)} fields"]
    }


def validate_context_agent(state: ContextValidationState) -> ContextValidationState:
    """Validate context"""
    print("\n✅ Validating Context...")
    
    validator = ContextValidator()
    validation_results = []
    
    # Run all validation checks
    checks = [
        validator.validate_completeness(state["context"]),
        validator.validate_types(state["context"]),
        validator.validate_consistency(state["context"]),
        validator.validate_security(state["context"])
    ]
    
    for check in checks:
        validation_results.append(check)
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        print(f"\n  {status}: {check['check']}")
        print(f"    {check['details']}")
    
    all_passed = all(check["passed"] for check in checks)
    
    print(f"\n  Overall Status: {'✓ VALID' if all_passed else '✗ INVALID'}")
    print(f"  Checks Passed: {sum(1 for c in checks if c['passed'])}/{len(checks)}")
    
    return {
        **state,
        "validation_results": validation_results,
        "is_valid": all_passed,
        "messages": [f"✓ Validation complete: {'VALID' if all_passed else 'INVALID'}"]
    }


def generate_context_validation_report_agent(state: ContextValidationState) -> ContextValidationState:
    """Generate context validation report"""
    print("\n" + "="*70)
    print("CONTEXT VALIDATION REPORT")
    print("="*70)
    
    print(f"\n📊 Context Under Validation:")
    for key, value in state["context"].items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Validation Results:")
    passed_count = sum(1 for r in state["validation_results"] if r["passed"])
    total_count = len(state["validation_results"])
    
    print(f"  Total Checks: {total_count}")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {total_count - passed_count}")
    print(f"  Overall Status: {'✓ VALID' if state['is_valid'] else '✗ INVALID'}")
    
    print(f"\n📋 Detailed Results:")
    for result in state["validation_results"]:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"\n  {status}: {result['check'].upper()}")
        print(f"    {result['details']}")
    
    print("\n💡 Context Validation Benefits:")
    print("  • Data quality assurance")
    print("  • Early error detection")
    print("  • Security protection")
    print("  • Consistency enforcement")
    print("  • Reliability improvement")
    print("  • Prevents downstream issues")
    
    print("\n="*70)
    if state["is_valid"]:
        print("✅ Context Validation Complete - VALID!")
    else:
        print("⚠️ Context Validation Complete - INVALID!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_validation_graph():
    workflow = StateGraph(ContextValidationState)
    workflow.add_node("load", load_context_for_validation_agent)
    workflow.add_node("validate", validate_context_agent)
    workflow.add_node("report", generate_context_validation_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "validate")
    workflow.add_edge("validate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 250: Context Validation MCP Pattern")
    print("="*70)
    
    app = create_context_validation_graph()
    final_state = app.invoke({
        "messages": [],
        "context": {},
        "validation_results": [],
        "is_valid": False
    })
    print("\n✅ Context Validation Pattern Complete!")


if __name__ == "__main__":
    main()

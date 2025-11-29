"""
Pattern 232: Backward Compatibility MCP Pattern

This pattern demonstrates maintaining backward compatibility - ensuring that
new versions work with code/data from older versions.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class BackwardCompatibilityState(TypedDict):
    """State for backward compatibility workflow"""
    messages: Annotated[List[str], add]
    api_versions: List[str]
    compatibility_tests: List[dict]
    compatibility_report: List[dict]


# API Version 1 (Old)
class APIv1:
    """Old API version that clients are using"""
    
    @staticmethod
    def get_user(user_id: int) -> Dict[str, Any]:
        """V1: Returns user with simple structure"""
        return {
            "id": user_id,
            "name": "John Doe",
            "email": "john@example.com"
        }
    
    @staticmethod
    def create_user(name: str, email: str) -> Dict[str, Any]:
        """V1: Create user with basic fields"""
        return {
            "id": 123,
            "name": name,
            "email": email
        }


# API Version 2 (New) - Backward Compatible
class APIv2:
    """New API version with additional features but backwards compatible"""
    
    @staticmethod
    def get_user(user_id: int, include_preferences: bool = False) -> Dict[str, Any]:
        """
        V2: Returns user with optional preferences
        Backward compatible: still works without include_preferences parameter
        """
        user = {
            "id": user_id,
            "name": "John Doe",
            "email": "john@example.com"
        }
        
        # New feature: optional preferences
        if include_preferences:
            user["preferences"] = {
                "theme": "dark",
                "language": "en",
                "notifications": True
            }
        
        return user
    
    @staticmethod
    def create_user(name: str, email: str, phone: str = None, age: int = None) -> Dict[str, Any]:
        """
        V2: Create user with optional new fields
        Backward compatible: phone and age are optional
        """
        user = {
            "id": 123,
            "name": name,
            "email": email
        }
        
        # New optional fields
        if phone:
            user["phone"] = phone
        if age:
            user["age"] = age
        
        return user


# Compatibility Layer
class BackwardCompatibleAPI:
    """
    API layer that maintains backward compatibility
    Handles both old and new request formats
    """
    
    def __init__(self):
        self.v1_api = APIv1()
        self.v2_api = APIv2()
    
    def handle_request(self, version: str, method: str, **kwargs) -> Dict[str, Any]:
        """
        Route requests to appropriate version
        Ensures backward compatibility
        """
        if version == "v1":
            # Route to v1 API
            if method == "get_user":
                return self.v1_api.get_user(kwargs["user_id"])
            elif method == "create_user":
                return self.v1_api.create_user(kwargs["name"], kwargs["email"])
        
        elif version == "v2":
            # Route to v2 API with backward compatible defaults
            if method == "get_user":
                return self.v2_api.get_user(
                    kwargs["user_id"],
                    kwargs.get("include_preferences", False)
                )
            elif method == "create_user":
                return self.v2_api.create_user(
                    kwargs["name"],
                    kwargs["email"],
                    kwargs.get("phone"),
                    kwargs.get("age")
                )
        
        return {"error": "Unsupported version or method"}


# Agent functions
def define_compatibility_requirements_agent(state: BackwardCompatibilityState) -> BackwardCompatibilityState:
    """Agent that defines backward compatibility requirements"""
    print("\n📋 Defining Backward Compatibility Requirements...")
    
    requirements = [
        "V1 API calls must continue to work",
        "V1 response format must be preserved",
        "V1 parameters must remain optional in V2",
        "New V2 features should be opt-in",
        "Error handling must be consistent"
    ]
    
    print("\n  Compatibility Requirements:")
    for i, req in enumerate(requirements, 1):
        print(f"    {i}. {req}")
    
    api_versions = ["v1", "v2"]
    
    return {
        **state,
        "api_versions": api_versions,
        "messages": [f"✓ Defined {len(requirements)} compatibility requirements"]
    }


def test_backward_compatibility_agent(state: BackwardCompatibilityState) -> BackwardCompatibilityState:
    """Agent that tests backward compatibility"""
    print("\n🧪 Testing Backward Compatibility...")
    
    api = BackwardCompatibleAPI()
    compatibility_tests = []
    
    # Test 1: V1 get_user still works
    print("\n  Test 1: V1 get_user compatibility")
    try:
        v1_result = api.handle_request("v1", "get_user", user_id=1)
        v1_has_required_fields = "id" in v1_result and "name" in v1_result and "email" in v1_result
        
        compatibility_tests.append({
            "test": "V1 get_user",
            "status": "PASS" if v1_has_required_fields else "FAIL",
            "details": "V1 API returns expected fields",
            "result": v1_result
        })
        print(f"    ✓ PASS - V1 API works as expected")
    except Exception as e:
        compatibility_tests.append({
            "test": "V1 get_user",
            "status": "FAIL",
            "details": f"Error: {str(e)}",
            "result": None
        })
        print(f"    ✗ FAIL - {str(e)}")
    
    # Test 2: V2 with V1 parameters (backward compatible)
    print("\n  Test 2: V2 with V1-style parameters")
    try:
        v2_v1_style = api.handle_request("v2", "get_user", user_id=1)
        # Should work like V1 but through V2
        v2_compatible = "id" in v2_v1_style and "preferences" not in v2_v1_style
        
        compatibility_tests.append({
            "test": "V2 with V1 parameters",
            "status": "PASS" if v2_compatible else "FAIL",
            "details": "V2 API works with V1-style calls",
            "result": v2_v1_style
        })
        print(f"    ✓ PASS - V2 is backward compatible")
    except Exception as e:
        compatibility_tests.append({
            "test": "V2 with V1 parameters",
            "status": "FAIL",
            "details": f"Error: {str(e)}",
            "result": None
        })
        print(f"    ✗ FAIL - {str(e)}")
    
    # Test 3: V2 with new features
    print("\n  Test 3: V2 with new features")
    try:
        v2_with_features = api.handle_request("v2", "get_user", user_id=1, include_preferences=True)
        has_new_features = "preferences" in v2_with_features
        
        compatibility_tests.append({
            "test": "V2 with new features",
            "status": "PASS" if has_new_features else "FAIL",
            "details": "V2 new features are accessible",
            "result": v2_with_features
        })
        print(f"    ✓ PASS - V2 new features work")
    except Exception as e:
        compatibility_tests.append({
            "test": "V2 with new features",
            "status": "FAIL",
            "details": f"Error: {str(e)}",
            "result": None
        })
        print(f"    ✗ FAIL - {str(e)}")
    
    # Test 4: V1 create_user
    print("\n  Test 4: V1 create_user compatibility")
    try:
        v1_create = api.handle_request("v1", "create_user", name="Jane", email="jane@example.com")
        v1_create_ok = "id" in v1_create and "name" in v1_create
        
        compatibility_tests.append({
            "test": "V1 create_user",
            "status": "PASS" if v1_create_ok else "FAIL",
            "details": "V1 create user works",
            "result": v1_create
        })
        print(f"    ✓ PASS - V1 create user works")
    except Exception as e:
        compatibility_tests.append({
            "test": "V1 create_user",
            "status": "FAIL",
            "details": f"Error: {str(e)}",
            "result": None
        })
        print(f"    ✗ FAIL - {str(e)}")
    
    # Test 5: V2 create_user with optional fields
    print("\n  Test 5: V2 create_user with optional fields")
    try:
        v2_create = api.handle_request("v2", "create_user", 
                                       name="Bob", email="bob@example.com",
                                       phone="555-0123", age=30)
        has_optional = "phone" in v2_create and "age" in v2_create
        
        compatibility_tests.append({
            "test": "V2 create_user with optional fields",
            "status": "PASS" if has_optional else "FAIL",
            "details": "V2 create user with optional fields works",
            "result": v2_create
        })
        print(f"    ✓ PASS - V2 create user with new fields works")
    except Exception as e:
        compatibility_tests.append({
            "test": "V2 create_user with optional fields",
            "status": "FAIL",
            "details": f"Error: {str(e)}",
            "result": None
        })
        print(f"    ✗ FAIL - {str(e)}")
    
    return {
        **state,
        "compatibility_tests": compatibility_tests,
        "messages": [f"✓ Ran {len(compatibility_tests)} compatibility tests"]
    }


def analyze_compatibility_agent(state: BackwardCompatibilityState) -> BackwardCompatibilityState:
    """Agent that analyzes compatibility results"""
    print("\n📊 Analyzing Compatibility Results...")
    
    passed = sum(1 for t in state["compatibility_tests"] if t["status"] == "PASS")
    failed = sum(1 for t in state["compatibility_tests"] if t["status"] == "FAIL")
    total = len(state["compatibility_tests"])
    
    compatibility_score = (passed / total * 100) if total > 0 else 0
    
    print(f"\n  Total Tests: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Compatibility Score: {compatibility_score:.1f}%")
    
    if compatibility_score == 100:
        status = "FULLY COMPATIBLE"
        recommendation = "Safe to deploy - all backward compatibility maintained"
    elif compatibility_score >= 80:
        status = "MOSTLY COMPATIBLE"
        recommendation = "Review failures before deployment"
    else:
        status = "COMPATIBILITY ISSUES"
        recommendation = "Fix compatibility issues before deployment"
    
    print(f"\n  Status: {status}")
    print(f"  Recommendation: {recommendation}")
    
    compatibility_report = [{
        "compatibility_score": compatibility_score,
        "status": status,
        "recommendation": recommendation,
        "passed": passed,
        "failed": failed,
        "total": total
    }]
    
    return {
        **state,
        "compatibility_report": compatibility_report,
        "messages": [f"✓ Compatibility score: {compatibility_score:.1f}%"]
    }


def generate_compatibility_report_agent(state: BackwardCompatibilityState) -> BackwardCompatibilityState:
    """Agent that generates compatibility report"""
    print("\n" + "="*70)
    print("BACKWARD COMPATIBILITY REPORT")
    print("="*70)
    
    print(f"\n📊 API Versions Tested: {', '.join(state['api_versions'])}")
    
    report = state["compatibility_report"][0]
    print(f"\n🎯 Compatibility Score: {report['compatibility_score']:.1f}%")
    print(f"   Status: {report['status']}")
    print(f"   Recommendation: {report['recommendation']}")
    
    print(f"\n📋 Test Results ({report['passed']}/{report['total']} passed):")
    for i, test in enumerate(state["compatibility_tests"], 1):
        status_icon = "✓" if test["status"] == "PASS" else "✗"
        print(f"\n  {status_icon} {i}. {test['test']}")
        print(f"      Status: {test['status']}")
        print(f"      Details: {test['details']}")
    
    print("\n💡 Backward Compatibility Best Practices:")
    print("  • Don't remove existing fields from responses")
    print("  • Make new parameters optional with sensible defaults")
    print("  • Maintain existing behavior when new features not requested")
    print("  • Version your API and support multiple versions")
    print("  • Provide clear migration guides for deprecated features")
    print("  • Test old client code against new API versions")
    
    print("\n" + "="*70)
    if report['compatibility_score'] == 100:
        print("✅ Backward Compatibility VERIFIED - Safe to Deploy!")
    else:
        print(f"⚠️ Backward Compatibility Issues - {report['failed']} tests failed")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Compatibility report generated"]
    }


# Create the graph
def create_backward_compatibility_graph():
    """Create the backward compatibility workflow graph"""
    workflow = StateGraph(BackwardCompatibilityState)
    
    # Add nodes
    workflow.add_node("define_requirements", define_compatibility_requirements_agent)
    workflow.add_node("test_compatibility", test_backward_compatibility_agent)
    workflow.add_node("analyze_results", analyze_compatibility_agent)
    workflow.add_node("generate_report", generate_compatibility_report_agent)
    
    # Add edges
    workflow.add_edge(START, "define_requirements")
    workflow.add_edge("define_requirements", "test_compatibility")
    workflow.add_edge("test_compatibility", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 232: Backward Compatibility MCP Pattern")
    print("="*70)
    print("\nBackward Compatibility: New versions work with old client code")
    print("Ensures existing clients don't break when you update!")
    
    # Create and run the workflow
    app = create_backward_compatibility_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "api_versions": [],
        "compatibility_tests": [],
        "compatibility_report": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Backward Compatibility Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Pattern 234: API Versioning MCP Pattern

This pattern demonstrates API versioning - managing multiple API versions
simultaneously to support different client requirements.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class APIVersioningState(TypedDict):
    """State for API versioning workflow"""
    messages: Annotated[List[str], add]
    supported_versions: List[str]
    routing_results: List[dict]


# API Versioning Manager
class APIVersionManager:
    """Manages multiple API versions"""
    
    def __init__(self):
        self.supported_versions = ["v1", "v2", "v3"]
        self.default_version = "v2"
    
    def route_request(self, version: str, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route request to appropriate version handler"""
        if version == "v1":
            return self._handle_v1(endpoint, params)
        elif version == "v2":
            return self._handle_v2(endpoint, params)
        elif version == "v3":
            return self._handle_v3(endpoint, params)
        else:
            return {"error": f"Unsupported version: {version}", "supported": self.supported_versions}
    
    def _handle_v1(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """V1 API handler - legacy format"""
        if endpoint == "/users":
            return {
                "version": "v1",
                "data": [
                    {"id": 1, "name": "John"},
                    {"id": 2, "name": "Jane"}
                ]
            }
        return {"error": "Endpoint not found in v1"}
    
    def _handle_v2(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """V2 API handler - current version with more features"""
        if endpoint == "/users":
            return {
                "version": "v2",
                "data": [
                    {"id": 1, "name": "John", "email": "john@example.com"},
                    {"id": 2, "name": "Jane", "email": "jane@example.com"}
                ],
                "metadata": {
                    "total": 2,
                    "page": 1
                }
            }
        return {"error": "Endpoint not found in v2"}
    
    def _handle_v3(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """V3 API handler - latest version with newest features"""
        if endpoint == "/users":
            return {
                "version": "v3",
                "data": [
                    {
                        "id": 1,
                        "name": "John",
                        "email": "john@example.com",
                        "profile": {"avatar": "john.jpg", "bio": "Developer"}
                    },
                    {
                        "id": 2,
                        "name": "Jane",
                        "email": "jane@example.com",
                        "profile": {"avatar": "jane.jpg", "bio": "Designer"}
                    }
                ],
                "metadata": {
                    "total": 2,
                    "page": 1,
                    "per_page": 10
                },
                "links": {
                    "self": "/v3/users?page=1",
                    "next": None
                }
            }
        return {"error": "Endpoint not found in v3"}


# Agent functions
def setup_version_manager_agent(state: APIVersioningState) -> APIVersioningState:
    """Setup API version manager"""
    print("\n⚙️ Setting Up API Version Manager...")
    
    manager = APIVersionManager()
    
    print(f"\n  Supported Versions: {', '.join(manager.supported_versions)}")
    print(f"  Default Version: {manager.default_version}")
    
    return {
        **state,
        "supported_versions": manager.supported_versions,
        "messages": [f"✓ Configured {len(manager.supported_versions)} API versions"]
    }


def test_version_routing_agent(state: APIVersioningState) -> APIVersioningState:
    """Test routing to different API versions"""
    print("\n🧪 Testing Version Routing...")
    
    manager = APIVersionManager()
    routing_results = []
    
    # Test V1
    print("\n  Testing V1 API...")
    v1_result = manager.route_request("v1", "/users", {})
    routing_results.append({
        "version": "v1",
        "status": "SUCCESS" if "data" in v1_result else "FAIL",
        "response": v1_result,
        "description": "Legacy API - minimal fields"
    })
    print(f"    ✓ V1 Response: {len(v1_result.get('data', []))} users")
    
    # Test V2
    print("\n  Testing V2 API...")
    v2_result = manager.route_request("v2", "/users", {})
    routing_results.append({
        "version": "v2",
        "status": "SUCCESS" if "metadata" in v2_result else "FAIL",
        "response": v2_result,
        "description": "Current API - added metadata"
    })
    print(f"    ✓ V2 Response: {len(v2_result.get('data', []))} users with metadata")
    
    # Test V3
    print("\n  Testing V3 API...")
    v3_result = manager.route_request("v3", "/users", {})
    routing_results.append({
        "version": "v3",
        "status": "SUCCESS" if "links" in v3_result else "FAIL",
        "response": v3_result,
        "description": "Latest API - full features with HATEOAS links"
    })
    print(f"    ✓ V3 Response: {len(v3_result.get('data', []))} users with links")
    
    # Test unsupported version
    print("\n  Testing Unsupported Version...")
    invalid_result = manager.route_request("v99", "/users", {})
    routing_results.append({
        "version": "v99",
        "status": "EXPECTED_ERROR" if "error" in invalid_result else "FAIL",
        "response": invalid_result,
        "description": "Should return error for unsupported version"
    })
    print(f"    ✓ V99 Response: {invalid_result.get('error', 'Unknown')}")
    
    return {
        **state,
        "routing_results": routing_results,
        "messages": [f"✓ Tested {len(routing_results)} version routes"]
    }


def generate_versioning_report_agent(state: APIVersioningState) -> APIVersioningState:
    """Generate API versioning report"""
    print("\n" + "="*70)
    print("API VERSIONING REPORT")
    print("="*70)
    
    print(f"\n📋 Supported Versions: {', '.join(state['supported_versions'])}")
    
    print(f"\n🧪 Routing Test Results:")
    for result in state["routing_results"]:
        status_icon = "✓" if result["status"] in ["SUCCESS", "EXPECTED_ERROR"] else "✗"
        print(f"\n  {status_icon} {result['version']}: {result['status']}")
        print(f"      {result['description']}")
    
    print("\n💡 API Versioning Strategies:")
    print("  1. URL Versioning: /v1/users, /v2/users")
    print("  2. Header Versioning: Accept: application/vnd.api.v1+json")
    print("  3. Query Parameter: /users?version=v1")
    print("  4. Content Negotiation: via Accept header")
    
    print("\n📚 Best Practices:")
    print("  • Support at least 2 versions simultaneously")
    print("  • Deprecate old versions gradually")
    print("  • Document version differences clearly")
    print("  • Use semantic versioning for clarity")
    print("  • Provide migration guides between versions")
    
    print("\n" + "="*70)
    print(f"✅ API Versioning Pattern Complete!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Versioning report generated"]
    }


# Create the graph
def create_api_versioning_graph():
    """Create the API versioning workflow graph"""
    workflow = StateGraph(APIVersioningState)
    
    # Add nodes
    workflow.add_node("setup_manager", setup_version_manager_agent)
    workflow.add_node("test_routing", test_version_routing_agent)
    workflow.add_node("generate_report", generate_versioning_report_agent)
    
    # Add edges
    workflow.add_edge(START, "setup_manager")
    workflow.add_edge("setup_manager", "test_routing")
    workflow.add_edge("test_routing", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 234: API Versioning MCP Pattern")
    print("="*70)
    print("\nAPI Versioning: Support multiple API versions simultaneously")
    
    # Create and run the workflow
    app = create_api_versioning_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "supported_versions": [],
        "routing_results": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ API Versioning Pattern Complete!")


if __name__ == "__main__":
    main()

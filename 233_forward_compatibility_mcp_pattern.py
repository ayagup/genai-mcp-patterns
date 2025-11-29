"""
Pattern 233: Forward Compatibility MCP Pattern

This pattern demonstrates forward compatibility - ensuring that old code
can handle data/features from newer versions gracefully.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ForwardCompatibilityState(TypedDict):
    """State for forward compatibility workflow"""
    messages: Annotated[List[str], add]
    test_results: List[dict]
    compatibility_score: float


# Old Client (V1) - Must handle new data formats
class OldClient:
    """
    Older client that might receive data from newer servers
    Should handle unknown fields gracefully
    """
    
    def parse_user_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse user data, ignoring unknown fields (forward compatible)
        """
        # Extract only the fields we know about
        known_fields = ["id", "name", "email"]
        parsed = {k: v for k, v in data.items() if k in known_fields}
        
        # Store unknown fields for future use (don't discard!)
        unknown_fields = {k: v for k, v in data.items() if k not in known_fields}
        if unknown_fields:
            parsed["_unknown_fields"] = unknown_fields
        
        return parsed
    
    def update_user(self, original_data: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update user, preserving unknown fields (round-trip compatibility)
        """
        # Merge updates with original data
        result = {**original_data, **updates}
        
        # Preserve unknown fields from original
        if "_unknown_fields" in original_data:
            result.update(original_data["_unknown_fields"])
        
        return result


# New Server (V2) - Sends additional fields
class NewServer:
    """
    Newer server that sends additional fields
    Old clients should handle this gracefully
    """
    
    def get_user(self, user_id: int) -> Dict[str, Any]:
        """
        Returns user with new fields that old clients don't expect
        """
        return {
            "id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            # New fields that old clients don't know about
            "phone": "555-0123",
            "age": 30,
            "preferences": {
                "theme": "dark",
                "language": "en"
            },
            "metadata": {
                "created_at": "2024-01-01",
                "updated_at": "2024-01-15"
            }
        }


# Agent functions
def test_ignore_unknown_fields_agent(state: ForwardCompatibilityState) -> ForwardCompatibilityState:
    """Test that old client ignores unknown fields"""
    print("\n🧪 Test 1: Ignore Unknown Fields...")
    
    server = NewServer()
    client = OldClient()
    
    # Server sends data with new fields
    server_data = server.get_user(123)
    
    # Old client parses it
    client_data = client.parse_user_data(server_data)
    
    # Verify old client extracted known fields
    has_known_fields = all(field in client_data for field in ["id", "name", "email"])
    
    # Verify unknown fields were preserved
    preserved_unknown = "_unknown_fields" in client_data
    
    test_result = {
        "test": "Ignore Unknown Fields",
        "status": "PASS" if has_known_fields and preserved_unknown else "FAIL",
        "details": f"Client extracted known fields and preserved {len(client_data.get('_unknown_fields', {}))} unknown fields",
        "known_fields_ok": has_known_fields,
        "unknown_preserved": preserved_unknown
    }
    
    if test_result["status"] == "PASS":
        print(f"  ✓ PASS - Client handled {len(client_data.get('_unknown_fields', {}))} unknown fields gracefully")
    else:
        print(f"  ✗ FAIL - Client did not handle unknown fields properly")
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Test 1 complete"]
    }


def test_round_trip_compatibility_agent(state: ForwardCompatibilityState) -> ForwardCompatibilityState:
    """Test round-trip compatibility (preserve unknown fields)"""
    print("\n🧪 Test 2: Round-Trip Compatibility...")
    
    server = NewServer()
    client = OldClient()
    
    # Server sends data with new fields
    original_data = server.get_user(123)
    
    # Old client receives and parses it
    parsed_data = client.parse_user_data(original_data)
    
    # Client makes an update (only knows about old fields)
    updated_data = client.update_user(parsed_data, {"email": "newemail@example.com"})
    
    # Verify unknown fields were preserved in the update
    new_fields_preserved = all(
        field in updated_data 
        for field in ["phone", "age", "preferences", "metadata"]
    )
    
    # Verify the known field was updated
    known_field_updated = updated_data["email"] == "newemail@example.com"
    
    test_result = {
        "test": "Round-Trip Compatibility",
        "status": "PASS" if new_fields_preserved and known_field_updated else "FAIL",
        "details": "Unknown fields preserved during update",
        "new_fields_preserved": new_fields_preserved,
        "known_field_updated": known_field_updated
    }
    
    if test_result["status"] == "PASS":
        print("  ✓ PASS - Unknown fields preserved in round-trip")
    else:
        print("  ✗ FAIL - Unknown fields lost in round-trip")
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Test 2 complete"]
    }


def test_graceful_degradation_agent(state: ForwardCompatibilityState) -> ForwardCompatibilityState:
    """Test graceful degradation with missing features"""
    print("\n🧪 Test 3: Graceful Degradation...")
    
    server = NewServer()
    client = OldClient()
    
    # Even with new server features, old client should work
    data = server.get_user(456)
    parsed = client.parse_user_data(data)
    
    # Old client can still perform basic operations
    can_read_id = "id" in parsed
    can_read_name = "name" in parsed
    can_read_email = "email" in parsed
    
    test_result = {
        "test": "Graceful Degradation",
        "status": "PASS" if can_read_id and can_read_name and can_read_email else "FAIL",
        "details": "Old client can still perform basic operations",
        "can_read_basic_fields": all([can_read_id, can_read_name, can_read_email])
    }
    
    if test_result["status"] == "PASS":
        print("  ✓ PASS - Old client works with new server")
    else:
        print("  ✗ FAIL - Old client broken by new server")
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Test 3 complete"]
    }


def calculate_compatibility_score_agent(state: ForwardCompatibilityState) -> ForwardCompatibilityState:
    """Calculate forward compatibility score"""
    print("\n📊 Calculating Forward Compatibility Score...")
    
    total_tests = len(state["test_results"])
    passed_tests = sum(1 for t in state["test_results"] if t["status"] == "PASS")
    
    score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Score: {score:.1f}%")
    
    if score == 100:
        print("  Rating: Excellent forward compatibility! ✨")
    elif score >= 75:
        print("  Rating: Good forward compatibility ✓")
    else:
        print("  Rating: Needs improvement ⚠️")
    
    return {
        **state,
        "compatibility_score": score,
        "messages": [f"✓ Compatibility score: {score:.1f}%"]
    }


def generate_forward_compatibility_report_agent(state: ForwardCompatibilityState) -> ForwardCompatibilityState:
    """Generate forward compatibility report"""
    print("\n" + "="*70)
    print("FORWARD COMPATIBILITY REPORT")
    print("="*70)
    
    print(f"\n🎯 Forward Compatibility Score: {state['compatibility_score']:.1f}%")
    
    print(f"\n📋 Test Results:")
    for i, test in enumerate(state["test_results"], 1):
        status_icon = "✓" if test["status"] == "PASS" else "✗"
        print(f"\n  {status_icon} {i}. {test['test']}")
        print(f"      Status: {test['status']}")
        print(f"      Details: {test['details']}")
    
    print("\n💡 Forward Compatibility Best Practices:")
    print("  • Ignore unknown fields in incoming data")
    print("  • Preserve unknown fields when updating (round-trip)")
    print("  • Provide default values for missing expected fields")
    print("  • Don't fail on unexpected data structures")
    print("  • Use optional/nullable types for new fields")
    print("  • Test old clients against new servers regularly")
    
    print("\n📚 Example Strategies:")
    print("  1. Must-Ignore: Clients must ignore unknown fields")
    print("  2. Must-Understand: Clients must preserve unknown fields")
    print("  3. Graceful Degradation: Work with subset of features")
    print("  4. Feature Detection: Check for new features before using")
    
    print("\n" + "="*70)
    if state["compatibility_score"] == 100:
        print("✅ Forward Compatibility VERIFIED!")
    else:
        failed = sum(1 for t in state["test_results"] if t["status"] == "FAIL")
        print(f"⚠️ Forward Compatibility Issues - {failed} tests failed")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Forward compatibility report generated"]
    }


# Create the graph
def create_forward_compatibility_graph():
    """Create the forward compatibility workflow graph"""
    workflow = StateGraph(ForwardCompatibilityState)
    
    # Add nodes
    workflow.add_node("test_ignore_unknown", test_ignore_unknown_fields_agent)
    workflow.add_node("test_round_trip", test_round_trip_compatibility_agent)
    workflow.add_node("test_degradation", test_graceful_degradation_agent)
    workflow.add_node("calculate_score", calculate_compatibility_score_agent)
    workflow.add_node("generate_report", generate_forward_compatibility_report_agent)
    
    # Add edges
    workflow.add_edge(START, "test_ignore_unknown")
    workflow.add_edge("test_ignore_unknown", "test_round_trip")
    workflow.add_edge("test_round_trip", "test_degradation")
    workflow.add_edge("test_degradation", "calculate_score")
    workflow.add_edge("calculate_score", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 233: Forward Compatibility MCP Pattern")
    print("="*70)
    print("\nForward Compatibility: Old code handles new data gracefully")
    print("Don't break when servers are updated!")
    
    # Create and run the workflow
    app = create_forward_compatibility_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "test_results": [],
        "compatibility_score": 0.0
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Forward Compatibility Pattern Complete!")


if __name__ == "__main__":
    main()

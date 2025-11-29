"""
Pattern 225: Stubbing MCP Pattern

This pattern demonstrates stubbing external dependencies to provide
controlled responses during testing without requiring actual external services.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class StubbingState(TypedDict):
    """State for stubbing pattern workflow"""
    messages: Annotated[List[str], add]
    stub_results: List[dict]
    test_results: List[dict]


# Stub for external API
class WeatherAPIStub:
    """Stub for external weather API"""
    
    def __init__(self):
        """Initialize stub with predefined responses"""
        self.call_count = 0
        self.predefined_responses = {
            "New York": {"temperature": 72, "condition": "Sunny", "humidity": 65},
            "London": {"temperature": 58, "condition": "Cloudy", "humidity": 80},
            "Tokyo": {"temperature": 68, "condition": "Rainy", "humidity": 75},
            "Default": {"temperature": 70, "condition": "Clear", "humidity": 60}
        }
    
    def get_weather(self, city: str) -> dict:
        """
        Stub method that returns predefined weather data
        No actual API calls are made
        """
        self.call_count += 1
        response = self.predefined_responses.get(city, self.predefined_responses["Default"])
        return {**response, "city": city}
    
    def get_call_count(self) -> int:
        """Return number of times stub was called"""
        return self.call_count


class DatabaseStub:
    """Stub for database operations"""
    
    def __init__(self):
        """Initialize stub with in-memory storage"""
        self.storage = {}
        self.call_log = []
    
    def save(self, key: str, value: dict) -> bool:
        """Stub save operation"""
        self.call_log.append(("save", key, value))
        self.storage[key] = value
        return True
    
    def get(self, key: str) -> dict:
        """Stub get operation"""
        self.call_log.append(("get", key))
        return self.storage.get(key, {})
    
    def delete(self, key: str) -> bool:
        """Stub delete operation"""
        self.call_log.append(("delete", key))
        if key in self.storage:
            del self.storage[key]
            return True
        return False
    
    def get_call_log(self) -> List[tuple]:
        """Return log of all calls made to stub"""
        return self.call_log


class EmailServiceStub:
    """Stub for email service"""
    
    def __init__(self):
        """Initialize stub"""
        self.sent_emails = []
        self.should_fail = False
    
    def send_email(self, to: str, subject: str, body: str) -> bool:
        """Stub email sending"""
        if self.should_fail:
            return False
        
        self.sent_emails.append({
            "to": to,
            "subject": subject,
            "body": body,
            "timestamp": "2024-01-01 12:00:00"  # Fixed timestamp for testing
        })
        return True
    
    def set_failure_mode(self, should_fail: bool):
        """Configure stub to simulate failures"""
        self.should_fail = should_fail
    
    def get_sent_emails(self) -> List[dict]:
        """Return all emails sent through stub"""
        return self.sent_emails


# Agent functions
def create_stubs_agent(state: StubbingState) -> StubbingState:
    """Agent that creates and configures stubs"""
    print("\n🔧 Creating Stubs...")
    
    # Create various stubs
    weather_stub = WeatherAPIStub()
    db_stub = DatabaseStub()
    email_stub = EmailServiceStub()
    
    stub_results = [
        {
            "stub_name": "WeatherAPIStub",
            "type": "External API",
            "purpose": "Provide predictable weather data for testing",
            "cities_configured": len(weather_stub.predefined_responses),
            "status": "Ready"
        },
        {
            "stub_name": "DatabaseStub",
            "type": "Database",
            "purpose": "In-memory storage for testing",
            "operations": ["save", "get", "delete"],
            "status": "Ready"
        },
        {
            "stub_name": "EmailServiceStub",
            "type": "Email Service",
            "purpose": "Track email sending without actual sends",
            "failure_mode": "Configurable",
            "status": "Ready"
        }
    ]
    
    return {
        **state,
        "stub_results": stub_results,
        "messages": [f"✓ Created {len(stub_results)} stubs"]
    }


def test_with_stubs_agent(state: StubbingState) -> StubbingState:
    """Agent that runs tests using stubs"""
    print("\n🧪 Running Tests with Stubs...")
    
    test_results = []
    
    # Test 1: Weather API Stub
    print("\n  Testing WeatherAPIStub...")
    weather_stub = WeatherAPIStub()
    
    # Test various cities
    test_cities = ["New York", "London", "Tokyo", "Paris"]
    for city in test_cities:
        result = weather_stub.get_weather(city)
        test_results.append({
            "test": "Weather API Call",
            "city": city,
            "temperature": result["temperature"],
            "condition": result["condition"],
            "status": "PASSED",
            "note": "Stub returned predictable data"
        })
    
    # Verify call count
    call_count_test = {
        "test": "Weather API Call Count",
        "expected_calls": len(test_cities),
        "actual_calls": weather_stub.get_call_count(),
        "status": "PASSED" if weather_stub.get_call_count() == len(test_cities) else "FAILED"
    }
    test_results.append(call_count_test)
    
    # Test 2: Database Stub
    print("\n  Testing DatabaseStub...")
    db_stub = DatabaseStub()
    
    # Test save and get
    test_data = {"name": "John Doe", "age": 30, "role": "Developer"}
    db_stub.save("user:1", test_data)
    retrieved = db_stub.get("user:1")
    
    test_results.append({
        "test": "Database Save and Get",
        "expected": test_data,
        "actual": retrieved,
        "status": "PASSED" if retrieved == test_data else "FAILED",
        "note": "Stub provided in-memory storage"
    })
    
    # Test delete
    delete_success = db_stub.delete("user:1")
    after_delete = db_stub.get("user:1")
    
    test_results.append({
        "test": "Database Delete",
        "delete_success": delete_success,
        "data_after_delete": after_delete,
        "status": "PASSED" if after_delete == {} else "FAILED",
        "note": "Stub simulated deletion"
    })
    
    # Test 3: Email Service Stub
    print("\n  Testing EmailServiceStub...")
    email_stub = EmailServiceStub()
    
    # Test successful email
    success = email_stub.send_email(
        to="test@example.com",
        subject="Test Subject",
        body="Test body content"
    )
    
    test_results.append({
        "test": "Email Send Success",
        "send_result": success,
        "emails_sent": len(email_stub.get_sent_emails()),
        "status": "PASSED" if success and len(email_stub.get_sent_emails()) == 1 else "FAILED",
        "note": "Stub tracked email without sending"
    })
    
    # Test failure mode
    email_stub.set_failure_mode(True)
    failure_result = email_stub.send_email(
        to="test2@example.com",
        subject="Should Fail",
        body="This should fail"
    )
    
    test_results.append({
        "test": "Email Send Failure Mode",
        "send_result": failure_result,
        "status": "PASSED" if not failure_result else "FAILED",
        "note": "Stub simulated failure"
    })
    
    return {
        **state,
        "test_results": test_results,
        "messages": [f"✓ Completed {len(test_results)} tests using stubs"]
    }


def verify_stub_benefits_agent(state: StubbingState) -> StubbingState:
    """Agent that verifies benefits of using stubs"""
    print("\n✅ Verifying Stub Benefits...")
    
    benefits = [
        "✓ Fast execution - no network calls",
        "✓ Predictable responses - always same data",
        "✓ No external dependencies - works offline",
        "✓ Easy to test edge cases",
        "✓ Call tracking - verify interactions",
        "✓ Configurable failures - test error handling"
    ]
    
    print("\nBenefits of Stubbing:")
    for benefit in benefits:
        print(f"  {benefit}")
    
    return {
        **state,
        "messages": ["✓ Verified stub benefits"]
    }


def generate_stub_report_agent(state: StubbingState) -> StubbingState:
    """Agent that generates stubbing report"""
    print("\n" + "="*70)
    print("STUBBING PATTERN REPORT")
    print("="*70)
    
    print(f"\nStubs Created: {len(state['stub_results'])}")
    print("\nStub Details:")
    for stub in state["stub_results"]:
        print(f"\n  📌 {stub['stub_name']}")
        print(f"     Type: {stub['type']}")
        print(f"     Purpose: {stub['purpose']}")
        print(f"     Status: {stub['status']}")
    
    print(f"\n\nTests Run: {len(state['test_results'])}")
    passed = sum(1 for t in state["test_results"] if t.get("status") == "PASSED")
    failed = sum(1 for t in state["test_results"] if t.get("status") == "FAILED")
    
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    
    print("\nTest Results:")
    for i, test in enumerate(state["test_results"], 1):
        status_icon = "✓" if test.get("status") == "PASSED" else "✗"
        print(f"  {status_icon} {i}. {test['test']}: {test['status']}")
        if "note" in test:
            print(f"      Note: {test['note']}")
    
    print("\n" + "="*70)
    print(f"✅ All tests completed using stubs - {passed}/{len(state['test_results'])} passed")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Stubbing report generated"]
    }


# Create the graph
def create_stubbing_graph():
    """Create the stubbing pattern workflow graph"""
    workflow = StateGraph(StubbingState)
    
    # Add nodes
    workflow.add_node("create_stubs", create_stubs_agent)
    workflow.add_node("test_with_stubs", test_with_stubs_agent)
    workflow.add_node("verify_benefits", verify_stub_benefits_agent)
    workflow.add_node("generate_report", generate_stub_report_agent)
    
    # Add edges
    workflow.add_edge(START, "create_stubs")
    workflow.add_edge("create_stubs", "test_with_stubs")
    workflow.add_edge("test_with_stubs", "verify_benefits")
    workflow.add_edge("verify_benefits", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 225: Stubbing MCP Pattern")
    print("="*70)
    print("\nStubbing provides controlled, predictable responses from dependencies")
    print("Perfect for testing without requiring actual external services")
    
    # Create and run the workflow
    app = create_stubbing_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "stub_results": [],
        "test_results": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Stubbing Pattern Complete!")


if __name__ == "__main__":
    main()

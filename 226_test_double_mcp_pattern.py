"""
Pattern 226: Test Double MCP Pattern

This pattern demonstrates using test doubles (generic term for stubs, mocks, fakes, spies)
to replace real components during testing for isolation and control.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from abc import ABC, abstractmethod


# State definition
class TestDoubleState(TypedDict):
    """State for test double pattern workflow"""
    messages: Annotated[List[str], add]
    test_doubles: List[dict]
    test_results: List[dict]


# Real interface (what we're doubling)
class PaymentProcessor(ABC):
    """Abstract payment processor interface"""
    
    @abstractmethod
    def process_payment(self, amount: float, card_number: str) -> dict:
        """Process a payment"""
        pass
    
    @abstractmethod
    def refund_payment(self, transaction_id: str) -> bool:
        """Refund a payment"""
        pass


# Test Double implementations
class PaymentProcessorDummy(PaymentProcessor):
    """Dummy: Simplest double, just satisfies interface but does nothing useful"""
    
    def process_payment(self, amount: float, card_number: str) -> dict:
        """Returns empty dict, doesn't actually process"""
        return {}
    
    def refund_payment(self, transaction_id: str) -> bool:
        """Returns False, doesn't actually refund"""
        return False


class PaymentProcessorStub(PaymentProcessor):
    """Stub: Returns predefined responses"""
    
    def __init__(self):
        self.predefined_result = {
            "success": True,
            "transaction_id": "STUB-12345",
            "amount": 100.00
        }
    
    def process_payment(self, amount: float, card_number: str) -> dict:
        """Returns predefined success result"""
        return {**self.predefined_result, "amount": amount}
    
    def refund_payment(self, transaction_id: str) -> bool:
        """Always returns True"""
        return True


class PaymentProcessorSpy(PaymentProcessor):
    """Spy: Records how it was called"""
    
    def __init__(self):
        self.calls = []
        self.refund_calls = []
    
    def process_payment(self, amount: float, card_number: str) -> dict:
        """Records call and returns success"""
        self.calls.append({"amount": amount, "card_number": card_number})
        return {
            "success": True,
            "transaction_id": f"SPY-{len(self.calls)}",
            "amount": amount
        }
    
    def refund_payment(self, transaction_id: str) -> bool:
        """Records refund call"""
        self.refund_calls.append(transaction_id)
        return True
    
    def get_call_count(self) -> int:
        """Get number of times process_payment was called"""
        return len(self.calls)
    
    def was_called_with_amount(self, amount: float) -> bool:
        """Check if called with specific amount"""
        return any(call["amount"] == amount for call in self.calls)


class PaymentProcessorMock(PaymentProcessor):
    """Mock: Verifies it was called correctly (expectations)"""
    
    def __init__(self):
        self.expectations = []
        self.calls = []
    
    def expect_payment(self, amount: float, card_number: str):
        """Set expectation for payment call"""
        self.expectations.append({
            "method": "process_payment",
            "amount": amount,
            "card_number": card_number
        })
    
    def process_payment(self, amount: float, card_number: str) -> dict:
        """Process payment and record call"""
        self.calls.append({
            "method": "process_payment",
            "amount": amount,
            "card_number": card_number
        })
        return {
            "success": True,
            "transaction_id": f"MOCK-{len(self.calls)}",
            "amount": amount
        }
    
    def refund_payment(self, transaction_id: str) -> bool:
        """Refund payment"""
        self.calls.append({
            "method": "refund_payment",
            "transaction_id": transaction_id
        })
        return True
    
    def verify(self) -> bool:
        """Verify all expectations were met"""
        if len(self.expectations) != len(self.calls):
            return False
        
        for exp, call in zip(self.expectations, self.calls):
            if exp != call:
                return False
        return True


class PaymentProcessorFake(PaymentProcessor):
    """Fake: Working implementation but simpler (in-memory)"""
    
    def __init__(self):
        self.transactions = {}
        self.next_id = 1
    
    def process_payment(self, amount: float, card_number: str) -> dict:
        """Fake payment processing with in-memory storage"""
        transaction_id = f"FAKE-{self.next_id}"
        self.next_id += 1
        
        self.transactions[transaction_id] = {
            "amount": amount,
            "card_number": card_number,
            "status": "completed"
        }
        
        return {
            "success": True,
            "transaction_id": transaction_id,
            "amount": amount
        }
    
    def refund_payment(self, transaction_id: str) -> bool:
        """Fake refund processing"""
        if transaction_id in self.transactions:
            self.transactions[transaction_id]["status"] = "refunded"
            return True
        return False
    
    def get_transaction(self, transaction_id: str) -> dict:
        """Get transaction details"""
        return self.transactions.get(transaction_id, {})


# Agent functions
def create_test_doubles_agent(state: TestDoubleState) -> TestDoubleState:
    """Agent that creates various test doubles"""
    print("\n🎭 Creating Test Doubles...")
    
    test_doubles = [
        {
            "type": "Dummy",
            "purpose": "Satisfy interface, no real behavior",
            "use_case": "When parameter needed but not used",
            "complexity": "Simplest"
        },
        {
            "type": "Stub",
            "purpose": "Return predefined responses",
            "use_case": "When you need specific return values",
            "complexity": "Simple"
        },
        {
            "type": "Spy",
            "purpose": "Record how it was called",
            "use_case": "Verify method calls and arguments",
            "complexity": "Medium"
        },
        {
            "type": "Mock",
            "purpose": "Verify expectations were met",
            "use_case": "Assert specific behavior occurred",
            "complexity": "Medium-High"
        },
        {
            "type": "Fake",
            "purpose": "Working but simplified implementation",
            "use_case": "When you need realistic behavior",
            "complexity": "Highest"
        }
    ]
    
    return {
        **state,
        "test_doubles": test_doubles,
        "messages": [f"✓ Created {len(test_doubles)} types of test doubles"]
    }


def test_dummy_agent(state: TestDoubleState) -> TestDoubleState:
    """Test using a dummy"""
    print("\n🎪 Testing with Dummy...")
    
    dummy = PaymentProcessorDummy()
    result = dummy.process_payment(100.0, "1234-5678-9012-3456")
    
    test_result = {
        "double_type": "Dummy",
        "test": "Payment Processing",
        "expected": "Empty or minimal response",
        "actual": result,
        "status": "PASSED",
        "note": "Dummy satisfied interface but did nothing useful"
    }
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Dummy test complete"]
    }


def test_stub_agent(state: TestDoubleState) -> TestDoubleState:
    """Test using a stub"""
    print("\n🎪 Testing with Stub...")
    
    stub = PaymentProcessorStub()
    result = stub.process_payment(250.0, "1234-5678-9012-3456")
    
    test_result = {
        "double_type": "Stub",
        "test": "Payment Processing",
        "expected": "Predefined success response",
        "actual": result,
        "status": "PASSED" if result["success"] else "FAILED",
        "note": "Stub returned predictable response"
    }
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Stub test complete"]
    }


def test_spy_agent(state: TestDoubleState) -> TestDoubleState:
    """Test using a spy"""
    print("\n🎪 Testing with Spy...")
    
    spy = PaymentProcessorSpy()
    
    # Make several calls
    spy.process_payment(100.0, "1111-2222-3333-4444")
    spy.process_payment(200.0, "5555-6666-7777-8888")
    spy.process_payment(100.0, "9999-0000-1111-2222")
    
    # Verify spy recorded calls
    call_count = spy.get_call_count()
    was_called_with_100 = spy.was_called_with_amount(100.0)
    
    test_result = {
        "double_type": "Spy",
        "test": "Call Recording",
        "expected_calls": 3,
        "actual_calls": call_count,
        "verified_amount_100": was_called_with_100,
        "status": "PASSED" if call_count == 3 and was_called_with_100 else "FAILED",
        "note": "Spy recorded all method calls"
    }
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Spy test complete"]
    }


def test_mock_agent(state: TestDoubleState) -> TestDoubleState:
    """Test using a mock"""
    print("\n🎪 Testing with Mock...")
    
    mock = PaymentProcessorMock()
    
    # Set expectations
    mock.expect_payment(150.0, "1234-5678-9012-3456")
    
    # Make call that meets expectation
    mock.process_payment(150.0, "1234-5678-9012-3456")
    
    # Verify expectations met
    verified = mock.verify()
    
    test_result = {
        "double_type": "Mock",
        "test": "Expectation Verification",
        "expectations_met": verified,
        "status": "PASSED" if verified else "FAILED",
        "note": "Mock verified all expectations were met"
    }
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Mock test complete"]
    }


def test_fake_agent(state: TestDoubleState) -> TestDoubleState:
    """Test using a fake"""
    print("\n🎪 Testing with Fake...")
    
    fake = PaymentProcessorFake()
    
    # Process payment
    result = fake.process_payment(300.0, "1234-5678-9012-3456")
    transaction_id = result["transaction_id"]
    
    # Verify transaction was stored
    transaction = fake.get_transaction(transaction_id)
    
    # Test refund
    refund_result = fake.refund_payment(transaction_id)
    transaction_after_refund = fake.get_transaction(transaction_id)
    
    test_result = {
        "double_type": "Fake",
        "test": "Payment and Refund",
        "payment_success": result["success"],
        "transaction_stored": transaction != {},
        "refund_success": refund_result,
        "status_after_refund": transaction_after_refund.get("status"),
        "status": "PASSED" if transaction_after_refund.get("status") == "refunded" else "FAILED",
        "note": "Fake provided realistic behavior with in-memory storage"
    }
    
    return {
        **state,
        "test_results": state.get("test_results", []) + [test_result],
        "messages": ["✓ Fake test complete"]
    }


def generate_comparison_report_agent(state: TestDoubleState) -> TestDoubleState:
    """Generate comparison report of all test doubles"""
    print("\n" + "="*70)
    print("TEST DOUBLE PATTERN COMPARISON REPORT")
    print("="*70)
    
    print("\n📚 Test Double Types:")
    for double in state["test_doubles"]:
        print(f"\n  🎭 {double['type']}")
        print(f"     Purpose: {double['purpose']}")
        print(f"     Use Case: {double['use_case']}")
        print(f"     Complexity: {double['complexity']}")
    
    print(f"\n\n🧪 Tests Run: {len(state['test_results'])}")
    
    print("\n📊 Test Results:")
    for i, result in enumerate(state["test_results"], 1):
        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        print(f"\n  {status_icon} {i}. {result['double_type']} - {result['test']}")
        print(f"      Status: {result['status']}")
        print(f"      Note: {result['note']}")
    
    print("\n" + "="*70)
    print("✅ All Test Double Types Demonstrated Successfully!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Comparison report generated"]
    }


# Create the graph
def create_test_double_graph():
    """Create the test double pattern workflow graph"""
    workflow = StateGraph(TestDoubleState)
    
    # Add nodes
    workflow.add_node("create_doubles", create_test_doubles_agent)
    workflow.add_node("test_dummy", test_dummy_agent)
    workflow.add_node("test_stub", test_stub_agent)
    workflow.add_node("test_spy", test_spy_agent)
    workflow.add_node("test_mock", test_mock_agent)
    workflow.add_node("test_fake", test_fake_agent)
    workflow.add_node("generate_report", generate_comparison_report_agent)
    
    # Add edges
    workflow.add_edge(START, "create_doubles")
    workflow.add_edge("create_doubles", "test_dummy")
    workflow.add_edge("test_dummy", "test_stub")
    workflow.add_edge("test_stub", "test_spy")
    workflow.add_edge("test_spy", "test_mock")
    workflow.add_edge("test_mock", "test_fake")
    workflow.add_edge("test_fake", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 226: Test Double MCP Pattern")
    print("="*70)
    print("\nTest Doubles: Dummy, Stub, Spy, Mock, Fake")
    print("Each serves a different testing purpose!")
    
    # Create and run the workflow
    app = create_test_double_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "test_doubles": [],
        "test_results": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Test Double Pattern Complete!")


if __name__ == "__main__":
    main()

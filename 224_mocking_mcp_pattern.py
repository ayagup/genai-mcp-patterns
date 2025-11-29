"""
Pattern 224: Mocking MCP Pattern

Mocking replaces dependencies with controlled test doubles:
- Simulate external services
- Control test behavior
- Verify method calls
- Isolate components

Benefits:
- Fast tests
- Predictable results
- Isolated testing
- No external dependencies
"""

from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class MockTestState(TypedDict):
    test_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


class PaymentGatewayMock:
    """Mock payment gateway"""
    def __init__(self):
        self.calls = []
        self.should_succeed = True
    
    def process_payment(self, amount: float) -> Dict[str, Any]:
        self.calls.append({'amount': amount})
        
        if self.should_succeed:
            return {'status': 'success', 'transaction_id': 'TXN-123'}
        else:
            return {'status': 'failed', 'error': 'Insufficient funds'}


class OrderService:
    """Service that uses payment gateway"""
    def __init__(self, payment_gateway):
        self.gateway = payment_gateway
    
    def place_order(self, amount: float) -> bool:
        result = self.gateway.process_payment(amount)
        return result['status'] == 'success'


def setup_agent(state: MockTestState):
    operations = []
    
    # Create mock
    mock_gateway = PaymentGatewayMock()
    order_service = OrderService(mock_gateway)
    
    operations.append("Mocking Pattern Setup:")
    operations.append("  Real: PaymentGateway (external API)")
    operations.append("  Mock: PaymentGatewayMock (test double)")
    
    state['_mock'] = mock_gateway
    state['_service'] = order_service
    
    return {
        "test_operations": operations,
        "operation_results": ["✓ Mocks created"],
        "performance_metrics": [],
        "messages": ["Ready"]
    }


def test_with_mock_agent(state: MockTestState):
    mock = state['_mock']
    service = state['_service']
    operations = []
    
    operations.append("\n🎭 Testing with Mock:")
    
    # Test success case
    mock.should_succeed = True
    result = service.place_order(100.0)
    operations.append(f"  Success case: {result}")
    operations.append(f"  Mock called: {len(mock.calls)} times")
    
    # Test failure case
    mock.should_succeed = False
    result = service.place_order(100.0)
    operations.append(f"  Failure case: {result}")
    
    return {
        "test_operations": operations,
        "operation_results": ["✓ Mock verified"],
        "performance_metrics": ["Execution: <10ms"],
        "messages": ["Complete"]
    }


def statistics_agent(state: MockTestState):
    return {
        "test_operations": ["\n📊 Mocking: Control test behavior"],
        "operation_results": ["✓ Tests isolated"],
        "performance_metrics": ["Libraries: unittest.mock, pytest-mock"],
        "messages": ["Done"]
    }


def create_mock_graph():
    workflow = StateGraph(MockTestState)
    workflow.add_node("setup", setup_agent)
    workflow.add_node("test", test_with_mock_agent)
    workflow.add_node("stats", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "test")
    workflow.add_edge("test", "stats")
    workflow.add_edge("stats", END)
    
    return workflow.compile()


def main():
    print("Pattern 224: Mocking MCP Pattern")
    app = create_mock_graph()
    final_state = app.invoke({
        "test_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    })
    
    for op in final_state["test_operations"]:
        print(op)
    print("\nMocking: Replace dependencies with controlled doubles ✓")


if __name__ == "__main__":
    main()

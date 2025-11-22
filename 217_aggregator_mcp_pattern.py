"""
Pattern 217: Aggregator MCP Pattern

Aggregator combines multiple service calls into one response:
- Calls multiple microservices in parallel
- Aggregates results into single response
- Reduces client-server round trips
- Improves performance
- Simplifies client logic

Use Cases:
- Dashboard data aggregation
- Product page (product + reviews + inventory)
- User profile (user + posts + followers)
- Search results from multiple sources
- Report generation

Benefits:
- Reduced network calls
- Better performance
- Simplified client code
- Centralized orchestration
- Error handling in one place
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class AggregatorState(TypedDict):
    """State for aggregator pattern operations"""
    aggregator_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


class UserService:
    def get_user(self, user_id: str) -> Dict[str, Any]:
        time.sleep(0.02)  # Simulate latency
        return {'id': user_id, 'name': 'John Doe', 'email': 'john@example.com'}


class OrderService:
    def get_orders(self, user_id: str) -> List[Dict[str, Any]]:
        time.sleep(0.03)
        return [
            {'id': 'ORD-1', 'total': 150.00, 'status': 'delivered'},
            {'id': 'ORD-2', 'total': 75.50, 'status': 'shipped'}
        ]


class PaymentService:
    def get_payment_methods(self, user_id: str) -> List[Dict[str, Any]]:
        time.sleep(0.02)
        return [
            {'type': 'credit_card', 'last4': '1234'},
            {'type': 'paypal', 'email': 'john@paypal.com'}
        ]


class AggregatorService:
    """
    Aggregator that calls multiple services and combines results
    """
    
    def __init__(self):
        self.user_service = UserService()
        self.order_service = OrderService()
        self.payment_service = PaymentService()
        
        self.aggregation_count = 0
    
    def get_user_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Aggregate data from multiple services"""
        start_time = time.time()
        self.aggregation_count += 1
        
        # Call all services (in parallel in real implementation)
        user = self.user_service.get_user(user_id)
        orders = self.order_service.get_orders(user_id)
        payments = self.payment_service.get_payment_methods(user_id)
        
        # Aggregate results
        aggregated = {
            'user': user,
            'orders': {
                'total_count': len(orders),
                'recent_orders': orders[:5],
                'total_spent': sum(o['total'] for o in orders)
            },
            'payment_methods': payments,
            'aggregation_time_ms': (time.time() - start_time) * 1000
        }
        
        return aggregated


def setup_agent(state: AggregatorState):
    """Agent to set up aggregator"""
    operations = []
    results = []
    
    aggregator = AggregatorService()
    
    operations.append("Aggregator Pattern Setup:")
    operations.append("\nMicroservices:")
    operations.append("  - UserService: User profile data")
    operations.append("  - OrderService: Order history")
    operations.append("  - PaymentService: Payment methods")
    
    operations.append("\nAggregator: Combines all services into dashboard")
    
    results.append("✓ Aggregator initialized")
    
    state['_aggregator'] = aggregator
    
    return {
        "aggregator_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def aggregation_demo_agent(state: AggregatorState):
    """Agent to demonstrate aggregation"""
    aggregator = state['_aggregator']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🔄 Aggregation Demo:")
    
    operations.append("\nClient makes ONE request to Aggregator")
    operations.append("Aggregator calls 3 services and combines results")
    
    dashboard = aggregator.get_user_dashboard("USER-123")
    
    operations.append("\nAggregated Response:")
    operations.append(f"  User: {dashboard['user']['name']}")
    operations.append(f"  Orders: {dashboard['orders']['total_count']} orders, ${dashboard['orders']['total_spent']:.2f} spent")
    operations.append(f"  Payment Methods: {len(dashboard['payment_methods'])} configured")
    operations.append(f"  Aggregation Time: {dashboard['aggregation_time_ms']:.1f}ms")
    
    operations.append("\nWithout Aggregator:")
    operations.append("  Client → UserService (1 call)")
    operations.append("  Client → OrderService (1 call)")
    operations.append("  Client → PaymentService (1 call)")
    operations.append("  Total: 3 round trips")
    
    operations.append("\nWith Aggregator:")
    operations.append("  Client → Aggregator (1 call)")
    operations.append("  Aggregator → All services (parallel)")
    operations.append("  Total: 1 round trip")
    
    metrics.append(f"Round trips saved: 2 (67% reduction)")
    results.append("✓ Aggregation successful")
    
    return {
        "aggregator_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Aggregation complete"]
    }


def statistics_agent(state: AggregatorState):
    """Agent to show statistics"""
    aggregator = state['_aggregator']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("AGGREGATOR STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nAggregations performed: {aggregator.aggregation_count}")
    
    metrics.append("\n📊 Aggregator Benefits:")
    metrics.append("  ✓ Reduced network calls")
    metrics.append("  ✓ Better performance")
    metrics.append("  ✓ Simplified client code")
    metrics.append("  ✓ Parallel service calls")
    metrics.append("  ✓ Centralized error handling")
    
    results.append("✓ Aggregator pattern demonstrated")
    
    return {
        "aggregator_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_aggregator_graph():
    """Create the aggregator workflow graph"""
    workflow = StateGraph(AggregatorState)
    
    workflow.add_node("setup", setup_agent)
    workflow.add_node("aggregate", aggregation_demo_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "aggregate")
    workflow.add_edge("aggregate", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 217: Aggregator MCP Pattern")
    print("=" * 80)
    
    app = create_aggregator_graph()
    initial_state = {
        "aggregator_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    
    for op in final_state["aggregator_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Aggregator: Combines multiple service calls

Benefits:
✓ 1 call instead of N calls
✓ Parallel service execution
✓ Reduced latency
✓ Simplified client
✓ Better error handling

Real-World:
- GraphQL (field-level aggregation)
- Netflix Falcor
- BFF pattern often includes aggregation
""")


if __name__ == "__main__":
    main()

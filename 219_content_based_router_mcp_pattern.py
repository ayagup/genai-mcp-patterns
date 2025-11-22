"""
Pattern 219: Content-Based Router MCP Pattern

Content-Based Router routes messages based on content/metadata:
- Inspects message content
- Routes to appropriate destination
- Enables dynamic routing
- Supports complex routing rules
- Decouples sender from receiver

Routing Criteria:
- Message type
- Header values
- Payload content
- Priority level
- Customer tier

Benefits:
- Flexible routing logic
- Easy to add new destinations
- Rule-based configuration
- Decoupled components
- Dynamic behavior

Use Cases:
- Order routing by product type
- Customer support ticket routing
- Log message routing
- Event processing
- Workflow orchestration
"""

from typing import TypedDict, Annotated, List, Dict, Any, Callable
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class RouterState(TypedDict):
    """State for content-based router operations"""
    router_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class Message:
    """Message to be routed"""
    id: str
    type: str
    priority: str
    content: Dict[str, Any]
    headers: Dict[str, str]


class Destination:
    """Message destination"""
    
    def __init__(self, name: str):
        self.name = name
        self.received_messages: List[Message] = []
    
    def deliver(self, message: Message):
        """Deliver message to this destination"""
        self.received_messages.append(message)


class RoutingRule:
    """Routing rule with condition and destination"""
    
    def __init__(self, name: str, condition: Callable[[Message], bool], destination: Destination):
        self.name = name
        self.condition = condition
        self.destination = destination
        self.match_count = 0
    
    def matches(self, message: Message) -> bool:
        """Check if message matches this rule"""
        return self.condition(message)
    
    def route(self, message: Message):
        """Route message to destination"""
        self.match_count += 1
        self.destination.deliver(message)


class ContentBasedRouter:
    """
    Content-Based Router that routes messages based on content
    """
    
    def __init__(self):
        self.rules: List[RoutingRule] = []
        self.default_destination: Destination = None
        
        self.total_messages = 0
        self.routed_messages = 0
        self.default_routed = 0
    
    def add_rule(self, rule: RoutingRule):
        """Add routing rule"""
        self.rules.append(rule)
    
    def set_default_destination(self, destination: Destination):
        """Set default destination for unmatched messages"""
        self.default_destination = destination
    
    def route(self, message: Message) -> str:
        """Route message based on content"""
        self.total_messages += 1
        
        # Check rules in order
        for rule in self.rules:
            if rule.matches(message):
                rule.route(message)
                self.routed_messages += 1
                return rule.destination.name
        
        # No rule matched - use default
        if self.default_destination:
            self.default_destination.deliver(message)
            self.default_routed += 1
            return self.default_destination.name
        
        return "UNROUTED"


def setup_router_agent(state: RouterState):
    """Agent to set up router"""
    operations = []
    results = []
    
    router = ContentBasedRouter()
    
    # Create destinations
    high_priority_queue = Destination("HighPriorityQueue")
    order_service = Destination("OrderService")
    support_service = Destination("SupportService")
    notification_service = Destination("NotificationService")
    dead_letter_queue = Destination("DeadLetterQueue")
    
    # Define routing rules
    
    # Rule 1: High priority messages
    router.add_rule(RoutingRule(
        "HighPriority",
        lambda msg: msg.priority == "high",
        high_priority_queue
    ))
    
    # Rule 2: Order messages
    router.add_rule(RoutingRule(
        "Orders",
        lambda msg: msg.type == "order",
        order_service
    ))
    
    # Rule 3: Support tickets
    router.add_rule(RoutingRule(
        "Support",
        lambda msg: msg.type == "support_ticket",
        support_service
    ))
    
    # Rule 4: Notifications
    router.add_rule(RoutingRule(
        "Notifications",
        lambda msg: msg.type == "notification",
        notification_service
    ))
    
    # Default destination
    router.set_default_destination(dead_letter_queue)
    
    operations.append("Content-Based Router Setup:")
    operations.append("\nRouting Rules:")
    operations.append("  1. HighPriority: priority='high' → HighPriorityQueue")
    operations.append("  2. Orders: type='order' → OrderService")
    operations.append("  3. Support: type='support_ticket' → SupportService")
    operations.append("  4. Notifications: type='notification' → NotificationService")
    operations.append("  Default: * → DeadLetterQueue")
    
    results.append(f"✓ Router configured with {len(router.rules)} rules")
    
    # Store in state
    state['_router'] = router
    state['_destinations'] = {
        'high_priority': high_priority_queue,
        'order': order_service,
        'support': support_service,
        'notification': notification_service,
        'dlq': dead_letter_queue
    }
    
    return {
        "router_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def routing_demo_agent(state: RouterState):
    """Agent to demonstrate routing"""
    router = state['_router']
    operations = []
    results = []
    
    operations.append("\n📨 Message Routing Demo:")
    
    # Create test messages
    messages = [
        Message("MSG-1", "order", "low", {"product": "laptop", "amount": 999}, {}),
        Message("MSG-2", "support_ticket", "high", {"issue": "billing"}, {}),
        Message("MSG-3", "notification", "low", {"type": "email"}, {}),
        Message("MSG-4", "payment", "high", {"amount": 500}, {}),
        Message("MSG-5", "unknown", "low", {"data": "test"}, {}),
    ]
    
    operations.append("\nRouting 5 messages:")
    
    for msg in messages:
        destination = router.route(msg)
        operations.append(f"  {msg.id} (type={msg.type}, priority={msg.priority}) → {destination}")
    
    results.append("✓ All messages routed successfully")
    
    return {
        "router_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Routing demo complete"]
    }


def destination_analysis_agent(state: RouterState):
    """Agent to analyze destinations"""
    destinations = state['_destinations']
    operations = []
    results = []
    
    operations.append("\n📊 Destination Analysis:")
    
    operations.append("\nMessages per destination:")
    for name, dest in destinations.items():
        count = len(dest.received_messages)
        operations.append(f"  {dest.name}: {count} messages")
        
        if count > 0:
            types = [msg.type for msg in dest.received_messages]
            operations.append(f"    Types: {', '.join(set(types))}")
    
    results.append("✓ Destination analysis complete")
    
    return {
        "router_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Analysis complete"]
    }


def statistics_agent(state: RouterState):
    """Agent to show statistics"""
    router = state['_router']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("ROUTER STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nTotal messages: {router.total_messages}")
    operations.append(f"Routed by rules: {router.routed_messages}")
    operations.append(f"Routed to default: {router.default_routed}")
    
    operations.append("\nRule Match Counts:")
    for rule in router.rules:
        operations.append(f"  {rule.name}: {rule.match_count} matches")
    
    metrics.append("\n📊 Content-Based Router Benefits:")
    metrics.append("  ✓ Dynamic routing logic")
    metrics.append("  ✓ Decoupled sender/receiver")
    metrics.append("  ✓ Easy to add destinations")
    metrics.append("  ✓ Rule-based configuration")
    metrics.append("  ✓ Content inspection")
    
    results.append("✓ Content-Based Router demonstrated")
    
    return {
        "router_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_router_graph():
    """Create the router workflow graph"""
    workflow = StateGraph(RouterState)
    
    workflow.add_node("setup", setup_router_agent)
    workflow.add_node("routing", routing_demo_agent)
    workflow.add_node("analysis", destination_analysis_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "routing")
    workflow.add_edge("routing", "analysis")
    workflow.add_edge("analysis", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 219: Content-Based Router MCP Pattern")
    print("=" * 80)
    
    app = create_router_graph()
    initial_state = {
        "router_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    
    for op in final_state["router_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Content-Based Router: Route by message content

Routing Criteria:
- Message type
- Priority level
- Header values
- Payload content
- Custom rules

Benefits:
✓ Dynamic routing
✓ Decoupled components
✓ Easy to extend
✓ Rule-based logic
✓ Flexible routing

Real-World:
- Apache Camel routing
- AWS EventBridge rules
- Message queue routing
- API Gateway routing
""")


if __name__ == "__main__":
    main()

"""
Pattern 179: Microservices MCP Pattern

This pattern demonstrates microservices architecture where the application is
decomposed into small, independently deployable services that communicate via
APIs. Each microservice owns its data, has its own lifecycle, and can be
developed, deployed, and scaled independently.

Key Concepts:
1. Microservice: Small, focused service with single responsibility
2. Independence: Each service independently deployable
3. Decentralized Data: Each service has own database
4. API Communication: Services communicate via REST, gRPC, or messaging
5. Service Discovery: Services find each other dynamically
6. Distributed System: Multiple processes, often on different machines
7. Polyglot: Services can use different technologies

Microservices Characteristics:
- Organized around business capabilities
- Independently deployable
- Decentralized governance
- Decentralized data management
- Infrastructure automation
- Design for failure
- Evolutionary design

Communication Patterns:
1. Synchronous: HTTP REST, gRPC (request-response)
2. Asynchronous: Message queues, event streaming
3. Service Mesh: Infrastructure layer for service-to-service communication
4. API Gateway: Single entry point for clients

Supporting Infrastructure:
- Service Discovery: Consul, Eureka
- API Gateway: Kong, NGINX
- Load Balancer: HAProxy, NGINX
- Message Broker: Kafka, RabbitMQ
- Container Orchestration: Kubernetes, Docker Swarm

Benefits:
- Independent Deployment: Deploy services separately
- Technology Diversity: Different services use different tech
- Scalability: Scale individual services
- Resilience: Service failure doesn't crash entire system
- Team Autonomy: Teams own specific services
- Faster Development: Parallel development

Trade-offs:
- Complexity: Distributed system challenges
- Network Latency: Service calls over network
- Data Consistency: Eventual consistency vs transactions
- Testing: Integration testing more complex
- Operations: More services to deploy and monitor
- Debugging: Distributed tracing needed

Use Cases:
- Large-scale applications: Netflix, Amazon, Uber
- E-commerce: Order, Payment, Inventory, Shipping services
- SaaS platforms: Multi-tenant isolated services
- Modernization: Break monolith into microservices
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from enum import Enum
import time

# Define the state for microservices architecture
class MicroservicesState(TypedDict):
    """State for microservices system"""
    request: str
    order_service_result: Optional[Dict[str, Any]]
    payment_service_result: Optional[Dict[str, Any]]
    inventory_service_result: Optional[Dict[str, Any]]
    notification_service_result: Optional[Dict[str, Any]]
    final_response: str
    service_calls: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# MICROSERVICE BASE
# ============================================================================

@dataclass
class ServiceMetadata:
    """Metadata for a microservice"""
    name: str
    version: str
    instance_id: str
    port: int
    endpoint: str
    health_endpoint: str
    database: str  # Own database

class ServiceStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class Microservice:
    """
    Base Microservice: Common functionality
    
    Each microservice:
    - Has own database (decentralized data)
    - Exposes REST API
    - Registers with service discovery
    - Implements health checks
    - Has independent deployment
    """
    
    def __init__(self, metadata: ServiceMetadata):
        self.metadata = metadata
        self.status = ServiceStatus.HEALTHY
        self.request_count = 0
        self.database = {}  # Simulated database (each service owns its data)
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            "service": self.metadata.name,
            "status": self.status.value,
            "instance": self.metadata.instance_id,
            "uptime": time.time(),
            "request_count": self.request_count
        }
    
    def call_service(self, service_name: str, endpoint: str, data: Any) -> Dict[str, Any]:
        """
        Call another microservice (service-to-service communication)
        
        In production:
        - Use service discovery to find service
        - Make HTTP/gRPC call
        - Handle retries and circuit breaking
        - Implement timeouts
        """
        return {
            "called_service": service_name,
            "endpoint": endpoint,
            "status": "success"
        }

# ============================================================================
# MICROSERVICE IMPLEMENTATIONS
# ============================================================================

class OrderService(Microservice):
    """
    Order Service: Manages customer orders
    
    Responsibilities:
    - Create orders
    - Update order status
    - Query orders
    
    Own Database: order_db
    API Endpoints:
    - POST /orders - Create order
    - GET /orders/{id} - Get order
    - PUT /orders/{id} - Update order
    """
    
    def __init__(self):
        metadata = ServiceMetadata(
            name="order-service",
            version="1.0.0",
            instance_id="order-001",
            port=8001,
            endpoint="http://order-service:8001",
            health_endpoint="/health",
            database="order_db"
        )
        super().__init__(metadata)
    
    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new order"""
        order_id = f"ORD-{len(self.database) + 1}"
        
        order = {
            "order_id": order_id,
            "customer": order_data.get("customer"),
            "items": order_data.get("items"),
            "total": order_data.get("total", 0),
            "status": "PENDING",
            "created_at": time.time()
        }
        
        # Save to own database
        self.database[order_id] = order
        self.request_count += 1
        
        return {
            "service": self.metadata.name,
            "order": order,
            "message": "Order created successfully"
        }

class PaymentService(Microservice):
    """
    Payment Service: Handles payments
    
    Responsibilities:
    - Process payments
    - Handle refunds
    - Payment status
    
    Own Database: payment_db
    API Endpoints:
    - POST /payments - Process payment
    - GET /payments/{id} - Get payment status
    - POST /refunds - Process refund
    """
    
    def __init__(self):
        metadata = ServiceMetadata(
            name="payment-service",
            version="2.0.0",
            instance_id="payment-001",
            port=8002,
            endpoint="http://payment-service:8002",
            health_endpoint="/health",
            database="payment_db"
        )
        super().__init__(metadata)
    
    def process_payment(self, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a payment"""
        payment_id = f"PAY-{len(self.database) + 1}"
        
        payment = {
            "payment_id": payment_id,
            "order_id": payment_data.get("order_id"),
            "amount": payment_data.get("amount"),
            "method": payment_data.get("method", "credit_card"),
            "status": "COMPLETED",
            "processed_at": time.time()
        }
        
        # Save to own database
        self.database[payment_id] = payment
        self.request_count += 1
        
        return {
            "service": self.metadata.name,
            "payment": payment,
            "message": "Payment processed successfully"
        }

class InventoryService(Microservice):
    """
    Inventory Service: Manages product inventory
    
    Responsibilities:
    - Check stock availability
    - Reserve items
    - Update inventory
    
    Own Database: inventory_db
    API Endpoints:
    - GET /inventory/{product_id} - Check stock
    - POST /inventory/reserve - Reserve items
    - PUT /inventory/{product_id} - Update stock
    """
    
    def __init__(self):
        metadata = ServiceMetadata(
            name="inventory-service",
            version="1.5.0",
            instance_id="inventory-001",
            port=8003,
            endpoint="http://inventory-service:8003",
            health_endpoint="/health",
            database="inventory_db"
        )
        super().__init__(metadata)
        
        # Initialize inventory
        self.database = {
            "PROD-001": {"product_id": "PROD-001", "stock": 100},
            "PROD-002": {"product_id": "PROD-002", "stock": 50}
        }
    
    def check_availability(self, product_id: str, quantity: int) -> Dict[str, Any]:
        """Check if product is available"""
        product = self.database.get(product_id, {})
        available_stock = product.get("stock", 0)
        
        is_available = available_stock >= quantity
        self.request_count += 1
        
        return {
            "service": self.metadata.name,
            "product_id": product_id,
            "requested_quantity": quantity,
            "available_stock": available_stock,
            "is_available": is_available
        }
    
    def reserve_items(self, product_id: str, quantity: int) -> Dict[str, Any]:
        """Reserve items"""
        if product_id in self.database:
            self.database[product_id]["stock"] -= quantity
            self.request_count += 1
            
            return {
                "service": self.metadata.name,
                "reserved": True,
                "remaining_stock": self.database[product_id]["stock"]
            }
        
        return {"service": self.metadata.name, "reserved": False}

class NotificationService(Microservice):
    """
    Notification Service: Sends notifications
    
    Responsibilities:
    - Send email notifications
    - Send SMS notifications
    - Push notifications
    
    Own Database: notification_db
    API Endpoints:
    - POST /notifications/email - Send email
    - POST /notifications/sms - Send SMS
    - GET /notifications/{id} - Get notification status
    """
    
    def __init__(self):
        metadata = ServiceMetadata(
            name="notification-service",
            version="1.2.0",
            instance_id="notification-001",
            port=8004,
            endpoint="http://notification-service:8004",
            health_endpoint="/health",
            database="notification_db"
        )
        super().__init__(metadata)
    
    def send_notification(self, notification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a notification"""
        notification_id = f"NOTIF-{len(self.database) + 1}"
        
        notification = {
            "notification_id": notification_id,
            "recipient": notification_data.get("recipient"),
            "type": notification_data.get("type", "email"),
            "message": notification_data.get("message"),
            "status": "SENT",
            "sent_at": time.time()
        }
        
        # Save to own database
        self.database[notification_id] = notification
        self.request_count += 1
        
        return {
            "service": self.metadata.name,
            "notification": notification,
            "message": "Notification sent successfully"
        }

# ============================================================================
# SERVICE REGISTRY (Service Discovery)
# ============================================================================

class ServiceRegistry:
    """
    Service Registry: Service discovery for microservices
    
    In production, use:
    - Consul
    - Eureka
    - etcd
    - Kubernetes DNS
    """
    
    def __init__(self):
        self.services: Dict[str, List[Microservice]] = {}
    
    def register(self, service: Microservice):
        """Register a service instance"""
        service_name = service.metadata.name
        
        if service_name not in self.services:
            self.services[service_name] = []
        
        self.services[service_name].append(service)
        print(f"Registered: {service_name} (instance: {service.metadata.instance_id})")
    
    def discover(self, service_name: str) -> Optional[Microservice]:
        """Discover a service (returns one instance, could load balance)"""
        instances = self.services.get(service_name, [])
        
        if instances:
            # Simple round-robin (in production, use load balancer)
            return instances[0]
        
        return None
    
    def health_check_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Check health of all services"""
        health_status = {}
        
        for service_name, instances in self.services.items():
            health_status[service_name] = [inst.health_check() for inst in instances]
        
        return health_status

# Create global service registry
service_registry = ServiceRegistry()

# ============================================================================
# API GATEWAY
# ============================================================================

class APIGateway:
    """
    API Gateway: Single entry point for clients
    
    Responsibilities:
    - Route requests to appropriate microservice
    - Aggregate responses
    - Authentication/Authorization
    - Rate limiting
    - Request/Response transformation
    
    In production, use:
    - Kong
    - NGINX
    - AWS API Gateway
    - Traefik
    """
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
    
    def route_request(self, service_name: str, method: str, endpoint: str, data: Any) -> Dict[str, Any]:
        """Route request to appropriate service"""
        service = self.registry.discover(service_name)
        
        if not service:
            return {"error": f"Service {service_name} not found"}
        
        # Route to service (simplified)
        return {
            "gateway": "API Gateway",
            "routed_to": service.metadata.endpoint,
            "service": service_name
        }

# Create API Gateway
api_gateway = APIGateway(service_registry)

# ============================================================================
# MICROSERVICES AGENTS (LangGraph Integration)
# ============================================================================

# Initialize services
order_service = OrderService()
payment_service = PaymentService()
inventory_service = InventoryService()
notification_service = NotificationService()

# Register services
service_registry.register(order_service)
service_registry.register(payment_service)
service_registry.register(inventory_service)
service_registry.register(notification_service)

def order_service_agent(state: MicroservicesState) -> MicroservicesState:
    """Order Service Agent"""
    request = state["request"]
    
    # Create order
    order_result = order_service.create_order({
        "customer": "customer_123",
        "items": [{"product_id": "PROD-001", "quantity": 2}],
        "total": 100.00
    })
    
    return {
        "order_service_result": order_result,
        "service_calls": [f"Called: {order_service.metadata.name} (POST /orders)"],
        "messages": [f"[Order Service] Order created: {order_result['order']['order_id']}"]
    }

def inventory_service_agent(state: MicroservicesState) -> MicroservicesState:
    """Inventory Service Agent"""
    
    # Check and reserve inventory
    check_result = inventory_service.check_availability("PROD-001", 2)
    
    if check_result["is_available"]:
        reserve_result = inventory_service.reserve_items("PROD-001", 2)
        inventory_result = {**check_result, **reserve_result}
    else:
        inventory_result = check_result
    
    return {
        "inventory_service_result": inventory_result,
        "service_calls": [f"Called: {inventory_service.metadata.name} (GET /inventory, POST /reserve)"],
        "messages": [f"[Inventory Service] Items reserved: {inventory_result.get('reserved', False)}"]
    }

def payment_service_agent(state: MicroservicesState) -> MicroservicesState:
    """Payment Service Agent"""
    order_result = state.get("order_service_result", {})
    order_id = order_result.get("order", {}).get("order_id")
    
    # Process payment
    payment_result = payment_service.process_payment({
        "order_id": order_id,
        "amount": 100.00,
        "method": "credit_card"
    })
    
    return {
        "payment_service_result": payment_result,
        "service_calls": [f"Called: {payment_service.metadata.name} (POST /payments)"],
        "messages": [f"[Payment Service] Payment processed: {payment_result['payment']['payment_id']}"]
    }

def notification_service_agent(state: MicroservicesState) -> MicroservicesState:
    """Notification Service Agent"""
    order_result = state.get("order_service_result", {})
    
    # Send notification
    notification_result = notification_service.send_notification({
        "recipient": "customer@example.com",
        "type": "email",
        "message": f"Order confirmed: {order_result.get('order', {}).get('order_id')}"
    })
    
    return {
        "notification_service_result": notification_result,
        "service_calls": [f"Called: {notification_service.metadata.name} (POST /notifications/email)"],
        "messages": [f"[Notification Service] Notification sent"]
    }

def api_gateway_aggregator(state: MicroservicesState) -> MicroservicesState:
    """API Gateway aggregates responses"""
    
    final_response = f"""Microservices Transaction Complete:
    
    Order Service: {state.get('order_service_result', {}).get('order', {}).get('order_id', 'N/A')}
    Inventory Service: Reserved = {state.get('inventory_service_result', {}).get('reserved', False)}
    Payment Service: {state.get('payment_service_result', {}).get('payment', {}).get('payment_id', 'N/A')}
    Notification Service: Sent = {state.get('notification_service_result', {}).get('notification', {}).get('status', 'N/A')}
    
    Total Service Calls: {len(state.get('service_calls', []))}
    """
    
    return {
        "final_response": final_response.strip(),
        "messages": ["[API Gateway] Aggregated microservices responses"]
    }

# ============================================================================
# BUILD THE MICROSERVICES GRAPH
# ============================================================================

def create_microservices_graph():
    """
    Create a StateGraph demonstrating microservices architecture.
    
    Flow (orchestrated by API Gateway):
    1. Order Service creates order
    2. Inventory Service checks and reserves items
    3. Payment Service processes payment
    4. Notification Service sends confirmation
    5. API Gateway aggregates responses
    
    Each service is independent, has own database, and can be scaled separately.
    """
    
    workflow = StateGraph(MicroservicesState)
    
    # Add microservice nodes
    workflow.add_node("order_service", order_service_agent)
    workflow.add_node("inventory_service", inventory_service_agent)
    workflow.add_node("payment_service", payment_service_agent)
    workflow.add_node("notification_service", notification_service_agent)
    workflow.add_node("api_gateway", api_gateway_aggregator)
    
    # Define flow (API Gateway orchestrates)
    workflow.add_edge(START, "order_service")
    workflow.add_edge("order_service", "inventory_service")
    workflow.add_edge("inventory_service", "payment_service")
    workflow.add_edge("payment_service", "notification_service")
    workflow.add_edge("notification_service", "api_gateway")
    workflow.add_edge("api_gateway", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Microservices MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Service Registry
    print("\n" + "=" * 80)
    print("Example 1: Service Registry and Discovery")
    print("=" * 80)
    
    print("\nRegistered Microservices:")
    for service_name, instances in service_registry.services.items():
        print(f"  {service_name}:")
        for instance in instances:
            print(f"    - Instance: {instance.metadata.instance_id}")
            print(f"      Endpoint: {instance.metadata.endpoint}")
            print(f"      Database: {instance.metadata.database}")
            print(f"      Version: {instance.metadata.version}")
    
    print("\nService Health Checks:")
    health_status = service_registry.health_check_all()
    for service_name, health_checks in health_status.items():
        for health in health_checks:
            print(f"  {health['service']}: {health['status']} (requests: {health['request_count']})")
    
    # Example 2: Microservices Execution
    print("\n" + "=" * 80)
    print("Example 2: E-Commerce Transaction via Microservices")
    print("=" * 80)
    
    microservices_graph = create_microservices_graph()
    
    initial_state: MicroservicesState = {
        "request": "Create order for product PROD-001, quantity 2",
        "order_service_result": None,
        "payment_service_result": None,
        "inventory_service_result": None,
        "notification_service_result": None,
        "final_response": "",
        "service_calls": [],
        "messages": []
    }
    
    result = microservices_graph.invoke(initial_state)
    
    print("\nService Execution Flow:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nService Calls Made:")
    for call in result["service_calls"]:
        print(f"  {call}")
    
    print("\nFinal Response:")
    print(result["final_response"])
    
    # Example 3: Decentralized Data
    print("\n" + "=" * 80)
    print("Example 3: Decentralized Data Management")
    print("=" * 80)
    
    print("\nEach microservice owns its data:")
    print(f"  Order Service DB (order_db): {len(order_service.database)} orders")
    print(f"  Payment Service DB (payment_db): {len(payment_service.database)} payments")
    print(f"  Inventory Service DB (inventory_db): {len(inventory_service.database)} products")
    print(f"  Notification Service DB (notification_db): {len(notification_service.database)} notifications")
    
    print("\nBenefits of Decentralized Data:")
    print("  ✓ Services can scale independently")
    print("  ✓ Services can choose best database for their needs")
    print("  ✓ No single point of failure")
    print("  ✓ Services are loosely coupled")
    
    print("\nChallenges:")
    print("  ✗ Data consistency across services")
    print("  ✗ Distributed transactions (Saga pattern needed)")
    print("  ✗ Query across services more complex")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Microservices decompose application into small, independent services
2. Each microservice has single responsibility (Order, Payment, Inventory, etc.)
3. Services independently deployable and scalable
4. Decentralized data: each service owns its database
5. Service Discovery: Registry helps services find each other
6. API Gateway: Single entry point, routes to services
7. Communication: REST APIs, gRPC, or message queues
8. Benefits: independent deployment, technology diversity, scalability, resilience
9. Trade-offs: complexity, network latency, eventual consistency, operations overhead
10. Infrastructure: Service registry, API gateway, load balancer, containers
11. Use cases: large-scale apps (Netflix, Amazon), e-commerce, SaaS platforms
12. vs Monolithic: distributed vs single process, polyglot vs single stack
    """)

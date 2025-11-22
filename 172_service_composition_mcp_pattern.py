"""
Pattern 172: Service Composition MCP Pattern

This pattern demonstrates Service-Oriented Architecture (SOA) principles where multiple
independent services are composed to create complex functionality. Services are discovered
through a registry, communicate via well-defined interfaces, and can be reused across
different compositions.

Key Concepts:
1. Service: Self-contained unit of functionality with defined interface
2. Service Registry: Central directory where services register and are discovered
3. Service Discovery: Mechanism to locate services by capability
4. Service Contract: Interface specification (inputs, outputs, protocols)
5. Service Orchestration: Coordinating multiple services to achieve goal
6. Service Choreography: Decentralized coordination via events
7. Loose Coupling: Services independent, interact only via contracts

Service Composition Patterns:
1. Orchestration: Central coordinator directs service interactions
2. Choreography: Services react to events, no central control
3. Service Mesh: Infrastructure layer handling service-to-service communication
4. API Gateway: Single entry point routing to multiple services
5. Service Aggregator: Combines results from multiple services

Service Registry Patterns:
- Client-Side Discovery: Client queries registry, calls service directly
- Server-Side Discovery: Load balancer queries registry
- Self-Registration: Services register themselves
- Third-Party Registration: Deployment platform registers services

Benefits:
- Reusability: Services used in multiple compositions
- Interoperability: Standard contracts enable integration
- Scalability: Scale individual services independently
- Flexibility: Swap service implementations
- Technology Agnostic: Services can use different tech stacks

Trade-offs:
- Network Overhead: Service calls across network
- Complexity: Distributed system coordination
- Service Discovery: Requires registry infrastructure
- Contract Management: Versioning and compatibility
- Testing: Integration testing more complex

Use Cases:
- E-commerce: payment + inventory + shipping + notification services
- Data processing: extract + transform + validate + load services
- AI pipeline: preprocessing + inference + postprocessing services
- Customer support: authentication + ticketing + knowledge base services
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from datetime import datetime
from enum import Enum

# Define the state for service composition
class ServiceCompositionState(TypedDict):
    """State for service composition workflow"""
    request: str
    service_calls: Annotated[List[str], operator.add]
    authentication_result: Optional[Dict[str, Any]]
    data_processing_result: Optional[Dict[str, Any]]
    storage_result: Optional[Dict[str, Any]]
    notification_result: Optional[Dict[str, Any]]
    final_response: str
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# SERVICE REGISTRY
# ============================================================================

class ServiceType(Enum):
    """Types of services available"""
    AUTHENTICATION = "authentication"
    DATA_PROCESSING = "data_processing"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    ANALYTICS = "analytics"
    VALIDATION = "validation"

class ServiceMetadata:
    """Metadata for a registered service"""
    def __init__(self, name: str, service_type: ServiceType, version: str, 
                 endpoint: str, capabilities: List[str]):
        self.name = name
        self.service_type = service_type
        self.version = version
        self.endpoint = endpoint
        self.capabilities = capabilities
        self.registered_at = datetime.now()
        self.health_status = "healthy"
        self.load = 0  # Current load for load balancing
    
    def to_dict(self):
        return {
            "name": self.name,
            "type": self.service_type.value,
            "version": self.version,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities,
            "health": self.health_status,
            "load": self.load
        }

class ServiceRegistry:
    """
    Service Registry: Central directory for service discovery
    
    Responsibilities:
    - Service registration and deregistration
    - Service discovery by type or capability
    - Health check tracking
    - Load balancing support
    """
    
    def __init__(self):
        self.services: Dict[str, ServiceMetadata] = {}
    
    def register(self, service: ServiceMetadata) -> str:
        """Register a service in the registry"""
        service_id = f"{service.name}_{service.version}"
        self.services[service_id] = service
        return f"Registered service: {service_id}"
    
    def deregister(self, service_id: str) -> str:
        """Remove a service from the registry"""
        if service_id in self.services:
            del self.services[service_id]
            return f"Deregistered service: {service_id}"
        return f"Service not found: {service_id}"
    
    def discover(self, service_type: ServiceType) -> List[ServiceMetadata]:
        """Discover all services of a given type"""
        return [s for s in self.services.values() 
                if s.service_type == service_type and s.health_status == "healthy"]
    
    def discover_by_capability(self, capability: str) -> List[ServiceMetadata]:
        """Discover services by specific capability"""
        return [s for s in self.services.values() 
                if capability in s.capabilities and s.health_status == "healthy"]
    
    def get_service(self, service_id: str) -> Optional[ServiceMetadata]:
        """Get a specific service by ID"""
        return self.services.get(service_id)
    
    def list_all(self) -> List[Dict[str, Any]]:
        """List all registered services"""
        return [s.to_dict() for s in self.services.values()]
    
    def health_check(self, service_id: str, is_healthy: bool):
        """Update health status of a service"""
        if service_id in self.services:
            self.services[service_id].health_status = "healthy" if is_healthy else "unhealthy"
    
    def update_load(self, service_id: str, load: int):
        """Update current load for load balancing"""
        if service_id in self.services:
            self.services[service_id].load = load
    
    def get_least_loaded(self, service_type: ServiceType) -> Optional[ServiceMetadata]:
        """Get the least loaded service of a given type (for load balancing)"""
        candidates = self.discover(service_type)
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.load)

# Create global service registry
service_registry = ServiceRegistry()

# ============================================================================
# SERVICE IMPLEMENTATIONS
# ============================================================================

def authentication_service(state: ServiceCompositionState) -> ServiceCompositionState:
    """
    Authentication Service: Validates user credentials and permissions
    
    Service Contract:
    - Input: request containing user info
    - Output: authentication result with token and permissions
    - SLA: < 100ms response time
    """
    request = state["request"]
    
    prompt = f"""You are an authentication service. Analyze the following request
    and determine if it contains valid authentication:
    
    Request: {request}
    
    Return a JSON-like response with:
    - authenticated: true/false
    - user_id: extracted user identifier
    - permissions: list of permissions
    - token: simulated auth token"""
    
    response = llm.invoke(prompt)
    
    # Simulate service response
    auth_result = {
        "service": "authentication_service",
        "version": "1.0",
        "authenticated": True,
        "user_id": "user_12345",
        "permissions": ["read", "write"],
        "token": "jwt_token_abc123",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "authentication_result": auth_result,
        "service_calls": [f"Called: authentication_service v1.0"],
        "messages": ["[Authentication Service] User authenticated successfully"]
    }

def data_processing_service(state: ServiceCompositionState) -> ServiceCompositionState:
    """
    Data Processing Service: Processes and transforms data
    
    Service Contract:
    - Input: raw request data + auth token
    - Output: processed and structured data
    - SLA: < 500ms for standard requests
    """
    request = state["request"]
    auth = state.get("authentication_result", {})
    
    if not auth.get("authenticated"):
        return {
            "data_processing_result": {"error": "Not authenticated"},
            "service_calls": ["Failed: data_processing_service - auth required"],
            "messages": ["[Data Processing Service] Authentication required"]
        }
    
    prompt = f"""You are a data processing service. Process the following request:
    
    Request: {request}
    User: {auth.get('user_id')}
    
    Extract key information and structure it appropriately."""
    
    response = llm.invoke(prompt)
    
    processing_result = {
        "service": "data_processing_service",
        "version": "2.1",
        "processed_data": response.content,
        "record_count": 1,
        "processing_time_ms": 234,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "data_processing_result": processing_result,
        "service_calls": [f"Called: data_processing_service v2.1"],
        "messages": ["[Data Processing Service] Data processed successfully"]
    }

def storage_service(state: ServiceCompositionState) -> ServiceCompositionState:
    """
    Storage Service: Persists data to database/storage
    
    Service Contract:
    - Input: processed data + auth token
    - Output: storage confirmation with record ID
    - SLA: < 200ms for writes
    """
    processing_result = state.get("data_processing_result", {})
    
    storage_result = {
        "service": "storage_service",
        "version": "1.5",
        "stored": True,
        "record_id": "rec_67890",
        "storage_location": "s3://bucket/records/67890",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "storage_result": storage_result,
        "service_calls": [f"Called: storage_service v1.5"],
        "messages": ["[Storage Service] Data stored successfully"]
    }

def notification_service(state: ServiceCompositionState) -> ServiceCompositionState:
    """
    Notification Service: Sends notifications to users
    
    Service Contract:
    - Input: notification message + recipient
    - Output: delivery confirmation
    - SLA: < 1000ms for delivery
    """
    auth = state.get("authentication_result", {})
    storage = state.get("storage_result", {})
    
    notification_result = {
        "service": "notification_service",
        "version": "1.2",
        "sent": True,
        "recipient": auth.get("user_id", "unknown"),
        "notification_id": "notif_abc",
        "channel": "email",
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "notification_result": notification_result,
        "service_calls": [f"Called: notification_service v1.2"],
        "messages": ["[Notification Service] Notification sent successfully"]
    }

# ============================================================================
# SERVICE ORCHESTRATOR
# ============================================================================

def service_orchestrator(state: ServiceCompositionState) -> ServiceCompositionState:
    """
    Service Orchestrator: Coordinates multiple services
    
    Orchestration Pattern:
    - Central coordinator (this function) directs all service interactions
    - Services are called in specific order
    - Error handling and compensation logic centralized
    - State passed through orchestrator
    
    Alternative: Choreography Pattern
    - Services react to events
    - No central coordinator
    - More resilient but harder to understand flow
    """
    
    return {
        "messages": ["[Service Orchestrator] Coordinating service composition"]
    }

def service_aggregator(state: ServiceCompositionState) -> ServiceCompositionState:
    """
    Service Aggregator: Combines results from all services
    
    Aggregation Patterns:
    - Collect all service responses
    - Combine into unified response
    - Handle partial failures
    - Format response for client
    """
    
    final_response = f"""Service Composition Complete:
    
    Authentication: {state.get('authentication_result', {}).get('authenticated', False)}
    Data Processing: {state.get('data_processing_result', {}).get('service', 'N/A')}
    Storage: {state.get('storage_result', {}).get('record_id', 'N/A')}
    Notification: {state.get('notification_result', {}).get('sent', False)}
    
    Total Services Called: {len(state.get('service_calls', []))}
    """
    
    return {
        "final_response": final_response,
        "messages": ["[Service Aggregator] Aggregated all service results"]
    }

# ============================================================================
# SERVICE DISCOVERY AND CLIENT
# ============================================================================

class ServiceClient:
    """
    Service Client: Discovers and invokes services
    
    Patterns:
    - Client-Side Discovery: Client queries registry and calls service
    - Load Balancing: Client selects least loaded service
    - Circuit Breaker: Client handles service failures
    - Retry Logic: Client retries failed calls
    """
    
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
    
    def call_service(self, service_type: ServiceType, request: Any) -> Dict[str, Any]:
        """
        Call a service by type (with client-side discovery)
        """
        # Discover available services
        services = self.registry.discover(service_type)
        
        if not services:
            return {"error": f"No services found for type {service_type.value}"}
        
        # Load balancing: select least loaded service
        service = min(services, key=lambda s: s.load)
        
        # Update load
        self.registry.update_load(f"{service.name}_{service.version}", service.load + 1)
        
        # Simulate service call
        result = {
            "service": service.name,
            "version": service.version,
            "endpoint": service.endpoint,
            "request": request,
            "status": "success"
        }
        
        # Decrease load after call
        self.registry.update_load(f"{service.name}_{service.version}", service.load - 1)
        
        return result
    
    def call_by_capability(self, capability: str, request: Any) -> Dict[str, Any]:
        """
        Call a service by specific capability
        """
        services = self.registry.discover_by_capability(capability)
        
        if not services:
            return {"error": f"No services found with capability {capability}"}
        
        service = services[0]  # Use first available
        
        return {
            "service": service.name,
            "capability": capability,
            "request": request,
            "status": "success"
        }

# ============================================================================
# REGISTER SERVICES
# ============================================================================

def initialize_service_registry():
    """Register all available services"""
    
    # Authentication Service
    auth_service = ServiceMetadata(
        name="authentication_service",
        service_type=ServiceType.AUTHENTICATION,
        version="1.0",
        endpoint="https://api.example.com/auth",
        capabilities=["jwt", "oauth", "session"]
    )
    service_registry.register(auth_service)
    
    # Data Processing Service
    processing_service = ServiceMetadata(
        name="data_processing_service",
        service_type=ServiceType.DATA_PROCESSING,
        version="2.1",
        endpoint="https://api.example.com/process",
        capabilities=["transform", "validate", "enrich"]
    )
    service_registry.register(processing_service)
    
    # Storage Service
    storage_svc = ServiceMetadata(
        name="storage_service",
        service_type=ServiceType.STORAGE,
        version="1.5",
        endpoint="https://api.example.com/storage",
        capabilities=["persist", "retrieve", "delete"]
    )
    service_registry.register(storage_svc)
    
    # Notification Service
    notif_service = ServiceMetadata(
        name="notification_service",
        service_type=ServiceType.NOTIFICATION,
        version="1.2",
        endpoint="https://api.example.com/notify",
        capabilities=["email", "sms", "push"]
    )
    service_registry.register(notif_service)

# ============================================================================
# BUILD THE SERVICE COMPOSITION GRAPH
# ============================================================================

def create_service_composition_graph():
    """
    Create a StateGraph that demonstrates service composition via orchestration.
    
    Flow (Orchestration Pattern):
    1. Orchestrator initializes
    2. Authentication service validates request
    3. Data processing service processes data
    4. Storage service persists data
    5. Notification service sends confirmation
    6. Aggregator combines all results
    """
    
    workflow = StateGraph(ServiceCompositionState)
    
    # Add service nodes
    workflow.add_node("orchestrator", service_orchestrator)
    workflow.add_node("authentication", authentication_service)
    workflow.add_node("data_processing", data_processing_service)
    workflow.add_node("storage", storage_service)
    workflow.add_node("notification", notification_service)
    workflow.add_node("aggregator", service_aggregator)
    
    # Define orchestrated flow
    workflow.add_edge(START, "orchestrator")
    workflow.add_edge("orchestrator", "authentication")
    workflow.add_edge("authentication", "data_processing")
    workflow.add_edge("data_processing", "storage")
    workflow.add_edge("storage", "notification")
    workflow.add_edge("notification", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Service Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Initialize service registry
    initialize_service_registry()
    
    # Example 1: Service Discovery
    print("\n" + "=" * 80)
    print("Example 1: Service Registry and Discovery")
    print("=" * 80)
    
    print("\nRegistered Services:")
    for service in service_registry.list_all():
        print(f"  - {service['name']} v{service['version']} ({service['type']})")
        print(f"    Capabilities: {', '.join(service['capabilities'])}")
        print(f"    Health: {service['health']}, Load: {service['load']}")
    
    print("\nService Discovery by Type:")
    auth_services = service_registry.discover(ServiceType.AUTHENTICATION)
    print(f"  Authentication services: {len(auth_services)}")
    for svc in auth_services:
        print(f"    - {svc.name} v{svc.version}")
    
    print("\nService Discovery by Capability:")
    email_services = service_registry.discover_by_capability("email")
    print(f"  Services with 'email' capability: {len(email_services)}")
    for svc in email_services:
        print(f"    - {svc.name} (can: {', '.join(svc.capabilities)})")
    
    # Example 2: Service Orchestration
    print("\n" + "=" * 80)
    print("Example 2: Service Orchestration Pattern")
    print("=" * 80)
    
    composition_graph = create_service_composition_graph()
    
    initial_state: ServiceCompositionState = {
        "request": "Process user data submission for user john@example.com",
        "service_calls": [],
        "authentication_result": None,
        "data_processing_result": None,
        "storage_result": None,
        "notification_result": None,
        "final_response": "",
        "messages": []
    }
    
    result = composition_graph.invoke(initial_state)
    
    print("\nService Execution Flow:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nService Calls Made:")
    for call in result["service_calls"]:
        print(f"  {call}")
    
    print("\nFinal Response:")
    print(result["final_response"])
    
    # Example 3: Service Client with Client-Side Discovery
    print("\n" + "=" * 80)
    print("Example 3: Service Client with Client-Side Discovery")
    print("=" * 80)
    
    client = ServiceClient(service_registry)
    
    # Call service by type
    result = client.call_service(
        ServiceType.DATA_PROCESSING, 
        {"data": "sample data"}
    )
    print("\nCalling service by type (DATA_PROCESSING):")
    print(f"  Service: {result['service']} v{result['version']}")
    print(f"  Endpoint: {result['endpoint']}")
    print(f"  Status: {result['status']}")
    
    # Call service by capability
    result = client.call_by_capability("email", {"to": "user@example.com"})
    print("\nCalling service by capability ('email'):")
    print(f"  Service: {result['service']}")
    print(f"  Capability: {result['capability']}")
    print(f"  Status: {result['status']}")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Service Composition builds complex functionality from independent services
2. Service Registry enables service discovery and health tracking
3. Orchestration Pattern: central coordinator directs service interactions
4. Choreography Pattern: services react to events (not shown, but alternative)
5. Service Contracts define inputs, outputs, and SLAs
6. Client-Side Discovery: client queries registry and calls service directly
7. Server-Side Discovery: load balancer/gateway handles discovery (not shown)
8. Load Balancing: select least loaded service for better distribution
9. Benefits: reusability, interoperability, independent scaling
10. Trade-offs: network overhead, complexity, distributed system challenges
11. Use cases: e-commerce, data pipelines, AI workflows, microservices
12. SOA principles: loose coupling, service contracts, discovery, reusability
    """)

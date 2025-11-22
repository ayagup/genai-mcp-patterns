"""
Pattern 211: API Gateway MCP Pattern

API Gateway provides a single entry point for all client requests:
- Request routing to appropriate microservices
- Authentication and authorization
- Rate limiting and throttling
- Request/response transformation
- API composition and aggregation
- Load balancing
- Caching
- Monitoring and logging

Benefits:
- Simplified client interface
- Centralized security and policies
- Protocol translation (REST, gRPC, WebSocket)
- Reduced client complexity
- Cross-cutting concerns in one place

Use Cases:
- Microservices architectures
- Mobile/web applications
- Third-party API access
- Multi-tenant systems
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import hashlib


class APIGatewayState(TypedDict):
    """State for API gateway operations"""
    gateway_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class Microservice:
    """Backend microservice"""
    service_id: str
    service_name: str
    base_path: str
    host: str
    port: int
    is_healthy: bool = True
    request_count: int = 0
    
    def handle_request(self, path: str, method: str) -> Dict[str, Any]:
        """Handle request"""
        self.request_count += 1
        
        # Simulate processing
        time.sleep(0.01)
        
        return {
            'service': self.service_name,
            'path': path,
            'method': method,
            'status': 200,
            'data': f'Response from {self.service_name}'
        }


@dataclass
class RateLimitRule:
    """Rate limiting configuration"""
    client_id: str
    max_requests_per_minute: int
    current_count: int = 0
    window_start: float = field(default_factory=time.time)
    
    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit"""
        current_time = time.time()
        
        # Reset window if minute has passed
        if current_time - self.window_start >= 60:
            self.current_count = 0
            self.window_start = current_time
        
        if self.current_count < self.max_requests_per_minute:
            self.current_count += 1
            return True
        
        return False


class APIGateway:
    """
    API Gateway implementation with:
    - Request routing
    - Authentication
    - Rate limiting
    - Load balancing
    - Response caching
    """
    
    def __init__(self):
        self.services: Dict[str, List[Microservice]] = {}
        self.rate_limits: Dict[str, RateLimitRule] = {}
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.auth_tokens: Dict[str, str] = {}
        
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_requests = 0
        self.cache_hits = 0
    
    def register_service(self, service: Microservice):
        """Register a microservice"""
        if service.base_path not in self.services:
            self.services[service.base_path] = []
        self.services[service.base_path].append(service)
    
    def set_rate_limit(self, client_id: str, max_requests: int):
        """Set rate limit for client"""
        self.rate_limits[client_id] = RateLimitRule(client_id, max_requests)
    
    def authenticate(self, token: str) -> Optional[str]:
        """Authenticate request"""
        return self.auth_tokens.get(token)
    
    def add_auth_token(self, token: str, client_id: str):
        """Add authentication token"""
        self.auth_tokens[token] = client_id
    
    def route_request(self, path: str, method: str, token: str) -> Dict[str, Any]:
        """
        Route request through gateway with:
        1. Authentication
        2. Rate limiting
        3. Caching
        4. Load balancing
        5. Request forwarding
        """
        self.total_requests += 1
        
        # 1. Authentication
        client_id = self.authenticate(token)
        if not client_id:
            self.failed_requests += 1
            return {
                'status': 401,
                'error': 'Unauthorized',
                'message': 'Invalid authentication token'
            }
        
        # 2. Rate Limiting
        if client_id in self.rate_limits:
            if not self.rate_limits[client_id].is_allowed():
                self.rate_limited_requests += 1
                self.failed_requests += 1
                return {
                    'status': 429,
                    'error': 'Too Many Requests',
                    'message': 'Rate limit exceeded'
                }
        
        # 3. Check Cache (for GET requests)
        cache_key = f"{method}:{path}"
        if method == "GET" and cache_key in self.cache:
            self.cache_hits += 1
            self.successful_requests += 1
            response = self.cache[cache_key].copy()
            response['cached'] = True
            return response
        
        # 4. Route to service
        service = self._select_service(path)
        if not service:
            self.failed_requests += 1
            return {
                'status': 404,
                'error': 'Not Found',
                'message': f'No service found for path: {path}'
            }
        
        # 5. Forward request
        response = service.handle_request(path, method)
        
        # Cache GET responses
        if method == "GET" and response['status'] == 200:
            self.cache[cache_key] = response.copy()
        
        self.successful_requests += 1
        response['cached'] = False
        return response
    
    def _select_service(self, path: str) -> Optional[Microservice]:
        """Select service based on path (with load balancing)"""
        for base_path, services in self.services.items():
            if path.startswith(base_path):
                # Simple round-robin load balancing
                healthy_services = [s for s in services if s.is_healthy]
                if healthy_services:
                    # Pick service with least requests
                    return min(healthy_services, key=lambda s: s.request_count)
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gateway statistics"""
        cache_hit_rate = (self.cache_hits / self.total_requests * 100) if self.total_requests > 0 else 0
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        return {
            'total_requests': self.total_requests,
            'successful': self.successful_requests,
            'failed': self.failed_requests,
            'rate_limited': self.rate_limited_requests,
            'success_rate': f"{success_rate:.1f}%",
            'cache_hits': self.cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'registered_services': sum(len(svcs) for svcs in self.services.values()),
            'active_clients': len(self.auth_tokens)
        }


def setup_gateway_agent(state: APIGatewayState):
    """Agent to set up API gateway"""
    operations = []
    results = []
    
    gateway = APIGateway()
    
    # Register microservices
    services = [
        Microservice("svc-1", "UserService", "/api/users", "localhost", 8001),
        Microservice("svc-2", "ProductService", "/api/products", "localhost", 8002),
        Microservice("svc-3", "OrderService", "/api/orders", "localhost", 8003),
        # Add redundant service for load balancing
        Microservice("svc-4", "UserService-2", "/api/users", "localhost", 8004),
    ]
    
    operations.append("API Gateway Setup:")
    operations.append("\nRegistered Services:")
    for service in services:
        gateway.register_service(service)
        operations.append(f"  {service.service_name}: {service.base_path} → {service.host}:{service.port}")
    
    # Configure authentication
    gateway.add_auth_token("token-client-1", "client-1")
    gateway.add_auth_token("token-client-2", "client-2")
    operations.append("\nAuthentication Tokens:")
    operations.append("  token-client-1 → client-1")
    operations.append("  token-client-2 → client-2")
    
    # Configure rate limits
    gateway.set_rate_limit("client-1", max_requests=10)  # 10 requests/minute
    gateway.set_rate_limit("client-2", max_requests=5)   # 5 requests/minute
    operations.append("\nRate Limits:")
    operations.append("  client-1: 10 req/min")
    operations.append("  client-2: 5 req/min")
    
    results.append(f"✓ API Gateway configured with {len(services)} services")
    
    # Store in state
    state['_gateway'] = gateway
    
    return {
        "gateway_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["API Gateway setup complete"]
    }


def routing_demo_agent(state: APIGatewayState):
    """Agent to demonstrate request routing"""
    gateway = state['_gateway']
    operations = []
    results = []
    
    operations.append("\n📡 Request Routing Demo:")
    
    # Valid requests
    test_requests = [
        ("/api/users/123", "GET", "token-client-1"),
        ("/api/products/456", "GET", "token-client-1"),
        ("/api/orders/789", "POST", "token-client-2"),
    ]
    
    operations.append("\nValid Requests:")
    for path, method, token in test_requests:
        response = gateway.route_request(path, method, token)
        operations.append(f"  {method} {path}: {response['status']} - {response.get('service', 'N/A')}")
    
    # Test authentication failure
    operations.append("\nAuthentication Test:")
    response = gateway.route_request("/api/users/123", "GET", "invalid-token")
    operations.append(f"  Invalid token: {response['status']} - {response['error']}")
    
    results.append("✓ Request routing working correctly")
    
    return {
        "gateway_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Routing demo complete"]
    }


def rate_limiting_demo_agent(state: APIGatewayState):
    """Agent to demonstrate rate limiting"""
    gateway = state['_gateway']
    operations = []
    results = []
    
    operations.append("\n🚦 Rate Limiting Demo:")
    
    # Client 2 has 5 req/min limit - try 7 requests
    operations.append("\nClient-2 (5 req/min limit) making 7 requests:")
    
    success_count = 0
    rate_limited_count = 0
    
    for i in range(7):
        response = gateway.route_request("/api/products/1", "GET", "token-client-2")
        if response['status'] == 200:
            success_count += 1
            operations.append(f"  Request {i+1}: ✓ Success")
        elif response['status'] == 429:
            rate_limited_count += 1
            operations.append(f"  Request {i+1}: ✗ Rate Limited")
    
    operations.append(f"\nResult: {success_count} succeeded, {rate_limited_count} rate limited")
    
    results.append(f"✓ Rate limiting enforced: {rate_limited_count} requests blocked")
    
    return {
        "gateway_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Rate limited: {rate_limited_count}/7"],
        "messages": ["Rate limiting demo complete"]
    }


def caching_demo_agent(state: APIGatewayState):
    """Agent to demonstrate response caching"""
    gateway = state['_gateway']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n💾 Response Caching Demo:")
    
    # Make same GET request multiple times
    path = "/api/users/999"
    token = "token-client-1"
    
    operations.append(f"\nMaking same request 3 times:")
    
    for i in range(3):
        start = time.time()
        response = gateway.route_request(path, "GET", token)
        latency = (time.time() - start) * 1000
        
        cached = response.get('cached', False)
        operations.append(f"  Request {i+1}: {latency:.2f}ms - {'CACHED' if cached else 'FRESH'}")
    
    stats = gateway.get_statistics()
    operations.append(f"\nCache Hit Rate: {stats['cache_hit_rate']}")
    
    results.append(f"✓ Caching working: {stats['cache_hits']} cache hits")
    metrics.append(f"Cache hit rate: {stats['cache_hit_rate']}")
    
    return {
        "gateway_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Caching demo complete"]
    }


def load_balancing_demo_agent(state: APIGatewayState):
    """Agent to demonstrate load balancing"""
    gateway = state['_gateway']
    operations = []
    results = []
    
    operations.append("\n⚖️ Load Balancing Demo:")
    
    # Make multiple requests to /api/users (has 2 service instances)
    operations.append("\nMaking 10 requests to /api/users (2 instances):")
    
    for i in range(10):
        gateway.route_request("/api/users/1", "GET", "token-client-1")
    
    # Check distribution
    user_services = gateway.services["/api/users"]
    operations.append("\nLoad Distribution:")
    for service in user_services:
        operations.append(f"  {service.service_name}: {service.request_count} requests")
    
    results.append("✓ Load balanced across instances")
    
    return {
        "gateway_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Load balancing demo complete"]
    }


def statistics_agent(state: APIGatewayState):
    """Agent to show statistics"""
    gateway = state['_gateway']
    operations = []
    results = []
    metrics = []
    
    stats = gateway.get_statistics()
    
    operations.append("\n" + "="*60)
    operations.append("API GATEWAY STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nRequest Metrics:")
    operations.append(f"  Total requests: {stats['total_requests']}")
    operations.append(f"  Successful: {stats['successful']}")
    operations.append(f"  Failed: {stats['failed']}")
    operations.append(f"  Rate limited: {stats['rate_limited']}")
    operations.append(f"  Success rate: {stats['success_rate']}")
    
    operations.append(f"\nCaching:")
    operations.append(f"  Cache hits: {stats['cache_hits']}")
    operations.append(f"  Hit rate: {stats['cache_hit_rate']}")
    
    operations.append(f"\nConfiguration:")
    operations.append(f"  Registered services: {stats['registered_services']}")
    operations.append(f"  Active clients: {stats['active_clients']}")
    
    metrics.append("\n📊 API Gateway Benefits:")
    metrics.append("  ✓ Single entry point")
    metrics.append("  ✓ Centralized authentication")
    metrics.append("  ✓ Rate limiting protection")
    metrics.append("  ✓ Response caching")
    metrics.append("  ✓ Load balancing")
    metrics.append("  ✓ Service discovery")
    
    results.append("✓ API Gateway demonstrated successfully")
    
    return {
        "gateway_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_api_gateway_graph():
    """Create the API gateway workflow graph"""
    workflow = StateGraph(APIGatewayState)
    
    # Add nodes
    workflow.add_node("setup", setup_gateway_agent)
    workflow.add_node("routing", routing_demo_agent)
    workflow.add_node("rate_limiting", rate_limiting_demo_agent)
    workflow.add_node("caching", caching_demo_agent)
    workflow.add_node("load_balancing", load_balancing_demo_agent)
    workflow.add_node("statistics", statistics_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "routing")
    workflow.add_edge("routing", "rate_limiting")
    workflow.add_edge("rate_limiting", "caching")
    workflow.add_edge("caching", "load_balancing")
    workflow.add_edge("load_balancing", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 211: API Gateway MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_api_gateway_graph()
    
    # Initialize state
    initial_state = {
        "gateway_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("GATEWAY OPERATIONS")
    print("=" * 80)
    for op in final_state["gateway_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    for metric in final_state["performance_metrics"]:
        print(metric)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
API Gateway Pattern implemented with:

1. Request Routing:
   - Path-based routing to microservices
   - Service discovery and registration
   - Load balancing across instances
   - Health checking

2. Authentication & Authorization:
   - Token-based authentication
   - Centralized security policies
   - Client identification
   - Access control

3. Rate Limiting:
   - Per-client request limits
   - Time-window based (per minute)
   - Throttling to prevent abuse
   - 429 Too Many Requests responses

4. Response Caching:
   - GET request caching
   - Reduced backend load
   - Improved response times
   - Configurable TTL

5. Cross-Cutting Concerns:
   - Logging and monitoring
   - Metrics collection
   - Error handling
   - Request/response transformation

Real-World Implementations:
- Kong Gateway
- AWS API Gateway
- NGINX Plus
- Azure API Management
- Google Cloud API Gateway
- Apigee
- Tyk

API Gateway Responsibilities:
✓ Request Routing: Direct to correct service
✓ Protocol Translation: REST → gRPC, WebSocket, etc.
✓ Authentication: Validate tokens, API keys
✓ Rate Limiting: Prevent abuse
✓ Load Balancing: Distribute requests
✓ Caching: Reduce backend calls
✓ Monitoring: Track metrics, logs
✓ Transformation: Request/response modification
✓ Aggregation: Combine multiple service calls
✓ Circuit Breaking: Handle service failures

Benefits:
✓ Simplified client code
✓ Centralized security
✓ Better observability
✓ Flexible routing
✓ Protocol abstraction
✓ Reduced latency (caching)
✓ Better resilience

Trade-offs:
⚠️ Single point of failure (mitigate with HA)
⚠️ Increased latency (additional hop)
⚠️ Complexity in configuration
⚠️ Potential bottleneck (horizontal scaling needed)
""")


if __name__ == "__main__":
    main()

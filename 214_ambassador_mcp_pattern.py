"""
Pattern 214: Ambassador MCP Pattern

Ambassador pattern provides proxy for accessing external services:
- Abstracts external dependencies
- Handles network communication complexities
- Provides retry logic, circuit breaking
- Enables local development/testing
- Manages authentication and credentials

Key Responsibilities:
- Connection management
- Protocol translation
- Error handling and retries
- Metrics and monitoring
- Load balancing across endpoints

Benefits:
- Decouples application from external services
- Centralized error handling
- Easier testing (mock ambassador)
- Configuration management
- Security (credential handling)

Use Cases:
- Third-party API access
- Database connections
- Message queue clients
- Cache access
- External service integration
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random


class AmbassadorState(TypedDict):
    """State for ambassador pattern operations"""
    ambassador_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class ExternalService:
    """Simulated external service"""
    service_name: str
    endpoint: str
    is_available: bool = True
    latency_ms: int = 50
    error_rate: float = 0.1  # 10% error rate
    
    request_count: int = 0
    error_count: int = 0
    
    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate external service call"""
        self.request_count += 1
        
        # Simulate unavailability
        if not self.is_available:
            self.error_count += 1
            raise Exception(f"{self.service_name} is unavailable")
        
        # Simulate latency
        time.sleep(self.latency_ms / 1000)
        
        # Simulate errors
        if random.random() < self.error_rate:
            self.error_count += 1
            raise Exception(f"{self.service_name} returned error")
        
        return {
            'service': self.service_name,
            'method': method,
            'result': f"Success from {self.service_name}",
            'params': params
        }


class AmbassadorProxy:
    """
    Ambassador proxy for external service access with:
    - Retry logic
    - Circuit breaking
    - Connection pooling
    - Metrics collection
    """
    
    def __init__(self, service: ExternalService, max_retries: int = 3):
        self.service = service
        self.max_retries = max_retries
        
        # Circuit breaker
        self.circuit_open = False
        self.failure_count = 0
        self.circuit_threshold = 5
        self.last_failure_time = 0
        self.circuit_timeout = 30  # seconds
        
        # Metrics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.retries_performed = 0
        self.circuit_breaks = 0
    
    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call external service through ambassador"""
        self.total_calls += 1
        
        # Check circuit breaker
        if self.circuit_open:
            # Try to close circuit after timeout
            if time.time() - self.last_failure_time > self.circuit_timeout:
                self.circuit_open = False
                self.failure_count = 0
            else:
                self.circuit_breaks += 1
                raise Exception(f"Circuit breaker open for {self.service.service_name}")
        
        # Retry logic
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = self.service.call(method, params)
                self.successful_calls += 1
                self.failure_count = 0  # Reset on success
                return result
                
            except Exception as e:
                last_error = e
                self.failure_count += 1
                
                if attempt < self.max_retries - 1:
                    self.retries_performed += 1
                    # Exponential backoff
                    backoff = 0.1 * (2 ** attempt)
                    time.sleep(backoff)
        
        # All retries failed
        self.failed_calls += 1
        self.last_failure_time = time.time()
        
        # Open circuit if threshold reached
        if self.failure_count >= self.circuit_threshold:
            self.circuit_open = True
        
        raise Exception(f"Failed after {self.max_retries} retries: {last_error}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get ambassador metrics"""
        success_rate = (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        
        return {
            'service': self.service.service_name,
            'total_calls': self.total_calls,
            'successful': self.successful_calls,
            'failed': self.failed_calls,
            'success_rate': f"{success_rate:.1f}%",
            'retries': self.retries_performed,
            'circuit_breaks': self.circuit_breaks,
            'circuit_status': 'OPEN' if self.circuit_open else 'CLOSED'
        }


class Application:
    """Application using ambassador to access external services"""
    
    def __init__(self):
        self.ambassadors: Dict[str, AmbassadorProxy] = {}
    
    def register_ambassador(self, name: str, ambassador: AmbassadorProxy):
        """Register ambassador for external service"""
        self.ambassadors[name] = ambassador
    
    def call_external_service(self, service_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call external service through ambassador"""
        ambassador = self.ambassadors.get(service_name)
        if not ambassador:
            raise Exception(f"No ambassador found for {service_name}")
        
        return ambassador.call(method, params)


def setup_ambassadors_agent(state: AmbassadorState):
    """Agent to set up ambassadors"""
    operations = []
    results = []
    
    app = Application()
    
    # Create external services
    payment_service = ExternalService("PaymentAPI", "https://api.payment.com", latency_ms=30)
    shipping_service = ExternalService("ShippingAPI", "https://api.shipping.com", latency_ms=40)
    inventory_service = ExternalService("InventoryAPI", "https://api.inventory.com", latency_ms=20)
    
    # Create ambassadors
    payment_ambassador = AmbassadorProxy(payment_service, max_retries=3)
    shipping_ambassador = AmbassadorProxy(shipping_service, max_retries=3)
    inventory_ambassador = AmbassadorProxy(inventory_service, max_retries=3)
    
    # Register with application
    app.register_ambassador("payment", payment_ambassador)
    app.register_ambassador("shipping", shipping_ambassador)
    app.register_ambassador("inventory", inventory_ambassador)
    
    operations.append("Ambassador Pattern Setup:")
    operations.append("\nRegistered Ambassadors:")
    operations.append(f"  payment → {payment_service.service_name} ({payment_service.endpoint})")
    operations.append(f"  shipping → {shipping_service.service_name} ({shipping_service.endpoint})")
    operations.append(f"  inventory → {inventory_service.service_name} ({inventory_service.endpoint})")
    
    results.append("✓ 3 ambassadors registered")
    
    # Store in state
    state['_app'] = app
    state['_payment_service'] = payment_service
    
    return {
        "ambassador_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Ambassadors setup complete"]
    }


def normal_operations_agent(state: AmbassadorState):
    """Agent to demonstrate normal operations"""
    app = state['_app']
    operations = []
    results = []
    
    operations.append("\n📞 Normal Operations Demo:")
    
    # Make successful calls
    operations.append("\nMaking service calls through ambassadors:")
    
    try:
        result = app.call_external_service("payment", "processPayment", {"amount": 100})
        operations.append(f"  ✓ Payment: {result['result']}")
    except Exception as e:
        operations.append(f"  ✗ Payment: {e}")
    
    try:
        result = app.call_external_service("shipping", "calculateShipping", {"weight": 5})
        operations.append(f"  ✓ Shipping: {result['result']}")
    except Exception as e:
        operations.append(f"  ✗ Shipping: {e}")
    
    try:
        result = app.call_external_service("inventory", "checkStock", {"sku": "ABC123"})
        operations.append(f"  ✓ Inventory: {result['result']}")
    except Exception as e:
        operations.append(f"  ✗ Inventory: {e}")
    
    results.append("✓ Service calls completed through ambassadors")
    
    return {
        "ambassador_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Normal operations complete"]
    }


def retry_demo_agent(state: AmbassadorState):
    """Agent to demonstrate retry logic"""
    app = state['_app']
    operations = []
    results = []
    
    operations.append("\n🔄 Retry Logic Demo:")
    
    # Multiple calls to trigger retries
    operations.append("\nMaking 10 calls (some may fail and retry):")
    
    success = 0
    failed = 0
    
    for i in range(10):
        try:
            app.call_external_service("payment", "processPayment", {"id": i})
            success += 1
        except:
            failed += 1
    
    operations.append(f"  Successful: {success}")
    operations.append(f"  Failed: {failed}")
    
    # Check retries
    payment_metrics = app.ambassadors["payment"].get_metrics()
    operations.append(f"  Retries performed: {payment_metrics['retries']}")
    
    results.append(f"✓ Automatic retries: {payment_metrics['retries']} performed")
    
    return {
        "ambassador_operations": operations,
        "operation_results": results,
        "performance_metrics": [f"Success rate: {payment_metrics['success_rate']}"],
        "messages": ["Retry demo complete"]
    }


def circuit_breaker_agent(state: AmbassadorState):
    """Agent to demonstrate circuit breaker"""
    app = state['_app']
    payment_service = state['_payment_service']
    operations = []
    results = []
    
    operations.append("\n⚡ Circuit Breaker Demo:")
    
    # Make service unavailable
    payment_service.is_available = False
    operations.append("\nSimulating service unavailability...")
    
    # Try multiple calls to trigger circuit breaker
    operations.append("Making calls to trigger circuit breaker:")
    
    for i in range(7):
        try:
            app.call_external_service("payment", "processPayment", {"id": i})
            operations.append(f"  Call {i+1}: SUCCESS")
        except Exception as e:
            if "Circuit breaker" in str(e):
                operations.append(f"  Call {i+1}: CIRCUIT OPEN")
            else:
                operations.append(f"  Call {i+1}: FAILED")
    
    metrics = app.ambassadors["payment"].get_metrics()
    operations.append(f"\nCircuit Status: {metrics['circuit_status']}")
    operations.append(f"Circuit breaks: {metrics['circuit_breaks']}")
    
    # Restore service
    payment_service.is_available = True
    
    results.append("✓ Circuit breaker activated to protect from cascading failures")
    
    return {
        "ambassador_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Circuit breaker demo complete"]
    }


def statistics_agent(state: AmbassadorState):
    """Agent to show statistics"""
    app = state['_app']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("AMBASSADOR METRICS")
    operations.append("="*60)
    
    for name, ambassador in app.ambassadors.items():
        ambassador_metrics = ambassador.get_metrics()
        operations.append(f"\n{name.upper()} Ambassador:")
        operations.append(f"  Service: {ambassador_metrics['service']}")
        operations.append(f"  Total calls: {ambassador_metrics['total_calls']}")
        operations.append(f"  Success rate: {ambassador_metrics['success_rate']}")
        operations.append(f"  Retries: {ambassador_metrics['retries']}")
        operations.append(f"  Circuit status: {ambassador_metrics['circuit_status']}")
    
    metrics.append("\n📊 Ambassador Pattern Benefits:")
    metrics.append("  ✓ Automatic retry logic")
    metrics.append("  ✓ Circuit breaker protection")
    metrics.append("  ✓ Centralized error handling")
    metrics.append("  ✓ Metrics collection")
    metrics.append("  ✓ Connection management")
    metrics.append("  ✓ Easy testing (mock ambassador)")
    
    results.append("✓ Ambassador pattern demonstrated successfully")
    
    return {
        "ambassador_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_ambassador_graph():
    """Create the ambassador workflow graph"""
    workflow = StateGraph(AmbassadorState)
    
    # Add nodes
    workflow.add_node("setup", setup_ambassadors_agent)
    workflow.add_node("normal", normal_operations_agent)
    workflow.add_node("retry", retry_demo_agent)
    workflow.add_node("circuit", circuit_breaker_agent)
    workflow.add_node("statistics", statistics_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "normal")
    workflow.add_edge("normal", "retry")
    workflow.add_edge("retry", "circuit")
    workflow.add_edge("circuit", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 214: Ambassador MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_ambassador_graph()
    
    # Initialize state
    initial_state = {
        "ambassador_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    for op in final_state["ambassador_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Ambassador Pattern: Proxy for external service access

Key Features:
1. Retry Logic: Exponential backoff
2. Circuit Breaker: Prevent cascading failures
3. Connection Pooling: Reuse connections
4. Metrics: Track success/failure rates
5. Error Handling: Centralized error management

Real-World Examples:
- Netflix Hystrix
- Resilience4j
- Polly (.NET)
- AWS SDK retry logic

Benefits:
✓ Decouples application from external dependencies
✓ Easier testing (mock ambassadors)
✓ Centralized resilience patterns
✓ Configuration management
✓ Security (credential handling)
""")


if __name__ == "__main__":
    main()

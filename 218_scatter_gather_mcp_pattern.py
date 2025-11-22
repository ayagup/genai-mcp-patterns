"""
Pattern 218: Scatter-Gather MCP Pattern

Scatter-Gather fans out requests to multiple services and gathers results:
- Scatter: Send request to multiple services in parallel
- Gather: Collect and combine responses
- Handle partial failures gracefully
- Set timeouts for slow services
- Aggregate best results

Use Cases:
- Price comparison across vendors
- Search across multiple data sources
- Multi-region queries
- Recommendation systems
- Auction systems

Benefits:
- Parallel processing
- Faster overall response
- Resilient to partial failures
- Best-of-breed results
- High availability
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import random


class ScatterGatherState(TypedDict):
    """State for scatter-gather pattern operations"""
    scatter_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class VendorService:
    """Individual vendor service"""
    name: str
    latency_ms: int
    availability: float  # 0.0 to 1.0
    
    def get_price(self, product_id: str) -> Dict[str, Any]:
        """Get price quote from vendor"""
        # Simulate latency
        time.sleep(self.latency_ms / 1000)
        
        # Simulate availability
        if random.random() > self.availability:
            raise Exception(f"{self.name} unavailable")
        
        # Return quote
        base_price = 100.0
        variance = random.uniform(-20, 20)
        
        return {
            'vendor': self.name,
            'price': base_price + variance,
            'shipping': random.choice([0, 5.99, 9.99]),
            'delivery_days': random.randint(2, 7)
        }


class ScatterGatherOrchestrator:
    """
    Orchestrator that scatters requests and gathers results
    """
    
    def __init__(self, vendors: List[VendorService], timeout_ms: int = 200):
        self.vendors = vendors
        self.timeout_ms = timeout_ms
        
        self.scatter_count = 0
        self.successful_gathers = 0
        self.failed_services = 0
        self.timeout_count = 0
    
    def get_best_price(self, product_id: str) -> Dict[str, Any]:
        """Scatter request to all vendors, gather best price"""
        start_time = time.time()
        self.scatter_count += 1
        
        results = []
        errors = []
        
        # Scatter: Send to all vendors (parallel in real implementation)
        for vendor in self.vendors:
            try:
                # Check timeout
                elapsed = (time.time() - start_time) * 1000
                if elapsed > self.timeout_ms:
                    self.timeout_count += 1
                    errors.append(f"{vendor.name}: Timeout")
                    continue
                
                quote = vendor.get_price(product_id)
                results.append(quote)
                self.successful_gathers += 1
                
            except Exception as e:
                self.failed_services += 1
                errors.append(f"{vendor.name}: {str(e)}")
        
        # Gather: Find best price
        if not results:
            return {
                'status': 'no_results',
                'errors': errors,
                'execution_time_ms': (time.time() - start_time) * 1000
            }
        
        # Calculate total cost (price + shipping)
        for result in results:
            result['total_cost'] = result['price'] + result['shipping']
        
        # Sort by total cost
        best = min(results, key=lambda x: x['total_cost'])
        
        return {
            'status': 'success',
            'best_offer': best,
            'all_offers': sorted(results, key=lambda x: x['total_cost']),
            'vendors_contacted': len(self.vendors),
            'vendors_responded': len(results),
            'errors': errors,
            'execution_time_ms': (time.time() - start_time) * 1000
        }


def setup_agent(state: ScatterGatherState):
    """Agent to set up scatter-gather"""
    operations = []
    results = []
    
    # Create vendor services with different characteristics
    vendors = [
        VendorService("Amazon", latency_ms=30, availability=0.99),
        VendorService("eBay", latency_ms=50, availability=0.95),
        VendorService("Walmart", latency_ms=40, availability=0.98),
        VendorService("BestBuy", latency_ms=60, availability=0.90),
        VendorService("Target", latency_ms=35, availability=0.97),
    ]
    
    orchestrator = ScatterGatherOrchestrator(vendors, timeout_ms=500)
    
    operations.append("Scatter-Gather Pattern Setup:")
    operations.append("\nVendor Services:")
    for vendor in vendors:
        operations.append(f"  {vendor.name}: {vendor.latency_ms}ms latency, {vendor.availability*100:.0f}% availability")
    
    operations.append(f"\nOrchestrator timeout: {orchestrator.timeout_ms}ms")
    
    results.append(f"✓ Orchestrator with {len(vendors)} vendors initialized")
    
    state['_orchestrator'] = orchestrator
    
    return {
        "scatter_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def scatter_gather_demo_agent(state: ScatterGatherState):
    """Agent to demonstrate scatter-gather"""
    orchestrator = state['_orchestrator']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n🎯 Scatter-Gather Demo:")
    
    operations.append("\nSearching for best price...")
    operations.append("Scattering request to 5 vendors in parallel...")
    
    result = orchestrator.get_best_price("PROD-789")
    
    if result['status'] == 'success':
        best = result['best_offer']
        
        operations.append(f"\n✓ Gathered {result['vendors_responded']}/{result['vendors_contacted']} responses")
        operations.append(f"  Execution time: {result['execution_time_ms']:.1f}ms")
        
        operations.append("\nBest Offer:")
        operations.append(f"  Vendor: {best['vendor']}")
        operations.append(f"  Price: ${best['price']:.2f}")
        operations.append(f"  Shipping: ${best['shipping']:.2f}")
        operations.append(f"  Total: ${best['total_cost']:.2f}")
        operations.append(f"  Delivery: {best['delivery_days']} days")
        
        operations.append("\nAll Offers (sorted by total cost):")
        for i, offer in enumerate(result['all_offers'][:3], 1):
            operations.append(f"  {i}. {offer['vendor']}: ${offer['total_cost']:.2f} ({offer['delivery_days']} days)")
        
        if result['errors']:
            operations.append(f"\nErrors: {len(result['errors'])}")
            for error in result['errors']:
                operations.append(f"  - {error}")
        
        metrics.append(f"Response time: {result['execution_time_ms']:.1f}ms")
        metrics.append(f"Success rate: {result['vendors_responded']}/{result['vendors_contacted']}")
        
        results.append("✓ Found best price across multiple vendors")
    else:
        operations.append("\n✗ No results available")
        results.append("✗ All vendors failed")
    
    return {
        "scatter_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Scatter-gather complete"]
    }


def multiple_requests_agent(state: ScatterGatherState):
    """Agent to demonstrate multiple requests"""
    orchestrator = state['_orchestrator']
    operations = []
    results = []
    
    operations.append("\n🔄 Multiple Requests Demo:")
    
    operations.append("\nProcessing 3 price comparisons...")
    
    for i in range(3):
        result = orchestrator.get_best_price(f"PROD-{i}")
        if result['status'] == 'success':
            best = result['best_offer']
            operations.append(f"  Product {i}: Best from {best['vendor']} - ${best['total_cost']:.2f}")
    
    results.append("✓ Multiple scatter-gather operations completed")
    
    return {
        "scatter_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Multiple requests complete"]
    }


def statistics_agent(state: ScatterGatherState):
    """Agent to show statistics"""
    orchestrator = state['_orchestrator']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("SCATTER-GATHER STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nTotal scatter operations: {orchestrator.scatter_count}")
    operations.append(f"Successful vendor responses: {orchestrator.successful_gathers}")
    operations.append(f"Failed services: {orchestrator.failed_services}")
    operations.append(f"Timeouts: {orchestrator.timeout_count}")
    
    metrics.append("\n📊 Scatter-Gather Benefits:")
    metrics.append("  ✓ Parallel service calls")
    metrics.append("  ✓ Fast overall response")
    metrics.append("  ✓ Best-of-breed results")
    metrics.append("  ✓ Partial failure handling")
    metrics.append("  ✓ Timeout protection")
    
    results.append("✓ Scatter-Gather pattern demonstrated")
    
    return {
        "scatter_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_scatter_gather_graph():
    """Create the scatter-gather workflow graph"""
    workflow = StateGraph(ScatterGatherState)
    
    workflow.add_node("setup", setup_agent)
    workflow.add_node("demo", scatter_gather_demo_agent)
    workflow.add_node("multiple", multiple_requests_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "demo")
    workflow.add_edge("demo", "multiple")
    workflow.add_edge("multiple", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 218: Scatter-Gather MCP Pattern")
    print("=" * 80)
    
    app = create_scatter_gather_graph()
    initial_state = {
        "scatter_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    
    for op in final_state["scatter_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Scatter-Gather: Fan out, collect best results

Process:
1. Scatter: Send to N services
2. Wait: With timeout
3. Gather: Collect responses
4. Aggregate: Find best result

Benefits:
✓ Parallel execution
✓ Faster response (slowest service wins, not sum)
✓ Resilient (partial failures OK)
✓ Best results (competition)

Real-World:
- Price comparison sites
- Flight search engines
- Hotel booking aggregators
- Multi-cloud queries
""")


if __name__ == "__main__":
    main()

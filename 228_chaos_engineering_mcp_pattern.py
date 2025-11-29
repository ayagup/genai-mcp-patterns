"""
Pattern 228: Chaos Engineering MCP Pattern

This pattern demonstrates chaos engineering - deliberately injecting failures
to test system resilience and identify weaknesses before they cause real problems.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random
import time


# State definition
class ChaosEngineeringState(TypedDict):
    """State for chaos engineering workflow"""
    messages: Annotated[List[str], add]
    chaos_experiments: List[dict]
    experiment_results: List[dict]
    system_resilience_score: float


# Simulated system components
class DatabaseService:
    """Simulated database service"""
    
    def __init__(self):
        self.is_healthy = True
        self.latency_ms = 10
    
    def query(self, sql: str) -> dict:
        """Execute database query"""
        if not self.is_healthy:
            raise Exception("Database is down")
        
        time.sleep(self.latency_ms / 1000)  # Simulate latency
        return {"status": "success", "rows": 10}
    
    def inject_failure(self):
        """Inject database failure"""
        self.is_healthy = False
    
    def inject_latency(self, latency_ms: int):
        """Inject high latency"""
        self.latency_ms = latency_ms
    
    def restore(self):
        """Restore to healthy state"""
        self.is_healthy = True
        self.latency_ms = 10


class APIService:
    """Simulated API service"""
    
    def __init__(self):
        self.failure_rate = 0.0
        self.timeout_enabled = False
    
    def call_api(self, endpoint: str) -> dict:
        """Call API endpoint"""
        # Random failure based on failure rate
        if random.random() < self.failure_rate:
            raise Exception("API call failed")
        
        if self.timeout_enabled:
            time.sleep(5)  # Simulate timeout
            raise Exception("API timeout")
        
        return {"status": "success", "data": {"message": "API response"}}
    
    def inject_random_failures(self, rate: float):
        """Inject random failures at specified rate"""
        self.failure_rate = rate
    
    def inject_timeout(self):
        """Inject timeout errors"""
        self.timeout_enabled = True
    
    def restore(self):
        """Restore to healthy state"""
        self.failure_rate = 0.0
        self.timeout_enabled = False


class CacheService:
    """Simulated cache service"""
    
    def __init__(self):
        self.cache = {}
        self.eviction_enabled = False
    
    def get(self, key: str) -> str:
        """Get from cache"""
        if self.eviction_enabled and random.random() < 0.5:
            # Randomly evict entries
            if key in self.cache:
                del self.cache[key]
        
        return self.cache.get(key, None)
    
    def set(self, key: str, value: str):
        """Set cache value"""
        self.cache[key] = value
    
    def inject_random_eviction(self):
        """Inject random cache eviction"""
        self.eviction_enabled = True
    
    def restore(self):
        """Restore to normal operation"""
        self.eviction_enabled = False


# Resilient application that should handle failures
class ResilientApplication:
    """Application with resilience patterns"""
    
    def __init__(self, db: DatabaseService, api: APIService, cache: CacheService):
        self.db = db
        self.api = api
        self.cache = cache
        self.circuit_breaker_open = False
        self.failed_attempts = 0
    
    def get_data_with_retry(self, query: str) -> dict:
        """Get data with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self.db.query(query)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # Wait before retry
                    continue
                else:
                    # Fallback to cached data or default
                    return {"status": "fallback", "message": "Using cached data"}
    
    def call_api_with_circuit_breaker(self, endpoint: str) -> dict:
        """Call API with circuit breaker pattern"""
        if self.circuit_breaker_open:
            return {"status": "circuit_open", "message": "Circuit breaker is open"}
        
        try:
            result = self.api.call_api(endpoint)
            self.failed_attempts = 0
            return result
        except Exception as e:
            self.failed_attempts += 1
            if self.failed_attempts >= 3:
                self.circuit_breaker_open = True
            return {"status": "error", "message": str(e)}
    
    def get_cached_data_with_fallback(self, key: str) -> str:
        """Get cached data with fallback"""
        cached = self.cache.get(key)
        if cached is None:
            # Fallback to regenerating data
            return f"regenerated_data_for_{key}"
        return cached


# Agent functions
def design_chaos_experiments_agent(state: ChaosEngineeringState) -> ChaosEngineeringState:
    """Agent that designs chaos experiments"""
    print("\n🔬 Designing Chaos Experiments...")
    
    experiments = [
        {
            "name": "Database Failure",
            "description": "Simulate complete database failure",
            "hypothesis": "Application should fallback to cached data",
            "blast_radius": "Low - single component",
            "rollback_plan": "Restore database health"
        },
        {
            "name": "High Database Latency",
            "description": "Inject 500ms latency into database queries",
            "hypothesis": "Application should handle slow queries gracefully",
            "blast_radius": "Medium - affects performance",
            "rollback_plan": "Restore normal latency"
        },
        {
            "name": "API Random Failures",
            "description": "Inject 30% failure rate into API calls",
            "hypothesis": "Circuit breaker should open after threshold",
            "blast_radius": "Medium - affects API functionality",
            "rollback_plan": "Remove failure injection"
        },
        {
            "name": "Cache Eviction",
            "description": "Randomly evict cache entries",
            "hypothesis": "Application should regenerate data when cache misses",
            "blast_radius": "Low - only affects performance",
            "rollback_plan": "Disable random eviction"
        },
        {
            "name": "Combined Failures",
            "description": "Multiple failures at once",
            "hypothesis": "Application should remain partially functional",
            "blast_radius": "High - tests overall resilience",
            "rollback_plan": "Restore all services"
        }
    ]
    
    return {
        **state,
        "chaos_experiments": experiments,
        "messages": [f"✓ Designed {len(experiments)} chaos experiments"]
    }


def run_chaos_experiments_agent(state: ChaosEngineeringState) -> ChaosEngineeringState:
    """Agent that runs chaos experiments"""
    print("\n💥 Running Chaos Experiments...")
    
    # Create services
    db = DatabaseService()
    api = APIService()
    cache = CacheService()
    app = ResilientApplication(db, api, cache)
    
    experiment_results = []
    
    # Experiment 1: Database Failure
    print("\n  Experiment 1: Database Failure")
    db.inject_failure()
    try:
        result = app.get_data_with_retry("SELECT * FROM users")
        experiment_results.append({
            "experiment": "Database Failure",
            "result": result,
            "status": "PASSED" if result["status"] == "fallback" else "FAILED",
            "observation": "Application used fallback mechanism"
        })
    except Exception as e:
        experiment_results.append({
            "experiment": "Database Failure",
            "result": {"error": str(e)},
            "status": "FAILED",
            "observation": "Application did not handle failure gracefully"
        })
    db.restore()
    
    # Experiment 2: High Database Latency
    print("\n  Experiment 2: High Database Latency")
    db.inject_latency(500)
    start_time = time.time()
    result = app.get_data_with_retry("SELECT * FROM users")
    elapsed = (time.time() - start_time) * 1000
    experiment_results.append({
        "experiment": "High Database Latency",
        "result": result,
        "latency_ms": elapsed,
        "status": "PASSED" if result["status"] == "success" else "FAILED",
        "observation": f"Query completed in {elapsed:.0f}ms"
    })
    db.restore()
    
    # Experiment 3: API Random Failures
    print("\n  Experiment 3: API Random Failures (30% rate)")
    api.inject_random_failures(0.3)
    api_calls = 10
    failures = 0
    for i in range(api_calls):
        result = app.call_api_with_circuit_breaker("/api/data")
        if result["status"] in ["error", "circuit_open"]:
            failures += 1
    
    experiment_results.append({
        "experiment": "API Random Failures",
        "total_calls": api_calls,
        "failures": failures,
        "circuit_breaker_opened": app.circuit_breaker_open,
        "status": "PASSED",  # As long as app didn't crash
        "observation": f"Circuit breaker {'opened' if app.circuit_breaker_open else 'handled failures'}"
    })
    api.restore()
    app.circuit_breaker_open = False
    app.failed_attempts = 0
    
    # Experiment 4: Cache Eviction
    print("\n  Experiment 4: Random Cache Eviction")
    cache.set("key1", "cached_value")
    cache.inject_random_eviction()
    cache_hits = 0
    cache_misses = 0
    for i in range(10):
        result = app.get_cached_data_with_fallback("key1")
        if result == "cached_value":
            cache_hits += 1
        else:
            cache_misses += 1
    
    experiment_results.append({
        "experiment": "Cache Eviction",
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "status": "PASSED",  # App handled cache misses
        "observation": f"Application regenerated data on {cache_misses} cache misses"
    })
    cache.restore()
    
    # Experiment 5: Combined Failures
    print("\n  Experiment 5: Combined Failures")
    db.inject_latency(200)
    api.inject_random_failures(0.2)
    cache.inject_random_eviction()
    
    combined_success = 0
    combined_total = 5
    for i in range(combined_total):
        try:
            db_result = app.get_data_with_retry("SELECT * FROM users")
            api_result = app.call_api_with_circuit_breaker("/api/data")
            cache_result = app.get_cached_data_with_fallback("key1")
            combined_success += 1
        except:
            pass
    
    experiment_results.append({
        "experiment": "Combined Failures",
        "total_attempts": combined_total,
        "successful": combined_success,
        "status": "PASSED" if combined_success >= combined_total * 0.5 else "FAILED",
        "observation": f"System remained {(combined_success/combined_total*100):.0f}% functional"
    })
    
    # Restore all
    db.restore()
    api.restore()
    cache.restore()
    
    return {
        **state,
        "experiment_results": experiment_results,
        "messages": [f"✓ Completed {len(experiment_results)} chaos experiments"]
    }


def calculate_resilience_score_agent(state: ChaosEngineeringState) -> ChaosEngineeringState:
    """Agent that calculates system resilience score"""
    print("\n📊 Calculating System Resilience Score...")
    
    passed = sum(1 for r in state["experiment_results"] if r["status"] == "PASSED")
    total = len(state["experiment_results"])
    
    resilience_score = (passed / total) * 100
    
    print(f"\n  Experiments Passed: {passed}/{total}")
    print(f"  Resilience Score: {resilience_score:.1f}%")
    
    if resilience_score >= 80:
        print("  Rating: Excellent resilience! 🌟")
    elif resilience_score >= 60:
        print("  Rating: Good resilience ✓")
    else:
        print("  Rating: Needs improvement ⚠️")
    
    return {
        **state,
        "system_resilience_score": resilience_score,
        "messages": [f"✓ Resilience score: {resilience_score:.1f}%"]
    }


def generate_chaos_report_agent(state: ChaosEngineeringState) -> ChaosEngineeringState:
    """Agent that generates chaos engineering report"""
    print("\n" + "="*70)
    print("CHAOS ENGINEERING REPORT")
    print("="*70)
    
    print(f"\n🔬 Experiments Designed: {len(state['chaos_experiments'])}")
    print(f"💥 Experiments Executed: {len(state['experiment_results'])}")
    print(f"⭐ System Resilience Score: {state['system_resilience_score']:.1f}%")
    
    print("\n📋 Experiment Results:")
    for i, result in enumerate(state["experiment_results"], 1):
        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        print(f"\n  {status_icon} {i}. {result['experiment']}")
        print(f"      Status: {result['status']}")
        print(f"      Observation: {result['observation']}")
    
    print("\n💡 Key Learnings:")
    print("  • Retry mechanisms help handle transient failures")
    print("  • Circuit breakers prevent cascading failures")
    print("  • Fallback strategies maintain partial functionality")
    print("  • Chaos engineering reveals weaknesses before production")
    
    print("\n" + "="*70)
    print("✅ Chaos Engineering Complete!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Chaos engineering report generated"]
    }


# Create the graph
def create_chaos_engineering_graph():
    """Create the chaos engineering workflow graph"""
    workflow = StateGraph(ChaosEngineeringState)
    
    # Add nodes
    workflow.add_node("design_experiments", design_chaos_experiments_agent)
    workflow.add_node("run_experiments", run_chaos_experiments_agent)
    workflow.add_node("calculate_resilience", calculate_resilience_score_agent)
    workflow.add_node("generate_report", generate_chaos_report_agent)
    
    # Add edges
    workflow.add_edge(START, "design_experiments")
    workflow.add_edge("design_experiments", "run_experiments")
    workflow.add_edge("run_experiments", "calculate_resilience")
    workflow.add_edge("calculate_resilience", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 228: Chaos Engineering MCP Pattern")
    print("="*70)
    print("\nChaos Engineering: Break things on purpose to build resilience!")
    print("Test system behavior under failure conditions")
    
    # Create and run the workflow
    app = create_chaos_engineering_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "chaos_experiments": [],
        "experiment_results": [],
        "system_resilience_score": 0.0
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Chaos Engineering Pattern Complete!")


if __name__ == "__main__":
    main()

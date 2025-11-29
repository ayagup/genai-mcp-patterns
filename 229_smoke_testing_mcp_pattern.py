"""
Pattern 229: Smoke Testing MCP Pattern

This pattern demonstrates smoke testing - quick, basic tests to verify
that the most critical functions of a system work before deeper testing.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


# State definition
class SmokeTestState(TypedDict):
    """State for smoke testing workflow"""
    messages: Annotated[List[str], add]
    smoke_tests: List[dict]
    test_results: List[dict]
    build_status: str


# Simulated system components
class Application:
    """Simulated application"""
    
    def __init__(self):
        self.is_running = True
        self.startup_time = time.time()
    
    def health_check(self) -> dict:
        """Check application health"""
        if not self.is_running:
            return {"status": "down", "uptime": 0}
        
        uptime = time.time() - self.startup_time
        return {
            "status": "up",
            "uptime": uptime,
            "version": "1.0.0"
        }
    
    def start(self):
        """Start application"""
        self.is_running = True
        self.startup_time = time.time()
    
    def stop(self):
        """Stop application"""
        self.is_running = False


class Database:
    """Simulated database"""
    
    def __init__(self):
        self.is_connected = True
    
    def connect(self) -> bool:
        """Connect to database"""
        return self.is_connected
    
    def ping(self) -> bool:
        """Ping database"""
        return self.is_connected
    
    def execute_query(self, query: str) -> dict:
        """Execute a simple query"""
        if not self.is_connected:
            raise Exception("Database not connected")
        return {"status": "success", "rows": 1}


class APIEndpoint:
    """Simulated API"""
    
    def __init__(self):
        self.is_available = True
    
    def get_health(self) -> dict:
        """Health endpoint"""
        if not self.is_available:
            return {"status": "unavailable"}
        return {"status": "healthy", "timestamp": time.time()}
    
    def get_version(self) -> dict:
        """Version endpoint"""
        return {"version": "1.0.0", "build": "12345"}


class AuthService:
    """Simulated authentication service"""
    
    def __init__(self):
        self.is_functional = True
    
    def login(self, username: str, password: str) -> dict:
        """Test login"""
        if not self.is_functional:
            return {"success": False, "error": "Auth service unavailable"}
        
        # Simple test login
        if username and password:
            return {"success": True, "token": "test_token_123"}
        return {"success": False, "error": "Invalid credentials"}


class FileSystem:
    """Simulated file system"""
    
    def __init__(self):
        self.can_read = True
        self.can_write = True
    
    def read_config(self) -> dict:
        """Read configuration file"""
        if not self.can_read:
            raise Exception("Cannot read config")
        return {"config": "loaded", "settings": {}}
    
    def write_log(self, message: str) -> bool:
        """Write to log file"""
        if not self.can_write:
            raise Exception("Cannot write log")
        return True


# Agent functions
def define_smoke_tests_agent(state: SmokeTestState) -> SmokeTestState:
    """Agent that defines critical smoke tests"""
    print("\n🚬 Defining Smoke Tests...")
    
    smoke_tests = [
        {
            "name": "Application Starts",
            "description": "Verify application can start successfully",
            "criticality": "CRITICAL",
            "expected_time": "< 5 seconds"
        },
        {
            "name": "Database Connectivity",
            "description": "Verify can connect to database",
            "criticality": "CRITICAL",
            "expected_time": "< 2 seconds"
        },
        {
            "name": "API Health Check",
            "description": "Verify API endpoints are responding",
            "criticality": "CRITICAL",
            "expected_time": "< 1 second"
        },
        {
            "name": "Authentication Works",
            "description": "Verify basic login functionality",
            "criticality": "HIGH",
            "expected_time": "< 3 seconds"
        },
        {
            "name": "Configuration Loads",
            "description": "Verify can read configuration files",
            "criticality": "HIGH",
            "expected_time": "< 1 second"
        },
        {
            "name": "Logging Works",
            "description": "Verify can write to log files",
            "criticality": "MEDIUM",
            "expected_time": "< 1 second"
        }
    ]
    
    print(f"\n  Defined {len(smoke_tests)} critical smoke tests")
    print("  These tests verify the system is ready for further testing")
    
    return {
        **state,
        "smoke_tests": smoke_tests,
        "messages": [f"✓ Defined {len(smoke_tests)} smoke tests"]
    }


def run_smoke_tests_agent(state: SmokeTestState) -> SmokeTestState:
    """Agent that runs smoke tests"""
    print("\n🔥 Running Smoke Tests...")
    
    # Initialize system components
    app = Application()
    db = Database()
    api = APIEndpoint()
    auth = AuthService()
    fs = FileSystem()
    
    test_results = []
    
    # Test 1: Application Starts
    print("\n  Test 1: Application Starts")
    start_time = time.time()
    try:
        app.start()
        health = app.health_check()
        elapsed = time.time() - start_time
        
        test_results.append({
            "test": "Application Starts",
            "status": "PASSED" if health["status"] == "up" else "FAILED",
            "time": elapsed,
            "details": f"Application is {health['status']}, version {health['version']}"
        })
        print(f"    ✓ PASSED ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        test_results.append({
            "test": "Application Starts",
            "status": "FAILED",
            "time": elapsed,
            "details": f"Error: {str(e)}"
        })
        print(f"    ✗ FAILED ({elapsed:.3f}s)")
    
    # Test 2: Database Connectivity
    print("\n  Test 2: Database Connectivity")
    start_time = time.time()
    try:
        connected = db.connect()
        can_ping = db.ping()
        elapsed = time.time() - start_time
        
        test_results.append({
            "test": "Database Connectivity",
            "status": "PASSED" if connected and can_ping else "FAILED",
            "time": elapsed,
            "details": f"Connected: {connected}, Ping: {can_ping}"
        })
        print(f"    ✓ PASSED ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        test_results.append({
            "test": "Database Connectivity",
            "status": "FAILED",
            "time": elapsed,
            "details": f"Error: {str(e)}"
        })
        print(f"    ✗ FAILED ({elapsed:.3f}s)")
    
    # Test 3: API Health Check
    print("\n  Test 3: API Health Check")
    start_time = time.time()
    try:
        health = api.get_health()
        version = api.get_version()
        elapsed = time.time() - start_time
        
        test_results.append({
            "test": "API Health Check",
            "status": "PASSED" if health["status"] == "healthy" else "FAILED",
            "time": elapsed,
            "details": f"Health: {health['status']}, Version: {version['version']}"
        })
        print(f"    ✓ PASSED ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        test_results.append({
            "test": "API Health Check",
            "status": "FAILED",
            "time": elapsed,
            "details": f"Error: {str(e)}"
        })
        print(f"    ✗ FAILED ({elapsed:.3f}s)")
    
    # Test 4: Authentication Works
    print("\n  Test 4: Authentication Works")
    start_time = time.time()
    try:
        login_result = auth.login("testuser", "testpass")
        elapsed = time.time() - start_time
        
        test_results.append({
            "test": "Authentication Works",
            "status": "PASSED" if login_result["success"] else "FAILED",
            "time": elapsed,
            "details": f"Login successful: {login_result['success']}"
        })
        print(f"    ✓ PASSED ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        test_results.append({
            "test": "Authentication Works",
            "status": "FAILED",
            "time": elapsed,
            "details": f"Error: {str(e)}"
        })
        print(f"    ✗ FAILED ({elapsed:.3f}s)")
    
    # Test 5: Configuration Loads
    print("\n  Test 5: Configuration Loads")
    start_time = time.time()
    try:
        config = fs.read_config()
        elapsed = time.time() - start_time
        
        test_results.append({
            "test": "Configuration Loads",
            "status": "PASSED" if config["config"] == "loaded" else "FAILED",
            "time": elapsed,
            "details": "Configuration loaded successfully"
        })
        print(f"    ✓ PASSED ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        test_results.append({
            "test": "Configuration Loads",
            "status": "FAILED",
            "time": elapsed,
            "details": f"Error: {str(e)}"
        })
        print(f"    ✗ FAILED ({elapsed:.3f}s)")
    
    # Test 6: Logging Works
    print("\n  Test 6: Logging Works")
    start_time = time.time()
    try:
        log_success = fs.write_log("Test log message")
        elapsed = time.time() - start_time
        
        test_results.append({
            "test": "Logging Works",
            "status": "PASSED" if log_success else "FAILED",
            "time": elapsed,
            "details": "Logging functional"
        })
        print(f"    ✓ PASSED ({elapsed:.3f}s)")
    except Exception as e:
        elapsed = time.time() - start_time
        test_results.append({
            "test": "Logging Works",
            "status": "FAILED",
            "time": elapsed,
            "details": f"Error: {str(e)}"
        })
        print(f"    ✗ FAILED ({elapsed:.3f}s)")
    
    return {
        **state,
        "test_results": test_results,
        "messages": [f"✓ Ran {len(test_results)} smoke tests"]
    }


def evaluate_build_agent(state: SmokeTestState) -> SmokeTestState:
    """Agent that evaluates build status based on smoke tests"""
    print("\n📊 Evaluating Build Status...")
    
    total_tests = len(state["test_results"])
    passed_tests = sum(1 for r in state["test_results"] if r["status"] == "PASSED")
    critical_failures = sum(1 for r, s in zip(state["test_results"], state["smoke_tests"]) 
                           if r["status"] == "FAILED" and s["criticality"] == "CRITICAL")
    
    # Determine build status
    if critical_failures > 0:
        build_status = "REJECT"
        recommendation = "Critical tests failed - do not proceed with further testing"
    elif passed_tests == total_tests:
        build_status = "PASS"
        recommendation = "All smoke tests passed - proceed with full test suite"
    elif passed_tests >= total_tests * 0.8:
        build_status = "CONDITIONAL"
        recommendation = "Most tests passed - investigate failures before proceeding"
    else:
        build_status = "REJECT"
        recommendation = "Too many failures - fix issues before retesting"
    
    total_time = sum(r["time"] for r in state["test_results"])
    
    print(f"\n  Tests Run: {total_tests}")
    print(f"  Tests Passed: {passed_tests}/{total_tests}")
    print(f"  Critical Failures: {critical_failures}")
    print(f"  Total Time: {total_time:.3f} seconds")
    print(f"  Build Status: {build_status}")
    print(f"  Recommendation: {recommendation}")
    
    return {
        **state,
        "build_status": build_status,
        "messages": [f"✓ Build status: {build_status}"]
    }


def generate_smoke_test_report_agent(state: SmokeTestState) -> SmokeTestState:
    """Agent that generates smoke test report"""
    print("\n" + "="*70)
    print("SMOKE TEST REPORT")
    print("="*70)
    
    print(f"\n🔥 Build Status: {state['build_status']}")
    
    passed = sum(1 for r in state["test_results"] if r["status"] == "PASSED")
    failed = sum(1 for r in state["test_results"] if r["status"] == "FAILED")
    total_time = sum(r["time"] for r in state["test_results"])
    
    print(f"\n📊 Summary:")
    print(f"  Total Tests: {len(state['test_results'])}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⏱ Total Time: {total_time:.3f} seconds")
    
    print("\n📋 Test Results:")
    for i, (result, smoke_test) in enumerate(zip(state["test_results"], state["smoke_tests"]), 1):
        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        print(f"\n  {status_icon} {i}. {result['test']}")
        print(f"      Status: {result['status']}")
        print(f"      Time: {result['time']:.3f}s")
        print(f"      Criticality: {smoke_test['criticality']}")
        print(f"      Details: {result['details']}")
    
    print("\n💡 Smoke Testing Benefits:")
    print("  • Fast execution (< 30 seconds typical)")
    print("  • Early detection of critical issues")
    print("  • Saves time by avoiding full test runs on broken builds")
    print("  • Verifies basic functionality before deployment")
    
    print("\n" + "="*70)
    if state["build_status"] == "PASS":
        print("✅ Smoke Tests PASSED - Proceed with Full Testing!")
    elif state["build_status"] == "CONDITIONAL":
        print("⚠️ Smoke Tests CONDITIONAL - Investigate Failures")
    else:
        print("❌ Smoke Tests FAILED - Fix Critical Issues")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Smoke test report generated"]
    }


# Create the graph
def create_smoke_testing_graph():
    """Create the smoke testing workflow graph"""
    workflow = StateGraph(SmokeTestState)
    
    # Add nodes
    workflow.add_node("define_tests", define_smoke_tests_agent)
    workflow.add_node("run_tests", run_smoke_tests_agent)
    workflow.add_node("evaluate_build", evaluate_build_agent)
    workflow.add_node("generate_report", generate_smoke_test_report_agent)
    
    # Add edges
    workflow.add_edge(START, "define_tests")
    workflow.add_edge("define_tests", "run_tests")
    workflow.add_edge("run_tests", "evaluate_build")
    workflow.add_edge("evaluate_build", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 229: Smoke Testing MCP Pattern")
    print("="*70)
    print("\nSmoke Testing: Quick sanity check before full testing")
    print("If it's smoking, don't proceed!")
    
    # Create and run the workflow
    app = create_smoke_testing_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "smoke_tests": [],
        "test_results": [],
        "build_status": ""
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Smoke Testing Pattern Complete!")


if __name__ == "__main__":
    main()

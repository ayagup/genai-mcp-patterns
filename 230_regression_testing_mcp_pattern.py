"""
Pattern 230: Regression Testing MCP Pattern

This pattern demonstrates regression testing - verifying that new code changes
haven't broken existing functionality that was previously working.
"""

from typing import TypedDict, Annotated, List, Dict
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json


# State definition
class RegressionTestState(TypedDict):
    """State for regression testing workflow"""
    messages: Annotated[List[str], add]
    baseline_results: Dict[str, dict]
    current_results: Dict[str, dict]
    regression_report: List[dict]
    regressions_found: int


# Baseline test suite (represents previous working version)
class BaselineTestSuite:
    """Test suite with baseline results"""
    
    @staticmethod
    def get_baseline_results() -> Dict[str, dict]:
        """
        Return baseline test results from previous version
        In real scenario, these would be loaded from a file/database
        """
        return {
            "test_user_registration": {
                "status": "PASSED",
                "execution_time": 0.15,
                "response_code": 200,
                "memory_usage_mb": 45
            },
            "test_user_login": {
                "status": "PASSED",
                "execution_time": 0.08,
                "response_code": 200,
                "memory_usage_mb": 32
            },
            "test_product_search": {
                "status": "PASSED",
                "execution_time": 0.25,
                "response_code": 200,
                "memory_usage_mb": 67
            },
            "test_add_to_cart": {
                "status": "PASSED",
                "execution_time": 0.12,
                "response_code": 200,
                "memory_usage_mb": 38
            },
            "test_checkout_process": {
                "status": "PASSED",
                "execution_time": 0.45,
                "response_code": 200,
                "memory_usage_mb": 89
            },
            "test_payment_processing": {
                "status": "PASSED",
                "execution_time": 0.35,
                "response_code": 200,
                "memory_usage_mb": 56
            },
            "test_order_confirmation": {
                "status": "PASSED",
                "execution_time": 0.18,
                "response_code": 200,
                "memory_usage_mb": 41
            },
            "test_user_profile_update": {
                "status": "PASSED",
                "execution_time": 0.11,
                "response_code": 200,
                "memory_usage_mb": 35
            }
        }


# Current version test suite (after code changes)
class CurrentTestSuite:
    """Current version tests (some may have regressions)"""
    
    @staticmethod
    def run_tests() -> Dict[str, dict]:
        """
        Run tests on current version
        Simulates some regressions introduced by recent changes
        """
        return {
            "test_user_registration": {
                "status": "PASSED",
                "execution_time": 0.14,  # Slightly improved
                "response_code": 200,
                "memory_usage_mb": 44
            },
            "test_user_login": {
                "status": "FAILED",  # REGRESSION: Now failing!
                "execution_time": 0.09,
                "response_code": 500,  # Error code
                "memory_usage_mb": 32,
                "error": "Authentication service timeout"
            },
            "test_product_search": {
                "status": "PASSED",
                "execution_time": 0.42,  # REGRESSION: 68% slower!
                "response_code": 200,
                "memory_usage_mb": 95  # REGRESSION: More memory
            },
            "test_add_to_cart": {
                "status": "PASSED",
                "execution_time": 0.11,  # Improved
                "response_code": 200,
                "memory_usage_mb": 36
            },
            "test_checkout_process": {
                "status": "PASSED",
                "execution_time": 0.43,  # Slightly improved
                "response_code": 200,
                "memory_usage_mb": 87
            },
            "test_payment_processing": {
                "status": "PASSED",
                "execution_time": 0.35,
                "response_code": 200,
                "memory_usage_mb": 56
            },
            "test_order_confirmation": {
                "status": "FAILED",  # REGRESSION: Now failing!
                "execution_time": 0.22,
                "response_code": 404,
                "memory_usage_mb": 41,
                "error": "Email template not found"
            },
            "test_user_profile_update": {
                "status": "PASSED",
                "execution_time": 0.11,
                "response_code": 200,
                "memory_usage_mb": 35
            }
        }


# Agent functions
def load_baseline_agent(state: RegressionTestState) -> RegressionTestState:
    """Agent that loads baseline test results"""
    print("\n📊 Loading Baseline Test Results...")
    
    baseline_results = BaselineTestSuite.get_baseline_results()
    
    print(f"\n  Loaded {len(baseline_results)} baseline test results")
    print("  These represent the known-good state from the previous version")
    
    passed = sum(1 for r in baseline_results.values() if r["status"] == "PASSED")
    print(f"  Baseline Status: {passed}/{len(baseline_results)} tests passing")
    
    return {
        **state,
        "baseline_results": baseline_results,
        "messages": [f"✓ Loaded {len(baseline_results)} baseline results"]
    }


def run_current_tests_agent(state: RegressionTestState) -> RegressionTestState:
    """Agent that runs tests on current version"""
    print("\n🧪 Running Tests on Current Version...")
    
    current_results = CurrentTestSuite.run_tests()
    
    print(f"\n  Executed {len(current_results)} tests on current version")
    
    passed = sum(1 for r in current_results.values() if r["status"] == "PASSED")
    failed = sum(1 for r in current_results.values() if r["status"] == "FAILED")
    
    print(f"  Current Status: {passed} passed, {failed} failed")
    
    return {
        **state,
        "current_results": current_results,
        "messages": [f"✓ Ran {len(current_results)} tests on current version"]
    }


def detect_regressions_agent(state: RegressionTestState) -> RegressionTestState:
    """Agent that detects regressions by comparing results"""
    print("\n🔍 Detecting Regressions...")
    
    regression_report = []
    regressions_found = 0
    
    baseline = state["baseline_results"]
    current = state["current_results"]
    
    for test_name in baseline.keys():
        if test_name not in current:
            regression_report.append({
                "test": test_name,
                "type": "MISSING_TEST",
                "severity": "CRITICAL",
                "details": "Test no longer exists in current version"
            })
            regressions_found += 1
            continue
        
        baseline_result = baseline[test_name]
        current_result = current[test_name]
        
        # Check for status regression
        if baseline_result["status"] == "PASSED" and current_result["status"] == "FAILED":
            regression_report.append({
                "test": test_name,
                "type": "STATUS_REGRESSION",
                "severity": "CRITICAL",
                "baseline_status": baseline_result["status"],
                "current_status": current_result["status"],
                "error": current_result.get("error", "Unknown error"),
                "details": f"Test was passing, now failing: {current_result.get('error', 'Unknown')}"
            })
            regressions_found += 1
            print(f"  ❌ CRITICAL: {test_name} - Status regression (PASSED → FAILED)")
        
        # Check for performance regression (>50% slower)
        baseline_time = baseline_result["execution_time"]
        current_time = current_result["execution_time"]
        time_increase = ((current_time - baseline_time) / baseline_time) * 100
        
        if time_increase > 50:
            regression_report.append({
                "test": test_name,
                "type": "PERFORMANCE_REGRESSION",
                "severity": "HIGH",
                "baseline_time": baseline_time,
                "current_time": current_time,
                "increase_percent": time_increase,
                "details": f"Execution time increased by {time_increase:.1f}%"
            })
            regressions_found += 1
            print(f"  ⚠️ HIGH: {test_name} - Performance regression ({time_increase:.1f}% slower)")
        
        # Check for memory regression (>30% more memory)
        baseline_memory = baseline_result.get("memory_usage_mb", 0)
        current_memory = current_result.get("memory_usage_mb", 0)
        
        if baseline_memory > 0:
            memory_increase = ((current_memory - baseline_memory) / baseline_memory) * 100
            
            if memory_increase > 30:
                regression_report.append({
                    "test": test_name,
                    "type": "MEMORY_REGRESSION",
                    "severity": "MEDIUM",
                    "baseline_memory": baseline_memory,
                    "current_memory": current_memory,
                    "increase_percent": memory_increase,
                    "details": f"Memory usage increased by {memory_increase:.1f}%"
                })
                regressions_found += 1
                print(f"  ⚠️ MEDIUM: {test_name} - Memory regression ({memory_increase:.1f}% more)")
    
    # Check for improvements too
    improvements = []
    for test_name in baseline.keys():
        if test_name in current:
            baseline_time = baseline[test_name]["execution_time"]
            current_time = current[test_name]["execution_time"]
            time_decrease = ((baseline_time - current_time) / baseline_time) * 100
            
            if time_decrease > 10 and current[test_name]["status"] == "PASSED":
                improvements.append(f"{test_name}: {time_decrease:.1f}% faster")
    
    if improvements:
        print(f"\n  ✅ Found {len(improvements)} performance improvements:")
        for improvement in improvements[:3]:  # Show first 3
            print(f"     {improvement}")
    
    return {
        **state,
        "regression_report": regression_report,
        "regressions_found": regressions_found,
        "messages": [f"✓ Found {regressions_found} regressions"]
    }


def analyze_impact_agent(state: RegressionTestState) -> RegressionTestState:
    """Agent that analyzes impact of regressions"""
    print("\n📈 Analyzing Regression Impact...")
    
    critical = sum(1 for r in state["regression_report"] if r["severity"] == "CRITICAL")
    high = sum(1 for r in state["regression_report"] if r["severity"] == "HIGH")
    medium = sum(1 for r in state["regression_report"] if r["severity"] == "MEDIUM")
    
    print(f"\n  Regression Breakdown:")
    print(f"    CRITICAL: {critical}")
    print(f"    HIGH: {high}")
    print(f"    MEDIUM: {medium}")
    
    # Determine if build should be rejected
    if critical > 0:
        recommendation = "REJECT BUILD - Critical regressions must be fixed"
    elif high > 2:
        recommendation = "CONDITIONAL - Multiple high-severity issues"
    elif high > 0 or medium > 0:
        recommendation = "REVIEW - Some regressions detected"
    else:
        recommendation = "ACCEPT - No regressions detected"
    
    print(f"\n  Recommendation: {recommendation}")
    
    return {
        **state,
        "messages": [f"✓ Analysis complete: {recommendation}"]
    }


def generate_regression_report_agent(state: RegressionTestState) -> RegressionTestState:
    """Agent that generates regression test report"""
    print("\n" + "="*70)
    print("REGRESSION TESTING REPORT")
    print("="*70)
    
    print(f"\n📊 Test Comparison:")
    print(f"  Baseline Tests: {len(state['baseline_results'])}")
    print(f"  Current Tests: {len(state['current_results'])}")
    
    baseline_passed = sum(1 for r in state["baseline_results"].values() if r["status"] == "PASSED")
    current_passed = sum(1 for r in state["current_results"].values() if r["status"] == "PASSED")
    
    print(f"\n  Baseline Passing: {baseline_passed}/{len(state['baseline_results'])}")
    print(f"  Current Passing: {current_passed}/{len(state['current_results'])}")
    
    print(f"\n🔍 Regressions Found: {state['regressions_found']}")
    
    if state["regressions_found"] > 0:
        # Group by severity
        critical = [r for r in state["regression_report"] if r["severity"] == "CRITICAL"]
        high = [r for r in state["regression_report"] if r["severity"] == "HIGH"]
        medium = [r for r in state["regression_report"] if r["severity"] == "MEDIUM"]
        
        if critical:
            print(f"\n  ❌ CRITICAL Regressions ({len(critical)}):")
            for reg in critical:
                print(f"     • {reg['test']}")
                print(f"       Type: {reg['type']}")
                print(f"       Details: {reg['details']}")
        
        if high:
            print(f"\n  ⚠️ HIGH Severity ({len(high)}):")
            for reg in high:
                print(f"     • {reg['test']}")
                print(f"       Details: {reg['details']}")
        
        if medium:
            print(f"\n  ⚠️ MEDIUM Severity ({len(medium)}):")
            for reg in medium:
                print(f"     • {reg['test']}")
                print(f"       Details: {reg['details']}")
    else:
        print("\n  ✅ No regressions detected!")
    
    print("\n💡 Regression Testing Benefits:")
    print("  • Catch unintended side effects of code changes")
    print("  • Ensure existing functionality remains working")
    print("  • Track performance over time")
    print("  • Build confidence in releases")
    
    print("\n" + "="*70)
    
    if state["regressions_found"] == 0:
        print("✅ Regression Testing PASSED - No Issues Found!")
    else:
        critical_count = sum(1 for r in state["regression_report"] if r["severity"] == "CRITICAL")
        if critical_count > 0:
            print(f"❌ Regression Testing FAILED - {critical_count} Critical Issues!")
        else:
            print(f"⚠️ Regression Testing WARNING - {state['regressions_found']} Issues Found")
    
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Regression report generated"]
    }


# Create the graph
def create_regression_testing_graph():
    """Create the regression testing workflow graph"""
    workflow = StateGraph(RegressionTestState)
    
    # Add nodes
    workflow.add_node("load_baseline", load_baseline_agent)
    workflow.add_node("run_current_tests", run_current_tests_agent)
    workflow.add_node("detect_regressions", detect_regressions_agent)
    workflow.add_node("analyze_impact", analyze_impact_agent)
    workflow.add_node("generate_report", generate_regression_report_agent)
    
    # Add edges
    workflow.add_edge(START, "load_baseline")
    workflow.add_edge("load_baseline", "run_current_tests")
    workflow.add_edge("run_current_tests", "detect_regressions")
    workflow.add_edge("detect_regressions", "analyze_impact")
    workflow.add_edge("analyze_impact", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 230: Regression Testing MCP Pattern")
    print("="*70)
    print("\nRegression Testing: Ensure new changes don't break old features")
    print("Compare current results against known-good baseline")
    
    # Create and run the workflow
    app = create_regression_testing_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "baseline_results": {},
        "current_results": {},
        "regression_report": [],
        "regressions_found": 0
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Regression Testing Pattern Complete!")


if __name__ == "__main__":
    main()

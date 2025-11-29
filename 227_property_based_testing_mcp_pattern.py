"""
Pattern 227: Property-Based Testing MCP Pattern

This pattern demonstrates property-based testing where you define properties
that should always hold true, and the system generates many test cases automatically.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random


# State definition
class PropertyBasedTestState(TypedDict):
    """State for property-based testing workflow"""
    messages: Annotated[List[str], add]
    properties: List[dict]
    test_results: List[dict]
    total_tests: int
    passed_tests: int
    failed_tests: int


# System under test
class MathOperations:
    """Math operations to test with properties"""
    
    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
    
    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b
    
    @staticmethod
    def reverse_list(lst: List[int]) -> List[int]:
        """Reverse a list"""
        return list(reversed(lst))
    
    @staticmethod
    def sort_list(lst: List[int]) -> List[int]:
        """Sort a list"""
        return sorted(lst)


class StringOperations:
    """String operations to test"""
    
    @staticmethod
    def concatenate(s1: str, s2: str) -> str:
        """Concatenate two strings"""
        return s1 + s2
    
    @staticmethod
    def reverse(s: str) -> str:
        """Reverse a string"""
        return s[::-1]
    
    @staticmethod
    def uppercase(s: str) -> str:
        """Convert to uppercase"""
        return s.upper()


# Property definitions
def property_commutative_addition(a: int, b: int) -> bool:
    """Property: Addition is commutative (a + b == b + a)"""
    return MathOperations.add(a, b) == MathOperations.add(b, a)


def property_associative_addition(a: int, b: int, c: int) -> bool:
    """Property: Addition is associative ((a + b) + c == a + (b + c))"""
    math_ops = MathOperations()
    left = math_ops.add(math_ops.add(a, b), c)
    right = math_ops.add(a, math_ops.add(b, c))
    return left == right


def property_identity_addition(a: int) -> bool:
    """Property: Adding zero is identity (a + 0 == a)"""
    return MathOperations.add(a, 0) == a


def property_reverse_twice_is_original(lst: List[int]) -> bool:
    """Property: Reversing twice gives original"""
    math_ops = MathOperations()
    once = math_ops.reverse_list(lst)
    twice = math_ops.reverse_list(once)
    return twice == lst


def property_sorted_is_ordered(lst: List[int]) -> bool:
    """Property: Sorted list is in ascending order"""
    sorted_lst = MathOperations.sort_list(lst)
    for i in range(len(sorted_lst) - 1):
        if sorted_lst[i] > sorted_lst[i + 1]:
            return False
    return True


def property_sorted_has_same_length(lst: List[int]) -> bool:
    """Property: Sorted list has same length as original"""
    return len(MathOperations.sort_list(lst)) == len(lst)


def property_string_concat_length(s1: str, s2: str) -> bool:
    """Property: Length of concatenation equals sum of lengths"""
    result = StringOperations.concatenate(s1, s2)
    return len(result) == len(s1) + len(s2)


def property_reverse_string_twice(s: str) -> bool:
    """Property: Reversing string twice gives original"""
    string_ops = StringOperations()
    once = string_ops.reverse(s)
    twice = string_ops.reverse(once)
    return twice == s


# Agent functions
def define_properties_agent(state: PropertyBasedTestState) -> PropertyBasedTestState:
    """Agent that defines properties to test"""
    print("\n📋 Defining Properties to Test...")
    
    properties = [
        {
            "name": "Commutative Addition",
            "property": "a + b == b + a",
            "function": property_commutative_addition,
            "num_params": 2
        },
        {
            "name": "Associative Addition",
            "property": "(a + b) + c == a + (b + c)",
            "function": property_associative_addition,
            "num_params": 3
        },
        {
            "name": "Identity Addition",
            "property": "a + 0 == a",
            "function": property_identity_addition,
            "num_params": 1
        },
        {
            "name": "Reverse Twice",
            "property": "reverse(reverse(list)) == list",
            "function": property_reverse_twice_is_original,
            "num_params": "list"
        },
        {
            "name": "Sorted is Ordered",
            "property": "sorted list is in ascending order",
            "function": property_sorted_is_ordered,
            "num_params": "list"
        },
        {
            "name": "Sorted Preserves Length",
            "property": "len(sorted(list)) == len(list)",
            "function": property_sorted_has_same_length,
            "num_params": "list"
        },
        {
            "name": "String Concat Length",
            "property": "len(s1 + s2) == len(s1) + len(s2)",
            "function": property_string_concat_length,
            "num_params": "strings"
        },
        {
            "name": "String Reverse Twice",
            "property": "reverse(reverse(s)) == s",
            "function": property_reverse_string_twice,
            "num_params": "string"
        }
    ]
    
    return {
        **state,
        "properties": properties,
        "messages": [f"✓ Defined {len(properties)} properties to test"]
    }


def generate_test_cases_agent(state: PropertyBasedTestState) -> PropertyBasedTestState:
    """Agent that generates random test cases"""
    print("\n🎲 Generating Random Test Cases...")
    
    num_cases_per_property = 20  # Generate 20 random cases per property
    test_results = []
    total = 0
    passed = 0
    failed = 0
    
    for prop in state["properties"]:
        print(f"\n  Testing property: {prop['name']}")
        property_func = prop["function"]
        num_params = prop["num_params"]
        
        property_passed = 0
        property_failed = 0
        
        for i in range(num_cases_per_property):
            try:
                # Generate random inputs based on parameter type
                if num_params == 1:
                    # Single integer
                    a = random.randint(-100, 100)
                    result = property_func(a)
                elif num_params == 2:
                    # Two integers
                    a = random.randint(-100, 100)
                    b = random.randint(-100, 100)
                    result = property_func(a, b)
                elif num_params == 3:
                    # Three integers
                    a = random.randint(-100, 100)
                    b = random.randint(-100, 100)
                    c = random.randint(-100, 100)
                    result = property_func(a, b, c)
                elif num_params == "list":
                    # Random list
                    length = random.randint(0, 10)
                    lst = [random.randint(-50, 50) for _ in range(length)]
                    result = property_func(lst)
                elif num_params == "string":
                    # Random string
                    length = random.randint(0, 20)
                    s = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(length))
                    result = property_func(s)
                elif num_params == "strings":
                    # Two random strings
                    len1 = random.randint(0, 15)
                    len2 = random.randint(0, 15)
                    s1 = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(len1))
                    s2 = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(len2))
                    result = property_func(s1, s2)
                
                if result:
                    property_passed += 1
                    passed += 1
                else:
                    property_failed += 1
                    failed += 1
                
                total += 1
                
            except Exception as e:
                property_failed += 1
                failed += 1
                total += 1
        
        test_results.append({
            "property": prop["name"],
            "formula": prop["property"],
            "total_cases": num_cases_per_property,
            "passed": property_passed,
            "failed": property_failed,
            "status": "PASSED" if property_failed == 0 else "FAILED"
        })
        
        print(f"    ✓ Passed: {property_passed}/{num_cases_per_property}")
        if property_failed > 0:
            print(f"    ✗ Failed: {property_failed}/{num_cases_per_property}")
    
    return {
        **state,
        "test_results": test_results,
        "total_tests": total,
        "passed_tests": passed,
        "failed_tests": failed,
        "messages": [f"✓ Generated and ran {total} test cases"]
    }


def analyze_properties_agent(state: PropertyBasedTestState) -> PropertyBasedTestState:
    """Agent that analyzes property test results"""
    print("\n📊 Analyzing Property Test Results...")
    
    total_properties = len(state["test_results"])
    passed_properties = sum(1 for r in state["test_results"] if r["status"] == "PASSED")
    
    print(f"\n  Properties Tested: {total_properties}")
    print(f"  Properties Valid: {passed_properties}/{total_properties}")
    print(f"  Total Test Cases: {state['total_tests']}")
    print(f"  Overall Pass Rate: {(state['passed_tests']/state['total_tests']*100):.1f}%")
    
    return {
        **state,
        "messages": ["✓ Property analysis complete"]
    }


def generate_report_agent(state: PropertyBasedTestState) -> PropertyBasedTestState:
    """Agent that generates property-based testing report"""
    print("\n" + "="*70)
    print("PROPERTY-BASED TESTING REPORT")
    print("="*70)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Properties Tested: {len(state['test_results'])}")
    print(f"  Total Test Cases Generated: {state['total_tests']}")
    print(f"  ✓ Passed: {state['passed_tests']}")
    print(f"  ✗ Failed: {state['failed_tests']}")
    print(f"  Pass Rate: {(state['passed_tests']/state['total_tests']*100):.1f}%")
    
    print(f"\n📋 Property Results:")
    for i, result in enumerate(state["test_results"], 1):
        status_icon = "✓" if result["status"] == "PASSED" else "✗"
        print(f"\n  {status_icon} {i}. {result['property']}")
        print(f"      Formula: {result['formula']}")
        print(f"      Test Cases: {result['total_cases']}")
        print(f"      Passed: {result['passed']}")
        if result["failed"] > 0:
            print(f"      Failed: {result['failed']}")
        print(f"      Status: {result['status']}")
    
    print("\n" + "="*70)
    print("✅ Property-Based Testing Complete!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Report generated"]
    }


# Create the graph
def create_property_testing_graph():
    """Create the property-based testing workflow graph"""
    workflow = StateGraph(PropertyBasedTestState)
    
    # Add nodes
    workflow.add_node("define_properties", define_properties_agent)
    workflow.add_node("generate_tests", generate_test_cases_agent)
    workflow.add_node("analyze_results", analyze_properties_agent)
    workflow.add_node("generate_report", generate_report_agent)
    
    # Add edges
    workflow.add_edge(START, "define_properties")
    workflow.add_edge("define_properties", "generate_tests")
    workflow.add_edge("generate_tests", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 227: Property-Based Testing MCP Pattern")
    print("="*70)
    print("\nProperty-based testing generates many random test cases")
    print("to verify that properties always hold true!")
    
    # Create and run the workflow
    app = create_property_testing_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "properties": [],
        "test_results": [],
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Property-Based Testing Pattern Complete!")


if __name__ == "__main__":
    main()

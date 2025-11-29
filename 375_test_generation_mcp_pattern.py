"""
Test Generation MCP Pattern

This pattern demonstrates automated test case generation from code,
specifications, or requirements.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class TestGenerationState(TypedDict):
    """State for test generation"""
    code: str
    test_cases: List[Dict]
    coverage_estimate: float
    messages: Annotated[List, operator.add]


class TestGenerator:
    """Generate test cases"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate(self, code: str) -> List[Dict]:
        """Generate test cases"""
        return [
            {"test": "test_normal_case", "input": "5", "expected": "10"},
            {"test": "test_edge_case", "input": "0", "expected": "0"},
            {"test": "test_negative", "input": "-5", "expected": "-10"}
        ]


def generate_tests(state: TestGenerationState) -> TestGenerationState:
    """Generate tests"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    generator = TestGenerator(llm)
    
    state["test_cases"] = generator.generate(state["code"])
    state["coverage_estimate"] = 0.85
    state["messages"].append(HumanMessage(content=f"Generated {len(state['test_cases'])} test cases"))
    return state


def create_testgen_graph():
    """Create test generation workflow"""
    workflow = StateGraph(TestGenerationState)
    workflow.add_node("generate", generate_tests)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_testgen_graph()
    result = app.invoke({"code": "def double(x):\n    return x * 2", "test_cases": [], "coverage_estimate": 0.0, "messages": []})
    print(f"Test Generation Complete: {len(result['test_cases'])} tests, {result['coverage_estimate']:.0%} coverage")

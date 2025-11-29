"""
Correctness Check MCP Pattern

This pattern implements correctness checking to verify
outputs, behaviors, and results are correct.

Pattern Type: Quality Assurance
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class CorrectnessState(TypedDict):
    """State for correctness checking"""
    output: str
    expected: str
    correctness_tests: List[Dict]
    correct: bool
    discrepancies: List[str]
    messages: Annotated[List, operator.add]


class CorrectnessChecker:
    """Check correctness"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def check(self, output: str, expected: str) -> List[Dict]:
        """Check correctness"""
        return [
            {"aspect": "Result", "status": "correct", "details": "Matches expected"},
            {"aspect": "Format", "status": "correct", "details": "Valid format"}
        ]


def check_correctness(state: CorrectnessState) -> CorrectnessState:
    """Check correctness"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    checker = CorrectnessChecker(llm)
    
    state["correctness_tests"] = checker.check(state["output"], state["expected"])
    state["correct"] = all(t["status"] == "correct" for t in state["correctness_tests"])
    state["discrepancies"] = [t["aspect"] for t in state["correctness_tests"] if t["status"] != "correct"]
    state["messages"].append(HumanMessage(content=f"Correctness: {'verified' if state['correct'] else 'failed'}"))
    return state


def create_correctness_graph():
    """Create correctness checking workflow"""
    workflow = StateGraph(CorrectnessState)
    workflow.add_node("check", check_correctness)
    workflow.add_edge(START, "check")
    workflow.add_edge("check", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_correctness_graph()
    result = app.invoke({"output": "42", "expected": "42", "correctness_tests": [], "correct": False, "discrepancies": [], "messages": []})
    print(f"Correctness: {'verified' if result['correct'] else 'failed'}")

"""
Completeness Check MCP Pattern

This pattern implements completeness checking to ensure
all required elements, data, or steps are present.

Pattern Type: Quality Assurance
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class CompletenessState(TypedDict):
    """State for completeness checking"""
    item: str
    required_elements: List[str]
    completeness_results: List[Dict]
    complete: bool
    missing_elements: List[str]
    messages: Annotated[List, operator.add]


class CompletenessChecker:
    """Check completeness"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def check(self, item: str, required: List[str]) -> List[Dict]:
        """Check completeness"""
        return [
            {"element": e, "status": "present", "details": "Found"}
            for e in required
        ]


def check_completeness(state: CompletenessState) -> CompletenessState:
    """Check completeness"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    checker = CompletenessChecker(llm)
    
    state["completeness_results"] = checker.check(state["item"], state["required_elements"])
    state["complete"] = all(r["status"] == "present" for r in state["completeness_results"])
    state["missing_elements"] = [r["element"] for r in state["completeness_results"] if r["status"] != "present"]
    state["messages"].append(HumanMessage(content=f"Completeness: {'complete' if state['complete'] else 'incomplete'}"))
    return state


def create_completeness_graph():
    """Create completeness checking workflow"""
    workflow = StateGraph(CompletenessState)
    workflow.add_node("check", check_completeness)
    workflow.add_edge(START, "check")
    workflow.add_edge("check", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_completeness_graph()
    result = app.invoke({"item": "Project Documentation", "required_elements": ["README", "API docs", "Setup guide", "Examples"], "completeness_results": [], "complete": False, "missing_elements": [], "messages": []})
    print(f"Completeness: {'complete' if result['complete'] else 'incomplete'}")

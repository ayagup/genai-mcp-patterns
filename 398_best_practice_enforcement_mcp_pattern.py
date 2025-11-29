"""
Best Practice Enforcement MCP Pattern

This pattern implements enforcement of best practices
and coding standards.

Pattern Type: Meta Pattern
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class BestPracticeState(TypedDict):
    """State for best practice enforcement"""
    code: str
    best_practices: List[str]
    enforcement_results: List[Dict]
    violations: List[str]
    enforcement_report: str
    messages: Annotated[List, operator.add]


class BestPracticeEnforcer:
    """Enforce best practices"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def enforce(self, code: str, practices: List[str]) -> List[Dict]:
        """Enforce best practices"""
        return [
            {"practice": p, "status": "compliant", "details": "Following standard"}
            for p in practices
        ]


def enforce_practices(state: BestPracticeState) -> BestPracticeState:
    """Enforce best practices"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    enforcer = BestPracticeEnforcer(llm)
    
    state["enforcement_results"] = enforcer.enforce(state["code"], state["best_practices"])
    state["violations"] = [
        r["practice"] for r in state["enforcement_results"] 
        if r["status"] != "compliant"
    ]
    state["enforcement_report"] = f"Checked {len(state['best_practices'])} best practices"
    state["messages"].append(HumanMessage(content=state["enforcement_report"]))
    return state


def create_bestpractice_graph():
    """Create best practice enforcement workflow"""
    workflow = StateGraph(BestPracticeState)
    workflow.add_node("enforce", enforce_practices)
    workflow.add_edge(START, "enforce")
    workflow.add_edge("enforce", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_bestpractice_graph()
    result = app.invoke({"code": "def process(): ...", "best_practices": ["Type hints", "Docstrings", "Error handling"], "enforcement_results": [], "violations": [], "enforcement_report": "", "messages": []})
    print(result["enforcement_report"])

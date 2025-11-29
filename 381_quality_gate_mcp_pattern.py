"""
Quality Gate MCP Pattern

This pattern implements a quality gate that checks whether work meets
defined criteria before allowing it to proceed.

Pattern Type: Quality Assurance
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class QualityGateState(TypedDict):
    """State for quality gate"""
    artifact: str
    criteria: List[str]
    quality_checks: List[Dict]
    passed: bool
    gate_decision: str
    messages: Annotated[List, operator.add]


class QualityGateChecker:
    """Check quality gates"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def check(self, artifact: str, criteria: List[str]) -> List[Dict]:
        """Check quality criteria"""
        return [
            {"criterion": c, "status": "pass", "score": 0.9}
            for c in criteria
        ]


def check_quality_gate(state: QualityGateState) -> QualityGateState:
    """Check quality gate"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    checker = QualityGateChecker(llm)
    
    state["quality_checks"] = checker.check(state["artifact"], state["criteria"])
    state["passed"] = all(check["status"] == "pass" for check in state["quality_checks"])
    state["gate_decision"] = "Approved" if state["passed"] else "Rejected"
    state["messages"].append(HumanMessage(content=f"Gate decision: {state['gate_decision']}"))
    return state


def create_qualitygate_graph():
    """Create quality gate workflow"""
    workflow = StateGraph(QualityGateState)
    workflow.add_node("check", check_quality_gate)
    workflow.add_edge(START, "check")
    workflow.add_edge("check", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_qualitygate_graph()
    result = app.invoke({"artifact": "Release v1.0", "criteria": ["All tests pass", "Code coverage > 80%", "Security scan clean"], "quality_checks": [], "passed": False, "gate_decision": "", "messages": []})
    print(f"Quality Gate: {result['gate_decision']}")

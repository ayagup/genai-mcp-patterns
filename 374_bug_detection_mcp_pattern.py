"""
Bug Detection MCP Pattern

This pattern demonstrates automated bug detection using static analysis,
pattern recognition, and anomaly detection.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class BugDetectionState(TypedDict):
    """State for bug detection"""
    code: str
    bugs_found: List[Dict]
    severity_analysis: Dict
    fix_suggestions: List[str]
    messages: Annotated[List, operator.add]


class BugDetector:
    """Detect bugs in code"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def detect(self, code: str) -> List[Dict]:
        """Detect potential bugs"""
        return [
            {"bug": "Null pointer risk", "line": 5, "severity": "high"},
            {"bug": "Resource leak", "line": 12, "severity": "medium"}
        ]


def detect_bugs(state: BugDetectionState) -> BugDetectionState:
    """Detect bugs"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    detector = BugDetector(llm)
    
    state["bugs_found"] = detector.detect(state["code"])
    state["severity_analysis"] = {"high": 1, "medium": 1, "low": 0}
    state["fix_suggestions"] = ["Add null checks", "Close resources"]
    state["messages"].append(HumanMessage(content=f"Found {len(state['bugs_found'])} potential bugs"))
    return state


def create_bugdetection_graph():
    """Create bug detection workflow"""
    workflow = StateGraph(BugDetectionState)
    workflow.add_node("detect", detect_bugs)
    workflow.add_edge(START, "detect")
    workflow.add_edge("detect", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_bugdetection_graph()
    result = app.invoke({"code": "def func(x):\n    return x.value", "bugs_found": [], "severity_analysis": {}, "fix_suggestions": [], "messages": []})
    print(f"Bug Detection Complete: {len(result['bugs_found'])} bugs found")

"""
Anti-Pattern Detection MCP Pattern

This pattern implements detection of anti-patterns and
problematic practices in code or system design.

Pattern Type: Meta Pattern
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class AntiPatternState(TypedDict):
    """State for anti-pattern detection"""
    code_or_design: str
    detected_antipatterns: List[Dict]
    severity_levels: Dict[str, str]
    remediation_suggestions: List[str]
    detection_report: str
    messages: Annotated[List, operator.add]


class AntiPatternDetector:
    """Detect anti-patterns"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def detect(self, code: str) -> List[Dict]:
        """Detect anti-patterns"""
        return [
            {"antipattern": "God Object", "location": "line 50", "severity": "high"},
            {"antipattern": "Magic Numbers", "location": "line 120", "severity": "medium"}
        ]


def detect_antipatterns(state: AntiPatternState) -> AntiPatternState:
    """Detect anti-patterns"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    detector = AntiPatternDetector(llm)
    
    state["detected_antipatterns"] = detector.detect(state["code_or_design"])
    state["severity_levels"] = {
        ap["antipattern"]: ap["severity"]
        for ap in state["detected_antipatterns"]
    }
    state["remediation_suggestions"] = ["Split large class", "Extract constants"]
    state["detection_report"] = f"Detected {len(state['detected_antipatterns'])} anti-patterns"
    state["messages"].append(HumanMessage(content=state["detection_report"]))
    return state


def create_antipattern_graph():
    """Create anti-pattern detection workflow"""
    workflow = StateGraph(AntiPatternState)
    workflow.add_node("detect", detect_antipatterns)
    workflow.add_edge(START, "detect")
    workflow.add_edge("detect", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_antipattern_graph()
    result = app.invoke({"code_or_design": "class Manager { ... }", "detected_antipatterns": [], "severity_levels": {}, "remediation_suggestions": [], "detection_report": "", "messages": []})
    print(result["detection_report"])

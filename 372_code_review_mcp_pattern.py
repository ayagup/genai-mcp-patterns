"""
Code Review MCP Pattern

This pattern demonstrates automated code review with quality checks,
best practices validation, and improvement suggestions.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


class CodeReviewState(TypedDict):
    """State for code review"""
    code: str
    language: str
    review_findings: List[Dict]
    quality_score: float
    suggestions: List[str]
    final_report: str
    messages: Annotated[List, operator.add]


class CodeReviewer:
    """Automated code reviewer"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def review(self, code: str, language: str) -> List[Dict]:
        """Review code for issues"""
        prompt = f"""Review this {language} code:

{code}

Check for:
1. Code quality issues
2. Security vulnerabilities
3. Performance problems
4. Best practice violations
5. Maintainability concerns

Return JSON array of findings."""
        
        messages = [
            SystemMessage(content="You are an expert code reviewer."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            return json.loads(content[json_start:json_end])
        except:
            return [{"issue": "Unable to parse review", "severity": "error"}]


def analyze_code(state: CodeReviewState) -> CodeReviewState:
    """Analyze code"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    reviewer = CodeReviewer(llm)
    
    state["review_findings"] = reviewer.review(state["code"], state["language"])
    state["quality_score"] = 0.75  # Simplified scoring
    
    state["messages"].append(HumanMessage(
        content=f"Review complete: {len(state['review_findings'])} findings"
    ))
    return state


def generate_suggestions(state: CodeReviewState) -> CodeReviewState:
    """Generate improvement suggestions"""
    state["suggestions"] = [
        "Add error handling",
        "Improve naming conventions",
        "Add unit tests"
    ]
    state["messages"].append(HumanMessage(content="Suggestions generated"))
    return state


def create_report(state: CodeReviewState) -> CodeReviewState:
    """Create review report"""
    report = f"""
CODE REVIEW REPORT
==================

Language: {state['language']}
Quality Score: {state['quality_score']:.0%}

Findings: {len(state['review_findings'])}
Suggestions: {len(state['suggestions'])}

Summary: Code reviewed with automated checks.
"""
    
    state["final_report"] = report
    state["messages"].append(HumanMessage(content=report))
    return state


def create_codereview_graph():
    """Create code review workflow"""
    workflow = StateGraph(CodeReviewState)
    
    workflow.add_node("analyze", analyze_code)
    workflow.add_node("suggest", generate_suggestions)
    workflow.add_node("report", create_report)
    
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "suggest")
    workflow.add_edge("suggest", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


if __name__ == "__main__":
    app = create_codereview_graph()
    result = app.invoke({
        "code": "def calc(x):\n    return x * 2",
        "language": "Python",
        "review_findings": [],
        "quality_score": 0.0,
        "suggestions": [],
        "final_report": "",
        "messages": []
    })
    print("Code Review MCP Pattern Complete")
    print(f"Quality Score: {result['quality_score']:.0%}")

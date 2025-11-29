"""
Review MCP Pattern

This pattern implements peer review or automated review
to ensure quality and catch issues.

Pattern Type: Quality Assurance
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class ReviewState(TypedDict):
    """State for review"""
    work_item: str
    review_criteria: List[str]
    review_findings: List[Dict]
    approved: bool
    review_comments: str
    messages: Annotated[List, operator.add]


class WorkReviewer:
    """Review work items"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def review(self, work: str, criteria: List[str]) -> List[Dict]:
        """Review against criteria"""
        return [
            {"criterion": c, "assessment": "satisfactory", "notes": "Meets expectations"}
            for c in criteria
        ]


def review_work(state: ReviewState) -> ReviewState:
    """Review work item"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    reviewer = WorkReviewer(llm)
    
    state["review_findings"] = reviewer.review(state["work_item"], state["review_criteria"])
    state["approved"] = all(f["assessment"] == "satisfactory" for f in state["review_findings"])
    state["review_comments"] = "Approved for merge" if state["approved"] else "Needs revision"
    state["messages"].append(HumanMessage(content=state["review_comments"]))
    return state


def create_review_graph():
    """Create review workflow"""
    workflow = StateGraph(ReviewState)
    workflow.add_node("review", review_work)
    workflow.add_edge(START, "review")
    workflow.add_edge("review", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_review_graph()
    result = app.invoke({"work_item": "PR #123", "review_criteria": ["Code quality", "Test coverage", "Documentation"], "review_findings": [], "approved": False, "review_comments": "", "messages": []})
    print(f"Review: {result['review_comments']}")

"""
Pattern Recommendation MCP Pattern

This pattern implements intelligent recommendation of patterns
based on context, requirements, and constraints.

Pattern Type: Meta Pattern
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class PatternRecommendationState(TypedDict):
    """State for pattern recommendation"""
    requirements: str
    constraints: Dict
    recommended_patterns: List[Dict]
    recommendation_scores: Dict[str, float]
    recommendation_explanation: str
    messages: Annotated[List, operator.add]


class PatternRecommender:
    """Recommend patterns"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def recommend(self, requirements: str, constraints: Dict) -> List[Dict]:
        """Recommend patterns"""
        return [
            {"pattern": "RAG", "score": 0.92, "reason": "Document-based QA"},
            {"pattern": "ReAct", "score": 0.85, "reason": "Requires reasoning"}
        ]


def recommend_patterns(state: PatternRecommendationState) -> PatternRecommendationState:
    """Recommend patterns"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    recommender = PatternRecommender(llm)
    
    state["recommended_patterns"] = recommender.recommend(
        state["requirements"],
        state["constraints"]
    )
    state["recommendation_scores"] = {
        p["pattern"]: p["score"]
        for p in state["recommended_patterns"]
    }
    state["recommendation_explanation"] = f"Top recommendation: {state['recommended_patterns'][0]['pattern']}"
    state["messages"].append(HumanMessage(content=state["recommendation_explanation"]))
    return state


def create_recommendation_graph():
    """Create pattern recommendation workflow"""
    workflow = StateGraph(PatternRecommendationState)
    workflow.add_node("recommend", recommend_patterns)
    workflow.add_edge(START, "recommend")
    workflow.add_edge("recommend", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_recommendation_graph()
    result = app.invoke({"requirements": "Answer questions from technical documentation", "constraints": {"latency": "< 2s", "accuracy": "> 0.9"}, "recommended_patterns": [], "recommendation_scores": {}, "recommendation_explanation": "", "messages": []})
    print(result["recommendation_explanation"])

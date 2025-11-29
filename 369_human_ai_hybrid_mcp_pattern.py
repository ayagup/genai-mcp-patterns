"""
Human-AI Hybrid MCP Pattern

This pattern combines human judgment and AI capabilities for
collaborative decision-making.

Pattern Type: Hybrid
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class HumanAIState(TypedDict):
    """State for human-AI hybrid"""
    task: str
    ai_recommendation: Dict[str, Any]
    human_input: Dict[str, Any]
    collaborative_decision: Dict[str, Any]
    confidence_scores: Dict[str, float]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class AIAssistant:
    """AI-powered assistant"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_and_recommend(self, task: str) -> Dict[str, Any]:
        """AI analysis and recommendation"""
        prompt = f"""Analyze this task and provide recommendation:

Task: {task}

Provide:
1. Analysis of the situation
2. Recommended action
3. Confidence level
4. Reasoning
5. Uncertainty factors

Return JSON:
{{
    "analysis": "situation analysis",
    "recommendation": "what to do",
    "confidence": 0.0-1.0,
    "reasoning": "why this recommendation",
    "uncertainties": ["factors where AI is uncertain"]
}}"""
        
        messages = [
            SystemMessage(content="You are an AI decision support system."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "analysis": "Unable to analyze",
            "recommendation": "Defer to human",
            "confidence": 0.0,
            "reasoning": "Parsing error",
            "uncertainties": ["Technical error"]
        }


class HumanExpert:
    """Simulated human expert input"""
    
    def review_and_decide(self, ai_recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """Human review of AI recommendation"""
        # In production, this would be actual human input
        # For demo, we simulate human judgment
        
        ai_confidence = ai_recommendation.get("confidence", 0.0)
        
        # Simulate human agreement/disagreement
        if ai_confidence > 0.7:
            return {
                "decision": "Agree with AI recommendation",
                "human_confidence": 0.85,
                "modifications": [],
                "rationale": "AI recommendation is sound and well-reasoned"
            }
        else:
            return {
                "decision": "Modify AI recommendation",
                "human_confidence": 0.75,
                "modifications": [
                    "Add human expertise consideration",
                    "Account for contextual factors AI missed"
                ],
                "rationale": "AI uncertain, human judgment needed"
            }


class CollaborativeDecisionMaker:
    """Combine human and AI input"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def synthesize(self, ai_rec: Dict[str, Any], 
                   human_input: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize human-AI collaboration"""
        prompt = f"""Synthesize human and AI input into final decision:

AI Recommendation:
{json.dumps(ai_rec, indent=2)}

Human Input:
{json.dumps(human_input, indent=2)}

Create collaborative decision that:
1. Leverages AI's data processing
2. Incorporates human judgment and context
3. Balances both strengths

Return JSON:
{{
    "final_decision": "the decision",
    "ai_contribution": "how AI helped",
    "human_contribution": "how human helped",
    "synergy": "how collaboration improved outcome",
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are a human-AI collaboration synthesizer."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "final_decision": "Unable to synthesize",
            "ai_contribution": "N/A",
            "human_contribution": "N/A",
            "synergy": "N/A",
            "confidence": 0.0
        }


# Agent functions
def initialize(state: HumanAIState) -> HumanAIState:
    """Initialize"""
    state["messages"].append(HumanMessage(
        content=f"Initializing human-AI collaboration for: {state['task']}"
    ))
    state["current_step"] = "initialized"
    return state


def get_ai_recommendation(state: HumanAIState) -> HumanAIState:
    """Get AI recommendation"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    assistant = AIAssistant(llm)
    
    recommendation = assistant.analyze_and_recommend(state["task"])
    state["ai_recommendation"] = recommendation
    
    state["messages"].append(HumanMessage(
        content=f"AI recommendation: {recommendation.get('recommendation', 'N/A')[:50]}..., "
                f"confidence: {recommendation.get('confidence', 0):.2f}"
    ))
    state["current_step"] = "ai_recommended"
    return state


def get_human_input(state: HumanAIState) -> HumanAIState:
    """Get human expert input"""
    expert = HumanExpert()
    
    human_review = expert.review_and_decide(state["ai_recommendation"])
    state["human_input"] = human_review
    
    state["messages"].append(HumanMessage(
        content=f"Human decision: {human_review.get('decision', 'N/A')}, "
                f"confidence: {human_review.get('human_confidence', 0):.2f}"
    ))
    state["current_step"] = "human_reviewed"
    return state


def collaborate(state: HumanAIState) -> HumanAIState:
    """Collaborative decision"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    decision_maker = CollaborativeDecisionMaker(llm)
    
    collaborative = decision_maker.synthesize(
        state["ai_recommendation"],
        state["human_input"]
    )
    
    state["collaborative_decision"] = collaborative
    state["confidence_scores"] = {
        "ai": state["ai_recommendation"].get("confidence", 0.0),
        "human": state["human_input"].get("human_confidence", 0.0),
        "collaborative": collaborative.get("confidence", 0.0)
    }
    
    state["messages"].append(HumanMessage(
        content=f"Collaborative decision made, confidence: {collaborative.get('confidence', 0):.2f}"
    ))
    state["current_step"] = "collaborated"
    return state


def generate_report(state: HumanAIState) -> HumanAIState:
    """Generate report"""
    report = f"""
HUMAN-AI HYBRID COLLABORATION REPORT
====================================

Task: {state['task']}

AI RECOMMENDATION:
------------------
Analysis: {state['ai_recommendation'].get('analysis', 'N/A')}
Recommendation: {state['ai_recommendation'].get('recommendation', 'N/A')}
Reasoning: {state['ai_recommendation'].get('reasoning', 'N/A')}
AI Confidence: {state['ai_recommendation'].get('confidence', 0):.2%}

Uncertainties:
{chr(10).join(f'- {u}' for u in state['ai_recommendation'].get('uncertainties', []))}

HUMAN INPUT:
------------
Decision: {state['human_input'].get('decision', 'N/A')}
Rationale: {state['human_input'].get('rationale', 'N/A')}
Human Confidence: {state['human_input'].get('human_confidence', 0):.2%}

Modifications:
{chr(10).join(f'- {m}' for m in state['human_input'].get('modifications', []))}

COLLABORATIVE DECISION:
-----------------------
Final Decision: {state['collaborative_decision'].get('final_decision', 'N/A')}

AI Contribution: {state['collaborative_decision'].get('ai_contribution', 'N/A')}
Human Contribution: {state['collaborative_decision'].get('human_contribution', 'N/A')}
Synergy: {state['collaborative_decision'].get('synergy', 'N/A')}

CONFIDENCE SCORES:
------------------
AI Only: {state['confidence_scores'].get('ai', 0):.2%}
Human Only: {state['confidence_scores'].get('human', 0):.2%}
Collaborative: {state['confidence_scores'].get('collaborative', 0):.2%}

Summary:
Human-AI collaboration leverages AI's analytical power with human judgment
and contextual understanding for superior decision-making.
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build graph
def create_human_ai_graph():
    """Create human-AI hybrid workflow"""
    workflow = StateGraph(HumanAIState)
    
    workflow.add_node("initialize", initialize)
    workflow.add_node("ai_recommend", get_ai_recommendation)
    workflow.add_node("human_review", get_human_input)
    workflow.add_node("collaborate", collaborate)
    workflow.add_node("report", generate_report)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "ai_recommend")
    workflow.add_edge("ai_recommend", "human_review")
    workflow.add_edge("human_review", "collaborate")
    workflow.add_edge("collaborate", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "task": "Should we launch the new product next quarter given current market conditions?",
        "ai_recommendation": {},
        "human_input": {},
        "collaborative_decision": {},
        "confidence_scores": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_human_ai_graph()
    
    print("Human-AI Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")
    
    print(f"\nCollaborative Confidence: {result['confidence_scores'].get('collaborative', 0):.2%}")

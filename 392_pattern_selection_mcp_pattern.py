"""
Pattern Selection MCP Pattern

This pattern implements automatic selection of the most
appropriate pattern for a given task or context.

Pattern Type: Meta Pattern
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class PatternSelectionState(TypedDict):
    """State for pattern selection"""
    task: str
    available_patterns: List[str]
    selection_criteria: Dict
    selected_pattern: str
    selection_rationale: str
    messages: Annotated[List, operator.add]


class PatternSelector:
    """Select patterns"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def select(self, task: str, patterns: List[str], criteria: Dict) -> str:
        """Select best pattern"""
        return patterns[0] if patterns else "None"


def select_pattern(state: PatternSelectionState) -> PatternSelectionState:
    """Select pattern"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    selector = PatternSelector(llm)
    
    state["selected_pattern"] = selector.select(
        state["task"], 
        state["available_patterns"], 
        state["selection_criteria"]
    )
    state["selection_rationale"] = f"Selected '{state['selected_pattern']}' for task: {state['task']}"
    state["messages"].append(HumanMessage(content=state["selection_rationale"]))
    return state


def create_selection_graph():
    """Create pattern selection workflow"""
    workflow = StateGraph(PatternSelectionState)
    workflow.add_node("select", select_pattern)
    workflow.add_edge(START, "select")
    workflow.add_edge("select", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_selection_graph()
    result = app.invoke({"task": "Answer question from documents", "available_patterns": ["RAG", "Chain-of-Thought", "ReAct"], "selection_criteria": {"accuracy": 0.9, "speed": 0.7}, "selected_pattern": "", "selection_rationale": "", "messages": []})
    print(result["selection_rationale"])

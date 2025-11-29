"""
Pattern Composition MCP Pattern

This pattern implements composition of multiple patterns
to create more sophisticated workflows.

Pattern Type: Meta Pattern
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class PatternCompositionState(TypedDict):
    """State for pattern composition"""
    patterns: List[str]
    composition_strategy: str
    composed_workflow: Dict
    execution_order: List[str]
    composition_result: str
    messages: Annotated[List, operator.add]


class PatternComposer:
    """Compose patterns"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def compose(self, patterns: List[str], strategy: str) -> Dict:
        """Compose patterns"""
        return {
            "patterns": patterns,
            "strategy": strategy,
            "workflow": "Sequential pipeline"
        }


def compose_patterns(state: PatternCompositionState) -> PatternCompositionState:
    """Compose patterns"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    composer = PatternComposer(llm)
    
    state["composed_workflow"] = composer.compose(state["patterns"], state["composition_strategy"])
    state["execution_order"] = state["patterns"]
    state["composition_result"] = f"Composed {len(state['patterns'])} patterns using {state['composition_strategy']}"
    state["messages"].append(HumanMessage(content=state["composition_result"]))
    return state


def create_composition_graph():
    """Create pattern composition workflow"""
    workflow = StateGraph(PatternCompositionState)
    workflow.add_node("compose", compose_patterns)
    workflow.add_edge(START, "compose")
    workflow.add_edge("compose", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_composition_graph()
    result = app.invoke({"patterns": ["Retrieval", "Generation", "Validation"], "composition_strategy": "Sequential", "composed_workflow": {}, "execution_order": [], "composition_result": "", "messages": []})
    print(result["composition_result"])

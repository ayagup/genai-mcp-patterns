"""
Centralized-Decentralized Hybrid MCP Pattern

This pattern combines centralized coordination with decentralized execution
for balanced control and autonomy.

Pattern Type: Hybrid
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class CentralizedDecentralizedState(TypedDict):
    """State for centralized-decentralized hybrid"""
    task: str
    central_plan: Dict
    decentralized_results: List[Dict]
    hybrid_coordination: Dict
    messages: Annotated[List, operator.add]


def central_planner(state: CentralizedDecentralizedState) -> CentralizedDecentralizedState:
    """Centralized planning"""
    state["central_plan"] = {
        "strategy": "Global optimization",
        "resource_allocation": "Centrally managed",
        "coordination_mode": "top-down"
    }
    state["messages"].append(HumanMessage(content="Central plan created"))
    return state


def decentralized_executor(state: CentralizedDecentralizedState) -> CentralizedDecentralizedState:
    """Decentralized execution"""
    state["decentralized_results"] = [
        {"agent": "A", "result": "Local decision 1"},
        {"agent": "B", "result": "Local decision 2"},
        {"agent": "C", "result": "Local decision 3"}
    ]
    state["messages"].append(HumanMessage(content="Decentralized execution complete"))
    return state


def hybrid_coordinator(state: CentralizedDecentralizedState) -> CentralizedDecentralizedState:
    """Hybrid coordination"""
    state["hybrid_coordination"] = {
        "approach": "Centralized planning with decentralized execution",
        "balance": "70% central, 30% decentralized"
    }
    state["messages"].append(HumanMessage(content="Hybrid coordination established"))
    return state


def create_centralized_decentralized_graph():
    """Create centralized-decentralized workflow"""
    workflow = StateGraph(CentralizedDecentralizedState)
    
    workflow.add_node("central", central_planner)
    workflow.add_node("decentral", decentralized_executor)
    workflow.add_node("hybrid", hybrid_coordinator)
    
    workflow.add_edge(START, "central")
    workflow.add_edge("central", "decentral")
    workflow.add_edge("decentral", "hybrid")
    workflow.add_edge("hybrid", END)
    
    return workflow.compile()


if __name__ == "__main__":
    app = create_centralized_decentralized_graph()
    result = app.invoke({
        "task": "Distributed task execution",
        "central_plan": {},
        "decentralized_results": [],
        "hybrid_coordination": {},
        "messages": []
    })
    print("Centralized-Decentralized Hybrid Pattern Complete")

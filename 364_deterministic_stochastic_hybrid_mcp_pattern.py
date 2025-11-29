"""
Deterministic-Stochastic Hybrid MCP Pattern

This pattern combines deterministic (rule-based, predictable) approaches
with stochastic (probabilistic, randomized) methods.

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
import random


# State definition
class DeterministicStochasticState(TypedDict):
    """State for deterministic-stochastic hybrid"""
    task: str
    deterministic_solution: Dict[str, Any]
    stochastic_samples: List[Dict[str, Any]]
    hybrid_result: Dict[str, Any]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class DeterministicSolver:
    """Deterministic problem solving"""
    
    def solve(self, task: str) -> Dict[str, Any]:
        """Deterministic solution"""
        # Example: sorting, search, mathematical computation
        return {
            "method": "deterministic",
            "solution": "Exact optimal solution using algorithm",
            "confidence": 1.0,
            "reproducible": True
        }


class StochasticSolver:
    """Stochastic problem solving"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def solve(self, task: str, num_samples: int = 3) -> List[Dict[str, Any]]:
        """Generate multiple stochastic solutions"""
        samples = []
        
        for i in range(num_samples):
            # Use LLM with temperature for variability
            prompt = f"Solve: {task} (Attempt {i+1})"
            
            messages = [
                SystemMessage(content="You are a creative problem solver."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            samples.append({
                "sample_id": i+1,
                "solution": response.content[:200],
                "method": "stochastic",
                "reproducible": False
            })
        
        return samples


class HybridCombiner:
    """Combine deterministic and stochastic approaches"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def combine(self, det_solution: Dict[str, Any], 
                stoch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine both approaches"""
        prompt = f"""Combine deterministic and stochastic solutions:

Deterministic: {json.dumps(det_solution, indent=2)}
Stochastic Samples: {json.dumps(stoch_samples, indent=2)}

Return JSON:
{{
    "hybrid_solution": "combined approach",
    "deterministic_weight": 0.0-1.0,
    "stochastic_weight": 0.0-1.0,
    "confidence": 0.0-1.0
}}"""
        
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        try:
            content = response.content
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            return json.loads(content[json_start:json_end])
        except:
            return {"hybrid_solution": "Error combining", "confidence": 0.0}


# Agent functions
def initialize(state: DeterministicStochasticState) -> DeterministicStochasticState:
    """Initialize"""
    state["messages"].append(HumanMessage(content="Initializing deterministic-stochastic hybrid"))
    state["current_step"] = "initialized"
    return state


def solve_deterministic(state: DeterministicStochasticState) -> DeterministicStochasticState:
    """Apply deterministic solving"""
    solver = DeterministicSolver()
    state["deterministic_solution"] = solver.solve(state["task"])
    state["messages"].append(HumanMessage(content="Deterministic solution computed"))
    state["current_step"] = "deterministic_solved"
    return state


def solve_stochastic(state: DeterministicStochasticState) -> DeterministicStochasticState:
    """Apply stochastic solving"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.9)  # High temp for variability
    solver = StochasticSolver(llm)
    state["stochastic_samples"] = solver.solve(state["task"], num_samples=3)
    state["messages"].append(HumanMessage(content=f"Generated {len(state['stochastic_samples'])} stochastic samples"))
    state["current_step"] = "stochastic_solved"
    return state


def combine_solutions(state: DeterministicStochasticState) -> DeterministicStochasticState:
    """Combine both approaches"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    combiner = HybridCombiner(llm)
    state["hybrid_result"] = combiner.combine(
        state["deterministic_solution"],
        state["stochastic_samples"]
    )
    state["messages"].append(HumanMessage(content="Combined deterministic and stochastic solutions"))
    state["current_step"] = "combined"
    return state


def generate_report(state: DeterministicStochasticState) -> DeterministicStochasticState:
    """Generate report"""
    report = f"""
DETERMINISTIC-STOCHASTIC HYBRID REPORT
======================================

Task: {state['task']}

Deterministic Solution:
{json.dumps(state['deterministic_solution'], indent=2)}

Stochastic Samples ({len(state['stochastic_samples'])}):
{json.dumps(state['stochastic_samples'], indent=2)}

Hybrid Result:
{json.dumps(state['hybrid_result'], indent=2)}
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build graph
def create_det_stoch_graph():
    """Create deterministic-stochastic workflow"""
    workflow = StateGraph(DeterministicStochasticState)
    
    workflow.add_node("initialize", initialize)
    workflow.add_node("deterministic", solve_deterministic)
    workflow.add_node("stochastic", solve_stochastic)
    workflow.add_node("combine", combine_solutions)
    workflow.add_node("report", generate_report)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "deterministic")
    workflow.add_edge("deterministic", "stochastic")
    workflow.add_edge("stochastic", "combine")
    workflow.add_edge("combine", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "task": "Find optimal path through network",
        "deterministic_solution": {},
        "stochastic_samples": [],
        "hybrid_result": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_det_stoch_graph()
    
    print("Deterministic-Stochastic Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")

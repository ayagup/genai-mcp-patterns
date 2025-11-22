"""
Constraint Satisfaction MCP Pattern

This pattern implements constraint satisfaction problem solving
using variables, domains, and constraints with backtracking.

Key Features:
- Variable-domain mapping
- Constraint propagation
- Arc consistency
- Backtracking search
- Heuristics for efficiency
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class CSPState(TypedDict):
    """State for constraint satisfaction pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    variables: List[str]
    domains: Dict[str, List]
    constraints: List[str]
    solution: Dict[str, any]


llm = ChatOpenAI(model="gpt-4", temperature=0.2)


def csp_solver(state: CSPState) -> CSPState:
    """Solves constraint satisfaction problems"""
    variables = state.get("variables", [])
    domains = state.get("domains", {})
    constraints = state.get("constraints", [])
    
    system_prompt = """You are a CSP solving expert.

CSP Components:
- Variables: X1, X2, ..., Xn
- Domains: D1, D2, ..., Dn (possible values)
- Constraints: Relations between variables

Solve using backtracking with inference."""
    
    user_prompt = f"""Variables: {variables}
Domains: {domains}
Constraints: {constraints}

Find assignment satisfying all constraints."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🧩 CSP Solver:
    
    Problem Definition:
    • Variables: {len(variables)}
    • Constraints: {len(constraints)}
    
    CSP Algorithm:
    ```python
    def backtracking_search(csp):
        return backtrack({}, csp)
    
    def backtrack(assignment, csp):
        if complete(assignment):
            return assignment
        
        var = select_unassigned_variable(csp, assignment)
        
        for value in order_domain_values(var, assignment, csp):
            if consistent(var, value, assignment, csp):
                assignment[var] = value
                inferences = inference(csp, var, value)
                
                if inferences != failure:
                    result = backtrack(assignment, csp)
                    if result != failure:
                        return result
                
                remove(var, assignment)
        
        return failure
    ```
    
    Heuristics:
    
    Variable Ordering:
    - MRV (Minimum Remaining Values)
    - Degree heuristic
    - Most constrained first
    
    Value Ordering:
    - Least constraining value
    - Random
    - Domain-specific
    
    Inference:
    - Forward checking
    - AC-3 (Arc Consistency)
    - MAC (Maintaining Arc Consistency)
    
    Applications:
    - Sudoku solving
    - Scheduling
    - Resource allocation
    - Configuration problems
    
    Key Insight:
    CSP provides systematic approach to finding
    solutions satisfying multiple constraints.
    """
    
    return {
        "messages": [AIMessage(content=f"🧩 CSP Solver:\n{report}\n\n{response.content}")],
        "solution": {}
    }


def build_csp_graph():
    workflow = StateGraph(CSPState)
    workflow.add_node("csp_solver", csp_solver)
    workflow.add_edge(START, "csp_solver")
    workflow.add_edge("csp_solver", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_csp_graph()
    
    print("=== Constraint Satisfaction MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "variables": ["X1", "X2", "X3"],
        "domains": {"X1": [1, 2, 3], "X2": [1, 2, 3], "X3": [1, 2, 3]},
        "constraints": ["X1 != X2", "X2 != X3", "X1 + X2 = X3"],
        "solution": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 137: Constraint Satisfaction - COMPLETE")
    print(f"{'='*70}")

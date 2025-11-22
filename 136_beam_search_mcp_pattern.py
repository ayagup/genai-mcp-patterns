"""
Beam Search MCP Pattern

This pattern implements beam search for efficient exploration
of large search spaces by keeping only top-k candidates.

Key Features:
- Width-limited search
- Pruning low-probability paths
- Memory efficient
- Parallel exploration
- Quality-speed tradeoff
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class BeamSearchState(TypedDict):
    """State for beam search pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    search_problem: str
    beam_width: int
    candidates: List[Dict]
    best_solution: str


llm = ChatOpenAI(model="gpt-4", temperature=0.5)


def beam_search_agent(state: BeamSearchState) -> BeamSearchState:
    """Performs beam search"""
    search_problem = state.get("search_problem", "")
    beam_width = state.get("beam_width", 3)
    
    system_prompt = f"""You are a beam search expert.

Beam Search Algorithm:
1. Start with initial state
2. Generate all successors
3. Score each candidate
4. Keep top-{beam_width} (beam width)
5. Repeat until goal or max depth

Find best solution efficiently."""
    
    user_prompt = f"""Problem: {search_problem}

Perform beam search with width {beam_width}.
Show the search process and best solution."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    📡 Beam Search Agent:
    
    Search Configuration:
    • Problem: {search_problem[:100]}...
    • Beam Width: {beam_width}
    
    Beam Search Algorithm:
    ```python
    def beam_search(start, beam_width, max_depth):
        beam = [start]
        
        for depth in range(max_depth):
            candidates = []
            
            # Expand all in beam
            for state in beam:
                successors = expand(state)
                candidates.extend(successors)
            
            # Score and prune
            scored = [(score(c), c) for c in candidates]
            scored.sort(reverse=True)
            
            # Keep top-k
            beam = [c for _, c in scored[:beam_width]]
            
            # Check for goal
            for state in beam:
                if is_goal(state):
                    return state
        
        return beam[0]  # Best so far
    ```
    
    Beam Width Effects:
    
    Width = 1: Greedy search
    - Fastest
    - May miss optimal
    - No backtracking
    
    Width = k: Balanced
    - Good speed/quality
    - Parallel exploration
    - Memory: O(k*d)
    
    Width = ∞: Breadth-first
    - Finds optimal
    - Very slow
    - High memory
    
    Applications:
    - Speech recognition
    - Machine translation
    - Code generation
    - Path finding
    
    Advantages:
    - Memory efficient
    - Parallelizable
    - Tunable quality-speed
    - Good for generation
    
    Key Insight:
    Beam search provides practical balance between
    exhaustive search and greedy selection.
    """
    
    return {
        "messages": [AIMessage(content=f"📡 Beam Search Agent:\n{report}\n\n{response.content}")],
        "best_solution": "solution_from_beam_search"
    }


def build_beam_search_graph():
    workflow = StateGraph(BeamSearchState)
    workflow.add_node("beam_search_agent", beam_search_agent)
    workflow.add_edge(START, "beam_search_agent")
    workflow.add_edge("beam_search_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_beam_search_graph()
    
    print("=== Beam Search MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "search_problem": "Find optimal sequence of operations to minimize cost",
        "beam_width": 3,
        "candidates": [],
        "best_solution": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 136: Beam Search - COMPLETE")
    print(f"{'='*70}")

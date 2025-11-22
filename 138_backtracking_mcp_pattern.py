"""
Backtracking MCP Pattern

This pattern implements backtracking search for systematic exploration
of solution spaces with ability to undo choices.

Key Features:
- Systematic search
- Choice points
- Backtrack on failure
- Constraint checking
- Solution enumeration
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class BacktrackingState(TypedDict):
    """State for backtracking pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    problem: str
    choices: List[Dict]
    solution_path: List[str]
    solutions_found: List[List[str]]


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def backtracking_agent(state: BacktrackingState) -> BacktrackingState:
    """Performs backtracking search"""
    problem = state.get("problem", "")
    
    system_prompt = """You are a backtracking search expert.

Backtracking Algorithm:
1. Make a choice
2. Check if valid
3. If valid, recurse
4. If reaches goal, solution found
5. If fails, backtrack (undo choice)
6. Try next option

Find all or first solution."""
    
    user_prompt = f"""Problem: {problem}

Use backtracking to find solution.
Show the search tree and backtrack points."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🔙 Backtracking Agent:
    
    Search Process:
    • Problem: {problem[:100]}...
    
    Backtracking Template:
    ```python
    def backtrack(state, choices):
        if is_solution(state):
            return state
        
        for choice in get_choices(state):
            if is_valid(choice, state):
                # Make choice
                new_state = apply_choice(state, choice)
                
                # Recurse
                result = backtrack(new_state, choices)
                
                if result is not None:
                    return result
                
                # Backtrack: undo choice
                state = undo_choice(state, choice)
        
        return None  # No solution
    ```
    
    Classic Problems:
    
    N-Queens:
    ```python
    def solve_n_queens(n):
        def backtrack(row, cols, diag1, diag2):
            if row == n:
                return [solution]
            
            solutions = []
            for col in range(n):
                if col not in cols and (row-col) not in diag1 
                   and (row+col) not in diag2:
                    # Place queen
                    solution.append((row, col))
                    solutions.extend(
                        backtrack(row+1, cols|{col}, 
                                 diag1|{row-col}, diag2|{row+col})
                    )
                    # Backtrack
                    solution.pop()
            
            return solutions
        
        return backtrack(0, set(), set(), set())
    ```
    
    Sudoku:
    ```python
    def solve_sudoku(board):
        def backtrack(row, col):
            if row == 9:
                return True
            
            next_row, next_col = (row, col+1) if col < 8 else (row+1, 0)
            
            if board[row][col] != 0:
                return backtrack(next_row, next_col)
            
            for num in range(1, 10):
                if is_valid(board, row, col, num):
                    board[row][col] = num
                    if backtrack(next_row, next_col):
                        return True
                    board[row][col] = 0  # Backtrack
            
            return False
        
        backtrack(0, 0)
        return board
    ```
    
    Optimization Techniques:
    
    Pruning:
    • Early termination
    • Constraint checking
    • Bound checking
    • Feasibility tests
    
    Ordering:
    • Variable ordering
    • Value ordering
    • Fail-first principle
    • Most constrained first
    
    Memoization:
    • Cache results
    • Avoid recomputation
    • State hashing
    • Dynamic programming
    
    Applications:
    - Puzzle solving (Sudoku, crosswords)
    - Path finding
    - Scheduling
    - Combinatorial optimization
    
    Key Insight:
    Backtracking systematically explores solution space
    by making choices and undoing them when they fail.
    """
    
    return {
        "messages": [AIMessage(content=f"🔙 Backtracking Agent:\n{report}\n\n{response.content}")],
        "solutions_found": [[]]
    }


def build_backtracking_graph():
    workflow = StateGraph(BacktrackingState)
    workflow.add_node("backtracking_agent", backtracking_agent)
    workflow.add_edge(START, "backtracking_agent")
    workflow.add_edge("backtracking_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_backtracking_graph()
    
    print("=== Backtracking MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "problem": "Find all valid arrangements satisfying placement constraints",
        "choices": [],
        "solution_path": [],
        "solutions_found": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 138: Backtracking - COMPLETE")
    print(f"{'='*70}")

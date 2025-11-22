"""
Tree-of-Thought MCP Pattern

This pattern implements multi-path reasoning exploration where the model
explores multiple reasoning branches like a tree search algorithm.

Key Features:
- Multiple reasoning paths
- Tree search exploration
- Path evaluation and comparison
- Backtracking support
- Best path selection
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TreeOfThoughtState(TypedDict):
    """State for tree-of-thought pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    problem: str
    search_strategy: str  # "bfs", "dfs", "beam"
    num_branches: int
    max_depth: int
    thought_tree: Dict
    best_path: List[str]
    final_solution: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)


# Thought Generator
def thought_generator(state: TreeOfThoughtState) -> TreeOfThoughtState:
    """Generates multiple reasoning paths (thought branches)"""
    problem = state.get("problem", "")
    num_branches = state.get("num_branches", 3)
    search_strategy = state.get("search_strategy", "bfs")
    
    system_prompt = """You are a creative reasoning assistant. Generate multiple distinct approaches to solve problems.

For each approach:
- Take a different angle
- Use different strategies
- Explore alternatives
- Be creative and diverse"""
    
    user_prompt = f"""Problem: {problem}

Generate {num_branches} different reasoning approaches. For each:
1. Describe the approach
2. Outline initial steps
3. Identify key assumptions

Make them diverse and creative."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse thought branches
    thought_branches = []
    lines = response.content.split("\n")
    current_branch = []
    
    for line in lines:
        if line.strip().startswith(("1.", "2.", "3.", "Approach")):
            if current_branch:
                thought_branches.append(" ".join(current_branch))
                current_branch = []
        if line.strip():
            current_branch.append(line.strip())
    
    if current_branch:
        thought_branches.append(" ".join(current_branch))
    
    # Build thought tree structure
    thought_tree = {
        "root": problem,
        "branches": [
            {
                "id": i,
                "thought": branch,
                "score": 0.0,
                "children": []
            }
            for i, branch in enumerate(thought_branches[:num_branches])
        ]
    }
    
    report = f"""
    🌳 Thought Generator:
    
    Tree Exploration:
    • Problem: {problem[:100]}...
    • Search Strategy: {search_strategy.upper()}
    • Branches Generated: {len(thought_branches[:num_branches])}
    
    Tree-of-Thought Concepts:
    
    Core Idea:
    Instead of single reasoning chain, explore multiple paths
    like a search tree, evaluating and selecting best routes.
    
    Search Strategies:
    
    Breadth-First Search (BFS):
    • Explore all branches at current level
    • Then move to next level
    • Guarantees shortest path
    • Good for balanced exploration
    
    Depth-First Search (DFS):
    • Explore one branch fully
    • Backtrack when stuck
    • Memory efficient
    • May miss better paths
    
    Beam Search:
    • Keep top-k best branches
    • Prune poor options
    • Balance breadth and depth
    • Efficient and effective
    
    Best-First Search:
    • Always expand most promising
    • Uses heuristic evaluation
    • Directed exploration
    • Fast convergence
    
    Generated Thought Branches:
    {chr(10).join(f"  Branch {i+1}: {branch[:150]}..." for i, branch in enumerate(thought_branches[:num_branches]))}
    
    ToT vs CoT:
    
    Chain-of-Thought:
    • Single linear path
    • Sequential reasoning
    • Faster
    • Less exploration
    
    Tree-of-Thought:
    • Multiple paths explored
    • Backtracking possible
    • More thorough
    • Better for complex problems
    
    ToT Process:
    
    1. Thought Generation:
    • Generate k diverse thoughts
    • Each represents a step
    • Multiple approaches
    • Creative exploration
    
    2. State Evaluation:
    • Score each thought
    • Assess promise
    • Estimate success probability
    • Rank options
    
    3. Search Algorithm:
    • BFS: explore all equally
    • DFS: deep dive one path
    • Beam: keep top-k
    • A*: use heuristics
    
    4. Deliberate Planning:
    • Lookahead reasoning
    • Compare alternatives
    • Backtrack if needed
    • Find optimal path
    
    Applications:
    
    Game Playing:
    • Chess moves
    • Strategy planning
    • Opponent modeling
    • Position evaluation
    
    Creative Writing:
    • Plot development
    • Character arcs
    • Multiple endings
    • Story branches
    
    Math Proofs:
    • Proof strategies
    • Lemma selection
    • Alternative approaches
    • Verification paths
    
    Code Generation:
    • Algorithm design
    • Data structure choice
    • Optimization paths
    • Refactoring options
    
    Research (Yao et al. 2023):
    
    Game of 24:
    • 74% success rate (vs 4% with CoT)
    • Explores multiple operations
    • Backtracks wrong paths
    • Finds creative solutions
    
    Creative Writing:
    • More coherent plots
    • Better story development
    • Explores alternatives
    • Higher quality output
    
    Crosswords:
    • Better constraint satisfaction
    • Considers word interactions
    • Backtracks conflicts
    • Improved completion rate
    """
    
    return {
        "messages": [AIMessage(content=f"🌳 Thought Generator:\n{report}\n\n{response.content}")],
        "thought_tree": thought_tree
    }


# Path Evaluator
def path_evaluator(state: TreeOfThoughtState) -> TreeOfThoughtState:
    """Evaluates and scores different reasoning paths"""
    thought_tree = state.get("thought_tree", {})
    problem = state.get("problem", "")
    
    branches = thought_tree.get("branches", [])
    
    # Evaluate each branch
    evaluated_branches = []
    
    for branch in branches:
        # Simple evaluation prompt
        eval_prompt = f"""Evaluate this reasoning approach for the problem:

Problem: {problem}

Approach: {branch['thought']}

Rate on scale 1-10:
1. Feasibility (can this work?)
2. Creativity (is this novel?)
3. Completeness (does it cover all aspects?)

Provide: Score (1-10) and brief reasoning."""
        
        messages = [HumanMessage(content=eval_prompt)]
        response = llm.invoke(messages)
        
        # Extract score (simplified parsing)
        score = 5.0  # default
        content = response.content.lower()
        
        # Look for numbers
        import re
        numbers = re.findall(r'\b([1-9]|10)\b', content)
        if numbers:
            scores = [float(n) for n in numbers[:3]]
            score = sum(scores) / len(scores) if scores else 5.0
        
        branch["score"] = score
        branch["evaluation"] = response.content
        evaluated_branches.append(branch)
    
    # Sort by score
    evaluated_branches.sort(key=lambda x: x["score"], reverse=True)
    
    # Select best path
    best_branch = evaluated_branches[0] if evaluated_branches else {}
    best_path = [best_branch.get("thought", "No path found")]
    
    thought_tree["branches"] = evaluated_branches
    
    summary = f"""
    🎯 Path Evaluator:
    
    Evaluation Results:
    • Paths Evaluated: {len(evaluated_branches)}
    • Best Path Score: {best_branch.get('score', 0):.1f}/10
    
    Path Scores:
    {chr(10).join(f"  Path {i+1}: {b['score']:.1f}/10 - {b['thought'][:100]}..." for i, b in enumerate(evaluated_branches))}
    
    Evaluation Strategies:
    
    Value Function:
    • Estimate solution quality
    • Predict success probability
    • Guide search direction
    • Prune bad branches
    
    Criteria:
    • Feasibility: Can this work?
    • Optimality: Is this best?
    • Creativity: Is this novel?
    • Completeness: All aspects covered?
    
    Voting Methods:
    • Generate multiple evaluations
    • Aggregate scores
    • Majority consensus
    • Ensemble judgment
    
    Self-Evaluation:
    • Model rates own thoughts
    • Metacognitive reasoning
    • Confidence scores
    • Uncertainty quantification
    
    ToT Implementation Patterns:
    
    Input-Output (IO):
    ```
    Input: Problem
    Output: Multiple solutions
    Evaluate: Score each
    Select: Best one
    ```
    
    Propose-Evaluate (PE):
    ```
    Propose: Generate k thoughts
    Evaluate: Score each thought
    Select: Top thoughts
    Repeat: Until solution
    ```
    
    Sample-Evaluate (SE):
    ```
    Sample: Random explorations
    Evaluate: Score outcomes
    Backtrack: From dead ends
    Converge: To best path
    ```
    
    Advanced Techniques:
    
    Monte Carlo Tree Search:
    • Random sampling
    • UCB selection
    • Backpropagation
    • Exploration-exploitation
    
    A* Search:
    • Heuristic guidance
    • Cost estimation
    • Optimal pathfinding
    • Admissible heuristics
    
    Iterative Deepening:
    • Depth-limited search
    • Gradually increase depth
    • Memory efficient
    • Complete exploration
    
    Alpha-Beta Pruning:
    • Cut unpromising branches
    • Reduce search space
    • Maintain optimality
    • Faster convergence
    
    Best Path Selected:
    {best_path[0][:200]}...
    
    Implementation Tips:
    
    Thought Diversity:
    • Use higher temperature
    • Different prompts
    • Multiple samples
    • Avoid repetition
    
    Pruning Strategies:
    • Set score threshold
    • Keep top-k only
    • Early stopping
    • Resource limits
    
    Combining Results:
    • Merge best ideas
    • Hybrid approaches
    • Ensemble solutions
    • Multi-path synthesis
    
    Key Insight:
    Tree-of-Thought excels when problems have multiple
    valid approaches and benefit from exploration and
    backtracking - especially creative and strategic tasks.
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Path Evaluator:\n{summary}")],
        "thought_tree": thought_tree,
        "best_path": best_path,
        "final_solution": best_branch.get("thought", "")
    }


# Build the graph
def build_tot_graph():
    """Build the tree-of-thought pattern graph"""
    workflow = StateGraph(TreeOfThoughtState)
    
    workflow.add_node("thought_generator", thought_generator)
    workflow.add_node("path_evaluator", path_evaluator)
    
    workflow.add_edge(START, "thought_generator")
    workflow.add_edge("thought_generator", "path_evaluator")
    workflow.add_edge("path_evaluator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_tot_graph()
    
    print("=== Tree-of-Thought MCP Pattern ===\n")
    
    # Test Case: Creative Problem Solving
    print("\n" + "="*70)
    print("TEST CASE: Creative Problem with Multiple Approaches")
    print("="*70)
    
    state = {
        "messages": [],
        "problem": "Design an innovative way to reduce food waste in restaurants",
        "search_strategy": "beam",
        "num_branches": 3,
        "max_depth": 2,
        "thought_tree": {},
        "best_path": [],
        "final_solution": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 124: Tree-of-Thought - COMPLETE")
    print(f"{'='*70}")

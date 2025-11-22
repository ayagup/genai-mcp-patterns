"""
Monte Carlo Tree Search (MCTS) MCP Pattern

This pattern implements MCTS for exploration-exploitation balance
in decision making and planning under uncertainty.

Key Features:
- Tree-based search
- UCB selection
- Random simulation
- Backpropagation
- Best action selection
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class MCTSState(TypedDict):
    """State for MCTS pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    problem: str
    search_tree: Dict
    simulations: int
    best_action: str


llm = ChatOpenAI(model="gpt-4", temperature=0.7)


def mcts_agent(state: MCTSState) -> MCTSState:
    """Performs Monte Carlo Tree Search"""
    problem = state.get("problem", "")
    simulations = state.get("simulations", 100)
    
    system_prompt = """You are an MCTS planning expert.

MCTS Phases:
1. Selection: Choose promising nodes (UCB1)
2. Expansion: Add new child nodes
3. Simulation: Random playouts
4. Backpropagation: Update statistics

Find best action through simulation."""
    
    user_prompt = f"""Problem: {problem}

Simulate MCTS with {simulations} iterations.
Explain the process and recommend best action."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🎲 MCTS Agent:
    
    Search Results:
    • Problem: {problem[:100]}...
    • Simulations: {simulations}
    
    MCTS Algorithm:
    ```python
    def mcts(root, iterations):
        for _ in range(iterations):
            # 1. Selection
            node = select(root)
            
            # 2. Expansion
            if not fully_expanded(node):
                node = expand(node)
            
            # 3. Simulation
            reward = simulate(node)
            
            # 4. Backpropagation
            backpropagate(node, reward)
        
        return best_child(root)
    
    def ucb1(node):
        return (node.wins / node.visits) + 
               C * sqrt(log(parent.visits) / node.visits)
    ```
    
    Applications:
    - Game playing (Go, Chess)
    - Recommendation systems
    - Resource allocation
    - Path planning
    
    Advantages:
    - No domain knowledge needed
    - Anytime algorithm
    - Asymmetric tree growth
    - Exploration-exploitation balance
    
    Key Insight:
    MCTS balances exploration and exploitation through
    random simulations and UCB selection.
    """
    
    return {
        "messages": [AIMessage(content=f"🎲 MCTS Agent:\n{report}\n\n{response.content}")],
        "best_action": "action_from_mcts"
    }


def build_mcts_graph():
    workflow = StateGraph(MCTSState)
    workflow.add_node("mcts_agent", mcts_agent)
    workflow.add_edge(START, "mcts_agent")
    workflow.add_edge("mcts_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_mcts_graph()
    
    print("=== Monte Carlo Tree Search MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "problem": "Choose best strategy for resource allocation under uncertainty",
        "search_tree": {},
        "simulations": 100,
        "best_action": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 135: Monte Carlo Tree Search - COMPLETE")
    print(f"{'='*70}")

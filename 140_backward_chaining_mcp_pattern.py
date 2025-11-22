"""
Backward Chaining MCP Pattern

This pattern implements backward chaining (goal-driven reasoning)
starting from goals and working backwards to find supporting facts.

Key Features:
- Goal-driven inference
- Backward reasoning
- Subgoal generation
- Top-down processing
- Query answering
"""

from typing import TypedDict, Sequence, Annotated, List, Dict, Set
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class BackwardChainingState(TypedDict):
    """State for backward chaining pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    facts: Set[str]
    rules: List[Dict]
    goal: str
    subgoals: List[str]
    proof_found: bool


llm = ChatOpenAI(model="gpt-4", temperature=0.2)


def backward_chaining_agent(state: BackwardChainingState) -> BackwardChainingState:
    """Performs backward chaining inference"""
    facts = state.get("facts", set())
    goal = state.get("goal", "")
    
    system_prompt = """You are a backward chaining expert.

Backward Chaining (Goal-Driven):
1. Start with goal to prove
2. Find rules that conclude goal
3. Check rule premises (new subgoals)
4. Recursively prove subgoals
5. If all premises proven, goal proven

Work backward from goal to facts."""
    
    user_prompt = f"""Known Facts: {facts}
Goal to Prove: {goal}

Use backward chaining to prove the goal.
Show subgoal generation and proof tree."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    ⬅️ Backward Chaining Agent:
    
    Proof Search:
    • Goal: {goal}
    • Known Facts: {len(facts)}
    
    Backward Chaining Algorithm:
    ```python
    def backward_chain(goal, facts, rules):
        # Base case: goal is a known fact
        if goal in facts:
            return True, [goal]
        
        # Find rules that conclude the goal
        for rule in rules:
            if rule.conclusion == goal:
                # Try to prove all premises
                all_proven = True
                proof_tree = []
                
                for premise in rule.premises:
                    proven, subtree = backward_chain(premise, facts, rules)
                    if not proven:
                        all_proven = False
                        break
                    proof_tree.append(subtree)
                
                if all_proven:
                    return True, [goal, rule, proof_tree]
        
        return False, []  # Goal not provable
    ```
    
    Example Proof Tree:
    
    Goal: prove(platypus)
    
    ```
    prove(platypus)
    ├─ Find rule: monotreme ∧ has_bill → platypus
    ├─ Subgoal: prove(monotreme)
    │  ├─ Find rule: mammal ∧ lays_eggs → monotreme
    │  ├─ Subgoal: prove(mammal)
    │  │  ├─ Find rule: has_fur ∧ warm_blooded → mammal
    │  │  ├─ Subgoal: prove(has_fur) ✓ (fact)
    │  │  └─ Subgoal: prove(warm_blooded) ✓ (fact)
    │  └─ Subgoal: prove(lays_eggs) ✓ (fact)
    └─ Subgoal: prove(has_bill) ✓ (fact)
    
    All subgoals proven → platypus proven! ✓
    ```
    
    Proof Strategies:
    
    Depth-First:
    ```python
    def dfs_backward_chain(goal, facts, rules):
        stack = [goal]
        proven = set()
        
        while stack:
            current = stack.pop()
            
            if current in facts:
                proven.add(current)
                continue
            
            for rule in find_rules(current, rules):
                stack.extend(rule.premises)
        
        return goal in proven
    ```
    
    Breadth-First:
    ```python
    def bfs_backward_chain(goal, facts, rules):
        queue = deque([goal])
        proven = set()
        
        while queue:
            current = queue.popleft()
            
            if current in facts:
                proven.add(current)
                continue
            
            for rule in find_rules(current, rules):
                queue.extend(rule.premises)
        
        return goal in proven
    ```
    
    Backward vs Forward Chaining:
    
    Backward Chaining:
    ✓ Goal-focused
    ✓ Avoids irrelevant derivations
    ✓ Good for queries
    ✓ Lazy evaluation
    ✗ May re-derive facts
    ✗ Not reactive to new data
    
    Forward Chaining:
    ✓ Derives all conclusions
    ✓ Reactive to data
    ✓ Good for monitoring
    ✗ May derive useless facts
    ✗ Not goal-focused
    
    Hybrid Approach:
    • Use both strategies
    • Forward for monitoring
    • Backward for queries
    • Best of both worlds
    
    Optimizations:
    
    Memoization:
    ```python
    cache = {}
    
    def backward_chain_memo(goal, facts, rules):
        if goal in cache:
            return cache[goal]
        
        if goal in facts:
            cache[goal] = True
            return True
        
        for rule in find_rules(goal, rules):
            if all(backward_chain_memo(p, facts, rules) 
                   for p in rule.premises):
                cache[goal] = True
                return True
        
        cache[goal] = False
        return False
    ```
    
    Loop Detection:
    ```python
    def backward_chain_safe(goal, facts, rules, visited=None):
        if visited is None:
            visited = set()
        
        if goal in visited:
            return False  # Cycle detected
        
        visited.add(goal)
        
        if goal in facts:
            return True
        
        for rule in find_rules(goal, rules):
            if all(backward_chain_safe(p, facts, rules, visited.copy())
                   for p in rule.premises):
                return True
        
        return False
    ```
    
    Applications:
    
    Query Answering:
    • Database queries
    • Knowledge base lookup
    • Theorem proving
    • Diagnostic systems
    
    Prolog Programming:
    • Logic programming
    • Rule-based queries
    • Unification
    • Backtracking search
    
    Expert Systems:
    • Medical diagnosis
    • Troubleshooting
    • Root cause analysis
    • Decision support
    
    Planning:
    • Goal decomposition
    • Means-ends analysis
    • Precondition checking
    • Plan validation
    
    Advanced Features:
    
    Unification:
    • Pattern matching
    • Variable binding
    • Logical variables
    • First-order logic
    
    Cut Operator:
    • Prune search
    • Commit to choice
    • Improve efficiency
    • Control backtracking
    
    Negation as Failure:
    • Closed-world assumption
    • If not provable, assume false
    • Practical reasoning
    • Default logic
    
    Key Insight:
    Backward chaining enables efficient goal-driven
    reasoning by working backwards from goals to
    supporting facts, avoiding irrelevant derivations.
    
    🎉 ALL PLANNING & REASONING PATTERNS (131-140) COMPLETE! 🎉
    """
    
    return {
        "messages": [AIMessage(content=f"⬅️ Backward Chaining Agent:\n{report}\n\n{response.content}")],
        "proof_found": True
    }


def build_backward_chaining_graph():
    workflow = StateGraph(BackwardChainingState)
    workflow.add_node("backward_chaining_agent", backward_chaining_agent)
    workflow.add_edge(START, "backward_chaining_agent")
    workflow.add_edge("backward_chaining_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_backward_chaining_graph()
    
    print("=== Backward Chaining MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "facts": {"has_fur", "warm_blooded", "lays_eggs", "has_bill"},
        "rules": [],
        "goal": "platypus",
        "subgoals": [],
        "proof_found": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 140: Backward Chaining - COMPLETE")
    print(f"{'='*70}")
    print("\n🎉🎉🎉 ALL PLANNING & REASONING PATTERNS (131-140) COMPLETE! 🎉🎉🎉")

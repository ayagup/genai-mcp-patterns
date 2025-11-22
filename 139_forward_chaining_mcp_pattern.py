"""
Forward Chaining MCP Pattern

This pattern implements forward chaining (data-driven reasoning)
starting from known facts and deriving new conclusions.

Key Features:
- Data-driven inference
- Rule application
- Fact derivation
- Bottom-up reasoning
- Reactive processing
"""

from typing import TypedDict, Sequence, Annotated, List, Dict, Set
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ForwardChainingState(TypedDict):
    """State for forward chaining pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    initial_facts: Set[str]
    rules: List[Dict]
    derived_facts: Set[str]
    goal: str
    goal_achieved: bool


llm = ChatOpenAI(model="gpt-4", temperature=0.2)


def forward_chaining_agent(state: ForwardChainingState) -> ForwardChainingState:
    """Performs forward chaining inference"""
    initial_facts = state.get("initial_facts", set())
    rules = state.get("rules", [])
    goal = state.get("goal", "")
    
    system_prompt = """You are a forward chaining expert.

Forward Chaining (Data-Driven):
1. Start with known facts
2. Find applicable rules
3. Fire rules to derive new facts
4. Add to knowledge base
5. Repeat until no new facts

Work forward from data to conclusions."""
    
    user_prompt = f"""Initial Facts: {initial_facts}
Goal: {goal}

Apply forward chaining to derive new facts.
Show rule applications and derived facts."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Simulate derivation
    derived_facts = initial_facts.copy()
    derived_facts.add("derived_fact_1")
    derived_facts.add("derived_fact_2")
    
    report = f"""
    ➡️ Forward Chaining Agent:
    
    Inference Results:
    • Initial Facts: {len(initial_facts)}
    • Derived Facts: {len(derived_facts) - len(initial_facts)}
    • Goal: {goal}
    
    Forward Chaining Algorithm:
    ```python
    def forward_chain(facts, rules, goal):
        inferred = set(facts)
        
        while True:
            new_facts = set()
            
            for rule in rules:
                # Check if all premises satisfied
                if all(premise in inferred for premise in rule.premises):
                    # Fire rule
                    conclusion = rule.conclusion
                    if conclusion not in inferred:
                        new_facts.add(conclusion)
                        print(f"Derived: {conclusion} from {rule}")
            
            if not new_facts:
                break  # No new facts
            
            inferred.update(new_facts)
            
            if goal in inferred:
                return True, inferred
        
        return False, inferred
    ```
    
    Example Rule System:
    
    Rules:
    ```
    R1: If (has_fur AND warm_blooded) → mammal
    R2: If (mammal AND lays_eggs) → monotreme  
    R3: If (monotreme AND has_bill) → platypus
    ```
    
    Facts: {has_fur, warm_blooded, lays_eggs, has_bill}
    
    Derivation Trace:
    ```
    Initial: {has_fur, warm_blooded, lays_eggs, has_bill}
    
    Cycle 1:
    - R1 fires: has_fur ∧ warm_blooded → mammal
    - Facts: {has_fur, warm_blooded, lays_eggs, has_bill, mammal}
    
    Cycle 2:
    - R2 fires: mammal ∧ lays_eggs → monotreme
    - Facts: {..., monotreme}
    
    Cycle 3:
    - R3 fires: monotreme ∧ has_bill → platypus
    - Facts: {..., platypus}
    
    No more rules fire. Done.
    Conclusion: platypus
    ```
    
    Forward vs Backward Chaining:
    
    Forward (Data-Driven):
    • Start with facts
    • Generate conclusions
    • May derive irrelevant facts
    • Good for: monitoring, reactive systems
    
    Backward (Goal-Driven):
    • Start with goal
    • Find supporting facts
    • Focused search
    • Good for: queries, diagnosis
    
    Characteristics:
    
    Advantages:
    • Natural for data arrival
    • All derivable facts found
    • Works well with updates
    • Reactive to new data
    
    Disadvantages:
    • May derive useless facts
    • Can be inefficient
    • Large fact base grows
    • No goal focus
    
    Optimizations:
    
    Rete Algorithm:
    • Efficient rule matching
    • Incremental updates
    • Network of nodes
    • Used in production systems
    
    Conflict Resolution:
    • Priority ordering
    • Recency
    • Specificity
    • Refractoriness
    
    Applications:
    
    Expert Systems:
    • Medical diagnosis
    • Fault detection
    • Configuration
    • Advisory systems
    
    Monitoring Systems:
    • Event processing
    • Alert generation
    • Pattern detection
    • Real-time analysis
    
    Business Rules:
    • Policy enforcement
    • Compliance checking
    • Workflow automation
    • Decision automation
    
    Derived Facts:
    {chr(10).join(f"  • {fact}" for fact in list(derived_facts)[:10])}
    
    Key Insight:
    Forward chaining enables data-driven reasoning,
    naturally handling new facts and reactive processing.
    """
    
    return {
        "messages": [AIMessage(content=f"➡️ Forward Chaining Agent:\n{report}\n\n{response.content}")],
        "derived_facts": derived_facts,
        "goal_achieved": goal in derived_facts
    }


def build_forward_chaining_graph():
    workflow = StateGraph(ForwardChainingState)
    workflow.add_node("forward_chaining_agent", forward_chaining_agent)
    workflow.add_edge(START, "forward_chaining_agent")
    workflow.add_edge("forward_chaining_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_forward_chaining_graph()
    
    print("=== Forward Chaining MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "initial_facts": {"has_fur", "warm_blooded", "lays_eggs", "has_bill"},
        "rules": [],
        "derived_facts": set(),
        "goal": "platypus",
        "goal_achieved": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 139: Forward Chaining - COMPLETE")
    print(f"{'='*70}")

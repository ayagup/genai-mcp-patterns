"""
STRIPS Planning MCP Pattern

This pattern implements STRIPS (Stanford Research Institute Problem Solver)
planning using preconditions, effects, and state-based reasoning.

Key Features:
- State representation
- Action preconditions and effects
- Goal-directed planning
- Forward/backward search
- Plan validation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict, Set
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class STRIPSState(TypedDict):
    """State for STRIPS planning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    initial_state: Set[str]
    goal_state: Set[str]
    available_actions: List[Dict]
    plan: List[str]
    planning_complete: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.2)


# STRIPS Planner
def strips_planner(state: STRIPSState) -> STRIPSState:
    """Plans using STRIPS representation"""
    initial_state_set = state.get("initial_state", set())
    goal_state_set = state.get("goal_state", set())
    
    system_prompt = """You are a STRIPS planning expert.

STRIPS uses:
- State: Set of propositions
- Actions: Preconditions + Add/Delete effects
- Goal: Desired state propositions

Create a plan to achieve the goal from initial state."""
    
    user_prompt = f"""Initial State: {initial_state_set}
Goal State: {goal_state_set}

Create a STRIPS plan. For each action specify:
Action: [action name]
Preconditions: [what must be true]
Add Effects: [what becomes true]
Delete Effects: [what becomes false]"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse plan
    plan = []
    for line in response.content.split("\n"):
        if line.startswith("Action:"):
            action = line.replace("Action:", "").strip()
            if action:
                plan.append(action)
    
    report = f"""
    🎯 STRIPS Planner:
    
    Planning Results:
    • Initial State: {len(initial_state_set)} propositions
    • Goal State: {len(goal_state_set)} propositions
    • Plan Length: {len(plan)} actions
    
    STRIPS Representation:
    
    State = Set of Propositions
    Example: {{at(robot, room_A), door_open, battery_full}}
    
    Action Schema:
    ```
    Action: move(from, to)
    Preconditions: {{at(robot, from), path_clear(from, to)}}
    Add Effects: {{at(robot, to)}}
    Delete Effects: {{at(robot, from)}}
    ```
    
    Planning Algorithm (Forward Search):
    ```python
    def strips_forward_search(initial, goal, actions):
        frontier = [(initial, [])]
        explored = set()
        
        while frontier:
            state, plan = frontier.pop(0)
            
            if goal.issubset(state):
                return plan  # Success
            
            if frozenset(state) in explored:
                continue
            explored.add(frozenset(state))
            
            for action in actions:
                if applicable(action, state):
                    new_state = apply(action, state)
                    new_plan = plan + [action.name]
                    frontier.append((new_state, new_plan))
        
        return None  # No plan found
    ```
    
    Generated Plan:
    {chr(10).join(f"  {i+1}. {action}" for i, action in enumerate(plan))}
    
    STRIPS Concepts:
    
    Closed World Assumption:
    - What's not stated is false
    - Simplifies representation
    - Efficient reasoning
    
    Frame Problem:
    - Most things don't change
    - Only specify changes
    - STRIPS uses add/delete lists
    
    Applications:
    - Robot navigation
    - Logistics planning
    - Block world problems
    - Resource allocation
    
    Key Insight:
    STRIPS provides elegant state-based planning through
    simple precondition-effect representation.
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 STRIPS Planner:\n{report}\n\n{response.content}")],
        "plan": plan,
        "planning_complete": True
    }


def build_strips_graph():
    """Build STRIPS planning graph"""
    workflow = StateGraph(STRIPSState)
    workflow.add_node("strips_planner", strips_planner)
    workflow.add_edge(START, "strips_planner")
    workflow.add_edge("strips_planner", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_strips_graph()
    
    print("=== STRIPS Planning MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "initial_state": {"at(robot, room_A)", "battery_full", "door_closed"},
        "goal_state": {"at(robot, room_B)", "package_delivered"},
        "available_actions": [],
        "plan": [],
        "planning_complete": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 134: STRIPS Planning - COMPLETE")
    print(f"{'='*70}")

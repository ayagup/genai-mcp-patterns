"""
Goal-Oriented MCP Pattern

This pattern implements goal-based planning where agents work backwards from
desired goals to determine necessary actions and subgoals.

Key Features:
- Goal definition and decomposition
- Backward reasoning from goals
- Subgoal identification
- Action planning
- Goal achievement tracking
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class GoalOrientedState(TypedDict):
    """State for goal-oriented pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    main_goal: str
    subgoals: List[Dict]
    actions: List[Dict]
    current_state: Dict
    goal_achieved: bool
    plan: List[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)


# Goal Analyzer
def goal_analyzer(state: GoalOrientedState) -> GoalOrientedState:
    """Analyzes main goal and decomposes into subgoals"""
    main_goal = state.get("main_goal", "")
    current_state_dict = state.get("current_state", {})
    
    system_prompt = """You are a goal analysis expert. Break down high-level goals into achievable subgoals.

For each goal:
1. Understand the desired end state
2. Identify necessary preconditions
3. Decompose into logical subgoals
4. Order subgoals by dependency
5. Ensure completeness

Use goal decomposition principles."""
    
    user_prompt = f"""Main Goal: {main_goal}

Current State: {current_state_dict}

Analyze this goal and identify subgoals needed to achieve it.

Format:
Subgoal 1: [description]
Preconditions: [what's needed]
Success Criteria: [how to verify]

Subgoal 2: ...
(and so on)"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse subgoals
    subgoals = []
    current_subgoal = {}
    
    for line in content.split("\n"):
        if line.startswith("Subgoal"):
            if current_subgoal:
                subgoals.append(current_subgoal)
            parts = line.split(":", 1)
            current_subgoal = {
                "description": parts[1].strip() if len(parts) > 1 else "",
                "preconditions": [],
                "success_criteria": "",
                "achieved": False
            }
        elif line.startswith("Preconditions:") and current_subgoal:
            current_subgoal["preconditions"] = [p.strip() for p in line.replace("Preconditions:", "").split(",")]
        elif line.startswith("Success Criteria:") and current_subgoal:
            current_subgoal["success_criteria"] = line.replace("Success Criteria:", "").strip()
    
    if current_subgoal:
        subgoals.append(current_subgoal)
    
    report = f"""
    🎯 Goal Analyzer:
    
    Goal Analysis:
    • Main Goal: {main_goal}
    • Subgoals Identified: {len(subgoals)}
    • Current State: {current_state_dict}
    
    Goal-Oriented Planning Concepts:
    
    Core Principles:
    
    Goal Definition:
    • Clear desired outcome
    • Measurable success criteria
    • Achievable and realistic
    • Time-bound when appropriate
    • Well-specified end state
    
    Goal Decomposition:
    • Break into subgoals
    • Identify dependencies
    • Logical ordering
    • Manageable chunks
    • Hierarchical structure
    
    Backward Planning:
    • Start from goal
    • Work backwards
    • Identify prerequisites
    • Find necessary steps
    • Chain to current state
    
    Goal Types:
    
    Achievement Goals:
    • Reach specific state
    • Accomplish task
    • Attain condition
    • Example: "Write report"
    
    Maintenance Goals:
    • Preserve state
    • Keep condition
    • Sustain level
    • Example: "Keep system online"
    
    Prevention Goals:
    • Avoid state
    • Prevent condition
    • Block outcome
    • Example: "Prevent errors"
    
    Optimization Goals:
    • Maximize value
    • Minimize cost
    • Optimize metric
    • Example: "Minimize time"
    
    Identified Subgoals:
    {chr(10).join(f"  {i+1}. {sg.get('description', '')[:100]}..." for i, sg in enumerate(subgoals))}
    
    Goal Decomposition Strategies:
    
    Temporal Decomposition:
    • Sequential subgoals
    • Time-ordered steps
    • Phase-based breakdown
    • Progressive achievement
    
    Example:
    Goal: "Launch product"
    → Develop MVP
    → Test with users
    → Gather feedback
    → Refine product
    → Marketing campaign
    → Launch event
    
    Functional Decomposition:
    • By capability
    • By component
    • By subsystem
    • Independent modules
    
    Example:
    Goal: "Build app"
    → Design UI
    → Implement backend
    → Create database
    → Add authentication
    → Deploy infrastructure
    
    Resource Decomposition:
    • By resource type
    • By team
    • By expertise
    • Parallel workstreams
    
    Example:
    Goal: "Complete project"
    → Development team: Code
    → Design team: UI/UX
    → QA team: Testing
    → DevOps: Infrastructure
    
    Hierarchical Decomposition:
    • Multiple levels
    • Parent-child goals
    • Tree structure
    • Recursive breakdown
    
    Example:
    Goal: "Improve customer satisfaction"
    → Enhance product quality
      → Better UX
        → User research
        → Design iteration
      → Fewer bugs
        → More testing
        → Code reviews
    → Faster support
      → Hire staff
      → Better tools
    
    Goal-Oriented vs Other Planning:
    
    Goal-Oriented:
    • Start with desired end state
    • Work backwards
    • Focus on objectives
    • Flexible means
    
    Procedural:
    • Start with actions
    • Work forwards
    • Focus on steps
    • Fixed procedures
    
    Opportunistic:
    • Start with resources
    • Explore possibilities
    • Focus on capabilities
    • Adaptive approach
    
    Benefits of Goal-Oriented Planning:
    
    Clarity:
    • Clear objectives
    • Defined success
    • Focused effort
    • Measurable progress
    
    Flexibility:
    • Multiple paths to goal
    • Alternative strategies
    • Adaptable plans
    • Creative solutions
    
    Motivation:
    • Purpose-driven
    • Progress visible
    • Achievement-focused
    • Meaningful work
    
    Efficiency:
    • Avoid unnecessary work
    • Direct path finding
    • Resource optimization
    • Priority-driven
    
    Goal Representation:
    
    Propositional:
    ```
    Goal: at(robot, location_B)
    Current: at(robot, location_A)
    Subgoal: path_clear(A, B)
    ```
    
    First-Order Logic:
    ```
    Goal: ∀x (package(x) → delivered(x))
    Current: package(p1) ∧ ¬delivered(p1)
    Subgoal: in_truck(p1)
    ```
    
    State-Based:
    ```
    Goal State: {temperature: 72, humidity: 45}
    Current State: {temperature: 68, humidity: 50}
    Subgoals: [adjust_temp(72), adjust_humidity(45)]
    ```
    
    Constraint-Based:
    ```
    Goal: minimize(cost) ∧ maximize(quality)
    Constraints: budget < 1000, time < 30_days
    Subgoals: optimize_design, efficient_implementation
    ```
    
    Goal Achievement Criteria:
    
    Completeness:
    • All aspects achieved
    • No missing pieces
    • Full satisfaction
    
    Correctness:
    • Right outcome
    • Meets specifications
    • No errors
    
    Optimality:
    • Best solution
    • Minimal resources
    • Maximum value
    
    Timeliness:
    • Within deadline
    • Appropriate timing
    • Not too early/late
    
    Research & Applications:
    
    STRIPS Planning:
    • Classic goal-oriented
    • Precondition-effect model
    • Backward search
    • Plan construction
    
    Goal-Directed Agents:
    • BDI architecture
    • Beliefs, Desires, Intentions
    • Goal adoption
    • Plan selection
    
    Use Cases:
    • Task planning
    • Project management
    • Problem solving
    • Strategic planning
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Goal Analyzer:\n{report}\n\n{response.content}")],
        "subgoals": subgoals
    }


# Action Planner
def action_planner(state: GoalOrientedState) -> GoalOrientedState:
    """Plans actions to achieve subgoals"""
    main_goal = state.get("main_goal", "")
    subgoals = state.get("subgoals", [])
    current_state_dict = state.get("current_state", {})
    
    system_prompt = """You are an action planning expert. Create concrete action plans to achieve goals.

For each subgoal:
1. Identify necessary actions
2. Determine action sequence
3. Specify parameters
4. Consider constraints
5. Verify feasibility"""
    
    # Build subgoals context
    subgoals_text = "\n".join(
        f"{i+1}. {sg.get('description', '')}" 
        for i, sg in enumerate(subgoals)
    )
    
    user_prompt = f"""Main Goal: {main_goal}

Subgoals:
{subgoals_text}

Current State: {current_state_dict}

Create an action plan to achieve these subgoals.

Format:
Action 1: [action name]
Target: [which subgoal]
Steps: [specific steps]
Expected Outcome: [result]

Action 2: ...
(and so on)"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse actions
    actions = []
    current_action = {}
    
    for line in content.split("\n"):
        if line.startswith("Action"):
            if current_action:
                actions.append(current_action)
            parts = line.split(":", 1)
            current_action = {
                "name": parts[1].strip() if len(parts) > 1 else "",
                "target": "",
                "steps": [],
                "expected_outcome": "",
                "completed": False
            }
        elif line.startswith("Target:") and current_action:
            current_action["target"] = line.replace("Target:", "").strip()
        elif line.startswith("Steps:") and current_action:
            current_action["steps"] = [line.replace("Steps:", "").strip()]
        elif line.startswith("Expected Outcome:") and current_action:
            current_action["expected_outcome"] = line.replace("Expected Outcome:", "").strip()
    
    if current_action:
        actions.append(current_action)
    
    # Create plan summary
    plan = [f"{a.get('name', '')}: {a.get('expected_outcome', '')}" for a in actions]
    
    summary = f"""
    📋 Action Planner:
    
    Planning Results:
    • Total Actions: {len(actions)}
    • Target Subgoals: {len(subgoals)}
    • Plan Created: Yes
    
    Action Plan:
    {chr(10).join(f"  {i+1}. {a.get('name', '')}" for i, a in enumerate(actions))}
    
    Goal-Oriented Action Planning:
    
    Action Selection Principles:
    
    Necessity:
    • Required for goal
    • No redundant actions
    • Essential steps only
    • Minimal sufficient set
    
    Sufficiency:
    • Complete coverage
    • Achieve all subgoals
    • No gaps
    • Full path to goal
    
    Efficiency:
    • Shortest path
    • Minimize resources
    • Optimize time
    • Reduce complexity
    
    Feasibility:
    • Executable actions
    • Available resources
    • Within constraints
    • Realistic assumptions
    
    Planning Algorithms:
    
    Forward Planning:
    ```python
    state = current_state
    plan = []
    while not goal_satisfied(state, goal):
        action = select_applicable_action(state)
        plan.append(action)
        state = apply(action, state)
    return plan
    ```
    
    Backward Planning:
    ```python
    plan = []
    subgoals = [goal]
    while subgoals:
        subgoal = subgoals.pop()
        if not satisfied(current_state, subgoal):
            action = find_achieving_action(subgoal)
            plan.insert(0, action)
            subgoals.extend(preconditions(action))
    return plan
    ```
    
    Hierarchical Planning:
    ```python
    def plan(goal, level):
        if is_primitive(goal):
            return [action_for(goal)]
        else:
            subgoals = decompose(goal)
            plan = []
            for subgoal in subgoals:
                plan.extend(plan(subgoal, level+1))
            return plan
    ```
    
    Partial-Order Planning:
    ```python
    # Actions with constraints, not total order
    plan = PartialPlan()
    plan.add_action(start)
    plan.add_action(goal)
    
    while plan.has_flaws():
        flaw = plan.select_flaw()
        resolvers = plan.find_resolvers(flaw)
        resolver = choose(resolvers)
        plan.add_resolver(resolver)
    
    return plan.linearize()
    ```
    
    Plan Qualities:
    
    Completeness:
    • Achieves all goals
    • No missing steps
    • Full solution
    • Covers all cases
    
    Correctness:
    • Valid actions
    • Proper ordering
    • No conflicts
    • Satisfies constraints
    
    Optimality:
    • Best path
    • Minimal cost
    • Maximum value
    • Pareto efficient
    
    Robustness:
    • Handles uncertainty
    • Error recovery
    • Adaptive
    • Fault tolerant
    
    Plan Execution Strategies:
    
    Deterministic Execution:
    • Fixed sequence
    • No branching
    • Predictable
    • Simple control
    
    Conditional Execution:
    • If-then branches
    • Context-dependent
    • Runtime decisions
    • Flexible adaptation
    
    Reactive Execution:
    • Sense-act loop
    • Environment feedback
    • Online replanning
    • Dynamic adjustment
    
    Deliberative Execution:
    • Look-ahead reasoning
    • Anticipate issues
    • Proactive planning
    • Strategic thinking
    
    Goal Monitoring:
    
    Progress Tracking:
    • Measure advancement
    • Milestones achieved
    • Percent complete
    • Time remaining
    
    Success Verification:
    • Check criteria
    • Validate outcome
    • Confirm achievement
    • Quality assessment
    
    Failure Detection:
    • Monitor deviations
    • Detect problems
    • Identify blockers
    • Early warning
    
    Adaptive Replanning:
    • Update plan
    • Find alternatives
    • Recover from failure
    • Learn from experience
    
    Best Practices:
    
    Clear Goals:
    • Specific outcomes
    • Measurable criteria
    • Achievable targets
    • Relevant objectives
    
    Structured Decomposition:
    • Logical breakdown
    • Manageable pieces
    • Clear dependencies
    • Appropriate granularity
    
    Flexible Planning:
    • Multiple strategies
    • Contingency plans
    • Adaptive approach
    • Open to revision
    
    Continuous Monitoring:
    • Track progress
    • Verify assumptions
    • Detect issues
    • Update as needed
    
    Key Insight:
    Goal-oriented planning enables purposeful, efficient action
    by working backward from desired outcomes to identify
    necessary subgoals and actions, creating flexible yet
    focused plans for achievement.
    """
    
    return {
        "messages": [AIMessage(content=f"📋 Action Planner:\n{summary}")],
        "actions": actions,
        "plan": plan
    }


# Build the graph
def build_goal_oriented_graph():
    """Build the goal-oriented pattern graph"""
    workflow = StateGraph(GoalOrientedState)
    
    workflow.add_node("goal_analyzer", goal_analyzer)
    workflow.add_node("action_planner", action_planner)
    
    workflow.add_edge(START, "goal_analyzer")
    workflow.add_edge("goal_analyzer", "action_planner")
    workflow.add_edge("action_planner", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_goal_oriented_graph()
    
    print("=== Goal-Oriented MCP Pattern ===\n")
    
    # Test Case: Project completion goal
    print("\n" + "="*70)
    print("TEST CASE: Goal-Oriented Planning for Project Completion")
    print("="*70)
    
    state = {
        "messages": [],
        "main_goal": "Complete and launch a new mobile app for task management",
        "subgoals": [],
        "actions": [],
        "current_state": {
            "team": "assembled",
            "requirements": "defined",
            "design": "not started",
            "development": "not started",
            "testing": "not started",
            "deployment": "not ready"
        },
        "goal_achieved": False,
        "plan": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 131: Goal-Oriented - COMPLETE")
    print(f"{'='*70}")

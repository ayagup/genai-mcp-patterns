"""
Plan-Execute MCP Pattern

This pattern implements a two-phase approach where planning and execution
are separated, allowing for robust planning followed by monitored execution.

Key Features:
- Separate planning and execution phases
- Plan creation and validation
- Execution monitoring
- Replanning on failure
- Progress tracking
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class PlanExecuteState(TypedDict):
    """State for plan-execute pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    plan: List[Dict]
    execution_log: List[Dict]
    current_step: int
    execution_status: str  # "planning", "executing", "completed", "failed"
    replanning_needed: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)


# Planner Agent
def planner_agent(state: PlanExecuteState) -> PlanExecuteState:
    """Creates comprehensive plan for the task"""
    task = state.get("task", "")
    replanning_needed = state.get("replanning_needed", False)
    execution_log = state.get("execution_log", [])
    
    # Build context from execution if replanning
    execution_context = ""
    if replanning_needed and execution_log:
        execution_context = "\n\nExecution History:\n"
        for entry in execution_log:
            execution_context += f"  Step {entry.get('step', 0)}: {entry.get('action', '')} - {entry.get('result', '')}\n"
    
    system_prompt = """You are an expert planner. Create detailed, executable plans.

For each plan:
1. Break down into discrete steps
2. Specify clear actions
3. Define success criteria
4. Identify dependencies
5. Consider edge cases
6. Estimate effort

Create robust, practical plans."""
    
    user_prompt = f"""Task: {task}{execution_context}

Create a {'new ' if replanning_needed else ''}detailed execution plan.

Format each step as:
Step N:
Action: [what to do]
Expected Outcome: [what should result]
Success Criteria: [how to verify]
Dependencies: [what's needed first]"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse plan
    plan = []
    current_step = {}
    
    for line in content.split("\n"):
        if line.startswith("Step"):
            if current_step:
                plan.append(current_step)
            current_step = {
                "step_number": len(plan) + 1,
                "action": "",
                "expected_outcome": "",
                "success_criteria": "",
                "dependencies": [],
                "status": "pending"
            }
        elif line.startswith("Action:") and current_step:
            current_step["action"] = line.replace("Action:", "").strip()
        elif line.startswith("Expected Outcome:") and current_step:
            current_step["expected_outcome"] = line.replace("Expected Outcome:", "").strip()
        elif line.startswith("Success Criteria:") and current_step:
            current_step["success_criteria"] = line.replace("Success Criteria:", "").strip()
        elif line.startswith("Dependencies:") and current_step:
            deps = line.replace("Dependencies:", "").strip()
            current_step["dependencies"] = [d.strip() for d in deps.split(",") if d.strip()]
    
    if current_step:
        plan.append(current_step)
    
    report = f"""
    📝 Planner Agent:
    
    Planning Results:
    • Task: {task[:100]}...
    • Steps Planned: {len(plan)}
    • Replanning: {replanning_needed}
    
    Plan-Execute Pattern:
    
    Core Concept:
    Separate planning (thinking) from execution (doing),
    allowing for thorough planning and robust execution
    with monitoring and adaptation.
    
    Two-Phase Approach:
    
    Phase 1 - Planning:
    • Analyze task
    • Create detailed plan
    • Validate feasibility
    • Identify resources
    • Estimate effort
    
    Phase 2 - Execution:
    • Execute step by step
    • Monitor progress
    • Verify success
    • Handle failures
    • Update status
    
    Benefits:
    
    Separation of Concerns:
    • Planning vs execution
    • Think then act
    • Clear boundaries
    • Focused phases
    
    Robustness:
    • Validated plans
    • Monitored execution
    • Error detection
    • Recovery mechanisms
    
    Visibility:
    • Clear progress
    • Explicit steps
    • Measurable outcomes
    • Trackable status
    
    Adaptability:
    • Replan when needed
    • Dynamic adjustment
    • Failure recovery
    • Learn from execution
    
    Generated Plan:
    {chr(10).join(f"  Step {s.get('step_number', 0)}: {s.get('action', '')[:80]}..." for s in plan)}
    
    Planning Strategies:
    
    Top-Down Planning:
    • Start with high-level
    • Decompose iteratively
    • Refine details
    • Hierarchical structure
    
    Bottom-Up Planning:
    • Start with primitives
    • Compose upward
    • Build complexity
    • Emergent structure
    
    Mixed-Initiative:
    • Human + AI collaboration
    • Interactive planning
    • Feedback integration
    • Shared control
    
    Contingency Planning:
    • Identify risks
    • Plan alternatives
    • Backup strategies
    • If-then branches
    
    Plan Validation:
    
    Completeness Check:
    ```python
    def validate_completeness(plan, task):
        # All task aspects covered?
        required = extract_requirements(task)
        covered = extract_coverage(plan)
        return all(r in covered for r in required)
    ```
    
    Consistency Check:
    ```python
    def validate_consistency(plan):
        # No conflicting steps?
        for step in plan:
            if conflicts(step, other_steps):
                return False
        return True
    ```
    
    Feasibility Check:
    ```python
    def validate_feasibility(plan, resources):
        # Can we execute this?
        for step in plan:
            if not has_capability(step):
                return False
            if not has_resources(step, resources):
                return False
        return True
    ```
    
    Dependency Check:
    ```python
    def validate_dependencies(plan):
        # Proper ordering?
        completed = set()
        for step in plan:
            if not all(d in completed for d in step.dependencies):
                return False
            completed.add(step.id)
        return True
    ```
    
    Plan Representation:
    
    Linear Plan:
    ```
    [Action1 → Action2 → Action3 → Goal]
    Simple sequence, deterministic
    ```
    
    Conditional Plan:
    ```
    Action1 →
      if success: Action2
      if failure: Action2_alt
    → Goal
    Handles contingencies
    ```
    
    Hierarchical Plan:
    ```
    High-Level: [Task1, Task2, Task3]
    Task1: [Subtask1.1, Subtask1.2]
    Task2: [Subtask2.1, Subtask2.2]
    Multiple abstraction levels
    ```
    
    Partial-Order Plan:
    ```
    {Action1, Action2} can run parallel
    Action3 requires Action1, Action2
    Flexible ordering
    ```
    
    Research & Applications:
    
    LangChain PlanAndExecute:
    • Built-in pattern
    • Planner + Executor agents
    • Automatic replanning
    • Tool integration
    
    Classical Planning:
    • STRIPS planners
    • GraphPlan
    • FF planner
    • Fast-Forward search
    
    Use Cases:
    • Task automation
    • Project management
    • Workflow orchestration
    • Multi-step reasoning
    
    Planning Algorithms:
    
    Forward Search:
    • Start from initial state
    • Apply actions
    • Search for goal
    • Breadth/depth first
    
    Backward Search:
    • Start from goal
    • Find achieving actions
    • Work to initial state
    • Regression planning
    
    Heuristic Search:
    • A* algorithm
    • Guided exploration
    • Cost estimation
    • Optimal paths
    
    HTN Planning:
    • Task decomposition
    • Method selection
    • Hierarchical
    • Domain-specific
    """
    
    return {
        "messages": [AIMessage(content=f"📝 Planner Agent:\n{report}\n\n{response.content}")],
        "plan": plan,
        "execution_status": "planning",
        "current_step": 0
    }


# Executor Agent
def executor_agent(state: PlanExecuteState) -> PlanExecuteState:
    """Executes the plan step by step"""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    execution_log = state.get("execution_log", [])
    
    if current_step >= len(plan):
        # All steps completed
        summary = f"""
        ✅ Executor Agent - Execution Complete
        
        Final Status:
        • Total Steps: {len(plan)}
        • Completed Steps: {current_step}
        • Execution Status: Completed
        
        Execution Summary:
        {chr(10).join(f"  ✓ Step {i+1}: {entry.get('action', '')}" for i, entry in enumerate(execution_log))}
        
        Plan-Execute Pattern Complete!
        """
        
        return {
            "messages": [AIMessage(content=summary)],
            "execution_status": "completed"
        }
    
    # Execute current step
    step = plan[current_step]
    action = step.get("action", "")
    expected_outcome = step.get("expected_outcome", "")
    success_criteria = step.get("success_criteria", "")
    
    # Simulate execution (in real implementation, actually execute)
    system_prompt = f"""You are executing a plan step. Simulate the execution and provide results.

Step to Execute:
Action: {action}
Expected Outcome: {expected_outcome}
Success Criteria: {success_criteria}

Simulate execution and report:
1. What happened
2. Was it successful
3. Any issues
4. Actual outcome"""
    
    user_prompt = f"Execute this step and report results."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Determine if step succeeded (simplified)
    execution_result = response.content
    success = "success" in execution_result.lower() or "completed" in execution_result.lower()
    
    # Log execution
    log_entry = {
        "step": current_step + 1,
        "action": action,
        "result": execution_result[:200],
        "success": success
    }
    
    report = f"""
    ⚙️ Executor Agent (Step {current_step + 1}/{len(plan)}):
    
    Execution Status:
    • Current Step: {current_step + 1}
    • Action: {action}
    • Expected: {expected_outcome}
    • Success: {success}
    
    Execution Result:
    {execution_result[:300]}...
    
    Execution Monitoring:
    
    Progress Tracking:
    • Steps completed: {current_step}/{len(plan)}
    • Current action: {action[:50]}...
    • Status: {'✅ Success' if success else '⚠️ Needs attention'}
    
    Execution Patterns:
    
    Sequential Execution:
    ```python
    for step in plan:
        result = execute(step)
        log(step, result)
        if not result.success:
            handle_failure(step, result)
    ```
    
    Monitored Execution:
    ```python
    for step in plan:
        # Pre-execution checks
        if not preconditions_met(step):
            replan()
            continue
        
        # Execute with monitoring
        result = execute_with_monitoring(step)
        
        # Post-execution verification
        if verify_success(step, result):
            mark_complete(step)
        else:
            handle_failure(step, result)
    ```
    
    Parallel Execution:
    ```python
    independent_steps = identify_parallel(plan)
    results = execute_parallel(independent_steps)
    
    for result in results:
        if not result.success:
            handle_failure(result.step, result)
    ```
    
    Adaptive Execution:
    ```python
    for step in plan:
        result = execute(step)
        
        if result.unexpected:
            # Adapt plan
            new_plan = replan(plan, result)
            plan = new_plan
        
        if not result.success:
            # Try alternative
            alt_step = find_alternative(step)
            result = execute(alt_step)
    ```
    
    Execution Monitoring Aspects:
    
    Success Verification:
    • Check criteria
    • Validate outcome
    • Confirm completion
    • Quality assessment
    
    Failure Detection:
    • Monitor errors
    • Detect deviations
    • Identify blockers
    • Exception handling
    
    Resource Tracking:
    • Monitor usage
    • Check limits
    • Optimize allocation
    • Prevent exhaustion
    
    Time Management:
    • Track duration
    • Check deadlines
    • Adjust priorities
    • Optimize schedule
    
    Error Handling Strategies:
    
    Retry:
    ```python
    for attempt in range(max_retries):
        result = execute(step)
        if result.success:
            break
        wait(backoff_time)
    ```
    
    Fallback:
    ```python
    result = execute(step)
    if not result.success:
        result = execute(fallback_step)
    ```
    
    Replan:
    ```python
    result = execute(step)
    if not result.success:
        new_plan = create_alternative_plan()
        execute_plan(new_plan)
    ```
    
    Skip and Continue:
    ```python
    result = execute(step)
    if not result.success and step.optional:
        log_warning(step)
        continue_execution()
    ```
    
    Execution Best Practices:
    
    Pre-execution Validation:
    • Check preconditions
    • Verify resources
    • Confirm readiness
    • Validate inputs
    
    During Execution:
    • Monitor progress
    • Log actions
    • Track metrics
    • Alert on issues
    
    Post-execution Verification:
    • Check outcomes
    • Validate results
    • Verify criteria
    • Update status
    
    Continuous Learning:
    • Collect feedback
    • Analyze failures
    • Improve plans
    • Update strategies
    
    Next: Step {current_step + 2 if current_step + 1 < len(plan) else 'Complete'}
    """
    
    return {
        "messages": [AIMessage(content=f"⚙️ Executor Agent:\n{report}")],
        "execution_log": execution_log + [log_entry],
        "current_step": current_step + 1,
        "execution_status": "completed" if current_step + 1 >= len(plan) else "executing",
        "replanning_needed": not success
    }


# Build the graph
def build_plan_execute_graph():
    """Build the plan-execute pattern graph"""
    workflow = StateGraph(PlanExecuteState)
    
    workflow.add_node("planner_agent", planner_agent)
    workflow.add_node("executor_agent", executor_agent)
    
    # Conditional routing
    def should_continue(state: PlanExecuteState) -> str:
        """Determine next step"""
        status = state.get("execution_status", "planning")
        replanning = state.get("replanning_needed", False)
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        
        if status == "completed":
            return "end"
        elif replanning:
            return "replan"
        elif current_step < len(plan):
            return "execute"
        else:
            return "end"
    
    workflow.add_edge(START, "planner_agent")
    
    workflow.add_conditional_edges(
        "planner_agent",
        lambda s: "execute",
        {"execute": "executor_agent"}
    )
    
    workflow.add_conditional_edges(
        "executor_agent",
        should_continue,
        {
            "execute": "executor_agent",
            "replan": "planner_agent",
            "end": END
        }
    )
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_plan_execute_graph()
    
    print("=== Plan-Execute MCP Pattern ===\n")
    
    # Test Case: Multi-step task
    print("\n" + "="*70)
    print("TEST CASE: Plan and Execute Data Analysis Task")
    print("="*70)
    
    state = {
        "messages": [],
        "task": "Analyze customer feedback data and create a summary report with insights and recommendations",
        "plan": [],
        "execution_log": [],
        "current_step": 0,
        "execution_status": "planning",
        "replanning_needed": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 132: Plan-Execute - COMPLETE")
    print(f"{'='*70}")

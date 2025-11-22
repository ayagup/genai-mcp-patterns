"""
Sequential Execution MCP Pattern

This pattern implements sequential execution where tasks are
performed one after another in a defined order.

Key Features:
- Ordered execution
- Dependency management
- State propagation
- Error handling
- Step tracking
"""

from typing import TypedDict, Sequence, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class SequentialState(TypedDict):
    """State for sequential execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: List[str]
    current_step: int
    results: List[str]
    intermediate_data: dict


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def step_1_agent(state: SequentialState) -> SequentialState:
    """Execute first step"""
    task = state.get("tasks", [])[0] if state.get("tasks") else "Step 1"
    
    system_prompt = """You are a sequential execution expert.

Sequential Execution:
• Execute in order
• Pass state forward
• Handle dependencies
• Track progress
• Ensure completion

One step at a time."""
    
    user_prompt = f"Execute: {task}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    results = state.get("results", [])
    results.append(f"Step 1 complete: {response.content[:100]}")
    
    return {
        "messages": [AIMessage(content=f"✅ Step 1: {response.content[:200]}")],
        "current_step": 1,
        "results": results
    }


def step_2_agent(state: SequentialState) -> SequentialState:
    """Execute second step - uses output from step 1"""
    previous_result = state.get("results", [])[-1] if state.get("results") else "No previous result"
    
    prompt = f"""Continue sequential execution.
Previous step: {previous_result}

Execute next step in sequence."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    results = state.get("results", [])
    results.append(f"Step 2 complete: {response.content[:100]}")
    
    return {
        "messages": [AIMessage(content=f"✅ Step 2: {response.content[:200]}")],
        "current_step": 2,
        "results": results
    }


def step_3_agent(state: SequentialState) -> SequentialState:
    """Execute third step - final step"""
    all_results = state.get("results", [])
    
    system_prompt = """You are completing a sequential execution.

Sequential Execution System:
```python
class SequentialExecutor:
    def __init__(self):
        self.steps = []
        self.current_index = 0
        self.state = {}
    
    def add_step(self, step_function):
        '''Add step to sequence'''
        self.steps.append(step_function)
    
    def execute(self):
        '''Execute all steps sequentially'''
        for i, step in enumerate(self.steps):
            print(f"Executing step {i+1}/{len(self.steps)}")
            
            # Execute step with current state
            result = step(self.state)
            
            # Update state with result
            self.state.update(result)
            
            # Check for errors
            if result.get('error'):
                return self.handle_error(result['error'], i)
        
        return self.state
```

Sequential Patterns:

Linear Pipeline:
```python
def linear_pipeline(input_data):
    '''Simple sequential pipeline'''
    # Step 1: Validate
    validated = validate(input_data)
    
    # Step 2: Transform
    transformed = transform(validated)
    
    # Step 3: Process
    processed = process(transformed)
    
    # Step 4: Output
    output = format_output(processed)
    
    return output
```

With State Passing:
```python
class PipelineState:
    def __init__(self, initial_data):
        self.data = initial_data
        self.metadata = {}
        self.history = []

def step_with_state(state):
    '''Each step receives and modifies state'''
    # Read from state
    input_data = state.data
    
    # Process
    result = perform_operation(input_data)
    
    # Update state
    state.data = result
    state.history.append(('operation', result))
    
    return state
```

Dependency Chain:
```python
def execute_with_dependencies(tasks):
    '''Execute tasks respecting dependencies'''
    completed = set()
    
    while len(completed) < len(tasks):
        for task in tasks:
            # Check if dependencies met
            deps = task.dependencies
            
            if all(d in completed for d in deps):
                # Execute task
                execute(task)
                completed.add(task.id)
```

Error Handling:
```python
def sequential_with_error_handling(steps):
    '''Handle errors in sequence'''
    for i, step in enumerate(steps):
        try:
            result = step()
        except Exception as e:
            # Rollback previous steps
            rollback(steps[:i])
            raise
    
    return result
```

Checkpoint Pattern:
```python
def sequential_with_checkpoints(steps):
    '''Save state at checkpoints'''
    checkpoints = []
    
    for i, step in enumerate(steps):
        # Execute step
        result = step()
        
        # Save checkpoint
        if i % CHECKPOINT_INTERVAL == 0:
            checkpoint = save_state(result)
            checkpoints.append(checkpoint)
    
    return result, checkpoints
```

Resume Capability:
```python
def resumable_sequence(steps, last_checkpoint=None):
    '''Resume from last checkpoint'''
    if last_checkpoint:
        # Skip completed steps
        start_index = last_checkpoint.step_index + 1
        state = last_checkpoint.state
    else:
        start_index = 0
        state = initialize_state()
    
    for i in range(start_index, len(steps)):
        state = steps[i](state)
    
    return state
```

Progress Tracking:
```python
def execute_with_progress(steps):
    '''Track execution progress'''
    total = len(steps)
    
    for i, step in enumerate(steps):
        # Update progress
        progress = (i + 1) / total * 100
        print(f"Progress: {progress:.1f}%")
        
        # Execute
        result = step()
    
    return result
```

Timing and Profiling:
```python
import time

def timed_sequence(steps):
    '''Measure step execution times'''
    timings = []
    
    for i, step in enumerate(steps):
        start = time.time()
        result = step()
        duration = time.time() - start
        
        timings.append({
            'step': i,
            'duration': duration
        })
    
    return result, timings
```

Conditional Steps:
```python
def conditional_sequence(steps):
    '''Skip steps based on conditions'''
    for step in steps:
        # Check if step should run
        if step.condition():
            step.execute()
        else:
            print(f"Skipping {step.name}")
```

Sequential with Validation:
```python
def validated_sequence(steps):
    '''Validate output of each step'''
    for step in steps:
        result = step()
        
        # Validate result
        if not validate(result):
            raise ValueError(f"Step {step.name} produced invalid output")
    
    return result
```

Best Practices:
✓ Define clear step order
✓ Pass state explicitly
✓ Handle errors gracefully
✓ Track progress
✓ Enable resume capability
✓ Validate at each step
✓ Log execution details

Key Insight:
Sequential execution ensures predictable order
and clear data flow through explicit dependencies.
"""
    
    user_prompt = f"Complete sequential execution. Previous steps: {len(all_results)}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    results = state.get("results", [])
    results.append(f"Step 3 complete: Finalized")
    
    return {
        "messages": [AIMessage(content=f"✅ Step 3 (Final):\n{response.content}")],
        "current_step": 3,
        "results": results
    }


def build_sequential_graph():
    workflow = StateGraph(SequentialState)
    
    # Add nodes in sequence
    workflow.add_node("step_1", step_1_agent)
    workflow.add_node("step_2", step_2_agent)
    workflow.add_node("step_3", step_3_agent)
    
    # Create linear sequence
    workflow.add_edge(START, "step_1")
    workflow.add_edge("step_1", "step_2")
    workflow.add_edge("step_2", "step_3")
    workflow.add_edge("step_3", END)
    
    return workflow.compile()


if __name__ == "__main__":
    graph = build_sequential_graph()
    
    print("=== Sequential Execution MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "tasks": ["Analyze data", "Transform results", "Generate report"],
        "current_step": 0,
        "results": [],
        "intermediate_data": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Execution Results:")
    for i, res in enumerate(result.get("results", []), 1):
        print(f"{i}. {res}")
    print(f"{'='*70}")
    print("Pattern 161: Sequential Execution - COMPLETE")
    print(f"{'='*70}")

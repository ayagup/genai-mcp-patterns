"""
Loop Execution MCP Pattern

This pattern implements loop execution for iterative
processing and repeated operations.

Key Features:
- Iteration control
- Loop conditions
- State accumulation
- Break/continue logic
- Convergence detection
"""

from typing import TypedDict, Sequence, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class LoopState(TypedDict):
    """State for loop execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    iteration: int
    max_iterations: int
    items: List[str]
    results: List[str]
    should_continue: bool


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def loop_condition(state: LoopState) -> str:
    """Determine if loop should continue"""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 3)
    should_continue = state.get("should_continue", True)
    
    if iteration >= max_iter or not should_continue:
        return "exit_loop"
    else:
        return "continue_loop"


def loop_body_agent(state: LoopState) -> LoopState:
    """Execute loop iteration"""
    iteration = state.get("iteration", 0)
    items = state.get("items", [])
    
    current_item = items[iteration] if iteration < len(items) else f"Item {iteration}"
    
    prompt = f"Process iteration {iteration + 1}: {current_item}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    results = state.get("results", [])
    results.append(f"Iter {iteration + 1}: {response.content[:80]}")
    
    # Increment iteration
    next_iteration = iteration + 1
    
    # Check if should continue
    should_continue = next_iteration < len(items)
    
    return {
        "messages": [AIMessage(content=f"🔄 Iteration {iteration + 1}: {response.content[:150]}")],
        "iteration": next_iteration,
        "results": results,
        "should_continue": should_continue
    }


def loop_exit_agent(state: LoopState) -> LoopState:
    """Handle loop exit and finalization"""
    iteration = state.get("iteration", 0)
    results = state.get("results", [])
    
    system_prompt = """You are a loop execution expert.

Loop Execution System:
```python
class LoopExecutor:
    def __init__(self):
        self.iteration = 0
        self.state = {}
    
    def execute_while(self, condition, body):
        '''While loop pattern'''
        while condition(self.state):
            self.state = body(self.state)
            self.iteration += 1
            
            # Safety: prevent infinite loops
            if self.iteration > MAX_ITERATIONS:
                raise LoopLimitExceeded()
        
        return self.state
    
    def execute_for(self, items, body):
        '''For loop pattern'''
        results = []
        
        for i, item in enumerate(items):
            result = body(item, i, self.state)
            results.append(result)
            
            # Update state
            self.state.update(result)
        
        return results
```

While Loop Pattern:
```python
def while_loop(initial_state, condition, body):
    '''Standard while loop'''
    state = initial_state
    
    while condition(state):
        state = body(state)
    
    return state

# Example: Retry until success
def retry_until_success(operation, max_retries=3):
    attempts = 0
    
    while attempts < max_retries:
        try:
            return operation()
        except Exception as e:
            attempts += 1
            if attempts >= max_retries:
                raise
```

For Loop with State:
```python
def for_loop_with_state(items):
    '''Accumulate state across iterations'''
    state = initialize_state()
    
    for item in items:
        # Process item with current state
        result = process(item, state)
        
        # Update state
        state = update_state(state, result)
    
    return state

# Example: Running average
def running_average(numbers):
    total = 0
    count = 0
    
    for num in numbers:
        total += num
        count += 1
        average = total / count
        print(f"Current average: {average}")
    
    return average
```

Do-While Pattern:
```python
def do_while(initial_state, body, condition):
    '''Execute at least once'''
    state = initial_state
    
    # Execute once
    state = body(state)
    
    # Then check condition
    while condition(state):
        state = body(state)
    
    return state

# Example: User input validation
def get_valid_input():
    user_input = None
    
    # Do-while: keep asking until valid
    while True:
        user_input = input("Enter value: ")
        if is_valid(user_input):
            break
    
    return user_input
```

Nested Loops:
```python
def nested_loops(outer_items, inner_items):
    '''Process with nested iteration'''
    results = []
    
    for outer in outer_items:
        for inner in inner_items:
            result = process_pair(outer, inner)
            results.append(result)
    
    return results

# Example: Matrix operations
def matrix_multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum_val = 0
            for k in range(len(B)):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)
    return result
```

Loop with Break/Continue:
```python
def loop_with_control(items):
    '''Use break and continue'''
    for item in items:
        # Skip certain items
        if should_skip(item):
            continue
        
        # Process item
        result = process(item)
        
        # Early exit on condition
        if meets_termination_condition(result):
            break
    
    return result

# Example: Find first match
def find_first_match(items, predicate):
    for item in items:
        if predicate(item):
            return item  # Early return = break
    return None
```

Convergence Loop:
```python
def converge(initial_value, update_function, tolerance=1e-6):
    '''Loop until convergence'''
    current = initial_value
    previous = None
    
    while True:
        previous = current
        current = update_function(current)
        
        # Check convergence
        if previous is not None:
            delta = abs(current - previous)
            if delta < tolerance:
                break
    
    return current

# Example: Gradient descent
def gradient_descent(initial_params, learning_rate=0.01):
    params = initial_params
    
    for iteration in range(MAX_ITERATIONS):
        gradient = compute_gradient(params)
        params = params - learning_rate * gradient
        
        # Check convergence
        if np.linalg.norm(gradient) < TOLERANCE:
            break
    
    return params
```

Iterative Refinement:
```python
def iterative_refinement(initial_solution):
    '''Improve solution iteratively'''
    solution = initial_solution
    
    for iteration in range(MAX_ITERATIONS):
        # Evaluate current solution
        score = evaluate(solution)
        
        # Refine
        refined = refine(solution)
        refined_score = evaluate(refined)
        
        # Accept if better
        if refined_score > score:
            solution = refined
        else:
            # Converged
            break
    
    return solution
```

Batch Processing Loop:
```python
def batch_loop(items, batch_size):
    '''Process in batches'''
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_result = process_batch(batch)
        results.extend(batch_result)
    
    return results

# Example: Batch API calls
def batch_api_calls(data, batch_size=100):
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        response = api.batch_request(batch)
        yield response
```

Enumerated Loop:
```python
def enumerated_loop(items):
    '''Loop with index'''
    for index, item in enumerate(items):
        print(f"Processing item {index}: {item}")
        result = process(item, index)

# Example: Compare with next
def compare_adjacent(items):
    for i, item in enumerate(items[:-1]):
        next_item = items[i + 1]
        if item > next_item:
            print(f"Descending at index {i}")
```

Parallel Loop:
```python
from concurrent.futures import ThreadPoolExecutor

def parallel_loop(items, worker_function):
    '''Execute loop iterations in parallel'''
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(worker_function, items))
    
    return results
```

Loop with State Accumulation:
```python
def accumulate_loop(items):
    '''Build up state across iterations'''
    accumulated = {
        'sum': 0,
        'count': 0,
        'max': float('-inf'),
        'min': float('inf')
    }
    
    for item in items:
        accumulated['sum'] += item
        accumulated['count'] += 1
        accumulated['max'] = max(accumulated['max'], item)
        accumulated['min'] = min(accumulated['min'], item)
    
    accumulated['average'] = accumulated['sum'] / accumulated['count']
    
    return accumulated
```

Generator Loop:
```python
def generator_loop(items):
    '''Memory-efficient iteration'''
    for item in items:
        # Process on-the-fly
        processed = process(item)
        yield processed

# Usage
for result in generator_loop(large_dataset):
    handle(result)  # Process one at a time
```

Best Practices:
✓ Set maximum iteration limits
✓ Have clear exit conditions
✓ Track iteration state
✓ Handle edge cases (empty lists)
✓ Consider early exit
✓ Log progress for long loops
✓ Use generators for large data

Key Insight:
Loop execution enables iterative processing
with controlled repetition and state accumulation.
"""
    
    user_prompt = f"Completed {iteration} iterations"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"✅ Loop Complete ({iteration} iterations):\n{response.content}")]
    }


def build_loop_graph():
    workflow = StateGraph(LoopState)
    
    # Add nodes
    workflow.add_node("loop_body", loop_body_agent)
    workflow.add_node("loop_exit", loop_exit_agent)
    
    # Start with loop body
    workflow.add_edge(START, "loop_body")
    
    # Conditional: continue or exit
    workflow.add_conditional_edges(
        "loop_body",
        loop_condition,
        {
            "continue_loop": "loop_body",  # Loop back
            "exit_loop": "loop_exit"
        }
    )
    
    workflow.add_edge("loop_exit", END)
    
    return workflow.compile()


if __name__ == "__main__":
    graph = build_loop_graph()
    
    print("=== Loop Execution MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "iteration": 0,
        "max_iterations": 3,
        "items": ["Task A", "Task B", "Task C"],
        "results": [],
        "should_continue": True
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Loop Results:")
    for res in result.get("results", []):
        print(f"  {res}")
    print(f"Total iterations: {result.get('iteration', 0)}")
    print(f"{'='*70}")
    print("Pattern 164: Loop Execution - COMPLETE")
    print(f"{'='*70}")

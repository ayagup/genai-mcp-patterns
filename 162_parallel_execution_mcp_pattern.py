"""
Parallel Execution MCP Pattern

This pattern implements parallel execution where multiple tasks
run concurrently to maximize throughput and minimize latency.

Key Features:
- Concurrent execution
- Resource management
- Result aggregation
- Synchronization
- Load distribution
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ParallelState(TypedDict):
    """State for parallel execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: List[str]
    results: Dict[str, str]
    execution_mode: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def parallel_task_1(state: ParallelState) -> ParallelState:
    """Execute task 1 in parallel"""
    task = state.get("tasks", [])[0] if len(state.get("tasks", [])) > 0 else "Task 1"
    
    prompt = f"Execute parallel task: {task}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    results = state.get("results", {})
    results["task_1"] = response.content[:100]
    
    return {
        "messages": [AIMessage(content=f"🔀 Task 1: {response.content[:150]}")],
        "results": results
    }


def parallel_task_2(state: ParallelState) -> ParallelState:
    """Execute task 2 in parallel"""
    task = state.get("tasks", [])[1] if len(state.get("tasks", [])) > 1 else "Task 2"
    
    prompt = f"Execute parallel task: {task}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    results = state.get("results", {})
    results["task_2"] = response.content[:100]
    
    return {
        "messages": [AIMessage(content=f"🔀 Task 2: {response.content[:150]}")],
        "results": results
    }


def parallel_task_3(state: ParallelState) -> ParallelState:
    """Execute task 3 in parallel"""
    task = state.get("tasks", [])[2] if len(state.get("tasks", [])) > 2 else "Task 3"
    
    prompt = f"Execute parallel task: {task}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    results = state.get("results", {})
    results["task_3"] = response.content[:100]
    
    return {
        "messages": [AIMessage(content=f"🔀 Task 3: {response.content[:150]}")],
        "results": results
    }


def aggregator_agent(state: ParallelState) -> ParallelState:
    """Aggregate results from parallel tasks"""
    results = state.get("results", {})
    
    system_prompt = """You are a parallel execution expert.

Parallel Execution System:
```python
import concurrent.futures
import asyncio
from typing import List, Callable

class ParallelExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.results = {}
    
    def execute_parallel(self, tasks: List[Callable]):
        '''Execute tasks in parallel using threads'''
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(task): task.__name__ 
                for task in tasks
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_task):
                task_name = future_to_task[future]
                try:
                    result = future.result()
                    self.results[task_name] = result
                except Exception as e:
                    self.results[task_name] = {'error': str(e)}
        
        return self.results
```

Async Parallel Execution:
```python
async def async_parallel_executor(tasks):
    '''Execute tasks concurrently with asyncio'''
    # Create tasks
    async_tasks = [asyncio.create_task(task()) for task in tasks]
    
    # Wait for all to complete
    results = await asyncio.gather(*async_tasks, return_exceptions=True)
    
    return results

# Usage
async def main():
    tasks = [fetch_data_1, fetch_data_2, fetch_data_3]
    results = await async_parallel_executor(tasks)
    return results
```

Process-Based Parallelism:
```python
from multiprocessing import Pool

def process_parallel(tasks, data_items):
    '''Use multiple processes for CPU-intensive tasks'''
    with Pool(processes=4) as pool:
        # Map tasks to data in parallel
        results = pool.map(cpu_intensive_task, data_items)
    
    return results
```

Parallel Map-Reduce:
```python
def parallel_map_reduce(data, map_func, reduce_func):
    '''Parallel map-reduce pattern'''
    # Map phase (parallel)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        mapped = list(executor.map(map_func, data))
    
    # Reduce phase (sequential)
    result = reduce_func(mapped)
    
    return result

# Example
def word_count_parallel(documents):
    # Map: count words in each document (parallel)
    def count_words(doc):
        return Counter(doc.split())
    
    # Reduce: combine counts (sequential)
    def combine_counts(counts):
        total = Counter()
        for count in counts:
            total.update(count)
        return total
    
    return parallel_map_reduce(documents, count_words, combine_counts)
```

Fan-Out Pattern:
```python
def fan_out(input_data, workers):
    '''Distribute work to multiple workers'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit work to all workers
        futures = [
            executor.submit(worker, input_data) 
            for worker in workers
        ]
        
        # Collect results
        results = [f.result() for f in futures]
    
    return results
```

Fan-In Pattern:
```python
def fan_in(results):
    '''Aggregate results from multiple sources'''
    aggregated = {}
    
    for result in results:
        # Merge results
        aggregated = merge(aggregated, result)
    
    return aggregated
```

Scatter-Gather:
```python
def scatter_gather(data, num_workers):
    '''Scatter data, process in parallel, gather results'''
    # Scatter: split data
    chunks = split_data(data, num_workers)
    
    # Process: parallel execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))
    
    # Gather: combine results
    combined = combine_results(results)
    
    return combined
```

Synchronization:

Barrier Pattern:
```python
from threading import Barrier

def parallel_with_barrier(num_threads):
    '''Synchronize threads at barrier'''
    barrier = Barrier(num_threads)
    
    def worker(worker_id):
        # Phase 1
        do_work_phase_1(worker_id)
        
        # Wait for all threads
        barrier.wait()
        
        # Phase 2 (after all complete phase 1)
        do_work_phase_2(worker_id)
    
    threads = [Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
```

Lock-Free Patterns:
```python
from queue import Queue

def producer_consumer_parallel():
    '''Lock-free parallel pattern'''
    queue = Queue()
    
    # Producers
    def producer():
        for item in generate_items():
            queue.put(item)
    
    # Consumers
    def consumer():
        while True:
            item = queue.get()
            if item is None:
                break
            process(item)
            queue.task_done()
    
    # Start workers
    producers = [Thread(target=producer) for _ in range(2)]
    consumers = [Thread(target=consumer) for _ in range(4)]
    
    for t in producers + consumers:
        t.start()
```

Result Aggregation:

First Completed:
```python
def first_completed(tasks):
    '''Return first result that completes'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task) for task in tasks]
        
        # Wait for first to complete
        done, pending = concurrent.futures.wait(
            futures, 
            return_when=concurrent.futures.FIRST_COMPLETED
        )
        
        # Cancel remaining
        for future in pending:
            future.cancel()
        
        # Return first result
        return list(done)[0].result()
```

All Completed:
```python
def all_completed(tasks):
    '''Wait for all results'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task) for task in tasks]
        
        # Wait for all
        results = [f.result() for f in futures]
    
    return results
```

Timeout Handling:
```python
def parallel_with_timeout(tasks, timeout):
    '''Execute with overall timeout'''
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(task) for task in tasks]
        
        done, pending = concurrent.futures.wait(
            futures,
            timeout=timeout
        )
        
        # Get completed results
        results = [f.result() for f in done]
        
        # Handle incomplete
        for f in pending:
            f.cancel()
        
        return results, len(pending)
```

Load Balancing:
```python
def dynamic_load_balancing(tasks):
    '''Distribute work dynamically'''
    task_queue = Queue()
    for task in tasks:
        task_queue.put(task)
    
    def worker():
        results = []
        while not task_queue.empty():
            try:
                task = task_queue.get_nowait()
                result = task()
                results.append(result)
            except Empty:
                break
        return results
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(worker) for _ in range(NUM_WORKERS)]
        all_results = [r for f in futures for r in f.result()]
    
    return all_results
```

Best Practices:
✓ Identify independent tasks
✓ Manage resource limits
✓ Handle exceptions properly
✓ Avoid shared state mutations
✓ Use appropriate parallelism type
✓ Monitor resource usage
✓ Test under load

Key Insight:
Parallel execution maximizes throughput by running
independent tasks concurrently while managing resources.
"""
    
    user_prompt = f"Aggregate {len(results)} parallel results"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"📊 Aggregation Complete:\n{response.content}")]
    }


def build_parallel_graph():
    workflow = StateGraph(ParallelState)
    
    # Add parallel task nodes
    workflow.add_node("task_1", parallel_task_1)
    workflow.add_node("task_2", parallel_task_2)
    workflow.add_node("task_3", parallel_task_3)
    workflow.add_node("aggregator", aggregator_agent)
    
    # Fan-out: all tasks start from START
    workflow.add_edge(START, "task_1")
    workflow.add_edge(START, "task_2")
    workflow.add_edge(START, "task_3")
    
    # Fan-in: all tasks feed into aggregator
    workflow.add_edge("task_1", "aggregator")
    workflow.add_edge("task_2", "aggregator")
    workflow.add_edge("task_3", "aggregator")
    
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


if __name__ == "__main__":
    graph = build_parallel_graph()
    
    print("=== Parallel Execution MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "tasks": ["Analyze dataset A", "Process dataset B", "Validate dataset C"],
        "results": {},
        "execution_mode": "parallel"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Parallel Execution Results:")
    for task, res in result.get("results", {}).items():
        print(f"{task}: {res}")
    print(f"{'='*70}")
    print("Pattern 162: Parallel Execution - COMPLETE")
    print(f"{'='*70}")

"""
Lazy Evaluation MCP Pattern

This pattern implements lazy evaluation where computation
is deferred until results are actually needed.

Key Features:
- Deferred computation
- On-demand execution
- Resource efficiency
- Generator patterns
- Memoization
"""

from typing import TypedDict, Sequence, Annotated, List, Callable
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class LazyState(TypedDict):
    """State for lazy evaluation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    computed_values: List[str]
    evaluation_count: int


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def lazy_evaluator_agent(state: LazyState) -> LazyState:
    """Demonstrate lazy evaluation concepts"""
    query = state.get("query", "")
    
    system_prompt = """You are a lazy evaluation expert.

Lazy Evaluation System:
```python
class LazyValue:
    def __init__(self, computation):
        self.computation = computation
        self._value = None
        self._computed = False
    
    @property
    def value(self):
        '''Compute only when accessed'''
        if not self._computed:
            print("Computing value...")
            self._value = self.computation()
            self._computed = True
        return self._value

# Usage
lazy_result = LazyValue(lambda: expensive_computation())
# Not computed yet...
result = lazy_result.value  # Computed now!
```

Generator Pattern:
```python
def lazy_sequence(n):
    '''Generate values on demand'''
    for i in range(n):
        # Value computed only when requested
        yield expensive_function(i)

# Usage
gen = lazy_sequence(1000000)  # No computation yet
first = next(gen)  # Compute first value only
```

Lazy Iterator:
```python
class LazyIterator:
    def __init__(self, data_source):
        self.data_source = data_source
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.data_source):
            raise StopIteration
        
        # Load data lazily
        item = self.data_source.get(self.index)
        self.index += 1
        return item

# Example: Lazy file reading
def lazy_file_reader(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()  # Read one line at a time

# Memory-efficient for large files
for line in lazy_file_reader('huge_file.txt'):
    process(line)
```

Lazy Property:
```python
class LazyObject:
    @property
    def expensive_property(self):
        '''Computed on first access, cached after'''
        if not hasattr(self, '_expensive_property'):
            print("Computing expensive property...")
            self._expensive_property = expensive_computation()
        return self._expensive_property

# Usage
obj = LazyObject()
# Not computed yet...
value = obj.expensive_property  # Computed now
value2 = obj.expensive_property  # Retrieved from cache
```

Lazy Import:
```python
def lazy_import(module_name):
    '''Import module only when used'''
    _module = None
    
    def get_module():
        nonlocal _module
        if _module is None:
            _module = __import__(module_name)
        return _module
    
    return get_module

# Usage
get_numpy = lazy_import('numpy')
# numpy not imported yet...
np = get_numpy()  # Imported now
```

Lazy Map:
```python
def lazy_map(func, iterable):
    '''Apply function lazily'''
    for item in iterable:
        yield func(item)

# More memory-efficient than list comprehension
lazy_results = lazy_map(expensive_func, huge_list)
# Nothing computed yet

first_result = next(lazy_results)  # Compute first only
```

Lazy Filter:
```python
def lazy_filter(predicate, iterable):
    '''Filter lazily'''
    for item in iterable:
        if predicate(item):
            yield item

# Example: Find first matching item
def find_first(items, condition):
    lazy_filtered = lazy_filter(condition, items)
    try:
        return next(lazy_filtered)  # Stop after first match
    except StopIteration:
        return None
```

Thunk Pattern:
```python
class Thunk:
    '''Delayed computation'''
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result = None
        self._evaluated = False
    
    def force(self):
        '''Force evaluation'''
        if not self._evaluated:
            self._result = self.func(*self.args, **self.kwargs)
            self._evaluated = True
        return self._result

# Usage
computation = Thunk(expensive_function, arg1, arg2)
# Not computed yet...
result = computation.force()  # Computed now
```

Lazy Data Loading:
```python
class LazyDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self._data = None
    
    def __getitem__(self, index):
        '''Load data on first access'''
        if self._data is None:
            self._data = load_data(self.file_path)
        return self._data[index]

# Usage
dataset = LazyDataset('large_file.parquet')
# File not loaded yet...
item = dataset[0]  # Loaded now
```

Lazy Evaluation with Caching:
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(x):
    '''Lazy + memoized'''
    print(f"Computing for {x}")
    return x ** 2

# First call computes
result1 = expensive_function(5)  # Prints "Computing for 5"

# Second call uses cache
result2 = expensive_function(5)  # No print (cached)
```

Stream Processing:
```python
def lazy_stream_processor(stream):
    '''Process stream lazily'''
    for item in stream:
        # Process one at a time
        processed = process(item)
        
        # Transform
        transformed = transform(processed)
        
        # Yield immediately
        yield transformed

# Usage with infinite stream
def infinite_stream():
    i = 0
    while True:
        yield generate_data(i)
        i += 1

processor = lazy_stream_processor(infinite_stream())
for result in processor:
    handle(result)
    if should_stop():
        break  # Can stop anytime
```

Lazy Reduce:
```python
def lazy_reduce(func, iterable, initial=None):
    '''Reduce without loading all data'''
    iterator = iter(iterable)
    
    if initial is None:
        value = next(iterator)
    else:
        value = initial
    
    for item in iterator:
        value = func(value, item)
    
    return value

# Example: Sum large file
def sum_large_file(filename):
    lines = lazy_file_reader(filename)
    numbers = lazy_map(int, lines)
    total = lazy_reduce(lambda a, b: a + b, numbers, 0)
    return total
```

Lazy Chain:
```python
def lazy_chain(*iterables):
    '''Chain iterables lazily'''
    for iterable in iterables:
        for item in iterable:
            yield item

# Combine multiple sources without loading all
combined = lazy_chain(source1, source2, source3)
for item in combined:
    process(item)
```

Conditional Lazy Evaluation:
```python
def lazy_or(conditions):
    '''Short-circuit OR evaluation'''
    for condition in conditions:
        if condition():  # Evaluate lazily
            return True
    return False

def lazy_and(conditions):
    '''Short-circuit AND evaluation'''
    for condition in conditions:
        if not condition():  # Stop at first False
            return False
    return True

# Example
if lazy_or([check1, check2, check3]):
    # check2 and check3 not called if check1 is True
    handle_true_case()
```

Lazy Pagination:
```python
class LazyPaginator:
    def __init__(self, fetch_page, total_pages):
        self.fetch_page = fetch_page
        self.total_pages = total_pages
        self.current_page = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_page >= self.total_pages:
            raise StopIteration
        
        # Fetch page on demand
        page_data = self.fetch_page(self.current_page)
        self.current_page += 1
        return page_data

# Usage
paginator = LazyPaginator(api.get_page, total_pages=100)
for page in paginator:
    process(page)
    if found_what_we_need():
        break  # Stop early, don't fetch remaining pages
```

Lazy Tree Traversal:
```python
def lazy_dfs(node):
    '''Depth-first search, yielding nodes lazily'''
    yield node
    
    for child in node.children:
        yield from lazy_dfs(child)

# Can stop traversal early
for node in lazy_dfs(root):
    if node.matches_criteria():
        return node  # Stop traversal
```

Advantages:
✓ Memory efficient for large datasets
✓ Faster startup (defer computation)
✓ Enable infinite sequences
✓ Short-circuit evaluation
✓ Process as you go
✓ Better composability

Trade-offs:
✗ Harder to debug
✗ Unpredictable timing
✗ May compute multiple times if not cached
✗ Hidden complexity

Best Practices:
✓ Use generators for large data
✓ Combine with caching for repeated access
✓ Document lazy behavior
✓ Consider memory vs CPU trade-offs
✓ Use for I/O-bound operations
✓ Be aware of side effects

Key Insight:
Lazy evaluation defers computation until needed,
improving efficiency for large or infinite datasets.
"""
    
    user_prompt = f"Explain lazy evaluation for: {query}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"💤 Lazy Evaluation:\n{response.content}")],
        "evaluation_count": 1
    }


def build_lazy_graph():
    workflow = StateGraph(LazyState)
    workflow.add_node("lazy_evaluator", lazy_evaluator_agent)
    workflow.add_edge(START, "lazy_evaluator")
    workflow.add_edge("lazy_evaluator", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_lazy_graph()
    
    print("=== Lazy Evaluation MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "query": "Process large dataset efficiently",
        "computed_values": [],
        "evaluation_count": 0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 165: Lazy Evaluation - COMPLETE")
    print(f"{'='*70}")

"""
Incremental Execution MCP Pattern

This pattern implements incremental execution for gradual
processing with intermediate checkpoints and resumability.

Key Features:
- Checkpoint management
- Progress tracking
- Resume capability
- Partial results
- Fault tolerance
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class IncrementalState(TypedDict):
    """State for incremental execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    total_steps: int
    completed_steps: int
    checkpoints: List[Dict]
    current_result: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def incremental_processor_agent(state: IncrementalState) -> IncrementalState:
    """Execute incrementally with checkpoints"""
    completed = state.get("completed_steps", 0)
    total = state.get("total_steps", 10)
    
    system_prompt = """You are an incremental execution expert.

Incremental Processing System:
```python
class IncrementalProcessor:
    def __init__(self):
        self.checkpoint_interval = 10
        self.current_state = {}
        self.checkpoints = []
    
    def process_incrementally(self, items):
        '''Process with checkpoints'''
        for i, item in enumerate(items):
            # Process item
            result = self.process_item(item)
            self.current_state['last_processed'] = i
            
            # Checkpoint periodically
            if i % self.checkpoint_interval == 0:
                self.save_checkpoint(i, self.current_state)
        
        return self.current_state
    
    def save_checkpoint(self, index, state):
        '''Save progress'''
        checkpoint = {
            'index': index,
            'state': state.copy(),
            'timestamp': time.time()
        }
        self.checkpoints.append(checkpoint)
        persist_checkpoint(checkpoint)
    
    def resume_from_checkpoint(self, checkpoint_id):
        '''Resume from saved point'''
        checkpoint = load_checkpoint(checkpoint_id)
        self.current_state = checkpoint['state']
        start_index = checkpoint['index'] + 1
        
        return start_index
```

Checkpoint-Based Processing:
```python
def checkpoint_process(items, checkpoint_file='checkpoint.pkl'):
    '''Process with resumability'''
    # Try to resume
    start_index = 0
    state = {}
    
    if os.path.exists(checkpoint_file):
        checkpoint = pickle.load(open(checkpoint_file, 'rb'))
        start_index = checkpoint['index']
        state = checkpoint['state']
        print(f"Resuming from index {start_index}")
    
    # Process from checkpoint
    for i in range(start_index, len(items)):
        state = process_item(items[i], state)
        
        # Save checkpoint every N items
        if i % 100 == 0:
            pickle.dump({
                'index': i,
                'state': state
            }, open(checkpoint_file, 'wb'))
    
    return state
```

Incremental Build:
```python
class IncrementalBuilder:
    def __init__(self):
        self.cache = {}
        self.dependencies = {}
    
    def build(self, target):
        '''Only rebuild changed components'''
        if self.is_up_to_date(target):
            return self.cache[target]
        
        # Build dependencies first
        for dep in self.dependencies.get(target, []):
            self.build(dep)
        
        # Build target
        result = self.build_target(target)
        self.cache[target] = result
        
        return result
    
    def is_up_to_date(self, target):
        '''Check if rebuild needed'''
        if target not in self.cache:
            return False
        
        # Check modification times
        target_time = get_modification_time(target)
        cache_time = self.cache[target]['timestamp']
        
        return target_time < cache_time

# Example: Incremental compilation
# Only recompile changed files
```

Incremental Indexing:
```python
class IncrementalIndexer:
    def __init__(self):
        self.index = {}
        self.last_indexed = {}
    
    def index_documents(self, documents):
        '''Index only new/modified documents'''
        for doc in documents:
            doc_id = doc['id']
            doc_modified = doc['modified_time']
            
            # Skip if already indexed and not modified
            if doc_id in self.last_indexed:
                if self.last_indexed[doc_id] >= doc_modified:
                    continue
            
            # Index document
            self.index[doc_id] = self.create_index(doc)
            self.last_indexed[doc_id] = doc_modified
        
        return self.index
```

Incremental Aggregation:
```python
class IncrementalAggregator:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.sum_squares = 0
    
    def add(self, value):
        '''Update aggregates incrementally'''
        self.count += 1
        self.sum += value
        self.sum_squares += value ** 2
    
    @property
    def mean(self):
        return self.sum / self.count if self.count > 0 else 0
    
    @property
    def variance(self):
        if self.count == 0:
            return 0
        return (self.sum_squares / self.count) - (self.mean ** 2)

# Process stream incrementally
aggregator = IncrementalAggregator()
for value in data_stream:
    aggregator.add(value)
    print(f"Current mean: {aggregator.mean}")
```

Incremental Training:
```python
def incremental_train(model, new_data, learning_rate=0.01):
    '''Update model with new data'''
    # Don't retrain from scratch
    # Just update with new samples
    
    for batch in batch_iterator(new_data):
        gradients = compute_gradients(model, batch)
        model.update_parameters(gradients, learning_rate)
    
    return model

# vs Full retrain:
# model = train_from_scratch(all_data)  # Expensive!
```

Incremental Search:
```python
class IncrementalSearch:
    def __init__(self):
        self.results = []
        self.offset = 0
    
    def next_page(self, query, page_size=10):
        '''Load results incrementally'''
        new_results = search(query, offset=self.offset, limit=page_size)
        self.results.extend(new_results)
        self.offset += page_size
        
        return new_results
    
    def has_more(self):
        return len(self.results) < self.total_count

# Usage
search = IncrementalSearch()
while search.has_more():
    page = search.next_page(query)
    display(page)
    
    if user_found_what_they_need():
        break  # Stop early
```

Version-Based Incremental Update:
```python
class VersionedData:
    def __init__(self):
        self.version = 0
        self.data = {}
        self.changes = []
    
    def update(self, key, value):
        '''Track incremental changes'''
        old_value = self.data.get(key)
        self.data[key] = value
        
        # Record change
        self.changes.append({
            'version': self.version,
            'key': key,
            'old': old_value,
            'new': value
        })
        
        self.version += 1
    
    def get_changes_since(self, version):
        '''Get incremental updates'''
        return [c for c in self.changes if c['version'] > version]

# Client can sync incrementally
local_version = 5
changes = server.get_changes_since(local_version)
apply_changes(changes)
```

Incremental Computation (Memoization):
```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    '''Compute incrementally using cache'''
    if n <= 1:
        return n
    
    # Reuse previously computed values
    return fibonacci(n-1) + fibonacci(n-2)

# fibonacci(100) will reuse fibonacci(99), fibonacci(98), etc.
```

Differential Update:
```python
def differential_update(old_data, new_data):
    '''Compute only the diff'''
    added = set(new_data) - set(old_data)
    removed = set(old_data) - set(new_data)
    unchanged = set(old_data) & set(new_data)
    
    # Only process changes
    for item in added:
        process_addition(item)
    
    for item in removed:
        process_removal(item)
    
    # Skip unchanged items
    return {
        'added': added,
        'removed': removed,
        'unchanged': len(unchanged)
    }
```

Incremental Rendering:
```python
class IncrementalRenderer:
    def __init__(self):
        self.rendered_chunks = {}
    
    def render(self, document, viewport):
        '''Render only visible portion'''
        visible_chunks = get_visible_chunks(document, viewport)
        
        to_render = []
        for chunk_id in visible_chunks:
            if chunk_id not in self.rendered_chunks:
                to_render.append(chunk_id)
        
        # Render only new chunks
        for chunk_id in to_render:
            self.rendered_chunks[chunk_id] = render_chunk(chunk_id)
        
        return [self.rendered_chunks[c] for c in visible_chunks]

# Only render what user sees, incrementally
```

Progress Tracking:
```python
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
    
    def update(self, increment=1):
        self.completed += increment
        
        # Calculate metrics
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        remaining = self.total - self.completed
        eta = remaining / rate if rate > 0 else float('inf')
        
        return {
            'completed': self.completed,
            'total': self.total,
            'percentage': (self.completed / self.total) * 100,
            'rate': rate,
            'eta': eta
        }

# Usage
tracker = ProgressTracker(total=10000)
for item in items:
    process(item)
    progress = tracker.update()
    print(f"{progress['percentage']:.1f}% complete, ETA: {progress['eta']:.0f}s")
```

Best Practices:
✓ Checkpoint frequently
✓ Save minimal state
✓ Handle resume gracefully
✓ Track progress
✓ Enable partial results
✓ Clean up old checkpoints
✓ Test resume logic

Benefits:
+ Fault tolerant
+ Shows progress
+ Can stop/resume
+ Saves intermediate results
+ Efficient for large tasks

Key Insight:
Incremental execution enables fault tolerance
and resumability through periodic checkpoints.
"""
    
    user_prompt = f"Process incrementally: {completed}/{total} steps done"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Simulate incremental progress
    next_checkpoint = {
        "step": completed + 1,
        "timestamp": "2024-01-15T10:30:00",
        "partial_result": "Checkpoint saved"
    }
    
    checkpoints = state.get("checkpoints", [])
    checkpoints.append(next_checkpoint)
    
    return {
        "messages": [AIMessage(content=f"📈 Incremental Progress:\n{response.content}")],
        "completed_steps": completed + 1,
        "checkpoints": checkpoints
    }


def build_incremental_graph():
    workflow = StateGraph(IncrementalState)
    workflow.add_node("incremental_processor", incremental_processor_agent)
    workflow.add_edge(START, "incremental_processor")
    workflow.add_edge("incremental_processor", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_incremental_graph()
    
    print("=== Incremental Execution MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "total_steps": 10,
        "completed_steps": 3,  # Resume from step 3
        "checkpoints": [],
        "current_result": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print(f"Progress: {result.get('completed_steps', 0)}/{state['total_steps']}")
    print(f"Checkpoints: {len(result.get('checkpoints', []))}")
    print(f"{'='*70}")
    print("Pattern 169: Incremental Execution - COMPLETE")
    print(f"{'='*70}")

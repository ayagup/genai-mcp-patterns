"""
Batch Execution MCP Pattern

This pattern implements batch execution for processing
groups of items together for efficiency.

Key Features:
- Batch accumulation
- Batch size optimization
- Bulk operations
- Resource efficiency
- Throughput maximization
"""

from typing import TypedDict, Sequence, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class BatchState(TypedDict):
    """State for batch execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    items: List[str]
    batch_size: int
    batches_processed: int
    results: List[str]


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def batch_processor_agent(state: BatchState) -> BatchState:
    """Process items in batches"""
    items = state.get("items", [])
    batch_size = state.get("batch_size", 10)
    
    system_prompt = """You are a batch execution expert.

Batch Processing System:
```python
class BatchProcessor:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.buffer = []
    
    def add(self, item):
        '''Add item to batch'''
        self.buffer.append(item)
        
        if len(self.buffer) >= self.batch_size:
            return self.flush()
        return None
    
    def flush(self):
        '''Process accumulated batch'''
        if not self.buffer:
            return None
        
        batch = self.buffer
        self.buffer = []
        
        return self.process_batch(batch)
    
    def process_batch(self, batch):
        '''Process entire batch together'''
        return bulk_operation(batch)
```

Fixed-Size Batching:
```python
def batch_process(items, batch_size):
    '''Process in fixed-size batches'''
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        result = process_batch(batch)
        yield result

# Usage
for batch_result in batch_process(data, batch_size=100):
    handle_result(batch_result)
```

Time-Based Batching:
```python
import time

class TimeBatchProcessor:
    def __init__(self, max_wait=1.0, max_size=100):
        self.max_wait = max_wait
        self.max_size = max_size
        self.buffer = []
        self.last_flush = time.time()
    
    def add(self, item):
        self.buffer.append(item)
        current_time = time.time()
        
        # Flush if size reached or time expired
        if (len(self.buffer) >= self.max_size or 
            current_time - self.last_flush >= self.max_wait):
            return self.flush()
        
        return None
    
    def flush(self):
        if not self.buffer:
            return None
        
        batch = self.buffer
        self.buffer = []
        self.last_flush = time.time()
        
        return process_batch(batch)
```

Database Batch Operations:
```python
def batch_insert(records, batch_size=1000):
    '''Batch database inserts'''
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        # Single bulk insert instead of many individual inserts
        db.execute_many(
            "INSERT INTO table VALUES (?)",
            batch
        )
        db.commit()

# Much faster than:
# for record in records:
#     db.execute("INSERT INTO table VALUES (?)", record)
```

Batch API Calls:
```python
def batch_api_requests(items, batch_size=50):
    '''Batch API requests'''
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Single API call for entire batch
        batch_response = api.batch_request(batch)
        results.extend(batch_response)
    
    return results

# vs individual calls:
# results = [api.request(item) for item in items]  # Slow!
```

Batch Inference:
```python
def batch_model_inference(inputs, batch_size=32):
    '''Batch ML model predictions'''
    predictions = []
    
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        
        # Single forward pass for batch (GPU efficient)
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
    
    return predictions

# GPU throughput: ~10x faster than one-at-a-time
```

Dynamic Batch Sizing:
```python
class DynamicBatchProcessor:
    def __init__(self):
        self.min_batch = 10
        self.max_batch = 1000
        self.current_batch = 100
        self.buffer = []
    
    def add(self, item):
        self.buffer.append(item)
        
        if len(self.buffer) >= self.current_batch:
            latency = self.flush()
            self.adjust_batch_size(latency)
    
    def adjust_batch_size(self, latency):
        '''Adapt batch size based on performance'''
        if latency < TARGET_LATENCY:
            # Increase batch size (more throughput)
            self.current_batch = min(
                self.current_batch * 1.5,
                self.max_batch
            )
        else:
            # Decrease batch size (lower latency)
            self.current_batch = max(
                self.current_batch * 0.75,
                self.min_batch
            )
```

Batch ETL Pipeline:
```python
def batch_etl(source, destination, batch_size=10000):
    '''Extract-Transform-Load in batches'''
    offset = 0
    
    while True:
        # Extract batch
        batch = source.read(offset, batch_size)
        
        if not batch:
            break
        
        # Transform batch
        transformed = [transform(record) for record in batch]
        
        # Load batch
        destination.write_batch(transformed)
        
        offset += batch_size
```

Parallel Batch Processing:
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_batch_process(items, batch_size=100):
    '''Process batches in parallel'''
    # Split into batches
    batches = [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]
    
    # Process batches in parallel
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_batch, batches))
    
    # Combine results
    return [item for batch_result in results for item in batch_result]
```

Batch Aggregation:
```python
def batch_aggregate(stream, batch_size=1000):
    '''Aggregate streaming data in batches'''
    batch = []
    
    for item in stream:
        batch.append(item)
        
        if len(batch) >= batch_size:
            # Aggregate batch
            aggregate = {
                'count': len(batch),
                'sum': sum(batch),
                'avg': sum(batch) / len(batch),
                'min': min(batch),
                'max': max(batch)
            }
            
            yield aggregate
            batch = []
```

Batch Retry Logic:
```python
def batch_with_retry(items, batch_size=100, max_retries=3):
    '''Retry failed batches'''
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        for attempt in range(max_retries):
            try:
                result = process_batch(batch)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    log_failed_batch(batch, e)
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
```

Batch Validation:
```python
def batch_validate(items, batch_size=500):
    '''Validate in batches'''
    errors = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Validate batch
        batch_errors = validate_batch(batch)
        
        if batch_errors:
            errors.extend(batch_errors)
    
    return errors

# Example: Batch schema validation
def validate_batch_schema(records):
    # Single schema compilation
    schema = compile_schema(SCHEMA_DEF)
    
    # Validate all at once
    for record in records:
        if not schema.validate(record):
            yield (record, schema.errors)
```

Batch Caching:
```python
class BatchCache:
    def __init__(self):
        self.cache = {}
        self.pending = []
    
    def get_batch(self, keys):
        '''Get multiple keys efficiently'''
        # Check cache
        results = {}
        missing = []
        
        for key in keys:
            if key in self.cache:
                results[key] = self.cache[key]
            else:
                missing.append(key)
        
        # Fetch missing in batch
        if missing:
            fetched = bulk_fetch(missing)
            self.cache.update(fetched)
            results.update(fetched)
        
        return results
```

Batch Write-Behind:
```python
class BatchWriter:
    def __init__(self, flush_interval=5.0, batch_size=1000):
        self.buffer = []
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.last_flush = time.time()
        self.start_background_flush()
    
    def write(self, data):
        '''Buffer writes'''
        self.buffer.append(data)
        
        if len(self.buffer) >= self.batch_size:
            self.flush()
    
    def flush(self):
        '''Write batch to storage'''
        if not self.buffer:
            return
        
        storage.write_batch(self.buffer)
        self.buffer = []
        self.last_flush = time.time()
    
    def start_background_flush(self):
        '''Periodic flush'''
        def periodic_flush():
            while True:
                time.sleep(self.flush_interval)
                self.flush()
        
        thread = Thread(target=periodic_flush, daemon=True)
        thread.start()
```

Batch Size Optimization:
```python
def find_optimal_batch_size(process_func, data_sample):
    '''Determine best batch size'''
    batch_sizes = [10, 50, 100, 500, 1000]
    results = {}
    
    for size in batch_sizes:
        start = time.time()
        process_func(data_sample[:size])
        duration = time.time() - start
        
        throughput = size / duration
        results[size] = throughput
    
    # Return size with best throughput
    return max(results, key=results.get)
```

Best Practices:
✓ Batch similar operations
✓ Optimize batch size for workload
✓ Handle partial batch at end
✓ Implement flush mechanism
✓ Consider memory limits
✓ Monitor batch metrics
✓ Use bulk APIs when available

Trade-offs:
+ Higher throughput
+ Better resource utilization
+ Amortized overhead
- Increased latency
- Memory buffering needed
- Complexity in error handling

Key Insight:
Batch execution trades latency for throughput
by processing groups of items together efficiently.
"""
    
    user_prompt = f"Process {len(items)} items in batches of {batch_size}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Simulate batch processing
    num_batches = (len(items) + batch_size - 1) // batch_size
    results = [f"Batch {i+1} processed" for i in range(num_batches)]
    
    return {
        "messages": [AIMessage(content=f"📦 Batch Processing:\n{response.content}")],
        "batches_processed": num_batches,
        "results": results
    }


def build_batch_graph():
    workflow = StateGraph(BatchState)
    workflow.add_node("batch_processor", batch_processor_agent)
    workflow.add_edge(START, "batch_processor")
    workflow.add_edge("batch_processor", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_batch_graph()
    
    print("=== Batch Execution MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "items": [f"Item_{i}" for i in range(1, 51)],  # 50 items
        "batch_size": 10,
        "batches_processed": 0,
        "results": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print(f"Total items: {len(state['items'])}")
    print(f"Batch size: {state['batch_size']}")
    print(f"Batches processed: {result.get('batches_processed', 0)}")
    print(f"{'='*70}")
    print("Pattern 168: Batch Execution - COMPLETE")
    print(f"{'='*70}")

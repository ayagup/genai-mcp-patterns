"""
Streaming Execution MCP Pattern

This pattern implements streaming execution for processing
data as it arrives without waiting for complete input.

Key Features:
- Real-time processing
- Incremental output
- Low latency
- Backpressure handling
- Event-driven flow
"""

from typing import TypedDict, Sequence, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class StreamingState(TypedDict):
    """State for streaming execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    stream_source: str
    chunks_processed: int
    buffer: List[str]


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def streaming_processor_agent(state: StreamingState) -> StreamingState:
    """Process streaming data"""
    source = state.get("stream_source", "")
    chunks = state.get("chunks_processed", 0)
    
    system_prompt = """You are a streaming execution expert.

Streaming System:
```python
class StreamProcessor:
    def __init__(self):
        self.buffer = []
    
    def process_stream(self, stream):
        '''Process data as it arrives'''
        for chunk in stream:
            # Process immediately
            result = self.process_chunk(chunk)
            
            # Emit result
            yield result
    
    def process_chunk(self, chunk):
        '''Process single chunk'''
        # Transform data
        processed = transform(chunk)
        
        # Update state
        self.buffer.append(processed)
        
        return processed
```

Generator-Based Streaming:
```python
def stream_processor(input_stream):
    '''Process and yield continuously'''
    for item in input_stream:
        # Process immediately
        processed = process(item)
        
        # Yield without waiting for more
        yield processed

# Usage
for result in stream_processor(data_stream):
    handle_result(result)  # Get results as they come
```

Async Streaming:
```python
import asyncio

async def async_stream_processor(stream):
    '''Async streaming processing'''
    async for chunk in stream:
        # Process asynchronously
        result = await process_async(chunk)
        
        # Yield immediately
        yield result

# Usage
async def consume_stream():
    async for result in async_stream_processor(input_stream):
        await handle_result(result)
```

Streaming Pipeline:
```python
def streaming_pipeline(source):
    '''Multi-stage streaming pipeline'''
    # Stage 1: Parse
    parsed = (parse(chunk) for chunk in source)
    
    # Stage 2: Filter
    filtered = (chunk for chunk in parsed if valid(chunk))
    
    # Stage 3: Transform
    transformed = (transform(chunk) for chunk in filtered)
    
    # Stage 4: Emit
    for result in transformed:
        yield result

# Lazy evaluation through entire pipeline
```

Chunked Streaming:
```python
def chunked_stream(stream, chunk_size):
    '''Buffer and emit in chunks'''
    buffer = []
    
    for item in stream:
        buffer.append(item)
        
        if len(buffer) >= chunk_size:
            # Emit chunk
            yield buffer
            buffer = []
    
    # Emit remaining
    if buffer:
        yield buffer

# Usage
for chunk in chunked_stream(large_stream, 100):
    batch_process(chunk)
```

Token Streaming (LLM):
```python
def stream_llm_response(prompt):
    '''Stream LLM tokens as generated'''
    for token in llm.stream(prompt):
        # Display token immediately
        print(token, end='', flush=True)
        yield token

# Usage
response = ""
for token in stream_llm_response("Write a story"):
    response += token
    # User sees response build up in real-time
```

Backpressure Handling:
```python
import asyncio
from asyncio import Queue

async def producer(queue, max_size=100):
    '''Produce with backpressure'''
    for i in range(1000):
        # Wait if queue full (backpressure)
        await queue.put(f"item_{i}")

async def consumer(queue):
    '''Consume at own pace'''
    while True:
        item = await queue.get()
        await process(item)
        queue.task_done()

# Queue provides natural backpressure
queue = Queue(maxsize=100)
```

Windowed Streaming:
```python
def windowed_stream(stream, window_size):
    '''Process with sliding window'''
    window = []
    
    for item in stream:
        window.append(item)
        
        if len(window) > window_size:
            window.pop(0)
        
        # Process window
        if len(window) == window_size:
            result = process_window(window)
            yield result

# Example: Moving average
def moving_average(numbers, window_size=5):
    for window in windowed_stream(numbers, window_size):
        avg = sum(window) / len(window)
        yield avg
```

Event Stream Processing:
```python
class EventStreamProcessor:
    def __init__(self):
        self.handlers = {}
    
    def register(self, event_type, handler):
        self.handlers[event_type] = handler
    
    def process_stream(self, event_stream):
        for event in event_stream:
            event_type = event['type']
            
            if event_type in self.handlers:
                result = self.handlers[event_type](event)
                yield result

# Usage
processor = EventStreamProcessor()
processor.register('click', handle_click)
processor.register('scroll', handle_scroll)

for result in processor.process_stream(events):
    emit(result)
```

Stream Merging:
```python
def merge_streams(*streams):
    '''Merge multiple streams'''
    iterators = [iter(stream) for stream in streams]
    
    while iterators:
        for it in list(iterators):
            try:
                yield next(it)
            except StopIteration:
                iterators.remove(it)

# Usage
combined = merge_streams(stream1, stream2, stream3)
for item in combined:
    process(item)
```

Stream Filtering:
```python
def filter_stream(stream, predicate):
    '''Filter streaming data'''
    for item in stream:
        if predicate(item):
            yield item

# Example: Real-time anomaly detection
def detect_anomalies(sensor_stream):
    for reading in filter_stream(sensor_stream, is_anomaly):
        alert(reading)
```

Stream Aggregation:
```python
def streaming_aggregate(stream, window_duration):
    '''Aggregate over time windows'''
    window_start = time.time()
    window_data = []
    
    for item in stream:
        current_time = time.time()
        
        # Check if window expired
        if current_time - window_start > window_duration:
            # Emit aggregated result
            yield aggregate(window_data)
            
            # Reset window
            window_start = current_time
            window_data = []
        
        window_data.append(item)

# Example: Requests per second
def requests_per_second(request_stream):
    for count in streaming_aggregate(request_stream, window_duration=1.0):
        print(f"RPS: {count}")
```

Stream Buffering:
```python
class StreamBuffer:
    def __init__(self, buffer_size=1000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def process(self, stream):
        for item in stream:
            self.buffer.append(item)
            
            # Flush when full
            if len(self.buffer) >= self.buffer_size:
                yield self.flush()
    
    def flush(self):
        result = process_batch(self.buffer)
        self.buffer = []
        return result
```

Real-Time Analytics:
```python
def real_time_analytics(event_stream):
    '''Compute metrics in real-time'''
    metrics = {
        'count': 0,
        'sum': 0,
        'recent': []
    }
    
    for event in event_stream:
        # Update metrics
        metrics['count'] += 1
        metrics['sum'] += event['value']
        metrics['recent'].append(event)
        
        # Keep recent bounded
        if len(metrics['recent']) > 1000:
            metrics['recent'].pop(0)
        
        # Emit current state
        yield {
            'count': metrics['count'],
            'average': metrics['sum'] / metrics['count'],
            'recent_avg': sum(e['value'] for e in metrics['recent']) / len(metrics['recent'])
        }
```

Stream Splitting:
```python
def split_stream(stream, predicate):
    '''Split stream into two based on condition'''
    for item in stream:
        if predicate(item):
            yield ('true_stream', item)
        else:
            yield ('false_stream', item)

# Usage
for stream_name, item in split_stream(data, is_valid):
    if stream_name == 'true_stream':
        handle_valid(item)
    else:
        handle_invalid(item)
```

Stream Rate Limiting:
```python
import time

def rate_limited_stream(stream, max_per_second):
    '''Limit stream processing rate'''
    min_interval = 1.0 / max_per_second
    last_time = 0
    
    for item in stream:
        current_time = time.time()
        elapsed = current_time - last_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        yield item
        last_time = time.time()
```

Best Practices:
✓ Process incrementally
✓ Handle backpressure
✓ Use generators for memory efficiency
✓ Implement error recovery
✓ Monitor stream health
✓ Bound buffer sizes
✓ Enable graceful shutdown

Use Cases:
• Real-time data processing
• Log processing
• Sensor data streams
• Live video/audio
• Chat applications
• Financial tick data
• IoT telemetry

Key Insight:
Streaming execution processes data continuously
as it arrives, enabling real-time, low-latency systems.
"""
    
    user_prompt = f"Process stream from: {source}"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"🌊 Streaming:\n{response.content}")],
        "chunks_processed": chunks + 1
    }


def build_streaming_graph():
    workflow = StateGraph(StreamingState)
    workflow.add_node("streaming_processor", streaming_processor_agent)
    workflow.add_edge(START, "streaming_processor")
    workflow.add_edge("streaming_processor", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_streaming_graph()
    
    print("=== Streaming Execution MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "stream_source": "real-time sensor data",
        "chunks_processed": 0,
        "buffer": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print(f"Chunks processed: {result.get('chunks_processed', 0)}")
    print(f"{'='*70}")
    print("Pattern 167: Streaming Execution - COMPLETE")
    print(f"{'='*70}")

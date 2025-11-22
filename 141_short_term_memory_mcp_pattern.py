"""
Short-Term Memory MCP Pattern

This pattern implements short-term memory for temporary information storage
during immediate task execution.

Key Features:
- Temporary storage
- Limited capacity
- Quick access
- Volatile memory
- Working context
"""

from typing import TypedDict, Sequence, Annotated, List, Dict, Deque
from collections import deque
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ShortTermMemoryState(TypedDict):
    """State for short-term memory pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    current_task: str
    short_term_buffer: Deque[Dict]
    buffer_size: int
    active_context: Dict


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def short_term_memory_agent(state: ShortTermMemoryState) -> ShortTermMemoryState:
    """Manages short-term memory operations"""
    current_task = state.get("current_task", "")
    buffer_size = state.get("buffer_size", 5)
    
    system_prompt = """You are a short-term memory expert.

Short-Term Memory (STM):
• Temporary storage (seconds to minutes)
• Limited capacity (7±2 items - Miller's Law)
• Fast access
• Volatile (lost when task ends)
• Working context for current task

Key operations: store, retrieve, forget."""
    
    user_prompt = f"""Task: {current_task}

Manage short-term memory for this task.
Show what to keep in immediate memory."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Simulate STM buffer
    stm_buffer = deque(maxlen=buffer_size)
    stm_buffer.append({
        "item": "current_input",
        "timestamp": "now",
        "relevance": "high"
    })
    
    report = f"""
    🧠 Short-Term Memory Agent:
    
    Memory Configuration:
    • Buffer Size: {buffer_size} items (7±2 rule)
    • Access Time: O(1) - immediate
    • Retention: Task duration only
    • Volatility: High (cleared on task end)
    
    Short-Term Memory Implementation:
    ```python
    from collections import deque
    
    class ShortTermMemory:
        def __init__(self, capacity=7):
            self.buffer = deque(maxlen=capacity)
            self.capacity = capacity
        
        def store(self, item):
            '''Store item in STM (FIFO if full)'''
            self.buffer.append({{
                'content': item,
                'timestamp': time.time(),
                'access_count': 0
            }})
        
        def retrieve(self, query):
            '''Quick retrieval from STM'''
            for item in self.buffer:
                if self.matches(item, query):
                    item['access_count'] += 1
                    return item['content']
            return None
        
        def clear(self):
            '''Clear STM (end of task)'''
            self.buffer.clear()
        
        def get_active_items(self):
            '''Get all items in STM'''
            return list(self.buffer)
    ```
    
    STM Characteristics:
    
    Capacity Limits (Miller's Law):
    ```
    Original: 7±2 items (5-9 range)
    Modern: 4±1 items (more accurate)
    
    Chunking helps:
    Individual: 7 1 4 9 2 6 5 → 7 items
    Chunked: 714-926-5 → 3 chunks
    ```
    
    Duration:
    ```
    Without rehearsal: 15-30 seconds
    With rehearsal: Minutes
    With attention: Extended
    
    Decay function:
    retention(t) = initial * e^(-t/τ)
    where τ = decay constant
    ```
    
    Memory Operations:
    
    Storage:
    ```python
    def store_in_stm(item, priority='normal'):
        if stm.is_full():
            if priority == 'high':
                # Evict lowest priority
                stm.evict_least_important()
            else:
                # FIFO: oldest item dropped
                pass
        
        stm.append(item)
    ```
    
    Retrieval:
    ```python
    def retrieve_from_stm(cue):
        # Scan STM buffer (fast - small size)
        for item in stm.buffer:
            similarity = compute_similarity(cue, item)
            if similarity > threshold:
                return item
        
        return None  # Not in STM
    ```
    
    Rehearsal:
    ```python
    def rehearse(item):
        '''Keep item in STM longer'''
        # Refresh timestamp
        item.last_access = now()
        
        # Move to front (recency)
        stm.buffer.remove(item)
        stm.buffer.append(item)
        
        # Increase strength
        item.strength += 1
    ```
    
    STM vs Registers vs Cache:
    
    Comparison:
    ```
    CPU Register:
    • Size: Bytes
    • Speed: ~1 cycle
    • Purpose: Immediate computation
    
    CPU Cache (L1):
    • Size: KB
    • Speed: ~4 cycles
    • Purpose: Fast data access
    
    Short-Term Memory:
    • Size: ~4-7 items
    • Speed: Milliseconds
    • Purpose: Active task context
    
    Working Memory:
    • Size: Task-dependent
    • Speed: Seconds
    • Purpose: Complex reasoning
    ```
    
    Use Cases:
    
    Conversation:
    ```python
    stm.store("User just asked about Python")
    stm.store("Topic: web scraping")
    stm.store("User skill level: intermediate")
    
    # Generate response using STM context
    context = stm.get_active_items()
    response = generate_response(query, context)
    ```
    
    Multi-Step Tasks:
    ```python
    # Step 1: Parse request
    stm.store({"step": 1, "result": parsed_data})
    
    # Step 2: Validate (uses step 1 result)
    previous = stm.retrieve("step 1")
    validated = validate(previous.result)
    stm.store({"step": 2, "result": validated})
    
    # Step 3: Process (uses step 2)
    data = stm.retrieve("step 2")
    final = process(data.result)
    ```
    
    Attention Management:
    ```python
    def update_attention(focus_item):
        # Highlight in STM
        for item in stm.buffer:
            if item == focus_item:
                item.attention = 'high'
            else:
                item.attention = 'low'
        
        # High attention = better retention
        high_attention_items = [
            i for i in stm.buffer 
            if i.attention == 'high'
        ]
    ```
    
    Integration Patterns:
    
    STM → LTM Transfer:
    ```python
    def consolidate_if_important(item):
        if item.importance > threshold:
            # Move to long-term memory
            ltm.store(item)
            
        if item.rehearsal_count > 3:
            # Frequently accessed → LTM
            ltm.store(item)
    ```
    
    STM + Working Memory:
    ```python
    class WorkingMemory:
        def __init__(self):
            self.stm = ShortTermMemory(capacity=7)
            self.focus = None
            self.goals = []
        
        def process_task(self, task):
            # Keep task goal in STM
            self.stm.store(task.goal)
            
            # Process subtasks
            for subtask in task.subtasks:
                self.focus = subtask
                self.stm.store(subtask)
                result = execute(subtask)
                self.stm.store(result)
    ```
    
    Performance Optimization:
    
    Chunking:
    ```python
    def chunk_information(items):
        '''Group related items to fit in STM'''
        chunks = []
        current_chunk = []
        
        for item in items:
            if len(current_chunk) < chunk_size:
                current_chunk.append(item)
            else:
                chunks.append(current_chunk)
                current_chunk = [item]
        
        return chunks
    ```
    
    Prioritization:
    ```python
    def prioritize_stm_items():
        '''Keep most important items'''
        scored = [
            (item, calculate_importance(item))
            for item in stm.buffer
        ]
        
        # Sort by importance
        sorted_items = sorted(scored, 
                             key=lambda x: x[1], 
                             reverse=True)
        
        # Keep top items
        stm.buffer = [item for item, _ in sorted_items[:stm.capacity]]
    ```
    
    Current STM State:
    • Active Items: {len(stm_buffer)}
    • Capacity: {buffer_size}
    • Utilization: {len(stm_buffer)/buffer_size*100:.1f}%
    
    Best Practices:
    ✓ Limit STM to 4-7 items
    ✓ Use chunking for more info
    ✓ Refresh important items
    ✓ Clear after task completion
    ✓ Promote to LTM if needed
    
    Key Insight:
    Short-term memory provides fast, temporary storage
    for immediate task context - like CPU cache for cognition.
    """
    
    return {
        "messages": [AIMessage(content=f"🧠 Short-Term Memory Agent:\n{report}\n\n{response.content}")],
        "short_term_buffer": stm_buffer,
        "active_context": {"items": len(stm_buffer)}
    }


def build_short_term_memory_graph():
    workflow = StateGraph(ShortTermMemoryState)
    workflow.add_node("short_term_memory_agent", short_term_memory_agent)
    workflow.add_edge(START, "short_term_memory_agent")
    workflow.add_edge("short_term_memory_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_short_term_memory_graph()
    
    print("=== Short-Term Memory MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "current_task": "Process customer inquiry and provide relevant response",
        "short_term_buffer": deque(maxlen=7),
        "buffer_size": 7,
        "active_context": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 141: Short-Term Memory - COMPLETE")
    print(f"{'='*70}")

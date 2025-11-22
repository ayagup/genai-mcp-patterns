"""
Context Window Management MCP Pattern

This pattern implements context window management for handling
limited context capacity in language models.

Key Features:
- Token budget management
- Sliding window
- Priority-based retention
- Compression strategies
- Context overflow handling
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ContextWindowState(TypedDict):
    """State for context window management pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    max_tokens: int
    current_tokens: int
    context_items: List[Dict]
    retention_strategy: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def context_window_manager(state: ContextWindowState) -> ContextWindowState:
    """Manages context window operations"""
    max_tokens = state.get("max_tokens", 8000)
    current_tokens = state.get("current_tokens", 0)
    strategy = state.get("retention_strategy", "sliding_window")
    
    system_prompt = """You are a context window management expert.

Context Window Management:
• Fixed token limit (e.g., 8K, 128K tokens)
• Must fit conversation + system prompts
• Evict old content when full
• Preserve important information
• Optimize token usage

Critical for long conversations."""
    
    user_prompt = f"""Max Tokens: {max_tokens}
Current: {current_tokens}
Strategy: {strategy}

Design context window management system.
Show how to handle token limits."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    📏 Context Window Manager:
    
    Window Status:
    • Max Capacity: {max_tokens} tokens
    • Current Usage: {current_tokens} tokens
    • Available: {max_tokens - current_tokens} tokens
    • Utilization: {current_tokens/max_tokens*100:.1f}%
    
    Context Window Management Implementation:
    ```python
    class ContextWindowManager:
        '''Manage limited context capacity'''
        
        def __init__(self, max_tokens=8000):
            self.max_tokens = max_tokens
            self.context = []
            self.system_prompt_tokens = 0
            self.reserved_tokens = 1000  # For response
        
        def add_to_context(self, item, priority='normal'):
            '''Add item to context window'''
            item_tokens = self.count_tokens(item)
            
            # Check if fits
            available = self.available_tokens()
            
            if item_tokens > available:
                # Evict to make room
                self.make_room(item_tokens, priority)
            
            # Add item
            self.context.append({{
                'content': item,
                'tokens': item_tokens,
                'priority': priority,
                'timestamp': time.time(),
                'access_count': 0
            }})
            
            return True
        
        def available_tokens(self):
            '''Calculate available space'''
            used = (
                self.system_prompt_tokens +
                sum(item['tokens'] for item in self.context) +
                self.reserved_tokens
            )
            return self.max_tokens - used
        
        def make_room(self, needed_tokens, new_item_priority='normal'):
            '''Evict items to free space'''
            freed = 0
            
            while freed < needed_tokens and self.context:
                # Select victim based on strategy
                victim = self.select_victim(new_item_priority)
                
                if victim:
                    freed += victim['tokens']
                    self.context.remove(victim)
                else:
                    break
            
            return freed >= needed_tokens
        
        def select_victim(self, new_priority):
            '''Choose item to evict'''
            # Don't evict high priority
            candidates = [
                item for item in self.context
                if item['priority'] != 'high'
            ]
            
            if not candidates:
                return None
            
            # Score each candidate
            scored = []
            for item in candidates:
                score = self.eviction_score(item)
                scored.append((item, score))
            
            # Evict lowest score (least valuable)
            scored.sort(key=lambda x: x[1])
            return scored[0][0] if scored else None
        
        def eviction_score(self, item):
            '''Calculate value for retention'''
            score = 0
            
            # Recency (newer = higher score)
            age = time.time() - item['timestamp']
            score += 100 / (1 + age/3600)  # Decay by hour
            
            # Frequency (more accessed = higher)
            score += item['access_count'] * 10
            
            # Priority
            priority_scores = {{'high': 1000, 'normal': 100, 'low': 10}}
            score += priority_scores.get(item['priority'], 100)
            
            # Size (smaller = easier to keep)
            score += 100 / (1 + item['tokens']/100)
            
            return score
    ```
    
    Management Strategies:
    
    Sliding Window:
    ```python
    def sliding_window(context, max_tokens):
        '''FIFO: keep most recent items'''
        tokens = 0
        window = []
        
        # Add from newest to oldest
        for item in reversed(context):
            if tokens + item['tokens'] <= max_tokens:
                window.insert(0, item)
                tokens += item['tokens']
            else:
                break
        
        return window
    ```
    
    Priority-Based:
    ```python
    def priority_retention(context, max_tokens):
        '''Keep highest priority items'''
        # Sort by priority and recency
        sorted_items = sorted(
            context,
            key=lambda x: (
                {{'high': 3, 'normal': 2, 'low': 1}}.get(x['priority'], 2),
                x['timestamp']
            ),
            reverse=True
        )
        
        tokens = 0
        retained = []
        
        for item in sorted_items:
            if tokens + item['tokens'] <= max_tokens:
                retained.append(item)
                tokens += item['tokens']
        
        # Restore chronological order
        retained.sort(key=lambda x: x['timestamp'])
        
        return retained
    ```
    
    Summarization:
    ```python
    def summarize_old_context(context, max_tokens):
        '''Compress old messages'''
        # Keep recent messages as-is
        recent_tokens = max_tokens * 0.6
        recent = []
        tokens = 0
        
        for item in reversed(context):
            if tokens + item['tokens'] <= recent_tokens:
                recent.insert(0, item)
                tokens += item['tokens']
            else:
                break
        
        # Summarize old messages
        old = context[:len(context)-len(recent)]
        if old:
            summary = summarize_messages(old)
            recent.insert(0, {{
                'content': summary,
                'tokens': count_tokens(summary),
                'type': 'summary',
                'priority': 'high'
            }})
        
        return recent
    ```
    
    Semantic Chunking:
    ```python
    def semantic_chunk_management(context, max_tokens):
        '''Group related messages, evict by topic'''
        # Cluster by topic
        chunks = cluster_by_topic(context)
        
        # Score chunks
        scored_chunks = []
        for chunk in chunks:
            relevance = calculate_relevance(chunk, current_topic)
            scored_chunks.append((chunk, relevance))
        
        # Keep most relevant
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        tokens = 0
        
        for chunk, relevance in scored_chunks:
            chunk_tokens = sum(item['tokens'] for item in chunk)
            if tokens + chunk_tokens <= max_tokens:
                selected.extend(chunk)
                tokens += chunk_tokens
        
        return selected
    ```
    
    Token Optimization:
    
    Compression Techniques:
    ```python
    def compress_context(messages):
        '''Reduce token usage'''
        compressed = []
        
        for msg in messages:
            # Remove redundancy
            deduplicated = remove_duplicate_info(msg)
            
            # Abbreviate verbose parts
            abbreviated = abbreviate(deduplicated)
            
            # Remove filler words
            concise = remove_filler(abbreviated)
            
            compressed.append(concise)
        
        return compressed
    ```
    
    Template Extraction:
    ```python
    def extract_templates(repeated_patterns):
        '''Extract common patterns to system prompt'''
        # Identify repeated instructions
        patterns = find_common_patterns(repeated_patterns)
        
        # Move to system prompt (counted once)
        for pattern in patterns:
            system_prompt += f"\\n{{pattern}}"
            
            # Remove from individual messages
            for msg in messages:
                if pattern in msg:
                    msg = msg.replace(pattern, "[ref:{{pattern.id}}]")
        
        # Saves tokens in long conversations
        return system_prompt, messages
    ```
    
    Context Patterns:
    
    Conversation Memory:
    ```python
    class ConversationContextManager:
        def __init__(self, max_tokens=8000):
            self.max_tokens = max_tokens
            self.system_prompt = ""  # Fixed
            self.messages = []  # Variable
            self.summary = None  # Compressed history
        
        def add_turn(self, user_msg, assistant_msg):
            '''Add conversation turn'''
            # Add messages
            self.messages.append(user_msg)
            self.messages.append(assistant_msg)
            
            # Check limit
            if self.total_tokens() > self.max_tokens * 0.8:
                self.compress_old_messages()
        
        def compress_old_messages(self):
            '''Summarize older part of conversation'''
            # Keep last N messages
            keep_recent = 10
            old_messages = self.messages[:-keep_recent]
            recent_messages = self.messages[-keep_recent:]
            
            # Summarize old
            self.summary = create_summary(old_messages)
            
            # Replace with summary
            self.messages = recent_messages
    ```
    
    Document Context:
    ```python
    class DocumentContextManager:
        '''Manage long document context'''
        
        def __init__(self, document, max_tokens=8000):
            self.full_document = document
            self.max_tokens = max_tokens
        
        def get_relevant_context(self, query):
            '''Extract relevant sections'''
            # Chunk document
            chunks = chunk_document(self.full_document)
            
            # Score relevance to query
            scored = [
                (chunk, relevance(chunk, query))
                for chunk in chunks
            ]
            
            # Select top chunks that fit
            scored.sort(key=lambda x: x[1], reverse=True)
            
            context = []
            tokens = 0
            
            for chunk, score in scored:
                chunk_tokens = count_tokens(chunk)
                if tokens + chunk_tokens <= self.max_tokens:
                    context.append(chunk)
                    tokens += chunk_tokens
            
            return context
    ```
    
    Multi-Turn Planning:
    
    Token Budget:
    ```python
    def plan_token_budget(task, max_tokens=8000):
        '''Allocate tokens across turns'''
        # Reserve for system
        system_tokens = 500
        
        # Reserve for final response
        response_tokens = 1500
        
        # Available for context
        context_budget = max_tokens - system_tokens - response_tokens
        
        # Plan turns
        estimated_turns = estimate_turns(task)
        tokens_per_turn = context_budget / (estimated_turns * 2)  # user + assistant
        
        return {{
            'system': system_tokens,
            'per_turn': tokens_per_turn,
            'response': response_tokens,
            'total_turns': estimated_turns
        }}
    ```
    
    Best Practices:
    ✓ Monitor token usage continuously
    ✓ Prioritize important information
    ✓ Summarize old context
    ✓ Use semantic compression
    ✓ Reserve tokens for responses
    ✓ Plan token budget for multi-turn
    
    Key Insight:
    Context window management is crucial for long interactions -
    must balance recency, relevance, and priority while staying
    within strict token limits.
    """
    
    return {
        "messages": [AIMessage(content=f"📏 Context Window Manager:\n{report}\n\n{response.content}")],
        "current_tokens": max_tokens // 2
    }


def build_context_window_graph():
    workflow = StateGraph(ContextWindowState)
    workflow.add_node("context_window_manager", context_window_manager)
    workflow.add_edge(START, "context_window_manager")
    workflow.add_edge("context_window_manager", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_context_window_graph()
    
    print("=== Context Window Management MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "max_tokens": 8000,
        "current_tokens": 3500,
        "context_items": [],
        "retention_strategy": "priority_based"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 149: Context Window Management - COMPLETE")
    print(f"{'='*70}")

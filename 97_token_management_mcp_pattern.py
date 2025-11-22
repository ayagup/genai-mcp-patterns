"""
Token Management MCP Pattern

This pattern optimizes LLM token usage to manage costs and stay within
API rate limits while maintaining response quality.

Key Features:
- Token counting and estimation
- Context window optimization
- Token budget management
- Response streaming
- Cost tracking per request
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TokenManagementState(TypedDict):
    """State for token management pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    model_name: str
    max_context_window: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    token_budget: int
    cost_per_1k_input: float
    cost_per_1k_output: float
    total_cost: float
    optimization_strategy: str  # "truncate", "summarize", "window", "streaming"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Token Counter
def token_counter(state: TokenManagementState) -> TokenManagementState:
    """Estimates and tracks token usage"""
    model_name = state.get("model_name", "gpt-4")
    max_context_window = state.get("max_context_window", 8192)
    
    system_message = SystemMessage(content="""You are a token counter.
    Estimate and track token usage for LLM interactions.""")
    
    user_message = HumanMessage(content=f"""Track token usage:

Model: {model_name}
Context Window: {max_context_window}

Estimate token consumption.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate token counting
    input_tokens = 150  # System + user message
    output_tokens = 100  # Response
    total_tokens = input_tokens + output_tokens
    
    report = f"""
    🔢 Token Counting:
    
    Usage Summary:
    • Model: {model_name}
    • Input Tokens: {input_tokens:,}
    • Output Tokens: {output_tokens:,}
    • Total Tokens: {total_tokens:,}
    • Context Window: {max_context_window:,}
    • Utilization: {(total_tokens/max_context_window*100):.1f}%
    
    Token Limits by Model:
    
    OpenAI:
    • GPT-4: 8,192 tokens
    • GPT-4-32K: 32,768 tokens
    • GPT-4-Turbo: 128,000 tokens
    • GPT-3.5-Turbo: 4,096 tokens
    • GPT-3.5-Turbo-16K: 16,384 tokens
    
    Anthropic Claude:
    • Claude 3 Opus: 200,000 tokens
    • Claude 3 Sonnet: 200,000 tokens
    • Claude 3 Haiku: 200,000 tokens
    • Claude 2.1: 200,000 tokens
    
    Google:
    • Gemini 1.5 Pro: 1,000,000 tokens
    • Gemini 1.5 Flash: 1,000,000 tokens
    • PaLM 2: 8,192 tokens
    
    Token Counting Methods:
    
    Exact Counting (tiktoken):
    ```python
    import tiktoken
    
    def count_tokens(text, model="gpt-4"):
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    # Count tokens
    tokens = count_tokens("Hello, world!")
    print(f"Tokens: {{tokens}}")
    ```
    
    Estimation (Rule of Thumb):
    • English: ~4 characters per token
    • Code: ~2-3 characters per token
    • Numbers: Variable (1-3 tokens)
    • Special chars: Usually 1 token each
    
    LangChain Integration:
    ```python
    from langchain.callbacks import get_openai_callback
    
    with get_openai_callback() as cb:
        response = llm.invoke(messages)
        print(f"Tokens: {{cb.total_tokens}}")
        print(f"Cost: ${{cb.total_cost:.4f}}")
    ```
    
    Token Distribution:
    
    Typical Breakdown:
    • System prompt: 50-200 tokens
    • User message: 100-500 tokens
    • Context/history: 500-2000 tokens
    • Response: 200-1000 tokens
    • Total: 850-3700 tokens
    
    What Counts as Tokens:
    • Input text (prompts)
    • Conversation history
    • System messages
    • Function definitions
    • Response text
    • Special tokens (<|im_start|>)
    
    Optimization Strategies:
    
    Reduce Input:
    • Concise prompts
    • Remove redundancy
    • Compress context
    • Smart summarization
    
    Limit Output:
    • max_tokens parameter
    • Stop sequences
    • Targeted responses
    • Streaming control
    
    Context Management:
    • Sliding window
    • Summarize old messages
    • Remove irrelevant context
    • Semantic search for relevant
    """
    
    return {
        "messages": [AIMessage(content=f"🔢 Token Counter:\n{response.content}\n{report}")],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    }


# Cost Calculator
def cost_calculator(state: TokenManagementState) -> TokenManagementState:
    """Calculates costs based on token usage"""
    model_name = state.get("model_name", "gpt-4")
    input_tokens = state.get("input_tokens", 0)
    output_tokens = state.get("output_tokens", 0)
    cost_per_1k_input = state.get("cost_per_1k_input", 0.03)
    cost_per_1k_output = state.get("cost_per_1k_output", 0.06)
    
    system_message = SystemMessage(content="""You are a cost calculator.
    Calculate LLM API costs based on token usage.""")
    
    user_message = HumanMessage(content=f"""Calculate costs:

Model: {model_name}
Input Tokens: {input_tokens}
Output Tokens: {output_tokens}
Input Rate: ${cost_per_1k_input}/1K
Output Rate: ${cost_per_1k_output}/1K

Calculate total cost.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate costs
    input_cost = (input_tokens / 1000) * cost_per_1k_input
    output_cost = (output_tokens / 1000) * cost_per_1k_output
    total_cost = input_cost + output_cost
    
    report = f"""
    💵 Cost Calculation:
    
    Cost Breakdown:
    • Input Cost: ${input_cost:.4f}
    • Output Cost: ${output_cost:.4f}
    • Total Cost: ${total_cost:.4f}
    
    Pricing (as of 2024):
    
    OpenAI GPT-4:
    • Input: $0.03/1K tokens
    • Output: $0.06/1K tokens
    • Example: 10K in + 2K out = $0.42
    
    OpenAI GPT-4-Turbo:
    • Input: $0.01/1K tokens
    • Output: $0.03/1K tokens
    • Example: 10K in + 2K out = $0.16
    
    OpenAI GPT-3.5-Turbo:
    • Input: $0.0015/1K tokens
    • Output: $0.002/1K tokens
    • Example: 10K in + 2K out = $0.019
    
    Anthropic Claude 3 Opus:
    • Input: $0.015/1K tokens
    • Output: $0.075/1K tokens
    
    Anthropic Claude 3 Sonnet:
    • Input: $0.003/1K tokens
    • Output: $0.015/1K tokens
    
    Google Gemini 1.5 Pro:
    • Input: $0.00125/1K tokens
    • Output: $0.00375/1K tokens
    
    Cost Optimization:
    
    Choose Right Model:
    • GPT-3.5 for simple tasks
    • GPT-4 for complex reasoning
    • Claude for long context
    • Gemini for cost-effective
    
    Reduce Token Usage:
    • Shorter prompts
    • Limit max_tokens
    • Summarize context
    • Cache responses
    
    Batch Processing:
    • Group similar requests
    • Parallel processing
    • Async operations
    • Reduce overhead
    
    Caching:
    • Cache common responses
    • Semantic caching
    • Prompt caching (Claude)
    • Response reuse
    
    Cost Tracking:
    ```python
    class TokenTracker:
        def __init__(self):
            self.total_input = 0
            self.total_output = 0
            self.total_cost = 0.0
        
        def track(self, input_tokens, output_tokens, 
                  input_rate, output_rate):
            self.total_input += input_tokens
            self.total_output += output_tokens
            
            cost = (input_tokens / 1000 * input_rate + 
                    output_tokens / 1000 * output_rate)
            self.total_cost += cost
            
            return cost
        
        def report(self):
            return {{
                'total_input': self.total_input,
                'total_output': self.total_output,
                'total_cost': self.total_cost
            }}
    ```
    
    Budget Management:
    • Set daily/monthly limits
    • Alert on threshold
    • Auto-cutoff at limit
    • Per-user quotas
    • Cost allocation tags
    
    Monthly Cost Examples:
    
    Light Usage (10K requests):
    • GPT-3.5: ~$20-50/month
    • GPT-4: ~$400-800/month
    
    Medium Usage (100K requests):
    • GPT-3.5: ~$200-500/month
    • GPT-4: ~$4,000-8,000/month
    
    Heavy Usage (1M requests):
    • GPT-3.5: ~$2,000-5,000/month
    • GPT-4: ~$40,000-80,000/month
    """
    
    return {
        "messages": [AIMessage(content=f"💵 Cost Calculator:\n{response.content}\n{report}")],
        "total_cost": total_cost
    }


# Token Monitor
def token_monitor(state: TokenManagementState) -> TokenManagementState:
    """Monitors token usage and provides optimization recommendations"""
    model_name = state.get("model_name", "")
    input_tokens = state.get("input_tokens", 0)
    output_tokens = state.get("output_tokens", 0)
    total_tokens = state.get("total_tokens", 0)
    total_cost = state.get("total_cost", 0.0)
    token_budget = state.get("token_budget", 1000000)
    
    budget_utilization = (total_tokens / token_budget * 100) if token_budget > 0 else 0
    
    summary = f"""
    📊 TOKEN MANAGEMENT COMPLETE
    
    Usage Summary:
    • Model: {model_name}
    • Input Tokens: {input_tokens:,}
    • Output Tokens: {output_tokens:,}
    • Total Tokens: {total_tokens:,}
    • Budget: {token_budget:,}
    • Utilization: {budget_utilization:.2f}%
    • Cost: ${total_cost:.4f}
    
    Token Management Pattern Process:
    1. Token Counter → Track token usage
    2. Cost Calculator → Calculate costs
    3. Monitor → Optimize and alert
    
    Advanced Optimization Techniques:
    
    Context Window Management:
    ```python
    class ContextManager:
        def __init__(self, max_tokens=4000):
            self.max_tokens = max_tokens
            self.messages = []
        
        def add_message(self, message):
            self.messages.append(message)
            self.trim_if_needed()
        
        def trim_if_needed(self):
            total = sum(count_tokens(m) for m in self.messages)
            
            while total > self.max_tokens and len(self.messages) > 1:
                # Remove oldest non-system message
                self.messages.pop(1)
                total = sum(count_tokens(m) for m in self.messages)
    ```
    
    Summarization Strategy:
    ```python
    def summarize_context(messages, max_summary_tokens=500):
        if count_tokens(messages) <= max_summary_tokens:
            return messages
        
        # Summarize old messages
        old_messages = messages[:-5]  # Keep last 5
        summary_prompt = f"Summarize: {{old_messages}}"
        summary = llm.invoke(summary_prompt)
        
        return [summary] + messages[-5:]
    ```
    
    Streaming for Token Control:
    ```python
    from openai import OpenAI
    
    client = OpenAI()
    
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{{"role": "user", "content": "Hello"}}],
        stream=True,
        max_tokens=100  # Limit output
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
    ```
    
    Best Practices:
    
    Prompt Engineering:
    • Be concise and specific
    • Use examples efficiently
    • Avoid redundancy
    • Structure with markdown
    • Use role definitions
    
    Context Management:
    • Keep only relevant history
    • Summarize old conversations
    • Use semantic search
    • Implement sliding window
    • Cache repeated content
    
    Output Control:
    • Set max_tokens appropriately
    • Use stop sequences
    • Request structured output
    • Limit response length
    • Stream for early stopping
    
    Monitoring:
    • Track per-request tokens
    • Daily/monthly aggregates
    • Cost per user
    • Cost per feature
    • Trend analysis
    
    Budgeting:
    • Set token quotas
    • Implement rate limits
    • Alert on thresholds
    • Auto-disable on exceed
    • Reserved budget pools
    
    Real-World Patterns:
    
    Chatbots:
    • Sliding window (10 messages)
    • Summarize after 20 messages
    • Budget: 4K tokens/conversation
    • Cost: ~$0.02-0.10/conversation
    
    Document Analysis:
    • Chunk large documents
    • Process in parallel
    • Summarize sections
    • Budget: Variable by doc size
    • Cost: ~$0.10-2.00/document
    
    Code Generation:
    • Targeted prompts
    • Limit output length
    • Use smaller models
    • Budget: 2K tokens/request
    • Cost: ~$0.01-0.05/request
    
    Key Metrics:
    • Tokens per request (avg, p95)
    • Cost per request
    • Cost per user
    • Daily token usage
    • Monthly cost trends
    • Budget burn rate
    
    Key Insight:
    Token management is critical for LLM cost control.
    Monitor usage, optimize prompts, and implement
    budgets to keep costs predictable and sustainable.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Token Monitor:\n{summary}")]
    }


# Build the graph
def build_token_management_graph():
    """Build the token management pattern graph"""
    workflow = StateGraph(TokenManagementState)
    
    workflow.add_node("counter", token_counter)
    workflow.add_node("calculator", cost_calculator)
    workflow.add_node("monitor", token_monitor)
    
    workflow.add_edge(START, "counter")
    workflow.add_edge("counter", "calculator")
    workflow.add_edge("calculator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_token_management_graph()
    
    print("=== Token Management MCP Pattern ===\n")
    
    # Test Case: GPT-4 token and cost tracking
    print("\n" + "="*70)
    print("TEST CASE: LLM Token Management")
    print("="*70)
    
    state = {
        "messages": [],
        "model_name": "gpt-4",
        "max_context_window": 8192,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "token_budget": 1000000,
        "cost_per_1k_input": 0.03,
        "cost_per_1k_output": 0.06,
        "total_cost": 0.0,
        "optimization_strategy": "window"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nTotal Tokens: {result.get('total_tokens', 0):,}")
    print(f"Total Cost: ${result.get('total_cost', 0):.4f}")

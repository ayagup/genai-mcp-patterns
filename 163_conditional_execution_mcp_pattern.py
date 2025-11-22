"""
Conditional Execution MCP Pattern

This pattern implements conditional execution where tasks
are executed based on runtime conditions and predicates.

Key Features:
- Predicate evaluation
- Branch selection
- Dynamic routing
- Condition chaining
- Fallback handling
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ConditionalState(TypedDict):
    """State for conditional execution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    condition_value: str
    score: float
    route_taken: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def evaluator_agent(state: ConditionalState) -> ConditionalState:
    """Evaluate conditions"""
    condition = state.get("condition_value", "unknown")
    
    prompt = f"Evaluate condition: {condition}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Simulate condition evaluation
    score = 0.7  # Would be calculated based on actual logic
    
    return {
        "messages": [AIMessage(content=f"🔍 Evaluation: {response.content[:200]}")],
        "score": score
    }


def route_decision(state: ConditionalState) -> str:
    """Decide which path to take based on conditions"""
    score = state.get("score", 0.5)
    condition_value = state.get("condition_value", "")
    
    # Routing logic
    if score > 0.8:
        return "high_priority_path"
    elif score > 0.5:
        return "medium_priority_path"
    else:
        return "low_priority_path"


def high_priority_agent(state: ConditionalState) -> ConditionalState:
    """Handle high priority cases"""
    system_prompt = """You are a conditional execution expert.

Conditional Execution System:
```python
class ConditionalExecutor:
    def __init__(self):
        self.conditions = []
        self.handlers = {}
    
    def add_condition(self, predicate, handler):
        '''Register condition and handler'''
        self.conditions.append((predicate, handler))
    
    def execute(self, context):
        '''Execute based on conditions'''
        for predicate, handler in self.conditions:
            if predicate(context):
                return handler(context)
        
        # No condition matched
        return self.default_handler(context)
```

If-Then-Else Pattern:
```python
def conditional_execute(data):
    '''Basic conditional execution'''
    if condition_a(data):
        return handle_a(data)
    elif condition_b(data):
        return handle_b(data)
    else:
        return default_handler(data)
```

Guard Clauses:
```python
def process_with_guards(data):
    '''Early return pattern'''
    # Guard: invalid input
    if not is_valid(data):
        return error_result("Invalid input")
    
    # Guard: already processed
    if is_processed(data):
        return cached_result(data)
    
    # Guard: insufficient resources
    if not has_resources():
        return queue_for_later(data)
    
    # Main processing
    return process(data)
```

Strategy Pattern:
```python
class ConditionalStrategy:
    def __init__(self):
        self.strategies = {}
    
    def register(self, condition, strategy):
        self.strategies[condition] = strategy
    
    def execute(self, context):
        for condition, strategy in self.strategies.items():
            if condition(context):
                return strategy.execute(context)

# Usage
executor = ConditionalStrategy()
executor.register(lambda x: x['type'] == 'A', StrategyA())
executor.register(lambda x: x['type'] == 'B', StrategyB())
result = executor.execute(context)
```

Multi-Condition Evaluation:
```python
def evaluate_conditions(data):
    '''Check multiple conditions'''
    results = {
        'is_valid': validate(data),
        'is_authorized': check_auth(data),
        'has_quota': check_quota(data),
        'is_urgent': check_urgency(data)
    }
    
    # All must pass
    if all(results.values()):
        return process_high_priority(data)
    
    # Some failed
    elif results['is_valid'] and results['is_authorized']:
        return process_normal(data)
    
    # Validation failed
    else:
        return reject(data)
```

Priority-Based Routing:
```python
def priority_router(task):
    '''Route based on priority'''
    priority = calculate_priority(task)
    
    if priority == 'critical':
        return immediate_handler(task)
    elif priority == 'high':
        return fast_queue(task)
    elif priority == 'medium':
        return normal_queue(task)
    else:
        return batch_queue(task)
```

Feature Flag Pattern:
```python
class FeatureFlags:
    def __init__(self):
        self.flags = {}
    
    def is_enabled(self, feature, context=None):
        '''Check if feature enabled'''
        flag = self.flags.get(feature)
        
        if flag is None:
            return False
        
        # Simple boolean
        if isinstance(flag, bool):
            return flag
        
        # Context-based
        if callable(flag):
            return flag(context)
        
        return False

def conditional_feature(user, feature):
    '''Execute based on feature flag'''
    if flags.is_enabled(feature, user):
        return new_implementation(user)
    else:
        return legacy_implementation(user)
```

Circuit Breaker Pattern:
```python
class CircuitBreaker:
    def __init__(self, threshold=5):
        self.failure_count = 0
        self.threshold = threshold
        self.state = 'closed'  # closed, open, half_open
    
    def execute(self, operation):
        if self.state == 'open':
            return fallback_operation()
        
        try:
            result = operation()
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            
            if self.state == 'open':
                return fallback_operation()
            raise

    def on_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.threshold:
            self.state = 'open'
    
    def on_success(self):
        self.failure_count = 0
        if self.state == 'half_open':
            self.state = 'closed'
```

Condition Chaining:
```python
class ConditionChain:
    def __init__(self):
        self.chain = []
    
    def add(self, condition, action):
        self.chain.append((condition, action))
        return self
    
    def execute(self, context):
        for condition, action in self.chain:
            if condition(context):
                result = action(context)
                # Update context with result
                context.update(result)
        
        return context

# Usage
chain = ConditionChain()
chain.add(lambda x: x['score'] > 0.8, high_quality_action)
chain.add(lambda x: x['urgent'], priority_action)
chain.add(lambda x: x['user_type'] == 'premium', premium_action)
result = chain.execute(data)
```

Predicate Composition:
```python
def and_predicate(*predicates):
    '''Combine predicates with AND'''
    return lambda x: all(p(x) for p in predicates)

def or_predicate(*predicates):
    '''Combine predicates with OR'''
    return lambda x: any(p(x) for p in predicates)

def not_predicate(predicate):
    '''Negate predicate'''
    return lambda x: not predicate(x)

# Usage
is_valid_premium = and_predicate(
    lambda x: x['is_valid'],
    lambda x: x['user_type'] == 'premium'
)

if is_valid_premium(user):
    process_premium(user)
```

Default Fallback:
```python
def execute_with_fallback(data):
    '''Always have a fallback'''
    try:
        if meets_condition_a(data):
            return strategy_a(data)
        elif meets_condition_b(data):
            return strategy_b(data)
        else:
            # Default fallback
            return default_strategy(data)
    except Exception as e:
        # Error fallback
        return error_handler(e, data)
```

Lazy Condition Evaluation:
```python
def lazy_conditions(data):
    '''Evaluate conditions only when needed'''
    # Short-circuit evaluation
    if expensive_check_1(data) and expensive_check_2(data):
        # expensive_check_2 only if check_1 passes
        return special_handling(data)
    
    return normal_handling(data)
```

Best Practices:
✓ Keep conditions simple and testable
✓ Use guard clauses for clarity
✓ Provide default/fallback handlers
✓ Log condition evaluations
✓ Avoid deep nesting
✓ Make conditions explicit
✓ Test all branches

Key Insight:
Conditional execution enables dynamic behavior
by routing execution based on runtime conditions.
"""
    
    user_prompt = "Handle high priority case"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"⚡ High Priority:\n{response.content}")],
        "route_taken": "high_priority"
    }


def medium_priority_agent(state: ConditionalState) -> ConditionalState:
    """Handle medium priority cases"""
    prompt = "Handle medium priority case"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=f"⚙️ Medium Priority: {response.content[:200]}")],
        "route_taken": "medium_priority"
    }


def low_priority_agent(state: ConditionalState) -> ConditionalState:
    """Handle low priority cases"""
    prompt = "Handle low priority case"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "messages": [AIMessage(content=f"📋 Low Priority: {response.content[:200]}")],
        "route_taken": "low_priority"
    }


def build_conditional_graph():
    workflow = StateGraph(ConditionalState)
    
    # Add nodes
    workflow.add_node("evaluator", evaluator_agent)
    workflow.add_node("high_priority", high_priority_agent)
    workflow.add_node("medium_priority", medium_priority_agent)
    workflow.add_node("low_priority", low_priority_agent)
    
    # Start with evaluation
    workflow.add_edge(START, "evaluator")
    
    # Conditional routing from evaluator
    workflow.add_conditional_edges(
        "evaluator",
        route_decision,
        {
            "high_priority_path": "high_priority",
            "medium_priority_path": "medium_priority",
            "low_priority_path": "low_priority"
        }
    )
    
    # All paths end
    workflow.add_edge("high_priority", END)
    workflow.add_edge("medium_priority", END)
    workflow.add_edge("low_priority", END)
    
    return workflow.compile()


if __name__ == "__main__":
    graph = build_conditional_graph()
    
    print("=== Conditional Execution MCP Pattern ===\n")
    
    # Test different conditions
    test_cases = [
        {"condition_value": "urgent_request", "score": 0.9},
        {"condition_value": "normal_request", "score": 0.6},
        {"condition_value": "low_priority_request", "score": 0.3}
    ]
    
    for i, test_state in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"Test Case {i}: {test_state['condition_value']} (score: {test_state['score']})")
        print(f"{'='*70}")
        
        state = {
            "messages": [],
            "condition_value": test_state["condition_value"],
            "score": test_state["score"],
            "route_taken": ""
        }
        
        result = graph.invoke(state)
        
        for msg in result["messages"]:
            print(f"\n{msg.content}")
        
        print(f"\nRoute taken: {result.get('route_taken', 'unknown')}")
    
    print(f"\n{'='*70}")
    print("Pattern 163: Conditional Execution - COMPLETE")
    print(f"{'='*70}")

"""
Confirmation MCP Pattern

This pattern implements confirmation mechanisms for verifying
user intent and critical actions before execution.

Key Features:
- Intent verification
- Action confirmation
- Risk assessment
- Explicit consent
- Undo mechanisms
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ConfirmationState(TypedDict):
    """State for confirmation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    action_to_confirm: str
    risk_level: str
    user_confirmed: bool
    confirmation_method: str


llm = ChatOpenAI(model="gpt-4", temperature=0.2)


def confirmation_agent(state: ConfirmationState) -> ConfirmationState:
    """Manages confirmation interactions"""
    action = state.get("action_to_confirm", "")
    risk_level = state.get("risk_level", "medium")
    
    system_prompt = """You are a confirmation expert.

Confirmation System:
• Verify user intent
• Assess action risk
• Request explicit consent
• Prevent errors
• Enable undo

Safety through verification."""
    
    user_prompt = f"""Action: {action}
Risk Level: {risk_level}

Design confirmation system.
Show when and how to confirm."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    ✅ Confirmation Agent:
    
    Confirmation Required:
    • Action: {action}
    • Risk: {risk_level}
    
    Implementation:
    ```python
    class ConfirmationSystem:
        def __init__(self):
            self.risk_assessor = RiskAssessor()
            self.confirmation_generator = ConfirmationGenerator()
        
        def should_confirm(self, action):
            '''Decide if confirmation needed'''
            risk = self.risk_assessor.assess(action)
            
            if risk == 'high':
                return True, 'explicit'
            elif risk == 'medium':
                return True, 'implicit'
            else:
                return False, None
        
        def get_confirmation(self, action, method='explicit'):
            '''Request user confirmation'''
            if method == 'explicit':
                return self.explicit_confirmation(action)
            elif method == 'implicit':
                return self.implicit_confirmation(action)
    ```
    
    Risk Assessment:
    ```python
    def assess_risk(action):
        risk_score = 0
        
        # Irreversibility
        if is_irreversible(action):
            risk_score += 40
        
        # Data loss potential
        if can_lose_data(action):
            risk_score += 30
        
        # Cost implications
        if has_cost(action):
            risk_score += 20
        
        # Privacy impact
        if affects_privacy(action):
            risk_score += 10
        
        if risk_score > 50:
            return 'high'
        elif risk_score > 20:
            return 'medium'
        else:
            return 'low'
    ```
    
    Confirmation Methods:
    
    Explicit Confirmation:
    ```python
    def explicit_confirmation(action):
        '''Clear yes/no question'''
        summary = summarize_action(action)
        consequences = list_consequences(action)
        
        message = f'''
        You are about to: {{summary}}
        
        This will:
        {{consequences}}
        
        Are you sure you want to proceed? (yes/no)
        '''
        
        return wait_for_response(message)
    ```
    
    Implicit Confirmation:
    ```python
    def implicit_confirmation(action):
        '''Confirmation through continuation'''
        preview = show_preview(action)
        
        message = f'''
        I'll {{action}}.
        
        Preview:
        {{preview}}
        
        Press Enter to continue or 'cancel' to abort.
        '''
        
        return wait_for_response(message)
    ```
    
    Two-Factor Confirmation:
    ```python
    def two_factor_confirmation(high_risk_action):
        '''Extra verification for high-risk actions'''
        # First confirmation
        confirmed_1 = explicit_confirmation(high_risk_action)
        
        if not confirmed_1:
            return False
        
        # Second verification (different method)
        code = generate_confirmation_code()
        message = f"Please type '{{code}}' to confirm"
        
        user_input = wait_for_response(message)
        confirmed_2 = (user_input == code)
        
        return confirmed_1 and confirmed_2
    ```
    
    Confirmation Patterns:
    
    Summary Confirmation:
    ```python
    "You want to delete 5 files. Confirm?"
    "This will email 100 contacts. Proceed?"
    ```
    
    Detailed Confirmation:
    ```python
    '''
    Action: Delete files
    Files: file1.txt, file2.py, file3.md
    Location: /home/user/docs
    
    WARNING: This cannot be undone.
    
    Type DELETE to confirm:
    '''
    ```
    
    Progressive Confirmation:
    ```python
    def progressive_confirmation(multi_step_action):
        '''Confirm each step'''
        for step in multi_step_action.steps:
            confirmed = confirm_step(step)
            
            if not confirmed:
                return cancel_remaining_steps()
            
            execute_step(step)
    ```
    
    Contextual Confirmation:
    
    User Experience:
    ```python
    def adaptive_confirmation(action, user_expertise):
        '''Adjust based on user'''
        if user_expertise == 'expert':
            # Minimal confirmation
            return quick_confirm(action)
        elif user_expertise == 'novice':
            # Detailed explanation
            return detailed_confirm(action)
    ```
    
    Action Frequency:
    ```python
    def frequency_based_confirmation(action):
        '''First time: detailed, repeated: quick'''
        if is_first_time(action):
            return detailed_confirmation(action)
        else:
            return quick_confirmation(action)
    ```
    
    Undo Mechanisms:
    
    Immediate Undo:
    ```python
    def execute_with_undo(action):
        '''Provide undo option'''
        # Execute
        result = execute(action)
        
        # Save undo state
        undo_state = save_undo_state(action)
        
        # Notify with undo option
        message = f'''
        Action completed: {{action}}
        
        Type 'undo' within 30 seconds to reverse.
        '''
        
        # Wait for potential undo
        response = wait_for_response(message, timeout=30)
        
        if response == 'undo':
            restore_state(undo_state)
            return 'undone'
        
        return result
    ```
    
    Confirmation Bypass:
    
    Batch Operations:
    ```python
    def batch_confirmation(actions):
        '''Confirm once for multiple items'''
        summary = f"Apply operation to {{len(actions)}} items?"
        
        confirmed = get_confirmation(summary)
        
        if confirmed:
            for action in actions:
                execute(action)  # No per-item confirmation
    ```
    
    Trusted Patterns:
    ```python
    def skip_for_safe_patterns(action):
        '''Don't confirm for known-safe actions'''
        if is_read_only(action):
            return execute_without_confirmation(action)
        
        if matches_safe_pattern(action):
            return execute_without_confirmation(action)
    ```
    
    Best Practices:
    ✓ Confirm high-risk actions
    ✓ Provide clear action summary
    ✓ Explain consequences
    ✓ Enable undo when possible
    ✓ Adapt to user expertise
    ✓ Don't over-confirm
    
    Key Insight:
    Confirmation prevents errors by verifying intent
    before irreversible or high-risk actions.
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Confirmation Agent:\n{report}\n\n{response.content}")],
        "user_confirmed": False
    }


def build_confirmation_graph():
    workflow = StateGraph(ConfirmationState)
    workflow.add_node("confirmation_agent", confirmation_agent)
    workflow.add_edge(START, "confirmation_agent")
    workflow.add_edge("confirmation_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_confirmation_graph()
    
    print("=== Confirmation MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "action_to_confirm": "Delete 100 files from production",
        "risk_level": "high",
        "user_confirmed": False,
        "confirmation_method": "explicit"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 154: Confirmation - COMPLETE")
    print(f"{'='*70}")

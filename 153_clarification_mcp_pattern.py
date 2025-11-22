"""
Clarification MCP Pattern

This pattern implements clarification mechanisms for handling
ambiguous or incomplete user inputs.

Key Features:
- Ambiguity detection
- Clarification questions
- Context disambiguation
- Progressive refinement
- Uncertainty handling
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ClarificationState(TypedDict):
    """State for clarification pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_input: str
    ambiguities: List[Dict]
    clarification_questions: List[str]
    resolved: bool


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def clarification_agent(state: ClarificationState) -> ClarificationState:
    """Manages clarification interactions"""
    user_input = state.get("user_input", "")
    
    system_prompt = """You are a clarification expert.

Clarification System:
• Detect ambiguity
• Ask targeted questions
• Disambiguate meaning
• Handle uncertainty
• Refine understanding

Ensure accurate understanding before proceeding."""
    
    user_prompt = f"""Input: {user_input}

Design clarification system.
Show ambiguity detection and resolution."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🤔 Clarification Agent:
    
    Analysis:
    • Input: {user_input[:100]}...
    • Goal: Resolve ambiguities
    
    Clarification System:
    ```python
    class ClarificationSystem:
        def __init__(self):
            self.ambiguity_detector = AmbiguityDetector()
            self.question_generator = ClarificationQuestionGenerator()
        
        def process_input(self, user_input):
            # Detect ambiguities
            ambiguities = self.ambiguity_detector.detect(user_input)
            
            if not ambiguities:
                return {{'status': 'clear', 'input': user_input}}
            
            # Generate clarification questions
            questions = self.question_generator.generate(ambiguities)
            
            return {{
                'status': 'needs_clarification',
                'ambiguities': ambiguities,
                'questions': questions
            }}
    ```
    
    Ambiguity Types:
    
    Lexical Ambiguity:
    ```python
    # Word has multiple meanings
    "Get me a Python book"
    # Python: programming language or snake?
    
    def detect_lexical_ambiguity(text):
        words = tokenize(text)
        ambiguous = []
        
        for word in words:
            senses = get_word_senses(word)
            if len(senses) > 1:
                ambiguous.append({{
                    'word': word,
                    'senses': senses,
                    'type': 'lexical'
                }})
        
        return ambiguous
    ```
    
    Referential Ambiguity:
    ```python
    # Unclear reference
    "The decorator and the function... it needs to return a value"
    # What does "it" refer to?
    
    def detect_referential_ambiguity(text):
        pronouns = extract_pronouns(text)
        ambiguous = []
        
        for pronoun in pronouns:
            antecedents = find_antecedents(pronoun, text)
            if len(antecedents) > 1:
                ambiguous.append({{
                    'pronoun': pronoun,
                    'possible_refs': antecedents,
                    'type': 'referential'
                }})
        
        return ambiguous
    ```
    
    Scope Ambiguity:
    ```python
    # Unclear scope
    "Show me files from last week"
    # Created last week or modified last week?
    
    def detect_scope_ambiguity(request):
        modifiers = extract_modifiers(request)
        
        for modifier in modifiers:
            if has_multiple_scopes(modifier):
                return {{
                    'modifier': modifier,
                    'possible_scopes': get_scopes(modifier),
                    'type': 'scope'
                }}
    ```
    
    Clarification Strategies:
    
    Targeted Questions:
    ```python
    def generate_clarification_question(ambiguity):
        if ambiguity['type'] == 'lexical':
            # Disambiguate word meaning
            return f"By '{{ambiguity['word']}}', do you mean {{option_a}} or {{option_b}}?"
        
        elif ambiguity['type'] == 'referential':
            # Clarify reference
            refs = ambiguity['possible_refs']
            return f"When you say 'it', are you referring to {{refs[0]}} or {{refs[1]}}?"
        
        elif ambiguity['type'] == 'scope':
            # Clarify scope
            return f"Do you mean {{scope_a}} or {{scope_b}}?"
        
        elif ambiguity['type'] == 'missing_info':
            # Request missing information
            return f"Could you specify {{ambiguity['missing_field']}}?"
    ```
    
    Progressive Clarification:
    ```python
    def progressive_clarification(user_input):
        '''Ask one question at a time'''
        ambiguities = detect_all_ambiguities(user_input)
        
        # Prioritize by importance
        sorted_ambiguities = sort_by_importance(ambiguities)
        
        # Ask about most critical first
        if sorted_ambiguities:
            return ask_about(sorted_ambiguities[0])
        
        return "Input is clear"
    ```
    
    Confirmation:
    ```python
    def confirm_understanding(interpretation):
        '''Check if interpretation is correct'''
        return f"Just to confirm, you want to {{interpretation}}. Is that right?"
    ```
    
    Missing Information:
    
    Detection:
    ```python
    def detect_missing_info(request, required_fields):
        '''Identify missing required information'''
        missing = []
        
        for field in required_fields:
            if not field in request:
                missing.append({{
                    'field': field,
                    'importance': get_importance(field),
                    'optional': is_optional(field)
                }})
        
        return missing
    ```
    
    Prioritization:
    ```python
    def prioritize_missing_info(missing_items):
        '''Ask for most important first'''
        # Required before optional
        required = [m for m in missing_items if not m['optional']]
        optional = [m for m in missing_items if m['optional']]
        
        # Sort by importance
        required.sort(key=lambda x: x['importance'], reverse=True)
        
        return required + optional
    ```
    
    Context-Based Resolution:
    
    Use Context:
    ```python
    def disambiguate_with_context(ambiguous_term, context):
        '''Use context to resolve ambiguity'''
        # "Show me the file"
        # Context: Last discussed file.txt
        
        if 'last_discussed' in context:
            return context['last_discussed']
        
        # Use domain context
        if in_programming_context(context):
            return resolve_as_programming_term(ambiguous_term)
        
        # Still ambiguous - need to ask
        return None
    ```
    
    Dialogue History:
    ```python
    def resolve_with_history(ambiguity, history):
        '''Use conversation history'''
        # Look for previous mentions
        for turn in reversed(history):
            if mentions(turn, ambiguity['candidates']):
                return extract_referent(turn)
        
        return None  # Can't resolve
    ```
    
    Default Assumptions:
    
    Most Likely Interpretation:
    ```python
    def use_default_interpretation(ambiguity):
        '''Make reasonable assumption'''
        # Use most common interpretation
        default = get_most_frequent_sense(ambiguity['term'])
        
        # Inform user of assumption
        notify = f"I'm assuming you mean {{default}}. Let me know if that's not right."
        
        return default, notify
    ```
    
    Graceful Degradation:
    ```python
    def handle_persistent_ambiguity():
        '''When can't resolve after multiple attempts'''
        return "I'm having trouble understanding. Could you rephrase?"
    ```
    
    Clarification Patterns:
    
    Yes/No Questions:
    ```python
    "Do you mean X or Y?"
    "Are you asking about A or B?"
    "Is this for [context]?"
    ```
    
    Open-Ended:
    ```python
    "Could you provide more details about...?"
    "What specifically would you like to know about...?"
    "Can you clarify what you mean by...?"
    ```
    
    Example-Based:
    ```python
    "For example, do you mean like X, or more like Y?"
    ```
    
    Best Practices:
    ✓ Detect ambiguity early
    ✓ Ask specific questions
    ✓ Use context when possible
    ✓ Prioritize critical ambiguities
    ✓ Confirm understanding
    ✓ Make reasonable defaults explicit
    
    Key Insight:
    Clarification prevents misunderstanding by identifying
    and resolving ambiguities through targeted questions.
    """
    
    return {
        "messages": [AIMessage(content=f"🤔 Clarification Agent:\n{report}\n\n{response.content}")],
        "resolved": False
    }


def build_clarification_graph():
    workflow = StateGraph(ClarificationState)
    workflow.add_node("clarification_agent", clarification_agent)
    workflow.add_edge(START, "clarification_agent")
    workflow.add_edge("clarification_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_clarification_graph()
    
    print("=== Clarification MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "user_input": "Show me the file",
        "ambiguities": [],
        "clarification_questions": [],
        "resolved": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 153: Clarification - COMPLETE")
    print(f"{'='*70}")

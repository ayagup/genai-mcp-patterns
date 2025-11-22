"""
Conversational MCP Pattern

This pattern implements conversational interaction with context management,
turn-taking, and natural dialogue flow.

Key Features:
- Turn-based dialogue
- Context tracking
- Topic management
- Natural flow
- Multi-turn coherence
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ConversationalState(TypedDict):
    """State for conversational pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    conversation_history: List[Dict]
    current_topic: str
    user_profile: Dict
    dialogue_act: str


llm = ChatOpenAI(model="gpt-4", temperature=0.7)


def conversational_agent(state: ConversationalState) -> ConversationalState:
    """Manages conversational interaction"""
    history = state.get("conversation_history", [])
    current_topic = state.get("current_topic", "")
    
    system_prompt = """You are a conversational AI expert.

Conversational Interaction:
• Natural dialogue flow
• Context awareness
• Turn-taking
• Topic tracking
• Coherence maintenance

Build engaging, contextual conversations."""
    
    user_prompt = f"""Topic: {current_topic}
History: {len(history)} turns

Design conversational system.
Show natural dialogue patterns."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    💬 Conversational Agent:
    
    Conversation State:
    • Current Topic: {current_topic}
    • Turn Count: {len(history)}
    • Dialogue Mode: Natural conversation
    
    Conversational System Implementation:
    ```python
    class ConversationalAgent:
        '''Natural multi-turn dialogue'''
        
        def __init__(self):
            self.conversation_history = []
            self.current_topic = None
            self.user_model = UserModel()
            self.dialogue_manager = DialogueManager()
        
        def process_turn(self, user_input):
            '''Process one conversation turn'''
            # Understand user input
            understanding = self.understand(user_input)
            
            # Update context
            self.update_context(understanding)
            
            # Generate response
            response = self.generate_response(understanding)
            
            # Track dialogue
            self.conversation_history.append({{
                'user': user_input,
                'understanding': understanding,
                'response': response,
                'timestamp': time.time()
            }})
            
            return response
        
        def understand(self, user_input):
            '''Understand user utterance'''
            return {{
                'text': user_input,
                'intent': self.classify_intent(user_input),
                'entities': self.extract_entities(user_input),
                'dialogue_act': self.classify_dialogue_act(user_input),
                'topic': self.detect_topic(user_input),
                'sentiment': self.analyze_sentiment(user_input)
            }}
        
        def update_context(self, understanding):
            '''Update conversation context'''
            # Topic tracking
            if understanding['topic'] != self.current_topic:
                if self.is_topic_shift(understanding):
                    self.current_topic = understanding['topic']
            
            # User model update
            self.user_model.update(understanding)
            
            # Dialogue state
            self.dialogue_manager.update_state(understanding)
        
        def generate_response(self, understanding):
            '''Generate contextual response'''
            # Select dialogue act
            response_act = self.dialogue_manager.select_response_act(
                understanding['dialogue_act']
            )
            
            # Generate content
            content = self.content_generation(
                understanding,
                response_act,
                self.conversation_history
            )
            
            # Apply persona
            styled_response = self.apply_style(content)
            
            return styled_response
    ```
    
    Dialogue Acts:
    
    Classification:
    ```python
    DIALOGUE_ACTS = {{
        # Assertives
        'inform': 'Provide information',
        'agree': 'Express agreement',
        'disagree': 'Express disagreement',
        
        # Directives
        'request': 'Ask for action',
        'question': 'Ask for information',
        'suggest': 'Propose something',
        
        # Commissives
        'promise': 'Commit to action',
        'offer': 'Offer to do something',
        
        # Expressives
        'thank': 'Express gratitude',
        'apologize': 'Express regret',
        'greet': 'Greeting',
        
        # Declarations
        'confirm': 'Confirm understanding',
        'clarify': 'Seek clarification'
    }}
    
    def classify_dialogue_act(utterance):
        '''Identify speech act'''
        # Question words → question
        if starts_with_question_word(utterance):
            return 'question'
        
        # Modal verbs → request
        if contains_modal(utterance, ['can', 'could', 'would']):
            return 'request'
        
        # Agreement markers
        if contains(['yes', 'agree', 'exactly'], utterance):
            return 'agree'
        
        return classify_with_model(utterance)
    ```
    
    Context Management:
    
    Conversation History:
    ```python
    class ConversationHistory:
        '''Manage dialogue history'''
        
        def __init__(self, max_turns=20):
            self.turns = []
            self.max_turns = max_turns
        
        def add_turn(self, user_msg, agent_msg):
            '''Add conversation turn'''
            self.turns.append({{
                'user': user_msg,
                'agent': agent_msg,
                'timestamp': time.time()
            }})
            
            # Keep only recent turns
            if len(self.turns) > self.max_turns:
                self.turns = self.turns[-self.max_turns:]
        
        def get_context_for_generation(self, n_turns=5):
            '''Get recent context'''
            recent = self.turns[-n_turns:]
            
            context = []
            for turn in recent:
                context.append(f"User: {{turn['user']}}")
                context.append(f"Agent: {{turn['agent']}}")
            
            return '\\n'.join(context)
    ```
    
    Topic Tracking:
    ```python
    class TopicTracker:
        '''Track conversation topics'''
        
        def __init__(self):
            self.topic_history = []
            self.current_topic = None
        
        def update(self, utterance):
            '''Update topic based on utterance'''
            new_topic = self.detect_topic(utterance)
            
            if new_topic != self.current_topic:
                # Check if topic shift
                if self.is_new_topic(new_topic):
                    self.handle_topic_shift(new_topic)
                elif self.is_topic_return(new_topic):
                    self.handle_topic_return(new_topic)
        
        def is_new_topic(self, topic):
            '''Check if genuinely new topic'''
            return topic not in self.topic_history
        
        def handle_topic_shift(self, new_topic):
            '''Smooth topic transition'''
            # Acknowledge shift
            acknowledgment = f"Switching to {{new_topic}}..."
            
            # Save old topic
            if self.current_topic:
                self.topic_history.append(self.current_topic)
            
            # Set new topic
            self.current_topic = new_topic
    ```
    
    Response Generation:
    
    Content Planning:
    ```python
    def plan_response(understanding, context):
        '''Plan what to say'''
        # Determine response type
        if understanding['dialogue_act'] == 'question':
            plan = {{'type': 'answer', 'content': answer_question()}}
        elif understanding['dialogue_act'] == 'request':
            plan = {{'type': 'fulfill', 'content': handle_request()}}
        else:
            plan = {{'type': 'acknowledge', 'content': acknowledge()}}
        
        # Add follow-up if appropriate
        if should_ask_followup(context):
            plan['followup'] = generate_followup_question()
        
        return plan
    ```
    
    Natural Language Generation:
    ```python
    def generate_natural_response(plan, style='friendly'):
        '''Convert plan to natural language'''
        templates = {{
            'friendly': {{
                'answer': "Great question! {{content}} Does that help?",
                'request': "I'd be happy to {{content}}",
                'acknowledge': "I see, {{content}}"
            }},
            'professional': {{
                'answer': "{{content}}",
                'request': "I can {{content}}",
                'acknowledge': "Understood. {{content}}"
            }}
        }}
        
        template = templates[style][plan['type']]
        response = template.format(content=plan['content'])
        
        return response
    ```
    
    Conversational Phenomena:
    
    Grounding:
    ```python
    def check_grounding(user_response):
        '''Ensure mutual understanding'''
        # User confirmed understanding
        if contains_acknowledgment(user_response):
            return 'grounded'
        
        # User confused
        if contains_confusion(user_response):
            return 'need_clarification'
        
        # Implicit grounding (continuation)
        if continues_topic(user_response):
            return 'implicitly_grounded'
    ```
    
    Turn-Taking:
    ```python
    def manage_turn_taking(state):
        '''Handle turn transitions'''
        # Yield turn signals
        yield_signals = [
            'What do you think?',
            'Does that make sense?',
            'Any questions?'
        ]
        
        # Hold turn signals
        hold_signals = [
            'And another thing...',
            'Also...',
            'Furthermore...'
        ]
        
        if state == 'want_to_yield':
            return add_yield_signal()
        elif state == 'want_to_continue':
            return add_hold_signal()
    ```
    
    Repair Strategies:
    
    Misunderstanding:
    ```python
    def detect_misunderstanding(user_response):
        '''Detect communication breakdown'''
        indicators = [
            "that's not what I meant",
            "no, I said",
            "I mean",
            "actually"
        ]
        
        if any(ind in user_response.lower() for ind in indicators):
            return True
        return False
    
    def repair_misunderstanding():
        '''Fix communication issue'''
        return "Let me clarify what I meant..."
    ```
    
    User Modeling:
    
    Preference Learning:
    ```python
    class UserModel:
        '''Model user preferences and traits'''
        
        def __init__(self):
            self.preferences = {{}}
            self.knowledge_level = 'intermediate'
            self.interaction_style = 'neutral'
        
        def update(self, interaction):
            '''Learn from interaction'''
            # Infer preferences
            if 'prefer' in interaction['text']:
                pref = extract_preference(interaction)
                self.preferences[pref['aspect']] = pref['value']
            
            # Assess knowledge
            if shows_expertise(interaction):
                self.knowledge_level = 'expert'
            elif shows_confusion(interaction):
                self.knowledge_level = 'beginner'
        
        def adapt_response(self, response):
            '''Customize for user'''
            if self.knowledge_level == 'expert':
                return add_technical_details(response)
            elif self.knowledge_level == 'beginner':
                return simplify(response)
            return response
    ```
    
    Engagement Strategies:
    
    Active Listening:
    ```python
    def show_active_listening(user_input):
        '''Demonstrate understanding'''
        # Paraphrase
        paraphrase = f"So you're saying {{summarize(user_input)}}..."
        
        # Reflect emotion
        emotion = detect_emotion(user_input)
        reflection = f"I sense you're {{emotion}}..."
        
        # Ask for confirmation
        confirmation = "Is that right?"
        
        return f"{{paraphrase}} {{reflection}} {{confirmation}}"
    ```
    
    Proactive Engagement:
    ```python
    def engage_proactively(context):
        '''Keep conversation going'''
        if conversation_lull(context):
            # Ask open-ended question
            return "What else would you like to know?"
        
        if can_offer_value(context):
            # Suggest related topic
            return "You might also be interested in..."
    ```
    
    Applications:
    
    Customer Support:
    ```python
    support_agent = ConversationalAgent(
        persona='helpful_support',
        goals=['resolve_issue', 'customer_satisfaction']
    )
    ```
    
    Teaching:
    ```python
    tutor_agent = ConversationalAgent(
        persona='patient_teacher',
        goals=['explain_concepts', 'check_understanding']
    )
    ```
    
    Best Practices:
    ✓ Track conversation history
    ✓ Maintain topic coherence
    ✓ Show active listening
    ✓ Handle topic shifts gracefully
    ✓ Adapt to user preferences
    ✓ Ensure grounding
    
    Key Insight:
    Natural conversation requires context awareness,
    turn management, and continuous adaptation to
    user needs and preferences.
    """
    
    return {
        "messages": [AIMessage(content=f"💬 Conversational Agent:\n{report}\n\n{response.content}")],
        "dialogue_act": "inform"
    }


def build_conversational_graph():
    workflow = StateGraph(ConversationalState)
    workflow.add_node("conversational_agent", conversational_agent)
    workflow.add_edge(START, "conversational_agent")
    workflow.add_edge("conversational_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_conversational_graph()
    
    print("=== Conversational MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "conversation_history": [
            {"user": "Hi!", "agent": "Hello! How can I help you today?"},
            {"user": "Tell me about Python", "agent": "Python is a versatile programming language..."}
        ],
        "current_topic": "Python programming",
        "user_profile": {"expertise": "intermediate"},
        "dialogue_act": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 151: Conversational - COMPLETE")
    print(f"{'='*70}")

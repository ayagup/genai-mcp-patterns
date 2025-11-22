"""
Persuasion MCP Pattern

This pattern implements persuasion strategies for ethical
influence through compelling arguments and evidence.

Key Features:
- Audience analysis
- Argument construction
- Evidence selection
- Emotional appeals
- Credibility building
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class PersuasionState(TypedDict):
    """State for persuasion pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    target_audience: str
    goal: str
    arguments: List[Dict]
    appeal_type: str
    credibility_score: float


llm = ChatOpenAI(model="gpt-4", temperature=0.6)


def persuasion_agent(state: PersuasionState) -> PersuasionState:
    """Manages persuasion strategies"""
    audience = state.get("target_audience", "")
    goal = state.get("goal", "")
    
    system_prompt = """You are a persuasion expert.

Persuasion Framework (Aristotle):
• Ethos: credibility
• Pathos: emotion
• Logos: logic

Ethical influence."""
    
    user_prompt = f"""Audience: {audience}
Goal: {goal}

Design persuasion strategy.
Show argument construction."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🎯 Persuasion Agent:
    
    Persuasion System:
    ```python
    class PersuasionSystem:
        def __init__(self):
            self.audience_analyzer = AudienceAnalyzer()
            self.argument_builder = ArgumentBuilder()
            self.appeal_selector = AppealSelector()
        
        def persuade(self, audience, goal):
            '''Main persuasion process'''
            # Analyze
            profile = self.analyze_audience(audience)
            
            # Construct
            arguments = self.build_arguments(goal, profile)
            
            # Deliver
            message = self.compose_message(arguments, profile)
            
            return message
    ```
    
    Audience Analysis:
    
    Psychographic Segmentation:
    ```python
    def analyze_audience(audience):
        '''Understand audience'''
        profile = {{
            'values': identify_values(audience),
            'beliefs': identify_beliefs(audience),
            'concerns': identify_concerns(audience),
            'knowledge_level': assess_knowledge(audience),
            'resistance_points': identify_objections(audience)
        }}
        
        return profile
    ```
    
    Attitude Assessment:
    ```python
    def assess_attitude(audience, topic):
        '''Where do they stand?'''
        attitudes = {{
            'favorable': 'reinforce, call to action',
            'neutral': 'inform, build case',
            'unfavorable': 'address concerns, reframe',
            'hostile': 'find common ground, incremental'
        }}
        
        return attitudes[determine_attitude(audience, topic)]
    ```
    
    Rhetorical Appeals:
    
    Ethos (Credibility):
    ```python
    def build_ethos():
        '''Establish credibility'''
        strategies = [
            # Expertise
            "Cite qualifications",
            "Demonstrate knowledge",
            "Reference experience",
            
            # Trustworthiness
            "Be transparent",
            "Acknowledge limitations",
            "Show good will",
            
            # Association
            "Reference authorities",
            "Use testimonials",
            "Show social proof"
        ]
        
        return strategies
    ```
    
    Example Ethos:
    ```python
    message = '''
    As someone who's spent 15 years studying climate science,
    I can tell you that the evidence is overwhelming.
    
    [Establishes expertise]
    '''
    ```
    
    Pathos (Emotion):
    ```python
    def build_pathos(audience):
        '''Appeal to emotions'''
        emotions = {{
            'fear': 'Show consequences of inaction',
            'hope': 'Paint vision of better future',
            'anger': 'Identify injustice',
            'pride': 'Appeal to identity',
            'compassion': 'Show impact on others'
        }}
        
        # Choose appropriate emotion
        emotion = select_emotion(audience)
        
        return craft_emotional_appeal(emotion)
    ```
    
    Example Pathos:
    ```python
    message = '''
    Imagine your children asking why you didn't act
    when you had the chance to make a difference.
    
    [Appeals to parental concern]
    '''
    ```
    
    Logos (Logic):
    ```python
    def build_logos(claim):
        '''Logical argument'''
        argument = {{
            'claim': state_claim(),
            'evidence': provide_evidence(),
            'reasoning': explain_connection(),
            'conclusion': draw_conclusion()
        }}
        
        return argument
    ```
    
    Example Logos:
    ```python
    argument = '''
    Studies show that companies with diverse teams
    are 35% more likely to outperform competitors.
    
    Therefore, investing in diversity improves
    business outcomes.
    
    [Data-driven reasoning]
    '''
    ```
    
    Argument Structure:
    
    Toulmin Model:
    ```python
    class ToulminArgument:
        def __init__(self):
            self.claim = None        # Conclusion
            self.data = None         # Evidence
            self.warrant = None      # Connection
            self.backing = None      # Support for warrant
            self.qualifier = None    # Confidence level
            self.rebuttal = None     # Counter-arguments
        
        def construct(self):
            return f'''
            Claim: {{self.claim}}
            
            Because: {{self.data}}
            
            Since: {{self.warrant}}
            
            {{self.qualifier}}
            
            Unless: {{self.rebuttal}}
            '''
    ```
    
    Example:
    ```python
    argument = ToulminArgument()
    argument.claim = "We should switch to renewable energy"
    argument.data = "Fossil fuels cause climate change"
    argument.warrant = "We should reduce harmful emissions"
    argument.backing = "Scientific consensus on climate change"
    argument.qualifier = "In most cases"
    argument.rebuttal = "Unless transition costs are prohibitive"
    ```
    
    Evidence Selection:
    
    Evidence Types:
    ```python
    def select_evidence(claim, audience):
        '''Choose compelling evidence'''
        evidence_types = {{
            'statistics': 'Quantitative data',
            'expert_testimony': 'Authority quotes',
            'examples': 'Specific cases',
            'analogies': 'Comparisons',
            'visual': 'Charts, images'
        }}
        
        # Match to audience preference
        if audience.prefers == 'data':
            return use_statistics()
        elif audience.prefers == 'stories':
            return use_examples()
    ```
    
    Evidence Quality:
    ```python
    def evaluate_evidence(source):
        '''Check credibility'''
        criteria = {{
            'recency': 'How recent?',
            'relevance': 'How related?',
            'authority': 'Who says so?',
            'accuracy': 'Is it true?',
            'purpose': 'Why published?'
        }}
        
        score = calculate_credibility(source, criteria)
        
        return score > threshold
    ```
    
    Persuasion Techniques:
    
    Reciprocity:
    ```python
    def reciprocity_principle():
        '''Give first, then ask'''
        # Provide value
        offer_free_resource()
        
        # Then request
        ask_for_action()
        
        # Example: Free trial → subscription
    ```
    
    Social Proof:
    ```python
    def social_proof():
        '''Show others doing it'''
        evidence = [
            "10,000 companies already use this",
            "Rated 4.8 stars by users",
            "Recommended by industry leaders"
        ]
        
        # People follow the crowd
    ```
    
    Scarcity:
    ```python
    def scarcity_principle():
        '''Limited availability'''
        messages = [
            "Only 5 spots remaining",
            "Offer expires in 24 hours",
            "Limited edition"
        ]
        
        # Creates urgency
    ```
    
    Commitment & Consistency:
    ```python
    def foot_in_door():
        '''Start small, build up'''
        # Small request first
        small_ask = "Would you sign this petition?"
        
        if agreed:
            # Larger request later
            large_ask = "Would you volunteer?"
        
        # People want to be consistent
    ```
    
    Framing:
    
    Positive vs Negative Framing:
    ```python
    def frame_message(goal):
        '''Choose frame'''
        positive = "This saves 90% of patients"
        negative = "This fails for 10% of patients"
        
        # Same fact, different impact
        
        # Use positive for risk-averse audiences
        # Use negative to motivate action
    ```
    
    Gain vs Loss Framing:
    ```python
    def frame_outcome():
        gain_frame = "You'll gain $100"
        loss_frame = "You'll avoid losing $100"
        
        # Loss aversion: people fear losses more than
        # they value gains
        
        # Use loss frame for motivation
    ```
    
    Counter-Persuasion:
    
    Inoculation:
    ```python
    def inoculate_against_counter_arguments():
        '''Prepare audience'''
        # Introduce weak counter-argument
        counter = "Some say this is too expensive"
        
        # Provide refutation
        refutation = "But the long-term savings outweigh costs"
        
        # Now audience is resistant to stronger version
    ```
    
    Two-Sided Arguments:
    ```python
    def two_sided_argument():
        '''Acknowledge other side'''
        # Your side
        our_position = "Product X has many benefits"
        
        # Acknowledge opposition
        opposition = "Critics say it's expensive"
        
        # Refute
        refutation = "But cost per use is actually lower"
        
        # More credible than one-sided
    ```
    
    Ethical Considerations:
    
    Ethical Persuasion:
    ```python
    def ethical_check(strategy):
        '''Ensure ethical persuasion'''
        questions = [
            "Is the information truthful?",
            "Are you respecting autonomy?",
            "Are you being transparent?",
            "Would you want to be persuaded this way?",
            "Are you manipulating or informing?"
        ]
        
        for question in questions:
            if not answer_yes(question):
                return revise_strategy()
    ```
    
    Message Composition:
    
    Attention-Interest-Desire-Action (AIDA):
    ```python
    def compose_aida_message(goal):
        '''Classic structure'''
        message = {{
            'attention': grab_attention(),
            'interest': build_interest(),
            'desire': create_desire(),
            'action': call_to_action()
        }}
        
        return format_message(message)
    ```
    
    Example:
    ```python
    message = '''
    [Attention]
    Your competitors are leaving you behind.
    
    [Interest]
    They're using AI to automate 80% of their workflow.
    
    [Desire]
    Imagine cutting your costs in half while doubling output.
    
    [Action]
    Schedule a demo today to see how.
    '''
    ```
    
    Best Practices:
    ✓ Know your audience
    ✓ Balance ethos, pathos, logos
    ✓ Use credible evidence
    ✓ Address counter-arguments
    ✓ Be transparent
    ✓ Respect autonomy
    ✓ Test and refine
    
    Key Insight:
    Ethical persuasion combines credibility, emotion,
    and logic to help audiences make informed decisions.
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Persuasion Agent:\n{report}\n\n{response.content}")]
    }


def build_persuasion_graph():
    workflow = StateGraph(PersuasionState)
    workflow.add_node("persuasion_agent", persuasion_agent)
    workflow.add_edge(START, "persuasion_agent")
    workflow.add_edge("persuasion_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_persuasion_graph()
    
    print("=== Persuasion MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "target_audience": "Small business owners",
        "goal": "Adopt cloud-based accounting software",
        "arguments": [],
        "appeal_type": "balanced",
        "credibility_score": 0.85
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 156: Persuasion - COMPLETE")
    print(f"{'='*70}")

"""
Negotiation MCP Pattern

This pattern implements negotiation strategies for reaching
agreements through multi-turn dialogue.

Key Features:
- Interest identification
- Proposal generation
- Concession strategy
- BATNA analysis
- Agreement formulation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class NegotiationState(TypedDict):
    """State for negotiation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    parties: List[str]
    interests: Dict[str, List[str]]
    proposals: List[Dict]
    agreements: List[str]
    batna: str


llm = ChatOpenAI(model="gpt-4", temperature=0.5)


def negotiation_agent(state: NegotiationState) -> NegotiationState:
    """Manages negotiation process"""
    parties = state.get("parties", [])
    
    system_prompt = """You are a negotiation expert.

Negotiation Framework:
• Identify interests
• Generate proposals
• Find common ground
• Make concessions
• Reach agreement

Win-win solutions."""
    
    user_prompt = f"""Parties: {', '.join(parties)}

Design negotiation system.
Show negotiation strategies."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🤝 Negotiation Agent:
    
    Negotiation System:
    ```python
    class NegotiationSystem:
        def __init__(self):
            self.interest_identifier = InterestIdentifier()
            self.proposal_generator = ProposalGenerator()
            self.concession_manager = ConcessionManager()
        
        def negotiate(self, parties):
            '''Main negotiation loop'''
            # Preparation
            interests = self.identify_interests(parties)
            batna = self.determine_batna(parties)
            
            # Negotiation
            while not agreement_reached():
                proposal = self.generate_proposal(interests)
                response = self.get_response(proposal)
                
                if response == 'accept':
                    return finalize_agreement(proposal)
                elif response == 'counter':
                    self.adjust_strategy()
                elif response == 'reject':
                    if better_than_batna(proposal):
                        continue
                    else:
                        return walk_away()
    ```
    
    Interest Identification:
    
    Positions vs Interests:
    ```python
    def identify_interests(position):
        '''Find underlying needs'''
        # Position: "I want the office"
        # Interest: "I need quiet space to work"
        
        interests = []
        
        # Ask why
        why = ask_why(position)
        interests.append(why)
        
        # Ask what for
        purpose = ask_purpose(position)
        interests.append(purpose)
        
        # Identify constraints
        constraints = identify_constraints(position)
        interests.extend(constraints)
        
        return interests
    ```
    
    Interest Types:
    ```python
    class InterestTypes:
        SUBSTANTIVE = 'tangible outcomes'    # Money, resources
        PROCEDURAL = 'how decided'           # Fair process
        PSYCHOLOGICAL = 'feelings'           # Respect, recognition
        RELATIONSHIP = 'connection'          # Ongoing relationship
    ```
    
    Proposal Generation:
    
    ZOPA Analysis:
    ```python
    def find_zopa(party_a, party_b):
        '''Zone of Possible Agreement'''
        a_min = party_a.reservation_point
        a_max = party_a.target_point
        
        b_min = party_b.reservation_point
        b_max = party_b.target_point
        
        # ZOPA exists if ranges overlap
        if a_min <= b_max and b_min <= a_max:
            zopa_start = max(a_min, b_min)
            zopa_end = min(a_max, b_max)
            return (zopa_start, zopa_end)
        
        return None  # No ZOPA
    ```
    
    Creative Options:
    ```python
    def generate_options(interests):
        '''Expand the pie'''
        options = []
        
        # Different currencies
        # Party A values: time
        # Party B values: money
        # Trade: A gets faster delivery, B pays less
        
        for interest_a in interests['A']:
            for interest_b in interests['B']:
                if compatible(interest_a, interest_b):
                    option = create_package(interest_a, interest_b)
                    options.append(option)
        
        return options
    ```
    
    Negotiation Strategies:
    
    Distributive (Win-Lose):
    ```python
    def distributive_strategy(issue):
        '''Fixed pie - claim value'''
        # Tactics:
        # - Start high/low
        # - Make small concessions
        # - Show reluctance
        # - Create deadline pressure
        
        opening_offer = calculate_extreme_offer()
        
        while negotiating:
            if counteroffer_received:
                concession = min_acceptable_concession()
                new_offer = adjust_offer(concession)
    ```
    
    Integrative (Win-Win):
    ```python
    def integrative_strategy(interests):
        '''Expand pie - create value'''
        # Tactics:
        # - Share information
        # - Identify interests
        # - Generate options
        # - Find trade-offs
        
        # Example: Job negotiation
        employer_interests = ['cost control', 'retain talent']
        candidate_interests = ['growth', 'compensation']
        
        # Solution: Lower base salary + stock options + training
        package = create_package([
            ('salary', medium),
            ('equity', high),
            ('training_budget', high)
        ])
    ```
    
    Concession Strategy:
    
    Concession Patterns:
    ```python
    def make_concessions():
        '''Strategic concession making'''
        concessions = [
            ('first', 'large'),   # Show willingness
            ('second', 'medium'), # Continue momentum
            ('third', 'small'),   # Signal limit
            ('fourth', 'tiny')    # Final offer
        ]
        
        for i, (position, size) in enumerate(concessions):
            if offer_accepted():
                return agreement
            
            # Diminishing concessions signal bottom
            next_concession = calculate_concession(size)
            make_offer(next_concession)
    ```
    
    Reciprocity:
    ```python
    def conditional_concession(issue):
        '''If-then concessions'''
        return f"If you agree to {{their_issue}}, then I'll agree to {{my_issue}}"
        
        # Example:
        # "If you can deliver by Friday, I'll pay the rush fee"
    ```
    
    BATNA Management:
    
    Determine BATNA:
    ```python
    def determine_batna():
        '''Best Alternative To Negotiated Agreement'''
        alternatives = generate_alternatives()
        
        best_alternative = max(alternatives, key=lambda x: x.value)
        
        # BATNA is your walk-away point
        return best_alternative
    ```
    
    Improve BATNA:
    ```python
    def improve_batna():
        '''Strengthen position before negotiating'''
        # Get competing offers
        # Build relationships
        # Develop alternatives
        
        # Example: Job seeker
        # - Apply to multiple companies
        # - Maintain current job
        # - Build skills
    ```
    
    Anchoring:
    
    First Offer Advantage:
    ```python
    def anchor_negotiation(target):
        '''Set reference point'''
        # Research suggests first offer influences outcome
        
        anchor = calculate_ambitious_anchor(target)
        
        # Make first offer
        opening = make_offer(anchor)
        
        # Subsequent offers adjust from this anchor
    ```
    
    Counter-Anchoring:
    ```python
    def respond_to_anchor(extreme_offer):
        '''Reject anchor, reframe'''
        # Don't accept their anchor
        
        # Provide own anchor
        counter_anchor = calculate_own_anchor()
        
        # Or reframe issue
        alternative_frame = reframe_negotiation()
    ```
    
    Negotiation Tactics:
    
    Information Exchange:
    ```python
    def share_information():
        '''Selective disclosure'''
        # Share interests, not positions
        # Signal priorities
        # Ask questions
        
        message = "We really need X (high priority), "
                  "but we're flexible on Y (low priority)"
    ```
    
    Packaging Issues:
    ```python
    def package_issues(issues):
        '''Bundle multiple issues'''
        # Don't negotiate one at a time
        
        package = {{
            'price': propose_price(),
            'delivery': propose_delivery(),
            'warranty': propose_warranty()
        }}
        
        # Allow trade-offs across issues
    ```
    
    Agreement Formulation:
    
    Test Agreement:
    ```python
    def test_agreement():
        '''Check mutual understanding'''
        summary = summarize_agreement()
        
        # Verify with other party
        confirmed = get_confirmation(summary)
        
        if not confirmed:
            clarify_differences()
    ```
    
    Formalize Agreement:
    ```python
    def formalize_agreement(terms):
        '''Document agreement'''
        agreement = {{
            'parties': list_parties(),
            'terms': list_terms(),
            'timeline': specify_timeline(),
            'consequences': specify_consequences()
        }}
        
        return create_contract(agreement)
    ```
    
    Best Practices:
    ✓ Prepare thoroughly (interests, BATNA)
    ✓ Listen actively
    ✓ Focus on interests, not positions
    ✓ Generate multiple options
    ✓ Use objective criteria
    ✓ Maintain relationships
    ✓ Know when to walk away
    
    Key Insight:
    Effective negotiation creates value through
    understanding interests and finding win-win solutions.
    """
    
    return {
        "messages": [AIMessage(content=f"🤝 Negotiation Agent:\n{report}\n\n{response.content}")]
    }


def build_negotiation_graph():
    workflow = StateGraph(NegotiationState)
    workflow.add_node("negotiation_agent", negotiation_agent)
    workflow.add_edge(START, "negotiation_agent")
    workflow.add_edge("negotiation_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_negotiation_graph()
    
    print("=== Negotiation MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "parties": ["Buyer", "Seller"],
        "interests": {
            "Buyer": ["low price", "quality", "fast delivery"],
            "Seller": ["profit margin", "repeat business", "reputation"]
        },
        "proposals": [],
        "agreements": [],
        "batna": "Alternative supplier with 10% higher price"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 155: Negotiation - COMPLETE")
    print(f"{'='*70}")

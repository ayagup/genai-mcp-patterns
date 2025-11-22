"""
Voting MCP Pattern

This pattern demonstrates agents casting votes on proposals and determining 
outcomes based on majority or weighted voting mechanisms.

Key Features:
- Multiple agents with voting rights
- Different voting strategies (simple majority, weighted, unanimous)
- Vote collection and tallying
- Decision determination based on voting results
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class VotingState(TypedDict):
    """State for voting-based decision making"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    proposal: str
    votes: dict[str, dict[str, any]]  # agent_name -> {vote: yes/no, weight: int, reasoning: str}
    voting_threshold: float  # e.g., 0.5 for simple majority, 0.66 for supermajority
    decision: str
    vote_tally: dict[str, int]  # yes/no counts


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Agent 1: Senior Engineer Voter (high weight)
def senior_engineer_voter(state: VotingState) -> VotingState:
    """Senior engineer with weighted vote (weight: 3)"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a senior software engineer with 15 years 
    of experience. Evaluate the technical proposal carefully considering feasibility, maintainability, 
    scalability, and technical debt. Vote YES or NO and provide detailed reasoning.""")
    
    user_message = HumanMessage(content=f"""Vote on this proposal:\n{proposal}
    
    Respond with:
    VOTE: [YES/NO]
    REASONING: [Your detailed reasoning]""")
    
    response = llm.invoke([system_message, user_message])
    
    # Parse vote
    vote_yes = "YES" in response.content.upper() and "VOTE: YES" in response.content.upper()
    
    votes = state.get("votes", {})
    votes["senior_engineer"] = {
        "vote": "yes" if vote_yes else "no",
        "weight": 3,
        "reasoning": response.content
    }
    
    return {
        "messages": [AIMessage(content=f"Senior Engineer (weight=3): {response.content}")],
        "votes": votes
    }


# Agent 2: Product Manager Voter (medium weight)
def product_manager_voter(state: VotingState) -> VotingState:
    """Product manager with weighted vote (weight: 2)"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a product manager focused on user value, 
    business impact, and market fit. Evaluate the proposal from a product perspective. 
    Vote YES or NO and provide reasoning.""")
    
    user_message = HumanMessage(content=f"""Vote on this proposal:\n{proposal}
    
    Respond with:
    VOTE: [YES/NO]
    REASONING: [Your detailed reasoning]""")
    
    response = llm.invoke([system_message, user_message])
    
    vote_yes = "YES" in response.content.upper() and "VOTE: YES" in response.content.upper()
    
    votes = state.get("votes", {})
    votes["product_manager"] = {
        "vote": "yes" if vote_yes else "no",
        "weight": 2,
        "reasoning": response.content
    }
    
    return {
        "messages": [AIMessage(content=f"Product Manager (weight=2): {response.content}")],
        "votes": votes
    }


# Agent 3: UX Designer Voter (medium weight)
def ux_designer_voter(state: VotingState) -> VotingState:
    """UX designer with weighted vote (weight: 2)"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a UX designer focused on user experience, 
    accessibility, and design consistency. Evaluate the proposal from a user-centric perspective. 
    Vote YES or NO and provide reasoning.""")
    
    user_message = HumanMessage(content=f"""Vote on this proposal:\n{proposal}
    
    Respond with:
    VOTE: [YES/NO]
    REASONING: [Your detailed reasoning]""")
    
    response = llm.invoke([system_message, user_message])
    
    vote_yes = "YES" in response.content.upper() and "VOTE: YES" in response.content.upper()
    
    votes = state.get("votes", {})
    votes["ux_designer"] = {
        "vote": "yes" if vote_yes else "no",
        "weight": 2,
        "reasoning": response.content
    }
    
    return {
        "messages": [AIMessage(content=f"UX Designer (weight=2): {response.content}")],
        "votes": votes
    }


# Agent 4: Junior Developer Voter (low weight)
def junior_developer_voter(state: VotingState) -> VotingState:
    """Junior developer with weighted vote (weight: 1)"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a junior developer with fresh perspectives 
    and eagerness to learn. Evaluate the proposal considering learning opportunities, code clarity, 
    and implementation effort. Vote YES or NO and provide reasoning.""")
    
    user_message = HumanMessage(content=f"""Vote on this proposal:\n{proposal}
    
    Respond with:
    VOTE: [YES/NO]
    REASONING: [Your detailed reasoning]""")
    
    response = llm.invoke([system_message, user_message])
    
    vote_yes = "YES" in response.content.upper() and "VOTE: YES" in response.content.upper()
    
    votes = state.get("votes", {})
    votes["junior_developer"] = {
        "vote": "yes" if vote_yes else "no",
        "weight": 1,
        "reasoning": response.content
    }
    
    return {
        "messages": [AIMessage(content=f"Junior Developer (weight=1): {response.content}")],
        "votes": votes
    }


# Agent 5: Security Specialist Voter (high weight)
def security_specialist_voter(state: VotingState) -> VotingState:
    """Security specialist with weighted vote (weight: 3)"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a security specialist focused on data protection, 
    vulnerability assessment, and compliance. Evaluate the proposal for security risks. 
    Vote YES or NO and provide reasoning.""")
    
    user_message = HumanMessage(content=f"""Vote on this proposal:\n{proposal}
    
    Respond with:
    VOTE: [YES/NO]
    REASONING: [Your detailed reasoning]""")
    
    response = llm.invoke([system_message, user_message])
    
    vote_yes = "YES" in response.content.upper() and "VOTE: YES" in response.content.upper()
    
    votes = state.get("votes", {})
    votes["security_specialist"] = {
        "vote": "yes" if vote_yes else "no",
        "weight": 3,
        "reasoning": response.content
    }
    
    return {
        "messages": [AIMessage(content=f"Security Specialist (weight=3): {response.content}")],
        "votes": votes
    }


# Vote Tally Agent
def vote_tally_agent(state: VotingState) -> VotingState:
    """Tallies votes and determines the decision"""
    votes = state.get("votes", {})
    threshold = state.get("voting_threshold", 0.5)
    
    # Calculate weighted votes
    yes_weight = sum(v["weight"] for v in votes.values() if v["vote"] == "yes")
    no_weight = sum(v["weight"] for v in votes.values() if v["vote"] == "no")
    total_weight = yes_weight + no_weight
    
    # Calculate percentages
    yes_percentage = yes_weight / total_weight if total_weight > 0 else 0
    
    # Determine decision
    decision = "APPROVED" if yes_percentage >= threshold else "REJECTED"
    
    summary = f"""
    === VOTING RESULTS ===
    Total Votes Cast: {len(votes)}
    YES Votes (weighted): {yes_weight}
    NO Votes (weighted): {no_weight}
    Total Weight: {total_weight}
    YES Percentage: {yes_percentage:.1%}
    Threshold: {threshold:.1%}
    
    DECISION: {decision}
    """
    
    vote_tally = {
        "yes": yes_weight,
        "no": no_weight
    }
    
    return {
        "messages": [AIMessage(content=summary)],
        "decision": decision,
        "vote_tally": vote_tally
    }


# Build the graph
def build_voting_graph():
    """Build the voting MCP pattern graph"""
    workflow = StateGraph(VotingState)
    
    # Add nodes
    workflow.add_node("senior_engineer", senior_engineer_voter)
    workflow.add_node("product_manager", product_manager_voter)
    workflow.add_node("ux_designer", ux_designer_voter)
    workflow.add_node("junior_developer", junior_developer_voter)
    workflow.add_node("security_specialist", security_specialist_voter)
    workflow.add_node("tally", vote_tally_agent)
    
    # Define edges - voters run in parallel, then tally
    workflow.add_edge(START, "senior_engineer")
    workflow.add_edge(START, "product_manager")
    workflow.add_edge(START, "ux_designer")
    workflow.add_edge(START, "junior_developer")
    workflow.add_edge(START, "security_specialist")
    
    # All voters feed into tally
    workflow.add_edge("senior_engineer", "tally")
    workflow.add_edge("product_manager", "tally")
    workflow.add_edge("ux_designer", "tally")
    workflow.add_edge("junior_developer", "tally")
    workflow.add_edge("security_specialist", "tally")
    
    workflow.add_edge("tally", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_voting_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "proposal": """Proposal: Implement Real-Time Collaboration Features
        
        We propose adding real-time collaborative editing capabilities to our document editor, 
        similar to Google Docs. This would require:
        - WebSocket infrastructure for real-time synchronization
        - Operational Transform (OT) or CRDT for conflict resolution
        - Presence awareness (showing active users)
        - Estimated timeline: 6 months
        - Estimated cost: $500K
        
        Benefits: Competitive advantage, increased user engagement
        Risks: Technical complexity, performance concerns, security challenges""",
        "votes": {},
        "voting_threshold": 0.6,  # 60% supermajority required
        "decision": "",
        "vote_tally": {}
    }
    
    # Run the voting process
    print("Starting Voting MCP Pattern...")
    print(f"Proposal: {initial_state['proposal'][:100]}...")
    print(f"Voting Threshold: {initial_state['voting_threshold']:.0%}\n")
    
    result = graph.invoke(initial_state)
    
    # Display results
    print("\n=== Individual Votes ===")
    for msg in result["messages"][:-1]:  # Exclude final tally
        print(f"\n{msg.content}")
    
    print("\n" + result["messages"][-1].content)  # Final tally
    
    print("\n=== Vote Details ===")
    for agent, vote_data in result["votes"].items():
        print(f"\n{agent.replace('_', ' ').title()}:")
        print(f"  Vote: {vote_data['vote'].upper()}")
        print(f"  Weight: {vote_data['weight']}")

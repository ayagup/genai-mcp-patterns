"""
Consensus-Based MCP Pattern

This pattern demonstrates multiple agents reaching consensus through iterative
voting and agreement. Agents propose solutions, evaluate proposals, and iterate
until consensus is achieved.

Key Features:
- Multiple agents participate in consensus building
- Iterative proposal and evaluation rounds
- Consensus threshold detection
- Conflict resolution through discussion
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ConsensusState(TypedDict):
    """State for consensus-based coordination"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    topic: str
    proposals: dict[str, str]
    votes: dict[str, dict[str, int]]  # agent_name -> {proposal_id -> score}
    round_number: int
    consensus_reached: bool
    final_decision: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Agent 1: Proposer Agent
def proposer_agent(state: ConsensusState) -> ConsensusState:
    """Agent that proposes initial solutions"""
    topic = state["topic"]
    round_num = state.get("round_number", 0)
    
    system_message = SystemMessage(content="""You are a proposer agent responsible for suggesting 
    solutions to problems. Analyze the topic and propose a concrete solution with clear reasoning.""")
    
    user_message = HumanMessage(content=f"""Round {round_num}: Propose a solution for: {topic}
    
    Previous proposals: {state.get('proposals', {})}
    
    Provide a clear, actionable proposal.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Store proposal
    proposal_id = f"proposer_round_{round_num}"
    proposals = state.get("proposals", {})
    proposals[proposal_id] = response.content
    
    return {
        "messages": [AIMessage(content=f"Proposer: {response.content}")],
        "proposals": proposals
    }


# Agent 2: Evaluator Agent 1
def evaluator_1_agent(state: ConsensusState) -> ConsensusState:
    """Agent that evaluates proposals from technical perspective"""
    proposals = state.get("proposals", {})
    
    system_message = SystemMessage(content="""You are a technical evaluator. Assess proposals 
    based on feasibility, technical soundness, and implementation complexity. 
    Rate each proposal from 1-10 and provide reasoning.""")
    
    proposals_text = "\n".join([f"{k}: {v}" for k, v in proposals.items()])
    user_message = HumanMessage(content=f"""Evaluate these proposals:\n{proposals_text}
    
    Provide scores (1-10) for each proposal with reasoning.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"Technical Evaluator: {response.content}")]
    }


# Agent 3: Evaluator Agent 2
def evaluator_2_agent(state: ConsensusState) -> ConsensusState:
    """Agent that evaluates proposals from business perspective"""
    proposals = state.get("proposals", {})
    
    system_message = SystemMessage(content="""You are a business evaluator. Assess proposals 
    based on cost-effectiveness, user impact, and business value. 
    Rate each proposal from 1-10 and provide reasoning.""")
    
    proposals_text = "\n".join([f"{k}: {v}" for k, v in proposals.items()])
    user_message = HumanMessage(content=f"""Evaluate these proposals:\n{proposals_text}
    
    Provide scores (1-10) for each proposal with reasoning.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"Business Evaluator: {response.content}")]
    }


# Agent 4: Evaluator Agent 3
def evaluator_3_agent(state: ConsensusState) -> ConsensusState:
    """Agent that evaluates proposals from user experience perspective"""
    proposals = state.get("proposals", {})
    
    system_message = SystemMessage(content="""You are a UX evaluator. Assess proposals 
    based on usability, accessibility, and user satisfaction. 
    Rate each proposal from 1-10 and provide reasoning.""")
    
    proposals_text = "\n".join([f"{k}: {v}" for k, v in proposals.items()])
    user_message = HumanMessage(content=f"""Evaluate these proposals:\n{proposals_text}
    
    Provide scores (1-10) for each proposal with reasoning.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"UX Evaluator: {response.content}")]
    }


# Agent 5: Consensus Checker
def consensus_checker(state: ConsensusState) -> ConsensusState:
    """Agent that determines if consensus is reached"""
    messages = state["messages"]
    proposals = state.get("proposals", {})
    
    system_message = SystemMessage(content="""You are a consensus checker. Analyze all evaluations 
    and determine if there is consensus (generally 70%+ agreement). If consensus is reached, 
    identify the winning proposal. If not, suggest what needs to be addressed in the next round.""")
    
    recent_messages = "\n".join([m.content for m in messages[-5:]])
    user_message = HumanMessage(content=f"""Analyze these evaluations:\n{recent_messages}
    
    Proposals: {proposals}
    
    Determine: 1) Is consensus reached? 2) What is the decision or what needs refinement?""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple heuristic: check if "consensus" or "agreement" in response
    consensus_reached = "consensus reached" in response.content.lower() or "agreement" in response.content.lower()
    
    return {
        "messages": [AIMessage(content=f"Consensus Checker: {response.content}")],
        "consensus_reached": consensus_reached,
        "final_decision": response.content if consensus_reached else ""
    }


# Define routing logic
def should_continue(state: ConsensusState) -> str:
    """Determine if we should continue iterating or end"""
    if state.get("consensus_reached", False):
        return "end"
    
    round_num = state.get("round_number", 0)
    if round_num >= 3:  # Max 3 rounds
        return "end"
    
    return "continue"


def increment_round(state: ConsensusState) -> ConsensusState:
    """Increment the round number for next iteration"""
    return {
        "round_number": state.get("round_number", 0) + 1
    }


# Build the graph
def build_consensus_graph():
    """Build the consensus-based MCP pattern graph"""
    workflow = StateGraph(ConsensusState)
    
    # Add nodes
    workflow.add_node("proposer", proposer_agent)
    workflow.add_node("evaluator_1", evaluator_1_agent)
    workflow.add_node("evaluator_2", evaluator_2_agent)
    workflow.add_node("evaluator_3", evaluator_3_agent)
    workflow.add_node("consensus_checker", consensus_checker)
    workflow.add_node("increment_round", increment_round)
    
    # Define edges
    workflow.add_edge(START, "proposer")
    workflow.add_edge("proposer", "evaluator_1")
    workflow.add_edge("evaluator_1", "evaluator_2")
    workflow.add_edge("evaluator_2", "evaluator_3")
    workflow.add_edge("evaluator_3", "consensus_checker")
    
    # Conditional edge for iteration
    workflow.add_conditional_edges(
        "consensus_checker",
        should_continue,
        {
            "continue": "increment_round",
            "end": END
        }
    )
    
    workflow.add_edge("increment_round", "proposer")
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_consensus_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "topic": "Should we migrate our monolithic application to microservices architecture?",
        "proposals": {},
        "votes": {},
        "round_number": 0,
        "consensus_reached": False,
        "final_decision": ""
    }
    
    # Run the consensus process
    print("Starting Consensus-Based MCP Pattern...")
    print(f"Topic: {initial_state['topic']}\n")
    
    result = graph.invoke(initial_state)
    
    # Display results
    print("\n=== Consensus Process ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
    
    print("\n=== Final Outcome ===")
    print(f"Consensus Reached: {result['consensus_reached']}")
    print(f"Rounds Completed: {result['round_number']}")
    if result['consensus_reached']:
        print(f"Final Decision: {result['final_decision']}")

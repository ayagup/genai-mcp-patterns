"""
Two-Phase Commit MCP Pattern

This pattern demonstrates distributed transaction coordination using prepare 
and commit phases to ensure atomicity across multiple agents.

Key Features:
- Prepare phase: All participants vote on readiness
- Commit phase: Execute if all participants are ready
- Rollback capability if any participant cannot commit
- Coordinator manages the 2PC protocol
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TwoPhaseCommitState(TypedDict):
    """State for two-phase commit pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    transaction: str
    phase: str  # "prepare", "commit", "rollback"
    participant_votes: dict[str, str]  # participant_id -> "ready"/"abort"
    participant_status: dict[str, str]  # participant_id -> "committed"/"aborted"
    coordinator_decision: str  # "commit" or "abort"
    transaction_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Coordinator Agent
def coordinator_prepare(state: TwoPhaseCommitState) -> TwoPhaseCommitState:
    """Coordinator initiates prepare phase"""
    transaction = state["transaction"]
    
    system_message = SystemMessage(content="""You are a transaction coordinator implementing 
    the two-phase commit protocol. You are initiating the PREPARE phase. Send prepare requests 
    to all participants and explain the transaction.""")
    
    user_message = HumanMessage(content=f"""Transaction to coordinate:\n{transaction}
    
    Initiate PREPARE phase: Ask all participants if they can commit this transaction.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🎯 Coordinator (PREPARE): {response.content}")],
        "phase": "prepare"
    }


# Participant 1: Database Service
def database_participant(state: TwoPhaseCommitState) -> TwoPhaseCommitState:
    """Database service participant"""
    transaction = state["transaction"]
    phase = state.get("phase", "")
    
    if phase == "prepare":
        system_message = SystemMessage(content="""You are the database service participant. 
        Evaluate if you can successfully prepare for this transaction (lock resources, validate 
        data, check constraints). Respond READY if you can commit, or ABORT if you cannot.""")
        
        user_message = HumanMessage(content=f"""PREPARE request for transaction:\n{transaction}
        
        Can you commit this transaction? Respond: READY or ABORT with reasoning.""")
        
        response = llm.invoke([system_message, user_message])
        
        vote = "ready" if "READY" in response.content.upper() else "abort"
        
        participant_votes = state.get("participant_votes", {})
        participant_votes["database"] = vote
        
        return {
            "messages": [AIMessage(content=f"💾 Database Service (PREPARE): {response.content}")],
            "participant_votes": participant_votes
        }
    
    elif phase == "commit":
        system_message = SystemMessage(content="""You are the database service. Execute the 
        COMMIT operation. Make the transaction changes permanent.""")
        
        user_message = HumanMessage(content=f"""COMMIT transaction:\n{transaction}
        
        Execute the commit and report success.""")
        
        response = llm.invoke([system_message, user_message])
        
        participant_status = state.get("participant_status", {})
        participant_status["database"] = "committed"
        
        return {
            "messages": [AIMessage(content=f"💾 Database Service (COMMIT): {response.content}")],
            "participant_status": participant_status
        }
    
    elif phase == "rollback":
        system_message = SystemMessage(content="""You are the database service. Execute 
        ROLLBACK operation. Undo any prepared changes and release locks.""")
        
        user_message = HumanMessage(content=f"""ROLLBACK transaction:\n{transaction}
        
        Undo prepared changes and report completion.""")
        
        response = llm.invoke([system_message, user_message])
        
        participant_status = state.get("participant_status", {})
        participant_status["database"] = "aborted"
        
        return {
            "messages": [AIMessage(content=f"💾 Database Service (ROLLBACK): {response.content}")],
            "participant_status": participant_status
        }
    
    return {"messages": []}


# Participant 2: Cache Service
def cache_participant(state: TwoPhaseCommitState) -> TwoPhaseCommitState:
    """Cache service participant"""
    transaction = state["transaction"]
    phase = state.get("phase", "")
    
    if phase == "prepare":
        system_message = SystemMessage(content="""You are the cache service participant. 
        Evaluate if you can prepare cache updates for this transaction. Respond READY or ABORT.""")
        
        user_message = HumanMessage(content=f"""PREPARE request for transaction:\n{transaction}
        
        Can you prepare cache updates? Respond: READY or ABORT with reasoning.""")
        
        response = llm.invoke([system_message, user_message])
        
        vote = "ready" if "READY" in response.content.upper() else "abort"
        
        participant_votes = state.get("participant_votes", {})
        participant_votes["cache"] = vote
        
        return {
            "messages": [AIMessage(content=f"⚡ Cache Service (PREPARE): {response.content}")],
            "participant_votes": participant_votes
        }
    
    elif phase == "commit":
        system_message = SystemMessage(content="""You are the cache service. Execute COMMIT 
        operation. Apply cache updates permanently.""")
        
        user_message = HumanMessage(content=f"""COMMIT transaction:\n{transaction}
        
        Apply cache updates and report success.""")
        
        response = llm.invoke([system_message, user_message])
        
        participant_status = state.get("participant_status", {})
        participant_status["cache"] = "committed"
        
        return {
            "messages": [AIMessage(content=f"⚡ Cache Service (COMMIT): {response.content}")],
            "participant_status": participant_status
        }
    
    elif phase == "rollback":
        system_message = SystemMessage(content="""You are the cache service. Execute ROLLBACK. 
        Discard prepared cache updates.""")
        
        user_message = HumanMessage(content=f"""ROLLBACK transaction:\n{transaction}
        
        Discard prepared updates and report completion.""")
        
        response = llm.invoke([system_message, user_message])
        
        participant_status = state.get("participant_status", {})
        participant_status["cache"] = "aborted"
        
        return {
            "messages": [AIMessage(content=f"⚡ Cache Service (ROLLBACK): {response.content}")],
            "participant_status": participant_status
        }
    
    return {"messages": []}


# Participant 3: Message Queue Service
def queue_participant(state: TwoPhaseCommitState) -> TwoPhaseCommitState:
    """Message queue service participant"""
    transaction = state["transaction"]
    phase = state.get("phase", "")
    
    if phase == "prepare":
        system_message = SystemMessage(content="""You are the message queue service participant. 
        Evaluate if you can prepare to publish messages for this transaction. Respond READY or ABORT.""")
        
        user_message = HumanMessage(content=f"""PREPARE request for transaction:\n{transaction}
        
        Can you prepare message publishing? Respond: READY or ABORT with reasoning.""")
        
        response = llm.invoke([system_message, user_message])
        
        vote = "ready" if "READY" in response.content.upper() else "abort"
        
        participant_votes = state.get("participant_votes", {})
        participant_votes["queue"] = vote
        
        return {
            "messages": [AIMessage(content=f"📨 Message Queue (PREPARE): {response.content}")],
            "participant_votes": participant_votes
        }
    
    elif phase == "commit":
        system_message = SystemMessage(content="""You are the message queue service. Execute 
        COMMIT. Publish the prepared messages to subscribers.""")
        
        user_message = HumanMessage(content=f"""COMMIT transaction:\n{transaction}
        
        Publish messages and report success.""")
        
        response = llm.invoke([system_message, user_message])
        
        participant_status = state.get("participant_status", {})
        participant_status["queue"] = "committed"
        
        return {
            "messages": [AIMessage(content=f"📨 Message Queue (COMMIT): {response.content}")],
            "participant_status": participant_status
        }
    
    elif phase == "rollback":
        system_message = SystemMessage(content="""You are the message queue service. Execute 
        ROLLBACK. Discard prepared messages without publishing.""")
        
        user_message = HumanMessage(content=f"""ROLLBACK transaction:\n{transaction}
        
        Discard messages and report completion.""")
        
        response = llm.invoke([system_message, user_message])
        
        participant_status = state.get("participant_status", {})
        participant_status["queue"] = "aborted"
        
        return {
            "messages": [AIMessage(content=f"📨 Message Queue (ROLLBACK): {response.content}")],
            "participant_status": participant_status
        }
    
    return {"messages": []}


# Coordinator Decision Node
def coordinator_decision(state: TwoPhaseCommitState) -> TwoPhaseCommitState:
    """Coordinator makes commit/abort decision based on votes"""
    participant_votes = state.get("participant_votes", {})
    
    # Check if all participants voted READY
    all_ready = all(vote == "ready" for vote in participant_votes.values())
    
    if all_ready:
        decision = "commit"
        summary = f"""
        ✅ ALL PARTICIPANTS READY - DECISION: COMMIT
        
        Votes received:
        {chr(10).join([f'  - {p}: {v.upper()}' for p, v in participant_votes.items()])}
        
        Proceeding to COMMIT phase...
        """
    else:
        decision = "abort"
        abort_reasons = [p for p, v in participant_votes.items() if v == "abort"]
        summary = f"""
        ❌ NOT ALL PARTICIPANTS READY - DECISION: ABORT
        
        Votes received:
        {chr(10).join([f'  - {p}: {v.upper()}' for p, v in participant_votes.items()])}
        
        Participants that voted ABORT: {', '.join(abort_reasons)}
        
        Proceeding to ROLLBACK phase...
        """
    
    return {
        "messages": [AIMessage(content=f"🎯 Coordinator (DECISION): {summary}")],
        "coordinator_decision": decision,
        "phase": decision if decision == "commit" else "rollback"
    }


# Final Result Node
def coordinator_finalize(state: TwoPhaseCommitState) -> TwoPhaseCommitState:
    """Coordinator announces final transaction result"""
    coordinator_decision = state.get("coordinator_decision", "")
    participant_status = state.get("participant_status", {})
    
    if coordinator_decision == "commit":
        result = "SUCCESS"
        summary = f"""
        🎉 TRANSACTION COMMITTED SUCCESSFULLY
        
        All participants have committed:
        {chr(10).join([f'  - {p}: {s}' for p, s in participant_status.items()])}
        
        Transaction is complete and durable across all services.
        """
    else:
        result = "ABORTED"
        summary = f"""
        🔄 TRANSACTION ABORTED
        
        All participants have rolled back:
        {chr(10).join([f'  - {p}: {s}' for p, s in participant_status.items()])}
        
        System returned to consistent state. No changes applied.
        """
    
    return {
        "messages": [AIMessage(content=f"🎯 Coordinator (FINAL): {summary}")],
        "transaction_result": result
    }


# Routing logic
def route_phase(state: TwoPhaseCommitState) -> str:
    """Route to commit or rollback based on decision"""
    decision = state.get("coordinator_decision", "")
    return "commit" if decision == "commit" else "rollback"


# Build the graph
def build_two_phase_commit_graph():
    """Build the two-phase commit MCP pattern graph"""
    workflow = StateGraph(TwoPhaseCommitState)
    
    # Add nodes
    workflow.add_node("prepare_coordinator", coordinator_prepare)
    workflow.add_node("prepare_database", database_participant)
    workflow.add_node("prepare_cache", cache_participant)
    workflow.add_node("prepare_queue", queue_participant)
    workflow.add_node("decision", coordinator_decision)
    workflow.add_node("commit_database", database_participant)
    workflow.add_node("commit_cache", cache_participant)
    workflow.add_node("commit_queue", queue_participant)
    workflow.add_node("finalize", coordinator_finalize)
    
    # Phase 1: PREPARE
    workflow.add_edge(START, "prepare_coordinator")
    workflow.add_edge("prepare_coordinator", "prepare_database")
    workflow.add_edge("prepare_coordinator", "prepare_cache")
    workflow.add_edge("prepare_coordinator", "prepare_queue")
    
    # Collect votes
    workflow.add_edge("prepare_database", "decision")
    workflow.add_edge("prepare_cache", "decision")
    workflow.add_edge("prepare_queue", "decision")
    
    # Phase 2: COMMIT or ROLLBACK
    workflow.add_conditional_edges(
        "decision",
        route_phase,
        {
            "commit": "commit_database",
            "rollback": "commit_database"  # Same node handles both based on state.phase
        }
    )
    
    # Execute commit/rollback on all participants
    workflow.add_edge("commit_database", "commit_cache")
    workflow.add_edge("commit_cache", "commit_queue")
    workflow.add_edge("commit_queue", "finalize")
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_two_phase_commit_graph()
    
    # Scenario 1: Successful transaction
    print("=== Scenario 1: Successful Distributed Transaction ===\n")
    
    initial_state = {
        "messages": [],
        "transaction": """Transfer $500 from Account A to Account B
        
        Operations:
        1. Database: Update account balances (A: -$500, B: +$500)
        2. Cache: Invalidate cached balance for Account A and B
        3. Message Queue: Publish transfer notification event
        
        Transaction ID: TXN-2024-001
        Timestamp: 2024-11-09 10:30:00 UTC""",
        "phase": "",
        "participant_votes": {},
        "participant_status": {},
        "coordinator_decision": "",
        "transaction_result": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Transaction Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
    
    print(f"\n\nFinal Result: {result['transaction_result']}")
    
    # Scenario 2: Transaction with abort
    print("\n\n" + "="*70)
    print("=== Scenario 2: Transaction Aborted (Participant Cannot Prepare) ===\n")
    
    initial_state_2 = {
        "messages": [],
        "transaction": """Large Bulk Update Operation
        
        Operations:
        1. Database: Update 1 million user records
        2. Cache: Invalidate entire user cache (500GB data)
        3. Message Queue: Publish 1 million update events
        
        Transaction ID: TXN-2024-002
        Timestamp: 2024-11-09 11:00:00 UTC
        
        Note: This is a high-load operation that may exceed resource limits.""",
        "phase": "",
        "participant_votes": {},
        "participant_status": {},
        "coordinator_decision": "",
        "transaction_result": ""
    }
    
    result_2 = graph.invoke(initial_state_2)
    
    print("\n=== Transaction Execution ===")
    for msg in result_2["messages"]:
        print(f"\n{msg.content}")
    
    print(f"\n\nFinal Result: {result_2['transaction_result']}")

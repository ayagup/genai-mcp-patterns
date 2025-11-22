"""
Quorum-Based MCP Pattern

This pattern demonstrates requiring a minimum number (quorum) of agents to 
agree before proceeding with a decision or action.

Key Features:
- Quorum threshold requirement
- Distributed decision making
- Agreement tracking and validation
- Fallback when quorum not met
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class QuorumState(TypedDict):
    """State for quorum-based decision making"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    proposal: str
    total_agents: int
    quorum_size: int  # Minimum number of approvals needed
    approvals: list[str]  # List of agent IDs that approved
    rejections: list[str]  # List of agent IDs that rejected
    quorum_met: bool
    final_decision: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Agent 1: Database Administrator
def db_admin_agent(state: QuorumState) -> QuorumState:
    """Database administrator evaluates data-related impacts"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a database administrator. Evaluate the 
    proposal from a data management perspective considering: data integrity, migration complexity, 
    performance impact, and backup/recovery. Respond with APPROVE or REJECT and explain why.""")
    
    user_message = HumanMessage(content=f"""Evaluate this proposal:\n{proposal}
    
    Decision: [APPROVE/REJECT]
    Reasoning: [Your detailed explanation]""")
    
    response = llm.invoke([system_message, user_message])
    
    approved = "APPROVE" in response.content.upper() and "Decision: APPROVE" in response.content.upper()
    
    approvals = state.get("approvals", [])
    rejections = state.get("rejections", [])
    
    if approved:
        approvals.append("db_admin")
    else:
        rejections.append("db_admin")
    
    return {
        "messages": [AIMessage(content=f"DB Admin: {response.content}")],
        "approvals": approvals,
        "rejections": rejections
    }


# Agent 2: Security Officer
def security_officer_agent(state: QuorumState) -> QuorumState:
    """Security officer evaluates security implications"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a security officer. Evaluate the proposal 
    for security risks, compliance requirements, authentication/authorization impacts, and 
    vulnerability concerns. Respond with APPROVE or REJECT and explain why.""")
    
    user_message = HumanMessage(content=f"""Evaluate this proposal:\n{proposal}
    
    Decision: [APPROVE/REJECT]
    Reasoning: [Your detailed explanation]""")
    
    response = llm.invoke([system_message, user_message])
    
    approved = "APPROVE" in response.content.upper() and "Decision: APPROVE" in response.content.upper()
    
    approvals = state.get("approvals", [])
    rejections = state.get("rejections", [])
    
    if approved:
        approvals.append("security_officer")
    else:
        rejections.append("security_officer")
    
    return {
        "messages": [AIMessage(content=f"Security Officer: {response.content}")],
        "approvals": approvals,
        "rejections": rejections
    }


# Agent 3: Operations Manager
def ops_manager_agent(state: QuorumState) -> QuorumState:
    """Operations manager evaluates operational impact"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are an operations manager. Evaluate the proposal 
    for deployment complexity, system reliability, monitoring needs, and operational overhead. 
    Respond with APPROVE or REJECT and explain why.""")
    
    user_message = HumanMessage(content=f"""Evaluate this proposal:\n{proposal}
    
    Decision: [APPROVE/REJECT]
    Reasoning: [Your detailed explanation]""")
    
    response = llm.invoke([system_message, user_message])
    
    approved = "APPROVE" in response.content.upper() and "Decision: APPROVE" in response.content.upper()
    
    approvals = state.get("approvals", [])
    rejections = state.get("rejections", [])
    
    if approved:
        approvals.append("ops_manager")
    else:
        rejections.append("ops_manager")
    
    return {
        "messages": [AIMessage(content=f"Operations Manager: {response.content}")],
        "approvals": approvals,
        "rejections": rejections
    }


# Agent 4: Development Lead
def dev_lead_agent(state: QuorumState) -> QuorumState:
    """Development lead evaluates technical implementation"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a development lead. Evaluate the proposal 
    for technical feasibility, code maintainability, testing requirements, and development effort. 
    Respond with APPROVE or REJECT and explain why.""")
    
    user_message = HumanMessage(content=f"""Evaluate this proposal:\n{proposal}
    
    Decision: [APPROVE/REJECT]
    Reasoning: [Your detailed explanation]""")
    
    response = llm.invoke([system_message, user_message])
    
    approved = "APPROVE" in response.content.upper() and "Decision: APPROVE" in response.content.upper()
    
    approvals = state.get("approvals", [])
    rejections = state.get("rejections", [])
    
    if approved:
        approvals.append("dev_lead")
    else:
        rejections.append("dev_lead")
    
    return {
        "messages": [AIMessage(content=f"Development Lead: {response.content}")],
        "approvals": approvals,
        "rejections": rejections
    }


# Agent 5: Product Owner
def product_owner_agent(state: QuorumState) -> QuorumState:
    """Product owner evaluates business value"""
    proposal = state["proposal"]
    
    system_message = SystemMessage(content="""You are a product owner. Evaluate the proposal 
    for business value, user impact, ROI, and strategic alignment. 
    Respond with APPROVE or REJECT and explain why.""")
    
    user_message = HumanMessage(content=f"""Evaluate this proposal:\n{proposal}
    
    Decision: [APPROVE/REJECT]
    Reasoning: [Your detailed explanation]""")
    
    response = llm.invoke([system_message, user_message])
    
    approved = "APPROVE" in response.content.upper() and "Decision: APPROVE" in response.content.upper()
    
    approvals = state.get("approvals", [])
    rejections = state.get("rejections", [])
    
    if approved:
        approvals.append("product_owner")
    else:
        rejections.append("product_owner")
    
    return {
        "messages": [AIMessage(content=f"Product Owner: {response.content}")],
        "approvals": approvals,
        "rejections": rejections
    }


# Quorum Checker Agent
def quorum_checker(state: QuorumState) -> QuorumState:
    """Checks if quorum is met and makes final decision"""
    approvals = state.get("approvals", [])
    rejections = state.get("rejections", [])
    quorum_size = state["quorum_size"]
    total_agents = state["total_agents"]
    
    approval_count = len(approvals)
    rejection_count = len(rejections)
    
    quorum_met = approval_count >= quorum_size
    
    if quorum_met:
        decision = "APPROVED"
        summary = f"""
        ✅ QUORUM ACHIEVED - PROPOSAL APPROVED
        
        Approvals: {approval_count}/{total_agents} (Required: {quorum_size})
        Rejections: {rejection_count}/{total_agents}
        
        Approved by: {', '.join(approvals)}
        Rejected by: {', '.join(rejections) if rejections else 'None'}
        
        The proposal has met the quorum requirement and is APPROVED for implementation.
        """
    else:
        # Check if quorum is impossible to achieve
        remaining = total_agents - (approval_count + rejection_count)
        max_possible = approval_count + remaining
        
        if max_possible < quorum_size:
            decision = "REJECTED"
            summary = f"""
            ❌ QUORUM CANNOT BE MET - PROPOSAL REJECTED
            
            Approvals: {approval_count}/{total_agents} (Required: {quorum_size})
            Rejections: {rejection_count}/{total_agents}
            Remaining votes: {remaining}
            Maximum possible approvals: {max_possible}
            
            Approved by: {', '.join(approvals) if approvals else 'None'}
            Rejected by: {', '.join(rejections)}
            
            Even with all remaining votes, quorum cannot be achieved. Proposal is REJECTED.
            """
        else:
            decision = "PENDING"
            summary = f"""
            ⏳ QUORUM PENDING
            
            Approvals: {approval_count}/{total_agents} (Required: {quorum_size})
            Rejections: {rejection_count}/{total_agents}
            Remaining votes needed: {quorum_size - approval_count}
            
            Approved by: {', '.join(approvals) if approvals else 'None'}
            Rejected by: {', '.join(rejections) if rejections else 'None'}
            
            Awaiting additional approvals to meet quorum requirement.
            """
    
    return {
        "messages": [AIMessage(content=f"Quorum Checker:\n{summary}")],
        "quorum_met": quorum_met,
        "final_decision": decision
    }


# Build the graph
def build_quorum_graph():
    """Build the quorum-based MCP pattern graph"""
    workflow = StateGraph(QuorumState)
    
    # Add nodes
    workflow.add_node("db_admin", db_admin_agent)
    workflow.add_node("security_officer", security_officer_agent)
    workflow.add_node("ops_manager", ops_manager_agent)
    workflow.add_node("dev_lead", dev_lead_agent)
    workflow.add_node("product_owner", product_owner_agent)
    workflow.add_node("quorum_checker", quorum_checker)
    
    # All agents evaluate in parallel
    workflow.add_edge(START, "db_admin")
    workflow.add_edge(START, "security_officer")
    workflow.add_edge(START, "ops_manager")
    workflow.add_edge(START, "dev_lead")
    workflow.add_edge(START, "product_owner")
    
    # All feed into quorum checker
    workflow.add_edge("db_admin", "quorum_checker")
    workflow.add_edge("security_officer", "quorum_checker")
    workflow.add_edge("ops_manager", "quorum_checker")
    workflow.add_edge("dev_lead", "quorum_checker")
    workflow.add_edge("product_owner", "quorum_checker")
    
    workflow.add_edge("quorum_checker", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_quorum_graph()
    
    # Scenario 1: Proposal that should meet quorum
    print("=== Scenario 1: Database Migration Proposal ===\n")
    
    initial_state = {
        "messages": [],
        "proposal": """Proposal: Migrate from PostgreSQL to PostgreSQL with Read Replicas
        
        Objective: Improve read performance and scalability
        
        Details:
        - Add 3 read replicas across different availability zones
        - Implement read/write splitting in application layer
        - Estimated cost: $50K setup + $10K/month operational
        - Timeline: 2 months planning + implementation
        - Minimal downtime: 2-hour maintenance window
        
        Benefits:
        - 5x improvement in read query performance
        - Better geographical distribution
        - Improved disaster recovery
        
        Risks:
        - Replication lag concerns
        - Application changes needed for read/write splitting
        - Increased operational complexity""",
        "total_agents": 5,
        "quorum_size": 3,  # Need 3 out of 5 approvals (60%)
        "approvals": [],
        "rejections": [],
        "quorum_met": False,
        "final_decision": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Agent Evaluations ===")
    for msg in result["messages"][:-1]:
        print(f"\n{msg.content}")
    
    print("\n" + "="*60)
    print(result["messages"][-1].content)
    
    # Scenario 2: Controversial proposal
    print("\n\n" + "="*60)
    print("=== Scenario 2: Controversial Proposal ===\n")
    
    initial_state_2 = {
        "messages": [],
        "proposal": """Proposal: Complete Rewrite in New Technology Stack
        
        Objective: Modernize application using latest technologies
        
        Details:
        - Rewrite entire backend from Java to Rust
        - Replace React frontend with Svelte
        - Migrate from REST to GraphQL
        - Estimated cost: $2M
        - Timeline: 18 months
        - Complete feature freeze during migration
        
        Benefits:
        - Better performance (claimed 10x improvement)
        - Modern developer experience
        - Reduced technical debt
        
        Risks:
        - Massive resource investment
        - Business feature development halted
        - Team needs to learn new technologies
        - Potential for bugs and regressions
        - No incremental rollback possible""",
        "total_agents": 5,
        "quorum_size": 4,  # Need 4 out of 5 approvals (80% supermajority)
        "approvals": [],
        "rejections": [],
        "quorum_met": False,
        "final_decision": ""
    }
    
    result_2 = graph.invoke(initial_state_2)
    
    print("\n=== Agent Evaluations ===")
    for msg in result_2["messages"][:-1]:
        print(f"\n{msg.content}")
    
    print("\n" + "="*60)
    print(result_2["messages"][-1].content)

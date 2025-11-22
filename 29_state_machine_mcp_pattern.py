"""
State Machine MCP Pattern

This pattern demonstrates workflow coordination using explicit state transitions 
with guards and actions. Agents execute tasks based on current state and transition 
rules.

Key Features:
- Explicit state definitions
- State transition rules with conditions
- Guards for transition validation
- Actions triggered on state changes
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class StateMachineState(TypedDict):
    """State for state machine pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    current_state: str  # draft, review, approved, published, rejected
    document_content: str
    review_feedback: str
    approval_decision: str
    revision_count: int
    max_revisions: int
    final_status: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# State: DRAFT
def draft_state_agent(state: StateMachineState) -> StateMachineState:
    """Agent in DRAFT state - creates initial document"""
    
    system_message = SystemMessage(content="""You are in the DRAFT state. Create an initial 
    document draft. Your document will transition to REVIEW state after completion.""")
    
    user_message = HumanMessage(content="""Create a draft document:
    
    Topic: "Best Practices for API Design"
    
    Write a draft covering key points.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📝 DRAFT State: {response.content}")],
        "document_content": response.content,
        "current_state": "review"  # Transition to REVIEW
    }


# State: REVIEW
def review_state_agent(state: StateMachineState) -> StateMachineState:
    """Agent in REVIEW state - reviews document and provides feedback"""
    document = state.get("document_content", "")
    
    system_message = SystemMessage(content="""You are in the REVIEW state. Review the document 
    for quality, accuracy, and completeness. Provide feedback and determine if it's ready for 
    approval or needs revision.""")
    
    user_message = HumanMessage(content=f"""Review this document:
    
    {document[:500]}...
    
    Provide review feedback and recommend: APPROVED or NEEDS_REVISION""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine next state based on review
    if "APPROVED" in response.content.upper() and "NEEDS_REVISION" not in response.content.upper():
        next_state = "approved"
    else:
        revision_count = state.get("revision_count", 0) + 1
        max_revisions = state.get("max_revisions", 3)
        
        if revision_count >= max_revisions:
            next_state = "rejected"
        else:
            next_state = "draft"  # Back to draft for revision
    
    return {
        "messages": [AIMessage(content=f"👁️ REVIEW State: {response.content}\n\nTransition to: {next_state.upper()}")],
        "review_feedback": response.content,
        "current_state": next_state,
        "revision_count": state.get("revision_count", 0) + 1 if next_state == "draft" else state.get("revision_count", 0)
    }


# State: APPROVED
def approved_state_agent(state: StateMachineState) -> StateMachineState:
    """Agent in APPROVED state - makes final approval decision"""
    document = state.get("document_content", "")
    review_feedback = state.get("review_feedback", "")
    
    system_message = SystemMessage(content="""You are in the APPROVED state. Make the final 
    approval decision. The document has passed review and is ready for publication unless there 
    are concerns.""")
    
    user_message = HumanMessage(content=f"""Final approval for document:
    
    Document: {document[:300]}...
    
    Review Feedback: {review_feedback[:300]}...
    
    Make final decision: PUBLISH or REJECT""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine next state
    if "PUBLISH" in response.content.upper() and "REJECT" not in response.content.upper():
        next_state = "published"
        decision = "approved_for_publication"
    else:
        next_state = "rejected"
        decision = "rejected_at_approval"
    
    return {
        "messages": [AIMessage(content=f"✅ APPROVED State: {response.content}\n\nTransition to: {next_state.upper()}")],
        "approval_decision": decision,
        "current_state": next_state
    }


# State: PUBLISHED
def published_state_agent(state: StateMachineState) -> StateMachineState:
    """Agent in PUBLISHED state - final state, publishes document"""
    document = state.get("document_content", "")
    
    system_message = SystemMessage(content="""You are in the PUBLISHED state (final state). 
    The document has been approved and is being published. Perform publication tasks.""")
    
    user_message = HumanMessage(content=f"""Publish the approved document:
    
    {document[:300]}...
    
    Execute publication process.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📤 PUBLISHED State (FINAL): {response.content}")],
        "final_status": "published"
    }


# State: REJECTED
def rejected_state_agent(state: StateMachineState) -> StateMachineState:
    """Agent in REJECTED state - final state, document rejected"""
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 3)
    
    if revision_count >= max_revisions:
        reason = f"Document rejected after {revision_count} revision attempts (max: {max_revisions})"
    else:
        reason = "Document rejected during approval process"
    
    system_message = SystemMessage(content="""You are in the REJECTED state (final state). 
    The document has been rejected. Provide closure and recommendations.""")
    
    user_message = HumanMessage(content=f"""Document rejected:
    
    Reason: {reason}
    
    Provide closure message and recommendations for future submissions.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"❌ REJECTED State (FINAL): {response.content}\n\nReason: {reason}")],
        "final_status": "rejected"
    }


# State Machine Controller
def state_machine_controller(state: StateMachineState) -> StateMachineState:
    """Controls state transitions and validates guards"""
    current_state = state.get("current_state", "draft")
    
    status_message = f"""
    🔄 STATE MACHINE CONTROLLER
    
    Current State: {current_state.upper()}
    Revision Count: {state.get('revision_count', 0)}/{state.get('max_revisions', 3)}
    
    State transition in progress...
    """
    
    return {
        "messages": [AIMessage(content=status_message)]
    }


# Routing logic
def route_to_state(state: StateMachineState) -> str:
    """Route to appropriate state handler"""
    current_state = state.get("current_state", "draft")
    
    state_routing = {
        "draft": "draft",
        "review": "review",
        "approved": "approved",
        "published": "published",
        "rejected": "rejected"
    }
    
    return state_routing.get(current_state, "end")


def check_final_state(state: StateMachineState) -> str:
    """Check if we've reached a final state"""
    current_state = state.get("current_state", "")
    
    if current_state in ["published", "rejected"]:
        return "end"
    else:
        return "continue"


# Build the graph
def build_state_machine_graph():
    """Build the state machine MCP pattern graph"""
    workflow = StateGraph(StateMachineState)
    
    # Add nodes for each state
    workflow.add_node("controller", state_machine_controller)
    workflow.add_node("draft", draft_state_agent)
    workflow.add_node("review", review_state_agent)
    workflow.add_node("approved", approved_state_agent)
    workflow.add_node("published", published_state_agent)
    workflow.add_node("rejected", rejected_state_agent)
    
    # Start with controller
    workflow.add_edge(START, "controller")
    
    # Controller routes to current state
    workflow.add_conditional_edges(
        "controller",
        route_to_state,
        {
            "draft": "draft",
            "review": "review",
            "approved": "approved",
            "published": "published",
            "rejected": "rejected",
            "end": END
        }
    )
    
    # State transitions back to controller (except final states)
    workflow.add_conditional_edges(
        "draft",
        check_final_state,
        {
            "continue": "controller",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "review",
        check_final_state,
        {
            "continue": "controller",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "approved",
        check_final_state,
        {
            "continue": "controller",
            "end": END
        }
    )
    
    # Final states go directly to END
    workflow.add_edge("published", END)
    workflow.add_edge("rejected", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_state_machine_graph()
    
    print("=== State Machine MCP Pattern: Document Approval Workflow ===\n")
    print("This demonstrates explicit state transitions with guards and conditions.")
    print("States: DRAFT → REVIEW → APPROVED → PUBLISHED")
    print("         ↓        ↓                    ↓")
    print("      (revise)  REJECTED            REJECTED\n")
    
    # Scenario 1: Successful publication
    print("=== Scenario 1: Document Approval Flow ===\n")
    
    initial_state = {
        "messages": [],
        "current_state": "draft",
        "document_content": "",
        "review_feedback": "",
        "approval_decision": "",
        "revision_count": 0,
        "max_revisions": 3,
        "final_status": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== State Machine Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n\n=== Final State Machine Status ===")
    print(f"Final State: {result['current_state'].upper()}")
    print(f"Final Status: {result['final_status'].upper()}")
    print(f"Total Revisions: {result['revision_count']}")
    
    # Show state transition diagram
    print("\n\n=== State Transition Diagram ===")
    print("""
    START
      │
      ▼
    DRAFT ─────────► REVIEW ─────────► APPROVED ─────────► PUBLISHED (END)
      ▲                │                    │
      │                │                    │
      └────(revise)────┘                    │
                       │                    │
                       ▼                    ▼
                    REJECTED (END) ◄──── REJECTED (END)
                   (max revisions)      (approval denied)
    """)

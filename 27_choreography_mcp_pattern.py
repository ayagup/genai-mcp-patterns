"""
Choreography MCP Pattern

This pattern demonstrates decentralized coordination where agents react to 
events without a central coordinator. Each agent knows what to do when specific 
events occur.

Key Features:
- Event-driven coordination
- No central coordinator
- Agents react independently to events
- Loosely coupled interactions
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ChoreographyState(TypedDict):
    """State for choreography pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    events: list[dict[str, str]]  # List of events: {type: str, data: str, source: str}
    process_name: str
    agents_completed: list[str]
    workflow_status: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Event Publisher (simulates external trigger)
def event_publisher(state: ChoreographyState) -> ChoreographyState:
    """Publishes initial event to trigger the workflow"""
    process_name = state["process_name"]
    
    # Publish initial event
    event = {
        "type": "customer_registration_submitted",
        "data": "New customer: John Doe, email: john@example.com",
        "source": "registration_service"
    }
    
    events = state.get("events", [])
    events.append(event)
    
    message = f"""
    📢 EVENT PUBLISHED
    Type: {event['type']}
    Source: {event['source']}
    Data: {event['data']}
    """
    
    return {
        "messages": [AIMessage(content=message)],
        "events": events
    }


# Agent 1: Customer Service (reacts to registration_submitted)
def customer_service_agent(state: ChoreographyState) -> ChoreographyState:
    """Reacts to customer_registration_submitted event"""
    events = state.get("events", [])
    
    # Check if relevant event exists
    relevant_event = None
    for event in events:
        if event["type"] == "customer_registration_submitted":
            relevant_event = event
            break
    
    if not relevant_event:
        return {"messages": []}
    
    system_message = SystemMessage(content="""You are the Customer Service agent. When a 
    customer_registration_submitted event occurs, you create a customer profile in the database 
    and publish a customer_profile_created event.""")
    
    user_message = HumanMessage(content=f"""Event received: {relevant_event['type']}
    Data: {relevant_event['data']}
    
    React to this event by:
    1. Creating customer profile
    2. Publishing customer_profile_created event""")
    
    response = llm.invoke([system_message, user_message])
    
    # Publish new event
    new_event = {
        "type": "customer_profile_created",
        "data": "Customer profile created: ID=12345, John Doe",
        "source": "customer_service"
    }
    
    events.append(new_event)
    agents_completed = state.get("agents_completed", [])
    agents_completed.append("customer_service")
    
    return {
        "messages": [AIMessage(content=f"👤 Customer Service Agent:\n{response.content}\n\n📢 Published Event: customer_profile_created")],
        "events": events,
        "agents_completed": agents_completed
    }


# Agent 2: Email Service (reacts to profile_created)
def email_service_agent(state: ChoreographyState) -> ChoreographyState:
    """Reacts to customer_profile_created event"""
    events = state.get("events", [])
    
    # Check if relevant event exists
    relevant_event = None
    for event in events:
        if event["type"] == "customer_profile_created":
            relevant_event = event
            break
    
    if not relevant_event:
        return {"messages": []}
    
    system_message = SystemMessage(content="""You are the Email Service agent. When a 
    customer_profile_created event occurs, you send a welcome email and publish a 
    welcome_email_sent event.""")
    
    user_message = HumanMessage(content=f"""Event received: {relevant_event['type']}
    Data: {relevant_event['data']}
    
    React to this event by:
    1. Sending welcome email to customer
    2. Publishing welcome_email_sent event""")
    
    response = llm.invoke([system_message, user_message])
    
    # Publish new event
    new_event = {
        "type": "welcome_email_sent",
        "data": "Welcome email sent to john@example.com",
        "source": "email_service"
    }
    
    events.append(new_event)
    agents_completed = state.get("agents_completed", [])
    agents_completed.append("email_service")
    
    return {
        "messages": [AIMessage(content=f"📧 Email Service Agent:\n{response.content}\n\n📢 Published Event: welcome_email_sent")],
        "events": events,
        "agents_completed": agents_completed
    }


# Agent 3: Analytics Service (reacts to profile_created)
def analytics_service_agent(state: ChoreographyState) -> ChoreographyState:
    """Reacts to customer_profile_created event (parallel to email)"""
    events = state.get("events", [])
    
    # Check if relevant event exists
    relevant_event = None
    for event in events:
        if event["type"] == "customer_profile_created":
            relevant_event = event
            break
    
    if not relevant_event:
        return {"messages": []}
    
    system_message = SystemMessage(content="""You are the Analytics Service agent. When a 
    customer_profile_created event occurs, you track the registration in analytics and publish 
    an analytics_tracked event.""")
    
    user_message = HumanMessage(content=f"""Event received: {relevant_event['type']}
    Data: {relevant_event['data']}
    
    React to this event by:
    1. Recording customer registration event in analytics
    2. Publishing analytics_tracked event""")
    
    response = llm.invoke([system_message, user_message])
    
    # Publish new event
    new_event = {
        "type": "analytics_tracked",
        "data": "Registration event tracked in analytics",
        "source": "analytics_service"
    }
    
    events.append(new_event)
    agents_completed = state.get("agents_completed", [])
    agents_completed.append("analytics_service")
    
    return {
        "messages": [AIMessage(content=f"📊 Analytics Service Agent:\n{response.content}\n\n📢 Published Event: analytics_tracked")],
        "events": events,
        "agents_completed": agents_completed
    }


# Agent 4: Loyalty Service (reacts to email_sent)
def loyalty_service_agent(state: ChoreographyState) -> ChoreographyState:
    """Reacts to welcome_email_sent event"""
    events = state.get("events", [])
    
    # Check if relevant event exists
    relevant_event = None
    for event in events:
        if event["type"] == "welcome_email_sent":
            relevant_event = event
            break
    
    if not relevant_event:
        return {"messages": []}
    
    system_message = SystemMessage(content="""You are the Loyalty Service agent. When a 
    welcome_email_sent event occurs, you create a loyalty account and award welcome points. 
    Publish a loyalty_account_created event.""")
    
    user_message = HumanMessage(content=f"""Event received: {relevant_event['type']}
    Data: {relevant_event['data']}
    
    React to this event by:
    1. Creating loyalty account
    2. Awarding 100 welcome points
    3. Publishing loyalty_account_created event""")
    
    response = llm.invoke([system_message, user_message])
    
    # Publish new event
    new_event = {
        "type": "loyalty_account_created",
        "data": "Loyalty account created with 100 welcome points",
        "source": "loyalty_service"
    }
    
    events.append(new_event)
    agents_completed = state.get("agents_completed", [])
    agents_completed.append("loyalty_service")
    
    return {
        "messages": [AIMessage(content=f"🎁 Loyalty Service Agent:\n{response.content}\n\n📢 Published Event: loyalty_account_created")],
        "events": events,
        "agents_completed": agents_completed
    }


# Workflow Monitor (checks completion)
def workflow_monitor(state: ChoreographyState) -> ChoreographyState:
    """Monitors the choreography and reports status"""
    events = state.get("events", [])
    agents_completed = state.get("agents_completed", [])
    
    system_message = SystemMessage(content="""You are the Workflow Monitor. Review all events 
    that occurred in this choreographed workflow and provide a summary.""")
    
    events_summary = "\n".join([
        f"  - {e['type']} (from {e['source']}): {e['data']}"
        for e in events
    ])
    
    user_message = HumanMessage(content=f"""Choreographed workflow completed.
    
    Events that occurred:
    {events_summary}
    
    Agents that participated:
    {', '.join(agents_completed)}
    
    Provide a summary of the choreography execution.""")
    
    response = llm.invoke([system_message, user_message])
    
    summary = f"""
    ✅ CHOREOGRAPHY COMPLETED
    
    Total Events: {len(events)}
    Agents Participated: {len(agents_completed)}
    
    Event Chain:
    {events_summary}
    
    Monitor Analysis:
    {response.content}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Workflow Monitor:\n{summary}")],
        "workflow_status": "completed"
    }


# Build the graph
def build_choreography_graph():
    """Build the choreography MCP pattern graph"""
    workflow = StateGraph(ChoreographyState)
    
    # Add nodes
    workflow.add_node("publisher", event_publisher)
    workflow.add_node("customer_service", customer_service_agent)
    workflow.add_node("email_service", email_service_agent)
    workflow.add_node("analytics_service", analytics_service_agent)
    workflow.add_node("loyalty_service", loyalty_service_agent)
    workflow.add_node("monitor", workflow_monitor)
    
    # Event-driven choreography flow
    workflow.add_edge(START, "publisher")
    
    # First event triggers customer service
    workflow.add_edge("publisher", "customer_service")
    
    # Customer service event triggers email and analytics in parallel
    workflow.add_edge("customer_service", "email_service")
    workflow.add_edge("customer_service", "analytics_service")
    
    # Email sent event triggers loyalty service
    workflow.add_edge("email_service", "loyalty_service")
    
    # Analytics and loyalty converge to monitor
    workflow.add_edge("analytics_service", "monitor")
    workflow.add_edge("loyalty_service", "monitor")
    
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_choreography_graph()
    
    print("=== Choreography MCP Pattern: Customer Registration Workflow ===\n")
    print("This demonstrates decentralized coordination through event reactions.")
    print("No central orchestrator - each service reacts to events independently.\n")
    
    initial_state = {
        "messages": [],
        "events": [],
        "process_name": "Customer Registration",
        "agents_completed": [],
        "workflow_status": "in_progress"
    }
    
    # Run the choreography
    result = graph.invoke(initial_state)
    
    print("\n=== Choreography Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n\n=== Final Event Chain ===")
    for i, event in enumerate(result["events"], 1):
        print(f"{i}. {event['type']}")
        print(f"   Source: {event['source']}")
        print(f"   Data: {event['data']}\n")

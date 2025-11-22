"""
Event-Driven MCP Pattern
=========================
Event-driven architecture where agents react to events emitted by other agents.
Events trigger specific handlers that process the event data.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator


# Define the state
class EventDrivenState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    events: List[Dict[str, str]]  # List of events: {type, data, source}
    event_log: List[str]  # Log of all events
    handlers_triggered: List[str]  # Which handlers were triggered


# Event Emitter Agent
def event_emitter_agent(state: EventDrivenState):
    """Agent that emits events based on conditions."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are an Event Emitter. Analyze the input and emit relevant events:\n"
                "Possible event types: 'user_action', 'system_alert', 'data_update'\n"
                "Emit events with appropriate data."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Emit multiple events
    events = state.get("events", [])
    event_log = state.get("event_log", [])
    
    # Emit user_action event
    events.append({
        "type": "user_action",
        "data": response.content[:100],
        "source": "event_emitter"
    })
    event_log.append(f"Event emitted: user_action")
    
    # Emit system_alert event
    events.append({
        "type": "system_alert",
        "data": "Alert: " + response.content[100:150],
        "source": "event_emitter"
    })
    event_log.append(f"Event emitted: system_alert")
    
    # Emit data_update event
    events.append({
        "type": "data_update",
        "data": "Update: " + response.content[150:200],
        "source": "event_emitter"
    })
    event_log.append(f"Event emitted: data_update")
    
    return {
        "messages": [AIMessage(content=f"Event Emitter: Emitted events - {response.content}")],
        "events": events,
        "event_log": event_log
    }


# User Action Handler
def user_action_handler(state: EventDrivenState):
    """Handler that processes user_action events."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    events = state.get("events", [])
    user_events = [e for e in events if e["type"] == "user_action"]
    
    if not user_events:
        return {"messages": [], "handlers_triggered": []}
    
    system_msg = SystemMessage(
        content=f"You are a User Action Handler. Process these user action events:\n"
                f"Events: {user_events}\n"
                "Respond appropriately to user actions."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    handlers = state.get("handlers_triggered", [])
    handlers.append("user_action_handler")
    
    return {
        "messages": [AIMessage(content=f"User Action Handler: {response.content}")],
        "handlers_triggered": handlers
    }


# System Alert Handler
def system_alert_handler(state: EventDrivenState):
    """Handler that processes system_alert events."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    events = state.get("events", [])
    alert_events = [e for e in events if e["type"] == "system_alert"]
    
    if not alert_events:
        return {"messages": [], "handlers_triggered": []}
    
    system_msg = SystemMessage(
        content=f"You are a System Alert Handler. Process these alert events:\n"
                f"Events: {alert_events}\n"
                "Take appropriate action on system alerts."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    handlers = state.get("handlers_triggered", [])
    handlers.append("system_alert_handler")
    
    return {
        "messages": [AIMessage(content=f"System Alert Handler: {response.content}")],
        "handlers_triggered": handlers
    }


# Data Update Handler
def data_update_handler(state: EventDrivenState):
    """Handler that processes data_update events."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    events = state.get("events", [])
    update_events = [e for e in events if e["type"] == "data_update"]
    
    if not update_events:
        return {"messages": [], "handlers_triggered": []}
    
    system_msg = SystemMessage(
        content=f"You are a Data Update Handler. Process these data update events:\n"
                f"Events: {update_events}\n"
                "Update relevant systems with new data."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    handlers = state.get("handlers_triggered", [])
    handlers.append("data_update_handler")
    
    return {
        "messages": [AIMessage(content=f"Data Update Handler: {response.content}")],
        "handlers_triggered": handlers
    }


# Event Coordinator
def event_coordinator(state: EventDrivenState):
    """Coordinates all event handling results."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    event_log = state.get("event_log", [])
    handlers = state.get("handlers_triggered", [])
    
    system_msg = SystemMessage(
        content=f"You are an Event Coordinator. Summarize event processing:\n"
                f"Event Log: {event_log}\n"
                f"Handlers Triggered: {handlers}\n"
                "Provide a summary of all event handling."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Event Coordinator: {response.content}")]
    }


# Build the event-driven graph
def create_event_driven_graph():
    """Create an event-driven workflow graph."""
    workflow = StateGraph(EventDrivenState)
    
    # Add nodes
    workflow.add_node("emitter", event_emitter_agent)
    workflow.add_node("user_handler", user_action_handler)
    workflow.add_node("alert_handler", system_alert_handler)
    workflow.add_node("update_handler", data_update_handler)
    workflow.add_node("coordinator", event_coordinator)
    
    # Event-driven flow: Emitter -> Handlers (parallel) -> Coordinator
    workflow.add_edge(START, "emitter")
    workflow.add_edge("emitter", "user_handler")
    workflow.add_edge("emitter", "alert_handler")
    workflow.add_edge("emitter", "update_handler")
    workflow.add_edge("user_handler", "coordinator")
    workflow.add_edge("alert_handler", "coordinator")
    workflow.add_edge("update_handler", "coordinator")
    workflow.add_edge("coordinator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_event_driven_graph()
    
    print("=" * 60)
    print("EVENT-DRIVEN MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Event-based system
    print("\n[Scenario: User Activity Monitoring]")
    result = graph.invoke({
        "messages": [HumanMessage(content="User John logged in, system detected unusual activity, database updated with new records")],
        "events": [],
        "event_log": [],
        "handlers_triggered": []
    })
    
    print("\n--- Event-Driven Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Event Log ---")
    for log in result.get("event_log", []):
        print(f"  - {log}")
    
    print(f"\n--- Events Emitted ---")
    for event in result.get("events", []):
        print(f"  Type: {event['type']}, Source: {event['source']}")
    
    print(f"\n--- Handlers Triggered ---")
    for handler in result.get("handlers_triggered", []):
        print(f"  - {handler}")
    
    print("\n" + "=" * 60)

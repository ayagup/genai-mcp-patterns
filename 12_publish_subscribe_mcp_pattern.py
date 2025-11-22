"""
Publish-Subscribe MCP Pattern
==============================
Pub-Sub messaging pattern where publishers send messages to topics,
and subscribers receive messages from topics they're interested in.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator


# Define the state
class PubSubState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    topics: Dict[str, List[str]]  # Topic -> messages
    subscriptions: Dict[str, List[str]]  # Subscriber -> topics
    published_messages: Dict[str, str]  # Topic -> message
    subscriber_inbox: Dict[str, List[str]]  # Subscriber -> received messages


# Publisher Agent
def publisher_agent(state: PubSubState):
    """Agent that publishes messages to topics."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Publisher Agent. Create messages to publish to different topics:\n"
                "Available topics: 'news', 'alerts', 'updates'\n"
                "Generate relevant content for each topic."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Publish to multiple topics
    topics = state.get("topics", {})
    published = {}
    
    topics.setdefault("news", []).append(f"News: {response.content[:50]}")
    topics.setdefault("alerts", []).append(f"Alert: {response.content[50:100]}")
    topics.setdefault("updates", []).append(f"Update: {response.content[100:150]}")
    
    published["news"] = response.content[:50]
    published["alerts"] = response.content[50:100]
    published["updates"] = response.content[100:150]
    
    return {
        "messages": [AIMessage(content=f"Publisher: Published to topics - {response.content}")],
        "topics": topics,
        "published_messages": published
    }


# Subscriber 1 - News and Alerts
def subscriber_1_agent(state: PubSubState):
    """Subscriber interested in news and alerts."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    subscriptions = state.get("subscriptions", {})
    subscriptions["subscriber_1"] = ["news", "alerts"]
    
    topics = state.get("topics", {})
    inbox = []
    
    # Receive messages from subscribed topics
    for topic in subscriptions["subscriber_1"]:
        if topic in topics:
            inbox.extend(topics[topic])
    
    system_msg = SystemMessage(
        content=f"You are Subscriber 1. You received messages from your subscribed topics (news, alerts):\n"
                f"Inbox: {inbox}\n"
                "Process and respond to these messages."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    subscriber_inbox = state.get("subscriber_inbox", {})
    subscriber_inbox["subscriber_1"] = inbox
    
    return {
        "messages": [AIMessage(content=f"Subscriber 1: {response.content}")],
        "subscriptions": subscriptions,
        "subscriber_inbox": subscriber_inbox
    }


# Subscriber 2 - Updates Only
def subscriber_2_agent(state: PubSubState):
    """Subscriber interested only in updates."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    subscriptions = state.get("subscriptions", {})
    subscriptions["subscriber_2"] = ["updates"]
    
    topics = state.get("topics", {})
    inbox = []
    
    # Receive messages from subscribed topics
    for topic in subscriptions["subscriber_2"]:
        if topic in topics:
            inbox.extend(topics[topic])
    
    system_msg = SystemMessage(
        content=f"You are Subscriber 2. You received messages from your subscribed topics (updates):\n"
                f"Inbox: {inbox}\n"
                "Process and respond to these messages."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    subscriber_inbox = state.get("subscriber_inbox", {})
    subscriber_inbox["subscriber_2"] = inbox
    
    return {
        "messages": [AIMessage(content=f"Subscriber 2: {response.content}")],
        "subscriptions": subscriptions,
        "subscriber_inbox": subscriber_inbox
    }


# Subscriber 3 - All Topics
def subscriber_3_agent(state: PubSubState):
    """Subscriber interested in all topics."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    subscriptions = state.get("subscriptions", {})
    subscriptions["subscriber_3"] = ["news", "alerts", "updates"]
    
    topics = state.get("topics", {})
    inbox = []
    
    # Receive messages from subscribed topics
    for topic in subscriptions["subscriber_3"]:
        if topic in topics:
            inbox.extend(topics[topic])
    
    system_msg = SystemMessage(
        content=f"You are Subscriber 3. You received messages from all topics:\n"
                f"Inbox: {inbox}\n"
                "Process and respond to these messages."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    subscriber_inbox = state.get("subscriber_inbox", {})
    subscriber_inbox["subscriber_3"] = inbox
    
    return {
        "messages": [AIMessage(content=f"Subscriber 3: {response.content}")],
        "subscriptions": subscriptions,
        "subscriber_inbox": subscriber_inbox
    }


# Build the pub-sub graph
def create_pubsub_graph():
    """Create a publish-subscribe workflow graph."""
    workflow = StateGraph(PubSubState)
    
    # Add nodes
    workflow.add_node("publisher", publisher_agent)
    workflow.add_node("subscriber_1", subscriber_1_agent)
    workflow.add_node("subscriber_2", subscriber_2_agent)
    workflow.add_node("subscriber_3", subscriber_3_agent)
    
    # Pub-Sub flow: Publisher -> All Subscribers in parallel
    workflow.add_edge(START, "publisher")
    workflow.add_edge("publisher", "subscriber_1")
    workflow.add_edge("publisher", "subscriber_2")
    workflow.add_edge("publisher", "subscriber_3")
    workflow.add_edge("subscriber_1", END)
    workflow.add_edge("subscriber_2", END)
    workflow.add_edge("subscriber_3", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_pubsub_graph()
    
    print("=" * 60)
    print("PUBLISH-SUBSCRIBE MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Topic-based messaging
    print("\n[Scenario: Multi-topic News Distribution]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Publish latest updates on AI technology, security alerts, and product releases")],
        "topics": {},
        "subscriptions": {},
        "published_messages": {},
        "subscriber_inbox": {}
    })
    
    print("\n--- Pub-Sub Message Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Published Topics ---")
    for topic, messages in result.get("topics", {}).items():
        print(f"{topic}: {len(messages)} message(s)")
    
    print(f"\n--- Subscriptions ---")
    for subscriber, topics in result.get("subscriptions", {}).items():
        print(f"{subscriber} -> {topics}")
    
    print(f"\n--- Subscriber Inboxes ---")
    for subscriber, inbox in result.get("subscriber_inbox", {}).items():
        print(f"{subscriber}: {len(inbox)} message(s)")
    
    print("\n" + "=" * 60)

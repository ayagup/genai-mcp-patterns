"""
Context Propagation MCP Pattern

This pattern demonstrates propagating context across multiple agent interactions
to maintain coherence and continuity in multi-step workflows.

Key Features:
- Context passing between agents
- Context enrichment at each step
- Maintaining conversation history
- Contextual awareness across workflow
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state with context
class ContextPropagationState(TypedDict):
    """State with propagating context"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    context: dict[str, any]  # Propagating context
    user_request: str
    final_response: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Context Initializer
def context_initializer(state: ContextPropagationState) -> ContextPropagationState:
    """Initializes context with user request"""
    user_request = state.get("user_request", "")
    context = state.get("context", {})
    
    system_message = SystemMessage(content="""You are a context initializer. Extract key 
    information from user request and initialize the context.""")
    
    user_message = HumanMessage(content=f"""Initialize context for request: {user_request}

Extract: user intent, entities, requirements, constraints.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize context
    context.update({
        "user_request": user_request,
        "user_intent": "extracted_from_request",
        "timestamp": "2025-11-09T10:00:00",
        "conversation_id": "conv_12345",
        "step": 1,
        "history": ["Context initialized"]
    })
    
    return {
        "messages": [AIMessage(content=f"🎯 Context Initializer: {response.content}\n\n✅ Context initialized")],
        "context": context
    }


# Authentication Agent
def authentication_agent(state: ContextPropagationState) -> ContextPropagationState:
    """Enriches context with authentication info"""
    context = state.get("context", {})
    
    system_message = SystemMessage(content="""You are an authentication agent. Verify user 
    and add authentication context.""")
    
    user_message = HumanMessage(content=f"""Authenticate user for conversation: {context.get('conversation_id')}

Add authentication details to context.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Enrich context with auth info
    context["authentication"] = {
        "user_id": "user_789",
        "role": "premium_customer",
        "permissions": ["read", "write", "premium_features"],
        "authenticated_at": "2025-11-09T10:01:00"
    }
    context["step"] = 2
    context["history"].append("User authenticated")
    
    return {
        "messages": [AIMessage(content=f"🔐 Authentication Agent: {response.content}\n\n✅ Auth context added")],
        "context": context
    }


# Personalization Agent
def personalization_agent(state: ContextPropagationState) -> ContextPropagationState:
    """Enriches context with user preferences"""
    context = state.get("context", {})
    
    system_message = SystemMessage(content="""You are a personalization agent. Add user 
    preferences and personalization data to context.""")
    
    auth_info = context.get("authentication", {})
    user_message = HumanMessage(content=f"""Add personalization for user: {auth_info.get('user_id')}

Enrich context with preferences and history.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Enrich context with personalization
    context["personalization"] = {
        "language": "English",
        "timezone": "America/New_York",
        "preferences": {
            "communication_style": "concise",
            "detail_level": "moderate"
        },
        "past_interactions": 47,
        "satisfaction_score": 4.5
    }
    context["step"] = 3
    context["history"].append("Personalization applied")
    
    return {
        "messages": [AIMessage(content=f"⭐ Personalization Agent: {response.content}\n\n✅ Personalization context added")],
        "context": context
    }


# Business Logic Agent
def business_logic_agent(state: ContextPropagationState) -> ContextPropagationState:
    """Processes request using full context"""
    context = state.get("context", {})
    
    system_message = SystemMessage(content="""You are a business logic agent. Process the 
    request using all accumulated context.""")
    
    user_request = context.get("user_request", "")
    auth = context.get("authentication", {})
    personalization = context.get("personalization", {})
    
    user_message = HumanMessage(content=f"""Process request: {user_request}

Context available:
- User: {auth.get('user_id')} ({auth.get('role')})
- Language: {personalization.get('language')}
- Preferences: {personalization.get('preferences')}
- Past interactions: {personalization.get('past_interactions')}

Apply business logic with full context awareness.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Enrich context with business logic results
    context["business_result"] = {
        "processed": True,
        "result": response.content,
        "used_premium_features": auth.get("role") == "premium_customer",
        "personalized": True
    }
    context["step"] = 4
    context["history"].append("Business logic executed")
    
    return {
        "messages": [AIMessage(content=f"⚙️ Business Logic Agent: {response.content}\n\n✅ Request processed with full context")],
        "context": context
    }


# Response Formatter
def response_formatter(state: ContextPropagationState) -> ContextPropagationState:
    """Formats final response using propagated context"""
    context = state.get("context", {})
    
    system_message = SystemMessage(content="""You are a response formatter. Format the 
    response according to user preferences from context.""")
    
    business_result = context.get("business_result", {})
    personalization = context.get("personalization", {})
    
    user_message = HumanMessage(content=f"""Format response:

Result: {business_result.get('result')}

User preferences:
- Communication style: {personalization.get('preferences', {}).get('communication_style')}
- Detail level: {personalization.get('preferences', {}).get('detail_level')}
- Language: {personalization.get('language')}

Format appropriately.""")
    
    response = llm.invoke([system_message, user_message])
    
    context["step"] = 5
    context["history"].append("Response formatted")
    
    return {
        "messages": [AIMessage(content=f"✨ Response Formatter: {response.content}")],
        "context": context,
        "final_response": response.content
    }


# Context Monitor
def context_monitor(state: ContextPropagationState) -> ContextPropagationState:
    """Monitors context propagation throughout workflow"""
    context = state.get("context", {})
    
    context_summary = f"""
    ✅ CONTEXT PROPAGATION COMPLETE
    
    Workflow Steps: {context.get('step', 0)}
    
    Context Evolution:
{chr(10).join([f"    {i+1}. {h}" for i, h in enumerate(context.get('history', []))])}
    
    Final Context Contains:
    • User Request: {context.get('user_request', 'N/A')}
    • Authentication: User {context.get('authentication', {}).get('user_id', 'N/A')}
    • Personalization: {context.get('personalization', {}).get('language', 'N/A')} language
    • Business Result: {context.get('business_result', {}).get('processed', False)}
    
    Context Propagation Benefits:
    • Maintained coherence across workflow
    • Each agent enriched context
    • No information loss between steps
    • Personalized experience enabled
    • Full traceability via history
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Context Monitor:\n{context_summary}")]
    }


# Build the graph
def build_context_propagation_graph():
    """Build the context propagation MCP pattern graph"""
    workflow = StateGraph(ContextPropagationState)
    
    workflow.add_node("initializer", context_initializer)
    workflow.add_node("auth", authentication_agent)
    workflow.add_node("personalization", personalization_agent)
    workflow.add_node("business", business_logic_agent)
    workflow.add_node("formatter", response_formatter)
    workflow.add_node("monitor", context_monitor)
    
    # Sequential context propagation
    workflow.add_edge(START, "initializer")
    workflow.add_edge("initializer", "auth")
    workflow.add_edge("auth", "personalization")
    workflow.add_edge("personalization", "business")
    workflow.add_edge("business", "formatter")
    workflow.add_edge("formatter", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_context_propagation_graph()
    
    print("=== Context Propagation MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "context": {},
        "user_request": "I need a detailed analysis of my account activity for the past quarter with recommendations for optimization",
        "final_response": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Context Propagation Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Response ===")
    print(result.get("final_response", "No response generated"))
    
    print(f"\n\n=== Context History ===")
    for step in result["context"].get("history", []):
        print(f"  • {step}")

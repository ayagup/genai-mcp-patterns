"""
Semantic Routing MCP Pattern

This pattern demonstrates routing queries to specialized agents based on
semantic understanding of the query content and intent.

Key Features:
- Intent-based routing
- Semantic query analysis
- Specialized agent selection
- Dynamic routing decisions
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SemanticRoutingState(TypedDict):
    """State for semantic routing"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    detected_intent: str
    routed_to: str
    final_response: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Semantic Router
def semantic_router(state: SemanticRoutingState) -> SemanticRoutingState:
    """Analyzes query semantically and routes to appropriate specialist"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a semantic router. Analyze the query's 
    intent and meaning to route it to the most appropriate specialist agent.
    
    Specialists available:
    - technical: Code, bugs, technical issues
    - billing: Payments, invoices, subscriptions
    - product: Features, functionality, product questions
    - support: General help, how-to, guidance""")
    
    user_message = HumanMessage(content=f"""Analyze and route this query: {query}

Determine the intent and select the best specialist.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Semantic intent detection (simplified - in practice use embeddings/classification)
    query_lower = query.lower()
    detected_intent = ""
    routed_to = ""
    
    if any(word in query_lower for word in ["code", "error", "bug", "api", "python", "deploy"]):
        detected_intent = "technical_assistance"
        routed_to = "technical"
    elif any(word in query_lower for word in ["payment", "bill", "invoice", "subscribe", "charge", "refund"]):
        detected_intent = "billing_inquiry"
        routed_to = "billing"
    elif any(word in query_lower for word in ["feature", "capability", "product", "functionality", "can it"]):
        detected_intent = "product_question"
        routed_to = "product"
    else:
        detected_intent = "general_support"
        routed_to = "support"
    
    return {
        "messages": [AIMessage(content=f"🎯 Semantic Router: {response.content}\n\n📍 Intent: {detected_intent}\n🔀 Routing to: {routed_to}")],
        "detected_intent": detected_intent,
        "routed_to": routed_to
    }


# Route decision
def route_decision(state: SemanticRoutingState) -> str:
    """Returns routing decision for conditional edges"""
    return state.get("routed_to", "support")


# Technical Specialist
def technical_specialist(state: SemanticRoutingState) -> SemanticRoutingState:
    """Handles technical queries"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a technical specialist. Handle code, 
    bugs, API questions, and technical implementation issues.""")
    
    user_message = HumanMessage(content=f"""Provide technical assistance for: {query}""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💻 Technical Specialist: {response.content}")],
        "final_response": response.content
    }


# Billing Specialist
def billing_specialist(state: SemanticRoutingState) -> SemanticRoutingState:
    """Handles billing queries"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a billing specialist. Handle payments, 
    invoices, subscriptions, and billing-related questions.""")
    
    user_message = HumanMessage(content=f"""Provide billing assistance for: {query}""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💳 Billing Specialist: {response.content}")],
        "final_response": response.content
    }


# Product Specialist
def product_specialist(state: SemanticRoutingState) -> SemanticRoutingState:
    """Handles product queries"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a product specialist. Handle questions 
    about features, capabilities, and product functionality.""")
    
    user_message = HumanMessage(content=f"""Provide product information for: {query}""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📦 Product Specialist: {response.content}")],
        "final_response": response.content
    }


# Support Specialist
def support_specialist(state: SemanticRoutingState) -> SemanticRoutingState:
    """Handles general support queries"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a support specialist. Handle general 
    inquiries, how-to questions, and provide guidance.""")
    
    user_message = HumanMessage(content=f"""Provide support for: {query}""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🤝 Support Specialist: {response.content}")],
        "final_response": response.content
    }


# Routing Monitor
def routing_monitor(state: SemanticRoutingState) -> SemanticRoutingState:
    """Monitors routing decisions"""
    detected_intent = state.get("detected_intent", "")
    routed_to = state.get("routed_to", "")
    
    summary = f"""
    ✅ SEMANTIC ROUTING COMPLETE
    
    Query: {state.get('query', '')}
    Detected Intent: {detected_intent}
    Routed To: {routed_to}
    
    Routing Benefits:
    • Semantic understanding of user intent
    • Automatic specialist selection
    • Improved response quality
    • Efficient query handling
    • No manual routing needed
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Routing Monitor:\n{summary}")]
    }


# Build the graph
def build_semantic_routing_graph():
    """Build the semantic routing MCP pattern graph"""
    workflow = StateGraph(SemanticRoutingState)
    
    workflow.add_node("router", semantic_router)
    workflow.add_node("technical", technical_specialist)
    workflow.add_node("billing", billing_specialist)
    workflow.add_node("product", product_specialist)
    workflow.add_node("support", support_specialist)
    workflow.add_node("monitor", routing_monitor)
    
    # Routing logic
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "technical": "technical",
            "billing": "billing",
            "product": "product",
            "support": "support"
        }
    )
    workflow.add_edge("technical", "monitor")
    workflow.add_edge("billing", "monitor")
    workflow.add_edge("product", "monitor")
    workflow.add_edge("support", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_semantic_routing_graph()
    
    print("=== Semantic Routing MCP Pattern ===\n")
    
    # Test different queries
    test_queries = [
        "My API is returning a 500 error when I try to authenticate",
        "I was charged twice for my subscription this month",
        "Does your product support multi-language translations?",
        "How do I reset my password?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}\n")
        
        initial_state = {
            "messages": [],
            "query": query,
            "detected_intent": "",
            "routed_to": "",
            "final_response": ""
        }
        
        result = graph.invoke(initial_state)
        
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)

"""
Request-Response MCP Pattern
=============================
Synchronous request-response communication where an agent sends a request
and waits for a response before continuing.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Optional
import operator


# Define the state
class RequestResponseState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    request: str
    response: Optional[str]
    requester: str
    responder: str
    status: str


# Requester Agent
def requester_agent(state: RequestResponseState):
    """Agent that sends a request and waits for response."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are a Requester Agent. Formulate a clear request "
                "that requires specific information or action from the responder."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Requester: {response.content}")],
        "request": response.content,
        "requester": "requester_agent",
        "status": "request_sent"
    }


# Responder Agent
def responder_agent(state: RequestResponseState):
    """Agent that processes request and sends response."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    request = state.get("request", "No request")
    
    system_msg = SystemMessage(
        content=f"You are a Responder Agent. Process this request and provide a response:\n"
                f"Request: {request}\n"
                "Provide a complete and accurate response."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Responder: {response.content}")],
        "response": response.content,
        "responder": "responder_agent",
        "status": "response_received"
    }


# Request Validator
def validate_response(state: RequestResponseState):
    """Validate that the response matches the request."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    request = state.get("request", "")
    response = state.get("response", "")
    
    system_msg = SystemMessage(
        content=f"Validate if the response adequately addresses the request:\n"
                f"Request: {request}\n"
                f"Response: {response}\n"
                "Confirm if the response is satisfactory."
    )
    
    validation = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Validator: {validation.content}")],
        "status": "validated"
    }


# Build the request-response graph
def create_request_response_graph():
    """Create a request-response workflow graph."""
    workflow = StateGraph(RequestResponseState)
    
    # Add nodes
    workflow.add_node("requester", requester_agent)
    workflow.add_node("responder", responder_agent)
    workflow.add_node("validator", validate_response)
    
    # Request-Response flow: Requester -> Responder -> Validator
    workflow.add_edge(START, "requester")
    workflow.add_edge("requester", "responder")
    workflow.add_edge("responder", "validator")
    workflow.add_edge("validator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = create_request_response_graph()
    
    print("=" * 60)
    print("REQUEST-RESPONSE MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: API-style request-response
    print("\n[Scenario: Database Query Request]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Get all users who registered in the last 30 days")],
        "request": "",
        "response": None,
        "requester": "",
        "responder": "",
        "status": "pending"
    })
    
    print("\n--- Request-Response Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Request Details ---")
    print(f"Requester: {result.get('requester')}")
    print(f"Request: {result.get('request', 'N/A')[:150]}...")
    
    print(f"\n--- Response Details ---")
    print(f"Responder: {result.get('responder')}")
    print(f"Response: {result.get('response', 'N/A')[:150]}...")
    
    print(f"\n--- Status: {result.get('status')} ---")
    
    print("\n" + "=" * 60)

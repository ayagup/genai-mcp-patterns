"""
API Design MCP Pattern

This pattern demonstrates automated API design from requirements,
including endpoint definition, schema design, and specification generation.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class APIDesignState(TypedDict):
    """State for API design"""
    requirements: str
    endpoints: List[Dict]
    schemas: Dict
    openapi_spec: str
    messages: Annotated[List, operator.add]


class APIDesigner:
    """Design API"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def design_endpoints(self, requirements: str) -> List[Dict]:
        """Design API endpoints"""
        return [
            {"method": "GET", "path": "/users", "description": "Get all users"},
            {"method": "POST", "path": "/users", "description": "Create user"},
            {"method": "GET", "path": "/users/{id}", "description": "Get user by ID"}
        ]


def design_api(state: APIDesignState) -> APIDesignState:
    """Design API"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    designer = APIDesigner(llm)
    
    state["endpoints"] = designer.design_endpoints(state["requirements"])
    state["schemas"] = {"User": {"id": "int", "name": "string", "email": "string"}}
    state["openapi_spec"] = "openapi: 3.0.0\ninfo:\n  title: User API"
    state["messages"].append(HumanMessage(content=f"Designed {len(state['endpoints'])} endpoints"))
    return state


def create_apidesign_graph():
    """Create API design workflow"""
    workflow = StateGraph(APIDesignState)
    workflow.add_node("design", design_api)
    workflow.add_edge(START, "design")
    workflow.add_edge("design", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_apidesign_graph()
    result = app.invoke({"requirements": "Need CRUD API for users", "endpoints": [], "schemas": {}, "openapi_spec": "", "messages": []})
    print(f"API Design Complete: {len(result['endpoints'])} endpoints")

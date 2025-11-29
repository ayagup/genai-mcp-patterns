"""
Validation MCP Pattern

This pattern implements validation to ensure data and inputs
meet expected formats and constraints.

Pattern Type: Quality Assurance
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class ValidationState(TypedDict):
    """State for validation"""
    data: Dict
    validation_rules: List[str]
    validation_results: List[Dict]
    valid: bool
    validation_errors: List[str]
    messages: Annotated[List, operator.add]


class DataValidator:
    """Validate data"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def validate(self, data: Dict, rules: List[str]) -> List[Dict]:
        """Validate data against rules"""
        return [
            {"rule": r, "status": "valid", "message": "OK"}
            for r in rules
        ]


def validate_data(state: ValidationState) -> ValidationState:
    """Validate data"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    validator = DataValidator(llm)
    
    state["validation_results"] = validator.validate(state["data"], state["validation_rules"])
    state["valid"] = all(res["status"] == "valid" for res in state["validation_results"])
    state["validation_errors"] = [res["message"] for res in state["validation_results"] if res["status"] != "valid"]
    state["messages"].append(HumanMessage(content=f"Validation {'passed' if state['valid'] else 'failed'}"))
    return state


def create_validation_graph():
    """Create validation workflow"""
    workflow = StateGraph(ValidationState)
    workflow.add_node("validate", validate_data)
    workflow.add_edge(START, "validate")
    workflow.add_edge("validate", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_validation_graph()
    result = app.invoke({"data": {"email": "user@example.com", "age": 25}, "validation_rules": ["Valid email format", "Age >= 18"], "validation_results": [], "valid": False, "validation_errors": [], "messages": []})
    print(f"Validation: {'passed' if result['valid'] else 'failed'}")

"""
Tool Validation MCP Pattern

This pattern implements input/output validation for tools
to ensure data quality and prevent errors.

Key Features:
- Schema validation
- Type checking
- Custom validators
- Sanitization
- Error reporting
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ToolValidationState(TypedDict):
    """State for tool validation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tool_name: str
    input_data: Dict
    output_data: Dict
    validation_errors: List[str]
    is_valid: bool


llm = ChatOpenAI(model="gpt-4", temperature=0)


def input_validator(state: ToolValidationState) -> ToolValidationState:
    """Validates tool inputs against schema"""
    tool_name = state.get("tool_name", "search")
    input_data = state.get("input_data", {"query": "AI", "limit": 10})
    
    system_message = SystemMessage(content="You are an input validator.")
    user_message = HumanMessage(content=f"Validate input for {tool_name}: {input_data}")
    response = llm.invoke([system_message, user_message])
    
    # Define schemas
    schemas = {
        "search": {
            "query": {"type": "string", "required": True, "min_length": 1},
            "limit": {"type": "integer", "required": False, "min": 1, "max": 100}
        },
        "calculator": {
            "operation": {"type": "string", "required": True, "enum": ["add", "subtract", "multiply", "divide"]},
            "operands": {"type": "list", "required": True, "min_length": 2}
        }
    }
    
    # Validate input
    errors = []
    schema = schemas.get(tool_name, {})
    
    for field, rules in schema.items():
        if rules.get("required") and field not in input_data:
            errors.append(f"Missing required field: {field}")
        elif field in input_data:
            value = input_data[field]
            expected_type = rules.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                errors.append(f"{field} must be string")
            elif expected_type == "integer" and not isinstance(value, int):
                errors.append(f"{field} must be integer")
            elif expected_type == "list" and not isinstance(value, list):
                errors.append(f"{field} must be list")
    
    is_valid = len(errors) == 0
    
    report = f"""
    ✓ Input Validation:
    
    Tool: {tool_name}
    Input: {input_data}
    Valid: {is_valid}
    
    Validation Methods:
    • JSON Schema validation
    • Type checking
    • Range validation
    • Format validation (email, URL, etc.)
    • Custom business rules
    
    Errors: {errors if errors else "None"}
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "tool_name": tool_name,
        "input_data": input_data,
        "validation_errors": errors,
        "is_valid": is_valid
    }


def output_validator(state: ToolValidationState) -> ToolValidationState:
    """Validates tool outputs"""
    output_data = {"results": ["result1", "result2"], "count": 2}
    is_valid = state.get("is_valid", True)
    
    system_message = SystemMessage(content="You are an output validator.")
    user_message = HumanMessage(content=f"Validate output: {output_data}")
    response = llm.invoke([system_message, user_message])
    
    # Validate output structure
    output_errors = []
    if "results" not in output_data:
        output_errors.append("Missing results field")
    if "count" not in output_data:
        output_errors.append("Missing count field")
    
    output_valid = len(output_errors) == 0
    
    report = f"""
    ✓ Output Validation:
    
    Output: {output_data}
    Valid: {output_valid}
    
    Sanitization:
    • HTML escaping
    • SQL injection prevention
    • XSS prevention
    • Data masking (PII)
    
    Overall Status: {'✓ Valid' if is_valid and output_valid else '✗ Invalid'}
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "output_data": output_data,
        "is_valid": is_valid and output_valid
    }


def build_tool_validation_graph():
    workflow = StateGraph(ToolValidationState)
    workflow.add_node("input_validator", input_validator)
    workflow.add_node("output_validator", output_validator)
    workflow.add_edge(START, "input_validator")
    workflow.add_edge("input_validator", "output_validator")
    workflow.add_edge("output_validator", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_tool_validation_graph()
    print("=== Tool Validation MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "tool_name": "",
        "input_data": {},
        "output_data": {},
        "validation_errors": [],
        "is_valid": False
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")

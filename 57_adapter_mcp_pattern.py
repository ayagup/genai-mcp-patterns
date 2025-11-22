"""
Adapter MCP Pattern

This pattern demonstrates adapting interfaces between incompatible systems,
allowing agents with different protocols to work together.

Key Features:
- Interface adaptation
- Protocol conversion
- Data format transformation
- Legacy system integration
- Backward compatibility
"""

from typing import TypedDict, Sequence, Annotated
import operator
import json
import xml.etree.ElementTree as ET
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AdapterState(TypedDict):
    """State for adapter pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    legacy_data: str  # Old format (XML)
    modern_data: str  # New format (JSON)
    adapted_data: str  # Converted data
    source_system: str
    target_system: str
    conversion_log: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Legacy System Agent (XML-based)
def legacy_system(state: AdapterState) -> AdapterState:
    """Simulates legacy system with XML output"""
    
    system_message = SystemMessage(content="""You are a legacy system agent. You work 
    with XML format and need adapters to communicate with modern JSON-based systems.""")
    
    user_message = HumanMessage(content="""Generate sample data in XML format:

Create user data with: id, name, email, status

Output in XML format.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate sample XML data
    legacy_data = """<?xml version="1.0"?>
<user>
    <id>12345</id>
    <name>John Doe</name>
    <email>john@example.com</email>
    <status>active</status>
</user>"""
    
    log_entry = "LEGACY_SYSTEM: Generated XML data"
    
    return {
        "messages": [AIMessage(content=f"🏛️ Legacy System: {response.content}\n\n✅ Generated XML data")],
        "legacy_data": legacy_data,
        "source_system": "LegacyXMLSystem",
        "conversion_log": [log_entry]
    }


# Format Detector
def format_detector(state: AdapterState) -> AdapterState:
    """Detects data format"""
    legacy_data = state.get("legacy_data", "")
    
    system_message = SystemMessage(content="""You are a format detector. Identify 
    the data format and structure for proper adaptation.""")
    
    user_message = HumanMessage(content=f"""Detect format:

Data: {legacy_data[:100]}...

Identify format and structure.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Detect format
    is_xml = legacy_data.strip().startswith("<?xml") or legacy_data.strip().startswith("<")
    detected_format = "XML" if is_xml else "Unknown"
    
    log_entry = f"FORMAT_DETECTOR: Detected {detected_format} format"
    
    return {
        "messages": [AIMessage(content=f"🔍 Format Detector: {response.content}\n\n✅ Detected: {detected_format}")],
        "conversion_log": [log_entry]
    }


# XML to JSON Adapter
def xml_to_json_adapter(state: AdapterState) -> AdapterState:
    """Adapts XML format to JSON"""
    legacy_data = state.get("legacy_data", "")
    
    system_message = SystemMessage(content="""You are an XML to JSON adapter. Convert 
    XML data structures to JSON format.""")
    
    user_message = HumanMessage(content=f"""Convert XML to JSON:

XML Data:
{legacy_data}

Perform conversion.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Parse XML and convert to JSON
    try:
        root = ET.fromstring(legacy_data)
        user_data = {}
        for child in root:
            user_data[child.tag] = child.text
        
        adapted_data = json.dumps(user_data, indent=2)
    except Exception as e:
        adapted_data = json.dumps({"error": f"Conversion failed: {str(e)}"})
    
    log_entry = "XML_TO_JSON_ADAPTER: Converted XML to JSON"
    
    return {
        "messages": [AIMessage(content=f"🔄 XML to JSON Adapter: {response.content}\n\n✅ Converted to JSON")],
        "adapted_data": adapted_data,
        "conversion_log": [log_entry]
    }


# Data Validator
def data_validator(state: AdapterState) -> AdapterState:
    """Validates adapted data"""
    adapted_data = state.get("adapted_data", "")
    
    system_message = SystemMessage(content="""You are a data validator. Verify that 
    adapted data maintains integrity and completeness.""")
    
    user_message = HumanMessage(content=f"""Validate adapted data:

Data: {adapted_data}

Verify:
- Valid JSON structure
- Required fields present
- Data integrity""")
    
    response = llm.invoke([system_message, user_message])
    
    # Validate JSON
    try:
        data = json.loads(adapted_data)
        required_fields = ["id", "name", "email", "status"]
        is_valid = all(field in data for field in required_fields)
        validation_status = "VALID" if is_valid else "INVALID"
    except:
        validation_status = "INVALID_JSON"
    
    log_entry = f"DATA_VALIDATOR: Validation {validation_status}"
    
    return {
        "messages": [AIMessage(content=f"✅ Data Validator: {response.content}\n\n✅ Status: {validation_status}")],
        "conversion_log": [log_entry]
    }


# Modern System Agent (JSON-based)
def modern_system(state: AdapterState) -> AdapterState:
    """Simulates modern system that consumes JSON"""
    adapted_data = state.get("adapted_data", "")
    
    system_message = SystemMessage(content="""You are a modern system agent. You work 
    with JSON format and process data from adapted legacy systems.""")
    
    user_message = HumanMessage(content=f"""Process JSON data:

{adapted_data}

Consume and process the data.""")
    
    response = llm.invoke([system_message, user_message])
    
    log_entry = "MODERN_SYSTEM: Successfully consumed JSON data"
    
    return {
        "messages": [AIMessage(content=f"🚀 Modern System: {response.content}\n\n✅ Data processed successfully")],
        "modern_data": adapted_data,
        "target_system": "ModernJSONSystem",
        "conversion_log": [log_entry]
    }


# Adapter Monitor
def adapter_monitor(state: AdapterState) -> AdapterState:
    """Monitors adaptation process"""
    source_system = state.get("source_system", "")
    target_system = state.get("target_system", "")
    legacy_data = state.get("legacy_data", "")
    adapted_data = state.get("adapted_data", "")
    conversion_log = state.get("conversion_log", [])
    
    logs_text = "\n".join([f"  {i+1}. {log}" for i, log in enumerate(conversion_log)])
    
    summary = f"""
    ✅ ADAPTER PATTERN COMPLETE
    
    Adaptation Summary:
    • Source System: {source_system}
    • Target System: {target_system}
    • Source Format: XML
    • Target Format: JSON
    • Conversion Steps: {len(conversion_log)}
    
    Conversion Log:
{logs_text}
    
    Original Data (XML):
{legacy_data[:150]}...
    
    Adapted Data (JSON):
{adapted_data[:150]}...
    
    Adapter Benefits:
    • Seamless integration of incompatible systems
    • Protocol/format conversion
    • Legacy system modernization
    • Backward compatibility
    • Data integrity preservation
    • Flexible interface adaptation
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Adapter Monitor:\n{summary}")]
    }


# Build the graph
def build_adapter_graph():
    """Build the adapter pattern graph"""
    workflow = StateGraph(AdapterState)
    
    workflow.add_node("legacy", legacy_system)
    workflow.add_node("detector", format_detector)
    workflow.add_node("adapter", xml_to_json_adapter)
    workflow.add_node("validator", data_validator)
    workflow.add_node("modern", modern_system)
    workflow.add_node("monitor", adapter_monitor)
    
    workflow.add_edge(START, "legacy")
    workflow.add_edge("legacy", "detector")
    workflow.add_edge("detector", "adapter")
    workflow.add_edge("adapter", "validator")
    workflow.add_edge("validator", "modern")
    workflow.add_edge("modern", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_adapter_graph()
    
    print("=== Adapter MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "legacy_data": "",
        "modern_data": "",
        "adapted_data": "",
        "source_system": "",
        "target_system": "",
        "conversion_log": []
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Adapter Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Conversion Details ===")
    print(f"\nOriginal XML:")
    print(result.get("legacy_data", ""))
    print(f"\nConverted JSON:")
    print(result.get("adapted_data", ""))

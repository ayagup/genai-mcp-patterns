"""
Pattern 220: Message Translator MCP Pattern

Message Translator transforms messages between different formats:
- Converts between data formats (JSON, XML, Protobuf, etc.)
- Adapts message structures
- Enables integration of incompatible systems
- Handles versioning
- Field mapping and transformation

Translation Types:
- Format translation (JSON ↔ XML)
- Structure translation (flat ↔ nested)
- Semantic translation (field renaming)
- Version translation (v1 ↔ v2)
- Protocol translation (REST ↔ gRPC)

Benefits:
- System interoperability
- Backward compatibility
- Gradual migration
- Decoupled systems
- Format flexibility

Use Cases:
- Legacy system integration
- Multi-vendor integration
- API versioning
- Data pipeline transformation
- Protocol bridging
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json


class TranslatorState(TypedDict):
    """State for message translator operations"""
    translator_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class CustomerV1:
    """Legacy customer format (v1)"""
    cust_id: int
    name: str
    email: str
    phone: str


@dataclass
class CustomerV2:
    """Modern customer format (v2)"""
    id: str
    full_name: str
    contact: Dict[str, str]
    metadata: Dict[str, Any]


class MessageTranslator:
    """
    Message Translator that converts between different formats
    """
    
    def __init__(self):
        self.translations = 0
        self.reverse_translations = 0
    
    def translate_v1_to_v2(self, v1: CustomerV1) -> CustomerV2:
        """Translate v1 customer to v2"""
        self.translations += 1
        
        return CustomerV2(
            id=f"CUST-{v1.cust_id:06d}",
            full_name=v1.name,
            contact={
                'email': v1.email,
                'phone': v1.phone
            },
            metadata={
                'version': 'v2',
                'migrated_from': 'v1'
            }
        )
    
    def translate_v2_to_v1(self, v2: CustomerV2) -> CustomerV1:
        """Translate v2 customer to v1 (backward compatibility)"""
        self.reverse_translations += 1
        
        # Extract ID number
        cust_id = int(v2.id.split('-')[1])
        
        return CustomerV1(
            cust_id=cust_id,
            name=v2.full_name,
            email=v2.contact.get('email', ''),
            phone=v2.contact.get('phone', '')
        )
    
    def json_to_xml(self, json_data: Dict[str, Any]) -> str:
        """Convert JSON to XML"""
        def dict_to_xml(d: Dict, root: str = "root") -> str:
            xml = f"<{root}>"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml += dict_to_xml(value, key)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            xml += dict_to_xml(item, key)
                        else:
                            xml += f"<{key}>{item}</{key}>"
                else:
                    xml += f"<{key}>{value}</{key}>"
            xml += f"</{root}>"
            return xml
        
        return dict_to_xml(json_data)
    
    def flatten_structure(self, nested: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested structure"""
        flat = {}
        
        def flatten(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_key = f"{prefix}_{key}" if prefix else key
                    flatten(value, new_key)
            else:
                flat[prefix] = obj
        
        flatten(nested)
        return flat


def setup_translator_agent(state: TranslatorState):
    """Agent to set up translator"""
    operations = []
    results = []
    
    translator = MessageTranslator()
    
    operations.append("Message Translator Setup:")
    operations.append("\nSupported Translations:")
    operations.append("  1. CustomerV1 ↔ CustomerV2 (version migration)")
    operations.append("  2. JSON → XML (format conversion)")
    operations.append("  3. Nested → Flat (structure transformation)")
    
    results.append("✓ Translator initialized")
    
    state['_translator'] = translator
    
    return {
        "translator_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def version_translation_agent(state: TranslatorState):
    """Agent to demonstrate version translation"""
    translator = state['_translator']
    operations = []
    results = []
    
    operations.append("\n🔄 Version Translation Demo:")
    
    # V1 → V2
    v1_customer = CustomerV1(
        cust_id=123,
        name="John Doe",
        email="john@example.com",
        phone="555-1234"
    )
    
    operations.append("\nOriginal (V1):")
    operations.append(f"  cust_id: {v1_customer.cust_id}")
    operations.append(f"  name: {v1_customer.name}")
    operations.append(f"  email: {v1_customer.email}")
    operations.append(f"  phone: {v1_customer.phone}")
    
    v2_customer = translator.translate_v1_to_v2(v1_customer)
    
    operations.append("\nTranslated (V2):")
    operations.append(f"  id: {v2_customer.id}")
    operations.append(f"  full_name: {v2_customer.full_name}")
    operations.append(f"  contact: {v2_customer.contact}")
    operations.append(f"  metadata: {v2_customer.metadata}")
    
    # V2 → V1 (backward compatibility)
    v1_again = translator.translate_v2_to_v1(v2_customer)
    
    operations.append("\nReverse Translated (V2 → V1):")
    operations.append(f"  cust_id: {v1_again.cust_id}")
    operations.append(f"  name: {v1_again.name}")
    
    results.append("✓ Bi-directional version translation working")
    
    return {
        "translator_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Version translation complete"]
    }


def format_translation_agent(state: TranslatorState):
    """Agent to demonstrate format translation"""
    translator = state['_translator']
    operations = []
    results = []
    
    operations.append("\n🔤 Format Translation Demo:")
    
    # JSON → XML
    json_data = {
        "order": {
            "id": "ORD-123",
            "customer": "John Doe",
            "items": [
                {"sku": "ITEM-1", "qty": 2},
                {"sku": "ITEM-2", "qty": 1}
            ],
            "total": 150.00
        }
    }
    
    operations.append("\nOriginal (JSON):")
    operations.append(f"  {json.dumps(json_data, indent=2)[:100]}...")
    
    xml_data = translator.json_to_xml(json_data)
    
    operations.append("\nTranslated (XML):")
    operations.append(f"  {xml_data[:150]}...")
    
    results.append("✓ Format translation (JSON → XML) successful")
    
    return {
        "translator_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Format translation complete"]
    }


def structure_translation_agent(state: TranslatorState):
    """Agent to demonstrate structure translation"""
    translator = state['_translator']
    operations = []
    results = []
    
    operations.append("\n🏗️ Structure Translation Demo:")
    
    # Nested → Flat
    nested = {
        "user": {
            "id": 123,
            "profile": {
                "name": "John Doe",
                "age": 30
            },
            "address": {
                "city": "NYC",
                "state": "NY"
            }
        }
    }
    
    operations.append("\nOriginal (Nested):")
    operations.append(f"  {json.dumps(nested, indent=2)}")
    
    flat = translator.flatten_structure(nested)
    
    operations.append("\nTranslated (Flat):")
    for key, value in flat.items():
        operations.append(f"  {key}: {value}")
    
    results.append("✓ Structure translation (Nested → Flat) successful")
    
    return {
        "translator_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Structure translation complete"]
    }


def statistics_agent(state: TranslatorState):
    """Agent to show statistics"""
    translator = state['_translator']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("TRANSLATOR STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nVersion translations (V1→V2): {translator.translations}")
    operations.append(f"Reverse translations (V2→V1): {translator.reverse_translations}")
    
    metrics.append("\n📊 Message Translator Benefits:")
    metrics.append("  ✓ System interoperability")
    metrics.append("  ✓ Format flexibility")
    metrics.append("  ✓ Backward compatibility")
    metrics.append("  ✓ Gradual migration")
    metrics.append("  ✓ Decoupled systems")
    
    results.append("✓ Message Translator demonstrated")
    
    return {
        "translator_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_translator_graph():
    """Create the translator workflow graph"""
    workflow = StateGraph(TranslatorState)
    
    workflow.add_node("setup", setup_translator_agent)
    workflow.add_node("version", version_translation_agent)
    workflow.add_node("format", format_translation_agent)
    workflow.add_node("structure", structure_translation_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "version")
    workflow.add_edge("version", "format")
    workflow.add_edge("format", "structure")
    workflow.add_edge("structure", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 220: Message Translator MCP Pattern")
    print("=" * 80)
    
    app = create_translator_graph()
    initial_state = {
        "translator_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    
    for op in final_state["translator_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Message Translator: Transform between formats

Translation Types:
1. Version: V1 ↔ V2
2. Format: JSON ↔ XML ↔ Protobuf
3. Structure: Nested ↔ Flat
4. Semantic: Field renaming/mapping
5. Protocol: REST ↔ gRPC

Benefits:
✓ System interoperability
✓ Backward compatibility
✓ Gradual migration
✓ Format flexibility
✓ Decoupled systems

Real-World:
- Apache Camel transformations
- AWS Step Functions transforms
- ETL pipelines
- API versioning layers
- Protocol adapters

Integration Patterns (211-220) Complete! 🎉
All 10 patterns demonstrate microservices integration:
✓ API Gateway (single entry point)
✓ Service Mesh (infrastructure layer)
✓ Sidecar (helper containers)
✓ Ambassador (proxy pattern)
✓ Anti-Corruption Layer (legacy isolation)
✓ BFF (client-specific backends)
✓ Aggregator (combine calls)
✓ Scatter-Gather (fan-out/in)
✓ Content Router (route by content)
✓ Message Translator (format conversion)
""")


if __name__ == "__main__":
    main()

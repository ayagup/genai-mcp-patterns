"""
Pattern 301: Data Transformation MCP Pattern

This pattern demonstrates transforming data from one structure or format
to another while preserving semantic meaning.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json


class DataTransformationPattern(TypedDict):
    """State for data transformation"""
    messages: Annotated[List[str], add]
    source_data: Dict[str, Any]
    target_schema: Dict[str, Any]
    transformation_rules: List[Dict[str, Any]]
    transformed_data: Dict[str, Any]
    transformation_statistics: Dict[str, Any]


class DataTransformer:
    """Data transformation engine"""
    
    def __init__(self):
        self.transformations = []
        self.field_mappings = {}
        self.type_conversions = {}
    
    def add_field_mapping(self, source_field: str, target_field: str, transform_fn=None):
        """Add a field mapping"""
        self.field_mappings[source_field] = {
            "target": target_field,
            "transform": transform_fn or (lambda x: x)
        }
    
    def add_type_conversion(self, field: str, target_type: str, converter=None):
        """Add type conversion"""
        self.type_conversions[field] = {
            "type": target_type,
            "converter": converter or str
        }
    
    def transform(self, source_data: dict, schema: dict = None) -> dict:
        """Transform source data"""
        result = {}
        
        # Apply field mappings
        for source_field, mapping in self.field_mappings.items():
            if source_field in source_data:
                value = source_data[source_field]
                transform_fn = mapping["transform"]
                target_field = mapping["target"]
                
                # Apply transformation
                transformed_value = transform_fn(value)
                result[target_field] = transformed_value
        
        # Apply type conversions
        for field, conversion in self.type_conversions.items():
            if field in result:
                converter = conversion["converter"]
                result[field] = converter(result[field])
        
        return result
    
    def validate_schema(self, data: dict, schema: dict) -> dict:
        """Validate data against schema"""
        errors = []
        
        for field, field_schema in schema.items():
            if field_schema.get("required", False) and field not in data:
                errors.append(f"Missing required field: {field}")
            
            if field in data:
                expected_type = field_schema.get("type")
                actual_type = type(data[field]).__name__
                
                if expected_type and actual_type != expected_type:
                    errors.append(f"Type mismatch for {field}: expected {expected_type}, got {actual_type}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


def initialize_transformation_agent(state: DataTransformationPattern) -> DataTransformationPattern:
    """Initialize data transformation system"""
    print("\n🔄 Initializing Data Transformation System...")
    
    # Source data (e.g., from legacy system)
    source_data = {
        "user_id": "12345",
        "first_name": "John",
        "last_name": "Doe",
        "email_addr": "john.doe@example.com",
        "phone_num": "555-0123",
        "birth_date": "1990-01-15",
        "account_balance": "1250.50",
        "is_active": "true",
        "signup_timestamp": "2024-01-01T10:30:00Z",
        "preferences": '{"theme": "dark", "notifications": true}'
    }
    
    # Target schema
    target_schema = {
        "id": {"type": "str", "required": True},
        "fullName": {"type": "str", "required": True},
        "email": {"type": "str", "required": True},
        "phone": {"type": "str", "required": False},
        "dateOfBirth": {"type": "str", "required": False},
        "balance": {"type": "float", "required": True},
        "active": {"type": "bool", "required": True},
        "createdAt": {"type": "str", "required": True},
        "settings": {"type": "dict", "required": False}
    }
    
    print(f"  Source Fields: {len(source_data)}")
    print(f"  Target Schema Fields: {len(target_schema)}")
    
    print(f"\n  Sample Source Data:")
    for key, value in list(source_data.items())[:3]:
        print(f"    {key}: {value}")
    
    return {
        **state,
        "source_data": source_data,
        "target_schema": target_schema,
        "transformation_rules": [],
        "transformed_data": {},
        "transformation_statistics": {},
        "messages": ["✓ Transformation system initialized"]
    }


def define_transformation_rules_agent(state: DataTransformationPattern) -> DataTransformationPattern:
    """Define transformation rules"""
    print("\n📋 Defining Transformation Rules...")
    
    transformer = DataTransformer()
    rules = []
    
    # Rule 1: Simple field rename
    transformer.add_field_mapping("user_id", "id")
    rules.append({
        "rule_id": "R1",
        "type": "field_mapping",
        "source": "user_id",
        "target": "id",
        "description": "Map user_id to id"
    })
    print(f"  Rule R1: user_id → id (simple mapping)")
    
    # Rule 2: Combine fields
    transformer.add_field_mapping(
        "first_name",
        "fullName",
        lambda fn: state["source_data"]["first_name"] + " " + state["source_data"]["last_name"]
    )
    rules.append({
        "rule_id": "R2",
        "type": "field_combination",
        "sources": ["first_name", "last_name"],
        "target": "fullName",
        "description": "Combine first_name and last_name"
    })
    print(f"  Rule R2: first_name + last_name → fullName (combination)")
    
    # Rule 3: Field rename
    transformer.add_field_mapping("email_addr", "email")
    rules.append({
        "rule_id": "R3",
        "type": "field_mapping",
        "source": "email_addr",
        "target": "email",
        "description": "Rename email_addr to email"
    })
    print(f"  Rule R3: email_addr → email (rename)")
    
    # Rule 4: Field rename
    transformer.add_field_mapping("phone_num", "phone")
    rules.append({
        "rule_id": "R4",
        "type": "field_mapping",
        "source": "phone_num",
        "target": "phone"
    })
    print(f"  Rule R4: phone_num → phone (rename)")
    
    # Rule 5: Field rename
    transformer.add_field_mapping("birth_date", "dateOfBirth")
    rules.append({
        "rule_id": "R5",
        "type": "field_mapping",
        "source": "birth_date",
        "target": "dateOfBirth"
    })
    print(f"  Rule R5: birth_date → dateOfBirth (rename)")
    
    # Rule 6: Type conversion (string to float)
    transformer.add_field_mapping("account_balance", "balance", lambda x: float(x))
    transformer.add_type_conversion("balance", "float", float)
    rules.append({
        "rule_id": "R6",
        "type": "type_conversion",
        "source": "account_balance",
        "target": "balance",
        "from_type": "str",
        "to_type": "float"
    })
    print(f"  Rule R6: account_balance → balance (str → float)")
    
    # Rule 7: Type conversion (string to bool)
    transformer.add_field_mapping(
        "is_active",
        "active",
        lambda x: x.lower() == "true" if isinstance(x, str) else bool(x)
    )
    transformer.add_type_conversion("active", "bool", bool)
    rules.append({
        "rule_id": "R7",
        "type": "type_conversion",
        "source": "is_active",
        "target": "active",
        "from_type": "str",
        "to_type": "bool"
    })
    print(f"  Rule R7: is_active → active (str → bool)")
    
    # Rule 8: Field rename
    transformer.add_field_mapping("signup_timestamp", "createdAt")
    rules.append({
        "rule_id": "R8",
        "type": "field_mapping",
        "source": "signup_timestamp",
        "target": "createdAt"
    })
    print(f"  Rule R8: signup_timestamp → createdAt (rename)")
    
    # Rule 9: Parse JSON string
    transformer.add_field_mapping(
        "preferences",
        "settings",
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    rules.append({
        "rule_id": "R9",
        "type": "parsing",
        "source": "preferences",
        "target": "settings",
        "description": "Parse JSON string"
    })
    print(f"  Rule R9: preferences → settings (JSON parse)")
    
    print(f"\n  Total Rules: {len(rules)}")
    
    return {
        **state,
        "transformation_rules": rules,
        "messages": [f"✓ Defined {len(rules)} transformation rules"]
    }


def execute_transformation_agent(state: DataTransformationPattern) -> DataTransformationPattern:
    """Execute data transformation"""
    print("\n⚙️ Executing Data Transformation...")
    
    transformer = DataTransformer()
    
    # Rebuild transformer from rules
    source_data = state["source_data"]
    
    # Add all mappings
    transformer.add_field_mapping("user_id", "id")
    transformer.add_field_mapping(
        "first_name",
        "fullName",
        lambda fn: source_data["first_name"] + " " + source_data["last_name"]
    )
    transformer.add_field_mapping("email_addr", "email")
    transformer.add_field_mapping("phone_num", "phone")
    transformer.add_field_mapping("birth_date", "dateOfBirth")
    transformer.add_field_mapping("account_balance", "balance", lambda x: float(x))
    transformer.add_field_mapping(
        "is_active",
        "active",
        lambda x: x.lower() == "true" if isinstance(x, str) else bool(x)
    )
    transformer.add_field_mapping("signup_timestamp", "createdAt")
    transformer.add_field_mapping(
        "preferences",
        "settings",
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    
    # Execute transformation
    transformed_data = transformer.transform(source_data, state["target_schema"])
    
    print(f"  Fields Transformed: {len(transformed_data)}")
    
    print(f"\n  Transformed Data:")
    for key, value in transformed_data.items():
        value_str = str(value)[:50]
        print(f"    {key}: {value_str} ({type(value).__name__})")
    
    return {
        **state,
        "transformed_data": transformed_data,
        "messages": ["✓ Data transformation executed"]
    }


def validate_transformation_agent(state: DataTransformationPattern) -> DataTransformationPattern:
    """Validate transformed data"""
    print("\n✅ Validating Transformed Data...")
    
    transformer = DataTransformer()
    
    # Validate against schema
    validation_result = transformer.validate_schema(
        state["transformed_data"],
        state["target_schema"]
    )
    
    if validation_result["valid"]:
        print(f"  ✓ Validation: PASSED")
        print(f"  All required fields present")
        print(f"  All type constraints satisfied")
    else:
        print(f"  ✗ Validation: FAILED")
        for error in validation_result["errors"]:
            print(f"    • {error}")
    
    # Calculate statistics
    source_fields = len(state["source_data"])
    target_fields = len(state["transformed_data"])
    rules_applied = len(state["transformation_rules"])
    
    statistics = {
        "source_fields": source_fields,
        "target_fields": target_fields,
        "rules_applied": rules_applied,
        "validation_passed": validation_result["valid"],
        "validation_errors": len(validation_result.get("errors", [])),
        "field_reduction": source_fields - target_fields,
        "transformation_ratio": target_fields / max(source_fields, 1)
    }
    
    print(f"\n  Statistics:")
    print(f"    Source Fields: {statistics['source_fields']}")
    print(f"    Target Fields: {statistics['target_fields']}")
    print(f"    Rules Applied: {statistics['rules_applied']}")
    print(f"    Transformation Ratio: {statistics['transformation_ratio']:.2f}")
    
    return {
        **state,
        "transformation_statistics": statistics,
        "messages": ["✓ Transformation validated"]
    }


def generate_transformation_report_agent(state: DataTransformationPattern) -> DataTransformationPattern:
    """Generate transformation report"""
    print("\n" + "="*70)
    print("DATA TRANSFORMATION REPORT")
    print("="*70)
    
    print(f"\n📥 Source Data:")
    for key, value in list(state["source_data"].items())[:5]:
        print(f"  {key}: {value}")
    print(f"  ... ({len(state['source_data'])} total fields)")
    
    print(f"\n📋 Transformation Rules:")
    for rule in state["transformation_rules"]:
        print(f"\n  {rule['rule_id']}: {rule['type']}")
        if "source" in rule:
            print(f"    Source: {rule['source']}")
        if "sources" in rule:
            print(f"    Sources: {', '.join(rule['sources'])}")
        print(f"    Target: {rule['target']}")
        if "description" in rule:
            print(f"    Description: {rule['description']}")
    
    print(f"\n📤 Transformed Data:")
    for key, value in state["transformed_data"].items():
        value_str = str(value)[:60]
        print(f"  {key}: {value_str} ({type(value).__name__})")
    
    print(f"\n📊 Transformation Statistics:")
    stats = state["transformation_statistics"]
    if stats:
        print(f"  Source Fields: {stats['source_fields']}")
        print(f"  Target Fields: {stats['target_fields']}")
        print(f"  Rules Applied: {stats['rules_applied']}")
        print(f"  Validation: {'✓ PASSED' if stats['validation_passed'] else '✗ FAILED'}")
        print(f"  Transformation Ratio: {stats['transformation_ratio']:.2f}")
    
    print(f"\n💡 Data Transformation Benefits:")
    print("  ✓ Schema migration")
    print("  ✓ API integration")
    print("  ✓ Legacy system modernization")
    print("  ✓ Data standardization")
    print("  ✓ Type safety")
    print("  ✓ Field normalization")
    
    print(f"\n🔧 Transformation Types:")
    print("  • Field mapping (rename)")
    print("  • Field combination (merge)")
    print("  • Type conversion (cast)")
    print("  • Parsing (JSON, XML)")
    print("  • Splitting (decompose)")
    print("  • Enrichment (add data)")
    
    print(f"\n⚙️ Use Cases:")
    print("  • ETL pipelines")
    print("  • API versioning")
    print("  • Database migration")
    print("  • Data integration")
    print("  • System modernization")
    print("  • Format conversion")
    
    print(f"\n🎯 Common Transformations:")
    print("  • Flatten nested objects")
    print("  • Normalize field names")
    print("  • Convert data types")
    print("  • Combine/split fields")
    print("  • Format dates/numbers")
    print("  • Parse/stringify JSON")
    
    print("\n" + "="*70)
    print("✅ Data Transformation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_data_transformation_graph():
    """Create data transformation workflow"""
    workflow = StateGraph(DataTransformationPattern)
    
    workflow.add_node("initialize", initialize_transformation_agent)
    workflow.add_node("define_rules", define_transformation_rules_agent)
    workflow.add_node("execute", execute_transformation_agent)
    workflow.add_node("validate", validate_transformation_agent)
    workflow.add_node("report", generate_transformation_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "execute")
    workflow.add_edge("execute", "validate")
    workflow.add_edge("validate", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 301: Data Transformation MCP Pattern")
    print("="*70)
    
    app = create_data_transformation_graph()
    final_state = app.invoke({
        "messages": [],
        "source_data": {},
        "target_schema": {},
        "transformation_rules": [],
        "transformed_data": {},
        "transformation_statistics": {}
    })
    
    print("\n✅ Data Transformation Pattern Complete!")


if __name__ == "__main__":
    main()

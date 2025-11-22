"""
Pattern 188: Data Transformation MCP Pattern

This pattern demonstrates converting data from one format, structure, or representation
to another. Transformation is central to data integration, ensuring data is in the
correct format for downstream processing and analysis.

Key Concepts:
1. Format Conversion: Change data format (JSON → CSV, XML → JSON)
2. Structure Mapping: Map source fields to target schema
3. Type Conversion: Change data types (string → int, date parsing)
4. Normalization: Convert to standard form
5. Denormalization: Flatten nested structures
6. Field Derivation: Compute new fields from existing
7. Data Reshaping: Pivot, unpivot, transpose

Transformation Types:
- Schema Transformation: Change structure/schema
- Value Transformation: Modify values (uppercase, truncate)
- Type Transformation: Convert data types
- Format Transformation: Change format (JSON, XML, CSV)
- Structural Transformation: Reshape data (pivot, flatten)
- Semantic Transformation: Change meaning (currency conversion)

Common Transformations:
- Map: Transform each record independently
- FlatMap: Transform one record to multiple
- Pivot: Rows to columns
- Unpivot: Columns to rows
- Join: Combine multiple sources
- Split: Separate compound fields

Benefits:
- Integration: Enable data from different sources to work together
- Standardization: Consistent format across systems
- Compatibility: Match target system requirements
- Optimization: Transform for performance (denormalization)
- Usability: Make data easier to work with

Trade-offs:
- Data Loss: Some transformations lose information
- Complexity: Complex mappings hard to maintain
- Performance: Transformations add processing time
- Errors: Transformation logic can have bugs
- Reversibility: Some transformations can't be reversed

Use Cases:
- ETL Pipelines: Transform data for warehouse
- API Integration: Convert API responses to internal format
- Data Migration: Transform for new system
- Report Generation: Transform data for presentation
- Data Normalization: Convert to standard formats
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json

# Define the state for data transformation
class DataTransformationState(TypedDict):
    """State for data transformation"""
    input_data: List[Dict[str, Any]]
    type_transformed: Optional[List[Dict[str, Any]]]
    structure_transformed: Optional[List[Dict[str, Any]]]
    value_transformed: Optional[List[Dict[str, Any]]]
    schema_mapped: Optional[List[Dict[str, Any]]]
    transformation_metadata: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# TRANSFORMATION ENGINES
# ============================================================================

class TypeTransformer:
    """
    Type Transformation: Convert data types
    
    Examples:
    - String to Integer: "123" → 123
    - String to Date: "2024-01-15" → datetime
    - Float to Integer: 3.14 → 3
    - Boolean conversion: "true" → True
    """
    
    def transform_types(self, data: List[Dict[str, Any]], 
                       type_mappings: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Transform field types according to mappings
        
        type_mappings: {"field_name": "target_type"}
        target_type: "int", "float", "str", "bool", "date"
        """
        transformed = []
        
        for record in data:
            transformed_record = record.copy()
            
            for field, target_type in type_mappings.items():
                if field in record:
                    value = record[field]
                    
                    try:
                        if target_type == "int":
                            transformed_record[field] = int(value)
                        elif target_type == "float":
                            transformed_record[field] = float(value)
                        elif target_type == "str":
                            transformed_record[field] = str(value)
                        elif target_type == "bool":
                            transformed_record[field] = bool(value)
                        elif target_type == "date":
                            transformed_record[field] = datetime.fromisoformat(str(value))
                    except:
                        # Keep original on error
                        pass
            
            transformed.append(transformed_record)
        
        return transformed

class StructureTransformer:
    """
    Structure Transformation: Change data structure
    
    Operations:
    - Flatten: Nested dict → flat dict
    - Nest: Flat dict → nested dict
    - Pivot: Rows → Columns
    - Unpivot: Columns → Rows
    """
    
    def flatten(self, data: List[Dict[str, Any]], 
                separator: str = "_") -> List[Dict[str, Any]]:
        """
        Flatten nested dictionaries
        
        {"user": {"name": "John", "age": 30}} 
        → {"user_name": "John", "user_age": 30}
        """
        def flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
            items = []
            
            for k, v in d.items():
                new_key = f"{parent_key}{separator}{k}" if parent_key else k
                
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            
            return dict(items)
        
        return [flatten_dict(record) for record in data]
    
    def nest(self, data: List[Dict[str, Any]], 
            field_groups: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Nest fields into sub-dictionaries
        
        field_groups: {"user": ["name", "age"], "address": ["city", "zip"]}
        """
        nested = []
        
        for record in data:
            nested_record = {}
            
            # Create nested groups
            for group_name, fields in field_groups.items():
                nested_record[group_name] = {}
                
                for field in fields:
                    if field in record:
                        nested_record[group_name][field] = record[field]
            
            # Add remaining fields
            for key, value in record.items():
                # Skip if already nested
                is_nested = any(key in fields for fields in field_groups.values())
                if not is_nested:
                    nested_record[key] = value
            
            nested.append(nested_record)
        
        return nested
    
    def pivot(self, data: List[Dict[str, Any]], 
             index_field: str,
             column_field: str,
             value_field: str) -> List[Dict[str, Any]]:
        """
        Pivot: Rows to columns
        
        Transform long format to wide format
        """
        # Group by index
        pivot_data = {}
        
        for record in data:
            index_val = record.get(index_field)
            column_val = record.get(column_field)
            value_val = record.get(value_field)
            
            if index_val not in pivot_data:
                pivot_data[index_val] = {index_field: index_val}
            
            pivot_data[index_val][column_val] = value_val
        
        return list(pivot_data.values())

class ValueTransformer:
    """
    Value Transformation: Modify field values
    
    Operations:
    - Uppercase/Lowercase
    - Trim whitespace
    - Replace values
    - Apply functions
    """
    
    def transform_values(self, data: List[Dict[str, Any]],
                        transformations: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """
        Transform field values using functions
        
        transformations: {"field": lambda x: transformation_function(x)}
        """
        transformed = []
        
        for record in data:
            transformed_record = record.copy()
            
            for field, transform_fn in transformations.items():
                if field in record:
                    try:
                        transformed_record[field] = transform_fn(record[field])
                    except:
                        # Keep original on error
                        pass
            
            transformed.append(transformed_record)
        
        return transformed
    
    def normalize_strings(self, data: List[Dict[str, Any]], 
                         fields: List[str]) -> List[Dict[str, Any]]:
        """Normalize string fields (trim, uppercase)"""
        transformations = {
            field: lambda x: str(x).strip().upper()
            for field in fields
        }
        
        return self.transform_values(data, transformations)
    
    def replace_values(self, data: List[Dict[str, Any]],
                      field: str,
                      replacements: Dict[Any, Any]) -> List[Dict[str, Any]]:
        """Replace values based on mapping"""
        def replace_fn(value):
            return replacements.get(value, value)
        
        return self.transform_values(data, {field: replace_fn})

class SchemaMapper:
    """
    Schema Mapping: Map source schema to target schema
    
    Example:
    Source: {"first_name": "John", "last_name": "Doe"}
    Target: {"name": "John Doe"}
    """
    
    def map_schema(self, data: List[Dict[str, Any]],
                  field_mappings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Map source fields to target schema
        
        field_mappings can be:
        - String: Simple rename {"source_field": "target_field"}
        - Function: Computed field {"target": lambda record: compute(record)}
        - List: Combine fields {"name": ["first_name", "last_name"]}
        """
        mapped = []
        
        for record in data:
            mapped_record = {}
            
            for target_field, mapping in field_mappings.items():
                if isinstance(mapping, str):
                    # Simple rename
                    if mapping in record:
                        mapped_record[target_field] = record[mapping]
                
                elif isinstance(mapping, list):
                    # Combine multiple fields
                    values = [str(record.get(f, "")) for f in mapping]
                    mapped_record[target_field] = " ".join(values).strip()
                
                elif callable(mapping):
                    # Computed field
                    try:
                        mapped_record[target_field] = mapping(record)
                    except:
                        pass
            
            mapped.append(mapped_record)
        
        return mapped

class DataReshaper:
    """
    Data Reshaping: Transform data shape
    
    Operations:
    - Transpose: Swap rows and columns
    - Split: One record → multiple records
    - Merge: Multiple records → one record
    """
    
    def split_field(self, data: List[Dict[str, Any]],
                   field: str,
                   delimiter: str = ",") -> List[Dict[str, Any]]:
        """
        Split field value into multiple records
        
        {"id": 1, "tags": "a,b,c"} 
        → [{"id": 1, "tag": "a"}, {"id": 1, "tag": "b"}, {"id": 1, "tag": "c"}]
        """
        result = []
        
        for record in data:
            if field in record:
                values = str(record[field]).split(delimiter)
                
                for value in values:
                    new_record = record.copy()
                    new_record[field] = value.strip()
                    result.append(new_record)
            else:
                result.append(record)
        
        return result

# Create transformer instances
type_transformer = TypeTransformer()
structure_transformer = StructureTransformer()
value_transformer = ValueTransformer()
schema_mapper = SchemaMapper()
data_reshaper = DataReshaper()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def transform_types(state: DataTransformationState) -> DataTransformationState:
    """Transform data types"""
    input_data = state["input_data"]
    
    # Convert string prices to floats, string IDs to ints
    type_mappings = {
        "price": "float",
        "id": "int",
        "in_stock": "bool"
    }
    
    transformed = type_transformer.transform_types(input_data, type_mappings)
    
    return {
        "type_transformed": transformed,
        "messages": [f"[Type Transform] Converted types for {len(type_mappings)} fields"]
    }

def transform_structure(state: DataTransformationState) -> DataTransformationState:
    """Transform data structure"""
    type_transformed = state.get("type_transformed", [])
    
    # Flatten nested structures
    transformed = structure_transformer.flatten(type_transformed)
    
    return {
        "structure_transformed": transformed,
        "messages": [f"[Structure Transform] Flattened {len(transformed)} records"]
    }

def transform_values(state: DataTransformationState) -> DataTransformationState:
    """Transform field values"""
    structure_transformed = state.get("structure_transformed", [])
    
    # Normalize string fields
    transformed = value_transformer.normalize_strings(
        structure_transformed,
        ["name", "category"]
    )
    
    return {
        "value_transformed": transformed,
        "messages": [f"[Value Transform] Normalized string fields"]
    }

def map_to_schema(state: DataTransformationState) -> DataTransformationState:
    """Map to target schema"""
    value_transformed = state.get("value_transformed", [])
    
    # Map to target schema
    field_mappings = {
        "product_id": "id",
        "product_name": "name",
        "product_price": "price",
        "product_category": "category"
    }
    
    mapped = schema_mapper.map_schema(value_transformed, field_mappings)
    
    return {
        "schema_mapped": mapped,
        "transformation_metadata": {
            "total_records": len(mapped),
            "transformations_applied": ["type", "structure", "value", "schema"]
        },
        "messages": [f"[Schema Mapping] Mapped to target schema with {len(field_mappings)} fields"]
    }

# ============================================================================
# BUILD THE TRANSFORMATION GRAPH
# ============================================================================

def create_transformation_graph():
    """
    Create a StateGraph demonstrating data transformation pattern.
    
    Flow:
    1. Type Transform: Convert data types
    2. Structure Transform: Reshape structure (flatten)
    3. Value Transform: Normalize values
    4. Schema Mapping: Map to target schema
    """
    
    workflow = StateGraph(DataTransformationState)
    
    # Add transformation nodes
    workflow.add_node("types", transform_types)
    workflow.add_node("structure", transform_structure)
    workflow.add_node("values", transform_values)
    workflow.add_node("schema", map_to_schema)
    
    # Define transformation flow
    workflow.add_edge(START, "types")
    workflow.add_edge("types", "structure")
    workflow.add_edge("structure", "values")
    workflow.add_edge("values", "schema")
    workflow.add_edge("schema", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Transformation MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Multi-Stage Transformation
    print("\n" + "=" * 80)
    print("Example 1: Multi-Stage Data Transformation Pipeline")
    print("=" * 80)
    
    transformation_graph = create_transformation_graph()
    
    # Sample data (messy format)
    input_data = [
        {"id": "1", "name": "  laptop  ", "price": "999.99", "category": "electronics", "in_stock": "1"},
        {"id": "2", "name": "Mouse", "price": "29.99", "category": "ELECTRONICS", "in_stock": "true"},
        {"id": "3", "name": " keyboard ", "price": "79.50", "category": "Electronics", "in_stock": "0"},
    ]
    
    initial_state: DataTransformationState = {
        "input_data": input_data,
        "type_transformed": None,
        "structure_transformed": None,
        "value_transformed": None,
        "schema_mapped": None,
        "transformation_metadata": {},
        "messages": []
    }
    
    result = transformation_graph.invoke(initial_state)
    
    print("\nTransformation Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nOriginal Data:")
    print(json.dumps(input_data[:1], indent=2))
    
    print("\nTransformed Data:")
    print(json.dumps(result["schema_mapped"][:1], indent=2))
    
    # Example 2: Structure Transformations
    print("\n" + "=" * 80)
    print("Example 2: Structure Transformations")
    print("=" * 80)
    
    nested_data = [
        {"id": 1, "user": {"name": "John", "age": 30}, "address": {"city": "NYC", "zip": "10001"}}
    ]
    
    print("\n1. Flatten Nested Structure:")
    print("Before:", json.dumps(nested_data[0], indent=2))
    flattened = structure_transformer.flatten(nested_data)
    print("After:", json.dumps(flattened[0], indent=2))
    
    print("\n2. Pivot (Rows to Columns):")
    sales_data = [
        {"region": "North", "month": "Jan", "sales": 1000},
        {"region": "North", "month": "Feb", "sales": 1500},
        {"region": "South", "month": "Jan", "sales": 800},
        {"region": "South", "month": "Feb", "sales": 900},
    ]
    
    pivoted = structure_transformer.pivot(sales_data, "region", "month", "sales")
    print("Pivoted:", json.dumps(pivoted, indent=2))
    
    # Example 3: Value Transformations
    print("\n" + "=" * 80)
    print("Example 3: Value Transformations")
    print("=" * 80)
    
    print("\n1. String Normalization:")
    messy_strings = [{"name": "  John Doe  ", "email": "JOHN@EXAMPLE.COM"}]
    normalized = value_transformer.normalize_strings(messy_strings, ["name", "email"])
    print(f"  Before: {messy_strings[0]}")
    print(f"  After: {normalized[0]}")
    
    print("\n2. Value Replacement:")
    status_data = [{"status": "Y"}, {"status": "N"}, {"status": "Y"}]
    replacements = {"Y": "Active", "N": "Inactive"}
    replaced = value_transformer.replace_values(status_data, "status", replacements)
    print(f"  Before: {[d['status'] for d in status_data]}")
    print(f"  After: {[d['status'] for d in replaced]}")
    
    # Example 4: Schema Mapping
    print("\n" + "=" * 80)
    print("Example 4: Schema Mapping")
    print("=" * 80)
    
    source_data = [
        {"first_name": "John", "last_name": "Doe", "email": "john@example.com", "age": 30}
    ]
    
    # Map to target schema
    field_mappings = {
        "full_name": ["first_name", "last_name"],  # Combine fields
        "contact_email": "email",  # Rename
        "years_old": "age",  # Rename
        "is_adult": lambda r: r.get("age", 0) >= 18  # Computed field
    }
    
    mapped = schema_mapper.map_schema(source_data, field_mappings)
    
    print("\nSource Schema:", list(source_data[0].keys()))
    print("Target Schema:", list(mapped[0].keys()))
    print("\nMapped Data:", json.dumps(mapped[0], indent=2))
    
    # Example 5: Common Transformations
    print("\n" + "=" * 80)
    print("Example 5: Common Transformation Patterns")
    print("=" * 80)
    
    print("\n1. Type Conversion:")
    print("   String → Integer: '123' → 123")
    print("   String → Date: '2024-01-15' → datetime object")
    print("   String → Boolean: 'true' → True")
    
    print("\n2. Structure Changes:")
    print("   Flatten: {a: {b: 1}} → {a_b: 1}")
    print("   Nest: {a_b: 1} → {a: {b: 1}}")
    print("   Pivot: Long → Wide format")
    
    print("\n3. Value Modifications:")
    print("   Uppercase: 'john' → 'JOHN'")
    print("   Trim: '  text  ' → 'text'")
    print("   Replace: 'Y' → 'Yes'")
    
    print("\n4. Schema Mapping:")
    print("   Rename: first_name → firstName")
    print("   Combine: [first, last] → full_name")
    print("   Derive: age → is_adult")
    
    print("\n" + "=" * 80)
    print("Data Transformation Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Transformation converts data between formats/structures
  • Type transformation: Convert data types (str → int, str → date)
  • Structure transformation: Reshape data (flatten, nest, pivot)
  • Value transformation: Modify values (uppercase, trim, replace)
  • Schema mapping: Map source to target schema
  • Common use cases: ETL, API integration, data migration
  • Tools: Pandas, dbt, Apache Spark, Talend
  • Best practices: Validate after transform, handle errors gracefully
    """)

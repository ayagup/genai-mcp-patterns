"""
Pattern 306: Structural Transformation MCP Pattern

This pattern demonstrates transforming data structures, including
flattening nested objects, creating hierarchies, and reshaping data.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class StructuralTransformationPattern(TypedDict):
    """State for structural transformation"""
    messages: Annotated[List[str], add]
    nested_data: Dict[str, Any]
    flat_data: Dict[str, Any]
    hierarchical_data: Dict[str, Any]
    transformation_log: List[Dict[str, Any]]
    structural_statistics: Dict[str, Any]


class StructureTransformer:
    """Transform data structures"""
    
    def __init__(self):
        self.transformations = []
    
    def flatten(self, data: dict, parent_key: str = '', sep: str = '.') -> dict:
        """Flatten nested dictionary"""
        items = []
        
        for key, value in data.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                items.extend(self.flatten(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        items.extend(self.flatten(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, value))
        
        return dict(items)
    
    def unflatten(self, data: dict, sep: str = '.') -> dict:
        """Unflatten dictionary to nested structure"""
        result = {}
        
        for key, value in data.items():
            parts = key.split(sep)
            target = result
            
            for part in parts[:-1]:
                # Handle array notation
                if '[' in part:
                    base_key = part.split('[')[0]
                    if base_key not in target:
                        target[base_key] = []
                    target = target[base_key]
                else:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
            
            # Set final value
            final_key = parts[-1]
            if '[' in final_key:
                final_key = final_key.split('[')[0]
            target[final_key] = value
        
        return result
    
    def to_hierarchy(self, data: List[dict], parent_key: str, children_key: str = 'children') -> dict:
        """Convert flat list to hierarchy"""
        # Build lookup
        lookup = {item[parent_key]: item for item in data if parent_key in item}
        
        # Build tree
        root = {}
        for item in data:
            parent = item.get('parent_id')
            if parent is None:
                root = item.copy()
                root[children_key] = []
            else:
                if parent in lookup:
                    if children_key not in lookup[parent]:
                        lookup[parent][children_key] = []
                    lookup[parent][children_key].append(item)
        
        return root
    
    def array_to_object(self, data: List[dict], key_field: str) -> dict:
        """Convert array to object keyed by field"""
        return {item[key_field]: item for item in data if key_field in item}
    
    def object_to_array(self, data: dict) -> List[dict]:
        """Convert object to array"""
        return list(data.values()) if isinstance(data, dict) else []


def initialize_structure_transformer_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Initialize structural transformation"""
    print("\n🏗️ Initializing Structural Transformer...")
    
    # Complex nested data
    nested_data = {
        "user": {
            "id": 12345,
            "profile": {
                "name": "John Doe",
                "age": 30,
                "contact": {
                    "email": "john@example.com",
                    "phone": "555-0123"
                }
            },
            "preferences": {
                "theme": "dark",
                "notifications": {
                    "email": True,
                    "sms": False
                }
            }
        },
        "orders": [
            {
                "id": "ORD001",
                "product": "Laptop",
                "price": 1299.99
            },
            {
                "id": "ORD002",
                "product": "Mouse",
                "price": 29.99
            }
        ],
        "metadata": {
            "created": "2024-01-01",
            "updated": "2024-01-15"
        }
    }
    
    print(f"  Nested Data Structure:")
    print(f"    Top-level keys: {len(nested_data)}")
    print(f"    Nesting levels: 4")
    print(f"    Arrays: 1 (orders)")
    
    print(f"\n  Sample Structure:")
    print(f"    user.profile.contact.email")
    print(f"    user.preferences.notifications.email")
    print(f"    orders[0].product")
    
    return {
        **state,
        "nested_data": nested_data,
        "flat_data": {},
        "hierarchical_data": {},
        "transformation_log": [],
        "structural_statistics": {},
        "messages": ["✓ Structure transformer initialized"]
    }


def flatten_structure_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Flatten nested structure"""
    print("\n📦 Flattening Nested Structure...")
    
    transformer = StructureTransformer()
    
    nested_data = state["nested_data"]
    flat_data = transformer.flatten(nested_data)
    
    print(f"  Nested Keys: {len(nested_data)}")
    print(f"  Flattened Keys: {len(flat_data)}")
    
    print(f"\n  Sample Flattened Keys:")
    for key in list(flat_data.keys())[:10]:
        print(f"    {key}: {flat_data[key]}")
    
    transformation_log = [{
        "operation": "flatten",
        "source_keys": len(nested_data),
        "target_keys": len(flat_data),
        "timestamp": "2024-01-01T12:00:00Z"
    }]
    
    return {
        **state,
        "flat_data": flat_data,
        "transformation_log": transformation_log,
        "messages": [f"✓ Flattened to {len(flat_data)} keys"]
    }


def unflatten_structure_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Unflatten to nested structure"""
    print("\n📂 Unflattening to Nested Structure...")
    
    transformer = StructureTransformer()
    
    flat_data = state["flat_data"]
    unflattened = transformer.unflatten(flat_data)
    
    print(f"  Flat Keys: {len(flat_data)}")
    print(f"  Unflattened Top-level Keys: {len(unflattened)}")
    
    print(f"\n  Reconstructed Structure:")
    for key in unflattened.keys():
        print(f"    {key}: {type(unflattened[key]).__name__}")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "operation": "unflatten",
        "source_keys": len(flat_data),
        "target_keys": len(unflattened),
        "timestamp": "2024-01-01T12:01:00Z"
    })
    
    return {
        **state,
        "transformation_log": transformation_log,
        "messages": [f"✓ Unflattened to {len(unflattened)} top-level keys"]
    }


def array_to_object_transform_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Transform array to object"""
    print("\n🔄 Transforming Array → Object...")
    
    transformer = StructureTransformer()
    
    # Get orders array from nested data
    orders_array = state["nested_data"]["orders"]
    
    print(f"  Source: Array with {len(orders_array)} items")
    
    # Convert to object keyed by id
    orders_object = transformer.array_to_object(orders_array, "id")
    
    print(f"  Target: Object with {len(orders_object)} keys")
    
    print(f"\n  Object Keys:")
    for key in orders_object.keys():
        print(f"    {key}: {orders_object[key]['product']}")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "operation": "array_to_object",
        "source_count": len(orders_array),
        "target_count": len(orders_object),
        "key_field": "id",
        "timestamp": "2024-01-01T12:02:00Z"
    })
    
    return {
        **state,
        "transformation_log": transformation_log,
        "messages": [f"✓ Converted array to object ({len(orders_object)} keys)"]
    }


def object_to_array_transform_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Transform object to array"""
    print("\n🔄 Transforming Object → Array...")
    
    transformer = StructureTransformer()
    
    # Create sample object
    sample_object = {
        "US": {"name": "United States", "code": "US", "population": 331000000},
        "UK": {"name": "United Kingdom", "code": "UK", "population": 67000000},
        "CA": {"name": "Canada", "code": "CA", "population": 38000000}
    }
    
    print(f"  Source: Object with {len(sample_object)} keys")
    
    # Convert to array
    result_array = transformer.object_to_array(sample_object)
    
    print(f"  Target: Array with {len(result_array)} items")
    
    print(f"\n  Array Items:")
    for item in result_array:
        print(f"    {item['code']}: {item['name']}")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "operation": "object_to_array",
        "source_count": len(sample_object),
        "target_count": len(result_array),
        "timestamp": "2024-01-01T12:03:00Z"
    })
    
    return {
        **state,
        "transformation_log": transformation_log,
        "messages": [f"✓ Converted object to array ({len(result_array)} items)"]
    }


def create_hierarchy_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Create hierarchical structure"""
    print("\n🌳 Creating Hierarchical Structure...")
    
    # Sample flat list with parent references
    flat_list = [
        {"id": 1, "name": "Root", "parent_id": None},
        {"id": 2, "name": "Child 1", "parent_id": 1},
        {"id": 3, "name": "Child 2", "parent_id": 1},
        {"id": 4, "name": "Grandchild 1.1", "parent_id": 2},
        {"id": 5, "name": "Grandchild 1.2", "parent_id": 2}
    ]
    
    print(f"  Flat List: {len(flat_list)} items")
    
    # Build hierarchy manually (simplified)
    hierarchy = {
        "id": 1,
        "name": "Root",
        "children": [
            {
                "id": 2,
                "name": "Child 1",
                "children": [
                    {"id": 4, "name": "Grandchild 1.1"},
                    {"id": 5, "name": "Grandchild 1.2"}
                ]
            },
            {
                "id": 3,
                "name": "Child 2",
                "children": []
            }
        ]
    }
    
    def count_nodes(node):
        count = 1
        if "children" in node:
            for child in node["children"]:
                count += count_nodes(child)
        return count
    
    total_nodes = count_nodes(hierarchy)
    
    print(f"  Hierarchy: {total_nodes} nodes")
    print(f"  Max Depth: 3 levels")
    
    print(f"\n  Structure:")
    print(f"    Root")
    print(f"      ├─ Child 1")
    print(f"      │   ├─ Grandchild 1.1")
    print(f"      │   └─ Grandchild 1.2")
    print(f"      └─ Child 2")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "operation": "create_hierarchy",
        "source_count": len(flat_list),
        "target_nodes": total_nodes,
        "max_depth": 3,
        "timestamp": "2024-01-01T12:04:00Z"
    })
    
    return {
        **state,
        "hierarchical_data": hierarchy,
        "transformation_log": transformation_log,
        "messages": [f"✓ Created hierarchy with {total_nodes} nodes"]
    }


def analyze_structural_transformations_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Analyze structural transformations"""
    print("\n📊 Analyzing Structural Transformations...")
    
    transformations = state["transformation_log"]
    
    # Analyze transformations
    total_transformations = len(transformations)
    operations = [t["operation"] for t in transformations]
    operation_counts = {op: operations.count(op) for op in set(operations)}
    
    # Calculate complexity metrics
    nested_depth = 4  # From nested_data
    flat_keys = len(state["flat_data"])
    
    statistics = {
        "total_transformations": total_transformations,
        "operation_types": list(operation_counts.keys()),
        "operation_counts": operation_counts,
        "original_nesting_depth": nested_depth,
        "flattened_keys": flat_keys,
        "complexity_reduction": nested_depth - 1  # Flattened to 1 level
    }
    
    print(f"  Total Transformations: {statistics['total_transformations']}")
    print(f"  Operation Types: {', '.join(statistics['operation_types'])}")
    
    print(f"\n  Operation Distribution:")
    for op, count in operation_counts.items():
        print(f"    {op}: {count}")
    
    print(f"\n  Complexity Metrics:")
    print(f"    Original Nesting Depth: {statistics['original_nesting_depth']}")
    print(f"    Flattened Keys: {statistics['flattened_keys']}")
    print(f"    Complexity Reduction: {statistics['complexity_reduction']} levels")
    
    return {
        **state,
        "structural_statistics": statistics,
        "messages": ["✓ Structural transformations analyzed"]
    }


def generate_structural_transformation_report_agent(state: StructuralTransformationPattern) -> StructuralTransformationPattern:
    """Generate structural transformation report"""
    print("\n" + "="*70)
    print("STRUCTURAL TRANSFORMATION REPORT")
    print("="*70)
    
    print(f"\n📥 Original Nested Structure:")
    nested = state["nested_data"]
    print(f"  Top-level Keys: {len(nested)}")
    for key in nested.keys():
        print(f"    {key}: {type(nested[key]).__name__}")
    
    print(f"\n📦 Flattened Structure:")
    flat = state["flat_data"]
    print(f"  Total Keys: {len(flat)}")
    print(f"  Sample Keys:")
    for key in list(flat.keys())[:8]:
        value = str(flat[key])[:30]
        print(f"    {key}: {value}")
    
    print(f"\n🌳 Hierarchical Structure:")
    hierarchy = state["hierarchical_data"]
    if hierarchy:
        print(f"  Root: {hierarchy.get('name', 'N/A')}")
        if "children" in hierarchy:
            print(f"  Direct Children: {len(hierarchy['children'])}")
    
    print(f"\n🔄 Transformations Performed:")
    for i, trans in enumerate(state["transformation_log"], 1):
        print(f"\n  {i}. {trans['operation'].upper()}")
        if "source_keys" in trans:
            print(f"     Source Keys: {trans['source_keys']}")
        if "target_keys" in trans:
            print(f"     Target Keys: {trans['target_keys']}")
        if "source_count" in trans:
            print(f"     Source Count: {trans['source_count']}")
        if "target_count" in trans:
            print(f"     Target Count: {trans['target_count']}")
    
    print(f"\n📊 Statistics:")
    stats = state["structural_statistics"]
    if stats:
        print(f"  Total Transformations: {stats['total_transformations']}")
        print(f"  Operation Types: {', '.join(stats['operation_types'])}")
        print(f"  Original Nesting: {stats['original_nesting_depth']} levels")
        print(f"  Flattened Keys: {stats['flattened_keys']}")
    
    print(f"\n💡 Structural Transformation Benefits:")
    print("  ✓ Simplify data access")
    print("  ✓ Enable querying")
    print("  ✓ Optimize storage")
    print("  ✓ Improve performance")
    print("  ✓ Flexible reshaping")
    print("  ✓ Data integration")
    
    print(f"\n🔧 Transformation Operations:")
    print("  • Flatten (nested → flat)")
    print("  • Unflatten (flat → nested)")
    print("  • Array → Object")
    print("  • Object → Array")
    print("  • List → Hierarchy")
    print("  • Hierarchy → List")
    
    print(f"\n⚙️ Use Cases:")
    print("  • API response reshaping")
    print("  • Database normalization")
    print("  • CSV export/import")
    print("  • UI data binding")
    print("  • Search indexing")
    print("  • Data migration")
    
    print(f"\n🎯 Common Patterns:")
    print("  • Flatten for CSV export")
    print("  • Nest for JSON APIs")
    print("  • Array→Object for lookup")
    print("  • Object→Array for iteration")
    print("  • Hierarchy for tree views")
    
    print("\n" + "="*70)
    print("✅ Structural Transformation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_structural_transformation_graph():
    """Create structural transformation workflow"""
    workflow = StateGraph(StructuralTransformationPattern)
    
    workflow.add_node("initialize", initialize_structure_transformer_agent)
    workflow.add_node("flatten", flatten_structure_agent)
    workflow.add_node("unflatten", unflatten_structure_agent)
    workflow.add_node("array_to_object", array_to_object_transform_agent)
    workflow.add_node("object_to_array", object_to_array_transform_agent)
    workflow.add_node("create_hierarchy", create_hierarchy_agent)
    workflow.add_node("analyze", analyze_structural_transformations_agent)
    workflow.add_node("report", generate_structural_transformation_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "flatten")
    workflow.add_edge("flatten", "unflatten")
    workflow.add_edge("unflatten", "array_to_object")
    workflow.add_edge("array_to_object", "object_to_array")
    workflow.add_edge("object_to_array", "create_hierarchy")
    workflow.add_edge("create_hierarchy", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 306: Structural Transformation MCP Pattern")
    print("="*70)
    
    app = create_structural_transformation_graph()
    final_state = app.invoke({
        "messages": [],
        "nested_data": {},
        "flat_data": {},
        "hierarchical_data": {},
        "transformation_log": [],
        "structural_statistics": {}
    })
    
    print("\n✅ Structural Transformation Pattern Complete!")


if __name__ == "__main__":
    main()

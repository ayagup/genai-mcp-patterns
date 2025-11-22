"""
Pattern 187: Data Filtering MCP Pattern

This pattern demonstrates selecting subsets of data based on conditions and predicates.
Filtering is fundamental for data processing, removing irrelevant records, and
focusing on data that meets specific criteria.

Key Concepts:
1. Predicates: Boolean conditions for selection
2. Inclusion Filters: Select records that match
3. Exclusion Filters: Reject records that match
4. Composite Filters: Combine multiple conditions
5. Dynamic Filters: Runtime-defined conditions
6. Filter Pushdown: Move filters closer to data source
7. Early Filtering: Filter early to reduce data volume

Filter Types:
- Equality: field == value
- Comparison: field > value, field < value
- Range: value BETWEEN min AND max
- Pattern Match: field LIKE pattern
- Set Membership: field IN (values)
- Null Check: field IS NULL, field IS NOT NULL
- Logical: AND, OR, NOT combinations

Filter Strategies:
- Conjunctive (AND): All conditions must be true
- Disjunctive (OR): Any condition can be true
- Negation (NOT): Condition must be false
- Complex: Nested combinations (A AND (B OR C))

Benefits:
- Data Reduction: Remove irrelevant records early
- Performance: Process less data downstream
- Focus: Work with relevant subset only
- Quality: Remove invalid/unwanted data
- Compliance: Filter sensitive data

Trade-offs:
- Complexity: Complex filters hard to understand
- Performance: Complex predicates can be slow
- False Positives/Negatives: Incorrect filtering
- Maintenance: Filter logic needs updates

Use Cases:
- Data Cleaning: Remove invalid records
- Security: Filter unauthorized data
- Analytics: Focus on specific segments
- Log Processing: Filter by severity level
- E-commerce: Filter products by category/price
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re

# Define the state for data filtering
class DataFilteringState(TypedDict):
    """State for data filtering"""
    raw_data: List[Dict[str, Any]]
    simple_filtered: Optional[List[Dict[str, Any]]]
    range_filtered: Optional[List[Dict[str, Any]]]
    pattern_filtered: Optional[List[Dict[str, Any]]]
    composite_filtered: Optional[List[Dict[str, Any]]]
    filter_statistics: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# FILTER PREDICATES
# ============================================================================

class FilterOperator(Enum):
    """Filter comparison operators"""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES = "matches"  # Regex
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

@dataclass
class FilterPredicate:
    """Filter predicate definition"""
    field: str
    operator: FilterOperator
    value: Any
    
    def evaluate(self, record: Dict[str, Any]) -> bool:
        """Evaluate predicate against record"""
        field_value = record.get(self.field)
        
        if self.operator == FilterOperator.EQUALS:
            return field_value == self.value
        elif self.operator == FilterOperator.NOT_EQUALS:
            return field_value != self.value
        elif self.operator == FilterOperator.GREATER_THAN:
            return field_value > self.value if field_value is not None else False
        elif self.operator == FilterOperator.GREATER_EQUAL:
            return field_value >= self.value if field_value is not None else False
        elif self.operator == FilterOperator.LESS_THAN:
            return field_value < self.value if field_value is not None else False
        elif self.operator == FilterOperator.LESS_EQUAL:
            return field_value <= self.value if field_value is not None else False
        elif self.operator == FilterOperator.IN:
            return field_value in self.value if field_value is not None else False
        elif self.operator == FilterOperator.NOT_IN:
            return field_value not in self.value if field_value is not None else False
        elif self.operator == FilterOperator.CONTAINS:
            return self.value in str(field_value) if field_value is not None else False
        elif self.operator == FilterOperator.STARTS_WITH:
            return str(field_value).startswith(str(self.value)) if field_value is not None else False
        elif self.operator == FilterOperator.ENDS_WITH:
            return str(field_value).endswith(str(self.value)) if field_value is not None else False
        elif self.operator == FilterOperator.MATCHES:
            return bool(re.match(str(self.value), str(field_value))) if field_value is not None else False
        elif self.operator == FilterOperator.IS_NULL:
            return field_value is None
        elif self.operator == FilterOperator.IS_NOT_NULL:
            return field_value is not None
        
        return False

# ============================================================================
# FILTER IMPLEMENTATIONS
# ============================================================================

class SimpleFilter:
    """
    Simple Filter: Single condition
    
    Example: SELECT * FROM users WHERE age > 18
    """
    
    def filter(self, data: List[Dict[str, Any]], predicate: FilterPredicate) -> List[Dict[str, Any]]:
        """Filter data with single predicate"""
        return [record for record in data if predicate.evaluate(record)]
    
    def filter_by_function(self, data: List[Dict[str, Any]], 
                          filter_fn: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
        """Filter using custom function"""
        return [record for record in data if filter_fn(record)]

class RangeFilter:
    """
    Range Filter: Value in range
    
    Example: SELECT * FROM products WHERE price BETWEEN 10 AND 100
    """
    
    def filter_range(self, data: List[Dict[str, Any]], 
                    field: str, 
                    min_value: Any, 
                    max_value: Any,
                    inclusive: bool = True) -> List[Dict[str, Any]]:
        """Filter records with field value in range"""
        filtered = []
        
        for record in data:
            value = record.get(field)
            
            if value is None:
                continue
            
            if inclusive:
                if min_value <= value <= max_value:
                    filtered.append(record)
            else:
                if min_value < value < max_value:
                    filtered.append(record)
        
        return filtered

class PatternFilter:
    """
    Pattern Filter: Match text patterns
    
    Examples:
    - LIKE '%@gmail.com' (email ends with gmail.com)
    - LIKE 'John%' (name starts with John)
    - MATCHES '^[0-9]{3}-[0-9]{4}$' (phone number pattern)
    """
    
    def filter_like(self, data: List[Dict[str, Any]], 
                   field: str, 
                   pattern: str) -> List[Dict[str, Any]]:
        """
        Filter using LIKE pattern
        
        % = wildcard (any characters)
        _ = single character
        """
        # Convert SQL LIKE to regex
        regex_pattern = pattern.replace('%', '.*').replace('_', '.')
        regex_pattern = f"^{regex_pattern}$"
        
        filtered = []
        
        for record in data:
            value = str(record.get(field, ""))
            
            if re.match(regex_pattern, value):
                filtered.append(record)
        
        return filtered
    
    def filter_regex(self, data: List[Dict[str, Any]], 
                    field: str, 
                    regex: str) -> List[Dict[str, Any]]:
        """Filter using regular expression"""
        filtered = []
        
        for record in data:
            value = str(record.get(field, ""))
            
            if re.search(regex, value):
                filtered.append(record)
        
        return filtered

class CompositeFilter:
    """
    Composite Filter: Multiple conditions combined
    
    Logical Operators:
    - AND: All conditions must be true
    - OR: Any condition can be true
    - NOT: Negate condition
    """
    
    def filter_and(self, data: List[Dict[str, Any]], 
                  predicates: List[FilterPredicate]) -> List[Dict[str, Any]]:
        """Filter with AND (all conditions must be true)"""
        filtered = []
        
        for record in data:
            if all(predicate.evaluate(record) for predicate in predicates):
                filtered.append(record)
        
        return filtered
    
    def filter_or(self, data: List[Dict[str, Any]], 
                 predicates: List[FilterPredicate]) -> List[Dict[str, Any]]:
        """Filter with OR (any condition can be true)"""
        filtered = []
        
        for record in data:
            if any(predicate.evaluate(record) for predicate in predicates):
                filtered.append(record)
        
        return filtered
    
    def filter_not(self, data: List[Dict[str, Any]], 
                  predicate: FilterPredicate) -> List[Dict[str, Any]]:
        """Filter with NOT (negate condition)"""
        filtered = []
        
        for record in data:
            if not predicate.evaluate(record):
                filtered.append(record)
        
        return filtered

class DynamicFilter:
    """
    Dynamic Filter: Build filters at runtime
    
    Useful for user-defined filters, query builders, etc.
    """
    
    def build_filter(self, field: str, operator: str, value: Any) -> FilterPredicate:
        """Build filter predicate from parameters"""
        op_enum = FilterOperator[operator.upper()]
        return FilterPredicate(field, op_enum, value)
    
    def apply_filters(self, data: List[Dict[str, Any]], 
                     filter_specs: List[Dict[str, Any]],
                     logic: str = "AND") -> List[Dict[str, Any]]:
        """
        Apply multiple dynamically-defined filters
        
        filter_specs: [{"field": "age", "operator": "greater_than", "value": 18}, ...]
        logic: "AND" or "OR"
        """
        predicates = []
        
        for spec in filter_specs:
            predicate = self.build_filter(
                spec["field"],
                spec["operator"],
                spec["value"]
            )
            predicates.append(predicate)
        
        # Apply filters
        composite = CompositeFilter()
        
        if logic == "AND":
            return composite.filter_and(data, predicates)
        else:
            return composite.filter_or(data, predicates)

# Create filter instances
simple_filter = SimpleFilter()
range_filter = RangeFilter()
pattern_filter = PatternFilter()
composite_filter = CompositeFilter()
dynamic_filter = DynamicFilter()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def apply_simple_filter(state: DataFilteringState) -> DataFilteringState:
    """Apply simple equality filter"""
    raw_data = state["raw_data"]
    
    # Filter: category == "Electronics"
    predicate = FilterPredicate("category", FilterOperator.EQUALS, "Electronics")
    filtered = simple_filter.filter(raw_data, predicate)
    
    return {
        "simple_filtered": filtered,
        "messages": [f"[Simple Filter] category == 'Electronics': {len(raw_data)} → {len(filtered)} records"]
    }

def apply_range_filter(state: DataFilteringState) -> DataFilteringState:
    """Apply range filter"""
    raw_data = state["raw_data"]
    
    # Filter: 100 <= price <= 2000
    filtered = range_filter.filter_range(raw_data, "price", 100, 2000)
    
    return {
        "range_filtered": filtered,
        "messages": [f"[Range Filter] 100 <= price <= 2000: {len(raw_data)} → {len(filtered)} records"]
    }

def apply_pattern_filter(state: DataFilteringState) -> DataFilteringState:
    """Apply pattern matching filter"""
    raw_data = state["raw_data"]
    
    # Filter: name starts with 'Product'
    filtered = pattern_filter.filter_like(raw_data, "name", "Product%")
    
    return {
        "pattern_filtered": filtered,
        "messages": [f"[Pattern Filter] name LIKE 'Product%': {len(raw_data)} → {len(filtered)} records"]
    }

def apply_composite_filter(state: DataFilteringState) -> DataFilteringState:
    """Apply composite filter (multiple conditions)"""
    raw_data = state["raw_data"]
    
    # Filter: category == "Electronics" AND price > 500
    predicates = [
        FilterPredicate("category", FilterOperator.EQUALS, "Electronics"),
        FilterPredicate("price", FilterOperator.GREATER_THAN, 500)
    ]
    
    filtered = composite_filter.filter_and(raw_data, predicates)
    
    return {
        "composite_filtered": filtered,
        "filter_statistics": {
            "total_input": len(raw_data),
            "total_output": len(filtered),
            "filtered_out": len(raw_data) - len(filtered),
            "filter_ratio": (len(raw_data) - len(filtered)) / len(raw_data) if raw_data else 0
        },
        "messages": [f"[Composite Filter] category == 'Electronics' AND price > 500: {len(raw_data)} → {len(filtered)} records"]
    }

# ============================================================================
# BUILD THE FILTERING GRAPH
# ============================================================================

def create_filtering_graph():
    """
    Create a StateGraph demonstrating data filtering pattern.
    
    Flow:
    1. Simple Filter: Single condition
    2. Range Filter: Value in range
    3. Pattern Filter: Text pattern matching
    4. Composite Filter: Multiple conditions (AND)
    """
    
    workflow = StateGraph(DataFilteringState)
    
    # Add filter nodes
    workflow.add_node("simple", apply_simple_filter)
    workflow.add_node("range", apply_range_filter)
    workflow.add_node("pattern", apply_pattern_filter)
    workflow.add_node("composite", apply_composite_filter)
    
    # Define filter flow
    workflow.add_edge(START, "simple")
    workflow.add_edge("simple", "range")
    workflow.add_edge("range", "pattern")
    workflow.add_edge("pattern", "composite")
    workflow.add_edge("composite", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Filtering MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Multi-Stage Filtering
    print("\n" + "=" * 80)
    print("Example 1: Multi-Stage Filtering Pipeline")
    print("=" * 80)
    
    filtering_graph = create_filtering_graph()
    
    # Sample product data
    products = [
        {"id": 1, "name": "Product_A", "category": "Electronics", "price": 500},
        {"id": 2, "name": "Product_B", "category": "Electronics", "price": 1500},
        {"id": 3, "name": "Item_C", "category": "Clothing", "price": 50},
        {"id": 4, "name": "Product_D", "category": "Electronics", "price": 2500},
        {"id": 5, "name": "Product_E", "category": "Food", "price": 20},
        {"id": 6, "name": "Product_F", "category": "Electronics", "price": 800},
    ]
    
    initial_state: DataFilteringState = {
        "raw_data": products,
        "simple_filtered": None,
        "range_filtered": None,
        "pattern_filtered": None,
        "composite_filtered": None,
        "filter_statistics": {},
        "messages": []
    }
    
    result = filtering_graph.invoke(initial_state)
    
    print("\nFiltering Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nFilter Statistics:")
    stats = result["filter_statistics"]
    print(f"  Input Records: {stats['total_input']}")
    print(f"  Output Records: {stats['total_output']}")
    print(f"  Filtered Out: {stats['filtered_out']}")
    print(f"  Filter Ratio: {stats['filter_ratio']:.1%}")
    
    print("\nFinal Filtered Results:")
    for product in result["composite_filtered"]:
        print(f"  {product['name']}: ${product['price']} ({product['category']})")
    
    # Example 2: Dynamic Filters
    print("\n" + "=" * 80)
    print("Example 2: Dynamic Runtime Filters")
    print("=" * 80)
    
    # Build filters dynamically
    filter_specs = [
        {"field": "category", "operator": "in", "value": ["Electronics", "Clothing"]},
        {"field": "price", "operator": "less_than", "value": 1000}
    ]
    
    dynamic_filtered = dynamic_filter.apply_filters(products, filter_specs, logic="AND")
    
    print(f"\nDynamic Filter: category IN ['Electronics', 'Clothing'] AND price < 1000")
    print(f"Result: {len(dynamic_filtered)} records")
    for product in dynamic_filtered:
        print(f"  {product['name']}: ${product['price']} ({product['category']})")
    
    # Example 3: Filter Patterns
    print("\n" + "=" * 80)
    print("Example 3: Common Filter Patterns")
    print("=" * 80)
    
    print("\n1. Equality Filter:")
    print("   WHERE status == 'active'")
    equality_pred = FilterPredicate("category", FilterOperator.EQUALS, "Electronics")
    equality_result = simple_filter.filter(products, equality_pred)
    print(f"   Result: {len(equality_result)} records")
    
    print("\n2. Range Filter:")
    print("   WHERE price BETWEEN 100 AND 1000")
    range_result = range_filter.filter_range(products, "price", 100, 1000)
    print(f"   Result: {len(range_result)} records")
    
    print("\n3. Pattern Filter:")
    print("   WHERE name LIKE 'Product%'")
    pattern_result = pattern_filter.filter_like(products, "name", "Product%")
    print(f"   Result: {len(pattern_result)} records")
    
    print("\n4. Set Membership:")
    print("   WHERE category IN ('Electronics', 'Food')")
    in_pred = FilterPredicate("category", FilterOperator.IN, ["Electronics", "Food"])
    in_result = simple_filter.filter(products, in_pred)
    print(f"   Result: {len(in_result)} records")
    
    # Example 4: Filter Optimization
    print("\n" + "=" * 80)
    print("Example 4: Filter Optimization Strategies")
    print("=" * 80)
    
    print("\n1. Filter Pushdown:")
    print("   Move filters closer to data source (database)")
    print("   Example: Filter in SQL WHERE clause, not in Python")
    
    print("\n2. Early Filtering:")
    print("   Filter early in pipeline to reduce data volume")
    print("   Example: Filter → Transform → Aggregate (not Transform → Filter → Aggregate)")
    
    print("\n3. Selective Conditions:")
    print("   Apply most selective filters first")
    print("   Example: If status=='active' filters 90%, apply it first")
    
    print("\n4. Index Usage:")
    print("   Filter on indexed columns")
    print("   Example: WHERE id = 123 (indexed) is fast")
    
    print("\n5. Avoid Functions in Filters:")
    print("   Don't: WHERE UPPER(name) = 'JOHN' (can't use index)")
    print("   Do: WHERE name = 'John' (can use index)")
    
    print("\n" + "=" * 80)
    print("Data Filtering Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Filtering selects data subsets based on conditions
  • Common operators: ==, !=, >, <, IN, LIKE, MATCHES
  • Composite filters: AND, OR, NOT combinations
  • Pattern matching: LIKE for SQL-style, regex for complex
  • Filter early: Reduce data volume in pipeline
  • Filter pushdown: Move filters to data source when possible
  • Dynamic filters: Build filters at runtime for flexibility
  • Use for: data cleaning, security, analytics, compliance
    """)

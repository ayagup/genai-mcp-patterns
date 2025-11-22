"""
Pattern 189: Data Validation MCP Pattern

This pattern demonstrates verifying that data meets quality standards, business rules,
and schema requirements. Validation ensures data integrity before processing or storage.

Key Concepts:
1. Schema Validation: Check structure matches expected format
2. Type Validation: Verify correct data types
3. Constraint Validation: Enforce business rules
4. Range Validation: Check values within bounds
5. Format Validation: Match patterns (email, phone)
6. Reference Validation: Verify foreign key integrity
7. Completeness Validation: Check required fields

Validation Levels:
- Syntax: Basic format (JSON valid, CSV parseable)
- Schema: Structure matches specification
- Semantic: Values make business sense
- Cross-field: Field relationships valid
- Reference: External dependencies exist
- Statistical: Values within normal distribution

Validation Strategies:
- Fail Fast: Stop on first error
- Collect All: Report all errors
- Warnings vs Errors: Distinguish severity
- Auto-correction: Fix common issues
- Quarantine: Isolate invalid records
- Progressive: Validate in stages

Benefits:
- Data Quality: Ensure high-quality data
- Error Prevention: Catch problems early
- Compliance: Meet regulatory requirements
- Trust: Increase confidence in data
- Debugging: Easier to find issues

Trade-offs:
- Performance: Validation adds processing time
- Complexity: Complex rules hard to maintain
- False Positives: Over-validation rejects good data
- Maintenance: Rules change with business
- Coverage: Can't validate everything

Use Cases:
- Data Ingestion: Validate before loading
- API Validation: Check request/response
- Form Validation: User input validation
- ETL Pipelines: Quality gates
- Data Migration: Verify after transfer
- Compliance: Regulatory requirements
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import re

# Define the state for data validation
class DataValidationState(TypedDict):
    """State for data validation"""
    input_data: List[Dict[str, Any]]
    schema_validation_results: Optional[Dict[str, Any]]
    constraint_validation_results: Optional[Dict[str, Any]]
    format_validation_results: Optional[Dict[str, Any]]
    cross_field_validation_results: Optional[Dict[str, Any]]
    validation_summary: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# VALIDATION RULES AND RESULTS
# ============================================================================

class ValidationSeverity(Enum):
    """Severity of validation issue"""
    ERROR = "error"  # Critical, reject record
    WARNING = "warning"  # Non-critical, accept with warning
    INFO = "info"  # Informational

@dataclass
class ValidationRule:
    """Validation rule definition"""
    rule_id: str
    rule_name: str
    field: Optional[str]
    severity: ValidationSeverity
    error_message: str

@dataclass
class ValidationIssue:
    """Validation issue found"""
    rule_id: str
    severity: ValidationSeverity
    field: Optional[str]
    message: str
    record_index: int
    actual_value: Any

# ============================================================================
# VALIDATORS
# ============================================================================

class SchemaValidator:
    """
    Schema Validation: Verify structure matches specification
    
    Checks:
    - Required fields present
    - No unexpected fields
    - Field types correct
    - Nested structure valid
    """
    
    def __init__(self, schema: Dict[str, Any]):
        """
        schema format:
        {
            "field_name": {
                "type": "str|int|float|bool|date|dict|list",
                "required": True|False,
                "nullable": True|False
            }
        }
        """
        self.schema = schema
    
    def validate(self, data: List[Dict[str, Any]]) -> List[ValidationIssue]:
        """Validate data against schema"""
        issues = []
        
        for idx, record in enumerate(data):
            # Check required fields
            for field, spec in self.schema.items():
                if spec.get("required", False):
                    if field not in record:
                        issues.append(ValidationIssue(
                            rule_id="SCHEMA_001",
                            severity=ValidationSeverity.ERROR,
                            field=field,
                            message=f"Required field '{field}' is missing",
                            record_index=idx,
                            actual_value=None
                        ))
                    elif record[field] is None and not spec.get("nullable", False):
                        issues.append(ValidationIssue(
                            rule_id="SCHEMA_002",
                            severity=ValidationSeverity.ERROR,
                            field=field,
                            message=f"Field '{field}' cannot be null",
                            record_index=idx,
                            actual_value=None
                        ))
            
            # Check field types
            for field, value in record.items():
                if field in self.schema and value is not None:
                    expected_type = self.schema[field].get("type")
                    
                    if expected_type:
                        is_valid = self._check_type(value, expected_type)
                        
                        if not is_valid:
                            issues.append(ValidationIssue(
                                rule_id="SCHEMA_003",
                                severity=ValidationSeverity.ERROR,
                                field=field,
                                message=f"Field '{field}' has wrong type. Expected {expected_type}, got {type(value).__name__}",
                                record_index=idx,
                                actual_value=value
                            ))
        
        return issues
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_checks = {
            "str": lambda v: isinstance(v, str),
            "int": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, (int, float)),
            "bool": lambda v: isinstance(v, bool),
            "dict": lambda v: isinstance(v, dict),
            "list": lambda v: isinstance(v, list),
            "date": lambda v: isinstance(v, (str, datetime))
        }
        
        check_fn = type_checks.get(expected_type)
        return check_fn(value) if check_fn else True

class ConstraintValidator:
    """
    Constraint Validation: Enforce business rules
    
    Constraints:
    - Range: Value within min/max
    - Enum: Value in allowed set
    - Pattern: Match regex
    - Length: String/list length
    - Uniqueness: No duplicates
    """
    
    def validate_range(self, data: List[Dict[str, Any]],
                      field: str,
                      min_value: Optional[float] = None,
                      max_value: Optional[float] = None) -> List[ValidationIssue]:
        """Validate numeric range"""
        issues = []
        
        for idx, record in enumerate(data):
            if field in record:
                value = record[field]
                
                if min_value is not None and value < min_value:
                    issues.append(ValidationIssue(
                        rule_id="RANGE_001",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Value {value} is below minimum {min_value}",
                        record_index=idx,
                        actual_value=value
                    ))
                
                if max_value is not None and value > max_value:
                    issues.append(ValidationIssue(
                        rule_id="RANGE_002",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Value {value} is above maximum {max_value}",
                        record_index=idx,
                        actual_value=value
                    ))
        
        return issues
    
    def validate_enum(self, data: List[Dict[str, Any]],
                     field: str,
                     allowed_values: List[Any]) -> List[ValidationIssue]:
        """Validate value in allowed set"""
        issues = []
        
        for idx, record in enumerate(data):
            if field in record:
                value = record[field]
                
                if value not in allowed_values:
                    issues.append(ValidationIssue(
                        rule_id="ENUM_001",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Value '{value}' not in allowed values: {allowed_values}",
                        record_index=idx,
                        actual_value=value
                    ))
        
        return issues
    
    def validate_length(self, data: List[Dict[str, Any]],
                       field: str,
                       min_length: Optional[int] = None,
                       max_length: Optional[int] = None) -> List[ValidationIssue]:
        """Validate string/list length"""
        issues = []
        
        for idx, record in enumerate(data):
            if field in record:
                value = record[field]
                length = len(value) if value else 0
                
                if min_length is not None and length < min_length:
                    issues.append(ValidationIssue(
                        rule_id="LENGTH_001",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Length {length} is below minimum {min_length}",
                        record_index=idx,
                        actual_value=value
                    ))
                
                if max_length is not None and length > max_length:
                    issues.append(ValidationIssue(
                        rule_id="LENGTH_002",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Length {length} is above maximum {max_length}",
                        record_index=idx,
                        actual_value=value
                    ))
        
        return issues
    
    def validate_uniqueness(self, data: List[Dict[str, Any]],
                          field: str) -> List[ValidationIssue]:
        """Validate field uniqueness"""
        issues = []
        seen = {}
        
        for idx, record in enumerate(data):
            if field in record:
                value = record[field]
                
                if value in seen:
                    issues.append(ValidationIssue(
                        rule_id="UNIQUE_001",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Duplicate value '{value}' (also in record {seen[value]})",
                        record_index=idx,
                        actual_value=value
                    ))
                else:
                    seen[value] = idx
        
        return issues

class FormatValidator:
    """
    Format Validation: Match patterns
    
    Common formats:
    - Email
    - Phone number
    - URL
    - IP address
    - Credit card
    - Date/time
    """
    
    PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "phone": r'^\+?1?\d{9,15}$',
        "url": r'^https?://[^\s]+$',
        "ipv4": r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$',
        "date_iso": r'^\d{4}-\d{2}-\d{2}$',
        "zip_us": r'^\d{5}(-\d{4})?$'
    }
    
    def validate_pattern(self, data: List[Dict[str, Any]],
                        field: str,
                        pattern_name: str) -> List[ValidationIssue]:
        """Validate against pattern"""
        issues = []
        pattern = self.PATTERNS.get(pattern_name)
        
        if not pattern:
            return issues
        
        for idx, record in enumerate(data):
            if field in record:
                value = str(record[field])
                
                if not re.match(pattern, value):
                    issues.append(ValidationIssue(
                        rule_id=f"FORMAT_{pattern_name.upper()}",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Value '{value}' doesn't match {pattern_name} format",
                        record_index=idx,
                        actual_value=value
                    ))
        
        return issues
    
    def validate_regex(self, data: List[Dict[str, Any]],
                      field: str,
                      regex: str) -> List[ValidationIssue]:
        """Validate against custom regex"""
        issues = []
        
        for idx, record in enumerate(data):
            if field in record:
                value = str(record[field])
                
                if not re.match(regex, value):
                    issues.append(ValidationIssue(
                        rule_id="FORMAT_REGEX",
                        severity=ValidationSeverity.ERROR,
                        field=field,
                        message=f"Value '{value}' doesn't match pattern {regex}",
                        record_index=idx,
                        actual_value=value
                    ))
        
        return issues

class CrossFieldValidator:
    """
    Cross-field Validation: Verify field relationships
    
    Examples:
    - end_date > start_date
    - total = price * quantity
    - if field A then field B required
    """
    
    def validate_comparison(self, data: List[Dict[str, Any]],
                          field1: str,
                          field2: str,
                          operator: str) -> List[ValidationIssue]:
        """Validate field comparison"""
        issues = []
        
        ops = {
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b
        }
        
        op_fn = ops.get(operator)
        if not op_fn:
            return issues
        
        for idx, record in enumerate(data):
            if field1 in record and field2 in record:
                value1 = record[field1]
                value2 = record[field2]
                
                if not op_fn(value1, value2):
                    issues.append(ValidationIssue(
                        rule_id="CROSS_FIELD_001",
                        severity=ValidationSeverity.ERROR,
                        field=None,
                        message=f"Condition '{field1} {operator} {field2}' failed ({value1} vs {value2})",
                        record_index=idx,
                        actual_value=f"{value1}, {value2}"
                    ))
        
        return issues
    
    def validate_conditional(self, data: List[Dict[str, Any]],
                           if_field: str,
                           if_value: Any,
                           then_field: str) -> List[ValidationIssue]:
        """Validate conditional requirement"""
        issues = []
        
        for idx, record in enumerate(data):
            if record.get(if_field) == if_value:
                if then_field not in record or record[then_field] is None:
                    issues.append(ValidationIssue(
                        rule_id="CONDITIONAL_001",
                        severity=ValidationSeverity.ERROR,
                        field=then_field,
                        message=f"If {if_field}={if_value}, then {then_field} is required",
                        record_index=idx,
                        actual_value=None
                    ))
        
        return issues

# Create validator instances
constraint_validator = ConstraintValidator()
format_validator = FormatValidator()
cross_field_validator = CrossFieldValidator()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def validate_schema(state: DataValidationState) -> DataValidationState:
    """Validate schema"""
    input_data = state["input_data"]
    
    # Define expected schema
    schema = {
        "id": {"type": "int", "required": True, "nullable": False},
        "email": {"type": "str", "required": True, "nullable": False},
        "age": {"type": "int", "required": True, "nullable": False},
        "status": {"type": "str", "required": False, "nullable": True}
    }
    
    validator = SchemaValidator(schema)
    issues = validator.validate(input_data)
    
    return {
        "schema_validation_results": {
            "issues": [vars(issue) for issue in issues],
            "total_issues": len(issues),
            "errors": sum(1 for i in issues if i.severity == ValidationSeverity.ERROR)
        },
        "messages": [f"[Schema] Found {len(issues)} schema validation issues"]
    }

def validate_constraints(state: DataValidationState) -> DataValidationState:
    """Validate constraints"""
    input_data = state["input_data"]
    all_issues = []
    
    # Age range
    all_issues.extend(constraint_validator.validate_range(input_data, "age", 0, 150))
    
    # Status enum
    all_issues.extend(constraint_validator.validate_enum(
        input_data, "status", ["active", "inactive", "pending"]
    ))
    
    # ID uniqueness
    all_issues.extend(constraint_validator.validate_uniqueness(input_data, "id"))
    
    return {
        "constraint_validation_results": {
            "issues": [vars(issue) for issue in all_issues],
            "total_issues": len(all_issues)
        },
        "messages": [f"[Constraints] Found {len(all_issues)} constraint violations"]
    }

def validate_formats(state: DataValidationState) -> DataValidationState:
    """Validate formats"""
    input_data = state["input_data"]
    all_issues = []
    
    # Email format
    all_issues.extend(format_validator.validate_pattern(input_data, "email", "email"))
    
    return {
        "format_validation_results": {
            "issues": [vars(issue) for issue in all_issues],
            "total_issues": len(all_issues)
        },
        "messages": [f"[Formats] Found {len(all_issues)} format issues"]
    }

def validate_cross_fields(state: DataValidationState) -> DataValidationState:
    """Validate cross-field rules"""
    input_data = state["input_data"]
    all_issues = []
    
    # No specific cross-field rules in this example
    # Example: end_date > start_date
    
    return {
        "cross_field_validation_results": {
            "issues": [vars(issue) for issue in all_issues],
            "total_issues": len(all_issues)
        },
        "messages": [f"[Cross-Field] Found {len(all_issues)} cross-field issues"]
    }

def summarize_validation(state: DataValidationState) -> DataValidationState:
    """Summarize validation results"""
    schema_results = state.get("schema_validation_results", {})
    constraint_results = state.get("constraint_validation_results", {})
    format_results = state.get("format_validation_results", {})
    cross_field_results = state.get("cross_field_validation_results", {})
    
    total_issues = (
        schema_results.get("total_issues", 0) +
        constraint_results.get("total_issues", 0) +
        format_results.get("total_issues", 0) +
        cross_field_results.get("total_issues", 0)
    )
    
    is_valid = total_issues == 0
    
    return {
        "validation_summary": {
            "is_valid": is_valid,
            "total_issues": total_issues,
            "schema_issues": schema_results.get("total_issues", 0),
            "constraint_issues": constraint_results.get("total_issues", 0),
            "format_issues": format_results.get("total_issues", 0),
            "cross_field_issues": cross_field_results.get("total_issues", 0)
        },
        "messages": [f"[Summary] Validation {'PASSED' if is_valid else 'FAILED'} - {total_issues} total issues"]
    }

# ============================================================================
# BUILD THE VALIDATION GRAPH
# ============================================================================

def create_validation_graph():
    """
    Create a StateGraph demonstrating data validation pattern.
    
    Flow:
    1. Schema Validation: Check structure
    2. Constraint Validation: Verify business rules
    3. Format Validation: Match patterns
    4. Cross-field Validation: Field relationships
    5. Summary: Aggregate results
    """
    
    workflow = StateGraph(DataValidationState)
    
    # Add validation nodes
    workflow.add_node("schema", validate_schema)
    workflow.add_node("constraints", validate_constraints)
    workflow.add_node("formats", validate_formats)
    workflow.add_node("cross_fields", validate_cross_fields)
    workflow.add_node("summary", summarize_validation)
    
    # Define validation flow
    workflow.add_edge(START, "schema")
    workflow.add_edge("schema", "constraints")
    workflow.add_edge("constraints", "formats")
    workflow.add_edge("formats", "cross_fields")
    workflow.add_edge("cross_fields", "summary")
    workflow.add_edge("summary", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Validation MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Comprehensive Validation
    print("\n" + "=" * 80)
    print("Example 1: Multi-Level Data Validation")
    print("=" * 80)
    
    validation_graph = create_validation_graph()
    
    # Sample data with various issues
    input_data = [
        {"id": 1, "email": "john@example.com", "age": 30, "status": "active"},  # Valid
        {"id": 2, "email": "invalid-email", "age": 200, "status": "unknown"},  # Multiple issues
        {"id": 1, "email": "jane@example.com", "age": 25, "status": "inactive"},  # Duplicate ID
        # Missing required field 'email'
        {"id": 4, "age": -5, "status": "pending"}  # Negative age, missing email
    ]
    
    initial_state: DataValidationState = {
        "input_data": input_data,
        "schema_validation_results": None,
        "constraint_validation_results": None,
        "format_validation_results": None,
        "cross_field_validation_results": None,
        "validation_summary": {},
        "messages": []
    }
    
    result = validation_graph.invoke(initial_state)
    
    print("\nValidation Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nValidation Summary:")
    summary = result["validation_summary"]
    print(f"  Overall Status: {'✓ PASSED' if summary['is_valid'] else '✗ FAILED'}")
    print(f"  Total Issues: {summary['total_issues']}")
    print(f"    Schema Issues: {summary['schema_issues']}")
    print(f"    Constraint Issues: {summary['constraint_issues']}")
    print(f"    Format Issues: {summary['format_issues']}")
    print(f"    Cross-field Issues: {summary['cross_field_issues']}")
    
    # Show some issues
    print("\nSample Issues:")
    schema_issues = result["schema_validation_results"]["issues"][:3]
    for issue in schema_issues:
        print(f"  [{issue['severity']}] Record {issue['record_index']}: {issue['message']}")
    
    # Example 2: Individual Validators
    print("\n" + "=" * 80)
    print("Example 2: Individual Validation Types")
    print("=" * 80)
    
    test_data = [
        {"price": 50, "quantity": 10},
        {"price": -10, "quantity": 5},  # Negative price
        {"price": 100, "quantity": 0}
    ]
    
    print("\n1. Range Validation:")
    issues = constraint_validator.validate_range(test_data, "price", min_value=0)
    print(f"   Found {len(issues)} range violations")
    if issues:
        print(f"   Example: {issues[0].message}")
    
    print("\n2. Enum Validation:")
    status_data = [{"status": "active"}, {"status": "invalid"}, {"status": "pending"}]
    issues = constraint_validator.validate_enum(status_data, "status", ["active", "inactive", "pending"])
    print(f"   Found {len(issues)} enum violations")
    if issues:
        print(f"   Example: {issues[0].message}")
    
    print("\n3. Format Validation:")
    email_data = [
        {"email": "valid@example.com"},
        {"email": "invalid-email"},
        {"email": "also.valid@test.co"}
    ]
    issues = format_validator.validate_pattern(email_data, "email", "email")
    print(f"   Found {len(issues)} format violations")
    if issues:
        print(f"   Example: {issues[0].message}")
    
    print("\n4. Uniqueness Validation:")
    id_data = [{"id": 1}, {"id": 2}, {"id": 1}, {"id": 3}]
    issues = constraint_validator.validate_uniqueness(id_data, "id")
    print(f"   Found {len(issues)} uniqueness violations")
    if issues:
        print(f"   Example: {issues[0].message}")
    
    # Example 3: Cross-field Validation
    print("\n" + "=" * 80)
    print("Example 3: Cross-Field Validation")
    print("=" * 80)
    
    date_data = [
        {"start_date": "2024-01-01", "end_date": "2024-01-31"},  # Valid
        {"start_date": "2024-02-01", "end_date": "2024-01-15"}   # Invalid: end before start
    ]
    
    # Note: This example uses string comparison for simplicity
    print("\n1. Date Comparison (end_date > start_date):")
    issues = cross_field_validator.validate_comparison(date_data, "end_date", "start_date", ">")
    print(f"   Found {len(issues)} violations")
    if issues:
        print(f"   Example: {issues[0].message}")
    
    print("\n2. Conditional Validation:")
    conditional_data = [
        {"type": "premium", "discount": 10},  # Valid
        {"type": "premium"},  # Invalid: premium requires discount
        {"type": "standard"}  # Valid: standard doesn't require discount
    ]
    issues = cross_field_validator.validate_conditional(
        conditional_data, "type", "premium", "discount"
    )
    print(f"   Found {len(issues)} violations")
    if issues:
        print(f"   Example: {issues[0].message}")
    
    print("\n" + "=" * 80)
    print("Data Validation Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Validation ensures data quality before processing
  • Schema validation: Check structure and types
  • Constraint validation: Enforce business rules (range, enum, uniqueness)
  • Format validation: Match patterns (email, phone, URL)
  • Cross-field validation: Verify field relationships
  • Multi-level approach: Schema → Constraints → Formats → Cross-field
  • Severity levels: ERROR (reject), WARNING (accept with warning), INFO
  • Strategies: Fail fast vs collect all errors, auto-correction, quarantine
  • Tools: JSON Schema, Cerberus, Great Expectations, Apache Griffin
  • Best practices: Validate early, clear error messages, log issues
    """)

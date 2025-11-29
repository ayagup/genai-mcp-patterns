"""
Constraint Validation MCP Pattern

This pattern demonstrates constraint validation in an agentic MCP system.
Constraint validation ensures data satisfies specific conditions, dependencies,
and integrity rules across multiple fields and entities.

Use cases:
- Database integrity constraints
- Referential integrity
- Uniqueness constraints
- Cross-field validation
- Temporal constraints
- State machine validation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Set
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime


# Define the state for constraint validation
class ConstraintValidationState(TypedDict):
    """State for tracking constraint validation"""
    messages: Annotated[List[str], add]
    data_records: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    constraint_violations: List[Dict[str, Any]]
    dependency_graph: Dict[str, List[str]]
    report: str


class ConstraintValidator:
    """Validate various types of constraints"""
    
    def __init__(self):
        # Mock database for referential integrity checks
        self.existing_customers = {'CUST001', 'CUST002', 'CUST003'}
        self.existing_products = {'PROD001', 'PROD002', 'PROD003', 'PROD004'}
        self.used_emails = {'john@example.com', 'jane@example.com'}
        self.used_order_ids = {'ORD001', 'ORD002'}
    
    def validate_uniqueness(self, records: List[Dict], field: str) -> List[Dict]:
        """Validate uniqueness constraint"""
        violations = []
        seen_values = set(self.used_order_ids if field == 'order_id' else self.used_emails)
        
        for idx, record in enumerate(records):
            value = record.get(field)
            if value in seen_values:
                violations.append({
                    'record_index': idx,
                    'constraint_type': 'UNIQUENESS',
                    'field': field,
                    'value': value,
                    'message': f"Duplicate {field}: {value}"
                })
            else:
                seen_values.add(value)
        
        return violations
    
    def validate_referential_integrity(self, records: List[Dict], 
                                      field: str, reference_table: str) -> List[Dict]:
        """Validate foreign key constraints"""
        violations = []
        
        reference_data = {
            'customers': self.existing_customers,
            'products': self.existing_products
        }
        
        valid_refs = reference_data.get(reference_table, set())
        
        for idx, record in enumerate(records):
            value = record.get(field)
            
            if value and value not in valid_refs:
                violations.append({
                    'record_index': idx,
                    'constraint_type': 'REFERENTIAL_INTEGRITY',
                    'field': field,
                    'value': value,
                    'message': f"Invalid reference to {reference_table}: {value}"
                })
            elif not value:
                violations.append({
                    'record_index': idx,
                    'constraint_type': 'REFERENTIAL_INTEGRITY',
                    'field': field,
                    'value': None,
                    'message': f"Missing required reference to {reference_table}"
                })
        
        return violations
    
    def validate_not_null(self, records: List[Dict], field: str) -> List[Dict]:
        """Validate NOT NULL constraint"""
        violations = []
        
        for idx, record in enumerate(records):
            value = record.get(field)
            
            if value is None or (isinstance(value, str) and value.strip() == ''):
                violations.append({
                    'record_index': idx,
                    'constraint_type': 'NOT_NULL',
                    'field': field,
                    'value': value,
                    'message': f"Field {field} cannot be null or empty"
                })
        
        return violations
    
    def validate_check_constraint(self, records: List[Dict], 
                                  field: str, condition: callable) -> List[Dict]:
        """Validate CHECK constraint with custom condition"""
        violations = []
        
        for idx, record in enumerate(records):
            value = record.get(field)
            
            try:
                if not condition(value):
                    violations.append({
                        'record_index': idx,
                        'constraint_type': 'CHECK',
                        'field': field,
                        'value': value,
                        'message': f"Field {field} violates check constraint"
                    })
            except Exception as e:
                violations.append({
                    'record_index': idx,
                    'constraint_type': 'CHECK',
                    'field': field,
                    'value': value,
                    'message': f"Error evaluating constraint: {str(e)}"
                })
        
        return violations
    
    def validate_cross_field_constraint(self, records: List[Dict],
                                       condition: callable,
                                       description: str) -> List[Dict]:
        """Validate cross-field constraints"""
        violations = []
        
        for idx, record in enumerate(records):
            try:
                if not condition(record):
                    violations.append({
                        'record_index': idx,
                        'constraint_type': 'CROSS_FIELD',
                        'fields': 'multiple',
                        'message': f"Cross-field constraint violated: {description}"
                    })
            except Exception as e:
                violations.append({
                    'record_index': idx,
                    'constraint_type': 'CROSS_FIELD',
                    'fields': 'multiple',
                    'message': f"Error evaluating cross-field constraint: {str(e)}"
                })
        
        return violations
    
    def validate_temporal_constraint(self, records: List[Dict],
                                    start_field: str, end_field: str) -> List[Dict]:
        """Validate temporal constraints (start < end)"""
        violations = []
        
        for idx, record in enumerate(records):
            start = record.get(start_field)
            end = record.get(end_field)
            
            if start and end:
                try:
                    start_dt = datetime.fromisoformat(start)
                    end_dt = datetime.fromisoformat(end)
                    
                    if start_dt >= end_dt:
                        violations.append({
                            'record_index': idx,
                            'constraint_type': 'TEMPORAL',
                            'fields': f"{start_field}, {end_field}",
                            'message': f"{start_field} must be before {end_field}"
                        })
                except ValueError:
                    violations.append({
                        'record_index': idx,
                        'constraint_type': 'TEMPORAL',
                        'fields': f"{start_field}, {end_field}",
                        'message': f"Invalid date format in temporal fields"
                    })
        
        return violations


class DependencyAnalyzer:
    """Analyze field dependencies"""
    
    def build_dependency_graph(self, constraints: List[Dict]) -> Dict[str, List[str]]:
        """Build dependency graph from constraints"""
        graph = {}
        
        for constraint in constraints:
            constraint_type = constraint.get('type')
            
            if constraint_type == 'REFERENTIAL_INTEGRITY':
                field = constraint.get('field')
                reference = constraint.get('reference_table')
                
                if field not in graph:
                    graph[field] = []
                graph[field].append(f"{reference} (foreign key)")
            
            elif constraint_type == 'CROSS_FIELD':
                fields = constraint.get('fields', [])
                for field in fields:
                    if field not in graph:
                        graph[field] = []
                    deps = [f for f in fields if f != field]
                    graph[field].extend(deps)
        
        return graph


# Agent functions
def initialize_constraint_validation_agent(state: ConstraintValidationState) -> ConstraintValidationState:
    """Initialize with sample data records"""
    
    # Sample order records with various constraint violations
    data_records = [
        {
            'record_id': 1,
            'order_id': 'ORD003',
            'customer_id': 'CUST001',  # Valid
            'order_date': '2024-03-15T10:00:00',
            'ship_date': '2024-03-20T10:00:00',
            'total': 1000.00,
            'discount': 100.00,  # Valid: discount < total
            'status': 'pending',
            'email': 'alice@example.com'
        },
        {
            'record_id': 2,
            'order_id': 'ORD003',  # Duplicate order_id
            'customer_id': 'CUST999',  # Invalid customer reference
            'order_date': '2024-03-16T14:00:00',
            'ship_date': '2024-03-15T10:00:00',  # ship_date before order_date
            'total': 500.00,
            'discount': 600.00,  # Invalid: discount > total
            'status': 'invalid_status',
            'email': 'john@example.com'  # Duplicate email
        },
        {
            'record_id': 3,
            'order_id': None,  # NULL violation
            'customer_id': None,  # Missing required reference
            'order_date': '2024-03-17T09:00:00',
            'ship_date': None,
            'total': -100.00,  # Negative total
            'discount': 0.00,
            'status': 'pending',
            'email': None  # NULL violation
        },
        {
            'record_id': 4,
            'order_id': 'ORD004',
            'customer_id': 'CUST002',  # Valid
            'order_date': '2024-03-18T11:00:00',
            'ship_date': '2024-03-22T11:00:00',
            'total': 2500.00,
            'discount': 250.00,
            'status': 'processing',
            'email': 'bob@example.com'
        }
    ]
    
    return {
        **state,
        'data_records': data_records,
        'messages': state['messages'] + [f'Initialized with {len(data_records)} data records']
    }


def define_constraints_agent(state: ConstraintValidationState) -> ConstraintValidationState:
    """Define constraints to validate"""
    
    constraints = [
        {
            'constraint_id': 'C001',
            'type': 'UNIQUENESS',
            'field': 'order_id',
            'description': 'Order ID must be unique'
        },
        {
            'constraint_id': 'C002',
            'type': 'UNIQUENESS',
            'field': 'email',
            'description': 'Email must be unique'
        },
        {
            'constraint_id': 'C003',
            'type': 'NOT_NULL',
            'field': 'order_id',
            'description': 'Order ID cannot be null'
        },
        {
            'constraint_id': 'C004',
            'type': 'NOT_NULL',
            'field': 'customer_id',
            'description': 'Customer ID cannot be null'
        },
        {
            'constraint_id': 'C005',
            'type': 'NOT_NULL',
            'field': 'email',
            'description': 'Email cannot be null'
        },
        {
            'constraint_id': 'C006',
            'type': 'REFERENTIAL_INTEGRITY',
            'field': 'customer_id',
            'reference_table': 'customers',
            'description': 'Customer ID must reference existing customer'
        },
        {
            'constraint_id': 'C007',
            'type': 'CHECK',
            'field': 'total',
            'condition': 'total > 0',
            'description': 'Total must be positive'
        },
        {
            'constraint_id': 'C008',
            'type': 'CHECK',
            'field': 'status',
            'allowed_values': ['pending', 'processing', 'shipped', 'delivered', 'cancelled'],
            'description': 'Status must be valid'
        },
        {
            'constraint_id': 'C009',
            'type': 'CROSS_FIELD',
            'fields': ['total', 'discount'],
            'condition': 'discount <= total',
            'description': 'Discount cannot exceed total'
        },
        {
            'constraint_id': 'C010',
            'type': 'TEMPORAL',
            'start_field': 'order_date',
            'end_field': 'ship_date',
            'description': 'Ship date must be after order date'
        }
    ]
    
    return {
        **state,
        'constraints': constraints,
        'messages': state['messages'] + [f'Defined {len(constraints)} constraints']
    }


def validate_constraints_agent(state: ConstraintValidationState) -> ConstraintValidationState:
    """Validate all constraints"""
    
    validator = ConstraintValidator()
    all_violations = []
    constraint_results = []
    
    for constraint in state['constraints']:
        constraint_type = constraint['type']
        violations = []
        
        if constraint_type == 'UNIQUENESS':
            violations = validator.validate_uniqueness(
                state['data_records'],
                constraint['field']
            )
        
        elif constraint_type == 'NOT_NULL':
            violations = validator.validate_not_null(
                state['data_records'],
                constraint['field']
            )
        
        elif constraint_type == 'REFERENTIAL_INTEGRITY':
            violations = validator.validate_referential_integrity(
                state['data_records'],
                constraint['field'],
                constraint['reference_table']
            )
        
        elif constraint_type == 'CHECK':
            field = constraint['field']
            if field == 'total':
                condition = lambda x: x is not None and x > 0
            elif field == 'status':
                allowed = constraint.get('allowed_values', [])
                condition = lambda x: x in allowed
            else:
                condition = lambda x: True
            
            violations = validator.validate_check_constraint(
                state['data_records'],
                field,
                condition
            )
        
        elif constraint_type == 'CROSS_FIELD':
            condition = lambda record: (
                record.get('discount', 0) <= record.get('total', 0)
            )
            violations = validator.validate_cross_field_constraint(
                state['data_records'],
                condition,
                constraint['description']
            )
        
        elif constraint_type == 'TEMPORAL':
            violations = validator.validate_temporal_constraint(
                state['data_records'],
                constraint['start_field'],
                constraint['end_field']
            )
        
        # Add constraint info to violations
        for violation in violations:
            violation['constraint_id'] = constraint['constraint_id']
            violation['constraint_description'] = constraint['description']
        
        all_violations.extend(violations)
        
        constraint_results.append({
            'constraint_id': constraint['constraint_id'],
            'constraint_type': constraint_type,
            'violations_found': len(violations),
            'is_satisfied': len(violations) == 0
        })
    
    records_with_violations = len(set(v['record_index'] for v in all_violations))
    
    return {
        **state,
        'validation_results': constraint_results,
        'constraint_violations': all_violations,
        'messages': state['messages'] + [
            f'Validated {len(state["constraints"])} constraints: '
            f'{len(all_violations)} violations in {records_with_violations} records'
        ]
    }


def analyze_dependencies_agent(state: ConstraintValidationState) -> ConstraintValidationState:
    """Analyze field dependencies"""
    
    analyzer = DependencyAnalyzer()
    dependency_graph = analyzer.build_dependency_graph(state['constraints'])
    
    return {
        **state,
        'dependency_graph': dependency_graph,
        'messages': state['messages'] + [
            f'Analyzed dependencies: {len(dependency_graph)} fields have dependencies'
        ]
    }


def generate_constraint_report_agent(state: ConstraintValidationState) -> ConstraintValidationState:
    """Generate comprehensive constraint validation report"""
    
    report_lines = [
        "=" * 80,
        "CONSTRAINT VALIDATION REPORT",
        "=" * 80,
        "",
        "VALIDATION SUMMARY:",
        "-" * 40,
        f"Total records validated: {len(state['data_records'])}",
        f"Total constraints checked: {len(state['constraints'])}",
        f"Total violations found: {len(state['constraint_violations'])}",
        f"Records with violations: {len(set(v['record_index'] for v in state['constraint_violations']))}",
        "",
        "CONSTRAINTS DEFINED:",
        "-" * 40
    ]
    
    for constraint in state['constraints']:
        report_lines.append(
            f"  {constraint['constraint_id']}: {constraint['type']} - {constraint['description']}"
        )
    
    report_lines.extend([
        "",
        "CONSTRAINT VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ SATISFIED" if result['is_satisfied'] else "✗ VIOLATED"
        report_lines.append(
            f"{result['constraint_id']} ({result['constraint_type']}): {status}"
        )
        if not result['is_satisfied']:
            report_lines.append(f"  Violations: {result['violations_found']}")
    
    report_lines.extend([
        "",
        "DETAILED VIOLATIONS:",
        "-" * 40
    ])
    
    # Group violations by record
    violations_by_record = {}
    for violation in state['constraint_violations']:
        rec_idx = violation['record_index']
        if rec_idx not in violations_by_record:
            violations_by_record[rec_idx] = []
        violations_by_record[rec_idx].append(violation)
    
    for rec_idx, violations in sorted(violations_by_record.items()):
        record = state['data_records'][rec_idx]
        report_lines.append(f"\nRecord #{rec_idx + 1} (ID: {record.get('order_id', 'N/A')}):")
        report_lines.append(f"  Violations: {len(violations)}")
        for violation in violations:
            report_lines.append(
                f"    ✗ [{violation['constraint_type']}] {violation['message']}"
            )
    
    report_lines.extend([
        "",
        "FIELD DEPENDENCIES:",
        "-" * 40
    ])
    
    if state['dependency_graph']:
        for field, deps in state['dependency_graph'].items():
            report_lines.append(f"{field} depends on:")
            for dep in deps:
                report_lines.append(f"    → {dep}")
    else:
        report_lines.append("No cross-field dependencies detected")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40
    ])
    
    if len(state['constraint_violations']) > 0:
        report_lines.append("⚠ Constraint violations detected:")
        report_lines.append("  • Review and fix data before processing")
        report_lines.append("  • Add validation at data entry points")
        report_lines.append("  • Implement database constraints")
        report_lines.append("  • Set up automated validation checks")
    
    report_lines.extend([
        "",
        "BEST PRACTICES:",
        "-" * 40,
        "✓ Define constraints at database level",
        "✓ Validate data before insert/update operations",
        "✓ Use transactions for maintaining referential integrity",
        "✓ Implement cascading updates/deletes carefully",
        "✓ Monitor constraint violations in production",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive constraint validation report']
    }


# Create the graph
def create_constraint_validation_graph():
    """Create the constraint validation workflow graph"""
    
    workflow = StateGraph(ConstraintValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_constraint_validation_agent)
    workflow.add_node("define_constraints", define_constraints_agent)
    workflow.add_node("validate", validate_constraints_agent)
    workflow.add_node("analyze_dependencies", analyze_dependencies_agent)
    workflow.add_node("generate_report", generate_constraint_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_constraints")
    workflow.add_edge("define_constraints", "validate")
    workflow.add_edge("validate", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the constraint validation graph
    app = create_constraint_validation_graph()
    
    # Initialize state
    initial_state: ConstraintValidationState = {
        'messages': [],
        'data_records': [],
        'constraints': [],
        'validation_results': [],
        'constraint_violations': [],
        'dependency_graph': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("CONSTRAINT VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nConstraint validation pattern execution complete! ✓")

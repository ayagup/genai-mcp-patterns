"""
Cross-Field Validation MCP Pattern

This pattern demonstrates cross-field validation in an agentic MCP system.
Cross-field validation ensures multiple fields in a record are consistent
with each other, enforcing complex multi-field business rules.

Use cases:
- Multi-field dependency validation
- Conditional field requirements
- Mutual exclusivity validation
- Field combination rules
- Complex business constraints
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta


# Define the state for cross-field validation
class CrossFieldValidationState(TypedDict):
    """State for tracking cross-field validation process"""
    messages: Annotated[List[str], add]
    records: List[Dict[str, Any]]
    validation_rules: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]]
    statistics: Dict[str, Any]
    report: str


class CrossFieldValidator:
    """Validator for cross-field business rules"""
    
    def validate_conditional_required(self, record: Dict[str, Any], 
                                     condition_field: str,
                                     condition_value: Any,
                                     required_fields: List[str]) -> tuple[bool, Optional[str]]:
        """Validate that fields are required when condition is met"""
        if record.get(condition_field) == condition_value:
            for field in required_fields:
                if field not in record or record[field] is None or record[field] == '':
                    return False, f"Field '{field}' is required when {condition_field}={condition_value}"
        return True, None
    
    def validate_mutual_exclusivity(self, record: Dict[str, Any],
                                   exclusive_fields: List[str]) -> tuple[bool, Optional[str]]:
        """Validate that only one of the exclusive fields is present"""
        present_fields = [f for f in exclusive_fields 
                         if f in record and record[f] is not None and record[f] != '']
        
        if len(present_fields) == 0:
            return False, f"At least one of {exclusive_fields} must be provided"
        elif len(present_fields) > 1:
            return False, f"Only one of {exclusive_fields} can be provided, found: {present_fields}"
        
        return True, None
    
    def validate_field_dependency(self, record: Dict[str, Any],
                                  parent_field: str,
                                  child_fields: List[str]) -> tuple[bool, Optional[str]]:
        """Validate that child fields are only present if parent field is present"""
        parent_present = parent_field in record and record[parent_field] is not None and record[parent_field] != ''
        
        for child in child_fields:
            child_present = child in record and record[child] is not None and record[child] != ''
            
            if child_present and not parent_present:
                return False, f"Field '{child}' requires '{parent_field}' to be present"
        
        return True, None
    
    def validate_at_least_one(self, record: Dict[str, Any],
                             required_group: List[str]) -> tuple[bool, Optional[str]]:
        """Validate that at least one field from the group is present"""
        present = [f for f in required_group 
                  if f in record and record[f] is not None and record[f] != '']
        
        if len(present) == 0:
            return False, f"At least one of {required_group} must be provided"
        
        return True, None
    
    def validate_field_combination(self, record: Dict[str, Any],
                                   field1: str, value1: Any,
                                   field2: str, allowed_values2: List[Any]) -> tuple[bool, Optional[str]]:
        """Validate field combinations are allowed"""
        if record.get(field1) == value1:
            value2 = record.get(field2)
            if value2 not in allowed_values2:
                return False, f"When {field1}={value1}, {field2} must be one of {allowed_values2}, got {value2}"
        
        return True, None
    
    def validate_range_consistency(self, record: Dict[str, Any],
                                   min_field: str, max_field: str) -> tuple[bool, Optional[str]]:
        """Validate min/max field consistency"""
        min_val = record.get(min_field)
        max_val = record.get(max_field)
        
        if min_val is not None and max_val is not None:
            try:
                if float(min_val) > float(max_val):
                    return False, f"{min_field} ({min_val}) cannot be greater than {max_field} ({max_val})"
            except (ValueError, TypeError):
                return False, f"Cannot compare {min_field} and {max_field} values"
        
        return True, None
    
    def validate_date_range_consistency(self, record: Dict[str, Any],
                                       start_field: str, end_field: str) -> tuple[bool, Optional[str]]:
        """Validate date range consistency"""
        start = record.get(start_field)
        end = record.get(end_field)
        
        if start and end:
            try:
                if isinstance(start, str):
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                else:
                    start_dt = start
                
                if isinstance(end, str):
                    end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                else:
                    end_dt = end
                
                if start_dt >= end_dt:
                    return False, f"{start_field} must be before {end_field}"
            except (ValueError, AttributeError) as e:
                return False, f"Invalid date format: {str(e)}"
        
        return True, None
    
    def validate_percentage_sum(self, record: Dict[str, Any],
                               percentage_fields: List[str],
                               expected_sum: float = 100.0,
                               tolerance: float = 0.1) -> tuple[bool, Optional[str]]:
        """Validate percentage fields sum to expected value"""
        values = []
        for field in percentage_fields:
            val = record.get(field)
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    return False, f"Field '{field}' is not a valid number"
        
        if values:
            total = sum(values)
            if abs(total - expected_sum) > tolerance:
                return False, f"Percentage fields sum to {total}, expected {expected_sum}"
        
        return True, None
    
    def validate_total_equals_sum(self, record: Dict[str, Any],
                                  total_field: str,
                                  component_fields: List[str],
                                  tolerance: float = 0.01) -> tuple[bool, Optional[str]]:
        """Validate total field equals sum of components"""
        total = record.get(total_field)
        if total is None:
            return True, None
        
        try:
            total_val = float(total)
        except (ValueError, TypeError):
            return False, f"Total field '{total_field}' is not a valid number"
        
        component_sum = 0.0
        for field in component_fields:
            val = record.get(field)
            if val is not None:
                try:
                    component_sum += float(val)
                except (ValueError, TypeError):
                    return False, f"Component field '{field}' is not a valid number"
        
        if abs(total_val - component_sum) > tolerance:
            return False, f"{total_field} ({total_val}) doesn't equal sum of components ({component_sum})"
        
        return True, None


# Agent functions
def initialize_cross_field_validation_agent(state: CrossFieldValidationState) -> CrossFieldValidationState:
    """Initialize with sample records"""
    
    today = datetime.now()
    
    records = [
        {
            'id': 1,
            'order_type': 'shipping',  # Requires shipping_address
            'shipping_address': '123 Main St',
            'pickup_location': None,  # Mutually exclusive with shipping
            'payment_method': 'credit_card',
            'card_number': '4111111111111111',
            'bank_account': None,  # Mutually exclusive
            'min_quantity': 1,
            'max_quantity': 10,
            'start_date': (today - timedelta(days=1)).isoformat(),
            'end_date': (today + timedelta(days=30)).isoformat(),
            'subtotal': 80.00,
            'tax': 10.00,
            'shipping': 5.00,
            'total': 95.00,
            'discount_a': 50,
            'discount_b': 30,
            'discount_c': 20
        },
        {
            'id': 2,
            'order_type': 'shipping',
            'shipping_address': None,  # VIOLATION: Required for shipping
            'pickup_location': None,
            'payment_method': 'credit_card',
            'card_number': None,  # VIOLATION: Required for credit_card
            'bank_account': None,
            'min_quantity': 20,
            'max_quantity': 10,  # VIOLATION: min > max
            'start_date': (today + timedelta(days=5)).isoformat(),
            'end_date': (today + timedelta(days=1)).isoformat(),  # VIOLATION: end before start
            'subtotal': 100.00,
            'tax': 10.00,
            'shipping': 5.00,
            'total': 200.00,  # VIOLATION: Doesn't match sum
            'discount_a': 40,
            'discount_b': 40,
            'discount_c': 40  # VIOLATION: Sum exceeds 100%
        },
        {
            'id': 3,
            'order_type': 'pickup',
            'shipping_address': '456 Oak Ave',  # VIOLATION: Shouldn't have both
            'pickup_location': 'Store #5',
            'payment_method': 'bank_transfer',
            'card_number': None,
            'bank_account': 'ACC123456',
            'min_quantity': 5,
            'max_quantity': 50,
            'start_date': (today).isoformat(),
            'end_date': (today + timedelta(days=15)).isoformat(),
            'subtotal': 200.00,
            'tax': 25.00,
            'shipping': 0.00,
            'total': 225.00,
            'discount_a': 60,
            'discount_b': 25,
            'discount_c': 15
        },
        {
            'id': 4,
            'order_type': 'pickup',
            'shipping_address': None,
            'pickup_location': 'Store #3',
            'payment_method': 'cash',
            'card_number': None,
            'bank_account': None,
            'min_quantity': 1,
            'max_quantity': 100,
            'start_date': (today - timedelta(days=2)).isoformat(),
            'end_date': (today + timedelta(days=7)).isoformat(),
            'subtotal': 150.00,
            'tax': 15.00,
            'shipping': 0.00,
            'total': 165.00,
            'discount_a': 33.33,
            'discount_b': 33.33,
            'discount_c': 33.34
        }
    ]
    
    return {
        **state,
        'records': records,
        'messages': state['messages'] + [f'Initialized with {len(records)} records']
    }


def define_validation_rules_agent(state: CrossFieldValidationState) -> CrossFieldValidationState:
    """Define cross-field validation rules"""
    
    validation_rules = [
        {
            'rule_id': 'CF001',
            'name': 'Shipping Address Required',
            'type': 'conditional_required',
            'description': 'Shipping address required when order_type is shipping',
            'params': {
                'condition_field': 'order_type',
                'condition_value': 'shipping',
                'required_fields': ['shipping_address']
            }
        },
        {
            'rule_id': 'CF002',
            'name': 'Pickup Location Required',
            'type': 'conditional_required',
            'description': 'Pickup location required when order_type is pickup',
            'params': {
                'condition_field': 'order_type',
                'condition_value': 'pickup',
                'required_fields': ['pickup_location']
            }
        },
        {
            'rule_id': 'CF003',
            'name': 'Delivery Method Mutual Exclusivity',
            'type': 'mutual_exclusivity',
            'description': 'Only one delivery method (shipping_address or pickup_location) allowed',
            'params': {
                'exclusive_fields': ['shipping_address', 'pickup_location']
            }
        },
        {
            'rule_id': 'CF004',
            'name': 'Payment Method Details',
            'type': 'conditional_required',
            'description': 'Card number required for credit_card payment',
            'params': {
                'condition_field': 'payment_method',
                'condition_value': 'credit_card',
                'required_fields': ['card_number']
            }
        },
        {
            'rule_id': 'CF005',
            'name': 'Bank Account for Transfers',
            'type': 'conditional_required',
            'description': 'Bank account required for bank_transfer payment',
            'params': {
                'condition_field': 'payment_method',
                'condition_value': 'bank_transfer',
                'required_fields': ['bank_account']
            }
        },
        {
            'rule_id': 'CF006',
            'name': 'Payment Mutual Exclusivity',
            'type': 'mutual_exclusivity',
            'description': 'Only one payment method details allowed',
            'params': {
                'exclusive_fields': ['card_number', 'bank_account']
            }
        },
        {
            'rule_id': 'CF007',
            'name': 'Quantity Range Consistency',
            'type': 'range_consistency',
            'description': 'Min quantity must be less than max quantity',
            'params': {
                'min_field': 'min_quantity',
                'max_field': 'max_quantity'
            }
        },
        {
            'rule_id': 'CF008',
            'name': 'Date Range Consistency',
            'type': 'date_range_consistency',
            'description': 'Start date must be before end date',
            'params': {
                'start_field': 'start_date',
                'end_field': 'end_date'
            }
        },
        {
            'rule_id': 'CF009',
            'name': 'Total Equals Components',
            'type': 'total_equals_sum',
            'description': 'Total must equal subtotal + tax + shipping',
            'params': {
                'total_field': 'total',
                'component_fields': ['subtotal', 'tax', 'shipping']
            }
        },
        {
            'rule_id': 'CF010',
            'name': 'Discount Percentages Sum',
            'type': 'percentage_sum',
            'description': 'Discount percentages must sum to 100',
            'params': {
                'percentage_fields': ['discount_a', 'discount_b', 'discount_c'],
                'expected_sum': 100.0
            }
        }
    ]
    
    # Build dependency graph
    dependencies = {
        'shipping_address': ['order_type'],
        'pickup_location': ['order_type'],
        'card_number': ['payment_method'],
        'bank_account': ['payment_method'],
        'max_quantity': ['min_quantity'],
        'end_date': ['start_date'],
        'total': ['subtotal', 'tax', 'shipping']
    }
    
    return {
        **state,
        'validation_rules': validation_rules,
        'dependencies': dependencies,
        'messages': state['messages'] + [f'Defined {len(validation_rules)} cross-field rules']
    }


def validate_cross_fields_agent(state: CrossFieldValidationState) -> CrossFieldValidationState:
    """Validate cross-field rules"""
    
    validator = CrossFieldValidator()
    validation_results = []
    all_violations = []
    
    for record in state['records']:
        record_id = record.get('id')
        violations = []
        
        for rule in state['validation_rules']:
            is_valid = False
            error = None
            
            if rule['type'] == 'conditional_required':
                is_valid, error = validator.validate_conditional_required(
                    record,
                    rule['params']['condition_field'],
                    rule['params']['condition_value'],
                    rule['params']['required_fields']
                )
            elif rule['type'] == 'mutual_exclusivity':
                is_valid, error = validator.validate_mutual_exclusivity(
                    record,
                    rule['params']['exclusive_fields']
                )
            elif rule['type'] == 'range_consistency':
                is_valid, error = validator.validate_range_consistency(
                    record,
                    rule['params']['min_field'],
                    rule['params']['max_field']
                )
            elif rule['type'] == 'date_range_consistency':
                is_valid, error = validator.validate_date_range_consistency(
                    record,
                    rule['params']['start_field'],
                    rule['params']['end_field']
                )
            elif rule['type'] == 'total_equals_sum':
                is_valid, error = validator.validate_total_equals_sum(
                    record,
                    rule['params']['total_field'],
                    rule['params']['component_fields']
                )
            elif rule['type'] == 'percentage_sum':
                is_valid, error = validator.validate_percentage_sum(
                    record,
                    rule['params']['percentage_fields'],
                    rule['params'].get('expected_sum', 100.0)
                )
            
            if not is_valid:
                violations.append({
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'description': rule['description'],
                    'error': error
                })
        
        validation_results.append({
            'record_id': record_id,
            'is_valid': len(violations) == 0,
            'rules_checked': len(state['validation_rules']),
            'violations_count': len(violations)
        })
        
        for violation in violations:
            all_violations.append({
                'record_id': record_id,
                **violation
            })
    
    valid_count = sum(1 for r in validation_results if r['is_valid'])
    
    return {
        **state,
        'validation_results': validation_results,
        'violations': all_violations,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} records: {valid_count} valid, '
            f'{len(validation_results) - valid_count} with violations'
        ]
    }


def analyze_dependencies_agent(state: CrossFieldValidationState) -> CrossFieldValidationState:
    """Analyze field dependencies"""
    
    # Count violations by dependency
    dependency_violations = {}
    for violation in state['violations']:
        # Extract fields from error message
        rule_id = violation['rule_id']
        rule = next((r for r in state['validation_rules'] if r['rule_id'] == rule_id), None)
        
        if rule:
            params = rule.get('params', {})
            # Count based on rule type
            if 'condition_field' in params:
                field = params['condition_field']
                dependency_violations[field] = dependency_violations.get(field, 0) + 1
    
    return {
        **state,
        'messages': state['messages'] + [f'Analyzed {len(state["dependencies"])} field dependencies']
    }


def calculate_statistics_agent(state: CrossFieldValidationState) -> CrossFieldValidationState:
    """Calculate validation statistics"""
    
    # Violations by rule
    violations_by_rule = {}
    for violation in state['violations']:
        rule_id = violation['rule_id']
        violations_by_rule[rule_id] = violations_by_rule.get(rule_id, 0) + 1
    
    # Violations by type
    violations_by_type = {}
    for violation in state['violations']:
        rule = next((r for r in state['validation_rules'] 
                    if r['rule_id'] == violation['rule_id']), None)
        if rule:
            rule_type = rule['type']
            violations_by_type[rule_type] = violations_by_type.get(rule_type, 0) + 1
    
    statistics = {
        'total_records': len(state['records']),
        'valid_records': sum(1 for r in state['validation_results'] if r['is_valid']),
        'invalid_records': sum(1 for r in state['validation_results'] if not r['is_valid']),
        'total_violations': len(state['violations']),
        'violations_by_rule': violations_by_rule,
        'violations_by_type': violations_by_type,
        'most_violated_rule': max(violations_by_rule.items(), key=lambda x: x[1])[0] if violations_by_rule else None,
        'pass_rate': (sum(1 for r in state['validation_results'] if r['is_valid']) / 
                     len(state['records']) * 100) if state['records'] else 0
    }
    
    return {
        **state,
        'statistics': statistics,
        'messages': state['messages'] + ['Calculated validation statistics']
    }


def generate_cross_field_report_agent(state: CrossFieldValidationState) -> CrossFieldValidationState:
    """Generate comprehensive cross-field validation report"""
    
    report_lines = [
        "=" * 80,
        "CROSS-FIELD VALIDATION REPORT",
        "=" * 80,
        "",
        "VALIDATION SUMMARY:",
        "-" * 40,
        f"Total records: {state['statistics']['total_records']}",
        f"Valid records: {state['statistics']['valid_records']}",
        f"Invalid records: {state['statistics']['invalid_records']}",
        f"Total violations: {state['statistics']['total_violations']}",
        f"Pass rate: {state['statistics']['pass_rate']:.1f}%",
        "",
        "CROSS-FIELD RULES:",
        "-" * 40
    ]
    
    for rule in state['validation_rules']:
        report_lines.append(f"• {rule['rule_id']}: {rule['name']}")
        report_lines.append(f"  Type: {rule['type']}")
        report_lines.append(f"  {rule['description']}")
    
    report_lines.extend([
        "",
        "FIELD DEPENDENCIES:",
        "-" * 40
    ])
    
    for field, depends_on in state['dependencies'].items():
        report_lines.append(f"• {field} depends on: {', '.join(depends_on)}")
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        report_lines.append(f"Record #{result['record_id']}: {status}")
        if not result['is_valid']:
            report_lines.append(f"  Violations: {result['violations_count']}/{result['rules_checked']}")
    
    report_lines.extend([
        "",
        "VIOLATIONS BY RULE:",
        "-" * 40
    ])
    
    for rule_id, count in sorted(state['statistics']['violations_by_rule'].items(),
                                 key=lambda x: x[1], reverse=True):
        rule = next((r for r in state['validation_rules'] if r['rule_id'] == rule_id), None)
        rule_name = rule['name'] if rule else rule_id
        report_lines.append(f"  {rule_id} ({rule_name}): {count} violations")
    
    report_lines.extend([
        "",
        "VIOLATIONS BY TYPE:",
        "-" * 40
    ])
    
    for vtype, count in sorted(state['statistics']['violations_by_type'].items(),
                               key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {vtype}: {count} violations")
    
    report_lines.extend([
        "",
        "DETAILED VIOLATIONS:",
        "-" * 40
    ])
    
    # Group by record
    violations_by_record = {}
    for violation in state['violations']:
        record_id = violation['record_id']
        if record_id not in violations_by_record:
            violations_by_record[record_id] = []
        violations_by_record[record_id].append(violation)
    
    for record_id, violations in sorted(violations_by_record.items()):
        report_lines.append(f"\nRecord #{record_id}:")
        for violation in violations:
            report_lines.append(f"  ✗ {violation['rule_id']}: {violation['rule_name']}")
            report_lines.append(f"     Error: {violation['error']}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most violated rule: {state['statistics']['most_violated_rule']}",
        f"✓ Overall pass rate: {state['statistics']['pass_rate']:.1f}%",
        f"✓ Total field dependencies: {len(state['dependencies'])}",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement dependency-aware form validation",
        "• Use conditional field requirements in UI",
        "• Validate cross-field rules before submission",
        "• Provide clear error messages for field dependencies",
        "• Consider using form libraries with cross-field validation",
        "• Document field dependencies in API schemas",
        "• Implement cascading validation (validate dependencies first)",
        "• Use database constraints for critical cross-field rules",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive cross-field validation report']
    }


# Create the graph
def create_cross_field_validation_graph():
    """Create the cross-field validation workflow graph"""
    
    workflow = StateGraph(CrossFieldValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_cross_field_validation_agent)
    workflow.add_node("define_rules", define_validation_rules_agent)
    workflow.add_node("validate", validate_cross_fields_agent)
    workflow.add_node("analyze_dependencies", analyze_dependencies_agent)
    workflow.add_node("calculate_stats", calculate_statistics_agent)
    workflow.add_node("generate_report", generate_cross_field_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "validate")
    workflow.add_edge("validate", "analyze_dependencies")
    workflow.add_edge("analyze_dependencies", "calculate_stats")
    workflow.add_edge("calculate_stats", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the cross-field validation graph
    app = create_cross_field_validation_graph()
    
    # Initialize state
    initial_state: CrossFieldValidationState = {
        'messages': [],
        'records': [],
        'validation_rules': [],
        'validation_results': [],
        'violations': [],
        'dependencies': {},
        'statistics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("CROSS-FIELD VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nCross-field validation pattern execution complete! ✓")

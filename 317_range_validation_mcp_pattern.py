"""
Range Validation MCP Pattern

This pattern demonstrates range validation in an agentic MCP system.
Range validation ensures numeric values, dates, and other orderable types
fall within acceptable bounds.

Use cases:
- Numeric range validation
- Date range validation
- String length validation
- Collection size validation
- Custom boundary validation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from decimal import Decimal


# Define the state for range validation
class RangeValidationState(TypedDict):
    """State for tracking range validation process"""
    messages: Annotated[List[str], add]
    test_data: List[Dict[str, Any]]
    range_rules: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    report: str


class RangeValidator:
    """Comprehensive range validator for various data types"""
    
    def validate_numeric_range(self, value: Any, min_value: Optional[float] = None,
                              max_value: Optional[float] = None,
                              inclusive_min: bool = True,
                              inclusive_max: bool = True) -> tuple[bool, Optional[str]]:
        """Validate numeric ranges"""
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            return False, f"Value '{value}' is not numeric"
        
        if min_value is not None:
            if inclusive_min and num_value < min_value:
                return False, f"Value {num_value} is less than minimum {min_value}"
            elif not inclusive_min and num_value <= min_value:
                return False, f"Value {num_value} is not greater than {min_value}"
        
        if max_value is not None:
            if inclusive_max and num_value > max_value:
                return False, f"Value {num_value} exceeds maximum {max_value}"
            elif not inclusive_max and num_value >= max_value:
                return False, f"Value {num_value} is not less than {max_value}"
        
        return True, None
    
    def validate_date_range(self, value: Any, min_date: Optional[datetime] = None,
                           max_date: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
        """Validate date ranges"""
        if isinstance(value, str):
            try:
                date_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                return False, f"Invalid date format: {value}"
        elif isinstance(value, datetime):
            date_value = value
        else:
            return False, f"Value is not a date: {type(value).__name__}"
        
        if min_date and date_value < min_date:
            return False, f"Date {date_value.date()} is before minimum {min_date.date()}"
        
        if max_date and date_value > max_date:
            return False, f"Date {date_value.date()} is after maximum {max_date.date()}"
        
        return True, None
    
    def validate_string_length(self, value: str, min_length: Optional[int] = None,
                              max_length: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """Validate string length ranges"""
        if not isinstance(value, str):
            return False, f"Value is not a string: {type(value).__name__}"
        
        length = len(value)
        
        if min_length is not None and length < min_length:
            return False, f"String length {length} is less than minimum {min_length}"
        
        if max_length is not None and length > max_length:
            return False, f"String length {length} exceeds maximum {max_length}"
        
        return True, None
    
    def validate_collection_size(self, value: Any, min_size: Optional[int] = None,
                                max_size: Optional[int] = None) -> tuple[bool, Optional[str]]:
        """Validate collection size ranges"""
        try:
            size = len(value)
        except TypeError:
            return False, f"Value does not have a length: {type(value).__name__}"
        
        if min_size is not None and size < min_size:
            return False, f"Collection size {size} is less than minimum {min_size}"
        
        if max_size is not None and size > max_size:
            return False, f"Collection size {size} exceeds maximum {max_size}"
        
        return True, None
    
    def validate_percentage_range(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate percentage values (0-100)"""
        return self.validate_numeric_range(value, 0, 100, inclusive_min=True, inclusive_max=True)
    
    def validate_probability_range(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate probability values (0-1)"""
        return self.validate_numeric_range(value, 0, 1, inclusive_min=True, inclusive_max=True)


class BoundaryConditionTester:
    """Test boundary conditions for ranges"""
    
    def test_boundaries(self, min_value: float, max_value: float,
                       test_values: List[float]) -> List[Dict[str, Any]]:
        """Test values at and around boundaries"""
        validator = RangeValidator()
        results = []
        
        for value in test_values:
            is_valid, error = validator.validate_numeric_range(value, min_value, max_value)
            results.append({
                'value': value,
                'is_valid': is_valid,
                'error': error,
                'position': self._categorize_position(value, min_value, max_value)
            })
        
        return results
    
    def _categorize_position(self, value: float, min_value: float, max_value: float) -> str:
        """Categorize value position relative to range"""
        if value < min_value:
            return 'below_min'
        elif value == min_value:
            return 'at_min'
        elif value > max_value:
            return 'above_max'
        elif value == max_value:
            return 'at_max'
        else:
            return 'in_range'


# Agent functions
def initialize_range_validation_agent(state: RangeValidationState) -> RangeValidationState:
    """Initialize with test data"""
    
    # Current date for date range validation
    today = datetime.now()
    past_date = today - timedelta(days=365)
    future_date = today + timedelta(days=365)
    
    test_data = [
        {
            'id': 1,
            'age': 25,  # Valid: 0-120
            'score': 85.5,  # Valid: 0-100
            'username': 'john_doe',  # Valid: 3-50 chars
            'items': [1, 2, 3, 4, 5],  # Valid: 1-10 items
            'discount': 15.0,  # Valid: 0-50%
            'probability': 0.75,  # Valid: 0-1
            'start_date': (today - timedelta(days=30)).isoformat(),  # Valid
            'end_date': (today + timedelta(days=30)).isoformat()  # Valid
        },
        {
            'id': 2,
            'age': -5,  # Invalid: negative age
            'score': 105,  # Invalid: exceeds max
            'username': 'ab',  # Invalid: too short
            'items': [],  # Invalid: empty list
            'discount': 75,  # Invalid: exceeds max
            'probability': 1.5,  # Invalid: exceeds max
            'start_date': (today - timedelta(days=400)).isoformat(),  # Invalid: too old
            'end_date': (today - timedelta(days=31)).isoformat()  # Invalid: in past
        },
        {
            'id': 3,
            'age': 150,  # Invalid: exceeds max
            'score': -10,  # Invalid: below min
            'username': 'a' * 100,  # Invalid: too long
            'items': list(range(20)),  # Invalid: too many items
            'discount': -5,  # Invalid: negative
            'probability': 0.0,  # Valid: boundary value
            'start_date': (today + timedelta(days=400)).isoformat(),  # Invalid: too far future
            'end_date': (today + timedelta(days=500)).isoformat()  # Invalid: too far future
        },
        {
            'id': 4,
            'age': 0,  # Valid: boundary value
            'score': 100,  # Valid: boundary value
            'username': 'bob',  # Valid
            'items': [1],  # Valid: minimum size
            'discount': 50,  # Valid: boundary value
            'probability': 1.0,  # Valid: boundary value
            'start_date': today.isoformat(),  # Valid
            'end_date': (today + timedelta(days=1)).isoformat()  # Valid
        }
    ]
    
    return {
        **state,
        'test_data': test_data,
        'messages': state['messages'] + [f'Initialized with {len(test_data)} test records']
    }


def define_range_rules_agent(state: RangeValidationState) -> RangeValidationState:
    """Define range validation rules"""
    
    today = datetime.now()
    
    range_rules = [
        {
            'field': 'age',
            'type': 'numeric',
            'min': 0,
            'max': 120,
            'description': 'Age must be between 0 and 120 years'
        },
        {
            'field': 'score',
            'type': 'numeric',
            'min': 0,
            'max': 100,
            'description': 'Score must be between 0 and 100'
        },
        {
            'field': 'username',
            'type': 'string_length',
            'min': 3,
            'max': 50,
            'description': 'Username must be 3-50 characters'
        },
        {
            'field': 'items',
            'type': 'collection_size',
            'min': 1,
            'max': 10,
            'description': 'Items list must contain 1-10 elements'
        },
        {
            'field': 'discount',
            'type': 'percentage',
            'min': 0,
            'max': 50,
            'description': 'Discount must be 0-50%'
        },
        {
            'field': 'probability',
            'type': 'probability',
            'min': 0,
            'max': 1,
            'description': 'Probability must be 0-1'
        },
        {
            'field': 'start_date',
            'type': 'date',
            'min': (today - timedelta(days=365)).isoformat(),
            'max': (today + timedelta(days=365)).isoformat(),
            'description': 'Start date must be within 1 year of today'
        },
        {
            'field': 'end_date',
            'type': 'date',
            'min': today.isoformat(),
            'max': (today + timedelta(days=365)).isoformat(),
            'description': 'End date must be in future within 1 year'
        }
    ]
    
    return {
        **state,
        'range_rules': range_rules,
        'messages': state['messages'] + [f'Defined {len(range_rules)} range validation rules']
    }


def validate_ranges_agent(state: RangeValidationState) -> RangeValidationState:
    """Validate all ranges"""
    
    validator = RangeValidator()
    validation_results = []
    all_violations = []
    
    for record in state['test_data']:
        record_id = record.get('id')
        violations = []
        
        for rule in state['range_rules']:
            field = rule['field']
            value = record.get(field)
            
            if value is None:
                violations.append({
                    'field': field,
                    'rule': rule['description'],
                    'error': 'Field is missing'
                })
                continue
            
            is_valid = False
            error = None
            
            if rule['type'] == 'numeric':
                is_valid, error = validator.validate_numeric_range(
                    value, rule.get('min'), rule.get('max')
                )
            elif rule['type'] == 'string_length':
                is_valid, error = validator.validate_string_length(
                    value, rule.get('min'), rule.get('max')
                )
            elif rule['type'] == 'collection_size':
                is_valid, error = validator.validate_collection_size(
                    value, rule.get('min'), rule.get('max')
                )
            elif rule['type'] == 'percentage':
                is_valid, error = validator.validate_percentage_range(value)
            elif rule['type'] == 'probability':
                is_valid, error = validator.validate_probability_range(value)
            elif rule['type'] == 'date':
                min_date = datetime.fromisoformat(rule['min']) if rule.get('min') else None
                max_date = datetime.fromisoformat(rule['max']) if rule.get('max') else None
                is_valid, error = validator.validate_date_range(value, min_date, max_date)
            
            if not is_valid:
                violations.append({
                    'field': field,
                    'rule': rule['description'],
                    'value': str(value)[:50],
                    'error': error
                })
        
        validation_results.append({
            'record_id': record_id,
            'is_valid': len(violations) == 0,
            'rules_checked': len(state['range_rules']),
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


def test_boundary_conditions_agent(state: RangeValidationState) -> RangeValidationState:
    """Test boundary conditions"""
    
    tester = BoundaryConditionTester()
    
    # Test age boundaries (0-120)
    test_values = [-1, 0, 1, 119, 120, 121]
    boundary_results = tester.test_boundaries(0, 120, test_values)
    
    boundary_summary = {
        'field': 'age',
        'min': 0,
        'max': 120,
        'test_values': len(test_values),
        'valid_count': sum(1 for r in boundary_results if r['is_valid']),
        'invalid_count': sum(1 for r in boundary_results if not r['is_valid']),
        'boundary_positions': {
            'below_min': sum(1 for r in boundary_results if r['position'] == 'below_min'),
            'at_min': sum(1 for r in boundary_results if r['position'] == 'at_min'),
            'in_range': sum(1 for r in boundary_results if r['position'] == 'in_range'),
            'at_max': sum(1 for r in boundary_results if r['position'] == 'at_max'),
            'above_max': sum(1 for r in boundary_results if r['position'] == 'above_max')
        }
    }
    
    return {
        **state,
        'messages': state['messages'] + [
            f'Tested boundary conditions: {boundary_summary["valid_count"]}/{len(test_values)} valid'
        ]
    }


def analyze_violations_agent(state: RangeValidationState) -> RangeValidationState:
    """Analyze range violations"""
    
    # Violations by field
    violations_by_field = {}
    for violation in state['violations']:
        field = violation['field']
        violations_by_field[field] = violations_by_field.get(field, 0) + 1
    
    # Violations by record
    violations_by_record = {}
    for violation in state['violations']:
        record_id = violation['record_id']
        violations_by_record[record_id] = violations_by_record.get(record_id, 0) + 1
    
    statistics = {
        'total_records': len(state['test_data']),
        'valid_records': sum(1 for r in state['validation_results'] if r['is_valid']),
        'invalid_records': sum(1 for r in state['validation_results'] if not r['is_valid']),
        'total_violations': len(state['violations']),
        'violations_by_field': violations_by_field,
        'violations_by_record': violations_by_record,
        'most_violated_field': max(violations_by_field.items(), key=lambda x: x[1])[0] if violations_by_field else None,
        'pass_rate': (sum(1 for r in state['validation_results'] if r['is_valid']) / 
                     len(state['test_data']) * 100) if state['test_data'] else 0
    }
    
    return {
        **state,
        'statistics': statistics,
        'messages': state['messages'] + ['Analyzed range violations']
    }


def generate_range_report_agent(state: RangeValidationState) -> RangeValidationState:
    """Generate comprehensive range validation report"""
    
    report_lines = [
        "=" * 80,
        "RANGE VALIDATION REPORT",
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
        "RANGE RULES:",
        "-" * 40
    ]
    
    for rule in state['range_rules']:
        report_lines.append(f"• {rule['field']}: {rule['description']}")
        if rule['type'] in ('numeric', 'string_length', 'collection_size', 'percentage', 'probability'):
            report_lines.append(f"  Range: [{rule.get('min', 'N/A')}, {rule.get('max', 'N/A')}]")
    
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
        "VIOLATIONS BY FIELD:",
        "-" * 40
    ])
    
    for field, count in sorted(state['statistics']['violations_by_field'].items(),
                               key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {field}: {count} violations")
    
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
            report_lines.append(f"  ✗ {violation['field']}: {violation['error']}")
            report_lines.append(f"     Value: {violation.get('value', 'N/A')}")
            report_lines.append(f"     Rule: {violation['rule']}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most violated field: {state['statistics']['most_violated_field']}",
        f"✓ Pass rate: {state['statistics']['pass_rate']:.1f}%",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement client-side range validation for better UX",
        "• Add detailed error messages indicating valid ranges",
        "• Consider using step validation for numeric inputs",
        "• Implement boundary value testing in QA",
        "• Document valid ranges in API specifications",
        "• Use type-safe languages with range types where possible",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive range validation report']
    }


# Create the graph
def create_range_validation_graph():
    """Create the range validation workflow graph"""
    
    workflow = StateGraph(RangeValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_range_validation_agent)
    workflow.add_node("define_rules", define_range_rules_agent)
    workflow.add_node("validate", validate_ranges_agent)
    workflow.add_node("test_boundaries", test_boundary_conditions_agent)
    workflow.add_node("analyze", analyze_violations_agent)
    workflow.add_node("generate_report", generate_range_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "validate")
    workflow.add_edge("validate", "test_boundaries")
    workflow.add_edge("test_boundaries", "analyze")
    workflow.add_edge("analyze", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the range validation graph
    app = create_range_validation_graph()
    
    # Initialize state
    initial_state: RangeValidationState = {
        'messages': [],
        'test_data': [],
        'range_rules': [],
        'validation_results': [],
        'violations': [],
        'statistics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("RANGE VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nRange validation pattern execution complete! ✓")

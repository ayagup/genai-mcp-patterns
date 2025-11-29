"""
Input Validation MCP Pattern

This pattern demonstrates input validation in an agentic MCP system.
Input validation ensures that data received from external sources meets
expected criteria before processing, preventing errors and security issues.

Use cases:
- API request validation
- Form data validation
- User input sanitization
- Data integrity checks
- Security hardening
- Error prevention
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
import re


# Define the state for input validation
class InputValidationState(TypedDict):
    """State for tracking input validation process"""
    messages: Annotated[List[str], add]
    raw_inputs: List[Dict[str, Any]]
    validation_rules: Dict[str, List[Dict[str, Any]]]
    validated_inputs: List[Dict[str, Any]]
    validation_errors: List[Dict[str, Any]]
    sanitized_inputs: List[Dict[str, Any]]
    validation_summary: Dict[str, Any]
    report: str


class InputValidator:
    """Class to handle various input validation operations"""
    
    def __init__(self):
        self.validators = {
            'required': self._validate_required,
            'type': self._validate_type,
            'min_length': self._validate_min_length,
            'max_length': self._validate_max_length,
            'pattern': self._validate_pattern,
            'email': self._validate_email,
            'url': self._validate_url,
            'range': self._validate_range,
            'enum': self._validate_enum,
            'custom': self._validate_custom
        }
    
    def _validate_required(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate that a value is present"""
        if value is None or (isinstance(value, str) and value.strip() == ''):
            return False, "Field is required"
        return True, None
    
    def _validate_type(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate value type"""
        expected_type = rule.get('expected_type')
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        expected_class = type_map.get(expected_type)
        if expected_class and not isinstance(value, expected_class):
            return False, f"Expected type {expected_type}, got {type(value).__name__}"
        return True, None
    
    def _validate_min_length(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate minimum length"""
        min_len = rule.get('min')
        if isinstance(value, (str, list)) and len(value) < min_len:
            return False, f"Minimum length is {min_len}, got {len(value)}"
        return True, None
    
    def _validate_max_length(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate maximum length"""
        max_len = rule.get('max')
        if isinstance(value, (str, list)) and len(value) > max_len:
            return False, f"Maximum length is {max_len}, got {len(value)}"
        return True, None
    
    def _validate_pattern(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate against regex pattern"""
        pattern = rule.get('pattern')
        if isinstance(value, str) and pattern:
            if not re.match(pattern, value):
                return False, f"Value does not match pattern {pattern}"
        return True, None
    
    def _validate_email(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate email format"""
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if isinstance(value, str):
            if not re.match(email_pattern, value):
                return False, "Invalid email format"
        return True, None
    
    def _validate_url(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate URL format"""
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if isinstance(value, str):
            if not re.match(url_pattern, value):
                return False, "Invalid URL format"
        return True, None
    
    def _validate_range(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate numeric range"""
        min_val = rule.get('min')
        max_val = rule.get('max')
        
        if isinstance(value, (int, float)):
            if min_val is not None and value < min_val:
                return False, f"Value must be >= {min_val}"
            if max_val is not None and value > max_val:
                return False, f"Value must be <= {max_val}"
        return True, None
    
    def _validate_enum(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Validate against allowed values"""
        allowed = rule.get('allowed', [])
        if value not in allowed:
            return False, f"Value must be one of {allowed}"
        return True, None
    
    def _validate_custom(self, value: Any, rule: Dict) -> tuple[bool, Optional[str]]:
        """Custom validation function"""
        validator_func = rule.get('validator')
        if validator_func and callable(validator_func):
            return validator_func(value)
        return True, None
    
    def validate_field(self, field_name: str, value: Any, 
                      rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate a single field against multiple rules"""
        errors = []
        
        for rule in rules:
            rule_type = rule.get('type')
            validator = self.validators.get(rule_type)
            
            if validator:
                is_valid, error_msg = validator(value, rule)
                if not is_valid:
                    errors.append({
                        'field': field_name,
                        'rule': rule_type,
                        'message': error_msg,
                        'value': str(value)[:50]  # Truncate long values
                    })
        
        return errors
    
    def validate_input(self, input_data: Dict[str, Any], 
                      validation_rules: Dict[str, List[Dict]]) -> tuple[bool, List[Dict]]:
        """Validate entire input against all rules"""
        all_errors = []
        
        for field_name, rules in validation_rules.items():
            value = input_data.get(field_name)
            field_errors = self.validate_field(field_name, value, rules)
            all_errors.extend(field_errors)
        
        return len(all_errors) == 0, all_errors


class InputSanitizer:
    """Class to sanitize input data"""
    
    def sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return value
        
        # Remove leading/trailing whitespace
        sanitized = value.strip()
        
        # Remove multiple consecutive spaces
        sanitized = re.sub(r'\s+', ' ', sanitized)
        
        # Remove potentially dangerous characters for basic SQL injection prevention
        # Note: This is basic - use parameterized queries in production
        dangerous_chars = ['--', ';--', '/*', '*/', 'xp_', 'sp_']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        return sanitized
    
    def sanitize_html(self, value: str) -> str:
        """Remove HTML tags"""
        if not isinstance(value, str):
            return value
        
        # Remove HTML tags
        sanitized = re.sub(r'<[^>]+>', '', value)
        
        # Decode HTML entities (basic)
        sanitized = sanitized.replace('&lt;', '<').replace('&gt;', '>')
        sanitized = sanitized.replace('&amp;', '&').replace('&quot;', '"')
        
        return sanitized
    
    def sanitize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize entire input"""
        sanitized = {}
        
        for key, value in input_data.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_input(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self.sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        
        return sanitized


# Agent functions
def initialize_validation_agent(state: InputValidationState) -> InputValidationState:
    """Initialize validation with sample inputs"""
    
    # Sample user registration inputs (some valid, some invalid)
    raw_inputs = [
        {
            'username': 'john_doe',
            'email': 'john@example.com',
            'age': 25,
            'password': 'SecurePass123!',
            'website': 'https://johndoe.com',
            'country': 'USA'
        },
        {
            'username': 'ab',  # Too short
            'email': 'invalid-email',  # Invalid format
            'age': 150,  # Out of range
            'password': '123',  # Too short
            'website': 'not-a-url',  # Invalid URL
            'country': 'INVALID'  # Not in enum
        },
        {
            'username': '',  # Required field missing
            'email': 'jane@example.com',
            'age': -5,  # Negative age
            'password': None,  # Required field missing
            'website': 'http://jane.com',
            'country': 'Canada'
        },
        {
            'username': 'alice_wonderland',
            'email': 'alice@example.com',
            'age': 30,
            'password': 'AlicePass456!',
            'website': 'https://alice.dev',
            'country': 'UK'
        },
        {
            'username': 'bob<script>alert("xss")</script>',  # XSS attempt
            'email': 'bob@example.com',
            'age': 28,
            'password': 'BobPass789!',
            'website': 'https://bob.io',
            'country': 'USA'
        }
    ]
    
    return {
        **state,
        'raw_inputs': raw_inputs,
        'messages': state['messages'] + [f'Initialized with {len(raw_inputs)} input records to validate']
    }


def define_validation_rules_agent(state: InputValidationState) -> InputValidationState:
    """Define validation rules for each field"""
    
    validation_rules = {
        'username': [
            {'type': 'required'},
            {'type': 'type', 'expected_type': 'string'},
            {'type': 'min_length', 'min': 3},
            {'type': 'max_length', 'max': 20},
            {'type': 'pattern', 'pattern': r'^[a-zA-Z0-9_]+$'}
        ],
        'email': [
            {'type': 'required'},
            {'type': 'type', 'expected_type': 'string'},
            {'type': 'email'}
        ],
        'age': [
            {'type': 'required'},
            {'type': 'type', 'expected_type': 'integer'},
            {'type': 'range', 'min': 0, 'max': 120}
        ],
        'password': [
            {'type': 'required'},
            {'type': 'type', 'expected_type': 'string'},
            {'type': 'min_length', 'min': 8},
            {'type': 'pattern', 'pattern': r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]+$'}
        ],
        'website': [
            {'type': 'type', 'expected_type': 'string'},
            {'type': 'url'}
        ],
        'country': [
            {'type': 'required'},
            {'type': 'enum', 'allowed': ['USA', 'Canada', 'UK', 'Australia', 'Germany', 'France']}
        ]
    }
    
    return {
        **state,
        'validation_rules': validation_rules,
        'messages': state['messages'] + [f'Defined validation rules for {len(validation_rules)} fields']
    }


def validate_inputs_agent(state: InputValidationState) -> InputValidationState:
    """Validate all inputs against rules"""
    
    validator = InputValidator()
    validated_inputs = []
    all_errors = []
    
    for idx, input_data in enumerate(state['raw_inputs']):
        is_valid, errors = validator.validate_input(input_data, state['validation_rules'])
        
        validated_inputs.append({
            'input_id': idx + 1,
            'data': input_data,
            'is_valid': is_valid,
            'error_count': len(errors)
        })
        
        for error in errors:
            all_errors.append({
                'input_id': idx + 1,
                **error
            })
    
    valid_count = sum(1 for v in validated_inputs if v['is_valid'])
    invalid_count = len(validated_inputs) - valid_count
    
    return {
        **state,
        'validated_inputs': validated_inputs,
        'validation_errors': all_errors,
        'messages': state['messages'] + [
            f'Validated {len(validated_inputs)} inputs: {valid_count} valid, {invalid_count} invalid, {len(all_errors)} total errors'
        ]
    }


def sanitize_inputs_agent(state: InputValidationState) -> InputValidationState:
    """Sanitize all inputs"""
    
    sanitizer = InputSanitizer()
    sanitized_inputs = []
    
    for validated in state['validated_inputs']:
        sanitized_data = sanitizer.sanitize_input(validated['data'])
        
        sanitized_inputs.append({
            'input_id': validated['input_id'],
            'original': validated['data'],
            'sanitized': sanitized_data,
            'was_modified': sanitized_data != validated['data'],
            'is_valid': validated['is_valid']
        })
    
    modified_count = sum(1 for s in sanitized_inputs if s['was_modified'])
    
    return {
        **state,
        'sanitized_inputs': sanitized_inputs,
        'messages': state['messages'] + [
            f'Sanitized {len(sanitized_inputs)} inputs: {modified_count} were modified'
        ]
    }


def analyze_validation_results_agent(state: InputValidationState) -> InputValidationState:
    """Analyze validation results and generate summary"""
    
    total_inputs = len(state['validated_inputs'])
    valid_inputs = sum(1 for v in state['validated_inputs'] if v['is_valid'])
    invalid_inputs = total_inputs - valid_inputs
    total_errors = len(state['validation_errors'])
    
    # Count errors by field
    errors_by_field = {}
    for error in state['validation_errors']:
        field = error['field']
        errors_by_field[field] = errors_by_field.get(field, 0) + 1
    
    # Count errors by rule type
    errors_by_rule = {}
    for error in state['validation_errors']:
        rule = error['rule']
        errors_by_rule[rule] = errors_by_rule.get(rule, 0) + 1
    
    validation_summary = {
        'total_inputs': total_inputs,
        'valid_inputs': valid_inputs,
        'invalid_inputs': invalid_inputs,
        'validation_rate': (valid_inputs / total_inputs * 100) if total_inputs > 0 else 0,
        'total_errors': total_errors,
        'errors_by_field': errors_by_field,
        'errors_by_rule': errors_by_rule,
        'most_problematic_field': max(errors_by_field.items(), key=lambda x: x[1])[0] if errors_by_field else None,
        'most_common_error': max(errors_by_rule.items(), key=lambda x: x[1])[0] if errors_by_rule else None
    }
    
    return {
        **state,
        'validation_summary': validation_summary,
        'messages': state['messages'] + ['Generated validation summary statistics']
    }


def generate_validation_report_agent(state: InputValidationState) -> InputValidationState:
    """Generate comprehensive validation report"""
    
    report_lines = [
        "=" * 80,
        "INPUT VALIDATION REPORT",
        "=" * 80,
        "",
        "VALIDATION SUMMARY:",
        "-" * 40,
        f"Total inputs processed: {state['validation_summary']['total_inputs']}",
        f"Valid inputs: {state['validation_summary']['valid_inputs']}",
        f"Invalid inputs: {state['validation_summary']['invalid_inputs']}",
        f"Validation success rate: {state['validation_summary']['validation_rate']:.1f}%",
        f"Total validation errors: {state['validation_summary']['total_errors']}",
        "",
        "VALIDATION RULES DEFINED:",
        "-" * 40
    ]
    
    for field, rules in state['validation_rules'].items():
        report_lines.append(f"\n{field}:")
        for rule in rules:
            rule_desc = f"  - {rule['type']}"
            if 'min' in rule:
                rule_desc += f" (min: {rule['min']})"
            if 'max' in rule:
                rule_desc += f" (max: {rule['max']})"
            if 'pattern' in rule:
                rule_desc += f" (pattern: {rule['pattern'][:30]}...)"
            if 'allowed' in rule:
                rule_desc += f" (allowed: {rule['allowed']})"
            report_lines.append(rule_desc)
    
    report_lines.extend([
        "",
        "ERRORS BY FIELD:",
        "-" * 40
    ])
    
    for field, count in sorted(state['validation_summary']['errors_by_field'].items(), 
                               key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {field}: {count} errors")
    
    report_lines.extend([
        "",
        "ERRORS BY RULE TYPE:",
        "-" * 40
    ])
    
    for rule, count in sorted(state['validation_summary']['errors_by_rule'].items(), 
                             key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {rule}: {count} occurrences")
    
    report_lines.extend([
        "",
        "DETAILED VALIDATION ERRORS:",
        "-" * 40
    ])
    
    # Group errors by input
    errors_by_input = {}
    for error in state['validation_errors']:
        input_id = error['input_id']
        if input_id not in errors_by_input:
            errors_by_input[input_id] = []
        errors_by_input[input_id].append(error)
    
    for input_id, errors in sorted(errors_by_input.items()):
        report_lines.append(f"\nInput #{input_id}: {len(errors)} errors")
        for error in errors:
            report_lines.append(f"  ✗ {error['field']}: {error['message']}")
    
    report_lines.extend([
        "",
        "SANITIZATION RESULTS:",
        "-" * 40
    ])
    
    modified_inputs = [s for s in state['sanitized_inputs'] if s['was_modified']]
    report_lines.append(f"Inputs modified during sanitization: {len(modified_inputs)}")
    
    for sanitized in modified_inputs[:3]:  # Show first 3
        report_lines.append(f"\nInput #{sanitized['input_id']}:")
        for key in sanitized['original'].keys():
            original = str(sanitized['original'].get(key, ''))[:40]
            cleaned = str(sanitized['sanitized'].get(key, ''))[:40]
            if original != cleaned:
                report_lines.append(f"  {key}:")
                report_lines.append(f"    Before: {original}")
                report_lines.append(f"    After:  {cleaned}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most problematic field: {state['validation_summary']['most_problematic_field']}",
        f"✓ Most common error type: {state['validation_summary']['most_common_error']}",
        f"✓ Validation success rate: {state['validation_summary']['validation_rate']:.1f}%",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Provide clear validation messages to users",
        "• Implement client-side validation to reduce invalid submissions",
        "• Sanitize all inputs before processing",
        "• Log validation failures for security monitoring",
        "• Consider rate limiting to prevent abuse",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive validation report']
    }


# Create the graph
def create_input_validation_graph():
    """Create the input validation workflow graph"""
    
    workflow = StateGraph(InputValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_validation_agent)
    workflow.add_node("define_rules", define_validation_rules_agent)
    workflow.add_node("validate", validate_inputs_agent)
    workflow.add_node("sanitize", sanitize_inputs_agent)
    workflow.add_node("analyze", analyze_validation_results_agent)
    workflow.add_node("generate_report", generate_validation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "validate")
    workflow.add_edge("validate", "sanitize")
    workflow.add_edge("sanitize", "analyze")
    workflow.add_edge("analyze", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the input validation graph
    app = create_input_validation_graph()
    
    # Initialize state
    initial_state: InputValidationState = {
        'messages': [],
        'raw_inputs': [],
        'validation_rules': {},
        'validated_inputs': [],
        'validation_errors': [],
        'sanitized_inputs': [],
        'validation_summary': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("INPUT VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nInput validation pattern execution complete! ✓")

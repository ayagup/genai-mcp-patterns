"""
Format Validation MCP Pattern

This pattern demonstrates format validation in an agentic MCP system.
Format validation ensures strings match expected patterns like emails,
phone numbers, URLs, credit cards, postal codes, etc.

Use cases:
- Email format validation
- Phone number validation
- URL validation
- Credit card validation
- Postal code validation
- SSN/Tax ID validation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
import re


# Define the state for format validation
class FormatValidationState(TypedDict):
    """State for tracking format validation process"""
    messages: Annotated[List[str], add]
    input_data: List[Dict[str, Any]]
    format_patterns: Dict[str, Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    format_violations: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    report: str


class FormatValidator:
    """Comprehensive format validator using regex patterns"""
    
    # Common regex patterns
    PATTERNS = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'us_phone': r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$',
        'url': r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)$',
        'ipv4': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$',
        'ipv6': r'^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4})$',
        'credit_card': r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12})$',
        'us_zip': r'^\d{5}(?:-\d{4})?$',
        'ssn': r'^\d{3}-\d{2}-\d{4}$',
        'date_iso': r'^\d{4}-\d{2}-\d{2}$',
        'time_24h': r'^([01]?[0-9]|2[0-3]):[0-5][0-9]$',
        'hex_color': r'^#?([a-fA-F0-9]{6}|[a-fA-F0-9]{3})$',
        'username': r'^[a-zA-Z0-9_-]{3,16}$',
        'password_strong': r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$',
        'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        'mac_address': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'
    }
    
    def validate_format(self, value: str, format_type: str,
                       custom_pattern: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Validate string against format pattern"""
        if not isinstance(value, str):
            return False, f"Value is not a string: {type(value).__name__}"
        
        # Use custom pattern if provided, otherwise use predefined
        pattern = custom_pattern if custom_pattern else self.PATTERNS.get(format_type)
        
        if not pattern:
            return False, f"Unknown format type: {format_type}"
        
        try:
            if re.match(pattern, value.strip()):
                return True, None
            else:
                return False, f"Value does not match {format_type} format"
        except re.error as e:
            return False, f"Invalid regex pattern: {str(e)}"
    
    def validate_email(self, email: str) -> tuple[bool, Optional[str]]:
        """Validate email format with additional checks"""
        is_valid, error = self.validate_format(email, 'email')
        if not is_valid:
            return is_valid, error
        
        # Additional checks
        local, domain = email.rsplit('@', 1)
        
        if len(local) > 64:
            return False, "Local part exceeds 64 characters"
        
        if len(domain) > 253:
            return False, "Domain exceeds 253 characters"
        
        if '..' in email:
            return False, "Consecutive dots not allowed"
        
        return True, None
    
    def validate_phone(self, phone: str, country_code: str = 'US') -> tuple[bool, Optional[str]]:
        """Validate phone number format"""
        # Remove common separators for validation
        cleaned = re.sub(r'[-.\s()]', '', phone)
        
        if country_code == 'US':
            if len(cleaned) == 10 or (len(cleaned) == 11 and cleaned[0] == '1'):
                return self.validate_format(phone, 'us_phone')
            else:
                return False, "US phone must be 10 digits (or 11 with country code)"
        
        return False, f"Country code {country_code} not supported"
    
    def validate_credit_card(self, card_number: str) -> tuple[bool, Optional[str]]:
        """Validate credit card with Luhn algorithm"""
        # Remove spaces and dashes
        cleaned = re.sub(r'[\s-]', '', card_number)
        
        # Check format
        is_valid, error = self.validate_format(cleaned, 'credit_card')
        if not is_valid:
            return is_valid, error
        
        # Luhn algorithm
        digits = [int(d) for d in cleaned]
        checksum = 0
        
        for i in range(len(digits) - 2, -1, -2):
            digits[i] *= 2
            if digits[i] > 9:
                digits[i] -= 9
        
        checksum = sum(digits)
        
        if checksum % 10 != 0:
            return False, "Credit card fails Luhn check"
        
        return True, None
    
    def detect_card_type(self, card_number: str) -> str:
        """Detect credit card type"""
        cleaned = re.sub(r'[\s-]', '', card_number)
        
        if re.match(r'^4', cleaned):
            return 'Visa'
        elif re.match(r'^5[1-5]', cleaned):
            return 'MasterCard'
        elif re.match(r'^3[47]', cleaned):
            return 'American Express'
        elif re.match(r'^6(?:011|5)', cleaned):
            return 'Discover'
        else:
            return 'Unknown'


# Agent functions
def initialize_format_validation_agent(state: FormatValidationState) -> FormatValidationState:
    """Initialize with sample data"""
    
    input_data = [
        {
            'id': 1,
            'email': 'john.doe@example.com',
            'phone': '(555) 123-4567',
            'website': 'https://www.example.com',
            'ip_address': '192.168.1.1',
            'credit_card': '4532-1488-0343-6467',
            'zip_code': '12345',
            'ssn': '123-45-6789',
            'username': 'john_doe',
            'password': 'SecurePass123!',
            'color': '#FF5733'
        },
        {
            'id': 2,
            'email': 'invalid.email@',  # Invalid: missing domain
            'phone': '555-123',  # Invalid: too short
            'website': 'not-a-url',  # Invalid: no protocol
            'ip_address': '256.1.1.1',  # Invalid: octet > 255
            'credit_card': '1234-5678-9012-3456',  # Invalid: wrong format
            'zip_code': '1234',  # Invalid: too short
            'ssn': '12-345-6789',  # Invalid: wrong format
            'username': 'a',  # Invalid: too short
            'password': 'weak',  # Invalid: not strong enough
            'color': 'red'  # Invalid: not hex format
        },
        {
            'id': 3,
            'email': 'alice@company.co.uk',
            'phone': '+1-555-987-6543',
            'website': 'http://subdomain.example.org/path?query=value',
            'ip_address': '10.0.0.255',
            'credit_card': '5425233430109903',  # MasterCard
            'zip_code': '98765-4321',
            'ssn': '987-65-4321',
            'username': 'alice_wonderland',
            'password': 'C0mpl3x!Pass',
            'color': '#3498db'
        },
        {
            'id': 4,
            'email': 'test@domain',  # Invalid: no TLD
            'phone': '5551234567',
            'website': 'https://example.com/page',
            'ip_address': '2001:0db8:85a3:0000:0000:8a2e:0370:7334',  # IPv6
            'credit_card': '378282246310005',  # Amex
            'zip_code': '00000',
            'ssn': '000-00-0000',  # Invalid: all zeros
            'username': 'user@name',  # Invalid: contains @
            'password': 'P@ssw0rd!Strong',
            'color': 'abc'  # Valid: 3-char hex
        }
    ]
    
    return {
        **state,
        'input_data': input_data,
        'messages': state['messages'] + [f'Initialized with {len(input_data)} records']
    }


def define_format_patterns_agent(state: FormatValidationState) -> FormatValidationState:
    """Define format validation patterns"""
    
    format_patterns = {
        'email': {
            'pattern_type': 'email',
            'description': 'Valid email address format',
            'examples': ['user@example.com', 'name.surname@company.co.uk']
        },
        'phone': {
            'pattern_type': 'us_phone',
            'description': 'US phone number (10 digits)',
            'examples': ['(555) 123-4567', '555-123-4567', '5551234567']
        },
        'website': {
            'pattern_type': 'url',
            'description': 'Valid HTTP/HTTPS URL',
            'examples': ['https://example.com', 'http://www.site.org/path']
        },
        'ip_address': {
            'pattern_type': 'ipv4',
            'description': 'IPv4 address format',
            'examples': ['192.168.1.1', '10.0.0.1']
        },
        'credit_card': {
            'pattern_type': 'credit_card',
            'description': 'Credit card number (with Luhn check)',
            'examples': ['4532148803436467', '5425233430109903']
        },
        'zip_code': {
            'pattern_type': 'us_zip',
            'description': 'US ZIP code (5 or 9 digits)',
            'examples': ['12345', '12345-6789']
        },
        'ssn': {
            'pattern_type': 'ssn',
            'description': 'US Social Security Number',
            'examples': ['123-45-6789']
        },
        'username': {
            'pattern_type': 'username',
            'description': 'Username (3-16 alphanumeric, dash, underscore)',
            'examples': ['john_doe', 'user-123']
        },
        'password': {
            'pattern_type': 'password_strong',
            'description': 'Strong password (8+ chars, upper, lower, digit, special)',
            'examples': ['SecurePass123!', 'C0mpl3x!Pass']
        },
        'color': {
            'pattern_type': 'hex_color',
            'description': 'Hex color code',
            'examples': ['#FF5733', '#abc', 'FF5733']
        }
    }
    
    return {
        **state,
        'format_patterns': format_patterns,
        'messages': state['messages'] + [f'Defined {len(format_patterns)} format patterns']
    }


def validate_formats_agent(state: FormatValidationState) -> FormatValidationState:
    """Validate all formats"""
    
    validator = FormatValidator()
    validation_results = []
    all_violations = []
    
    for record in state['input_data']:
        record_id = record.get('id')
        violations = []
        
        for field, pattern_info in state['format_patterns'].items():
            value = record.get(field)
            
            if value is None:
                violations.append({
                    'field': field,
                    'pattern': pattern_info['description'],
                    'error': 'Field is missing'
                })
                continue
            
            # Special handling for specific fields
            if field == 'email':
                is_valid, error = validator.validate_email(value)
            elif field == 'phone':
                is_valid, error = validator.validate_phone(value)
            elif field == 'credit_card':
                is_valid, error = validator.validate_credit_card(value)
                if is_valid:
                    card_type = validator.detect_card_type(value)
            else:
                is_valid, error = validator.validate_format(value, pattern_info['pattern_type'])
            
            if not is_valid:
                violations.append({
                    'field': field,
                    'pattern': pattern_info['description'],
                    'value': str(value)[:50],
                    'error': error
                })
        
        validation_results.append({
            'record_id': record_id,
            'is_valid': len(violations) == 0,
            'fields_checked': len(state['format_patterns']),
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
        'format_violations': all_violations,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} records: {valid_count} valid, '
            f'{len(validation_results) - valid_count} with format violations'
        ]
    }


def analyze_format_violations_agent(state: FormatValidationState) -> FormatValidationState:
    """Analyze format validation violations"""
    
    # Violations by field
    violations_by_field = {}
    for violation in state['format_violations']:
        field = violation['field']
        violations_by_field[field] = violations_by_field.get(field, 0) + 1
    
    # Common error types
    error_types = {}
    for violation in state['format_violations']:
        error = violation['error']
        # Categorize errors
        if 'missing' in error.lower():
            category = 'Missing field'
        elif 'does not match' in error.lower():
            category = 'Pattern mismatch'
        elif 'format' in error.lower():
            category = 'Invalid format'
        else:
            category = 'Other'
        
        error_types[category] = error_types.get(category, 0) + 1
    
    statistics = {
        'total_records': len(state['input_data']),
        'valid_records': sum(1 for r in state['validation_results'] if r['is_valid']),
        'invalid_records': sum(1 for r in state['validation_results'] if not r['is_valid']),
        'total_violations': len(state['format_violations']),
        'violations_by_field': violations_by_field,
        'error_types': error_types,
        'most_violated_field': max(violations_by_field.items(), key=lambda x: x[1])[0] if violations_by_field else None,
        'pass_rate': (sum(1 for r in state['validation_results'] if r['is_valid']) / 
                     len(state['input_data']) * 100) if state['input_data'] else 0
    }
    
    return {
        **state,
        'statistics': statistics,
        'messages': state['messages'] + ['Analyzed format violations']
    }


def generate_format_report_agent(state: FormatValidationState) -> FormatValidationState:
    """Generate comprehensive format validation report"""
    
    report_lines = [
        "=" * 80,
        "FORMAT VALIDATION REPORT",
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
        "FORMAT PATTERNS:",
        "-" * 40
    ]
    
    for field, pattern_info in state['format_patterns'].items():
        report_lines.append(f"• {field}: {pattern_info['description']}")
        report_lines.append(f"  Examples: {', '.join(pattern_info['examples'])}")
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        report_lines.append(f"Record #{result['record_id']}: {status}")
        if not result['is_valid']:
            report_lines.append(f"  Violations: {result['violations_count']}/{result['fields_checked']}")
    
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
        "ERROR CATEGORIES:",
        "-" * 40
    ])
    
    for category, count in sorted(state['statistics']['error_types'].items(),
                                  key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {category}: {count} occurrences")
    
    report_lines.extend([
        "",
        "DETAILED VIOLATIONS:",
        "-" * 40
    ])
    
    # Group by record
    violations_by_record = {}
    for violation in state['format_violations']:
        record_id = violation['record_id']
        if record_id not in violations_by_record:
            violations_by_record[record_id] = []
        violations_by_record[record_id].append(violation)
    
    for record_id, violations in sorted(violations_by_record.items()):
        report_lines.append(f"\nRecord #{record_id}:")
        for violation in violations:
            report_lines.append(f"  ✗ {violation['field']}: {violation['error']}")
            report_lines.append(f"     Value: {violation.get('value', 'N/A')}")
            report_lines.append(f"     Expected: {violation['pattern']}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most violated format: {state['statistics']['most_violated_field']}",
        f"✓ Overall pass rate: {state['statistics']['pass_rate']:.1f}%",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Use input masks to guide users toward correct formats",
        "• Provide real-time format validation feedback",
        "• Show format examples in placeholder text",
        "• Consider auto-formatting (e.g., phone numbers, credit cards)",
        "• Implement progressive validation (validate as user types)",
        "• Use specialized UI components (date pickers, color pickers)",
        "• Provide clear error messages with format requirements",
        "• Consider international format variations",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive format validation report']
    }


# Create the graph
def create_format_validation_graph():
    """Create the format validation workflow graph"""
    
    workflow = StateGraph(FormatValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_format_validation_agent)
    workflow.add_node("define_patterns", define_format_patterns_agent)
    workflow.add_node("validate", validate_formats_agent)
    workflow.add_node("analyze", analyze_format_violations_agent)
    workflow.add_node("generate_report", generate_format_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_patterns")
    workflow.add_edge("define_patterns", "validate")
    workflow.add_edge("validate", "analyze")
    workflow.add_edge("analyze", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the format validation graph
    app = create_format_validation_graph()
    
    # Initialize state
    initial_state: FormatValidationState = {
        'messages': [],
        'input_data': [],
        'format_patterns': {},
        'validation_results': [],
        'format_violations': [],
        'statistics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("FORMAT VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nFormat validation pattern execution complete! ✓")

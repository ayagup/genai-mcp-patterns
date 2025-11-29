"""
Output Validation MCP Pattern

This pattern demonstrates output validation in an agentic MCP system.
Output validation ensures that data produced by agents meets quality standards,
format requirements, and business rules before being sent to clients or other systems.

Use cases:
- API response validation
- Generated content verification
- Data quality assurance
- Contract compliance
- Output sanitization
- Format consistency
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
import json


# Define the state for output validation
class OutputValidationState(TypedDict):
    """State for tracking output validation process"""
    messages: Annotated[List[str], add]
    raw_outputs: List[Dict[str, Any]]
    output_schema: Dict[str, Any]
    validation_rules: List[Dict[str, Any]]
    validated_outputs: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]
    report: str


class OutputValidator:
    """Class to validate output data"""
    
    def validate_structure(self, output: Dict[str, Any], 
                          schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate output structure matches schema"""
        errors = []
        
        # Check required fields
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        field_types = schema.get('fields', {})
        for field, expected_type in field_types.items():
            if field in output:
                value = output[field]
                if not self._check_type(value, expected_type):
                    errors.append(
                        f"Field '{field}' has wrong type. Expected {expected_type}, "
                        f"got {type(value).__name__}"
                    )
        
        # Check for unexpected fields
        if schema.get('strict', False):
            allowed_fields = set(field_types.keys())
            actual_fields = set(output.keys())
            unexpected = actual_fields - allowed_fields
            if unexpected:
                errors.append(f"Unexpected fields: {unexpected}")
        
        return len(errors) == 0, errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        type_map = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict,
            'number': (int, float)
        }
        
        expected_class = type_map.get(expected_type)
        if expected_class:
            return isinstance(value, expected_class)
        return True
    
    def validate_completeness(self, output: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate output completeness"""
        errors = []
        
        # Check for empty values in important fields
        important_fields = ['content', 'result', 'data', 'response']
        for field in important_fields:
            if field in output:
                value = output[field]
                if value is None or (isinstance(value, str) and value.strip() == ''):
                    errors.append(f"Field '{field}' is empty")
                elif isinstance(value, (list, dict)) and len(value) == 0:
                    errors.append(f"Field '{field}' is empty collection")
        
        return len(errors) == 0, errors
    
    def validate_format(self, output: Dict[str, Any], 
                       format_rules: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate output format"""
        errors = []
        
        for field, format_type in format_rules.items():
            if field not in output:
                continue
            
            value = output[field]
            
            if format_type == 'json' and isinstance(value, str):
                try:
                    json.loads(value)
                except json.JSONDecodeError:
                    errors.append(f"Field '{field}' is not valid JSON")
            
            elif format_type == 'non_empty_string':
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"Field '{field}' must be non-empty string")
            
            elif format_type == 'positive_number':
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"Field '{field}' must be positive number")
        
        return len(errors) == 0, errors
    
    def validate_business_rules(self, output: Dict[str, Any], 
                               rules: List[Dict]) -> tuple[bool, List[str]]:
        """Validate against business rules"""
        errors = []
        
        for rule in rules:
            rule_type = rule.get('type')
            
            if rule_type == 'min_value':
                field = rule.get('field')
                min_val = rule.get('min')
                if field in output and output[field] < min_val:
                    errors.append(f"{field} must be >= {min_val}")
            
            elif rule_type == 'max_value':
                field = rule.get('field')
                max_val = rule.get('max')
                if field in output and output[field] > max_val:
                    errors.append(f"{field} must be <= {max_val}")
            
            elif rule_type == 'field_relationship':
                field1 = rule.get('field1')
                field2 = rule.get('field2')
                operator = rule.get('operator', '>')
                
                if field1 in output and field2 in output:
                    val1, val2 = output[field1], output[field2]
                    if operator == '>' and not (val1 > val2):
                        errors.append(f"{field1} must be > {field2}")
                    elif operator == '<' and not (val1 < val2):
                        errors.append(f"{field1} must be < {field2}")
        
        return len(errors) == 0, errors


class OutputQualityAnalyzer:
    """Analyze output quality"""
    
    def calculate_completeness_score(self, output: Dict[str, Any], 
                                     expected_fields: List[str]) -> float:
        """Calculate completeness score (0-100)"""
        if not expected_fields:
            return 100.0
        
        present_fields = sum(1 for field in expected_fields if field in output)
        non_empty_fields = sum(
            1 for field in expected_fields 
            if field in output and output[field] not in [None, '', [], {}]
        )
        
        return (non_empty_fields / len(expected_fields)) * 100
    
    def calculate_accuracy_score(self, validation_errors: List[str]) -> float:
        """Calculate accuracy score based on errors (0-100)"""
        # Simple scoring: fewer errors = higher score
        if not validation_errors:
            return 100.0
        
        # Penalize each error
        penalty_per_error = 10
        score = max(0, 100 - (len(validation_errors) * penalty_per_error))
        return score
    
    def calculate_quality_score(self, completeness: float, accuracy: float) -> float:
        """Calculate overall quality score"""
        # Weighted average
        return (completeness * 0.4) + (accuracy * 0.6)


# Agent functions
def initialize_output_validation_agent(state: OutputValidationState) -> OutputValidationState:
    """Initialize with sample outputs to validate"""
    
    # Sample API responses from different agents
    raw_outputs = [
        {
            'output_id': 1,
            'agent': 'summarization_agent',
            'status': 'success',
            'summary': 'The quarterly sales report shows a 15% increase...',
            'word_count': 150,
            'confidence': 0.92,
            'timestamp': '2024-03-15T10:30:00Z'
        },
        {
            'output_id': 2,
            'agent': 'data_extraction_agent',
            'status': 'success',
            'data': {
                'entities': ['Company A', 'Product X'],
                'dates': ['2024-01-01'],
                'amounts': [10000, 25000]
            },
            'count': 5,
            'confidence': 0.88
        },
        {
            'output_id': 3,
            'agent': 'translation_agent',
            'status': 'error',  # Invalid status
            'translation': '',  # Empty output
            'source_lang': 'en',
            'target_lang': 'es'
            # Missing 'confidence' field
        },
        {
            'output_id': 4,
            'agent': 'recommendation_agent',
            'status': 'success',
            'recommendations': [
                {'item_id': 'PROD001', 'score': 0.95},
                {'item_id': 'PROD002', 'score': 0.87},
                {'item_id': 'PROD003', 'score': 0.82}
            ],
            'count': 3,
            'confidence': 0.91
        },
        {
            'output_id': 5,
            'agent': 'sentiment_agent',
            'status': 'success',
            'sentiment': 'positive',
            'score': 1.5,  # Invalid: should be 0-1
            'confidence': -0.1  # Invalid: negative confidence
        }
    ]
    
    return {
        **state,
        'raw_outputs': raw_outputs,
        'messages': state['messages'] + [f'Initialized with {len(raw_outputs)} outputs to validate']
    }


def define_output_schema_agent(state: OutputValidationState) -> OutputValidationState:
    """Define expected output schema"""
    
    output_schema = {
        'required': ['output_id', 'agent', 'status', 'confidence'],
        'fields': {
            'output_id': 'integer',
            'agent': 'string',
            'status': 'string',
            'confidence': 'float',
            'summary': 'string',
            'translation': 'string',
            'data': 'dict',
            'recommendations': 'list',
            'sentiment': 'string',
            'score': 'float',
            'count': 'integer',
            'word_count': 'integer',
            'timestamp': 'string',
            'source_lang': 'string',
            'target_lang': 'string'
        },
        'strict': False  # Allow additional fields
    }
    
    return {
        **state,
        'output_schema': output_schema,
        'messages': state['messages'] + ['Defined output schema with required and optional fields']
    }


def define_validation_rules_agent(state: OutputValidationState) -> OutputValidationState:
    """Define validation rules"""
    
    validation_rules = [
        {
            'name': 'Structure Validation',
            'type': 'structure',
            'enabled': True
        },
        {
            'name': 'Completeness Check',
            'type': 'completeness',
            'enabled': True
        },
        {
            'name': 'Format Validation',
            'type': 'format',
            'format_rules': {
                'summary': 'non_empty_string',
                'translation': 'non_empty_string',
                'count': 'positive_number'
            }
        },
        {
            'name': 'Business Rules',
            'type': 'business_rules',
            'rules': [
                {'type': 'min_value', 'field': 'confidence', 'min': 0.0},
                {'type': 'max_value', 'field': 'confidence', 'max': 1.0},
                {'type': 'min_value', 'field': 'score', 'min': 0.0},
                {'type': 'max_value', 'field': 'score', 'max': 1.0}
            ]
        }
    ]
    
    return {
        **state,
        'validation_rules': validation_rules,
        'messages': state['messages'] + [f'Defined {len(validation_rules)} validation rule sets']
    }


def validate_outputs_agent(state: OutputValidationState) -> OutputValidationState:
    """Validate all outputs"""
    
    validator = OutputValidator()
    validation_results = []
    
    for output in state['raw_outputs']:
        all_errors = []
        validations_passed = 0
        validations_total = 0
        
        # Structure validation
        is_valid, errors = validator.validate_structure(output, state['output_schema'])
        validations_total += 1
        if is_valid:
            validations_passed += 1
        else:
            all_errors.extend(errors)
        
        # Completeness validation
        is_valid, errors = validator.validate_completeness(output)
        validations_total += 1
        if is_valid:
            validations_passed += 1
        else:
            all_errors.extend(errors)
        
        # Format validation
        format_rule = next((r for r in state['validation_rules'] if r['type'] == 'format'), None)
        if format_rule:
            is_valid, errors = validator.validate_format(
                output, 
                format_rule.get('format_rules', {})
            )
            validations_total += 1
            if is_valid:
                validations_passed += 1
            else:
                all_errors.extend(errors)
        
        # Business rules validation
        business_rule = next((r for r in state['validation_rules'] 
                            if r['type'] == 'business_rules'), None)
        if business_rule:
            is_valid, errors = validator.validate_business_rules(
                output,
                business_rule.get('rules', [])
            )
            validations_total += 1
            if is_valid:
                validations_passed += 1
            else:
                all_errors.extend(errors)
        
        validation_results.append({
            'output_id': output.get('output_id'),
            'agent': output.get('agent'),
            'is_valid': len(all_errors) == 0,
            'validations_passed': validations_passed,
            'validations_total': validations_total,
            'errors': all_errors,
            'error_count': len(all_errors)
        })
    
    valid_count = sum(1 for r in validation_results if r['is_valid'])
    
    return {
        **state,
        'validation_results': validation_results,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} outputs: {valid_count} valid, '
            f'{len(validation_results) - valid_count} invalid'
        ]
    }


def calculate_quality_metrics_agent(state: OutputValidationState) -> OutputValidationState:
    """Calculate quality metrics for outputs"""
    
    analyzer = OutputQualityAnalyzer()
    quality_metrics = {
        'outputs': [],
        'overall_quality': 0.0,
        'pass_rate': 0.0
    }
    
    expected_fields = state['output_schema']['required']
    
    for output, validation in zip(state['raw_outputs'], state['validation_results']):
        completeness = analyzer.calculate_completeness_score(output, expected_fields)
        accuracy = analyzer.calculate_accuracy_score(validation['errors'])
        quality = analyzer.calculate_quality_score(completeness, accuracy)
        
        quality_metrics['outputs'].append({
            'output_id': output.get('output_id'),
            'completeness_score': round(completeness, 2),
            'accuracy_score': round(accuracy, 2),
            'quality_score': round(quality, 2),
            'passed_validation': validation['is_valid']
        })
    
    # Calculate overall metrics
    avg_quality = sum(o['quality_score'] for o in quality_metrics['outputs']) / len(quality_metrics['outputs'])
    pass_rate = sum(1 for o in quality_metrics['outputs'] if o['passed_validation']) / len(quality_metrics['outputs']) * 100
    
    quality_metrics['overall_quality'] = round(avg_quality, 2)
    quality_metrics['pass_rate'] = round(pass_rate, 2)
    
    return {
        **state,
        'quality_metrics': quality_metrics,
        'messages': state['messages'] + [
            f'Calculated quality metrics: {quality_metrics["overall_quality"]}/100 overall quality, '
            f'{quality_metrics["pass_rate"]}% pass rate'
        ]
    }


def generate_validation_report_agent(state: OutputValidationState) -> OutputValidationState:
    """Generate comprehensive validation report"""
    
    report_lines = [
        "=" * 80,
        "OUTPUT VALIDATION REPORT",
        "=" * 80,
        "",
        "VALIDATION SUMMARY:",
        "-" * 40,
        f"Total outputs validated: {len(state['validation_results'])}",
        f"Outputs passed: {sum(1 for r in state['validation_results'] if r['is_valid'])}",
        f"Outputs failed: {sum(1 for r in state['validation_results'] if not r['is_valid'])}",
        f"Pass rate: {state['quality_metrics']['pass_rate']:.1f}%",
        f"Overall quality score: {state['quality_metrics']['overall_quality']:.1f}/100",
        "",
        "OUTPUT SCHEMA:",
        "-" * 40,
        f"Required fields: {', '.join(state['output_schema']['required'])}",
        f"Total defined fields: {len(state['output_schema']['fields'])}",
        f"Strict mode: {state['output_schema'].get('strict', False)}",
        "",
        "VALIDATION RULES:",
        "-" * 40
    ]
    
    for rule in state['validation_rules']:
        report_lines.append(f"  ✓ {rule['name']} ({rule['type']})")
    
    report_lines.extend([
        "",
        "DETAILED VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ PASSED" if result['is_valid'] else "✗ FAILED"
        report_lines.append(
            f"\nOutput #{result['output_id']} ({result['agent']}): {status}"
        )
        report_lines.append(
            f"  Validations: {result['validations_passed']}/{result['validations_total']} passed"
        )
        
        if result['errors']:
            report_lines.append(f"  Errors ({len(result['errors'])}):")
            for error in result['errors']:
                report_lines.append(f"    • {error}")
    
    report_lines.extend([
        "",
        "QUALITY METRICS:",
        "-" * 40
    ])
    
    for metric in state['quality_metrics']['outputs']:
        status = "✓" if metric['passed_validation'] else "✗"
        report_lines.append(
            f"\nOutput #{metric['output_id']}: {status}"
        )
        report_lines.append(f"  Completeness: {metric['completeness_score']}/100")
        report_lines.append(f"  Accuracy: {metric['accuracy_score']}/100")
        report_lines.append(f"  Quality: {metric['quality_score']}/100")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40
    ])
    
    if state['quality_metrics']['pass_rate'] < 80:
        report_lines.append("⚠ Low pass rate detected:")
        report_lines.append("  • Review agent output generation logic")
        report_lines.append("  • Add output post-processing steps")
        report_lines.append("  • Implement output templates")
    
    if state['quality_metrics']['overall_quality'] < 70:
        report_lines.append("⚠ Low quality scores detected:")
        report_lines.append("  • Improve completeness of outputs")
        report_lines.append("  • Add validation before returning results")
        report_lines.append("  • Implement quality gates")
    
    report_lines.extend([
        "",
        "BEST PRACTICES:",
        "-" * 40,
        "✓ Always validate outputs before sending to clients",
        "✓ Define clear output schemas and enforce them",
        "✓ Monitor quality metrics over time",
        "✓ Implement automated quality gates",
        "✓ Log validation failures for debugging",
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
def create_output_validation_graph():
    """Create the output validation workflow graph"""
    
    workflow = StateGraph(OutputValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_output_validation_agent)
    workflow.add_node("define_schema", define_output_schema_agent)
    workflow.add_node("define_rules", define_validation_rules_agent)
    workflow.add_node("validate", validate_outputs_agent)
    workflow.add_node("calculate_quality", calculate_quality_metrics_agent)
    workflow.add_node("generate_report", generate_validation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_schema")
    workflow.add_edge("define_schema", "define_rules")
    workflow.add_edge("define_rules", "validate")
    workflow.add_edge("validate", "calculate_quality")
    workflow.add_edge("calculate_quality", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the output validation graph
    app = create_output_validation_graph()
    
    # Initialize state
    initial_state: OutputValidationState = {
        'messages': [],
        'raw_outputs': [],
        'output_schema': {},
        'validation_rules': [],
        'validated_outputs': [],
        'validation_results': [],
        'quality_metrics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("OUTPUT VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nOutput validation pattern execution complete! ✓")

"""
Type Validation MCP Pattern

This pattern demonstrates type validation in an agentic MCP system.
Type validation ensures data conforms to expected types at runtime,
catching type mismatches and preventing type-related errors.

Use cases:
- Runtime type checking
- Dynamic type validation
- Type coercion validation
- Generic type validation
- Union type checking
- Complex type validation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Union, get_args, get_origin
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI


# Define the state for type validation
class TypeValidationState(TypedDict):
    """State for tracking type validation process"""
    messages: Annotated[List[str], add]
    data_samples: List[Dict[str, Any]]
    type_definitions: Dict[str, Any]
    validation_results: List[Dict[str, Any]]
    type_mismatches: List[Dict[str, Any]]
    type_statistics: Dict[str, Any]
    report: str


class TypeValidator:
    """Comprehensive type validator"""
    
    def validate_primitive_type(self, value: Any, expected_type: type) -> tuple[bool, Optional[str]]:
        """Validate primitive types (int, float, str, bool)"""
        if expected_type == int:
            if not isinstance(value, int) or isinstance(value, bool):
                return False, f"Expected int, got {type(value).__name__}"
        elif expected_type == float:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return False, f"Expected float, got {type(value).__name__}"
        elif expected_type == str:
            if not isinstance(value, str):
                return False, f"Expected str, got {type(value).__name__}"
        elif expected_type == bool:
            if not isinstance(value, bool):
                return False, f"Expected bool, got {type(value).__name__}"
        else:
            if not isinstance(value, expected_type):
                return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"
        
        return True, None
    
    def validate_collection_type(self, value: Any, expected_type: type) -> tuple[bool, Optional[str]]:
        """Validate collection types (list, dict, set, tuple)"""
        origin = get_origin(expected_type)
        
        if origin is list or expected_type is list:
            if not isinstance(value, list):
                return False, f"Expected list, got {type(value).__name__}"
        elif origin is dict or expected_type is dict:
            if not isinstance(value, dict):
                return False, f"Expected dict, got {type(value).__name__}"
        elif origin is set or expected_type is set:
            if not isinstance(value, set):
                return False, f"Expected set, got {type(value).__name__}"
        elif origin is tuple or expected_type is tuple:
            if not isinstance(value, tuple):
                return False, f"Expected tuple, got {type(value).__name__}"
        
        return True, None
    
    def validate_union_type(self, value: Any, union_types: tuple) -> tuple[bool, Optional[str]]:
        """Validate union types (e.g., Union[int, str])"""
        for expected_type in union_types:
            is_valid, _ = self.validate_type(value, expected_type)
            if is_valid:
                return True, None
        
        type_names = ', '.join(t.__name__ if hasattr(t, '__name__') else str(t) for t in union_types)
        return False, f"Value doesn't match any type in Union[{type_names}]"
    
    def validate_optional_type(self, value: Any, inner_type: type) -> tuple[bool, Optional[str]]:
        """Validate Optional types (Union[type, None])"""
        if value is None:
            return True, None
        
        return self.validate_type(value, inner_type)
    
    def validate_list_items(self, value: List, item_type: type) -> tuple[bool, Optional[str]]:
        """Validate all items in a list"""
        if not isinstance(value, list):
            return False, "Value is not a list"
        
        for idx, item in enumerate(value):
            is_valid, error = self.validate_type(item, item_type)
            if not is_valid:
                return False, f"Item at index {idx}: {error}"
        
        return True, None
    
    def validate_dict_types(self, value: Dict, key_type: type, value_type: type) -> tuple[bool, Optional[str]]:
        """Validate dictionary key and value types"""
        if not isinstance(value, dict):
            return False, "Value is not a dict"
        
        for k, v in value.items():
            # Validate key type
            is_valid, error = self.validate_type(k, key_type)
            if not is_valid:
                return False, f"Key {k}: {error}"
            
            # Validate value type
            is_valid, error = self.validate_type(v, value_type)
            if not is_valid:
                return False, f"Value for key {k}: {error}"
        
        return True, None
    
    def validate_type(self, value: Any, expected_type: Any) -> tuple[bool, Optional[str]]:
        """Main type validation method"""
        # Handle None/null
        if value is None:
            if expected_type is type(None):
                return True, None
            origin = get_origin(expected_type)
            if origin is Union:
                args = get_args(expected_type)
                if type(None) in args:
                    return True, None
            return False, "Value is None"
        
        # Handle Union types
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            return self.validate_union_type(value, args)
        
        # Handle primitive types
        if expected_type in (int, float, str, bool):
            return self.validate_primitive_type(value, expected_type)
        
        # Handle collection types
        if expected_type in (list, dict, set, tuple) or origin in (list, dict, set, tuple):
            is_valid, error = self.validate_collection_type(value, expected_type)
            if not is_valid:
                return is_valid, error
            
            # Validate generic parameters if available
            args = get_args(expected_type)
            if args:
                if origin is list and len(args) == 1:
                    return self.validate_list_items(value, args[0])
                elif origin is dict and len(args) == 2:
                    return self.validate_dict_types(value, args[0], args[1])
            
            return True, None
        
        # Handle custom classes
        if isinstance(expected_type, type):
            if not isinstance(value, expected_type):
                return False, f"Expected {expected_type.__name__}, got {type(value).__name__}"
            return True, None
        
        return True, None


class TypeCoercionValidator:
    """Validate type coercion attempts"""
    
    def can_coerce(self, value: Any, target_type: type) -> tuple[bool, Any, Optional[str]]:
        """Check if value can be coerced to target type"""
        try:
            if target_type == int:
                coerced = int(value)
                return True, coerced, None
            elif target_type == float:
                coerced = float(value)
                return True, coerced, None
            elif target_type == str:
                coerced = str(value)
                return True, coerced, None
            elif target_type == bool:
                # Explicit boolean conversion
                if isinstance(value, str):
                    if value.lower() in ('true', '1', 'yes'):
                        return True, True, None
                    elif value.lower() in ('false', '0', 'no'):
                        return True, False, None
                coerced = bool(value)
                return True, coerced, None
            else:
                return False, value, f"Cannot coerce to {target_type.__name__}"
        except (ValueError, TypeError) as e:
            return False, value, f"Coercion failed: {str(e)}"


# Agent functions
def initialize_type_validation_agent(state: TypeValidationState) -> TypeValidationState:
    """Initialize with sample data for type validation"""
    
    data_samples = [
        {
            'sample_id': 1,
            'user_id': 12345,  # int
            'username': 'john_doe',  # str
            'score': 95.5,  # float
            'is_active': True,  # bool
            'tags': ['python', 'ai', 'ml'],  # List[str]
            'metadata': {'role': 'admin', 'level': 5},  # Dict[str, Any]
            'optional_field': 'present'  # Optional[str]
        },
        {
            'sample_id': 2,
            'user_id': '67890',  # Wrong type: str instead of int
            'username': 12345,  # Wrong type: int instead of str
            'score': 'invalid',  # Wrong type: str instead of float
            'is_active': 'yes',  # Wrong type: str instead of bool
            'tags': 'not_a_list',  # Wrong type: str instead of list
            'metadata': ['not', 'a', 'dict'],  # Wrong type: list instead of dict
            'optional_field': None  # Valid: Optional allows None
        },
        {
            'sample_id': 3,
            'user_id': 11111,
            'username': 'alice',
            'score': 88,  # int but should be acceptable as float
            'is_active': False,
            'tags': [1, 2, 3],  # Wrong item type: List[int] instead of List[str]
            'metadata': {1: 'invalid_key', 'valid': 'value'},  # Wrong key type
            'optional_field': None
        },
        {
            'sample_id': 4,
            'user_id': 99999,
            'username': 'bob',
            'score': 92.0,
            'is_active': True,
            'tags': [],  # Empty list (valid)
            'metadata': {},  # Empty dict (valid)
            'optional_field': 'data'
        }
    ]
    
    return {
        **state,
        'data_samples': data_samples,
        'messages': state['messages'] + [f'Initialized with {len(data_samples)} data samples']
    }


def define_type_definitions_agent(state: TypeValidationState) -> TypeValidationState:
    """Define expected types for each field"""
    
    type_definitions = {
        'sample_id': int,
        'user_id': int,
        'username': str,
        'score': float,
        'is_active': bool,
        'tags': 'List[str]',  # String representation for complex types
        'metadata': 'Dict[str, Any]',
        'optional_field': 'Optional[str]'
    }
    
    return {
        **state,
        'type_definitions': type_definitions,
        'messages': state['messages'] + [f'Defined types for {len(type_definitions)} fields']
    }


def validate_types_agent(state: TypeValidationState) -> TypeValidationState:
    """Validate types for all samples"""
    
    validator = TypeValidator()
    validation_results = []
    all_mismatches = []
    
    # Type mapping for validation
    type_map = {
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
        'List[str]': List[str],
        'Dict[str, Any]': Dict[str, Any],
        'Optional[str]': Optional[str]
    }
    
    for sample in state['data_samples']:
        sample_id = sample.get('sample_id')
        mismatches = []
        
        for field, expected_type_str in state['type_definitions'].items():
            if field == 'sample_id':
                continue
            
            value = sample.get(field)
            expected_type = type_map.get(expected_type_str, str)
            
            is_valid, error = validator.validate_type(value, expected_type)
            
            if not is_valid:
                mismatches.append({
                    'field': field,
                    'expected_type': expected_type_str,
                    'actual_type': type(value).__name__,
                    'value': str(value)[:50],
                    'error': error
                })
        
        validation_results.append({
            'sample_id': sample_id,
            'is_valid': len(mismatches) == 0,
            'fields_checked': len(state['type_definitions']) - 1,
            'type_errors': len(mismatches)
        })
        
        for mismatch in mismatches:
            all_mismatches.append({
                'sample_id': sample_id,
                **mismatch
            })
    
    valid_count = sum(1 for r in validation_results if r['is_valid'])
    
    return {
        **state,
        'validation_results': validation_results,
        'type_mismatches': all_mismatches,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} samples: {valid_count} valid, '
            f'{len(validation_results) - valid_count} with type errors'
        ]
    }


def analyze_type_errors_agent(state: TypeValidationState) -> TypeValidationState:
    """Analyze type error patterns"""
    
    # Count errors by field
    errors_by_field = {}
    for mismatch in state['type_mismatches']:
        field = mismatch['field']
        errors_by_field[field] = errors_by_field.get(field, 0) + 1
    
    # Count errors by type
    errors_by_type = {}
    for mismatch in state['type_mismatches']:
        actual_type = mismatch['actual_type']
        errors_by_type[actual_type] = errors_by_type.get(actual_type, 0) + 1
    
    type_statistics = {
        'total_samples': len(state['data_samples']),
        'valid_samples': sum(1 for r in state['validation_results'] if r['is_valid']),
        'invalid_samples': sum(1 for r in state['validation_results'] if not r['is_valid']),
        'total_type_errors': len(state['type_mismatches']),
        'errors_by_field': errors_by_field,
        'errors_by_type': errors_by_type,
        'most_problematic_field': max(errors_by_field.items(), key=lambda x: x[1])[0] if errors_by_field else None,
        'most_common_wrong_type': max(errors_by_type.items(), key=lambda x: x[1])[0] if errors_by_type else None
    }
    
    return {
        **state,
        'type_statistics': type_statistics,
        'messages': state['messages'] + ['Analyzed type error patterns']
    }


def test_type_coercion_agent(state: TypeValidationState) -> TypeValidationState:
    """Test type coercion possibilities"""
    
    coercion_validator = TypeCoercionValidator()
    coercion_results = []
    
    # Test coercion for samples with type errors
    for sample in state['data_samples']:
        if sample.get('sample_id') == 2:  # Sample with many type errors
            # Try coercing user_id from str to int
            can_coerce, coerced, error = coercion_validator.can_coerce(
                sample.get('user_id'), int
            )
            coercion_results.append({
                'field': 'user_id',
                'original_value': sample.get('user_id'),
                'target_type': 'int',
                'can_coerce': can_coerce,
                'coerced_value': coerced if can_coerce else None,
                'error': error
            })
            
            # Try coercing score from str to float
            can_coerce, coerced, error = coercion_validator.can_coerce(
                sample.get('score'), float
            )
            coercion_results.append({
                'field': 'score',
                'original_value': sample.get('score'),
                'target_type': 'float',
                'can_coerce': can_coerce,
                'coerced_value': coerced if can_coerce else None,
                'error': error
            })
    
    successful_coercions = sum(1 for r in coercion_results if r['can_coerce'])
    
    return {
        **state,
        'messages': state['messages'] + [
            f'Tested type coercion: {successful_coercions}/{len(coercion_results)} successful'
        ]
    }


def generate_type_validation_report_agent(state: TypeValidationState) -> TypeValidationState:
    """Generate comprehensive type validation report"""
    
    report_lines = [
        "=" * 80,
        "TYPE VALIDATION REPORT",
        "=" * 80,
        "",
        "VALIDATION SUMMARY:",
        "-" * 40,
        f"Total samples: {state['type_statistics']['total_samples']}",
        f"Valid samples: {state['type_statistics']['valid_samples']}",
        f"Invalid samples: {state['type_statistics']['invalid_samples']}",
        f"Total type errors: {state['type_statistics']['total_type_errors']}",
        "",
        "TYPE DEFINITIONS:",
        "-" * 40
    ]
    
    for field, expected_type in state['type_definitions'].items():
        report_lines.append(f"  {field}: {expected_type}")
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        report_lines.append(f"Sample #{result['sample_id']}: {status}")
        if not result['is_valid']:
            report_lines.append(f"  Type errors: {result['type_errors']}/{result['fields_checked']}")
    
    report_lines.extend([
        "",
        "TYPE ERRORS BY FIELD:",
        "-" * 40
    ])
    
    for field, count in sorted(state['type_statistics']['errors_by_field'].items(),
                               key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {field}: {count} errors")
    
    report_lines.extend([
        "",
        "WRONG TYPES ENCOUNTERED:",
        "-" * 40
    ])
    
    for wrong_type, count in sorted(state['type_statistics']['errors_by_type'].items(),
                                    key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {wrong_type}: {count} occurrences")
    
    report_lines.extend([
        "",
        "DETAILED TYPE MISMATCHES:",
        "-" * 40
    ])
    
    # Group by sample
    mismatches_by_sample = {}
    for mismatch in state['type_mismatches']:
        sample_id = mismatch['sample_id']
        if sample_id not in mismatches_by_sample:
            mismatches_by_sample[sample_id] = []
        mismatches_by_sample[sample_id].append(mismatch)
    
    for sample_id, mismatches in sorted(mismatches_by_sample.items()):
        report_lines.append(f"\nSample #{sample_id}:")
        for mismatch in mismatches:
            report_lines.append(
                f"  ✗ {mismatch['field']}: expected {mismatch['expected_type']}, "
                f"got {mismatch['actual_type']}"
            )
            if mismatch.get('error'):
                report_lines.append(f"     Error: {mismatch['error']}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most problematic field: {state['type_statistics']['most_problematic_field']}",
        f"✓ Most common wrong type: {state['type_statistics']['most_common_wrong_type']}",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Use static type checkers (mypy, pyright) during development",
        "• Implement runtime type validation at API boundaries",
        "• Consider using Pydantic for data validation",
        "• Add type hints to all functions and classes",
        "• Enable strict type checking in development",
        "• Document expected types clearly",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive type validation report']
    }


# Create the graph
def create_type_validation_graph():
    """Create the type validation workflow graph"""
    
    workflow = StateGraph(TypeValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_type_validation_agent)
    workflow.add_node("define_types", define_type_definitions_agent)
    workflow.add_node("validate", validate_types_agent)
    workflow.add_node("analyze", analyze_type_errors_agent)
    workflow.add_node("test_coercion", test_type_coercion_agent)
    workflow.add_node("generate_report", generate_type_validation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_types")
    workflow.add_edge("define_types", "validate")
    workflow.add_edge("validate", "analyze")
    workflow.add_edge("analyze", "test_coercion")
    workflow.add_edge("test_coercion", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the type validation graph
    app = create_type_validation_graph()
    
    # Initialize state
    initial_state: TypeValidationState = {
        'messages': [],
        'data_samples': [],
        'type_definitions': {},
        'validation_results': [],
        'type_mismatches': [],
        'type_statistics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("TYPE VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nType validation pattern execution complete! ✓")

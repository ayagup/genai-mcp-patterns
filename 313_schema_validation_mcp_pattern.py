"""
Schema Validation MCP Pattern

This pattern demonstrates schema validation in an agentic MCP system.
Schema validation ensures data conforms to a defined structure, types, and constraints,
providing a contract between data producers and consumers.

Use cases:
- API contract enforcement
- Data interchange validation
- Configuration file validation
- Database schema compliance
- Message format validation
- JSON/XML schema validation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from enum import Enum


# Define the state for schema validation
class SchemaValidationState(TypedDict):
    """State for tracking schema validation process"""
    messages: Annotated[List[str], add]
    data_samples: List[Dict[str, Any]]
    schemas: Dict[str, Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    schema_compatibility: Dict[str, Any]
    migration_plan: List[Dict[str, Any]]
    report: str


class SchemaType(Enum):
    """Schema type definitions"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    ANY = "any"


class SchemaValidator:
    """Comprehensive schema validator"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate data against schema"""
        errors = []
        
        # Validate required fields
        errors.extend(self._validate_required(data))
        
        # Validate field types
        errors.extend(self._validate_types(data))
        
        # Validate constraints
        errors.extend(self._validate_constraints(data))
        
        # Validate nested objects
        errors.extend(self._validate_nested(data))
        
        return len(errors) == 0, errors
    
    def _validate_required(self, data: Dict[str, Any]) -> List[str]:
        """Check required fields"""
        errors = []
        required = self.schema.get('required', [])
        
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
            elif data[field] is None:
                errors.append(f"Required field '{field}' is null")
        
        return errors
    
    def _validate_types(self, data: Dict[str, Any]) -> List[str]:
        """Validate field types"""
        errors = []
        properties = self.schema.get('properties', {})
        
        for field, field_schema in properties.items():
            if field not in data:
                continue
            
            value = data[field]
            expected_type = field_schema.get('type')
            
            if not self._check_type(value, expected_type):
                errors.append(
                    f"Field '{field}': expected type '{expected_type}', "
                    f"got '{type(value).__name__}'"
                )
        
        return errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        if expected_type == SchemaType.STRING.value:
            return isinstance(value, str)
        elif expected_type == SchemaType.INTEGER.value:
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == SchemaType.FLOAT.value:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == SchemaType.BOOLEAN.value:
            return isinstance(value, bool)
        elif expected_type == SchemaType.ARRAY.value:
            return isinstance(value, list)
        elif expected_type == SchemaType.OBJECT.value:
            return isinstance(value, dict)
        elif expected_type == SchemaType.NULL.value:
            return value is None
        elif expected_type == SchemaType.ANY.value:
            return True
        return False
    
    def _validate_constraints(self, data: Dict[str, Any]) -> List[str]:
        """Validate field constraints"""
        errors = []
        properties = self.schema.get('properties', {})
        
        for field, field_schema in properties.items():
            if field not in data:
                continue
            
            value = data[field]
            
            # String constraints
            if isinstance(value, str):
                min_length = field_schema.get('minLength')
                max_length = field_schema.get('maxLength')
                pattern = field_schema.get('pattern')
                enum = field_schema.get('enum')
                
                if min_length and len(value) < min_length:
                    errors.append(f"Field '{field}': length {len(value)} < minimum {min_length}")
                if max_length and len(value) > max_length:
                    errors.append(f"Field '{field}': length {len(value)} > maximum {max_length}")
                if pattern and not self._match_pattern(value, pattern):
                    errors.append(f"Field '{field}': does not match pattern '{pattern}'")
                if enum and value not in enum:
                    errors.append(f"Field '{field}': value not in allowed values {enum}")
            
            # Numeric constraints
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                minimum = field_schema.get('minimum')
                maximum = field_schema.get('maximum')
                
                if minimum is not None and value < minimum:
                    errors.append(f"Field '{field}': value {value} < minimum {minimum}")
                if maximum is not None and value > maximum:
                    errors.append(f"Field '{field}': value {value} > maximum {maximum}")
            
            # Array constraints
            if isinstance(value, list):
                min_items = field_schema.get('minItems')
                max_items = field_schema.get('maxItems')
                
                if min_items and len(value) < min_items:
                    errors.append(f"Field '{field}': {len(value)} items < minimum {min_items}")
                if max_items and len(value) > max_items:
                    errors.append(f"Field '{field}': {len(value)} items > maximum {max_items}")
        
        return errors
    
    def _match_pattern(self, value: str, pattern: str) -> bool:
        """Check if value matches regex pattern"""
        import re
        try:
            return bool(re.match(pattern, value))
        except:
            return False
    
    def _validate_nested(self, data: Dict[str, Any]) -> List[str]:
        """Validate nested objects"""
        errors = []
        properties = self.schema.get('properties', {})
        
        for field, field_schema in properties.items():
            if field not in data:
                continue
            
            value = data[field]
            
            # Validate nested object
            if field_schema.get('type') == 'object' and isinstance(value, dict):
                nested_schema = field_schema.get('properties')
                if nested_schema:
                    nested_validator = SchemaValidator({'properties': nested_schema})
                    _, nested_errors = nested_validator.validate(value)
                    errors.extend([f"{field}.{e}" for e in nested_errors])
            
            # Validate array items
            if field_schema.get('type') == 'array' and isinstance(value, list):
                item_schema = field_schema.get('items')
                if item_schema:
                    for idx, item in enumerate(value):
                        if isinstance(item, dict) and item_schema.get('type') == 'object':
                            item_properties = item_schema.get('properties')
                            if item_properties:
                                item_validator = SchemaValidator({'properties': item_properties})
                                _, item_errors = item_validator.validate(item)
                                errors.extend([f"{field}[{idx}].{e}" for e in item_errors])
        
        return errors


class SchemaCompatibilityChecker:
    """Check schema compatibility for evolution"""
    
    def check_compatibility(self, old_schema: Dict, new_schema: Dict) -> Dict[str, Any]:
        """Check if new schema is compatible with old schema"""
        
        result = {
            'is_compatible': True,
            'breaking_changes': [],
            'additions': [],
            'deprecations': [],
            'warnings': []
        }
        
        old_props = old_schema.get('properties', {})
        new_props = new_schema.get('properties', {})
        old_required = set(old_schema.get('required', []))
        new_required = set(new_schema.get('required', []))
        
        # Check for removed fields
        removed_fields = set(old_props.keys()) - set(new_props.keys())
        if removed_fields:
            result['breaking_changes'].append(f"Removed fields: {removed_fields}")
            result['is_compatible'] = False
        
        # Check for new required fields
        new_required_fields = new_required - old_required
        if new_required_fields:
            result['breaking_changes'].append(f"New required fields: {new_required_fields}")
            result['is_compatible'] = False
        
        # Check for type changes
        for field in set(old_props.keys()) & set(new_props.keys()):
            old_type = old_props[field].get('type')
            new_type = new_props[field].get('type')
            
            if old_type != new_type:
                result['breaking_changes'].append(
                    f"Field '{field}': type changed from {old_type} to {new_type}"
                )
                result['is_compatible'] = False
        
        # Check for added fields
        added_fields = set(new_props.keys()) - set(old_props.keys())
        if added_fields:
            result['additions'].append(f"Added fields: {added_fields}")
        
        # Check for removed required fields
        removed_required = old_required - new_required
        if removed_required:
            result['deprecations'].append(f"Fields no longer required: {removed_required}")
        
        return result


# Agent functions
def initialize_schema_validation_agent(state: SchemaValidationState) -> SchemaValidationState:
    """Initialize with data samples and schemas"""
    
    # Sample data for user profile
    data_samples = [
        {
            'sample_id': 1,
            'user_id': 12345,
            'username': 'john_doe',
            'email': 'john@example.com',
            'age': 30,
            'is_active': True,
            'roles': ['user', 'moderator'],
            'profile': {
                'first_name': 'John',
                'last_name': 'Doe',
                'bio': 'Software engineer'
            }
        },
        {
            'sample_id': 2,
            'user_id': 67890,
            'username': 'jane',  # Too short
            'email': 'invalid-email',  # Invalid format
            'age': 150,  # Out of range
            'is_active': 'yes',  # Wrong type
            'roles': [],  # Empty array (min 1)
            'profile': {
                'first_name': 'Jane'
                # Missing 'last_name'
            }
        },
        {
            'sample_id': 3,
            # Missing 'user_id' (required)
            'username': 'alice_wonderland',
            'email': 'alice@example.com',
            'age': 25,
            'is_active': True,
            'roles': ['user'],
            'profile': {
                'first_name': 'Alice',
                'last_name': 'Wonderland',
                'bio': 'Product manager'
            }
        }
    ]
    
    return {
        **state,
        'data_samples': data_samples,
        'messages': state['messages'] + [f'Initialized with {len(data_samples)} data samples']
    }


def define_schemas_agent(state: SchemaValidationState) -> SchemaValidationState:
    """Define validation schemas"""
    
    schemas = {
        'user_profile_v1': {
            'version': '1.0',
            'type': 'object',
            'required': ['user_id', 'username', 'email', 'is_active'],
            'properties': {
                'user_id': {
                    'type': 'integer',
                    'minimum': 1
                },
                'username': {
                    'type': 'string',
                    'minLength': 3,
                    'maxLength': 20,
                    'pattern': r'^[a-zA-Z0-9_]+$'
                },
                'email': {
                    'type': 'string',
                    'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                },
                'age': {
                    'type': 'integer',
                    'minimum': 0,
                    'maximum': 120
                },
                'is_active': {
                    'type': 'boolean'
                },
                'roles': {
                    'type': 'array',
                    'minItems': 1,
                    'items': {
                        'type': 'string',
                        'enum': ['user', 'moderator', 'admin']
                    }
                },
                'profile': {
                    'type': 'object',
                    'properties': {
                        'first_name': {'type': 'string', 'minLength': 1},
                        'last_name': {'type': 'string', 'minLength': 1},
                        'bio': {'type': 'string', 'maxLength': 500}
                    }
                }
            }
        },
        'user_profile_v2': {
            'version': '2.0',
            'type': 'object',
            'required': ['user_id', 'username', 'email'],  # Removed 'is_active' requirement
            'properties': {
                'user_id': {
                    'type': 'integer',
                    'minimum': 1
                },
                'username': {
                    'type': 'string',
                    'minLength': 3,
                    'maxLength': 20,
                    'pattern': r'^[a-zA-Z0-9_]+$'
                },
                'email': {
                    'type': 'string',
                    'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                },
                'age': {
                    'type': 'integer',
                    'minimum': 0,
                    'maximum': 120
                },
                'is_active': {
                    'type': 'boolean'
                },
                'roles': {
                    'type': 'array',
                    'minItems': 1,
                    'items': {
                        'type': 'string',
                        'enum': ['user', 'moderator', 'admin']
                    }
                },
                'profile': {
                    'type': 'object',
                    'properties': {
                        'first_name': {'type': 'string', 'minLength': 1},
                        'last_name': {'type': 'string', 'minLength': 1},
                        'bio': {'type': 'string', 'maxLength': 500}
                    }
                },
                'preferences': {  # New field
                    'type': 'object',
                    'properties': {
                        'theme': {'type': 'string', 'enum': ['light', 'dark']},
                        'notifications': {'type': 'boolean'}
                    }
                }
            }
        }
    }
    
    return {
        **state,
        'schemas': schemas,
        'messages': state['messages'] + [f'Defined {len(schemas)} schema versions']
    }


def validate_data_agent(state: SchemaValidationState) -> SchemaValidationState:
    """Validate data samples against schema"""
    
    schema = state['schemas']['user_profile_v1']
    validator = SchemaValidator(schema)
    validation_results = []
    
    for sample in state['data_samples']:
        is_valid, errors = validator.validate(sample)
        
        validation_results.append({
            'sample_id': sample.get('sample_id'),
            'is_valid': is_valid,
            'error_count': len(errors),
            'errors': errors
        })
    
    valid_count = sum(1 for r in validation_results if r['is_valid'])
    
    return {
        **state,
        'validation_results': validation_results,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} samples: {valid_count} valid, '
            f'{len(validation_results) - valid_count} invalid'
        ]
    }


def check_schema_compatibility_agent(state: SchemaValidationState) -> SchemaValidationState:
    """Check compatibility between schema versions"""
    
    checker = SchemaCompatibilityChecker()
    
    v1 = state['schemas']['user_profile_v1']
    v2 = state['schemas']['user_profile_v2']
    
    compatibility = checker.check_compatibility(v1, v2)
    
    return {
        **state,
        'schema_compatibility': compatibility,
        'messages': state['messages'] + [
            f'Schema compatibility check: {"Compatible" if compatibility["is_compatible"] else "Breaking changes detected"}'
        ]
    }


def generate_migration_plan_agent(state: SchemaValidationState) -> SchemaValidationState:
    """Generate migration plan for schema evolution"""
    
    migration_plan = []
    
    compatibility = state['schema_compatibility']
    
    if compatibility['breaking_changes']:
        migration_plan.append({
            'step': 1,
            'action': 'Handle Breaking Changes',
            'details': compatibility['breaking_changes'],
            'priority': 'HIGH',
            'tasks': [
                'Review all breaking changes',
                'Create migration scripts',
                'Update data transformation logic',
                'Notify API consumers'
            ]
        })
    
    if compatibility['additions']:
        migration_plan.append({
            'step': 2,
            'action': 'Add New Fields',
            'details': compatibility['additions'],
            'priority': 'MEDIUM',
            'tasks': [
                'Add new fields to database schema',
                'Set default values for existing records',
                'Update API documentation'
            ]
        })
    
    if compatibility['deprecations']:
        migration_plan.append({
            'step': 3,
            'action': 'Handle Deprecations',
            'details': compatibility['deprecations'],
            'priority': 'LOW',
            'tasks': [
                'Mark fields as optional',
                'Add deprecation warnings',
                'Plan removal timeline'
            ]
        })
    
    migration_plan.append({
        'step': len(migration_plan) + 1,
        'action': 'Testing & Validation',
        'priority': 'HIGH',
        'tasks': [
            'Test with existing data',
            'Validate migration scripts',
            'Perform rollback test',
            'Load testing with new schema'
        ]
    })
    
    return {
        **state,
        'migration_plan': migration_plan,
        'messages': state['messages'] + [f'Generated migration plan with {len(migration_plan)} steps']
    }


def generate_schema_report_agent(state: SchemaValidationState) -> SchemaValidationState:
    """Generate comprehensive schema validation report"""
    
    report_lines = [
        "=" * 80,
        "SCHEMA VALIDATION REPORT",
        "=" * 80,
        "",
        "SCHEMAS DEFINED:",
        "-" * 40
    ]
    
    for schema_name, schema in state['schemas'].items():
        report_lines.extend([
            f"\n{schema_name} (v{schema.get('version', 'unknown')}):",
            f"  Required fields: {', '.join(schema.get('required', []))}",
            f"  Total properties: {len(schema.get('properties', {}))}",
            f"  Type: {schema.get('type', 'unknown')}"
        ])
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        report_lines.append(f"\nSample #{result['sample_id']}: {status}")
        
        if result['errors']:
            report_lines.append(f"  Errors ({result['error_count']}):")
            for error in result['errors']:
                report_lines.append(f"    • {error}")
    
    report_lines.extend([
        "",
        "SCHEMA COMPATIBILITY (v1 → v2):",
        "-" * 40,
        f"Status: {'✓ Compatible' if state['schema_compatibility']['is_compatible'] else '✗ Breaking Changes'}"
    ])
    
    if state['schema_compatibility']['breaking_changes']:
        report_lines.append("\nBreaking Changes:")
        for change in state['schema_compatibility']['breaking_changes']:
            report_lines.append(f"  ✗ {change}")
    
    if state['schema_compatibility']['additions']:
        report_lines.append("\nAdditions:")
        for addition in state['schema_compatibility']['additions']:
            report_lines.append(f"  + {addition}")
    
    if state['schema_compatibility']['deprecations']:
        report_lines.append("\nDeprecations:")
        for deprecation in state['schema_compatibility']['deprecations']:
            report_lines.append(f"  - {deprecation}")
    
    report_lines.extend([
        "",
        "MIGRATION PLAN:",
        "-" * 40
    ])
    
    for step in state['migration_plan']:
        report_lines.extend([
            f"\nStep {step['step']}: {step['action']} [Priority: {step['priority']}]",
            "  Tasks:"
        ])
        for task in step['tasks']:
            report_lines.append(f"    □ {task}")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "✓ Use schema versioning for all data contracts",
        "✓ Implement backward compatibility when possible",
        "✓ Document all schema changes",
        "✓ Validate data at API boundaries",
        "✓ Automate schema compatibility checks in CI/CD",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive schema validation report']
    }


# Create the graph
def create_schema_validation_graph():
    """Create the schema validation workflow graph"""
    
    workflow = StateGraph(SchemaValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_schema_validation_agent)
    workflow.add_node("define_schemas", define_schemas_agent)
    workflow.add_node("validate_data", validate_data_agent)
    workflow.add_node("check_compatibility", check_schema_compatibility_agent)
    workflow.add_node("generate_migration", generate_migration_plan_agent)
    workflow.add_node("generate_report", generate_schema_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_schemas")
    workflow.add_edge("define_schemas", "validate_data")
    workflow.add_edge("validate_data", "check_compatibility")
    workflow.add_edge("check_compatibility", "generate_migration")
    workflow.add_edge("generate_migration", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the schema validation graph
    app = create_schema_validation_graph()
    
    # Initialize state
    initial_state: SchemaValidationState = {
        'messages': [],
        'data_samples': [],
        'schemas': {},
        'validation_results': [],
        'schema_compatibility': {},
        'migration_plan': [],
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("SCHEMA VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nSchema validation pattern execution complete! ✓")

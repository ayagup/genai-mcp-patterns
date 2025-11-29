"""
Pattern 304: Schema Transformation MCP Pattern

This pattern demonstrates transforming data schemas, including version
migration, schema evolution, and structural changes.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import copy


class SchemaTransformationPattern(TypedDict):
    """State for schema transformation"""
    messages: Annotated[List[str], add]
    source_schema: Dict[str, Any]
    target_schema: Dict[str, Any]
    migration_rules: List[Dict[str, Any]]
    sample_data: Dict[str, Any]
    migrated_data: Dict[str, Any]
    schema_statistics: Dict[str, Any]


class SchemaField:
    """Schema field definition"""
    
    def __init__(self, name: str, field_type: str, required: bool = False, 
                 default: Any = None, constraints: Optional[Dict] = None):
        self.name = name
        self.field_type = field_type
        self.required = required
        self.default = default
        self.constraints = constraints or {}
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "name": self.name,
            "type": self.field_type,
            "required": self.required,
            "default": self.default,
            "constraints": self.constraints
        }


class Schema:
    """Schema definition"""
    
    def __init__(self, version: str):
        self.version = version
        self.fields = {}
    
    def add_field(self, field: SchemaField):
        """Add a field to schema"""
        self.fields[field.name] = field
    
    def remove_field(self, field_name: str):
        """Remove a field from schema"""
        if field_name in self.fields:
            del self.fields[field_name]
    
    def get_field(self, field_name: str) -> Optional[SchemaField]:
        """Get a field by name"""
        return self.fields.get(field_name)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "version": self.version,
            "fields": {name: field.to_dict() for name, field in self.fields.items()}
        }


class SchemaMigrator:
    """Migrate data between schema versions"""
    
    def __init__(self):
        self.migrations = []
    
    def add_migration(self, migration: dict):
        """Add a migration rule"""
        self.migrations.append(migration)
    
    def migrate(self, data: dict, source_schema: Schema, target_schema: Schema) -> dict:
        """Migrate data from source to target schema"""
        migrated = {}
        
        # Process each target field
        for field_name, target_field in target_schema.fields.items():
            
            # Check if field exists in source
            if field_name in data:
                # Direct copy
                migrated[field_name] = data[field_name]
            
            # Check for field renaming in migrations
            else:
                renamed_from = self._find_renamed_field(field_name)
                if renamed_from and renamed_from in data:
                    migrated[field_name] = data[renamed_from]
                
                # Check for computed fields
                elif self._is_computed_field(field_name):
                    migrated[field_name] = self._compute_field(field_name, data)
                
                # Use default value
                elif target_field.default is not None:
                    migrated[field_name] = target_field.default
                
                # Required field without value
                elif target_field.required:
                    raise ValueError(f"Required field '{field_name}' missing and no default provided")
        
        return migrated
    
    def _find_renamed_field(self, new_name: str) -> Optional[str]:
        """Find if field was renamed"""
        for migration in self.migrations:
            if migration.get("type") == "rename_field":
                if migration.get("new_name") == new_name:
                    return migration.get("old_name")
        return None
    
    def _is_computed_field(self, field_name: str) -> bool:
        """Check if field is computed"""
        for migration in self.migrations:
            if migration.get("type") == "add_computed_field":
                if migration.get("field_name") == field_name:
                    return True
        return False
    
    def _compute_field(self, field_name: str, data: dict) -> Any:
        """Compute field value"""
        for migration in self.migrations:
            if migration.get("type") == "add_computed_field":
                if migration.get("field_name") == field_name:
                    compute_fn = migration.get("compute_function")
                    if compute_fn:
                        return compute_fn(data)
        return None


def initialize_schema_migration_agent(state: SchemaTransformationPattern) -> SchemaTransformationPattern:
    """Initialize schema migration system"""
    print("\n🔄 Initializing Schema Migration System...")
    
    # Source schema (v1.0)
    source_schema = Schema("1.0")
    source_schema.add_field(SchemaField("user_id", "string", required=True))
    source_schema.add_field(SchemaField("first_name", "string", required=True))
    source_schema.add_field(SchemaField("last_name", "string", required=True))
    source_schema.add_field(SchemaField("email", "string", required=True))
    source_schema.add_field(SchemaField("age", "integer", required=False))
    source_schema.add_field(SchemaField("created_date", "string", required=True))
    
    # Target schema (v2.0)
    target_schema = Schema("2.0")
    target_schema.add_field(SchemaField("id", "string", required=True))
    target_schema.add_field(SchemaField("full_name", "string", required=True))
    target_schema.add_field(SchemaField("email_address", "string", required=True))
    target_schema.add_field(SchemaField("date_of_birth", "string", required=False))
    target_schema.add_field(SchemaField("created_at", "string", required=True))
    target_schema.add_field(SchemaField("updated_at", "string", required=True, default="2024-01-01T00:00:00Z"))
    target_schema.add_field(SchemaField("status", "string", required=True, default="active"))
    
    print(f"  Source Schema Version: {source_schema.version}")
    print(f"  Source Fields: {len(source_schema.fields)}")
    
    print(f"\n  Target Schema Version: {target_schema.version}")
    print(f"  Target Fields: {len(target_schema.fields)}")
    
    print(f"\n  Migration Type: v{source_schema.version} → v{target_schema.version}")
    
    # Sample data
    sample_data = {
        "user_id": "U12345",
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "created_date": "2023-01-15"
    }
    
    return {
        **state,
        "source_schema": source_schema.to_dict(),
        "target_schema": target_schema.to_dict(),
        "migration_rules": [],
        "sample_data": sample_data,
        "migrated_data": {},
        "schema_statistics": {},
        "messages": ["✓ Schema migration system initialized"]
    }


def define_migration_rules_agent(state: SchemaTransformationPattern) -> SchemaTransformationPattern:
    """Define migration rules"""
    print("\n📋 Defining Migration Rules...")
    
    migration_rules = []
    
    # Rule 1: Rename field
    migration_rules.append({
        "rule_id": "M1",
        "type": "rename_field",
        "old_name": "user_id",
        "new_name": "id",
        "description": "Rename user_id to id"
    })
    print(f"  M1: Rename field 'user_id' → 'id'")
    
    # Rule 2: Combine fields
    migration_rules.append({
        "rule_id": "M2",
        "type": "add_computed_field",
        "field_name": "full_name",
        "source_fields": ["first_name", "last_name"],
        "description": "Combine first_name and last_name into full_name"
    })
    print(f"  M2: Combine 'first_name' + 'last_name' → 'full_name'")
    
    # Rule 3: Rename field
    migration_rules.append({
        "rule_id": "M3",
        "type": "rename_field",
        "old_name": "email",
        "new_name": "email_address",
        "description": "Rename email to email_address"
    })
    print(f"  M3: Rename field 'email' → 'email_address'")
    
    # Rule 4: Remove field
    migration_rules.append({
        "rule_id": "M4",
        "type": "remove_field",
        "field_name": "age",
        "description": "Remove age field (replaced by date_of_birth)"
    })
    print(f"  M4: Remove field 'age'")
    
    # Rule 5: Rename field
    migration_rules.append({
        "rule_id": "M5",
        "type": "rename_field",
        "old_name": "created_date",
        "new_name": "created_at",
        "description": "Rename created_date to created_at"
    })
    print(f"  M5: Rename field 'created_date' → 'created_at'")
    
    # Rule 6: Add new field
    migration_rules.append({
        "rule_id": "M6",
        "type": "add_field",
        "field_name": "updated_at",
        "default_value": "2024-01-01T00:00:00Z",
        "description": "Add updated_at field with default value"
    })
    print(f"  M6: Add field 'updated_at' with default")
    
    # Rule 7: Add new field
    migration_rules.append({
        "rule_id": "M7",
        "type": "add_field",
        "field_name": "status",
        "default_value": "active",
        "description": "Add status field with default 'active'"
    })
    print(f"  M7: Add field 'status' with default 'active'")
    
    print(f"\n  Total Migration Rules: {len(migration_rules)}")
    
    return {
        **state,
        "migration_rules": migration_rules,
        "messages": [f"✓ Defined {len(migration_rules)} migration rules"]
    }


def execute_schema_migration_agent(state: SchemaTransformationPattern) -> SchemaTransformationPattern:
    """Execute schema migration"""
    print("\n⚙️ Executing Schema Migration...")
    
    # Rebuild schemas
    source_schema = Schema(state["source_schema"]["version"])
    for field_name, field_data in state["source_schema"]["fields"].items():
        field = SchemaField(
            field_data["name"],
            field_data["type"],
            field_data["required"],
            field_data["default"]
        )
        source_schema.add_field(field)
    
    target_schema = Schema(state["target_schema"]["version"])
    for field_name, field_data in state["target_schema"]["fields"].items():
        field = SchemaField(
            field_data["name"],
            field_data["type"],
            field_data["required"],
            field_data["default"]
        )
        target_schema.add_field(field)
    
    # Setup migrator
    migrator = SchemaMigrator()
    
    # Add migrations with compute functions
    for rule in state["migration_rules"]:
        if rule["type"] == "rename_field":
            migrator.add_migration(rule)
        elif rule["type"] == "add_computed_field" and rule["field_name"] == "full_name":
            migrator.add_migration({
                **rule,
                "compute_function": lambda data: f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
            })
    
    # Execute migration
    sample_data = state["sample_data"]
    migrated_data = migrator.migrate(sample_data, source_schema, target_schema)
    
    print(f"  Source Data Fields: {len(sample_data)}")
    print(f"  Migrated Data Fields: {len(migrated_data)}")
    
    print(f"\n  Source Data:")
    for key, value in sample_data.items():
        print(f"    {key}: {value}")
    
    print(f"\n  Migrated Data:")
    for key, value in migrated_data.items():
        print(f"    {key}: {value}")
    
    return {
        **state,
        "migrated_data": migrated_data,
        "messages": ["✓ Schema migration executed"]
    }


def validate_migrated_schema_agent(state: SchemaTransformationPattern) -> SchemaTransformationPattern:
    """Validate migrated data against target schema"""
    print("\n✅ Validating Migrated Data...")
    
    target_schema = state["target_schema"]
    migrated_data = state["migrated_data"]
    
    validation_errors = []
    
    # Check all required fields
    for field_name, field_data in target_schema["fields"].items():
        if field_data["required"]:
            if field_name not in migrated_data:
                validation_errors.append(f"Missing required field: {field_name}")
    
    # Check field types (simplified)
    for field_name, value in migrated_data.items():
        if field_name in target_schema["fields"]:
            expected_type = target_schema["fields"][field_name]["type"]
            actual_type = type(value).__name__
            
            type_map = {
                "string": "str",
                "integer": "int",
                "float": "float",
                "boolean": "bool"
            }
            
            if type_map.get(expected_type) and actual_type != type_map[expected_type]:
                validation_errors.append(f"Type mismatch for {field_name}: expected {expected_type}, got {actual_type}")
    
    if validation_errors:
        print(f"  ✗ Validation Failed: {len(validation_errors)} errors")
        for error in validation_errors:
            print(f"    • {error}")
    else:
        print(f"  ✓ Validation Passed")
        print(f"  All required fields present")
        print(f"  All type constraints satisfied")
    
    # Calculate statistics
    source_fields = len(state["source_schema"]["fields"])
    target_fields = len(state["target_schema"]["fields"])
    migrated_fields = len(migrated_data)
    
    rules_applied = len(state["migration_rules"])
    
    statistics = {
        "source_fields": source_fields,
        "target_fields": target_fields,
        "migrated_fields": migrated_fields,
        "migration_rules": rules_applied,
        "validation_passed": len(validation_errors) == 0,
        "validation_errors": len(validation_errors),
        "fields_added": target_fields - source_fields,
        "fields_removed": max(0, source_fields - target_fields)
    }
    
    print(f"\n  Statistics:")
    print(f"    Source Fields: {statistics['source_fields']}")
    print(f"    Target Fields: {statistics['target_fields']}")
    print(f"    Migration Rules: {statistics['migration_rules']}")
    print(f"    Validation: {'✓ PASSED' if statistics['validation_passed'] else '✗ FAILED'}")
    
    return {
        **state,
        "schema_statistics": statistics,
        "messages": ["✓ Migration validated"]
    }


def generate_schema_migration_report_agent(state: SchemaTransformationPattern) -> SchemaTransformationPattern:
    """Generate schema migration report"""
    print("\n" + "="*70)
    print("SCHEMA TRANSFORMATION REPORT")
    print("="*70)
    
    print(f"\n📥 Source Schema (v{state['source_schema']['version']}):")
    for field_name, field_data in state["source_schema"]["fields"].items():
        req_marker = "*" if field_data["required"] else ""
        print(f"  {field_name}{req_marker}: {field_data['type']}")
    
    print(f"\n📤 Target Schema (v{state['target_schema']['version']}):")
    for field_name, field_data in state["target_schema"]["fields"].items():
        req_marker = "*" if field_data["required"] else ""
        default_info = f" = {field_data['default']}" if field_data["default"] else ""
        print(f"  {field_name}{req_marker}: {field_data['type']}{default_info}")
    
    print(f"\n🔄 Migration Rules:")
    for rule in state["migration_rules"]:
        print(f"\n  {rule['rule_id']}: {rule['type']}")
        if "old_name" in rule:
            print(f"    From: {rule['old_name']}")
        if "new_name" in rule:
            print(f"    To: {rule['new_name']}")
        if "field_name" in rule:
            print(f"    Field: {rule['field_name']}")
        if "description" in rule:
            print(f"    Description: {rule['description']}")
    
    print(f"\n📊 Migration Result:")
    print(f"  Source Data:")
    for key, value in state["sample_data"].items():
        print(f"    {key}: {value}")
    
    print(f"\n  Migrated Data:")
    for key, value in state["migrated_data"].items():
        print(f"    {key}: {value}")
    
    print(f"\n📈 Statistics:")
    stats = state["schema_statistics"]
    if stats:
        print(f"  Source Fields: {stats['source_fields']}")
        print(f"  Target Fields: {stats['target_fields']}")
        print(f"  Migration Rules: {stats['migration_rules']}")
        print(f"  Fields Added: {stats['fields_added']}")
        print(f"  Fields Removed: {stats['fields_removed']}")
        print(f"  Validation: {'✓ PASSED' if stats['validation_passed'] else '✗ FAILED'}")
    
    print(f"\n💡 Schema Transformation Benefits:")
    print("  ✓ Version migration")
    print("  ✓ Database evolution")
    print("  ✓ API versioning")
    print("  ✓ Backward compatibility")
    print("  ✓ Data integrity")
    print("  ✓ Schema validation")
    
    print(f"\n🔧 Migration Types:")
    print("  • Add field")
    print("  • Remove field")
    print("  • Rename field")
    print("  • Change type")
    print("  • Combine fields")
    print("  • Split fields")
    print("  • Add default values")
    print("  • Add constraints")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Database migrations")
    print("  • API versioning")
    print("  • Data model evolution")
    print("  • Legacy system updates")
    print("  • Breaking changes")
    print("  • Schema refactoring")
    
    print(f"\n🎯 Best Practices:")
    print("  • Version schemas")
    print("  • Use migration scripts")
    print("  • Validate before/after")
    print("  • Support rollback")
    print("  • Document changes")
    print("  • Test migrations")
    
    print("\n" + "="*70)
    print("✅ Schema Transformation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_schema_transformation_graph():
    """Create schema transformation workflow"""
    workflow = StateGraph(SchemaTransformationPattern)
    
    workflow.add_node("initialize", initialize_schema_migration_agent)
    workflow.add_node("define_rules", define_migration_rules_agent)
    workflow.add_node("execute", execute_schema_migration_agent)
    workflow.add_node("validate", validate_migrated_schema_agent)
    workflow.add_node("report", generate_schema_migration_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "execute")
    workflow.add_edge("execute", "validate")
    workflow.add_edge("validate", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 304: Schema Transformation MCP Pattern")
    print("="*70)
    
    app = create_schema_transformation_graph()
    final_state = app.invoke({
        "messages": [],
        "source_schema": {},
        "target_schema": {},
        "migration_rules": [],
        "sample_data": {},
        "migrated_data": {},
        "schema_statistics": {}
    })
    
    print("\n✅ Schema Transformation Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Pattern 235: Schema Evolution MCP Pattern

This pattern demonstrates schema evolution - managing changes to data schemas
over time while maintaining compatibility.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class SchemaEvolutionState(TypedDict):
    """State for schema evolution workflow"""
    messages: Annotated[List[str], add]
    schema_versions: List[dict]
    migration_results: List[dict]


# Schema Definitions
class SchemaV1:
    """Original schema version"""
    
    @staticmethod
    def get_schema() -> Dict[str, str]:
        return {
            "id": "integer",
            "name": "string",
            "email": "string"
        }
    
    @staticmethod
    def create_record(id: int, name: str, email: str) -> Dict[str, Any]:
        return {"id": id, "name": name, "email": email}


class SchemaV2:
    """Evolved schema with additional fields"""
    
    @staticmethod
    def get_schema() -> Dict[str, str]:
        return {
            "id": "integer",
            "name": "string",
            "email": "string",
            "phone": "string (optional)",
            "created_at": "timestamp"
        }
    
    @staticmethod
    def create_record(id: int, name: str, email: str, phone: str = "", created_at: str = "2024-01-01") -> Dict[str, Any]:
        return {
            "id": id,
            "name": name,
            "email": email,
            "phone": phone,
            "created_at": created_at
        }


class SchemaV3:
    """Further evolved schema with restructured fields"""
    
    @staticmethod
    def get_schema() -> Dict[str, str]:
        return {
            "id": "integer",
            "full_name": "string (renamed from 'name')",
            "contact": "object {email, phone}",
            "metadata": "object {created_at, updated_at}"
        }
    
    @staticmethod
    def create_record(id: int, full_name: str, email: str, phone: str = "", created_at: str = "2024-01-01") -> Dict[str, Any]:
        return {
            "id": id,
            "full_name": full_name,
            "contact": {
                "email": email,
                "phone": phone
            },
            "metadata": {
                "created_at": created_at,
                "updated_at": "2024-01-01"
            }
        }


# Schema Migration
class SchemaMigrator:
    """Handles schema migrations"""
    
    @staticmethod
    def migrate_v1_to_v2(v1_record: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from V1 to V2 schema"""
        return {
            **v1_record,
            "phone": "",  # Add default value for new field
            "created_at": "2024-01-01"  # Add default timestamp
        }
    
    @staticmethod
    def migrate_v2_to_v3(v2_record: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from V2 to V3 schema"""
        return {
            "id": v2_record["id"],
            "full_name": v2_record["name"],  # Rename field
            "contact": {
                "email": v2_record["email"],
                "phone": v2_record.get("phone", "")
            },
            "metadata": {
                "created_at": v2_record.get("created_at", "2024-01-01"),
                "updated_at": "2024-01-01"
            }
        }
    
    @staticmethod
    def migrate_v1_to_v3(v1_record: Dict[str, Any]) -> Dict[str, Any]:
        """Direct migration from V1 to V3"""
        v2_record = SchemaMigrator.migrate_v1_to_v2(v1_record)
        return SchemaMigrator.migrate_v2_to_v3(v2_record)


# Agent functions
def define_schema_versions_agent(state: SchemaEvolutionState) -> SchemaEvolutionState:
    """Define all schema versions"""
    print("\n📋 Defining Schema Versions...")
    
    schema_versions = [
        {
            "version": "v1",
            "description": "Original schema - basic fields only",
            "fields": SchemaV1.get_schema(),
            "breaking_changes": []
        },
        {
            "version": "v2",
            "description": "Added optional phone and created_at",
            "fields": SchemaV2.get_schema(),
            "breaking_changes": []
        },
        {
            "version": "v3",
            "description": "Renamed 'name' to 'full_name', nested contact info",
            "fields": SchemaV3.get_schema(),
            "breaking_changes": ["Renamed field: name → full_name", "Nested structure: contact object"]
        }
    ]
    
    print("\n  Schema Evolution Timeline:")
    for schema in schema_versions:
        print(f"\n    {schema['version']}: {schema['description']}")
        if schema['breaking_changes']:
            print(f"      Breaking: {', '.join(schema['breaking_changes'])}")
    
    return {
        **state,
        "schema_versions": schema_versions,
        "messages": [f"✓ Defined {len(schema_versions)} schema versions"]
    }


def test_schema_migrations_agent(state: SchemaEvolutionState) -> SchemaEvolutionState:
    """Test schema migrations"""
    print("\n🔄 Testing Schema Migrations...")
    
    migrator = SchemaMigrator()
    migration_results = []
    
    # Create V1 record
    v1_record = SchemaV1.create_record(1, "John Doe", "john@example.com")
    print(f"\n  Original V1 Record: {v1_record}")
    
    # Migrate V1 → V2
    print("\n  Migrating V1 → V2...")
    v2_record = migrator.migrate_v1_to_v2(v1_record)
    migration_results.append({
        "migration": "V1 → V2",
        "status": "SUCCESS",
        "original": v1_record,
        "migrated": v2_record,
        "added_fields": ["phone", "created_at"]
    })
    print(f"    ✓ V2 Record: {v2_record}")
    
    # Migrate V2 → V3
    print("\n  Migrating V2 → V3...")
    v3_record = migrator.migrate_v2_to_v3(v2_record)
    migration_results.append({
        "migration": "V2 → V3",
        "status": "SUCCESS",
        "original": v2_record,
        "migrated": v3_record,
        "changes": ["Renamed 'name' to 'full_name'", "Nested contact info"]
    })
    print(f"    ✓ V3 Record: {v3_record}")
    
    # Direct migration V1 → V3
    print("\n  Direct Migration V1 → V3...")
    v3_direct = migrator.migrate_v1_to_v3(v1_record)
    migration_results.append({
        "migration": "V1 → V3 (direct)",
        "status": "SUCCESS",
        "original": v1_record,
        "migrated": v3_direct,
        "note": "Skips intermediate V2 schema"
    })
    print(f"    ✓ V3 Record (direct): {v3_direct}")
    
    return {
        **state,
        "migration_results": migration_results,
        "messages": [f"✓ Tested {len(migration_results)} migrations"]
    }


def generate_schema_evolution_report_agent(state: SchemaEvolutionState) -> SchemaEvolutionState:
    """Generate schema evolution report"""
    print("\n" + "="*70)
    print("SCHEMA EVOLUTION REPORT")
    print("="*70)
    
    print(f"\n📚 Schema Versions: {len(state['schema_versions'])}")
    for schema in state["schema_versions"]:
        print(f"\n  {schema['version']}: {schema['description']}")
        print(f"    Fields: {list(schema['fields'].keys())}")
        if schema['breaking_changes']:
            print(f"    Breaking Changes: {', '.join(schema['breaking_changes'])}")
    
    print(f"\n🔄 Migration Results:")
    for migration in state["migration_results"]:
        print(f"\n  ✓ {migration['migration']}: {migration['status']}")
        if "added_fields" in migration:
            print(f"    Added: {', '.join(migration['added_fields'])}")
        if "changes" in migration:
            for change in migration["changes"]:
                print(f"    • {change}")
    
    print("\n💡 Schema Evolution Best Practices:")
    print("  • Add fields with default values (non-breaking)")
    print("  • Use optional fields for new additions")
    print("  • Provide migration tools for breaking changes")
    print("  • Version your schema explicitly")
    print("  • Support multiple schema versions during transition")
    print("  • Test migrations thoroughly with real data")
    
    print("\n" + "="*70)
    print("✅ Schema Evolution Pattern Complete!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Schema evolution report generated"]
    }


# Create the graph
def create_schema_evolution_graph():
    """Create the schema evolution workflow graph"""
    workflow = StateGraph(SchemaEvolutionState)
    
    # Add nodes
    workflow.add_node("define_schemas", define_schema_versions_agent)
    workflow.add_node("test_migrations", test_schema_migrations_agent)
    workflow.add_node("generate_report", generate_schema_evolution_report_agent)
    
    # Add edges
    workflow.add_edge(START, "define_schemas")
    workflow.add_edge("define_schemas", "test_migrations")
    workflow.add_edge("test_migrations", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 235: Schema Evolution MCP Pattern")
    print("="*70)
    print("\nSchema Evolution: Manage data schema changes over time")
    
    # Create and run the workflow
    app = create_schema_evolution_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "schema_versions": [],
        "migration_results": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Schema Evolution Pattern Complete!")


if __name__ == "__main__":
    main()

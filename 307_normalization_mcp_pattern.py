"""
Pattern 307: Normalization MCP Pattern

This pattern demonstrates data normalization, including database normalization,
value standardization, and data cleaning.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import re


class NormalizationPattern(TypedDict):
    """State for normalization"""
    messages: Annotated[List[str], add]
    raw_data: List[Dict[str, Any]]
    normalized_tables: Dict[str, List[Dict[str, Any]]]
    normalization_rules: List[Dict[str, Any]]
    normalization_statistics: Dict[str, Any]


class DataNormalizer:
    """Normalize data"""
    
    def __init__(self):
        self.rules = []
    
    def normalize_phone(self, phone: str) -> str:
        """Normalize phone number"""
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)
        
        # Format as (XXX) XXX-XXXX
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        return phone
    
    def normalize_email(self, email: str) -> str:
        """Normalize email address"""
        return email.lower().strip()
    
    def normalize_name(self, name: str) -> str:
        """Normalize name"""
        # Title case
        return name.strip().title()
    
    def normalize_string(self, value: str) -> str:
        """General string normalization"""
        # Trim whitespace, normalize case
        return value.strip()
    
    def normalize_date(self, date_str: str) -> str:
        """Normalize date format"""
        # Simple normalization (in production, use date parsing library)
        # Convert various formats to ISO format
        return date_str  # Simplified for demo


class DatabaseNormalizer:
    """Normalize database schema"""
    
    def normalize_to_1nf(self, data: List[dict]) -> List[dict]:
        """Convert to First Normal Form (1NF)"""
        # Remove repeating groups, ensure atomic values
        normalized = []
        
        for record in data:
            new_record = {}
            for key, value in record.items():
                if isinstance(value, list):
                    # Split into multiple records
                    for item in value:
                        new_rec = new_record.copy()
                        new_rec[key] = item
                        normalized.append(new_rec)
                else:
                    new_record[key] = value
            
            if not any(isinstance(v, list) for v in record.values()):
                normalized.append(new_record)
        
        return normalized if normalized else data
    
    def normalize_to_2nf(self, table: List[dict], primary_key: str) -> dict:
        """Convert to Second Normal Form (2NF)"""
        # Remove partial dependencies
        # Split into multiple tables based on dependencies
        
        main_table = []
        dependent_tables = {}
        
        for record in table:
            # Keep only full-key dependent fields
            main_record = {primary_key: record.get(primary_key)}
            main_table.append(main_record)
        
        return {
            "main": main_table,
            "dependent": dependent_tables
        }
    
    def normalize_to_3nf(self, tables: dict) -> dict:
        """Convert to Third Normal Form (3NF)"""
        # Remove transitive dependencies
        # Further split tables
        
        result = {}
        for table_name, records in tables.items():
            result[table_name] = records
        
        return result


def initialize_normalization_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Initialize normalization system"""
    print("\n🔧 Initializing Normalization System...")
    
    # Raw unnormalized data
    raw_data = [
        {
            "customer_id": 1,
            "name": "  JOHN DOE  ",
            "email": "John.Doe@EXAMPLE.COM",
            "phone": "555-0123",
            "orders": ["ORD001", "ORD002"],
            "address": "123 Main St, New York, NY 10001"
        },
        {
            "customer_id": 2,
            "name": "jane smith",
            "email": "  Jane@Example.com  ",
            "phone": "(555) 456-7890",
            "orders": ["ORD003"],
            "address": "456 Oak Ave, Los Angeles, CA 90001"
        },
        {
            "customer_id": 3,
            "name": "Bob JOHNSON",
            "email": "BOB.J@SAMPLE.ORG",
            "phone": "5551234567",
            "orders": ["ORD004", "ORD005", "ORD006"],
            "address": "789 Pine Rd, Chicago, IL 60601"
        }
    ]
    
    print(f"  Raw Data Records: {len(raw_data)}")
    print(f"\n  Data Quality Issues:")
    print(f"    • Inconsistent name casing")
    print(f"    • Email whitespace and casing")
    print(f"    • Phone number formats vary")
    print(f"    • Repeating groups (orders)")
    print(f"    • Composite fields (address)")
    
    return {
        **state,
        "raw_data": raw_data,
        "normalized_tables": {},
        "normalization_rules": [],
        "normalization_statistics": {},
        "messages": ["✓ Normalization system initialized"]
    }


def normalize_values_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Normalize field values"""
    print("\n🧹 Normalizing Field Values...")
    
    normalizer = DataNormalizer()
    raw_data = state["raw_data"]
    
    normalized_data = []
    rules_applied = []
    
    for record in raw_data:
        normalized_record = record.copy()
        
        # Normalize name
        if "name" in normalized_record:
            original = normalized_record["name"]
            normalized_record["name"] = normalizer.normalize_name(original)
            if original != normalized_record["name"]:
                rules_applied.append({
                    "field": "name",
                    "rule": "title_case",
                    "before": original,
                    "after": normalized_record["name"]
                })
        
        # Normalize email
        if "email" in normalized_record:
            original = normalized_record["email"]
            normalized_record["email"] = normalizer.normalize_email(original)
            if original != normalized_record["email"]:
                rules_applied.append({
                    "field": "email",
                    "rule": "lowercase_trim",
                    "before": original,
                    "after": normalized_record["email"]
                })
        
        # Normalize phone
        if "phone" in normalized_record:
            original = normalized_record["phone"]
            normalized_record["phone"] = normalizer.normalize_phone(original)
            if original != normalized_record["phone"]:
                rules_applied.append({
                    "field": "phone",
                    "rule": "format_phone",
                    "before": original,
                    "after": normalized_record["phone"]
                })
        
        normalized_data.append(normalized_record)
    
    print(f"  Normalization Rules Applied: {len(rules_applied)}")
    print(f"\n  Sample Normalizations:")
    for rule in rules_applied[:5]:
        print(f"    {rule['field']}: '{rule['before']}' → '{rule['after']}'")
    
    return {
        **state,
        "raw_data": normalized_data,
        "normalization_rules": rules_applied,
        "messages": [f"✓ Applied {len(rules_applied)} normalization rules"]
    }


def normalize_to_1nf_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Normalize to First Normal Form (1NF)"""
    print("\n📋 Normalizing to 1NF (First Normal Form)...")
    
    print(f"  1NF Requirements:")
    print(f"    • Atomic values (no repeating groups)")
    print(f"    • Each field contains only one value")
    print(f"    • No duplicate rows")
    
    # Separate customers and orders
    customers = []
    orders = []
    
    for record in state["raw_data"]:
        # Customer table
        customer = {
            "customer_id": record["customer_id"],
            "name": record["name"],
            "email": record["email"],
            "phone": record["phone"],
            "address": record["address"]
        }
        customers.append(customer)
        
        # Orders table (expand repeating group)
        for order_id in record.get("orders", []):
            orders.append({
                "order_id": order_id,
                "customer_id": record["customer_id"]
            })
    
    print(f"\n  Result:")
    print(f"    Customers Table: {len(customers)} records")
    print(f"    Orders Table: {len(orders)} records")
    print(f"    ✓ Removed repeating groups")
    
    normalized_tables = {
        "customers": customers,
        "orders": orders
    }
    
    return {
        **state,
        "normalized_tables": normalized_tables,
        "messages": ["✓ Normalized to 1NF"]
    }


def normalize_to_2nf_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Normalize to Second Normal Form (2NF)"""
    print("\n📋 Normalizing to 2NF (Second Normal Form)...")
    
    print(f"  2NF Requirements:")
    print(f"    • Must be in 1NF")
    print(f"    • Remove partial dependencies")
    print(f"    • Non-key fields depend on entire primary key")
    
    # Split address into separate table
    customers = state["normalized_tables"]["customers"]
    
    customer_info = []
    addresses = []
    
    address_id = 1
    for customer in customers:
        # Parse address (simplified)
        address_parts = customer["address"].split(", ")
        
        # Customer info (without address details)
        customer_info.append({
            "customer_id": customer["customer_id"],
            "name": customer["name"],
            "email": customer["email"],
            "phone": customer["phone"],
            "address_id": address_id
        })
        
        # Address table
        addresses.append({
            "address_id": address_id,
            "street": address_parts[0] if len(address_parts) > 0 else "",
            "city": address_parts[1] if len(address_parts) > 1 else "",
            "state_zip": address_parts[2] if len(address_parts) > 2 else ""
        })
        
        address_id += 1
    
    print(f"\n  Result:")
    print(f"    Customer Info: {len(customer_info)} records")
    print(f"    Addresses: {len(addresses)} records")
    print(f"    ✓ Removed partial dependencies")
    
    normalized_tables = state["normalized_tables"].copy()
    normalized_tables["customers"] = customer_info
    normalized_tables["addresses"] = addresses
    
    return {
        **state,
        "normalized_tables": normalized_tables,
        "messages": ["✓ Normalized to 2NF"]
    }


def normalize_to_3nf_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Normalize to Third Normal Form (3NF)"""
    print("\n📋 Normalizing to 3NF (Third Normal Form)...")
    
    print(f"  3NF Requirements:")
    print(f"    • Must be in 2NF")
    print(f"    • Remove transitive dependencies")
    print(f"    • Non-key fields depend only on primary key")
    
    # Split state/zip into states table
    addresses = state["normalized_tables"]["addresses"]
    
    addresses_normalized = []
    states = []
    state_id_map = {}
    state_id = 1
    
    for address in addresses:
        # Parse state and zip
        state_zip = address.get("state_zip", "")
        parts = state_zip.split(" ")
        state_code = parts[0] if len(parts) > 0 else ""
        zip_code = parts[1] if len(parts) > 1 else ""
        
        # Check if state already in states table
        if state_code not in state_id_map:
            states.append({
                "state_id": state_id,
                "state_code": state_code,
                "state_name": self._get_state_name(state_code)
            })
            state_id_map[state_code] = state_id
            state_id += 1
        
        # Address without state name (only reference)
        addresses_normalized.append({
            "address_id": address["address_id"],
            "street": address["street"],
            "city": address["city"],
            "state_id": state_id_map.get(state_code, 0),
            "zip_code": zip_code
        })
    
    print(f"\n  Result:")
    print(f"    Addresses: {len(addresses_normalized)} records")
    print(f"    States: {len(states)} records")
    print(f"    ✓ Removed transitive dependencies")
    
    normalized_tables = state["normalized_tables"].copy()
    normalized_tables["addresses"] = addresses_normalized
    normalized_tables["states"] = states
    
    return {
        **state,
        "normalized_tables": normalized_tables,
        "messages": ["✓ Normalized to 3NF"]
    }

def _get_state_name(state_code: str) -> str:
    """Get full state name from code"""
    state_names = {
        "NY": "New York",
        "CA": "California",
        "IL": "Illinois"
    }
    return state_names.get(state_code, state_code)


def analyze_normalization_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Analyze normalization results"""
    print("\n📊 Analyzing Normalization Results...")
    
    # Count tables and records
    tables = state["normalized_tables"]
    table_count = len(tables)
    total_records = sum(len(records) for records in tables.values())
    
    # Original data
    original_records = len(state["raw_data"])
    
    # Normalization rules
    rules_applied = len(state["normalization_rules"])
    
    # Calculate redundancy reduction
    original_fields = sum(len(record.keys()) for record in state["raw_data"])
    normalized_fields = sum(
        sum(len(record.keys()) for record in records)
        for records in tables.values()
    )
    
    statistics = {
        "original_records": original_records,
        "normalized_tables": table_count,
        "total_records": total_records,
        "rules_applied": rules_applied,
        "original_fields": original_fields,
        "normalized_fields": normalized_fields,
        "tables": {name: len(records) for name, records in tables.items()}
    }
    
    print(f"  Original Records: {statistics['original_records']}")
    print(f"  Normalized Tables: {statistics['normalized_tables']}")
    print(f"  Total Records: {statistics['total_records']}")
    print(f"  Rules Applied: {statistics['rules_applied']}")
    
    print(f"\n  Table Distribution:")
    for table_name, count in statistics["tables"].items():
        print(f"    {table_name}: {count} records")
    
    return {
        **state,
        "normalization_statistics": statistics,
        "messages": ["✓ Normalization analysis complete"]
    }


def generate_normalization_report_agent(state: NormalizationPattern) -> NormalizationPattern:
    """Generate normalization report"""
    print("\n" + "="*70)
    print("NORMALIZATION REPORT")
    print("="*70)
    
    print(f"\n📥 Original Data (Unnormalized):")
    for i, record in enumerate(state["raw_data"][:2], 1):
        print(f"\n  Record {i}:")
        for key, value in record.items():
            value_str = str(value)[:50]
            print(f"    {key}: {value_str}")
    
    print(f"\n🔧 Normalization Rules Applied:")
    for i, rule in enumerate(state["normalization_rules"][:6], 1):
        print(f"\n  {i}. {rule['field']} - {rule['rule']}")
        print(f"     Before: {rule['before']}")
        print(f"     After: {rule['after']}")
    
    print(f"\n📋 Normalized Tables:")
    for table_name, records in state["normalized_tables"].items():
        print(f"\n  {table_name.upper()} ({len(records)} records):")
        if records:
            # Show first record as example
            print(f"    Fields: {', '.join(records[0].keys())}")
            print(f"    Sample: {records[0]}")
    
    print(f"\n📊 Statistics:")
    stats = state["normalization_statistics"]
    if stats:
        print(f"  Original Records: {stats['original_records']}")
        print(f"  Normalized Tables: {stats['normalized_tables']}")
        print(f"  Total Records: {stats['total_records']}")
        print(f"  Rules Applied: {stats['rules_applied']}")
    
    print(f"\n💡 Normalization Benefits:")
    print("  ✓ Eliminate data redundancy")
    print("  ✓ Reduce update anomalies")
    print("  ✓ Improve data integrity")
    print("  ✓ Efficient storage")
    print("  ✓ Easier maintenance")
    print("  ✓ Better query performance")
    
    print(f"\n🔧 Normal Forms:")
    print("  • 1NF: Atomic values, no repeating groups")
    print("  • 2NF: 1NF + no partial dependencies")
    print("  • 3NF: 2NF + no transitive dependencies")
    print("  • BCNF: 3NF + every determinant is a key")
    print("  • 4NF: BCNF + no multi-valued dependencies")
    print("  • 5NF: 4NF + no join dependencies")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Database design")
    print("  • Data warehousing")
    print("  • ETL processes")
    print("  • Data migration")
    print("  • Schema optimization")
    print("  • Data quality")
    
    print(f"\n🎯 Normalization vs. Denormalization:")
    print("  Normalization: Minimize redundancy")
    print("  Denormalization: Optimize read performance")
    print("  Balance based on use case")
    
    print("\n" + "="*70)
    print("✅ Normalization Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_normalization_graph():
    """Create normalization workflow"""
    workflow = StateGraph(NormalizationPattern)
    
    workflow.add_node("initialize", initialize_normalization_agent)
    workflow.add_node("normalize_values", normalize_values_agent)
    workflow.add_node("to_1nf", normalize_to_1nf_agent)
    workflow.add_node("to_2nf", normalize_to_2nf_agent)
    workflow.add_node("to_3nf", normalize_to_3nf_agent)
    workflow.add_node("analyze", analyze_normalization_agent)
    workflow.add_node("report", generate_normalization_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "normalize_values")
    workflow.add_edge("normalize_values", "to_1nf")
    workflow.add_edge("to_1nf", "to_2nf")
    workflow.add_edge("to_2nf", "to_3nf")
    workflow.add_edge("to_3nf", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 307: Normalization MCP Pattern")
    print("="*70)
    
    app = create_normalization_graph()
    final_state = app.invoke({
        "messages": [],
        "raw_data": [],
        "normalized_tables": {},
        "normalization_rules": [],
        "normalization_statistics": {}
    })
    
    print("\n✅ Normalization Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Pattern 215: Anti-Corruption Layer MCP Pattern

Anti-Corruption Layer isolates modern systems from legacy systems:
- Translates between different domain models
- Prevents legacy concepts from "corrupting" new design
- Provides clean interface to legacy functionality
- Enables gradual migration
- Protects bounded contexts

Purpose:
- Isolate subsystems with different semantics
- Translate between domain models
- Maintain clean architecture
- Enable independent evolution

Benefits:
- Clean separation of concerns
- Easier testing of new code
- Gradual refactoring possible
- Prevents technical debt spread
- Maintains domain integrity

Use Cases:
- Legacy system integration
- Third-party API integration
- Modernization projects
- Microservices migration
- Multi-vendor integration
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class ACLState(TypedDict):
    """State for Anti-Corruption Layer operations"""
    acl_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class LegacyCustomer:
    """Legacy system customer representation"""
    cust_id: int
    f_name: str
    l_name: str
    addr_line_1: str
    addr_line_2: str
    city: str
    state: str
    zip_code: str
    ph_num: str
    email_addr: str
    cust_type: str  # "R" for retail, "W" for wholesale


class LegacySystem:
    """Simulated legacy system with old data model"""
    
    def __init__(self):
        self.customers = {
            1: LegacyCustomer(1, "John", "Doe", "123 Main St", "", "Springfield", "IL", "62701", "555-1234", "john@example.com", "R"),
            2: LegacyCustomer(2, "Jane", "Smith", "456 Oak Ave", "Apt 2B", "Chicago", "IL", "60601", "555-5678", "jane@example.com", "W")
        }
    
    def get_customer(self, cust_id: int) -> LegacyCustomer:
        """Get customer from legacy system"""
        return self.customers.get(cust_id)
    
    def update_customer(self, cust: LegacyCustomer):
        """Update customer in legacy system"""
        self.customers[cust.cust_id] = cust


@dataclass
class ModernCustomer:
    """Modern system customer representation"""
    id: str
    full_name: str
    email: str
    phone: str
    address: Dict[str, str]
    customer_type: str  # "individual" or "business"


class ModernCRM:
    """Modern CRM system with clean domain model"""
    
    def __init__(self):
        self.customers: Dict[str, ModernCustomer] = {}
    
    def create_customer(self, customer: ModernCustomer):
        """Create customer in modern system"""
        self.customers[customer.id] = customer
    
    def get_customer(self, customer_id: str) -> ModernCustomer:
        """Get customer from modern system"""
        return self.customers.get(customer_id)
    
    def update_customer(self, customer: ModernCustomer):
        """Update customer"""
        self.customers[customer.id] = customer


class AntiCorruptionLayer:
    """
    Anti-Corruption Layer that translates between legacy and modern systems
    """
    
    def __init__(self, legacy: LegacySystem, modern: ModernCRM):
        self.legacy = legacy
        self.modern = modern
        
        self.translations = 0
        self.reverse_translations = 0
    
    def get_customer_from_legacy(self, legacy_id: int) -> ModernCustomer:
        """Get customer from legacy and translate to modern model"""
        legacy_customer = self.legacy.get_customer(legacy_id)
        if not legacy_customer:
            return None
        
        # Translate from legacy to modern
        modern_customer = self._translate_to_modern(legacy_customer)
        self.translations += 1
        
        return modern_customer
    
    def sync_to_legacy(self, modern_customer: ModernCustomer):
        """Sync modern customer back to legacy system"""
        legacy_customer = self._translate_to_legacy(modern_customer)
        self.legacy.update_customer(legacy_customer)
        self.reverse_translations += 1
    
    def _translate_to_modern(self, legacy: LegacyCustomer) -> ModernCustomer:
        """Translate legacy model to modern model"""
        return ModernCustomer(
            id=f"CUST-{legacy.cust_id:06d}",
            full_name=f"{legacy.f_name} {legacy.l_name}",
            email=legacy.email_addr,
            phone=legacy.ph_num,
            address={
                'street': f"{legacy.addr_line_1} {legacy.addr_line_2}".strip(),
                'city': legacy.city,
                'state': legacy.state,
                'postal_code': legacy.zip_code
            },
            customer_type='individual' if legacy.cust_type == 'R' else 'business'
        )
    
    def _translate_to_legacy(self, modern: ModernCustomer) -> LegacyCustomer:
        """Translate modern model to legacy model"""
        legacy_id = int(modern.id.split('-')[1])
        name_parts = modern.full_name.split(' ', 1)
        
        # Parse address
        street = modern.address.get('street', '')
        addr_parts = street.split(' ', 2) if len(street.split()) > 2 else [street, '']
        
        return LegacyCustomer(
            cust_id=legacy_id,
            f_name=name_parts[0],
            l_name=name_parts[1] if len(name_parts) > 1 else '',
            addr_line_1=addr_parts[0] if addr_parts else '',
            addr_line_2=addr_parts[1] if len(addr_parts) > 1 else '',
            city=modern.address.get('city', ''),
            state=modern.address.get('state', ''),
            zip_code=modern.address.get('postal_code', ''),
            ph_num=modern.phone,
            email_addr=modern.email,
            cust_type='R' if modern.customer_type == 'individual' else 'W'
        )


def setup_systems_agent(state: ACLState):
    """Agent to set up systems and ACL"""
    operations = []
    results = []
    
    legacy = LegacySystem()
    modern = ModernCRM()
    acl = AntiCorruptionLayer(legacy, modern)
    
    operations.append("Anti-Corruption Layer Setup:")
    operations.append("\nLegacy System (Old Model):")
    operations.append("  Fields: cust_id, f_name, l_name, addr_line_1, ph_num, etc.")
    operations.append("  Customer types: 'R' (retail), 'W' (wholesale)")
    operations.append("  Database schema from 1990s")
    
    operations.append("\nModern CRM (Clean Model):")
    operations.append("  Fields: id, full_name, email, phone, address (dict)")
    operations.append("  Customer types: 'individual', 'business'")
    operations.append("  Modern RESTful API design")
    
    operations.append("\nACL: Translates between the two systems")
    
    results.append("✓ Systems and ACL initialized")
    
    state['_acl'] = acl
    state['_modern'] = modern
    
    return {
        "acl_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def translation_demo_agent(state: ACLState):
    """Agent to demonstrate translation"""
    acl = state['_acl']
    modern = state['_modern']
    operations = []
    results = []
    
    operations.append("\n🔄 Translation Demo (Legacy → Modern):")
    
    # Get customer from legacy through ACL
    operations.append("\nRetrieving customer ID 1 from legacy system:")
    modern_customer = acl.get_customer_from_legacy(1)
    
    operations.append("\nLegacy Format:")
    operations.append("  cust_id=1, f_name='John', l_name='Doe'")
    operations.append("  addr_line_1='123 Main St', city='Springfield'")
    operations.append("  cust_type='R'")
    
    operations.append("\nTranslated to Modern Format:")
    operations.append(f"  id: {modern_customer.id}")
    operations.append(f"  full_name: {modern_customer.full_name}")
    operations.append(f"  email: {modern_customer.email}")
    operations.append(f"  address: {modern_customer.address}")
    operations.append(f"  customer_type: {modern_customer.customer_type}")
    
    # Store in modern system
    modern.create_customer(modern_customer)
    
    results.append("✓ Successfully translated legacy data to modern model")
    
    return {
        "acl_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Translation complete"]
    }


def reverse_sync_agent(state: ACLState):
    """Agent to demonstrate reverse synchronization"""
    acl = state['_acl']
    modern = state['_modern']
    operations = []
    results = []
    
    operations.append("\n⬅️ Reverse Sync Demo (Modern → Legacy):")
    
    # Update modern customer
    customer = modern.get_customer("CUST-000001")
    customer.phone = "555-9999"
    customer.address['city'] = "New Springfield"
    
    operations.append("\nUpdating customer in modern system:")
    operations.append(f"  Changed phone to: {customer.phone}")
    operations.append(f"  Changed city to: {customer.address['city']}")
    
    # Sync back to legacy
    acl.sync_to_legacy(customer)
    
    operations.append("\n✓ Changes synchronized back to legacy system")
    operations.append("  ACL translated modern model back to legacy format")
    
    results.append("✓ Bi-directional sync working")
    
    return {
        "acl_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Reverse sync complete"]
    }


def statistics_agent(state: ACLState):
    """Agent to show statistics"""
    acl = state['_acl']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("ACL STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nTranslations (Legacy → Modern): {acl.translations}")
    operations.append(f"Reverse Translations (Modern → Legacy): {acl.reverse_translations}")
    
    metrics.append("\n📊 ACL Pattern Benefits:")
    metrics.append("  ✓ Isolates legacy complexity")
    metrics.append("  ✓ Clean modern domain model")
    metrics.append("  ✓ Bi-directional translation")
    metrics.append("  ✓ Enables gradual migration")
    metrics.append("  ✓ Prevents corruption of new design")
    
    results.append("✓ Anti-Corruption Layer working successfully")
    
    return {
        "acl_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_acl_graph():
    """Create the ACL workflow graph"""
    workflow = StateGraph(ACLState)
    
    workflow.add_node("setup", setup_systems_agent)
    workflow.add_node("translate", translation_demo_agent)
    workflow.add_node("reverse", reverse_sync_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "translate")
    workflow.add_edge("translate", "reverse")
    workflow.add_edge("reverse", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 215: Anti-Corruption Layer MCP Pattern")
    print("=" * 80)
    
    app = create_acl_graph()
    initial_state = {
        "acl_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    
    for op in final_state["acl_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Anti-Corruption Layer: Protects modern systems from legacy complexity

Pattern Structure:
1. Legacy System: Old data model, naming conventions
2. ACL: Translation layer
3. Modern System: Clean domain model

Benefits:
✓ Domain isolation
✓ Independent evolution
✓ Gradual migration
✓ Clean architecture
✓ Prevents technical debt spread

Real-World Use:
- Strangler Fig pattern migrations
- SAP/Oracle integration
- Mainframe modernization
- Multi-vendor systems
""")


if __name__ == "__main__":
    main()

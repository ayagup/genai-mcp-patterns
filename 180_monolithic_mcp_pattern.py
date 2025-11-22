"""
Pattern 180: Monolithic MCP Pattern

This pattern demonstrates monolithic architecture where the entire application
is built as a single, unified unit. All components run in a single process,
share the same database, and are deployed together as one application.

Key Concepts:
1. Single Application: All functionality in one codebase
2. Unified Deployment: Deploy entire application as single unit
3. Shared Database: One database for all components
4. In-Process Communication: Components call each other directly (no network)
5. Tightly Coupled: Components often have interdependencies
6. Single Technology Stack: Typically one programming language/framework
7. Shared Resources: Memory, CPU, threads shared across components

Monolithic Characteristics:
- All code in single repository
- Single deployment artifact (JAR, WAR, EXE)
- Components communicate via function calls
- Shared data model and schema
- Vertical scaling (scale entire app)
- Single runtime process
- Simple deployment model

Monolithic Patterns:
1. Layered Monolith: Layers within single application
2. Modular Monolith: Well-defined modules but deployed together
3. Big Ball of Mud: Unstructured monolith (anti-pattern)

Component Organization:
- Presentation Layer: UI components
- Business Logic: Core business rules
- Data Access: Database interaction
- Utilities: Shared helpers

Benefits:
- Simplicity: Easier to understand and develop initially
- Easy Debugging: Single process, use standard debugger
- Simple Deployment: One artifact to deploy
- Performance: No network calls between components
- Easy Testing: Test entire app in one go
- Transaction Management: Easy ACID transactions
- Development Velocity: Fast initial development

Trade-offs:
- Scalability: Must scale entire app, not individual components
- Technology Lock-In: Hard to use different technologies
- Deployment Risk: Small change requires full redeployment
- Team Coordination: Large teams step on each other
- Long-Term Maintenance: Can become complex over time
- Startup Time: Large app takes longer to start
- IDE Performance: Large codebase can slow IDE

When to Use Monolithic:
- New projects/startups (start simple)
- Small to medium applications
- Simple business domain
- Small team (< 10 developers)
- Need fast time to market
- Don't need independent scaling

Migration Path:
1. Start: Monolith (simple, fast)
2. Grow: Modular Monolith (organize better)
3. Scale: Extract microservices (if needed)

Use Cases:
- Startup MVPs: Get to market quickly
- Small business applications: Internal tools, CRM
- Traditional enterprise apps: ERP systems
- Simple e-commerce: Small online stores
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass

# Define the state for monolithic architecture
class MonolithicState(TypedDict):
    """State for monolithic application"""
    request: str
    ui_processed: Optional[Dict[str, Any]]
    business_result: Optional[Dict[str, Any]]
    data_result: Optional[Dict[str, Any]]
    final_response: str
    component_calls: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# SHARED DATABASE (Centralized Data)
# ============================================================================

class SharedDatabase:
    """
    Shared Database: Single database for entire application
    
    In monolithic architecture:
    - All components access same database
    - Shared schema and tables
    - Easy transactions across components
    - Strong consistency
    
    This is opposite of microservices where each service has own database.
    """
    
    def __init__(self):
        self.tables = {
            "users": {},
            "orders": {},
            "products": {},
            "payments": {}
        }
        self.transaction_active = False
    
    def begin_transaction(self):
        """Start transaction (ACID)"""
        self.transaction_active = True
        print("Transaction started")
    
    def commit(self):
        """Commit transaction"""
        self.transaction_active = False
        print("Transaction committed")
    
    def rollback(self):
        """Rollback transaction"""
        self.transaction_active = False
        print("Transaction rolled back")
    
    def insert(self, table: str, key: str, data: Dict[str, Any]):
        """Insert data into table"""
        if table in self.tables:
            self.tables[table][key] = data
    
    def select(self, table: str, key: str) -> Optional[Dict[str, Any]]:
        """Select data from table"""
        return self.tables.get(table, {}).get(key)
    
    def update(self, table: str, key: str, data: Dict[str, Any]):
        """Update data in table"""
        if table in self.tables and key in self.tables[table]:
            self.tables[table][key].update(data)
    
    def delete(self, table: str, key: str):
        """Delete data from table"""
        if table in self.tables and key in self.tables[table]:
            del self.tables[table][key]

# Create shared database (singleton)
shared_db = SharedDatabase()

# ============================================================================
# MONOLITHIC COMPONENTS (In-Process)
# ============================================================================

class UIComponent:
    """
    UI Component: Presentation layer
    
    In monolithic architecture:
    - Part of same application
    - Calls business layer directly (no HTTP)
    - Shares same process and memory
    - Can access shared utilities
    """
    
    def __init__(self, business_layer: 'BusinessLayer'):
        self.business_layer = business_layer
    
    def handle_request(self, user_input: str) -> Dict[str, Any]:
        """Handle user request"""
        # Validate input
        if not user_input or len(user_input) == 0:
            return {"error": "Invalid input"}
        
        # Direct function call to business layer (in-process)
        business_result = self.business_layer.process_order(user_input)
        
        # Format response for user
        return {
            "component": "UI",
            "user_input": user_input,
            "business_result": business_result,
            "formatted": f"Processed: {user_input}"
        }

class BusinessLayer:
    """
    Business Layer: Core business logic
    
    Contains:
    - Business rules
    - Workflow orchestration
    - Domain logic
    
    Direct access to data layer (no network calls)
    """
    
    def __init__(self, data_layer: 'DataLayer'):
        self.data_layer = data_layer
    
    def process_order(self, order_data: str) -> Dict[str, Any]:
        """Process an order"""
        # Business rule: validate order
        if "cancel" in order_data.lower():
            return {"status": "rejected", "reason": "Order cancelled"}
        
        # Create order ID
        order_id = f"ORD-{len(shared_db.tables['orders']) + 1}"
        
        # Direct function call to data layer (in-process)
        save_result = self.data_layer.save_order(order_id, {
            "order_id": order_id,
            "data": order_data,
            "status": "CONFIRMED"
        })
        
        # Business logic: calculate total, apply discounts, etc.
        total = 100.00  # Simplified
        
        return {
            "component": "Business",
            "order_id": order_id,
            "total": total,
            "status": "processed",
            "data_saved": save_result["saved"]
        }
    
    def calculate_discount(self, total: float, customer_type: str) -> float:
        """Business rule: calculate discount"""
        if customer_type == "premium":
            return total * 0.10
        elif customer_type == "regular":
            return total * 0.05
        return 0.0

class DataLayer:
    """
    Data Layer: Database access
    
    Accesses shared database
    All components use same database (centralized)
    Easy to maintain data consistency
    """
    
    def __init__(self, database: SharedDatabase):
        self.db = database
    
    def save_order(self, order_id: str, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save order to database"""
        self.db.insert("orders", order_id, order_data)
        
        return {
            "component": "Data",
            "saved": True,
            "order_id": order_id,
            "table": "orders"
        }
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order from database"""
        return self.db.select("orders", order_id)
    
    def save_payment(self, payment_id: str, payment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save payment"""
        self.db.insert("payments", payment_id, payment_data)
        return {"saved": True, "payment_id": payment_id}

class UtilityComponents:
    """
    Shared Utilities: Common functions used across application
    
    In monolith:
    - Shared code used by all components
    - In-process access (fast)
    - Easy to update (single codebase)
    """
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        return "@" in email and "." in email
    
    @staticmethod
    def format_currency(amount: float) -> str:
        """Format currency"""
        return f"${amount:.2f}"
    
    @staticmethod
    def generate_id(prefix: str) -> str:
        """Generate unique ID"""
        import time
        return f"{prefix}-{int(time.time() * 1000)}"

# ============================================================================
# MONOLITHIC APPLICATION
# ============================================================================

class MonolithicApplication:
    """
    Monolithic Application: All components in single application
    
    Characteristics:
    - Single process
    - Shared memory
    - Direct function calls
    - One database
    - Deployed as single unit
    """
    
    def __init__(self):
        # Initialize shared database
        self.database = shared_db
        
        # Initialize layers (bottom-up)
        self.data_layer = DataLayer(self.database)
        self.business_layer = BusinessLayer(self.data_layer)
        self.ui_component = UIComponent(self.business_layer)
        
        # Shared utilities
        self.utils = UtilityComponents()
        
        print("Monolithic Application initialized")
    
    def process_request(self, request: str) -> Dict[str, Any]:
        """
        Process request through all layers
        
        Flow (in single process):
        UI → Business → Data → Database
        
        All function calls, no network
        """
        # UI handles request
        ui_result = self.ui_component.handle_request(request)
        
        return {
            "application": "Monolithic",
            "deployment": "Single Unit",
            "communication": "In-Process Function Calls",
            "database": "Shared Database",
            "result": ui_result
        }
    
    def process_with_transaction(self, request: str) -> Dict[str, Any]:
        """
        Process with ACID transaction
        
        Benefit of monolith: Easy transactions across all components
        """
        try:
            # Begin transaction
            self.database.begin_transaction()
            
            # Process order
            result = self.process_request(request)
            
            # Commit transaction
            self.database.commit()
            
            return result
        
        except Exception as e:
            # Rollback on error
            self.database.rollback()
            return {"error": str(e), "transaction": "rolled back"}

# Create monolithic application instance
monolithic_app = MonolithicApplication()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def ui_component_agent(state: MonolithicState) -> MonolithicState:
    """UI Component in monolithic app"""
    request = state["request"]
    
    # In monolith, UI directly calls business layer
    ui_result = monolithic_app.ui_component.handle_request(request)
    
    return {
        "ui_processed": ui_result,
        "component_calls": ["UI Component → Business Layer (in-process call)"],
        "messages": ["[UI Component] Request processed"]
    }

def business_component_agent(state: MonolithicState) -> MonolithicState:
    """Business Component in monolithic app"""
    request = state["request"]
    
    # Business layer processes
    business_result = monolithic_app.business_layer.process_order(request)
    
    return {
        "business_result": business_result,
        "component_calls": ["Business Layer → Data Layer (in-process call)"],
        "messages": [f"[Business Layer] Order processed: {business_result.get('order_id', 'N/A')}"]
    }

def data_component_agent(state: MonolithicState) -> MonolithicState:
    """Data Component in monolithic app"""
    business_result = state.get("business_result", {})
    order_id = business_result.get("order_id", "N/A")
    
    # Data layer accesses shared database
    data_result = {
        "component": "Data",
        "database": "Shared Database",
        "order_saved": order_id,
        "tables_accessed": ["orders", "payments"]
    }
    
    return {
        "data_result": data_result,
        "component_calls": ["Data Layer → Shared Database (SQL queries)"],
        "messages": ["[Data Layer] Data persisted to shared database"]
    }

def monolithic_aggregator(state: MonolithicState) -> MonolithicState:
    """Aggregate monolithic components"""
    
    final_response = f"""Monolithic Application Results:
    
    Architecture: Single Process, Unified Application
    Database: Shared Database (all components access same DB)
    Communication: In-Process Function Calls (no network overhead)
    Deployment: Single Deployment Unit
    
    Components Executed:
    {chr(10).join(f"  - {call}" for call in state.get('component_calls', []))}
    
    Transaction: ACID guaranteed (single database)
    Performance: Fast (no network latency)
    
    All components executed in single process.
    """
    
    return {
        "final_response": final_response.strip(),
        "messages": ["[Monolithic App] All components completed"]
    }

# ============================================================================
# BUILD THE MONOLITHIC GRAPH
# ============================================================================

def create_monolithic_graph():
    """
    Create a StateGraph demonstrating monolithic architecture.
    
    Flow (all in single process):
    1. UI Component receives request
    2. Business Component processes logic
    3. Data Component persists to shared database
    4. Aggregator combines results
    
    All communication via direct function calls (no network).
    """
    
    workflow = StateGraph(MonolithicState)
    
    # Add component nodes
    workflow.add_node("ui", ui_component_agent)
    workflow.add_node("business", business_component_agent)
    workflow.add_node("data", data_component_agent)
    workflow.add_node("aggregator", monolithic_aggregator)
    
    # Define flow (sequential in-process calls)
    workflow.add_edge(START, "ui")
    workflow.add_edge("ui", "business")
    workflow.add_edge("business", "data")
    workflow.add_edge("data", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Monolithic MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Monolithic Application
    print("\n" + "=" * 80)
    print("Example 1: Monolithic Application Architecture")
    print("=" * 80)
    
    monolithic_graph = create_monolithic_graph()
    
    initial_state: MonolithicState = {
        "request": "Create order for customer_123 with product_XYZ",
        "ui_processed": None,
        "business_result": None,
        "data_result": None,
        "final_response": "",
        "component_calls": [],
        "messages": []
    }
    
    result = monolithic_graph.invoke(initial_state)
    
    print("\nComponent Execution:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nComponent Calls (In-Process):")
    for call in result["component_calls"]:
        print(f"  {call}")
    
    print("\nFinal Response:")
    print(result["final_response"])
    
    # Example 2: ACID Transaction
    print("\n" + "=" * 80)
    print("Example 2: ACID Transactions in Monolithic Architecture")
    print("=" * 80)
    
    print("\nExecuting transaction...")
    transaction_result = monolithic_app.process_with_transaction(
        "Create premium order for customer_456"
    )
    
    print(f"Transaction Result: {transaction_result.get('result', {}).get('application', 'N/A')}")
    print("\nBenefit: All operations in single transaction")
    print("  ✓ Atomic: All succeed or all fail")
    print("  ✓ Consistent: Database always in valid state")
    print("  ✓ Isolated: Concurrent transactions don't interfere")
    print("  ✓ Durable: Committed changes permanent")
    
    # Example 3: Shared Database
    print("\n" + "=" * 80)
    print("Example 3: Shared Database Access")
    print("=" * 80)
    
    print("\nShared Database Tables:")
    for table_name, data in shared_db.tables.items():
        print(f"  {table_name}: {len(data)} records")
    
    print("\nAll components access same database:")
    print("  ✓ Strong consistency")
    print("  ✓ Easy joins across tables")
    print("  ✓ ACID transactions")
    print("  ✓ No data synchronization needed")
    
    # Example 4: Monolithic vs Microservices
    print("\n" + "=" * 80)
    print("Example 4: Monolithic vs Microservices Comparison")
    print("=" * 80)
    
    print("\nMonolithic Architecture:")
    print("  Structure: Single unified application")
    print("  Deployment: One deployment unit")
    print("  Database: Shared database")
    print("  Communication: In-process function calls")
    print("  Scaling: Vertical (scale entire app)")
    print("  Technology: Single stack")
    print("  Transactions: Easy ACID")
    print("  Best for: Startups, small apps, simple domains")
    
    print("\nMicroservices Architecture:")
    print("  Structure: Multiple independent services")
    print("  Deployment: Independent deployments")
    print("  Database: Database per service")
    print("  Communication: HTTP/gRPC/messaging")
    print("  Scaling: Horizontal (scale individual services)")
    print("  Technology: Polyglot (different stacks)")
    print("  Transactions: Eventual consistency (Saga pattern)")
    print("  Best for: Large scale, complex domains, large teams")
    
    # Example 5: Evolution Path
    print("\n" + "=" * 80)
    print("Example 5: Evolution from Monolith to Microservices")
    print("=" * 80)
    
    print("\nPhase 1: Start with Monolith")
    print("  Why: Fast development, simple deployment, small team")
    print("  Duration: 6-24 months typically")
    
    print("\nPhase 2: Modular Monolith")
    print("  Why: Better organization, prepare for potential split")
    print("  Duration: As long as needed")
    
    print("\nPhase 3: Extract Microservices (if needed)")
    print("  Why: Need independent scaling, large team, complex domain")
    print("  Strategy: Extract gradually, start with boundaries")
    
    print("\nRecommendation:")
    print("  Start monolithic, evolve to microservices ONLY if truly needed")
    print("  Many successful companies run on well-structured monoliths!")
    
    print("\n" + "=" * 80)
    print("🎉 Composition Patterns (171-180) - ALL COMPLETE! 🎉")
    print("=" * 80)
    print(f"""
Progress: 180/400 patterns (45% complete!)

Composition Patterns Summary:
  171. Agent Composition - Combine agents into unified system
  172. Service Composition - SOA with service registry
  173. Vertical Composition - Layered stack (presentation→business→data)
  174. Horizontal Composition - Peer components side-by-side
  175. Nested Composition - Recursive tree structures
  176. Layered Composition - Clear layer boundaries and dependencies
  177. Modular Composition - Independent pluggable modules
  178. Plugin Architecture - Dynamic extension points
  179. Microservices - Independent deployable services
  180. Monolithic - Unified single-process application

Key Insights:
  - Composition is about HOW to combine components
  - Vertical: layers stacked (UI → Business → Data)
  - Horizontal: peers side-by-side (Service A | B | C)
  - Nested: hierarchical trees (Parent → Child → Grandchild)
  - Modular: pluggable independent modules
  - Microservices: distributed independent services
  - Monolithic: unified single application
  - Choose based on: team size, scale needs, complexity, maturity

Next Category: Data Flow Patterns (181-190)
    """)

"""
Pattern 176: Layered Composition MCP Pattern

This pattern demonstrates layered architecture where the system is decomposed into
horizontal layers, each providing services to the layer above and using services
from the layer below. This creates clear separation of concerns with well-defined
layer boundaries and dependencies.

Key Concepts:
1. Layer: Horizontal slice of the system with specific responsibility
2. Layer Boundary: Clear interface between adjacent layers
3. Dependency Rule: Layer N can only depend on Layer N-1 (below)
4. Abstraction Levels: Higher layers more abstract, lower more concrete
5. Closed Layers: Request must pass through all layers (strict)
6. Open Layers: Can skip layers (relaxed)
7. Layer Services: Each layer exposes services to layer above

Common Layer Architectures:
1. Presentation Layer: UI, user interaction, display logic
2. Application Layer: Use cases, application workflows, coordination
3. Domain/Business Layer: Business rules, domain model, core logic
4. Infrastructure Layer: Database, external services, frameworks

Layered Patterns:
- Three-Tier: Presentation + Business + Data
- Four-Tier: Presentation + Application + Domain + Infrastructure
- Clean Architecture: Entities + Use Cases + Interface Adapters + Frameworks
- Hexagonal: Core domain + Ports + Adapters
- Onion: Domain Model (center) + Domain Services + Application + Infrastructure

Benefits:
- Separation of Concerns: Each layer has single responsibility
- Testability: Test layers independently with mocks
- Maintainability: Changes isolated to specific layers
- Reusability: Layers can be reused in different applications
- Team Organization: Different teams can own different layers
- Technology Independence: Swap layer implementations

Trade-offs:
- Performance: Layer transitions add overhead
- Complexity: More structure to understand
- Rigidity: Strict layering can be inflexible
- Cascade Changes: Interface changes affect multiple layers
- Over-Engineering: Simple apps may not need all layers

Use Cases:
- Enterprise applications: Web apps, business systems
- Data processing: ETL pipelines with processing layers
- API design: Controller + Service + Repository layers
- Mobile apps: UI + ViewModel + Model + Data layers
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from abc import ABC, abstractmethod
from enum import Enum

# Define the state for layered composition
class LayeredCompositionState(TypedDict):
    """State flowing through layered architecture"""
    user_request: str
    presentation_output: Optional[str]
    application_result: Optional[Dict[str, Any]]
    domain_result: Optional[Dict[str, Any]]
    infrastructure_result: Optional[Dict[str, Any]]
    final_response: str
    layer_calls: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# LAYER 1: PRESENTATION LAYER (Outermost)
# ============================================================================

def presentation_layer(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Presentation Layer: User Interface and Interaction
    
    Responsibilities:
    - Handle user input
    - Display output
    - Input validation
    - Format data for display
    - User session management
    
    Dependencies:
    - Depends ONLY on Application Layer (below)
    - Does NOT know about Domain or Infrastructure layers
    
    Technology Examples:
    - Web: React, Vue, Angular
    - Mobile: iOS/Android UI
    - CLI: Command-line interface
    - API: REST/GraphQL endpoints
    """
    user_request = state["user_request"]
    
    # Presentation layer formats the request for the application layer
    presentation_output = f"Validated request: {user_request}"
    
    return {
        "presentation_output": presentation_output,
        "layer_calls": ["Layer 1 (Presentation) → Layer 2 (Application)"],
        "messages": ["[Presentation Layer] User request received and validated"]
    }

def presentation_formatter(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Format response from lower layers for presentation
    """
    app_result = state.get("application_result", {})
    
    formatted = f"User-friendly response: {app_result.get('status', 'N/A')}"
    
    return {
        "final_response": formatted,
        "layer_calls": ["Layer 2 (Application) → Layer 1 (Presentation)"],
        "messages": ["[Presentation Layer] Response formatted for user"]
    }

# ============================================================================
# LAYER 2: APPLICATION LAYER
# ============================================================================

def application_layer(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Application Layer: Use Cases and Application Logic
    
    Responsibilities:
    - Implement use cases
    - Orchestrate workflows
    - Coordinate domain objects
    - Application-specific rules
    - Transaction management
    
    Dependencies:
    - Depends on Domain Layer (below)
    - Used by Presentation Layer (above)
    - Does NOT know about Infrastructure details
    
    Examples:
    - CreateOrderUseCase
    - ProcessPaymentWorkflow
    - GenerateReportService
    """
    presentation_output = state.get("presentation_output", "")
    
    prompt = f"""You are the application layer implementing a use case.
    Orchestrate the business logic for this request:
    
    Request: {presentation_output}
    
    Coordinate domain operations needed."""
    
    response = llm.invoke(prompt)
    
    application_result = {
        "use_case": "process_request",
        "status": "coordinated",
        "domain_operations": ["validate", "process", "persist"],
        "workflow": response.content[:150]
    }
    
    return {
        "application_result": application_result,
        "layer_calls": ["Layer 2 (Application) → Layer 3 (Domain)"],
        "messages": ["[Application Layer] Use case orchestrated"]
    }

# ============================================================================
# LAYER 3: DOMAIN LAYER (Core Business Logic)
# ============================================================================

def domain_layer(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Domain Layer: Business Rules and Domain Model
    
    Responsibilities:
    - Business rules
    - Domain entities and value objects
    - Domain services
    - Business invariants
    - Core business logic
    
    Dependencies:
    - Self-contained (minimal dependencies)
    - Does NOT depend on outer layers
    - Infrastructure injected via interfaces (Dependency Inversion)
    
    Examples:
    - Order (entity)
    - Money (value object)
    - PricingService (domain service)
    - OrderMustHaveItems (business rule)
    
    This is the HEART of the application.
    """
    app_result = state.get("application_result", {})
    
    prompt = f"""You are the domain layer with core business logic.
    Apply business rules for this operation:
    
    Operations: {app_result.get('domain_operations', [])}
    
    Enforce business constraints and rules."""
    
    response = llm.invoke(prompt)
    
    domain_result = {
        "rules_applied": ["business_constraint_1", "validation_rule_2"],
        "domain_objects": ["Order", "Customer", "Payment"],
        "business_logic_result": response.content[:150],
        "invariants_satisfied": True
    }
    
    return {
        "domain_result": domain_result,
        "layer_calls": ["Layer 3 (Domain) → Layer 4 (Infrastructure)"],
        "messages": ["[Domain Layer] Business rules applied"]
    }

# ============================================================================
# LAYER 4: INFRASTRUCTURE LAYER (Innermost)
# ============================================================================

def infrastructure_layer(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Infrastructure Layer: External Concerns and Frameworks
    
    Responsibilities:
    - Database access
    - File system
    - External APIs
    - Framework integration
    - Messaging systems
    - Caching
    
    Dependencies:
    - Implements interfaces defined by Domain layer
    - Provides implementations for Application layer
    - Outermost layer in Clean Architecture thinking
    
    Examples:
    - PostgresOrderRepository
    - RedisCache
    - S3FileStorage
    - StripePaymentGateway
    
    Technology can be swapped without affecting business logic.
    """
    domain_result = state.get("domain_result", {})
    
    infrastructure_result = {
        "database": "PostgreSQL",
        "cache": "Redis",
        "storage": "S3",
        "records_persisted": 1,
        "transaction_id": "txn_123",
        "infrastructure_operations": ["db_insert", "cache_update", "s3_upload"]
    }
    
    return {
        "infrastructure_result": infrastructure_result,
        "layer_calls": ["Layer 4 (Infrastructure) executed"],
        "messages": ["[Infrastructure Layer] Data persisted"]
    }

# ============================================================================
# LAYER INTERFACES (Dependency Inversion)
# ============================================================================

class LayerInterface(ABC):
    """
    Abstract interface between layers
    
    Benefits of layer interfaces:
    - Decoupling: layers depend on abstractions, not concrete implementations
    - Testability: easy to mock interfaces
    - Flexibility: swap implementations
    """
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """Execute layer's responsibility"""
        pass

class IPresentationLayer(LayerInterface):
    """Interface for presentation layer"""
    @abstractmethod
    def render(self, data: Any) -> str:
        """Render data for display"""
        pass

class IApplicationLayer(LayerInterface):
    """Interface for application layer"""
    @abstractmethod
    def execute_use_case(self, use_case: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an application use case"""
        pass

class IDomainService(ABC):
    """Interface for domain services"""
    @abstractmethod
    def apply_business_rule(self, entity: Any) -> bool:
        """Apply business rule to entity"""
        pass

class IRepository(ABC):
    """
    Repository interface (Domain layer defines, Infrastructure implements)
    
    This is Dependency Inversion:
    - Domain defines what it needs (interface)
    - Infrastructure provides implementation
    - Domain doesn't depend on Infrastructure
    """
    @abstractmethod
    def save(self, entity: Any) -> str:
        """Save entity"""
        pass
    
    @abstractmethod
    def find_by_id(self, entity_id: str) -> Optional[Any]:
        """Find entity by ID"""
        pass

# ============================================================================
# CLEAN ARCHITECTURE (Dependency Inversion)
# ============================================================================

def clean_architecture_coordinator(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Clean Architecture Coordinator
    
    Dependency Rule (Clean Architecture):
    - Dependencies point INWARD toward domain
    - Inner layers don't know about outer layers
    - Outer layers depend on inner layers
    
    Layers (inside to outside):
    1. Entities (most inner) - business objects
    2. Use Cases - application-specific business rules
    3. Interface Adapters - convert data formats
    4. Frameworks & Drivers (most outer) - UI, DB, Web
    
    This is opposite of traditional layering!
    """
    
    return {
        "layer_calls": ["Clean Architecture: Dependencies point inward"],
        "messages": ["[Clean Architecture] Dependency inversion applied"]
    }

# ============================================================================
# LAYER BYPASS (Open Layers)
# ============================================================================

def layer_bypass_example(state: LayeredCompositionState) -> LayeredCompositionState:
    """
    Open Layers: Allow bypassing intermediate layers
    
    Closed Layers (Strict):
    - Every request must pass through all layers in order
    - Presentation → Application → Domain → Infrastructure
    - More rigid but enforces separation
    
    Open Layers (Relaxed):
    - Can skip layers when appropriate
    - Presentation → Domain (skip Application)
    - More flexible but can lead to coupling
    
    Trade-off: strictness vs flexibility
    """
    
    return {
        "layer_calls": ["Layer bypass: Presentation → Domain (skipped Application)"],
        "messages": ["[Layer Bypass] Skipped intermediate layer for simple operation"]
    }

# ============================================================================
# BUILD THE LAYERED COMPOSITION GRAPH
# ============================================================================

def create_layered_composition_graph():
    """
    Create a StateGraph demonstrating layered architecture.
    
    Four-Tier Architecture:
    Layer 1: Presentation (UI, display)
    Layer 2: Application (use cases, workflows)
    Layer 3: Domain (business rules, entities)
    Layer 4: Infrastructure (database, external services)
    
    Flow (Request flows down, Response flows up):
    1. User → Presentation Layer
    2. Presentation → Application Layer
    3. Application → Domain Layer
    4. Domain → Infrastructure Layer
    5. Infrastructure → Domain (results)
    6. Domain → Application (results)
    7. Application → Presentation (results)
    8. Presentation → User (formatted response)
    """
    
    workflow = StateGraph(LayeredCompositionState)
    
    # Add layer nodes (in order from outer to inner)
    workflow.add_node("presentation_input", presentation_layer)
    workflow.add_node("application", application_layer)
    workflow.add_node("domain", domain_layer)
    workflow.add_node("infrastructure", infrastructure_layer)
    workflow.add_node("presentation_output", presentation_formatter)
    
    # Flow down through layers
    workflow.add_edge(START, "presentation_input")
    workflow.add_edge("presentation_input", "application")
    workflow.add_edge("application", "domain")
    workflow.add_edge("domain", "infrastructure")
    
    # Flow back up (in real app, this happens implicitly)
    workflow.add_edge("infrastructure", "presentation_output")
    workflow.add_edge("presentation_output", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Layered Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Four-Tier Layered Architecture
    print("\n" + "=" * 80)
    print("Example 1: Four-Tier Layered Architecture")
    print("=" * 80)
    
    layered_graph = create_layered_composition_graph()
    
    initial_state: LayeredCompositionState = {
        "user_request": "Create new customer order for product XYZ, quantity 5",
        "presentation_output": None,
        "application_result": None,
        "domain_result": None,
        "infrastructure_result": None,
        "final_response": "",
        "layer_calls": [],
        "messages": []
    }
    
    result = layered_graph.invoke(initial_state)
    
    print("\nLayer Execution Flow:")
    for call in result["layer_calls"]:
        print(f"  {call}")
    
    print("\nLayer Processing:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nLayer Results:")
    print(f"  Application: {result.get('application_result', {}).get('use_case', 'N/A')}")
    print(f"  Domain: {len(result.get('domain_result', {}).get('rules_applied', []))} rules applied")
    print(f"  Infrastructure: {result.get('infrastructure_result', {}).get('database', 'N/A')}")
    
    print("\nFinal Response:")
    print(f"  {result['final_response']}")
    
    # Example 2: Layer Responsibilities
    print("\n" + "=" * 80)
    print("Example 2: Layer Responsibilities and Dependencies")
    print("=" * 80)
    
    layers = {
        "Presentation": {
            "responsibilities": ["UI rendering", "Input validation", "Display formatting"],
            "depends_on": ["Application"],
            "examples": ["React UI", "REST API", "CLI"]
        },
        "Application": {
            "responsibilities": ["Use cases", "Workflows", "Coordination"],
            "depends_on": ["Domain"],
            "examples": ["CreateOrderUseCase", "ProcessPayment", "GenerateReport"]
        },
        "Domain": {
            "responsibilities": ["Business rules", "Entities", "Domain logic"],
            "depends_on": ["None (core)"],
            "examples": ["Order", "Customer", "PricingService"]
        },
        "Infrastructure": {
            "responsibilities": ["Database", "External APIs", "Frameworks"],
            "depends_on": ["Implements Domain interfaces"],
            "examples": ["PostgresRepository", "S3Storage", "StripeGateway"]
        }
    }
    
    for layer_name, details in layers.items():
        print(f"\n{layer_name} Layer:")
        print(f"  Responsibilities: {', '.join(details['responsibilities'])}")
        print(f"  Depends On: {', '.join(details['depends_on'])}")
        print(f"  Examples: {', '.join(details['examples'])}")
    
    # Example 3: Dependency Direction
    print("\n" + "=" * 80)
    print("Example 3: Dependency Direction (Traditional vs Clean Architecture)")
    print("=" * 80)
    
    print("\nTraditional Layered Architecture:")
    print("  Presentation → Application → Domain → Infrastructure")
    print("  Dependencies flow downward through layers")
    print("  Lower layers don't know about upper layers")
    print("  Issue: Domain depends on Infrastructure (database details)")
    
    print("\nClean Architecture (Dependency Inversion):")
    print("  Infrastructure → Interface Adapters → Use Cases → Entities")
    print("  Dependencies point INWARD toward domain")
    print("  Domain defines interfaces, Infrastructure implements")
    print("  Benefit: Domain independent of infrastructure details")
    
    print("\nDependency Inversion Example:")
    print("  Domain defines: IOrderRepository (interface)")
    print("  Infrastructure implements: PostgresOrderRepository")
    print("  Domain doesn't know about PostgreSQL, only the interface")
    
    # Example 4: Closed vs Open Layers
    print("\n" + "=" * 80)
    print("Example 4: Closed vs Open Layers")
    print("=" * 80)
    
    print("\nClosed Layers (Strict Layering):")
    print("  ✓ Request must pass through ALL layers in order")
    print("  ✓ Presentation → Application → Domain → Infrastructure")
    print("  ✓ Cannot skip layers")
    print("  ✓ Benefits: Enforces separation, prevents coupling")
    print("  ✗ Drawback: Less flexible, potential performance overhead")
    
    print("\nOpen Layers (Relaxed Layering):")
    print("  ✓ Can skip intermediate layers")
    print("  ✓ Presentation → Domain (bypass Application)")
    print("  ✓ Benefits: More flexible, better performance")
    print("  ✗ Drawback: Risk of layer coupling, harder to maintain")
    
    print("\nRecommendation:")
    print("  Default: Closed (strict) for most cases")
    print("  Exception: Open for performance-critical simple operations")
    
    # Example 5: Technology Independence
    print("\n" + "=" * 80)
    print("Example 5: Technology Independence Through Layering")
    print("=" * 80)
    
    print("\nInfrastructure Layer (can swap technologies):")
    print("  Database:")
    print("    - Current: PostgreSQL")
    print("    - Alternative: MongoDB, MySQL, DynamoDB")
    print("    - Impact: Change ONLY Infrastructure layer")
    
    print("\n  Cache:")
    print("    - Current: Redis")
    print("    - Alternative: Memcached, In-Memory")
    print("    - Impact: Change ONLY Infrastructure layer")
    
    print("\n  Storage:")
    print("    - Current: AWS S3")
    print("    - Alternative: Azure Blob, Local Filesystem")
    print("    - Impact: Change ONLY Infrastructure layer")
    
    print("\nDomain and Application layers remain unchanged!")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Layered Architecture decomposes system into horizontal layers
2. Four-Tier: Presentation + Application + Domain + Infrastructure
3. Each layer has specific responsibility and dependencies
4. Dependency Rule: Layer N depends ONLY on Layer N-1 (below)
5. Presentation: UI, display, user interaction
6. Application: Use cases, workflows, coordination
7. Domain: Business rules, entities, core logic (HEART of system)
8. Infrastructure: Database, external services, frameworks
9. Closed Layers: Must pass through all layers (strict)
10. Open Layers: Can skip layers (relaxed)
11. Clean Architecture: Dependencies point INWARD (Dependency Inversion)
12. Repository Pattern: Domain defines interface, Infrastructure implements
13. Benefits: separation of concerns, testability, maintainability, technology independence
14. Trade-offs: performance overhead, complexity, potential rigidity
15. Use cases: enterprise apps, web services, mobile apps, APIs
    """)

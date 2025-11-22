"""
Pattern 173: Vertical Composition MCP Pattern

This pattern demonstrates vertical composition where components are organized in
a layered stack, with each layer building upon the layer below it. Data flows
vertically through the stack (top-down for requests, bottom-up for responses).

Key Concepts:
1. Layered Architecture: Components organized in vertical layers
2. Presentation Layer: User interface and interaction (top)
3. Business Logic Layer: Core business rules and processing (middle)
4. Data Access Layer: Database and storage operations (bottom)
5. Dependency Direction: Each layer depends only on layer directly below
6. Abstraction Levels: Higher layers more abstract, lower layers more concrete
7. Vertical Flow: Requests flow down, responses flow up

Vertical Composition Patterns:
1. Three-Tier: Presentation + Business + Data
2. N-Tier: Multiple intermediate layers
3. Hexagonal: Business logic at center, adapters around edges
4. Onion: Domain model at core, dependencies point inward
5. Clean Architecture: Entities -> Use Cases -> Interface Adapters -> Frameworks

Layer Responsibilities:
- Presentation: UI rendering, input validation, user experience
- Application: Use cases, workflow coordination, authorization
- Domain/Business: Business rules, entities, domain logic
- Infrastructure: Persistence, external services, frameworks

Benefits:
- Separation of Concerns: Each layer has clear responsibility
- Testability: Test layers independently with mocks
- Maintainability: Changes in one layer don't affect others
- Scalability: Scale layers independently
- Portability: Swap out layers (e.g., change database)

Trade-offs:
- Performance: Multiple layer transitions add overhead
- Complexity: More structure to navigate
- Over-Engineering: Simple apps may not need all layers
- Rigidity: Strict layering can be inflexible

Use Cases:
- Web applications: UI -> API -> Service -> Repository -> Database
- Enterprise systems: Portal -> Workflow -> Business Rules -> Data Access
- AI pipelines: Interface -> Orchestration -> Model -> Data
- Mobile apps: View -> ViewModel -> Model -> Storage
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from abc import ABC, abstractmethod

# Define the state for vertical composition
class VerticalCompositionState(TypedDict):
    """State that flows vertically through all layers"""
    user_request: str
    presentation_data: Optional[Dict[str, Any]]
    business_result: Optional[Dict[str, Any]]
    data_result: Optional[Dict[str, Any]]
    response: str
    layer_trace: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# LAYER 1: PRESENTATION LAYER (Top)
# ============================================================================

def presentation_layer(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Presentation Layer: Handles user interface and input/output formatting
    
    Responsibilities:
    - Receive user requests
    - Validate input format
    - Format data for display
    - Handle UI concerns
    - Delegate to business layer
    
    Does NOT:
    - Contain business logic
    - Access database directly
    - Make business decisions
    """
    user_request = state["user_request"]
    
    prompt = f"""You are the presentation layer of a vertical architecture.
    Process this user request and prepare it for the business layer:
    
    User Request: {user_request}
    
    Extract and structure:
    1. User intent
    2. Required parameters
    3. Display preferences
    4. Validation status"""
    
    response = llm.invoke(prompt)
    
    presentation_data = {
        "layer": "presentation",
        "user_intent": "process_request",
        "parameters": {"request": user_request},
        "validated": True,
        "formatted_for_business_layer": response.content[:200]
    }
    
    return {
        "presentation_data": presentation_data,
        "layer_trace": ["Presentation Layer → Business Layer"],
        "messages": ["[Presentation Layer] Request validated and formatted"]
    }

def presentation_formatter(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Format response from lower layers for user presentation
    """
    business_result = state.get("business_result", {})
    data_result = state.get("data_result", {})
    
    # Format results for user display
    formatted_response = f"""
    === User Response ===
    
    Request Processing Complete
    
    Business Layer Result: {business_result.get('layer', 'N/A')}
    Data Layer Result: {data_result.get('layer', 'N/A')}
    
    Status: Success
    """
    
    return {
        "response": formatted_response.strip(),
        "layer_trace": ["Business Layer → Presentation Layer"],
        "messages": ["[Presentation Layer] Formatted response for user"]
    }

# ============================================================================
# LAYER 2: BUSINESS LOGIC LAYER (Middle)
# ============================================================================

def business_logic_layer(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Business Logic Layer: Core business rules and processing
    
    Responsibilities:
    - Implement business rules
    - Coordinate workflows
    - Apply domain logic
    - Validate business constraints
    - Delegate to data layer for persistence
    
    Does NOT:
    - Handle UI formatting
    - Know about database schema
    - Make infrastructure decisions
    """
    presentation_data = state.get("presentation_data", {})
    user_request = presentation_data.get("parameters", {}).get("request", "")
    
    prompt = f"""You are the business logic layer. Apply business rules and 
    processing to this request:
    
    Request: {user_request}
    
    Process according to business rules:
    1. Validate business constraints
    2. Apply business logic
    3. Determine data operations needed
    4. Calculate business metrics"""
    
    response = llm.invoke(prompt)
    
    business_result = {
        "layer": "business_logic",
        "rules_applied": ["validation", "processing", "calculation"],
        "business_decision": "approved",
        "data_requirements": ["fetch_user_data", "update_records"],
        "metrics": {"confidence": 0.95, "priority": "high"},
        "processing_result": response.content[:200]
    }
    
    return {
        "business_result": business_result,
        "layer_trace": ["Business Layer → Data Layer"],
        "messages": ["[Business Logic Layer] Business rules applied"]
    }

def business_orchestrator(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Orchestrate complex business workflows across multiple operations
    """
    return {
        "layer_trace": ["Business Layer Orchestration"],
        "messages": ["[Business Logic Layer] Workflow orchestrated"]
    }

# ============================================================================
# LAYER 3: DATA ACCESS LAYER (Bottom)
# ============================================================================

def data_access_layer(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Data Access Layer: Database and storage operations
    
    Responsibilities:
    - Execute database queries
    - Manage connections
    - Handle transactions
    - Map between objects and data
    - Cache data
    
    Does NOT:
    - Contain business logic
    - Know about UI
    - Make business decisions
    """
    business_result = state.get("business_result", {})
    data_requirements = business_result.get("data_requirements", [])
    
    # Simulate data operations
    data_result = {
        "layer": "data_access",
        "operations": data_requirements,
        "records_affected": 5,
        "query_time_ms": 45,
        "cache_hit": False,
        "transaction_id": "txn_12345",
        "data_retrieved": {
            "users": ["user1", "user2", "user3"],
            "metadata": {"total": 3, "timestamp": "2025-11-15"}
        }
    }
    
    return {
        "data_result": data_result,
        "layer_trace": ["Data Layer → Business Layer"],
        "messages": ["[Data Access Layer] Data operations completed"]
    }

def repository_pattern(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Repository Pattern: Abstract data access behind collections
    
    Benefits:
    - Business layer works with collections, not database
    - Easy to swap data sources
    - Centralized query logic
    - Testable with in-memory repositories
    """
    return {
        "layer_trace": ["Repository Pattern Applied"],
        "messages": ["[Data Access Layer] Repository pattern used"]
    }

# ============================================================================
# LAYER ABSTRACTIONS (Clean Architecture)
# ============================================================================

class LayerInterface(ABC):
    """
    Abstract interface that defines contract between layers
    
    This enables:
    - Dependency inversion (layers depend on abstractions, not concrete)
    - Easy testing (mock implementations)
    - Flexibility (swap implementations)
    """
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data according to layer's responsibility"""
        pass
    
    @abstractmethod
    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input meets layer's requirements"""
        pass

class PresentationLayerImpl(LayerInterface):
    """Concrete implementation of presentation layer"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format and validate user input"""
        return {
            "layer": "presentation",
            "validated": self.validate(input_data),
            "formatted": input_data
        }
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input format"""
        return "user_request" in input_data

class BusinessLayerImpl(LayerInterface):
    """Concrete implementation of business layer"""
    
    def __init__(self, data_layer: 'DataLayerImpl'):
        """Business layer depends on data layer interface"""
        self.data_layer = data_layer
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply business rules and delegate to data layer"""
        if not self.validate(input_data):
            return {"error": "Invalid business input"}
        
        # Apply business logic
        business_result = {"rules_applied": True, "approved": True}
        
        # Delegate to data layer
        data_result = self.data_layer.process(input_data)
        
        return {**business_result, "data": data_result}
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate business constraints"""
        return input_data.get("validated", False)

class DataLayerImpl(LayerInterface):
    """Concrete implementation of data layer"""
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data operations"""
        return {
            "layer": "data",
            "records_affected": 1,
            "success": True
        }
    
    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate data requirements"""
        return True

# ============================================================================
# VERTICAL FLOW COORDINATOR
# ============================================================================

def vertical_flow_coordinator(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Coordinate vertical flow through all layers
    
    Flow:
    1. Request enters at presentation layer (top)
    2. Flows down through business layer
    3. Reaches data layer (bottom)
    4. Response flows back up through all layers
    5. Final response returned to user
    """
    return {
        "messages": ["[Vertical Flow Coordinator] Managing vertical composition"]
    }

# ============================================================================
# N-TIER EXAMPLE (Extended Vertical Composition)
# ============================================================================

def application_layer(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Application Layer: Use cases and application workflows
    
    In N-tier, this sits between presentation and business:
    - Presentation Layer (UI)
    - Application Layer (Use Cases) <- this
    - Business Layer (Domain Logic)
    - Data Layer (Persistence)
    """
    return {
        "layer_trace": ["Application Layer (Use Cases)"],
        "messages": ["[Application Layer] Use case executed"]
    }

def infrastructure_layer(state: VerticalCompositionState) -> VerticalCompositionState:
    """
    Infrastructure Layer: External services, frameworks, tools
    
    In Clean Architecture, this is the outermost layer:
    - Infrastructure (Frameworks, Drivers) <- this
    - Interface Adapters (Controllers, Gateways)
    - Use Cases (Application Business Rules)
    - Entities (Enterprise Business Rules)
    """
    return {
        "layer_trace": ["Infrastructure Layer (External Services)"],
        "messages": ["[Infrastructure Layer] External services integrated"]
    }

# ============================================================================
# BUILD THE VERTICAL COMPOSITION GRAPH
# ============================================================================

def create_vertical_composition_graph():
    """
    Create a StateGraph demonstrating vertical composition.
    
    Flow (Top-down, then bottom-up):
    1. Coordinator initializes
    2. Presentation layer receives request (top)
    3. Business logic layer processes (middle)
    4. Data access layer persists (bottom)
    5. Business layer receives data results
    6. Presentation layer formats response
    """
    
    workflow = StateGraph(VerticalCompositionState)
    
    # Add layer nodes
    workflow.add_node("coordinator", vertical_flow_coordinator)
    workflow.add_node("presentation_input", presentation_layer)
    workflow.add_node("business_logic", business_logic_layer)
    workflow.add_node("data_access", data_access_layer)
    workflow.add_node("presentation_output", presentation_formatter)
    
    # Define vertical flow (down then up)
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "presentation_input")
    workflow.add_edge("presentation_input", "business_logic")  # Down
    workflow.add_edge("business_logic", "data_access")         # Down
    workflow.add_edge("data_access", "presentation_output")    # Up (skip business for this example)
    workflow.add_edge("presentation_output", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Vertical Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Three-Tier Vertical Composition
    print("\n" + "=" * 80)
    print("Example 1: Three-Tier Vertical Composition (Presentation-Business-Data)")
    print("=" * 80)
    
    vertical_graph = create_vertical_composition_graph()
    
    initial_state: VerticalCompositionState = {
        "user_request": "Process customer order for product XYZ with quantity 5",
        "presentation_data": None,
        "business_result": None,
        "data_result": None,
        "response": "",
        "layer_trace": [],
        "messages": []
    }
    
    result = vertical_graph.invoke(initial_state)
    
    print("\nVertical Flow Trace (Request flows down, Response flows up):")
    for trace in result["layer_trace"]:
        print(f"  {trace}")
    
    print("\nLayer Processing:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nLayer Results:")
    print(f"  Presentation Data: {result.get('presentation_data', {}).get('layer', 'N/A')}")
    print(f"  Business Result: {result.get('business_result', {}).get('layer', 'N/A')}")
    print(f"  Data Result: {result.get('data_result', {}).get('layer', 'N/A')}")
    
    print("\nFinal Response:")
    print(result["response"])
    
    # Example 2: Layer Abstractions (Clean Architecture)
    print("\n" + "=" * 80)
    print("Example 2: Layer Abstractions with Dependency Inversion")
    print("=" * 80)
    
    # Create layers with dependency injection
    data_layer = DataLayerImpl()
    business_layer = BusinessLayerImpl(data_layer)  # Inject data layer
    presentation_layer_impl = PresentationLayerImpl()
    
    # Process request through layers
    user_input = {"user_request": "Sample request"}
    
    print("\nProcessing through layers:")
    
    # Layer 1: Presentation
    presentation_result = presentation_layer_impl.process(user_input)
    print(f"  1. Presentation Layer: {presentation_result}")
    
    # Layer 2: Business (depends on data layer)
    business_result = business_layer.process(presentation_result)
    print(f"  2. Business Layer: {business_result}")
    
    # Example 3: Layer Responsibilities
    print("\n" + "=" * 80)
    print("Example 3: Layer Responsibilities in Vertical Composition")
    print("=" * 80)
    
    print("\nPresentation Layer Responsibilities:")
    print("  ✓ Input validation and formatting")
    print("  ✓ Output rendering and display")
    print("  ✓ User interaction handling")
    print("  ✓ UI state management")
    print("  ✗ Business logic")
    print("  ✗ Database access")
    
    print("\nBusiness Logic Layer Responsibilities:")
    print("  ✓ Business rules and constraints")
    print("  ✓ Domain logic and calculations")
    print("  ✓ Workflow orchestration")
    print("  ✓ Authorization decisions")
    print("  ✗ UI formatting")
    print("  ✗ SQL queries")
    
    print("\nData Access Layer Responsibilities:")
    print("  ✓ Database queries and commands")
    print("  ✓ Transaction management")
    print("  ✓ Data mapping (ORM)")
    print("  ✓ Caching strategies")
    print("  ✗ Business rules")
    print("  ✗ User interface")
    
    # Example 4: Vertical vs Horizontal Comparison
    print("\n" + "=" * 80)
    print("Example 4: Vertical vs Horizontal Composition")
    print("=" * 80)
    
    print("\nVertical Composition (Layers):")
    print("  Structure: Components stacked in layers")
    print("  Flow: Top-down (request) and bottom-up (response)")
    print("  Dependencies: Each layer depends on layer below")
    print("  Example: Presentation → Business → Data")
    print("  Use case: Traditional enterprise applications")
    
    print("\nHorizontal Composition (Peers):")
    print("  Structure: Components side-by-side at same level")
    print("  Flow: Left-to-right or concurrent")
    print("  Dependencies: Minimal, mostly independent")
    print("  Example: Service A | Service B | Service C")
    print("  Use case: Microservices, parallel processing")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Vertical Composition organizes components in layered stack
2. Three-tier: Presentation + Business + Data (classic)
3. N-tier: Multiple intermediate layers (application, domain, etc.)
4. Flow: Requests flow down layers, responses flow up
5. Dependency Direction: Each layer depends only on layer below
6. Separation of Concerns: Each layer has distinct responsibility
7. Presentation Layer: UI, input validation, output formatting
8. Business Layer: Business rules, domain logic, workflows
9. Data Layer: Database access, persistence, caching
10. Benefits: testability, maintainability, scalability, portability
11. Trade-offs: performance overhead, complexity, potential over-engineering
12. Clean Architecture: Dependencies point inward toward domain
13. Repository Pattern: Abstract data access behind collections
14. Use cases: web apps, enterprise systems, mobile apps, APIs
    """)

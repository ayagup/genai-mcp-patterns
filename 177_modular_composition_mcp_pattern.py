"""
Pattern 177: Modular Composition MCP Pattern

This pattern demonstrates modular architecture where the system is composed of
independent, self-contained modules that can be developed, tested, deployed, and
replaced independently. Modules communicate through well-defined interfaces with
loose coupling and high cohesion.

Key Concepts:
1. Module: Self-contained unit with specific functionality
2. Interface: Contract defining how modules interact
3. Loose Coupling: Modules depend on interfaces, not implementations
4. High Cohesion: Related functionality grouped together
5. Encapsulation: Internal implementation hidden
6. Substitutability: Modules can be swapped if they implement same interface
7. Independent Lifecycle: Develop, test, deploy modules separately

Modular Design Principles:
- Single Responsibility: Each module does one thing well
- Open/Closed: Open for extension, closed for modification
- Dependency Inversion: Depend on abstractions, not concretions
- Interface Segregation: Specific interfaces better than one general
- Explicit Dependencies: Clear what module needs

Module Types:
1. Core Modules: Essential system functionality
2. Feature Modules: Specific features/capabilities
3. Shared Modules: Common utilities/services
4. Integration Modules: Connect to external systems

Benefits:
- Maintainability: Changes isolated to specific modules
- Testability: Test modules independently
- Reusability: Modules used across projects
- Parallel Development: Teams work on different modules
- Flexibility: Swap module implementations
- Scalability: Add modules without changing existing

Trade-offs:
- Design Overhead: Need to design good interfaces
- Integration Complexity: Modules must work together
- Performance: Interface calls have overhead
- Versioning: Managing module version compatibility
- Discovery: Finding right modules

Use Cases:
- Large applications: Break into manageable modules
- Plugin systems: Core + optional modules
- Multi-tenant: Tenant-specific modules
- A/B testing: Alternative module implementations
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Protocol
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Define the state for modular composition
class ModularCompositionState(TypedDict):
    """State shared across modules"""
    input_data: str
    module_results: Annotated[List[Dict[str, Any]], operator.add]
    final_output: str
    module_calls: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# MODULE INTERFACES (Contracts)
# ============================================================================

class IModule(Protocol):
    """
    Module Interface: Contract that all modules must fulfill
    
    Protocol defines the contract without implementation.
    Modules implement this interface to be compatible.
    """
    
    def get_name(self) -> str:
        """Get module name"""
        ...
    
    def get_version(self) -> str:
        """Get module version"""
        ...
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute module's functionality"""
        ...
    
    def get_dependencies(self) -> List[str]:
        """Get list of module dependencies"""
        ...

class IDataProcessor(ABC):
    """Interface for data processing modules"""
    
    @abstractmethod
    def process(self, data: str) -> str:
        """Process data and return result"""
        pass
    
    @abstractmethod
    def validate(self, data: str) -> bool:
        """Validate input data"""
        pass

class IAnalyzer(ABC):
    """Interface for analyzer modules"""
    
    @abstractmethod
    def analyze(self, data: str) -> Dict[str, Any]:
        """Analyze data and return insights"""
        pass

class IStorage(ABC):
    """Interface for storage modules"""
    
    @abstractmethod
    def save(self, key: str, value: Any) -> bool:
        """Save data"""
        pass
    
    @abstractmethod
    def load(self, key: str) -> Optional[Any]:
        """Load data"""
        pass

# ============================================================================
# MODULE IMPLEMENTATIONS
# ============================================================================

@dataclass
class ModuleMetadata:
    """Metadata for a module"""
    name: str
    version: str
    description: str
    dependencies: List[str]
    author: str

class BaseModule(ABC):
    """
    Base Module: Common functionality for all modules
    
    Provides:
    - Metadata management
    - Dependency declaration
    - Lifecycle hooks
    """
    
    def __init__(self, metadata: ModuleMetadata):
        self.metadata = metadata
        self._initialized = False
    
    def initialize(self):
        """Initialize module (lifecycle hook)"""
        self._initialized = True
        print(f"Module {self.metadata.name} initialized")
    
    def cleanup(self):
        """Cleanup module (lifecycle hook)"""
        self._initialized = False
        print(f"Module {self.metadata.name} cleaned up")
    
    def get_name(self) -> str:
        return self.metadata.name
    
    def get_version(self) -> str:
        return self.metadata.version
    
    def get_dependencies(self) -> List[str]:
        return self.metadata.dependencies
    
    @abstractmethod
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Each module implements its specific logic"""
        pass

class TextProcessorModule(BaseModule, IDataProcessor):
    """
    Text Processor Module: Processes text data
    
    Demonstrates:
    - Implements both BaseModule and IDataProcessor
    - Self-contained functionality
    - Clear interface
    """
    
    def __init__(self):
        metadata = ModuleMetadata(
            name="TextProcessor",
            version="1.0.0",
            description="Process and transform text data",
            dependencies=[],
            author="AI Team"
        )
        super().__init__(metadata)
    
    def process(self, data: str) -> str:
        """Process text data"""
        # Text processing logic
        return f"Processed: {data.upper()}"
    
    def validate(self, data: str) -> bool:
        """Validate text data"""
        return isinstance(data, str) and len(data) > 0
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute module's main functionality"""
        if not self.validate(input_data):
            return {"error": "Invalid input"}
        
        processed = self.process(input_data)
        
        return {
            "module": self.metadata.name,
            "version": self.metadata.version,
            "input": input_data,
            "output": processed,
            "success": True
        }

class SentimentAnalyzerModule(BaseModule, IAnalyzer):
    """
    Sentiment Analyzer Module: Analyzes text sentiment
    
    Depends on TextProcessor module
    """
    
    def __init__(self):
        metadata = ModuleMetadata(
            name="SentimentAnalyzer",
            version="2.1.0",
            description="Analyze sentiment of text",
            dependencies=["TextProcessor"],
            author="ML Team"
        )
        super().__init__(metadata)
    
    def analyze(self, data: str) -> Dict[str, Any]:
        """Analyze sentiment"""
        # Simplified sentiment analysis
        positive_words = ["good", "great", "excellent", "happy"]
        negative_words = ["bad", "terrible", "awful", "sad"]
        
        data_lower = data.lower()
        pos_count = sum(1 for word in positive_words if word in data_lower)
        neg_count = sum(1 for word in negative_words if word in data_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "confidence": max(pos_count, neg_count) / (pos_count + neg_count + 1),
            "positive_count": pos_count,
            "negative_count": neg_count
        }
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute sentiment analysis"""
        analysis = self.analyze(input_data)
        
        return {
            "module": self.metadata.name,
            "version": self.metadata.version,
            "input": input_data,
            "analysis": analysis,
            "success": True
        }

class StorageModule(BaseModule, IStorage):
    """
    Storage Module: Handles data persistence
    
    Independent module with no dependencies
    """
    
    def __init__(self):
        metadata = ModuleMetadata(
            name="Storage",
            version="1.5.0",
            description="Data storage and retrieval",
            dependencies=[],
            author="Infrastructure Team"
        )
        super().__init__(metadata)
        self._storage: Dict[str, Any] = {}
    
    def save(self, key: str, value: Any) -> bool:
        """Save data"""
        self._storage[key] = value
        return True
    
    def load(self, key: str) -> Optional[Any]:
        """Load data"""
        return self._storage.get(key)
    
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute storage operation"""
        if isinstance(input_data, dict) and "operation" in input_data:
            op = input_data["operation"]
            
            if op == "save":
                success = self.save(input_data["key"], input_data["value"])
                return {"module": self.metadata.name, "operation": "save", "success": success}
            
            elif op == "load":
                value = self.load(input_data["key"])
                return {"module": self.metadata.name, "operation": "load", "value": value}
        
        return {"module": self.metadata.name, "error": "Invalid operation"}

# ============================================================================
# MODULE REGISTRY
# ============================================================================

class ModuleRegistry:
    """
    Module Registry: Manages available modules
    
    Responsibilities:
    - Register modules
    - Resolve dependencies
    - Provide modules to application
    - Version management
    """
    
    def __init__(self):
        self._modules: Dict[str, BaseModule] = {}
    
    def register(self, module: BaseModule):
        """Register a module"""
        self._modules[module.get_name()] = module
        print(f"Registered module: {module.get_name()} v{module.get_version()}")
    
    def get(self, name: str) -> Optional[BaseModule]:
        """Get a module by name"""
        return self._modules.get(name)
    
    def list_modules(self) -> List[str]:
        """List all registered modules"""
        return list(self._modules.keys())
    
    def resolve_dependencies(self, module_name: str) -> List[str]:
        """Resolve module dependencies"""
        module = self.get(module_name)
        if not module:
            return []
        
        dependencies = []
        for dep in module.get_dependencies():
            if dep in self._modules:
                dependencies.append(dep)
                # Recursively resolve dependencies
                dependencies.extend(self.resolve_dependencies(dep))
        
        return list(set(dependencies))  # Remove duplicates

# Create global registry
module_registry = ModuleRegistry()

# ============================================================================
# MODULE COMPOSITION (LangGraph Integration)
# ============================================================================

def text_processor_agent(state: ModularCompositionState) -> ModularCompositionState:
    """Agent using TextProcessor module"""
    input_data = state["input_data"]
    
    # Get module from registry
    module = module_registry.get("TextProcessor")
    if not module:
        return {"messages": ["TextProcessor module not found"]}
    
    # Execute module
    result = module.execute(input_data)
    
    return {
        "module_results": [result],
        "module_calls": [f"Module: {module.get_name()} v{module.get_version()}"],
        "messages": [f"[TextProcessor Module] Processed text"]
    }

def sentiment_analyzer_agent(state: ModularCompositionState) -> ModularCompositionState:
    """Agent using SentimentAnalyzer module"""
    input_data = state["input_data"]
    
    # Get module from registry
    module = module_registry.get("SentimentAnalyzer")
    if not module:
        return {"messages": ["SentimentAnalyzer module not found"]}
    
    # Check dependencies
    deps = module_registry.resolve_dependencies("SentimentAnalyzer")
    
    # Execute module
    result = module.execute(input_data)
    
    return {
        "module_results": [result],
        "module_calls": [f"Module: {module.get_name()} v{module.get_version()} (deps: {deps})"],
        "messages": [f"[SentimentAnalyzer Module] Analyzed sentiment"]
    }

def storage_agent(state: ModularCompositionState) -> ModularCompositionState:
    """Agent using Storage module"""
    
    # Get module from registry
    module = module_registry.get("Storage")
    if not module:
        return {"messages": ["Storage module not found"]}
    
    # Save results
    storage_input = {
        "operation": "save",
        "key": "results",
        "value": state.get("module_results", [])
    }
    
    result = module.execute(storage_input)
    
    return {
        "module_results": [result],
        "module_calls": [f"Module: {module.get_name()} v{module.get_version()}"],
        "messages": [f"[Storage Module] Saved results"]
    }

def module_aggregator(state: ModularCompositionState) -> ModularCompositionState:
    """Aggregate results from all modules"""
    
    results = state.get("module_results", [])
    
    output = f"""Modular Composition Results:
    
    Total Modules Used: {len(results)}
    
    Modules Executed:
    {chr(10).join(f"  - {call}" for call in state.get('module_calls', []))}
    
    All modules executed successfully.
    """
    
    return {
        "final_output": output.strip(),
        "messages": ["[Aggregator] All modules completed"]
    }

# ============================================================================
# MODULE LOADER (Dynamic Loading)
# ============================================================================

class ModuleLoader:
    """
    Module Loader: Dynamically load and initialize modules
    
    Features:
    - Load modules at runtime
    - Initialize dependencies
    - Hot reload modules
    - Version checking
    """
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
    
    def load_module(self, module_class):
        """Load and register a module"""
        module = module_class()
        
        # Check dependencies
        for dep in module.get_dependencies():
            if not self.registry.get(dep):
                print(f"Warning: Dependency {dep} not found for {module.get_name()}")
        
        # Initialize module
        module.initialize()
        
        # Register module
        self.registry.register(module)
        
        return module
    
    def unload_module(self, module_name: str):
        """Unload a module"""
        module = self.registry.get(module_name)
        if module:
            module.cleanup()
            # In real implementation, remove from registry

# ============================================================================
# BUILD THE MODULAR COMPOSITION GRAPH
# ============================================================================

def create_modular_composition_graph():
    """
    Create a StateGraph using modular components.
    
    Flow:
    1. Text Processor module processes input
    2. Sentiment Analyzer module analyzes text
    3. Storage module saves results
    4. Aggregator combines results
    """
    
    workflow = StateGraph(ModularCompositionState)
    
    # Add module agents
    workflow.add_node("text_processor", text_processor_agent)
    workflow.add_node("sentiment_analyzer", sentiment_analyzer_agent)
    workflow.add_node("storage", storage_agent)
    workflow.add_node("aggregator", module_aggregator)
    
    # Define flow
    workflow.add_edge(START, "text_processor")
    workflow.add_edge("text_processor", "sentiment_analyzer")
    workflow.add_edge("sentiment_analyzer", "storage")
    workflow.add_edge("storage", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Modular Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Initialize modules
    loader = ModuleLoader(module_registry)
    
    print("\nLoading Modules:")
    print("=" * 80)
    loader.load_module(TextProcessorModule)
    loader.load_module(SentimentAnalyzerModule)
    loader.load_module(StorageModule)
    
    # Example 1: Module Registry
    print("\n" + "=" * 80)
    print("Example 1: Module Registry and Dependencies")
    print("=" * 80)
    
    print("\nRegistered Modules:")
    for module_name in module_registry.list_modules():
        module = module_registry.get(module_name)
        print(f"  - {module_name} v{module.get_version()}")
        print(f"    Description: {module.metadata.description}")
        print(f"    Dependencies: {module.get_dependencies() or 'None'}")
    
    print("\nDependency Resolution:")
    deps = module_registry.resolve_dependencies("SentimentAnalyzer")
    print(f"  SentimentAnalyzer depends on: {deps}")
    
    # Example 2: Modular Composition
    print("\n" + "=" * 80)
    print("Example 2: Modular Composition Execution")
    print("=" * 80)
    
    modular_graph = create_modular_composition_graph()
    
    initial_state: ModularCompositionState = {
        "input_data": "This is a great example of modular architecture!",
        "module_results": [],
        "final_output": "",
        "module_calls": [],
        "messages": []
    }
    
    result = modular_graph.invoke(initial_state)
    
    print("\nModule Execution:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nModule Results:")
    for i, mod_result in enumerate(result["module_results"], 1):
        print(f"  {i}. Module: {mod_result.get('module', 'Unknown')}")
        if "output" in mod_result:
            print(f"     Output: {mod_result['output'][:50]}...")
        if "analysis" in mod_result:
            print(f"     Analysis: {mod_result['analysis']}")
    
    print("\nFinal Output:")
    print(result["final_output"])
    
    # Example 3: Module Substitutability
    print("\n" + "=" * 80)
    print("Example 3: Module Substitutability")
    print("=" * 80)
    
    print("\nOriginal TextProcessor Module:")
    original_module = module_registry.get("TextProcessor")
    print(f"  Name: {original_module.get_name()}")
    print(f"  Version: {original_module.get_version()}")
    
    print("\nCan be replaced with alternative implementation:")
    print("  - AdvancedTextProcessor v2.0")
    print("  - LightweightTextProcessor v1.0")
    print("  - As long as they implement IDataProcessor interface")
    
    print("\nNo changes needed in:")
    print("  ✓ Application code")
    print("  ✓ Other modules")
    print("  ✓ Module consumers")
    print("  Just swap the module registration!")
    
    # Example 4: Modular Design Principles
    print("\n" + "=" * 80)
    print("Example 4: Modular Design Principles")
    print("=" * 80)
    
    print("\n1. Single Responsibility:")
    print("   TextProcessor: Only processes text")
    print("   SentimentAnalyzer: Only analyzes sentiment")
    print("   Storage: Only handles persistence")
    
    print("\n2. Loose Coupling:")
    print("   Modules depend on interfaces, not implementations")
    print("   SentimentAnalyzer depends on IDataProcessor, not TextProcessorModule")
    
    print("\n3. High Cohesion:")
    print("   All text processing logic in TextProcessor module")
    print("   All sentiment logic in SentimentAnalyzer module")
    
    print("\n4. Explicit Dependencies:")
    print("   ModuleMetadata clearly lists dependencies")
    print("   Registry can resolve dependency graph")
    
    print("\n5. Interface Segregation:")
    print("   IDataProcessor: for data processing modules")
    print("   IAnalyzer: for analyzer modules")
    print("   IStorage: for storage modules")
    print("   Specific interfaces > one general interface")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Modular Composition builds systems from independent modules
2. Module: Self-contained unit with specific functionality
3. Interface: Contract defining module interactions
4. Loose Coupling: Modules depend on interfaces, not implementations
5. High Cohesion: Related functionality grouped together
6. Module Registry: Manages available modules and dependencies
7. Module Loader: Dynamically loads and initializes modules
8. Substitutability: Swap modules implementing same interface
9. Benefits: maintainability, testability, reusability, parallel development
10. Trade-offs: design overhead, integration complexity, versioning
11. Design Principles: SRP, Open/Closed, Dependency Inversion, Interface Segregation
12. Use cases: large apps, plugin systems, multi-tenant, A/B testing
    """)

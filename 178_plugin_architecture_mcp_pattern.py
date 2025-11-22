"""
Pattern 178: Plugin Architecture MCP Pattern

This pattern demonstrates a plugin-based architecture where the core system defines
extension points, and plugins can be dynamically discovered, loaded, and integrated
to extend functionality without modifying the core system.

Key Concepts:
1. Core System: Minimal base functionality with extension points
2. Plugin: Independent module that extends core functionality
3. Extension Point: Well-defined interface where plugins can hook in
4. Plugin Discovery: Mechanism to find available plugins
5. Plugin Loader: Dynamically loads and initializes plugins
6. Plugin Registry: Tracks loaded plugins
7. Hot-Plug: Load/unload plugins at runtime without restart

Plugin Architecture Components:
- Plugin Interface: Contract plugins must implement
- Plugin Manifest: Metadata about plugin (name, version, dependencies)
- Plugin Manager: Discovers, loads, unloads, and manages plugins
- Hook Points: Specific places in core where plugins execute
- Plugin Lifecycle: initialize → activate → execute → deactivate → cleanup

Plugin Discovery Methods:
1. Convention-Based: Scan specific directories for plugins
2. Manifest-Based: Read plugin configuration files
3. Registry-Based: Plugins register themselves
4. Annotation-Based: Discover via decorators/attributes

Benefits:
- Extensibility: Add features without modifying core
- Flexibility: Enable/disable features via plugins
- Customization: Users can create custom plugins
- Isolation: Plugin failures don't crash core
- Distribution: Distribute plugins separately
- Third-Party: Allow external plugin development

Trade-offs:
- Complexity: More complex than monolithic
- Security: Plugins can be malicious
- Versioning: Plugin compatibility with core versions
- Performance: Plugin loading and discovery overhead
- Debugging: Harder to debug across plugin boundaries

Use Cases:
- IDEs: VS Code extensions, IntelliJ plugins
- Browsers: Chrome extensions, Firefox add-ons
- CMS: WordPress plugins, Drupal modules
- Frameworks: Flask extensions, Pytest plugins
- Applications: Photoshop plugins, Jenkins plugins
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import importlib
import inspect

# Define the state for plugin architecture
class PluginArchitectureState(TypedDict):
    """State for plugin-based system"""
    input_data: str
    core_result: Optional[Dict[str, Any]]
    plugin_results: Annotated[List[Dict[str, Any]], operator.add]
    final_output: str
    plugins_loaded: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# PLUGIN INTERFACE AND BASE
# ============================================================================

class PluginState(Enum):
    """Plugin lifecycle states"""
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"

@dataclass
class PluginManifest:
    """
    Plugin Manifest: Metadata about a plugin
    
    Contains all information needed to load and manage the plugin.
    """
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    core_version: str  # Compatible core version
    entry_point: str   # Main class/function
    extension_points: List[str]  # Which extension points this plugin uses

class IPlugin(ABC):
    """
    Plugin Interface: Contract all plugins must implement
    
    This defines the lifecycle and capabilities that core system expects
    from every plugin.
    """
    
    @abstractmethod
    def get_manifest(self) -> PluginManifest:
        """Get plugin manifest"""
        pass
    
    @abstractmethod
    def initialize(self, core_context: Dict[str, Any]) -> bool:
        """Initialize plugin with core context"""
        pass
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Execute plugin functionality"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup when plugin is unloaded"""
        pass
    
    @abstractmethod
    def get_hooks(self) -> Dict[str, Callable]:
        """Get hook functions for extension points"""
        pass

class BasePlugin(IPlugin):
    """
    Base Plugin: Common functionality for all plugins
    
    Provides default implementation of lifecycle methods.
    """
    
    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest
        self.state = PluginState.DISCOVERED
        self.core_context: Optional[Dict[str, Any]] = None
    
    def get_manifest(self) -> PluginManifest:
        return self.manifest
    
    def initialize(self, core_context: Dict[str, Any]) -> bool:
        """Initialize plugin"""
        self.core_context = core_context
        self.state = PluginState.INITIALIZED
        return True
    
    def cleanup(self):
        """Cleanup plugin"""
        self.state = PluginState.INACTIVE
        self.core_context = None
    
    @abstractmethod
    def execute(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Plugins must implement execute"""
        pass
    
    @abstractmethod
    def get_hooks(self) -> Dict[str, Callable]:
        """Plugins must define their hooks"""
        pass

# ============================================================================
# CONCRETE PLUGIN IMPLEMENTATIONS
# ============================================================================

class DataValidationPlugin(BasePlugin):
    """
    Plugin 1: Data Validation
    
    Hooks into: pre_process extension point
    Function: Validates data before processing
    """
    
    def __init__(self):
        manifest = PluginManifest(
            name="DataValidation",
            version="1.0.0",
            description="Validates input data",
            author="Core Team",
            dependencies=[],
            core_version=">=1.0.0",
            entry_point="DataValidationPlugin",
            extension_points=["pre_process"]
        )
        super().__init__(manifest)
    
    def execute(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Validate data"""
        is_valid = isinstance(data, str) and len(data) > 0
        
        return {
            "plugin": self.manifest.name,
            "valid": is_valid,
            "message": "Data is valid" if is_valid else "Data is invalid"
        }
    
    def get_hooks(self) -> Dict[str, Callable]:
        """Define hook functions"""
        return {
            "pre_process": self._validate_hook
        }
    
    def _validate_hook(self, data: Any) -> Dict[str, Any]:
        """Hook function executed at pre_process extension point"""
        return self.execute(data)

class SentimentAnalysisPlugin(BasePlugin):
    """
    Plugin 2: Sentiment Analysis
    
    Hooks into: post_process extension point
    Function: Analyzes sentiment after processing
    """
    
    def __init__(self):
        manifest = PluginManifest(
            name="SentimentAnalysis",
            version="2.0.0",
            description="Analyzes text sentiment",
            author="ML Team",
            dependencies=["DataValidation"],
            core_version=">=1.0.0",
            entry_point="SentimentAnalysisPlugin",
            extension_points=["post_process"]
        )
        super().__init__(manifest)
    
    def execute(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Analyze sentiment"""
        # Simplified sentiment analysis
        data_str = str(data).lower()
        
        if any(word in data_str for word in ["good", "great", "excellent"]):
            sentiment = "positive"
        elif any(word in data_str for word in ["bad", "terrible", "awful"]):
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "plugin": self.manifest.name,
            "sentiment": sentiment,
            "analyzed": True
        }
    
    def get_hooks(self) -> Dict[str, Callable]:
        return {
            "post_process": self._analyze_hook
        }
    
    def _analyze_hook(self, data: Any) -> Dict[str, Any]:
        """Hook function for post-processing"""
        return self.execute(data)

class LoggingPlugin(BasePlugin):
    """
    Plugin 3: Logging
    
    Hooks into: Multiple extension points
    Function: Logs operations throughout the system
    """
    
    def __init__(self):
        manifest = PluginManifest(
            name="Logging",
            version="1.5.0",
            description="Logs system operations",
            author="DevOps Team",
            dependencies=[],
            core_version=">=1.0.0",
            entry_point="LoggingPlugin",
            extension_points=["pre_process", "post_process", "error"]
        )
        super().__init__(manifest)
        self.logs: List[str] = []
    
    def execute(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Log data"""
        log_entry = f"[{self.manifest.name}] {kwargs.get('event', 'UNKNOWN')}: {str(data)[:50]}"
        self.logs.append(log_entry)
        
        return {
            "plugin": self.manifest.name,
            "logged": True,
            "log_count": len(self.logs)
        }
    
    def get_hooks(self) -> Dict[str, Callable]:
        return {
            "pre_process": lambda data: self.execute(data, event="PRE_PROCESS"),
            "post_process": lambda data: self.execute(data, event="POST_PROCESS"),
            "error": lambda data: self.execute(data, event="ERROR")
        }
    
    def get_logs(self) -> List[str]:
        """Get all logged entries"""
        return self.logs

# ============================================================================
# PLUGIN MANAGER
# ============================================================================

class PluginManager:
    """
    Plugin Manager: Discovers, loads, and manages plugins
    
    Responsibilities:
    - Discover available plugins
    - Load and initialize plugins
    - Execute plugins at extension points
    - Manage plugin lifecycle
    - Handle plugin dependencies
    """
    
    def __init__(self, core_version: str = "1.0.0"):
        self.core_version = core_version
        self.plugins: Dict[str, IPlugin] = {}
        self.extension_points: Dict[str, List[IPlugin]] = {}
        self.core_context = {"version": core_version}
    
    def register_plugin(self, plugin: IPlugin) -> bool:
        """Register a plugin"""
        manifest = plugin.get_manifest()
        
        # Check core version compatibility
        if not self._is_compatible(manifest.core_version):
            print(f"Plugin {manifest.name} not compatible with core {self.core_version}")
            return False
        
        # Check dependencies
        for dep in manifest.dependencies:
            if dep not in self.plugins:
                print(f"Plugin {manifest.name} missing dependency: {dep}")
                return False
        
        # Initialize plugin
        if not plugin.initialize(self.core_context):
            print(f"Failed to initialize plugin {manifest.name}")
            return False
        
        # Register plugin
        self.plugins[manifest.name] = plugin
        
        # Register hooks at extension points
        hooks = plugin.get_hooks()
        for extension_point, hook_fn in hooks.items():
            if extension_point not in self.extension_points:
                self.extension_points[extension_point] = []
            self.extension_points[extension_point].append(plugin)
        
        print(f"Registered plugin: {manifest.name} v{manifest.version}")
        return True
    
    def unregister_plugin(self, plugin_name: str):
        """Unregister a plugin"""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            plugin.cleanup()
            
            # Remove from extension points
            for plugins in self.extension_points.values():
                if plugin in plugins:
                    plugins.remove(plugin)
            
            del self.plugins[plugin_name]
            print(f"Unregistered plugin: {plugin_name}")
    
    def execute_extension_point(self, extension_point: str, data: Any) -> List[Dict[str, Any]]:
        """
        Execute all plugins registered at an extension point
        
        This is where plugins actually run!
        """
        results = []
        
        if extension_point in self.extension_points:
            for plugin in self.extension_points[extension_point]:
                hooks = plugin.get_hooks()
                if extension_point in hooks:
                    result = hooks[extension_point](data)
                    results.append(result)
        
        return results
    
    def list_plugins(self) -> List[PluginManifest]:
        """List all registered plugins"""
        return [p.get_manifest() for p in self.plugins.values()]
    
    def get_plugin(self, name: str) -> Optional[IPlugin]:
        """Get a plugin by name"""
        return self.plugins.get(name)
    
    def _is_compatible(self, required_version: str) -> bool:
        """Check version compatibility"""
        # Simplified version check
        return True  # In real implementation, parse and compare versions

# Create global plugin manager
plugin_manager = PluginManager()

# ============================================================================
# CORE SYSTEM (LangGraph Integration)
# ============================================================================

def core_system(state: PluginArchitectureState) -> PluginArchitectureState:
    """
    Core System: Minimal base functionality
    
    The core is kept minimal. Functionality is added through plugins!
    """
    input_data = state["input_data"]
    
    core_result = {
        "system": "core",
        "version": "1.0.0",
        "processed": input_data,
        "extension_points_available": ["pre_process", "post_process", "error"]
    }
    
    return {
        "core_result": core_result,
        "messages": ["[Core System] Base processing completed"]
    }

def pre_process_extension_point(state: PluginArchitectureState) -> PluginArchitectureState:
    """
    Extension Point: pre_process
    
    Executes all plugins registered at pre_process extension point.
    """
    input_data = state["input_data"]
    
    # Execute all plugins at this extension point
    results = plugin_manager.execute_extension_point("pre_process", input_data)
    
    return {
        "plugin_results": results,
        "messages": [f"[Extension Point: pre_process] Executed {len(results)} plugins"]
    }

def post_process_extension_point(state: PluginArchitectureState) -> PluginArchitectureState:
    """
    Extension Point: post_process
    
    Executes all plugins registered at post_process extension point.
    """
    core_result = state.get("core_result", {})
    processed_data = core_result.get("processed", "")
    
    # Execute all plugins at this extension point
    results = plugin_manager.execute_extension_point("post_process", processed_data)
    
    return {
        "plugin_results": results,
        "messages": [f"[Extension Point: post_process] Executed {len(results)} plugins"]
    }

def plugin_aggregator(state: PluginArchitectureState) -> PluginArchitectureState:
    """Aggregate results from core and plugins"""
    
    plugin_results = state.get("plugin_results", [])
    plugins_loaded = [p.get_manifest().name for p in plugin_manager.plugins.values()]
    
    output = f"""Plugin Architecture Results:
    
    Core System: v{plugin_manager.core_version}
    Total Plugins Loaded: {len(plugins_loaded)}
    
    Loaded Plugins:
    {chr(10).join(f"  - {name}" for name in plugins_loaded)}
    
    Plugin Results: {len(plugin_results)} executions
    
    Extension points successfully executed.
    """
    
    return {
        "plugins_loaded": plugins_loaded,
        "final_output": output.strip(),
        "messages": ["[Aggregator] Plugin execution completed"]
    }

# ============================================================================
# PLUGIN DISCOVERY
# ============================================================================

class PluginDiscovery:
    """
    Plugin Discovery: Find available plugins
    
    Methods:
    - Scan directory for plugin files
    - Read plugin manifests
    - Detect plugin classes
    """
    
    @staticmethod
    def discover_plugins() -> List[type]:
        """
        Discover plugin classes
        
        In real implementation:
        - Scan plugin directories
        - Load Python modules
        - Find classes implementing IPlugin
        - Read plugin manifests
        """
        # For this example, return hardcoded plugins
        return [
            DataValidationPlugin,
            SentimentAnalysisPlugin,
            LoggingPlugin
        ]
    
    @staticmethod
    def load_plugin_class(plugin_class: type) -> Optional[IPlugin]:
        """Load and instantiate a plugin class"""
        try:
            plugin = plugin_class()
            return plugin
        except Exception as e:
            print(f"Failed to load plugin {plugin_class.__name__}: {e}")
            return None

# ============================================================================
# BUILD THE PLUGIN ARCHITECTURE GRAPH
# ============================================================================

def create_plugin_architecture_graph():
    """
    Create a StateGraph demonstrating plugin architecture.
    
    Flow:
    1. pre_process extension point (plugins execute)
    2. Core system processing
    3. post_process extension point (plugins execute)
    4. Aggregator combines results
    
    Plugins hook into extension points to extend functionality.
    """
    
    workflow = StateGraph(PluginArchitectureState)
    
    # Add nodes
    workflow.add_node("pre_process_hook", pre_process_extension_point)
    workflow.add_node("core", core_system)
    workflow.add_node("post_process_hook", post_process_extension_point)
    workflow.add_node("aggregator", plugin_aggregator)
    
    # Define flow with extension points
    workflow.add_edge(START, "pre_process_hook")
    workflow.add_edge("pre_process_hook", "core")
    workflow.add_edge("core", "post_process_hook")
    workflow.add_edge("post_process_hook", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Plugin Architecture MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Plugin Discovery and Loading
    print("\n" + "=" * 80)
    print("Example 1: Discover and Load Plugins")
    print("=" * 80)
    
    # Discover plugins
    print("\nDiscovering plugins...")
    plugin_classes = PluginDiscovery.discover_plugins()
    print(f"Found {len(plugin_classes)} plugins")
    
    # Load and register plugins
    print("\nLoading plugins:")
    for plugin_class in plugin_classes:
        plugin = PluginDiscovery.load_plugin_class(plugin_class)
        if plugin:
            plugin_manager.register_plugin(plugin)
    
    # List loaded plugins
    print("\nLoaded Plugins:")
    for manifest in plugin_manager.list_plugins():
        print(f"  - {manifest.name} v{manifest.version}")
        print(f"    Description: {manifest.description}")
        print(f"    Extension Points: {', '.join(manifest.extension_points)}")
        print(f"    Dependencies: {', '.join(manifest.dependencies) or 'None'}")
    
    # Example 2: Plugin Execution
    print("\n" + "=" * 80)
    print("Example 2: Execute Core System with Plugins")
    print("=" * 80)
    
    plugin_graph = create_plugin_architecture_graph()
    
    initial_state: PluginArchitectureState = {
        "input_data": "This is a great example of plugin architecture!",
        "core_result": None,
        "plugin_results": [],
        "final_output": "",
        "plugins_loaded": [],
        "messages": []
    }
    
    result = plugin_graph.invoke(initial_state)
    
    print("\nExecution Flow:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nPlugin Results:")
    for plugin_result in result["plugin_results"]:
        print(f"  Plugin: {plugin_result.get('plugin', 'Unknown')}")
        for key, value in plugin_result.items():
            if key != 'plugin':
                print(f"    {key}: {value}")
    
    print("\nFinal Output:")
    print(result["final_output"])
    
    # Example 3: Hot-Plug (Runtime Plugin Management)
    print("\n" + "=" * 80)
    print("Example 3: Hot-Plug (Add/Remove Plugins at Runtime)")
    print("=" * 80)
    
    print("\nCurrent plugins:", list(plugin_manager.plugins.keys()))
    
    print("\nRemoving SentimentAnalysis plugin...")
    plugin_manager.unregister_plugin("SentimentAnalysis")
    
    print("Remaining plugins:", list(plugin_manager.plugins.keys()))
    
    print("\nRe-adding SentimentAnalysis plugin...")
    new_plugin = SentimentAnalysisPlugin()
    plugin_manager.register_plugin(new_plugin)
    
    print("Plugins after re-adding:", list(plugin_manager.plugins.keys()))
    
    # Example 4: Extension Points
    print("\n" + "=" * 80)
    print("Example 4: Extension Points in Plugin Architecture")
    print("=" * 80)
    
    print("\nExtension Points:")
    for ext_point, plugins in plugin_manager.extension_points.items():
        print(f"  {ext_point}:")
        for plugin in plugins:
            print(f"    - {plugin.get_manifest().name}")
    
    print("\nHow Extension Points Work:")
    print("  1. Core system defines extension points (pre_process, post_process, etc.)")
    print("  2. Plugins register hooks at extension points")
    print("  3. When core reaches extension point, all registered plugin hooks execute")
    print("  4. Results aggregated and passed to next stage")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Plugin Architecture extends core system without modification
2. Core System: Minimal base with extension points
3. Plugin: Independent module implementing IPlugin interface
4. Extension Point: Location where plugins can hook in
5. Plugin Manager: Discovers, loads, manages plugins
6. Plugin Manifest: Metadata (name, version, dependencies)
7. Plugin Discovery: Find available plugins (scan, manifest, registry)
8. Plugin Lifecycle: discover → load → initialize → execute → cleanup
9. Hot-Plug: Add/remove plugins at runtime
10. Hook Points: pre_process, post_process, error, etc.
11. Benefits: extensibility, flexibility, customization, third-party development
12. Trade-offs: complexity, security, versioning, performance
13. Use cases: IDEs, browsers, CMS, frameworks, applications
14. Real-world: VS Code extensions, Chrome plugins, WordPress plugins
    """)

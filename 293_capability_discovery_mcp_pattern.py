"""
Pattern 293: Capability Discovery MCP Pattern

This pattern demonstrates discovering and querying capabilities of agents
and services to determine what operations they can perform.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class CapabilityDiscoveryPattern(TypedDict):
    """State for capability discovery"""
    messages: Annotated[List[str], add]
    capability_catalog: Dict[str, Any]
    capability_queries: List[Dict[str, Any]]
    compatibility_matrix: Dict[str, List[str]]
    recommendations: List[Dict[str, Any]]


class Capability:
    """Represents a capability"""
    
    def __init__(self, capability_id: str, name: str, category: str):
        self.capability_id = capability_id
        self.name = name
        self.category = category
        self.description = ""
        self.parameters = []
        self.returns = None
        self.dependencies = []
        self.version = "1.0.0"
        self.provider = None
    
    def add_parameter(self, param_name: str, param_type: str, required: bool = True):
        """Add parameter"""
        self.parameters.append({
            "name": param_name,
            "type": param_type,
            "required": required
        })
    
    def add_dependency(self, capability_id: str):
        """Add dependency"""
        if capability_id not in self.dependencies:
            self.dependencies.append(capability_id)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "dependencies": self.dependencies,
            "version": self.version,
            "provider": self.provider
        }


class CapabilityCatalog:
    """Catalog of capabilities"""
    
    def __init__(self):
        self.capabilities = {}
        self.category_index = {}
        self.provider_index = {}
    
    def register_capability(self, capability: Capability):
        """Register a capability"""
        self.capabilities[capability.capability_id] = capability
        
        # Index by category
        if capability.category not in self.category_index:
            self.category_index[capability.category] = []
        self.category_index[capability.category].append(capability.capability_id)
        
        # Index by provider
        if capability.provider:
            if capability.provider not in self.provider_index:
                self.provider_index[capability.provider] = []
            self.provider_index[capability.provider].append(capability.capability_id)
    
    def discover_by_category(self, category: str):
        """Discover capabilities by category"""
        cap_ids = self.category_index.get(category, [])
        return [self.capabilities[cid].to_dict() for cid in cap_ids]
    
    def discover_by_provider(self, provider: str):
        """Discover capabilities by provider"""
        cap_ids = self.provider_index.get(provider, [])
        return [self.capabilities[cid].to_dict() for cid in cap_ids]
    
    def get_capability(self, capability_id: str):
        """Get specific capability"""
        if capability_id in self.capabilities:
            return self.capabilities[capability_id].to_dict()
        return None
    
    def check_dependencies(self, capability_id: str):
        """Check if all dependencies are available"""
        if capability_id not in self.capabilities:
            return False, []
        
        cap = self.capabilities[capability_id]
        missing = []
        
        for dep_id in cap.dependencies:
            if dep_id not in self.capabilities:
                missing.append(dep_id)
        
        return len(missing) == 0, missing
    
    def find_compatible_capabilities(self, output_type: str):
        """Find capabilities that can consume specific output"""
        compatible = []
        
        for cap in self.capabilities.values():
            for param in cap.parameters:
                if param["type"] == output_type:
                    compatible.append(cap.capability_id)
                    break
        
        return compatible


def initialize_capability_catalog_agent(state: CapabilityDiscoveryPattern) -> CapabilityDiscoveryPattern:
    """Initialize capability catalog"""
    print("\n📚 Initializing Capability Catalog...")
    
    catalog = CapabilityCatalog()
    
    print(f"  Catalog: Ready")
    print(f"  Features:")
    print(f"    • Capability registration")
    print(f"    • Category indexing")
    print(f"    • Dependency tracking")
    print(f"    • Compatibility checking")
    
    return {
        **state,
        "capability_catalog": {},
        "capability_queries": [],
        "compatibility_matrix": {},
        "recommendations": [],
        "messages": ["✓ Capability catalog initialized"]
    }


def register_capabilities_agent(state: CapabilityDiscoveryPattern) -> CapabilityDiscoveryPattern:
    """Register various capabilities"""
    print("\n📝 Registering Capabilities...")
    
    catalog = CapabilityCatalog()
    
    # Define capabilities
    capabilities_config = [
        {
            "id": "text_analysis",
            "name": "Text Analysis",
            "category": "nlp",
            "description": "Analyze text for various attributes",
            "params": [("text", "string", True)],
            "returns": "analysis_result",
            "provider": "nlp_service"
        },
        {
            "id": "sentiment_detection",
            "name": "Sentiment Detection",
            "category": "nlp",
            "description": "Detect sentiment in text",
            "params": [("text", "string", True)],
            "returns": "sentiment_score",
            "provider": "nlp_service",
            "deps": ["text_analysis"]
        },
        {
            "id": "entity_extraction",
            "name": "Entity Extraction",
            "category": "nlp",
            "description": "Extract named entities",
            "params": [("text", "string", True)],
            "returns": "entity_list",
            "provider": "nlp_service",
            "deps": ["text_analysis"]
        },
        {
            "id": "image_recognition",
            "name": "Image Recognition",
            "category": "vision",
            "description": "Recognize objects in images",
            "params": [("image", "binary", True)],
            "returns": "recognition_result",
            "provider": "vision_service"
        },
        {
            "id": "ocr",
            "name": "Optical Character Recognition",
            "category": "vision",
            "description": "Extract text from images",
            "params": [("image", "binary", True)],
            "returns": "string",
            "provider": "vision_service",
            "deps": ["image_recognition"]
        },
        {
            "id": "data_transformation",
            "name": "Data Transformation",
            "category": "data",
            "description": "Transform data formats",
            "params": [("data", "any", True), ("target_format", "string", True)],
            "returns": "transformed_data",
            "provider": "data_service"
        },
        {
            "id": "statistical_analysis",
            "name": "Statistical Analysis",
            "category": "data",
            "description": "Perform statistical analysis",
            "params": [("dataset", "array", True)],
            "returns": "statistics",
            "provider": "data_service"
        }
    ]
    
    for config in capabilities_config:
        cap = Capability(config["id"], config["name"], config["category"])
        cap.description = config["description"]
        cap.returns = config["returns"]
        cap.provider = config["provider"]
        
        for param_name, param_type, required in config["params"]:
            cap.add_parameter(param_name, param_type, required)
        
        for dep in config.get("deps", []):
            cap.add_dependency(dep)
        
        catalog.register_capability(cap)
        
        print(f"  ✓ Registered: {cap.name}")
        print(f"    ID: {cap.capability_id}")
        print(f"    Category: {cap.category}")
        print(f"    Parameters: {len(cap.parameters)}")
    
    print(f"\n  Total Capabilities: {len(catalog.capabilities)}")
    print(f"  Categories: {len(catalog.category_index)}")
    
    cap_dict = {cid: cap.to_dict() for cid, cap in catalog.capabilities.items()}
    
    return {
        **state,
        "capability_catalog": cap_dict,
        "messages": [f"✓ Registered {len(capabilities_config)} capabilities"]
    }


def query_capabilities_agent(state: CapabilityDiscoveryPattern) -> CapabilityDiscoveryPattern:
    """Query capabilities"""
    print("\n🔍 Querying Capabilities...")
    
    catalog = CapabilityCatalog()
    
    # Recreate catalog
    for cap_id, cap_data in state["capability_catalog"].items():
        cap = Capability(cap_data["capability_id"], cap_data["name"], cap_data["category"])
        cap.description = cap_data["description"]
        cap.parameters = cap_data["parameters"]
        cap.returns = cap_data["returns"]
        cap.dependencies = cap_data["dependencies"]
        cap.provider = cap_data["provider"]
        catalog.register_capability(cap)
    
    # Execute queries
    queries = [
        {"type": "category", "value": "nlp", "name": "NLP Capabilities"},
        {"type": "category", "value": "vision", "name": "Vision Capabilities"},
        {"type": "provider", "value": "nlp_service", "name": "NLP Service Capabilities"}
    ]
    
    query_results = []
    
    for query in queries:
        if query["type"] == "category":
            results = catalog.discover_by_category(query["value"])
        elif query["type"] == "provider":
            results = catalog.discover_by_provider(query["value"])
        else:
            results = []
        
        print(f"\n  Query: {query['name']}")
        print(f"  Results: {len(results)} capability(ies)")
        
        for cap in results:
            print(f"    • {cap['name']}")
            print(f"      Returns: {cap['returns']}")
        
        query_results.append({
            "query": query["name"],
            "results": results
        })
    
    return {
        **state,
        "capability_queries": query_results,
        "messages": [f"✓ Executed {len(queries)} queries"]
    }


def check_compatibility_agent(state: CapabilityDiscoveryPattern) -> CapabilityDiscoveryPattern:
    """Check capability compatibility"""
    print("\n🔗 Checking Capability Compatibility...")
    
    catalog = CapabilityCatalog()
    
    # Recreate catalog
    for cap_id, cap_data in state["capability_catalog"].items():
        cap = Capability(cap_data["capability_id"], cap_data["name"], cap_data["category"])
        cap.returns = cap_data["returns"]
        cap.parameters = cap_data["parameters"]
        cap.dependencies = cap_data["dependencies"]
        catalog.register_capability(cap)
    
    # Build compatibility matrix
    compatibility_matrix = {}
    
    for cap_id, cap in catalog.capabilities.items():
        if cap.returns:
            compatible = catalog.find_compatible_capabilities(cap.returns)
            if compatible:
                compatibility_matrix[cap_id] = compatible
                
                print(f"\n  {cap.name} ({cap.returns}):")
                print(f"    Compatible with {len(compatible)} capability(ies)")
                for comp_id in compatible[:3]:
                    comp_cap = catalog.capabilities.get(comp_id)
                    if comp_cap:
                        print(f"      • {comp_cap.name}")
    
    # Check dependencies
    print(f"\n  Dependency Checks:")
    for cap_id, cap in catalog.capabilities.items():
        if cap.dependencies:
            satisfied, missing = catalog.check_dependencies(cap_id)
            status = "✓" if satisfied else "✗"
            print(f"    {status} {cap.name}: {len(cap.dependencies)} dependency(ies)")
            if missing:
                print(f"      Missing: {', '.join(missing)}")
    
    return {
        **state,
        "compatibility_matrix": compatibility_matrix,
        "messages": ["✓ Compatibility check complete"]
    }


def generate_capability_discovery_report_agent(state: CapabilityDiscoveryPattern) -> CapabilityDiscoveryPattern:
    """Generate capability discovery report"""
    print("\n" + "="*70)
    print("CAPABILITY DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n📚 Capability Catalog:")
    print(f"  Total Capabilities: {len(state['capability_catalog'])}")
    
    # Group by category
    by_category = {}
    for cap_data in state['capability_catalog'].values():
        cat = cap_data['category']
        by_category[cat] = by_category.get(cat, 0) + 1
    
    print(f"\n  Capabilities by Category:")
    for cat, count in by_category.items():
        print(f"    {cat}: {count}")
    
    print(f"\n🔧 Registered Capabilities:")
    for cap_data in list(state['capability_catalog'].values())[:6]:
        print(f"\n  • {cap_data['name']}:")
        print(f"      ID: {cap_data['capability_id']}")
        print(f"      Category: {cap_data['category']}")
        print(f"      Parameters: {len(cap_data['parameters'])}")
        if cap_data['parameters']:
            for param in cap_data['parameters'][:2]:
                print(f"        - {param['name']}: {param['type']}")
        print(f"      Returns: {cap_data['returns']}")
        if cap_data['dependencies']:
            print(f"      Dependencies: {', '.join(cap_data['dependencies'])}")
    
    print(f"\n🔍 Query Results:")
    for query_result in state['capability_queries']:
        print(f"  {query_result['query']}: {len(query_result['results'])} result(s)")
    
    print(f"\n🔗 Compatibility Matrix:")
    print(f"  Compatible Pairs: {sum(len(v) for v in state['compatibility_matrix'].values())}")
    for cap_id, compatible_ids in list(state['compatibility_matrix'].items())[:3]:
        cap_name = state['capability_catalog'][cap_id]['name']
        print(f"  {cap_name} → {len(compatible_ids)} capability(ies)")
    
    print(f"\n💡 Capability Discovery Benefits:")
    print("  ✓ Dynamic capability detection")
    print("  ✓ Service composition")
    print("  ✓ Dependency resolution")
    print("  ✓ Compatibility checking")
    print("  ✓ Provider selection")
    print("  ✓ Version management")
    
    print(f"\n🔧 Discovery Methods:")
    print("  • Category-based")
    print("  • Provider-based")
    print("  • Dependency-based")
    print("  • Compatibility-based")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Service composition")
    print("  • Workflow building")
    print("  • Agent selection")
    print("  • API discovery")
    print("  • Capability matching")
    print("  • Dynamic binding")
    
    print("\n" + "="*70)
    print("✅ Capability Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_capability_discovery_graph():
    """Create capability discovery workflow"""
    workflow = StateGraph(CapabilityDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_capability_catalog_agent)
    workflow.add_node("register", register_capabilities_agent)
    workflow.add_node("query", query_capabilities_agent)
    workflow.add_node("compatibility", check_compatibility_agent)
    workflow.add_node("report", generate_capability_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "query")
    workflow.add_edge("query", "compatibility")
    workflow.add_edge("compatibility", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 293: Capability Discovery MCP Pattern")
    print("="*70)
    
    app = create_capability_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "capability_catalog": {},
        "capability_queries": [],
        "compatibility_matrix": {},
        "recommendations": []
    })
    
    print("\n✅ Capability Discovery Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Tool Discovery MCP Pattern

This pattern implements automatic tool discovery with capability
matching and intelligent tool selection based on requirements.

Key Features:
- Automatic tool discovery
- Capability-based matching
- Dynamic tool loading
- Service discovery integration
- Tool recommendation engine
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ToolDiscoveryState(TypedDict):
    """State for tool discovery pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    available_tools: List[Dict]  # [{name, capabilities, metadata}]
    required_capabilities: List[str]
    discovered_tools: List[Dict]
    recommended_tools: List[str]
    discovery_method: str  # "registry", "filesystem", "network"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Tool Discovery Agent
def tool_discovery_agent(state: ToolDiscoveryState) -> ToolDiscoveryState:
    """Discovers available tools in the system"""
    discovery_method = state.get("discovery_method", "registry")
    
    system_message = SystemMessage(content="""You are a tool discovery agent.
    Find and catalog available tools in the system.""")
    
    user_message = HumanMessage(content=f"""Discover tools:

Method: {discovery_method}

Find all available tools.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate tool discovery
    available_tools = [
        {
            "name": "web_search",
            "capabilities": ["search_web", "fetch_url", "scrape_page"],
            "category": "information_retrieval",
            "priority": 8,
            "latency_ms": 500,
            "reliability": 0.95
        },
        {
            "name": "database_reader",
            "capabilities": ["read_data", "query_sql", "aggregate"],
            "category": "data_access",
            "priority": 9,
            "latency_ms": 100,
            "reliability": 0.99
        },
        {
            "name": "calculator",
            "capabilities": ["math_operations", "statistics", "conversions"],
            "category": "computation",
            "priority": 10,
            "latency_ms": 10,
            "reliability": 1.0
        },
        {
            "name": "text_analyzer",
            "capabilities": ["sentiment_analysis", "entity_extraction", "summarize"],
            "category": "nlp",
            "priority": 7,
            "latency_ms": 200,
            "reliability": 0.92
        },
        {
            "name": "api_caller",
            "capabilities": ["http_request", "rest_api", "graphql"],
            "category": "integration",
            "priority": 6,
            "latency_ms": 800,
            "reliability": 0.88
        }
    ]
    
    report = f"""
    🔍 Tool Discovery Agent:
    
    Discovery Results:
    • Method: {discovery_method.upper()}
    • Tools Found: {len(available_tools)}
    • Categories: {len(set(t['category'] for t in available_tools))}
    • Total Capabilities: {sum(len(t['capabilities']) for t in available_tools)}
    
    Tool Discovery Methods:
    
    Registry-Based Discovery:
    ```python
    class RegistryDiscovery:
        def __init__(self, registry_url):
            self.registry_url = registry_url
        
        def discover_tools(self):
            # Query central registry
            response = requests.get(f"{{self.registry_url}}/tools")
            tools = response.json()
            
            # Filter active tools
            active_tools = [
                t for t in tools 
                if t.get("status") == "active"
            ]
            
            return active_tools
    ```
    
    Filesystem Discovery:
    ```python
    import os
    import importlib
    import inspect
    
    class FilesystemDiscovery:
        def __init__(self, tools_dir):
            self.tools_dir = tools_dir
        
        def discover_tools(self):
            tools = []
            
            # Scan directory for Python files
            for filename in os.listdir(self.tools_dir):
                if filename.endswith('.py'):
                    module_name = filename[:-3]
                    
                    # Import module
                    module = importlib.import_module(
                        f"tools.{{module_name}}"
                    )
                    
                    # Find tool classes
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            hasattr(obj, '__tool_metadata__')):
                            tools.append({{
                                "name": name,
                                "class": obj,
                                "metadata": obj.__tool_metadata__
                            }})
            
            return tools
    ```
    
    Network Discovery (Service Discovery):
    ```python
    import consul
    
    class NetworkDiscovery:
        def __init__(self, consul_host='localhost'):
            self.consul = consul.Consul(host=consul_host)
        
        def discover_tools(self):
            # Query Consul for tool services
            _, services = self.consul.catalog.services()
            
            tools = []
            for service_name in services:
                if service_name.startswith('tool-'):
                    # Get service details
                    _, nodes = self.consul.health.service(
                        service_name, 
                        passing=True
                    )
                    
                    for node in nodes:
                        tools.append({{
                            "name": service_name,
                            "address": node['Service']['Address'],
                            "port": node['Service']['Port'],
                            "metadata": node['Service'].get('Meta', {{}})
                        }})
            
            return tools
    ```
    
    Plugin Discovery (Entry Points):
    ```python
    from importlib.metadata import entry_points
    
    class PluginDiscovery:
        def discover_tools(self):
            tools = []
            
            # Discover via entry points
            discovered = entry_points(group='mcp.tools')
            
            for entry_point in discovered:
                try:
                    tool_class = entry_point.load()
                    tools.append({{
                        "name": entry_point.name,
                        "class": tool_class,
                        "module": entry_point.value
                    }})
                except Exception as e:
                    print(f"Failed to load {{entry_point.name}}: {{e}}")
            
            return tools
    
    # In setup.py:
    # entry_points={{
    #     'mcp.tools': [
    #         'calculator = mytools.calculator:CalculatorTool',
    #         'web_search = mytools.search:WebSearchTool',
    #     ]
    # }}
    ```
    
    Docker Service Discovery:
    ```python
    import docker
    
    class DockerDiscovery:
        def __init__(self):
            self.client = docker.from_env()
        
        def discover_tools(self):
            tools = []
            
            # Find containers with tool label
            containers = self.client.containers.list(
                filters={{"label": "type=mcp-tool"}}
            )
            
            for container in containers:
                labels = container.labels
                tools.append({{
                    "name": labels.get('tool.name'),
                    "endpoint": f"http://{{container.name}}:{{labels.get('tool.port')}}",
                    "capabilities": labels.get('tool.capabilities', '').split(','),
                    "version": labels.get('tool.version')
                }})
            
            return tools
    ```
    
    Kubernetes Service Discovery:
    ```python
    from kubernetes import client, config
    
    class K8sDiscovery:
        def __init__(self):
            config.load_incluster_config()
            self.v1 = client.CoreV1Api()
        
        def discover_tools(self):
            tools = []
            
            # List services with tool label
            services = self.v1.list_service_for_all_namespaces(
                label_selector="type=mcp-tool"
            )
            
            for svc in services.items:
                metadata = svc.metadata
                spec = svc.spec
                
                tools.append({{
                    "name": metadata.labels.get('tool.name'),
                    "namespace": metadata.namespace,
                    "endpoint": f"http://{{metadata.name}}.{{metadata.namespace}}.svc.cluster.local:{{spec.ports[0].port}}",
                    "capabilities": metadata.labels.get('tool.capabilities', '').split(',')
                }})
            
            return tools
    ```
    
    Discovered Tools:
    {chr(10).join(f"• {tool['name']} ({tool['category']}) - {len(tool['capabilities'])} capabilities, Priority: {tool['priority']}" for tool in available_tools)}
    
    Capability Matching Strategies:
    
    Exact Match:
    • Required capability exactly matches
    • No approximation
    • Strict matching
    • Fast lookup
    
    Fuzzy Match:
    • Similar capabilities
    • Synonym matching
    • Embedding similarity
    • Flexible matching
    
    Semantic Match:
    • NLP-based matching
    • Context understanding
    • Intent recognition
    • Advanced matching
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Discovery Agent:\n{response.content}\n{report}")],
        "available_tools": available_tools
    }


# Capability Matcher
def capability_matcher(state: ToolDiscoveryState) -> ToolDiscoveryState:
    """Matches tools to required capabilities"""
    available_tools = state.get("available_tools", [])
    required_capabilities = state.get("required_capabilities", ["search_web", "summarize"])
    
    system_message = SystemMessage(content="""You are a capability matcher.
    Match available tools to required capabilities and recommend best options.""")
    
    user_message = HumanMessage(content=f"""Match capabilities:

Required: {', '.join(required_capabilities)}
Available Tools: {len(available_tools)}

Find best matches.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Match tools to capabilities
    discovered_tools = []
    
    for req_cap in required_capabilities:
        matching_tools = [
            tool for tool in available_tools
            if req_cap in tool["capabilities"]
        ]
        
        for tool in matching_tools:
            if tool not in discovered_tools:
                discovered_tools.append(tool)
    
    # Recommend tools based on priority and reliability
    recommended_tools = sorted(
        discovered_tools,
        key=lambda t: (t["priority"], t["reliability"]),
        reverse=True
    )[:3]  # Top 3 recommendations
    
    summary = f"""
    📊 TOOL DISCOVERY COMPLETE
    
    Discovery Summary:
    • Available Tools: {len(available_tools)}
    • Required Capabilities: {len(required_capabilities)}
    • Matching Tools: {len(discovered_tools)}
    • Recommended: {len(recommended_tools)}
    
    Required Capabilities:
    {chr(10).join(f"  • {cap}" for cap in required_capabilities)}
    
    Discovered Matches:
    {chr(10).join(f"  • {tool['name']}: {', '.join(c for c in tool['capabilities'] if c in required_capabilities)}" for tool in discovered_tools)}
    
    Top Recommendations:
    {chr(10).join(f"  {i+1}. {tool['name']} (Priority: {tool['priority']}, Reliability: {tool['reliability']:.0%}, Latency: {tool['latency_ms']}ms)" for i, tool in enumerate(recommended_tools))}
    
    Tool Discovery Pattern Process:
    1. Discovery Agent → Find available tools
    2. Capability Matcher → Match to requirements
    
    Advanced Matching Algorithms:
    
    Weighted Scoring:
    ```python
    def score_tool(tool, requirements, weights):
        score = 0
        
        # Capability match (40%)
        matched_caps = sum(
            1 for cap in requirements 
            if cap in tool["capabilities"]
        )
        cap_score = (matched_caps / len(requirements)) * weights["capabilities"]
        
        # Reliability (30%)
        rel_score = tool["reliability"] * weights["reliability"]
        
        # Performance (20%)
        perf_score = (1 - (tool["latency_ms"] / 1000)) * weights["performance"]
        
        # Priority (10%)
        pri_score = (tool["priority"] / 10) * weights["priority"]
        
        score = cap_score + rel_score + perf_score + pri_score
        return score
    
    weights = {{
        "capabilities": 0.4,
        "reliability": 0.3,
        "performance": 0.2,
        "priority": 0.1
    }}
    ```
    
    Semantic Similarity:
    ```python
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    class SemanticMatcher:
        def __init__(self):
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def match_capabilities(self, required, available_tools):
            # Encode required capabilities
            req_embeddings = self.model.encode(required)
            
            matches = []
            for tool in available_tools:
                # Encode tool capabilities
                tool_embeddings = self.model.encode(
                    tool["capabilities"]
                )
                
                # Compute similarity
                similarities = np.dot(
                    req_embeddings, 
                    tool_embeddings.T
                )
                
                max_similarity = similarities.max(axis=1).mean()
                
                if max_similarity > 0.7:  # Threshold
                    matches.append((tool, max_similarity))
            
            # Sort by similarity
            return sorted(matches, key=lambda x: x[1], reverse=True)
    ```
    
    Multi-Criteria Decision Making:
    ```python
    import numpy as np
    
    def topsis_ranking(tools, criteria, weights):
        # Convert to decision matrix
        matrix = np.array([
            [t[c] for c in criteria] 
            for t in tools
        ])
        
        # Normalize
        norm_matrix = matrix / np.sqrt((matrix ** 2).sum(axis=0))
        
        # Apply weights
        weighted_matrix = norm_matrix * weights
        
        # Ideal solutions
        ideal_best = weighted_matrix.max(axis=0)
        ideal_worst = weighted_matrix.min(axis=0)
        
        # Calculate distances
        dist_best = np.sqrt(
            ((weighted_matrix - ideal_best) ** 2).sum(axis=1)
        )
        dist_worst = np.sqrt(
            ((weighted_matrix - ideal_worst) ** 2).sum(axis=1)
        )
        
        # Calculate scores
        scores = dist_worst / (dist_best + dist_worst)
        
        # Rank tools
        ranked_indices = scores.argsort()[::-1]
        return [tools[i] for i in ranked_indices]
    ```
    
    Best Practices:
    
    Discovery:
    • Cache discovered tools
    • Periodic refresh
    • Health checks
    • Fallback mechanisms
    
    Matching:
    • Consider latency
    • Check reliability
    • Verify availability
    • Test compatibility
    
    Recommendation:
    • Score transparently
    • Allow overrides
    • Log decisions
    • Monitor performance
    
    Key Insight:
    Intelligent tool discovery with capability-based
    matching enables dynamic tool selection and
    optimal resource utilization in multi-agent systems.
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Capability Matcher:\n{response.content}\n{summary}")],
        "discovered_tools": discovered_tools,
        "recommended_tools": [t["name"] for t in recommended_tools]
    }


# Build the graph
def build_tool_discovery_graph():
    """Build the tool discovery pattern graph"""
    workflow = StateGraph(ToolDiscoveryState)
    
    workflow.add_node("discovery_agent", tool_discovery_agent)
    workflow.add_node("capability_matcher", capability_matcher)
    
    workflow.add_edge(START, "discovery_agent")
    workflow.add_edge("discovery_agent", "capability_matcher")
    workflow.add_edge("capability_matcher", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_tool_discovery_graph()
    
    print("=== Tool Discovery MCP Pattern ===\n")
    
    # Test Case: Discover tools and match capabilities
    print("\n" + "="*70)
    print("TEST CASE: Capability-Based Tool Discovery")
    print("="*70)
    
    state = {
        "messages": [],
        "available_tools": [],
        "required_capabilities": ["search_web", "summarize"],
        "discovered_tools": [],
        "recommended_tools": [],
        "discovery_method": "registry"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nDiscovery Results:")
    print(f"Available: {len(result.get('available_tools', []))}")
    print(f"Discovered: {len(result.get('discovered_tools', []))}")
    print(f"Recommended: {len(result.get('recommended_tools', []))}")

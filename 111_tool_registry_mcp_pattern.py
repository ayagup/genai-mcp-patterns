"""
Tool Registry MCP Pattern

This pattern implements a centralized tool registry for managing,
discovering, and accessing tools in a multi-agent system.

Key Features:
- Tool registration and metadata management
- Tool lookup by name, category, or capability
- Version tracking and compatibility
- Dynamic tool loading
- Tool lifecycle management
"""

from typing import TypedDict, Sequence, Annotated, List, Dict, Any, Optional
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ToolRegistryState(TypedDict):
    """State for tool registry pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    registry: Dict[str, Dict]  # {tool_name: {metadata, version, capabilities}}
    registered_tools: List[str]
    categories: List[str]
    query_type: str  # "name", "category", "capability"
    query_value: str
    search_results: List[Dict]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Registry Manager
def registry_manager(state: ToolRegistryState) -> ToolRegistryState:
    """Manages tool registration and metadata"""
    registry = state.get("registry", {})
    
    system_message = SystemMessage(content="""You are a tool registry manager.
    Maintain a comprehensive registry of tools with metadata and capabilities.""")
    
    user_message = HumanMessage(content=f"""Manage tool registry:

Current Tools: {len(registry) if registry else 'None registered'}

Initialize and populate registry.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize registry if empty
    if not registry:
        registry = {
            "calculator": {
                "name": "calculator",
                "version": "1.0.0",
                "category": "math",
                "description": "Performs mathematical calculations",
                "capabilities": ["add", "subtract", "multiply", "divide"],
                "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
                "output_schema": {"type": "number"},
                "author": "System",
                "status": "active"
            },
            "web_search": {
                "name": "web_search",
                "version": "2.1.0",
                "category": "information_retrieval",
                "description": "Searches the web for information",
                "capabilities": ["search", "fetch", "scrape"],
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
                "output_schema": {"type": "array", "items": {"type": "object"}},
                "author": "External",
                "status": "active"
            },
            "database_query": {
                "name": "database_query",
                "version": "1.5.2",
                "category": "data_access",
                "description": "Queries database for data",
                "capabilities": ["select", "insert", "update", "delete"],
                "input_schema": {"type": "object", "properties": {"sql": {"type": "string"}}},
                "output_schema": {"type": "array", "items": {"type": "object"}},
                "author": "Internal",
                "status": "active"
            },
            "text_analyzer": {
                "name": "text_analyzer",
                "version": "1.0.0",
                "category": "nlp",
                "description": "Analyzes text for sentiment, entities, keywords",
                "capabilities": ["sentiment", "ner", "keywords", "summarize"],
                "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}},
                "output_schema": {"type": "object"},
                "author": "ML Team",
                "status": "active"
            },
            "image_processor": {
                "name": "image_processor",
                "version": "2.0.0",
                "category": "computer_vision",
                "description": "Processes and analyzes images",
                "capabilities": ["resize", "crop", "detect_objects", "classify"],
                "input_schema": {"type": "object", "properties": {"image_url": {"type": "string"}}},
                "output_schema": {"type": "object"},
                "author": "Vision Team",
                "status": "active"
            },
            "email_sender": {
                "name": "email_sender",
                "version": "1.2.0",
                "category": "communication",
                "description": "Sends emails to recipients",
                "capabilities": ["send", "schedule", "template"],
                "input_schema": {"type": "object", "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
                "author": "Comms Team",
                "status": "active"
            }
        }
    
    registered_tools = list(registry.keys())
    categories = list(set(tool["category"] for tool in registry.values()))
    
    report = f"""
    📚 Tool Registry Manager:
    
    Registry Overview:
    • Registered Tools: {len(registry)}
    • Categories: {len(categories)}
    • Active Tools: {sum(1 for t in registry.values() if t['status'] == 'active')}
    
    Tool Registry Concepts:
    
    Registry Pattern Benefits:
    
    Centralization:
    • Single source of truth
    • Consistent metadata
    • Easy discovery
    • Version management
    • Lifecycle control
    
    Discoverability:
    • Search by name
    • Filter by category
    • Match by capability
    • Version compatibility
    • Status filtering
    
    Metadata Management:
    • Tool description
    • Input/output schemas
    • Version information
    • Capabilities list
    • Author/ownership
    • Status tracking
    
    Tool Registration Process:
    
    1. Tool Definition:
    ```python
    tool_metadata = {{
        "name": "calculator",
        "version": "1.0.0",
        "category": "math",
        "description": "Performs calculations",
        "capabilities": ["add", "subtract", "multiply", "divide"],
        "input_schema": {{
            "type": "object",
            "properties": {{
                "expression": {{"type": "string"}}
            }},
            "required": ["expression"]
        }},
        "output_schema": {{
            "type": "number"
        }},
        "implementation": calculator_function,
        "author": "Math Team",
        "status": "active"
    }}
    ```
    
    2. Registration:
    ```python
    class ToolRegistry:
        def __init__(self):
            self.tools = {{}}
        
        def register(self, tool_metadata):
            # Validate metadata
            self._validate_metadata(tool_metadata)
            
            # Check for conflicts
            if tool_metadata["name"] in self.tools:
                raise ValueError(f"Tool {{tool_metadata['name']}} already registered")
            
            # Register tool
            self.tools[tool_metadata["name"]] = tool_metadata
            
            print(f"✓ Registered: {{tool_metadata['name']}} v{{tool_metadata['version']}}")
        
        def _validate_metadata(self, metadata):
            required = ["name", "version", "category", "capabilities"]
            for field in required:
                if field not in metadata:
                    raise ValueError(f"Missing required field: {{field}}")
    ```
    
    3. Lookup:
    ```python
    def get_tool(self, name):
        if name not in self.tools:
            raise KeyError(f"Tool {{name}} not found")
        return self.tools[name]
    
    def find_by_category(self, category):
        return [t for t in self.tools.values() 
                if t["category"] == category]
    
    def find_by_capability(self, capability):
        return [t for t in self.tools.values() 
                if capability in t["capabilities"]]
    ```
    
    Tool Categories in Registry:
    
    {chr(10).join(f"• {cat.replace('_', ' ').title()}: {sum(1 for t in registry.values() if t['category'] == cat)} tools" for cat in sorted(categories))}
    
    Registered Tools:
    {chr(10).join(f"• {tool['name']} v{tool['version']} ({tool['category']}) - {len(tool['capabilities'])} capabilities" for tool in registry.values())}
    
    Advanced Registry Features:
    
    Versioning:
    • Semantic versioning
    • Backward compatibility
    • Deprecation notices
    • Migration guides
    • Breaking changes
    
    Schema Validation:
    • JSON Schema
    • Pydantic models
    • Type checking
    • Required fields
    • Default values
    
    Dependency Management:
    • Tool dependencies
    • Version constraints
    • Conflict resolution
    • Installation tracking
    
    Access Control:
    • Permission levels
    • Role-based access
    • API keys
    • Usage quotas
    • Rate limiting
    
    Registry Implementation Patterns:
    
    In-Memory Registry:
    ```python
    class InMemoryRegistry:
        def __init__(self):
            self._tools = {{}}
            self._index_by_category = {{}}
            self._index_by_capability = {{}}
        
        def register(self, tool):
            self._tools[tool["name"]] = tool
            
            # Build indexes
            cat = tool["category"]
            if cat not in self._index_by_category:
                self._index_by_category[cat] = []
            self._index_by_category[cat].append(tool["name"])
            
            for cap in tool["capabilities"]:
                if cap not in self._index_by_capability:
                    self._index_by_capability[cap] = []
                self._index_by_capability[cap].append(tool["name"])
    ```
    
    Persistent Registry (Database):
    ```python
    from sqlalchemy import Column, String, JSON
    from sqlalchemy.ext.declarative import declarative_base
    
    Base = declarative_base()
    
    class Tool(Base):
        __tablename__ = 'tools'
        
        name = Column(String, primary_key=True)
        version = Column(String)
        category = Column(String, index=True)
        description = Column(String)
        capabilities = Column(JSON)
        input_schema = Column(JSON)
        output_schema = Column(JSON)
        metadata = Column(JSON)
    
    class DatabaseRegistry:
        def __init__(self, session):
            self.session = session
        
        def register(self, tool_data):
            tool = Tool(**tool_data)
            self.session.add(tool)
            self.session.commit()
        
        def find_by_category(self, category):
            return self.session.query(Tool).filter_by(
                category=category
            ).all()
    ```
    
    Distributed Registry (Service):
    ```python
    import requests
    
    class RemoteRegistry:
        def __init__(self, registry_url):
            self.url = registry_url
        
        def register(self, tool):
            response = requests.post(
                f"{{self.url}}/tools",
                json=tool
            )
            return response.json()
        
        def get_tool(self, name):
            response = requests.get(
                f"{{self.url}}/tools/{{name}}"
            )
            return response.json()
        
        def search(self, **filters):
            response = requests.get(
                f"{{self.url}}/tools/search",
                params=filters
            )
            return response.json()
    ```
    
    LangChain Tool Registry:
    ```python
    from langchain.tools import BaseTool
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    
    class ToolRegistryManager:
        def __init__(self):
            self.tools = {{}}
        
        def register_langchain_tool(self, tool: BaseTool):
            self.tools[tool.name] = {{
                "tool": tool,
                "name": tool.name,
                "description": tool.description,
                "schema": tool.args_schema
            }}
        
        def get_tools_list(self):
            return [t["tool"] for t in self.tools.values()]
        
        def create_agent(self, llm):
            tools = self.get_tools_list()
            return create_openai_functions_agent(llm, tools)
    ```
    """
    
    return {
        "messages": [AIMessage(content=f"📚 Registry Manager:\n{response.content}\n{report}")],
        "registry": registry,
        "registered_tools": registered_tools,
        "categories": categories
    }


# Tool Searcher
def tool_searcher(state: ToolRegistryState) -> ToolRegistryState:
    """Searches registry for tools matching criteria"""
    registry = state.get("registry", {})
    query_type = state.get("query_type", "capability")
    query_value = state.get("query_value", "search")
    
    system_message = SystemMessage(content="""You are a tool searcher.
    Find tools in the registry matching search criteria.""")
    
    user_message = HumanMessage(content=f"""Search registry:

Query Type: {query_type}
Query Value: {query_value}
Total Tools: {len(registry)}

Find matching tools.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Search based on query type
    search_results = []
    
    if query_type == "name":
        if query_value in registry:
            search_results.append(registry[query_value])
    elif query_type == "category":
        search_results = [tool for tool in registry.values() if tool["category"] == query_value]
    elif query_type == "capability":
        search_results = [tool for tool in registry.values() if query_value in tool["capabilities"]]
    
    summary = f"""
    📊 TOOL REGISTRY COMPLETE
    
    Registry Summary:
    • Total Registered: {len(registry)}
    • Categories: {len(state.get('categories', []))}
    • Search Query: {query_type}={query_value}
    • Results Found: {len(search_results)}
    
    Search Results:
    {chr(10).join(f"  • {tool['name']} v{tool['version']} - {tool['description']}" for tool in search_results) if search_results else "  • No tools found matching criteria"}
    
    Tool Registry Pattern Process:
    1. Registry Manager → Register and maintain tools
    2. Tool Searcher → Find tools by criteria
    
    Registry Query Examples:
    
    By Name:
    ```python
    tool = registry.get_tool("calculator")
    print(f"Found: {{tool['name']}} - {{tool['description']}}")
    ```
    
    By Category:
    ```python
    math_tools = registry.find_by_category("math")
    for tool in math_tools:
        print(f"- {{tool['name']}}: {{tool['capabilities']}}")
    ```
    
    By Capability:
    ```python
    search_tools = registry.find_by_capability("search")
    for tool in search_tools:
        print(f"- {{tool['name']}} can search")
    ```
    
    Complex Queries:
    ```python
    # Find tools with multiple capabilities
    multi_capable = [
        t for t in registry.tools.values()
        if len(t["capabilities"]) > 3
    ]
    
    # Find active tools by version
    latest_tools = {{}}
    for tool in registry.tools.values():
        name = tool["name"]
        if (name not in latest_tools or 
            tool["version"] > latest_tools[name]["version"]):
            latest_tools[name] = tool
    
    # Find deprecated tools
    deprecated = [
        t for t in registry.tools.values()
        if t.get("status") == "deprecated"
    ]
    ```
    
    Best Practices:
    
    Registration:
    • Validate all metadata
    • Check version conflicts
    • Verify schemas
    • Test tool function
    • Document clearly
    
    Discovery:
    • Index by category
    • Index by capability
    • Cache search results
    • Provide suggestions
    • Fast lookup
    
    Maintenance:
    • Regular audits
    • Remove unused tools
    • Update versions
    • Monitor usage
    • Deprecation policy
    
    Key Insight:
    A well-designed tool registry enables efficient
    tool discovery, version management, and consistent
    metadata across multi-agent systems.
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Tool Searcher:\n{response.content}\n{summary}")],
        "search_results": search_results
    }


# Build the graph
def build_tool_registry_graph():
    """Build the tool registry pattern graph"""
    workflow = StateGraph(ToolRegistryState)
    
    workflow.add_node("registry_manager", registry_manager)
    workflow.add_node("tool_searcher", tool_searcher)
    
    workflow.add_edge(START, "registry_manager")
    workflow.add_edge("registry_manager", "tool_searcher")
    workflow.add_edge("tool_searcher", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_tool_registry_graph()
    
    print("=== Tool Registry MCP Pattern ===\n")
    
    # Test Case: Register tools and search by capability
    print("\n" + "="*70)
    print("TEST CASE: Tool Registry with Capability Search")
    print("="*70)
    
    state = {
        "messages": [],
        "registry": {},
        "registered_tools": [],
        "categories": [],
        "query_type": "capability",
        "query_value": "search",
        "search_results": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nRegistry Statistics:")
    print(f"Total Tools: {len(result.get('registry', {}))}")
    print(f"Categories: {len(result.get('categories', []))}")
    print(f"Search Results: {len(result.get('search_results', []))}")

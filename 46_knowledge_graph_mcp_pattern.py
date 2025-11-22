"""
Knowledge Graph MCP Pattern

This pattern demonstrates using a knowledge graph to represent entities,
relationships, and enabling graph-based reasoning and traversal.

Key Features:
- Entity-relationship modeling
- Graph traversal
- Multi-hop reasoning
- Connected knowledge discovery
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class KnowledgeGraphState(TypedDict):
    """State with knowledge graph"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    knowledge_graph: dict[str, any]  # Graph structure
    query: str
    traversal_path: list[str]
    discovered_facts: list[str]
    answer: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Graph Builder
def graph_builder(state: KnowledgeGraphState) -> KnowledgeGraphState:
    """Builds knowledge graph from information"""
    
    system_message = SystemMessage(content="""You are a knowledge graph builder. Structure 
    information as entities and relationships in a graph.""")
    
    user_message = HumanMessage(content="""Build knowledge graph for tech company domain:

Entities: Companies, Products, People, Technologies
Relationships: works_at, develops, uses, competes_with

Create graph structure.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Build knowledge graph structure
    knowledge_graph = {
        "entities": {
            "Alice": {"type": "Person", "role": "Engineer"},
            "Bob": {"type": "Person", "role": "PM"},
            "TechCorp": {"type": "Company", "industry": "AI"},
            "CloudAI": {"type": "Product", "category": "Platform"},
            "Python": {"type": "Technology", "domain": "Programming"},
            "LangChain": {"type": "Technology", "domain": "LLM Framework"}
        },
        "relationships": [
            {"from": "Alice", "to": "TechCorp", "type": "works_at"},
            {"from": "Bob", "to": "TechCorp", "type": "works_at"},
            {"from": "Alice", "to": "CloudAI", "type": "develops"},
            {"from": "Bob", "to": "CloudAI", "type": "manages"},
            {"from": "CloudAI", "to": "Python", "type": "built_with"},
            {"from": "CloudAI", "to": "LangChain", "type": "uses"},
            {"from": "Alice", "to": "Python", "type": "expert_in"},
            {"from": "Alice", "to": "LangChain", "type": "uses"}
        ]
    }
    
    return {
        "messages": [AIMessage(content=f"🔗 Graph Builder: {response.content}\n\n✅ Knowledge graph built with {len(knowledge_graph['entities'])} entities and {len(knowledge_graph['relationships'])} relationships")],
        "knowledge_graph": knowledge_graph
    }


# Query Analyzer
def query_analyzer(state: KnowledgeGraphState) -> KnowledgeGraphState:
    """Analyzes query to determine graph traversal strategy"""
    query = state.get("query", "")
    knowledge_graph = state.get("knowledge_graph", {})
    
    system_message = SystemMessage(content="""You are a query analyzer. Determine which 
    entities and relationships to explore in the knowledge graph.""")
    
    entities_list = ", ".join(knowledge_graph.get("entities", {}).keys())
    
    user_message = HumanMessage(content=f"""Analyze query: {query}

Available entities: {entities_list}

Determine starting entity and relationships to explore.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🔍 Query Analyzer: {response.content}")]
    }


# Graph Traverser
def graph_traverser(state: KnowledgeGraphState) -> KnowledgeGraphState:
    """Traverses knowledge graph to discover relevant information"""
    query = state.get("query", "")
    knowledge_graph = state.get("knowledge_graph", {})
    
    system_message = SystemMessage(content="""You are a graph traverser. Navigate the 
    knowledge graph to discover relevant facts through entity relationships.""")
    
    user_message = HumanMessage(content=f"""Traverse graph for query: {query}

Follow relationships to discover connected information.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate graph traversal (simplified)
    query_lower = query.lower()
    traversal_path = []
    discovered_facts = []
    
    entities = knowledge_graph.get("entities", {})
    relationships = knowledge_graph.get("relationships", [])
    
    # Find starting entity
    start_entity = None
    if "alice" in query_lower:
        start_entity = "Alice"
    elif "bob" in query_lower:
        start_entity = "Bob"
    elif "cloudai" in query_lower or "product" in query_lower:
        start_entity = "CloudAI"
    
    if start_entity:
        traversal_path.append(start_entity)
        discovered_facts.append(f"{start_entity} is a {entities[start_entity]['type']}")
        
        # Find connected entities (1-hop)
        for rel in relationships:
            if rel["from"] == start_entity:
                traversal_path.append(f"{rel['type']} -> {rel['to']}")
                target_entity = entities.get(rel["to"], {})
                discovered_facts.append(
                    f"{start_entity} {rel['type']} {rel['to']} ({target_entity.get('type', 'unknown')})"
                )
            elif rel["to"] == start_entity:
                traversal_path.append(f"{rel['from']} -> {rel['type']}")
                source_entity = entities.get(rel["from"], {})
                discovered_facts.append(
                    f"{rel['from']} ({source_entity.get('type', 'unknown')}) {rel['type']} {start_entity}"
                )
    
    return {
        "messages": [AIMessage(content=f"🗺️ Graph Traverser: {response.content}\n\n📍 Traversed {len(traversal_path)} nodes")],
        "traversal_path": traversal_path,
        "discovered_facts": discovered_facts
    }


# Fact Connector
def fact_connector(state: KnowledgeGraphState) -> KnowledgeGraphState:
    """Connects discovered facts into coherent insights"""
    discovered_facts = state.get("discovered_facts", [])
    traversal_path = state.get("traversal_path", [])
    
    system_message = SystemMessage(content="""You are a fact connector. Synthesize discovered 
    facts from graph traversal into coherent insights.""")
    
    facts_text = "\n".join([f"- {fact}" for fact in discovered_facts])
    
    user_message = HumanMessage(content=f"""Connect these discovered facts:

{facts_text}

Create coherent insights from graph exploration.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🔗 Fact Connector: {response.content}")]
    }


# Answer Generator
def answer_generator(state: KnowledgeGraphState) -> KnowledgeGraphState:
    """Generates final answer using graph-derived knowledge"""
    query = state.get("query", "")
    discovered_facts = state.get("discovered_facts", [])
    
    system_message = SystemMessage(content="""You are an answer generator. Use facts 
    discovered from knowledge graph to answer the query comprehensively.""")
    
    facts_text = "\n".join(discovered_facts)
    
    user_message = HumanMessage(content=f"""Answer query: {query}

Using these facts from knowledge graph:
{facts_text}

Provide comprehensive answer.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💡 Answer Generator: {response.content}")],
        "answer": response.content
    }


# Graph Monitor
def graph_monitor(state: KnowledgeGraphState) -> KnowledgeGraphState:
    """Monitors knowledge graph usage"""
    knowledge_graph = state.get("knowledge_graph", {})
    traversal_path = state.get("traversal_path", [])
    discovered_facts = state.get("discovered_facts", [])
    
    summary = f"""
    ✅ KNOWLEDGE GRAPH PATTERN COMPLETE
    
    Graph Statistics:
    • Entities: {len(knowledge_graph.get('entities', {}))}
    • Relationships: {len(knowledge_graph.get('relationships', []))}
    • Traversal Hops: {len(traversal_path)}
    • Facts Discovered: {len(discovered_facts)}
    
    Traversal Path:
{chr(10).join([f"    {i+1}. {step}" for i, step in enumerate(traversal_path)])}
    
    Knowledge Graph Benefits:
    • Connected knowledge representation
    • Multi-hop reasoning enabled
    • Relationship discovery
    • Semantic querying
    • Explainable inference paths
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Graph Monitor:\n{summary}")]
    }


# Build the graph
def build_knowledge_graph_pattern():
    """Build the knowledge graph MCP pattern graph"""
    workflow = StateGraph(KnowledgeGraphState)
    
    workflow.add_node("builder", graph_builder)
    workflow.add_node("analyzer", query_analyzer)
    workflow.add_node("traverser", graph_traverser)
    workflow.add_node("connector", fact_connector)
    workflow.add_node("generator", answer_generator)
    workflow.add_node("monitor", graph_monitor)
    
    workflow.add_edge(START, "builder")
    workflow.add_edge("builder", "analyzer")
    workflow.add_edge("analyzer", "traverser")
    workflow.add_edge("traverser", "connector")
    workflow.add_edge("connector", "generator")
    workflow.add_edge("generator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_knowledge_graph_pattern()
    
    print("=== Knowledge Graph MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "knowledge_graph": {},
        "query": "What technologies does Alice use and what product does she work on?",
        "traversal_path": [],
        "discovered_facts": [],
        "answer": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Knowledge Graph Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Answer ===")
    print(result.get("answer", "No answer generated"))

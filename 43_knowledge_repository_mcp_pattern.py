"""
Knowledge Repository MCP Pattern

This pattern demonstrates centralized knowledge storage with versioning,
retrieval, and management capabilities for multi-agent systems.

Key Features:
- Centralized knowledge storage
- Version control
- Query and retrieval operations
- Knowledge categorization
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class KnowledgeRepositoryState(TypedDict):
    """State with knowledge repository"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    repository: dict[str, list[dict]]  # category -> knowledge items
    query: str
    retrieved_knowledge: list[dict]
    final_answer: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Knowledge Ingestion Agent
def knowledge_ingester(state: KnowledgeRepositoryState) -> KnowledgeRepositoryState:
    """Ingests and stores knowledge in repository"""
    repository = state.get("repository", {
        "technical": [],
        "business": [],
        "product": [],
        "customer": []
    })
    
    system_message = SystemMessage(content="""You are a knowledge ingester. Organize and 
    store knowledge items in the appropriate repository categories.""")
    
    user_message = HumanMessage(content="""Ingest these knowledge items:

1. Python best practices for error handling
2. Q4 revenue targets and KPIs
3. Product roadmap for new AI features
4. Customer feedback on mobile app performance

Categorize and store them in the repository.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Store knowledge items
    repository["technical"].append({
        "id": "tech_001",
        "content": "Python best practices: Use try-except blocks, log errors, fail gracefully",
        "version": "1.0",
        "timestamp": "2025-11-09T10:00:00"
    })
    
    repository["business"].append({
        "id": "bus_001",
        "content": "Q4 Targets: $10M revenue, 15% growth, customer satisfaction > 90%",
        "version": "1.0",
        "timestamp": "2025-11-09T10:00:00"
    })
    
    repository["product"].append({
        "id": "prod_001",
        "content": "AI Features Roadmap: NLP improvements, image generation, voice synthesis",
        "version": "1.0",
        "timestamp": "2025-11-09T10:00:00"
    })
    
    repository["customer"].append({
        "id": "cust_001",
        "content": "Mobile App Feedback: Users want faster load times and offline mode",
        "version": "1.0",
        "timestamp": "2025-11-09T10:00:00"
    })
    
    return {
        "messages": [AIMessage(content=f"📥 Knowledge Ingester: {response.content}\n\n✅ 4 knowledge items stored in repository")],
        "repository": repository
    }


# Query Processor
def query_processor(state: KnowledgeRepositoryState) -> KnowledgeRepositoryState:
    """Processes queries and determines search strategy"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a query processor. Analyze queries 
    and determine which repository categories to search.""")
    
    user_message = HumanMessage(content=f"""Process this query: {query}

Determine which categories to search: technical, business, product, or customer.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine categories to search (simple keyword matching)
    categories_to_search = []
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["technical", "code", "python", "error"]):
        categories_to_search.append("technical")
    if any(word in query_lower for word in ["business", "revenue", "kpi", "target"]):
        categories_to_search.append("business")
    if any(word in query_lower for word in ["product", "feature", "roadmap", "ai"]):
        categories_to_search.append("product")
    if any(word in query_lower for word in ["customer", "feedback", "user"]):
        categories_to_search.append("customer")
    
    if not categories_to_search:
        categories_to_search = ["technical", "business", "product", "customer"]
    
    return {
        "messages": [AIMessage(content=f"🔍 Query Processor: {response.content}\n\n📋 Categories to search: {', '.join(categories_to_search)}")]
    }


# Knowledge Retriever
def knowledge_retriever(state: KnowledgeRepositoryState) -> KnowledgeRepositoryState:
    """Retrieves relevant knowledge from repository"""
    query = state.get("query", "")
    repository = state.get("repository", {})
    
    system_message = SystemMessage(content="""You are a knowledge retriever. Search the 
    repository and retrieve relevant knowledge items based on the query.""")
    
    user_message = HumanMessage(content=f"""Retrieve knowledge for query: {query}

Search all repository categories and find relevant items.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Retrieve relevant items (simple retrieval - in practice, use semantic search)
    retrieved_knowledge = []
    query_lower = query.lower()
    
    for category, items in repository.items():
        for item in items:
            content_lower = item["content"].lower()
            # Simple keyword matching (in practice, use embeddings/semantic search)
            if any(word in content_lower for word in query_lower.split()):
                retrieved_knowledge.append({
                    **item,
                    "category": category,
                    "relevance_score": 0.85  # Placeholder
                })
    
    return {
        "messages": [AIMessage(content=f"📚 Knowledge Retriever: {response.content}\n\n✅ Retrieved {len(retrieved_knowledge)} knowledge items")],
        "retrieved_knowledge": retrieved_knowledge
    }


# Answer Generator
def answer_generator(state: KnowledgeRepositoryState) -> KnowledgeRepositoryState:
    """Generates answers using retrieved knowledge"""
    query = state.get("query", "")
    retrieved_knowledge = state.get("retrieved_knowledge", [])
    
    system_message = SystemMessage(content="""You are an answer generator. Use retrieved 
    knowledge from the repository to generate comprehensive answers.""")
    
    knowledge_context = "\n\n".join([
        f"[{item['category']}] {item['content']}"
        for item in retrieved_knowledge
    ])
    
    user_message = HumanMessage(content=f"""Generate an answer for: {query}

Using this knowledge from repository:

{knowledge_context if knowledge_context else 'No relevant knowledge found'}

Provide a comprehensive answer.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💡 Answer Generator: {response.content}")],
        "final_answer": response.content
    }


# Repository Monitor
def repository_monitor(state: KnowledgeRepositoryState) -> KnowledgeRepositoryState:
    """Monitors repository state and statistics"""
    repository = state.get("repository", {})
    retrieved_knowledge = state.get("retrieved_knowledge", [])
    
    total_items = sum(len(items) for items in repository.values())
    
    category_stats = "\n".join([
        f"  {category}: {len(items)} items"
        for category, items in repository.items()
    ])
    
    final_summary = f"""
    ✅ KNOWLEDGE REPOSITORY PATTERN COMPLETE
    
    Repository Statistics:
    • Total Knowledge Items: {total_items}
    • Categories: {len(repository)}
    
    Category Breakdown:
{category_stats}
    
    Query Results:
    • Items Retrieved: {len(retrieved_knowledge)}
    • Answer Generated: Yes
    
    Repository Benefits:
    • Centralized knowledge management
    • Organized categorization
    • Version control support
    • Efficient retrieval
    • Scalable storage
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Repository Monitor:\n{final_summary}")]
    }


# Build the graph
def build_knowledge_repository_graph():
    """Build the knowledge repository MCP pattern graph"""
    workflow = StateGraph(KnowledgeRepositoryState)
    
    workflow.add_node("ingester", knowledge_ingester)
    workflow.add_node("query_processor", query_processor)
    workflow.add_node("retriever", knowledge_retriever)
    workflow.add_node("generator", answer_generator)
    workflow.add_node("monitor", repository_monitor)
    
    # First ingest knowledge, then process query
    workflow.add_edge(START, "ingester")
    workflow.add_edge("ingester", "query_processor")
    workflow.add_edge("query_processor", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_knowledge_repository_graph()
    
    print("=== Knowledge Repository MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "repository": {
            "technical": [],
            "business": [],
            "product": [],
            "customer": []
        },
        "query": "What are the AI features in our product roadmap and how do customers feel about our mobile app?",
        "retrieved_knowledge": [],
        "final_answer": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Knowledge Repository Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Answer ===")
    print(result.get("final_answer", "No answer generated"))

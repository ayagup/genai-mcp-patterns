"""
Embedding-Based Retrieval MCP Pattern

This pattern demonstrates semantic similarity search using embeddings
in multi-agent systems.

Key Features:
- Semantic similarity matching
- Embedding-based ranking
- Multi-modal retrieval
- Contextual relevance scoring
"""

from typing import TypedDict, Sequence, Annotated
import operator
import math
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class EmbeddingRetrievalState(TypedDict):
    """State for embedding-based retrieval"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    knowledge_base: list[dict]  # Documents with embeddings
    query: str
    query_embedding: list[float]
    candidates: list[dict]  # Initial candidates
    reranked_results: list[dict]  # Reranked by relevance
    final_results: list[dict]  # Top-k results
    context_summary: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Embedding utilities
def generate_embedding(text: str) -> list[float]:
    """Generate embedding (simplified for demonstration)"""
    words = text.lower().split()
    embedding = [0.0] * 10
    for word in words[:10]:
        for i, char in enumerate(word[:10]):
            embedding[i] += ord(char) / 1000.0
    # Normalize
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    return embedding


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity"""
    return sum(a * b for a, b in zip(vec1, vec2))


# Knowledge Base Indexer
def knowledge_indexer(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Indexes knowledge base with embeddings"""
    
    system_message = SystemMessage(content="""You are a knowledge indexer. Create semantic 
    embeddings for all documents in the knowledge base.""")
    
    user_message = HumanMessage(content="""Index these documents with semantic embeddings:

Technical Docs:
- "FastAPI is a modern Python web framework for building APIs"
- "Docker containers provide isolated runtime environments"
- "Kubernetes orchestrates containerized applications at scale"

Business Docs:
- "Market analysis shows growing demand for AI solutions"
- "Customer retention strategies improve long-term revenue"
- "Product roadmap prioritizes user feedback and metrics"

Research Docs:
- "Neural networks learn patterns from large datasets"
- "Transfer learning applies pre-trained models to new tasks"
- "Reinforcement learning optimizes decision-making policies"

Create indexed knowledge base.""")
    
    response = llm.invoke([system_message, user_message])
    
    documents = [
        {"text": "FastAPI is a modern Python web framework for building APIs", "category": "technical", "priority": "high"},
        {"text": "Docker containers provide isolated runtime environments", "category": "technical", "priority": "medium"},
        {"text": "Kubernetes orchestrates containerized applications at scale", "category": "technical", "priority": "medium"},
        {"text": "Market analysis shows growing demand for AI solutions", "category": "business", "priority": "high"},
        {"text": "Customer retention strategies improve long-term revenue", "category": "business", "priority": "high"},
        {"text": "Product roadmap prioritizes user feedback and metrics", "category": "business", "priority": "medium"},
        {"text": "Neural networks learn patterns from large datasets", "category": "research", "priority": "high"},
        {"text": "Transfer learning applies pre-trained models to new tasks", "category": "research", "priority": "medium"},
        {"text": "Reinforcement learning optimizes decision-making policies", "category": "research", "priority": "medium"}
    ]
    
    knowledge_base = []
    for i, doc in enumerate(documents):
        knowledge_base.append({
            "id": f"kb_{i+1}",
            "text": doc["text"],
            "embedding": generate_embedding(doc["text"]),
            "category": doc["category"],
            "priority": doc["priority"]
        })
    
    return {
        "messages": [AIMessage(content=f"📚 Knowledge Indexer: {response.content}\n\n✅ Indexed {len(knowledge_base)} documents across 3 categories")],
        "knowledge_base": knowledge_base
    }


# Query Analyzer
def query_analyzer(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Analyzes query and generates embedding"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a query analyzer. Understand the query 
    intent and generate semantic embedding for retrieval.""")
    
    user_message = HumanMessage(content=f"""Analyze query: "{query}"

Generate semantic embedding for retrieval.""")
    
    response = llm.invoke([system_message, user_message])
    
    query_embedding = generate_embedding(query)
    
    return {
        "messages": [AIMessage(content=f"🔍 Query Analyzer: {response.content}\n\n✅ Generated query embedding")],
        "query_embedding": query_embedding
    }


# Candidate Retriever
def candidate_retriever(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Retrieves candidate documents based on similarity"""
    query_embedding = state.get("query_embedding", [])
    knowledge_base = state.get("knowledge_base", [])
    
    system_message = SystemMessage(content="""You are a candidate retriever. Find documents 
    semantically similar to the query using embedding similarity.""")
    
    user_message = HumanMessage(content=f"""Retrieve candidate documents from knowledge base.

Knowledge base: {len(knowledge_base)} documents

Find top 5 candidates based on semantic similarity.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate similarities
    candidates = []
    for doc in knowledge_base:
        similarity = cosine_similarity(query_embedding, doc["embedding"])
        candidates.append({
            **doc,
            "similarity": similarity
        })
    
    # Sort by similarity and take top 5
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    candidates = candidates[:5]
    
    return {
        "messages": [AIMessage(content=f"📥 Candidate Retriever: {response.content}\n\n✅ Retrieved {len(candidates)} candidate documents")],
        "candidates": candidates
    }


# Relevance Reranker
def relevance_reranker(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Reranks candidates based on contextual relevance"""
    query = state.get("query", "")
    candidates = state.get("candidates", [])
    
    system_message = SystemMessage(content="""You are a relevance reranker. Analyze candidates 
    and rerank based on contextual relevance to the query, considering priority and category.""")
    
    candidate_list = "\n".join([
        f"{i+1}. [{c['category']}|{c['priority']}] {c['text'][:60]}... (sim: {c['similarity']:.3f})"
        for i, c in enumerate(candidates)
    ])
    
    user_message = HumanMessage(content=f"""Rerank candidates for query: "{query}"

Candidates:
{candidate_list}

Apply contextual relevance scoring.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Rerank with combined score (similarity + priority boost)
    priority_boost = {"high": 0.15, "medium": 0.05, "low": 0.0}
    
    reranked = []
    for doc in candidates:
        relevance_score = doc["similarity"] + priority_boost.get(doc["priority"], 0)
        reranked.append({
            **doc,
            "relevance_score": relevance_score
        })
    
    reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return {
        "messages": [AIMessage(content=f"🎯 Relevance Reranker: {response.content}\n\n✅ Reranked {len(reranked)} documents by contextual relevance")],
        "reranked_results": reranked
    }


# Result Selector
def result_selector(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Selects top-k most relevant results"""
    reranked_results = state.get("reranked_results", [])
    
    system_message = SystemMessage(content="""You are a result selector. Choose the most 
    relevant documents for the final result set.""")
    
    user_message = HumanMessage(content=f"""Select top 3 most relevant documents from {len(reranked_results)} reranked results.

Apply final selection criteria.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Select top 3
    final_results = reranked_results[:3]
    
    return {
        "messages": [AIMessage(content=f"✅ Result Selector: {response.content}\n\n✅ Selected {len(final_results)} final results")],
        "final_results": final_results
    }


# Context Summarizer
def context_summarizer(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Summarizes retrieved context"""
    query = state.get("query", "")
    final_results = state.get("final_results", [])
    
    system_message = SystemMessage(content="""You are a context summarizer. Create a 
    comprehensive summary of retrieved documents relevant to the query.""")
    
    context_docs = "\n\n".join([
        f"Document {i+1} [{doc['category']}|{doc['priority']}]:\n{doc['text']}\n(Relevance: {doc['relevance_score']:.3f})"
        for i, doc in enumerate(final_results)
    ])
    
    user_message = HumanMessage(content=f"""Summarize retrieved context for query: "{query}"

Retrieved Documents:
{context_docs}

Create comprehensive summary.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📝 Context Summarizer: {response.content}")],
        "context_summary": response.content
    }


# Retrieval Monitor
def retrieval_monitor(state: EmbeddingRetrievalState) -> EmbeddingRetrievalState:
    """Monitors retrieval pipeline statistics"""
    knowledge_base = state.get("knowledge_base", [])
    candidates = state.get("candidates", [])
    final_results = state.get("final_results", [])
    
    results_info = "\n".join([
        f"  {i+1}. [{r['category']}|{r['priority']}] Relevance: {r['relevance_score']:.3f}\n     {r['text'][:70]}..."
        for i, r in enumerate(final_results)
    ])
    
    summary = f"""
    ✅ EMBEDDING-BASED RETRIEVAL COMPLETE
    
    Pipeline Statistics:
    • Knowledge Base Size: {len(knowledge_base)}
    • Candidates Retrieved: {len(candidates)}
    • Final Results: {len(final_results)}
    
    Top Results:
{results_info}
    
    Retrieval Features:
    • Semantic similarity matching
    • Contextual relevance scoring
    • Priority-based boosting
    • Multi-stage ranking
    • Category-aware retrieval
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Retrieval Monitor:\n{summary}")]
    }


# Build the graph
def build_embedding_retrieval_graph():
    """Build the embedding-based retrieval pattern graph"""
    workflow = StateGraph(EmbeddingRetrievalState)
    
    workflow.add_node("indexer", knowledge_indexer)
    workflow.add_node("analyzer", query_analyzer)
    workflow.add_node("retriever", candidate_retriever)
    workflow.add_node("reranker", relevance_reranker)
    workflow.add_node("selector", result_selector)
    workflow.add_node("summarizer", context_summarizer)
    workflow.add_node("monitor", retrieval_monitor)
    
    workflow.add_edge(START, "indexer")
    workflow.add_edge("indexer", "analyzer")
    workflow.add_edge("analyzer", "retriever")
    workflow.add_edge("retriever", "reranker")
    workflow.add_edge("reranker", "selector")
    workflow.add_edge("selector", "summarizer")
    workflow.add_edge("summarizer", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_embedding_retrieval_graph()
    
    print("=== Embedding-Based Retrieval MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "knowledge_base": [],
        "query": "What machine learning techniques are available?",
        "query_embedding": [],
        "candidates": [],
        "reranked_results": [],
        "final_results": [],
        "context_summary": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Retrieval Pipeline Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Context Summary ===")
    print(result.get("context_summary", "No summary generated"))

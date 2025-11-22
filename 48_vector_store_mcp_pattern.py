"""
Vector Store MCP Pattern

This pattern demonstrates using vector embeddings for semantic knowledge storage
and retrieval in multi-agent systems.

Key Features:
- Embedding-based storage
- Semantic search capabilities
- Similarity-based retrieval
- Vector indexing
"""

from typing import TypedDict, Sequence, Annotated
import operator
import math
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class VectorStoreState(TypedDict):
    """State with vector store"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    vector_store: list[dict]  # List of {id, text, embedding, metadata}
    query: str
    query_embedding: list[float]
    retrieved_docs: list[dict]
    final_answer: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Simple embedding function (placeholder - in practice use actual embeddings)
def simple_embedding(text: str) -> list[float]:
    """Generate simple embedding (placeholder)"""
    # In practice, use OpenAI embeddings or similar
    # This is a simplified version for demonstration
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
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    return dot_product


# Document Ingestion Agent
def document_ingester(state: VectorStoreState) -> VectorStoreState:
    """Ingests documents and stores them as vectors"""
    
    system_message = SystemMessage(content="""You are a document ingester. Convert documents 
    to vector embeddings and store them in the vector database.""")
    
    user_message = HumanMessage(content="""Ingest these documents into vector store:

1. "Python is a versatile programming language for AI and data science"
2. "LangChain enables building applications with large language models"
3. "Vector databases store embeddings for semantic search"
4. "Machine learning models can understand natural language"
5. "AI agents can collaborate to solve complex problems"

Generate embeddings and store.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create vector store with embeddings
    documents = [
        "Python is a versatile programming language for AI and data science",
        "LangChain enables building applications with large language models",
        "Vector databases store embeddings for semantic search",
        "Machine learning models can understand natural language",
        "AI agents can collaborate to solve complex problems"
    ]
    
    vector_store = []
    for i, doc in enumerate(documents):
        vector_store.append({
            "id": f"doc_{i+1}",
            "text": doc,
            "embedding": simple_embedding(doc),
            "metadata": {"source": "knowledge_base", "index": i}
        })
    
    return {
        "messages": [AIMessage(content=f"📥 Document Ingester: {response.content}\n\n✅ Ingested {len(vector_store)} documents with embeddings")],
        "vector_store": vector_store
    }


# Query Embedding Agent
def query_embedder(state: VectorStoreState) -> VectorStoreState:
    """Converts query to vector embedding"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a query embedder. Convert the user 
    query to a vector embedding for semantic search.""")
    
    user_message = HumanMessage(content=f"""Convert query to embedding: {query}

Generate query vector.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate query embedding
    query_embedding = simple_embedding(query)
    
    return {
        "messages": [AIMessage(content=f"🔢 Query Embedder: {response.content}\n\n✅ Generated query embedding")],
        "query_embedding": query_embedding
    }


# Similarity Search Agent
def similarity_searcher(state: VectorStoreState) -> VectorStoreState:
    """Performs similarity search in vector store"""
    query_embedding = state.get("query_embedding", [])
    vector_store = state.get("vector_store", [])
    
    system_message = SystemMessage(content="""You are a similarity searcher. Find documents 
    in vector store that are semantically similar to the query.""")
    
    user_message = HumanMessage(content=f"""Search vector store for documents similar to query embedding.

Vector store size: {len(vector_store)} documents

Find top 3 most similar documents.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate similarities and rank
    similarities = []
    for doc in vector_store:
        sim = cosine_similarity(query_embedding, doc["embedding"])
        similarities.append({
            **doc,
            "similarity_score": sim
        })
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
    
    # Get top 3
    retrieved_docs = similarities[:3]
    
    return {
        "messages": [AIMessage(content=f"🔍 Similarity Searcher: {response.content}\n\n✅ Retrieved {len(retrieved_docs)} similar documents")],
        "retrieved_docs": retrieved_docs
    }


# Answer Generator with Retrieved Context
def answer_generator(state: VectorStoreState) -> VectorStoreState:
    """Generates answer using retrieved documents"""
    query = state.get("query", "")
    retrieved_docs = state.get("retrieved_docs", [])
    
    system_message = SystemMessage(content="""You are an answer generator. Use retrieved 
    documents from vector store to answer the query comprehensively.""")
    
    context = "\n\n".join([
        f"[Similarity: {doc['similarity_score']:.3f}] {doc['text']}"
        for doc in retrieved_docs
    ])
    
    user_message = HumanMessage(content=f"""Answer query: {query}

Using these semantically similar documents:

{context}

Provide comprehensive answer.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💡 Answer Generator: {response.content}")],
        "final_answer": response.content
    }


# Vector Store Monitor
def vector_store_monitor(state: VectorStoreState) -> VectorStoreState:
    """Monitors vector store statistics"""
    vector_store = state.get("vector_store", [])
    retrieved_docs = state.get("retrieved_docs", [])
    
    retrieval_info = "\n".join([
        f"  {i+1}. [{doc['id']}] Similarity: {doc['similarity_score']:.3f}\n     {doc['text'][:80]}..."
        for i, doc in enumerate(retrieved_docs)
    ])
    
    summary = f"""
    ✅ VECTOR STORE PATTERN COMPLETE
    
    Vector Store Statistics:
    • Total Documents: {len(vector_store)}
    • Embedding Dimension: {len(vector_store[0]['embedding']) if vector_store else 0}
    • Retrieved Documents: {len(retrieved_docs)}
    
    Top Retrieved Documents:
{retrieval_info}
    
    Vector Store Benefits:
    • Semantic search capabilities
    • Similarity-based retrieval
    • No keyword matching needed
    • Understands meaning and context
    • Scalable to large document collections
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Vector Store Monitor:\n{summary}")]
    }


# Build the graph
def build_vector_store_graph():
    """Build the vector store MCP pattern graph"""
    workflow = StateGraph(VectorStoreState)
    
    workflow.add_node("ingester", document_ingester)
    workflow.add_node("embedder", query_embedder)
    workflow.add_node("searcher", similarity_searcher)
    workflow.add_node("generator", answer_generator)
    workflow.add_node("monitor", vector_store_monitor)
    
    workflow.add_edge(START, "ingester")
    workflow.add_edge("ingester", "embedder")
    workflow.add_edge("embedder", "searcher")
    workflow.add_edge("searcher", "generator")
    workflow.add_edge("generator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_vector_store_graph()
    
    print("=== Vector Store MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "vector_store": [],
        "query": "How can AI systems work together effectively?",
        "query_embedding": [],
        "retrieved_docs": [],
        "final_answer": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Vector Store Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Answer ===")
    print(result.get("final_answer", "No answer generated"))

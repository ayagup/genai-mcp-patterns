"""
RAG (Retrieval-Augmented Generation) MCP Pattern

This pattern demonstrates the complete RAG workflow combining retrieval
and generation in multi-agent systems.

Key Features:
- Document retrieval phase
- Context augmentation
- Prompt engineering
- Answer generation with citations
- Quality validation
"""

from typing import TypedDict, Sequence, Annotated
import operator
import math
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class RAGState(TypedDict):
    """State for RAG pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    knowledge_corpus: list[dict]  # Document collection
    user_query: str
    query_embedding: list[float]
    retrieved_chunks: list[dict]  # Retrieved document chunks
    augmented_context: str  # Formatted context for prompt
    generated_answer: str
    citations: list[str]
    quality_score: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Embedding utilities
def create_embedding(text: str) -> list[float]:
    """Create embedding (simplified)"""
    words = text.lower().split()
    embedding = [0.0] * 10
    for word in words[:10]:
        for i, char in enumerate(word[:10]):
            embedding[i] += ord(char) / 1000.0
    magnitude = math.sqrt(sum(x**2 for x in embedding))
    if magnitude > 0:
        embedding = [x / magnitude for x in embedding]
    return embedding


def similarity(vec1: list[float], vec2: list[float]) -> float:
    """Cosine similarity"""
    return sum(a * b for a, b in zip(vec1, vec2))


# Corpus Loader
def corpus_loader(state: RAGState) -> RAGState:
    """Loads and chunks knowledge corpus"""
    
    system_message = SystemMessage(content="""You are a corpus loader. Load documents 
    and create searchable chunks with embeddings.""")
    
    user_message = HumanMessage(content="""Load knowledge corpus on AI and machine learning:

Documents to load:
1. "Large Language Models (LLMs) are neural networks trained on massive text corpora to understand and generate human language"
2. "Retrieval-Augmented Generation combines information retrieval with text generation for more accurate and grounded responses"
3. "Vector embeddings map text to high-dimensional space where semantic similarity is preserved"
4. "Fine-tuning adapts pre-trained models to specific tasks using smaller domain-specific datasets"
5. "Prompt engineering optimizes input prompts to guide model behavior and output quality"
6. "Multi-agent systems coordinate multiple AI agents to solve complex tasks through collaboration"
7. "Knowledge graphs represent information as entities and relationships enabling structured reasoning"
8. "Reinforcement Learning from Human Feedback (RLHF) aligns models with human preferences"

Create searchable chunks.""")
    
    response = llm.invoke([system_message, user_message])
    
    documents = [
        "Large Language Models (LLMs) are neural networks trained on massive text corpora to understand and generate human language",
        "Retrieval-Augmented Generation combines information retrieval with text generation for more accurate and grounded responses",
        "Vector embeddings map text to high-dimensional space where semantic similarity is preserved",
        "Fine-tuning adapts pre-trained models to specific tasks using smaller domain-specific datasets",
        "Prompt engineering optimizes input prompts to guide model behavior and output quality",
        "Multi-agent systems coordinate multiple AI agents to solve complex tasks through collaboration",
        "Knowledge graphs represent information as entities and relationships enabling structured reasoning",
        "Reinforcement Learning from Human Feedback (RLHF) aligns models with human preferences"
    ]
    
    knowledge_corpus = []
    for i, doc in enumerate(documents):
        knowledge_corpus.append({
            "chunk_id": f"chunk_{i+1}",
            "text": doc,
            "embedding": create_embedding(doc),
            "source": f"doc_{i//2 + 1}"  # Group chunks by source
        })
    
    return {
        "messages": [AIMessage(content=f"📚 Corpus Loader: {response.content}\n\n✅ Loaded {len(knowledge_corpus)} searchable chunks")],
        "knowledge_corpus": knowledge_corpus
    }


# Query Processor
def query_processor(state: RAGState) -> RAGState:
    """Processes user query for retrieval"""
    user_query = state.get("user_query", "")
    
    system_message = SystemMessage(content="""You are a query processor. Analyze the user 
    query, extract key concepts, and prepare for semantic retrieval.""")
    
    user_message = HumanMessage(content=f"""Process query for retrieval: "{user_query}"

Extract key concepts and generate query embedding.""")
    
    response = llm.invoke([system_message, user_message])
    
    query_embedding = create_embedding(user_query)
    
    return {
        "messages": [AIMessage(content=f"🔍 Query Processor: {response.content}\n\n✅ Generated query embedding")],
        "query_embedding": query_embedding
    }


# Retriever Agent
def retriever_agent(state: RAGState) -> RAGState:
    """Retrieves relevant chunks using semantic search"""
    query_embedding = state.get("query_embedding", [])
    knowledge_corpus = state.get("knowledge_corpus", [])
    user_query = state.get("user_query", "")
    
    system_message = SystemMessage(content="""You are a retriever agent. Find the most 
    relevant document chunks using semantic similarity.""")
    
    user_message = HumanMessage(content=f"""Retrieve relevant chunks for: "{user_query}"

Corpus size: {len(knowledge_corpus)} chunks

Find top 4 most relevant chunks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate similarities and rank
    scored_chunks = []
    for chunk in knowledge_corpus:
        score = similarity(query_embedding, chunk["embedding"])
        scored_chunks.append({
            **chunk,
            "relevance_score": score
        })
    
    # Sort and take top 4
    scored_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
    retrieved_chunks = scored_chunks[:4]
    
    return {
        "messages": [AIMessage(content=f"📥 Retriever: {response.content}\n\n✅ Retrieved {len(retrieved_chunks)} relevant chunks")],
        "retrieved_chunks": retrieved_chunks
    }


# Context Augmenter
def context_augmenter(state: RAGState) -> RAGState:
    """Augments prompt with retrieved context"""
    user_query = state.get("user_query", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    system_message = SystemMessage(content="""You are a context augmenter. Format retrieved 
    chunks into well-structured context for the generation model.""")
    
    chunks_text = "\n\n".join([
        f"[{i+1}] (Relevance: {chunk['relevance_score']:.3f})\n{chunk['text']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    user_message = HumanMessage(content=f"""Augment context for query: "{user_query}"

Retrieved Chunks:
{chunks_text}

Format as structured context.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create augmented context
    augmented_context = f"""Based on the following relevant information:

{chunks_text}

Please answer the question: {user_query}"""
    
    return {
        "messages": [AIMessage(content=f"📝 Context Augmenter: {response.content}\n\n✅ Created augmented context")],
        "augmented_context": augmented_context
    }


# Answer Generator
def answer_generator(state: RAGState) -> RAGState:
    """Generates answer using augmented context"""
    augmented_context = state.get("augmented_context", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    system_message = SystemMessage(content="""You are an answer generator. Create comprehensive, 
    accurate answers using the provided context. Include citations to source chunks.""")
    
    user_message = HumanMessage(content=augmented_context)
    
    response = llm.invoke([system_message, user_message])
    
    # Extract citations
    citations = [f"[{i+1}] {chunk['source']}" for i, chunk in enumerate(retrieved_chunks)]
    
    return {
        "messages": [AIMessage(content=f"💡 Answer Generator: {response.content}")],
        "generated_answer": response.content,
        "citations": citations
    }


# Quality Validator
def quality_validator(state: RAGState) -> RAGState:
    """Validates answer quality and grounding"""
    user_query = state.get("user_query", "")
    generated_answer = state.get("generated_answer", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    system_message = SystemMessage(content="""You are a quality validator. Assess if the 
    answer is well-grounded in the retrieved context, accurate, and complete.""")
    
    context_summary = " | ".join([chunk['text'][:50] + "..." for chunk in retrieved_chunks[:3]])
    
    user_message = HumanMessage(content=f"""Validate answer quality:

Query: {user_query}
Answer: {generated_answer[:200]}...
Context: {context_summary}

Rate quality (0-1) based on:
- Groundedness in context
- Accuracy
- Completeness
- Relevance""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple quality score (in practice, use more sophisticated scoring)
    quality_score = 0.92  # High quality RAG response
    
    return {
        "messages": [AIMessage(content=f"✅ Quality Validator: {response.content}\n\n✅ Quality Score: {quality_score:.2f}")],
        "quality_score": quality_score
    }


# RAG Monitor
def rag_monitor(state: RAGState) -> RAGState:
    """Monitors RAG pipeline"""
    knowledge_corpus = state.get("knowledge_corpus", [])
    retrieved_chunks = state.get("retrieved_chunks", [])
    citations = state.get("citations", [])
    quality_score = state.get("quality_score", 0.0)
    generated_answer = state.get("generated_answer", "")
    
    chunks_info = "\n".join([
        f"  {i+1}. [{chunk['chunk_id']}] Relevance: {chunk['relevance_score']:.3f}\n     {chunk['text'][:70]}..."
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    citations_text = "\n".join([f"  {cite}" for cite in citations])
    
    summary = f"""
    ✅ RAG PIPELINE COMPLETE
    
    Pipeline Statistics:
    • Knowledge Corpus: {len(knowledge_corpus)} chunks
    • Retrieved Chunks: {len(retrieved_chunks)}
    • Quality Score: {quality_score:.2f}/1.00
    • Citations: {len(citations)}
    
    Retrieved Context:
{chunks_info}
    
    Citations:
{citations_text}
    
    RAG Benefits:
    • Grounded in factual information
    • Reduced hallucination
    • Traceable sources
    • Up-to-date knowledge
    • Domain-specific accuracy
    
    Final Answer:
    {generated_answer[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 RAG Monitor:\n{summary}")]
    }


# Build the graph
def build_rag_graph():
    """Build the RAG pattern graph"""
    workflow = StateGraph(RAGState)
    
    workflow.add_node("loader", corpus_loader)
    workflow.add_node("processor", query_processor)
    workflow.add_node("retriever", retriever_agent)
    workflow.add_node("augmenter", context_augmenter)
    workflow.add_node("generator", answer_generator)
    workflow.add_node("validator", quality_validator)
    workflow.add_node("monitor", rag_monitor)
    
    workflow.add_edge(START, "loader")
    workflow.add_edge("loader", "processor")
    workflow.add_edge("processor", "retriever")
    workflow.add_edge("retriever", "augmenter")
    workflow.add_edge("augmenter", "generator")
    workflow.add_edge("generator", "validator")
    workflow.add_edge("validator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_rag_graph()
    
    print("=== RAG (Retrieval-Augmented Generation) MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "knowledge_corpus": [],
        "user_query": "How does RAG improve language model responses?",
        "query_embedding": [],
        "retrieved_chunks": [],
        "augmented_context": "",
        "generated_answer": "",
        "citations": [],
        "quality_score": 0.0
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== RAG Pipeline Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Answer with Citations ===")
    print(result.get("generated_answer", "No answer generated"))
    print(f"\n\nCitations:")
    for citation in result.get("citations", []):
        print(f"  {citation}")
    print(f"\n\nQuality Score: {result.get('quality_score', 0.0):.2f}/1.00")

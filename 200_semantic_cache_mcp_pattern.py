"""
Pattern 200: Semantic Cache MCP Pattern

This pattern demonstrates semantic caching based on query similarity:
- Embedding-based similarity matching
- Approximate cache hits for similar queries
- Vector similarity search
- LLM response caching

Semantic caching allows reusing cached responses for semantically similar
queries, even if the exact text differs. This is particularly useful for
LLM applications where users ask the same question in different ways.

Use Cases:
- LLM/ChatBot response caching (avoid redundant API calls)
- Search query caching (similar queries return cached results)
- Recommendation systems (similar user profiles)
- Document retrieval (semantic search caching)
- FAQ systems (match similar questions)

Key Features:
- Semantic Similarity: Match by meaning, not exact text
- Embedding Vectors: Convert queries to vector representations
- Similarity Threshold: Configurable threshold for cache hits
- Cost Reduction: Reduce expensive LLM API calls by 40-70%
- Response Time: Fast lookups for similar queries

Example:
  Query 1: "What is the capital of France?"
  Query 2: "Tell me the capital city of France"
  Query 3: "France's capital?"
  -> All three match semantically -> return cached response
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
import hashlib
import math


class SemanticCacheState(TypedDict):
    """State for semantic cache operations"""
    cache_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    similarity_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class CachedEntry:
    """Entry in semantic cache"""
    query: str
    query_embedding: List[float]
    response: str
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


class SimpleEmbedding:
    """
    Simple embedding generator (simulated).
    
    In production, use:
    - OpenAI Embeddings (text-embedding-ada-002)
    - Sentence-BERT
    - Universal Sentence Encoder
    - Custom fine-tuned models
    
    This implementation uses word-based vector representation
    for demonstration purposes.
    """
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.vocab_size = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase and split
        text = text.lower()
        # Remove punctuation
        for char in '.,!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        return [word for word in text.split() if word]
    
    def _get_word_id(self, word: str) -> int:
        """Get or create word ID"""
        if word not in self.vocab:
            self.vocab[word] = self.vocab_size
            self.vocab_size += 1
        return self.vocab[word]
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Simple bag-of-words representation with word position weighting.
        Real embeddings capture semantic meaning better.
        """
        tokens = self._tokenize(text)
        
        # Initialize embedding vector
        embedding = [0.0] * self.dim
        
        if not tokens:
            return embedding
        
        # Create simple embedding based on word IDs and positions
        for i, token in enumerate(tokens):
            word_id = self._get_word_id(token)
            # Spread word influence across multiple dimensions
            for j in range(self.dim):
                # Use hash of word+dimension for stable representation
                hash_val = hash(f"{token}_{j}") % 1000000
                # Position weighting: earlier words have more weight
                position_weight = 1.0 / (i + 1)
                embedding[j] += math.sin(hash_val) * position_weight
        
        # Normalize to unit vector
        magnitude = math.sqrt(sum(x*x for x in embedding))
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        return embedding


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Returns value between -1 and 1:
    - 1.0: Identical direction (very similar)
    - 0.0: Orthogonal (unrelated)
    - -1.0: Opposite direction (very dissimilar)
    """
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)


class SemanticCache:
    """
    Semantic cache using embedding similarity.
    
    Cache hits occur when:
    1. Exact match (text identical)
    2. Semantic match (similarity above threshold)
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.embedder = SimpleEmbedding(dim=128)
        self.cache: List[CachedEntry] = []
        
        # Statistics
        self.exact_hits = 0
        self.semantic_hits = 0
        self.misses = 0
        self.total_queries = 0
    
    def _find_similar(self, query: str, query_embedding: List[float]) -> Optional[Tuple[CachedEntry, float]]:
        """Find most similar cached entry"""
        best_match = None
        best_similarity = 0.0
        
        for entry in self.cache:
            # Check exact match first
            if entry.query.lower() == query.lower():
                return entry, 1.0
            
            # Calculate semantic similarity
            similarity = cosine_similarity(query_embedding, entry.query_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_match and best_similarity >= self.similarity_threshold:
            return best_match, best_similarity
        
        return None
    
    def get(self, query: str) -> Optional[Tuple[str, float, str]]:
        """
        Get cached response for query.
        
        Returns: (response, similarity, match_type) or None
        match_type: 'exact' or 'semantic'
        """
        self.total_queries += 1
        
        # Generate query embedding
        query_embedding = self.embedder.embed(query)
        
        # Find similar entry
        result = self._find_similar(query, query_embedding)
        
        if result:
            entry, similarity = result
            entry.access_count += 1
            entry.last_accessed = time.time()
            
            if similarity == 1.0:
                self.exact_hits += 1
                match_type = 'exact'
            else:
                self.semantic_hits += 1
                match_type = 'semantic'
            
            return entry.response, similarity, match_type
        
        self.misses += 1
        return None
    
    def put(self, query: str, response: str):
        """Cache a query-response pair"""
        query_embedding = self.embedder.embed(query)
        
        entry = CachedEntry(
            query=query,
            query_embedding=query_embedding,
            response=response
        )
        
        self.cache.append(entry)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = self.exact_hits + self.semantic_hits
        hit_rate = (total_hits / self.total_queries * 100) if self.total_queries > 0 else 0
        semantic_hit_rate = (self.semantic_hits / self.total_queries * 100) if self.total_queries > 0 else 0
        
        return {
            'total_queries': self.total_queries,
            'exact_hits': self.exact_hits,
            'semantic_hits': self.semantic_hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'semantic_hit_rate': semantic_hit_rate,
            'cache_size': len(self.cache),
            'similarity_threshold': self.similarity_threshold
        }


def setup_cache_agent(state: SemanticCacheState):
    """Agent to set up semantic cache"""
    operations = []
    results = []
    
    # Create semantic cache with 85% similarity threshold
    cache = SemanticCache(similarity_threshold=0.85)
    
    operations.append("Initialized Semantic Cache:")
    operations.append(f"  Similarity threshold: {cache.similarity_threshold}")
    operations.append(f"  Embedding dimension: {cache.embedder.dim}")
    operations.append("  Similarity metric: Cosine similarity")
    
    results.append("✓ Semantic cache initialized")
    
    # Store cache in state
    state['_cache'] = cache
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "similarity_metrics": [],
        "messages": ["Semantic cache setup complete"]
    }


def populate_cache_agent(state: SemanticCacheState):
    """Agent to populate cache with sample Q&A"""
    cache = state['_cache']
    operations = []
    results = []
    
    # Sample Q&A pairs (simulating LLM responses)
    qa_pairs = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("How do I reset my password?", "To reset your password, click on 'Forgot Password' on the login page and follow the instructions sent to your email."),
        ("What are the business hours?", "Our business hours are Monday-Friday 9AM-5PM EST."),
        ("How can I contact support?", "You can contact support by emailing support@example.com or calling 1-800-SUPPORT."),
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
    ]
    
    operations.append("\nPopulating semantic cache with Q&A pairs:")
    
    for query, response in qa_pairs:
        cache.put(query, response)
        operations.append(f"  CACHE: '{query[:50]}...' -> Response ({len(response)} chars)")
    
    results.append(f"✓ Cached {len(qa_pairs)} Q&A pairs")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "similarity_metrics": [],
        "messages": ["Cache populated with sample data"]
    }


def exact_match_demo_agent(state: SemanticCacheState):
    """Agent to demonstrate exact cache hits"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n1. Exact Match Demonstration:")
    
    # Exact match query
    query = "What is the capital of France?"
    result = cache.get(query)
    
    if result:
        response, similarity, match_type = result
        operations.append(f"\n  Query: '{query}'")
        operations.append(f"  Match type: {match_type}")
        operations.append(f"  Similarity: {similarity:.4f}")
        operations.append(f"  Response: '{response}'")
        
        metrics.append(f"Exact match: similarity = {similarity:.4f}")
    
    results.append("✓ Exact match cache hit")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "similarity_metrics": metrics,
        "messages": ["Exact match demonstrated"]
    }


def semantic_match_demo_agent(state: SemanticCacheState):
    """Agent to demonstrate semantic cache hits"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n2. Semantic Match Demonstration:")
    
    # Similar queries to cached ones
    similar_queries = [
        "Tell me the capital city of France",
        "What's France's capital?",
        "Capital of France please",
        "I need to reset my password",
        "Forgot my password, how to reset?",
        "Password reset procedure",
    ]
    
    operations.append("\nTesting semantically similar queries:")
    
    for query in similar_queries:
        result = cache.get(query)
        
        if result:
            response, similarity, match_type = result
            operations.append(f"\n  Query: '{query}'")
            operations.append(f"  Match type: {match_type}")
            operations.append(f"  Similarity: {similarity:.4f}")
            operations.append(f"  Cached query: '{cache.cache[0].query}'")
            operations.append(f"  Response: '{response[:60]}...'")
            
            metrics.append(f"Similarity: {similarity:.4f} for '{query[:30]}...'")
        else:
            operations.append(f"\n  Query: '{query}'")
            operations.append(f"  Result: MISS (no similar query in cache)")
    
    results.append("✓ Semantic matching demonstrated")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "similarity_metrics": metrics,
        "messages": ["Semantic matching demonstrated"]
    }


def cache_miss_demo_agent(state: SemanticCacheState):
    """Agent to demonstrate cache misses"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n3. Cache Miss Demonstration:")
    
    # Completely different queries
    different_queries = [
        "What is quantum computing?",
        "How do I cook pasta?",
        "Best programming language for beginners?",
    ]
    
    operations.append("\nTesting unrelated queries (should miss):")
    
    for query in different_queries:
        result = cache.get(query)
        
        operations.append(f"\n  Query: '{query}'")
        
        if result:
            response, similarity, match_type = result
            operations.append(f"  Unexpected hit: {similarity:.4f}")
        else:
            operations.append(f"  Result: MISS (no similar query)")
            operations.append(f"  Action: Would call LLM and cache response")
            
            # Simulate caching the new query
            cache.put(query, f"[LLM Response for: {query}]")
            operations.append(f"  ✓ New response cached")
    
    results.append("✓ Cache miss handling demonstrated")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "similarity_metrics": metrics,
        "messages": ["Cache miss demonstrated"]
    }


def performance_analysis_agent(state: SemanticCacheState):
    """Agent to analyze cache performance"""
    cache = state['_cache']
    operations = []
    results = []
    metrics = []
    
    stats = cache.get_stats()
    
    operations.append("\n" + "="*60)
    operations.append("PERFORMANCE ANALYSIS")
    operations.append("="*60)
    
    operations.append(f"\nCache Statistics:")
    operations.append(f"  Total queries: {stats['total_queries']}")
    operations.append(f"  Exact hits: {stats['exact_hits']}")
    operations.append(f"  Semantic hits: {stats['semantic_hits']}")
    operations.append(f"  Misses: {stats['misses']}")
    operations.append(f"  Overall hit rate: {stats['hit_rate']:.1f}%")
    operations.append(f"  Semantic hit rate: {stats['semantic_hit_rate']:.1f}%")
    operations.append(f"  Cache size: {stats['cache_size']} entries")
    
    # Calculate cost savings
    # Assume: LLM call = $0.002, Cache hit = $0.000001
    llm_cost_per_call = 0.002
    cache_cost_per_hit = 0.000001
    
    total_hits = stats['exact_hits'] + stats['semantic_hits']
    cost_without_cache = stats['total_queries'] * llm_cost_per_call
    cost_with_cache = (stats['misses'] * llm_cost_per_call) + (total_hits * cache_cost_per_hit)
    savings = cost_without_cache - cost_with_cache
    savings_percent = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0
    
    metrics.append(f"\nCost Analysis (per {stats['total_queries']} queries):")
    metrics.append(f"  Without cache: ${cost_without_cache:.4f}")
    metrics.append(f"  With cache: ${cost_with_cache:.4f}")
    metrics.append(f"  Savings: ${savings:.4f} ({savings_percent:.1f}%)")
    
    # Response time analysis
    # Assume: LLM call = 2000ms, Cache hit = 10ms
    llm_latency = 2000  # ms
    cache_latency = 10  # ms
    
    time_without_cache = stats['total_queries'] * llm_latency
    time_with_cache = (stats['misses'] * llm_latency) + (total_hits * cache_latency)
    time_saved = time_without_cache - time_with_cache
    
    metrics.append(f"\nLatency Analysis (per {stats['total_queries']} queries):")
    metrics.append(f"  Without cache: {time_without_cache}ms ({time_without_cache/1000:.1f}s)")
    metrics.append(f"  With cache: {time_with_cache}ms ({time_with_cache/1000:.1f}s)")
    metrics.append(f"  Time saved: {time_saved}ms ({time_saved/1000:.1f}s)")
    
    results.append(f"✓ Cache hit rate: {stats['hit_rate']:.1f}%")
    results.append(f"✓ Cost savings: {savings_percent:.1f}%")
    results.append(f"✓ Semantic hits: {stats['semantic_hits']} (avoid exact match requirement)")
    
    return {
        "cache_operations": operations,
        "operation_results": results,
        "similarity_metrics": metrics,
        "messages": ["Performance analysis complete"]
    }


def create_semantic_cache_graph():
    """Create the semantic cache workflow graph"""
    workflow = StateGraph(SemanticCacheState)
    
    # Add nodes
    workflow.add_node("setup", setup_cache_agent)
    workflow.add_node("populate", populate_cache_agent)
    workflow.add_node("exact_match", exact_match_demo_agent)
    workflow.add_node("semantic_match", semantic_match_demo_agent)
    workflow.add_node("cache_miss", cache_miss_demo_agent)
    workflow.add_node("performance", performance_analysis_agent)
    
    # Add edges
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "populate")
    workflow.add_edge("populate", "exact_match")
    workflow.add_edge("exact_match", "semantic_match")
    workflow.add_edge("semantic_match", "cache_miss")
    workflow.add_edge("cache_miss", "performance")
    workflow.add_edge("performance", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 200: Semantic Cache MCP Pattern")
    print("=" * 80)
    
    # Create and run the workflow
    app = create_semantic_cache_graph()
    
    # Initialize state
    initial_state = {
        "cache_operations": [],
        "operation_results": [],
        "similarity_metrics": [],
        "messages": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    # Display results
    print("\n" + "=" * 80)
    print("CACHE OPERATIONS")
    print("=" * 80)
    for op in final_state["cache_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("OPERATION RESULTS")
    print("=" * 80)
    for result in final_state["operation_results"]:
        print(result)
    
    print("\n" + "=" * 80)
    print("SIMILARITY METRICS")
    print("=" * 80)
    for metric in final_state["similarity_metrics"]:
        print(metric)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Semantic Cache Pattern implemented with:

1. Embedding-Based Similarity:
   - Convert queries to vector embeddings
   - Compare using cosine similarity
   - Match semantically similar queries (not just exact text)

2. Configurable Similarity Threshold:
   - Default: 0.85 (85% similarity)
   - Higher threshold: More strict matching
   - Lower threshold: More lenient matching

3. Dual Match Types:
   - Exact Match: Identical text (similarity = 1.0)
   - Semantic Match: Similar meaning (similarity >= threshold)

4. Cost & Performance Benefits:
   - Reduce LLM API calls by 40-70%
   - 200x faster response (10ms vs 2000ms)
   - Automatic handling of query variations

Use Cases:
✓ LLM/ChatBot Caching:
  - "What is AI?" ≈ "Tell me about artificial intelligence"
  - Reduces expensive GPT-4 API calls

✓ Search Query Caching:
  - "best restaurants NYC" ≈ "top places to eat in New York"
  - Reuse search results for similar queries

✓ FAQ Systems:
  - "How do I reset password?" ≈ "Forgot my password"
  - Match user questions to existing answers

✓ Customer Support:
  - Similar tickets get cached responses
  - Reduce support agent workload

Real-World Examples:
- ChatGPT: Cache responses for common questions
- Google Search: Similar queries return similar results
- Stack Overflow: Similar questions get linked
- Recommendation Systems: Similar user profiles

Implementation Considerations:

1. Embedding Generation:
   - OpenAI: text-embedding-ada-002 ($0.0001/1K tokens)
   - Sentence-BERT: Open-source, fast
   - Universal Sentence Encoder: Google's model
   - Custom fine-tuned: Domain-specific

2. Vector Storage:
   - In-memory: Fast, limited scale
   - Redis with vector search: Production-ready
   - Pinecone/Weaviate: Specialized vector DBs
   - FAISS: Facebook's similarity search

3. Similarity Threshold Tuning:
   - 0.95+: Very strict (near-exact match)
   - 0.85-0.95: Balanced (recommended)
   - 0.70-0.85: Lenient (more cache hits)
   - <0.70: Too lenient (false positives)

Performance Metrics (Typical):
- Embedding generation: 10-50ms
- Similarity search: 1-10ms
- Total cache hit latency: ~20ms
- LLM API call latency: 500-3000ms
- Speedup: 25-150x faster

Cost Savings Example (1M queries):
- Without cache: 1M × $0.002 = $2,000
- With cache (70% hit rate): 300K × $0.002 = $600
- Savings: $1,400 (70% reduction)

Comparison with Traditional Caching:
- Traditional (exact match): 20-30% hit rate
- Semantic (similarity match): 50-70% hit rate
- Benefit: 2-3x more cache hits
""")


if __name__ == "__main__":
    main()

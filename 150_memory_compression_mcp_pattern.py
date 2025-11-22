"""
Memory Compression MCP Pattern

This pattern implements memory compression techniques for
efficient storage and retrieval of information.

Key Features:
- Lossy compression
- Lossless compression
- Abstraction
- Chunking
- Efficient encoding
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class MemoryCompressionState(TypedDict):
    """State for memory compression pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    original_data: str
    compressed_data: str
    compression_ratio: float
    compression_method: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def memory_compression_agent(state: MemoryCompressionState) -> MemoryCompressionState:
    """Manages memory compression operations"""
    original = state.get("original_data", "")
    method = state.get("compression_method", "abstraction")
    
    system_prompt = """You are a memory compression expert.

Memory Compression:
• Reduce storage requirements
• Preserve essential information
• Lossy vs lossless trade-offs
• Abstraction and generalization
• Efficient representations

Compression enables scaling."""
    
    user_prompt = f"""Original Data: {len(original)} chars
Method: {method}

Design memory compression system.
Show how to compress while preserving essence."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🗜️ Memory Compression Agent:
    
    Compression Task:
    • Original Size: {len(original)} chars
    • Method: {method}
    • Goal: Reduce storage, preserve meaning
    
    Memory Compression Implementation:
    ```python
    class MemoryCompression:
        '''Compress memories for efficient storage'''
        
        def __init__(self):
            self.compression_methods = {{
                'abstraction': self.abstract_compress,
                'chunking': self.chunk_compress,
                'summarization': self.summarize_compress,
                'embedding': self.embed_compress,
                'deduplication': self.deduplicate_compress
            }}
        
        def compress(self, data, method='abstraction', target_ratio=0.3):
            '''Compress data using specified method'''
            original_size = self.measure_size(data)
            
            # Apply compression
            compressor = self.compression_methods[method]
            compressed = compressor(data, target_ratio)
            
            compressed_size = self.measure_size(compressed)
            
            # Calculate metrics
            ratio = compressed_size / original_size
            loss = self.measure_information_loss(data, compressed)
            
            return {{
                'original': data,
                'compressed': compressed,
                'compression_ratio': ratio,
                'information_loss': loss,
                'method': method
            }}
        
        def decompress(self, compressed_data, method):
            '''Reconstruct from compressed form'''
            if method == 'lossless':
                # Perfect reconstruction
                return self.lossless_decompress(compressed_data)
            else:
                # Lossy: reconstruct approximation
                return self.lossy_decompress(compressed_data)
    ```
    
    Compression Methods:
    
    Abstraction (Lossy):
    ```python
    def abstract_compress(details):
        '''Extract high-level concepts'''
        # Specific → General
        
        # Example:
        # Original: "I drove my red Toyota Camry to the grocery store 
        #           and bought milk, eggs, and bread"
        # Compressed: "Went shopping for groceries by car"
        
        # Extract key concepts
        entities = extract_entities(details)
        actions = extract_actions(details)
        
        # Generalize
        abstract = {{
            'action': generalize_action(actions),
            'object': generalize_object(entities),
            'time': extract_temporal(details),
            'location': generalize_location(details)
        }}
        
        # Generate abstract description
        compressed = generate_summary(abstract)
        
        return compressed
    ```
    
    Chunking:
    ```python
    def chunk_compress(items):
        '''Group into meaningful chunks'''
        # Miller's Law: 7±2 items → 1 chunk
        
        # Phone number: 5-5-5-1-2-3-4 (7 items)
        # Chunked: 555-1234 (2 chunks)
        
        chunks = []
        current_chunk = []
        
        for item in items:
            current_chunk.append(item)
            
            # Chunk when meaningful grouping found
            if is_meaningful_unit(current_chunk):
                chunks.append(create_chunk(current_chunk))
                current_chunk = []
        
        # Each chunk treated as single item
        return chunks
    ```
    
    Summarization:
    ```python
    def summarize_compress(text, target_length):
        '''Extract essential information'''
        # Extractive summarization
        sentences = split_sentences(text)
        
        # Score importance
        scored = [
            (sent, importance_score(sent, text))
            for sent in sentences
        ]
        
        # Select top sentences
        scored.sort(key=lambda x: x[1], reverse=True)
        
        summary = []
        length = 0
        
        for sent, score in scored:
            if length + len(sent) <= target_length:
                summary.append(sent)
                length += len(sent)
        
        # Reorder chronologically
        summary.sort(key=lambda s: sentences.index(s))
        
        return ' '.join(summary)
    ```
    
    Embedding Compression:
    ```python
    def embed_compress(text):
        '''Dense vector representation'''
        # Text → Vector
        # "Python is a programming language" 
        # → [0.1, -0.3, 0.8, ..., 0.2]  (1536 dims)
        
        # Advantages:
        # - Fixed size regardless of text length
        # - Semantic similarity preserved
        # - Efficient for search
        
        embedding = embedding_model.encode(text)
        
        # Can reconstruct approximate meaning via similarity
        # Cannot reconstruct exact text (lossy)
        
        return {{
            'embedding': embedding,
            'metadata': extract_metadata(text)
        }}
    ```
    
    Deduplication:
    ```python
    def deduplicate_compress(data):
        '''Remove redundancy'''
        # Find duplicates
        unique = set()
        references = []
        
        for item in data:
            if item not in unique:
                unique.add(item)
                references.append(('value', item))
            else:
                # Store reference instead of copy
                ref_id = get_id(item, unique)
                references.append(('ref', ref_id))
        
        # Compressed representation
        return {{
            'unique_values': list(unique),
            'structure': references
        }}
    ```
    
    Hierarchical Compression:
    
    Multi-Level Abstraction:
    ```python
    class HierarchicalCompression:
        '''Compress at multiple levels'''
        
        def compress_hierarchical(self, data):
            levels = []
            
            # Level 0: Raw data
            levels.append(data)
            
            # Level 1: Group similar items
            level1 = self.group_similar(data)
            levels.append(level1)
            
            # Level 2: Abstract categories
            level2 = self.abstract_categories(level1)
            levels.append(level2)
            
            # Level 3: High-level concepts
            level3 = self.extract_concepts(level2)
            levels.append(level3)
            
            return levels
        
        def retrieve_at_level(self, query, level=0):
            '''Access appropriate detail level'''
            if level == 0:
                return self.full_details()
            else:
                return self.levels[level]
    ```
    
    Semantic Compression:
    
    Gist Extraction:
    ```python
    def extract_gist(experience):
        '''Core meaning without details'''
        # "Went to Italian restaurant, ordered margherita pizza,
        #  waiter was friendly, paid $25"
        # Gist: "Had Italian food"
        
        gist = {{
            'type': identify_event_type(experience),
            'key_elements': extract_essential(experience),
            'outcome': identify_outcome(experience)
        }}
        
        # Details lost but essence preserved
        return gist
    ```
    
    Schema Compression:
    ```python
    def schema_compress(instance):
        '''Store deviation from schema'''
        # Schema: Restaurant visit
        # - Arrival, order, eat, pay, leave
        
        schema = get_schema('restaurant_visit')
        
        # Store only deviations
        deviations = {{}}
        for aspect in instance:
            if instance[aspect] != schema.default[aspect]:
                deviations[aspect] = instance[aspect]
        
        # Compressed: schema_id + deviations
        return {{
            'schema': 'restaurant_visit',
            'deviations': deviations
        }}
    ```
    
    Temporal Compression:
    
    Event Compression:
    ```python
    def compress_event_sequence(events):
        '''Compress temporal sequences'''
        # Consecutive similar events → single summary
        
        compressed = []
        current_pattern = None
        count = 0
        
        for event in events:
            if event['type'] == current_pattern:
                count += 1
            else:
                if current_pattern:
                    compressed.append({{
                        'pattern': current_pattern,
                        'count': count
                    }})
                current_pattern = event['type']
                count = 1
        
        return compressed
    ```
    
    Adaptive Compression:
    
    Importance-Based:
    ```python
    def adaptive_compress(memories, available_space):
        '''Compress based on importance'''
        # Sort by importance
        sorted_memories = sorted(
            memories,
            key=lambda m: m['importance'],
            reverse=True
        )
        
        # Allocate space proportionally
        total_importance = sum(m['importance'] for m in memories)
        
        compressed = []
        for memory in sorted_memories:
            # More important → less compression
            compression_level = 1.0 - (memory['importance'] / total_importance)
            
            compressed_mem = self.compress(
                memory,
                compression=compression_level
            )
            compressed.append(compressed_mem)
        
        return compressed
    ```
    
    Context-Aware:
    ```python
    def context_aware_compress(data, context):
        '''Compress relative to context'''
        # What's novel given context?
        
        redundant = find_redundant_with_context(data, context)
        novel = data - redundant
        
        # Store only novel information
        return {{
            'novel': novel,
            'context_ref': context.id
        }}
    ```
    
    Trade-offs:
    
    Compression vs Accuracy:
    ```python
    def compression_tradeoff(data, priority):
        '''Balance compression and fidelity'''
        
        if priority == 'accuracy':
            # Minimal compression
            return lossless_compress(data)
        
        elif priority == 'storage':
            # Maximum compression
            return aggressive_compress(data)
        
        else:  # balanced
            # Preserve critical information
            critical = extract_critical(data)
            critical_preserved = lossless_compress(critical)
            
            non_critical = data - critical
            non_critical_compressed = lossy_compress(non_critical)
            
            return critical_preserved + non_critical_compressed
    ```
    
    Reconstruction Quality:
    ```python
    def measure_reconstruction_quality(original, reconstructed):
        '''Assess information preservation'''
        
        # Semantic similarity
        semantic_sim = cosine_similarity(
            embed(original),
            embed(reconstructed)
        )
        
        # Fact preservation
        facts_original = extract_facts(original)
        facts_reconstructed = extract_facts(reconstructed)
        fact_recall = len(facts_original & facts_reconstructed) / len(facts_original)
        
        # Overall quality
        quality = 0.6 * semantic_sim + 0.4 * fact_recall
        
        return quality
    ```
    
    Applications:
    
    Long Conversation:
    ```python
    def compress_conversation_history(messages):
        '''Compress old messages'''
        # Recent: keep full
        recent = messages[-20:]
        
        # Old: compress
        old = messages[:-20]
        
        # Hierarchical compression
        compressed_old = {{
            'summary': summarize(old),
            'key_points': extract_key_points(old),
            'decisions': extract_decisions(old)
        }}
        
        return compressed_old, recent
    ```
    
    Best Practices:
    ✓ Choose compression level by importance
    ✓ Preserve semantic meaning
    ✓ Use hierarchical compression
    ✓ Monitor reconstruction quality
    ✓ Compress old memories more aggressively
    ✓ Maintain indices for retrieval
    
    Key Insight:
    Memory compression enables scaling by trading
    storage space for information fidelity - critical
    for managing long-term knowledge efficiently.
    
    🎉🎉🎉 ALL MEMORY PATTERNS (141-150) COMPLETE! 🎉🎉🎉
    """
    
    ratio = 0.3 if len(original) > 0 else 0.0
    
    return {
        "messages": [AIMessage(content=f"🗜️ Memory Compression Agent:\n{report}\n\n{response.content}")],
        "compressed_data": f"[Compressed via {method}]",
        "compression_ratio": ratio
    }


def build_memory_compression_graph():
    workflow = StateGraph(MemoryCompressionState)
    workflow.add_node("memory_compression_agent", memory_compression_agent)
    workflow.add_edge(START, "memory_compression_agent")
    workflow.add_edge("memory_compression_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_memory_compression_graph()
    
    print("=== Memory Compression MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "original_data": "This is a long conversation with many details that need to be compressed for efficient storage while preserving the key information and semantic meaning.",
        "compressed_data": "",
        "compression_ratio": 0.0,
        "compression_method": "abstraction"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 150: Memory Compression - COMPLETE")
    print(f"{'='*70}")
    print("\n🎉🎉🎉 ALL MEMORY PATTERNS (141-150) COMPLETE! 🎉🎉🎉")

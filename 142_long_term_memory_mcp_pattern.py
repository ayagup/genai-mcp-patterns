"""
Long-Term Memory MCP Pattern

This pattern implements long-term memory for persistent knowledge
storage and retrieval across sessions.

Key Features:
- Persistent storage
- Large capacity
- Knowledge retention
- Cross-session memory
- Structured retrieval
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class LongTermMemoryState(TypedDict):
    """State for long-term memory pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    memory_store: Dict
    retrieval_results: List[Dict]
    storage_type: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def long_term_memory_agent(state: LongTermMemoryState) -> LongTermMemoryState:
    """Manages long-term memory operations"""
    query = state.get("query", "")
    storage_type = state.get("storage_type", "vector_db")
    
    system_prompt = """You are a long-term memory expert.

Long-Term Memory (LTM):
• Persistent storage (unlimited duration)
• Large capacity (virtually unlimited)
• Slower access than STM
• Structured organization
• Survives across sessions

Key operations: encode, store, retrieve, consolidate."""
    
    user_prompt = f"""Query: {query}
Storage: {storage_type}

Design long-term memory system.
Show storage and retrieval strategies."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    💾 Long-Term Memory Agent:
    
    Memory System:
    • Storage Type: {storage_type}
    • Capacity: Unlimited
    • Persistence: Permanent
    • Access Pattern: Retrieval-based
    
    Long-Term Memory Architecture:
    ```python
    class LongTermMemory:
        def __init__(self, storage_backend='vector_db'):
            self.storage = self._init_storage(storage_backend)
            self.index = self._build_index()
            self.metadata = {{}}
        
        def store(self, content, metadata=None):
            '''Store information in LTM'''
            # Generate embedding
            embedding = self.embed(content)
            
            # Create memory entry
            memory_entry = {{
                'id': generate_id(),
                'content': content,
                'embedding': embedding,
                'metadata': metadata or {{}},
                'timestamp': time.time(),
                'access_count': 0,
                'importance': self.calculate_importance(content)
            }}
            
            # Store persistently
            self.storage.save(memory_entry)
            self.index.add(memory_entry)
            
            return memory_entry['id']
        
        def retrieve(self, query, top_k=5):
            '''Retrieve relevant memories'''
            # Generate query embedding
            query_embedding = self.embed(query)
            
            # Similarity search
            results = self.index.search(
                query_embedding, 
                top_k=top_k
            )
            
            # Update access statistics
            for result in results:
                result['access_count'] += 1
                result['last_accessed'] = time.time()
            
            return results
        
        def consolidate(self):
            '''Strengthen important memories'''
            for memory in self.storage.all():
                if self.should_consolidate(memory):
                    memory['importance'] += 1
                    memory['strength'] *= 1.5
    ```
    
    Storage Backends:
    
    Vector Database (Recommended):
    ```python
    from langchain.vectorstores import Chroma, Pinecone, Weaviate
    
    class VectorStoreLTM:
        def __init__(self):
            self.vectorstore = Chroma(
                embedding_function=OpenAIEmbeddings(),
                persist_directory="./ltm_store"
            )
        
        def store(self, text, metadata):
            self.vectorstore.add_texts(
                texts=[text],
                metadatas=[metadata]
            )
        
        def retrieve(self, query, k=5):
            return self.vectorstore.similarity_search(
                query, k=k
            )
    ```
    
    Graph Database:
    ```python
    from neo4j import GraphDatabase
    
    class GraphLTM:
        def __init__(self, uri, user, password):
            self.driver = GraphDatabase.driver(uri, 
                                              auth=(user, password))
        
        def store_concept(self, concept, relations):
            with self.driver.session() as session:
                # Create node
                session.run(
                    "CREATE (c:Concept {{name: $name, properties: $props}})",
                    name=concept.name,
                    props=concept.properties
                )
                
                # Create relationships
                for relation in relations:
                    session.run(
                        "MATCH (a:Concept {{name: $from}}), "
                        "(b:Concept {{name: $to}}) "
                        "CREATE (a)-[r:RELATES {{type: $type}}]->(b)",
                        from=relation.from_node,
                        to=relation.to_node,
                        type=relation.type
                    )
        
        def retrieve_related(self, concept, depth=2):
            with self.driver.session() as session:
                result = session.run(
                    "MATCH path = (c:Concept {{name: $name}})"
                    "-[*1..$depth]-(related) "
                    "RETURN path",
                    name=concept,
                    depth=depth
                )
                return [record['path'] for record in result]
    ```
    
    Document Store:
    ```python
    class DocumentStoreLTM:
        def __init__(self, db_path):
            self.db = sqlite3.connect(db_path)
            self._create_tables()
        
        def _create_tables(self):
            self.db.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT,
                    embedding BLOB,
                    metadata TEXT,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    access_count INTEGER,
                    importance REAL
                )
            ''')
            
            self.db.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance 
                ON memories(importance DESC)
            ''')
    ```
    
    Memory Types in LTM:
    
    Episodic (Events):
    ```python
    episodic_memory = {{
        'type': 'episodic',
        'event': 'User asked about Python decorators',
        'context': 'During code review discussion',
        'timestamp': '2024-01-15T10:30:00',
        'participants': ['user', 'assistant'],
        'outcome': 'Explained with examples'
    }}
    ```
    
    Semantic (Facts):
    ```python
    semantic_memory = {{
        'type': 'semantic',
        'concept': 'Python decorator',
        'definition': 'Function that modifies another function',
        'properties': ['syntax: @decorator', 'wrapper pattern'],
        'related_concepts': ['closure', 'higher-order function']
    }}
    ```
    
    Procedural (Skills):
    ```python
    procedural_memory = {{
        'type': 'procedural',
        'skill': 'Writing decorators',
        'steps': [
            'Define wrapper function',
            'Use functools.wraps',
            'Return wrapper',
            'Apply with @ syntax'
        ],
        'proficiency': 'expert'
    }}
    ```
    
    Retrieval Strategies:
    
    Similarity-Based:
    ```python
    def retrieve_by_similarity(query, top_k=5):
        query_emb = embed(query)
        
        # Cosine similarity
        results = []
        for memory in ltm.all_memories():
            similarity = cosine_similarity(query_emb, memory.embedding)
            results.append((memory, similarity))
        
        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, score in results[:top_k]]
    ```
    
    Associative Retrieval:
    ```python
    def retrieve_by_association(cue, hops=2):
        '''Retrieve memories connected to cue'''
        visited = set()
        queue = [cue]
        memories = []
        
        for _ in range(hops):
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            related = ltm.get_related(current)
            memories.extend(related)
            queue.extend([r.target for r in related])
        
        return memories
    ```
    
    Context-Based:
    ```python
    def retrieve_by_context(context_filters):
        '''Filter by metadata context'''
        return ltm.query(
            topic=context_filters.get('topic'),
            time_range=context_filters.get('time_range'),
            importance_min=context_filters.get('importance', 0.5),
            tags=context_filters.get('tags', [])
        )
    ```
    
    Memory Consolidation:
    
    Spacing Effect:
    ```python
    def spaced_rehearsal(memory):
        '''Strengthen through spaced repetition'''
        intervals = [1, 3, 7, 14, 30]  # days
        
        for interval in intervals:
            schedule_rehearsal(memory, days=interval)
            memory.strength *= 1.2
    ```
    
    Importance-Based:
    ```python
    def consolidate_important():
        '''Strengthen important memories'''
        for memory in ltm.all():
            importance = calculate_importance(memory)
            
            if importance > 0.8:
                memory.strength *= 2.0
                memory.persist = True
            elif importance < 0.2:
                if memory.age > 30 and memory.access_count == 0:
                    ltm.archive(memory)  # Move to cold storage
    ```
    
    Integration Patterns:
    
    STM → LTM Transfer:
    ```python
    def transfer_to_ltm(stm_item):
        if should_remember_long_term(stm_item):
            ltm.store(
                content=stm_item.content,
                metadata={{
                    'source': 'stm',
                    'task': current_task,
                    'importance': stm_item.importance
                }}
            )
    ```
    
    LTM → Working Memory:
    ```python
    def load_context(task):
        # Retrieve relevant LTM
        relevant = ltm.retrieve(task.description, top_k=10)
        
        # Load into working memory
        for memory in relevant:
            working_memory.load(memory)
        
        return working_memory
    ```
    
    Performance Optimization:
    
    Indexing:
    ```python
    # FAISS for fast similarity search
    import faiss
    
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(memory_embeddings)
    
    # Search
    distances, indices = index.search(query_embedding, k=5)
    ```
    
    Caching:
    ```python
    from functools import lru_cache
    
    @lru_cache(maxsize=100)
    def retrieve_cached(query_hash):
        return ltm.retrieve(query_hash)
    ```
    
    Best Practices:
    ✓ Use vector DB for semantic search
    ✓ Add rich metadata for filtering
    ✓ Implement importance scoring
    ✓ Regular consolidation
    ✓ Archive rarely used memories
    
    Key Insight:
    Long-term memory enables persistent knowledge
    across sessions - the foundation for continuous learning.
    """
    
    return {
        "messages": [AIMessage(content=f"💾 Long-Term Memory Agent:\n{report}\n\n{response.content}")],
        "retrieval_results": [{"memory": "example", "relevance": 0.95}]
    }


def build_long_term_memory_graph():
    workflow = StateGraph(LongTermMemoryState)
    workflow.add_node("long_term_memory_agent", long_term_memory_agent)
    workflow.add_edge(START, "long_term_memory_agent")
    workflow.add_edge("long_term_memory_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_long_term_memory_graph()
    
    print("=== Long-Term Memory MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "query": "Retrieve information about Python decorators",
        "memory_store": {},
        "retrieval_results": [],
        "storage_type": "vector_db"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 142: Long-Term Memory - COMPLETE")
    print(f"{'='*70}")

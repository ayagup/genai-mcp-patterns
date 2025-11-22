"""
Semantic Memory MCP Pattern

This pattern implements semantic memory for storing and retrieving
general knowledge, facts, concepts, and their relationships.

Key Features:
- Conceptual knowledge
- Fact storage
- Relationship mapping
- Category hierarchies
- Context-independent
"""

from typing import TypedDict, Sequence, Annotated, List, Dict, Set
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class SemanticMemoryState(TypedDict):
    """State for semantic memory pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    concepts: Dict
    relationships: List[Dict]
    knowledge_graph: Dict


llm = ChatOpenAI(model="gpt-4", temperature=0.2)


def semantic_memory_agent(state: SemanticMemoryState) -> SemanticMemoryState:
    """Manages semantic memory operations"""
    query = state.get("query", "")
    
    system_prompt = """You are a semantic memory expert.

Semantic Memory:
• General knowledge and facts
• Concepts and their meanings
• Categorical relationships
• Context-independent
• Shared cultural knowledge

"I know that..." knowledge."""
    
    user_prompt = f"""Query: {query}

Design semantic memory system.
Show knowledge organization and retrieval."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    📚 Semantic Memory Agent:
    
    Memory Type: Semantic (General Knowledge)
    • Facts: "Python is a programming language"
    • Concepts: "Decorator modifies functions"
    • Categories: "Python ∈ Programming Languages"
    • Relationships: "Python uses decorators"
    
    Semantic Memory Implementation:
    ```python
    class SemanticMemory:
        '''Store and retrieve general knowledge'''
        
        def __init__(self):
            self.concepts = {{}}
            self.facts = []
            self.relations = []
            self.categories = CategoryHierarchy()
            self.knowledge_graph = KnowledgeGraph()
        
        def store_concept(self, concept_name, properties):
            '''Add concept to semantic memory'''
            concept = {{
                'name': concept_name,
                'definition': properties.get('definition'),
                'properties': properties.get('properties', []),
                'examples': properties.get('examples', []),
                'category': properties.get('category'),
                'related_concepts': set()
            }}
            
            self.concepts[concept_name] = concept
            self.categories.add(concept_name, concept['category'])
            self.knowledge_graph.add_node(concept)
            
            return concept
        
        def store_fact(self, subject, predicate, object):
            '''Store factual knowledge (triple)'''
            fact = {{
                'subject': subject,
                'predicate': predicate,
                'object': object,
                'confidence': 1.0,
                'source': None
            }}
            
            self.facts.append(fact)
            self.knowledge_graph.add_edge(subject, predicate, object)
            
            return fact
        
        def retrieve_concept(self, concept_name):
            '''Retrieve concept and related knowledge'''
            if concept_name not in self.concepts:
                # Try semantic similarity
                similar = self.find_similar_concepts(concept_name)
                if similar:
                    concept_name = similar[0]
            
            concept = self.concepts.get(concept_name)
            
            if concept:
                # Enrich with related information
                concept['related'] = self.get_related_concepts(concept_name)
                concept['facts'] = self.get_facts_about(concept_name)
                concept['category_members'] = self.categories.get_members(
                    concept['category']
                )
            
            return concept
        
        def query_knowledge(self, query):
            '''Query semantic knowledge'''
            # Parse query into semantic components
            parsed = self.parse_semantic_query(query)
            
            if parsed['type'] == 'fact':
                # Retrieve specific fact
                return self.get_fact(parsed['subject'], parsed['predicate'])
            elif parsed['type'] == 'concept':
                # Retrieve concept
                return self.retrieve_concept(parsed['concept'])
            elif parsed['type'] == 'category':
                # Retrieve category members
                return self.categories.get_all_members(parsed['category'])
    ```
    
    Knowledge Representation:
    
    Concept Structure:
    ```python
    concept_decorator = {{
        'name': 'decorator',
        'domain': 'programming',
        'definition': 'A function that modifies another function',
        'properties': [
            'takes function as argument',
            'returns modified function',
            'uses @ syntax',
            'implements wrapper pattern'
        ],
        'examples': [
            '@staticmethod',
            '@property',
            '@lru_cache'
        ],
        'category': 'design_pattern',
        'superclass': 'higher_order_function',
        'related_concepts': ['closure', 'wrapper', 'metaprogramming'],
        'typical_use': 'add functionality without modifying original'
    }}
    ```
    
    Fact Triples:
    ```python
    facts = [
        ('Python', 'is_a', 'programming_language'),
        ('Python', 'supports', 'decorators'),
        ('Python', 'created_by', 'Guido_van_Rossum'),
        ('Python', 'first_released', '1991'),
        ('decorator', 'is_a', 'design_pattern'),
        ('decorator', 'part_of', 'Python'),
        ('@staticmethod', 'is_a', 'decorator'),
        ('decorator', 'uses', 'closure')
    ]
    ```
    
    Category Hierarchy:
    ```python
    class CategoryHierarchy:
        def __init__(self):
            self.hierarchy = {{}}
        
        def build_hierarchy(self):
            '''Create ISA hierarchy'''
            self.hierarchy = {{
                'entity': {{
                    'abstract': ['concept', 'relation'],
                    'concrete': ['object', 'event']
                }},
                'programming_concept': {{
                    'paradigm': ['OOP', 'functional', 'procedural'],
                    'structure': ['function', 'class', 'module'],
                    'pattern': ['decorator', 'singleton', 'factory']
                }},
                'language': {{
                    'programming': ['Python', 'Java', 'C++'],
                    'natural': ['English', 'Spanish', 'Chinese']
                }}
            }}
        
        def isa(self, instance, category):
            '''Check category membership'''
            return self.is_member_of(instance, category)
        
        def get_superordinate(self, concept):
            '''Get broader category'''
            # decorator → design_pattern → programming_concept
            return self.parent_category(concept)
        
        def get_subordinates(self, category):
            '''Get specific instances'''
            # design_pattern → [decorator, singleton, ...]
            return self.children(category)
    ```
    
    Knowledge Graph:
    ```python
    class KnowledgeGraph:
        '''Graph-based knowledge representation'''
        
        def __init__(self):
            self.nodes = {{}}  # Concepts/entities
            self.edges = []    # Relations
        
        def add_triple(self, subject, predicate, obj):
            '''Add (subject, predicate, object) triple'''
            # Ensure nodes exist
            for entity in [subject, obj]:
                if entity not in self.nodes:
                    self.nodes[entity] = {{'name': entity, 'type': 'entity'}}
            
            # Add edge
            self.edges.append({{
                'from': subject,
                'relation': predicate,
                'to': obj
            }})
        
        def query_path(self, start, end, max_hops=3):
            '''Find connection between concepts'''
            # BFS to find path
            queue = [(start, [start])]
            visited = set()
            
            while queue:
                current, path = queue.pop(0)
                
                if current == end:
                    return path
                
                if len(path) > max_hops:
                    continue
                
                if current in visited:
                    continue
                visited.add(current)
                
                # Explore connected nodes
                for edge in self.edges:
                    if edge['from'] == current:
                        queue.append((edge['to'], path + [edge['to']]))
            
            return None  # No path found
        
        def get_related(self, concept, depth=1):
            '''Get related concepts'''
            related = set()
            
            for edge in self.edges:
                if edge['from'] == concept:
                    related.add(edge['to'])
                elif edge['to'] == concept:
                    related.add(edge['from'])
            
            return related
    ```
    
    Retrieval Mechanisms:
    
    Spreading Activation:
    ```python
    def spreading_activation(source_concept, threshold=0.3):
        '''Activate related concepts'''
        activation = {{source_concept: 1.0}}
        
        for iteration in range(3):  # Spread for 3 hops
            new_activation = {{}}
            
            for concept, act in activation.items():
                if act < threshold:
                    continue
                
                # Spread to related concepts
                related = knowledge_graph.get_related(concept)
                for rel in related:
                    # Decay activation
                    new_act = act * 0.7
                    new_activation[rel] = max(
                        new_activation.get(rel, 0),
                        new_act
                    )
            
            activation.update(new_activation)
        
        return activation
    ```
    
    Semantic Similarity:
    ```python
    def semantic_similarity(concept1, concept2):
        '''Calculate concept similarity'''
        # Common properties
        props1 = set(concepts[concept1]['properties'])
        props2 = set(concepts[concept2]['properties'])
        jaccard = len(props1 & props2) / len(props1 | props2)
        
        # Shared category
        cat_sim = category_similarity(
            concepts[concept1]['category'],
            concepts[concept2]['category']
        )
        
        # Graph distance
        path = knowledge_graph.query_path(concept1, concept2)
        path_sim = 1.0 / (1 + len(path)) if path else 0
        
        return 0.4 * jaccard + 0.3 * cat_sim + 0.3 * path_sim
    ```
    
    Inference:
    
    Property Inheritance:
    ```python
    def inherit_properties(concept):
        '''Inherit from superclasses'''
        properties = set(concept['properties'])
        
        # Climb ISA hierarchy
        current = concept['category']
        while current:
            category_props = categories.get_properties(current)
            properties.update(category_props)
            current = categories.get_parent(current)
        
        return properties
    ```
    
    Transitive Closure:
    ```python
    def infer_transitive(relation):
        '''Infer transitive relations'''
        # If A → B and B → C, then A → C
        inferred = []
        
        for fact1 in facts:
            if fact1['predicate'] == relation:
                for fact2 in facts:
                    if (fact2['predicate'] == relation and
                        fact1['object'] == fact2['subject']):
                        inferred.append({{
                            'subject': fact1['subject'],
                            'predicate': relation,
                            'object': fact2['object'],
                            'derived': True
                        }})
        
        return inferred
    ```
    
    Schema Theory:
    ```python
    class Schema:
        '''Organized knowledge structure'''
        
        def __init__(self, name):
            self.name = name
            self.slots = {{}}
            self.defaults = {{}}
        
        def instantiate(self, values):
            '''Create instance with filled slots'''
            instance = self.defaults.copy()
            instance.update(values)
            return instance
    
    # Restaurant schema
    restaurant_schema = Schema('restaurant')
    restaurant_schema.slots = {{
        'location': str,
        'cuisine': str,
        'serves': ['breakfast', 'lunch', 'dinner'],
        'has_tables': True,
        'has_menu': True,
        'accepts_payment': True
    }}
    ```
    
    Applications:
    
    Question Answering:
    ```python
    def answer_question(question):
        '''Answer using semantic knowledge'''
        # "What is a decorator?"
        if 'what is' in question.lower():
            concept = extract_concept(question)
            info = retrieve_concept(concept)
            return info['definition']
        
        # "Is Python a programming language?"
        elif 'is' in question:
            subject, category = parse_is_question(question)
            return isa(subject, category)
    ```
    
    Best Practices:
    ✓ Use knowledge graphs
    ✓ Build category hierarchies  
    ✓ Enable spreading activation
    ✓ Support inference
    ✓ Link to episodic memory
    
    Key Insight:
    Semantic memory stores general knowledge and facts
    independent of personal experience - the foundation
    for reasoning and understanding.
    """
    
    return {
        "messages": [AIMessage(content=f"📚 Semantic Memory Agent:\n{report}\n\n{response.content}")],
        "knowledge_graph": {"nodes": 10, "edges": 25}
    }


def build_semantic_memory_graph():
    workflow = StateGraph(SemanticMemoryState)
    workflow.add_node("semantic_memory_agent", semantic_memory_agent)
    workflow.add_edge(START, "semantic_memory_agent")
    workflow.add_edge("semantic_memory_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_semantic_memory_graph()
    
    print("=== Semantic Memory MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "query": "What is a decorator in Python?",
        "concepts": {},
        "relationships": [],
        "knowledge_graph": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 145: Semantic Memory - COMPLETE")
    print(f"{'='*70}")

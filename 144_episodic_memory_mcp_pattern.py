"""
Episodic Memory MCP Pattern

This pattern implements episodic memory for storing and retrieving
personal experiences and events with temporal and spatial context.

Key Features:
- Event-based storage
- Temporal context
- Spatial context
- Autobiographical recall
- Experience replay
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class EpisodicMemoryState(TypedDict):
    """State for episodic memory pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    episodes: List[Dict]
    retrieval_cues: Dict
    reconstructed_episode: Dict


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def episodic_memory_agent(state: EpisodicMemoryState) -> EpisodicMemoryState:
    """Manages episodic memory operations"""
    query = state.get("query", "")
    
    system_prompt = """You are an episodic memory expert.

Episodic Memory:
• Personal experiences and events
• What, when, where context
• Temporal relationships
• Spatial context
• Autobiographical nature

"I remember when..." memories."""
    
    user_prompt = f"""Query: {query}

Design episodic memory system.
Show how to store and retrieve experiences."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    📸 Episodic Memory Agent:
    
    Memory Type: Episodic (Personal Experiences)
    • What happened: Event content
    • When: Temporal context
    • Where: Spatial context
    • Who: Participants
    • How felt: Emotional state
    
    Episodic Memory Implementation:
    ```python
    class EpisodicMemory:
        '''Store and retrieve personal experiences'''
        
        def __init__(self):
            self.episodes = []
            self.timeline = TemporalIndex()
            self.spatial_map = SpatialIndex()
        
        def encode_episode(self, event):
            '''Create episodic memory from experience'''
            episode = {{
                'what': event.content,
                'when': {{
                    'timestamp': event.time,
                    'date': event.date,
                    'time_of_day': event.hour,
                    'temporal_relations': self.get_temporal_context(event)
                }},
                'where': {{
                    'location': event.place,
                    'spatial_context': self.get_spatial_context(event),
                    'coordinates': event.coordinates
                }},
                'who': event.participants,
                'how': {{
                    'emotional_state': event.emotion,
                    'sensory_details': event.sensory,
                    'significance': event.importance
                }},
                'context': {{
                    'preceding_events': self.get_recent_events(),
                    'concurrent_activities': event.concurrent,
                    'goals': event.active_goals
                }}
            }}
            
            self.episodes.append(episode)
            self.timeline.index(episode)
            self.spatial_map.index(episode)
            
            return episode
        
        def retrieve_episode(self, cues):
            '''Retrieve using multiple cues'''
            candidates = self.episodes
            
            # Filter by temporal cues
            if 'when' in cues:
                candidates = self.timeline.search(
                    cues['when'], 
                    candidates
                )
            
            # Filter by spatial cues
            if 'where' in cues:
                candidates = self.spatial_map.search(
                    cues['where'],
                    candidates
                )
            
            # Filter by content
            if 'what' in cues:
                candidates = [
                    e for e in candidates
                    if self.content_match(e, cues['what'])
                ]
            
            # Reconstruct most relevant
            if candidates:
                return self.reconstruct(candidates[0])
            
            return None
        
        def reconstruct(self, episode_fragment):
            '''Reconstruct full episode from partial cues'''
            # Start with retrieved fragment
            reconstruction = episode_fragment.copy()
            
            # Fill in missing details from related episodes
            related = self.find_related_episodes(episode_fragment)
            reconstruction = self.infer_missing(reconstruction, related)
            
            # Add schema-based expectations
            reconstruction = self.apply_schema(reconstruction)
            
            return reconstruction
    ```
    
    Episode Structure:
    
    Example Episode:
    ```python
    episode = {{
        'id': 'ep_20240115_143000',
        'what': {{
            'event_type': 'conversation',
            'content': 'Discussed Python decorators with user',
            'actions': ['explained', 'provided examples', 'answered questions'],
            'outcome': 'User understood decorator pattern'
        }},
        'when': {{
            'timestamp': '2024-01-15T14:30:00Z',
            'duration': '15 minutes',
            'temporal_context': {{
                'before': 'Code review session',
                'after': 'Implementation task',
                'relative': '2 days before deadline'
            }}
        }},
        'where': {{
            'location': 'virtual_meeting',
            'context': 'project_workspace',
            'related_locations': ['code_repository', 'documentation']
        }},
        'who': {{
            'participants': ['user', 'assistant'],
            'roles': {{'user': 'learner', 'assistant': 'teacher'}}
        }},
        'how': {{
            'emotional': 'engaged, curious',
            'confidence': 0.8,
            'attention': 'high',
            'sensory': ['visual: code examples', 'linguistic: technical terms']
        }},
        'why': {{
            'goals': ['learn decorators', 'improve code quality'],
            'motivations': ['project requirement', 'skill development']
        }},
        'significance': 0.7,
        'vividness': 0.85,
        'accessibility': 0.9
    }}
    ```
    
    Retrieval Mechanisms:
    
    Direct Retrieval:
    ```python
    def retrieve_by_time(when):
        '''Retrieve episodes from specific time'''
        if isinstance(when, str):
            # Exact time
            return timeline.get_exact(when)
        elif isinstance(when, tuple):
            # Time range
            start, end = when
            return timeline.get_range(start, end)
    
    def retrieve_by_location(where):
        '''Retrieve episodes from location'''
        return spatial_map.get_at_location(where)
    
    def retrieve_by_participant(who):
        '''Retrieve episodes with specific person'''
        return [e for e in episodes if who in e['who']['participants']]
    ```
    
    Pattern Completion:
    ```python
    def pattern_complete(partial_cue):
        '''Complete pattern from partial information'''
        # Example: "That time we..." → full episode
        
        # Extract available cues
        cues = extract_cues(partial_cue)
        
        # Find matching episodes
        matches = fuzzy_match(cues, episodes)
        
        # Rank by completeness and relevance
        ranked = rank_episodes(matches, cues)
        
        return ranked[0] if ranked else None
    ```
    
    Mental Time Travel:
    ```python
    def mental_time_travel(target_time):
        '''Re-experience past episode'''
        episode = timeline.get(target_time)
        
        if episode:
            # Reconstruct sensory details
            reconstruction = {{
                'visual': episode['sensory']['visual'],
                'auditory': episode['sensory']['auditory'],
                'emotional': episode['how']['emotional'],
                'thoughts': episode['thoughts'],
                'perspective': 'first_person'  # Field memory
            }}
            
            # Simulate re-experiencing
            return reconstruction
    ```
    
    Temporal Organization:
    
    Timeline Structure:
    ```python
    class TemporalIndex:
        def __init__(self):
            self.chronological = []  # Time-ordered
            self.landmark_events = {{}}  # Major events
            self.periods = {{}}  # Life periods
        
        def organize_temporally(self, episodes):
            # Absolute time
            self.chronological = sorted(episodes, 
                                       key=lambda e: e['when']['timestamp'])
            
            # Relative to landmarks
            for episode in episodes:
                nearest_landmark = self.find_nearest_landmark(episode)
                episode['temporal_relation'] = {{
                    'landmark': nearest_landmark,
                    'offset': calculate_offset(episode, nearest_landmark)
                }}
            
            # Life periods
            self.organize_into_periods(episodes)
    ```
    
    Spatial Organization:
    ```python
    class SpatialIndex:
        def __init__(self):
            self.locations = {{}}
            self.routes = []
            self.spatial_hierarchies = {{}}
        
        def index_spatially(self, episode):
            location = episode['where']['location']
            
            # Add to location index
            if location not in self.locations:
                self.locations[location] = []
            self.locations[location].append(episode)
            
            # Spatial hierarchy
            # e.g., room → building → city → country
            self.add_to_hierarchy(location, episode)
    ```
    
    Memory Consolidation:
    
    Sleep Consolidation:
    ```python
    def consolidate_episodes():
        '''Strengthen important episodes'''
        for episode in recent_episodes:
            # Calculate importance
            importance = (
                episode['emotional_intensity'] * 0.4 +
                episode['novelty'] * 0.3 +
                episode['goal_relevance'] * 0.3
            )
            
            if importance > threshold:
                # Strengthen
                episode['strength'] *= 1.5
                
                # Extract to semantic memory
                semantic_memory.extract_facts(episode)
                
                # Integrate with existing knowledge
                integrate_with_schemas(episode)
    ```
    
    Forgetting:
    ```python
    def apply_forgetting_curve(episode, time_elapsed):
        '''Model memory decay over time'''
        # Ebbinghaus forgetting curve
        retention = np.exp(-time_elapsed / decay_rate)
        
        # Modulated by importance
        retention = retention * (1 + episode['significance'])
        
        episode['accessibility'] = retention
        
        # If too low, move to inaccessible storage
        if retention < 0.1:
            archive_episode(episode)
    ```
    
    Integration with Other Memory:
    
    Episodic → Semantic:
    ```python
    def extract_semantic_knowledge(episode):
        '''Derive general knowledge from specific episode'''
        # Example: "Paris is in France"
        # From episode: "I visited Paris, France"
        
        facts = []
        
        # Extract facts
        if 'location' in episode['where']:
            facts.append({{
                'type': 'location_fact',
                'content': f"{{episode['what']['subject']}} is in {{episode['where']['location']}}"
            }})
        
        # Generalize from specific
        semantic_memory.store_facts(facts)
    ```
    
    Applications:
    
    Conversation History:
    ```python
    # Store conversation as episodes
    episode = encode_episode({{
        'what': conversation_turn,
        'when': timestamp,
        'who': [user, assistant],
        'context': conversation_context
    }})
    
    # Retrieve for context
    previous_discussion = retrieve_episode({{
        'what': 'Python decorators',
        'who': current_user
    }})
    ```
    
    Experience Replay:
    ```python
    def replay_for_learning(task_type):
        '''Replay relevant episodes to improve'''
        # Find similar past experiences
        relevant = retrieve_episode({{'what': task_type}})
        
        # Extract what worked
        successful_strategies = [
            e['actions'] for e in relevant
            if e['outcome']['success']
        ]
        
        return successful_strategies
    ```
    
    Best Practices:
    ✓ Rich contextual encoding
    ✓ Multiple retrieval cues
    ✓ Temporal organization
    ✓ Consolidate important episodes
    ✓ Extract to semantic memory
    
    Key Insight:
    Episodic memory stores personal experiences with
    rich context, enabling mental time travel and
    learning from past experiences.
    """
    
    return {
        "messages": [AIMessage(content=f"📸 Episodic Memory Agent:\n{report}\n\n{response.content}")],
        "reconstructed_episode": {"event": "example", "when": "past", "where": "context"}
    }


def build_episodic_memory_graph():
    workflow = StateGraph(EpisodicMemoryState)
    workflow.add_node("episodic_memory_agent", episodic_memory_agent)
    workflow.add_edge(START, "episodic_memory_agent")
    workflow.add_edge("episodic_memory_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_episodic_memory_graph()
    
    print("=== Episodic Memory MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "query": "Remember when we discussed Python decorators",
        "episodes": [],
        "retrieval_cues": {"what": "Python decorators", "when": "last week"},
        "reconstructed_episode": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 144: Episodic Memory - COMPLETE")
    print(f"{'='*70}")

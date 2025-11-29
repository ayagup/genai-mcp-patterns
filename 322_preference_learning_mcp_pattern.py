"""
Preference Learning MCP Pattern

This pattern demonstrates preference learning in an agentic MCP system.
The system learns and adapts to user preferences over time through
implicit and explicit feedback mechanisms.

Use cases:
- Implicit preference extraction
- Explicit preference capture
- Preference evolution tracking
- Preference conflict resolution
- Multi-dimensional preference modeling
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from collections import defaultdict


# Define the state for preference learning
class PreferenceLearningState(TypedDict):
    """State for tracking preference learning process"""
    messages: Annotated[List[str], add]
    user_interactions: List[Dict[str, Any]]
    explicit_preferences: Dict[str, Any]
    learned_preferences: Dict[str, Any]
    preference_history: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    report: str


class PreferenceLearner:
    """Learn user preferences from interactions"""
    
    def __init__(self):
        self.preference_weights = defaultdict(float)
        self.interaction_counts = defaultdict(int)
    
    def learn_from_view(self, item: Dict[str, Any], duration: int):
        """Learn preferences from viewing behavior"""
        # Weight by duration (longer = stronger preference)
        weight = min(duration / 60.0, 3.0)  # Cap at 3x weight for very long views
        
        for category in item.get('categories', []):
            self.preference_weights[f'category_{category}'] += weight
            self.interaction_counts[f'category_{category}'] += 1
        
        topic = item.get('topic')
        if topic:
            self.preference_weights[f'topic_{topic}'] += weight
            self.interaction_counts[f'topic_{topic}'] += 1
    
    def learn_from_like(self, item: Dict[str, Any]):
        """Learn preferences from explicit likes"""
        weight = 5.0  # Strong signal
        
        for category in item.get('categories', []):
            self.preference_weights[f'category_{category}'] += weight
            self.interaction_counts[f'category_{category}'] += 1
        
        topic = item.get('topic')
        if topic:
            self.preference_weights[f'topic_{topic}'] += weight
            self.interaction_counts[f'topic_{topic}'] += 1
    
    def learn_from_dislike(self, item: Dict[str, Any]):
        """Learn preferences from explicit dislikes"""
        weight = -3.0  # Negative signal
        
        for category in item.get('categories', []):
            self.preference_weights[f'category_{category}'] += weight
            self.interaction_counts[f'category_{category}'] += 1
        
        topic = item.get('topic')
        if topic:
            self.preference_weights[f'topic_{topic}'] += weight
            self.interaction_counts[f'topic_{topic}'] += 1
    
    def learn_from_skip(self, item: Dict[str, Any]):
        """Learn preferences from skips"""
        weight = -1.0  # Mild negative signal
        
        for category in item.get('categories', []):
            self.preference_weights[f'category_{category}'] += weight
            self.interaction_counts[f'category_{category}'] += 1
    
    def get_preferences(self) -> Dict[str, float]:
        """Get normalized preference scores"""
        preferences = {}
        
        for key, weight in self.preference_weights.items():
            count = self.interaction_counts[key]
            # Normalize by number of interactions
            normalized_score = weight / count if count > 0 else 0
            preferences[key] = normalized_score
        
        return preferences
    
    def get_confidence_scores(self) -> Dict[str, float]:
        """Calculate confidence in each preference"""
        confidence = {}
        
        for key, count in self.interaction_counts.items():
            # Confidence increases with more interactions, capped at 1.0
            confidence[key] = min(count / 10.0, 1.0)
        
        return confidence


class PreferenceEvolutionTracker:
    """Track how preferences evolve over time"""
    
    def __init__(self):
        self.snapshots = []
    
    def add_snapshot(self, timestamp: datetime, preferences: Dict[str, float]):
        """Add a preference snapshot"""
        self.snapshots.append({
            'timestamp': timestamp,
            'preferences': preferences.copy()
        })
    
    def get_preference_trend(self, preference_key: str) -> List[tuple]:
        """Get trend for a specific preference"""
        return [
            (snapshot['timestamp'], snapshot['preferences'].get(preference_key, 0))
            for snapshot in self.snapshots
        ]
    
    def detect_preference_shifts(self, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect significant preference changes"""
        if len(self.snapshots) < 2:
            return []
        
        shifts = []
        latest = self.snapshots[-1]['preferences']
        previous = self.snapshots[-2]['preferences']
        
        all_keys = set(latest.keys()) | set(previous.keys())
        
        for key in all_keys:
            latest_val = latest.get(key, 0)
            previous_val = previous.get(key, 0)
            change = latest_val - previous_val
            
            if abs(change) >= threshold:
                shifts.append({
                    'preference': key,
                    'previous_value': previous_val,
                    'current_value': latest_val,
                    'change': change,
                    'direction': 'increased' if change > 0 else 'decreased'
                })
        
        return shifts


class PreferenceConflictResolver:
    """Resolve conflicts between different preference signals"""
    
    def resolve_explicit_vs_implicit(self, explicit: Dict[str, Any], 
                                     implicit: Dict[str, float]) -> Dict[str, float]:
        """Resolve conflicts between explicit and implicit preferences"""
        resolved = implicit.copy()
        
        # Explicit preferences override implicit ones with high confidence
        for key, value in explicit.items():
            pref_key = f'category_{key}' if not key.startswith('category_') else key
            
            if value == 'like':
                resolved[pref_key] = max(resolved.get(pref_key, 0), 5.0)
            elif value == 'dislike':
                resolved[pref_key] = min(resolved.get(pref_key, 0), -3.0)
        
        return resolved
    
    def resolve_temporal_conflicts(self, preferences: Dict[str, float],
                                   recency_weight: float = 0.7) -> Dict[str, float]:
        """Weight recent preferences more heavily"""
        # In a real system, this would use temporal data
        # Here we simulate by applying recency weight
        return {k: v * recency_weight for k, v in preferences.items()}


# Agent functions
def initialize_interactions_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Initialize with user interaction data"""
    
    # Simulate user interactions over time
    base_time = datetime.now() - timedelta(days=30)
    
    interactions = [
        # Week 1 - Exploring tech content
        {'timestamp': (base_time + timedelta(days=1)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_1', 'categories': ['technology', 'ai'], 'topic': 'machine_learning'}, 
         'duration': 180},
        {'timestamp': (base_time + timedelta(days=1)).isoformat(), 'type': 'like', 
         'item': {'id': 'article_2', 'categories': ['technology', 'programming'], 'topic': 'python'}},
        {'timestamp': (base_time + timedelta(days=2)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_3', 'categories': ['technology', 'ai'], 'topic': 'deep_learning'}, 
         'duration': 240},
        
        # Week 2 - Discovering interest in business
        {'timestamp': (base_time + timedelta(days=8)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_4', 'categories': ['business', 'startup'], 'topic': 'entrepreneurship'}, 
         'duration': 120},
        {'timestamp': (base_time + timedelta(days=9)).isoformat(), 'type': 'like', 
         'item': {'id': 'article_5', 'categories': ['business', 'finance'], 'topic': 'investing'}},
        {'timestamp': (base_time + timedelta(days=10)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_6', 'categories': ['sports'], 'topic': 'football'}, 
         'duration': 30},
        {'timestamp': (base_time + timedelta(days=10)).isoformat(), 'type': 'skip', 
         'item': {'id': 'article_7', 'categories': ['sports'], 'topic': 'basketball'}},
        
        # Week 3 - Strong AI preference emerging
        {'timestamp': (base_time + timedelta(days=15)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_8', 'categories': ['technology', 'ai'], 'topic': 'nlp'}, 
         'duration': 300},
        {'timestamp': (base_time + timedelta(days=16)).isoformat(), 'type': 'like', 
         'item': {'id': 'article_9', 'categories': ['technology', 'ai'], 'topic': 'computer_vision'}},
        {'timestamp': (base_time + timedelta(days=17)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_10', 'categories': ['technology', 'ai'], 'topic': 'reinforcement_learning'}, 
         'duration': 200},
        
        # Week 4 - Disliking politics
        {'timestamp': (base_time + timedelta(days=22)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_11', 'categories': ['politics'], 'topic': 'elections'}, 
         'duration': 20},
        {'timestamp': (base_time + timedelta(days=22)).isoformat(), 'type': 'dislike', 
         'item': {'id': 'article_12', 'categories': ['politics'], 'topic': 'policy'}},
        {'timestamp': (base_time + timedelta(days=23)).isoformat(), 'type': 'skip', 
         'item': {'id': 'article_13', 'categories': ['politics'], 'topic': 'debate'}},
        
        # Recent - Continued tech/business interest
        {'timestamp': (base_time + timedelta(days=28)).isoformat(), 'type': 'view', 
         'item': {'id': 'article_14', 'categories': ['technology', 'business'], 'topic': 'saas'}, 
         'duration': 150},
        {'timestamp': (base_time + timedelta(days=29)).isoformat(), 'type': 'like', 
         'item': {'id': 'article_15', 'categories': ['technology', 'ai'], 'topic': 'llm'}}
    ]
    
    return {
        **state,
        'user_interactions': interactions,
        'messages': state['messages'] + [f'Initialized with {len(interactions)} user interactions']
    }


def capture_explicit_preferences_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Capture explicit user preferences"""
    
    # Simulate explicit preferences from settings/profile
    explicit_preferences = {
        'preferred_categories': ['technology', 'business'],
        'disliked_categories': ['politics'],
        'notification_preferences': {
            'ai': 'like',
            'sports': 'dislike'
        },
        'content_length': 'medium',  # short, medium, long
        'reading_time': 'evening'
    }
    
    return {
        **state,
        'explicit_preferences': explicit_preferences,
        'messages': state['messages'] + ['Captured explicit user preferences']
    }


def learn_implicit_preferences_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Learn preferences from implicit signals"""
    
    learner = PreferenceLearner()
    
    # Process each interaction
    for interaction in state['user_interactions']:
        int_type = interaction['type']
        item = interaction['item']
        
        if int_type == 'view':
            learner.learn_from_view(item, interaction.get('duration', 0))
        elif int_type == 'like':
            learner.learn_from_like(item)
        elif int_type == 'dislike':
            learner.learn_from_dislike(item)
        elif int_type == 'skip':
            learner.learn_from_skip(item)
    
    learned_preferences = learner.get_preferences()
    confidence_scores = learner.get_confidence_scores()
    
    # Sort by preference score
    sorted_prefs = sorted(learned_preferences.items(), key=lambda x: x[1], reverse=True)
    
    return {
        **state,
        'learned_preferences': dict(sorted_prefs[:20]),  # Top 20
        'confidence_scores': confidence_scores,
        'messages': state['messages'] + [f'Learned {len(learned_preferences)} preference signals']
    }


def track_preference_evolution_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Track how preferences evolve over time"""
    
    tracker = PreferenceEvolutionTracker()
    
    # Create snapshots at different time periods
    # Simulate weekly snapshots
    base_time = datetime.now() - timedelta(days=30)
    
    # Week 1 snapshot (early preferences)
    week1_prefs = {
        'category_technology': 2.5,
        'category_ai': 2.0,
        'topic_machine_learning': 3.0
    }
    tracker.add_snapshot(base_time + timedelta(days=7), week1_prefs)
    
    # Week 2 snapshot (discovering business)
    week2_prefs = {
        'category_technology': 2.8,
        'category_ai': 2.5,
        'category_business': 2.0,
        'topic_machine_learning': 3.0,
        'topic_entrepreneurship': 2.0
    }
    tracker.add_snapshot(base_time + timedelta(days=14), week2_prefs)
    
    # Week 3 snapshot (strong AI preference)
    week3_prefs = {
        'category_technology': 3.5,
        'category_ai': 4.2,
        'category_business': 2.3,
        'topic_machine_learning': 3.5,
        'topic_nlp': 5.0,
        'category_politics': -2.0
    }
    tracker.add_snapshot(base_time + timedelta(days=21), week3_prefs)
    
    # Current snapshot
    tracker.add_snapshot(datetime.now(), state['learned_preferences'])
    
    # Detect shifts
    shifts = tracker.detect_preference_shifts(threshold=1.5)
    
    return {
        **state,
        'preference_history': tracker.snapshots,
        'messages': state['messages'] + [
            f'Tracked preference evolution: {len(shifts)} significant shifts detected'
        ]
    }


def resolve_preference_conflicts_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Resolve conflicts between different preference signals"""
    
    resolver = PreferenceConflictResolver()
    
    # Combine explicit and implicit preferences
    final_preferences = resolver.resolve_explicit_vs_implicit(
        state['explicit_preferences'].get('notification_preferences', {}),
        state['learned_preferences']
    )
    
    # Apply recency weighting
    final_preferences = resolver.resolve_temporal_conflicts(final_preferences)
    
    return {
        **state,
        'learned_preferences': final_preferences,
        'messages': state['messages'] + ['Resolved preference conflicts']
    }


def generate_recommendations_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Generate personalized recommendations based on learned preferences"""
    
    # Get top preferences
    top_prefs = sorted(state['learned_preferences'].items(), 
                      key=lambda x: x[1], reverse=True)[:5]
    
    recommendations = []
    
    for pref_key, score in top_prefs:
        # Extract category/topic from key
        pref_type, pref_value = pref_key.split('_', 1)
        
        confidence = state['confidence_scores'].get(pref_key, 0)
        
        recommendations.append({
            'type': pref_type,
            'value': pref_value,
            'preference_score': round(score, 2),
            'confidence': round(confidence, 2),
            'recommendation': f'Show more {pref_value} content' if score > 0 else f'Avoid {pref_value} content'
        })
    
    return {
        **state,
        'recommendations': recommendations,
        'messages': state['messages'] + [f'Generated {len(recommendations)} recommendations']
    }


def generate_preference_learning_report_agent(state: PreferenceLearningState) -> PreferenceLearningState:
    """Generate comprehensive preference learning report"""
    
    report_lines = [
        "=" * 80,
        "PREFERENCE LEARNING REPORT",
        "=" * 80,
        "",
        "LEARNING SUMMARY:",
        "-" * 40,
        f"Total interactions analyzed: {len(state['user_interactions'])}",
        f"Explicit preferences: {len(state['explicit_preferences'])}",
        f"Learned preferences: {len(state['learned_preferences'])}",
        f"Preference snapshots: {len(state['preference_history'])}",
        "",
        "EXPLICIT PREFERENCES:",
        "-" * 40,
        f"Preferred categories: {', '.join(state['explicit_preferences'].get('preferred_categories', []))}",
        f"Disliked categories: {', '.join(state['explicit_preferences'].get('disliked_categories', []))}",
        f"Content length: {state['explicit_preferences'].get('content_length')}",
        f"Reading time: {state['explicit_preferences'].get('reading_time')}",
        "",
        "TOP LEARNED PREFERENCES:",
        "-" * 40
    ]
    
    for pref, score in list(state['learned_preferences'].items())[:10]:
        confidence = state['confidence_scores'].get(pref, 0)
        sentiment = "positive" if score > 0 else "negative"
        report_lines.append(
            f"  {pref}: {score:.2f} ({sentiment}, confidence: {confidence:.2f})"
        )
    
    report_lines.extend([
        "",
        "INTERACTION BREAKDOWN:",
        "-" * 40
    ])
    
    interaction_types = {}
    for interaction in state['user_interactions']:
        int_type = interaction['type']
        interaction_types[int_type] = interaction_types.get(int_type, 0) + 1
    
    for int_type, count in sorted(interaction_types.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {int_type}: {count}")
    
    report_lines.extend([
        "",
        "PREFERENCE EVOLUTION:",
        "-" * 40
    ])
    
    for i, snapshot in enumerate(state['preference_history'], 1):
        timestamp = snapshot['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        report_lines.append(f"\nSnapshot {i} ({timestamp.strftime('%Y-%m-%d')}):")
        
        top_3 = sorted(snapshot['preferences'].items(), key=lambda x: x[1], reverse=True)[:3]
        for pref, score in top_3:
            report_lines.append(f"  {pref}: {score:.2f}")
    
    report_lines.extend([
        "",
        "PERSONALIZED RECOMMENDATIONS:",
        "-" * 40
    ])
    
    for rec in state['recommendations']:
        report_lines.append(f"\n{rec['type'].upper()}: {rec['value']}")
        report_lines.append(f"  Preference Score: {rec['preference_score']}")
        report_lines.append(f"  Confidence: {rec['confidence']}")
        report_lines.append(f"  Action: {rec['recommendation']}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Strongest preference: {max(state['learned_preferences'].items(), key=lambda x: x[1])[0] if state['learned_preferences'] else 'N/A'}",
        f"✓ Most confident prediction: {max(state['confidence_scores'].items(), key=lambda x: x[1])[0] if state['confidence_scores'] else 'N/A'}",
        f"✓ Learning trajectory: Preferences becoming more refined over time",
        "",
        "RECOMMENDATIONS FOR SYSTEM:",
        "-" * 40,
        "• Continue tracking implicit signals for preference refinement",
        "• Periodically ask for explicit preference confirmation",
        "• Use A/B testing to validate learned preferences",
        "• Implement preference decay for outdated signals",
        "• Allow users to correct mislearned preferences",
        "• Weight recent interactions more heavily",
        "• Detect and handle preference shifts proactively",
        "• Provide transparency in how preferences are used",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive preference learning report']
    }


# Create the graph
def create_preference_learning_graph():
    """Create the preference learning workflow graph"""
    
    workflow = StateGraph(PreferenceLearningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_interactions_agent)
    workflow.add_node("capture_explicit", capture_explicit_preferences_agent)
    workflow.add_node("learn_implicit", learn_implicit_preferences_agent)
    workflow.add_node("track_evolution", track_preference_evolution_agent)
    workflow.add_node("resolve_conflicts", resolve_preference_conflicts_agent)
    workflow.add_node("generate_recommendations", generate_recommendations_agent)
    workflow.add_node("generate_report", generate_preference_learning_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "capture_explicit")
    workflow.add_edge("capture_explicit", "learn_implicit")
    workflow.add_edge("learn_implicit", "track_evolution")
    workflow.add_edge("track_evolution", "resolve_conflicts")
    workflow.add_edge("resolve_conflicts", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the preference learning graph
    app = create_preference_learning_graph()
    
    # Initialize state
    initial_state: PreferenceLearningState = {
        'messages': [],
        'user_interactions': [],
        'explicit_preferences': {},
        'learned_preferences': {},
        'preference_history': [],
        'confidence_scores': {},
        'recommendations': [],
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("PREFERENCE LEARNING MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nPreference learning pattern execution complete! ✓")

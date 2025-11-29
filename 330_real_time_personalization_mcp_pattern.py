"""
Real-Time Personalization MCP Pattern

This pattern demonstrates real-time personalization in an agentic MCP system.
The system processes streaming user events and adapts personalization
immediately, providing instant responses to user behavior.

Use cases:
- Live recommendation updates
- Real-time content adaptation
- Session-based personalization
- Instant feedback incorporation
- Streaming analytics personalization
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from collections import deque, defaultdict
import math


# Define the state for real-time personalization
class RealTimePersonalizationState(TypedDict):
    """State for tracking real-time personalization process"""
    messages: Annotated[List[str], add]
    event_stream: List[Dict[str, Any]]
    user_state: Dict[str, Any]
    real_time_profile: Dict[str, Any]
    personalization_updates: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    adaptation_actions: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class EventProcessor:
    """Process streaming user events in real-time"""
    
    def __init__(self):
        self.event_buffer = deque(maxlen=100)
    
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single event"""
        self.event_buffer.append(event)
        
        processed = {
            'event_id': event.get('event_id'),
            'event_type': event.get('type'),
            'timestamp': event.get('timestamp'),
            'data': event.get('data', {}),
            'processed_at': datetime.now().isoformat()
        }
        
        # Extract signals
        if event['type'] == 'page_view':
            processed['signals'] = {
                'interest': event['data'].get('category'),
                'engagement_level': 'browsing'
            }
        elif event['type'] == 'click':
            processed['signals'] = {
                'interest': event['data'].get('item_category'),
                'engagement_level': 'interested',
                'intent': 'exploration'
            }
        elif event['type'] == 'add_to_cart':
            processed['signals'] = {
                'interest': event['data'].get('item_category'),
                'engagement_level': 'high',
                'intent': 'purchase'
            }
        elif event['type'] == 'search':
            processed['signals'] = {
                'query': event['data'].get('query'),
                'engagement_level': 'active',
                'intent': 'seeking'
            }
        
        return processed
    
    def detect_patterns(self, recent_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in recent events"""
        if not recent_events:
            return {}
        
        # Event frequency
        event_types = [e['event_type'] for e in recent_events]
        
        # Detect browsing pattern
        if event_types.count('page_view') > 3:
            pattern = 'heavy_browsing'
        elif event_types.count('search') > 2:
            pattern = 'active_searching'
        elif 'add_to_cart' in event_types:
            pattern = 'purchase_intent'
        elif event_types.count('click') > event_types.count('page_view'):
            pattern = 'engaged_exploration'
        else:
            pattern = 'casual_browsing'
        
        # Calculate velocity
        if len(recent_events) > 1:
            time_span = (datetime.fromisoformat(recent_events[-1]['timestamp']) - 
                        datetime.fromisoformat(recent_events[0]['timestamp'])).total_seconds()
            velocity = len(recent_events) / max(time_span, 1)  # events per second
        else:
            velocity = 0
        
        return {
            'pattern': pattern,
            'velocity': velocity,
            'event_count': len(recent_events)
        }


class RealTimeProfiler:
    """Maintain real-time user profile"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.interests = defaultdict(float)
        self.intent_scores = defaultdict(float)
    
    def update_from_event(self, processed_event: Dict[str, Any]) -> Dict[str, Any]:
        """Update profile from processed event"""
        signals = processed_event.get('signals', {})
        
        # Update interests with decay
        if 'interest' in signals:
            interest = signals['interest']
            if interest:
                self.interests[interest] = self.interests[interest] * 0.9 + 1.0
        
        # Update intent
        if 'intent' in signals:
            intent = signals['intent']
            self.intent_scores[intent] = self.intent_scores[intent] * 0.8 + 1.0
        
        # Build current profile
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        top_interests = sorted(self.interests.items(), key=lambda x: x[1], reverse=True)[:3]
        top_intent = max(self.intent_scores.items(), key=lambda x: x[1])[0] if self.intent_scores else 'unknown'
        
        return {
            'session_duration': round(session_duration, 2),
            'top_interests': [i for i, _ in top_interests],
            'interest_scores': dict(top_interests),
            'current_intent': top_intent,
            'intent_confidence': self.intent_scores.get(top_intent, 0) / sum(self.intent_scores.values()) if self.intent_scores else 0
        }


class RealTimeRecommender:
    """Generate recommendations in real-time"""
    
    def generate_instant_recommendations(self, profile: Dict[str, Any],
                                        recent_pattern: Dict[str, Any],
                                        catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations instantly based on current state"""
        recommendations = []
        
        top_interests = profile.get('top_interests', [])
        current_intent = profile.get('current_intent', 'unknown')
        pattern = recent_pattern.get('pattern', 'casual_browsing')
        
        for item in catalog:
            score = 0
            
            # Interest matching
            if item.get('category') in top_interests:
                position = top_interests.index(item['category'])
                score += (3 - position)  # Higher score for top interests
            
            # Intent matching
            if current_intent == 'purchase' and item.get('available', True):
                score += 2
            elif current_intent == 'exploration' and item.get('popular', False):
                score += 1
            
            # Pattern matching
            if pattern == 'purchase_intent' and item.get('recommended', False):
                score += 2
            elif pattern == 'active_searching' and item.get('searchable', True):
                score += 1
            
            if score > 0:
                recommendations.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'category': item['category'],
                    'score': score,
                    'reason': self._generate_reason(top_interests, current_intent, pattern, item)
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Add real-time metadata
        for rec in recommendations:
            rec['generated_at'] = datetime.now().isoformat()
            rec['freshness'] = 1.0  # Maximum freshness
        
        return recommendations[:5]
    
    def _generate_reason(self, interests: List[str], intent: str, 
                        pattern: str, item: Dict[str, Any]) -> str:
        """Generate explanation for recommendation"""
        reasons = []
        
        if item.get('category') in interests:
            reasons.append(f"matches your interest in {item['category']}")
        
        if intent == 'purchase':
            reasons.append("you seem ready to buy")
        
        if pattern == 'active_searching':
            reasons.append("based on your search activity")
        
        return "; ".join(reasons) if reasons else "popular choice"


class AdaptationEngine:
    """Adapt UI and content in real-time"""
    
    def determine_adaptations(self, profile: Dict[str, Any],
                             pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine what adaptations to apply"""
        adaptations = []
        
        intent = profile.get('current_intent', 'unknown')
        velocity = pattern.get('velocity', 0)
        session_duration = profile.get('session_duration', 0)
        
        # Intent-based adaptations
        if intent == 'purchase':
            adaptations.append({
                'type': 'ui_adaptation',
                'action': 'highlight_buy_buttons',
                'reason': 'purchase intent detected'
            })
            adaptations.append({
                'type': 'content_injection',
                'action': 'show_trust_signals',
                'reason': 'facilitate conversion'
            })
        elif intent == 'exploration':
            adaptations.append({
                'type': 'content_adaptation',
                'action': 'increase_variety',
                'reason': 'user exploring options'
            })
        
        # Velocity-based adaptations
        if velocity > 0.5:  # Fast interaction
            adaptations.append({
                'type': 'ui_adaptation',
                'action': 'quick_access_mode',
                'reason': 'high interaction velocity'
            })
        
        # Session duration adaptations
        if session_duration > 10:
            adaptations.append({
                'type': 'engagement',
                'action': 'show_loyalty_prompt',
                'reason': 'extended session'
            })
        elif session_duration < 2:
            adaptations.append({
                'type': 'engagement',
                'action': 'welcome_orientation',
                'reason': 'new session'
            })
        
        # Pattern-based adaptations
        if pattern.get('pattern') == 'active_searching':
            adaptations.append({
                'type': 'search_enhancement',
                'action': 'improve_filters',
                'reason': 'active search pattern'
            })
        
        return adaptations


class PerformanceMonitor:
    """Monitor real-time personalization performance"""
    
    def __init__(self):
        self.update_count = 0
        self.update_times = []
    
    def track_update(self, update_time_ms: float):
        """Track personalization update"""
        self.update_count += 1
        self.update_times.append(update_time_ms)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.update_times:
            return {}
        
        avg_latency = sum(self.update_times) / len(self.update_times)
        max_latency = max(self.update_times)
        min_latency = min(self.update_times)
        
        return {
            'total_updates': self.update_count,
            'avg_latency_ms': round(avg_latency, 2),
            'max_latency_ms': round(max_latency, 2),
            'min_latency_ms': round(min_latency, 2),
            'updates_per_minute': self.update_count  # Simplified
        }


# Agent functions
def initialize_real_time_session_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Initialize real-time session"""
    
    user_state = {
        'user_id': 'user_rt_001',
        'session_id': 'session_rt_12345',
        'session_start': datetime.now().isoformat(),
        'device': 'mobile',
        'connection': 'wifi'
    }
    
    return {
        **state,
        'user_state': user_state,
        'messages': state['messages'] + [f'Initialized real-time session {user_state["session_id"]}']
    }


def process_event_stream_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Process streaming events"""
    
    processor = EventProcessor()
    
    # Simulate event stream (would be live in production)
    base_time = datetime.now()
    events = [
        {
            'event_id': 'evt_001',
            'type': 'page_view',
            'timestamp': base_time.isoformat(),
            'data': {'page': 'home', 'category': 'electronics'}
        },
        {
            'event_id': 'evt_002',
            'type': 'search',
            'timestamp': (base_time + timedelta(seconds=5)).isoformat(),
            'data': {'query': 'wireless headphones'}
        },
        {
            'event_id': 'evt_003',
            'type': 'click',
            'timestamp': (base_time + timedelta(seconds=8)).isoformat(),
            'data': {'item_id': 'item_001', 'item_category': 'electronics'}
        },
        {
            'event_id': 'evt_004',
            'type': 'page_view',
            'timestamp': (base_time + timedelta(seconds=12)).isoformat(),
            'data': {'page': 'product', 'category': 'electronics'}
        },
        {
            'event_id': 'evt_005',
            'type': 'add_to_cart',
            'timestamp': (base_time + timedelta(seconds=20)).isoformat(),
            'data': {'item_id': 'item_001', 'item_category': 'electronics'}
        },
        {
            'event_id': 'evt_006',
            'type': 'page_view',
            'timestamp': (base_time + timedelta(seconds=25)).isoformat(),
            'data': {'page': 'cart', 'category': 'electronics'}
        }
    ]
    
    processed_events = [processor.process_event(e) for e in events]
    
    return {
        **state,
        'event_stream': processed_events,
        'messages': state['messages'] + [f'Processed {len(processed_events)} streaming events']
    }


def update_real_time_profile_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Update user profile in real-time"""
    
    profiler = RealTimeProfiler()
    
    # Process all events to build profile
    for event in state['event_stream']:
        profile = profiler.update_from_event(event)
    
    # Detect patterns
    processor = EventProcessor()
    patterns = processor.detect_patterns(state['event_stream'])
    
    profile['patterns'] = patterns
    
    return {
        **state,
        'real_time_profile': profile,
        'messages': state['messages'] + [f'Updated real-time profile (intent: {profile.get("current_intent")}, pattern: {patterns.get("pattern")})']
    }


def generate_real_time_recommendations_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Generate recommendations in real-time"""
    
    recommender = RealTimeRecommender()
    
    # Mock catalog
    catalog = [
        {'item_id': 'item_001', 'title': 'Premium Wireless Headphones', 'category': 'electronics', 'available': True, 'recommended': True, 'popular': True},
        {'item_id': 'item_002', 'title': 'Noise Cancelling Earbuds', 'category': 'electronics', 'available': True, 'recommended': True, 'popular': True},
        {'item_id': 'item_003', 'title': 'Bluetooth Speaker', 'category': 'electronics', 'available': True, 'recommended': False, 'popular': True},
        {'item_id': 'item_004', 'title': 'Smart Watch', 'category': 'wearables', 'available': True, 'recommended': True, 'popular': False},
        {'item_id': 'item_005', 'title': 'Phone Case', 'category': 'accessories', 'available': True, 'recommended': False, 'popular': True},
        {'item_id': 'item_006', 'title': 'Laptop Stand', 'category': 'electronics', 'available': True, 'recommended': False, 'popular': False},
    ]
    
    recommendations = recommender.generate_instant_recommendations(
        state['real_time_profile'],
        state['real_time_profile'].get('patterns', {}),
        catalog
    )
    
    return {
        **state,
        'recommendations': recommendations,
        'messages': state['messages'] + [f'Generated {len(recommendations)} real-time recommendations']
    }


def apply_real_time_adaptations_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Apply real-time UI/content adaptations"""
    
    engine = AdaptationEngine()
    
    adaptations = engine.determine_adaptations(
        state['real_time_profile'],
        state['real_time_profile'].get('patterns', {})
    )
    
    return {
        **state,
        'adaptation_actions': adaptations,
        'messages': state['messages'] + [f'Applied {len(adaptations)} real-time adaptations']
    }


def track_personalization_updates_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Track personalization updates"""
    
    monitor = PerformanceMonitor()
    
    # Simulate update tracking
    update_times = [15, 12, 18, 14, 16, 13]  # ms
    for time in update_times:
        monitor.track_update(time)
    
    metrics = monitor.get_metrics()
    
    updates = []
    for i, event in enumerate(state['event_stream']):
        updates.append({
            'update_id': f'upd_{i+1:03d}',
            'triggered_by': event['event_type'],
            'timestamp': event['timestamp'],
            'latency_ms': update_times[min(i, len(update_times)-1)]
        })
    
    return {
        **state,
        'personalization_updates': updates,
        'performance_metrics': metrics,
        'messages': state['messages'] + [f'Tracked {len(updates)} personalization updates']
    }


def analyze_real_time_performance_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Analyze real-time personalization performance"""
    
    analytics = {
        'session_stats': {
            'session_duration': state['real_time_profile'].get('session_duration', 0),
            'events_processed': len(state['event_stream']),
            'updates_triggered': len(state['personalization_updates'])
        },
        'profile_stats': {
            'top_interests': state['real_time_profile'].get('top_interests', []),
            'current_intent': state['real_time_profile'].get('current_intent', 'unknown'),
            'intent_confidence': state['real_time_profile'].get('intent_confidence', 0),
            'detected_pattern': state['real_time_profile'].get('patterns', {}).get('pattern', 'unknown')
        },
        'recommendation_stats': {
            'recommendations_generated': len(state['recommendations']),
            'avg_score': sum(r['score'] for r in state['recommendations']) / len(state['recommendations']) if state['recommendations'] else 0
        },
        'adaptation_stats': {
            'adaptations_applied': len(state['adaptation_actions']),
            'adaptation_types': list(set(a['type'] for a in state['adaptation_actions']))
        },
        'performance_stats': state['performance_metrics']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed real-time personalization performance']
    }


def generate_real_time_personalization_report_agent(state: RealTimePersonalizationState) -> RealTimePersonalizationState:
    """Generate comprehensive real-time personalization report"""
    
    report_lines = [
        "=" * 80,
        "REAL-TIME PERSONALIZATION REPORT",
        "=" * 80,
        "",
        "SESSION OVERVIEW:",
        "-" * 40,
        f"User ID: {state['user_state']['user_id']}",
        f"Session ID: {state['user_state']['session_id']}",
        f"Duration: {state['analytics']['session_stats']['session_duration']:.2f} minutes",
        f"Device: {state['user_state']['device']}",
        "",
        "EVENT STREAM:",
        "-" * 40,
        f"Events Processed: {state['analytics']['session_stats']['events_processed']}",
        f"Updates Triggered: {state['analytics']['session_stats']['updates_triggered']}",
        ""
    ]
    
    for event in state['event_stream'][:5]:
        report_lines.append(f"  • {event['event_type']}: {event['data']} ({event.get('signals', {})})")
    
    report_lines.extend([
        "",
        "REAL-TIME PROFILE:",
        "-" * 40,
        f"Top Interests: {', '.join(state['analytics']['profile_stats']['top_interests'])}",
        f"Current Intent: {state['analytics']['profile_stats']['current_intent']} (confidence: {state['analytics']['profile_stats']['intent_confidence']:.2%})",
        f"Detected Pattern: {state['analytics']['profile_stats']['detected_pattern']}",
        f"Interaction Velocity: {state['real_time_profile'].get('patterns', {}).get('velocity', 0):.3f} events/sec",
        "",
        "REAL-TIME RECOMMENDATIONS:",
        "-" * 40,
        f"Generated: {state['analytics']['recommendation_stats']['recommendations_generated']} recommendations",
        f"Average Score: {state['analytics']['recommendation_stats']['avg_score']:.2f}",
        ""
    ]
    
    for rec in state['recommendations']:
        report_lines.append(f"\n  {rec['item_id']}: {rec['title']}")
        report_lines.append(f"    Category: {rec['category']} | Score: {rec['score']}")
        report_lines.append(f"    Reason: {rec['reason']}")
        report_lines.append(f"    Freshness: {rec['freshness']:.2f}")
    
    report_lines.extend([
        "",
        "",
        "REAL-TIME ADAPTATIONS:",
        "-" * 40,
        f"Adaptations Applied: {state['analytics']['adaptation_stats']['adaptations_applied']}",
        f"Adaptation Types: {', '.join(state['analytics']['adaptation_stats']['adaptation_types'])}",
        ""
    ])
    
    for adapt in state['adaptation_actions']:
        report_lines.append(f"\n  {adapt['type'].upper()}: {adapt['action']}")
        report_lines.append(f"    Reason: {adapt['reason']}")
    
    report_lines.extend([
        "",
        "",
        "PERFORMANCE METRICS:",
        "-" * 40,
        f"Total Updates: {state['performance_metrics'].get('total_updates', 0)}",
        f"Average Latency: {state['performance_metrics'].get('avg_latency_ms', 0):.2f}ms",
        f"Max Latency: {state['performance_metrics'].get('max_latency_ms', 0):.2f}ms",
        f"Min Latency: {state['performance_metrics'].get('min_latency_ms', 0):.2f}ms",
        f"Updates per Minute: {state['performance_metrics'].get('updates_per_minute', 0)}",
        "",
        "PERSONALIZATION UPDATES:",
        "-" * 40
    ])
    
    for update in state['personalization_updates'][:5]:
        report_lines.append(f"  • {update['update_id']}: triggered by {update['triggered_by']} ({update['latency_ms']}ms)")
    
    report_lines.extend([
        "",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Real-time profile updated {len(state['event_stream'])} times",
        f"✓ Average personalization latency: {state['performance_metrics'].get('avg_latency_ms', 0):.1f}ms (excellent)",
        f"✓ Intent detected as '{state['analytics']['profile_stats']['current_intent']}' with {state['analytics']['profile_stats']['intent_confidence']:.0%} confidence",
        f"✓ {len(state['adaptation_actions'])} UI/content adaptations applied instantly",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Monitor event processing latency (target <50ms)",
        "• Implement event buffering for burst handling",
        "• Use streaming analytics for pattern detection",
        "• Cache frequently accessed recommendations",
        "• Implement A/B testing for real-time strategies",
        "• Add predictive pre-computation",
        "• Use WebSockets for instant updates",
        "• Implement circuit breakers for resilience",
        "",
        "=" * 80
    ]
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive real-time personalization report']
    }


# Create the graph
def create_real_time_personalization_graph():
    """Create the real-time personalization workflow graph"""
    
    workflow = StateGraph(RealTimePersonalizationState)
    
    # Add nodes
    workflow.add_node("initialize_session", initialize_real_time_session_agent)
    workflow.add_node("process_events", process_event_stream_agent)
    workflow.add_node("update_profile", update_real_time_profile_agent)
    workflow.add_node("generate_recommendations", generate_real_time_recommendations_agent)
    workflow.add_node("apply_adaptations", apply_real_time_adaptations_agent)
    workflow.add_node("track_updates", track_personalization_updates_agent)
    workflow.add_node("analyze_performance", analyze_real_time_performance_agent)
    workflow.add_node("generate_report", generate_real_time_personalization_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize_session")
    workflow.add_edge("initialize_session", "process_events")
    workflow.add_edge("process_events", "update_profile")
    workflow.add_edge("update_profile", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "apply_adaptations")
    workflow.add_edge("apply_adaptations", "track_updates")
    workflow.add_edge("track_updates", "analyze_performance")
    workflow.add_edge("analyze_performance", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the real-time personalization graph
    app = create_real_time_personalization_graph()
    
    # Initialize state
    initial_state: RealTimePersonalizationState = {
        'messages': [],
        'event_stream': [],
        'user_state': {},
        'real_time_profile': {},
        'personalization_updates': [],
        'recommendations': [],
        'adaptation_actions': [],
        'performance_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("REAL-TIME PERSONALIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nReal-time personalization pattern execution complete! ✓")

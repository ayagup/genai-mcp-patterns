"""
Dynamic Personalization MCP Pattern

This pattern demonstrates dynamic personalization in an agentic MCP system.
The system adapts personalization strategies in real-time based on changing
contexts, user states, and environmental factors.

Use cases:
- Context-aware recommendations
- Real-time adaptation
- Session-based personalization
- Dynamic content adjustment
- Adaptive user experiences
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from collections import defaultdict
import math


# Define the state for dynamic personalization
class DynamicPersonalizationState(TypedDict):
    """State for tracking dynamic personalization process"""
    messages: Annotated[List[str], add]
    user_session: Dict[str, Any]
    context_signals: List[Dict[str, Any]]
    personalization_strategy: Dict[str, Any]
    dynamic_adjustments: List[Dict[str, Any]]
    content_variants: List[Dict[str, Any]]
    selected_content: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class SessionAnalyzer:
    """Analyze user session for dynamic personalization"""
    
    def analyze_session_intent(self, session_data: Dict[str, Any]) -> str:
        """Determine user's current intent"""
        actions = session_data.get('actions', [])
        
        if not actions:
            return 'exploration'
        
        # Analyze action patterns
        action_types = [a['type'] for a in actions]
        
        if action_types.count('search') > 2:
            return 'searching'
        elif action_types.count('purchase') > 0:
            return 'transactional'
        elif action_types.count('view') > 3:
            return 'browsing'
        elif action_types.count('compare') > 1:
            return 'comparing'
        else:
            return 'exploration'
    
    def calculate_session_engagement(self, session_data: Dict[str, Any]) -> float:
        """Calculate engagement score for current session"""
        duration = session_data.get('duration_minutes', 0)
        actions = len(session_data.get('actions', []))
        interactions = session_data.get('interaction_count', 0)
        
        # Weighted score
        engagement = (
            (duration * 0.3) +
            (actions * 2.0) +
            (interactions * 1.5)
        )
        
        # Normalize to 0-100
        return min(engagement / 50.0 * 100, 100)
    
    def detect_session_phase(self, session_data: Dict[str, Any]) -> str:
        """Detect which phase of session user is in"""
        duration = session_data.get('duration_minutes', 0)
        actions = len(session_data.get('actions', []))
        
        if duration < 2 and actions < 3:
            return 'entry'
        elif duration < 10 and actions < 10:
            return 'active'
        elif session_data.get('last_action_minutes_ago', 0) > 5:
            return 'cooling'
        else:
            return 'engaged'


class ContextMonitor:
    """Monitor contextual signals for dynamic adaptation"""
    
    def capture_context_signals(self, session: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Capture all relevant contextual signals"""
        signals = []
        
        # Time-based signals
        current_hour = datetime.now().hour
        signals.append({
            'type': 'temporal',
            'signal': 'time_of_day',
            'value': 'business_hours' if 9 <= current_hour < 17 else 'off_hours',
            'weight': 0.7
        })
        
        # Device signals
        device_type = session.get('device_type', 'desktop')
        signals.append({
            'type': 'device',
            'signal': 'device_type',
            'value': device_type,
            'weight': 0.9
        })
        
        # Network signals
        connection_speed = session.get('connection_speed', 'fast')
        signals.append({
            'type': 'network',
            'signal': 'connection_quality',
            'value': connection_speed,
            'weight': 0.6
        })
        
        # Location signals
        location = session.get('location', {})
        if location:
            signals.append({
                'type': 'location',
                'signal': 'user_location',
                'value': location.get('type', 'unknown'),
                'weight': 0.8
            })
        
        # Behavioral signals
        intent = session.get('intent', 'exploration')
        signals.append({
            'type': 'behavioral',
            'signal': 'user_intent',
            'value': intent,
            'weight': 1.0
        })
        
        return signals
    
    def detect_context_changes(self, previous_signals: List[Dict[str, Any]],
                               current_signals: List[Dict[str, Any]]) -> List[str]:
        """Detect significant context changes"""
        changes = []
        
        prev_dict = {s['signal']: s['value'] for s in previous_signals}
        curr_dict = {s['signal']: s['value'] for s in current_signals}
        
        for signal, value in curr_dict.items():
            if signal in prev_dict and prev_dict[signal] != value:
                changes.append(f"{signal}: {prev_dict[signal]} -> {value}")
        
        return changes


class DynamicStrategySelector:
    """Select personalization strategy dynamically"""
    
    def select_strategy(self, session: Dict[str, Any],
                       signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select optimal personalization strategy"""
        intent = session.get('intent', 'exploration')
        engagement = session.get('engagement_score', 50)
        phase = session.get('session_phase', 'entry')
        
        # Strategy selection logic
        if intent == 'transactional':
            strategy = {
                'name': 'conversion_focused',
                'priority': 'high',
                'tactics': ['show_promotions', 'reduce_friction', 'highlight_trust_signals'],
                'content_style': 'direct',
                'recommendation_count': 3
            }
        elif intent == 'searching':
            strategy = {
                'name': 'discovery_oriented',
                'priority': 'medium',
                'tactics': ['diverse_results', 'filter_support', 'related_searches'],
                'content_style': 'informative',
                'recommendation_count': 5
            }
        elif engagement > 70:
            strategy = {
                'name': 'engagement_maximization',
                'priority': 'high',
                'tactics': ['personalized_content', 'deep_recommendations', 'interactive_elements'],
                'content_style': 'engaging',
                'recommendation_count': 7
            }
        elif phase == 'cooling':
            strategy = {
                'name': 'retention_focused',
                'priority': 'critical',
                'tactics': ['re_engagement', 'value_proposition', 'limited_time_offers'],
                'content_style': 'urgent',
                'recommendation_count': 3
            }
        else:
            strategy = {
                'name': 'exploration_support',
                'priority': 'medium',
                'tactics': ['broad_recommendations', 'category_highlights', 'trending_items'],
                'content_style': 'inspirational',
                'recommendation_count': 5
            }
        
        return strategy
    
    def adjust_strategy_for_context(self, strategy: Dict[str, Any],
                                    signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust strategy based on contextual signals"""
        adjusted = strategy.copy()
        
        # Adjust for device
        device_signal = next((s for s in signals if s['signal'] == 'device_type'), None)
        if device_signal and device_signal['value'] == 'mobile':
            adjusted['content_style'] = 'concise'
            adjusted['recommendation_count'] = max(3, adjusted['recommendation_count'] - 2)
        
        # Adjust for connection
        connection_signal = next((s for s in signals if s['signal'] == 'connection_quality'), None)
        if connection_signal and connection_signal['value'] == 'slow':
            adjusted['tactics'].append('lightweight_content')
            adjusted['tactics'].append('lazy_loading')
        
        return adjusted


class ContentAdapter:
    """Adapt content dynamically based on strategy"""
    
    def generate_content_variants(self, base_content: Dict[str, Any],
                                  strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple content variants"""
        variants = []
        
        content_style = strategy.get('content_style', 'standard')
        
        # Generate variants based on style
        if content_style == 'direct':
            variants.append({
                'variant_id': 'direct_v1',
                'title': f"BUY NOW: {base_content['title']}",
                'description': f"Limited offer! {base_content['description'][:100]}",
                'cta': 'Buy Now',
                'urgency': 'high'
            })
        elif content_style == 'informative':
            variants.append({
                'variant_id': 'info_v1',
                'title': base_content['title'],
                'description': base_content['description'],
                'cta': 'Learn More',
                'urgency': 'low'
            })
        elif content_style == 'engaging':
            variants.append({
                'variant_id': 'engage_v1',
                'title': f"✨ {base_content['title']}",
                'description': f"Discover: {base_content['description']}",
                'cta': 'Explore',
                'urgency': 'medium'
            })
        elif content_style == 'urgent':
            variants.append({
                'variant_id': 'urgent_v1',
                'title': f"⚡ Don't Miss: {base_content['title']}",
                'description': f"Hurry! {base_content['description'][:80]}",
                'cta': 'Act Now',
                'urgency': 'critical'
            })
        else:
            variants.append({
                'variant_id': 'standard_v1',
                'title': base_content['title'],
                'description': base_content['description'],
                'cta': 'View Details',
                'urgency': 'low'
            })
        
        return variants
    
    def select_optimal_variant(self, variants: List[Dict[str, Any]],
                              user_history: Dict[str, Any]) -> Dict[str, Any]:
        """Select best variant based on user history"""
        if not variants:
            return {}
        
        # Simple selection based on past performance
        preferred_urgency = user_history.get('preferred_urgency', 'medium')
        
        for variant in variants:
            if variant.get('urgency') == preferred_urgency:
                return variant
        
        return variants[0]


class PerformanceTracker:
    """Track performance of dynamic personalization"""
    
    def track_interaction(self, content_id: str, interaction_type: str,
                         timestamp: datetime) -> Dict[str, Any]:
        """Track user interaction with personalized content"""
        return {
            'content_id': content_id,
            'interaction_type': interaction_type,
            'timestamp': timestamp.isoformat(),
            'value': 1.0 if interaction_type in ['click', 'purchase'] else 0.5
        }
    
    def calculate_effectiveness(self, interactions: List[Dict[str, Any]],
                               total_impressions: int) -> Dict[str, float]:
        """Calculate effectiveness metrics"""
        if total_impressions == 0:
            return {'ctr': 0, 'engagement_rate': 0, 'conversion_rate': 0}
        
        clicks = sum(1 for i in interactions if i['interaction_type'] == 'click')
        engagements = sum(1 for i in interactions if i['interaction_type'] in ['click', 'view', 'share'])
        conversions = sum(1 for i in interactions if i['interaction_type'] == 'purchase')
        
        return {
            'ctr': clicks / total_impressions,
            'engagement_rate': engagements / total_impressions,
            'conversion_rate': conversions / total_impressions
        }


# Agent functions
def initialize_session_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Initialize user session"""
    
    # Simulate user session
    session = {
        'session_id': 'session_12345',
        'user_id': 'user_001',
        'start_time': datetime.now() - timedelta(minutes=15),
        'duration_minutes': 15,
        'device_type': 'mobile',
        'connection_speed': 'fast',
        'location': {'type': 'home', 'city': 'San Francisco'},
        'actions': [
            {'type': 'view', 'item': 'product_a', 'duration': 30},
            {'type': 'search', 'query': 'wireless headphones'},
            {'type': 'view', 'item': 'product_b', 'duration': 45},
            {'type': 'compare', 'items': ['product_a', 'product_b']},
            {'type': 'view', 'item': 'product_c', 'duration': 60}
        ],
        'interaction_count': 12,
        'last_action_minutes_ago': 2
    }
    
    # Analyze session
    analyzer = SessionAnalyzer()
    session['intent'] = analyzer.analyze_session_intent(session)
    session['engagement_score'] = analyzer.calculate_session_engagement(session)
    session['session_phase'] = analyzer.detect_session_phase(session)
    
    return {
        **state,
        'user_session': session,
        'messages': state['messages'] + [f'Initialized session {session["session_id"]} (intent: {session["intent"]}, engagement: {session["engagement_score"]:.1f})']
    }


def capture_context_signals_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Capture contextual signals"""
    
    monitor = ContextMonitor()
    signals = monitor.capture_context_signals(state['user_session'])
    
    return {
        **state,
        'context_signals': signals,
        'messages': state['messages'] + [f'Captured {len(signals)} contextual signals']
    }


def select_personalization_strategy_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Select dynamic personalization strategy"""
    
    selector = DynamicStrategySelector()
    
    # Select base strategy
    strategy = selector.select_strategy(
        state['user_session'],
        state['context_signals']
    )
    
    # Adjust for context
    adjusted_strategy = selector.adjust_strategy_for_context(
        strategy,
        state['context_signals']
    )
    
    return {
        **state,
        'personalization_strategy': adjusted_strategy,
        'messages': state['messages'] + [f'Selected strategy: {adjusted_strategy["name"]} (priority: {adjusted_strategy["priority"]})']
    }


def apply_dynamic_adjustments_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Apply dynamic adjustments based on real-time signals"""
    
    adjustments = []
    strategy = state['personalization_strategy']
    
    # Apply tactics from strategy
    for tactic in strategy.get('tactics', []):
        if tactic == 'show_promotions':
            adjustments.append({
                'type': 'content_injection',
                'action': 'inject_promotion_banner',
                'reason': 'conversion_focused strategy'
            })
        elif tactic == 'diverse_results':
            adjustments.append({
                'type': 'ranking_adjustment',
                'action': 'increase_diversity',
                'reason': 'discovery_oriented strategy'
            })
        elif tactic == 're_engagement':
            adjustments.append({
                'type': 'notification',
                'action': 'trigger_retention_popup',
                'reason': 'user in cooling phase'
            })
        elif tactic == 'lightweight_content':
            adjustments.append({
                'type': 'content_optimization',
                'action': 'compress_media',
                'reason': 'slow connection detected'
            })
    
    return {
        **state,
        'dynamic_adjustments': adjustments,
        'messages': state['messages'] + [f'Applied {len(adjustments)} dynamic adjustments']
    }


def generate_personalized_content_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Generate personalized content variants"""
    
    adapter = ContentAdapter()
    
    # Base content items
    base_items = [
        {
            'item_id': 'product_001',
            'title': 'Premium Wireless Headphones',
            'description': 'High-quality audio experience with advanced noise cancellation and 30-hour battery life'
        },
        {
            'item_id': 'product_002',
            'title': 'Smart Fitness Watch',
            'description': 'Track your health and fitness goals with advanced sensors and GPS'
        },
        {
            'item_id': 'product_003',
            'title': 'Portable Bluetooth Speaker',
            'description': 'Powerful sound in a compact design, perfect for on-the-go listening'
        }
    ]
    
    # Generate variants for each item
    all_variants = []
    for item in base_items:
        variants = adapter.generate_content_variants(
            item,
            state['personalization_strategy']
        )
        all_variants.extend(variants)
    
    return {
        **state,
        'content_variants': all_variants,
        'messages': state['messages'] + [f'Generated {len(all_variants)} content variants']
    }


def select_optimal_content_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Select optimal content for user"""
    
    adapter = ContentAdapter()
    
    # User history (simulated)
    user_history = {
        'preferred_urgency': 'high',  # Based on past interactions
        'click_rate': 0.15
    }
    
    # Select best variant
    selected = []
    for i in range(min(state['personalization_strategy']['recommendation_count'], len(state['content_variants']))):
        if i < len(state['content_variants']):
            variant = state['content_variants'][i]
            selected.append({
                **variant,
                'position': i + 1,
                'personalization_score': 0.85 - (i * 0.1)  # Decreasing score
            })
    
    return {
        **state,
        'selected_content': selected,
        'messages': state['messages'] + [f'Selected {len(selected)} optimal content items']
    }


def track_performance_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Track performance of dynamic personalization"""
    
    tracker = PerformanceTracker()
    
    # Simulate interactions
    interactions = [
        tracker.track_interaction('product_001', 'view', datetime.now()),
        tracker.track_interaction('product_001', 'click', datetime.now()),
        tracker.track_interaction('product_002', 'view', datetime.now()),
        tracker.track_interaction('product_003', 'view', datetime.now()),
        tracker.track_interaction('product_003', 'click', datetime.now()),
    ]
    
    total_impressions = len(state['selected_content'])
    metrics = tracker.calculate_effectiveness(interactions, total_impressions)
    
    return {
        **state,
        'performance_metrics': {
            'impressions': total_impressions,
            'interactions': len(interactions),
            'ctr': metrics['ctr'],
            'engagement_rate': metrics['engagement_rate'],
            'conversion_rate': metrics['conversion_rate']
        },
        'messages': state['messages'] + [f'Tracked performance: CTR={metrics["ctr"]:.2%}, Engagement={metrics["engagement_rate"]:.2%}']
    }


def analyze_dynamic_personalization_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Analyze dynamic personalization effectiveness"""
    
    analytics = {
        'session_stats': {
            'intent': state['user_session']['intent'],
            'engagement_score': state['user_session']['engagement_score'],
            'session_phase': state['user_session']['session_phase'],
            'duration': state['user_session']['duration_minutes']
        },
        'strategy_stats': {
            'strategy_name': state['personalization_strategy']['name'],
            'tactics_applied': len(state['personalization_strategy']['tactics']),
            'adjustments_made': len(state['dynamic_adjustments'])
        },
        'content_stats': {
            'variants_generated': len(state['content_variants']),
            'content_selected': len(state['selected_content']),
            'avg_personalization_score': sum(c['personalization_score'] for c in state['selected_content']) / len(state['selected_content']) if state['selected_content'] else 0
        },
        'performance_stats': state['performance_metrics']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed dynamic personalization effectiveness']
    }


def generate_dynamic_personalization_report_agent(state: DynamicPersonalizationState) -> DynamicPersonalizationState:
    """Generate comprehensive dynamic personalization report"""
    
    report_lines = [
        "=" * 80,
        "DYNAMIC PERSONALIZATION REPORT",
        "=" * 80,
        "",
        "SESSION OVERVIEW:",
        "-" * 40,
        f"Session ID: {state['user_session']['session_id']}",
        f"User ID: {state['user_session']['user_id']}",
        f"Duration: {state['user_session']['duration_minutes']} minutes",
        f"Device: {state['user_session']['device_type']}",
        f"Connection: {state['user_session']['connection_speed']}",
        "",
        "SESSION ANALYSIS:",
        "-" * 40,
        f"Detected Intent: {state['analytics']['session_stats']['intent']}",
        f"Engagement Score: {state['analytics']['session_stats']['engagement_score']:.1f}/100",
        f"Session Phase: {state['analytics']['session_stats']['session_phase']}",
        f"Total Actions: {len(state['user_session']['actions'])}",
        "",
        "CONTEXTUAL SIGNALS:",
        "-" * 40
    ]
    
    for signal in state['context_signals']:
        report_lines.append(f"  • {signal['type'].upper()}: {signal['signal']} = {signal['value']} (weight: {signal['weight']})")
    
    report_lines.extend([
        "",
        "PERSONALIZATION STRATEGY:",
        "-" * 40,
        f"Strategy: {state['personalization_strategy']['name']}",
        f"Priority: {state['personalization_strategy']['priority']}",
        f"Content Style: {state['personalization_strategy']['content_style']}",
        f"Recommendation Count: {state['personalization_strategy']['recommendation_count']}",
        "",
        "Applied Tactics:"
    ])
    
    for tactic in state['personalization_strategy']['tactics']:
        report_lines.append(f"  • {tactic}")
    
    report_lines.extend([
        "",
        "DYNAMIC ADJUSTMENTS:",
        "-" * 40
    ])
    
    for adj in state['dynamic_adjustments']:
        report_lines.append(f"\n  Type: {adj['type']}")
        report_lines.append(f"  Action: {adj['action']}")
        report_lines.append(f"  Reason: {adj['reason']}")
    
    report_lines.extend([
        "",
        "",
        "PERSONALIZED CONTENT:",
        "-" * 40,
        f"Variants Generated: {state['analytics']['content_stats']['variants_generated']}",
        f"Content Selected: {state['analytics']['content_stats']['content_selected']}",
        f"Avg Personalization Score: {state['analytics']['content_stats']['avg_personalization_score']:.2f}",
        ""
    ])
    
    for content in state['selected_content']:
        report_lines.append(f"\n{content['position']}. {content['title']}")
        report_lines.append(f"   Description: {content['description']}")
        report_lines.append(f"   CTA: {content['cta']} | Urgency: {content['urgency']}")
        report_lines.append(f"   Personalization Score: {content['personalization_score']:.2f}")
    
    report_lines.extend([
        "",
        "",
        "PERFORMANCE METRICS:",
        "-" * 40,
        f"Impressions: {state['performance_metrics']['impressions']}",
        f"Interactions: {state['performance_metrics']['interactions']}",
        f"Click-Through Rate: {state['performance_metrics']['ctr']:.2%}",
        f"Engagement Rate: {state['performance_metrics']['engagement_rate']:.2%}",
        f"Conversion Rate: {state['performance_metrics']['conversion_rate']:.2%}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Strategy '{state['personalization_strategy']['name']}' applied based on {state['user_session']['intent']} intent",
        f"✓ {len(state['dynamic_adjustments'])} real-time adjustments made",
        f"✓ Content adapted to {state['personalization_strategy']['content_style']} style",
        f"✓ {state['performance_metrics']['ctr']:.1%} CTR achieved with dynamic personalization",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Continue monitoring session phase for retention triggers",
        "• A/B test content variants to optimize performance",
        "• Adjust strategy weights based on conversion data",
        "• Implement predictive intent detection",
        "• Add more contextual signals (weather, events)",
        "• Use reinforcement learning for strategy selection",
        "• Implement multi-armed bandit for content selection",
        "• Add real-time feedback loops for instant adaptation",
        "",
        "=" * 80
    ]
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive dynamic personalization report']
    }


# Create the graph
def create_dynamic_personalization_graph():
    """Create the dynamic personalization workflow graph"""
    
    workflow = StateGraph(DynamicPersonalizationState)
    
    # Add nodes
    workflow.add_node("initialize_session", initialize_session_agent)
    workflow.add_node("capture_signals", capture_context_signals_agent)
    workflow.add_node("select_strategy", select_personalization_strategy_agent)
    workflow.add_node("apply_adjustments", apply_dynamic_adjustments_agent)
    workflow.add_node("generate_content", generate_personalized_content_agent)
    workflow.add_node("select_content", select_optimal_content_agent)
    workflow.add_node("track_performance", track_performance_agent)
    workflow.add_node("analyze_results", analyze_dynamic_personalization_agent)
    workflow.add_node("generate_report", generate_dynamic_personalization_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize_session")
    workflow.add_edge("initialize_session", "capture_signals")
    workflow.add_edge("capture_signals", "select_strategy")
    workflow.add_edge("select_strategy", "apply_adjustments")
    workflow.add_edge("apply_adjustments", "generate_content")
    workflow.add_edge("generate_content", "select_content")
    workflow.add_edge("select_content", "track_performance")
    workflow.add_edge("track_performance", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the dynamic personalization graph
    app = create_dynamic_personalization_graph()
    
    # Initialize state
    initial_state: DynamicPersonalizationState = {
        'messages': [],
        'user_session': {},
        'context_signals': [],
        'personalization_strategy': {},
        'dynamic_adjustments': [],
        'content_variants': [],
        'selected_content': [],
        'performance_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("DYNAMIC PERSONALIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nDynamic personalization pattern execution complete! ✓")

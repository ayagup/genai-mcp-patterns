"""
Contextual Personalization MCP Pattern

This pattern demonstrates contextual personalization in an agentic MCP system.
The system adapts content and recommendations based on the user's current
context, including location, time, device, activity, and environmental factors.

Use cases:
- Location-based personalization
- Time-sensitive recommendations
- Device-specific adaptation
- Activity-aware content
- Environmental context adaptation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta


# Define the state for contextual personalization
class ContextualPersonalizationState(TypedDict):
    """State for tracking contextual personalization process"""
    messages: Annotated[List[str], add]
    user_context: Dict[str, Any]
    context_history: List[Dict[str, Any]]
    personalization_rules: List[Dict[str, Any]]
    personalized_content: List[Dict[str, Any]]
    context_transitions: List[Dict[str, Any]]
    analytics: Dict[str, Any]
    report: str


class ContextExtractor:
    """Extract and enrich user context"""
    
    def extract_temporal_context(self, timestamp: datetime) -> Dict[str, Any]:
        """Extract time-based context"""
        hour = timestamp.hour
        day_of_week = timestamp.strftime('%A')
        
        # Determine time of day
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
        
        # Determine day type
        day_type = 'weekend' if day_of_week in ['Saturday', 'Sunday'] else 'weekday'
        
        return {
            'hour': hour,
            'day_of_week': day_of_week,
            'time_of_day': time_of_day,
            'day_type': day_type,
            'is_business_hours': day_type == 'weekday' and 9 <= hour < 17
        }
    
    def extract_location_context(self, location_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract location-based context"""
        return {
            'country': location_data.get('country', 'unknown'),
            'city': location_data.get('city', 'unknown'),
            'timezone': location_data.get('timezone', 'UTC'),
            'location_type': location_data.get('type', 'unknown'),  # home, work, travel
            'weather': location_data.get('weather', 'unknown'),
            'is_traveling': location_data.get('is_traveling', False)
        }
    
    def extract_device_context(self, device_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract device-based context"""
        return {
            'device_type': device_data.get('type', 'unknown'),  # mobile, desktop, tablet
            'screen_size': device_data.get('screen_size', 'unknown'),
            'connection_type': device_data.get('connection', 'unknown'),  # wifi, mobile, offline
            'battery_level': device_data.get('battery', 100),
            'is_low_bandwidth': device_data.get('connection') == 'mobile'
        }
    
    def extract_activity_context(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract user activity context"""
        return {
            'current_activity': activity_data.get('activity', 'unknown'),  # browsing, working, commuting
            'focus_level': activity_data.get('focus', 'medium'),  # low, medium, high
            'multitasking': activity_data.get('multitasking', False),
            'session_duration': activity_data.get('session_duration', 0),
            'interaction_count': activity_data.get('interaction_count', 0)
        }


class ContextualPersonalizer:
    """Personalize content based on context"""
    
    def personalize_for_time(self, content: Dict[str, Any], 
                            temporal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize content based on time context"""
        time_of_day = temporal_context['time_of_day']
        
        personalized = content.copy()
        
        if time_of_day == 'morning':
            personalized['greeting'] = 'Good morning!'
            personalized['suggested_content_type'] = 'news_briefing'
            personalized['tone'] = 'energetic'
        elif time_of_day == 'afternoon':
            personalized['greeting'] = 'Good afternoon!'
            personalized['suggested_content_type'] = 'productivity_tips'
            personalized['tone'] = 'professional'
        elif time_of_day == 'evening':
            personalized['greeting'] = 'Good evening!'
            personalized['suggested_content_type'] = 'entertainment'
            personalized['tone'] = 'relaxed'
        else:  # night
            personalized['greeting'] = 'Still up?'
            personalized['suggested_content_type'] = 'light_reading'
            personalized['tone'] = 'calm'
        
        return personalized
    
    def personalize_for_location(self, content: Dict[str, Any],
                                 location_context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize content based on location"""
        personalized = content.copy()
        
        if location_context['is_traveling']:
            personalized['show_local_recommendations'] = True
            personalized['language_suggestions'] = True
            personalized['currency_conversion'] = True
        
        if location_context['weather'] in ['rain', 'snow']:
            personalized['indoor_activities'] = True
        
        location_type = location_context['location_type']
        if location_type == 'work':
            personalized['work_related_content'] = True
            personalized['notifications_muted'] = True
        elif location_type == 'home':
            personalized['personal_content'] = True
            personalized['full_notifications'] = True
        
        return personalized
    
    def personalize_for_device(self, content: Dict[str, Any],
                               device_context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize content based on device"""
        personalized = content.copy()
        
        device_type = device_context['device_type']
        
        if device_type == 'mobile':
            personalized['layout'] = 'mobile_optimized'
            personalized['media_quality'] = 'compressed'
            personalized['max_content_length'] = 'short'
        elif device_type == 'tablet':
            personalized['layout'] = 'tablet_optimized'
            personalized['media_quality'] = 'standard'
            personalized['max_content_length'] = 'medium'
        else:  # desktop
            personalized['layout'] = 'full_width'
            personalized['media_quality'] = 'high'
            personalized['max_content_length'] = 'long'
        
        if device_context['is_low_bandwidth']:
            personalized['media_quality'] = 'compressed'
            personalized['lazy_loading'] = True
        
        if device_context['battery_level'] < 20:
            personalized['power_saving_mode'] = True
            personalized['reduce_animations'] = True
        
        return personalized
    
    def personalize_for_activity(self, content: Dict[str, Any],
                                 activity_context: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize content based on activity"""
        personalized = content.copy()
        
        activity = activity_context['current_activity']
        focus_level = activity_context['focus_level']
        
        if activity == 'commuting':
            personalized['audio_version'] = True
            personalized['offline_available'] = True
            personalized['brief_format'] = True
        elif activity == 'working':
            personalized['work_relevant'] = True
            personalized['minimal_distractions'] = True
        elif activity == 'browsing':
            personalized['exploration_mode'] = True
            personalized['diverse_content'] = True
        
        if focus_level == 'low':
            personalized['interactive_content'] = True
            personalized['short_form'] = True
        elif focus_level == 'high':
            personalized['deep_content'] = True
            personalized['no_interruptions'] = True
        
        if activity_context['multitasking']:
            personalized['quick_access'] = True
            personalized['resumable'] = True
        
        return personalized


class ContextTransitionDetector:
    """Detect transitions between contexts"""
    
    def detect_transition(self, previous_context: Dict[str, Any],
                         current_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect significant context changes"""
        transitions = []
        
        # Location transition
        if previous_context.get('location', {}).get('location_type') != current_context.get('location', {}).get('location_type'):
            transitions.append({
                'type': 'location',
                'from': previous_context.get('location', {}).get('location_type'),
                'to': current_context.get('location', {}).get('location_type')
            })
        
        # Time transition
        prev_time = previous_context.get('temporal', {}).get('time_of_day')
        curr_time = current_context.get('temporal', {}).get('time_of_day')
        if prev_time != curr_time:
            transitions.append({
                'type': 'temporal',
                'from': prev_time,
                'to': curr_time
            })
        
        # Device transition
        prev_device = previous_context.get('device', {}).get('device_type')
        curr_device = current_context.get('device', {}).get('device_type')
        if prev_device != curr_device:
            transitions.append({
                'type': 'device',
                'from': prev_device,
                'to': curr_device
            })
        
        # Activity transition
        prev_activity = previous_context.get('activity', {}).get('current_activity')
        curr_activity = current_context.get('activity', {}).get('current_activity')
        if prev_activity != curr_activity:
            transitions.append({
                'type': 'activity',
                'from': prev_activity,
                'to': curr_activity
            })
        
        return transitions if transitions else None


# Agent functions
def extract_user_context_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Extract current user context"""
    
    extractor = ContextExtractor()
    current_time = datetime.now()
    
    # Simulate context data
    temporal_context = extractor.extract_temporal_context(current_time)
    
    location_context = extractor.extract_location_context({
        'country': 'USA',
        'city': 'San Francisco',
        'timezone': 'America/Los_Angeles',
        'type': 'work',
        'weather': 'sunny',
        'is_traveling': False
    })
    
    device_context = extractor.extract_device_context({
        'type': 'mobile',
        'screen_size': 'small',
        'connection': 'wifi',
        'battery': 45
    })
    
    activity_context = extractor.extract_activity_context({
        'activity': 'commuting',
        'focus': 'medium',
        'multitasking': True,
        'session_duration': 15,
        'interaction_count': 5
    })
    
    user_context = {
        'timestamp': current_time.isoformat(),
        'temporal': temporal_context,
        'location': location_context,
        'device': device_context,
        'activity': activity_context
    }
    
    return {
        **state,
        'user_context': user_context,
        'messages': state['messages'] + ['Extracted comprehensive user context']
    }


def track_context_history_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Track context changes over time"""
    
    # Simulate historical contexts
    base_time = datetime.now() - timedelta(hours=8)
    
    context_history = [
        # Morning at home
        {
            'timestamp': (base_time).isoformat(),
            'temporal': {'time_of_day': 'morning'},
            'location': {'location_type': 'home'},
            'device': {'device_type': 'mobile'},
            'activity': {'current_activity': 'browsing'}
        },
        # Commute to work
        {
            'timestamp': (base_time + timedelta(hours=1)).isoformat(),
            'temporal': {'time_of_day': 'morning'},
            'location': {'location_type': 'transit'},
            'device': {'device_type': 'mobile'},
            'activity': {'current_activity': 'commuting'}
        },
        # At work
        {
            'timestamp': (base_time + timedelta(hours=2)).isoformat(),
            'temporal': {'time_of_day': 'morning'},
            'location': {'location_type': 'work'},
            'device': {'device_type': 'desktop'},
            'activity': {'current_activity': 'working'}
        },
        # Lunch break
        {
            'timestamp': (base_time + timedelta(hours=5)).isoformat(),
            'temporal': {'time_of_day': 'afternoon'},
            'location': {'location_type': 'restaurant'},
            'device': {'device_type': 'mobile'},
            'activity': {'current_activity': 'browsing'}
        },
        # Current (commuting home)
        state['user_context']
    ]
    
    return {
        **state,
        'context_history': context_history,
        'messages': state['messages'] + [f'Tracked {len(context_history)} context snapshots']
    }


def define_personalization_rules_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Define contextual personalization rules"""
    
    personalization_rules = [
        {
            'rule_id': 'CP001',
            'name': 'Morning News Briefing',
            'condition': 'temporal.time_of_day == "morning" AND location.location_type == "home"',
            'action': 'Show curated news briefing',
            'priority': 'high'
        },
        {
            'rule_id': 'CP002',
            'name': 'Commute Content',
            'condition': 'activity.current_activity == "commuting"',
            'action': 'Provide audio-friendly, resumable content',
            'priority': 'high'
        },
        {
            'rule_id': 'CP003',
            'name': 'Work Focus Mode',
            'condition': 'location.location_type == "work" AND temporal.is_business_hours',
            'action': 'Minimize distractions, show work-relevant content',
            'priority': 'high'
        },
        {
            'rule_id': 'CP004',
            'name': 'Mobile Optimization',
            'condition': 'device.device_type == "mobile"',
            'action': 'Optimize layout for mobile, compress media',
            'priority': 'medium'
        },
        {
            'rule_id': 'CP005',
            'name': 'Low Battery Mode',
            'condition': 'device.battery_level < 20',
            'action': 'Enable power saving, reduce animations',
            'priority': 'high'
        },
        {
            'rule_id': 'CP006',
            'name': 'Evening Relaxation',
            'condition': 'temporal.time_of_day == "evening" AND location.location_type == "home"',
            'action': 'Show entertainment and leisure content',
            'priority': 'medium'
        },
        {
            'rule_id': 'CP007',
            'name': 'Travel Assistance',
            'condition': 'location.is_traveling',
            'action': 'Show local recommendations, enable translation',
            'priority': 'high'
        },
        {
            'rule_id': 'CP008',
            'name': 'Multitasking Support',
            'condition': 'activity.multitasking',
            'action': 'Enable quick-access, make content resumable',
            'priority': 'medium'
        }
    ]
    
    return {
        **state,
        'personalization_rules': personalization_rules,
        'messages': state['messages'] + [f'Defined {len(personalization_rules)} personalization rules']
    }


def apply_contextual_personalization_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Apply contextual personalization"""
    
    personalizer = ContextualPersonalizer()
    context = state['user_context']
    
    # Base content
    base_content = {
        'title': 'Welcome to the App',
        'content_type': 'general',
        'priority': 'normal'
    }
    
    # Apply all personalizations
    personalized = base_content.copy()
    personalized = personalizer.personalize_for_time(personalized, context['temporal'])
    personalized = personalizer.personalize_for_location(personalized, context['location'])
    personalized = personalizer.personalize_for_device(personalized, context['device'])
    personalized = personalizer.personalize_for_activity(personalized, context['activity'])
    
    personalized_content = [{
        'base_content': base_content,
        'personalized_content': personalized,
        'context_applied': ['temporal', 'location', 'device', 'activity']
    }]
    
    return {
        **state,
        'personalized_content': personalized_content,
        'messages': state['messages'] + ['Applied contextual personalization']
    }


def detect_context_transitions_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Detect context transitions"""
    
    detector = ContextTransitionDetector()
    transitions = []
    
    # Detect transitions between historical contexts
    for i in range(1, len(state['context_history'])):
        prev = state['context_history'][i-1]
        curr = state['context_history'][i]
        
        detected = detector.detect_transition(prev, curr)
        if detected:
            transitions.extend([{
                'from_timestamp': prev.get('timestamp'),
                'to_timestamp': curr.get('timestamp'),
                **trans
            } for trans in detected])
    
    return {
        **state,
        'context_transitions': transitions,
        'messages': state['messages'] + [f'Detected {len(transitions)} context transitions']
    }


def analyze_context_patterns_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Analyze contextual patterns"""
    
    # Analyze transition patterns
    transition_types = {}
    for trans in state['context_transitions']:
        trans_type = trans['type']
        transition_types[trans_type] = transition_types.get(trans_type, 0) + 1
    
    # Context diversity
    unique_activities = set(c.get('activity', {}).get('current_activity') 
                          for c in state['context_history'])
    unique_locations = set(c.get('location', {}).get('location_type') 
                         for c in state['context_history'])
    
    analytics = {
        'total_context_snapshots': len(state['context_history']),
        'total_transitions': len(state['context_transitions']),
        'transition_types': transition_types,
        'context_diversity': {
            'unique_activities': len(unique_activities),
            'unique_locations': len(unique_locations)
        },
        'personalization_rules_applicable': sum(1 for rule in state['personalization_rules'] 
                                               if rule['priority'] == 'high')
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed contextual patterns']
    }


def generate_contextual_personalization_report_agent(state: ContextualPersonalizationState) -> ContextualPersonalizationState:
    """Generate comprehensive contextual personalization report"""
    
    report_lines = [
        "=" * 80,
        "CONTEXTUAL PERSONALIZATION REPORT",
        "=" * 80,
        "",
        "CURRENT CONTEXT:",
        "-" * 40,
        f"Time: {state['user_context']['temporal']['time_of_day']} ({state['user_context']['temporal']['day_type']})",
        f"Location: {state['user_context']['location']['location_type']} in {state['user_context']['location']['city']}",
        f"Device: {state['user_context']['device']['device_type']} (battery: {state['user_context']['device']['battery_level']}%)",
        f"Activity: {state['user_context']['activity']['current_activity']} (focus: {state['user_context']['activity']['focus_level']})",
        "",
        "CONTEXT HISTORY:",
        "-" * 40,
        f"Total snapshots: {state['analytics']['total_context_snapshots']}",
        f"Unique activities: {state['analytics']['context_diversity']['unique_activities']}",
        f"Unique locations: {state['analytics']['context_diversity']['unique_locations']}",
        "",
        "CONTEXT TRANSITIONS:",
        "-" * 40,
        f"Total transitions: {state['analytics']['total_transitions']}"
    ]
    
    for trans_type, count in state['analytics']['transition_types'].items():
        report_lines.append(f"  {trans_type}: {count} transitions")
    
    report_lines.extend([
        "",
        "RECENT TRANSITIONS:",
        "-" * 40
    ])
    
    for trans in state['context_transitions'][-5:]:
        report_lines.append(f"• {trans['type'].upper()}: {trans.get('from', 'N/A')} → {trans.get('to', 'N/A')}")
    
    report_lines.extend([
        "",
        "PERSONALIZATION RULES:",
        "-" * 40
    ])
    
    for rule in state['personalization_rules']:
        report_lines.append(f"\n{rule['rule_id']}: {rule['name']} (Priority: {rule['priority']})")
        report_lines.append(f"  Condition: {rule['condition']}")
        report_lines.append(f"  Action: {rule['action']}")
    
    report_lines.extend([
        "",
        "PERSONALIZED CONTENT EXAMPLE:",
        "-" * 40
    ])
    
    if state['personalized_content']:
        example = state['personalized_content'][0]
        report_lines.append("\nBase Content:")
        for key, value in example['base_content'].items():
            report_lines.append(f"  {key}: {value}")
        
        report_lines.append("\nPersonalized Content:")
        for key, value in example['personalized_content'].items():
            if key not in example['base_content']:
                report_lines.append(f"  + {key}: {value}")
        
        report_lines.append(f"\nContext Applied: {', '.join(example['context_applied'])}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most common transition: {max(state['analytics']['transition_types'].items(), key=lambda x: x[1])[0] if state['analytics']['transition_types'] else 'N/A'}",
        f"✓ High-priority rules applicable: {state['analytics']['personalization_rules_applicable']}",
        f"✓ Context changes detected: {state['analytics']['total_transitions']}",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement predictive context switching based on patterns",
        "• Pre-load content for anticipated contexts",
        "• Smooth transitions between context states",
        "• Allow manual context override when needed",
        "• Cache context-specific preferences",
        "• Use geofencing for location-based triggers",
        "• Implement gradual personalization intensity",
        "• Provide context-awareness transparency to users",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive contextual personalization report']
    }


# Create the graph
def create_contextual_personalization_graph():
    """Create the contextual personalization workflow graph"""
    
    workflow = StateGraph(ContextualPersonalizationState)
    
    # Add nodes
    workflow.add_node("extract_context", extract_user_context_agent)
    workflow.add_node("track_history", track_context_history_agent)
    workflow.add_node("define_rules", define_personalization_rules_agent)
    workflow.add_node("apply_personalization", apply_contextual_personalization_agent)
    workflow.add_node("detect_transitions", detect_context_transitions_agent)
    workflow.add_node("analyze_patterns", analyze_context_patterns_agent)
    workflow.add_node("generate_report", generate_contextual_personalization_report_agent)
    
    # Add edges
    workflow.add_edge(START, "extract_context")
    workflow.add_edge("extract_context", "track_history")
    workflow.add_edge("track_history", "define_rules")
    workflow.add_edge("define_rules", "apply_personalization")
    workflow.add_edge("apply_personalization", "detect_transitions")
    workflow.add_edge("detect_transitions", "analyze_patterns")
    workflow.add_edge("analyze_patterns", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the contextual personalization graph
    app = create_contextual_personalization_graph()
    
    # Initialize state
    initial_state: ContextualPersonalizationState = {
        'messages': [],
        'user_context': {},
        'context_history': [],
        'personalization_rules': [],
        'personalized_content': [],
        'context_transitions': [],
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("CONTEXTUAL PERSONALIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nContextual personalization pattern execution complete! ✓")

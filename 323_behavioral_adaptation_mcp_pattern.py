"""
Behavioral Adaptation MCP Pattern

This pattern demonstrates behavioral adaptation in an agentic MCP system.
The system adapts its behavior and responses based on observed user
behavior patterns and interaction styles.

Use cases:
- Interaction style adaptation
- Response personalization
- Communication tone adjustment
- Complexity level adaptation
- Timing and frequency optimization
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
from collections import Counter


# Define the state for behavioral adaptation
class BehavioralAdaptationState(TypedDict):
    """State for tracking behavioral adaptation process"""
    messages: Annotated[List[str], add]
    user_behaviors: List[Dict[str, Any]]
    behavior_patterns: Dict[str, Any]
    adaptation_rules: List[Dict[str, Any]]
    adapted_responses: List[Dict[str, Any]]
    effectiveness_metrics: Dict[str, Any]
    report: str


class BehaviorAnalyzer:
    """Analyze user behavior patterns"""
    
    def analyze_interaction_speed(self, behaviors: List[Dict[str, Any]]) -> str:
        """Determine user's interaction speed"""
        response_times = [b.get('response_time', 0) for b in behaviors if 'response_time' in b]
        
        if not response_times:
            return 'unknown'
        
        avg_time = sum(response_times) / len(response_times)
        
        if avg_time < 2:
            return 'very_fast'
        elif avg_time < 5:
            return 'fast'
        elif avg_time < 15:
            return 'moderate'
        else:
            return 'slow'
    
    def analyze_detail_preference(self, behaviors: List[Dict[str, Any]]) -> str:
        """Determine user's preference for detail level"""
        detail_signals = []
        
        for behavior in behaviors:
            if behavior.get('type') == 'expand_detail':
                detail_signals.append('high')
            elif behavior.get('type') == 'collapse_detail':
                detail_signals.append('low')
            elif behavior.get('type') == 'tldr_click':
                detail_signals.append('low')
            elif behavior.get('type') == 'read_full':
                detail_signals.append('high')
        
        if not detail_signals:
            return 'moderate'
        
        detail_counter = Counter(detail_signals)
        return detail_counter.most_common(1)[0][0]
    
    def analyze_communication_style(self, behaviors: List[Dict[str, Any]]) -> str:
        """Determine preferred communication style"""
        style_signals = []
        
        for behavior in behaviors:
            if behavior.get('type') == 'like_formal':
                style_signals.append('formal')
            elif behavior.get('type') == 'like_casual':
                style_signals.append('casual')
            elif behavior.get('type') == 'prefer_technical':
                style_signals.append('technical')
            elif behavior.get('type') == 'prefer_simple':
                style_signals.append('simple')
        
        if not style_signals:
            return 'balanced'
        
        style_counter = Counter(style_signals)
        return style_counter.most_common(1)[0][0]
    
    def analyze_time_preferences(self, behaviors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze when user is most active"""
        active_hours = []
        active_days = []
        
        for behavior in behaviors:
            timestamp = behavior.get('timestamp')
            if timestamp:
                dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
                active_hours.append(dt.hour)
                active_days.append(dt.strftime('%A'))
        
        if not active_hours:
            return {'preferred_hours': [], 'preferred_days': []}
        
        hour_counter = Counter(active_hours)
        day_counter = Counter(active_days)
        
        return {
            'preferred_hours': [h for h, _ in hour_counter.most_common(3)],
            'preferred_days': [d for d, _ in day_counter.most_common(2)],
            'peak_hour': hour_counter.most_common(1)[0][0] if hour_counter else None
        }
    
    def analyze_error_tolerance(self, behaviors: List[Dict[str, Any]]) -> str:
        """Determine user's error tolerance"""
        error_reactions = []
        
        for behavior in behaviors:
            if behavior.get('type') == 'report_error':
                error_reactions.append('low_tolerance')
            elif behavior.get('type') == 'ignore_error':
                error_reactions.append('high_tolerance')
            elif behavior.get('type') == 'retry_after_error':
                error_reactions.append('moderate_tolerance')
        
        if not error_reactions:
            return 'unknown'
        
        reaction_counter = Counter(error_reactions)
        return reaction_counter.most_common(1)[0][0]


class BehaviorAdaptationEngine:
    """Adapt system behavior based on user patterns"""
    
    def adapt_response_length(self, user_detail_pref: str, base_response: str) -> str:
        """Adapt response length to user preference"""
        if user_detail_pref == 'low':
            # Shorten response
            sentences = base_response.split('.')
            return '. '.join(sentences[:2]) + '.'
        elif user_detail_pref == 'high':
            # Expand response with more details
            return base_response + " Additionally, here are more details..."
        else:
            return base_response
    
    def adapt_tone(self, user_style: str, message: str) -> str:
        """Adapt communication tone"""
        if user_style == 'formal':
            return f"Dear User, {message} Respectfully,"
        elif user_style == 'casual':
            return f"Hey! {message} Cheers!"
        elif user_style == 'technical':
            return f"[TECHNICAL] {message}"
        else:
            return message
    
    def adapt_timing(self, user_time_prefs: Dict[str, Any], 
                     current_time: datetime) -> Dict[str, Any]:
        """Adapt notification timing"""
        preferred_hours = user_time_prefs.get('preferred_hours', [])
        current_hour = current_time.hour
        
        if preferred_hours and current_hour not in preferred_hours:
            # Suggest better time
            next_preferred = min(preferred_hours, 
                               key=lambda h: ((h - current_hour) % 24))
            return {
                'should_send_now': False,
                'suggested_hour': next_preferred,
                'reason': 'User typically inactive at this hour'
            }
        else:
            return {
                'should_send_now': True,
                'suggested_hour': current_hour,
                'reason': 'User typically active at this hour'
            }
    
    def adapt_complexity(self, user_style: str, content: str) -> str:
        """Adapt content complexity"""
        if user_style == 'simple':
            return f"[SIMPLIFIED] {content}"
        elif user_style == 'technical':
            return f"[TECHNICAL DEPTH] {content}"
        else:
            return content
    
    def adapt_error_handling(self, error_tolerance: str, error_msg: str) -> Dict[str, Any]:
        """Adapt error presentation based on tolerance"""
        if error_tolerance == 'low_tolerance':
            return {
                'show_error': True,
                'detail_level': 'high',
                'suggestion': 'Provide detailed error and multiple recovery options'
            }
        elif error_tolerance == 'high_tolerance':
            return {
                'show_error': True,
                'detail_level': 'low',
                'suggestion': 'Brief error message, auto-retry'
            }
        else:
            return {
                'show_error': True,
                'detail_level': 'moderate',
                'suggestion': 'Standard error handling'
            }


# Agent functions
def collect_user_behaviors_agent(state: BehavioralAdaptationState) -> BehavioralAdaptationState:
    """Collect user behavior data"""
    
    base_time = datetime.now() - timedelta(days=7)
    
    user_behaviors = [
        # Fast interaction speed
        {'timestamp': (base_time + timedelta(hours=10)).isoformat(), 'type': 'response', 'response_time': 1.5},
        {'timestamp': (base_time + timedelta(hours=10, minutes=5)).isoformat(), 'type': 'response', 'response_time': 2.0},
        {'timestamp': (base_time + timedelta(hours=10, minutes=10)).isoformat(), 'type': 'response', 'response_time': 1.8},
        
        # Preference for concise content
        {'timestamp': (base_time + timedelta(days=1, hours=14)).isoformat(), 'type': 'tldr_click', 'item': 'article_1'},
        {'timestamp': (base_time + timedelta(days=1, hours=14, minutes=5)).isoformat(), 'type': 'collapse_detail', 'item': 'section_2'},
        
        # Technical communication style
        {'timestamp': (base_time + timedelta(days=2, hours=9)).isoformat(), 'type': 'prefer_technical', 'content': 'api_docs'},
        {'timestamp': (base_time + timedelta(days=2, hours=9, minutes=30)).isoformat(), 'type': 'like_formal', 'content': 'documentation'},
        
        # Active in mornings and evenings
        {'timestamp': (base_time + timedelta(days=3, hours=8)).isoformat(), 'type': 'session_start'},
        {'timestamp': (base_time + timedelta(days=3, hours=19)).isoformat(), 'type': 'session_start'},
        {'timestamp': (base_time + timedelta(days=4, hours=7)).isoformat(), 'type': 'session_start'},
        {'timestamp': (base_time + timedelta(days=4, hours=20)).isoformat(), 'type': 'session_start'},
        
        # Low error tolerance
        {'timestamp': (base_time + timedelta(days=5, hours=11)).isoformat(), 'type': 'report_error', 'error_id': 'err_001'},
        {'timestamp': (base_time + timedelta(days=5, hours=11, minutes=2)).isoformat(), 'type': 'report_error', 'error_id': 'err_002'},
        
        # More detail preference signals
        {'timestamp': (base_time + timedelta(days=6, hours=15)).isoformat(), 'type': 'response', 'response_time': 1.2},
        {'timestamp': (base_time + timedelta(days=6, hours=16)).isoformat(), 'type': 'tldr_click', 'item': 'article_5'},
        
        # Weekday preference
        {'timestamp': (base_time + timedelta(days=7, hours=9)).isoformat(), 'type': 'session_start'}
    ]
    
    return {
        **state,
        'user_behaviors': user_behaviors,
        'messages': state['messages'] + [f'Collected {len(user_behaviors)} user behavior signals']
    }


def analyze_behavior_patterns_agent(state: BehavioralAdaptationState) -> BehavioralAdaptationState:
    """Analyze behavior patterns"""
    
    analyzer = BehaviorAnalyzer()
    
    behavior_patterns = {
        'interaction_speed': analyzer.analyze_interaction_speed(state['user_behaviors']),
        'detail_preference': analyzer.analyze_detail_preference(state['user_behaviors']),
        'communication_style': analyzer.analyze_communication_style(state['user_behaviors']),
        'time_preferences': analyzer.analyze_time_preferences(state['user_behaviors']),
        'error_tolerance': analyzer.analyze_error_tolerance(state['user_behaviors'])
    }
    
    return {
        **state,
        'behavior_patterns': behavior_patterns,
        'messages': state['messages'] + [f'Analyzed {len(behavior_patterns)} behavior patterns']
    }


def create_adaptation_rules_agent(state: BehavioralAdaptationState) -> BehavioralAdaptationState:
    """Create adaptation rules based on patterns"""
    
    patterns = state['behavior_patterns']
    
    adaptation_rules = [
        {
            'rule_id': 'AR001',
            'name': 'Response Length Adaptation',
            'condition': f"detail_preference == '{patterns['detail_preference']}'",
            'action': f"Adjust response length to {patterns['detail_preference']} detail",
            'priority': 'high'
        },
        {
            'rule_id': 'AR002',
            'name': 'Communication Tone Adaptation',
            'condition': f"communication_style == '{patterns['communication_style']}'",
            'action': f"Use {patterns['communication_style']} communication style",
            'priority': 'high'
        },
        {
            'rule_id': 'AR003',
            'name': 'Response Timing Optimization',
            'condition': 'notification_needed',
            'action': f"Send during preferred hours: {patterns['time_preferences'].get('preferred_hours', [])}",
            'priority': 'medium'
        },
        {
            'rule_id': 'AR004',
            'name': 'Interaction Speed Matching',
            'condition': f"interaction_speed == '{patterns['interaction_speed']}'",
            'action': f"Match user's {patterns['interaction_speed']} interaction pace",
            'priority': 'medium'
        },
        {
            'rule_id': 'AR005',
            'name': 'Error Handling Adaptation',
            'condition': f"error_tolerance == '{patterns['error_tolerance']}'",
            'action': f"Adapt error handling for {patterns['error_tolerance']} users",
            'priority': 'high'
        }
    ]
    
    return {
        **state,
        'adaptation_rules': adaptation_rules,
        'messages': state['messages'] + [f'Created {len(adaptation_rules)} adaptation rules']
    }


def apply_behavioral_adaptations_agent(state: BehavioralAdaptationState) -> BehavioralAdaptationState:
    """Apply behavioral adaptations to sample scenarios"""
    
    engine = BehaviorAdaptationEngine()
    patterns = state['behavior_patterns']
    adapted_responses = []
    
    # Scenario 1: Standard response
    base_response = "Your request has been processed. The system analyzed your data and generated a report. You can download it from the dashboard."
    adapted_length = engine.adapt_response_length(patterns['detail_preference'], base_response)
    adapted_tone = engine.adapt_tone(patterns['communication_style'], adapted_length)
    
    adapted_responses.append({
        'scenario': 'Standard Response',
        'original': base_response,
        'adapted': adapted_tone,
        'adaptations_applied': ['length', 'tone']
    })
    
    # Scenario 2: Notification timing
    current_time = datetime.now().replace(hour=23, minute=0)  # Late night
    timing_adaptation = engine.adapt_timing(patterns['time_preferences'], current_time)
    
    adapted_responses.append({
        'scenario': 'Notification Timing',
        'current_hour': current_time.hour,
        'timing_decision': timing_adaptation,
        'adaptations_applied': ['timing']
    })
    
    # Scenario 3: Technical content
    tech_content = "The API endpoint returned a 200 OK status code."
    adapted_complexity = engine.adapt_complexity(patterns['communication_style'], tech_content)
    
    adapted_responses.append({
        'scenario': 'Technical Content',
        'original': tech_content,
        'adapted': adapted_complexity,
        'adaptations_applied': ['complexity']
    })
    
    # Scenario 4: Error handling
    error_msg = "Database connection failed"
    error_adaptation = engine.adapt_error_handling(patterns['error_tolerance'], error_msg)
    
    adapted_responses.append({
        'scenario': 'Error Handling',
        'error': error_msg,
        'adaptation': error_adaptation,
        'adaptations_applied': ['error_presentation']
    })
    
    return {
        **state,
        'adapted_responses': adapted_responses,
        'messages': state['messages'] + [f'Applied adaptations to {len(adapted_responses)} scenarios']
    }


def measure_adaptation_effectiveness_agent(state: BehavioralAdaptationState) -> BehavioralAdaptationState:
    """Measure effectiveness of adaptations"""
    
    # Simulate effectiveness metrics
    effectiveness_metrics = {
        'user_satisfaction': {
            'before_adaptation': 3.2,
            'after_adaptation': 4.5,
            'improvement': 1.3
        },
        'engagement_rate': {
            'before_adaptation': 0.45,
            'after_adaptation': 0.72,
            'improvement': 0.27
        },
        'response_time': {
            'before_adaptation': 5.8,  # seconds
            'after_adaptation': 2.1,
            'improvement': -3.7  # negative is good (faster)
        },
        'error_reports': {
            'before_adaptation': 12,
            'after_adaptation': 4,
            'improvement': -8  # negative is good (fewer errors reported)
        },
        'adaptation_accuracy': {
            'correct_adaptations': 18,
            'total_adaptations': 20,
            'accuracy_rate': 0.90
        }
    }
    
    return {
        **state,
        'effectiveness_metrics': effectiveness_metrics,
        'messages': state['messages'] + ['Measured adaptation effectiveness']
    }


def generate_behavioral_adaptation_report_agent(state: BehavioralAdaptationState) -> BehavioralAdaptationState:
    """Generate comprehensive behavioral adaptation report"""
    
    report_lines = [
        "=" * 80,
        "BEHAVIORAL ADAPTATION REPORT",
        "=" * 80,
        "",
        "ADAPTATION SUMMARY:",
        "-" * 40,
        f"Behavior signals collected: {len(state['user_behaviors'])}",
        f"Patterns identified: {len(state['behavior_patterns'])}",
        f"Adaptation rules created: {len(state['adaptation_rules'])}",
        f"Scenarios adapted: {len(state['adapted_responses'])}",
        "",
        "IDENTIFIED BEHAVIOR PATTERNS:",
        "-" * 40,
        f"• Interaction Speed: {state['behavior_patterns']['interaction_speed']}",
        f"• Detail Preference: {state['behavior_patterns']['detail_preference']}",
        f"• Communication Style: {state['behavior_patterns']['communication_style']}",
        f"• Error Tolerance: {state['behavior_patterns']['error_tolerance']}",
        f"• Preferred Hours: {state['behavior_patterns']['time_preferences'].get('preferred_hours', [])}",
        f"• Preferred Days: {state['behavior_patterns']['time_preferences'].get('preferred_days', [])}",
        "",
        "ADAPTATION RULES:",
        "-" * 40
    ]
    
    for rule in state['adaptation_rules']:
        report_lines.append(f"\n{rule['rule_id']}: {rule['name']}")
        report_lines.append(f"  Priority: {rule['priority']}")
        report_lines.append(f"  Condition: {rule['condition']}")
        report_lines.append(f"  Action: {rule['action']}")
    
    report_lines.extend([
        "",
        "ADAPTATION EXAMPLES:",
        "-" * 40
    ])
    
    for response in state['adapted_responses']:
        report_lines.append(f"\nScenario: {response['scenario']}")
        if 'original' in response:
            report_lines.append(f"  Original: {response['original']}")
            report_lines.append(f"  Adapted: {response['adapted']}")
        elif 'timing_decision' in response:
            decision = response['timing_decision']
            report_lines.append(f"  Current Hour: {response['current_hour']}")
            report_lines.append(f"  Should Send Now: {decision['should_send_now']}")
            report_lines.append(f"  Suggested Hour: {decision['suggested_hour']}")
            report_lines.append(f"  Reason: {decision['reason']}")
        elif 'adaptation' in response:
            adaptation = response['adaptation']
            report_lines.append(f"  Error: {response['error']}")
            report_lines.append(f"  Show Error: {adaptation['show_error']}")
            report_lines.append(f"  Detail Level: {adaptation['detail_level']}")
            report_lines.append(f"  Suggestion: {adaptation['suggestion']}")
        
        report_lines.append(f"  Adaptations Applied: {', '.join(response['adaptations_applied'])}")
    
    report_lines.extend([
        "",
        "EFFECTIVENESS METRICS:",
        "-" * 40
    ])
    
    metrics = state['effectiveness_metrics']
    for metric_name, metric_data in metrics.items():
        if isinstance(metric_data, dict):
            report_lines.append(f"\n{metric_name.replace('_', ' ').title()}:")
            for key, value in metric_data.items():
                if isinstance(value, float):
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    report_lines.append(f"  {key.replace('_', ' ').title()}: {value}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ User prefers {state['behavior_patterns']['detail_preference']} detail responses",
        f"✓ Communication style: {state['behavior_patterns']['communication_style']}",
        f"✓ Adaptation accuracy: {metrics['adaptation_accuracy']['accuracy_rate']*100:.0f}%",
        f"✓ User satisfaction improved by {metrics['user_satisfaction']['improvement']} points",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Continue monitoring behavior patterns for drift detection",
        "• A/B test adaptations to validate effectiveness",
        "• Implement gradual adaptation to avoid jarring changes",
        "• Allow users to provide feedback on adaptations",
        "• Use multi-armed bandit for adaptation strategy selection",
        "• Track long-term impact of behavioral adaptations",
        "• Implement fallback to default behavior if adaptation fails",
        "• Consider contextual factors in adaptation decisions",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive behavioral adaptation report']
    }


# Create the graph
def create_behavioral_adaptation_graph():
    """Create the behavioral adaptation workflow graph"""
    
    workflow = StateGraph(BehavioralAdaptationState)
    
    # Add nodes
    workflow.add_node("collect_behaviors", collect_user_behaviors_agent)
    workflow.add_node("analyze_patterns", analyze_behavior_patterns_agent)
    workflow.add_node("create_rules", create_adaptation_rules_agent)
    workflow.add_node("apply_adaptations", apply_behavioral_adaptations_agent)
    workflow.add_node("measure_effectiveness", measure_adaptation_effectiveness_agent)
    workflow.add_node("generate_report", generate_behavioral_adaptation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "collect_behaviors")
    workflow.add_edge("collect_behaviors", "analyze_patterns")
    workflow.add_edge("analyze_patterns", "create_rules")
    workflow.add_edge("create_rules", "apply_adaptations")
    workflow.add_edge("apply_adaptations", "measure_effectiveness")
    workflow.add_edge("measure_effectiveness", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the behavioral adaptation graph
    app = create_behavioral_adaptation_graph()
    
    # Initialize state
    initial_state: BehavioralAdaptationState = {
        'messages': [],
        'user_behaviors': [],
        'behavior_patterns': {},
        'adaptation_rules': [],
        'adapted_responses': [],
        'effectiveness_metrics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("BEHAVIORAL ADAPTATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nBehavioral adaptation pattern execution complete! ✓")

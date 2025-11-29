"""
User Profile MCP Pattern

This pattern demonstrates user profile management in an agentic MCP system.
User profiles store and manage user-specific information that drives
personalization across the system.

Use cases:
- User attribute management
- Profile creation and updates
- Profile enrichment
- Profile segmentation
- Multi-dimensional user modeling
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta
import json


# Define the state for user profile management
class UserProfileState(TypedDict):
    """State for tracking user profile management"""
    messages: Annotated[List[str], add]
    user_profiles: List[Dict[str, Any]]
    profile_schema: Dict[str, Any]
    enrichment_results: List[Dict[str, Any]]
    segments: Dict[str, List[str]]
    analytics: Dict[str, Any]
    report: str


class UserProfile:
    """Comprehensive user profile model"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.demographics = {}
        self.preferences = {}
        self.behavior = {}
        self.interactions = []
        self.metadata = {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def update_demographics(self, **kwargs):
        """Update demographic information"""
        self.demographics.update(kwargs)
        self.updated_at = datetime.now()
    
    def update_preferences(self, **kwargs):
        """Update user preferences"""
        self.preferences.update(kwargs)
        self.updated_at = datetime.now()
    
    def add_interaction(self, interaction_type: str, data: Dict[str, Any]):
        """Record user interaction"""
        self.interactions.append({
            'type': interaction_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        self.updated_at = datetime.now()
    
    def calculate_engagement_score(self) -> float:
        """Calculate user engagement score"""
        if not self.interactions:
            return 0.0
        
        # Simple engagement calculation based on interaction count and recency
        recent_interactions = [
            i for i in self.interactions
            if datetime.fromisoformat(i['timestamp']) > datetime.now() - timedelta(days=30)
        ]
        
        interaction_score = min(len(recent_interactions) / 50.0, 1.0) * 50
        recency_score = 50 if recent_interactions else 0
        
        return interaction_score + recency_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            'user_id': self.user_id,
            'demographics': self.demographics,
            'preferences': self.preferences,
            'behavior': self.behavior,
            'interaction_count': len(self.interactions),
            'engagement_score': self.calculate_engagement_score(),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }


class ProfileEnricher:
    """Enrich user profiles with derived attributes"""
    
    def enrich_with_segment(self, profile: Dict[str, Any]) -> str:
        """Determine user segment"""
        engagement = profile.get('engagement_score', 0)
        interaction_count = profile.get('interaction_count', 0)
        
        if engagement > 75 and interaction_count > 30:
            return 'power_user'
        elif engagement > 50 and interaction_count > 15:
            return 'active_user'
        elif engagement > 25 and interaction_count > 5:
            return 'casual_user'
        else:
            return 'new_user'
    
    def enrich_with_lifetime_value(self, profile: Dict[str, Any]) -> float:
        """Calculate predicted lifetime value"""
        # Simplified LTV calculation
        engagement = profile.get('engagement_score', 0)
        tenure_days = (datetime.now() - datetime.fromisoformat(profile.get('created_at', datetime.now().isoformat()))).days
        
        base_value = engagement * 10
        tenure_multiplier = min(tenure_days / 365.0, 2.0)
        
        return base_value * (1 + tenure_multiplier)
    
    def enrich_with_churn_risk(self, profile: Dict[str, Any]) -> str:
        """Assess churn risk"""
        engagement = profile.get('engagement_score', 0)
        
        if engagement < 20:
            return 'high'
        elif engagement < 50:
            return 'medium'
        else:
            return 'low'
    
    def enrich_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Add enriched attributes to profile"""
        enriched = profile.copy()
        enriched['segment'] = self.enrich_with_segment(profile)
        enriched['predicted_ltv'] = self.enrich_with_lifetime_value(profile)
        enriched['churn_risk'] = self.enrich_with_churn_risk(profile)
        
        return enriched


class ProfileAnalyzer:
    """Analyze user profile patterns"""
    
    def analyze_segment_distribution(self, profiles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze distribution across segments"""
        segments = {}
        for profile in profiles:
            segment = profile.get('segment', 'unknown')
            segments[segment] = segments.get(segment, 0) + 1
        return segments
    
    def analyze_engagement_stats(self, profiles: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate engagement statistics"""
        if not profiles:
            return {'mean': 0, 'median': 0, 'min': 0, 'max': 0}
        
        scores = [p.get('engagement_score', 0) for p in profiles]
        scores.sort()
        
        return {
            'mean': sum(scores) / len(scores),
            'median': scores[len(scores) // 2],
            'min': min(scores),
            'max': max(scores)
        }
    
    def analyze_churn_risk_distribution(self, profiles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze churn risk distribution"""
        risks = {}
        for profile in profiles:
            risk = profile.get('churn_risk', 'unknown')
            risks[risk] = risks.get(risk, 0) + 1
        return risks


# Agent functions
def initialize_user_profiles_agent(state: UserProfileState) -> UserProfileState:
    """Initialize user profiles"""
    
    # Create sample user profiles
    profiles = []
    
    # Power user
    profile1 = UserProfile('user_001')
    profile1.update_demographics(age=28, gender='F', location='New York', occupation='Software Engineer')
    profile1.update_preferences(language='en', theme='dark', notifications=True, categories=['tech', 'ai', 'programming'])
    for i in range(40):
        profile1.add_interaction('page_view', {'page': f'article_{i}', 'duration': 120 + i*2})
    profiles.append(profile1.to_dict())
    
    # Active user
    profile2 = UserProfile('user_002')
    profile2.update_demographics(age=35, gender='M', location='San Francisco', occupation='Product Manager')
    profile2.update_preferences(language='en', theme='light', notifications=True, categories=['business', 'tech'])
    for i in range(20):
        profile2.add_interaction('page_view', {'page': f'article_{i}', 'duration': 90})
    profiles.append(profile2.to_dict())
    
    # Casual user
    profile3 = UserProfile('user_003')
    profile3.update_demographics(age=42, gender='F', location='Austin', occupation='Teacher')
    profile3.update_preferences(language='en', theme='light', notifications=False, categories=['education', 'news'])
    for i in range(8):
        profile3.add_interaction('page_view', {'page': f'article_{i}', 'duration': 60})
    profiles.append(profile3.to_dict())
    
    # New user
    profile4 = UserProfile('user_004')
    profile4.update_demographics(age=25, gender='M', location='Seattle', occupation='Student')
    profile4.update_preferences(language='en', theme='dark', notifications=True, categories=['science'])
    for i in range(2):
        profile4.add_interaction('page_view', {'page': f'article_{i}', 'duration': 30})
    profiles.append(profile4.to_dict())
    
    # Inactive user (high churn risk)
    profile5 = UserProfile('user_005')
    profile5.update_demographics(age=50, gender='F', location='Boston', occupation='Consultant')
    profile5.update_preferences(language='en', theme='light', notifications=False, categories=['finance'])
    # No recent interactions - high churn risk
    profiles.append(profile5.to_dict())
    
    return {
        **state,
        'user_profiles': profiles,
        'messages': state['messages'] + [f'Initialized {len(profiles)} user profiles']
    }


def define_profile_schema_agent(state: UserProfileState) -> UserProfileState:
    """Define comprehensive profile schema"""
    
    profile_schema = {
        'required_fields': ['user_id', 'created_at'],
        'demographic_fields': ['age', 'gender', 'location', 'occupation', 'education'],
        'preference_fields': ['language', 'theme', 'notifications', 'categories', 'interests'],
        'behavioral_fields': ['interaction_count', 'engagement_score', 'last_active'],
        'derived_fields': ['segment', 'predicted_ltv', 'churn_risk'],
        'metadata_fields': ['created_at', 'updated_at', 'source', 'version']
    }
    
    return {
        **state,
        'profile_schema': profile_schema,
        'messages': state['messages'] + ['Defined comprehensive profile schema']
    }


def enrich_profiles_agent(state: UserProfileState) -> UserProfileState:
    """Enrich profiles with derived attributes"""
    
    enricher = ProfileEnricher()
    enrichment_results = []
    enriched_profiles = []
    
    for profile in state['user_profiles']:
        enriched = enricher.enrich_profile(profile)
        enriched_profiles.append(enriched)
        
        enrichment_results.append({
            'user_id': profile['user_id'],
            'segment': enriched['segment'],
            'predicted_ltv': round(enriched['predicted_ltv'], 2),
            'churn_risk': enriched['churn_risk'],
            'engagement_score': round(enriched['engagement_score'], 2)
        })
    
    return {
        **state,
        'user_profiles': enriched_profiles,
        'enrichment_results': enrichment_results,
        'messages': state['messages'] + [f'Enriched {len(enriched_profiles)} profiles with derived attributes']
    }


def segment_users_agent(state: UserProfileState) -> UserProfileState:
    """Segment users into groups"""
    
    segments = {
        'power_user': [],
        'active_user': [],
        'casual_user': [],
        'new_user': [],
        'at_risk': []
    }
    
    for profile in state['user_profiles']:
        user_id = profile['user_id']
        segment = profile.get('segment', 'unknown')
        churn_risk = profile.get('churn_risk', 'unknown')
        
        # Add to segment
        if segment in segments:
            segments[segment].append(user_id)
        
        # Add to at-risk if high churn risk
        if churn_risk == 'high':
            segments['at_risk'].append(user_id)
    
    segment_summary = {k: len(v) for k, v in segments.items()}
    
    return {
        **state,
        'segments': segments,
        'messages': state['messages'] + [f'Segmented users: {segment_summary}']
    }


def analyze_profiles_agent(state: UserProfileState) -> UserProfileState:
    """Analyze user profiles"""
    
    analyzer = ProfileAnalyzer()
    
    # Segment distribution
    segment_dist = analyzer.analyze_segment_distribution(state['user_profiles'])
    
    # Engagement statistics
    engagement_stats = analyzer.analyze_engagement_stats(state['user_profiles'])
    
    # Churn risk distribution
    churn_dist = analyzer.analyze_churn_risk_distribution(state['user_profiles'])
    
    # LTV statistics
    ltv_values = [p.get('predicted_ltv', 0) for p in state['user_profiles']]
    ltv_stats = {
        'total_ltv': sum(ltv_values),
        'average_ltv': sum(ltv_values) / len(ltv_values) if ltv_values else 0,
        'max_ltv': max(ltv_values) if ltv_values else 0
    }
    
    analytics = {
        'total_users': len(state['user_profiles']),
        'segment_distribution': segment_dist,
        'engagement_statistics': engagement_stats,
        'churn_risk_distribution': churn_dist,
        'ltv_statistics': ltv_stats
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed user profile patterns']
    }


def generate_profile_report_agent(state: UserProfileState) -> UserProfileState:
    """Generate comprehensive user profile report"""
    
    report_lines = [
        "=" * 80,
        "USER PROFILE MANAGEMENT REPORT",
        "=" * 80,
        "",
        "PROFILE OVERVIEW:",
        "-" * 40,
        f"Total users: {state['analytics']['total_users']}",
        "",
        "PROFILE SCHEMA:",
        "-" * 40,
        f"Required fields: {', '.join(state['profile_schema']['required_fields'])}",
        f"Demographic fields: {', '.join(state['profile_schema']['demographic_fields'])}",
        f"Preference fields: {', '.join(state['profile_schema']['preference_fields'])}",
        f"Behavioral fields: {', '.join(state['profile_schema']['behavioral_fields'])}",
        f"Derived fields: {', '.join(state['profile_schema']['derived_fields'])}",
        "",
        "SEGMENT DISTRIBUTION:",
        "-" * 40
    ]
    
    for segment, count in sorted(state['analytics']['segment_distribution'].items(),
                                 key=lambda x: x[1], reverse=True):
        percentage = (count / state['analytics']['total_users'] * 100) if state['analytics']['total_users'] > 0 else 0
        report_lines.append(f"  {segment}: {count} users ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "ENGAGEMENT STATISTICS:",
        "-" * 40,
        f"  Mean engagement: {state['analytics']['engagement_statistics']['mean']:.2f}",
        f"  Median engagement: {state['analytics']['engagement_statistics']['median']:.2f}",
        f"  Min engagement: {state['analytics']['engagement_statistics']['min']:.2f}",
        f"  Max engagement: {state['analytics']['engagement_statistics']['max']:.2f}",
        "",
        "CHURN RISK DISTRIBUTION:",
        "-" * 40
    ])
    
    for risk, count in sorted(state['analytics']['churn_risk_distribution'].items()):
        percentage = (count / state['analytics']['total_users'] * 100) if state['analytics']['total_users'] > 0 else 0
        report_lines.append(f"  {risk} risk: {count} users ({percentage:.1f}%)")
    
    report_lines.extend([
        "",
        "LIFETIME VALUE STATISTICS:",
        "-" * 40,
        f"  Total LTV: ${state['analytics']['ltv_statistics']['total_ltv']:.2f}",
        f"  Average LTV: ${state['analytics']['ltv_statistics']['average_ltv']:.2f}",
        f"  Max LTV: ${state['analytics']['ltv_statistics']['max_ltv']:.2f}",
        "",
        "PROFILE ENRICHMENT RESULTS:",
        "-" * 40
    ])
    
    for result in state['enrichment_results']:
        report_lines.append(f"\n{result['user_id']}:")
        report_lines.append(f"  Segment: {result['segment']}")
        report_lines.append(f"  Engagement: {result['engagement_score']}")
        report_lines.append(f"  Predicted LTV: ${result['predicted_ltv']}")
        report_lines.append(f"  Churn Risk: {result['churn_risk']}")
    
    report_lines.extend([
        "",
        "USER SEGMENTS:",
        "-" * 40
    ])
    
    for segment, users in state['segments'].items():
        if users:
            report_lines.append(f"\n{segment} ({len(users)} users):")
            report_lines.append(f"  Users: {', '.join(users)}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Largest segment: {max(state['analytics']['segment_distribution'].items(), key=lambda x: x[1])[0]}",
        f"✓ Users at risk: {len(state['segments'].get('at_risk', []))}",
        f"✓ Average engagement: {state['analytics']['engagement_statistics']['mean']:.1f}/100",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement re-engagement campaigns for at-risk users",
        "• Develop premium features for power users",
        "• Create onboarding flows for new users",
        "• Personalize content based on user preferences",
        "• Track profile completeness and encourage updates",
        "• Use predictive LTV for resource allocation",
        "• Monitor churn risk indicators in real-time",
        "• Segment-specific communication strategies",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive user profile report']
    }


# Create the graph
def create_user_profile_graph():
    """Create the user profile management workflow graph"""
    
    workflow = StateGraph(UserProfileState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_user_profiles_agent)
    workflow.add_node("define_schema", define_profile_schema_agent)
    workflow.add_node("enrich", enrich_profiles_agent)
    workflow.add_node("segment", segment_users_agent)
    workflow.add_node("analyze", analyze_profiles_agent)
    workflow.add_node("generate_report", generate_profile_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_schema")
    workflow.add_edge("define_schema", "enrich")
    workflow.add_edge("enrich", "segment")
    workflow.add_edge("segment", "analyze")
    workflow.add_edge("analyze", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the user profile graph
    app = create_user_profile_graph()
    
    # Initialize state
    initial_state: UserProfileState = {
        'messages': [],
        'user_profiles': [],
        'profile_schema': {},
        'enrichment_results': [],
        'segments': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("USER PROFILE MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nUser profile pattern execution complete! ✓")

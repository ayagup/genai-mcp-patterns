"""
Segment-Based Personalization MCP Pattern

This pattern demonstrates segment-based personalization in an agentic MCP system.
The system groups users into segments and applies cohort-level personalization
strategies while maintaining individual customization.

Use cases:
- Cohort-based marketing
- Customer segmentation
- Targeted campaigns
- Group-level optimization
- Demographic personalization
"""

from typing import TypedDict, Annotated, List, Dict, Any, Set
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from collections import defaultdict, Counter
import math


# Define the state for segment-based personalization
class SegmentBasedPersonalizationState(TypedDict):
    """State for tracking segment-based personalization process"""
    messages: Annotated[List[str], add]
    user_profiles: List[Dict[str, Any]]
    segments: Dict[str, List[str]]
    segment_profiles: Dict[str, Dict[str, Any]]
    personalization_strategies: Dict[str, Dict[str, Any]]
    segment_recommendations: Dict[str, List[Dict[str, Any]]]
    individual_recommendations: Dict[str, List[Dict[str, Any]]]
    analytics: Dict[str, Any]
    report: str


class UserSegmenter:
    """Segment users based on multiple criteria"""
    
    def segment_by_demographics(self, users: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Segment users by demographic attributes"""
        segments = defaultdict(list)
        
        for user in users:
            user_id = user['user_id']
            age = user.get('age', 0)
            gender = user.get('gender', 'unknown')
            
            # Age-based segments
            if age < 25:
                segments['age_young'].append(user_id)
            elif age < 40:
                segments['age_adult'].append(user_id)
            else:
                segments['age_mature'].append(user_id)
            
            # Gender-based segments
            segments[f'gender_{gender}'].append(user_id)
        
        return dict(segments)
    
    def segment_by_behavior(self, users: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Segment users by behavioral patterns"""
        segments = defaultdict(list)
        
        for user in users:
            user_id = user['user_id']
            engagement = user.get('engagement_score', 0)
            purchase_frequency = user.get('purchase_frequency', 0)
            
            # Engagement-based segments
            if engagement > 75:
                segments['highly_engaged'].append(user_id)
            elif engagement > 40:
                segments['moderately_engaged'].append(user_id)
            else:
                segments['low_engaged'].append(user_id)
            
            # Purchase behavior
            if purchase_frequency > 10:
                segments['frequent_buyer'].append(user_id)
            elif purchase_frequency > 3:
                segments['regular_buyer'].append(user_id)
            elif purchase_frequency > 0:
                segments['occasional_buyer'].append(user_id)
            else:
                segments['non_buyer'].append(user_id)
        
        return dict(segments)
    
    def segment_by_value(self, users: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Segment users by customer value"""
        segments = defaultdict(list)
        
        for user in users:
            user_id = user['user_id']
            ltv = user.get('lifetime_value', 0)
            
            if ltv > 1000:
                segments['high_value'].append(user_id)
            elif ltv > 500:
                segments['medium_value'].append(user_id)
            elif ltv > 0:
                segments['low_value'].append(user_id)
            else:
                segments['potential_value'].append(user_id)
        
        return dict(segments)
    
    def segment_by_interests(self, users: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Segment users by interests"""
        segments = defaultdict(list)
        
        for user in users:
            user_id = user['user_id']
            interests = user.get('interests', [])
            
            if 'technology' in interests:
                segments['tech_enthusiast'].append(user_id)
            if 'sports' in interests:
                segments['sports_fan'].append(user_id)
            if 'fashion' in interests:
                segments['fashion_forward'].append(user_id)
            if 'travel' in interests:
                segments['travel_lover'].append(user_id)
            if 'food' in interests:
                segments['foodie'].append(user_id)
        
        return dict(segments)


class SegmentProfileBuilder:
    """Build aggregate profiles for segments"""
    
    def build_segment_profile(self, segment_name: str,
                             user_ids: List[str],
                             all_users: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build aggregate profile for a segment"""
        segment_users = [u for u in all_users if u['user_id'] in user_ids]
        
        if not segment_users:
            return {}
        
        # Aggregate demographics
        avg_age = sum(u.get('age', 0) for u in segment_users) / len(segment_users)
        gender_dist = Counter(u.get('gender', 'unknown') for u in segment_users)
        
        # Aggregate behavior
        avg_engagement = sum(u.get('engagement_score', 0) for u in segment_users) / len(segment_users)
        avg_purchases = sum(u.get('purchase_frequency', 0) for u in segment_users) / len(segment_users)
        
        # Aggregate value
        avg_ltv = sum(u.get('lifetime_value', 0) for u in segment_users) / len(segment_users)
        total_revenue = sum(u.get('lifetime_value', 0) for u in segment_users)
        
        # Aggregate interests
        all_interests = []
        for u in segment_users:
            all_interests.extend(u.get('interests', []))
        interest_dist = Counter(all_interests)
        top_interests = [i for i, _ in interest_dist.most_common(5)]
        
        # Aggregate preferences
        all_categories = []
        for u in segment_users:
            all_categories.extend(u.get('preferred_categories', []))
        category_dist = Counter(all_categories)
        top_categories = [c for c, _ in category_dist.most_common(5)]
        
        return {
            'segment_name': segment_name,
            'size': len(segment_users),
            'demographics': {
                'avg_age': round(avg_age, 1),
                'gender_distribution': dict(gender_dist)
            },
            'behavior': {
                'avg_engagement': round(avg_engagement, 1),
                'avg_purchases': round(avg_purchases, 1)
            },
            'value': {
                'avg_ltv': round(avg_ltv, 2),
                'total_revenue': round(total_revenue, 2)
            },
            'interests': top_interests,
            'preferred_categories': top_categories
        }


class SegmentStrategyDesigner:
    """Design personalization strategies for segments"""
    
    def design_strategy(self, segment_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Design personalization strategy for a segment"""
        segment_name = segment_profile['segment_name']
        size = segment_profile['size']
        avg_engagement = segment_profile['behavior']['avg_engagement']
        avg_ltv = segment_profile['value']['avg_ltv']
        
        # Strategy design based on segment characteristics
        if 'high_value' in segment_name:
            strategy = {
                'name': 'vip_treatment',
                'objective': 'retention',
                'tactics': ['exclusive_offers', 'priority_support', 'early_access'],
                'content_focus': 'premium',
                'communication_frequency': 'high',
                'promotion_type': 'exclusive'
            }
        elif 'frequent_buyer' in segment_name:
            strategy = {
                'name': 'loyalty_nurture',
                'objective': 'increase_frequency',
                'tactics': ['loyalty_rewards', 'personalized_bundles', 'replenishment_reminders'],
                'content_focus': 'recommendations',
                'communication_frequency': 'medium',
                'promotion_type': 'loyalty_bonus'
            }
        elif 'non_buyer' in segment_name or 'potential' in segment_name:
            strategy = {
                'name': 'conversion_activation',
                'objective': 'first_purchase',
                'tactics': ['first_time_discount', 'social_proof', 'limited_time_offers'],
                'content_focus': 'value_proposition',
                'communication_frequency': 'medium',
                'promotion_type': 'acquisition'
            }
        elif avg_engagement < 40:
            strategy = {
                'name': 'reengagement',
                'objective': 'activation',
                'tactics': ['win_back_campaign', 'personalized_content', 'special_incentives'],
                'content_focus': 'engaging',
                'communication_frequency': 'low',
                'promotion_type': 'reactivation'
            }
        else:
            strategy = {
                'name': 'standard_nurture',
                'objective': 'engagement',
                'tactics': ['personalized_recommendations', 'educational_content', 'seasonal_offers'],
                'content_focus': 'balanced',
                'communication_frequency': 'medium',
                'promotion_type': 'standard'
            }
        
        strategy['segment_size'] = size
        return strategy
    
    def customize_for_interests(self, strategy: Dict[str, Any],
                                interests: List[str]) -> Dict[str, Any]:
        """Customize strategy based on segment interests"""
        customized = strategy.copy()
        
        # Add interest-specific tactics
        interest_tactics = []
        if 'technology' in interests:
            interest_tactics.append('tech_product_highlights')
        if 'fashion' in interests:
            interest_tactics.append('style_recommendations')
        if 'travel' in interests:
            interest_tactics.append('destination_guides')
        
        customized['tactics'].extend(interest_tactics)
        customized['interests'] = interests
        
        return customized


class SegmentRecommender:
    """Generate recommendations for segments"""
    
    def generate_segment_recommendations(self, segment_profile: Dict[str, Any],
                                        strategy: Dict[str, Any],
                                        catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations for entire segment"""
        recommendations = []
        
        top_categories = segment_profile.get('preferred_categories', [])
        interests = segment_profile.get('interests', [])
        
        # Filter catalog based on segment preferences
        for item in catalog:
            score = 0
            
            # Category match
            if item.get('category') in top_categories:
                score += 3
            
            # Interest match
            item_tags = set(item.get('tags', []))
            if item_tags & set(interests):
                score += 2
            
            # Strategy alignment
            if strategy.get('content_focus') == 'premium' and item.get('premium', False):
                score += 2
            
            if score > 0:
                recommendations.append({
                    'item_id': item['item_id'],
                    'title': item['title'],
                    'category': item.get('category'),
                    'segment_score': score,
                    'strategy_alignment': strategy['name']
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['segment_score'], reverse=True)
        return recommendations[:10]
    
    def personalize_for_individual(self, segment_recs: List[Dict[str, Any]],
                                   user_profile: Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Personalize segment recommendations for individual user"""
        personalized = []
        
        user_interests = set(user_profile.get('interests', []))
        user_categories = set(user_profile.get('preferred_categories', []))
        
        for rec in segment_recs:
            # Individual scoring
            individual_score = rec['segment_score']
            
            # Boost if matches user's specific interests
            rec_tags = set(rec.get('tags', []))
            if rec_tags & user_interests:
                individual_score += 1
            
            # Boost if matches user's categories
            if rec.get('category') in user_categories:
                individual_score += 1
            
            personalized.append({
                **rec,
                'individual_score': individual_score,
                'user_id': user_profile['user_id']
            })
        
        personalized.sort(key=lambda x: x['individual_score'], reverse=True)
        return personalized[:5]


# Agent functions
def initialize_user_profiles_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Initialize user profiles"""
    
    # Simulate diverse user profiles
    users = [
        {
            'user_id': 'user_001',
            'age': 28,
            'gender': 'female',
            'engagement_score': 85,
            'purchase_frequency': 12,
            'lifetime_value': 1200,
            'interests': ['technology', 'fashion'],
            'preferred_categories': ['electronics', 'clothing']
        },
        {
            'user_id': 'user_002',
            'age': 35,
            'gender': 'male',
            'engagement_score': 65,
            'purchase_frequency': 6,
            'lifetime_value': 600,
            'interests': ['sports', 'technology'],
            'preferred_categories': ['sports', 'electronics']
        },
        {
            'user_id': 'user_003',
            'age': 42,
            'gender': 'female',
            'engagement_score': 30,
            'purchase_frequency': 2,
            'lifetime_value': 150,
            'interests': ['travel', 'food'],
            'preferred_categories': ['travel', 'books']
        },
        {
            'user_id': 'user_004',
            'age': 22,
            'gender': 'male',
            'engagement_score': 90,
            'purchase_frequency': 15,
            'lifetime_value': 800,
            'interests': ['technology', 'sports'],
            'preferred_categories': ['electronics', 'sports']
        },
        {
            'user_id': 'user_005',
            'age': 50,
            'gender': 'male',
            'engagement_score': 45,
            'purchase_frequency': 8,
            'lifetime_value': 900,
            'interests': ['travel', 'food'],
            'preferred_categories': ['travel', 'home']
        },
        {
            'user_id': 'user_006',
            'age': 26,
            'gender': 'female',
            'engagement_score': 75,
            'purchase_frequency': 0,
            'lifetime_value': 0,
            'interests': ['fashion', 'food'],
            'preferred_categories': ['clothing', 'beauty']
        },
        {
            'user_id': 'user_007',
            'age': 38,
            'gender': 'male',
            'engagement_score': 55,
            'purchase_frequency': 4,
            'lifetime_value': 400,
            'interests': ['technology', 'travel'],
            'preferred_categories': ['electronics', 'travel']
        },
        {
            'user_id': 'user_008',
            'age': 31,
            'gender': 'female',
            'engagement_score': 88,
            'purchase_frequency': 20,
            'lifetime_value': 1500,
            'interests': ['fashion', 'technology'],
            'preferred_categories': ['clothing', 'electronics']
        }
    ]
    
    return {
        **state,
        'user_profiles': users,
        'messages': state['messages'] + [f'Initialized {len(users)} user profiles']
    }


def segment_users_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Segment users based on multiple criteria"""
    
    segmenter = UserSegmenter()
    
    # Apply multiple segmentation strategies
    demographic_segments = segmenter.segment_by_demographics(state['user_profiles'])
    behavioral_segments = segmenter.segment_by_behavior(state['user_profiles'])
    value_segments = segmenter.segment_by_value(state['user_profiles'])
    interest_segments = segmenter.segment_by_interests(state['user_profiles'])
    
    # Combine all segments
    all_segments = {
        **demographic_segments,
        **behavioral_segments,
        **value_segments,
        **interest_segments
    }
    
    return {
        **state,
        'segments': all_segments,
        'messages': state['messages'] + [f'Created {len(all_segments)} user segments']
    }


def build_segment_profiles_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Build profiles for each segment"""
    
    builder = SegmentProfileBuilder()
    segment_profiles = {}
    
    for segment_name, user_ids in state['segments'].items():
        profile = builder.build_segment_profile(
            segment_name,
            user_ids,
            state['user_profiles']
        )
        if profile:
            segment_profiles[segment_name] = profile
    
    return {
        **state,
        'segment_profiles': segment_profiles,
        'messages': state['messages'] + [f'Built profiles for {len(segment_profiles)} segments']
    }


def design_segment_strategies_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Design personalization strategies for each segment"""
    
    designer = SegmentStrategyDesigner()
    strategies = {}
    
    # Design strategy for key segments
    key_segments = ['high_value', 'frequent_buyer', 'non_buyer', 'low_engaged']
    
    for segment_name in key_segments:
        if segment_name in state['segment_profiles']:
            profile = state['segment_profiles'][segment_name]
            strategy = designer.design_strategy(profile)
            
            # Customize based on interests
            if profile.get('interests'):
                strategy = designer.customize_for_interests(
                    strategy,
                    profile['interests']
                )
            
            strategies[segment_name] = strategy
    
    return {
        **state,
        'personalization_strategies': strategies,
        'messages': state['messages'] + [f'Designed strategies for {len(strategies)} key segments']
    }


def generate_segment_recommendations_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Generate recommendations for segments"""
    
    recommender = SegmentRecommender()
    
    # Mock catalog
    catalog = [
        {'item_id': 'item_001', 'title': 'Premium Wireless Headphones', 'category': 'electronics', 'tags': ['technology', 'audio'], 'premium': True},
        {'item_id': 'item_002', 'title': 'Smart Fitness Watch', 'category': 'sports', 'tags': ['sports', 'technology'], 'premium': True},
        {'item_id': 'item_003', 'title': 'Designer Handbag', 'category': 'clothing', 'tags': ['fashion', 'luxury'], 'premium': True},
        {'item_id': 'item_004', 'title': 'Travel Backpack', 'category': 'travel', 'tags': ['travel', 'outdoor'], 'premium': False},
        {'item_id': 'item_005', 'title': 'Gourmet Coffee Set', 'category': 'food', 'tags': ['food', 'gourmet'], 'premium': False},
        {'item_id': 'item_006', 'title': 'Running Shoes', 'category': 'sports', 'tags': ['sports', 'fitness'], 'premium': False},
        {'item_id': 'item_007', 'title': 'Skincare Bundle', 'category': 'beauty', 'tags': ['fashion', 'beauty'], 'premium': False},
        {'item_id': 'item_008', 'title': 'Tablet Device', 'category': 'electronics', 'tags': ['technology', 'productivity'], 'premium': True},
    ]
    
    segment_recs = {}
    
    for segment_name, strategy in state['personalization_strategies'].items():
        if segment_name in state['segment_profiles']:
            profile = state['segment_profiles'][segment_name]
            recs = recommender.generate_segment_recommendations(
                profile,
                strategy,
                catalog
            )
            segment_recs[segment_name] = recs
    
    return {
        **state,
        'segment_recommendations': segment_recs,
        'messages': state['messages'] + [f'Generated recommendations for {len(segment_recs)} segments']
    }


def personalize_for_individuals_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Personalize segment recommendations for individuals"""
    
    recommender = SegmentRecommender()
    individual_recs = {}
    
    # Map users to their primary segment (simplified)
    user_segment_map = {}
    for segment_name, user_ids in state['segments'].items():
        if segment_name in ['high_value', 'frequent_buyer', 'non_buyer', 'low_engaged']:
            for user_id in user_ids:
                if user_id not in user_segment_map:  # First match wins
                    user_segment_map[user_id] = segment_name
    
    # Personalize for each user
    for user in state['user_profiles']:
        user_id = user['user_id']
        segment = user_segment_map.get(user_id)
        
        if segment and segment in state['segment_recommendations']:
            segment_recs = state['segment_recommendations'][segment]
            personalized = recommender.personalize_for_individual(
                segment_recs,
                user
            )
            individual_recs[user_id] = personalized
    
    return {
        **state,
        'individual_recommendations': individual_recs,
        'messages': state['messages'] + [f'Personalized recommendations for {len(individual_recs)} users']
    }


def analyze_segment_personalization_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Analyze segment-based personalization"""
    
    # Segment size distribution
    segment_sizes = {name: profile['size'] 
                    for name, profile in state['segment_profiles'].items()}
    
    # Strategy distribution
    strategy_types = Counter(s['name'] for s in state['personalization_strategies'].values())
    
    # Coverage analysis
    total_users = len(state['user_profiles'])
    users_with_recs = len(state['individual_recommendations'])
    
    analytics = {
        'segmentation_stats': {
            'total_segments': len(state['segments']),
            'profiled_segments': len(state['segment_profiles']),
            'avg_segment_size': sum(segment_sizes.values()) / len(segment_sizes) if segment_sizes else 0
        },
        'strategy_stats': {
            'strategies_designed': len(state['personalization_strategies']),
            'strategy_types': dict(strategy_types)
        },
        'recommendation_stats': {
            'segments_with_recs': len(state['segment_recommendations']),
            'users_with_individual_recs': users_with_recs,
            'coverage': users_with_recs / total_users if total_users > 0 else 0
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed segment-based personalization']
    }


def generate_segment_personalization_report_agent(state: SegmentBasedPersonalizationState) -> SegmentBasedPersonalizationState:
    """Generate comprehensive segment-based personalization report"""
    
    report_lines = [
        "=" * 80,
        "SEGMENT-BASED PERSONALIZATION REPORT",
        "=" * 80,
        "",
        "SEGMENTATION OVERVIEW:",
        "-" * 40,
        f"Total Segments: {state['analytics']['segmentation_stats']['total_segments']}",
        f"Profiled Segments: {state['analytics']['segmentation_stats']['profiled_segments']}",
        f"Average Segment Size: {state['analytics']['segmentation_stats']['avg_segment_size']:.1f} users",
        "",
        "KEY SEGMENTS:",
        "-" * 40
    ]
    
    for segment_name, profile in list(state['segment_profiles'].items())[:5]:
        report_lines.append(f"\n{segment_name.upper().replace('_', ' ')}:")
        report_lines.append(f"  Size: {profile['size']} users")
        report_lines.append(f"  Avg Age: {profile['demographics']['avg_age']}")
        report_lines.append(f"  Avg Engagement: {profile['behavior']['avg_engagement']:.1f}")
        report_lines.append(f"  Avg LTV: ${profile['value']['avg_ltv']:.2f}")
        report_lines.append(f"  Top Interests: {', '.join(profile['interests'][:3])}")
    
    report_lines.extend([
        "",
        "",
        "PERSONALIZATION STRATEGIES:",
        "-" * 40
    ])
    
    for segment_name, strategy in state['personalization_strategies'].items():
        report_lines.append(f"\n{segment_name.upper()} - {strategy['name'].upper()}")
        report_lines.append(f"  Objective: {strategy['objective']}")
        report_lines.append(f"  Tactics: {', '.join(strategy['tactics'][:3])}")
        report_lines.append(f"  Content Focus: {strategy['content_focus']}")
        report_lines.append(f"  Communication Frequency: {strategy['communication_frequency']}")
    
    report_lines.extend([
        "",
        "",
        "SEGMENT RECOMMENDATIONS:",
        "-" * 40
    ])
    
    for segment_name, recs in state['segment_recommendations'].items():
        report_lines.append(f"\n{segment_name.upper()} (Top 3):")
        for rec in recs[:3]:
            report_lines.append(f"  • {rec['title']} (score: {rec['segment_score']})")
    
    report_lines.extend([
        "",
        "",
        "INDIVIDUAL PERSONALIZATION EXAMPLES:",
        "-" * 40
    ])
    
    for user_id, recs in list(state['individual_recommendations'].items())[:2]:
        user = next(u for u in state['user_profiles'] if u['user_id'] == user_id)
        report_lines.append(f"\n{user_id} (Age: {user['age']}, Interests: {', '.join(user['interests'])}):")
        for rec in recs[:3]:
            report_lines.append(f"  • {rec['title']} (individual score: {rec['individual_score']})")
    
    report_lines.extend([
        "",
        "",
        "PERFORMANCE METRICS:",
        "-" * 40,
        f"Segments with Recommendations: {state['analytics']['recommendation_stats']['segments_with_recs']}",
        f"Users with Individual Recs: {state['analytics']['recommendation_stats']['users_with_individual_recs']}",
        f"Coverage: {state['analytics']['recommendation_stats']['coverage']:.1%}",
        f"Strategy Types: {len(state['analytics']['strategy_stats']['strategy_types'])}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ {state['analytics']['segmentation_stats']['total_segments']} segments created across multiple dimensions",
        f"✓ {len(state['personalization_strategies'])} distinct strategies designed",
        f"✓ {state['analytics']['recommendation_stats']['coverage']:.0%} of users received personalized recommendations",
        "✓ Segment-level + individual-level personalization applied",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Monitor segment performance and adjust strategies",
        "• A/B test different strategies within segments",
        "• Refine segmentation criteria based on outcomes",
        "• Implement dynamic segment assignment",
        "• Use lookalike modeling to expand segments",
        "• Apply micro-segmentation for high-value users",
        "• Combine with real-time personalization",
        "• Track segment migration patterns",
        "",
        "=" * 80
    ]
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive segment-based personalization report']
    }


# Create the graph
def create_segment_based_personalization_graph():
    """Create the segment-based personalization workflow graph"""
    
    workflow = StateGraph(SegmentBasedPersonalizationState)
    
    # Add nodes
    workflow.add_node("initialize_profiles", initialize_user_profiles_agent)
    workflow.add_node("segment_users", segment_users_agent)
    workflow.add_node("build_profiles", build_segment_profiles_agent)
    workflow.add_node("design_strategies", design_segment_strategies_agent)
    workflow.add_node("generate_segment_recs", generate_segment_recommendations_agent)
    workflow.add_node("personalize_individual", personalize_for_individuals_agent)
    workflow.add_node("analyze_results", analyze_segment_personalization_agent)
    workflow.add_node("generate_report", generate_segment_personalization_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize_profiles")
    workflow.add_edge("initialize_profiles", "segment_users")
    workflow.add_edge("segment_users", "build_profiles")
    workflow.add_edge("build_profiles", "design_strategies")
    workflow.add_edge("design_strategies", "generate_segment_recs")
    workflow.add_edge("generate_segment_recs", "personalize_individual")
    workflow.add_edge("personalize_individual", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the segment-based personalization graph
    app = create_segment_based_personalization_graph()
    
    # Initialize state
    initial_state: SegmentBasedPersonalizationState = {
        'messages': [],
        'user_profiles': [],
        'segments': {},
        'segment_profiles': {},
        'personalization_strategies': {},
        'segment_recommendations': {},
        'individual_recommendations': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("SEGMENT-BASED PERSONALIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nSegment-based personalization pattern execution complete! ✓")

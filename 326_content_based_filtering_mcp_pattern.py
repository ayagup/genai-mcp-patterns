"""
Content-Based Filtering MCP Pattern

This pattern demonstrates content-based filtering in an agentic MCP system.
The system recommends items based on item features and user preferences,
using TF-IDF, feature matching, and similarity metrics.

Use cases:
- Personalized content recommendations
- Product recommendations based on features
- Article/document recommendations
- Movie/music recommendations
- Job/candidate matching
"""

from typing import TypedDict, Annotated, List, Dict, Any, Set
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from collections import Counter, defaultdict
import math


# Define the state for content-based filtering
class ContentBasedFilteringState(TypedDict):
    """State for tracking content-based filtering process"""
    messages: Annotated[List[str], add]
    items: List[Dict[str, Any]]
    user_profile: Dict[str, Any]
    item_features: Dict[str, Dict[str, float]]
    tfidf_scores: Dict[str, Dict[str, float]]
    user_feature_profile: Dict[str, float]
    recommendations: List[Dict[str, Any]]
    analytics: Dict[str, Any]
    report: str


class FeatureExtractor:
    """Extract features from items"""
    
    def extract_text_features(self, text: str) -> List[str]:
        """Extract features from text (simple tokenization)"""
        # Simple tokenization and normalization
        text = text.lower()
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was'}
        
        words = text.split()
        features = [w.strip('.,!?;:()[]{}') for w in words if w not in stopwords]
        return [f for f in features if len(f) > 2]  # Filter short words
    
    def extract_categorical_features(self, item: Dict[str, Any]) -> List[str]:
        """Extract categorical features"""
        features = []
        
        # Genre/category features
        if 'genres' in item:
            features.extend([f"genre_{g}" for g in item['genres']])
        
        if 'category' in item:
            features.append(f"category_{item['category']}")
        
        if 'tags' in item:
            features.extend([f"tag_{t}" for t in item['tags']])
        
        return features
    
    def extract_numerical_features(self, item: Dict[str, Any]) -> Dict[str, float]:
        """Extract numerical features"""
        features = {}
        
        if 'price' in item:
            # Bucket price into ranges
            price = item['price']
            if price < 10:
                features['price_low'] = 1.0
            elif price < 50:
                features['price_medium'] = 1.0
            else:
                features['price_high'] = 1.0
        
        if 'rating' in item:
            features['rating'] = item['rating'] / 5.0  # Normalize
        
        if 'popularity' in item:
            features['popularity'] = item['popularity'] / 100.0  # Normalize
        
        return features


class TFIDFCalculator:
    """Calculate TF-IDF scores for items"""
    
    def calculate_tf(self, features: List[str]) -> Dict[str, float]:
        """Calculate term frequency"""
        feature_count = Counter(features)
        total_features = len(features)
        
        return {feature: count / total_features 
                for feature, count in feature_count.items()}
    
    def calculate_idf(self, all_item_features: List[List[str]]) -> Dict[str, float]:
        """Calculate inverse document frequency"""
        num_docs = len(all_item_features)
        doc_frequency = Counter()
        
        for features in all_item_features:
            unique_features = set(features)
            doc_frequency.update(unique_features)
        
        idf = {}
        for feature, df in doc_frequency.items():
            idf[feature] = math.log(num_docs / df)
        
        return idf
    
    def calculate_tfidf(self, item_features: List[str], 
                       idf: Dict[str, float]) -> Dict[str, float]:
        """Calculate TF-IDF scores"""
        tf = self.calculate_tf(item_features)
        
        tfidf = {}
        for feature, tf_score in tf.items():
            tfidf[feature] = tf_score * idf.get(feature, 0)
        
        return tfidf


class UserProfileBuilder:
    """Build user feature profile from preferences"""
    
    def build_profile_from_ratings(self, user_ratings: List[Dict[str, Any]],
                                   item_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Build user profile from rated items"""
        weighted_features = defaultdict(float)
        total_weight = 0
        
        for rating in user_ratings:
            item_id = rating['item_id']
            rating_value = rating['rating']
            
            if item_id in item_features:
                features = item_features[item_id]
                
                for feature, score in features.items():
                    weighted_features[feature] += score * rating_value
                    total_weight += rating_value
        
        # Normalize by total weight
        if total_weight > 0:
            user_profile = {feature: score / total_weight 
                          for feature, score in weighted_features.items()}
        else:
            user_profile = {}
        
        return user_profile
    
    def update_profile_with_implicit(self, user_profile: Dict[str, float],
                                    implicit_signals: List[Dict[str, Any]],
                                    item_features: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Update profile with implicit feedback"""
        updated_profile = user_profile.copy()
        
        # Weight for implicit signals
        signal_weights = {
            'view': 0.5,
            'click': 1.0,
            'share': 2.0,
            'bookmark': 1.5
        }
        
        for signal in implicit_signals:
            item_id = signal['item_id']
            signal_type = signal['type']
            weight = signal_weights.get(signal_type, 0.5)
            
            if item_id in item_features:
                features = item_features[item_id]
                
                for feature, score in features.items():
                    current = updated_profile.get(feature, 0)
                    updated_profile[feature] = current + (score * weight * 0.1)  # Dampen implicit signals
        
        return updated_profile


class ContentBasedRecommender:
    """Generate content-based recommendations"""
    
    def calculate_similarity(self, profile: Dict[str, float],
                            item_features: Dict[str, float]) -> float:
        """Calculate cosine similarity between profile and item"""
        # Get common features
        common_features = set(profile.keys()) & set(item_features.keys())
        
        if not common_features:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(profile[f] * item_features[f] for f in common_features)
        
        # Calculate magnitudes
        profile_magnitude = math.sqrt(sum(v ** 2 for v in profile.values()))
        item_magnitude = math.sqrt(sum(v ** 2 for v in item_features.values()))
        
        if profile_magnitude == 0 or item_magnitude == 0:
            return 0.0
        
        return dot_product / (profile_magnitude * item_magnitude)
    
    def generate_recommendations(self, user_profile: Dict[str, float],
                                item_features: Dict[str, Dict[str, float]],
                                exclude_items: Set[str],
                                n: int = 5) -> List[Dict[str, Any]]:
        """Generate top N recommendations"""
        recommendations = []
        
        for item_id, features in item_features.items():
            if item_id in exclude_items:
                continue
            
            similarity = self.calculate_similarity(user_profile, features)
            
            if similarity > 0:
                # Feature overlap
                overlap = len(set(user_profile.keys()) & set(features.keys()))
                
                recommendations.append({
                    'item_id': item_id,
                    'similarity_score': round(similarity, 3),
                    'feature_overlap': overlap,
                    'top_matching_features': self._get_top_features(
                        user_profile, features, k=3
                    )
                })
        
        # Sort by similarity
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        return recommendations[:n]
    
    def _get_top_features(self, profile: Dict[str, float],
                         item_features: Dict[str, float],
                         k: int = 3) -> List[str]:
        """Get top matching features"""
        feature_scores = []
        
        for feature in set(profile.keys()) & set(item_features.keys()):
            score = profile[feature] * item_features[feature]
            feature_scores.append((feature, score))
        
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in feature_scores[:k]]


# Agent functions
def initialize_items_agent(state: ContentBasedFilteringState) -> ContentBasedFilteringState:
    """Initialize item catalog"""
    
    # Simulate item catalog
    items = [
        {
            'item_id': 'item_001',
            'title': 'Introduction to Machine Learning',
            'description': 'Learn the fundamentals of machine learning algorithms and applications',
            'genres': ['education', 'technology'],
            'category': 'course',
            'tags': ['machine-learning', 'ai', 'python'],
            'rating': 4.5,
            'popularity': 85
        },
        {
            'item_id': 'item_002',
            'title': 'Deep Learning Specialization',
            'description': 'Master deep learning with neural networks and advanced techniques',
            'genres': ['education', 'technology'],
            'category': 'course',
            'tags': ['deep-learning', 'neural-networks', 'ai'],
            'rating': 4.8,
            'popularity': 92
        },
        {
            'item_id': 'item_003',
            'title': 'Data Science for Business',
            'description': 'Apply data science techniques to solve business problems',
            'genres': ['education', 'business'],
            'category': 'course',
            'tags': ['data-science', 'analytics', 'business'],
            'rating': 4.3,
            'popularity': 78
        },
        {
            'item_id': 'item_004',
            'title': 'Web Development Bootcamp',
            'description': 'Full-stack web development with modern frameworks',
            'genres': ['education', 'technology'],
            'category': 'course',
            'tags': ['web-development', 'javascript', 'react'],
            'rating': 4.6,
            'popularity': 88
        },
        {
            'item_id': 'item_005',
            'title': 'Python for Data Analysis',
            'description': 'Data analysis and visualization with Python',
            'genres': ['education', 'technology'],
            'category': 'course',
            'tags': ['python', 'data-analysis', 'pandas'],
            'rating': 4.4,
            'popularity': 80
        },
        {
            'item_id': 'item_006',
            'title': 'Cloud Computing Fundamentals',
            'description': 'Introduction to cloud platforms and services',
            'genres': ['education', 'technology'],
            'category': 'course',
            'tags': ['cloud', 'aws', 'devops'],
            'rating': 4.2,
            'popularity': 75
        },
        {
            'item_id': 'item_007',
            'title': 'Digital Marketing Mastery',
            'description': 'Comprehensive digital marketing strategies',
            'genres': ['education', 'business'],
            'category': 'course',
            'tags': ['marketing', 'seo', 'social-media'],
            'rating': 4.1,
            'popularity': 70
        },
        {
            'item_id': 'item_008',
            'title': 'Natural Language Processing',
            'description': 'NLP techniques and applications with transformers',
            'genres': ['education', 'technology'],
            'category': 'course',
            'tags': ['nlp', 'ai', 'transformers'],
            'rating': 4.7,
            'popularity': 82
        }
    ]
    
    return {
        **state,
        'items': items,
        'messages': state['messages'] + [f'Initialized {len(items)} items in catalog']
    }


def extract_item_features_agent(state: ContentBasedFilteringState) -> ContentBasedFilteringState:
    """Extract features from all items"""
    
    extractor = FeatureExtractor()
    tfidf_calc = TFIDFCalculator()
    
    item_features = {}
    all_text_features = []
    
    # Extract features for each item
    for item in state['items']:
        # Text features
        text = f"{item['title']} {item['description']}"
        text_features = extractor.extract_text_features(text)
        all_text_features.append(text_features)
        
        # Categorical features
        categorical_features = extractor.extract_categorical_features(item)
        
        # Combine features
        all_features = text_features + categorical_features
        
        item_features[item['item_id']] = all_features
    
    # Calculate IDF
    idf = tfidf_calc.calculate_idf(all_text_features)
    
    # Calculate TF-IDF for each item
    tfidf_scores = {}
    for item_id, features in item_features.items():
        tfidf = tfidf_calc.calculate_tfidf(features, idf)
        tfidf_scores[item_id] = tfidf
    
    return {
        **state,
        'item_features': item_features,
        'tfidf_scores': tfidf_scores,
        'messages': state['messages'] + [f'Extracted features for {len(item_features)} items']
    }


def build_user_profile_agent(state: ContentBasedFilteringState) -> ContentBasedFilteringState:
    """Build user feature profile"""
    
    # Simulate user ratings (explicit feedback)
    user_ratings = [
        {'item_id': 'item_001', 'rating': 5.0},  # Loved ML course
        {'item_id': 'item_002', 'rating': 4.5},  # Liked DL course
        {'item_id': 'item_005', 'rating': 4.0},  # Liked Python course
    ]
    
    # Simulate implicit signals
    implicit_signals = [
        {'item_id': 'item_008', 'type': 'view'},    # Viewed NLP course
        {'item_id': 'item_004', 'type': 'click'},   # Clicked web dev
        {'item_id': 'item_003', 'type': 'bookmark'} # Bookmarked data science
    ]
    
    profile_builder = UserProfileBuilder()
    
    # Build profile from ratings
    user_profile = profile_builder.build_profile_from_ratings(
        user_ratings,
        state['tfidf_scores']
    )
    
    # Update with implicit signals
    user_profile = profile_builder.update_profile_with_implicit(
        user_profile,
        implicit_signals,
        state['tfidf_scores']
    )
    
    user_data = {
        'profile': user_profile,
        'rated_items': [r['item_id'] for r in user_ratings],
        'implicit_items': [s['item_id'] for s in implicit_signals],
        'num_features': len(user_profile)
    }
    
    return {
        **state,
        'user_feature_profile': user_profile,
        'user_profile': user_data,
        'messages': state['messages'] + [f'Built user profile with {len(user_profile)} features']
    }


def generate_content_based_recommendations_agent(state: ContentBasedFilteringState) -> ContentBasedFilteringState:
    """Generate content-based recommendations"""
    
    recommender = ContentBasedRecommender()
    
    # Items to exclude (already rated or interacted with)
    exclude_items = set(state['user_profile']['rated_items'] + 
                       state['user_profile']['implicit_items'])
    
    # Generate recommendations
    recommendations = recommender.generate_recommendations(
        state['user_feature_profile'],
        state['tfidf_scores'],
        exclude_items,
        n=5
    )
    
    # Enrich with item details
    item_map = {item['item_id']: item for item in state['items']}
    
    enriched_recommendations = []
    for rec in recommendations:
        item = item_map.get(rec['item_id'])
        if item:
            enriched_recommendations.append({
                **rec,
                'title': item['title'],
                'category': item['category'],
                'genres': item['genres'],
                'rating': item['rating']
            })
    
    return {
        **state,
        'recommendations': enriched_recommendations,
        'messages': state['messages'] + [f'Generated {len(enriched_recommendations)} recommendations']
    }


def analyze_content_filtering_agent(state: ContentBasedFilteringState) -> ContentBasedFilteringState:
    """Analyze content-based filtering results"""
    
    # Analyze feature distribution
    all_features = set()
    for features in state['item_features'].values():
        all_features.update(features)
    
    # Top user features
    top_user_features = sorted(
        state['user_feature_profile'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Recommendation diversity
    rec_categories = Counter(rec['category'] for rec in state['recommendations'])
    rec_genres = Counter(genre for rec in state['recommendations'] for genre in rec['genres'])
    
    analytics = {
        'item_stats': {
            'total_items': len(state['items']),
            'total_features': len(all_features),
            'avg_features_per_item': sum(len(f) for f in state['item_features'].values()) / len(state['items'])
        },
        'user_profile_stats': {
            'num_rated_items': len(state['user_profile']['rated_items']),
            'num_implicit_signals': len(state['user_profile']['implicit_items']),
            'profile_feature_count': state['user_profile']['num_features'],
            'top_features': [f"{f}: {s:.3f}" for f, s in top_user_features[:5]]
        },
        'recommendation_stats': {
            'total_recommendations': len(state['recommendations']),
            'avg_similarity': sum(r['similarity_score'] for r in state['recommendations']) / len(state['recommendations']) if state['recommendations'] else 0,
            'category_diversity': len(rec_categories),
            'genre_diversity': len(rec_genres)
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed content-based filtering results']
    }


def generate_content_filtering_report_agent(state: ContentBasedFilteringState) -> ContentBasedFilteringState:
    """Generate comprehensive content-based filtering report"""
    
    report_lines = [
        "=" * 80,
        "CONTENT-BASED FILTERING REPORT",
        "=" * 80,
        "",
        "ITEM CATALOG:",
        "-" * 40,
        f"Total Items: {state['analytics']['item_stats']['total_items']}",
        f"Total Unique Features: {state['analytics']['item_stats']['total_features']}",
        f"Avg Features per Item: {state['analytics']['item_stats']['avg_features_per_item']:.1f}",
        "",
        "USER PROFILE:",
        "-" * 40,
        f"Rated Items: {state['analytics']['user_profile_stats']['num_rated_items']}",
        f"Implicit Signals: {state['analytics']['user_profile_stats']['num_implicit_signals']}",
        f"Profile Features: {state['analytics']['user_profile_stats']['profile_feature_count']}",
        "",
        "Top User Features:"
    ]
    
    for feature in state['analytics']['user_profile_stats']['top_features']:
        report_lines.append(f"  • {feature}")
    
    report_lines.extend([
        "",
        "RATED ITEMS:",
        "-" * 40
    ])
    
    item_map = {item['item_id']: item for item in state['items']}
    for item_id in state['user_profile']['rated_items']:
        item = item_map.get(item_id)
        if item:
            report_lines.append(f"  • {item['title']} ({item['category']})")
    
    report_lines.extend([
        "",
        "CONTENT-BASED RECOMMENDATIONS:",
        "-" * 40,
        f"Total Generated: {state['analytics']['recommendation_stats']['total_recommendations']}",
        f"Average Similarity: {state['analytics']['recommendation_stats']['avg_similarity']:.3f}",
        f"Category Diversity: {state['analytics']['recommendation_stats']['category_diversity']} categories",
        f"Genre Diversity: {state['analytics']['recommendation_stats']['genre_diversity']} genres",
        ""
    ])
    
    for i, rec in enumerate(state['recommendations'], 1):
        report_lines.append(f"\n{i}. {rec['title']}")
        report_lines.append(f"   Similarity Score: {rec['similarity_score']:.3f}")
        report_lines.append(f"   Category: {rec['category']}")
        report_lines.append(f"   Genres: {', '.join(rec['genres'])}")
        report_lines.append(f"   Rating: {rec['rating']}/5.0")
        report_lines.append(f"   Feature Overlap: {rec['feature_overlap']} features")
        report_lines.append(f"   Top Matching Features: {', '.join(rec['top_matching_features'])}")
    
    report_lines.extend([
        "",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ User shows strong preference for: {state['analytics']['user_profile_stats']['top_features'][0].split(':')[0]}",
        f"✓ Recommendations have {state['analytics']['recommendation_stats']['avg_similarity']:.1%} avg similarity to profile",
        f"✓ Diverse recommendations across {state['analytics']['recommendation_stats']['category_diversity']} categories",
        f"✓ Feature-based matching provides explainable recommendations",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Combine with collaborative filtering for better coverage",
        "• Use TF-IDF for text-heavy content",
        "• Include temporal features for time-sensitive items",
        "• Implement feature weighting based on importance",
        "• Consider user feedback to refine feature extraction",
        "• Use deep learning embeddings for richer representations",
        "• Apply diversity re-ranking to avoid filter bubbles",
        "• Explain recommendations using top matching features",
        "",
        "=" * 80
    ]
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive content-based filtering report']
    }


# Create the graph
def create_content_based_filtering_graph():
    """Create the content-based filtering workflow graph"""
    
    workflow = StateGraph(ContentBasedFilteringState)
    
    # Add nodes
    workflow.add_node("initialize_items", initialize_items_agent)
    workflow.add_node("extract_features", extract_item_features_agent)
    workflow.add_node("build_profile", build_user_profile_agent)
    workflow.add_node("generate_recommendations", generate_content_based_recommendations_agent)
    workflow.add_node("analyze_results", analyze_content_filtering_agent)
    workflow.add_node("generate_report", generate_content_filtering_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize_items")
    workflow.add_edge("initialize_items", "extract_features")
    workflow.add_edge("extract_features", "build_profile")
    workflow.add_edge("build_profile", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the content-based filtering graph
    app = create_content_based_filtering_graph()
    
    # Initialize state
    initial_state: ContentBasedFilteringState = {
        'messages': [],
        'items': [],
        'user_profile': {},
        'item_features': {},
        'tfidf_scores': {},
        'user_feature_profile': {},
        'recommendations': [],
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("CONTENT-BASED FILTERING MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nContent-based filtering pattern execution complete! ✓")

"""
Hybrid Filtering MCP Pattern

This pattern demonstrates hybrid filtering in an agentic MCP system.
The system combines collaborative and content-based filtering using
multiple hybridization strategies: weighted, switching, cascade, and feature combination.

Use cases:
- Comprehensive recommendation systems
- Cold-start problem mitigation
- Balanced recommendations
- Multi-strategy personalization
- Adaptive recommendation systems
"""

from typing import TypedDict, Annotated, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from collections import defaultdict
import math


# Define the state for hybrid filtering
class HybridFilteringState(TypedDict):
    """State for tracking hybrid filtering process"""
    messages: Annotated[List[str], add]
    user_item_matrix: Dict[str, Dict[str, float]]
    item_features: Dict[str, Dict[str, Any]]
    user_profile: Dict[str, Any]
    collaborative_recs: List[Dict[str, Any]]
    content_based_recs: List[Dict[str, Any]]
    weighted_hybrid_recs: List[Dict[str, Any]]
    switching_hybrid_recs: List[Dict[str, Any]]
    cascade_hybrid_recs: List[Dict[str, Any]]
    feature_combination_recs: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
    analytics: Dict[str, Any]
    report: str


class CollaborativeComponent:
    """Collaborative filtering component"""
    
    def cosine_similarity(self, vec1: Dict[str, float], 
                         vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity"""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        
        dot = sum(vec1[k] * vec2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0
    
    def generate_recommendations(self, user_id: str,
                                user_item_matrix: Dict[str, Dict[str, float]],
                                k_neighbors: int = 3,
                                n: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate collaborative recommendations with confidence"""
        user_ratings = user_item_matrix.get(user_id, {})
        
        # Find similar users
        similarities = []
        for other_user, other_ratings in user_item_matrix.items():
            if other_user != user_id:
                sim = self.cosine_similarity(user_ratings, other_ratings)
                if sim > 0:
                    similarities.append((other_user, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_users = similarities[:k_neighbors]
        
        # Generate recommendations
        item_scores = defaultdict(float)
        item_weights = defaultdict(float)
        
        for similar_user, similarity in similar_users:
            for item, rating in user_item_matrix[similar_user].items():
                if item not in user_ratings:
                    item_scores[item] += rating * similarity
                    item_weights[item] += similarity
        
        recommendations = []
        for item, score in item_scores.items():
            if item_weights[item] > 0:
                predicted_rating = score / item_weights[item]
                recommendations.append({
                    'item': item,
                    'score': round(predicted_rating, 2),
                    'confidence': round(item_weights[item] / sum(s for _, s in similar_users), 2),
                    'method': 'collaborative'
                })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate component confidence
        confidence_score = len(similar_users) / k_neighbors if similar_users else 0
        
        metadata = {
            'neighbors_found': len(similar_users),
            'confidence': confidence_score,
            'coverage': len(recommendations)
        }
        
        return recommendations[:n], metadata


class ContentBasedComponent:
    """Content-based filtering component"""
    
    def calculate_item_similarity(self, features1: Dict[str, Any],
                                  features2: Dict[str, Any]) -> float:
        """Calculate similarity between item features"""
        # Convert features to comparable format
        vec1 = self._featurize(features1)
        vec2 = self._featurize(features2)
        
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        
        dot = sum(vec1[k] * vec2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0
    
    def _featurize(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Convert features to numeric vector"""
        vector = {}
        
        # Genre features
        if 'genres' in features:
            for genre in features['genres']:
                vector[f'genre_{genre}'] = 1.0
        
        # Tags
        if 'tags' in features:
            for tag in features['tags']:
                vector[f'tag_{tag}'] = 1.0
        
        # Numeric features
        for key in ['rating', 'popularity', 'year']:
            if key in features:
                vector[key] = features[key] / 100.0  # Normalize
        
        return vector
    
    def build_user_profile(self, user_id: str,
                          user_item_matrix: Dict[str, Dict[str, float]],
                          item_features: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Build user profile from rated items"""
        user_ratings = user_item_matrix.get(user_id, {})
        profile = defaultdict(float)
        total_weight = 0
        
        for item, rating in user_ratings.items():
            if item in item_features:
                features = self._featurize(item_features[item])
                for feature, value in features.items():
                    profile[feature] += value * rating
                    total_weight += rating
        
        if total_weight > 0:
            profile = {k: v / total_weight for k, v in profile.items()}
        
        return dict(profile)
    
    def generate_recommendations(self, user_id: str,
                                user_item_matrix: Dict[str, Dict[str, float]],
                                item_features: Dict[str, Dict[str, Any]],
                                n: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Generate content-based recommendations with confidence"""
        user_profile = self.build_user_profile(user_id, user_item_matrix, item_features)
        user_items = set(user_item_matrix.get(user_id, {}).keys())
        
        recommendations = []
        
        for item, features in item_features.items():
            if item not in user_items:
                item_vec = self._featurize(features)
                similarity = self._cosine_similarity(user_profile, item_vec)
                
                if similarity > 0:
                    recommendations.append({
                        'item': item,
                        'score': round(similarity * 5.0, 2),  # Scale to rating
                        'confidence': round(similarity, 2),
                        'method': 'content_based'
                    })
        
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate component confidence
        confidence_score = len(user_profile) / 10.0 if user_profile else 0  # Assume 10 features is good
        confidence_score = min(confidence_score, 1.0)
        
        metadata = {
            'profile_features': len(user_profile),
            'confidence': confidence_score,
            'coverage': len(recommendations)
        }
        
        return recommendations[:n], metadata
    
    def _cosine_similarity(self, vec1: Dict[str, float], 
                          vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity"""
        common = set(vec1.keys()) & set(vec2.keys())
        if not common:
            return 0.0
        
        dot = sum(vec1[k] * vec2[k] for k in common)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0


class HybridStrategies:
    """Different hybridization strategies"""
    
    def weighted_hybrid(self, collab_recs: List[Dict[str, Any]],
                       content_recs: List[Dict[str, Any]],
                       collab_weight: float = 0.6,
                       content_weight: float = 0.4,
                       n: int = 5) -> List[Dict[str, Any]]:
        """Weighted combination of recommendations"""
        combined = {}
        
        # Add collaborative recommendations
        for rec in collab_recs:
            item = rec['item']
            combined[item] = {
                'item': item,
                'collab_score': rec['score'] * collab_weight,
                'content_score': 0,
                'method': 'weighted'
            }
        
        # Add content-based recommendations
        for rec in content_recs:
            item = rec['item']
            if item in combined:
                combined[item]['content_score'] = rec['score'] * content_weight
            else:
                combined[item] = {
                    'item': item,
                    'collab_score': 0,
                    'content_score': rec['score'] * content_weight,
                    'method': 'weighted'
                }
        
        # Calculate final scores
        results = []
        for item, scores in combined.items():
            final_score = scores['collab_score'] + scores['content_score']
            results.append({
                'item': item,
                'score': round(final_score, 2),
                'collab_contribution': round(scores['collab_score'], 2),
                'content_contribution': round(scores['content_score'], 2),
                'method': 'weighted'
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n]
    
    def switching_hybrid(self, collab_recs: List[Dict[str, Any]],
                        content_recs: List[Dict[str, Any]],
                        collab_metadata: Dict[str, Any],
                        content_metadata: Dict[str, Any],
                        n: int = 5) -> List[Dict[str, Any]]:
        """Switch between methods based on confidence"""
        # Decide which method to use based on metadata
        collab_conf = collab_metadata.get('confidence', 0)
        content_conf = content_metadata.get('confidence', 0)
        
        if collab_conf > content_conf:
            selected = collab_recs[:n]
            method = 'collaborative'
            reason = f'Higher confidence ({collab_conf:.2f} vs {content_conf:.2f})'
        else:
            selected = content_recs[:n]
            method = 'content_based'
            reason = f'Higher confidence ({content_conf:.2f} vs {collab_conf:.2f})'
        
        results = []
        for rec in selected:
            results.append({
                **rec,
                'method': f'switching->{method}',
                'switch_reason': reason
            })
        
        return results
    
    def cascade_hybrid(self, collab_recs: List[Dict[str, Any]],
                      content_recs: List[Dict[str, Any]],
                      threshold: float = 3.5,
                      n: int = 5) -> List[Dict[str, Any]]:
        """Cascade: use collaborative first, fill with content-based"""
        results = []
        
        # Add high-confidence collaborative recommendations
        for rec in collab_recs:
            if rec['score'] >= threshold:
                results.append({
                    **rec,
                    'method': 'cascade->collaborative',
                    'cascade_stage': 'primary'
                })
        
        # Fill remaining slots with content-based
        remaining = n - len(results)
        if remaining > 0:
            for rec in content_recs[:remaining]:
                results.append({
                    **rec,
                    'method': 'cascade->content_based',
                    'cascade_stage': 'secondary'
                })
        
        return results[:n]
    
    def feature_combination(self, collab_recs: List[Dict[str, Any]],
                           content_recs: List[Dict[str, Any]],
                           n: int = 5) -> List[Dict[str, Any]]:
        """Combine as features for a meta-recommender"""
        combined = {}
        
        for rec in collab_recs:
            item = rec['item']
            combined[item] = {
                'item': item,
                'collab_score': rec['score'],
                'collab_confidence': rec.get('confidence', 0),
                'content_score': 0,
                'content_confidence': 0
            }
        
        for rec in content_recs:
            item = rec['item']
            if item in combined:
                combined[item]['content_score'] = rec['score']
                combined[item]['content_confidence'] = rec.get('confidence', 0)
            else:
                combined[item] = {
                    'item': item,
                    'collab_score': 0,
                    'collab_confidence': 0,
                    'content_score': rec['score'],
                    'content_confidence': rec.get('confidence', 0)
                }
        
        # Meta-scoring: weighted by confidence
        results = []
        for item, features in combined.items():
            total_conf = features['collab_confidence'] + features['content_confidence']
            if total_conf > 0:
                meta_score = (
                    features['collab_score'] * features['collab_confidence'] +
                    features['content_score'] * features['content_confidence']
                ) / total_conf
            else:
                meta_score = (features['collab_score'] + features['content_score']) / 2
            
            results.append({
                'item': item,
                'score': round(meta_score, 2),
                'collab_score': features['collab_score'],
                'content_score': features['content_score'],
                'collab_conf': features['collab_confidence'],
                'content_conf': features['content_confidence'],
                'method': 'feature_combination'
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n]


# Agent functions
def initialize_hybrid_data_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Initialize data for hybrid filtering"""
    
    # User-item ratings
    user_item_matrix = {
        'user_001': {'item_a': 5, 'item_b': 4, 'item_e': 5, 'item_f': 4},
        'user_002': {'item_a': 4, 'item_b': 5, 'item_c': 4, 'item_g': 4},
        'user_003': {'item_b': 3, 'item_c': 5, 'item_d': 4, 'item_h': 5},
        'user_004': {'item_a': 5, 'item_f': 5, 'item_g': 3, 'item_i': 4},
        'user_005': {'item_b': 4, 'item_d': 5, 'item_e': 3, 'item_h': 4},
        'target_user': {'item_a': 5, 'item_b': 4}  # New user with limited ratings
    }
    
    # Item features
    item_features = {
        'item_a': {'genres': ['action', 'scifi'], 'tags': ['robots', 'future'], 'rating': 85, 'year': 2020},
        'item_b': {'genres': ['action', 'adventure'], 'tags': ['hero', 'quest'], 'rating': 80, 'year': 2019},
        'item_c': {'genres': ['drama', 'romance'], 'tags': ['love', 'emotional'], 'rating': 75, 'year': 2021},
        'item_d': {'genres': ['drama', 'mystery'], 'tags': ['detective', 'suspense'], 'rating': 82, 'year': 2020},
        'item_e': {'genres': ['action', 'scifi'], 'tags': ['space', 'future'], 'rating': 88, 'year': 2021},
        'item_f': {'genres': ['action', 'thriller'], 'tags': ['spy', 'intrigue'], 'rating': 83, 'year': 2020},
        'item_g': {'genres': ['comedy', 'adventure'], 'tags': ['funny', 'journey'], 'rating': 70, 'year': 2018},
        'item_h': {'genres': ['drama', 'historical'], 'tags': ['period', 'war'], 'rating': 86, 'year': 2019},
        'item_i': {'genres': ['action', 'crime'], 'tags': ['heist', 'team'], 'rating': 84, 'year': 2021}
    }
    
    user_profile = {
        'user_id': 'target_user',
        'rated_items': list(user_item_matrix['target_user'].keys()),
        'num_ratings': len(user_item_matrix['target_user'])
    }
    
    return {
        **state,
        'user_item_matrix': user_item_matrix,
        'item_features': item_features,
        'user_profile': user_profile,
        'messages': state['messages'] + ['Initialized hybrid filtering data']
    }


def generate_collaborative_recommendations_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Generate collaborative filtering recommendations"""
    
    collab = CollaborativeComponent()
    recommendations, metadata = collab.generate_recommendations(
        'target_user',
        state['user_item_matrix'],
        k_neighbors=3,
        n=5
    )
    
    return {
        **state,
        'collaborative_recs': recommendations,
        'messages': state['messages'] + [f'Generated {len(recommendations)} collaborative recommendations (confidence: {metadata["confidence"]:.2f})']
    }


def generate_content_based_recommendations_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Generate content-based filtering recommendations"""
    
    content = ContentBasedComponent()
    recommendations, metadata = content.generate_recommendations(
        'target_user',
        state['user_item_matrix'],
        state['item_features'],
        n=5
    )
    
    return {
        **state,
        'content_based_recs': recommendations,
        'messages': state['messages'] + [f'Generated {len(recommendations)} content-based recommendations (confidence: {metadata["confidence"]:.2f})']
    }


def apply_weighted_hybrid_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Apply weighted hybrid strategy"""
    
    hybrid = HybridStrategies()
    recommendations = hybrid.weighted_hybrid(
        state['collaborative_recs'],
        state['content_based_recs'],
        collab_weight=0.6,
        content_weight=0.4,
        n=5
    )
    
    return {
        **state,
        'weighted_hybrid_recs': recommendations,
        'messages': state['messages'] + ['Applied weighted hybrid strategy (60% collab, 40% content)']
    }


def apply_switching_hybrid_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Apply switching hybrid strategy"""
    
    # Extract metadata from recommendations
    collab_metadata = {
        'confidence': sum(r.get('confidence', 0) for r in state['collaborative_recs']) / len(state['collaborative_recs']) if state['collaborative_recs'] else 0
    }
    content_metadata = {
        'confidence': sum(r.get('confidence', 0) for r in state['content_based_recs']) / len(state['content_based_recs']) if state['content_based_recs'] else 0
    }
    
    hybrid = HybridStrategies()
    recommendations = hybrid.switching_hybrid(
        state['collaborative_recs'],
        state['content_based_recs'],
        collab_metadata,
        content_metadata,
        n=5
    )
    
    return {
        **state,
        'switching_hybrid_recs': recommendations,
        'messages': state['messages'] + ['Applied switching hybrid strategy']
    }


def apply_cascade_hybrid_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Apply cascade hybrid strategy"""
    
    hybrid = HybridStrategies()
    recommendations = hybrid.cascade_hybrid(
        state['collaborative_recs'],
        state['content_based_recs'],
        threshold=3.5,
        n=5
    )
    
    return {
        **state,
        'cascade_hybrid_recs': recommendations,
        'messages': state['messages'] + ['Applied cascade hybrid strategy (threshold: 3.5)']
    }


def apply_feature_combination_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Apply feature combination strategy"""
    
    hybrid = HybridStrategies()
    recommendations = hybrid.feature_combination(
        state['collaborative_recs'],
        state['content_based_recs'],
        n=5
    )
    
    return {
        **state,
        'feature_combination_recs': recommendations,
        'messages': state['messages'] + ['Applied feature combination strategy']
    }


def select_final_recommendations_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Select final recommendations using ensemble"""
    
    # Use weighted hybrid as final (best overall performance)
    final_recommendations = state['weighted_hybrid_recs']
    
    return {
        **state,
        'final_recommendations': final_recommendations,
        'messages': state['messages'] + ['Selected weighted hybrid as final recommendations']
    }


def analyze_hybrid_results_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Analyze hybrid filtering results"""
    
    # Analyze overlap between methods
    collab_items = set(r['item'] for r in state['collaborative_recs'])
    content_items = set(r['item'] for r in state['content_based_recs'])
    weighted_items = set(r['item'] for r in state['weighted_hybrid_recs'])
    
    overlap_collab_content = len(collab_items & content_items)
    
    analytics = {
        'component_stats': {
            'collaborative_count': len(state['collaborative_recs']),
            'content_based_count': len(state['content_based_recs']),
            'method_overlap': overlap_collab_content
        },
        'hybrid_stats': {
            'weighted_count': len(state['weighted_hybrid_recs']),
            'switching_count': len(state['switching_hybrid_recs']),
            'cascade_count': len(state['cascade_hybrid_recs']),
            'feature_combination_count': len(state['feature_combination_recs'])
        },
        'user_stats': {
            'user_id': state['user_profile']['user_id'],
            'num_ratings': state['user_profile']['num_ratings']
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed hybrid filtering results']
    }


def generate_hybrid_filtering_report_agent(state: HybridFilteringState) -> HybridFilteringState:
    """Generate comprehensive hybrid filtering report"""
    
    report_lines = [
        "=" * 80,
        "HYBRID FILTERING REPORT",
        "=" * 80,
        "",
        "USER PROFILE:",
        "-" * 40,
        f"User ID: {state['user_profile']['user_id']}",
        f"Rated Items: {state['user_profile']['num_ratings']}",
        f"Items: {', '.join(state['user_profile']['rated_items'])}",
        "",
        "COMPONENT METHODS:",
        "-" * 40,
        f"Collaborative Recommendations: {state['analytics']['component_stats']['collaborative_count']}",
        f"Content-Based Recommendations: {state['analytics']['component_stats']['content_based_count']}",
        f"Method Overlap: {state['analytics']['component_stats']['method_overlap']} items",
        "",
        "COLLABORATIVE FILTERING (User-Based):",
        "-" * 40
    ]
    
    for rec in state['collaborative_recs'][:3]:
        report_lines.append(f"  • {rec['item']}: score={rec['score']}, confidence={rec.get('confidence', 0):.2f}")
    
    report_lines.extend([
        "",
        "CONTENT-BASED FILTERING (Feature-Based):",
        "-" * 40
    ])
    
    for rec in state['content_based_recs'][:3]:
        report_lines.append(f"  • {rec['item']}: score={rec['score']}, confidence={rec.get('confidence', 0):.2f}")
    
    report_lines.extend([
        "",
        "",
        "HYBRID STRATEGIES:",
        "=" * 80,
        "",
        "1. WEIGHTED HYBRID (60% Collaborative + 40% Content):",
        "-" * 40
    ])
    
    for rec in state['weighted_hybrid_recs']:
        report_lines.append(f"\n  • {rec['item']}")
        report_lines.append(f"    Final Score: {rec['score']}")
        report_lines.append(f"    Collab: {rec['collab_contribution']} | Content: {rec['content_contribution']}")
    
    report_lines.extend([
        "",
        "",
        "2. SWITCHING HYBRID (Confidence-Based):",
        "-" * 40
    ])
    
    for rec in state['switching_hybrid_recs']:
        report_lines.append(f"  • {rec['item']}: score={rec['score']} ({rec.get('switch_reason', 'N/A')})")
    
    report_lines.extend([
        "",
        "3. CASCADE HYBRID (Threshold=3.5):",
        "-" * 40
    ])
    
    for rec in state['cascade_hybrid_recs']:
        report_lines.append(f"  • {rec['item']}: score={rec['score']} (stage: {rec.get('cascade_stage', 'N/A')})")
    
    report_lines.extend([
        "",
        "4. FEATURE COMBINATION (Meta-Recommender):",
        "-" * 40
    ])
    
    for rec in state['feature_combination_recs']:
        report_lines.append(f"\n  • {rec['item']}")
        report_lines.append(f"    Meta Score: {rec['score']}")
        report_lines.append(f"    Collab: {rec['collab_score']} (conf: {rec['collab_conf']:.2f})")
        report_lines.append(f"    Content: {rec['content_score']} (conf: {rec['content_conf']:.2f})")
    
    report_lines.extend([
        "",
        "",
        "FINAL RECOMMENDATIONS (Weighted Hybrid):",
        "=" * 80
    ])
    
    for i, rec in enumerate(state['final_recommendations'], 1):
        report_lines.append(f"\n{i}. {rec['item']}")
        report_lines.append(f"   Score: {rec['score']}")
        report_lines.append(f"   Collaborative: {rec['collab_contribution']}")
        report_lines.append(f"   Content-Based: {rec['content_contribution']}")
    
    report_lines.extend([
        "",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Methods agree on {state['analytics']['component_stats']['method_overlap']} items",
        "✓ Weighted hybrid provides balanced recommendations",
        "✓ Switching adapts to data availability",
        "✓ Cascade ensures quality threshold",
        "✓ Feature combination leverages both signals",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Use weighted hybrid for general cases",
        "• Switch to content-based for new users (cold-start)",
        "• Apply cascade when quality is critical",
        "• Feature combination works well with sufficient data",
        "• Monitor component performance separately",
        "• Adjust weights based on A/B testing",
        "• Consider user context for dynamic weighting",
        "• Implement fallback strategies for edge cases",
        "",
        "=" * 80
    ]
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive hybrid filtering report']
    }


# Create the graph
def create_hybrid_filtering_graph():
    """Create the hybrid filtering workflow graph"""
    
    workflow = StateGraph(HybridFilteringState)
    
    # Add nodes
    workflow.add_node("initialize_data", initialize_hybrid_data_agent)
    workflow.add_node("generate_collaborative", generate_collaborative_recommendations_agent)
    workflow.add_node("generate_content", generate_content_based_recommendations_agent)
    workflow.add_node("apply_weighted", apply_weighted_hybrid_agent)
    workflow.add_node("apply_switching", apply_switching_hybrid_agent)
    workflow.add_node("apply_cascade", apply_cascade_hybrid_agent)
    workflow.add_node("apply_feature_combination", apply_feature_combination_agent)
    workflow.add_node("select_final", select_final_recommendations_agent)
    workflow.add_node("analyze_results", analyze_hybrid_results_agent)
    workflow.add_node("generate_report", generate_hybrid_filtering_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize_data")
    workflow.add_edge("initialize_data", "generate_collaborative")
    workflow.add_edge("generate_collaborative", "generate_content")
    workflow.add_edge("generate_content", "apply_weighted")
    workflow.add_edge("apply_weighted", "apply_switching")
    workflow.add_edge("apply_switching", "apply_cascade")
    workflow.add_edge("apply_cascade", "apply_feature_combination")
    workflow.add_edge("apply_feature_combination", "select_final")
    workflow.add_edge("select_final", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the hybrid filtering graph
    app = create_hybrid_filtering_graph()
    
    # Initialize state
    initial_state: HybridFilteringState = {
        'messages': [],
        'user_item_matrix': {},
        'item_features': {},
        'user_profile': {},
        'collaborative_recs': [],
        'content_based_recs': [],
        'weighted_hybrid_recs': [],
        'switching_hybrid_recs': [],
        'cascade_hybrid_recs': [],
        'feature_combination_recs': [],
        'final_recommendations': [],
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("HYBRID FILTERING MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nHybrid filtering pattern execution complete! ✓")

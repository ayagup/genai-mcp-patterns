"""
Collaborative Filtering MCP Pattern

This pattern demonstrates collaborative filtering in an agentic MCP system.
The system recommends items based on user-user similarity and collective
behavior patterns, using both user-based and item-based collaborative filtering.

Use cases:
- Product recommendations
- Content discovery
- Social recommendations
- Similar user finding
- Trend identification
"""

from typing import TypedDict, Annotated, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from collections import defaultdict
import math


# Define the state for collaborative filtering
class CollaborativeFilteringState(TypedDict):
    """State for tracking collaborative filtering process"""
    messages: Annotated[List[str], add]
    user_item_matrix: Dict[str, Dict[str, float]]
    user_similarities: Dict[str, List[Tuple[str, float]]]
    item_similarities: Dict[str, List[Tuple[str, float]]]
    user_based_recommendations: Dict[str, List[Dict[str, Any]]]
    item_based_recommendations: Dict[str, List[Dict[str, Any]]]
    hybrid_recommendations: Dict[str, List[Dict[str, Any]]]
    analytics: Dict[str, Any]
    report: str


class SimilarityCalculator:
    """Calculate similarity between users and items"""
    
    def cosine_similarity(self, vector1: Dict[str, float], 
                         vector2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Get common items
        common_items = set(vector1.keys()) & set(vector2.keys())
        
        if not common_items:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vector1[item] * vector2[item] for item in common_items)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(v ** 2 for v in vector1.values()))
        magnitude2 = math.sqrt(sum(v ** 2 for v in vector2.values()))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def pearson_correlation(self, vector1: Dict[str, float],
                           vector2: Dict[str, float]) -> float:
        """Calculate Pearson correlation coefficient"""
        common_items = set(vector1.keys()) & set(vector2.keys())
        
        if len(common_items) < 2:
            return 0.0
        
        # Calculate means
        mean1 = sum(vector1[item] for item in common_items) / len(common_items)
        mean2 = sum(vector2[item] for item in common_items) / len(common_items)
        
        # Calculate correlation
        numerator = sum((vector1[item] - mean1) * (vector2[item] - mean2) 
                       for item in common_items)
        
        denominator1 = math.sqrt(sum((vector1[item] - mean1) ** 2 
                                    for item in common_items))
        denominator2 = math.sqrt(sum((vector2[item] - mean2) ** 2 
                                    for item in common_items))
        
        if denominator1 == 0 or denominator2 == 0:
            return 0.0
        
        return numerator / (denominator1 * denominator2)
    
    def jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets"""
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class UserBasedCF:
    """User-based collaborative filtering"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
    
    def find_similar_users(self, target_user: str, 
                          user_item_matrix: Dict[str, Dict[str, float]],
                          k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar users"""
        target_ratings = user_item_matrix.get(target_user, {})
        similarities = []
        
        for user, ratings in user_item_matrix.items():
            if user == target_user:
                continue
            
            similarity = self.similarity_calculator.cosine_similarity(
                target_ratings, ratings
            )
            
            if similarity > 0:
                similarities.append((user, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def generate_recommendations(self, target_user: str,
                                user_item_matrix: Dict[str, Dict[str, float]],
                                similar_users: List[Tuple[str, float]],
                                n: int = 5) -> List[Dict[str, Any]]:
        """Generate recommendations based on similar users"""
        target_items = set(user_item_matrix.get(target_user, {}).keys())
        
        # Weighted ratings for items
        item_scores = defaultdict(float)
        item_weights = defaultdict(float)
        
        for similar_user, similarity in similar_users:
            similar_user_ratings = user_item_matrix.get(similar_user, {})
            
            for item, rating in similar_user_ratings.items():
                if item not in target_items:
                    item_scores[item] += rating * similarity
                    item_weights[item] += similarity
        
        # Calculate weighted average
        recommendations = []
        for item, score in item_scores.items():
            if item_weights[item] > 0:
                predicted_rating = score / item_weights[item]
                recommendations.append({
                    'item': item,
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': round(item_weights[item] / sum(s for _, s in similar_users), 2)
                })
        
        # Sort by predicted rating
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:n]


class ItemBasedCF:
    """Item-based collaborative filtering"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
    
    def build_item_vectors(self, user_item_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Build item vectors from user-item matrix"""
        item_vectors = defaultdict(dict)
        
        for user, items in user_item_matrix.items():
            for item, rating in items.items():
                item_vectors[item][user] = rating
        
        return dict(item_vectors)
    
    def find_similar_items(self, target_item: str,
                          item_vectors: Dict[str, Dict[str, float]],
                          k: int = 5) -> List[Tuple[str, float]]:
        """Find k most similar items"""
        target_vector = item_vectors.get(target_item, {})
        similarities = []
        
        for item, vector in item_vectors.items():
            if item == target_item:
                continue
            
            similarity = self.similarity_calculator.cosine_similarity(
                target_vector, vector
            )
            
            if similarity > 0:
                similarities.append((item, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def generate_recommendations(self, target_user: str,
                                user_item_matrix: Dict[str, Dict[str, float]],
                                item_vectors: Dict[str, Dict[str, float]],
                                n: int = 5) -> List[Dict[str, Any]]:
        """Generate recommendations based on similar items"""
        target_items = user_item_matrix.get(target_user, {})
        all_items = set(item_vectors.keys())
        candidate_items = all_items - set(target_items.keys())
        
        recommendations = []
        
        for candidate_item in candidate_items:
            # Find similar items that user has rated
            similar_items = self.find_similar_items(candidate_item, item_vectors, k=5)
            
            # Calculate weighted score
            weighted_sum = 0
            similarity_sum = 0
            
            for similar_item, similarity in similar_items:
                if similar_item in target_items:
                    weighted_sum += similarity * target_items[similar_item]
                    similarity_sum += similarity
            
            if similarity_sum > 0:
                predicted_rating = weighted_sum / similarity_sum
                recommendations.append({
                    'item': candidate_item,
                    'predicted_rating': round(predicted_rating, 2),
                    'confidence': round(min(similarity_sum / 5.0, 1.0), 2)
                })
        
        recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
        return recommendations[:n]


# Agent functions
def initialize_user_item_matrix_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Initialize user-item interaction matrix"""
    
    # Simulate user-item ratings (1-5 scale)
    user_item_matrix = {
        'user_001': {
            'item_a': 5.0, 'item_b': 4.0, 'item_c': 3.0, 'item_e': 5.0,
            'item_f': 4.0, 'item_j': 5.0
        },
        'user_002': {
            'item_a': 4.0, 'item_b': 5.0, 'item_c': 4.0, 'item_d': 3.0,
            'item_g': 4.0
        },
        'user_003': {
            'item_b': 3.0, 'item_c': 5.0, 'item_d': 4.0, 'item_e': 2.0,
            'item_h': 5.0
        },
        'user_004': {
            'item_a': 5.0, 'item_c': 2.0, 'item_f': 5.0, 'item_g': 3.0,
            'item_i': 4.0
        },
        'user_005': {
            'item_b': 4.0, 'item_d': 5.0, 'item_e': 3.0, 'item_h': 4.0,
            'item_j': 3.0
        },
        'user_006': {
            'item_a': 4.0, 'item_e': 5.0, 'item_f': 4.0, 'item_i': 5.0,
            'item_j': 4.0
        },
        'user_007': {
            'item_c': 3.0, 'item_d': 4.0, 'item_g': 5.0, 'item_h': 3.0
        },
        'user_008': {
            'item_a': 5.0, 'item_b': 3.0, 'item_f': 5.0, 'item_i': 4.0
        },
        # Target user with some ratings
        'target_user': {
            'item_a': 5.0, 'item_c': 4.0, 'item_f': 5.0
        }
    }
    
    total_ratings = sum(len(ratings) for ratings in user_item_matrix.values())
    
    return {
        **state,
        'user_item_matrix': user_item_matrix,
        'messages': state['messages'] + [f'Initialized user-item matrix with {len(user_item_matrix)} users and {total_ratings} ratings']
    }


def calculate_user_similarities_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Calculate user-user similarities"""
    
    calculator = SimilarityCalculator()
    user_cf = UserBasedCF(calculator)
    
    user_similarities = {}
    
    # Calculate similarities for all users
    for user in state['user_item_matrix']:
        similar_users = user_cf.find_similar_users(
            user, 
            state['user_item_matrix'],
            k=5
        )
        user_similarities[user] = similar_users
    
    return {
        **state,
        'user_similarities': user_similarities,
        'messages': state['messages'] + [f'Calculated similarities for {len(user_similarities)} users']
    }


def calculate_item_similarities_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Calculate item-item similarities"""
    
    calculator = SimilarityCalculator()
    item_cf = ItemBasedCF(calculator)
    
    # Build item vectors
    item_vectors = item_cf.build_item_vectors(state['user_item_matrix'])
    
    item_similarities = {}
    
    # Calculate similarities for all items
    for item in item_vectors:
        similar_items = item_cf.find_similar_items(item, item_vectors, k=5)
        item_similarities[item] = similar_items
    
    return {
        **state,
        'item_similarities': item_similarities,
        'messages': state['messages'] + [f'Calculated similarities for {len(item_similarities)} items']
    }


def generate_user_based_recommendations_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Generate user-based collaborative filtering recommendations"""
    
    calculator = SimilarityCalculator()
    user_cf = UserBasedCF(calculator)
    
    user_based_recommendations = {}
    
    # Generate recommendations for target user
    target_user = 'target_user'
    similar_users = state['user_similarities'].get(target_user, [])
    
    recommendations = user_cf.generate_recommendations(
        target_user,
        state['user_item_matrix'],
        similar_users,
        n=5
    )
    
    user_based_recommendations[target_user] = recommendations
    
    return {
        **state,
        'user_based_recommendations': user_based_recommendations,
        'messages': state['messages'] + [f'Generated user-based recommendations for target user']
    }


def generate_item_based_recommendations_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Generate item-based collaborative filtering recommendations"""
    
    calculator = SimilarityCalculator()
    item_cf = ItemBasedCF(calculator)
    
    # Build item vectors
    item_vectors = item_cf.build_item_vectors(state['user_item_matrix'])
    
    item_based_recommendations = {}
    
    # Generate recommendations for target user
    target_user = 'target_user'
    
    recommendations = item_cf.generate_recommendations(
        target_user,
        state['user_item_matrix'],
        item_vectors,
        n=5
    )
    
    item_based_recommendations[target_user] = recommendations
    
    return {
        **state,
        'item_based_recommendations': item_based_recommendations,
        'messages': state['messages'] + [f'Generated item-based recommendations for target user']
    }


def generate_hybrid_recommendations_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Generate hybrid recommendations combining both approaches"""
    
    target_user = 'target_user'
    
    # Combine user-based and item-based recommendations
    user_based = {rec['item']: rec for rec in state['user_based_recommendations'].get(target_user, [])}
    item_based = {rec['item']: rec for rec in state['item_based_recommendations'].get(target_user, [])}
    
    all_items = set(user_based.keys()) | set(item_based.keys())
    
    hybrid_recommendations = []
    
    for item in all_items:
        # Weighted combination (60% user-based, 40% item-based)
        user_score = user_based.get(item, {}).get('predicted_rating', 0) * 0.6
        item_score = item_based.get(item, {}).get('predicted_rating', 0) * 0.4
        
        # Average confidence
        user_conf = user_based.get(item, {}).get('confidence', 0)
        item_conf = item_based.get(item, {}).get('confidence', 0)
        avg_confidence = (user_conf + item_conf) / 2 if (user_conf or item_conf) else 0
        
        hybrid_score = user_score + item_score
        
        if hybrid_score > 0:
            hybrid_recommendations.append({
                'item': item,
                'hybrid_score': round(hybrid_score, 2),
                'user_based_score': round(user_based.get(item, {}).get('predicted_rating', 0), 2),
                'item_based_score': round(item_based.get(item, {}).get('predicted_rating', 0), 2),
                'confidence': round(avg_confidence, 2),
                'source': 'both' if (item in user_based and item in item_based) else 
                         ('user_based' if item in user_based else 'item_based')
            })
    
    # Sort by hybrid score
    hybrid_recommendations.sort(key=lambda x: x['hybrid_score'], reverse=True)
    
    return {
        **state,
        'hybrid_recommendations': {target_user: hybrid_recommendations[:5]},
        'messages': state['messages'] + [f'Generated hybrid recommendations combining both approaches']
    }


def analyze_collaborative_filtering_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Analyze collaborative filtering results"""
    
    target_user = 'target_user'
    
    # Analyze user similarities
    similar_users = state['user_similarities'].get(target_user, [])
    avg_user_similarity = sum(sim for _, sim in similar_users) / len(similar_users) if similar_users else 0
    
    # Analyze item coverage
    all_items = set()
    for ratings in state['user_item_matrix'].values():
        all_items.update(ratings.keys())
    
    target_rated = set(state['user_item_matrix'].get(target_user, {}).keys())
    
    user_recs = set(rec['item'] for rec in state['user_based_recommendations'].get(target_user, []))
    item_recs = set(rec['item'] for rec in state['item_based_recommendations'].get(target_user, []))
    hybrid_recs = set(rec['item'] for rec in state['hybrid_recommendations'].get(target_user, []))
    
    # Agreement between methods
    agreement = len(user_recs & item_recs)
    
    analytics = {
        'matrix_stats': {
            'total_users': len(state['user_item_matrix']),
            'total_items': len(all_items),
            'total_ratings': sum(len(r) for r in state['user_item_matrix'].values()),
            'sparsity': 1 - (sum(len(r) for r in state['user_item_matrix'].values()) / 
                           (len(state['user_item_matrix']) * len(all_items)))
        },
        'target_user_stats': {
            'items_rated': len(target_rated),
            'similar_users_found': len(similar_users),
            'avg_user_similarity': round(avg_user_similarity, 3)
        },
        'recommendation_stats': {
            'user_based_count': len(state['user_based_recommendations'].get(target_user, [])),
            'item_based_count': len(state['item_based_recommendations'].get(target_user, [])),
            'hybrid_count': len(state['hybrid_recommendations'].get(target_user, [])),
            'method_agreement': agreement,
            'unique_recommendations': len(user_recs | item_recs)
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed collaborative filtering results']
    }


def generate_collaborative_filtering_report_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Generate comprehensive collaborative filtering report"""
    
    target_user = 'target_user'
    
    report_lines = [
        "=" * 80,
        "COLLABORATIVE FILTERING REPORT",
        "=" * 80,
        "",
        "MATRIX STATISTICS:",
        "-" * 40,
        f"Total Users: {state['analytics']['matrix_stats']['total_users']}",
        f"Total Items: {state['analytics']['matrix_stats']['total_items']}",
        f"Total Ratings: {state['analytics']['matrix_stats']['total_ratings']}",
        f"Matrix Sparsity: {state['analytics']['matrix_stats']['sparsity']:.2%}",
        "",
        "TARGET USER PROFILE:",
        "-" * 40,
        f"User ID: {target_user}",
        f"Items Rated: {state['analytics']['target_user_stats']['items_rated']}",
    ]
    
    # Show rated items
    target_ratings = state['user_item_matrix'].get(target_user, {})
    report_lines.append("\nRated Items:")
    for item, rating in sorted(target_ratings.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  • {item}: {rating}/5.0")
    
    report_lines.extend([
        "",
        "SIMILAR USERS:",
        "-" * 40,
        f"Found: {state['analytics']['target_user_stats']['similar_users_found']} similar users",
        f"Average Similarity: {state['analytics']['target_user_stats']['avg_user_similarity']:.3f}",
        ""
    ])
    
    similar_users = state['user_similarities'].get(target_user, [])
    for user, similarity in similar_users[:5]:
        report_lines.append(f"  {user}: similarity = {similarity:.3f}")
    
    report_lines.extend([
        "",
        "USER-BASED RECOMMENDATIONS:",
        "-" * 40
    ])
    
    for rec in state['user_based_recommendations'].get(target_user, [])[:5]:
        report_lines.append(f"\n• {rec['item']}")
        report_lines.append(f"  Predicted Rating: {rec['predicted_rating']}/5.0")
        report_lines.append(f"  Confidence: {rec['confidence']:.2f}")
    
    report_lines.extend([
        "",
        "",
        "ITEM-BASED RECOMMENDATIONS:",
        "-" * 40
    ])
    
    for rec in state['item_based_recommendations'].get(target_user, [])[:5]:
        report_lines.append(f"\n• {rec['item']}")
        report_lines.append(f"  Predicted Rating: {rec['predicted_rating']}/5.0")
        report_lines.append(f"  Confidence: {rec['confidence']:.2f}")
    
    report_lines.extend([
        "",
        "",
        "HYBRID RECOMMENDATIONS (60% User + 40% Item):",
        "-" * 40
    ])
    
    for rec in state['hybrid_recommendations'].get(target_user, [])[:5]:
        report_lines.append(f"\n• {rec['item']}")
        report_lines.append(f"  Hybrid Score: {rec['hybrid_score']:.2f}")
        report_lines.append(f"  User-Based: {rec['user_based_score']:.2f} | Item-Based: {rec['item_based_score']:.2f}")
        report_lines.append(f"  Confidence: {rec['confidence']:.2f}")
        report_lines.append(f"  Source: {rec['source']}")
    
    report_lines.extend([
        "",
        "",
        "RECOMMENDATION ANALYSIS:",
        "-" * 40,
        f"User-Based Recommendations: {state['analytics']['recommendation_stats']['user_based_count']}",
        f"Item-Based Recommendations: {state['analytics']['recommendation_stats']['item_based_count']}",
        f"Hybrid Recommendations: {state['analytics']['recommendation_stats']['hybrid_count']}",
        f"Method Agreement: {state['analytics']['recommendation_stats']['method_agreement']} items",
        f"Unique Recommendations: {state['analytics']['recommendation_stats']['unique_recommendations']} items",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Matrix sparsity: {state['analytics']['matrix_stats']['sparsity']:.1%} (typical for collaborative filtering)",
        f"✓ Found {len(similar_users)} similar users with avg similarity {state['analytics']['target_user_stats']['avg_user_similarity']:.3f}",
        f"✓ Both methods agree on {state['analytics']['recommendation_stats']['method_agreement']} recommendations",
        f"✓ Hybrid approach provides balanced recommendations",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Use hybrid approach for best results",
        "• User-based CF works well for finding niche items",
        "• Item-based CF is more stable and scalable",
        "• Consider matrix factorization for large-scale systems",
        "• Implement implicit feedback for better coverage",
        "• Use neighborhood selection carefully (k=5-20)",
        "• Apply significance weighting for sparse data",
        "• Combine with content-based filtering for cold-start",
        "",
        "=" * 80
    ]
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive collaborative filtering report']
    }


# Create the graph
def create_collaborative_filtering_graph():
    """Create the collaborative filtering workflow graph"""
    
    workflow = StateGraph(CollaborativeFilteringState)
    
    # Add nodes
    workflow.add_node("initialize_matrix", initialize_user_item_matrix_agent)
    workflow.add_node("calculate_user_similarities", calculate_user_similarities_agent)
    workflow.add_node("calculate_item_similarities", calculate_item_similarities_agent)
    workflow.add_node("generate_user_based", generate_user_based_recommendations_agent)
    workflow.add_node("generate_item_based", generate_item_based_recommendations_agent)
    workflow.add_node("generate_hybrid", generate_hybrid_recommendations_agent)
    workflow.add_node("analyze_results", analyze_collaborative_filtering_agent)
    workflow.add_node("generate_report", generate_collaborative_filtering_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize_matrix")
    workflow.add_edge("initialize_matrix", "calculate_user_similarities")
    workflow.add_edge("calculate_user_similarities", "calculate_item_similarities")
    workflow.add_edge("calculate_item_similarities", "generate_user_based")
    workflow.add_edge("generate_user_based", "generate_item_based")
    workflow.add_edge("generate_item_based", "generate_hybrid")
    workflow.add_edge("generate_hybrid", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the collaborative filtering graph
    app = create_collaborative_filtering_graph()
    
    # Initialize state
    initial_state: CollaborativeFilteringState = {
        'messages': [],
        'user_item_matrix': {},
        'user_similarities': {},
        'item_similarities': {},
        'user_based_recommendations': {},
        'item_based_recommendations': {},
        'hybrid_recommendations': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("COLLABORATIVE FILTERING MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nCollaborative filtering pattern execution complete! ✓")

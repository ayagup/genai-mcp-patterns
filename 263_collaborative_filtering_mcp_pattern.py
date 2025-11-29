"""
Pattern 263: Collaborative Filtering MCP Pattern

This pattern demonstrates collaborative filtering - leveraging collective
preferences and behaviors to make predictions and recommendations.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CollaborativeFilteringState(TypedDict):
    """State for collaborative filtering workflow"""
    messages: Annotated[List[str], add]
    user_ratings: Dict[str, Dict[str, float]]
    target_user: str
    recommendations: List[Dict[str, Any]]
    similarity_scores: Dict[str, float]


class CollaborativeFilter:
    """Implements collaborative filtering algorithm"""
    
    def __init__(self):
        self.user_item_matrix = {}
        self.similarities = {}
    
    def add_ratings(self, user_id: str, ratings: Dict[str, float]):
        """Add user ratings"""
        self.user_item_matrix[user_id] = ratings
    
    def calculate_similarity(self, user1: str, user2: str) -> float:
        """Calculate similarity between two users (cosine similarity)"""
        ratings1 = self.user_item_matrix.get(user1, {})
        ratings2 = self.user_item_matrix.get(user2, {})
        
        # Find common items
        common_items = set(ratings1.keys()) & set(ratings2.keys())
        
        if not common_items:
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(ratings1[item] * ratings2[item] for item in common_items)
        norm1 = sum(ratings1[item]**2 for item in common_items) ** 0.5
        norm2 = sum(ratings2[item]**2 for item in common_items) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_similar_users(self, target_user: str, k: int = 3) -> List[tuple]:
        """Find k most similar users"""
        similarities = []
        
        for user in self.user_item_matrix:
            if user != target_user:
                sim = self.calculate_similarity(target_user, user)
                similarities.append((user, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def predict_rating(self, target_user: str, item: str, similar_users: List[tuple]) -> float:
        """Predict rating using weighted average"""
        if not similar_users:
            return 0.0
        
        weighted_sum = 0.0
        similarity_sum = 0.0
        
        for user, similarity in similar_users:
            user_ratings = self.user_item_matrix.get(user, {})
            if item in user_ratings:
                weighted_sum += similarity * user_ratings[item]
                similarity_sum += similarity
        
        if similarity_sum == 0:
            return 0.0
        
        return weighted_sum / similarity_sum
    
    def recommend(self, target_user: str, n: int = 5) -> List[Dict[str, Any]]:
        """Generate top N recommendations"""
        target_ratings = self.user_item_matrix.get(target_user, {})
        similar_users = self.find_similar_users(target_user)
        
        # Find items not rated by target user
        all_items = set()
        for user_ratings in self.user_item_matrix.values():
            all_items.update(user_ratings.keys())
        
        unrated_items = all_items - set(target_ratings.keys())
        
        # Predict ratings for unrated items
        predictions = []
        for item in unrated_items:
            predicted_rating = self.predict_rating(target_user, item, similar_users)
            if predicted_rating > 0:
                predictions.append({
                    "item": item,
                    "predicted_rating": predicted_rating,
                    "confidence": min(len(similar_users) / 5, 1.0)  # More similar users = higher confidence
                })
        
        # Sort by predicted rating (descending)
        predictions.sort(key=lambda x: x["predicted_rating"], reverse=True)
        return predictions[:n]


def load_user_ratings_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Load user rating data"""
    print("\n📊 Loading User Ratings...")
    
    # Sample user-item ratings (1-5 scale)
    user_ratings = {
        "Alice": {
            "Movie_A": 5.0,
            "Movie_B": 3.0,
            "Movie_C": 4.0,
            "Movie_D": 1.0,
            "Movie_E": 5.0
        },
        "Bob": {
            "Movie_A": 4.0,
            "Movie_B": 3.0,
            "Movie_C": 5.0,
            "Movie_F": 4.0,
            "Movie_G": 3.0
        },
        "Carol": {
            "Movie_A": 5.0,
            "Movie_C": 4.0,
            "Movie_D": 2.0,
            "Movie_E": 5.0,
            "Movie_F": 3.0
        },
        "David": {
            "Movie_B": 4.0,
            "Movie_C": 3.0,
            "Movie_F": 5.0,
            "Movie_G": 4.0,
            "Movie_H": 5.0
        },
        "Eve": {
            "Movie_A": 4.0,
            "Movie_B": 2.0,
            "Movie_C": 4.0,
            "Movie_E": 4.0
        }
    }
    
    target_user = "Eve"
    
    print(f"\n  Users: {len(user_ratings)}")
    print(f"  Target User: {target_user}")
    print(f"\n  Ratings Matrix:")
    for user, ratings in user_ratings.items():
        marker = " (TARGET)" if user == target_user else ""
        print(f"    {user}{marker}: {len(ratings)} items rated")
    
    return {
        **state,
        "user_ratings": user_ratings,
        "target_user": target_user,
        "messages": [f"✓ Loaded ratings for {len(user_ratings)} users"]
    }


def generate_recommendations_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Generate collaborative filtering recommendations"""
    print("\n🎯 Generating Recommendations...")
    
    # Build collaborative filter
    cf = CollaborativeFilter()
    for user, ratings in state["user_ratings"].items():
        cf.add_ratings(user, ratings)
    
    # Find similar users
    similar_users = cf.find_similar_users(state["target_user"], k=3)
    
    print(f"\n  Similar Users to {state['target_user']}:")
    similarity_scores = {}
    for user, similarity in similar_users:
        print(f"    • {user}: {similarity:.3f}")
        similarity_scores[user] = similarity
    
    # Generate recommendations
    recommendations = cf.recommend(state["target_user"], n=5)
    
    print(f"\n  Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec['item']}")
        print(f"       Predicted Rating: {rec['predicted_rating']:.2f}/5.0")
        print(f"       Confidence: {rec['confidence']:.0%}")
    
    return {
        **state,
        "recommendations": recommendations,
        "similarity_scores": similarity_scores,
        "messages": [f"✓ Generated {len(recommendations)} recommendations"]
    }


def generate_collaborative_report_agent(state: CollaborativeFilteringState) -> CollaborativeFilteringState:
    """Generate collaborative filtering report"""
    print("\n" + "="*70)
    print("COLLABORATIVE FILTERING REPORT")
    print("="*70)
    
    target = state["target_user"]
    print(f"\n👤 Target User: {target}")
    print(f"\n  Rated Items:")
    target_ratings = state["user_ratings"].get(target, {})
    for item, rating in sorted(target_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"    • {item}: {rating:.1f}/5.0")
    
    print(f"\n🔍 Similar Users:")
    for user, similarity in state["similarity_scores"].items():
        print(f"\n  {user} (Similarity: {similarity:.3f})")
        user_ratings = state["user_ratings"].get(user, {})
        print(f"    Rated Items:")
        for item, rating in sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"      • {item}: {rating:.1f}/5.0")
    
    print(f"\n🎯 Top Recommendations for {target}:")
    for i, rec in enumerate(state["recommendations"], 1):
        print(f"\n  {i}. {rec['item']}")
        print(f"     Predicted Rating: {rec['predicted_rating']:.2f}/5.0")
        print(f"     Confidence: {rec['confidence']:.0%}")
        
        # Show which similar users rated this
        print(f"     Based on ratings from:")
        for user in state["similarity_scores"].keys():
            user_ratings = state["user_ratings"].get(user, {})
            if rec["item"] in user_ratings:
                print(f"       • {user}: {user_ratings[rec['item']]:.1f}/5.0")
    
    print(f"\n💡 Collaborative Filtering Benefits:")
    print("  • Leverages collective wisdom")
    print("  • Discovers hidden patterns")
    print("  • Personalized recommendations")
    print("  • No content analysis needed")
    print("  • Serendipitous discoveries")
    print("  • Scalable approach")
    
    print(f"\n📊 Filtering Statistics:")
    all_items = set()
    for ratings in state["user_ratings"].values():
        all_items.update(ratings.keys())
    print(f"  Total Items: {len(all_items)}")
    print(f"  Items Rated by {target}: {len(target_ratings)}")
    print(f"  Items Not Rated: {len(all_items) - len(target_ratings)}")
    print(f"  Recommendations Generated: {len(state['recommendations'])}")
    
    if state["recommendations"]:
        avg_predicted = sum(r["predicted_rating"] for r in state["recommendations"]) / len(state["recommendations"])
        print(f"  Average Predicted Rating: {avg_predicted:.2f}/5.0")
    
    print("\n="*70)
    print("✅ Collaborative Filtering Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_collaborative_filtering_graph():
    workflow = StateGraph(CollaborativeFilteringState)
    workflow.add_node("load", load_user_ratings_agent)
    workflow.add_node("recommend", generate_recommendations_agent)
    workflow.add_node("report", generate_collaborative_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "recommend")
    workflow.add_edge("recommend", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 263: Collaborative Filtering MCP Pattern")
    print("="*70)
    
    app = create_collaborative_filtering_graph()
    final_state = app.invoke({
        "messages": [],
        "user_ratings": {},
        "target_user": "",
        "recommendations": [],
        "similarity_scores": {}
    })
    print("\n✅ Collaborative Filtering Pattern Complete!")


if __name__ == "__main__":
    main()

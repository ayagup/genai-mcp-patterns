"""
Pattern 283: Session State MCP Pattern

This pattern demonstrates session-based state management for maintaining
user-specific state across multiple requests within a session.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class SessionStatePattern(TypedDict):
    """State for session management"""
    messages: Annotated[List[str], add]
    session_data: Dict[str, Any]
    session_history: List[Dict[str, Any]]
    session_analytics: Dict[str, Any]


class SessionManager:
    """Manages user sessions"""
    
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 1800  # 30 minutes
    
    def create_session(self, user_id: str) -> Dict[str, Any]:
        """Create new session"""
        session_id = f"sess_{user_id}_{int(time.time())}"
        
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "data": {},
            "interaction_count": 0,
            "context": [],
            "preferences": {},
            "cart": [],
            "navigation_history": []
        }
        
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Check timeout
            if time.time() - session["last_activity"] < self.session_timeout:
                session["last_activity"] = time.time()
                return session
            else:
                # Session expired
                del self.sessions[session_id]
                return None
        return None
    
    def update_session(self, session_id: str, updates: Dict[str, Any]):
        """Update session data"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session["last_activity"] = time.time()
            session["interaction_count"] += 1
            
            for key, value in updates.items():
                if key == "cart":
                    session["cart"].extend(value if isinstance(value, list) else [value])
                elif key == "navigation":
                    session["navigation_history"].append(value)
                elif key == "context":
                    session["context"].append(value)
                else:
                    session["data"][key] = value


def create_session_agent(state: SessionStatePattern) -> SessionStatePattern:
    """Create user session"""
    print("\n🔐 Creating Session...")
    
    manager = SessionManager()
    user_id = "user_123"
    session = manager.create_session(user_id)
    
    print(f"  Session ID: {session['session_id']}")
    print(f"  User ID: {session['user_id']}")
    print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session['created_at']))}")
    
    return {
        **state,
        "session_data": session,
        "messages": [f"✓ Session created: {session['session_id']}"]
    }


def track_interactions_agent(state: SessionStatePattern) -> SessionStatePattern:
    """Track user interactions within session"""
    print("\n👤 Tracking Session Interactions...")
    
    manager = SessionManager()
    session_id = state["session_data"]["session_id"]
    
    # Simulate user interactions
    interactions = [
        {"action": "view_product", "product_id": "P001", "page": "/products/laptop"},
        {"action": "add_to_cart", "product_id": "P001", "quantity": 1},
        {"action": "view_product", "product_id": "P002", "page": "/products/mouse"},
        {"action": "update_preferences", "theme": "dark", "language": "en"},
        {"action": "add_to_cart", "product_id": "P002", "quantity": 2}
    ]
    
    history = []
    
    for i, interaction in enumerate(interactions, 1):
        # Update session
        updates = {}
        if interaction["action"] == "add_to_cart":
            updates["cart"] = {"product_id": interaction["product_id"], "quantity": interaction["quantity"]}
        
        if "page" in interaction:
            updates["navigation"] = interaction["page"]
        
        if interaction["action"] == "update_preferences":
            updates["theme"] = interaction.get("theme")
            updates["language"] = interaction.get("language")
        
        manager.update_session(session_id, updates)
        
        history.append({
            "sequence": i,
            "timestamp": time.time(),
            "action": interaction["action"],
            "details": interaction
        })
        
        print(f"  #{i}: {interaction['action']}")
    
    # Get updated session
    updated_session = manager.get_session(session_id)
    
    return {
        **state,
        "session_data": updated_session,
        "session_history": history,
        "messages": [f"✓ Tracked {len(interactions)} interactions"]
    }


def analyze_session_agent(state: SessionStatePattern) -> SessionStatePattern:
    """Analyze session data"""
    print("\n📊 Analyzing Session...")
    
    session = state["session_data"]
    history = state["session_history"]
    
    analytics = {
        "session_duration": time.time() - session["created_at"],
        "total_interactions": session["interaction_count"],
        "cart_items": len(session["cart"]),
        "pages_visited": len(session["navigation_history"]),
        "unique_products_viewed": len(set(h["details"].get("product_id", "") for h in history if h["action"] == "view_product")),
        "conversion_rate": len([h for h in history if h["action"] == "add_to_cart"]) / max(len([h for h in history if h["action"] == "view_product"]), 1),
        "session_value": sum(item.get("quantity", 0) * 100 for item in session["cart"])  # Assuming $100 per item
    }
    
    print(f"  Duration: {analytics['session_duration']:.1f}s")
    print(f"  Interactions: {analytics['total_interactions']}")
    print(f"  Cart Items: {analytics['cart_items']}")
    print(f"  Conversion Rate: {analytics['conversion_rate']:.1%}")
    
    return {
        **state,
        "session_analytics": analytics,
        "messages": ["✓ Session analytics generated"]
    }


def generate_session_report_agent(state: SessionStatePattern) -> SessionStatePattern:
    """Generate session state report"""
    print("\n" + "="*70)
    print("SESSION STATE REPORT")
    print("="*70)
    
    session = state["session_data"]
    history = state["session_history"]
    analytics = state["session_analytics"]
    
    print(f"\n🔐 Session Information:")
    print(f"  Session ID: {session['session_id']}")
    print(f"  User ID: {session['user_id']}")
    print(f"  Created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session['created_at']))}")
    print(f"  Last Activity: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(session['last_activity']))}")
    print(f"  Interaction Count: {session['interaction_count']}")
    
    print(f"\n🛒 Shopping Cart ({len(session['cart'])} items):")
    for item in session["cart"]:
        print(f"  • Product {item['product_id']}: Quantity {item['quantity']}")
    
    print(f"\n🧭 Navigation History:")
    for i, page in enumerate(session["navigation_history"][-5:], 1):
        print(f"  {i}. {page}")
    
    print(f"\n📋 Session Data:")
    for key, value in session["data"].items():
        print(f"  {key}: {value}")
    
    print(f"\n📊 Session Analytics:")
    print(f"  Session Duration: {analytics['session_duration']:.1f} seconds")
    print(f"  Total Interactions: {analytics['total_interactions']}")
    print(f"  Pages Visited: {analytics['pages_visited']}")
    print(f"  Products Viewed: {analytics['unique_products_viewed']}")
    print(f"  Cart Items: {analytics['cart_items']}")
    print(f"  Conversion Rate: {analytics['conversion_rate']:.1%}")
    print(f"  Session Value: ${analytics['session_value']:.2f}")
    
    print(f"\n📜 Interaction Timeline:")
    for entry in history:
        print(f"  #{entry['sequence']}: {entry['action']}")
        for key, value in entry["details"].items():
            if key != "action":
                print(f"    {key}: {value}")
    
    print(f"\n💡 Session State Benefits:")
    print("  ✓ User-specific context")
    print("  ✓ Shopping cart persistence")
    print("  ✓ Navigation tracking")
    print("  ✓ Personalized experience")
    print("  ✓ Temporary data storage")
    print("  ✓ Session timeout management")
    
    print(f"\n🔄 Session Lifecycle:")
    print("  1. Session creation (login/visit)")
    print("  2. Activity tracking")
    print("  3. State updates")
    print("  4. Timeout monitoring")
    print("  5. Session cleanup (logout/timeout)")
    
    print(f"\n⚙️ Use Cases:")
    print("  • E-commerce shopping carts")
    print("  • Multi-step forms")
    print("  • User preferences")
    print("  • Temporary data storage")
    print("  • Activity tracking")
    
    print("\n" + "="*70)
    print("✅ Session State Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_session_state_graph():
    """Create session state workflow"""
    workflow = StateGraph(SessionStatePattern)
    
    workflow.add_node("create", create_session_agent)
    workflow.add_node("track", track_interactions_agent)
    workflow.add_node("analyze", analyze_session_agent)
    workflow.add_node("report", generate_session_report_agent)
    
    workflow.add_edge(START, "create")
    workflow.add_edge("create", "track")
    workflow.add_edge("track", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 283: Session State MCP Pattern")
    print("="*70)
    
    app = create_session_state_graph()
    final_state = app.invoke({
        "messages": [],
        "session_data": {},
        "session_history": [],
        "session_analytics": {}
    })
    
    print("\n✅ Session State Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Pattern 282: Stateful MCP Pattern

This pattern demonstrates stateful agent architecture where state is maintained
across multiple interactions and requests.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class StatefulState(TypedDict):
    """State maintained across interactions"""
    messages: Annotated[List[str], add]
    session_id: str
    user_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    accumulated_data: Dict[str, Any]
    state_snapshot: Dict[str, Any]


class StatefulAgent:
    """Agent that maintains state across interactions"""
    
    def __init__(self):
        self.sessions = {}  # Session storage
        self.state_version = 0
    
    def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "created_at": "2025-11-29T10:00:00Z",
                "interactions": 0,
                "context": {},
                "history": [],
                "preferences": {},
                "accumulated_score": 0
            }
        return self.sessions[session_id]
    
    def update_session(self, session_id: str, interaction: Dict[str, Any]):
        """Update session state with new interaction"""
        session = self.get_or_create_session(session_id)
        
        session["interactions"] += 1
        session["history"].append(interaction)
        session["last_interaction"] = interaction
        
        # Update context based on interaction
        if "user_input" in interaction:
            self._extract_context(session, interaction["user_input"])
        
        self.state_version += 1
    
    def _extract_context(self, session: Dict[str, Any], user_input: str):
        """Extract and accumulate context from user input"""
        # Simple context extraction
        if "prefer" in user_input.lower():
            session["preferences"]["last_preference"] = user_input
        
        if "name" in user_input.lower():
            words = user_input.split()
            if len(words) > 1:
                session["context"]["mentioned_name"] = True
        
        # Accumulate score based on interaction quality
        session["accumulated_score"] += len(user_input.split())


def initialize_session_agent(state: StatefulState) -> StatefulState:
    """Initialize stateful session"""
    print("\n🔄 Initializing Stateful Session...")
    
    session_id = "SESSION-12345"
    
    agent = StatefulAgent()
    session = agent.get_or_create_session(session_id)
    
    print(f"  Session ID: {session_id}")
    print(f"  Created: {session['created_at']}")
    print(f"  Interactions: {session['interactions']}")
    
    return {
        **state,
        "session_id": session_id,
        "user_context": session.get("context", {}),
        "conversation_history": session.get("history", []),
        "messages": [f"✓ Session initialized: {session_id}"]
    }


def process_interaction_agent(state: StatefulState) -> StatefulState:
    """Process interaction and update state"""
    print("\n💬 Processing Interaction...")
    
    # Simulate user interactions
    interactions = [
        {"turn": 1, "user_input": "Hello, my name is Alice", "timestamp": "10:00:00"},
        {"turn": 2, "user_input": "I prefer technical content", "timestamp": "10:01:00"},
        {"turn": 3, "user_input": "Show me advanced topics", "timestamp": "10:02:00"}
    ]
    
    agent = StatefulAgent()
    conversation_history = []
    
    for interaction in interactions:
        agent.update_session(state["session_id"], interaction)
        session = agent.get_or_create_session(state["session_id"])
        
        print(f"\n  Turn {interaction['turn']}: {interaction['user_input']}")
        print(f"    Total Interactions: {session['interactions']}")
        print(f"    Accumulated Score: {session['accumulated_score']}")
        
        conversation_history.append(interaction)
    
    # Get updated session
    final_session = agent.get_or_create_session(state["session_id"])
    
    return {
        **state,
        "conversation_history": conversation_history,
        "user_context": final_session["context"],
        "accumulated_data": {
            "total_interactions": final_session["interactions"],
            "accumulated_score": final_session["accumulated_score"],
            "preferences": final_session["preferences"]
        },
        "messages": [f"✓ Processed {len(interactions)} interactions"]
    }


def maintain_state_agent(state: StatefulState) -> StatefulState:
    """Maintain and persist state"""
    print("\n💾 Maintaining State...")
    
    # Create state snapshot
    snapshot = {
        "session_id": state["session_id"],
        "timestamp": "2025-11-29T10:03:00Z",
        "conversation_length": len(state["conversation_history"]),
        "context_keys": list(state["user_context"].keys()),
        "accumulated_data": state["accumulated_data"],
        "state_hash": hash(str(state["accumulated_data"]))
    }
    
    print(f"  Session: {snapshot['session_id']}")
    print(f"  Conversation Length: {snapshot['conversation_length']}")
    print(f"  Context Keys: {snapshot['context_keys']}")
    print(f"  State Version: {abs(snapshot['state_hash']) % 1000}")
    
    return {
        **state,
        "state_snapshot": snapshot,
        "messages": ["✓ State snapshot created"]
    }


def generate_stateful_report_agent(state: StatefulState) -> StatefulState:
    """Generate stateful processing report"""
    print("\n" + "="*70)
    print("STATEFUL PROCESSING REPORT")
    print("="*70)
    
    print(f"\n🔄 Session Information:")
    print(f"  Session ID: {state['session_id']}")
    print(f"  Total Interactions: {state['accumulated_data']['total_interactions']}")
    print(f"  Accumulated Score: {state['accumulated_data']['accumulated_score']}")
    
    print(f"\n💬 Conversation History:")
    for interaction in state["conversation_history"]:
        print(f"  Turn {interaction['turn']}: \"{interaction['user_input']}\"")
        print(f"    Timestamp: {interaction['timestamp']}")
    
    print(f"\n📊 User Context:")
    if state["user_context"]:
        for key, value in state["user_context"].items():
            print(f"  {key}: {value}")
    else:
        print("  (No context extracted yet)")
    
    print(f"\n💾 Accumulated Data:")
    for key, value in state["accumulated_data"].items():
        if key == "preferences" and isinstance(value, dict):
            print(f"  {key}:")
            for pref_key, pref_val in value.items():
                print(f"    {pref_key}: {pref_val}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📸 State Snapshot:")
    snapshot = state["state_snapshot"]
    print(f"  Timestamp: {snapshot['timestamp']}")
    print(f"  Conversation Length: {snapshot['conversation_length']}")
    print(f"  State Version: {abs(snapshot['state_hash']) % 1000}")
    
    print(f"\n💡 Stateful Architecture Benefits:")
    print("  ✓ Maintains context across interactions")
    print("  ✓ Personalized experiences")
    print("  ✓ Conversation continuity")
    print("  ✓ Progressive learning")
    print("  ✓ User preferences retained")
    print("  ✓ Historical tracking")
    
    print(f"\n⚠️ Stateful Considerations:")
    print("  • Requires session management")
    print("  • State synchronization needed")
    print("  • Memory overhead")
    print("  • Scaling complexity")
    print("  • State persistence required")
    
    print(f"\n🔄 State Lifecycle:")
    print("  1. Session initialization")
    print("  2. State accumulation")
    print("  3. Context updates")
    print("  4. State persistence")
    print("  5. Session cleanup (eventual)")
    
    print("\n" + "="*70)
    print("✅ Stateful Processing Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_stateful_graph():
    """Create stateful processing workflow"""
    workflow = StateGraph(StatefulState)
    
    workflow.add_node("initialize", initialize_session_agent)
    workflow.add_node("interact", process_interaction_agent)
    workflow.add_node("maintain", maintain_state_agent)
    workflow.add_node("report", generate_stateful_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "interact")
    workflow.add_edge("interact", "maintain")
    workflow.add_edge("maintain", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 282: Stateful MCP Pattern")
    print("="*70)
    print("\n💡 Demonstrating stateful architecture:")
    print("  • State maintained across interactions")
    print("  • Session-based processing")
    print("  • Context accumulation")
    
    app = create_stateful_graph()
    final_state = app.invoke({
        "messages": [],
        "session_id": "",
        "user_context": {},
        "conversation_history": [],
        "accumulated_data": {},
        "state_snapshot": {}
    })
    
    print("\n✅ Stateful Pattern Complete!")


if __name__ == "__main__":
    main()

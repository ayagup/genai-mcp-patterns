"""
Pattern 247: Context Prioritization MCP Pattern

This pattern demonstrates context prioritization - ranking and ordering context
items based on importance, urgency, and relevance.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextPrioritizationState(TypedDict):
    """State for context prioritization workflow"""
    messages: Annotated[List[str], add]
    context_items: List[Dict[str, Any]]
    prioritized_items: List[Dict[str, Any]]


class ContextPrioritizer:
    """Prioritizes context items"""
    
    def calculate_priority_score(self, item: Dict[str, Any]) -> float:
        """Calculate priority score"""
        importance = item.get("importance", 5)  # 1-10
        urgency = item.get("urgency", 5)  # 1-10
        relevance = item.get("relevance", 5)  # 1-10
        
        # Weighted score
        score = (importance * 0.4) + (urgency * 0.35) + (relevance * 0.25)
        return round(score, 2)
    
    def prioritize(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize items by score"""
        prioritized = []
        
        for item in items:
            item_copy = item.copy()
            item_copy["priority_score"] = self.calculate_priority_score(item)
            prioritized.append(item_copy)
        
        # Sort by priority score (descending)
        prioritized.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return prioritized


def load_context_items_agent(state: ContextPrioritizationState) -> ContextPrioritizationState:
    """Load context items to prioritize"""
    print("\n📋 Loading Context Items...")
    
    items = [
        {
            "id": "ctx_1",
            "name": "Production issue alert",
            "importance": 10,
            "urgency": 10,
            "relevance": 9
        },
        {
            "id": "ctx_2",
            "name": "User feedback",
            "importance": 7,
            "urgency": 4,
            "relevance": 8
        },
        {
            "id": "ctx_3",
            "name": "Code review request",
            "importance": 6,
            "urgency": 7,
            "relevance": 9
        },
        {
            "id": "ctx_4",
            "name": "Documentation update",
            "importance": 5,
            "urgency": 3,
            "relevance": 6
        },
        {
            "id": "ctx_5",
            "name": "Security vulnerability",
            "importance": 10,
            "urgency": 9,
            "relevance": 10
        },
        {
            "id": "ctx_6",
            "name": "Feature request",
            "importance": 6,
            "urgency": 4,
            "relevance": 7
        }
    ]
    
    print(f"\n  Loaded {len(items)} context items")
    for item in items:
        print(f"    • {item['name']}")
    
    return {
        **state,
        "context_items": items,
        "messages": [f"✓ Loaded {len(items)} context items"]
    }


def prioritize_items_agent(state: ContextPrioritizationState) -> ContextPrioritizationState:
    """Prioritize context items"""
    print("\n📊 Prioritizing Context Items...")
    
    prioritizer = ContextPrioritizer()
    prioritized = prioritizer.prioritize(state["context_items"])
    
    print(f"\n  Prioritization Results:")
    for i, item in enumerate(prioritized, 1):
        print(f"\n  {i}. {item['name']} (Score: {item['priority_score']})")
        print(f"     Importance: {item['importance']}, Urgency: {item['urgency']}, Relevance: {item['relevance']}")
    
    return {
        **state,
        "prioritized_items": prioritized,
        "messages": [f"✓ Prioritized {len(prioritized)} items"]
    }


def generate_context_prioritization_report_agent(state: ContextPrioritizationState) -> ContextPrioritizationState:
    """Generate context prioritization report"""
    print("\n" + "="*70)
    print("CONTEXT PRIORITIZATION REPORT")
    print("="*70)
    
    print(f"\n📊 Total Items: {len(state['prioritized_items'])}")
    
    print(f"\n🏆 Priority Ranking:")
    for i, item in enumerate(state["prioritized_items"], 1):
        tier = "CRITICAL" if item["priority_score"] >= 9 else \
               "HIGH" if item["priority_score"] >= 7 else \
               "MEDIUM" if item["priority_score"] >= 5 else "LOW"
        
        print(f"\n  {i}. {item['name']}")
        print(f"     Priority Tier: {tier}")
        print(f"     Score: {item['priority_score']}")
        print(f"     Importance: {item['importance']}/10")
        print(f"     Urgency: {item['urgency']}/10")
        print(f"     Relevance: {item['relevance']}/10")
    
    # Statistics
    avg_score = sum(item["priority_score"] for item in state["prioritized_items"]) / len(state["prioritized_items"])
    critical_count = sum(1 for item in state["prioritized_items"] if item["priority_score"] >= 9)
    
    print(f"\n📈 Statistics:")
    print(f"  Average Priority Score: {avg_score:.2f}")
    print(f"  Critical Items: {critical_count}")
    print(f"  High Priority Items: {sum(1 for item in state['prioritized_items'] if 7 <= item['priority_score'] < 9)}")
    
    print("\n💡 Context Prioritization Benefits:")
    print("  • Focus on what matters most")
    print("  • Efficient resource allocation")
    print("  • Clear decision-making")
    print("  • Better time management")
    print("  • Reduced cognitive load")
    
    print("\n="*70)
    print("✅ Context Prioritization Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_prioritization_graph():
    workflow = StateGraph(ContextPrioritizationState)
    workflow.add_node("load", load_context_items_agent)
    workflow.add_node("prioritize", prioritize_items_agent)
    workflow.add_node("report", generate_context_prioritization_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "prioritize")
    workflow.add_edge("prioritize", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 247: Context Prioritization MCP Pattern")
    print("="*70)
    
    app = create_context_prioritization_graph()
    final_state = app.invoke({
        "messages": [],
        "context_items": [],
        "prioritized_items": []
    })
    print("\n✅ Context Prioritization Pattern Complete!")


if __name__ == "__main__":
    main()

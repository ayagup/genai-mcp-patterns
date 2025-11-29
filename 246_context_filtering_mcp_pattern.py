"""
Pattern 246: Context Filtering MCP Pattern

This pattern demonstrates context filtering - selecting only relevant context
based on specific criteria to improve efficiency and focus.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextFilteringState(TypedDict):
    """State for context filtering workflow"""
    messages: Annotated[List[str], add]
    raw_context: Dict[str, Any]
    filters: List[Dict[str, Any]]
    filtered_context: Dict[str, Any]


class ContextFilter:
    """Filters context based on criteria"""
    
    def filter_by_relevance(self, context: Dict[str, Any], keywords: List[str]) -> Dict[str, Any]:
        """Filter by relevance to keywords"""
        filtered = {}
        for key, value in context.items():
            value_str = str(value).lower()
            if any(kw.lower() in value_str or kw.lower() in key.lower() for kw in keywords):
                filtered[key] = value
        return filtered
    
    def filter_by_priority(self, context: Dict[str, Any], min_priority: int) -> Dict[str, Any]:
        """Filter by priority level"""
        filtered = {}
        for key, value in context.items():
            if isinstance(value, dict) and value.get("priority", 0) >= min_priority:
                filtered[key] = value
            elif not isinstance(value, dict):
                filtered[key] = value
        return filtered
    
    def filter_by_recency(self, context: Dict[str, Any], days: int) -> Dict[str, Any]:
        """Filter by recency"""
        # Simplified - in real scenario would check timestamps
        filtered = {}
        for key, value in context.items():
            if isinstance(value, dict) and value.get("is_recent", True):
                filtered[key] = value
            elif not isinstance(value, dict):
                filtered[key] = value
        return filtered


def load_raw_context_agent(state: ContextFilteringState) -> ContextFilteringState:
    """Load raw context"""
    print("\n📥 Loading Raw Context...")
    
    raw_context = {
        "user_name": "John Doe",
        "user_role": "developer",
        "last_login": {"value": "2024-11-29", "priority": 5, "is_recent": True},
        "preferences": {"theme": "dark", "priority": 3, "is_recent": True},
        "old_settings": {"value": "deprecated", "priority": 1, "is_recent": False},
        "current_project": {"name": "AI Platform", "priority": 10, "is_recent": True},
        "archived_data": {"value": "old_data", "priority": 1, "is_recent": False},
        "recent_activity": {"actions": ["code", "test", "deploy"], "priority": 8, "is_recent": True}
    }
    
    print(f"\n  Total Context Items: {len(raw_context)}")
    for key in raw_context.keys():
        print(f"    • {key}")
    
    return {
        **state,
        "raw_context": raw_context,
        "messages": [f"✓ Loaded {len(raw_context)} context items"]
    }


def apply_filters_agent(state: ContextFilteringState) -> ContextFilteringState:
    """Apply filters to context"""
    print("\n🔍 Applying Filters...")
    
    context_filter = ContextFilter()
    filters_applied = []
    
    # Filter 1: By keywords (relevance)
    print("\n  Filter 1: By Relevance (keywords: project, recent, current)")
    filtered_1 = context_filter.filter_by_relevance(
        state["raw_context"],
        ["project", "recent", "current", "activity"]
    )
    print(f"    Kept {len(filtered_1)}/{len(state['raw_context'])} items")
    filters_applied.append({"name": "relevance", "kept": len(filtered_1)})
    
    # Filter 2: By priority
    print("\n  Filter 2: By Priority (minimum: 5)")
    filtered_2 = context_filter.filter_by_priority(state["raw_context"], min_priority=5)
    print(f"    Kept {len(filtered_2)}/{len(state['raw_context'])} items")
    filters_applied.append({"name": "priority", "kept": len(filtered_2)})
    
    # Filter 3: By recency
    print("\n  Filter 3: By Recency (recent only)")
    filtered_3 = context_filter.filter_by_recency(state["raw_context"], days=30)
    print(f"    Kept {len(filtered_3)}/{len(state['raw_context'])} items")
    filters_applied.append({"name": "recency", "kept": len(filtered_3)})
    
    # Combined filter (priority + recency)
    combined = context_filter.filter_by_priority(state["raw_context"], min_priority=5)
    combined = context_filter.filter_by_recency(combined, days=30)
    
    print(f"\n  Combined Filters Result: {len(combined)} items")
    
    return {
        **state,
        "filters": filters_applied,
        "filtered_context": combined,
        "messages": [f"✓ Applied {len(filters_applied)} filters"]
    }


def generate_context_filtering_report_agent(state: ContextFilteringState) -> ContextFilteringState:
    """Generate context filtering report"""
    print("\n" + "="*70)
    print("CONTEXT FILTERING REPORT")
    print("="*70)
    
    print(f"\n📊 Original Context: {len(state['raw_context'])} items")
    print(f"📊 Filtered Context: {len(state['filtered_context'])} items")
    print(f"📊 Reduction: {len(state['raw_context']) - len(state['filtered_context'])} items removed")
    
    print(f"\n🔍 Filters Applied:")
    for filter_info in state["filters"]:
        print(f"  • {filter_info['name']}: kept {filter_info['kept']} items")
    
    print(f"\n✅ Final Filtered Context:")
    for key, value in state["filtered_context"].items():
        print(f"  • {key}: {value}")
    
    print("\n💡 Context Filtering Benefits:")
    print("  • Reduced noise")
    print("  • Improved focus on relevant data")
    print("  • Better performance")
    print("  • Lower token usage")
    print("  • Faster processing")
    
    print("\n="*70)
    print("✅ Context Filtering Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_filtering_graph():
    workflow = StateGraph(ContextFilteringState)
    workflow.add_node("load", load_raw_context_agent)
    workflow.add_node("filter", apply_filters_agent)
    workflow.add_node("report", generate_context_filtering_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "filter")
    workflow.add_edge("filter", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 246: Context Filtering MCP Pattern")
    print("="*70)
    
    app = create_context_filtering_graph()
    final_state = app.invoke({
        "messages": [],
        "raw_context": {},
        "filters": [],
        "filtered_context": {}
    })
    print("\n✅ Context Filtering Pattern Complete!")


if __name__ == "__main__":
    main()

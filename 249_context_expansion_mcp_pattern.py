"""
Pattern 249: Context Expansion MCP Pattern

This pattern demonstrates context expansion - enriching minimal context with
additional relevant information from various sources.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextExpansionState(TypedDict):
    """State for context expansion workflow"""
    messages: Annotated[List[str], add]
    minimal_context: str
    expanded_context: Dict[str, Any]
    expansion_sources: List[str]


class ContextExpander:
    """Expands context with additional information"""
    
    def __init__(self):
        self.knowledge_base = {
            "user": {
                "name": "John Doe",
                "role": "Senior Developer",
                "team": "Platform Engineering",
                "location": "San Francisco"
            },
            "project": {
                "name": "AI Platform",
                "status": "active",
                "deadline": "2024-12-31",
                "priority": "high"
            },
            "system": {
                "environment": "production",
                "version": "2.5.0",
                "last_deployment": "2024-11-15",
                "health": "healthy"
            },
            "historical": {
                "similar_issues": ["AUTH-101", "AUTH-205"],
                "resolution_time_avg": "2.5 hours",
                "success_rate": "94%"
            }
        }
    
    def expand_user_context(self, query: str) -> Dict[str, Any]:
        """Expand with user information"""
        if "user" in query.lower() or "i" in query.lower() or "my" in query.lower():
            return self.knowledge_base["user"]
        return {}
    
    def expand_project_context(self, query: str) -> Dict[str, Any]:
        """Expand with project information"""
        if "project" in query.lower() or "work" in query.lower():
            return self.knowledge_base["project"]
        return {}
    
    def expand_system_context(self, query: str) -> Dict[str, Any]:
        """Expand with system information"""
        if "system" in query.lower() or "deploy" in query.lower() or "environment" in query.lower():
            return self.knowledge_base["system"]
        return {}
    
    def expand_historical_context(self, query: str) -> Dict[str, Any]:
        """Expand with historical information"""
        if "issue" in query.lower() or "problem" in query.lower() or "error" in query.lower():
            return self.knowledge_base["historical"]
        return {}


def load_minimal_context_agent(state: ContextExpansionState) -> ContextExpansionState:
    """Load minimal context"""
    print("\n📥 Loading Minimal Context...")
    
    minimal = "Fix authentication issue in production system"
    
    print(f"\n  Minimal Context: '{minimal}'")
    print(f"  Length: {len(minimal)} characters")
    
    return {
        **state,
        "minimal_context": minimal,
        "messages": [f"✓ Loaded minimal context ({len(minimal)} chars)"]
    }


def expand_context_agent(state: ContextExpansionState) -> ContextExpansionState:
    """Expand context with additional information"""
    print("\n📈 Expanding Context...")
    
    expander = ContextExpander()
    query = state["minimal_context"]
    expanded = {"original_query": query}
    sources = []
    
    # Expand from different sources
    user_ctx = expander.expand_user_context(query)
    if user_ctx:
        expanded["user_context"] = user_ctx
        sources.append("user_profile")
        print(f"\n  ✓ Added user context: {len(user_ctx)} fields")
    
    project_ctx = expander.expand_project_context(query)
    if project_ctx:
        expanded["project_context"] = project_ctx
        sources.append("project_info")
        print(f"  ✓ Added project context: {len(project_ctx)} fields")
    
    system_ctx = expander.expand_system_context(query)
    if system_ctx:
        expanded["system_context"] = system_ctx
        sources.append("system_status")
        print(f"  ✓ Added system context: {len(system_ctx)} fields")
    
    historical_ctx = expander.expand_historical_context(query)
    if historical_ctx:
        expanded["historical_context"] = historical_ctx
        sources.append("historical_data")
        print(f"  ✓ Added historical context: {len(historical_ctx)} fields")
    
    print(f"\n  Total Sources: {len(sources)}")
    print(f"  Total Fields: {sum(len(v) if isinstance(v, dict) else 0 for v in expanded.values()) + 1}")
    
    return {
        **state,
        "expanded_context": expanded,
        "expansion_sources": sources,
        "messages": [f"✓ Expanded from {len(sources)} sources"]
    }


def generate_context_expansion_report_agent(state: ContextExpansionState) -> ContextExpansionState:
    """Generate context expansion report"""
    print("\n" + "="*70)
    print("CONTEXT EXPANSION REPORT")
    print("="*70)
    
    print(f"\n📊 Expansion Statistics:")
    print(f"  Original: '{state['minimal_context']}'")
    print(f"  Sources Added: {len(state['expansion_sources'])}")
    
    total_fields = sum(
        len(v) if isinstance(v, dict) else 0 
        for v in state['expanded_context'].values()
    )
    print(f"  Total Fields Added: {total_fields}")
    
    print(f"\n📈 Expanded Context:")
    for key, value in state["expanded_context"].items():
        if isinstance(value, dict):
            print(f"\n  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n📚 Expansion Sources:")
    for source in state["expansion_sources"]:
        print(f"  • {source}")
    
    print("\n💡 Context Expansion Benefits:")
    print("  • Richer understanding")
    print("  • Better decision-making")
    print("  • Comprehensive view")
    print("  • Reduced ambiguity")
    print("  • Personalized responses")
    print("  • Historical insights")
    
    print("\n="*70)
    print("✅ Context Expansion Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_expansion_graph():
    workflow = StateGraph(ContextExpansionState)
    workflow.add_node("load", load_minimal_context_agent)
    workflow.add_node("expand", expand_context_agent)
    workflow.add_node("report", generate_context_expansion_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "expand")
    workflow.add_edge("expand", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 249: Context Expansion MCP Pattern")
    print("="*70)
    
    app = create_context_expansion_graph()
    final_state = app.invoke({
        "messages": [],
        "minimal_context": "",
        "expanded_context": {},
        "expansion_sources": []
    })
    print("\n✅ Context Expansion Pattern Complete!")


if __name__ == "__main__":
    main()

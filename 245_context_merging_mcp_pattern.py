"""
Pattern 245: Context Merging MCP Pattern

This pattern demonstrates context merging - combining multiple contexts
into a unified view while resolving conflicts.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextMergingState(TypedDict):
    """State for context merging workflow"""
    messages: Annotated[List[str], add]
    source_contexts: List[Dict[str, Any]]
    merged_context: Dict[str, Any]
    conflicts: List[Dict[str, Any]]


class ContextMerger:
    """Merges multiple contexts"""
    
    def __init__(self):
        self.merge_strategies = {
            "override": self._override_merge,
            "combine": self._combine_merge,
            "latest": self._latest_merge
        }
    
    def _override_merge(self, values: List[Any]) -> Any:
        """Last value wins"""
        return values[-1] if values else None
    
    def _combine_merge(self, values: List[Any]) -> Any:
        """Combine all values"""
        if all(isinstance(v, list) for v in values):
            combined = []
            for v in values:
                combined.extend(v)
            return combined
        return values[-1]
    
    def _latest_merge(self, values: List[Any]) -> Any:
        """Use latest timestamped value"""
        return values[-1] if values else None
    
    def merge(self, contexts: List[Dict[str, Any]], strategy: str = "override") -> tuple:
        """Merge contexts using specified strategy"""
        merged = {}
        conflicts = []
        
        all_keys = set()
        for ctx in contexts:
            all_keys.update(ctx.keys())
        
        for key in all_keys:
            values = [ctx[key] for ctx in contexts if key in ctx]
            
            if len(values) > 1 and len(set(str(v) for v in values)) > 1:
                # Conflict detected
                conflicts.append({
                    "key": key,
                    "values": values,
                    "resolution": "using strategy: " + strategy
                })
            
            merge_func = self.merge_strategies.get(strategy, self._override_merge)
            merged[key] = merge_func(values)
        
        return merged, conflicts


def load_source_contexts_agent(state: ContextMergingState) -> ContextMergingState:
    """Load source contexts to merge"""
    print("\n📥 Loading Source Contexts...")
    
    contexts = [
        {
            "source": "user_preferences",
            "theme": "dark",
            "language": "en",
            "notifications": True,
            "timezone": "UTC"
        },
        {
            "source": "system_defaults",
            "theme": "light",
            "language": "en",
            "font_size": 14,
            "timezone": "UTC-5"
        },
        {
            "source": "session_overrides",
            "theme": "auto",
            "notifications": False,
            "font_size": 16
        }
    ]
    
    print(f"\n  Loaded {len(contexts)} source contexts:")
    for ctx in contexts:
        print(f"    • {ctx.pop('source')}: {len(ctx)} properties")
    
    return {
        **state,
        "source_contexts": contexts,
        "messages": [f"✓ Loaded {len(contexts)} source contexts"]
    }


def merge_contexts_agent(state: ContextMergingState) -> ContextMergingState:
    """Merge source contexts"""
    print("\n🔀 Merging Contexts...")
    
    merger = ContextMerger()
    merged, conflicts = merger.merge(state["source_contexts"], strategy="override")
    
    print(f"\n  Merged Properties: {len(merged)}")
    for key, value in merged.items():
        print(f"    {key}: {value}")
    
    if conflicts:
        print(f"\n  Conflicts Resolved: {len(conflicts)}")
        for conflict in conflicts:
            print(f"    {conflict['key']}: {conflict['values']} → {merged[conflict['key']]}")
    
    return {
        **state,
        "merged_context": merged,
        "conflicts": conflicts,
        "messages": [f"✓ Merged {len(state['source_contexts'])} contexts"]
    }


def generate_context_merging_report_agent(state: ContextMergingState) -> ContextMergingState:
    """Generate context merging report"""
    print("\n" + "="*70)
    print("CONTEXT MERGING REPORT")
    print("="*70)
    
    print(f"\n📊 Source Contexts: {len(state['source_contexts'])}")
    for i, ctx in enumerate(state['source_contexts'], 1):
        print(f"\n  Context {i}:")
        for key, value in ctx.items():
            print(f"    {key}: {value}")
    
    print(f"\n🔀 Merged Context:")
    for key, value in state["merged_context"].items():
        print(f"  {key}: {value}")
    
    if state["conflicts"]:
        print(f"\n⚠️ Conflicts Resolved ({len(state['conflicts'])}):")
        for conflict in state["conflicts"]:
            print(f"\n  • {conflict['key']}")
            print(f"    Conflicting values: {conflict['values']}")
            print(f"    Resolution: {conflict['resolution']}")
            print(f"    Final value: {state['merged_context'][conflict['key']]}")
    
    print("\n💡 Context Merging Benefits:")
    print("  • Unified view from multiple sources")
    print("  • Automatic conflict resolution")
    print("  • Flexible merge strategies")
    print("  • Preserve important context")
    print("  • Reduce complexity")
    
    print("\n="*70)
    print("✅ Context Merging Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_merging_graph():
    workflow = StateGraph(ContextMergingState)
    workflow.add_node("load", load_source_contexts_agent)
    workflow.add_node("merge", merge_contexts_agent)
    workflow.add_node("report", generate_context_merging_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "merge")
    workflow.add_edge("merge", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 245: Context Merging MCP Pattern")
    print("="*70)
    
    app = create_context_merging_graph()
    final_state = app.invoke({
        "messages": [],
        "source_contexts": [],
        "merged_context": {},
        "conflicts": []
    })
    print("\n✅ Context Merging Pattern Complete!")


if __name__ == "__main__":
    main()

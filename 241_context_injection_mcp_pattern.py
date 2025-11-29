"""
Pattern 241: Context Injection MCP Pattern

This pattern demonstrates context injection - dynamically providing relevant
context to agents at runtime to enhance their decision-making capabilities.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextInjectionState(TypedDict):
    """State for context injection workflow"""
    messages: Annotated[List[str], add]
    base_query: str
    injected_contexts: List[Dict[str, Any]]
    enriched_query: str


class ContextInjector:
    """Manages context injection"""
    
    def __init__(self):
        self.context_sources = {
            "user_profile": {"name": "John Doe", "role": "developer", "experience": "5 years"},
            "system_state": {"cpu_usage": "45%", "memory": "8GB/16GB", "status": "healthy"},
            "environment": {"timezone": "UTC-5", "locale": "en_US", "platform": "linux"},
            "business_rules": {"max_retry": 3, "timeout": 30, "priority": "high"}
        }
    
    def inject_context(self, query: str, context_types: List[str]) -> Dict[str, Any]:
        """Inject specified contexts into query"""
        injected = {}
        for context_type in context_types:
            if context_type in self.context_sources:
                injected[context_type] = self.context_sources[context_type]
        return injected


def analyze_query_agent(state: ContextInjectionState) -> ContextInjectionState:
    """Analyze query to determine needed context"""
    print("\n🔍 Analyzing Query for Context Needs...")
    
    query = "Deploy new service with user permissions"
    print(f"\n  Base Query: {query}")
    
    # Determine which contexts are needed
    needed_contexts = []
    if "user" in query.lower():
        needed_contexts.append("user_profile")
    if "deploy" in query.lower():
        needed_contexts.append("system_state")
        needed_contexts.append("environment")
        needed_contexts.append("business_rules")
    
    print(f"  Identified Needed Contexts: {', '.join(needed_contexts)}")
    
    return {
        **state,
        "base_query": query,
        "messages": [f"✓ Analyzed query, need {len(needed_contexts)} contexts"]
    }


def inject_context_agent(state: ContextInjectionState) -> ContextInjectionState:
    """Inject relevant context"""
    print("\n💉 Injecting Context...")
    
    injector = ContextInjector()
    context_types = ["user_profile", "system_state", "environment", "business_rules"]
    
    injected_contexts = []
    for context_type in context_types:
        context_data = injector.inject_context(state["base_query"], [context_type])
        if context_data:
            injected_contexts.append({
                "type": context_type,
                "data": context_data[context_type]
            })
            print(f"\n  ✓ Injected {context_type}:")
            for key, value in context_data[context_type].items():
                print(f"    {key}: {value}")
    
    return {
        **state,
        "injected_contexts": injected_contexts,
        "messages": [f"✓ Injected {len(injected_contexts)} contexts"]
    }


def enrich_query_agent(state: ContextInjectionState) -> ContextInjectionState:
    """Enrich query with injected context"""
    print("\n🎯 Enriching Query with Context...")
    
    enriched_parts = [f"Base Query: {state['base_query']}"]
    
    for context in state["injected_contexts"]:
        enriched_parts.append(f"\nContext [{context['type']}]:")
        for key, value in context["data"].items():
            enriched_parts.append(f"  - {key}: {value}")
    
    enriched_query = "\n".join(enriched_parts)
    
    print(f"\n  Enriched Query Length: {len(enriched_query)} characters")
    print(f"  Context Sources: {len(state['injected_contexts'])}")
    
    return {
        **state,
        "enriched_query": enriched_query,
        "messages": ["✓ Query enriched with context"]
    }


def generate_context_injection_report_agent(state: ContextInjectionState) -> ContextInjectionState:
    """Generate context injection report"""
    print("\n" + "="*70)
    print("CONTEXT INJECTION REPORT")
    print("="*70)
    
    print(f"\n📝 Base Query:")
    print(f"  {state['base_query']}")
    
    print(f"\n💉 Injected Contexts ({len(state['injected_contexts'])}):")
    for context in state["injected_contexts"]:
        print(f"\n  • {context['type']}:")
        for key, value in context["data"].items():
            print(f"    {key}: {value}")
    
    print(f"\n🎯 Enriched Query Preview:")
    lines = state["enriched_query"].split("\n")
    for line in lines[:10]:  # Show first 10 lines
        print(f"  {line}")
    if len(lines) > 10:
        print(f"  ... ({len(lines) - 10} more lines)")
    
    print("\n💡 Context Injection Benefits:")
    print("  • Enhanced agent understanding")
    print("  • Personalized responses")
    print("  • Context-aware decisions")
    print("  • Reduced ambiguity")
    print("  • Better accuracy")
    
    print("\n="*70)
    print("✅ Context Injection Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_injection_graph():
    workflow = StateGraph(ContextInjectionState)
    workflow.add_node("analyze", analyze_query_agent)
    workflow.add_node("inject", inject_context_agent)
    workflow.add_node("enrich", enrich_query_agent)
    workflow.add_node("report", generate_context_injection_report_agent)
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "inject")
    workflow.add_edge("inject", "enrich")
    workflow.add_edge("enrich", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 241: Context Injection MCP Pattern")
    print("="*70)
    
    app = create_context_injection_graph()
    final_state = app.invoke({
        "messages": [],
        "base_query": "",
        "injected_contexts": [],
        "enriched_query": ""
    })
    print("\n✅ Context Injection Pattern Complete!")


if __name__ == "__main__":
    main()

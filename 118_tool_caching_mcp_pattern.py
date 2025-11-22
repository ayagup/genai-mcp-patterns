"""
Tool Caching MCP Pattern

This pattern implements intelligent caching of tool results
to improve performance and reduce redundant operations.

Key Features:
- Result caching
- Cache invalidation
- TTL management
- Cache warming
- Distributed caching
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ToolCachingState(TypedDict):
    """State for tool caching pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    cache: Dict[str, Dict]
    cache_key: str
    cache_hit: bool
    cached_result: str


llm = ChatOpenAI(model="gpt-4", temperature=0)


def cache_manager(state: ToolCachingState) -> ToolCachingState:
    """Manages tool result caching"""
    cache_key = state.get("cache_key", "search:AI trends")
    
    system_message = SystemMessage(content="You are a cache manager.")
    user_message = HumanMessage(content=f"Check cache for: {cache_key}")
    response = llm.invoke([system_message, user_message])
    
    # Simulate cache with TTL
    cache = {
        "search:AI trends": {
            "result": "AI trends data...",
            "timestamp": 1699876543,
            "ttl": 3600,
            "hits": 5
        },
        "calc:2+2": {
            "result": "4",
            "timestamp": 1699876000,
            "ttl": 86400,
            "hits": 12
        }
    }
    
    cache_hit = cache_key in cache
    cached_result = cache.get(cache_key, {}).get("result", "")
    
    report = f"""
    💾 Tool Caching:
    
    Cache Key: {cache_key}
    Cache Hit: {cache_hit}
    Cache Size: {len(cache)} entries
    
    Caching Strategies:
    • LRU (Least Recently Used)
    • LFU (Least Frequently Used)
    • TTL (Time To Live)
    • Write-through
    • Write-back
    
    Cache Statistics:
    {chr(10).join(f"  • {key}: {val['hits']} hits, TTL: {val['ttl']}s" for key, val in cache.items())}
    
    Cache Benefits:
    • Reduced latency
    • Lower API costs
    • Improved throughput
    • Resilience to failures
    """
    
    return {
        "messages": [AIMessage(content=f"{response.content}\n{report}")],
        "cache": cache,
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "cached_result": cached_result
    }


def build_tool_caching_graph():
    workflow = StateGraph(ToolCachingState)
    workflow.add_node("manager", cache_manager)
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_tool_caching_graph()
    print("=== Tool Caching MCP Pattern ===\n")
    
    result = graph.invoke({
        "messages": [],
        "cache": {},
        "cache_key": "",
        "cache_hit": False,
        "cached_result": ""
    })
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")

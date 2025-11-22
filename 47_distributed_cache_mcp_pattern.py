"""
Distributed Cache MCP Pattern

This pattern demonstrates distributed caching for knowledge sharing across
multiple agents with cache synchronization and invalidation.

Key Features:
- Multi-level caching
- Cache synchronization
- Cache invalidation strategies
- Distributed cache access
"""

from typing import TypedDict, Sequence, Annotated
import operator
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class DistributedCacheState(TypedDict):
    """State with distributed cache"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    cache_nodes: dict[str, dict]  # node_id -> cache data
    query: str
    cache_hit: bool
    cached_result: str
    final_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Cache Manager
def cache_manager(state: DistributedCacheState) -> DistributedCacheState:
    """Manages distributed cache initialization"""
    
    system_message = SystemMessage(content="""You are a cache manager. Initialize and 
    manage distributed cache nodes for knowledge sharing.""")
    
    user_message = HumanMessage(content="""Initialize distributed cache with 3 nodes:
- Node 1: Primary cache
- Node 2: Replica cache
- Node 3: Local cache

Set up cache infrastructure.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize cache nodes
    cache_nodes = {
        "node_1": {
            "type": "primary",
            "location": "us-east",
            "data": {},
            "hits": 0,
            "misses": 0
        },
        "node_2": {
            "type": "replica",
            "location": "eu-west",
            "data": {},
            "hits": 0,
            "misses": 0
        },
        "node_3": {
            "type": "local",
            "location": "ap-south",
            "data": {},
            "hits": 0,
            "misses": 0
        }
    }
    
    return {
        "messages": [AIMessage(content=f"🗄️ Cache Manager: {response.content}\n\n✅ Initialized {len(cache_nodes)} cache nodes")],
        "cache_nodes": cache_nodes
    }


# Cache Lookup
def cache_lookup(state: DistributedCacheState) -> DistributedCacheState:
    """Looks up query in distributed cache"""
    query = state.get("query", "")
    cache_nodes = state.get("cache_nodes", {})
    
    system_message = SystemMessage(content="""You are a cache lookup agent. Search 
    distributed cache nodes for cached query results.""")
    
    user_message = HumanMessage(content=f"""Lookup query in cache: {query}

Search all cache nodes for matching result.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate cache key
    cache_key = hashlib.md5(query.encode()).hexdigest()[:8]
    
    # Check cache nodes
    cache_hit = False
    cached_result = ""
    
    for node_id, node in cache_nodes.items():
        if cache_key in node.get("data", {}):
            cache_hit = True
            cached_result = node["data"][cache_key]
            node["hits"] += 1
            break
    
    if not cache_hit:
        for node in cache_nodes.values():
            node["misses"] += 1
    
    return {
        "messages": [AIMessage(content=f"🔍 Cache Lookup: {response.content}\n\n{'✅ Cache HIT' if cache_hit else '❌ Cache MISS'}")],
        "cache_nodes": cache_nodes,
        "cache_hit": cache_hit,
        "cached_result": cached_result
    }


# Query Processor (on cache miss)
def query_processor(state: DistributedCacheState) -> DistributedCacheState:
    """Processes query when cache miss occurs"""
    query = state.get("query", "")
    cache_hit = state.get("cache_hit", False)
    
    if cache_hit:
        return {"messages": [AIMessage(content="⏭️ Query Processor: Skipped (cache hit)")]}
    
    system_message = SystemMessage(content="""You are a query processor. Process queries 
    that are not found in cache.""")
    
    user_message = HumanMessage(content=f"""Process query: {query}

Generate fresh result.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"⚙️ Query Processor: {response.content}")],
        "final_result": response.content
    }


# Cache Writer
def cache_writer(state: DistributedCacheState) -> DistributedCacheState:
    """Writes results to distributed cache"""
    query = state.get("query", "")
    cache_hit = state.get("cache_hit", False)
    cache_nodes = state.get("cache_nodes", {})
    final_result = state.get("final_result", "")
    
    if cache_hit:
        return {"messages": [AIMessage(content="⏭️ Cache Writer: Skipped (cache hit)")]}
    
    system_message = SystemMessage(content="""You are a cache writer. Write results to 
    distributed cache and synchronize across nodes.""")
    
    user_message = HumanMessage(content=f"""Write result to cache:

Query: {query}
Result: {final_result}

Synchronize across all cache nodes.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate cache key
    cache_key = hashlib.md5(query.encode()).hexdigest()[:8]
    
    # Write to all cache nodes (synchronization)
    for node_id, node in cache_nodes.items():
        node["data"][cache_key] = final_result
    
    return {
        "messages": [AIMessage(content=f"💾 Cache Writer: {response.content}\n\n✅ Result cached and synchronized across {len(cache_nodes)} nodes")],
        "cache_nodes": cache_nodes
    }


# Result Aggregator
def result_aggregator(state: DistributedCacheState) -> DistributedCacheState:
    """Aggregates final result from cache or fresh processing"""
    cache_hit = state.get("cache_hit", False)
    cached_result = state.get("cached_result", "")
    final_result = state.get("final_result", "")
    
    result = cached_result if cache_hit else final_result
    
    system_message = SystemMessage(content="""You are a result aggregator. Return the final 
    result from either cache or fresh processing.""")
    
    source = "distributed cache" if cache_hit else "fresh processing"
    user_message = HumanMessage(content=f"""Aggregate result from {source}:

{result}""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📊 Result Aggregator: {response.content}")],
        "final_result": result
    }


# Cache Monitor
def cache_monitor(state: DistributedCacheState) -> DistributedCacheState:
    """Monitors cache performance"""
    cache_nodes = state.get("cache_nodes", {})
    cache_hit = state.get("cache_hit", False)
    
    total_hits = sum(node.get("hits", 0) for node in cache_nodes.values())
    total_misses = sum(node.get("misses", 0) for node in cache_nodes.values())
    hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
    
    node_stats = "\n".join([
        f"  {node_id} ({node['type']}):\n    Location: {node['location']}\n    Cached items: {len(node.get('data', {}))}\n    Hits: {node.get('hits', 0)}, Misses: {node.get('misses', 0)}"
        for node_id, node in cache_nodes.items()
    ])
    
    summary = f"""
    ✅ DISTRIBUTED CACHE PATTERN COMPLETE
    
    Cache Performance:
    • Total Hits: {total_hits}
    • Total Misses: {total_misses}
    • Hit Rate: {hit_rate:.1f}%
    • This Query: {'HIT ✅' if cache_hit else 'MISS ❌'}
    
    Cache Nodes:
{node_stats}
    
    Distributed Cache Benefits:
    • Reduced latency through caching
    • Geographic distribution
    • High availability with replication
    • Automatic synchronization
    • Improved throughput
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Cache Monitor:\n{summary}")]
    }


# Build the graph
def build_distributed_cache_graph():
    """Build the distributed cache MCP pattern graph"""
    workflow = StateGraph(DistributedCacheState)
    
    workflow.add_node("manager", cache_manager)
    workflow.add_node("lookup", cache_lookup)
    workflow.add_node("processor", query_processor)
    workflow.add_node("writer", cache_writer)
    workflow.add_node("aggregator", result_aggregator)
    workflow.add_node("monitor", cache_monitor)
    
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", "lookup")
    workflow.add_edge("lookup", "processor")
    workflow.add_edge("processor", "writer")
    workflow.add_edge("writer", "aggregator")
    workflow.add_edge("aggregator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_distributed_cache_graph()
    
    print("=== Distributed Cache MCP Pattern ===\n")
    
    # First query - cache miss
    print("\n" + "="*70)
    print("QUERY 1 (Cache Miss Expected)")
    print("="*70)
    
    state1 = {
        "messages": [],
        "cache_nodes": {},
        "query": "What are the benefits of using LangChain for AI applications?",
        "cache_hit": False,
        "cached_result": "",
        "final_result": ""
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Second query - same query, cache hit
    print("\n\n" + "="*70)
    print("QUERY 2 (Cache Hit Expected - Same Query)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "cache_nodes": result1["cache_nodes"],  # Reuse cache
        "query": "What are the benefits of using LangChain for AI applications?",
        "cache_hit": False,
        "cached_result": "",
        "final_result": ""
    }
    
    result2 = graph.invoke(state2)
    
    for msg in result2["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)

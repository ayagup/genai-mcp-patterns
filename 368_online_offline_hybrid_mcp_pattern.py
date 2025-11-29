"""
Online-Offline Hybrid MCP Pattern

This pattern combines online (real-time, connected) and offline (cached, local)
processing for resilience and performance.

Pattern Type: Hybrid
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator
import json


# State definition
class OnlineOfflineState(TypedDict):
    """State for online-offline hybrid"""
    query: str
    online_available: bool
    online_result: Dict[str, Any]
    offline_cache: List[Dict[str, Any]]
    offline_result: Dict[str, Any]
    hybrid_response: Dict[str, Any]
    sync_status: Dict[str, Any]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class OnlineProcessor:
    """Online processing with live connection"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def process_online(self, query: str) -> Dict[str, Any]:
        """Process using online resources"""
        messages = [HumanMessage(content=f"Online query: {query}")]
        response = self.llm.invoke(messages)
        
        return {
            "mode": "online",
            "result": response.content[:200],
            "freshness": "real-time",
            "latency": "network-dependent"
        }


class OfflineProcessor:
    """Offline processing with local cache"""
    
    def __init__(self):
        self.cache = self._load_cache()
    
    def _load_cache(self) -> List[Dict[str, Any]]:
        """Load offline cache"""
        return [
            {"query": "weather", "result": "Cached: 72°F sunny"},
            {"query": "time", "result": "Cached: 12:00 PM"},
            {"query": "news", "result": "Cached: Headlines from today"}
        ]
    
    def process_offline(self, query: str) -> Dict[str, Any]:
        """Process using offline cache"""
        # Simple cache lookup
        for item in self.cache:
            if query.lower() in item["query"].lower():
                return {
                    "mode": "offline",
                    "result": item["result"],
                    "freshness": "cached",
                    "latency": "instant"
                }
        
        return {
            "mode": "offline",
            "result": "No cached data available",
            "freshness": "none",
            "latency": "instant"
        }


class HybridDecider:
    """Decide between online and offline"""
    
    def decide(self, online_available: bool, online_result: Dict, 
               offline_result: Dict) -> Dict[str, Any]:
        """Decide which result to use"""
        if online_available and online_result.get("result"):
            return {
                "source": "online",
                "result": online_result["result"],
                "freshness": "real-time",
                "fallback_available": bool(offline_result.get("result"))
            }
        elif offline_result.get("result") != "No cached data available":
            return {
                "source": "offline",
                "result": offline_result["result"],
                "freshness": "cached",
                "online_unavailable": not online_available
            }
        else:
            return {
                "source": "none",
                "result": "No data available",
                "freshness": "none",
                "error": "Both online and offline failed"
            }


# Agent functions
def initialize(state: OnlineOfflineState) -> OnlineOfflineState:
    """Initialize"""
    state["messages"].append(HumanMessage(content="Initializing online-offline hybrid"))
    state["offline_cache"] = []
    state["current_step"] = "initialized"
    return state


def try_online(state: OnlineOfflineState) -> OnlineOfflineState:
    """Try online processing"""
    if state["online_available"]:
        llm = ChatOpenAI(model="gpt-4", temperature=0.3)
        processor = OnlineProcessor(llm)
        state["online_result"] = processor.process_online(state["query"])
        state["messages"].append(HumanMessage(content="Online processing successful"))
    else:
        state["online_result"] = {"mode": "online", "result": None, "error": "Offline"}
        state["messages"].append(HumanMessage(content="Online unavailable, using offline"))
    
    state["current_step"] = "online_attempted"
    return state


def try_offline(state: OnlineOfflineState) -> OnlineOfflineState:
    """Try offline processing"""
    processor = OfflineProcessor()
    state["offline_cache"] = processor.cache
    state["offline_result"] = processor.process_offline(state["query"])
    state["messages"].append(HumanMessage(content="Offline cache checked"))
    state["current_step"] = "offline_attempted"
    return state


def decide_hybrid(state: OnlineOfflineState) -> OnlineOfflineState:
    """Decide hybrid approach"""
    decider = HybridDecider()
    state["hybrid_response"] = decider.decide(
        state["online_available"],
        state["online_result"],
        state["offline_result"]
    )
    state["messages"].append(HumanMessage(
        content=f"Using {state['hybrid_response']['source']} source"
    ))
    state["current_step"] = "decided"
    return state


def generate_report(state: OnlineOfflineState) -> OnlineOfflineState:
    """Generate report"""
    report = f"""
ONLINE-OFFLINE HYBRID REPORT
=============================

Query: {state['query']}
Online Available: {state['online_available']}

ONLINE RESULT:
{json.dumps(state['online_result'], indent=2)}

OFFLINE RESULT:
{json.dumps(state['offline_result'], indent=2)}

HYBRID RESPONSE:
{json.dumps(state['hybrid_response'], indent=2)}

Summary:
Hybrid approach ensures availability with online freshness and offline resilience.
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build graph
def create_online_offline_graph():
    """Create online-offline workflow"""
    workflow = StateGraph(OnlineOfflineState)
    
    workflow.add_node("initialize", initialize)
    workflow.add_node("online", try_online)
    workflow.add_node("offline", try_offline)
    workflow.add_node("decide", decide_hybrid)
    workflow.add_node("report", generate_report)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "online")
    workflow.add_edge("online", "offline")
    workflow.add_edge("offline", "decide")
    workflow.add_edge("decide", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "query": "What's the weather?",
        "online_available": True,  # Set to False to test offline mode
        "online_result": {},
        "offline_cache": [],
        "offline_result": {},
        "hybrid_response": {},
        "sync_status": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_online_offline_graph()
    
    print("Online-Offline Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")

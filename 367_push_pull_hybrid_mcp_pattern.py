"""
Push-Pull Hybrid MCP Pattern

This pattern combines push (proactive notification) and pull (on-demand request)
communication models for flexible data flow.

Pattern Type: Hybrid
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class PushPullState(TypedDict):
    """State for push-pull hybrid"""
    data_streams: List[str]
    push_subscriptions: List[Dict[str, Any]]
    pull_requests: List[Dict[str, Any]]
    push_data: List[Dict[str, Any]]
    pull_data: List[Dict[str, Any]]
    hybrid_strategy: Dict[str, Any]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class PushProvider:
    """Push-based data provider (proactive)"""
    
    def subscribe(self, streams: List[str]) -> List[Dict[str, Any]]:
        """Subscribe to push notifications"""
        subscriptions = []
        
        for stream in streams:
            subscriptions.append({
                "stream": stream,
                "mode": "push",
                "frequency": "real-time",
                "active": True
            })
        
        return subscriptions
    
    def push_updates(self, subscriptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Push updates to subscribers"""
        updates = []
        
        for sub in subscriptions:
            if sub.get("active"):
                updates.append({
                    "stream": sub["stream"],
                    "data": f"Push update from {sub['stream']}",
                    "timestamp": "2024-11-29T12:00:00Z",
                    "mode": "push"
                })
        
        return updates


class PullConsumer:
    """Pull-based data consumer (on-demand)"""
    
    def request_data(self, streams: List[str]) -> List[Dict[str, Any]]:
        """Request data on-demand"""
        requests = []
        
        for stream in streams:
            requests.append({
                "stream": stream,
                "mode": "pull",
                "timing": "on-demand",
                "cached": False
            })
        
        return requests
    
    def fetch_data(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fetch data based on requests"""
        data = []
        
        for req in requests:
            data.append({
                "stream": req["stream"],
                "data": f"Pull data from {req['stream']}",
                "timestamp": "2024-11-29T12:00:01Z",
                "mode": "pull"
            })
        
        return data


class HybridCoordinator:
    """Coordinate push and pull strategies"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def determine_strategy(self, streams: List[str]) -> Dict[str, Any]:
        """Determine push vs pull for each stream"""
        prompt = f"""Determine optimal push/pull strategy for data streams:

Streams:
{json.dumps(streams, indent=2)}

Classify each stream:
- Push: High-frequency updates, time-critical, subscribers need real-time data
- Pull: Low-frequency updates, on-demand needs, resource-intensive

Return JSON:
{{
    "push_streams": ["streams for push model"],
    "pull_streams": ["streams for pull model"],
    "rationale": "reasoning for classification"
}}"""
        
        messages = [
            SystemMessage(content="You are a data flow optimization expert."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "push_streams": streams[:len(streams)//2],
            "pull_streams": streams[len(streams)//2:],
            "rationale": "Default split"
        }


# Agent functions
def initialize_system(state: PushPullState) -> PushPullState:
    """Initialize push-pull system"""
    state["messages"].append(HumanMessage(
        content=f"Initializing push-pull hybrid for {len(state['data_streams'])} streams"
    ))
    state["current_step"] = "initialized"
    return state


def setup_push(state: PushPullState) -> PushPullState:
    """Setup push subscriptions"""
    provider = PushProvider()
    
    # Subscribe to push streams (first half for demo)
    push_streams = state["data_streams"][:len(state["data_streams"])//2]
    state["push_subscriptions"] = provider.subscribe(push_streams)
    
    state["messages"].append(HumanMessage(
        content=f"Subscribed to {len(state['push_subscriptions'])} push streams"
    ))
    state["current_step"] = "push_setup"
    return state


def setup_pull(state: PushPullState) -> PushPullState:
    """Setup pull requests"""
    consumer = PullConsumer()
    
    # Request pull streams (second half for demo)
    pull_streams = state["data_streams"][len(state["data_streams"])//2:]
    state["pull_requests"] = consumer.request_data(pull_streams)
    
    state["messages"].append(HumanMessage(
        content=f"Created {len(state['pull_requests'])} pull requests"
    ))
    state["current_step"] = "pull_setup"
    return state


def receive_push_data(state: PushPullState) -> PushPullState:
    """Receive push updates"""
    provider = PushProvider()
    
    state["push_data"] = provider.push_updates(state["push_subscriptions"])
    
    state["messages"].append(HumanMessage(
        content=f"Received {len(state['push_data'])} push updates"
    ))
    state["current_step"] = "push_received"
    return state


def fetch_pull_data(state: PushPullState) -> PushPullState:
    """Fetch pull data"""
    consumer = PullConsumer()
    
    state["pull_data"] = consumer.fetch_data(state["pull_requests"])
    
    state["messages"].append(HumanMessage(
        content=f"Fetched {len(state['pull_data'])} pull responses"
    ))
    state["current_step"] = "pull_fetched"
    return state


def optimize_strategy(state: PushPullState) -> PushPullState:
    """Optimize hybrid strategy"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    coordinator = HybridCoordinator(llm)
    
    strategy = coordinator.determine_strategy(state["data_streams"])
    state["hybrid_strategy"] = strategy
    
    state["messages"].append(HumanMessage(
        content=f"Optimized: {len(strategy.get('push_streams', []))} push, "
                f"{len(strategy.get('pull_streams', []))} pull"
    ))
    state["current_step"] = "optimized"
    return state


def generate_report(state: PushPullState) -> PushPullState:
    """Generate final report"""
    report = f"""
PUSH-PULL HYBRID REPORT
=======================

Data Streams: {len(state['data_streams'])}

PUSH MODEL (Proactive):
-----------------------
Subscriptions: {len(state['push_subscriptions'])}

"""
    
    for sub in state['push_subscriptions']:
        report += f"- {sub['stream']}: {sub['frequency']} updates\n"
    
    report += f"""
Push Data Received: {len(state['push_data'])} updates

PULL MODEL (On-Demand):
-----------------------
Requests: {len(state['pull_requests'])}

"""
    
    for req in state['pull_requests']:
        report += f"- {req['stream']}: {req['timing']} access\n"
    
    report += f"""
Pull Data Fetched: {len(state['pull_data'])} responses

HYBRID STRATEGY:
----------------
{state['hybrid_strategy'].get('rationale', 'N/A')}

Push Streams: {', '.join(state['hybrid_strategy'].get('push_streams', []))}
Pull Streams: {', '.join(state['hybrid_strategy'].get('pull_streams', []))}

Summary:
Push-pull hybrid balances real-time updates with on-demand efficiency.
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_push_pull_graph():
    """Create push-pull hybrid workflow"""
    workflow = StateGraph(PushPullState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_system)
    workflow.add_node("setup_push", setup_push)
    workflow.add_node("setup_pull", setup_pull)
    workflow.add_node("receive_push", receive_push_data)
    workflow.add_node("fetch_pull", fetch_pull_data)
    workflow.add_node("optimize", optimize_strategy)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "setup_push")
    workflow.add_edge("setup_push", "setup_pull")
    workflow.add_edge("setup_pull", "receive_push")
    workflow.add_edge("receive_push", "fetch_pull")
    workflow.add_edge("fetch_pull", "optimize")
    workflow.add_edge("optimize", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "data_streams": [
            "stock_prices",  # Push - real-time critical
            "weather_updates",  # Push - frequent changes
            "user_reports",  # Pull - on-demand
            "analytics_data"  # Pull - resource-intensive
        ],
        "push_subscriptions": [],
        "pull_requests": [],
        "push_data": [],
        "pull_data": [],
        "hybrid_strategy": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_push_pull_graph()
    
    print("Push-Pull Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")

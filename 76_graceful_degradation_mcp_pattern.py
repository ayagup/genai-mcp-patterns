"""
Graceful Degradation MCP Pattern

This pattern maintains partial functionality when full service is unavailable,
providing reduced but acceptable service instead of complete failure.

Key Features:
- Feature prioritization
- Partial service maintenance
- Progressive degradation levels
- Capability assessment
- Service recovery
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class DegradationState(TypedDict):
    """State for graceful degradation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    service_name: str
    available_resources: list[str]
    unavailable_resources: list[str]
    degradation_level: int  # 0=full, 1=minor, 2=moderate, 3=severe
    enabled_features: list[str]
    disabled_features: list[str]
    user_request: str
    response_quality: str
    can_fulfill: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Health Assessor
def health_assessor(state: DegradationState) -> DegradationState:
    """Assesses system health and available capabilities"""
    service_name = state.get("service_name", "")
    available_resources = state.get("available_resources", [])
    unavailable_resources = state.get("unavailable_resources", [])
    
    system_message = SystemMessage(content="""You are a health assessor. 
    Evaluate system health and determine degradation level.""")
    
    user_message = HumanMessage(content=f"""Assess system health:

Service: {service_name}
Available Resources: {', '.join(available_resources)}
Unavailable Resources: {', '.join(unavailable_resources)}

Determine the system degradation level.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate degradation level
    total_resources = len(available_resources) + len(unavailable_resources)
    availability_ratio = len(available_resources) / total_resources if total_resources > 0 else 0
    
    if availability_ratio >= 0.9:
        degradation_level = 0  # Full service
        level_name = "🟢 FULL SERVICE"
    elif availability_ratio >= 0.7:
        degradation_level = 1  # Minor degradation
        level_name = "🟡 MINOR DEGRADATION"
    elif availability_ratio >= 0.5:
        degradation_level = 2  # Moderate degradation
        level_name = "🟠 MODERATE DEGRADATION"
    else:
        degradation_level = 3  # Severe degradation
        level_name = "🔴 SEVERE DEGRADATION"
    
    assessment = f"""
    {level_name}
    
    System Health:
    • Total Resources: {total_resources}
    • Available: {len(available_resources)} ({availability_ratio*100:.0f}%)
    • Unavailable: {len(unavailable_resources)} ({(1-availability_ratio)*100:.0f}%)
    • Degradation Level: {degradation_level}/3
    
    Available: {', '.join(available_resources) if available_resources else 'None'}
    Unavailable: {', '.join(unavailable_resources) if unavailable_resources else 'None'}
    """
    
    return {
        "messages": [AIMessage(content=f"🏥 Health Assessor:\n{response.content}\n{assessment}")],
        "degradation_level": degradation_level
    }


# Feature Controller
def feature_controller(state: DegradationState) -> DegradationState:
    """Controls which features are enabled based on degradation level"""
    degradation_level = state.get("degradation_level", 0)
    service_name = state.get("service_name", "")
    available_resources = state.get("available_resources", [])
    
    system_message = SystemMessage(content="""You are a feature controller. 
    Enable/disable features based on available resources and degradation level.""")
    
    user_message = HumanMessage(content=f"""Control features:

Service: {service_name}
Degradation Level: {degradation_level}
Available Resources: {', '.join(available_resources)}

Determine which features should be enabled.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define feature sets for each degradation level
    all_features = [
        "full_search",
        "recommendations",
        "analytics",
        "real_time_updates",
        "advanced_filtering",
        "basic_search",
        "cached_data",
        "error_messages"
    ]
    
    if degradation_level == 0:  # Full service
        enabled_features = all_features
        disabled_features = []
    elif degradation_level == 1:  # Minor degradation
        enabled_features = [
            "full_search", "basic_search", "cached_data",
            "advanced_filtering", "error_messages"
        ]
        disabled_features = ["recommendations", "analytics", "real_time_updates"]
    elif degradation_level == 2:  # Moderate degradation
        enabled_features = ["basic_search", "cached_data", "error_messages"]
        disabled_features = [
            "full_search", "recommendations", "analytics",
            "real_time_updates", "advanced_filtering"
        ]
    else:  # Severe degradation
        enabled_features = ["cached_data", "error_messages"]
        disabled_features = [
            "full_search", "basic_search", "recommendations",
            "analytics", "real_time_updates", "advanced_filtering"
        ]
    
    feature_status = f"""
    Feature Configuration:
    
    ✅ Enabled ({len(enabled_features)}):
    {chr(10).join(f'   • {f}' for f in enabled_features)}
    
    ❌ Disabled ({len(disabled_features)}):
    {chr(10).join(f'   • {f}' for f in disabled_features) if disabled_features else '   (none)'}
    
    Strategy: Prioritize core functionality, disable non-essential features
    """
    
    return {
        "messages": [AIMessage(content=f"⚙️ Feature Controller:\n{response.content}\n{feature_status}")],
        "enabled_features": enabled_features,
        "disabled_features": disabled_features
    }


# Request Handler
def request_handler(state: DegradationState) -> DegradationState:
    """Handles requests with degraded service capabilities"""
    user_request = state.get("user_request", "")
    enabled_features = state.get("enabled_features", [])
    disabled_features = state.get("disabled_features", [])
    degradation_level = state.get("degradation_level", 0)
    
    system_message = SystemMessage(content="""You are a request handler. 
    Process user requests with available features, providing best possible service.""")
    
    user_message = HumanMessage(content=f"""Handle request:

User Request: {user_request}
Enabled Features: {', '.join(enabled_features)}
Disabled Features: {', '.join(disabled_features)}
Degradation Level: {degradation_level}

Process the request with available capabilities.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine if request can be fulfilled
    request_lower = user_request.lower()
    
    if degradation_level == 0:
        can_fulfill = True
        response_quality = "FULL - All features available"
    elif degradation_level == 1:
        if "recommendation" in request_lower or "analytics" in request_lower:
            can_fulfill = True
            response_quality = "PARTIAL - Using cached/fallback data"
        else:
            can_fulfill = True
            response_quality = "FULL - Core features functional"
    elif degradation_level == 2:
        if "search" in request_lower and "basic_search" in enabled_features:
            can_fulfill = True
            response_quality = "DEGRADED - Basic search only, limited results"
        elif "cached_data" in enabled_features:
            can_fulfill = True
            response_quality = "DEGRADED - Cached data only (may be stale)"
        else:
            can_fulfill = False
            response_quality = "UNAVAILABLE - Required features offline"
    else:  # Severe
        if "cached_data" in enabled_features:
            can_fulfill = True
            response_quality = "MINIMAL - Very limited cached data"
        else:
            can_fulfill = False
            response_quality = "UNAVAILABLE - Service severely degraded"
    
    quality_icon = "✅" if "FULL" in response_quality else "⚠️" if can_fulfill else "❌"
    
    handling_result = f"""
    {quality_icon} Request Handling Result:
    
    • Request: {user_request}
    • Can Fulfill: {'Yes' if can_fulfill else 'No'}
    • Response Quality: {response_quality}
    
    {'User Experience: Degraded but functional' if can_fulfill and degradation_level > 0 else ''}
    {'User Experience: Full functionality' if can_fulfill and degradation_level == 0 else ''}
    {'User Experience: Service unavailable' if not can_fulfill else ''}
    """
    
    return {
        "messages": [AIMessage(content=f"🔧 Request Handler:\n{response.content}\n{handling_result}")],
        "response_quality": response_quality,
        "can_fulfill": can_fulfill
    }


# Degradation Monitor
def degradation_monitor(state: DegradationState) -> DegradationState:
    """Monitors degradation status and provides recovery guidance"""
    service_name = state.get("service_name", "")
    degradation_level = state.get("degradation_level", 0)
    enabled_features = state.get("enabled_features", [])
    disabled_features = state.get("disabled_features", [])
    available_resources = state.get("available_resources", [])
    unavailable_resources = state.get("unavailable_resources", [])
    user_request = state.get("user_request", "")
    response_quality = state.get("response_quality", "")
    can_fulfill = state.get("can_fulfill", False)
    
    level_names = {
        0: "🟢 Full Service",
        1: "🟡 Minor Degradation",
        2: "🟠 Moderate Degradation",
        3: "🔴 Severe Degradation"
    }
    
    level_emoji = ["🟢", "🟡", "🟠", "🔴"][degradation_level]
    
    summary = f"""
    {level_emoji} GRACEFUL DEGRADATION STATUS
    
    Service: {service_name}
    Degradation Level: {degradation_level}/3 - {level_names.get(degradation_level, 'Unknown')}
    
    Request Status:
    • User Request: {user_request}
    • Fulfillment: {'✅ Success' if can_fulfill else '❌ Failed'}
    • Quality: {response_quality}
    
    System Resources:
    • Available ({len(available_resources)}): {', '.join(available_resources)}
    • Unavailable ({len(unavailable_resources)}): {', '.join(unavailable_resources)}
    
    Features Status:
    • Enabled: {len(enabled_features)} features
    • Disabled: {len(disabled_features)} features
    
    Degradation Strategy:
    1. Level 0 (Full): All features enabled, optimal performance
    2. Level 1 (Minor): Disable non-critical features (analytics, etc.)
    3. Level 2 (Moderate): Core features only, use cached data
    4. Level 3 (Severe): Minimal service, cached data only
    
    Graceful Degradation Process:
    1. Assess Health → Determine resource availability
    2. Set Degradation Level → Based on available resources
    3. Control Features → Enable/disable based on level
    4. Handle Requests → Provide best possible service
    5. Monitor & Recover → Watch for resource restoration
    
    Pattern Benefits:
    • Maintains partial service vs complete failure
    • Prioritizes critical functionality
    • Transparent to users (with notifications)
    • Automatic recovery when resources return
    • Better user experience
    • Improved availability
    
    Recovery Strategy:
    • Monitor unavailable resources continuously
    • Restore features progressively as resources return
    • Notify users of service restoration
    • Log degradation events for analysis
    
    User Communication:
    Level 0: Normal operation
    Level 1: "Some features temporarily limited"
    Level 2: "Service operating in limited mode"
    Level 3: "Minimal service available, please try again later"
    
    Key Insight:
    Graceful degradation maintains partial functionality during failures,
    providing degraded but acceptable service instead of complete outage.
    Critical for user experience and service availability.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Degradation Monitor:\n{summary}")]
    }


# Build the graph
def build_degradation_graph():
    """Build the graceful degradation pattern graph"""
    workflow = StateGraph(DegradationState)
    
    workflow.add_node("assessor", health_assessor)
    workflow.add_node("controller", feature_controller)
    workflow.add_node("handler", request_handler)
    workflow.add_node("monitor", degradation_monitor)
    
    workflow.add_edge(START, "assessor")
    workflow.add_edge("assessor", "controller")
    workflow.add_edge("controller", "handler")
    workflow.add_edge("handler", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_degradation_graph()
    
    print("=== Graceful Degradation MCP Pattern ===\n")
    
    # Simulate different degradation scenarios
    scenarios = [
        {
            "name": "Full Service",
            "available": ["database", "cache", "search_engine", "analytics", "recommendations"],
            "unavailable": [],
            "request": "Show me product recommendations with analytics"
        },
        {
            "name": "Minor Degradation",
            "available": ["database", "cache", "search_engine"],
            "unavailable": ["analytics", "recommendations"],
            "request": "Search for products with filters"
        },
        {
            "name": "Moderate Degradation",
            "available": ["cache"],
            "unavailable": ["database", "search_engine", "analytics", "recommendations"],
            "request": "Search for products"
        },
        {
            "name": "Severe Degradation",
            "available": ["cache"],
            "unavailable": ["database", "search_engine", "analytics", "recommendations", "filters"],
            "request": "Get any product data"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario['name']}")
        print('='*70)
        
        state = {
            "messages": [],
            "service_name": "ProductCatalogService",
            "available_resources": scenario["available"],
            "unavailable_resources": scenario["unavailable"],
            "degradation_level": 0,
            "enabled_features": [],
            "disabled_features": [],
            "user_request": scenario["request"],
            "response_quality": "",
            "can_fulfill": False
        }
        
        result = graph.invoke(state)
        
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)

"""
Delegate MCP Pattern

This pattern demonstrates delegating tasks to other specialized agents
based on task type and complexity.

Key Features:
- Task analysis and decomposition
- Intelligent delegation to specialists
- Result aggregation
- Delegation tracking
- Dynamic delegation strategies
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class DelegateState(TypedDict):
    """State for delegate pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    task_type: str
    complexity: str
    delegated_to: str
    specialist_results: dict[str, str]
    aggregated_result: str
    delegation_log: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Task Analyzer
def task_analyzer(state: DelegateState) -> DelegateState:
    """Analyzes task to determine type and complexity"""
    task = state.get("task", "")
    
    system_message = SystemMessage(content="""You are a task analyzer. Analyze tasks 
    to determine their type and complexity for proper delegation.""")
    
    user_message = HumanMessage(content=f"""Analyze task: {task}

Determine:
- Task type (technical, business, creative, analytical)
- Complexity (simple, moderate, complex)
- Best specialist to delegate to""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simple task classification
    task_lower = task.lower()
    if any(word in task_lower for word in ["code", "program", "software", "api"]):
        task_type = "technical"
    elif any(word in task_lower for word in ["analyze", "data", "metrics", "statistics"]):
        task_type = "analytical"
    elif any(word in task_lower for word in ["design", "creative", "marketing", "brand"]):
        task_type = "creative"
    else:
        task_type = "business"
    
    # Determine complexity
    word_count = len(task.split())
    complexity = "complex" if word_count > 15 else "moderate" if word_count > 8 else "simple"
    
    log_entry = f"[ANALYSIS] Task classified as {task_type.upper()} with {complexity.upper()} complexity"
    
    return {
        "messages": [AIMessage(content=f"🔍 Task Analyzer: {response.content}\n\n✅ Classified as {task_type} ({complexity})")],
        "task_type": task_type,
        "complexity": complexity,
        "delegation_log": [log_entry]
    }


# Delegation Manager
def delegation_manager(state: DelegateState) -> DelegateState:
    """Manages delegation to appropriate specialists"""
    task = state.get("task", "")
    task_type = state.get("task_type", "")
    complexity = state.get("complexity", "")
    
    system_message = SystemMessage(content="""You are a delegation manager. Assign tasks 
    to the most appropriate specialist based on task type and complexity.""")
    
    user_message = HumanMessage(content=f"""Delegate task: {task}

Task Type: {task_type}
Complexity: {complexity}

Select best specialist for delegation.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine which specialist to delegate to
    specialist_map = {
        "technical": "technical_specialist",
        "analytical": "data_specialist",
        "creative": "creative_specialist",
        "business": "business_specialist"
    }
    
    delegated_to = specialist_map.get(task_type, "general_specialist")
    
    log_entry = f"[DELEGATION] Task delegated to {delegated_to.upper()}"
    
    return {
        "messages": [AIMessage(content=f"📋 Delegation Manager: {response.content}\n\n✅ Delegating to {delegated_to}")],
        "delegated_to": delegated_to,
        "delegation_log": [log_entry]
    }


# Technical Specialist
def technical_specialist(state: DelegateState) -> DelegateState:
    """Handles technical tasks"""
    task = state.get("task", "")
    delegated_to = state.get("delegated_to", "")
    
    if delegated_to != "technical_specialist":
        return {"messages": [AIMessage(content="⏭️ Technical Specialist: Skipped (not delegated)")]}
    
    system_message = SystemMessage(content="""You are a technical specialist with expertise 
    in software development, APIs, and technical solutions.""")
    
    user_message = HumanMessage(content=f"""Execute technical task: {task}

Provide technical solution.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💻 Technical Specialist: {response.content}")],
        "specialist_results": {"technical": response.content}
    }


# Data Specialist
def data_specialist(state: DelegateState) -> DelegateState:
    """Handles analytical and data tasks"""
    task = state.get("task", "")
    delegated_to = state.get("delegated_to", "")
    
    if delegated_to != "data_specialist":
        return {"messages": [AIMessage(content="⏭️ Data Specialist: Skipped (not delegated)")]}
    
    system_message = SystemMessage(content="""You are a data specialist with expertise 
    in data analysis, statistics, and metrics.""")
    
    user_message = HumanMessage(content=f"""Execute analytical task: {task}

Provide data-driven analysis.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📊 Data Specialist: {response.content}")],
        "specialist_results": {"analytical": response.content}
    }


# Creative Specialist
def creative_specialist(state: DelegateState) -> DelegateState:
    """Handles creative tasks"""
    task = state.get("task", "")
    delegated_to = state.get("delegated_to", "")
    
    if delegated_to != "creative_specialist":
        return {"messages": [AIMessage(content="⏭️ Creative Specialist: Skipped (not delegated)")]}
    
    system_message = SystemMessage(content="""You are a creative specialist with expertise 
    in design, marketing, and creative content.""")
    
    user_message = HumanMessage(content=f"""Execute creative task: {task}

Provide creative solution.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🎨 Creative Specialist: {response.content}")],
        "specialist_results": {"creative": response.content}
    }


# Business Specialist
def business_specialist(state: DelegateState) -> DelegateState:
    """Handles business tasks"""
    task = state.get("task", "")
    delegated_to = state.get("delegated_to", "")
    
    if delegated_to != "business_specialist":
        return {"messages": [AIMessage(content="⏭️ Business Specialist: Skipped (not delegated)")]}
    
    system_message = SystemMessage(content="""You are a business specialist with expertise 
    in strategy, operations, and business processes.""")
    
    user_message = HumanMessage(content=f"""Execute business task: {task}

Provide business solution.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"💼 Business Specialist: {response.content}")],
        "specialist_results": {"business": response.content}
    }


# Result Aggregator
def result_aggregator(state: DelegateState) -> DelegateState:
    """Aggregates results from specialists"""
    specialist_results = state.get("specialist_results", {})
    task = state.get("task", "")
    delegated_to = state.get("delegated_to", "")
    
    system_message = SystemMessage(content="""You are a result aggregator. Compile and 
    format specialist results into a comprehensive response.""")
    
    results_text = "\n".join([f"{k}: {v[:150]}..." for k, v in specialist_results.items()])
    
    user_message = HumanMessage(content=f"""Aggregate results for task: {task}

Specialist Results:
{results_text}

Create final comprehensive response.""")
    
    response = llm.invoke([system_message, user_message])
    
    log_entry = f"[AGGREGATION] Results compiled from {delegated_to}"
    
    return {
        "messages": [AIMessage(content=f"📦 Result Aggregator: {response.content}")],
        "aggregated_result": response.content,
        "delegation_log": [log_entry]
    }


# Delegation Monitor
def delegation_monitor(state: DelegateState) -> DelegateState:
    """Monitors delegation process"""
    task = state.get("task", "")
    task_type = state.get("task_type", "")
    complexity = state.get("complexity", "")
    delegated_to = state.get("delegated_to", "")
    delegation_log = state.get("delegation_log", [])
    aggregated_result = state.get("aggregated_result", "")
    
    logs_text = "\n".join([f"  {log}" for log in delegation_log])
    
    summary = f"""
    ✅ DELEGATE PATTERN COMPLETE
    
    Delegation Summary:
    • Original Task: {task[:80]}...
    • Task Type: {task_type.upper()}
    • Complexity: {complexity.upper()}
    • Delegated To: {delegated_to.upper()}
    • Log Entries: {len(delegation_log)}
    
    Delegation Log:
{logs_text}
    
    Delegation Benefits:
    • Specialized expertise for each task type
    • Efficient task distribution
    • Quality results from domain experts
    • Scalable task management
    • Clear delegation tracking
    
    Final Result:
    {aggregated_result[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Delegation Monitor:\n{summary}")]
    }


# Build the graph
def build_delegate_graph():
    """Build the delegate pattern graph"""
    workflow = StateGraph(DelegateState)
    
    workflow.add_node("analyzer", task_analyzer)
    workflow.add_node("manager", delegation_manager)
    workflow.add_node("technical", technical_specialist)
    workflow.add_node("data", data_specialist)
    workflow.add_node("creative", creative_specialist)
    workflow.add_node("business", business_specialist)
    workflow.add_node("aggregator", result_aggregator)
    workflow.add_node("monitor", delegation_monitor)
    
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "manager")
    workflow.add_edge("manager", "technical")
    workflow.add_edge("technical", "data")
    workflow.add_edge("data", "creative")
    workflow.add_edge("creative", "business")
    workflow.add_edge("business", "aggregator")
    workflow.add_edge("aggregator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_delegate_graph()
    
    print("=== Delegate MCP Pattern ===\n")
    
    # Technical task
    print("\n--- Example 1: Technical Task ---")
    initial_state = {
        "messages": [],
        "task": "Design a REST API for user authentication with JWT tokens",
        "task_type": "",
        "complexity": "",
        "delegated_to": "",
        "specialist_results": {},
        "aggregated_result": "",
        "delegation_log": []
    }
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Analytical task
    print("\n\n--- Example 2: Analytical Task ---")
    initial_state["messages"] = []
    initial_state["task"] = "Analyze customer retention metrics and identify trends"
    initial_state["delegation_log"] = []
    initial_state["specialist_results"] = {}
    
    result = graph.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)

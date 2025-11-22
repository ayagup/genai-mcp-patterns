"""
Mediator MCP Pattern

This pattern demonstrates centralizing communication between agents,
reducing direct dependencies and coupling.

Key Features:
- Centralized communication
- Reduced coupling
- Coordinated interactions
- Simplified agent logic
- Flexible collaboration
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class MediatorState(TypedDict):
    """State for mediator pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    agent_a_output: str
    agent_b_output: str
    agent_c_output: str
    mediator_decisions: list[str]
    collaboration_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Mediator - Central Coordinator
def mediator(state: MediatorState) -> MediatorState:
    """Centralizes communication and coordinates agents"""
    task = state.get("task", "")
    
    system_message = SystemMessage(content="""You are a mediator. Coordinate communication 
    between agents, manage their interactions, and orchestrate collaboration.""")
    
    user_message = HumanMessage(content=f"""Coordinate agents for task: {task}

Available agents:
- Agent A: Data collection and preprocessing
- Agent B: Analysis and insights
- Agent C: Reporting and visualization

Create coordination plan.""")
    
    response = llm.invoke([system_message, user_message])
    
    mediator_decisions = [
        "DECISION: Assign data collection to Agent A",
        "DECISION: Agent B will analyze after Agent A completes",
        "DECISION: Agent C will create report from Agent B's analysis"
    ]
    
    return {
        "messages": [AIMessage(content=f"🎯 Mediator: {response.content}\n\n✅ Created coordination plan")],
        "mediator_decisions": mediator_decisions
    }


# Agent A - Data Collector
def agent_a(state: MediatorState) -> MediatorState:
    """Collects and preprocesses data"""
    task = state.get("task", "")
    mediator_decisions = state.get("mediator_decisions", [])
    
    system_message = SystemMessage(content="""You are Agent A specializing in data 
    collection and preprocessing. Communicate through the mediator.""")
    
    user_message = HumanMessage(content=f"""Collect data for: {task}

Mediator Instructions:
{chr(10).join(mediator_decisions[:1])}

Perform data collection.""")
    
    response = llm.invoke([system_message, user_message])
    
    agent_a_output = "Collected 1000 data points, cleaned and preprocessed, ready for analysis"
    
    return {
        "messages": [AIMessage(content=f"🅰️ Agent A: {response.content}\n\n➡️ Sending to Mediator: Data collection complete")],
        "agent_a_output": agent_a_output
    }


# Mediator Coordination 1
def mediator_coord_1(state: MediatorState) -> MediatorState:
    """Mediator coordinates after Agent A"""
    agent_a_output = state.get("agent_a_output", "")
    
    system_message = SystemMessage(content="""You are the mediator coordinating the workflow. 
    Agent A has completed, now coordinate Agent B.""")
    
    user_message = HumanMessage(content=f"""Agent A completed: {agent_a_output}

Coordinate next step with Agent B.""")
    
    response = llm.invoke([system_message, user_message])
    
    decision = "COORDINATION: Agent A complete, forwarding data to Agent B for analysis"
    
    return {
        "messages": [AIMessage(content=f"🎯 Mediator Coordination: {response.content}")],
        "mediator_decisions": [decision]
    }


# Agent B - Analyzer
def agent_b(state: MediatorState) -> MediatorState:
    """Analyzes data"""
    agent_a_output = state.get("agent_a_output", "")
    
    system_message = SystemMessage(content="""You are Agent B specializing in analysis 
    and insights. Communicate through the mediator.""")
    
    user_message = HumanMessage(content=f"""Analyze data from Agent A:

Data: {agent_a_output}

Perform analysis and generate insights.""")
    
    response = llm.invoke([system_message, user_message])
    
    agent_b_output = "Analysis complete: Identified 3 key trends, 2 anomalies, confidence: 0.87"
    
    return {
        "messages": [AIMessage(content=f"🅱️ Agent B: {response.content}\n\n➡️ Sending to Mediator: Analysis complete")],
        "agent_b_output": agent_b_output
    }


# Mediator Coordination 2
def mediator_coord_2(state: MediatorState) -> MediatorState:
    """Mediator coordinates after Agent B"""
    agent_b_output = state.get("agent_b_output", "")
    
    system_message = SystemMessage(content="""You are the mediator coordinating the workflow. 
    Agent B has completed, now coordinate Agent C.""")
    
    user_message = HumanMessage(content=f"""Agent B completed: {agent_b_output}

Coordinate final step with Agent C.""")
    
    response = llm.invoke([system_message, user_message])
    
    decision = "COORDINATION: Agent B complete, forwarding insights to Agent C for reporting"
    
    return {
        "messages": [AIMessage(content=f"🎯 Mediator Coordination: {response.content}")],
        "mediator_decisions": [decision]
    }


# Agent C - Reporter
def agent_c(state: MediatorState) -> MediatorState:
    """Creates reports"""
    agent_b_output = state.get("agent_b_output", "")
    
    system_message = SystemMessage(content="""You are Agent C specializing in reporting 
    and visualization. Communicate through the mediator.""")
    
    user_message = HumanMessage(content=f"""Create report from Agent B's analysis:

Analysis: {agent_b_output}

Generate comprehensive report.""")
    
    response = llm.invoke([system_message, user_message])
    
    agent_c_output = "Report generated: 5 visualizations, executive summary, detailed findings"
    
    return {
        "messages": [AIMessage(content=f"🅲 Agent C: {response.content}\n\n➡️ Sending to Mediator: Report complete")],
        "agent_c_output": agent_c_output
    }


# Mediator Final Coordination
def mediator_final(state: MediatorState) -> MediatorState:
    """Mediator finalizes and aggregates results"""
    agent_a_output = state.get("agent_a_output", "")
    agent_b_output = state.get("agent_b_output", "")
    agent_c_output = state.get("agent_c_output", "")
    
    system_message = SystemMessage(content="""You are the mediator finalizing the collaboration. 
    Aggregate results from all agents.""")
    
    user_message = HumanMessage(content=f"""Finalize collaboration:

Agent A: {agent_a_output}
Agent B: {agent_b_output}
Agent C: {agent_c_output}

Create final result.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🎯 Mediator Final: {response.content}")],
        "collaboration_result": response.content
    }


# Mediator Monitor
def mediator_monitor(state: MediatorState) -> MediatorState:
    """Monitors mediated collaboration"""
    task = state.get("task", "")
    mediator_decisions = state.get("mediator_decisions", [])
    agent_a_output = state.get("agent_a_output", "")
    agent_b_output = state.get("agent_b_output", "")
    agent_c_output = state.get("agent_c_output", "")
    collaboration_result = state.get("collaboration_result", "")
    
    decisions_text = "\n".join([f"  • {decision}" for decision in mediator_decisions])
    
    summary = f"""
    ✅ MEDIATOR PATTERN COMPLETE
    
    Mediation Summary:
    • Task: {task[:80]}...
    • Agents Coordinated: 3
    • Mediator Decisions: {len(mediator_decisions)}
    
    Mediator Decisions:
{decisions_text}
    
    Agent Outputs:
    • Agent A: {agent_a_output[:60]}...
    • Agent B: {agent_b_output[:60]}...
    • Agent C: {agent_c_output[:60]}...
    
    Communication Pattern:
    Task → Mediator → Agent A → Mediator → Agent B → Mediator → Agent C → Mediator → Result
    
    Mediator Benefits:
    • Centralized communication
    • Reduced agent coupling
    • Coordinated interactions
    • Simplified agent logic
    • Flexible collaboration
    • Single point of control
    
    Collaboration Result:
    {collaboration_result[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Mediator Monitor:\n{summary}")]
    }


# Build the graph
def build_mediator_graph():
    """Build the mediator pattern graph"""
    workflow = StateGraph(MediatorState)
    
    workflow.add_node("mediator_start", mediator)
    workflow.add_node("agent_a", agent_a)
    workflow.add_node("mediator_1", mediator_coord_1)
    workflow.add_node("agent_b", agent_b)
    workflow.add_node("mediator_2", mediator_coord_2)
    workflow.add_node("agent_c", agent_c)
    workflow.add_node("mediator_final", mediator_final)
    workflow.add_node("monitor", mediator_monitor)
    
    workflow.add_edge(START, "mediator_start")
    workflow.add_edge("mediator_start", "agent_a")
    workflow.add_edge("agent_a", "mediator_1")
    workflow.add_edge("mediator_1", "agent_b")
    workflow.add_edge("agent_b", "mediator_2")
    workflow.add_edge("mediator_2", "agent_c")
    workflow.add_edge("agent_c", "mediator_final")
    workflow.add_edge("mediator_final", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_mediator_graph()
    
    print("=== Mediator MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "task": "Analyze customer behavior patterns and create executive report",
        "agent_a_output": "",
        "agent_b_output": "",
        "agent_c_output": "",
        "mediator_decisions": [],
        "collaboration_result": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Mediated Collaboration Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Collaboration Result ===")
    print(result.get("collaboration_result", "No result generated"))

"""
Master-Slave MCP Pattern
=========================
A master agent controls and directs multiple slave agents that execute tasks.
The master has full authority over task assignment and slave behavior.
Slaves report back to master and have limited autonomy.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Dict, List
import operator
import json


# Define the state
class MasterSlaveState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    master_commands: Dict[str, str]  # Commands from master to each slave
    slave_status: Dict[str, str]  # Status of each slave
    slave_outputs: Dict[str, str]  # Output from each slave
    master_decision: str
    execution_complete: bool


# Master Agent
def master_agent(state: MasterSlaveState):
    """Master agent that controls and directs slave agents."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are the Master Agent with full control over slave agents.\n"
                "Available slaves:\n"
                "- Slave 1: Data processor\n"
                "- Slave 2: Calculator\n"
                "- Slave 3: Reporter\n"
                "Issue specific commands to each slave. Format as JSON:\n"
                "{\"commands\": {\"slave_1\": \"command\", \"slave_2\": \"command\", \"slave_3\": \"command\"}}"
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Parse master commands
    try:
        parsed = json.loads(response.content)
        commands = parsed.get("commands", {})
    except:
        commands = {
            "slave_1": "Process the input data",
            "slave_2": "Perform calculations",
            "slave_3": "Generate report"
        }
    
    return {
        "messages": [AIMessage(content=f"Master: Issuing commands to slaves - {response.content}")],
        "master_commands": commands,
        "slave_status": {k: "assigned" for k in commands.keys()}
    }


# Slave 1 - Data Processor
def slave_1_agent(state: MasterSlaveState):
    """Slave agent 1 - executes master's commands for data processing."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    command = state.get("master_commands", {}).get("slave_1", "No command")
    
    system_msg = SystemMessage(
        content=f"You are Slave 1 - Data Processor. You must follow master's commands.\n"
                f"Master's command: {command}\n"
                "Execute the command exactly as directed and report completion."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update slave status
    status = state.get("slave_status", {})
    status["slave_1"] = "completed"
    
    return {
        "messages": [AIMessage(content=f"Slave 1: Command executed - {response.content}")],
        "slave_outputs": {"slave_1": response.content},
        "slave_status": status
    }


# Slave 2 - Calculator
def slave_2_agent(state: MasterSlaveState):
    """Slave agent 2 - executes master's commands for calculations."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    command = state.get("master_commands", {}).get("slave_2", "No command")
    slave_1_output = state.get("slave_outputs", {}).get("slave_1", "No input")
    
    system_msg = SystemMessage(
        content=f"You are Slave 2 - Calculator. You must follow master's commands.\n"
                f"Master's command: {command}\n"
                f"Input from Slave 1: {slave_1_output}\n"
                "Execute the command exactly as directed and report completion."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update slave status
    status = state.get("slave_status", {})
    status["slave_2"] = "completed"
    
    return {
        "messages": [AIMessage(content=f"Slave 2: Command executed - {response.content}")],
        "slave_outputs": {"slave_2": response.content},
        "slave_status": status
    }


# Slave 3 - Reporter
def slave_3_agent(state: MasterSlaveState):
    """Slave agent 3 - executes master's commands for reporting."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    command = state.get("master_commands", {}).get("slave_3", "No command")
    slave_1_output = state.get("slave_outputs", {}).get("slave_1", "No data")
    slave_2_output = state.get("slave_outputs", {}).get("slave_2", "No calculations")
    
    system_msg = SystemMessage(
        content=f"You are Slave 3 - Reporter. You must follow master's commands.\n"
                f"Master's command: {command}\n"
                f"Input from Slave 1: {slave_1_output}\n"
                f"Input from Slave 2: {slave_2_output}\n"
                "Execute the command exactly as directed and report completion."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    # Update slave status
    status = state.get("slave_status", {})
    status["slave_3"] = "completed"
    
    return {
        "messages": [AIMessage(content=f"Slave 3: Command executed - {response.content}")],
        "slave_outputs": {"slave_3": response.content},
        "slave_status": status
    }


# Master Review
def master_review(state: MasterSlaveState):
    """Master reviews slave outputs and makes final decision."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    slave_outputs = state.get("slave_outputs", {})
    slave_status = state.get("slave_status", {})
    
    system_msg = SystemMessage(
        content=f"You are the Master Agent. Review slave outputs and provide final decision:\n"
                f"Slave 1 Status: {slave_status.get('slave_1')} - Output: {slave_outputs.get('slave_1', 'N/A')}\n"
                f"Slave 2 Status: {slave_status.get('slave_2')} - Output: {slave_outputs.get('slave_2', 'N/A')}\n"
                f"Slave 3 Status: {slave_status.get('slave_3')} - Output: {slave_outputs.get('slave_3', 'N/A')}\n"
                "Validate slave work and provide master's final decision."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Master Review: {response.content}")],
        "master_decision": response.content,
        "execution_complete": True
    }


# Build the master-slave graph
def create_master_slave_graph():
    """Create a master-slave workflow graph."""
    workflow = StateGraph(MasterSlaveState)
    
    # Add master and slave nodes
    workflow.add_node("master", master_agent)
    workflow.add_node("slave_1", slave_1_agent)
    workflow.add_node("slave_2", slave_2_agent)
    workflow.add_node("slave_3", slave_3_agent)
    workflow.add_node("master_review", master_review)
    
    # Master-Slave topology: Master commands -> Slaves execute -> Master reviews
    workflow.add_edge(START, "master")
    workflow.add_edge("master", "slave_1")
    workflow.add_edge("slave_1", "slave_2")
    workflow.add_edge("slave_2", "slave_3")
    workflow.add_edge("slave_3", "master_review")
    workflow.add_edge("master_review", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the master-slave agent system
    graph = create_master_slave_graph()
    
    print("=" * 60)
    print("MASTER-SLAVE MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Master-controlled task execution
    print("\n[Task: Process sales data and generate report]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Process Q4 sales data: revenue=$500K, costs=$300K, units=1000")],
        "master_commands": {},
        "slave_status": {},
        "slave_outputs": {},
        "master_decision": "",
        "execution_complete": False
    })
    
    print("\n--- Master-Slave Execution Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Slave Status ---")
    for slave, status in result.get("slave_status", {}).items():
        print(f"{slave}: {status}")
    
    print(f"\n--- Master's Final Decision ---")
    print(f"{result.get('master_decision', 'N/A')[:300]}...")
    
    print(f"\n--- Execution Complete: {result.get('execution_complete', False)} ---")
    
    print("\n" + "=" * 60)

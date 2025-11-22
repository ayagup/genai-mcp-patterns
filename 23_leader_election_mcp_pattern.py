"""
Leader Election MCP Pattern

This pattern demonstrates dynamic leader election among peer agents using 
voting mechanisms like Bully algorithm or Raft consensus.

Key Features:
- Dynamic leader election process
- Peer agents with equal status
- Leader failure detection and re-election
- Coordination through elected leader
"""

from typing import TypedDict, Sequence, Annotated
import operator
import random
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class LeaderElectionState(TypedDict):
    """State for leader election pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    agents: dict[str, dict[str, any]]  # agent_id -> {priority: int, health: str, capabilities: list}
    current_leader: str
    election_round: int
    task: str
    task_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Agent Election Coordinator
def election_coordinator(state: LeaderElectionState) -> LeaderElectionState:
    """Coordinates the leader election process"""
    agents = state.get("agents", {})
    election_round = state.get("election_round", 0)
    
    system_message = SystemMessage(content="""You are an election coordinator. Analyze the 
    available agents and their attributes (priority, health status, capabilities) to conduct 
    a fair leader election. Use a bully algorithm approach: highest priority agent becomes leader.""")
    
    agents_info = "\n".join([
        f"Agent {aid}: Priority={data['priority']}, Health={data['health']}, Capabilities={data['capabilities']}"
        for aid, data in agents.items()
    ])
    
    user_message = HumanMessage(content=f"""Election Round {election_round}
    
    Available Agents:
    {agents_info}
    
    Determine which agent should be elected as leader based on:
    1. Health status (only healthy agents eligible)
    2. Highest priority wins
    3. If tie, most capabilities wins
    
    Announce the elected leader.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine leader by highest priority among healthy agents
    healthy_agents = {aid: data for aid, data in agents.items() if data['health'] == 'healthy'}
    if healthy_agents:
        leader = max(healthy_agents.items(), key=lambda x: (x[1]['priority'], len(x[1]['capabilities'])))[0]
    else:
        leader = "none"
    
    return {
        "messages": [AIMessage(content=f"Election Coordinator: {response.content}")],
        "current_leader": leader,
        "election_round": election_round + 1
    }


# Agent 1: High Priority Agent
def agent_alpha(state: LeaderElectionState) -> LeaderElectionState:
    """High priority agent that can become leader"""
    current_leader = state.get("current_leader", "")
    task = state.get("task", "")
    
    agent_id = "agent_alpha"
    is_leader = current_leader == agent_id
    
    if is_leader:
        system_message = SystemMessage(content="""You are Agent Alpha, the elected LEADER. 
        You have high priority and strong analytical capabilities. Coordinate the team and 
        execute the task with authority.""")
        
        user_message = HumanMessage(content=f"""As the leader, execute this task:\n{task}
        
        Provide strategic direction and coordinate the team.""")
        
        response = llm.invoke([system_message, user_message])
        
        return {
            "messages": [AIMessage(content=f"🌟 LEADER Agent Alpha: {response.content}")],
            "task_result": response.content
        }
    else:
        return {
            "messages": [AIMessage(content=f"Agent Alpha: Standing by as follower. Current leader: {current_leader}")]
        }


# Agent 2: Medium Priority Agent
def agent_beta(state: LeaderElectionState) -> LeaderElectionState:
    """Medium priority agent that can become leader if alpha is unavailable"""
    current_leader = state.get("current_leader", "")
    task = state.get("task", "")
    
    agent_id = "agent_beta"
    is_leader = current_leader == agent_id
    
    if is_leader:
        system_message = SystemMessage(content="""You are Agent Beta, the elected LEADER. 
        You have good coordination and communication capabilities. Lead the team effectively.""")
        
        user_message = HumanMessage(content=f"""As the leader, execute this task:\n{task}
        
        Coordinate team efforts and ensure completion.""")
        
        response = llm.invoke([system_message, user_message])
        
        return {
            "messages": [AIMessage(content=f"🌟 LEADER Agent Beta: {response.content}")],
            "task_result": response.content
        }
    else:
        # Assist the leader
        system_message = SystemMessage(content="""You are Agent Beta, a follower. 
        Provide support and assistance to the current leader.""")
        
        user_message = HumanMessage(content=f"""Support the leader ({current_leader}) in this task:\n{task}
        
        Offer assistance and insights.""")
        
        response = llm.invoke([system_message, user_message])
        
        return {
            "messages": [AIMessage(content=f"Agent Beta (Follower): {response.content}")]
        }


# Agent 3: Lower Priority Agent
def agent_gamma(state: LeaderElectionState) -> LeaderElectionState:
    """Lower priority agent, typically a follower"""
    current_leader = state.get("current_leader", "")
    task = state.get("task", "")
    
    agent_id = "agent_gamma"
    is_leader = current_leader == agent_id
    
    if is_leader:
        system_message = SystemMessage(content="""You are Agent Gamma, the elected LEADER. 
        Though you have lower priority, you have specialized technical skills. Lead with your expertise.""")
        
        user_message = HumanMessage(content=f"""As the leader, execute this task:\n{task}
        
        Apply your technical expertise to guide the team.""")
        
        response = llm.invoke([system_message, user_message])
        
        return {
            "messages": [AIMessage(content=f"🌟 LEADER Agent Gamma: {response.content}")],
            "task_result": response.content
        }
    else:
        # Execute assigned work
        system_message = SystemMessage(content="""You are Agent Gamma, a follower with 
        technical expertise. Execute tasks assigned by the leader.""")
        
        user_message = HumanMessage(content=f"""Work under leader ({current_leader}) on this task:\n{task}
        
        Execute your part diligently.""")
        
        response = llm.invoke([system_message, user_message])
        
        return {
            "messages": [AIMessage(content=f"Agent Gamma (Follower): {response.content}")]
        }


# Health Monitor Agent
def health_monitor(state: LeaderElectionState) -> LeaderElectionState:
    """Monitors agent health and triggers re-election if leader fails"""
    agents = state.get("agents", {})
    current_leader = state.get("current_leader", "")
    
    # Simulate health check
    health_status = "\n".join([f"{aid}: {data['health']}" for aid, data in agents.items()])
    
    leader_health = agents.get(current_leader, {}).get('health', 'unknown')
    
    if leader_health != 'healthy':
        status = f"⚠️ ALERT: Leader {current_leader} health is {leader_health}. Re-election required!"
    else:
        status = f"✓ Leader {current_leader} is healthy. System stable."
    
    return {
        "messages": [AIMessage(content=f"Health Monitor: {status}\n\nAgent Status:\n{health_status}")]
    }


# Routing logic
def route_to_leader(state: LeaderElectionState) -> str:
    """Route to the appropriate leader agent"""
    leader = state.get("current_leader", "")
    
    if leader == "agent_alpha":
        return "agent_alpha"
    elif leader == "agent_beta":
        return "agent_beta"
    elif leader == "agent_gamma":
        return "agent_gamma"
    else:
        return "end"


# Build the graph
def build_leader_election_graph():
    """Build the leader election MCP pattern graph"""
    workflow = StateGraph(LeaderElectionState)
    
    # Add nodes
    workflow.add_node("election", election_coordinator)
    workflow.add_node("health_monitor", health_monitor)
    workflow.add_node("agent_alpha", agent_alpha)
    workflow.add_node("agent_beta", agent_beta)
    workflow.add_node("agent_gamma", agent_gamma)
    
    # Define edges
    workflow.add_edge(START, "election")
    workflow.add_edge("election", "health_monitor")
    
    # Route to elected leader
    workflow.add_conditional_edges(
        "health_monitor",
        route_to_leader,
        {
            "agent_alpha": "agent_alpha",
            "agent_beta": "agent_beta",
            "agent_gamma": "agent_gamma",
            "end": END
        }
    )
    
    # After leader executes, followers assist
    workflow.add_edge("agent_alpha", "agent_beta")
    workflow.add_edge("agent_beta", "agent_gamma")
    workflow.add_edge("agent_gamma", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_leader_election_graph()
    
    # Initial state - Scenario 1: Normal election
    print("=== Scenario 1: Normal Leader Election ===\n")
    initial_state = {
        "messages": [],
        "agents": {
            "agent_alpha": {
                "priority": 10,
                "health": "healthy",
                "capabilities": ["analysis", "strategy", "coordination"]
            },
            "agent_beta": {
                "priority": 7,
                "health": "healthy",
                "capabilities": ["communication", "coordination"]
            },
            "agent_gamma": {
                "priority": 5,
                "health": "healthy",
                "capabilities": ["technical", "implementation"]
            }
        },
        "current_leader": "",
        "election_round": 0,
        "task": "Analyze customer feedback data and create an improvement plan for our mobile app",
        "task_result": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Election Process ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
    
    print(f"\n\n=== Final Leader: {result['current_leader'].upper()} ===")
    
    # Scenario 2: Leader failure and re-election
    print("\n\n" + "="*60)
    print("=== Scenario 2: Leader Failure & Re-election ===\n")
    
    # Alpha fails, beta should be elected
    initial_state_2 = {
        "messages": [],
        "agents": {
            "agent_alpha": {
                "priority": 10,
                "health": "failed",  # Alpha is down!
                "capabilities": ["analysis", "strategy", "coordination"]
            },
            "agent_beta": {
                "priority": 7,
                "health": "healthy",
                "capabilities": ["communication", "coordination"]
            },
            "agent_gamma": {
                "priority": 5,
                "health": "healthy",
                "capabilities": ["technical", "implementation"]
            }
        },
        "current_leader": "",
        "election_round": 0,
        "task": "Emergency: Handle critical production bug affecting user authentication",
        "task_result": ""
    }
    
    result_2 = graph.invoke(initial_state_2)
    
    print("\n=== Re-election Process ===")
    for msg in result_2["messages"]:
        print(f"\n{msg.content}")
    
    print(f"\n\n=== New Leader After Failure: {result_2['current_leader'].upper()} ===")

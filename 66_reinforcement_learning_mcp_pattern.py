"""
Reinforcement Learning MCP Pattern

This pattern demonstrates agents learning through rewards and penalties,
optimizing their behavior based on environmental feedback.

Key Features:
- Reward-based learning
- Action selection
- Policy optimization
- Environment interaction
- Value estimation
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ReinforcementLearningState(TypedDict):
    """State for reinforcement learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    environment: str
    current_state: str
    available_actions: list[str]
    action_history: list[dict[str, str | float]]
    total_reward: float
    episode: int
    policy: dict[str, str]
    value_estimates: dict[str, float]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Environment Simulator
def environment_simulator(state: ReinforcementLearningState) -> ReinforcementLearningState:
    """Simulates the environment and provides current state"""
    environment = state.get("environment", "")
    episode = state.get("episode", 0)
    
    system_message = SystemMessage(content="""You are an environment simulator. 
    Present the current state and available actions to the agent.""")
    
    user_message = HumanMessage(content=f"""Simulate environment: {environment}

Episode: {episode + 1}

Describe the current state and available actions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define states and actions based on environment
    if "customer_service" in environment.lower():
        current_state = "Customer inquiry received"
        available_actions = [
            "Provide immediate solution",
            "Escalate to specialist",
            "Request more information",
            "Offer alternative options"
        ]
    elif "resource" in environment.lower():
        current_state = "High demand detected"
        available_actions = [
            "Scale up resources",
            "Optimize current usage",
            "Queue requests",
            "Load balance"
        ]
    else:
        current_state = "Initial state"
        available_actions = [
            "Action A",
            "Action B",
            "Action C",
            "Action D"
        ]
    
    return {
        "messages": [AIMessage(content=f"🌍 Environment ({environment}):\n{response.content}\n\n📍 State: {current_state}\n🎯 Actions: {len(available_actions)}")],
        "current_state": current_state,
        "available_actions": available_actions
    }


# Agent (Policy-based)
def policy_agent(state: ReinforcementLearningState) -> ReinforcementLearningState:
    """Agent selects action based on current policy"""
    current_state = state.get("current_state", "")
    available_actions = state.get("available_actions", [])
    policy = state.get("policy", {})
    value_estimates = state.get("value_estimates", {})
    episode = state.get("episode", 0)
    
    system_message = SystemMessage(content="""You are a reinforcement learning agent. 
    Select the best action based on your current policy and value estimates.""")
    
    # Show current policy
    policy_info = f"Policy for '{current_state}': {policy.get(current_state, 'Random (exploring)')}"
    
    # Show value estimates
    values_info = "\n".join([
        f"  • {action}: {value_estimates.get(action, 0.0):.2f}"
        for action in available_actions
    ])
    
    user_message = HumanMessage(content=f"""Select action:

Current State: {current_state}
{policy_info}

Available Actions & Values:
{values_info}

Choose action (explore vs exploit).""")
    
    response = llm.invoke([system_message, user_message])
    
    # Action selection (epsilon-greedy)
    epsilon = max(0.1, 1.0 - episode * 0.1)  # Decrease exploration over time
    
    # For simplicity, select action with highest value estimate
    # In early episodes, more random (exploration)
    if value_estimates and episode > 2:
        # Exploit: choose best action
        selected_action = max(
            available_actions,
            key=lambda a: value_estimates.get(a, 0.0)
        )
    else:
        # Explore: choose first action (simplified)
        selected_action = available_actions[episode % len(available_actions)]
    
    return {
        "messages": [AIMessage(content=f"🤖 Policy Agent (Episode {episode + 1}):\n{response.content}\n\n✅ Selected: {selected_action}\n📊 Exploration rate: {epsilon:.1%}")]
    }


# Reward Calculator
def reward_calculator(state: ReinforcementLearningState) -> ReinforcementLearningState:
    """Calculates reward for the action taken"""
    current_state = state.get("current_state", "")
    available_actions = state.get("available_actions", [])
    episode = state.get("episode", 0)
    policy = state.get("policy", {})
    
    system_message = SystemMessage(content="""You are a reward calculator. 
    Evaluate the action's effectiveness and assign a reward.""")
    
    # Determine which action was selected (from available actions)
    if episode > 2 and current_state in policy:
        selected_action = policy.get(current_state, available_actions[0])
    else:
        selected_action = available_actions[episode % len(available_actions)]
    
    user_message = HumanMessage(content=f"""Calculate reward:

State: {current_state}
Action Taken: {selected_action}

Evaluate effectiveness and assign reward.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate rewards (would be from actual environment)
    # Better actions get higher rewards
    action_rewards = {
        "Provide immediate solution": 10.0,
        "Escalate to specialist": 5.0,
        "Request more information": 3.0,
        "Offer alternative options": 7.0,
        "Scale up resources": 8.0,
        "Optimize current usage": 9.0,
        "Queue requests": 4.0,
        "Load balance": 8.0
    }
    
    reward = action_rewards.get(selected_action, 5.0)
    
    # Add to action history
    action_record = {
        "episode": str(episode + 1),
        "state": current_state,
        "action": selected_action,
        "reward": reward
    }
    
    return {
        "messages": [AIMessage(content=f"💰 Reward Calculator:\n{response.content}\n\n✅ Reward: {reward:+.1f}")],
        "action_history": [action_record]
    }


# Value Updater
def value_updater(state: ReinforcementLearningState) -> ReinforcementLearningState:
    """Updates value estimates based on received rewards"""
    action_history = state.get("action_history", [])
    value_estimates = state.get("value_estimates", {})
    episode = state.get("episode", 0)
    
    system_message = SystemMessage(content="""You are a value updater. 
    Update action value estimates using the reward received.""")
    
    # Get latest action
    latest_action = action_history[-1] if action_history else {}
    action_name = latest_action.get("action", "")
    reward = latest_action.get("reward", 0.0)
    
    user_message = HumanMessage(content=f"""Update values:

Action: {action_name}
Reward: {reward}

Use learning rate α=0.1 to update value estimate.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Update value estimate (simplified Q-learning update)
    learning_rate = 0.1
    current_value = value_estimates.get(action_name, 0.0)
    
    if isinstance(reward, (int, float)):
        new_value = current_value + learning_rate * (reward - current_value)
    else:
        new_value = current_value
    
    value_estimates[action_name] = new_value
    
    avg_value = sum(value_estimates.values()) / len(value_estimates) if value_estimates else 0
    
    return {
        "messages": [AIMessage(content=f"📈 Value Updater:\n{response.content}\n\n✅ Updated: {action_name} → {new_value:.2f}\n📊 Avg Value: {avg_value:.2f}")],
        "value_estimates": value_estimates
    }


# Policy Improver
def policy_improver(state: ReinforcementLearningState) -> ReinforcementLearningState:
    """Improves policy based on value estimates"""
    current_state = state.get("current_state", "")
    available_actions = state.get("available_actions", [])
    value_estimates = state.get("value_estimates", {})
    policy = state.get("policy", {})
    action_history = state.get("action_history", [])
    total_reward = state.get("total_reward", 0.0)
    episode = state.get("episode", 0)
    
    system_message = SystemMessage(content="""You are a policy improver. 
    Update the policy to choose actions with highest estimated values.""")
    
    user_message = HumanMessage(content=f"""Improve policy:

State: {current_state}
Value Estimates: {len(value_estimates)} actions evaluated

Select best action for this state.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Update policy: choose action with highest value
    if value_estimates:
        best_action = max(
            [a for a in available_actions if a in value_estimates],
            key=lambda a: value_estimates.get(a, 0.0),
            default=available_actions[0] if available_actions else ""
        )
        policy[current_state] = best_action
    
    # Update total reward
    latest_reward = action_history[-1].get("reward", 0.0) if action_history else 0.0
    if isinstance(latest_reward, (int, float)):
        total_reward += latest_reward
    
    return {
        "messages": [AIMessage(content=f"🎯 Policy Improver:\n{response.content}\n\n✅ Policy updated for '{current_state}'\n💰 Total Reward: {total_reward:.1f}")],
        "policy": policy,
        "total_reward": total_reward,
        "episode": episode + 1
    }


# RL Monitor
def rl_monitor(state: ReinforcementLearningState) -> ReinforcementLearningState:
    """Monitors reinforcement learning progress"""
    environment = state.get("environment", "")
    current_state = state.get("current_state", "")
    action_history = state.get("action_history", [])
    total_reward = state.get("total_reward", 0.0)
    episode = state.get("episode", 0)
    policy = state.get("policy", {})
    value_estimates = state.get("value_estimates", {})
    
    recent_actions = "\n".join([
        f"    Episode {action['episode']}: {action['action']} → Reward: {action['reward']:+.1f}"
        for action in action_history[-3:]
    ])
    
    policy_summary = "\n".join([
        f"    • {state_name}: {action}"
        for state_name, action in policy.items()
    ])
    
    top_actions = sorted(
        value_estimates.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    top_actions_summary = "\n".join([
        f"    {i+1}. {action}: {value:.2f}"
        for i, (action, value) in enumerate(top_actions)
    ])
    
    avg_reward = total_reward / episode if episode > 0 else 0
    
    summary = f"""
    ✅ REINFORCEMENT LEARNING PATTERN - Episode {episode}
    
    Learning Summary:
    • Environment: {environment}
    • Episodes Completed: {episode}
    • Total Reward: {total_reward:.1f}
    • Average Reward: {avg_reward:.1f}
    • Actions Tried: {len(value_estimates)}
    
    Current Policy:
{policy_summary if policy_summary else "    • Still learning (exploration phase)"}
    
    Top Value Actions:
{top_actions_summary if top_actions_summary else "    • No values yet"}
    
    Recent Actions:
{recent_actions if recent_actions else "    • None yet"}
    
    RL Learning Cycle:
    1. Observe State → 2. Select Action (Policy) → 3. Receive Reward → 
    4. Update Values → 5. Improve Policy → 6. Repeat
    
    Reinforcement Learning Benefits:
    • Learns from experience
    • Optimizes through rewards
    • Adapts to environment
    • Balances exploration/exploitation
    • Continuous improvement
    • Goal-oriented behavior
    
    Learning Strategy:
    • Epsilon-greedy exploration (decreasing over time)
    • Value-based action selection
    • Policy improvement through experience
    • Reward maximization
    
    Progress:
    • {len(policy)} states have optimal policies
    • {len(value_estimates)} action values learned
    • Total reward accumulated: {total_reward:.1f}
    • Average reward per episode: {avg_reward:.1f}
    """
    
    return {
        "messages": [AIMessage(content=f"📊 RL Monitor:\n{summary}")]
    }


# Build the graph
def build_rl_graph():
    """Build the reinforcement learning pattern graph"""
    workflow = StateGraph(ReinforcementLearningState)
    
    workflow.add_node("environment", environment_simulator)
    workflow.add_node("agent", policy_agent)
    workflow.add_node("reward", reward_calculator)
    workflow.add_node("value_updater", value_updater)
    workflow.add_node("policy_improver", policy_improver)
    workflow.add_node("monitor", rl_monitor)
    
    workflow.add_edge(START, "environment")
    workflow.add_edge("environment", "agent")
    workflow.add_edge("agent", "reward")
    workflow.add_edge("reward", "value_updater")
    workflow.add_edge("value_updater", "policy_improver")
    workflow.add_edge("policy_improver", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage - Multiple RL episodes
if __name__ == "__main__":
    graph = build_rl_graph()
    
    print("=== Reinforcement Learning MCP Pattern ===\n")
    
    # Initial state
    state = {
        "messages": [],
        "environment": "Customer Service Optimization",
        "current_state": "",
        "available_actions": [],
        "action_history": [],
        "total_reward": 0.0,
        "episode": 0,
        "policy": {},
        "value_estimates": {}
    }
    
    # Run multiple RL episodes
    for i in range(5):
        print(f"\n{'=' * 70}")
        print(f"REINFORCEMENT LEARNING EPISODE {i + 1}")
        print('=' * 70)
        
        result = graph.invoke(state)
        
        # Show messages for this episode
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next episode
        state = {
            "messages": [],
            "environment": state["environment"],
            "current_state": "",
            "available_actions": [],
            "action_history": result.get("action_history", []),
            "total_reward": result.get("total_reward", 0.0),
            "episode": result.get("episode", i + 1),
            "policy": result.get("policy", {}),
            "value_estimates": result.get("value_estimates", {})
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL REINFORCEMENT LEARNING RESULTS")
    print('=' * 70)
    print(f"\nTotal Episodes: {state['episode']}")
    print(f"Total Reward: {state['total_reward']:.1f}")
    print(f"Average Reward: {state['total_reward'] / state['episode']:.1f}")
    print(f"Policy Learned: {len(state['policy'])} states")
    print(f"Action Values: {len(state['value_estimates'])} actions")

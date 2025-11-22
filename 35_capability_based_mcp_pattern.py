"""
Capability-Based Distribution MCP Pattern

This pattern demonstrates distributing tasks based on agent capabilities
and skills, ensuring tasks are assigned to the most qualified agents.

Key Features:
- Skill-based task matching
- Capability-aware distribution
- Optimal agent-task alignment
- Expertise utilization
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class CapabilityDistributionState(TypedDict):
    """State for capability-based distribution pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tasks: list[dict[str, any]]  # includes required_skills field
    agent_capabilities: dict[str, list[str]]  # agent -> skill list
    task_assignments: dict[str, list[str]]  # agent -> task_ids
    completed_tasks: list[str]
    results: dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Capability Matcher
def capability_matcher(state: CapabilityDistributionState) -> CapabilityDistributionState:
    """Matches tasks to agents based on capabilities"""
    tasks = state.get("tasks", [])
    agent_capabilities = state.get("agent_capabilities", {})
    task_assignments = state.get("task_assignments", {agent: [] for agent in agent_capabilities.keys()})
    
    system_message = SystemMessage(content="""You are a capability matcher. Analyze task requirements 
    and agent skills to create optimal assignments based on expertise.""")
    
    tasks_info = "\n".join([
        f"Task {t['id']}: {t['description']} (Requires: {', '.join(t.get('required_skills', []))})"
        for t in tasks
    ])
    
    capabilities_info = "\n".join([
        f"{agent}: {', '.join(skills)}"
        for agent, skills in agent_capabilities.items()
    ])
    
    user_message = HumanMessage(content=f"""Match tasks to agents based on capabilities:
    
Tasks:
{tasks_info}

Agent Capabilities:
{capabilities_info}

Assign each task to the most qualified agent.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Match tasks to agents based on skill overlap
    for task in tasks:
        required_skills = set(task.get("required_skills", []))
        best_agent = None
        best_match_score = -1
        
        for agent, skills in agent_capabilities.items():
            agent_skills = set(skills)
            match_score = len(required_skills & agent_skills)
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_agent = agent
        
        if best_agent:
            task_assignments[best_agent].append(task['id'])
    
    assignment_summary = "\n".join([
        f"{agent}: {len(tasks)} tasks"
        for agent, tasks in task_assignments.items() if tasks
    ])
    
    return {
        "messages": [AIMessage(content=f"🎯 Capability Matcher: {response.content}\n\nAssignments:\n{assignment_summary}")],
        "task_assignments": task_assignments
    }


# Database Specialist
def database_specialist(state: CapabilityDistributionState) -> CapabilityDistributionState:
    """Handles database-related tasks"""
    assignments = state.get("task_assignments", {}).get("db_specialist", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="💾 Database Specialist: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a database specialist with expertise in 
    SQL, database design, optimization, and data migration. Handle database tasks professionally.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these database tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"DB Specialist: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"💾 Database Specialist: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Frontend Developer
def frontend_developer(state: CapabilityDistributionState) -> CapabilityDistributionState:
    """Handles frontend development tasks"""
    assignments = state.get("task_assignments", {}).get("frontend_dev", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🎨 Frontend Developer: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a frontend developer with expertise in 
    React, JavaScript, CSS, and UI/UX. Handle frontend tasks professionally.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these frontend tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Frontend Dev: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"🎨 Frontend Developer: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Backend Engineer
def backend_engineer(state: CapabilityDistributionState) -> CapabilityDistributionState:
    """Handles backend engineering tasks"""
    assignments = state.get("task_assignments", {}).get("backend_engineer", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="⚙️ Backend Engineer: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a backend engineer with expertise in 
    Python, API design, microservices, and system architecture. Handle backend tasks professionally.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these backend tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"Backend Engineer: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"⚙️ Backend Engineer: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# DevOps Engineer
def devops_engineer(state: CapabilityDistributionState) -> CapabilityDistributionState:
    """Handles DevOps tasks"""
    assignments = state.get("task_assignments", {}).get("devops_engineer", [])
    tasks = state.get("tasks", [])
    
    if not assignments:
        return {"messages": [AIMessage(content="🔧 DevOps Engineer: No tasks assigned")]}
    
    my_tasks = [t for t in tasks if t['id'] in assignments]
    
    system_message = SystemMessage(content="""You are a DevOps engineer with expertise in 
    Docker, Kubernetes, CI/CD, and cloud infrastructure. Handle DevOps tasks professionally.""")
    
    tasks_desc = "\n".join([f"- {t['description']}" for t in my_tasks])
    user_message = HumanMessage(content=f"""Process these DevOps tasks:\n{tasks_desc}""")
    
    response = llm.invoke([system_message, user_message])
    
    completed = state.get("completed_tasks", [])
    results = state.get("results", {})
    
    for task in my_tasks:
        completed.append(task['id'])
        results[task['id']] = f"DevOps Engineer: {task['description']}"
    
    return {
        "messages": [AIMessage(content=f"🔧 DevOps Engineer: {response.content}\n\nCompleted {len(my_tasks)} tasks")],
        "completed_tasks": completed,
        "results": results
    }


# Results Aggregator
def results_aggregator(state: CapabilityDistributionState) -> CapabilityDistributionState:
    """Aggregates capability-based distribution results"""
    task_assignments = state.get("task_assignments", {})
    completed = state.get("completed_tasks", [])
    
    distribution = "\n".join([
        f"  {agent}: {len(tasks)} tasks (matched by skills)"
        for agent, tasks in task_assignments.items() if tasks
    ])
    
    summary = f"""
    ✅ CAPABILITY-BASED DISTRIBUTION COMPLETE
    
    Total Tasks: {len(completed)}
    
    Task Distribution:
{distribution}
    
    Each task assigned to agent with best matching capabilities.
    Skills and expertise optimally utilized.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Results Aggregator:\n{summary}")]
    }


# Build the graph
def build_capability_distribution_graph():
    """Build the capability-based distribution MCP pattern graph"""
    workflow = StateGraph(CapabilityDistributionState)
    
    workflow.add_node("matcher", capability_matcher)
    workflow.add_node("db_specialist", database_specialist)
    workflow.add_node("frontend_dev", frontend_developer)
    workflow.add_node("backend_engineer", backend_engineer)
    workflow.add_node("devops_engineer", devops_engineer)
    workflow.add_node("aggregator", results_aggregator)
    
    workflow.add_edge(START, "matcher")
    workflow.add_edge("matcher", "db_specialist")
    workflow.add_edge("matcher", "frontend_dev")
    workflow.add_edge("matcher", "backend_engineer")
    workflow.add_edge("matcher", "devops_engineer")
    workflow.add_edge("db_specialist", "aggregator")
    workflow.add_edge("frontend_dev", "aggregator")
    workflow.add_edge("backend_engineer", "aggregator")
    workflow.add_edge("devops_engineer", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_capability_distribution_graph()
    
    print("=== Capability-Based Distribution MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "tasks": [
            {"id": "task_1", "description": "Optimize database queries", "required_skills": ["SQL", "database"]},
            {"id": "task_2", "description": "Build React dashboard", "required_skills": ["React", "JavaScript"]},
            {"id": "task_3", "description": "Design REST API", "required_skills": ["Python", "API"]},
            {"id": "task_4", "description": "Setup CI/CD pipeline", "required_skills": ["Docker", "CI/CD"]},
            {"id": "task_5", "description": "Migrate database schema", "required_skills": ["SQL", "database"]},
            {"id": "task_6", "description": "Implement responsive UI", "required_skills": ["CSS", "React"]},
            {"id": "task_7", "description": "Configure Kubernetes cluster", "required_skills": ["Kubernetes", "Docker"]},
            {"id": "task_8", "description": "Build microservices", "required_skills": ["Python", "API"]},
        ],
        "agent_capabilities": {
            "db_specialist": ["SQL", "database", "optimization"],
            "frontend_dev": ["React", "JavaScript", "CSS", "UI/UX"],
            "backend_engineer": ["Python", "API", "microservices"],
            "devops_engineer": ["Docker", "Kubernetes", "CI/CD", "cloud"]
        },
        "task_assignments": {
            "db_specialist": [],
            "frontend_dev": [],
            "backend_engineer": [],
            "devops_engineer": []
        },
        "completed_tasks": [],
        "results": {}
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Capability-Based Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)

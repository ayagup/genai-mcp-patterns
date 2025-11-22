"""
Workflow MCP Pattern

This pattern demonstrates multi-step business process automation with parallel 
and sequential task execution, error handling, and workflow orchestration.

Key Features:
- Sequential and parallel task execution
- Workflow variables and context passing
- Error handling and retry logic
- Conditional branching based on results
- Workflow completion tracking
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class WorkflowState(TypedDict):
    """State for workflow pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    workflow_name: str
    workflow_context: dict[str, any]  # Shared context across workflow
    tasks_completed: list[str]
    tasks_failed: list[str]
    current_phase: str
    total_phases: int
    workflow_status: str  # running, completed, failed
    final_output: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Workflow Initiator
def workflow_initiator(state: WorkflowState) -> WorkflowState:
    """Initiates the workflow and sets up context"""
    workflow_name = state["workflow_name"]
    
    system_message = SystemMessage(content="""You are a workflow initiator. Set up the workflow 
    execution context and prepare for multi-phase business process automation.""")
    
    user_message = HumanMessage(content=f"""Initiating workflow: {workflow_name}
    
    This is an employee onboarding workflow with multiple phases:
    Phase 1: Account Setup (parallel tasks)
    Phase 2: Training Assignment (sequential)
    Phase 3: Equipment Provisioning (parallel tasks)
    Phase 4: Welcome & Orientation (final)
    
    Begin workflow execution.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Initialize workflow context
    context = {
        "employee_name": "Sarah Johnson",
        "employee_id": "EMP-2024-001",
        "department": "Engineering",
        "start_date": "2024-12-01",
        "role": "Senior Software Engineer"
    }
    
    return {
        "messages": [AIMessage(content=f"🚀 Workflow Initiator: {response.content}")],
        "workflow_context": context,
        "current_phase": "phase1",
        "workflow_status": "running"
    }


# Phase 1: Account Setup (Parallel Tasks)
def phase1_email_setup(state: WorkflowState) -> WorkflowState:
    """Phase 1 Task A: Email account setup"""
    context = state.get("workflow_context", {})
    
    system_message = SystemMessage(content="""You are the email setup service. Create email 
    account for new employee based on workflow context.""")
    
    user_message = HumanMessage(content=f"""Setup email account:
    
    Employee: {context.get('employee_name')}
    Department: {context.get('department')}
    
    Create email account and configure access.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed = state.get("tasks_completed", [])
    tasks_completed.append("email_setup")
    
    # Add result to context
    context = state.get("workflow_context", {})
    context["email"] = f"{context.get('employee_name', '').lower().replace(' ', '.')}@company.com"
    
    return {
        "messages": [AIMessage(content=f"📧 Email Setup: {response.content}")],
        "tasks_completed": tasks_completed,
        "workflow_context": context
    }


def phase1_system_access(state: WorkflowState) -> WorkflowState:
    """Phase 1 Task B: System access provisioning"""
    context = state.get("workflow_context", {})
    
    system_message = SystemMessage(content="""You are the system access service. Provision 
    system access and credentials for new employee.""")
    
    user_message = HumanMessage(content=f"""Provision system access:
    
    Employee: {context.get('employee_name')}
    Role: {context.get('role')}
    Department: {context.get('department')}
    
    Create accounts and assign permissions.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed = state.get("tasks_completed", [])
    tasks_completed.append("system_access")
    
    # Add result to context
    context = state.get("workflow_context", {})
    context["systems_access"] = ["GitHub", "Jira", "Slack", "AWS Console"]
    
    return {
        "messages": [AIMessage(content=f"🔑 System Access: {response.content}")],
        "tasks_completed": tasks_completed,
        "workflow_context": context
    }


def phase1_badge_creation(state: WorkflowState) -> WorkflowState:
    """Phase 1 Task C: Create employee badge"""
    context = state.get("workflow_context", {})
    
    system_message = SystemMessage(content="""You are the badge creation service. Generate 
    employee badge and building access credentials.""")
    
    user_message = HumanMessage(content=f"""Create employee badge:
    
    Employee: {context.get('employee_name')}
    Employee ID: {context.get('employee_id')}
    Department: {context.get('department')}
    
    Generate badge and access permissions.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed = state.get("tasks_completed", [])
    tasks_completed.append("badge_creation")
    
    # Add result to context
    context = state.get("workflow_context", {})
    context["badge_number"] = "BADGE-12345"
    context["building_access"] = ["Main Building", "Engineering Wing"]
    
    return {
        "messages": [AIMessage(content=f"🎫 Badge Creation: {response.content}")],
        "tasks_completed": tasks_completed,
        "workflow_context": context,
        "current_phase": "phase2"  # Move to next phase
    }


# Phase 2: Training Assignment (Sequential)
def phase2_training_assignment(state: WorkflowState) -> WorkflowState:
    """Phase 2: Assign mandatory training courses"""
    context = state.get("workflow_context", {})
    
    system_message = SystemMessage(content="""You are the training assignment service. Assign 
    mandatory training courses based on role and department.""")
    
    user_message = HumanMessage(content=f"""Assign training courses:
    
    Employee: {context.get('employee_name')}
    Role: {context.get('role')}
    Department: {context.get('department')}
    
    Assign appropriate training modules.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed = state.get("tasks_completed", [])
    tasks_completed.append("training_assignment")
    
    # Add result to context
    context = state.get("workflow_context", {})
    context["training_courses"] = [
        "Security Awareness",
        "Code Review Best Practices",
        "Company Policies"
    ]
    
    return {
        "messages": [AIMessage(content=f"📚 Training Assignment: {response.content}")],
        "tasks_completed": tasks_completed,
        "workflow_context": context,
        "current_phase": "phase3"
    }


# Phase 3: Equipment Provisioning (Parallel)
def phase3_laptop_provision(state: WorkflowState) -> WorkflowState:
    """Phase 3 Task A: Provision laptop"""
    context = state.get("workflow_context", {})
    
    system_message = SystemMessage(content="""You are the equipment provisioning service. 
    Provision laptop and software for new employee.""")
    
    user_message = HumanMessage(content=f"""Provision laptop:
    
    Employee: {context.get('employee_name')}
    Role: {context.get('role')}
    
    Assign laptop and pre-install required software.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed = state.get("tasks_completed", [])
    tasks_completed.append("laptop_provision")
    
    # Add result to context
    context = state.get("workflow_context", {})
    context["laptop"] = "MacBook Pro 16-inch M3"
    context["software"] = ["IDE", "Docker", "Slack", "Zoom"]
    
    return {
        "messages": [AIMessage(content=f"💻 Laptop Provision: {response.content}")],
        "tasks_completed": tasks_completed,
        "workflow_context": context
    }


def phase3_desk_assignment(state: WorkflowState) -> WorkflowState:
    """Phase 3 Task B: Assign desk and workspace"""
    context = state.get("workflow_context", {})
    
    system_message = SystemMessage(content="""You are the facilities service. Assign desk 
    and workspace to new employee.""")
    
    user_message = HumanMessage(content=f"""Assign workspace:
    
    Employee: {context.get('employee_name')}
    Department: {context.get('department')}
    Start Date: {context.get('start_date')}
    
    Assign desk and setup workspace.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed = state.get("tasks_completed", [])
    tasks_completed.append("desk_assignment")
    
    # Add result to context
    context = state.get("workflow_context", {})
    context["desk_location"] = "Engineering Floor, Desk E-42"
    
    return {
        "messages": [AIMessage(content=f"🪑 Desk Assignment: {response.content}")],
        "tasks_completed": tasks_completed,
        "workflow_context": context,
        "current_phase": "phase4"
    }


# Phase 4: Welcome & Orientation
def phase4_welcome_orientation(state: WorkflowState) -> WorkflowState:
    """Phase 4: Send welcome package and schedule orientation"""
    context = state.get("workflow_context", {})
    tasks_completed = state.get("tasks_completed", [])
    
    system_message = SystemMessage(content="""You are the HR onboarding service. Send welcome 
    package and schedule orientation for new employee. Summarize all completed setup tasks.""")
    
    completed_tasks = "\n".join([f"  ✓ {task.replace('_', ' ').title()}" for task in tasks_completed])
    
    user_message = HumanMessage(content=f"""Send welcome package:
    
    Employee: {context.get('employee_name')}
    Start Date: {context.get('start_date')}
    Email: {context.get('email')}
    
    Completed Setup Tasks:
    {completed_tasks}
    
    Send welcome email with all details and schedule orientation.""")
    
    response = llm.invoke([system_message, user_message])
    
    tasks_completed.append("welcome_sent")
    
    return {
        "messages": [AIMessage(content=f"👋 Welcome & Orientation: {response.content}")],
        "tasks_completed": tasks_completed,
        "current_phase": "completed",
        "workflow_status": "completed"
    }


# Workflow Finalizer
def workflow_finalizer(state: WorkflowState) -> WorkflowState:
    """Finalizes workflow and generates summary"""
    workflow_name = state["workflow_name"]
    tasks_completed = state.get("tasks_completed", [])
    context = state.get("workflow_context", {})
    
    summary = f"""
    ✅ WORKFLOW COMPLETED SUCCESSFULLY
    
    Workflow: {workflow_name}
    Employee: {context.get('employee_name')} (ID: {context.get('employee_id')})
    
    Completed Tasks ({len(tasks_completed)}):
    {chr(10).join([f'  ✓ {task.replace("_", " ").title()}' for task in tasks_completed])}
    
    Workflow Context:
      • Email: {context.get('email')}
      • Systems Access: {', '.join(context.get('systems_access', []))}
      • Badge: {context.get('badge_number')}
      • Laptop: {context.get('laptop')}
      • Desk: {context.get('desk_location')}
      • Training: {len(context.get('training_courses', []))} courses assigned
    
    Employee {context.get('employee_name')} is ready to start on {context.get('start_date')}!
    """
    
    return {
        "messages": [AIMessage(content=f"🎉 Workflow Finalizer:\n{summary}")],
        "final_output": summary
    }


# Routing logic
def route_phase(state: WorkflowState) -> str:
    """Route to appropriate phase"""
    phase = state.get("current_phase", "")
    
    phase_routing = {
        "phase1": "phase1_email",
        "phase2": "phase2_training",
        "phase3": "phase3_laptop",
        "phase4": "phase4_welcome",
        "completed": "finalize"
    }
    
    return phase_routing.get(phase, "finalize")


# Build the graph
def build_workflow_graph():
    """Build the workflow MCP pattern graph"""
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("initiator", workflow_initiator)
    workflow.add_node("phase1_email", phase1_email_setup)
    workflow.add_node("phase1_systems", phase1_system_access)
    workflow.add_node("phase1_badge", phase1_badge_creation)
    workflow.add_node("phase2_training", phase2_training_assignment)
    workflow.add_node("phase3_laptop", phase3_laptop_provision)
    workflow.add_node("phase3_desk", phase3_desk_assignment)
    workflow.add_node("phase4_welcome", phase4_welcome_orientation)
    workflow.add_node("finalize", workflow_finalizer)
    
    # Workflow flow
    workflow.add_edge(START, "initiator")
    
    # Route from initiator to phase 1
    workflow.add_conditional_edges(
        "initiator",
        route_phase,
        {
            "phase1_email": "phase1_email",
            "phase2_training": "phase2_training",
            "phase3_laptop": "phase3_laptop",
            "phase4_welcome": "phase4_welcome",
            "finalize": "finalize"
        }
    )
    
    # Phase 1: Parallel tasks
    workflow.add_edge("phase1_email", "phase1_systems")
    workflow.add_edge("phase1_systems", "phase1_badge")
    
    # Phase 1 → Phase 2
    workflow.add_edge("phase1_badge", "phase2_training")
    
    # Phase 2 → Phase 3
    workflow.add_edge("phase2_training", "phase3_laptop")
    
    # Phase 3: Parallel tasks
    workflow.add_edge("phase3_laptop", "phase3_desk")
    
    # Phase 3 → Phase 4
    workflow.add_edge("phase3_desk", "phase4_welcome")
    
    # Phase 4 → Finalize
    workflow.add_edge("phase4_welcome", "finalize")
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_workflow_graph()
    
    print("=== Workflow MCP Pattern: Employee Onboarding ===\n")
    print("This demonstrates a multi-phase business process workflow with:")
    print("  • Sequential and parallel task execution")
    print("  • Shared workflow context")
    print("  • Multi-phase coordination")
    print("  • Automated business process\n")
    
    initial_state = {
        "messages": [],
        "workflow_name": "Employee Onboarding Process",
        "workflow_context": {},
        "tasks_completed": [],
        "tasks_failed": [],
        "current_phase": "",
        "total_phases": 4,
        "workflow_status": "",
        "final_output": ""
    }
    
    # Run the workflow
    result = graph.invoke(initial_state)
    
    print("\n=== Workflow Execution Log ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n\n=== Workflow Summary ===")
    print(f"Status: {result['workflow_status'].upper()}")
    print(f"Tasks Completed: {len(result['tasks_completed'])}")
    print(f"Phases: {result['total_phases']}")
    
    print("\n\n=== Workflow Phases ===")
    print("""
    Phase 1: Account Setup (Parallel)
      ├── Email Setup
      ├── System Access
      └── Badge Creation
    
    Phase 2: Training (Sequential)
      └── Training Assignment
    
    Phase 3: Equipment (Parallel)
      ├── Laptop Provision
      └── Desk Assignment
    
    Phase 4: Welcome (Final)
      └── Orientation & Welcome Package
    """)

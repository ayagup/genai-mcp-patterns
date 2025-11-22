"""
Orchestration MCP Pattern

This pattern demonstrates centralized workflow coordination where an orchestrator 
agent explicitly controls the execution flow and delegates tasks to worker agents.

Key Features:
- Central orchestrator controls workflow
- Explicit task delegation and sequencing
- Orchestrator maintains workflow state
- Worker agents execute assigned tasks
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class OrchestrationState(TypedDict):
    """State for orchestration pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    workflow_name: str
    current_step: int
    total_steps: int
    step_results: dict[str, str]  # step_name -> result
    orchestrator_decisions: list[str]
    workflow_complete: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Orchestrator Agent
def orchestrator_agent(state: OrchestrationState) -> OrchestrationState:
    """Central orchestrator that coordinates the workflow"""
    workflow_name = state["workflow_name"]
    current_step = state.get("current_step", 0)
    step_results = state.get("step_results", {})
    
    if current_step == 0:
        # Initiate workflow
        system_message = SystemMessage(content="""You are a workflow orchestrator. You coordinate 
        and manage the execution of multi-step workflows by delegating tasks to specialized workers 
        and making decisions based on their results.""")
        
        user_message = HumanMessage(content=f"""Initiating workflow: {workflow_name}
        
        You will orchestrate a multi-step content creation workflow:
        1. Research topic
        2. Create outline
        3. Write content
        4. Review and edit
        5. Format and publish
        
        Begin orchestration.""")
        
        response = llm.invoke([system_message, user_message])
        
        return {
            "messages": [AIMessage(content=f"🎯 Orchestrator (INIT): {response.content}")],
            "current_step": 1
        }
    
    else:
        # Make decisions between steps
        system_message = SystemMessage(content="""You are the orchestrator reviewing step results 
        and deciding next actions. Analyze the previous step's output and provide direction for 
        the next step.""")
        
        previous_results = "\n".join([f"{k}: {v}" for k, v in step_results.items()])
        
        user_message = HumanMessage(content=f"""Step {current_step - 1} completed.
        
        Results so far:
        {previous_results}
        
        Analyze results and provide guidance for step {current_step}.""")
        
        response = llm.invoke([system_message, user_message])
        
        decisions = state.get("orchestrator_decisions", [])
        decisions.append(response.content)
        
        return {
            "messages": [AIMessage(content=f"🎯 Orchestrator (STEP {current_step}): {response.content}")],
            "orchestrator_decisions": decisions
        }


# Worker 1: Research Agent
def research_worker(state: OrchestrationState) -> OrchestrationState:
    """Worker agent that performs research"""
    
    system_message = SystemMessage(content="""You are a research worker agent. When the 
    orchestrator assigns you a research task, you gather information and provide comprehensive 
    research findings.""")
    
    user_message = HumanMessage(content="""Orchestrator has assigned you to research:
    
    Topic: "Benefits and challenges of implementing microservices architecture"
    
    Conduct research and provide key findings.""")
    
    response = llm.invoke([system_message, user_message])
    
    step_results = state.get("step_results", {})
    step_results["research"] = response.content
    
    return {
        "messages": [AIMessage(content=f"🔍 Research Worker: {response.content}")],
        "step_results": step_results,
        "current_step": 2
    }


# Worker 2: Outline Agent
def outline_worker(state: OrchestrationState) -> OrchestrationState:
    """Worker agent that creates content outlines"""
    step_results = state.get("step_results", {})
    research_findings = step_results.get("research", "")
    
    system_message = SystemMessage(content="""You are an outline worker agent. Based on research 
    findings provided by the orchestrator, you create a structured outline for content.""")
    
    user_message = HumanMessage(content=f"""Orchestrator has assigned you to create an outline.
    
    Based on research findings:
    {research_findings[:500]}...
    
    Create a detailed content outline.""")
    
    response = llm.invoke([system_message, user_message])
    
    step_results["outline"] = response.content
    
    return {
        "messages": [AIMessage(content=f"📋 Outline Worker: {response.content}")],
        "step_results": step_results,
        "current_step": 3
    }


# Worker 3: Writing Agent
def writing_worker(state: OrchestrationState) -> OrchestrationState:
    """Worker agent that writes content"""
    step_results = state.get("step_results", {})
    outline = step_results.get("outline", "")
    
    system_message = SystemMessage(content="""You are a writing worker agent. Based on the 
    outline provided by the orchestrator, you write comprehensive, well-structured content.""")
    
    user_message = HumanMessage(content=f"""Orchestrator has assigned you to write content.
    
    Based on outline:
    {outline[:500]}...
    
    Write the first section of the content (introduction and overview).""")
    
    response = llm.invoke([system_message, user_message])
    
    step_results["content"] = response.content
    
    return {
        "messages": [AIMessage(content=f"✍️ Writing Worker: {response.content}")],
        "step_results": step_results,
        "current_step": 4
    }


# Worker 4: Review Agent
def review_worker(state: OrchestrationState) -> OrchestrationState:
    """Worker agent that reviews and edits content"""
    step_results = state.get("step_results", {})
    content = step_results.get("content", "")
    
    system_message = SystemMessage(content="""You are a review worker agent. You review content 
    for quality, clarity, grammar, and coherence. Provide feedback and suggest improvements.""")
    
    user_message = HumanMessage(content=f"""Orchestrator has assigned you to review content.
    
    Content to review:
    {content[:500]}...
    
    Provide review feedback and suggested edits.""")
    
    response = llm.invoke([system_message, user_message])
    
    step_results["review"] = response.content
    
    return {
        "messages": [AIMessage(content=f"👁️ Review Worker: {response.content}")],
        "step_results": step_results,
        "current_step": 5
    }


# Worker 5: Publishing Agent
def publishing_worker(state: OrchestrationState) -> OrchestrationState:
    """Worker agent that formats and publishes content"""
    step_results = state.get("step_results", {})
    
    system_message = SystemMessage(content="""You are a publishing worker agent. You format 
    content according to publishing guidelines and prepare it for publication.""")
    
    user_message = HumanMessage(content=f"""Orchestrator has assigned you to publish content.
    
    All previous steps completed:
    - Research: ✓
    - Outline: ✓
    - Content: ✓
    - Review: ✓
    
    Format and prepare for publication.""")
    
    response = llm.invoke([system_message, user_message])
    
    step_results["published"] = response.content
    
    return {
        "messages": [AIMessage(content=f"📤 Publishing Worker: {response.content}")],
        "step_results": step_results,
        "current_step": 6,
        "workflow_complete": True
    }


# Finalizer
def orchestrator_finalizer(state: OrchestrationState) -> OrchestrationState:
    """Orchestrator provides final workflow summary"""
    step_results = state.get("step_results", {})
    workflow_name = state["workflow_name"]
    
    summary = f"""
    ✅ WORKFLOW ORCHESTRATION COMPLETE
    
    Workflow: {workflow_name}
    Total Steps: {state['total_steps']}
    Steps Completed: {len(step_results)}
    
    Step Results:
    {chr(10).join([f'  ✓ {step.title()}: Completed' for step in step_results.keys()])}
    
    All workers successfully executed their assigned tasks under orchestrator coordination.
    Workflow completed successfully!
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Orchestrator (FINAL):\n{summary}")]
    }


# Routing logic
def route_next_step(state: OrchestrationState) -> str:
    """Route to next step based on current step"""
    current_step = state.get("current_step", 0)
    
    step_routing = {
        1: "research",
        2: "outline",
        3: "writing",
        4: "review",
        5: "publishing",
        6: "finalize"
    }
    
    return step_routing.get(current_step, "finalize")


# Build the graph
def build_orchestration_graph():
    """Build the orchestration MCP pattern graph"""
    workflow = StateGraph(OrchestrationState)
    
    # Add nodes
    workflow.add_node("orchestrator", orchestrator_agent)
    workflow.add_node("research", research_worker)
    workflow.add_node("outline", outline_worker)
    workflow.add_node("writing", writing_worker)
    workflow.add_node("review", review_worker)
    workflow.add_node("publishing", publishing_worker)
    workflow.add_node("finalize", orchestrator_finalizer)
    
    # Orchestrated flow
    workflow.add_edge(START, "orchestrator")
    
    # After orchestrator, route to appropriate worker
    workflow.add_conditional_edges(
        "orchestrator",
        route_next_step,
        {
            "research": "research",
            "outline": "outline",
            "writing": "writing",
            "review": "review",
            "publishing": "publishing",
            "finalize": "finalize"
        }
    )
    
    # Each worker reports back to orchestrator (except publishing)
    workflow.add_edge("research", "orchestrator")
    workflow.add_edge("outline", "orchestrator")
    workflow.add_edge("writing", "orchestrator")
    workflow.add_edge("review", "orchestrator")
    workflow.add_edge("publishing", "finalize")
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_orchestration_graph()
    
    print("=== Orchestration MCP Pattern: Content Creation Workflow ===\n")
    print("This demonstrates centralized orchestration where the orchestrator")
    print("explicitly coordinates each step and delegates to specialized workers.\n")
    
    initial_state = {
        "messages": [],
        "workflow_name": "Technical Article Creation",
        "current_step": 0,
        "total_steps": 5,
        "step_results": {},
        "orchestrator_decisions": [],
        "workflow_complete": False
    }
    
    # Run the orchestrated workflow
    result = graph.invoke(initial_state)
    
    print("\n=== Orchestrated Workflow Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n\n=== Workflow Summary ===")
    print(f"Workflow: {result['workflow_name']}")
    print(f"Total Steps: {result['total_steps']}")
    print(f"Completed: {result['workflow_complete']}")
    print(f"\nSteps Executed:")
    for step_name in result['step_results'].keys():
        print(f"  ✓ {step_name.title()}")

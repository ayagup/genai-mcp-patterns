"""
Blackboard MCP Pattern

This pattern demonstrates a shared knowledge space (blackboard) where multiple
specialist agents contribute their expertise to solve complex problems.

Key Features:
- Centralized blackboard for knowledge sharing
- Specialist agents contribute domain knowledge
- Incremental problem solving
- Opportunistic reasoning
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state with blackboard
class BlackboardState(TypedDict):
    """State with blackboard knowledge space"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    blackboard: dict[str, any]  # Shared blackboard
    problem: str
    solution: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Problem Definition Agent
def problem_definer(state: BlackboardState) -> BlackboardState:
    """Defines the problem on the blackboard"""
    problem = state.get("problem", "")
    blackboard = state.get("blackboard", {})
    
    system_message = SystemMessage(content="""You are a problem definer. Analyze the problem 
    and break it down into components on the blackboard.""")
    
    user_message = HumanMessage(content=f"""Define and structure this problem:

{problem}

Write problem structure to the blackboard.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write to blackboard
    blackboard["problem_definition"] = {
        "original": problem,
        "structured": response.content,
        "components": ["requirements", "constraints", "objectives"],
        "contributor": "problem_definer"
    }
    
    return {
        "messages": [AIMessage(content=f"📝 Problem Definer: {response.content}\n\n✅ Problem definition posted to blackboard")],
        "blackboard": blackboard
    }


# Domain Expert 1: Technical Specialist
def technical_specialist(state: BlackboardState) -> BlackboardState:
    """Contributes technical expertise to blackboard"""
    blackboard = state.get("blackboard", {})
    problem_def = blackboard.get("problem_definition", {})
    
    system_message = SystemMessage(content="""You are a technical specialist. Contribute 
    technical insights and solutions to the blackboard.""")
    
    user_message = HumanMessage(content=f"""Provide technical expertise for:

{problem_def.get('structured', 'No problem defined')}

Write your technical insights to the blackboard.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write to blackboard
    blackboard["technical_insights"] = {
        "insights": response.content,
        "technologies": ["Python", "LangChain", "API"],
        "recommendations": "See content",
        "contributor": "technical_specialist"
    }
    
    return {
        "messages": [AIMessage(content=f"💻 Technical Specialist: {response.content}\n\n✅ Technical insights posted to blackboard")],
        "blackboard": blackboard
    }


# Domain Expert 2: Business Analyst
def business_analyst(state: BlackboardState) -> BlackboardState:
    """Contributes business perspective to blackboard"""
    blackboard = state.get("blackboard", {})
    problem_def = blackboard.get("problem_definition", {})
    tech_insights = blackboard.get("technical_insights", {})
    
    system_message = SystemMessage(content="""You are a business analyst. Contribute 
    business requirements and constraints to the blackboard.""")
    
    user_message = HumanMessage(content=f"""Provide business analysis for:

Problem: {problem_def.get('structured', 'No problem')}
Technical Insights: {tech_insights.get('insights', 'No insights yet')}

Write your business analysis to the blackboard.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write to blackboard
    blackboard["business_analysis"] = {
        "analysis": response.content,
        "requirements": ["ROI", "Scalability", "User Experience"],
        "constraints": ["Budget", "Timeline"],
        "contributor": "business_analyst"
    }
    
    return {
        "messages": [AIMessage(content=f"📊 Business Analyst: {response.content}\n\n✅ Business analysis posted to blackboard")],
        "blackboard": blackboard
    }


# Domain Expert 3: UX Designer
def ux_designer(state: BlackboardState) -> BlackboardState:
    """Contributes UX perspective to blackboard"""
    blackboard = state.get("blackboard", {})
    business_analysis = blackboard.get("business_analysis", {})
    
    system_message = SystemMessage(content="""You are a UX designer. Contribute 
    user experience insights to the blackboard.""")
    
    user_message = HumanMessage(content=f"""Provide UX design insights for:

Business Requirements: {business_analysis.get('analysis', 'No analysis yet')}

Write your UX recommendations to the blackboard.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write to blackboard
    blackboard["ux_insights"] = {
        "insights": response.content,
        "principles": ["Simplicity", "Accessibility", "Consistency"],
        "wireframes": "Described in content",
        "contributor": "ux_designer"
    }
    
    return {
        "messages": [AIMessage(content=f"🎨 UX Designer: {response.content}\n\n✅ UX insights posted to blackboard")],
        "blackboard": blackboard
    }


# Solution Integrator
def solution_integrator(state: BlackboardState) -> BlackboardState:
    """Integrates all contributions from blackboard into cohesive solution"""
    blackboard = state.get("blackboard", {})
    
    system_message = SystemMessage(content="""You are a solution integrator. Read all 
    contributions from the blackboard and synthesize them into a comprehensive solution.""")
    
    contributions = "\n\n".join([
        f"{key}:\n{value.get('insights', value.get('analysis', value.get('structured', str(value))))}"
        for key, value in blackboard.items()
    ])
    
    user_message = HumanMessage(content=f"""Integrate these contributions into a solution:

{contributions}

Create a comprehensive, unified solution.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Write final solution to blackboard
    blackboard["integrated_solution"] = {
        "solution": response.content,
        "contributors": list(blackboard.keys()),
        "integration_status": "complete"
    }
    
    return {
        "messages": [AIMessage(content=f"🔗 Solution Integrator: {response.content}\n\n✅ Integrated solution posted to blackboard")],
        "blackboard": blackboard,
        "solution": response.content
    }


# Blackboard Monitor
def blackboard_monitor(state: BlackboardState) -> BlackboardState:
    """Monitors and reports blackboard state"""
    blackboard = state.get("blackboard", {})
    
    blackboard_summary = "📋 BLACKBOARD CONTENTS:\n\n"
    for key, value in blackboard.items():
        blackboard_summary += f"  {key}:\n"
        blackboard_summary += f"    Contributor: {value.get('contributor', 'unknown')}\n"
        blackboard_summary += "\n"
    
    final_summary = f"""
    ✅ BLACKBOARD PATTERN COMPLETE
    
    Total Contributions: {len(blackboard)}
    Expert Agents: {len([v for v in blackboard.values() if 'contributor' in v])}
    
{blackboard_summary}
    
    Blackboard Benefits:
    • Multiple specialists contribute expertise
    • Incremental problem solving
    • Shared knowledge space
    • Opportunistic reasoning from diverse perspectives
    • Integrated solution from partial contributions
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Blackboard Monitor:\n{final_summary}")]
    }


# Build the graph
def build_blackboard_graph():
    """Build the blackboard MCP pattern graph"""
    workflow = StateGraph(BlackboardState)
    
    workflow.add_node("problem_definer", problem_definer)
    workflow.add_node("technical", technical_specialist)
    workflow.add_node("business", business_analyst)
    workflow.add_node("ux", ux_designer)
    workflow.add_node("integrator", solution_integrator)
    workflow.add_node("monitor", blackboard_monitor)
    
    # Sequential contributions to blackboard
    workflow.add_edge(START, "problem_definer")
    workflow.add_edge("problem_definer", "technical")
    workflow.add_edge("technical", "business")
    workflow.add_edge("business", "ux")
    workflow.add_edge("ux", "integrator")
    workflow.add_edge("integrator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_blackboard_graph()
    
    print("=== Blackboard MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "blackboard": {},
        "problem": "Design and implement a customer support chatbot that handles inquiries, escalates complex issues, and learns from interactions",
        "solution": ""
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Blackboard Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Integrated Solution ===")
    print(result.get("solution", "No solution generated"))

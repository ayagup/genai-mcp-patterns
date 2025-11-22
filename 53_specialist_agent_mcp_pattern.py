"""
Specialist Agent MCP Pattern

This pattern demonstrates using domain-specific specialist agents
that excel in particular areas of expertise.

Key Features:
- Domain expertise specialization
- Capability matching
- Multi-specialist collaboration
- Quality assurance per domain
- Expert knowledge application
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SpecialistState(TypedDict):
    """State for specialist agent pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    problem: str
    required_specialists: list[str]
    specialist_contributions: dict[str, str]
    integrated_solution: str
    quality_score: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Problem Analyzer
def problem_analyzer(state: SpecialistState) -> SpecialistState:
    """Analyzes problem to identify required specialists"""
    problem = state.get("problem", "")
    
    system_message = SystemMessage(content="""You are a problem analyzer. Identify 
    which domain specialists are needed to solve the given problem.""")
    
    user_message = HumanMessage(content=f"""Analyze problem: {problem}

Identify required specialist expertise areas.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Identify required specialists based on problem keywords
    problem_lower = problem.lower()
    required_specialists = []
    
    if any(word in problem_lower for word in ["security", "authentication", "encryption"]):
        required_specialists.append("security_specialist")
    if any(word in problem_lower for word in ["database", "sql", "data model", "schema"]):
        required_specialists.append("database_specialist")
    if any(word in problem_lower for word in ["ui", "user interface", "ux", "design"]):
        required_specialists.append("ux_specialist")
    if any(word in problem_lower for word in ["performance", "optimize", "scalability"]):
        required_specialists.append("performance_specialist")
    
    # Default to architecture specialist if no specific match
    if not required_specialists:
        required_specialists.append("architecture_specialist")
    
    return {
        "messages": [AIMessage(content=f"🔍 Problem Analyzer: {response.content}\n\n✅ Required specialists: {', '.join(required_specialists)}")],
        "required_specialists": required_specialists
    }


# Security Specialist
def security_specialist(state: SpecialistState) -> SpecialistState:
    """Expert in security, authentication, and encryption"""
    problem = state.get("problem", "")
    required_specialists = state.get("required_specialists", [])
    
    if "security_specialist" not in required_specialists:
        return {"messages": [AIMessage(content="⏭️ Security Specialist: Not required")]}
    
    system_message = SystemMessage(content="""You are a security specialist with deep 
    expertise in authentication, authorization, encryption, and security best practices.""")
    
    user_message = HumanMessage(content=f"""Provide security expertise for: {problem}

Focus on:
- Authentication mechanisms
- Authorization strategies
- Data encryption
- Security vulnerabilities
- Best practices""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🔒 Security Specialist: {response.content}")],
        "specialist_contributions": {"security": response.content}
    }


# Database Specialist
def database_specialist(state: SpecialistState) -> SpecialistState:
    """Expert in database design and optimization"""
    problem = state.get("problem", "")
    required_specialists = state.get("required_specialists", [])
    
    if "database_specialist" not in required_specialists:
        return {"messages": [AIMessage(content="⏭️ Database Specialist: Not required")]}
    
    system_message = SystemMessage(content="""You are a database specialist with deep 
    expertise in database design, normalization, indexing, and query optimization.""")
    
    user_message = HumanMessage(content=f"""Provide database expertise for: {problem}

Focus on:
- Schema design
- Normalization
- Indexing strategies
- Query optimization
- Data integrity""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🗄️ Database Specialist: {response.content}")],
        "specialist_contributions": {"database": response.content}
    }


# UX Specialist
def ux_specialist(state: SpecialistState) -> SpecialistState:
    """Expert in user experience and interface design"""
    problem = state.get("problem", "")
    required_specialists = state.get("required_specialists", [])
    
    if "ux_specialist" not in required_specialists:
        return {"messages": [AIMessage(content="⏭️ UX Specialist: Not required")]}
    
    system_message = SystemMessage(content="""You are a UX specialist with deep expertise 
    in user interface design, usability, and user experience best practices.""")
    
    user_message = HumanMessage(content=f"""Provide UX expertise for: {problem}

Focus on:
- User interface design
- Usability principles
- Accessibility
- User flow
- Design patterns""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🎨 UX Specialist: {response.content}")],
        "specialist_contributions": {"ux": response.content}
    }


# Performance Specialist
def performance_specialist(state: SpecialistState) -> SpecialistState:
    """Expert in performance optimization and scalability"""
    problem = state.get("problem", "")
    required_specialists = state.get("required_specialists", [])
    
    if "performance_specialist" not in required_specialists:
        return {"messages": [AIMessage(content="⏭️ Performance Specialist: Not required")]}
    
    system_message = SystemMessage(content="""You are a performance specialist with deep 
    expertise in optimization, caching, load balancing, and scalability.""")
    
    user_message = HumanMessage(content=f"""Provide performance expertise for: {problem}

Focus on:
- Performance optimization
- Caching strategies
- Load balancing
- Scalability patterns
- Bottleneck identification""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"⚡ Performance Specialist: {response.content}")],
        "specialist_contributions": {"performance": response.content}
    }


# Architecture Specialist
def architecture_specialist(state: SpecialistState) -> SpecialistState:
    """Expert in system architecture and design patterns"""
    problem = state.get("problem", "")
    required_specialists = state.get("required_specialists", [])
    
    if "architecture_specialist" not in required_specialists:
        return {"messages": [AIMessage(content="⏭️ Architecture Specialist: Not required")]}
    
    system_message = SystemMessage(content="""You are an architecture specialist with deep 
    expertise in system design, design patterns, and architectural best practices.""")
    
    user_message = HumanMessage(content=f"""Provide architectural expertise for: {problem}

Focus on:
- System architecture
- Design patterns
- Component design
- Integration patterns
- Best practices""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🏗️ Architecture Specialist: {response.content}")],
        "specialist_contributions": {"architecture": response.content}
    }


# Solution Integrator
def solution_integrator(state: SpecialistState) -> SpecialistState:
    """Integrates specialist contributions into unified solution"""
    problem = state.get("problem", "")
    specialist_contributions = state.get("specialist_contributions", {})
    
    system_message = SystemMessage(content="""You are a solution integrator. Combine 
    insights from multiple specialists into a coherent, comprehensive solution.""")
    
    contributions_text = "\n\n".join([
        f"{domain.upper()} Specialist:\n{contribution[:200]}..."
        for domain, contribution in specialist_contributions.items()
    ])
    
    user_message = HumanMessage(content=f"""Integrate specialist insights for: {problem}

Specialist Contributions:
{contributions_text}

Create unified solution incorporating all expert recommendations.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🔗 Solution Integrator: {response.content}")],
        "integrated_solution": response.content
    }


# Quality Validator
def quality_validator(state: SpecialistState) -> SpecialistState:
    """Validates solution quality across all domains"""
    integrated_solution = state.get("integrated_solution", "")
    specialist_contributions = state.get("specialist_contributions", {})
    
    system_message = SystemMessage(content="""You are a quality validator. Assess the 
    completeness and quality of the integrated solution.""")
    
    user_message = HumanMessage(content=f"""Validate solution quality:

Solution: {integrated_solution[:200]}...
Specialist areas covered: {len(specialist_contributions)}

Rate quality (0-1) based on:
- Completeness
- Integration coherence
- Expert recommendations followed
- Best practices adherence""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate quality score based on coverage
    quality_score = min(0.95, 0.65 + (len(specialist_contributions) * 0.1))
    
    return {
        "messages": [AIMessage(content=f"✅ Quality Validator: {response.content}\n\n✅ Quality Score: {quality_score:.2f}")],
        "quality_score": quality_score
    }


# Specialist Monitor
def specialist_monitor(state: SpecialistState) -> SpecialistState:
    """Monitors specialist collaboration"""
    problem = state.get("problem", "")
    required_specialists = state.get("required_specialists", [])
    specialist_contributions = state.get("specialist_contributions", {})
    quality_score = state.get("quality_score", 0.0)
    integrated_solution = state.get("integrated_solution", "")
    
    specialists_info = "\n".join([
        f"  • {specialist.replace('_', ' ').title()}"
        for specialist in required_specialists
    ])
    
    contributions_info = "\n".join([
        f"  • {domain.title()}: {len(contribution)} characters"
        for domain, contribution in specialist_contributions.items()
    ])
    
    summary = f"""
    ✅ SPECIALIST AGENT PATTERN COMPLETE
    
    Collaboration Summary:
    • Problem: {problem[:80]}...
    • Specialists Engaged: {len(required_specialists)}
    • Contributions Received: {len(specialist_contributions)}
    • Quality Score: {quality_score:.2f}/1.00
    
    Specialists Involved:
{specialists_info}
    
    Contributions:
{contributions_info}
    
    Specialist Pattern Benefits:
    • Deep domain expertise
    • Multi-perspective solutions
    • High-quality recommendations
    • Specialized knowledge application
    • Comprehensive problem coverage
    
    Integrated Solution:
    {integrated_solution[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Specialist Monitor:\n{summary}")]
    }


# Build the graph
def build_specialist_graph():
    """Build the specialist agent pattern graph"""
    workflow = StateGraph(SpecialistState)
    
    workflow.add_node("analyzer", problem_analyzer)
    workflow.add_node("security", security_specialist)
    workflow.add_node("database", database_specialist)
    workflow.add_node("ux", ux_specialist)
    workflow.add_node("performance", performance_specialist)
    workflow.add_node("architecture", architecture_specialist)
    workflow.add_node("integrator", solution_integrator)
    workflow.add_node("validator", quality_validator)
    workflow.add_node("monitor", specialist_monitor)
    
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "security")
    workflow.add_edge("security", "database")
    workflow.add_edge("database", "ux")
    workflow.add_edge("ux", "performance")
    workflow.add_edge("performance", "architecture")
    workflow.add_edge("architecture", "integrator")
    workflow.add_edge("integrator", "validator")
    workflow.add_edge("validator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_specialist_graph()
    
    print("=== Specialist Agent MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "problem": "Design a secure, high-performance user authentication system with database storage and intuitive UI",
        "required_specialists": [],
        "specialist_contributions": {},
        "integrated_solution": "",
        "quality_score": 0.0
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Specialist Collaboration Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Integrated Solution ===")
    print(result.get("integrated_solution", "No solution generated"))
    print(f"\n\nQuality Score: {result.get('quality_score', 0.0):.2f}/1.00")

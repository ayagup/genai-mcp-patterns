"""
Continual Learning MCP Pattern

This pattern demonstrates agents learning continuously from new experiences
while retaining previously learned knowledge without catastrophic forgetting.

Key Features:
- Continuous knowledge acquisition
- Knowledge retention
- Anti-forgetting mechanisms
- Incremental learning
- Long-term memory preservation
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ContinualLearningState(TypedDict):
    """State for continual learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    domain: str
    knowledge_base: dict[str, list[str]]
    new_information: str
    learning_phase: int
    retention_scores: dict[str, float]
    forgetting_rate: float
    total_knowledge_items: int


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Knowledge Integrator
def knowledge_integrator(state: ContinualLearningState) -> ContinualLearningState:
    """Integrates new information into existing knowledge base"""
    domain = state.get("domain", "")
    knowledge_base = state.get("knowledge_base", {})
    new_information = state.get("new_information", "")
    learning_phase = state.get("learning_phase", 0)
    
    system_message = SystemMessage(content="""You are a knowledge integrator. 
    Integrate new information while maintaining existing knowledge.""")
    
    kb_summary = "\n".join([
        f"  • {category}: {len(items)} items"
        for category, items in knowledge_base.items()
    ]) if knowledge_base else "  • Empty knowledge base"
    
    user_message = HumanMessage(content=f"""Integrate new knowledge:

Domain: {domain}
Learning Phase: {learning_phase + 1}

Current Knowledge Base:
{kb_summary}

New Information: {new_information}

Integrate without forgetting existing knowledge.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Integrate new information
    if "concepts" not in knowledge_base:
        knowledge_base["concepts"] = []
    if "skills" not in knowledge_base:
        knowledge_base["skills"] = []
    if "facts" not in knowledge_base:
        knowledge_base["facts"] = []
    
    # Add new knowledge based on phase
    if learning_phase % 3 == 0:
        knowledge_base["concepts"].append(f"Phase {learning_phase + 1} Concept: {new_information[:50]}")
    elif learning_phase % 3 == 1:
        knowledge_base["skills"].append(f"Phase {learning_phase + 1} Skill: {new_information[:50]}")
    else:
        knowledge_base["facts"].append(f"Phase {learning_phase + 1} Fact: {new_information[:50]}")
    
    total_items = sum(len(items) for items in knowledge_base.values())
    
    return {
        "messages": [AIMessage(content=f"🔄 Knowledge Integrator (Phase {learning_phase + 1}):\n{response.content}\n\n✅ Knowledge Base: {total_items} items")],
        "knowledge_base": knowledge_base,
        "total_knowledge_items": total_items
    }


# Memory Consolidator
def memory_consolidator(state: ContinualLearningState) -> ContinualLearningState:
    """Consolidates memories to prevent forgetting"""
    domain = state.get("domain", "")
    knowledge_base = state.get("knowledge_base", {})
    learning_phase = state.get("learning_phase", 0)
    
    system_message = SystemMessage(content="""You are a memory consolidator. 
    Strengthen important memories and connections to prevent forgetting.""")
    
    total_knowledge = sum(len(items) for items in knowledge_base.values())
    
    user_message = HumanMessage(content=f"""Consolidate memories:

Domain: {domain}
Phase: {learning_phase + 1}
Total Knowledge: {total_knowledge} items

Strengthen connections and prevent forgetting.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🧠 Memory Consolidator:\n{response.content}\n\n✅ Memories consolidated")]
    }


# Retention Tester
def retention_tester(state: ContinualLearningState) -> ContinualLearningState:
    """Tests retention of previously learned knowledge"""
    domain = state.get("domain", "")
    knowledge_base = state.get("knowledge_base", {})
    learning_phase = state.get("learning_phase", 0)
    retention_scores = state.get("retention_scores", {})
    
    system_message = SystemMessage(content="""You are a retention tester. 
    Test how well previous knowledge is retained after learning new information.""")
    
    user_message = HumanMessage(content=f"""Test retention:

Domain: {domain}
Current Phase: {learning_phase + 1}
Knowledge Categories: {len(knowledge_base)}

Test if old knowledge is still accessible.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Test retention for each category
    for category, items in knowledge_base.items():
        if items:
            # Simulate retention (high retention due to continual learning)
            base_retention = 0.90
            phase_penalty = min(0.1, learning_phase * 0.01)  # Small degradation
            retention_scores[category] = base_retention - phase_penalty
    
    avg_retention = sum(retention_scores.values()) / len(retention_scores) if retention_scores else 0
    
    return {
        "messages": [AIMessage(content=f"✅ Retention Tester:\n{response.content}\n\n📊 Average Retention: {avg_retention:.1%}")],
        "retention_scores": retention_scores
    }


# Forgetting Monitor
def forgetting_monitor(state: ContinualLearningState) -> ContinualLearningState:
    """Monitors and prevents catastrophic forgetting"""
    domain = state.get("domain", "")
    knowledge_base = state.get("knowledge_base", {})
    retention_scores = state.get("retention_scores", {})
    learning_phase = state.get("learning_phase", 0)
    
    system_message = SystemMessage(content="""You are a forgetting monitor. 
    Detect and prevent catastrophic forgetting of old knowledge.""")
    
    retention_summary = "\n".join([
        f"  • {category}: {score:.1%} retained"
        for category, score in retention_scores.items()
    ]) if retention_scores else "  • No retention data yet"
    
    user_message = HumanMessage(content=f"""Monitor forgetting:

Domain: {domain}
Phase: {learning_phase + 1}

Retention Scores:
{retention_summary}

Check for catastrophic forgetting.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate forgetting rate
    avg_retention = sum(retention_scores.values()) / len(retention_scores) if retention_scores else 1.0
    forgetting_rate = 1.0 - avg_retention
    
    status = "✅ Healthy" if forgetting_rate < 0.2 else "⚠️ Warning" if forgetting_rate < 0.4 else "❌ Critical"
    
    return {
        "messages": [AIMessage(content=f"🛡️ Forgetting Monitor:\n{response.content}\n\n📊 Forgetting Rate: {forgetting_rate:.1%} {status}")],
        "forgetting_rate": forgetting_rate
    }


# Continual Learning Controller
def continual_learning_controller(state: ContinualLearningState) -> ContinualLearningState:
    """Controls and optimizes continual learning process"""
    domain = state.get("domain", "")
    knowledge_base = state.get("knowledge_base", {})
    new_information = state.get("new_information", "")
    learning_phase = state.get("learning_phase", 0)
    retention_scores = state.get("retention_scores", {})
    forgetting_rate = state.get("forgetting_rate", 0.0)
    total_knowledge_items = state.get("total_knowledge_items", 0)
    
    kb_details = "\n".join([
        f"    • {category.title()}: {len(items)} items (retention: {retention_scores.get(category, 0):.1%})"
        for category, items in knowledge_base.items()
    ])
    
    recent_knowledge = []
    for category, items in knowledge_base.items():
        if items:
            recent_knowledge.append(f"    • {items[-1]}")
    
    recent_summary = "\n".join(recent_knowledge[-3:]) if recent_knowledge else "    • None yet"
    
    summary = f"""
    ✅ CONTINUAL LEARNING PATTERN - Phase {learning_phase + 1}
    
    Learning Summary:
    • Domain: {domain}
    • Learning Phase: {learning_phase + 1}
    • Total Knowledge Items: {total_knowledge_items}
    • Forgetting Rate: {forgetting_rate:.1%}
    • Knowledge Categories: {len(knowledge_base)}
    
    Knowledge Base Status:
{kb_details if kb_details else "    • Empty"}
    
    Recently Learned:
{recent_summary}
    
    Continual Learning Process:
    1. Integrate New Knowledge → 2. Consolidate Memories → 
    3. Test Retention → 4. Monitor Forgetting → 5. Optimize → 6. Repeat
    
    Continual Learning Benefits:
    • Learn continuously over time
    • Retain old knowledge
    • Prevent catastrophic forgetting
    • Accumulate knowledge progressively
    • Long-term memory preservation
    • Adaptive knowledge management
    
    Anti-Forgetting Mechanisms:
    • Memory consolidation after each phase
    • Regular retention testing
    • Forgetting rate monitoring
    • Knowledge reinforcement
    • Incremental integration
    
    Performance Metrics:
    • Total Knowledge: {total_knowledge_items} items accumulated
    • Forgetting Rate: {forgetting_rate:.1%} (target: <20%)
    • Average Retention: {(1 - forgetting_rate):.1%}
    • Learning Phases: {learning_phase + 1}
    
    Key Insight:
    Continual learning allows the agent to grow its knowledge base indefinitely
    while maintaining access to previously learned information, avoiding the
    catastrophic forgetting problem common in traditional learning systems.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Continual Learning Controller:\n{summary}")],
        "learning_phase": learning_phase + 1
    }


# Build the graph
def build_continual_learning_graph():
    """Build the continual learning pattern graph"""
    workflow = StateGraph(ContinualLearningState)
    
    workflow.add_node("integrator", knowledge_integrator)
    workflow.add_node("consolidator", memory_consolidator)
    workflow.add_node("tester", retention_tester)
    workflow.add_node("monitor", forgetting_monitor)
    workflow.add_node("controller", continual_learning_controller)
    
    workflow.add_edge(START, "integrator")
    workflow.add_edge("integrator", "consolidator")
    workflow.add_edge("consolidator", "tester")
    workflow.add_edge("tester", "monitor")
    workflow.add_edge("monitor", "controller")
    workflow.add_edge("controller", END)
    
    return workflow.compile()


# Example usage - Multiple learning phases
if __name__ == "__main__":
    graph = build_continual_learning_graph()
    
    print("=== Continual Learning MCP Pattern ===\n")
    
    domain = "Software Development Best Practices"
    learning_content = [
        "Design patterns improve code maintainability and reusability",
        "Test-driven development ensures code quality and reliability",
        "Code reviews facilitate knowledge sharing and error detection",
        "Continuous integration automates testing and deployment",
        "Refactoring improves code structure without changing functionality",
        "Version control enables collaboration and change tracking",
        "Documentation aids understanding and maintenance"
    ]
    
    # Initial state
    state = {
        "messages": [],
        "domain": domain,
        "knowledge_base": {},
        "new_information": "",
        "learning_phase": 0,
        "retention_scores": {},
        "forgetting_rate": 0.0,
        "total_knowledge_items": 0
    }
    
    # Continuous learning across multiple phases
    for i, content in enumerate(learning_content):
        print(f"\n{'=' * 70}")
        print(f"CONTINUAL LEARNING PHASE {i + 1}")
        print('=' * 70)
        
        state["new_information"] = content
        
        result = graph.invoke(state)
        
        # Show messages for this phase
        for msg in result["messages"]:
            print(f"\n{msg.content}")
            print("-" * 70)
        
        # Update state for next phase
        state = {
            "messages": [],
            "domain": domain,
            "knowledge_base": result.get("knowledge_base", {}),
            "new_information": "",
            "learning_phase": result.get("learning_phase", i + 1),
            "retention_scores": result.get("retention_scores", {}),
            "forgetting_rate": result.get("forgetting_rate", 0.0),
            "total_knowledge_items": result.get("total_knowledge_items", 0)
        }
    
    print(f"\n\n{'=' * 70}")
    print("FINAL CONTINUAL LEARNING RESULTS")
    print('=' * 70)
    print(f"\nDomain: {domain}")
    print(f"Learning Phases: {state['learning_phase']}")
    print(f"Total Knowledge: {state['total_knowledge_items']} items")
    print(f"Forgetting Rate: {state['forgetting_rate']:.1%}")
    print(f"Retention Rate: {(1 - state['forgetting_rate']):.1%}")
    print(f"\nKnowledge Categories:")
    for category, items in state['knowledge_base'].items():
        print(f"  • {category.title()}: {len(items)} items")

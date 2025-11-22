"""
Transfer Learning MCP Pattern

This pattern demonstrates agents applying knowledge learned from one domain
to improve performance in a different but related domain.

Key Features:
- Cross-domain knowledge transfer
- Domain adaptation
- Feature extraction
- Knowledge reuse
- Learning efficiency
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class TransferLearningState(TypedDict):
    """State for transfer learning pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    source_domain: str
    target_domain: str
    source_knowledge: dict[str, list[str]]
    transferred_knowledge: dict[str, list[str]]
    adaptations: list[str]
    transfer_effectiveness: float
    performance_improvement: float


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Source Domain Learner
def source_learner(state: TransferLearningState) -> TransferLearningState:
    """Learns from source domain"""
    source_domain = state.get("source_domain", "")
    
    system_message = SystemMessage(content="""You are a source domain learner. 
    Extract general knowledge and patterns from the source domain that can be 
    transferred to other domains.""")
    
    user_message = HumanMessage(content=f"""Learn from source domain: {source_domain}

Extract:
1. General principles
2. Core patterns
3. Best practices
4. Common strategies
5. Universal concepts

Focus on knowledge that can apply to other domains.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Build source knowledge base
    source_knowledge = {
        "principles": [
            "Modularity improves maintainability",
            "Clear interfaces reduce coupling",
            "Validation prevents errors"
        ],
        "patterns": [
            "Layered architecture separates concerns",
            "Caching improves performance",
            "Error handling ensures reliability"
        ],
        "strategies": [
            "Incremental development reduces risk",
            "Testing ensures quality",
            "Documentation aids understanding"
        ]
    }
    
    total_knowledge = sum(len(items) for items in source_knowledge.values())
    
    return {
        "messages": [AIMessage(content=f"📚 Source Domain Learner ({source_domain}):\n{response.content}\n\n✅ Extracted: {total_knowledge} knowledge items")],
        "source_knowledge": source_knowledge
    }


# Knowledge Mapper
def knowledge_mapper(state: TransferLearningState) -> TransferLearningState:
    """Maps source knowledge to target domain"""
    source_domain = state.get("source_domain", "")
    target_domain = state.get("target_domain", "")
    source_knowledge = state.get("source_knowledge", {})
    
    system_message = SystemMessage(content="""You are a knowledge mapper. 
    Identify which source domain knowledge can be applied to the target domain.""")
    
    knowledge_summary = "\n".join([
        f"{category.title()}:\n" + "\n".join([f"  • {item}" for item in items[:2]])
        for category, items in source_knowledge.items()
    ])
    
    user_message = HumanMessage(content=f"""Map knowledge transfer:

Source Domain: {source_domain}
Target Domain: {target_domain}

Source Knowledge:
{knowledge_summary}

Identify transferable knowledge and necessary adaptations.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Identify transferable knowledge
    transferable = {
        "applicable_principles": [
            "Modularity concept applies to target domain",
            "Validation principle is universal"
        ],
        "adaptable_patterns": [
            "Layered architecture can be adapted",
            "Caching strategy is transferable"
        ],
        "relevant_strategies": [
            "Incremental development works here too",
            "Testing approach is applicable"
        ]
    }
    
    total_transferable = sum(len(items) for items in transferable.values())
    
    return {
        "messages": [AIMessage(content=f"🗺️ Knowledge Mapper:\n{response.content}\n\n✅ Transferable: {total_transferable} items")],
        "transferred_knowledge": transferable
    }


# Domain Adapter
def domain_adapter(state: TransferLearningState) -> TransferLearningState:
    """Adapts transferred knowledge to target domain"""
    source_domain = state.get("source_domain", "")
    target_domain = state.get("target_domain", "")
    transferred_knowledge = state.get("transferred_knowledge", {})
    
    system_message = SystemMessage(content="""You are a domain adapter. 
    Adapt transferred knowledge to fit the target domain's specific requirements.""")
    
    knowledge_summary = "\n".join([
        f"{category.replace('_', ' ').title()}:\n" + "\n".join([f"  • {item}" for item in items[:2]])
        for category, items in transferred_knowledge.items()
    ])
    
    user_message = HumanMessage(content=f"""Adapt knowledge:

From: {source_domain}
To: {target_domain}

Transferred Knowledge:
{knowledge_summary}

Adapt for target domain specifics.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Document adaptations
    adaptations = [
        f"Adapted modularity: Apply domain-specific module boundaries",
        f"Adapted validation: Use {target_domain}-specific validation rules",
        f"Adapted caching: Implement {target_domain}-optimized cache strategy",
        f"Adapted testing: Create {target_domain}-specific test cases"
    ]
    
    return {
        "messages": [AIMessage(content=f"🔧 Domain Adapter:\n{response.content}\n\n✅ Adaptations: {len(adaptations)}")],
        "adaptations": adaptations
    }


# Performance Validator
def performance_validator(state: TransferLearningState) -> TransferLearningState:
    """Validates transfer learning effectiveness"""
    source_domain = state.get("source_domain", "")
    target_domain = state.get("target_domain", "")
    source_knowledge = state.get("source_knowledge", {})
    transferred_knowledge = state.get("transferred_knowledge", {})
    adaptations = state.get("adaptations", [])
    
    system_message = SystemMessage(content="""You are a performance validator. 
    Assess how effectively knowledge transferred from source to target domain.""")
    
    user_message = HumanMessage(content=f"""Validate transfer learning:

Transfer: {source_domain} → {target_domain}
Source Knowledge Items: {sum(len(items) for items in source_knowledge.values())}
Transferred Items: {sum(len(items) for items in transferred_knowledge.values())}
Adaptations Made: {len(adaptations)}

Evaluate transfer effectiveness and performance improvement.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate metrics
    source_total = sum(len(items) for items in source_knowledge.values())
    transferred_total = sum(len(items) for items in transferred_knowledge.values())
    
    transfer_effectiveness = (transferred_total / source_total * 100) if source_total > 0 else 0
    
    # Simulate performance improvement
    base_performance = 0.5  # Without transfer learning
    transfer_boost = 0.25  # Boost from transfer learning
    adaptation_boost = len(adaptations) * 0.03
    performance_improvement = base_performance + transfer_boost + adaptation_boost
    
    return {
        "messages": [AIMessage(content=f"✅ Performance Validator:\n{response.content}\n\n📊 Effectiveness: {transfer_effectiveness:.1f}% | Improvement: {performance_improvement:.1%}")],
        "transfer_effectiveness": transfer_effectiveness,
        "performance_improvement": performance_improvement
    }


# Transfer Learning Monitor
def transfer_monitor(state: TransferLearningState) -> TransferLearningState:
    """Monitors overall transfer learning process"""
    source_domain = state.get("source_domain", "")
    target_domain = state.get("target_domain", "")
    source_knowledge = state.get("source_knowledge", {})
    transferred_knowledge = state.get("transferred_knowledge", {})
    adaptations = state.get("adaptations", [])
    transfer_effectiveness = state.get("transfer_effectiveness", 0.0)
    performance_improvement = state.get("performance_improvement", 0.0)
    
    source_summary = "\n".join([
        f"    • {category.replace('_', ' ').title()}: {len(items)} items"
        for category, items in source_knowledge.items()
    ])
    
    transferred_summary = "\n".join([
        f"    • {category.replace('_', ' ').title()}: {len(items)} items"
        for category, items in transferred_knowledge.items()
    ])
    
    adaptations_summary = "\n".join([
        f"    • {adaptation}" for adaptation in adaptations[:4]
    ])
    
    summary = f"""
    ✅ TRANSFER LEARNING PATTERN COMPLETE
    
    Transfer Summary:
    • Source Domain: {source_domain}
    • Target Domain: {target_domain}
    • Transfer Effectiveness: {transfer_effectiveness:.1f}%
    • Performance Improvement: {performance_improvement:.1%}
    
    Source Knowledge:
{source_summary}
    
    Transferred Knowledge:
{transferred_summary}
    
    Domain Adaptations:
{adaptations_summary}
    
    Transfer Learning Process:
    1. Learn from Source Domain → 2. Map Transferable Knowledge → 
    3. Adapt to Target Domain → 4. Validate Performance → 5. Monitor Results
    
    Transfer Learning Benefits:
    • Faster learning in new domains
    • Reduced training requirements
    • Leverages existing knowledge
    • Cross-domain applicability
    • Improved efficiency
    • Knowledge reusability
    
    Key Insights:
    • {sum(len(items) for items in source_knowledge.values())} source concepts extracted
    • {sum(len(items) for items in transferred_knowledge.values())} concepts transferred
    • {len(adaptations)} domain-specific adaptations
    • {transfer_effectiveness:.1f}% knowledge successfully transferred
    • {performance_improvement:.1%} performance gain in target domain
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Transfer Learning Monitor:\n{summary}")]
    }


# Build the graph
def build_transfer_learning_graph():
    """Build the transfer learning pattern graph"""
    workflow = StateGraph(TransferLearningState)
    
    workflow.add_node("source_learner", source_learner)
    workflow.add_node("mapper", knowledge_mapper)
    workflow.add_node("adapter", domain_adapter)
    workflow.add_node("validator", performance_validator)
    workflow.add_node("monitor", transfer_monitor)
    
    workflow.add_edge(START, "source_learner")
    workflow.add_edge("source_learner", "mapper")
    workflow.add_edge("mapper", "adapter")
    workflow.add_edge("adapter", "validator")
    workflow.add_edge("validator", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_transfer_learning_graph()
    
    print("=== Transfer Learning MCP Pattern ===\n")
    
    # Example: Transfer from Web Development to Mobile App Development
    state = {
        "messages": [],
        "source_domain": "Web Development",
        "target_domain": "Mobile App Development",
        "source_knowledge": {},
        "transferred_knowledge": {},
        "adaptations": [],
        "transfer_effectiveness": 0.0,
        "performance_improvement": 0.0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "=" * 70)
    print("TRANSFER LEARNING COMPLETE")
    print("=" * 70)
    print(f"\nTransfer Path: {state['source_domain']} → {state['target_domain']}")
    print(f"Effectiveness: {result['transfer_effectiveness']:.1f}%")
    print(f"Performance Gain: {result['performance_improvement']:.1%}")

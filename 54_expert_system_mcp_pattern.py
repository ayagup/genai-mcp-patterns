"""
Expert System MCP Pattern

This pattern demonstrates a knowledge-based reasoning system that uses
rules and facts to make intelligent decisions.

Key Features:
- Rule-based reasoning
- Fact collection and inference
- Knowledge base management
- Explanation of reasoning
- Confidence scoring
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ExpertSystemState(TypedDict):
    """State for expert system pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    facts: list[str]
    rules: list[dict[str, str]]
    inferences: list[str]
    recommendation: str
    confidence: float
    reasoning_trace: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Knowledge Base (Rules and Facts)
RULE_BASE = [
    {"id": "R1", "condition": "high_traffic AND slow_response", "conclusion": "scale_horizontally", "confidence": 0.9},
    {"id": "R2", "condition": "memory_high AND cpu_low", "conclusion": "memory_leak", "confidence": 0.85},
    {"id": "R3", "condition": "database_slow AND high_connections", "conclusion": "add_connection_pool", "confidence": 0.88},
    {"id": "R4", "condition": "security_breach AND unauthorized_access", "conclusion": "implement_mfa", "confidence": 0.95},
    {"id": "R5", "condition": "data_loss AND no_backup", "conclusion": "implement_backup_strategy", "confidence": 0.92},
]


# Fact Collector
def fact_collector(state: ExpertSystemState) -> ExpertSystemState:
    """Collects facts from the query"""
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a fact collector. Extract relevant 
    facts and conditions from the user query that can be used for rule-based reasoning.""")
    
    user_message = HumanMessage(content=f"""Extract facts from query: {query}

Identify conditions like:
- Performance issues
- Resource utilization
- Security concerns
- Data management issues""")
    
    response = llm.invoke([system_message, user_message])
    
    # Extract facts from query
    query_lower = query.lower()
    facts = []
    
    if "high traffic" in query_lower or "many users" in query_lower:
        facts.append("high_traffic")
    if "slow" in query_lower or "latency" in query_lower:
        facts.append("slow_response")
    if "memory" in query_lower:
        facts.append("memory_high")
    if "database" in query_lower and "slow" in query_lower:
        facts.append("database_slow")
    if "connections" in query_lower:
        facts.append("high_connections")
    if "security" in query_lower or "breach" in query_lower:
        facts.append("security_breach")
    if "unauthorized" in query_lower:
        facts.append("unauthorized_access")
    
    return {
        "messages": [AIMessage(content=f"📋 Fact Collector: {response.content}\n\n✅ Collected {len(facts)} facts: {', '.join(facts)}")],
        "facts": facts
    }


# Rule Matcher
def rule_matcher(state: ExpertSystemState) -> ExpertSystemState:
    """Matches facts against rule base"""
    facts = state.get("facts", [])
    
    system_message = SystemMessage(content="""You are a rule matcher. Match collected 
    facts against the rule base to identify applicable rules.""")
    
    user_message = HumanMessage(content=f"""Match facts against rules:

Facts: {', '.join(facts)}

Available Rules: {len(RULE_BASE)}

Identify matching rules.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Match rules
    matched_rules = []
    for rule in RULE_BASE:
        conditions = rule["condition"].replace("AND", " ").split()
        if all(cond.strip() in facts for cond in conditions):
            matched_rules.append(rule)
    
    return {
        "messages": [AIMessage(content=f"🔍 Rule Matcher: {response.content}\n\n✅ Matched {len(matched_rules)} rules")],
        "rules": matched_rules
    }


# Inference Engine
def inference_engine(state: ExpertSystemState) -> ExpertSystemState:
    """Performs inference based on matched rules"""
    rules = state.get("rules", [])
    facts = state.get("facts", [])
    
    system_message = SystemMessage(content="""You are an inference engine. Apply matched 
    rules to derive conclusions and recommendations.""")
    
    rules_text = "\n".join([
        f"{rule['id']}: IF {rule['condition']} THEN {rule['conclusion']} (confidence: {rule['confidence']})"
        for rule in rules
    ])
    
    user_message = HumanMessage(content=f"""Apply rules to derive inferences:

Facts: {', '.join(facts)}

Applicable Rules:
{rules_text}

Generate inferences.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate inferences
    inferences = []
    reasoning_trace = []
    
    for rule in rules:
        inference = f"{rule['conclusion']} (from {rule['id']}, confidence: {rule['confidence']})"
        inferences.append(inference)
        reasoning_trace.append(f"Applied {rule['id']}: {rule['condition']} → {rule['conclusion']}")
    
    return {
        "messages": [AIMessage(content=f"🧠 Inference Engine: {response.content}\n\n✅ Generated {len(inferences)} inferences")],
        "inferences": inferences,
        "reasoning_trace": reasoning_trace
    }


# Recommendation Generator
def recommendation_generator(state: ExpertSystemState) -> ExpertSystemState:
    """Generates final recommendation"""
    inferences = state.get("inferences", [])
    rules = state.get("rules", [])
    query = state.get("query", "")
    
    system_message = SystemMessage(content="""You are a recommendation generator. Based 
    on inferences, create a comprehensive recommendation with actionable steps.""")
    
    inferences_text = "\n".join([f"• {inf}" for inf in inferences])
    
    user_message = HumanMessage(content=f"""Generate recommendation for: {query}

Inferences:
{inferences_text}

Create actionable recommendation.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate overall confidence
    if rules:
        confidence = sum(rule["confidence"] for rule in rules) / len(rules)
    else:
        confidence = 0.0
    
    return {
        "messages": [AIMessage(content=f"💡 Recommendation Generator: {response.content}")],
        "recommendation": response.content,
        "confidence": confidence
    }


# Explanation Generator
def explanation_generator(state: ExpertSystemState) -> ExpertSystemState:
    """Explains the reasoning process"""
    reasoning_trace = state.get("reasoning_trace", [])
    recommendation = state.get("recommendation", "")
    confidence = state.get("confidence", 0.0)
    
    system_message = SystemMessage(content="""You are an explanation generator. Provide 
    clear explanation of how the expert system arrived at its recommendation.""")
    
    trace_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(reasoning_trace)])
    
    user_message = HumanMessage(content=f"""Explain reasoning:

Recommendation: {recommendation[:150]}...
Confidence: {confidence:.2f}

Reasoning Steps:
{trace_text}

Provide clear explanation.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"📖 Explanation Generator: {response.content}")]
    }


# Expert System Monitor
def expert_system_monitor(state: ExpertSystemState) -> ExpertSystemState:
    """Monitors expert system execution"""
    query = state.get("query", "")
    facts = state.get("facts", [])
    rules = state.get("rules", [])
    inferences = state.get("inferences", [])
    confidence = state.get("confidence", 0.0)
    reasoning_trace = state.get("reasoning_trace", [])
    recommendation = state.get("recommendation", "")
    
    facts_text = "\n".join([f"  • {fact}" for fact in facts])
    rules_text = "\n".join([f"  • {rule['id']}: {rule['conclusion']}" for rule in rules])
    inferences_text = "\n".join([f"  • {inf[:80]}..." for inf in inferences])
    trace_text = "\n".join([f"  {i+1}. {step}" for i, step in enumerate(reasoning_trace)])
    
    summary = f"""
    ✅ EXPERT SYSTEM PATTERN COMPLETE
    
    Expert System Execution:
    • Query: {query[:80]}...
    • Facts Collected: {len(facts)}
    • Rules Matched: {len(rules)}
    • Inferences Generated: {len(inferences)}
    • Confidence: {confidence:.2f}/1.00
    
    Collected Facts:
{facts_text}
    
    Applied Rules:
{rules_text}
    
    Inferences:
{inferences_text}
    
    Reasoning Trace:
{trace_text}
    
    Expert System Benefits:
    • Knowledge-based reasoning
    • Transparent decision making
    • Explainable recommendations
    • Rule-based consistency
    • Confidence scoring
    
    Final Recommendation:
    {recommendation[:300]}...
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Expert System Monitor:\n{summary}")]
    }


# Build the graph
def build_expert_system_graph():
    """Build the expert system pattern graph"""
    workflow = StateGraph(ExpertSystemState)
    
    workflow.add_node("collector", fact_collector)
    workflow.add_node("matcher", rule_matcher)
    workflow.add_node("inference", inference_engine)
    workflow.add_node("recommender", recommendation_generator)
    workflow.add_node("explainer", explanation_generator)
    workflow.add_node("monitor", expert_system_monitor)
    
    workflow.add_edge(START, "collector")
    workflow.add_edge("collector", "matcher")
    workflow.add_edge("matcher", "inference")
    workflow.add_edge("inference", "recommender")
    workflow.add_edge("recommender", "explainer")
    workflow.add_edge("explainer", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_expert_system_graph()
    
    print("=== Expert System MCP Pattern ===\n")
    
    initial_state = {
        "messages": [],
        "query": "Our application is experiencing high traffic and slow response times. What should we do?",
        "facts": [],
        "rules": [],
        "inferences": [],
        "recommendation": "",
        "confidence": 0.0,
        "reasoning_trace": []
    }
    
    result = graph.invoke(initial_state)
    
    print("\n=== Expert System Execution ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n\n=== Final Recommendation ===")
    print(result.get("recommendation", "No recommendation generated"))
    print(f"\nConfidence: {result.get('confidence', 0.0):.2f}/1.00")

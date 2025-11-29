"""
Symbolic-Neural Hybrid MCP Pattern

This pattern demonstrates combining symbolic reasoning (rules, logic)
with neural approaches (LLMs, embeddings) for enhanced reasoning.

Pattern Type: Hybrid
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class SymbolicNeuralState(TypedDict):
    """State for symbolic-neural hybrid reasoning"""
    query: str
    domain: str
    symbolic_rules: List[Dict[str, Any]]
    neural_insights: Dict[str, Any]
    rule_matches: List[Dict[str, Any]]
    neural_analysis: Dict[str, Any]
    hybrid_reasoning: Dict[str, Any]
    symbolic_confidence: float
    neural_confidence: float
    final_answer: Dict[str, Any]
    reasoning_trace: str
    messages: Annotated[List, operator.add]
    current_step: str


class SymbolicRuleEngine:
    """Symbolic rule-based reasoning"""
    
    def __init__(self):
        # Predefined symbolic rules
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize domain rules"""
        return [
            {
                "rule_id": "R1",
                "name": "Transitivity",
                "pattern": "IF A > B AND B > C THEN A > C",
                "domain": "general",
                "confidence": 1.0
            },
            {
                "rule_id": "R2",
                "name": "Modus Ponens",
                "pattern": "IF P IMPLIES Q AND P THEN Q",
                "domain": "logic",
                "confidence": 1.0
            },
            {
                "rule_id": "R3",
                "name": "Conservation",
                "pattern": "IF system is closed THEN total is conserved",
                "domain": "physics",
                "confidence": 0.95
            },
            {
                "rule_id": "R4",
                "name": "Causality",
                "pattern": "IF effect observed THEN cause must precede",
                "domain": "general",
                "confidence": 0.98
            },
            {
                "rule_id": "R5",
                "name": "Consistency",
                "pattern": "IF A is true THEN NOT A is false",
                "domain": "logic",
                "confidence": 1.0
            }
        ]
    
    def match_rules(self, query: str, domain: str) -> List[Dict[str, Any]]:
        """Match query against symbolic rules"""
        matches = []
        
        # Simple keyword matching (in production, use proper rule engine)
        query_lower = query.lower()
        
        for rule in self.rules:
            # Check domain match
            if rule["domain"] == "general" or rule["domain"] == domain.lower():
                # Simple pattern matching
                keywords = self._extract_keywords(rule["pattern"])
                
                if any(kw in query_lower for kw in keywords):
                    matches.append({
                        "rule_id": rule["rule_id"],
                        "rule_name": rule["name"],
                        "pattern": rule["pattern"],
                        "match_score": 0.8,  # Simplified
                        "confidence": rule["confidence"]
                    })
        
        return matches
    
    def _extract_keywords(self, pattern: str) -> List[str]:
        """Extract keywords from rule pattern"""
        keywords = []
        pattern_lower = pattern.lower()
        
        # Extract meaningful words
        words = ["greater", "implies", "then", "if", "and", "conserved", 
                 "cause", "effect", "true", "false", "closed", "total"]
        
        for word in words:
            if word in pattern_lower:
                keywords.append(word)
        
        return keywords
    
    def apply_rules(self, matches: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Apply matched rules"""
        if not matches:
            return {
                "applicable": False,
                "conclusion": None,
                "confidence": 0.0
            }
        
        # Apply highest confidence rule
        best_match = max(matches, key=lambda m: m["confidence"])
        
        return {
            "applicable": True,
            "applied_rule": best_match["rule_id"],
            "rule_name": best_match["rule_name"],
            "conclusion": f"Based on {best_match['rule_name']}: symbolic inference applied",
            "confidence": best_match["confidence"]
        }


class NeuralReasoningEngine:
    """Neural network-based reasoning using LLM"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_query(self, query: str, domain: str) -> Dict[str, Any]:
        """Neural analysis of query"""
        prompt = f"""Analyze this query using neural/pattern-based reasoning:

Query: {query}
Domain: {domain}

Provide:
1. Semantic understanding
2. Context and implications
3. Patterns recognized
4. Confidence in understanding

Return JSON:
{{
    "semantic_understanding": "what query means",
    "context": "relevant context",
    "patterns": ["recognized patterns"],
    "implications": ["what it implies"],
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are a neural reasoning engine."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "semantic_understanding": "Unable to parse",
            "context": "N/A",
            "patterns": [],
            "implications": [],
            "confidence": 0.0
        }
    
    def generate_answer(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate neural answer"""
        prompt = f"""Answer this query using neural reasoning:

Query: {query}

Context:
{json.dumps(context, indent=2)}

Provide a comprehensive answer based on learned patterns.

Return JSON:
{{
    "answer": "the answer",
    "reasoning": "how you arrived at answer",
    "confidence": 0.0-1.0,
    "limitations": ["potential limitations"]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert neural reasoning system."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "answer": "Unable to generate answer",
            "reasoning": "Parsing error",
            "confidence": 0.0,
            "limitations": ["parsing failed"]
        }


class HybridIntegrator:
    """Integrate symbolic and neural reasoning"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def integrate_reasoning(self, symbolic_result: Dict[str, Any],
                          neural_result: Dict[str, Any],
                          query: str) -> Dict[str, Any]:
        """Integrate symbolic and neural insights"""
        prompt = f"""Integrate symbolic and neural reasoning:

Query: {query}

Symbolic Reasoning:
{json.dumps(symbolic_result, indent=2)}

Neural Reasoning:
{json.dumps(neural_result, indent=2)}

Combine both approaches:
1. Where do they agree?
2. Where do they differ?
3. Which is more reliable for this query?
4. Synthesized answer

Return JSON:
{{
    "agreement_areas": ["where both agree"],
    "disagreement_areas": ["where they differ"],
    "symbolic_strength": "where symbolic excels",
    "neural_strength": "where neural excels",
    "integrated_answer": "combined answer",
    "confidence": 0.0-1.0,
    "reasoning_type": "symbolic|neural|hybrid"
}}"""
        
        messages = [
            SystemMessage(content="You are a hybrid reasoning integrator."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "agreement_areas": [],
            "disagreement_areas": [],
            "symbolic_strength": "N/A",
            "neural_strength": "N/A",
            "integrated_answer": "Unable to integrate",
            "confidence": 0.0,
            "reasoning_type": "hybrid"
        }


# Agent functions
def initialize_hybrid(state: SymbolicNeuralState) -> SymbolicNeuralState:
    """Initialize hybrid reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing symbolic-neural hybrid reasoning for: {state['query']}"
    ))
    state["symbolic_rules"] = []
    state["current_step"] = "initialized"
    return state


def apply_symbolic_rules(state: SymbolicNeuralState) -> SymbolicNeuralState:
    """Apply symbolic rule-based reasoning"""
    engine = SymbolicRuleEngine()
    
    # Load rules
    state["symbolic_rules"] = engine.rules
    
    # Match rules
    matches = engine.match_rules(state["query"], state["domain"])
    state["rule_matches"] = matches
    
    # Apply rules
    result = engine.apply_rules(matches, state["query"])
    state["symbolic_confidence"] = result.get("confidence", 0.0)
    
    state["messages"].append(HumanMessage(
        content=f"Symbolic: {len(matches)} rules matched, confidence: {state['symbolic_confidence']:.2f}"
    ))
    state["current_step"] = "symbolic_applied"
    return state


def apply_neural_reasoning(state: SymbolicNeuralState) -> SymbolicNeuralState:
    """Apply neural reasoning"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    engine = NeuralReasoningEngine(llm)
    
    # Analyze query
    analysis = engine.analyze_query(state["query"], state["domain"])
    state["neural_analysis"] = analysis
    
    # Generate answer
    neural_answer = engine.generate_answer(state["query"], analysis)
    state["neural_insights"] = neural_answer
    state["neural_confidence"] = neural_answer.get("confidence", 0.0)
    
    state["messages"].append(HumanMessage(
        content=f"Neural: Analysis complete, confidence: {state['neural_confidence']:.2f}"
    ))
    state["current_step"] = "neural_applied"
    return state


def integrate_hybrid_reasoning(state: SymbolicNeuralState) -> SymbolicNeuralState:
    """Integrate symbolic and neural"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    integrator = HybridIntegrator(llm)
    
    # Prepare symbolic result
    symbolic_result = {
        "rule_matches": state["rule_matches"],
        "confidence": state["symbolic_confidence"],
        "type": "symbolic"
    }
    
    # Integrate
    hybrid_result = integrator.integrate_reasoning(
        symbolic_result,
        state["neural_insights"],
        state["query"]
    )
    
    state["hybrid_reasoning"] = hybrid_result
    
    state["messages"].append(HumanMessage(
        content=f"Hybrid: Integrated both approaches, type: {hybrid_result.get('reasoning_type', 'unknown')}"
    ))
    state["current_step"] = "integrated"
    return state


def synthesize_answer(state: SymbolicNeuralState) -> SymbolicNeuralState:
    """Synthesize final answer"""
    hybrid = state["hybrid_reasoning"]
    
    final_answer = {
        "answer": hybrid.get("integrated_answer", "No answer"),
        "symbolic_contribution": {
            "rules_used": len(state["rule_matches"]),
            "confidence": state["symbolic_confidence"],
            "strength": hybrid.get("symbolic_strength", "N/A")
        },
        "neural_contribution": {
            "patterns": state["neural_analysis"].get("patterns", []),
            "confidence": state["neural_confidence"],
            "strength": hybrid.get("neural_strength", "N/A")
        },
        "hybrid_confidence": hybrid.get("confidence", 0.0),
        "reasoning_type": hybrid.get("reasoning_type", "hybrid")
    }
    
    state["final_answer"] = final_answer
    
    state["messages"].append(HumanMessage(
        content=f"Answer synthesized: {final_answer['answer'][:100]}..."
    ))
    state["current_step"] = "synthesized"
    return state


def generate_report(state: SymbolicNeuralState) -> SymbolicNeuralState:
    """Generate reasoning trace report"""
    report = f"""
SYMBOLIC-NEURAL HYBRID REASONING REPORT
=======================================

Query: {state['query']}
Domain: {state['domain']}

SYMBOLIC REASONING:
-------------------
Available Rules: {len(state['symbolic_rules'])}
Matched Rules: {len(state['rule_matches'])}

Rule Matches:
"""
    
    for match in state['rule_matches']:
        report += f"\n- {match['rule_name']} ({match['rule_id']})\n"
        report += f"  Pattern: {match['pattern']}\n"
        report += f"  Confidence: {match['confidence']:.2f}\n"
    
    report += f"""
Symbolic Confidence: {state['symbolic_confidence']:.2%}

NEURAL REASONING:
-----------------
Semantic Understanding: {state['neural_analysis'].get('semantic_understanding', 'N/A')}

Patterns Recognized:
{chr(10).join(f'- {p}' for p in state['neural_analysis'].get('patterns', []))}

Neural Answer: {state['neural_insights'].get('answer', 'N/A')}
Neural Confidence: {state['neural_confidence']:.2%}

HYBRID INTEGRATION:
-------------------
Agreement Areas:
{chr(10).join(f'- {a}' for a in state['hybrid_reasoning'].get('agreement_areas', []))}

Disagreement Areas:
{chr(10).join(f'- {d}' for d in state['hybrid_reasoning'].get('disagreement_areas', []))}

Symbolic Strength: {state['hybrid_reasoning'].get('symbolic_strength', 'N/A')}
Neural Strength: {state['hybrid_reasoning'].get('neural_strength', 'N/A')}

FINAL ANSWER:
-------------
{state['final_answer']['answer']}

Reasoning Type: {state['final_answer']['reasoning_type']}
Hybrid Confidence: {state['final_answer']['hybrid_confidence']:.2%}

Contributions:
- Symbolic: {state['final_answer']['symbolic_contribution']['rules_used']} rules, 
  confidence {state['final_answer']['symbolic_contribution']['confidence']:.2%}
- Neural: {len(state['final_answer']['neural_contribution']['patterns'])} patterns, 
  confidence {state['final_answer']['neural_contribution']['confidence']:.2%}

Summary:
This answer combines the precision of symbolic rules with the flexibility
of neural pattern recognition for optimal reasoning.
"""
    
    state["reasoning_trace"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_hybrid_graph():
    """Create symbolic-neural hybrid workflow"""
    workflow = StateGraph(SymbolicNeuralState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_hybrid)
    workflow.add_node("symbolic", apply_symbolic_rules)
    workflow.add_node("neural", apply_neural_reasoning)
    workflow.add_node("integrate", integrate_hybrid_reasoning)
    workflow.add_node("synthesize", synthesize_answer)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "symbolic")
    workflow.add_edge("symbolic", "neural")
    workflow.add_edge("neural", "integrate")
    workflow.add_edge("integrate", "synthesize")
    workflow.add_edge("synthesize", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample query
    initial_state = {
        "query": "If temperature increases, then pressure increases. Temperature is increasing. What happens to pressure?",
        "domain": "physics",
        "symbolic_rules": [],
        "neural_insights": {},
        "rule_matches": [],
        "neural_analysis": {},
        "hybrid_reasoning": {},
        "symbolic_confidence": 0.0,
        "neural_confidence": 0.0,
        "final_answer": {},
        "reasoning_trace": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run
    app = create_hybrid_graph()
    
    print("Symbolic-Neural Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")
    
    print(f"\nFinal Answer: {result['final_answer'].get('answer', 'Unknown')}")
    print(f"Reasoning Type: {result['final_answer'].get('reasoning_type', 'Unknown')}")

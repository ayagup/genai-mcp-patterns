"""
Abductive Reasoning MCP Pattern

This pattern demonstrates inference to the best explanation, generating and
evaluating hypotheses to explain observations.

Pattern Type: Advanced Reasoning
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
class AbductiveReasoningState(TypedDict):
    """State for abductive reasoning workflow"""
    observations: List[str]
    domain_context: str
    hypothesis_space: List[Dict[str, Any]]
    explanatory_hypotheses: List[Dict[str, Any]]
    hypothesis_evaluations: List[Dict[str, Any]]
    best_explanation: Dict[str, Any]
    alternative_explanations: List[Dict[str, Any]]
    explanatory_coherence: Dict[str, Any]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class HypothesisGenerator:
    """Generate explanatory hypotheses"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_hypotheses(self, observations: List[str], 
                           context: str) -> List[Dict[str, Any]]:
        """Generate potential explanatory hypotheses"""
        prompt = f"""Generate explanatory hypotheses for these observations:

Observations:
{chr(10).join(f'{i+1}. {obs}' for i, obs in enumerate(observations))}

Context: {context}

Generate 5-7 different hypotheses that could explain these observations.
For each hypothesis:
1. State the explanation clearly
2. Identify what it explains
3. Note assumptions made
4. Assess initial plausibility

Return JSON array:
[
    {{
        "id": 1,
        "hypothesis": "the proposed explanation",
        "explains": ["observation1", "observation2"],
        "assumptions": ["assumption1", "assumption2"],
        "initial_plausibility": "high|medium|low",
        "type": "causal|descriptive|mechanistic|functional"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at generating explanatory hypotheses."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class ExplanatoryPowerAnalyzer:
    """Analyze explanatory power of hypotheses"""
    
    def analyze_explanatory_power(self, hypotheses: List[Dict[str, Any]],
                                  observations: List[str]) -> List[Dict[str, Any]]:
        """Analyze how well each hypothesis explains the observations"""
        analyzed = []
        
        for hypothesis in hypotheses:
            explained_obs = hypothesis.get("explains", [])
            
            # Calculate coverage
            coverage = len(explained_obs) / len(observations) if observations else 0
            
            # Calculate specificity (how specific vs generic)
            specificity = self._calculate_specificity(hypothesis)
            
            # Calculate parsimony (simplicity)
            parsimony = self._calculate_parsimony(hypothesis)
            
            analyzed.append({
                "hypothesis_id": hypothesis.get("id", 0),
                "hypothesis": hypothesis.get("hypothesis", ""),
                "coverage": coverage,
                "specificity": specificity,
                "parsimony": parsimony,
                "explanatory_power": (coverage * 0.4 + specificity * 0.3 + parsimony * 0.3)
            })
        
        return analyzed
    
    def _calculate_specificity(self, hypothesis: Dict[str, Any]) -> float:
        """Calculate how specific the hypothesis is"""
        explanation = hypothesis.get("hypothesis", "")
        
        # Simple heuristic: longer, more detailed explanations are more specific
        word_count = len(explanation.split())
        
        if word_count > 30:
            return 0.9
        elif word_count > 20:
            return 0.7
        elif word_count > 10:
            return 0.5
        else:
            return 0.3
    
    def _calculate_parsimony(self, hypothesis: Dict[str, Any]) -> float:
        """Calculate simplicity (fewer assumptions = higher parsimony)"""
        assumptions = hypothesis.get("assumptions", [])
        
        # Fewer assumptions = higher parsimony
        if len(assumptions) <= 1:
            return 1.0
        elif len(assumptions) <= 2:
            return 0.8
        elif len(assumptions) <= 3:
            return 0.6
        else:
            return 0.4


class HypothesisEvaluator:
    """Evaluate hypotheses across multiple criteria"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def evaluate_hypotheses(self, hypotheses: List[Dict[str, Any]],
                           observations: List[str]) -> List[Dict[str, Any]]:
        """Evaluate each hypothesis comprehensively"""
        evaluations = []
        
        for hypothesis in hypotheses:
            evaluation = self._evaluate_single_hypothesis(
                hypothesis,
                observations
            )
            evaluations.append(evaluation)
        
        return evaluations
    
    def _evaluate_single_hypothesis(self, hypothesis: Dict[str, Any],
                                    observations: List[str]) -> Dict[str, Any]:
        """Evaluate a single hypothesis"""
        prompt = f"""Evaluate this explanatory hypothesis:

Hypothesis: {hypothesis.get('hypothesis', '')}

Observations:
{chr(10).join(f'- {obs}' for obs in observations)}

Evaluate on these criteria (scale 0.0-1.0):
1. Consistency: Does it contradict any observations?
2. Completeness: Does it explain all relevant observations?
3. Simplicity: Is it the simplest viable explanation?
4. Testability: Can it be tested or verified?
5. Coherence: Does it fit with existing knowledge?

Return JSON:
{{
    "consistency": 0.0-1.0,
    "completeness": 0.0-1.0,
    "simplicity": 0.0-1.0,
    "testability": 0.0-1.0,
    "coherence": 0.0-1.0,
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "potential_falsifiers": ["what could prove it wrong"]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at evaluating explanatory hypotheses."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                evaluation = json.loads(content[json_start:json_end])
                evaluation["hypothesis_id"] = hypothesis.get("id", 0)
                
                # Calculate overall score
                evaluation["overall_score"] = (
                    evaluation.get("consistency", 0) * 0.25 +
                    evaluation.get("completeness", 0) * 0.25 +
                    evaluation.get("simplicity", 0) * 0.2 +
                    evaluation.get("testability", 0) * 0.15 +
                    evaluation.get("coherence", 0) * 0.15
                )
                
                return evaluation
            except:
                pass
        
        return {
            "hypothesis_id": hypothesis.get("id", 0),
            "consistency": 0.5,
            "completeness": 0.5,
            "simplicity": 0.5,
            "testability": 0.5,
            "coherence": 0.5,
            "overall_score": 0.5,
            "strengths": [],
            "weaknesses": [],
            "potential_falsifiers": []
        }


class BestExplanationSelector:
    """Select the best explanation using inference to best explanation"""
    
    def select_best(self, hypotheses: List[Dict[str, Any]],
                   evaluations: List[Dict[str, Any]],
                   explanatory_power: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best explanation"""
        # Combine scores from evaluations and explanatory power
        combined_scores = []
        
        for hyp in hypotheses:
            hyp_id = hyp.get("id", 0)
            
            # Get evaluation
            evaluation = next((e for e in evaluations if e.get("hypothesis_id") == hyp_id), {})
            eval_score = evaluation.get("overall_score", 0)
            
            # Get explanatory power
            exp_power = next((e for e in explanatory_power if e.get("hypothesis_id") == hyp_id), {})
            power_score = exp_power.get("explanatory_power", 0)
            
            # Combined score
            combined = (eval_score * 0.6 + power_score * 0.4)
            
            combined_scores.append({
                "hypothesis_id": hyp_id,
                "hypothesis": hyp.get("hypothesis", ""),
                "combined_score": combined,
                "evaluation_score": eval_score,
                "power_score": power_score,
                "evaluation": evaluation,
                "explanatory_power": exp_power
            })
        
        # Sort by combined score
        combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        
        if combined_scores:
            best = combined_scores[0]
            best["alternatives"] = combined_scores[1:4]  # Top 3 alternatives
            return best
        
        return {}


class CoherenceAnalyzer:
    """Analyze explanatory coherence"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_coherence(self, best_explanation: Dict[str, Any],
                         observations: List[str]) -> Dict[str, Any]:
        """Analyze how coherently the best explanation fits together"""
        prompt = f"""Analyze the explanatory coherence:

Best Explanation: {best_explanation.get('hypothesis', '')}

Observations:
{chr(10).join(f'- {obs}' for obs in observations)}

Analyze:
1. How well does this explanation form a coherent story?
2. Are there any gaps or inconsistencies?
3. Does it provide a unified understanding?
4. What evidence would strengthen it?
5. What would weaken it?

Return JSON:
{{
    "coherence_score": 0.0-1.0,
    "narrative_quality": "assessment of story quality",
    "gaps": ["gap1", "gap2"],
    "supporting_evidence_needed": ["evidence1", "evidence2"],
    "potential_challenges": ["challenge1", "challenge2"],
    "explanatory_scope": "what it explains beyond observations"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing explanatory coherence."),
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
            "coherence_score": 0.5,
            "narrative_quality": "Unknown",
            "gaps": [],
            "supporting_evidence_needed": [],
            "potential_challenges": [],
            "explanatory_scope": "Limited"
        }


# Agent functions
def initialize_abductive_reasoning(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Initialize abductive reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing abductive reasoning for {len(state['observations'])} observations"
    ))
    state["hypothesis_space"] = []
    state["explanatory_hypotheses"] = []
    state["current_step"] = "initialized"
    return state


def generate_hypotheses(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Generate explanatory hypotheses"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = HypothesisGenerator(llm)
    
    hypotheses = generator.generate_hypotheses(
        state["observations"],
        state["domain_context"]
    )
    
    state["hypothesis_space"] = hypotheses
    
    state["messages"].append(HumanMessage(
        content=f"Generated {len(hypotheses)} explanatory hypotheses"
    ))
    state["current_step"] = "hypotheses_generated"
    return state


def analyze_explanatory_power(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Analyze explanatory power"""
    analyzer = ExplanatoryPowerAnalyzer()
    
    power_analysis = analyzer.analyze_explanatory_power(
        state["hypothesis_space"],
        state["observations"]
    )
    
    state["explanatory_hypotheses"] = power_analysis
    
    avg_power = sum(h.get("explanatory_power", 0) for h in power_analysis) / len(power_analysis) if power_analysis else 0
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed explanatory power: avg {avg_power:.2f}"
    ))
    state["current_step"] = "power_analyzed"
    return state


def evaluate_hypotheses(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Evaluate hypotheses"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    evaluator = HypothesisEvaluator(llm)
    
    evaluations = evaluator.evaluate_hypotheses(
        state["hypothesis_space"],
        state["observations"]
    )
    
    state["hypothesis_evaluations"] = evaluations
    
    avg_score = sum(e.get("overall_score", 0) for e in evaluations) / len(evaluations) if evaluations else 0
    
    state["messages"].append(HumanMessage(
        content=f"Evaluated {len(evaluations)} hypotheses: avg score {avg_score:.2f}"
    ))
    state["current_step"] = "hypotheses_evaluated"
    return state


def select_best_explanation(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Select best explanation"""
    selector = BestExplanationSelector()
    
    best = selector.select_best(
        state["hypothesis_space"],
        state["hypothesis_evaluations"],
        state["explanatory_hypotheses"]
    )
    
    state["best_explanation"] = best
    state["alternative_explanations"] = best.get("alternatives", [])
    
    state["messages"].append(HumanMessage(
        content=f"Selected best explanation (score: {best.get('combined_score', 0):.2f})"
    ))
    state["current_step"] = "best_selected"
    return state


def analyze_coherence(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Analyze explanatory coherence"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    analyzer = CoherenceAnalyzer(llm)
    
    coherence = analyzer.analyze_coherence(
        state["best_explanation"],
        state["observations"]
    )
    
    state["explanatory_coherence"] = coherence
    
    state["messages"].append(HumanMessage(
        content=f"Coherence analysis: score {coherence.get('coherence_score', 0):.2f}"
    ))
    state["current_step"] = "coherence_analyzed"
    return state


def synthesize_explanation(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Synthesize final explanation"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    best = state["best_explanation"]
    coherence = state["explanatory_coherence"]
    
    prompt = f"""Synthesize a comprehensive abductive explanation:

Observations:
{chr(10).join(f'- {obs}' for obs in state['observations'])}

Best Explanation: {best.get('hypothesis', '')}
Score: {best.get('combined_score', 0):.2f}

Evaluation:
{json.dumps(best.get('evaluation', {}), indent=2)}

Coherence Analysis:
{json.dumps(coherence, indent=2)}

Create a comprehensive explanation that:
1. States the conclusion (best explanation)
2. Explains why it's the best explanation
3. Shows how it accounts for observations
4. Acknowledges limitations
5. Discusses alternatives briefly"""
    
    messages = [
        SystemMessage(content="You are an expert at synthesizing abductive explanations."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    state["messages"].append(HumanMessage(
        content=f"Synthesized explanation ({len(response.content.split())} words)"
    ))
    state["current_step"] = "explanation_synthesized"
    return state


def generate_report(state: AbductiveReasoningState) -> AbductiveReasoningState:
    """Generate final report"""
    best = state["best_explanation"]
    
    report = f"""
ABDUCTIVE REASONING REPORT
==========================

Observations ({len(state['observations'])}):
{chr(10).join(f'{i+1}. {obs}' for i, obs in enumerate(state['observations']))}

Context: {state['domain_context']}

Hypothesis Space ({len(state['hypothesis_space'])} hypotheses):

"""
    
    for hyp in state['hypothesis_space'][:5]:
        report += f"H{hyp.get('id', 0)}: {hyp.get('hypothesis', 'Unknown')}\n"
        report += f"    Plausibility: {hyp.get('initial_plausibility', 'unknown')}\n"
    
    report += f"""
Explanatory Power Analysis:
"""
    
    for exp in state['explanatory_hypotheses'][:5]:
        report += f"H{exp.get('hypothesis_id', 0)}: Power={exp.get('explanatory_power', 0):.2f} "
        report += f"(coverage={exp.get('coverage', 0):.2f}, "
        report += f"specificity={exp.get('specificity', 0):.2f}, "
        report += f"parsimony={exp.get('parsimony', 0):.2f})\n"
    
    report += f"""
Hypothesis Evaluations:
"""
    
    for ev in state['hypothesis_evaluations'][:5]:
        report += f"\nH{ev.get('hypothesis_id', 0)}: Score={ev.get('overall_score', 0):.2f}\n"
        report += f"  Consistency: {ev.get('consistency', 0):.2f}\n"
        report += f"  Completeness: {ev.get('completeness', 0):.2f}\n"
        report += f"  Simplicity: {ev.get('simplicity', 0):.2f}\n"
        report += f"  Strengths: {', '.join(ev.get('strengths', [])[:2])}\n"
    
    report += f"""
BEST EXPLANATION (Score: {best.get('combined_score', 0):.2f}):
{best.get('hypothesis', 'None selected')}

Evaluation Details:
- Evaluation Score: {best.get('evaluation_score', 0):.2f}
- Explanatory Power: {best.get('power_score', 0):.2f}
- Combined Score: {best.get('combined_score', 0):.2f}

Coherence Analysis:
- Coherence Score: {state['explanatory_coherence'].get('coherence_score', 0):.2f}
- Narrative Quality: {state['explanatory_coherence'].get('narrative_quality', 'Unknown')}
- Gaps: {len(state['explanatory_coherence'].get('gaps', []))}
- Evidence Needed: {len(state['explanatory_coherence'].get('supporting_evidence_needed', []))}

Alternative Explanations ({len(state['alternative_explanations'])}):
"""
    
    for i, alt in enumerate(state['alternative_explanations'][:3], 1):
        report += f"{i}. {alt.get('hypothesis', 'Unknown')} (score: {alt.get('combined_score', 0):.2f})\n"
    
    report += f"""
Supporting Evidence Needed:
{chr(10).join(f'- {ev}' for ev in state['explanatory_coherence'].get('supporting_evidence_needed', []))}

Potential Challenges:
{chr(10).join(f'- {ch}' for ch in state['explanatory_coherence'].get('potential_challenges', []))}

Reasoning Summary:
- Hypotheses Generated: {len(state['hypothesis_space'])}
- Best Explanation Score: {best.get('combined_score', 0):.2%}
- Coherence: {state['explanatory_coherence'].get('coherence_score', 0):.2%}
- Alternative Explanations: {len(state['alternative_explanations'])}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_abductive_reasoning_graph():
    """Create the abductive reasoning workflow graph"""
    workflow = StateGraph(AbductiveReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_abductive_reasoning)
    workflow.add_node("generate_hypotheses", generate_hypotheses)
    workflow.add_node("analyze_power", analyze_explanatory_power)
    workflow.add_node("evaluate", evaluate_hypotheses)
    workflow.add_node("select_best", select_best_explanation)
    workflow.add_node("analyze_coherence", analyze_coherence)
    workflow.add_node("synthesize", synthesize_explanation)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "generate_hypotheses")
    workflow.add_edge("generate_hypotheses", "analyze_power")
    workflow.add_edge("analyze_power", "evaluate")
    workflow.add_edge("evaluate", "select_best")
    workflow.add_edge("select_best", "analyze_coherence")
    workflow.add_edge("analyze_coherence", "synthesize")
    workflow.add_edge("synthesize", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample abductive reasoning task
    initial_state = {
        "observations": [
            "Sales dropped by 30% in the last quarter",
            "Customer complaints increased by 50%",
            "Two major competitors launched similar products at lower prices",
            "The company's social media engagement decreased significantly",
            "Employee turnover in customer service increased by 40%"
        ],
        "domain_context": "Business performance analysis and competitive dynamics",
        "hypothesis_space": [],
        "explanatory_hypotheses": [],
        "hypothesis_evaluations": [],
        "best_explanation": {},
        "alternative_explanations": [],
        "explanatory_coherence": {},
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_abductive_reasoning_graph()
    
    print("Abductive Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Best Explanation Score: {result['best_explanation'].get('combined_score', 0):.2%}")

"""
Inductive Reasoning MCP Pattern

This pattern demonstrates reasoning from specific observations to general
principles, finding patterns and forming generalizations.

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
from collections import Counter


# State definition
class InductiveReasoningState(TypedDict):
    """State for inductive reasoning workflow"""
    observations: List[Dict[str, Any]]
    domain_context: str
    patterns_identified: List[Dict[str, Any]]
    generalizations: List[Dict[str, Any]]
    inductive_strength: Dict[str, Any]
    counter_examples: List[Dict[str, Any]]
    scope_analysis: Dict[str, Any]
    refined_generalizations: List[Dict[str, Any]]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class PatternIdentifier:
    """Identify patterns in observations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def identify_patterns(self, observations: List[Dict[str, Any]],
                         context: str) -> List[Dict[str, Any]]:
        """Identify patterns across observations"""
        # Use both statistical and LLM-based pattern finding
        statistical_patterns = self._find_statistical_patterns(observations)
        semantic_patterns = self._find_semantic_patterns(observations, context)
        
        # Combine patterns
        all_patterns = statistical_patterns + semantic_patterns
        
        return all_patterns
    
    def _find_statistical_patterns(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find statistical patterns"""
        patterns = []
        
        # Extract common attributes
        if not observations:
            return patterns
        
        # Get all keys from observations
        all_keys = set()
        for obs in observations:
            all_keys.update(obs.keys())
        
        # For each key, analyze values
        for key in all_keys:
            if key in ['id', 'timestamp']:  # Skip metadata
                continue
            
            values = [obs.get(key) for obs in observations if key in obs]
            
            if values:
                # Check for frequency patterns
                value_counts = Counter(str(v) for v in values)
                most_common = value_counts.most_common(1)[0]
                
                if most_common[1] / len(values) >= 0.6:  # Appears in 60%+
                    patterns.append({
                        "type": "frequency",
                        "attribute": key,
                        "pattern": f"{key} is frequently '{most_common[0]}'",
                        "support": most_common[1] / len(values),
                        "evidence_count": most_common[1]
                    })
        
        return patterns
    
    def _find_semantic_patterns(self, observations: List[Dict[str, Any]],
                               context: str) -> List[Dict[str, Any]]:
        """Find semantic patterns using LLM"""
        # Format observations for LLM
        obs_text = json.dumps(observations[:10], indent=2)  # Limit to first 10
        
        prompt = f"""Identify patterns across these observations:

Context: {context}

Observations:
{obs_text}

Identify semantic patterns such as:
1. Recurring themes or concepts
2. Relationships between attributes
3. Conditional patterns (if X then Y)
4. Temporal patterns
5. Causal patterns

Return JSON array:
[
    {{
        "type": "semantic|relational|conditional|temporal|causal",
        "pattern": "description of pattern",
        "evidence": ["example1", "example2"],
        "confidence": 0.0-1.0
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying patterns in data."),
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


class GeneralizationBuilder:
    """Build generalizations from patterns"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def build_generalizations(self, patterns: List[Dict[str, Any]],
                             observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build general principles from patterns"""
        prompt = f"""Build generalizations from these patterns:

Patterns Identified:
{json.dumps(patterns, indent=2)}

Number of Observations: {len(observations)}

For each pattern or group of related patterns, formulate a generalization.

Return JSON array:
[
    {{
        "id": 1,
        "generalization": "general principle or rule",
        "based_on_patterns": ["pattern1", "pattern2"],
        "scope": "what it applies to",
        "strength": "universal|strong|moderate|weak",
        "confidence": 0.0-1.0,
        "exceptions_expected": "yes|no|maybe"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at formulating generalizations."),
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


class InductiveStrengthAnalyzer:
    """Analyze strength of inductive inferences"""
    
    def analyze_strength(self, generalizations: List[Dict[str, Any]],
                        observations: List[Dict[str, Any]],
                        patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the strength of inductive inferences"""
        analysis = {}
        
        # Sample size adequacy
        sample_size = len(observations)
        sample_adequacy = min(1.0, sample_size / 50)  # 50+ is adequate
        
        # Pattern consistency
        avg_confidence = sum(p.get("confidence", 0.5) for p in patterns) / len(patterns) if patterns else 0
        
        # Generalization diversity
        gen_types = set(g.get("strength", "unknown") for g in generalizations)
        diversity_score = len(gen_types) / 4  # 4 possible types
        
        # Overall inductive strength
        overall_strength = (
            sample_adequacy * 0.3 +
            avg_confidence * 0.4 +
            diversity_score * 0.3
        )
        
        analysis = {
            "overall_strength": overall_strength,
            "sample_size": sample_size,
            "sample_adequacy": sample_adequacy,
            "pattern_confidence": avg_confidence,
            "generalization_count": len(generalizations),
            "assessment": self._assess_strength(overall_strength),
            "limitations": self._identify_limitations(sample_size, patterns)
        }
        
        return analysis
    
    def _assess_strength(self, strength: float) -> str:
        """Assess inductive strength"""
        if strength >= 0.8:
            return "Strong inductive inference"
        elif strength >= 0.6:
            return "Moderate inductive inference"
        elif strength >= 0.4:
            return "Weak inductive inference"
        else:
            return "Very weak inductive inference"
    
    def _identify_limitations(self, sample_size: int,
                            patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify limitations of the inductive reasoning"""
        limitations = []
        
        if sample_size < 20:
            limitations.append("Small sample size limits generalization")
        
        if sample_size < 50:
            limitations.append("Sample may not be fully representative")
        
        semantic_patterns = [p for p in patterns if p.get("type") == "semantic"]
        if not semantic_patterns:
            limitations.append("Lacks semantic pattern analysis")
        
        if len(patterns) < 3:
            limitations.append("Limited number of patterns identified")
        
        return limitations


class CounterExampleFinder:
    """Find potential counter-examples to generalizations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def find_counter_examples(self, generalizations: List[Dict[str, Any]],
                             observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find potential counter-examples"""
        counter_examples = []
        
        for gen in generalizations:
            counter = self._find_for_generalization(gen, observations)
            if counter:
                counter_examples.append(counter)
        
        return counter_examples
    
    def _find_for_generalization(self, generalization: Dict[str, Any],
                                observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find counter-examples for a specific generalization"""
        prompt = f"""Find potential counter-examples:

Generalization: {generalization.get('generalization', '')}
Scope: {generalization.get('scope', '')}

Observations:
{json.dumps(observations[:10], indent=2)}

1. Check if any observations contradict the generalization
2. Imagine plausible counter-examples
3. Assess whether generalization needs refinement

Return JSON:
{{
    "actual_counter_examples": ["example from observations"],
    "potential_counter_examples": ["plausible scenarios"],
    "requires_refinement": true|false,
    "suggested_refinement": "how to refine if needed"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at finding counter-examples."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                counter = json.loads(content[json_start:json_end])
                counter["generalization_id"] = generalization.get("id", 0)
                return counter
            except:
                pass
        
        return None


class ScopeAnalyzer:
    """Analyze scope and applicability of generalizations"""
    
    def analyze_scope(self, generalizations: List[Dict[str, Any]],
                     observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the scope of generalizations"""
        scope_analysis = {
            "temporal_scope": self._analyze_temporal(observations),
            "domain_scope": self._analyze_domain(generalizations),
            "population_scope": self._analyze_population(observations),
            "applicability_limits": self._identify_limits(generalizations)
        }
        
        return scope_analysis
    
    def _analyze_temporal(self, observations: List[Dict[str, Any]]) -> str:
        """Analyze temporal scope"""
        # Check if observations have timestamps
        has_time = any('timestamp' in obs or 'date' in obs or 'time' in obs 
                      for obs in observations)
        
        if has_time:
            return "Time-bound - generalizations may change over time"
        else:
            return "Unknown temporal scope"
    
    def _analyze_domain(self, generalizations: List[Dict[str, Any]]) -> str:
        """Analyze domain scope"""
        scopes = [g.get("scope", "unknown") for g in generalizations]
        
        if all("specific" in s.lower() or "limited" in s.lower() for s in scopes):
            return "Narrow domain - limited applicability"
        elif all("broad" in s.lower() or "general" in s.lower() for s in scopes):
            return "Broad domain - wide applicability"
        else:
            return "Mixed domain - some generalizations narrow, some broad"
    
    def _analyze_population(self, observations: List[Dict[str, Any]]) -> str:
        """Analyze population scope"""
        sample_size = len(observations)
        
        if sample_size < 20:
            return "Small sample - limited population generalization"
        elif sample_size < 100:
            return "Moderate sample - reasonable population generalization"
        else:
            return "Large sample - strong population generalization"
    
    def _identify_limits(self, generalizations: List[Dict[str, Any]]) -> List[str]:
        """Identify applicability limits"""
        limits = []
        
        for gen in generalizations:
            if gen.get("exceptions_expected") == "yes":
                limits.append(f"Generalization '{gen.get('generalization', 'unknown')}' expects exceptions")
            
            if gen.get("strength") in ["weak", "moderate"]:
                limits.append(f"Generalization '{gen.get('generalization', 'unknown')}' has {gen.get('strength')} support")
        
        return limits


# Agent functions
def initialize_inductive_reasoning(state: InductiveReasoningState) -> InductiveReasoningState:
    """Initialize inductive reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing inductive reasoning with {len(state['observations'])} observations"
    ))
    state["patterns_identified"] = []
    state["generalizations"] = []
    state["current_step"] = "initialized"
    return state


def identify_patterns(state: InductiveReasoningState) -> InductiveReasoningState:
    """Identify patterns in observations"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    identifier = PatternIdentifier(llm)
    
    patterns = identifier.identify_patterns(
        state["observations"],
        state["domain_context"]
    )
    
    state["patterns_identified"] = patterns
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(patterns)} patterns"
    ))
    state["current_step"] = "patterns_identified"
    return state


def build_generalizations(state: InductiveReasoningState) -> InductiveReasoningState:
    """Build generalizations from patterns"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    builder = GeneralizationBuilder(llm)
    
    generalizations = builder.build_generalizations(
        state["patterns_identified"],
        state["observations"]
    )
    
    state["generalizations"] = generalizations
    
    state["messages"].append(HumanMessage(
        content=f"Built {len(generalizations)} generalizations"
    ))
    state["current_step"] = "generalizations_built"
    return state


def analyze_inductive_strength(state: InductiveReasoningState) -> InductiveReasoningState:
    """Analyze inductive strength"""
    analyzer = InductiveStrengthAnalyzer()
    
    strength = analyzer.analyze_strength(
        state["generalizations"],
        state["observations"],
        state["patterns_identified"]
    )
    
    state["inductive_strength"] = strength
    
    state["messages"].append(HumanMessage(
        content=f"Inductive strength: {strength['overall_strength']:.2f} - {strength['assessment']}"
    ))
    state["current_step"] = "strength_analyzed"
    return state


def find_counter_examples(state: InductiveReasoningState) -> InductiveReasoningState:
    """Find counter-examples"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    finder = CounterExampleFinder(llm)
    
    counter_examples = finder.find_counter_examples(
        state["generalizations"],
        state["observations"]
    )
    
    # Filter out None values
    counter_examples = [ce for ce in counter_examples if ce is not None]
    
    state["counter_examples"] = counter_examples
    
    needing_refinement = sum(1 for ce in counter_examples if ce.get("requires_refinement"))
    
    state["messages"].append(HumanMessage(
        content=f"Found counter-examples: {needing_refinement}/{len(counter_examples)} need refinement"
    ))
    state["current_step"] = "counter_examples_found"
    return state


def analyze_scope(state: InductiveReasoningState) -> InductiveReasoningState:
    """Analyze scope of generalizations"""
    analyzer = ScopeAnalyzer()
    
    scope = analyzer.analyze_scope(
        state["generalizations"],
        state["observations"]
    )
    
    state["scope_analysis"] = scope
    
    state["messages"].append(HumanMessage(
        content=f"Scope analysis complete: {scope['domain_scope']}"
    ))
    state["current_step"] = "scope_analyzed"
    return state


def refine_generalizations(state: InductiveReasoningState) -> InductiveReasoningState:
    """Refine generalizations based on counter-examples"""
    refined = []
    
    for gen in state["generalizations"]:
        gen_id = gen.get("id", 0)
        
        # Find counter-examples for this generalization
        counter = next((ce for ce in state["counter_examples"] 
                       if ce.get("generalization_id") == gen_id), None)
        
        if counter and counter.get("requires_refinement"):
            # Create refined version
            refined_gen = gen.copy()
            refined_gen["original"] = gen.get("generalization")
            refined_gen["generalization"] = counter.get("suggested_refinement", gen.get("generalization"))
            refined_gen["refined"] = True
            refined.append(refined_gen)
        else:
            # Keep original
            gen_copy = gen.copy()
            gen_copy["refined"] = False
            refined.append(gen_copy)
    
    state["refined_generalizations"] = refined
    
    refined_count = sum(1 for g in refined if g.get("refined"))
    
    state["messages"].append(HumanMessage(
        content=f"Refined {refined_count}/{len(refined)} generalizations"
    ))
    state["current_step"] = "generalizations_refined"
    return state


def generate_report(state: InductiveReasoningState) -> InductiveReasoningState:
    """Generate final report"""
    report = f"""
INDUCTIVE REASONING REPORT
==========================

Context: {state['domain_context']}
Observations: {len(state['observations'])}

Patterns Identified ({len(state['patterns_identified'])}):
"""
    
    for i, pattern in enumerate(state['patterns_identified'][:5], 1):
        report += f"{i}. {pattern.get('pattern', 'Unknown')}\n"
        report += f"   Type: {pattern.get('type', 'unknown')}\n"
        if 'support' in pattern:
            report += f"   Support: {pattern['support']:.2%}\n"
        if 'confidence' in pattern:
            report += f"   Confidence: {pattern['confidence']:.2f}\n"
    
    if len(state['patterns_identified']) > 5:
        report += f"... and {len(state['patterns_identified']) - 5} more\n"
    
    report += f"""
Generalizations Formulated ({len(state['generalizations'])}):
"""
    
    for gen in state['generalizations']:
        report += f"\n{gen.get('id', 0)}. {gen.get('generalization', 'Unknown')}\n"
        report += f"   Scope: {gen.get('scope', 'unknown')}\n"
        report += f"   Strength: {gen.get('strength', 'unknown')}\n"
        report += f"   Confidence: {gen.get('confidence', 0):.2f}\n"
    
    report += f"""
Inductive Strength Analysis:
- Overall Strength: {state['inductive_strength']['overall_strength']:.2f}
- Assessment: {state['inductive_strength']['assessment']}
- Sample Size: {state['inductive_strength']['sample_size']}
- Sample Adequacy: {state['inductive_strength']['sample_adequacy']:.2%}
- Pattern Confidence: {state['inductive_strength']['pattern_confidence']:.2f}

Limitations:
{chr(10).join(f'- {lim}' for lim in state['inductive_strength']['limitations'])}

Counter-Examples Analysis:
"""
    
    for ce in state['counter_examples']:
        gen_id = ce.get('generalization_id', 0)
        report += f"\nGeneralization {gen_id}:\n"
        if ce.get('actual_counter_examples'):
            report += f"  Actual: {len(ce['actual_counter_examples'])} found\n"
        if ce.get('potential_counter_examples'):
            report += f"  Potential: {len(ce['potential_counter_examples'])} identified\n"
        report += f"  Needs Refinement: {ce.get('requires_refinement', False)}\n"
    
    report += f"""
Scope Analysis:
- Temporal: {state['scope_analysis']['temporal_scope']}
- Domain: {state['scope_analysis']['domain_scope']}
- Population: {state['scope_analysis']['population_scope']}

Applicability Limits:
{chr(10).join(f'- {lim}' for lim in state['scope_analysis']['applicability_limits'])}

Refined Generalizations ({len(state['refined_generalizations'])}):
"""
    
    for gen in state['refined_generalizations']:
        if gen.get('refined'):
            report += f"\n{gen.get('id', 0)}. REFINED:\n"
            report += f"   Original: {gen.get('original', 'N/A')}\n"
            report += f"   Refined: {gen.get('generalization', 'N/A')}\n"
        else:
            report += f"\n{gen.get('id', 0)}. {gen.get('generalization', 'Unknown')} (unchanged)\n"
    
    report += f"""
Reasoning Summary:
- Observations: {len(state['observations'])}
- Patterns Found: {len(state['patterns_identified'])}
- Generalizations: {len(state['generalizations'])}
- Refined: {sum(1 for g in state['refined_generalizations'] if g.get('refined'))}
- Inductive Strength: {state['inductive_strength']['overall_strength']:.2%}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_inductive_reasoning_graph():
    """Create the inductive reasoning workflow graph"""
    workflow = StateGraph(InductiveReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_inductive_reasoning)
    workflow.add_node("identify_patterns", identify_patterns)
    workflow.add_node("build_generalizations", build_generalizations)
    workflow.add_node("analyze_strength", analyze_inductive_strength)
    workflow.add_node("find_counter_examples", find_counter_examples)
    workflow.add_node("analyze_scope", analyze_scope)
    workflow.add_node("refine", refine_generalizations)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "identify_patterns")
    workflow.add_edge("identify_patterns", "build_generalizations")
    workflow.add_edge("build_generalizations", "analyze_strength")
    workflow.add_edge("analyze_strength", "find_counter_examples")
    workflow.add_edge("find_counter_examples", "analyze_scope")
    workflow.add_edge("analyze_scope", "refine")
    workflow.add_edge("refine", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample inductive reasoning task
    initial_state = {
        "observations": [
            {"id": 1, "customer_type": "premium", "purchase_frequency": "high", "satisfaction": "high", "referrals": 5},
            {"id": 2, "customer_type": "premium", "purchase_frequency": "high", "satisfaction": "high", "referrals": 4},
            {"id": 3, "customer_type": "standard", "purchase_frequency": "medium", "satisfaction": "medium", "referrals": 1},
            {"id": 4, "customer_type": "premium", "purchase_frequency": "high", "satisfaction": "high", "referrals": 6},
            {"id": 5, "customer_type": "standard", "purchase_frequency": "low", "satisfaction": "low", "referrals": 0},
            {"id": 6, "customer_type": "premium", "purchase_frequency": "high", "satisfaction": "high", "referrals": 5},
            {"id": 7, "customer_type": "standard", "purchase_frequency": "medium", "satisfaction": "medium", "referrals": 2},
            {"id": 8, "customer_type": "basic", "purchase_frequency": "low", "satisfaction": "low", "referrals": 0}
        ],
        "domain_context": "Customer behavior and satisfaction analysis",
        "patterns_identified": [],
        "generalizations": [],
        "inductive_strength": {},
        "counter_examples": [],
        "scope_analysis": {},
        "refined_generalizations": [],
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_inductive_reasoning_graph()
    
    print("Inductive Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Inductive Strength: {result['inductive_strength']['overall_strength']:.2%}")

"""
Causal Reasoning MCP Pattern

This pattern demonstrates reasoning about cause-and-effect relationships,
identifying causal chains, interventions, and counterfactual scenarios.

Pattern Type: Advanced Reasoning
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class CausalReasoningState(TypedDict):
    """State for causal reasoning workflow"""
    observation: str
    domain_context: str
    causal_graph: Dict[str, Any]
    causal_chains: List[Dict[str, Any]]
    root_causes: List[Dict[str, str]]
    direct_effects: List[Dict[str, str]]
    indirect_effects: List[Dict[str, str]]
    confounding_factors: List[Dict[str, str]]
    intervention_analysis: Dict[str, Any]
    causal_explanation: str
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class CausalGraphBuilder:
    """Build causal graph from observations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def build_causal_graph(self, observation: str, context: str) -> Dict[str, Any]:
        """Build causal graph structure"""
        prompt = f"""Analyze this observation and build a causal graph:

Observation: {observation}
Context: {context}

Identify:
1. Variables involved (causes and effects)
2. Causal relationships between variables
3. Direction of causation
4. Strength of causal links

Provide a causal graph in JSON format:
{{
    "variables": [
        {{"id": "var1", "name": "Variable Name", "type": "cause|effect|mediator"}}
    ],
    "causal_links": [
        {{"from": "var1", "to": "var2", "strength": "strong|moderate|weak", "mechanism": "how it causes"}}
    ],
    "temporal_order": ["var1", "var2", "var3"]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at causal analysis and building causal graphs."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except:
                pass
        
        return {
            "variables": [],
            "causal_links": [],
            "temporal_order": []
        }


class CausalChainIdentifier:
    """Identify causal chains and pathways"""
    
    def identify_chains(self, causal_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify causal chains in the graph"""
        chains = []
        
        variables = causal_graph.get("variables", [])
        links = causal_graph.get("causal_links", [])
        
        # Build adjacency map
        adjacency = {}
        for link in links:
            from_var = link.get("from")
            to_var = link.get("to")
            
            if from_var not in adjacency:
                adjacency[from_var] = []
            
            adjacency[from_var].append({
                "to": to_var,
                "strength": link.get("strength", "unknown"),
                "mechanism": link.get("mechanism", "unknown")
            })
        
        # Find chains (paths from causes to effects)
        cause_vars = [v["id"] for v in variables if v.get("type") == "cause"]
        effect_vars = [v["id"] for v in variables if v.get("type") == "effect"]
        
        for cause in cause_vars:
            for effect in effect_vars:
                paths = self._find_paths(cause, effect, adjacency, [], set())
                
                for path in paths:
                    chains.append({
                        "start": cause,
                        "end": effect,
                        "path": path,
                        "length": len(path),
                        "type": "direct" if len(path) == 2 else "indirect"
                    })
        
        return chains
    
    def _find_paths(self, current: str, target: str, 
                   adjacency: Dict, path: List[str],
                   visited: set, max_depth: int = 5) -> List[List[str]]:
        """Find all paths from current to target"""
        if len(path) >= max_depth:
            return []
        
        path = path + [current]
        visited = visited | {current}
        
        if current == target:
            return [path]
        
        if current not in adjacency:
            return []
        
        paths = []
        for next_node in adjacency[current]:
            next_id = next_node["to"]
            if next_id not in visited:
                new_paths = self._find_paths(next_id, target, adjacency, path, visited, max_depth)
                paths.extend(new_paths)
        
        return paths


class RootCauseAnalyzer:
    """Analyze root causes"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def identify_root_causes(self, observation: str, causal_graph: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify root causes"""
        # Find variables with no incoming causal links
        variables = causal_graph.get("variables", [])
        links = causal_graph.get("causal_links", [])
        
        # Build set of variables that are effects (have incoming links)
        effect_vars = set(link.get("to") for link in links)
        
        # Root causes are variables not in effect_vars
        root_causes = []
        for var in variables:
            if var["id"] not in effect_vars:
                root_causes.append({
                    "variable": var["name"],
                    "id": var["id"],
                    "why_root": "No upstream causes identified",
                    "type": var.get("type", "unknown")
                })
        
        # Use LLM to analyze significance
        if root_causes:
            root_causes = self._analyze_root_cause_significance(
                observation, root_causes
            )
        
        return root_causes
    
    def _analyze_root_cause_significance(self, observation: str,
                                        root_causes: List[Dict]) -> List[Dict[str, str]]:
        """Analyze significance of root causes"""
        prompt = f"""Analyze the significance of these root causes:

Observation: {observation}

Root Causes Identified:
{json.dumps(root_causes, indent=2)}

For each root cause, assess:
1. How fundamental is it?
2. How actionable is it?
3. What is its relative importance?

Return JSON with added 'significance' and 'actionability' fields."""
        
        messages = [
            SystemMessage(content="You are an expert at root cause analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Try to extract enhanced root causes
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                enhanced = json.loads(json_str)
                return enhanced
            except:
                pass
        
        # Return original if parsing fails
        return root_causes


class EffectAnalyzer:
    """Analyze direct and indirect effects"""
    
    def analyze_effects(self, causal_graph: Dict[str, Any],
                       chains: List[Dict[str, Any]]) -> tuple:
        """Analyze direct and indirect effects"""
        direct_effects = []
        indirect_effects = []
        
        links = causal_graph.get("causal_links", [])
        variables = {v["id"]: v["name"] for v in causal_graph.get("variables", [])}
        
        # Direct effects: single-hop causal links
        for link in links:
            direct_effects.append({
                "cause": variables.get(link["from"], link["from"]),
                "effect": variables.get(link["to"], link["to"]),
                "strength": link.get("strength", "unknown"),
                "mechanism": link.get("mechanism", "unknown")
            })
        
        # Indirect effects: multi-hop chains
        for chain in chains:
            if chain["type"] == "indirect":
                path_names = [variables.get(v, v) for v in chain["path"]]
                indirect_effects.append({
                    "cause": path_names[0],
                    "effect": path_names[-1],
                    "pathway": " → ".join(path_names),
                    "length": chain["length"] - 1,
                    "mediators": path_names[1:-1]
                })
        
        return direct_effects, indirect_effects


class ConfoundingFactorDetector:
    """Detect potential confounding factors"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def detect_confounders(self, observation: str, 
                          causal_graph: Dict[str, Any]) -> List[Dict[str, str]]:
        """Detect potential confounding factors"""
        prompt = f"""Identify potential confounding factors:

Observation: {observation}

Causal Graph:
{json.dumps(causal_graph, indent=2)}

Identify variables or factors that:
1. Are not yet in the graph but might influence multiple variables
2. Could create spurious correlations
3. Might explain the observed relationships differently

Return JSON array:
[
    {{"factor": "name", "affects": ["var1", "var2"], "why_confounding": "explanation"}}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying confounding variables."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except:
                pass
        
        return []


class InterventionAnalyzer:
    """Analyze potential interventions"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_interventions(self, observation: str,
                             causal_graph: Dict[str, Any],
                             root_causes: List[Dict]) -> Dict[str, Any]:
        """Analyze potential causal interventions"""
        prompt = f"""Analyze potential interventions based on causal structure:

Observation: {observation}

Root Causes:
{json.dumps(root_causes, indent=2)}

Causal Structure:
Variables: {len(causal_graph.get('variables', []))}
Links: {len(causal_graph.get('causal_links', []))}

Propose interventions:
1. Which variables to intervene on
2. Expected effects of intervention
3. Potential side effects
4. Feasibility and cost

Return JSON:
{{
    "recommended_interventions": [
        {{
            "target": "variable to intervene on",
            "intervention": "type of intervention",
            "expected_effect": "what should happen",
            "side_effects": ["effect1", "effect2"],
            "feasibility": "high|medium|low"
        }}
    ],
    "intervention_strategy": "overall strategy"
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at causal intervention analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except:
                pass
        
        return {
            "recommended_interventions": [],
            "intervention_strategy": "No clear intervention strategy identified"
        }


# Agent functions
def initialize_causal_reasoning(state: CausalReasoningState) -> CausalReasoningState:
    """Initialize causal reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing causal reasoning for: {state['observation'][:100]}..."
    ))
    state["causal_chains"] = []
    state["root_causes"] = []
    state["direct_effects"] = []
    state["indirect_effects"] = []
    state["current_step"] = "initialized"
    return state


def build_causal_graph(state: CausalReasoningState) -> CausalReasoningState:
    """Build causal graph"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    builder = CausalGraphBuilder(llm)
    
    graph = builder.build_causal_graph(
        state["observation"],
        state["domain_context"]
    )
    
    state["causal_graph"] = graph
    
    state["messages"].append(HumanMessage(
        content=f"Built causal graph: {len(graph.get('variables', []))} variables, "
                f"{len(graph.get('causal_links', []))} causal links"
    ))
    state["current_step"] = "graph_built"
    return state


def identify_causal_chains(state: CausalReasoningState) -> CausalReasoningState:
    """Identify causal chains"""
    identifier = CausalChainIdentifier()
    
    chains = identifier.identify_chains(state["causal_graph"])
    
    state["causal_chains"] = chains
    
    direct_count = sum(1 for c in chains if c["type"] == "direct")
    indirect_count = sum(1 for c in chains if c["type"] == "indirect")
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(chains)} causal chains: {direct_count} direct, {indirect_count} indirect"
    ))
    state["current_step"] = "chains_identified"
    return state


def analyze_root_causes(state: CausalReasoningState) -> CausalReasoningState:
    """Analyze root causes"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = RootCauseAnalyzer(llm)
    
    root_causes = analyzer.identify_root_causes(
        state["observation"],
        state["causal_graph"]
    )
    
    state["root_causes"] = root_causes
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(root_causes)} root causes"
    ))
    state["current_step"] = "root_causes_analyzed"
    return state


def analyze_effects(state: CausalReasoningState) -> CausalReasoningState:
    """Analyze direct and indirect effects"""
    analyzer = EffectAnalyzer()
    
    direct, indirect = analyzer.analyze_effects(
        state["causal_graph"],
        state["causal_chains"]
    )
    
    state["direct_effects"] = direct
    state["indirect_effects"] = indirect
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed effects: {len(direct)} direct, {len(indirect)} indirect"
    ))
    state["current_step"] = "effects_analyzed"
    return state


def detect_confounders(state: CausalReasoningState) -> CausalReasoningState:
    """Detect confounding factors"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    detector = ConfoundingFactorDetector(llm)
    
    confounders = detector.detect_confounders(
        state["observation"],
        state["causal_graph"]
    )
    
    state["confounding_factors"] = confounders
    
    state["messages"].append(HumanMessage(
        content=f"Detected {len(confounders)} potential confounding factors"
    ))
    state["current_step"] = "confounders_detected"
    return state


def analyze_interventions(state: CausalReasoningState) -> CausalReasoningState:
    """Analyze potential interventions"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    analyzer = InterventionAnalyzer(llm)
    
    intervention_analysis = analyzer.analyze_interventions(
        state["observation"],
        state["causal_graph"],
        state["root_causes"]
    )
    
    state["intervention_analysis"] = intervention_analysis
    
    interventions = intervention_analysis.get("recommended_interventions", [])
    state["messages"].append(HumanMessage(
        content=f"Analyzed interventions: {len(interventions)} recommendations"
    ))
    state["current_step"] = "interventions_analyzed"
    return state


def generate_causal_explanation(state: CausalReasoningState) -> CausalReasoningState:
    """Generate comprehensive causal explanation"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    
    prompt = f"""Generate a comprehensive causal explanation:

Observation: {state['observation']}

Causal Structure:
- Variables: {len(state['causal_graph'].get('variables', []))}
- Causal Links: {len(state['causal_graph'].get('causal_links', []))}
- Causal Chains: {len(state['causal_chains'])}

Root Causes: {len(state['root_causes'])}
Direct Effects: {len(state['direct_effects'])}
Indirect Effects: {len(state['indirect_effects'])}
Confounders: {len(state['confounding_factors'])}

Create a clear narrative explanation of:
1. What causes what
2. The causal mechanisms involved
3. Root causes and their significance
4. Direct and indirect causal pathways
5. Potential confounding factors
6. Overall causal story"""
    
    messages = [
        SystemMessage(content="You are an expert at explaining causal relationships clearly."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    state["causal_explanation"] = response.content
    
    state["messages"].append(HumanMessage(
        content=f"Generated causal explanation ({len(response.content.split())} words)"
    ))
    state["current_step"] = "explanation_generated"
    return state


def generate_report(state: CausalReasoningState) -> CausalReasoningState:
    """Generate final causal reasoning report"""
    report = f"""
CAUSAL REASONING REPORT
=======================

Observation: {state['observation']}
Context: {state['domain_context']}

Causal Graph Structure:
- Variables: {len(state['causal_graph'].get('variables', []))}
- Causal Links: {len(state['causal_graph'].get('causal_links', []))}
- Temporal Order: {len(state['causal_graph'].get('temporal_order', []))} steps

Root Causes ({len(state['root_causes'])}):
"""
    
    for i, cause in enumerate(state['root_causes'][:5], 1):
        report += f"{i}. {cause.get('variable', 'Unknown')}\n"
        report += f"   - Why root: {cause.get('why_root', 'N/A')}\n"
        if cause.get('significance'):
            report += f"   - Significance: {cause['significance']}\n"
    
    report += f"""
Direct Causal Effects ({len(state['direct_effects'])}):
"""
    
    for i, effect in enumerate(state['direct_effects'][:5], 1):
        report += f"{i}. {effect['cause']} → {effect['effect']} ({effect['strength']})\n"
        report += f"   Mechanism: {effect['mechanism']}\n"
    
    if len(state['direct_effects']) > 5:
        report += f"... and {len(state['direct_effects']) - 5} more\n"
    
    report += f"""
Indirect Causal Effects ({len(state['indirect_effects'])}):
"""
    
    for i, effect in enumerate(state['indirect_effects'][:3], 1):
        report += f"{i}. {effect['cause']} → {effect['effect']}\n"
        report += f"   Pathway: {effect['pathway']}\n"
    
    if len(state['indirect_effects']) > 3:
        report += f"... and {len(state['indirect_effects']) - 3} more\n"
    
    report += f"""
Causal Chains ({len(state['causal_chains'])}):
- Direct chains: {sum(1 for c in state['causal_chains'] if c['type'] == 'direct')}
- Indirect chains: {sum(1 for c in state['causal_chains'] if c['type'] == 'indirect')}

Confounding Factors ({len(state['confounding_factors'])}):
"""
    
    for i, confounder in enumerate(state['confounding_factors'], 1):
        report += f"{i}. {confounder.get('factor', 'Unknown')}\n"
        report += f"   - Affects: {', '.join(confounder.get('affects', []))}\n"
        report += f"   - Why confounding: {confounder.get('why_confounding', 'N/A')}\n"
    
    report += f"""
Intervention Analysis:

Strategy: {state['intervention_analysis'].get('intervention_strategy', 'None')}

Recommended Interventions:
"""
    
    for i, intervention in enumerate(state['intervention_analysis'].get('recommended_interventions', []), 1):
        report += f"{i}. Target: {intervention.get('target', 'Unknown')}\n"
        report += f"   - Intervention: {intervention.get('intervention', 'N/A')}\n"
        report += f"   - Expected Effect: {intervention.get('expected_effect', 'N/A')}\n"
        report += f"   - Feasibility: {intervention.get('feasibility', 'Unknown')}\n"
    
    report += f"""
CAUSAL EXPLANATION:
{'-' * 50}
{state['causal_explanation']}
{'-' * 50}

Reasoning Summary:
- Causal Complexity: {len(state['causal_graph'].get('causal_links', []))} links
- Root Causes Identified: {len(state['root_causes'])}
- Effect Pathways: {len(state['direct_effects'])} + {len(state['indirect_effects'])}
- Intervention Points: {len(state['intervention_analysis'].get('recommended_interventions', []))}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_causal_reasoning_graph():
    """Create the causal reasoning workflow graph"""
    workflow = StateGraph(CausalReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_causal_reasoning)
    workflow.add_node("build_graph", build_causal_graph)
    workflow.add_node("identify_chains", identify_causal_chains)
    workflow.add_node("analyze_roots", analyze_root_causes)
    workflow.add_node("analyze_effects", analyze_effects)
    workflow.add_node("detect_confounders", detect_confounders)
    workflow.add_node("analyze_interventions", analyze_interventions)
    workflow.add_node("generate_explanation", generate_causal_explanation)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "build_graph")
    workflow.add_edge("build_graph", "identify_chains")
    workflow.add_edge("identify_chains", "analyze_roots")
    workflow.add_edge("analyze_roots", "analyze_effects")
    workflow.add_edge("analyze_effects", "detect_confounders")
    workflow.add_edge("detect_confounders", "analyze_interventions")
    workflow.add_edge("analyze_interventions", "generate_explanation")
    workflow.add_edge("generate_explanation", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample causal reasoning task
    initial_state = {
        "observation": "Students who attend tutoring sessions perform better on exams, but the correlation seems stronger for students from higher-income families",
        "domain_context": "Educational outcomes and intervention effectiveness",
        "causal_graph": {},
        "causal_chains": [],
        "root_causes": [],
        "direct_effects": [],
        "indirect_effects": [],
        "confounding_factors": [],
        "intervention_analysis": {},
        "causal_explanation": "",
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_causal_reasoning_graph()
    
    print("Causal Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Causal Links: {len(result['causal_graph'].get('causal_links', []))}")

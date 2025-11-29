"""
Controlled Generation MCP Pattern

This pattern demonstrates content generation with control parameters including style,
tone, sentiment, formality, and creativity levels for fine-grained output control.

Pattern Type: Generation
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
class ControlledGenerationState(TypedDict):
    """State for controlled generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    control_parameters: Dict[str, float]
    style_profile: Dict[str, Any]
    generated_variants: List[Dict[str, Any]]
    control_analysis: Dict[str, Any]
    selected_variant: Dict[str, Any]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class ControlParameterManager:
    """Manage generation control parameters"""
    
    def __init__(self):
        self.parameters = self._initialize_parameters()
    
    def _initialize_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize control parameter definitions"""
        return {
            "formality": {
                "description": "Level of formal vs casual language",
                "range": [0.0, 1.0],
                "levels": {
                    0.0: "very casual",
                    0.25: "casual",
                    0.5: "neutral",
                    0.75: "formal",
                    1.0: "very formal"
                },
                "indicators": {
                    "formal": ["utilize", "accordingly", "therefore", "furthermore", "consequently"],
                    "casual": ["use", "so", "also", "besides", "that's why"]
                }
            },
            "creativity": {
                "description": "Level of creative vs conventional expression",
                "range": [0.0, 1.0],
                "levels": {
                    0.0: "very conventional",
                    0.25: "conventional",
                    0.5: "balanced",
                    0.75: "creative",
                    1.0: "very creative"
                },
                "indicators": {
                    "creative": ["metaphor", "analogy", "imagine", "envision", "like"],
                    "conventional": ["standard", "typical", "normal", "usual", "traditional"]
                }
            },
            "sentiment": {
                "description": "Emotional tone from negative to positive",
                "range": [-1.0, 1.0],
                "levels": {
                    -1.0: "very negative",
                    -0.5: "negative",
                    0.0: "neutral",
                    0.5: "positive",
                    1.0: "very positive"
                },
                "indicators": {
                    "positive": ["excellent", "great", "wonderful", "amazing", "fantastic"],
                    "negative": ["poor", "bad", "terrible", "awful", "disappointing"]
                }
            },
            "technicality": {
                "description": "Level of technical vs layman language",
                "range": [0.0, 1.0],
                "levels": {
                    0.0: "very simple",
                    0.25: "simple",
                    0.5: "moderate",
                    0.75: "technical",
                    1.0: "very technical"
                },
                "indicators": {
                    "technical": ["algorithm", "protocol", "methodology", "paradigm", "infrastructure"],
                    "simple": ["way", "method", "how", "approach", "system"]
                }
            },
            "conciseness": {
                "description": "Level of concise vs elaborate expression",
                "range": [0.0, 1.0],
                "levels": {
                    0.0: "very elaborate",
                    0.25: "elaborate",
                    0.5: "balanced",
                    0.75: "concise",
                    1.0: "very concise"
                },
                "impact": "word_count"
            },
            "assertiveness": {
                "description": "Level of assertive vs tentative language",
                "range": [0.0, 1.0],
                "levels": {
                    0.0: "very tentative",
                    0.25: "tentative",
                    0.5: "neutral",
                    0.75: "assertive",
                    1.0: "very assertive"
                },
                "indicators": {
                    "assertive": ["will", "must", "definitely", "certainly", "undoubtedly"],
                    "tentative": ["might", "could", "perhaps", "possibly", "may"]
                }
            }
        }
    
    def get_parameter_description(self, param_name: str, value: float) -> str:
        """Get human-readable description of parameter value"""
        if param_name not in self.parameters:
            return "Unknown parameter"
        
        param = self.parameters[param_name]
        levels = param["levels"]
        
        # Find closest level
        closest_level = min(levels.keys(), key=lambda k: abs(k - value))
        return levels[closest_level]
    
    def build_control_prompt(self, parameters: Dict[str, float]) -> str:
        """Build prompt section describing control parameters"""
        prompt_parts = ["Control Parameters:"]
        
        for param_name, value in parameters.items():
            if param_name in self.parameters:
                description = self.get_parameter_description(param_name, value)
                prompt_parts.append(f"- {param_name.capitalize()}: {description} ({value:.2f})")
        
        return "\n".join(prompt_parts)


class StyleProfileBuilder:
    """Build comprehensive style profiles from control parameters"""
    
    def build_profile(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """Build style profile from parameters"""
        profile = {
            "parameters": parameters,
            "style_guide": self._generate_style_guide(parameters),
            "example_phrases": self._generate_example_phrases(parameters),
            "avoid_list": self._generate_avoid_list(parameters)
        }
        
        return profile
    
    def _generate_style_guide(self, parameters: Dict[str, float]) -> List[str]:
        """Generate style guidelines"""
        guidelines = []
        
        # Formality
        if parameters.get("formality", 0.5) >= 0.7:
            guidelines.append("Use formal, professional language")
            guidelines.append("Avoid contractions and colloquialisms")
        elif parameters.get("formality", 0.5) <= 0.3:
            guidelines.append("Use casual, conversational language")
            guidelines.append("Contractions and informal expressions welcome")
        
        # Creativity
        if parameters.get("creativity", 0.5) >= 0.7:
            guidelines.append("Use creative metaphors and analogies")
            guidelines.append("Employ vivid, imaginative language")
        elif parameters.get("creativity", 0.5) <= 0.3:
            guidelines.append("Stick to conventional, straightforward expression")
            guidelines.append("Focus on clarity over creativity")
        
        # Sentiment
        sentiment = parameters.get("sentiment", 0.0)
        if sentiment >= 0.5:
            guidelines.append("Maintain positive, optimistic tone")
            guidelines.append("Highlight benefits and opportunities")
        elif sentiment <= -0.5:
            guidelines.append("Address challenges and concerns")
            guidelines.append("Be realistic about limitations")
        
        # Technicality
        if parameters.get("technicality", 0.5) >= 0.7:
            guidelines.append("Use technical terminology and jargon")
            guidelines.append("Include specific technical details")
        elif parameters.get("technicality", 0.5) <= 0.3:
            guidelines.append("Use simple, accessible language")
            guidelines.append("Explain concepts in everyday terms")
        
        # Conciseness
        if parameters.get("conciseness", 0.5) >= 0.7:
            guidelines.append("Be brief and to the point")
            guidelines.append("Eliminate unnecessary words")
        elif parameters.get("conciseness", 0.5) <= 0.3:
            guidelines.append("Provide detailed explanations")
            guidelines.append("Include context and background")
        
        # Assertiveness
        if parameters.get("assertiveness", 0.5) >= 0.7:
            guidelines.append("Use confident, definitive statements")
            guidelines.append("Avoid hedging language")
        elif parameters.get("assertiveness", 0.5) <= 0.3:
            guidelines.append("Use tentative, qualified language")
            guidelines.append("Acknowledge uncertainty where appropriate")
        
        return guidelines
    
    def _generate_example_phrases(self, parameters: Dict[str, float]) -> Dict[str, List[str]]:
        """Generate example phrases for the style"""
        examples = {}
        
        # Formality examples
        if parameters.get("formality", 0.5) >= 0.7:
            examples["formality"] = [
                "We would like to inform you that...",
                "It is imperative to note that...",
                "Upon careful consideration..."
            ]
        else:
            examples["formality"] = [
                "We want to let you know...",
                "It's important to remember...",
                "When you think about it..."
            ]
        
        # Assertiveness examples
        if parameters.get("assertiveness", 0.5) >= 0.7:
            examples["assertiveness"] = [
                "This will definitely improve...",
                "The solution is clearly...",
                "We must take action..."
            ]
        else:
            examples["assertiveness"] = [
                "This might improve...",
                "The solution could be...",
                "We should consider..."
            ]
        
        return examples
    
    def _generate_avoid_list(self, parameters: Dict[str, float]) -> List[str]:
        """Generate list of things to avoid"""
        avoid = []
        
        if parameters.get("formality", 0.5) >= 0.7:
            avoid.extend(["contractions", "slang", "colloquialisms"])
        
        if parameters.get("creativity", 0.5) <= 0.3:
            avoid.extend(["unusual metaphors", "flowery language"])
        
        if parameters.get("sentiment", 0.0) >= 0.5:
            avoid.extend(["negative language", "criticism"])
        elif parameters.get("sentiment", 0.0) <= -0.5:
            avoid.extend(["overly optimistic claims", "unrealistic promises"])
        
        if parameters.get("conciseness", 0.5) >= 0.7:
            avoid.extend(["redundancy", "wordiness", "unnecessary details"])
        
        return avoid


class ControlledContentGenerator:
    """Generate content with specific control parameters"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.param_manager = ControlParameterManager()
    
    def generate_variant(self, requirements: Dict[str, Any],
                        parameters: Dict[str, float],
                        style_profile: Dict[str, Any]) -> str:
        """Generate single variant with specific controls"""
        control_prompt = self.param_manager.build_control_prompt(parameters)
        
        style_guide = "\n".join([f"  • {g}" for g in style_profile["style_guide"]])
        avoid_list = ", ".join(style_profile["avoid_list"])
        
        prompt = f"""Generate content following these precise control parameters:

{control_prompt}

Style Guidelines:
{style_guide}

Things to Avoid: {avoid_list}

Content Requirements:
- Topic: {requirements.get('topic', 'general')}
- Purpose: {requirements.get('purpose', 'inform')}
- Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 300)} words
- Audience: {requirements.get('audience', 'general')}

Generate content that precisely matches ALL control parameters."""
        
        messages = [
            SystemMessage(content="You are a precision content generator that exactly matches specified control parameters."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class ControlAnalyzer:
    """Analyze how well generated content matches control parameters"""
    
    def __init__(self):
        self.param_manager = ControlParameterManager()
    
    def analyze_control_adherence(self, content: str,
                                  target_parameters: Dict[str, float]) -> Dict[str, Any]:
        """Analyze how well content matches target parameters"""
        results = {}
        
        for param_name, target_value in target_parameters.items():
            if param_name in self.param_manager.parameters:
                measured_value = self._measure_parameter(content, param_name)
                deviation = abs(target_value - measured_value)
                
                results[param_name] = {
                    "target": target_value,
                    "measured": measured_value,
                    "deviation": deviation,
                    "adherence_score": max(0.0, 1.0 - deviation)
                }
        
        # Calculate overall adherence
        if results:
            overall_score = sum(r["adherence_score"] for r in results.values()) / len(results)
        else:
            overall_score = 0.0
        
        return {
            "parameter_results": results,
            "overall_adherence": overall_score
        }
    
    def _measure_parameter(self, content: str, param_name: str) -> float:
        """Measure actual parameter value in content"""
        param_def = self.param_manager.parameters[param_name]
        
        if "indicators" not in param_def:
            return 0.5  # Default neutral value
        
        indicators = param_def["indicators"]
        content_lower = content.lower()
        
        # Count indicator occurrences
        if param_name in ["formality", "creativity", "technicality", "assertiveness"]:
            high_key = list(indicators.keys())[0]
            low_key = list(indicators.keys())[1]
            
            high_count = sum(1 for word in indicators[high_key] if word in content_lower)
            low_count = sum(1 for word in indicators[low_key] if word in content_lower)
            
            total_count = high_count + low_count
            if total_count == 0:
                return 0.5
            
            return high_count / total_count
        
        elif param_name == "sentiment":
            pos_count = sum(1 for word in indicators["positive"] if word in content_lower)
            neg_count = sum(1 for word in indicators["negative"] if word in content_lower)
            
            total_count = pos_count + neg_count
            if total_count == 0:
                return 0.0
            
            # Map to -1.0 to 1.0 range
            return (pos_count - neg_count) / total_count
        
        elif param_name == "conciseness":
            # Measure by average sentence length
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            if not sentences:
                return 0.5
            
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            
            # Map sentence length to conciseness (shorter = more concise)
            # Optimal concise: 10-15 words, elaborate: 20-30 words
            if avg_length <= 15:
                return 0.8  # Concise
            elif avg_length >= 25:
                return 0.2  # Elaborate
            else:
                return 0.5  # Balanced
        
        return 0.5


# Agent functions
def initialize_generation(state: ControlledGenerationState) -> ControlledGenerationState:
    """Initialize controlled generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing controlled generation: {state['task_description']}"
    ))
    state["current_step"] = "initialized"
    return state


def setup_controls(state: ControlledGenerationState) -> ControlledGenerationState:
    """Set up control parameters and style profile"""
    # Get control parameters from requirements
    parameters = state["generation_requirements"].get("control_parameters", {
        "formality": 0.7,
        "creativity": 0.5,
        "sentiment": 0.5,
        "technicality": 0.6,
        "conciseness": 0.6,
        "assertiveness": 0.7
    })
    
    # Build style profile
    profile_builder = StyleProfileBuilder()
    style_profile = profile_builder.build_profile(parameters)
    
    state["control_parameters"] = parameters
    state["style_profile"] = style_profile
    
    param_manager = ControlParameterManager()
    param_desc = [
        f"{name}: {param_manager.get_parameter_description(name, value)}"
        for name, value in parameters.items()
    ]
    
    state["messages"].append(HumanMessage(
        content=f"Configured {len(parameters)} control parameters: {', '.join(param_desc)}"
    ))
    state["current_step"] = "controls_setup"
    return state


def generate_variants(state: ControlledGenerationState) -> ControlledGenerationState:
    """Generate multiple variants with different control settings"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = ControlledContentGenerator(llm)
    
    # Generate primary variant with target parameters
    primary_content = generator.generate_variant(
        state["generation_requirements"],
        state["control_parameters"],
        state["style_profile"]
    )
    
    variants = [
        {
            "name": "primary",
            "parameters": state["control_parameters"].copy(),
            "content": primary_content
        }
    ]
    
    # Generate alternative variants if requested
    num_variants = state["generation_requirements"].get("num_variants", 1)
    
    if num_variants > 1:
        # Create slight variations
        base_params = state["control_parameters"].copy()
        
        # More creative variant
        creative_params = base_params.copy()
        creative_params["creativity"] = min(1.0, creative_params.get("creativity", 0.5) + 0.3)
        
        creative_profile = StyleProfileBuilder().build_profile(creative_params)
        creative_content = generator.generate_variant(
            state["generation_requirements"],
            creative_params,
            creative_profile
        )
        
        variants.append({
            "name": "creative",
            "parameters": creative_params,
            "content": creative_content
        })
    
    if num_variants > 2:
        # More concise variant
        concise_params = base_params.copy()
        concise_params["conciseness"] = min(1.0, concise_params.get("conciseness", 0.5) + 0.3)
        
        concise_profile = StyleProfileBuilder().build_profile(concise_params)
        concise_content = generator.generate_variant(
            state["generation_requirements"],
            concise_params,
            concise_profile
        )
        
        variants.append({
            "name": "concise",
            "parameters": concise_params,
            "content": concise_content
        })
    
    state["generated_variants"] = variants
    state["messages"].append(HumanMessage(
        content=f"Generated {len(variants)} variant(s)"
    ))
    state["current_step"] = "variants_generated"
    return state


def analyze_variants(state: ControlledGenerationState) -> ControlledGenerationState:
    """Analyze control adherence of all variants"""
    analyzer = ControlAnalyzer()
    
    analyses = []
    for variant in state["generated_variants"]:
        analysis = analyzer.analyze_control_adherence(
            variant["content"],
            variant["parameters"]
        )
        
        analyses.append({
            "variant_name": variant["name"],
            "analysis": analysis,
            "overall_adherence": analysis["overall_adherence"]
        })
    
    state["control_analysis"] = {
        "variant_analyses": analyses
    }
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed {len(analyses)} variant(s) for control adherence"
    ))
    state["current_step"] = "variants_analyzed"
    return state


def select_best_variant(state: ControlledGenerationState) -> ControlledGenerationState:
    """Select best variant based on control adherence"""
    # Find variant with highest adherence
    best_variant = None
    best_adherence = 0.0
    
    for i, analysis in enumerate(state["control_analysis"]["variant_analyses"]):
        if analysis["overall_adherence"] > best_adherence:
            best_adherence = analysis["overall_adherence"]
            best_variant = state["generated_variants"][i]
    
    if not best_variant:
        best_variant = state["generated_variants"][0]
    
    state["selected_variant"] = best_variant
    state["final_output"] = best_variant["content"]
    
    state["messages"].append(HumanMessage(
        content=f"Selected '{best_variant['name']}' variant (adherence: {best_adherence:.1%})"
    ))
    state["current_step"] = "variant_selected"
    return state


def generate_report(state: ControlledGenerationState) -> ControlledGenerationState:
    """Generate final controlled generation report"""
    selected_analysis = None
    for analysis in state["control_analysis"]["variant_analyses"]:
        if analysis["variant_name"] == state["selected_variant"]["name"]:
            selected_analysis = analysis
            break
    
    report = f"""
CONTROLLED GENERATION REPORT
============================

Task: {state['task_description']}

Control Parameters (Target):
"""
    
    param_manager = ControlParameterManager()
    for param, value in state["control_parameters"].items():
        desc = param_manager.get_parameter_description(param, value)
        report += f"- {param.capitalize()}: {value:.2f} ({desc})\n"
    
    report += f"""
Generated Variants: {len(state['generated_variants'])}
Selected Variant: {state['selected_variant']['name']}

Control Adherence Analysis:
"""
    
    if selected_analysis:
        for param, result in selected_analysis["analysis"]["parameter_results"].items():
            report += f"- {param.capitalize()}:\n"
            report += f"    Target: {result['target']:.2f} | Measured: {result['measured']:.2f} | "
            report += f"Adherence: {result['adherence_score']:.1%}\n"
        
        report += f"\nOverall Control Adherence: {selected_analysis['overall_adherence']:.1%}\n"
    
    report += f"""
GENERATED CONTENT ({state['selected_variant']['name']} variant):
{'-' * 50}
{state['final_output']}
{'-' * 50}

Content Stats:
- Words: {len(state['final_output'].split())}
- Sentences: {len([s for s in state['final_output'].split('.') if s.strip()])}
"""
    
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_controlled_generation_graph():
    """Create the controlled generation workflow graph"""
    workflow = StateGraph(ControlledGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("setup_controls", setup_controls)
    workflow.add_node("generate_variants", generate_variants)
    workflow.add_node("analyze_variants", analyze_variants)
    workflow.add_node("select_variant", select_best_variant)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "setup_controls")
    workflow.add_edge("setup_controls", "generate_variants")
    workflow.add_edge("generate_variants", "analyze_variants")
    workflow.add_edge("analyze_variants", "select_variant")
    workflow.add_edge("select_variant", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample controlled generation task
    initial_state = {
        "task_description": "Generate a controlled business proposal",
        "generation_requirements": {
            "topic": "AI-Powered Customer Service Platform",
            "purpose": "business proposal",
            "audience": "executive leadership",
            "min_words": 200,
            "max_words": 350,
            "num_variants": 3,
            "control_parameters": {
                "formality": 0.85,       # Very formal
                "creativity": 0.6,       # Moderately creative
                "sentiment": 0.7,        # Positive
                "technicality": 0.7,     # Technical
                "conciseness": 0.7,      # Concise
                "assertiveness": 0.8     # Assertive
            }
        },
        "control_parameters": {},
        "style_profile": {},
        "generated_variants": [],
        "control_analysis": {},
        "selected_variant": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_controlled_generation_graph()
    
    print("Controlled Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")

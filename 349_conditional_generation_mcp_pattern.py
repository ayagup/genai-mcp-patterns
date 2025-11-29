"""
Conditional Generation MCP Pattern

This pattern demonstrates content generation with conditional logic, branching
based on context, requirements, and dynamic decision points during generation.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any, Callable
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
import re


# State definition
class ConditionalGenerationState(TypedDict):
    """State for conditional generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    context_analysis: Dict[str, Any]
    decision_points: List[Dict[str, Any]]
    generation_path: List[str]
    conditional_branches: Dict[str, Any]
    generated_sections: Dict[str, str]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class ContextAnalyzer:
    """Analyze context to inform conditional decisions"""
    
    def analyze_context(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze generation context"""
        return {
            "audience_level": self._determine_audience_level(requirements),
            "content_depth": self._determine_content_depth(requirements),
            "format_preference": self._determine_format(requirements),
            "tone_requirement": self._determine_tone(requirements),
            "length_category": self._categorize_length(requirements)
        }
    
    def _determine_audience_level(self, requirements: Dict[str, Any]) -> str:
        """Determine audience expertise level"""
        audience = requirements.get('audience', 'general').lower()
        
        if any(term in audience for term in ['expert', 'professional', 'technical']):
            return "expert"
        elif any(term in audience for term in ['intermediate', 'practitioner']):
            return "intermediate"
        else:
            return "beginner"
    
    def _determine_content_depth(self, requirements: Dict[str, Any]) -> str:
        """Determine required content depth"""
        purpose = requirements.get('purpose', 'inform').lower()
        
        if any(term in purpose for term in ['comprehensive', 'detailed', 'in-depth']):
            return "deep"
        elif any(term in purpose for term in ['overview', 'introduction', 'summary']):
            return "shallow"
        else:
            return "moderate"
    
    def _determine_format(self, requirements: Dict[str, Any]) -> str:
        """Determine content format preference"""
        format_pref = requirements.get('format', 'article').lower()
        
        format_map = {
            'list': 'list_based',
            'tutorial': 'step_by_step',
            'story': 'narrative',
            'report': 'formal',
            'guide': 'instructional'
        }
        
        for key, value in format_map.items():
            if key in format_pref:
                return value
        
        return 'standard'
    
    def _determine_tone(self, requirements: Dict[str, Any]) -> str:
        """Determine tone requirement"""
        tone = requirements.get('tone', 'professional').lower()
        
        if any(term in tone for term in ['casual', 'friendly', 'conversational']):
            return "casual"
        elif any(term in tone for term in ['formal', 'professional', 'academic']):
            return "formal"
        else:
            return "neutral"
    
    def _categorize_length(self, requirements: Dict[str, Any]) -> str:
        """Categorize target length"""
        max_words = requirements.get('max_words', 500)
        
        if max_words < 200:
            return "brief"
        elif max_words < 500:
            return "standard"
        else:
            return "extended"


class ConditionalDecisionEngine:
    """Make conditional decisions during generation"""
    
    def __init__(self):
        self.decision_rules = self._initialize_decision_rules()
    
    def _initialize_decision_rules(self) -> Dict[str, Callable]:
        """Initialize conditional decision rules"""
        return {
            "include_technical_details": lambda ctx: ctx["audience_level"] in ["expert", "intermediate"],
            "include_examples": lambda ctx: ctx["audience_level"] in ["beginner", "intermediate"],
            "use_analogies": lambda ctx: ctx["audience_level"] == "beginner",
            "include_code_samples": lambda ctx: ctx["format_preference"] in ["tutorial", "step_by_step"],
            "add_visual_descriptions": lambda ctx: ctx["format_preference"] == "narrative",
            "include_citations": lambda ctx: ctx["tone_requirement"] == "formal",
            "add_conversational_elements": lambda ctx: ctx["tone_requirement"] == "casual",
            "deep_dive_sections": lambda ctx: ctx["content_depth"] == "deep",
            "quick_tips": lambda ctx: ctx["length_category"] == "brief",
            "extended_analysis": lambda ctx: ctx["length_category"] == "extended",
            "step_by_step_breakdown": lambda ctx: ctx["format_preference"] == "step_by_step",
            "bullet_point_format": lambda ctx: ctx["format_preference"] == "list_based"
        }
    
    def make_decision(self, decision_id: str, context: Dict[str, Any]) -> bool:
        """Make a conditional decision"""
        rule = self.decision_rules.get(decision_id)
        if rule:
            return rule(context)
        return False
    
    def evaluate_all_decisions(self, context: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate all decision points"""
        return {
            decision_id: self.make_decision(decision_id, context)
            for decision_id in self.decision_rules.keys()
        }


class ConditionalContentGenerator:
    """Generate content based on conditional logic"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_section(self, section_name: str, requirements: Dict[str, Any],
                        context: Dict[str, Any], decisions: Dict[str, bool],
                        previous_sections: Dict[str, str]) -> str:
        """Generate section based on conditional logic"""
        # Build conditional instructions
        conditional_instructions = self._build_conditional_instructions(decisions)
        
        # Build context from previous sections
        previous_context = self._build_previous_context(previous_sections)
        
        prompt = f"""Generate the {section_name} section based on these conditions:

Topic: {requirements.get('topic', 'general')}

Context Analysis:
- Audience Level: {context['audience_level']}
- Content Depth: {context['content_depth']}
- Format: {context['format_preference']}
- Tone: {context['tone_requirement']}

Conditional Instructions (IMPORTANT - Follow these):
{conditional_instructions}

{previous_context}

Generate the {section_name} section following ALL conditional instructions."""
        
        messages = [
            SystemMessage(content="You are a conditional content generator that adapts output based on context and rules."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _build_conditional_instructions(self, decisions: Dict[str, bool]) -> str:
        """Build human-readable conditional instructions"""
        instructions = []
        
        instruction_map = {
            "include_technical_details": "✓ Include technical details and terminology",
            "include_examples": "✓ Include concrete examples",
            "use_analogies": "✓ Use analogies to explain concepts",
            "include_code_samples": "✓ Include code samples where relevant",
            "add_visual_descriptions": "✓ Add vivid visual descriptions",
            "include_citations": "✓ Include citations and references",
            "add_conversational_elements": "✓ Use conversational language",
            "deep_dive_sections": "✓ Provide in-depth analysis",
            "quick_tips": "✓ Focus on quick, actionable tips",
            "extended_analysis": "✓ Provide extended analysis",
            "step_by_step_breakdown": "✓ Use step-by-step breakdown",
            "bullet_point_format": "✓ Use bullet-point format"
        }
        
        for decision_id, should_include in decisions.items():
            if should_include and decision_id in instruction_map:
                instructions.append(instruction_map[decision_id])
            elif not should_include and decision_id in instruction_map:
                negative_instruction = instruction_map[decision_id].replace("✓", "✗")
                instructions.append(negative_instruction)
        
        return "\n".join(instructions) if instructions else "No special conditions"
    
    def _build_previous_context(self, previous_sections: Dict[str, str]) -> str:
        """Build context from previous sections"""
        if not previous_sections:
            return "This is the first section."
        
        context = "Previous Sections:\n"
        for section_name, content in previous_sections.items():
            preview = content[:200] + "..." if len(content) > 200 else content
            context += f"\n{section_name}:\n{preview}\n"
        
        return context


class GenerationPathPlanner:
    """Plan generation path based on conditions"""
    
    def plan_path(self, context: Dict[str, Any], decisions: Dict[str, bool],
                 requirements: Dict[str, Any]) -> List[str]:
        """Plan which sections to generate"""
        sections = ["introduction"]
        
        # Conditional section inclusion
        if decisions.get("include_technical_details"):
            sections.append("technical_background")
        
        if decisions.get("include_examples"):
            sections.append("examples")
        
        if decisions.get("step_by_step_breakdown"):
            sections.extend(["step1", "step2", "step3"])
        else:
            sections.append("main_content")
        
        if decisions.get("deep_dive_sections"):
            sections.append("detailed_analysis")
        
        if decisions.get("use_analogies"):
            sections.append("analogies_section")
        
        if decisions.get("quick_tips"):
            sections.append("quick_tips")
        
        if decisions.get("include_code_samples"):
            sections.append("code_examples")
        
        sections.append("conclusion")
        
        return sections


# Agent functions
def initialize_generation(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Initialize conditional generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing conditional generation: {state['task_description']}"
    ))
    state["generation_path"] = []
    state["generated_sections"] = {}
    state["current_step"] = "initialized"
    return state


def analyze_context(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Analyze generation context"""
    analyzer = ContextAnalyzer()
    
    context = analyzer.analyze_context(state["generation_requirements"])
    
    state["context_analysis"] = context
    
    state["messages"].append(HumanMessage(
        content=f"Context: audience={context['audience_level']}, "
                f"depth={context['content_depth']}, "
                f"format={context['format_preference']}, "
                f"tone={context['tone_requirement']}"
    ))
    state["current_step"] = "context_analyzed"
    return state


def make_conditional_decisions(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Make conditional decisions"""
    engine = ConditionalDecisionEngine()
    
    decisions = engine.evaluate_all_decisions(state["context_analysis"])
    
    # Convert to decision points
    decision_points = [
        {
            "decision_id": decision_id,
            "condition": decision_id.replace("_", " "),
            "result": result
        }
        for decision_id, result in decisions.items()
    ]
    
    state["decision_points"] = decision_points
    state["conditional_branches"] = decisions
    
    active_decisions = [d["decision_id"] for d in decision_points if d["result"]]
    state["messages"].append(HumanMessage(
        content=f"Made {len(decision_points)} decisions, {len(active_decisions)} active: "
                f"{', '.join(active_decisions[:5])}"
    ))
    state["current_step"] = "decisions_made"
    return state


def plan_generation_path(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Plan generation path based on decisions"""
    planner = GenerationPathPlanner()
    
    path = planner.plan_path(
        state["context_analysis"],
        state["conditional_branches"],
        state["generation_requirements"]
    )
    
    state["generation_path"] = path
    
    state["messages"].append(HumanMessage(
        content=f"Planned {len(path)}-section path: {' → '.join(path)}"
    ))
    state["current_step"] = "path_planned"
    return state


def generate_sections(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Generate all sections based on path"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = ConditionalContentGenerator(llm)
    
    for section_name in state["generation_path"]:
        section_content = generator.generate_section(
            section_name,
            state["generation_requirements"],
            state["context_analysis"],
            state["conditional_branches"],
            state["generated_sections"]
        )
        
        state["generated_sections"][section_name] = section_content
        
        state["messages"].append(HumanMessage(
            content=f"Generated {section_name}: {len(section_content.split())} words"
        ))
    
    state["current_step"] = "sections_generated"
    return state


def assemble_output(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Assemble final output from sections"""
    # Combine all sections
    output_parts = []
    
    for section_name in state["generation_path"]:
        if section_name in state["generated_sections"]:
            output_parts.append(state["generated_sections"][section_name])
    
    final_output = "\n\n".join(output_parts)
    
    state["final_output"] = final_output
    
    state["messages"].append(HumanMessage(
        content=f"Assembled {len(state['generation_path'])} sections into final output"
    ))
    state["current_step"] = "output_assembled"
    return state


def generate_report(state: ConditionalGenerationState) -> ConditionalGenerationState:
    """Generate final conditional generation report"""
    active_decisions = [d for d in state["decision_points"] if d["result"]]
    inactive_decisions = [d for d in state["decision_points"] if not d["result"]]
    
    report = f"""
CONDITIONAL GENERATION REPORT
==============================

Task: {state['task_description']}

Context Analysis:
- Audience Level: {state['context_analysis']['audience_level']}
- Content Depth: {state['context_analysis']['content_depth']}
- Format Preference: {state['context_analysis']['format_preference']}
- Tone Requirement: {state['context_analysis']['tone_requirement']}
- Length Category: {state['context_analysis']['length_category']}

Conditional Decisions ({len(active_decisions)} active / {len(state['decision_points'])} total):

Active Conditions:
"""
    
    for decision in active_decisions:
        report += f"✓ {decision['condition']}\n"
    
    report += "\nInactive Conditions:\n"
    for decision in inactive_decisions[:5]:  # Show first 5
        report += f"✗ {decision['condition']}\n"
    
    report += f"""
Generation Path ({len(state['generation_path'])} sections):
{' → '.join(state['generation_path'])}

Section Breakdown:
"""
    
    for section_name in state['generation_path']:
        if section_name in state['generated_sections']:
            content = state['generated_sections'][section_name]
            report += f"- {section_name}: {len(content.split())} words\n"
    
    report += f"""
FINAL OUTPUT:
{'-' * 50}
{state['final_output']}
{'-' * 50}

Generation Statistics:
- Total Sections: {len(state['generated_sections'])}
- Total Words: {len(state['final_output'].split())}
- Conditional Branches Used: {len(active_decisions)}
- Generation Path Complexity: {len(state['generation_path'])} steps
"""
    
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_conditional_generation_graph():
    """Create the conditional generation workflow graph"""
    workflow = StateGraph(ConditionalGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("analyze_context", analyze_context)
    workflow.add_node("make_decisions", make_conditional_decisions)
    workflow.add_node("plan_path", plan_generation_path)
    workflow.add_node("generate_sections", generate_sections)
    workflow.add_node("assemble_output", assemble_output)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_context")
    workflow.add_edge("analyze_context", "make_decisions")
    workflow.add_edge("make_decisions", "plan_path")
    workflow.add_edge("plan_path", "generate_sections")
    workflow.add_edge("generate_sections", "assemble_output")
    workflow.add_edge("assemble_output", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample conditional generation task
    initial_state = {
        "task_description": "Generate content with conditional logic based on context",
        "generation_requirements": {
            "topic": "Introduction to Machine Learning",
            "audience": "beginner programmers",
            "purpose": "comprehensive tutorial",
            "format": "tutorial",
            "tone": "friendly and approachable",
            "max_words": 600,
            "keywords": ["machine learning", "algorithms", "training", "prediction"]
        },
        "context_analysis": {},
        "decision_points": [],
        "generation_path": [],
        "conditional_branches": {},
        "generated_sections": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_conditional_generation_graph()
    
    print("Conditional Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Active Conditions: {sum(1 for d in result['decision_points'] if d['result'])}/{len(result['decision_points'])}")

"""
Multi-Stage Generation MCP Pattern

This pattern demonstrates content generation across multiple distinct stages,
each with specific goals, transforming content through a pipeline of generation phases.

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
class MultiStageGenerationState(TypedDict):
    """State for multi-stage generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    stage_pipeline: List[Dict[str, Any]]
    current_stage: int
    stage_outputs: Dict[str, Any]
    intermediate_artifacts: List[Dict[str, Any]]
    stage_transitions: List[Dict[str, Any]]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class StagePipeline Builder:
    """Build multi-stage generation pipeline"""
    
    def build_pipeline(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build stage pipeline based on content type"""
        content_type = requirements.get('content_type', 'article')
        
        pipelines = {
            "article": self._article_pipeline(),
            "story": self._story_pipeline(),
            "report": self._report_pipeline(),
            "tutorial": self._tutorial_pipeline(),
            "proposal": self._proposal_pipeline()
        }
        
        return pipelines.get(content_type, self._default_pipeline())
    
    def _article_pipeline(self) -> List[Dict[str, Any]]:
        """Pipeline for article generation"""
        return [
            {
                "stage": 1,
                "name": "brainstorming",
                "goal": "Generate key ideas and angles",
                "output_type": "bullet_points",
                "temperature": 0.9,
                "focus": "creativity and ideation"
            },
            {
                "stage": 2,
                "name": "outlining",
                "goal": "Create structured outline",
                "output_type": "hierarchical_outline",
                "temperature": 0.5,
                "focus": "structure and organization"
            },
            {
                "stage": 3,
                "name": "drafting",
                "goal": "Write full draft",
                "output_type": "full_text",
                "temperature": 0.7,
                "focus": "content development"
            },
            {
                "stage": 4,
                "name": "enhancement",
                "goal": "Enhance with examples and details",
                "output_type": "enhanced_text",
                "temperature": 0.6,
                "focus": "depth and richness"
            },
            {
                "stage": 5,
                "name": "polishing",
                "goal": "Polish language and flow",
                "output_type": "polished_text",
                "temperature": 0.3,
                "focus": "clarity and style"
            }
        ]
    
    def _story_pipeline(self) -> List[Dict[str, Any]]:
        """Pipeline for story generation"""
        return [
            {
                "stage": 1,
                "name": "premise",
                "goal": "Develop story premise and theme",
                "output_type": "concept",
                "temperature": 0.9,
                "focus": "core idea"
            },
            {
                "stage": 2,
                "name": "characters",
                "goal": "Create characters and relationships",
                "output_type": "character_profiles",
                "temperature": 0.8,
                "focus": "character development"
            },
            {
                "stage": 3,
                "name": "plot",
                "goal": "Structure plot and key events",
                "output_type": "plot_outline",
                "temperature": 0.6,
                "focus": "narrative arc"
            },
            {
                "stage": 4,
                "name": "scenes",
                "goal": "Write individual scenes",
                "output_type": "scene_text",
                "temperature": 0.7,
                "focus": "vivid storytelling"
            },
            {
                "stage": 5,
                "name": "narrative",
                "goal": "Weave scenes into cohesive narrative",
                "output_type": "complete_story",
                "temperature": 0.5,
                "focus": "flow and coherence"
            }
        ]
    
    def _report_pipeline(self) -> List[Dict[str, Any]]:
        """Pipeline for report generation"""
        return [
            {
                "stage": 1,
                "name": "data_collection",
                "goal": "Identify key data points and facts",
                "output_type": "data_summary",
                "temperature": 0.3,
                "focus": "factual information"
            },
            {
                "stage": 2,
                "name": "analysis",
                "goal": "Analyze data and identify insights",
                "output_type": "analysis_points",
                "temperature": 0.5,
                "focus": "analytical thinking"
            },
            {
                "stage": 3,
                "name": "structuring",
                "goal": "Structure findings into sections",
                "output_type": "report_structure",
                "temperature": 0.4,
                "focus": "logical organization"
            },
            {
                "stage": 4,
                "name": "writing",
                "goal": "Write formal report content",
                "output_type": "report_draft",
                "temperature": 0.4,
                "focus": "professional writing"
            },
            {
                "stage": 5,
                "name": "executive_summary",
                "goal": "Create executive summary",
                "output_type": "complete_report",
                "temperature": 0.3,
                "focus": "concise synthesis"
            }
        ]
    
    def _tutorial_pipeline(self) -> List[Dict[str, Any]]:
        """Pipeline for tutorial generation"""
        return [
            {
                "stage": 1,
                "name": "learning_objectives",
                "goal": "Define learning objectives",
                "output_type": "objectives_list",
                "temperature": 0.4,
                "focus": "educational goals"
            },
            {
                "stage": 2,
                "name": "prerequisites",
                "goal": "Identify prerequisites and context",
                "output_type": "prereq_section",
                "temperature": 0.5,
                "focus": "foundational knowledge"
            },
            {
                "stage": 3,
                "name": "step_breakdown",
                "goal": "Break down into detailed steps",
                "output_type": "step_outline",
                "temperature": 0.4,
                "focus": "logical progression"
            },
            {
                "stage": 4,
                "name": "instruction_writing",
                "goal": "Write clear instructions with examples",
                "output_type": "tutorial_content",
                "temperature": 0.6,
                "focus": "clarity and examples"
            },
            {
                "stage": 5,
                "name": "practice_exercises",
                "goal": "Add practice exercises and summary",
                "output_type": "complete_tutorial",
                "temperature": 0.5,
                "focus": "reinforcement"
            }
        ]
    
    def _proposal_pipeline(self) -> List[Dict[str, Any]]:
        """Pipeline for proposal generation"""
        return [
            {
                "stage": 1,
                "name": "problem_definition",
                "goal": "Define problem and opportunity",
                "output_type": "problem_statement",
                "temperature": 0.5,
                "focus": "problem clarity"
            },
            {
                "stage": 2,
                "name": "solution_design",
                "goal": "Design proposed solution",
                "output_type": "solution_overview",
                "temperature": 0.7,
                "focus": "innovative solution"
            },
            {
                "stage": 3,
                "name": "benefits_analysis",
                "goal": "Analyze benefits and ROI",
                "output_type": "benefits_section",
                "temperature": 0.6,
                "focus": "value proposition"
            },
            {
                "stage": 4,
                "name": "implementation_plan",
                "goal": "Create implementation roadmap",
                "output_type": "plan_section",
                "temperature": 0.4,
                "focus": "actionable steps"
            },
            {
                "stage": 5,
                "name": "proposal_assembly",
                "goal": "Assemble complete proposal",
                "output_type": "final_proposal",
                "temperature": 0.3,
                "focus": "professional presentation"
            }
        ]
    
    def _default_pipeline(self) -> List[Dict[str, Any]]:
        """Default pipeline"""
        return [
            {
                "stage": 1,
                "name": "ideation",
                "goal": "Generate ideas",
                "output_type": "ideas",
                "temperature": 0.8,
                "focus": "creativity"
            },
            {
                "stage": 2,
                "name": "planning",
                "goal": "Plan structure",
                "output_type": "plan",
                "temperature": 0.5,
                "focus": "organization"
            },
            {
                "stage": 3,
                "name": "creation",
                "goal": "Create content",
                "output_type": "content",
                "temperature": 0.7,
                "focus": "development"
            },
            {
                "stage": 4,
                "name": "refinement",
                "goal": "Refine content",
                "output_type": "refined_content",
                "temperature": 0.4,
                "focus": "quality"
            }
        ]


class StageExecutor:
    """Execute individual generation stages"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def execute_stage(self, stage_def: Dict[str, Any],
                     previous_outputs: Dict[str, Any],
                     requirements: Dict[str, Any]) -> str:
        """Execute a single stage"""
        # Build context from previous stages
        context = self._build_context(previous_outputs)
        
        # Create stage-specific LLM with appropriate temperature
        stage_llm = ChatOpenAI(
            model="gpt-4",
            temperature=stage_def.get("temperature", 0.7)
        )
        
        prompt = f"""Execute Stage {stage_def['stage']}: {stage_def['name'].upper()}

Goal: {stage_def['goal']}
Output Type: {stage_def['output_type']}
Focus: {stage_def['focus']}

{context}

Topic: {requirements.get('topic', 'general')}
Requirements: {requirements.get('specific_requirements', 'Follow best practices')}

Generate the output for this stage, focusing specifically on the stated goal.
Do not try to complete the entire project - just this stage."""
        
        messages = [
            SystemMessage(content=f"You are executing stage {stage_def['stage']} of a multi-stage generation process. Focus on this stage's specific goal."),
            HumanMessage(content=prompt)
        ]
        
        response = stage_llm.invoke(messages)
        return response.content
    
    def _build_context(self, previous_outputs: Dict[str, Any]) -> str:
        """Build context from previous stage outputs"""
        if not previous_outputs:
            return "This is the first stage - no previous context."
        
        context_parts = ["Previous Stage Outputs:"]
        for stage_name, output in previous_outputs.items():
            context_parts.append(f"\n{stage_name.upper()}:")
            # Truncate long outputs
            output_preview = output[:500] + "..." if len(output) > 500 else output
            context_parts.append(output_preview)
        
        return "\n".join(context_parts)


class StageTransitionAnalyzer:
    """Analyze transitions between stages"""
    
    def analyze_transition(self, from_stage: Dict[str, Any],
                          to_stage: Dict[str, Any],
                          from_output: str,
                          to_output: str) -> Dict[str, Any]:
        """Analyze transition between stages"""
        return {
            "from_stage": from_stage["name"],
            "to_stage": to_stage["name"],
            "transformation": self._describe_transformation(from_stage, to_stage),
            "output_growth": len(to_output.split()) - len(from_output.split()),
            "focus_shift": f"{from_stage['focus']} → {to_stage['focus']}"
        }
    
    def _describe_transformation(self, from_stage: Dict[str, Any],
                                to_stage: Dict[str, Any]) -> str:
        """Describe transformation between stages"""
        transforms = {
            ("brainstorming", "outlining"): "Ideas organized into structure",
            ("outlining", "drafting"): "Structure expanded into full text",
            ("drafting", "enhancement"): "Content enriched with details",
            ("enhancement", "polishing"): "Language refined and polished",
            ("premise", "characters"): "Concept expanded into characters",
            ("characters", "plot"): "Characters woven into plot",
            ("plot", "scenes"): "Plot broken into vivid scenes",
            ("scenes", "narrative"): "Scenes unified into narrative"
        }
        
        key = (from_stage["name"], to_stage["name"])
        return transforms.get(key, f"{from_stage['output_type']} transformed to {to_stage['output_type']}")


# Agent functions
def initialize_generation(state: MultiStageGenerationState) -> MultiStageGenerationState:
    """Initialize multi-stage generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing multi-stage generation: {state['task_description']}"
    ))
    state["current_stage"] = 0
    state["stage_outputs"] = {}
    state["current_step"] = "initialized"
    return state


def build_pipeline(state: MultiStageGenerationState) -> MultiStageGenerationState:
    """Build generation pipeline"""
    builder = StagePipelineBuilder()
    
    pipeline = builder.build_pipeline(state["generation_requirements"])
    
    state["stage_pipeline"] = pipeline
    
    stage_names = [s["name"] for s in pipeline]
    state["messages"].append(HumanMessage(
        content=f"Built {len(pipeline)}-stage pipeline: {' → '.join(stage_names)}"
    ))
    state["current_step"] = "pipeline_built"
    return state


def execute_current_stage(state: MultiStageGenerationState) -> MultiStageGenerationState:
    """Execute current stage"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    executor = StageExecutor(llm)
    
    stage_index = state["current_stage"]
    stage_def = state["stage_pipeline"][stage_index]
    
    output = executor.execute_stage(
        stage_def,
        state["stage_outputs"],
        state["generation_requirements"]
    )
    
    # Store output
    state["stage_outputs"][stage_def["name"]] = output
    
    # Store artifact
    state["intermediate_artifacts"].append({
        "stage": stage_def["stage"],
        "name": stage_def["name"],
        "output": output,
        "word_count": len(output.split())
    })
    
    state["messages"].append(HumanMessage(
        content=f"Executed Stage {stage_def['stage']} ({stage_def['name']}): "
                f"{len(output.split())} words, focus: {stage_def['focus']}"
    ))
    state["current_step"] = "stage_executed"
    return state


def analyze_transition(state: MultiStageGenerationState) -> MultiStageGenerationState:
    """Analyze transition to next stage"""
    stage_index = state["current_stage"]
    
    # Only analyze if not the first stage
    if stage_index > 0:
        analyzer = StageTransitionAnalyzer()
        
        from_stage = state["stage_pipeline"][stage_index - 1]
        to_stage = state["stage_pipeline"][stage_index]
        
        from_output = state["stage_outputs"][from_stage["name"]]
        to_output = state["stage_outputs"][to_stage["name"]]
        
        transition = analyzer.analyze_transition(
            from_stage, to_stage, from_output, to_output
        )
        
        state["stage_transitions"].append(transition)
        
        state["messages"].append(HumanMessage(
            content=f"Transition: {transition['transformation']} "
                    f"(growth: {transition['output_growth']:+d} words)"
        ))
    
    state["current_stage"] += 1
    state["current_step"] = "transition_analyzed"
    return state


def check_pipeline_complete(state: MultiStageGenerationState) -> str:
    """Check if pipeline is complete"""
    if state["current_stage"] >= len(state["stage_pipeline"]):
        return "generate_report"
    else:
        return "execute_stage"


def generate_report(state: MultiStageGenerationState) -> MultiStageGenerationState:
    """Generate final multi-stage generation report"""
    # Get final output from last stage
    final_stage = state["stage_pipeline"][-1]
    final_output = state["stage_outputs"][final_stage["name"]]
    
    report = f"""
MULTI-STAGE GENERATION REPORT
==============================

Task: {state['task_description']}

Pipeline: {len(state['stage_pipeline'])} stages
Total Stages Executed: {len(state['intermediate_artifacts'])}

Stage Pipeline:
"""
    
    for stage in state['stage_pipeline']:
        report += f"{stage['stage']}. {stage['name'].upper()}\n"
        report += f"   Goal: {stage['goal']}\n"
        report += f"   Focus: {stage['focus']}\n"
        report += f"   Output Type: {stage['output_type']}\n"
        
        if stage['name'] in state['stage_outputs']:
            output = state['stage_outputs'][stage['name']]
            report += f"   Generated: {len(output.split())} words\n"
        report += "\n"
    
    if state['stage_transitions']:
        report += "Stage Transitions:\n"
        for trans in state['stage_transitions']:
            report += f"- {trans['from_stage']} → {trans['to_stage']}: {trans['transformation']}\n"
            report += f"  Focus shift: {trans['focus_shift']}\n"
            report += f"  Growth: {trans['output_growth']:+d} words\n"
    
    report += f"""
FINAL OUTPUT ({final_stage['name']}):
{'-' * 50}
{final_output}
{'-' * 50}

Generation Statistics:
- Total Stages: {len(state['stage_pipeline'])}
- Final Word Count: {len(final_output.split())}
- Pipeline Completion: 100%
"""
    
    state["final_output"] = final_output
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_multi_stage_generation_graph():
    """Create the multi-stage generation workflow graph"""
    workflow = StateGraph(MultiStageGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("build_pipeline", build_pipeline)
    workflow.add_node("execute_stage", execute_current_stage)
    workflow.add_node("analyze_transition", analyze_transition)
    workflow.add_node("generate_report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "build_pipeline")
    workflow.add_edge("build_pipeline", "execute_stage")
    workflow.add_edge("execute_stage", "analyze_transition")
    
    # Conditional edge - continue pipeline or finish
    workflow.add_conditional_edges(
        "analyze_transition",
        check_pipeline_complete,
        {
            "execute_stage": "execute_stage",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample multi-stage generation task
    initial_state = {
        "task_description": "Generate a comprehensive article through multi-stage pipeline",
        "generation_requirements": {
            "topic": "The Future of Remote Work: Technology and Culture",
            "content_type": "article",
            "specific_requirements": "Professional, well-researched, engaging for business audience"
        },
        "stage_pipeline": [],
        "current_stage": 0,
        "stage_outputs": {},
        "intermediate_artifacts": [],
        "stage_transitions": [],
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_multi_stage_generation_graph()
    
    print("Multi-Stage Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Stages Completed: {len(result['intermediate_artifacts'])}")

"""
Guided Generation MCP Pattern

This pattern demonstrates generation guided by human feedback, intermediate
checkpoints, and iterative steering throughout the generation process.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
from datetime import datetime


# State definition
class GuidedGenerationState(TypedDict):
    """State for guided generation workflow"""
    generation_goal: str
    guidance_mode: str  # 'interactive', 'checkpoint', 'continuous'
    generation_outline: Dict[str, Any]
    checkpoints: List[Dict[str, Any]]
    current_checkpoint: int
    generated_content: str
    checkpoint_outputs: List[Dict[str, str]]
    guidance_history: List[Dict[str, Any]]
    user_feedback: Optional[Dict[str, Any]]
    revision_needed: bool
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class OutlineGenerator:
    """Generate structured outline for guided generation"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_outline(self, goal: str, guidance_mode: str) -> Dict[str, Any]:
        """Generate outline with checkpoints"""
        prompt = f"""Create a structured outline for this generation task:

Goal: {goal}

Guidance Mode: {guidance_mode}

Create an outline with 4-6 checkpoints where guidance/feedback can be provided.
Each checkpoint should represent a logical section or milestone.

Return JSON with this structure:
{{
    "title": "overall title",
    "checkpoints": [
        {{
            "id": 1,
            "name": "checkpoint name",
            "description": "what to generate",
            "guidance_questions": ["question1", "question2"]
        }}
    ]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at creating structured outlines for guided generation."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        
        # Fallback outline
        return {
            "title": goal,
            "checkpoints": [
                {
                    "id": 1,
                    "name": "Introduction",
                    "description": "Create introduction",
                    "guidance_questions": ["Is the scope clear?", "Is the tone appropriate?"]
                },
                {
                    "id": 2,
                    "name": "Main Content",
                    "description": "Develop main content",
                    "guidance_questions": ["Are key points covered?", "Is depth sufficient?"]
                },
                {
                    "id": 3,
                    "name": "Conclusion",
                    "description": "Write conclusion",
                    "guidance_questions": ["Does it summarize well?", "Is there a clear takeaway?"]
                }
            ]
        }


class CheckpointGenerator:
    """Generate content for individual checkpoints"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_checkpoint(self, checkpoint: Dict[str, Any], 
                          previous_content: str,
                          goal: str,
                          guidance_feedback: Optional[Dict[str, Any]] = None) -> str:
        """Generate content for a checkpoint"""
        # Build prompt with guidance
        guidance_context = self._build_guidance_context(guidance_feedback)
        previous_context = self._build_previous_context(previous_content)
        
        prompt = f"""Generate content for this checkpoint:

Overall Goal: {goal}

Checkpoint: {checkpoint['name']}
Description: {checkpoint['description']}

Guidance Questions to Consider:
{self._format_guidance_questions(checkpoint.get('guidance_questions', []))}

{guidance_context}

{previous_context}

Generate content for this checkpoint that:
1. Addresses the checkpoint description
2. Considers the guidance questions
3. Incorporates any provided feedback
4. Builds naturally on previous content"""
        
        messages = [
            SystemMessage(content="You are a content generator that follows guidance and feedback carefully."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _build_guidance_context(self, feedback: Optional[Dict[str, Any]]) -> str:
        """Build context from guidance feedback"""
        if not feedback:
            return "No specific guidance provided yet."
        
        context = "Guidance Feedback:\n"
        
        if feedback.get("direction"):
            context += f"- Direction: {feedback['direction']}\n"
        
        if feedback.get("adjustments"):
            context += f"- Adjustments: {feedback['adjustments']}\n"
        
        if feedback.get("focus_areas"):
            context += f"- Focus on: {', '.join(feedback['focus_areas'])}\n"
        
        if feedback.get("avoid"):
            context += f"- Avoid: {', '.join(feedback['avoid'])}\n"
        
        return context
    
    def _build_previous_context(self, previous_content: str) -> str:
        """Build context from previous content"""
        if not previous_content:
            return "This is the first checkpoint."
        
        preview = previous_content[:400] + "..." if len(previous_content) > 400 else previous_content
        return f"Previous Content:\n{preview}"
    
    def _format_guidance_questions(self, questions: List[str]) -> str:
        """Format guidance questions"""
        if not questions:
            return "No specific guidance questions."
        
        return "\n".join(f"- {q}" for q in questions)


class FeedbackSimulator:
    """Simulate user feedback for guided generation (in real scenario, this would be actual user input)"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_feedback(self, checkpoint: Dict[str, Any], 
                         generated_content: str,
                         goal: str) -> Dict[str, Any]:
        """Simulate realistic user feedback"""
        prompt = f"""As a user reviewing generated content, provide constructive feedback:

Generation Goal: {goal}
Checkpoint: {checkpoint['name']}

Generated Content:
{generated_content[:500]}...

Guidance Questions:
{', '.join(checkpoint.get('guidance_questions', []))}

Provide feedback in JSON format:
{{
    "approval": "approved|needs_revision",
    "direction": "brief guidance direction",
    "adjustments": "specific adjustments if needed",
    "focus_areas": ["area1", "area2"],
    "avoid": ["thing1", "thing2"],
    "suggestions": ["suggestion1", "suggestion2"]
}}

Be constructive and specific."""
        
        messages = [
            SystemMessage(content="You are a constructive reviewer providing helpful feedback."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        
        # Fallback feedback
        return {
            "approval": "approved",
            "direction": "Continue as planned",
            "adjustments": "None",
            "focus_areas": ["clarity", "completeness"],
            "avoid": ["redundancy"],
            "suggestions": ["Maintain current quality"]
        }


class GuidanceIntegrator:
    """Integrate guidance and feedback into generation process"""
    
    def analyze_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze feedback to determine next steps"""
        return {
            "needs_revision": feedback.get("approval") == "needs_revision",
            "has_direction": bool(feedback.get("direction")),
            "has_adjustments": bool(feedback.get("adjustments")) and feedback.get("adjustments") != "None",
            "focus_count": len(feedback.get("focus_areas", [])),
            "suggestion_count": len(feedback.get("suggestions", []))
        }
    
    def apply_guidance(self, content: str, feedback: Dict[str, Any],
                      llm: ChatOpenAI) -> str:
        """Apply guidance to revise content"""
        if feedback.get("approval") == "approved":
            return content  # No revision needed
        
        prompt = f"""Revise this content based on the following feedback:

Original Content:
{content}

Feedback:
- Direction: {feedback.get('direction', 'N/A')}
- Adjustments: {feedback.get('adjustments', 'N/A')}
- Focus Areas: {', '.join(feedback.get('focus_areas', []))}
- Avoid: {', '.join(feedback.get('avoid', []))}
- Suggestions: {', '.join(feedback.get('suggestions', []))}

Provide revised content that addresses all feedback points."""
        
        messages = [
            SystemMessage(content="You are a content reviser that carefully applies feedback."),
            HumanMessage(content=prompt)
        ]
        
        response = llm.invoke(messages)
        return response.content


# Agent functions
def initialize_guided_generation(state: GuidedGenerationState) -> GuidedGenerationState:
    """Initialize guided generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing guided generation: {state['generation_goal']} (mode: {state['guidance_mode']})"
    ))
    state["current_checkpoint"] = 0
    state["generated_content"] = ""
    state["checkpoint_outputs"] = []
    state["guidance_history"] = []
    state["revision_needed"] = False
    state["current_step"] = "initialized"
    return state


def create_outline(state: GuidedGenerationState) -> GuidedGenerationState:
    """Create generation outline with checkpoints"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    generator = OutlineGenerator(llm)
    
    outline = generator.generate_outline(
        state["generation_goal"],
        state["guidance_mode"]
    )
    
    state["generation_outline"] = outline
    state["checkpoints"] = outline.get("checkpoints", [])
    
    state["messages"].append(HumanMessage(
        content=f"Created outline: {outline.get('title', 'Untitled')} with {len(state['checkpoints'])} checkpoints"
    ))
    state["current_step"] = "outline_created"
    return state


def generate_checkpoint_content(state: GuidedGenerationState) -> GuidedGenerationState:
    """Generate content for current checkpoint"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = CheckpointGenerator(llm)
    
    checkpoint_idx = state["current_checkpoint"]
    
    if checkpoint_idx >= len(state["checkpoints"]):
        state["current_step"] = "all_checkpoints_complete"
        return state
    
    checkpoint = state["checkpoints"][checkpoint_idx]
    
    # Generate content with any previous guidance
    content = generator.generate_checkpoint(
        checkpoint,
        state["generated_content"],
        state["generation_goal"],
        state.get("user_feedback")
    )
    
    # Store checkpoint output
    checkpoint_output = {
        "checkpoint_id": checkpoint["id"],
        "checkpoint_name": checkpoint["name"],
        "content": content,
        "timestamp": datetime.now().isoformat(),
        "word_count": len(content.split())
    }
    
    state["checkpoint_outputs"].append(checkpoint_output)
    
    state["messages"].append(HumanMessage(
        content=f"Generated checkpoint {checkpoint_idx + 1}/{len(state['checkpoints'])}: "
                f"{checkpoint['name']} ({len(content.split())} words)"
    ))
    state["current_step"] = "checkpoint_generated"
    return state


def collect_feedback(state: GuidedGenerationState) -> GuidedGenerationState:
    """Collect feedback on current checkpoint"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    simulator = FeedbackSimulator(llm)
    
    checkpoint_idx = state["current_checkpoint"]
    
    if checkpoint_idx >= len(state["checkpoints"]):
        state["current_step"] = "feedback_complete"
        return state
    
    checkpoint = state["checkpoints"][checkpoint_idx]
    current_output = state["checkpoint_outputs"][-1]
    
    # Simulate user feedback (in real scenario, this would be actual user input)
    feedback = simulator.generate_feedback(
        checkpoint,
        current_output["content"],
        state["generation_goal"]
    )
    
    # Store feedback
    guidance_entry = {
        "checkpoint_id": checkpoint["id"],
        "checkpoint_name": checkpoint["name"],
        "feedback": feedback,
        "timestamp": datetime.now().isoformat()
    }
    
    state["guidance_history"].append(guidance_entry)
    state["user_feedback"] = feedback
    
    state["messages"].append(HumanMessage(
        content=f"Received feedback for {checkpoint['name']}: {feedback.get('approval', 'unknown')}"
    ))
    state["current_step"] = "feedback_collected"
    return state


def process_feedback(state: GuidedGenerationState) -> GuidedGenerationState:
    """Process feedback and determine if revision needed"""
    integrator = GuidanceIntegrator()
    
    if not state.get("user_feedback"):
        state["revision_needed"] = False
        state["current_step"] = "no_feedback"
        return state
    
    analysis = integrator.analyze_feedback(state["user_feedback"])
    
    state["revision_needed"] = analysis["needs_revision"]
    
    if state["revision_needed"]:
        state["messages"].append(HumanMessage(
            content=f"Revision needed based on feedback"
        ))
    else:
        state["messages"].append(HumanMessage(
            content=f"Checkpoint approved, proceeding to next"
        ))
    
    state["current_step"] = "feedback_processed"
    return state


def apply_feedback(state: GuidedGenerationState) -> GuidedGenerationState:
    """Apply feedback to revise content"""
    if not state["revision_needed"]:
        state["current_step"] = "no_revision_needed"
        return state
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    integrator = GuidanceIntegrator()
    
    # Get current checkpoint content
    current_output = state["checkpoint_outputs"][-1]
    
    # Apply guidance to revise
    revised_content = integrator.apply_guidance(
        current_output["content"],
        state["user_feedback"],
        llm
    )
    
    # Update checkpoint output
    state["checkpoint_outputs"][-1]["content"] = revised_content
    state["checkpoint_outputs"][-1]["revised"] = True
    state["checkpoint_outputs"][-1]["word_count"] = len(revised_content.split())
    
    state["messages"].append(HumanMessage(
        content=f"Applied feedback and revised checkpoint {state['current_checkpoint'] + 1}"
    ))
    
    state["revision_needed"] = False
    state["current_step"] = "feedback_applied"
    return state


def accumulate_content(state: GuidedGenerationState) -> GuidedGenerationState:
    """Accumulate approved checkpoint content"""
    if state["checkpoint_outputs"]:
        current_output = state["checkpoint_outputs"][-1]
        
        # Add to accumulated content
        if state["generated_content"]:
            state["generated_content"] += "\n\n"
        
        state["generated_content"] += current_output["content"]
    
    # Move to next checkpoint
    state["current_checkpoint"] += 1
    state["user_feedback"] = None  # Clear for next checkpoint
    
    state["messages"].append(HumanMessage(
        content=f"Accumulated content, moving to checkpoint {state['current_checkpoint'] + 1}"
    ))
    state["current_step"] = "content_accumulated"
    return state


def check_completion(state: GuidedGenerationState) -> GuidedGenerationState:
    """Check if all checkpoints are complete"""
    if state["current_checkpoint"] >= len(state["checkpoints"]):
        state["current_step"] = "generation_complete"
    else:
        state["current_step"] = "continue_generation"
    
    return state


def finalize_output(state: GuidedGenerationState) -> GuidedGenerationState:
    """Finalize guided generation output"""
    state["final_output"] = state["generated_content"]
    
    state["messages"].append(HumanMessage(
        content=f"Finalized output: {len(state['final_output'].split())} words total"
    ))
    state["current_step"] = "finalized"
    return state


def generate_report(state: GuidedGenerationState) -> GuidedGenerationState:
    """Generate final guided generation report"""
    revisions = sum(1 for output in state["checkpoint_outputs"] if output.get("revised", False))
    approvals = len([g for g in state["guidance_history"] if g["feedback"].get("approval") == "approved"])
    
    report = f"""
GUIDED GENERATION REPORT
========================

Goal: {state['generation_goal']}
Mode: {state['guidance_mode']}

Outline: {state['generation_outline'].get('title', 'Untitled')}
Checkpoints: {len(state['checkpoints'])}

Checkpoint Breakdown:
"""
    
    for i, checkpoint in enumerate(state['checkpoints'], 1):
        output = state['checkpoint_outputs'][i-1] if i-1 < len(state['checkpoint_outputs']) else None
        if output:
            revised_marker = " (revised)" if output.get("revised") else ""
            report += f"{i}. {checkpoint['name']}: {output['word_count']} words{revised_marker}\n"
    
    report += f"""
Guidance Statistics:
- Total Guidance Sessions: {len(state['guidance_history'])}
- Approvals: {approvals}
- Revisions Required: {revisions}
- Feedback Integration Rate: {(revisions / len(state['guidance_history']) * 100) if state['guidance_history'] else 0:.1f}%

Guidance History:
"""
    
    for i, guidance in enumerate(state['guidance_history'], 1):
        feedback = guidance['feedback']
        report += f"\n{i}. {guidance['checkpoint_name']}:\n"
        report += f"   - Approval: {feedback.get('approval', 'N/A')}\n"
        report += f"   - Direction: {feedback.get('direction', 'N/A')}\n"
        if feedback.get('adjustments') and feedback['adjustments'] != 'None':
            report += f"   - Adjustments: {feedback['adjustments']}\n"
    
    report += f"""
FINAL OUTPUT:
{'-' * 50}
{state['final_output']}
{'-' * 50}

Generation Summary:
- Total Words: {len(state['final_output'].split())}
- Checkpoints Completed: {len(state['checkpoint_outputs'])}
- Guidance Sessions: {len(state['guidance_history'])}
- Revisions Made: {revisions}
"""
    
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_guided_generation_graph():
    """Create the guided generation workflow graph"""
    workflow = StateGraph(GuidedGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_guided_generation)
    workflow.add_node("create_outline", create_outline)
    workflow.add_node("generate_checkpoint", generate_checkpoint_content)
    workflow.add_node("collect_feedback", collect_feedback)
    workflow.add_node("process_feedback", process_feedback)
    workflow.add_node("apply_feedback", apply_feedback)
    workflow.add_node("accumulate", accumulate_content)
    workflow.add_node("check_completion", check_completion)
    workflow.add_node("finalize", finalize_output)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "create_outline")
    workflow.add_edge("create_outline", "generate_checkpoint")
    workflow.add_edge("generate_checkpoint", "collect_feedback")
    workflow.add_edge("collect_feedback", "process_feedback")
    workflow.add_edge("process_feedback", "apply_feedback")
    workflow.add_edge("apply_feedback", "accumulate")
    workflow.add_edge("accumulate", "check_completion")
    
    # Conditional routing from check_completion
    def route_after_check(state: GuidedGenerationState) -> str:
        if state["current_step"] == "generation_complete":
            return "finalize"
        else:
            return "generate_checkpoint"
    
    workflow.add_conditional_edges(
        "check_completion",
        route_after_check,
        {
            "finalize": "finalize",
            "generate_checkpoint": "generate_checkpoint"
        }
    )
    
    workflow.add_edge("finalize", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample guided generation task
    initial_state = {
        "generation_goal": "Write a comprehensive guide to sustainable living",
        "guidance_mode": "checkpoint",  # interactive, checkpoint, continuous
        "generation_outline": {},
        "checkpoints": [],
        "current_checkpoint": 0,
        "generated_content": "",
        "checkpoint_outputs": [],
        "guidance_history": [],
        "user_feedback": None,
        "revision_needed": False,
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_guided_generation_graph()
    
    print("Guided Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Checkpoints: {len(result['checkpoints'])}, Guidance Sessions: {len(result['guidance_history'])}")

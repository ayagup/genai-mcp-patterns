"""
Incremental Generation MCP Pattern

This pattern demonstrates content generation in incremental steps, building upon
previous outputs progressively to create comprehensive, layered content.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Intermediate
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class IncrementalGenerationState(TypedDict):
    """State for incremental generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    generation_plan: Dict[str, Any]
    increments: List[Dict[str, Any]]
    current_increment: int
    accumulated_content: str
    increment_quality_scores: List[float]
    continuation_context: str
    should_continue: bool
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class IncrementalPlanner:
    """Plan incremental generation steps"""
    
    def create_plan(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create incremental generation plan"""
        total_length = requirements.get('target_length', 500)
        num_increments = requirements.get('num_increments', 5)
        
        # Calculate length per increment
        base_increment = total_length // num_increments
        
        # Define increment types based on content structure
        increment_types = self._determine_increment_types(
            requirements.get('content_type', 'article'),
            num_increments
        )
        
        plan = {
            "num_increments": num_increments,
            "total_target_length": total_length,
            "increments": []
        }
        
        for i, inc_type in enumerate(increment_types):
            plan["increments"].append({
                "increment_number": i + 1,
                "type": inc_type,
                "target_length": base_increment,
                "focus": self._get_increment_focus(inc_type, requirements),
                "dependencies": list(range(1, i + 1)) if i > 0 else []
            })
        
        return plan
    
    def _determine_increment_types(self, content_type: str, num_increments: int) -> List[str]:
        """Determine types for each increment based on content type"""
        type_templates = {
            "article": ["introduction", "context", "main_points", "elaboration", "conclusion"],
            "story": ["setup", "characters", "conflict", "development", "resolution"],
            "tutorial": ["overview", "prerequisites", "step1", "step2", "conclusion"],
            "report": ["executive_summary", "background", "findings", "analysis", "recommendations"],
            "essay": ["thesis", "argument1", "argument2", "counter_argument", "conclusion"]
        }
        
        template = type_templates.get(content_type, ["intro", "body1", "body2", "body3", "conclusion"])
        
        # Adjust template to match num_increments
        if len(template) > num_increments:
            return template[:num_increments]
        elif len(template) < num_increments:
            # Expand middle sections
            middle = template[1:-1]
            expanded = template[:1]
            for i in range(num_increments - 2):
                expanded.append(middle[i % len(middle)])
            expanded.append(template[-1])
            return expanded
        
        return template
    
    def _get_increment_focus(self, inc_type: str, requirements: Dict[str, Any]) -> str:
        """Get focus description for increment type"""
        topic = requirements.get('topic', 'the subject')
        
        focus_map = {
            "introduction": f"Introduce {topic} and set context",
            "thesis": f"Present main thesis about {topic}",
            "setup": f"Set up the scenario for {topic}",
            "overview": f"Provide overview of {topic}",
            "executive_summary": f"Summarize key points about {topic}",
            "context": f"Provide background context for {topic}",
            "background": f"Explain background of {topic}",
            "characters": f"Introduce key elements/actors in {topic}",
            "prerequisites": f"Explain prerequisites for understanding {topic}",
            "main_points": f"Present main points about {topic}",
            "conflict": f"Present challenges or conflicts in {topic}",
            "findings": f"Present findings about {topic}",
            "argument1": f"Present first argument about {topic}",
            "step1": f"Explain first step of {topic}",
            "elaboration": f"Elaborate on key aspects of {topic}",
            "development": f"Develop the discussion of {topic}",
            "analysis": f"Analyze {topic} in depth",
            "argument2": f"Present second argument about {topic}",
            "step2": f"Explain second step of {topic}",
            "counter_argument": f"Address counter-arguments about {topic}",
            "conclusion": f"Conclude discussion of {topic}",
            "resolution": f"Resolve the discussion of {topic}",
            "recommendations": f"Provide recommendations regarding {topic}"
        }
        
        return focus_map.get(inc_type, f"Discuss {topic}")


class IncrementalGenerator:
    """Generate content incrementally"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_increment(self, increment_spec: Dict[str, Any],
                          previous_content: str,
                          requirements: Dict[str, Any]) -> str:
        """Generate single increment of content"""
        context_section = ""
        if previous_content:
            context_section = f"""
Previous Content:
{previous_content}

Build upon the above content naturally.
"""
        
        prompt = f"""Generate the next increment of content following this specification:

Increment #{increment_spec['increment_number']} - {increment_spec['type'].upper()}
Focus: {increment_spec['focus']}
Target Length: approximately {increment_spec['target_length']} words

{context_section}

Requirements:
- Topic: {requirements.get('topic', 'general')}
- Tone: {requirements.get('tone', 'professional')}
- Style: {requirements.get('style', 'clear and concise')}

Important Guidelines:
1. Continue seamlessly from previous content (if any)
2. Focus specifically on this increment's purpose
3. Maintain consistent tone and style
4. Do NOT repeat information from previous increments
5. Ensure smooth transitions
6. Write approximately {increment_spec['target_length']} words

Generate only this increment, not the entire piece."""
        
        messages = [
            SystemMessage(content="You are an expert content writer who creates content incrementally, building naturally on previous work."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class IncrementEvaluator:
    """Evaluate quality of each increment"""
    
    def evaluate_increment(self, increment_content: str,
                          increment_spec: Dict[str, Any],
                          previous_content: str) -> Dict[str, float]:
        """Evaluate increment quality"""
        return {
            "length_score": self._evaluate_length(increment_content, increment_spec),
            "coherence_score": self._evaluate_coherence(increment_content, previous_content),
            "focus_score": self._evaluate_focus(increment_content, increment_spec),
            "quality_score": self._evaluate_quality(increment_content)
        }
    
    def _evaluate_length(self, content: str, spec: Dict[str, Any]) -> float:
        """Evaluate if length meets target"""
        actual_length = len(content.split())
        target_length = spec['target_length']
        
        # Allow 20% variance
        min_acceptable = target_length * 0.8
        max_acceptable = target_length * 1.2
        
        if min_acceptable <= actual_length <= max_acceptable:
            return 1.0
        elif actual_length < min_acceptable:
            return max(0.5, actual_length / min_acceptable)
        else:
            return max(0.5, max_acceptable / actual_length)
    
    def _evaluate_coherence(self, content: str, previous_content: str) -> float:
        """Evaluate coherence with previous content"""
        if not previous_content:
            return 1.0  # First increment
        
        # Check for transition words/phrases
        transitions = ['furthermore', 'moreover', 'additionally', 'however', 
                      'therefore', 'consequently', 'building on', 'as mentioned',
                      'following', 'next', 'continuing', 'in addition']
        
        content_lower = content.lower()
        has_transition = any(trans in content_lower for trans in transitions)
        
        # Check for repetition (bad)
        prev_sentences = set(previous_content.lower().split('.'))
        curr_sentences = set(content.lower().split('.'))
        overlap = len(prev_sentences & curr_sentences) / max(len(curr_sentences), 1)
        
        transition_score = 0.7 if has_transition else 0.5
        repetition_penalty = max(0, 0.3 - overlap)
        
        return min(1.0, transition_score + repetition_penalty)
    
    def _evaluate_focus(self, content: str, spec: Dict[str, Any]) -> float:
        """Evaluate if content stays focused on increment purpose"""
        focus = spec['focus'].lower()
        content_lower = content.lower()
        
        # Extract key terms from focus
        focus_terms = [word for word in focus.split() 
                      if len(word) > 3 and word not in ['the', 'and', 'for', 'about']]
        
        # Count presence of focus terms
        term_presence = sum(1 for term in focus_terms if term in content_lower)
        
        return min(1.0, term_presence / max(len(focus_terms), 1))
    
    def _evaluate_quality(self, content: str) -> float:
        """Evaluate overall content quality"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Check sentence variety (length variance)
        lengths = [len(s.split()) for s in sentences]
        avg_length = sum(lengths) / len(lengths)
        
        # Ideal: varied sentence lengths
        if 12 <= avg_length <= 20:
            return 0.9
        elif 8 <= avg_length <= 25:
            return 0.7
        else:
            return 0.5


class ContinuationDecider:
    """Decide whether to continue generation"""
    
    def should_continue(self, current_increment: int,
                       total_increments: int,
                       quality_scores: List[float],
                       requirements: Dict[str, Any]) -> bool:
        """Decide if generation should continue"""
        # Check if all increments completed
        if current_increment >= total_increments:
            return False
        
        # Check quality threshold
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = requirements.get('min_quality_threshold', 0.5)
            
            if avg_quality < min_quality:
                # Quality too low, stop
                return False
        
        return True


# Agent functions
def initialize_generation(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Initialize incremental generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing incremental generation: {state['task_description']}"
    ))
    state["current_increment"] = 0
    state["accumulated_content"] = ""
    state["should_continue"] = True
    state["current_step"] = "initialized"
    return state


def create_generation_plan(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Create incremental generation plan"""
    planner = IncrementalPlanner()
    
    plan = planner.create_plan(state["generation_requirements"])
    
    state["generation_plan"] = plan
    state["increments"] = []
    
    state["messages"].append(HumanMessage(
        content=f"Created plan with {plan['num_increments']} increments: "
                f"{', '.join([inc['type'] for inc in plan['increments']])}"
    ))
    state["current_step"] = "plan_created"
    return state


def generate_next_increment(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Generate next increment of content"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = IncrementalGenerator(llm)
    
    increment_number = state["current_increment"]
    increment_spec = state["generation_plan"]["increments"][increment_number]
    
    increment_content = generator.generate_increment(
        increment_spec,
        state["accumulated_content"],
        state["generation_requirements"]
    )
    
    state["increments"].append({
        "number": increment_number + 1,
        "type": increment_spec["type"],
        "content": increment_content
    })
    
    state["messages"].append(HumanMessage(
        content=f"Generated increment #{increment_number + 1} ({increment_spec['type']}): "
                f"{len(increment_content.split())} words"
    ))
    state["current_step"] = "increment_generated"
    return state


def evaluate_increment(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Evaluate the generated increment"""
    evaluator = IncrementEvaluator()
    
    current_increment = state["increments"][-1]
    increment_number = state["current_increment"]
    increment_spec = state["generation_plan"]["increments"][increment_number]
    
    scores = evaluator.evaluate_increment(
        current_increment["content"],
        increment_spec,
        state["accumulated_content"]
    )
    
    # Calculate overall score
    overall_score = sum(scores.values()) / len(scores)
    state["increment_quality_scores"].append(overall_score)
    
    state["messages"].append(HumanMessage(
        content=f"Increment #{increment_number + 1} quality: {overall_score:.2f} "
                f"(length: {scores['length_score']:.2f}, "
                f"coherence: {scores['coherence_score']:.2f}, "
                f"focus: {scores['focus_score']:.2f})"
    ))
    state["current_step"] = "increment_evaluated"
    return state


def accumulate_content(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Accumulate the increment into overall content"""
    current_increment = state["increments"][-1]
    
    if state["accumulated_content"]:
        # Add spacing between increments
        state["accumulated_content"] += "\n\n" + current_increment["content"]
    else:
        state["accumulated_content"] = current_increment["content"]
    
    state["current_increment"] += 1
    
    state["messages"].append(HumanMessage(
        content=f"Accumulated increment. Total content: {len(state['accumulated_content'].split())} words"
    ))
    state["current_step"] = "content_accumulated"
    return state


def check_continuation(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Check if generation should continue"""
    decider = ContinuationDecider()
    
    should_continue = decider.should_continue(
        state["current_increment"],
        state["generation_plan"]["num_increments"],
        state["increment_quality_scores"],
        state["generation_requirements"]
    )
    
    state["should_continue"] = should_continue
    
    if should_continue:
        state["messages"].append(HumanMessage(
            content=f"Continuing to next increment ({state['current_increment'] + 1}/{state['generation_plan']['num_increments']})"
        ))
    else:
        reason = "All increments completed" if state["current_increment"] >= state["generation_plan"]["num_increments"] else "Quality threshold not met"
        state["messages"].append(HumanMessage(
            content=f"Generation complete. Reason: {reason}"
        ))
    
    state["current_step"] = "continuation_checked"
    return state


def decide_next_step(state: IncrementalGenerationState) -> str:
    """Decide next step in workflow"""
    if state["should_continue"]:
        return "generate_increment"
    else:
        return "generate_report"


def generate_report(state: IncrementalGenerationState) -> IncrementalGenerationState:
    """Generate final incremental generation report"""
    avg_quality = (sum(state["increment_quality_scores"]) / len(state["increment_quality_scores"]) 
                   if state["increment_quality_scores"] else 0.0)
    
    report = f"""
INCREMENTAL GENERATION REPORT
==============================

Task: {state['task_description']}

Generation Plan:
- Total Increments: {state['generation_plan']['num_increments']}
- Increments Generated: {len(state['increments'])}
- Target Length: {state['generation_plan']['total_target_length']} words
- Actual Length: {len(state['accumulated_content'].split())} words

Increment Breakdown:
"""
    
    for i, increment in enumerate(state['increments'], 1):
        quality = state['increment_quality_scores'][i-1] if i <= len(state['increment_quality_scores']) else 0.0
        report += f"{i}. {increment['type'].upper()}: {len(increment['content'].split())} words (quality: {quality:.2f})\n"
    
    report += f"""
Quality Metrics:
- Average Quality Score: {avg_quality:.2f}
- Completion Rate: {len(state['increments'])}/{state['generation_plan']['num_increments']} ({len(state['increments'])/state['generation_plan']['num_increments']:.1%})

GENERATED CONTENT:
{'-' * 50}
{state['accumulated_content']}
{'-' * 50}

Content Statistics:
- Total Words: {len(state['accumulated_content'].split())}
- Total Sentences: {len([s for s in state['accumulated_content'].split('.') if s.strip()])}
- Total Paragraphs: {len([p for p in state['accumulated_content'].split('\n\n') if p.strip()])}
"""
    
    state["final_output"] = state["accumulated_content"]
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_incremental_generation_graph():
    """Create the incremental generation workflow graph"""
    workflow = StateGraph(IncrementalGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("create_plan", create_generation_plan)
    workflow.add_node("generate_increment", generate_next_increment)
    workflow.add_node("evaluate_increment", evaluate_increment)
    workflow.add_node("accumulate", accumulate_content)
    workflow.add_node("check_continuation", check_continuation)
    workflow.add_node("generate_report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "create_plan")
    workflow.add_edge("create_plan", "generate_increment")
    workflow.add_edge("generate_increment", "evaluate_increment")
    workflow.add_edge("evaluate_increment", "accumulate")
    workflow.add_edge("accumulate", "check_continuation")
    
    # Conditional edge - continue or finish
    workflow.add_conditional_edges(
        "check_continuation",
        decide_next_step,
        {
            "generate_increment": "generate_increment",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample incremental generation task
    initial_state = {
        "task_description": "Generate an article about sustainable energy incrementally",
        "generation_requirements": {
            "topic": "Sustainable Energy Solutions for Urban Development",
            "content_type": "article",
            "target_length": 400,
            "num_increments": 5,
            "tone": "professional",
            "style": "informative and engaging",
            "min_quality_threshold": 0.6
        },
        "generation_plan": {},
        "increments": [],
        "current_increment": 0,
        "accumulated_content": "",
        "increment_quality_scores": [],
        "continuation_context": "",
        "should_continue": True,
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_incremental_generation_graph()
    
    print("Incremental Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Increments Generated: {len(result['increments'])}/{result['generation_plan']['num_increments']}")

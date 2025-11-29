"""
Iterative Refinement MCP Pattern

This pattern demonstrates content generation through iterative refinement cycles,
progressively improving quality through feedback analysis and targeted improvements.

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
class IterativeRefinementState(TypedDict):
    """State for iterative refinement workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    initial_content: str
    refinement_iterations: List[Dict[str, Any]]
    current_iteration: int
    current_content: str
    feedback_analysis: Dict[str, Any]
    improvement_areas: List[Dict[str, Any]]
    quality_progression: List[float]
    convergence_detected: bool
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class InitialGenerator:
    """Generate initial draft content"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_initial(self, requirements: Dict[str, Any]) -> str:
        """Generate initial draft"""
        prompt = f"""Generate initial draft content based on these requirements:

Topic: {requirements.get('topic', 'general')}
Purpose: {requirements.get('purpose', 'inform')}
Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words
Tone: {requirements.get('tone', 'professional')}
Audience: {requirements.get('audience', 'general')}

Focus on getting the core ideas down. This is an initial draft that will be refined."""
        
        messages = [
            SystemMessage(content="You are a content generator creating initial drafts for iterative refinement."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class FeedbackAnalyzer:
    """Analyze content and provide detailed feedback"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, content: str, requirements: Dict[str, Any],
                iteration: int) -> Dict[str, Any]:
        """Analyze content and generate feedback"""
        automated_feedback = self._automated_analysis(content, requirements)
        llm_feedback = self._llm_analysis(content, requirements, iteration)
        
        return {
            "iteration": iteration,
            "automated_feedback": automated_feedback,
            "llm_feedback": llm_feedback,
            "combined_score": (automated_feedback["overall_score"] + llm_feedback["score"]) / 2
        }
    
    def _automated_analysis(self, content: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Automated analysis of content metrics"""
        words = content.split()
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        # Length compliance
        min_words = requirements.get('min_words', 100)
        max_words = requirements.get('max_words', 500)
        word_count = len(words)
        
        length_score = 1.0 if min_words <= word_count <= max_words else 0.5
        
        # Readability (sentence length)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        readability_score = 1.0 if 12 <= avg_sentence_length <= 20 else 0.7
        
        # Structure (paragraphs)
        structure_score = min(1.0, len(paragraphs) / 3)
        
        # Keyword presence
        keywords = requirements.get('keywords', [])
        keyword_presence = sum(1 for kw in keywords if kw.lower() in content.lower())
        keyword_score = keyword_presence / max(len(keywords), 1) if keywords else 0.8
        
        overall = (length_score + readability_score + structure_score + keyword_score) / 4
        
        return {
            "word_count": word_count,
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "avg_sentence_length": avg_sentence_length,
            "length_score": length_score,
            "readability_score": readability_score,
            "structure_score": structure_score,
            "keyword_score": keyword_score,
            "overall_score": overall
        }
    
    def _llm_analysis(self, content: str, requirements: Dict[str, Any],
                     iteration: int) -> Dict[str, Any]:
        """LLM-based qualitative analysis"""
        prompt = f"""Analyze this content (iteration {iteration}) and provide detailed feedback:

Content:
{content}

Requirements:
- Topic: {requirements.get('topic', 'general')}
- Tone: {requirements.get('tone', 'professional')}
- Purpose: {requirements.get('purpose', 'inform')}

Provide feedback in JSON format:
{{
    "strengths": ["strength 1", "strength 2", ...],
    "weaknesses": ["weakness 1", "weakness 2", ...],
    "specific_improvements": ["improvement 1", "improvement 2", ...],
    "score": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are an expert content critic providing constructive feedback."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "strengths": ["Content generated"],
            "weaknesses": ["Needs refinement"],
            "specific_improvements": ["Continue refining"],
            "score": 0.7
        }


class ImprovementIdentifier:
    """Identify specific areas for improvement"""
    
    def identify_improvements(self, feedback: Dict[str, Any],
                            requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify prioritized improvement areas"""
        improvements = []
        
        auto_feedback = feedback["automated_feedback"]
        llm_feedback = feedback["llm_feedback"]
        
        # Length issues
        if auto_feedback["length_score"] < 0.9:
            target_words = (requirements.get('min_words', 100) + requirements.get('max_words', 500)) / 2
            current_words = auto_feedback["word_count"]
            
            improvements.append({
                "area": "length",
                "priority": "high",
                "issue": f"Word count {current_words} not optimal",
                "target": f"Adjust to approximately {int(target_words)} words",
                "current_score": auto_feedback["length_score"]
            })
        
        # Readability issues
        if auto_feedback["readability_score"] < 0.8:
            improvements.append({
                "area": "readability",
                "priority": "medium",
                "issue": f"Average sentence length {auto_feedback['avg_sentence_length']:.1f} words",
                "target": "Aim for 12-20 words per sentence",
                "current_score": auto_feedback["readability_score"]
            })
        
        # Structure issues
        if auto_feedback["structure_score"] < 0.8:
            improvements.append({
                "area": "structure",
                "priority": "medium",
                "issue": f"Only {auto_feedback['paragraph_count']} paragraphs",
                "target": "Add more paragraphs for better structure",
                "current_score": auto_feedback["structure_score"]
            })
        
        # Keyword coverage
        if auto_feedback["keyword_score"] < 0.8:
            improvements.append({
                "area": "keywords",
                "priority": "high",
                "issue": "Missing required keywords",
                "target": f"Include keywords: {', '.join(requirements.get('keywords', []))}",
                "current_score": auto_feedback["keyword_score"]
            })
        
        # LLM-identified improvements
        for imp in llm_feedback.get("specific_improvements", [])[:3]:
            improvements.append({
                "area": "content_quality",
                "priority": "high",
                "issue": imp,
                "target": "Address this feedback",
                "current_score": llm_feedback["score"]
            })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        improvements.sort(key=lambda x: priority_order.get(x["priority"], 2))
        
        return improvements


class ContentRefiner:
    """Refine content based on feedback"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def refine(self, current_content: str, improvements: List[Dict[str, Any]],
              requirements: Dict[str, Any], iteration: int) -> str:
        """Refine content addressing improvements"""
        improvement_text = "\n".join([
            f"{i+1}. [{imp['priority'].upper()}] {imp['area']}: {imp['issue']} → {imp['target']}"
            for i, imp in enumerate(improvements[:5])  # Top 5 improvements
        ])
        
        prompt = f"""Refine this content (iteration {iteration}) addressing the following improvements:

Current Content:
{current_content}

Improvements Needed:
{improvement_text}

Requirements:
- Topic: {requirements.get('topic', 'general')}
- Tone: {requirements.get('tone', 'professional')}
- Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words

Generate an improved version that:
1. Addresses ALL listed improvements
2. Maintains the core message and strengths
3. Improves clarity, structure, and quality
4. Meets all requirements

Generate the refined content:"""
        
        messages = [
            SystemMessage(content="You are an expert content editor who iteratively refines content to perfection."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class ConvergenceDetector:
    """Detect when refinement has converged"""
    
    def check_convergence(self, quality_progression: List[float],
                         max_iterations: int, current_iteration: int) -> bool:
        """Check if refinement should stop"""
        # Max iterations reached
        if current_iteration >= max_iterations:
            return True
        
        # Quality threshold reached
        if quality_progression and quality_progression[-1] >= 0.9:
            return True
        
        # Improvement plateaued (last 2 iterations show minimal improvement)
        if len(quality_progression) >= 3:
            recent_improvements = [
                quality_progression[i] - quality_progression[i-1]
                for i in range(-2, 0)
            ]
            
            if all(imp < 0.05 for imp in recent_improvements):
                return True  # Minimal improvement
        
        return False


# Agent functions
def initialize_refinement(state: IterativeRefinementState) -> IterativeRefinementState:
    """Initialize iterative refinement"""
    state["messages"].append(HumanMessage(
        content=f"Initializing iterative refinement: {state['task_description']}"
    ))
    state["current_iteration"] = 0
    state["convergence_detected"] = False
    state["current_step"] = "initialized"
    return state


def generate_initial_content(state: IterativeRefinementState) -> IterativeRefinementState:
    """Generate initial draft"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = InitialGenerator(llm)
    
    initial = generator.generate_initial(state["generation_requirements"])
    
    state["initial_content"] = initial
    state["current_content"] = initial
    
    state["messages"].append(HumanMessage(
        content=f"Generated initial draft: {len(initial.split())} words"
    ))
    state["current_step"] = "initial_generated"
    return state


def analyze_content(state: IterativeRefinementState) -> IterativeRefinementState:
    """Analyze current content and provide feedback"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    analyzer = FeedbackAnalyzer(llm)
    
    feedback = analyzer.analyze(
        state["current_content"],
        state["generation_requirements"],
        state["current_iteration"]
    )
    
    state["feedback_analysis"] = feedback
    state["quality_progression"].append(feedback["combined_score"])
    
    state["messages"].append(HumanMessage(
        content=f"Iteration {state['current_iteration']} analysis: "
                f"Quality score: {feedback['combined_score']:.2f} "
                f"(auto: {feedback['automated_feedback']['overall_score']:.2f}, "
                f"llm: {feedback['llm_feedback']['score']:.2f})"
    ))
    state["current_step"] = "content_analyzed"
    return state


def identify_improvements(state: IterativeRefinementState) -> IterativeRefinementState:
    """Identify areas for improvement"""
    identifier = ImprovementIdentifier()
    
    improvements = identifier.identify_improvements(
        state["feedback_analysis"],
        state["generation_requirements"]
    )
    
    state["improvement_areas"] = improvements
    
    high_priority = sum(1 for imp in improvements if imp["priority"] == "high")
    state["messages"].append(HumanMessage(
        content=f"Identified {len(improvements)} improvements "
                f"({high_priority} high priority)"
    ))
    state["current_step"] = "improvements_identified"
    return state


def check_convergence(state: IterativeRefinementState) -> IterativeRefinementState:
    """Check if refinement has converged"""
    detector = ConvergenceDetector()
    
    max_iterations = state["generation_requirements"].get("max_iterations", 5)
    
    converged = detector.check_convergence(
        state["quality_progression"],
        max_iterations,
        state["current_iteration"]
    )
    
    state["convergence_detected"] = converged
    
    if converged:
        if state["current_iteration"] >= max_iterations:
            reason = "Maximum iterations reached"
        elif state["quality_progression"][-1] >= 0.9:
            reason = "Quality threshold achieved"
        else:
            reason = "Improvement plateaued"
        
        state["messages"].append(HumanMessage(
            content=f"Convergence detected: {reason}"
        ))
    else:
        state["messages"].append(HumanMessage(
            content=f"Continuing refinement (iteration {state['current_iteration'] + 1})"
        ))
    
    state["current_step"] = "convergence_checked"
    return state


def refine_content(state: IterativeRefinementState) -> IterativeRefinementState:
    """Refine content based on feedback"""
    if state["convergence_detected"]:
        state["final_output"] = state["current_content"]
        return state
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    refiner = ContentRefiner(llm)
    
    refined = refiner.refine(
        state["current_content"],
        state["improvement_areas"],
        state["generation_requirements"],
        state["current_iteration"] + 1
    )
    
    # Store iteration
    state["refinement_iterations"].append({
        "iteration": state["current_iteration"],
        "content": state["current_content"],
        "quality": state["quality_progression"][-1],
        "improvements_addressed": len(state["improvement_areas"])
    })
    
    state["current_content"] = refined
    state["current_iteration"] += 1
    
    state["messages"].append(HumanMessage(
        content=f"Refined content for iteration {state['current_iteration']}"
    ))
    state["current_step"] = "content_refined"
    return state


def decide_next_step(state: IterativeRefinementState) -> str:
    """Decide next step based on convergence"""
    if state["convergence_detected"]:
        return "generate_report"
    else:
        return "analyze_content"


def generate_report(state: IterativeRefinementState) -> IterativeRefinementState:
    """Generate final refinement report"""
    initial_quality = state["quality_progression"][0] if state["quality_progression"] else 0.0
    final_quality = state["quality_progression"][-1] if state["quality_progression"] else 0.0
    improvement = final_quality - initial_quality
    
    report = f"""
ITERATIVE REFINEMENT REPORT
===========================

Task: {state['task_description']}

Refinement Summary:
- Total Iterations: {state['current_iteration']}
- Initial Quality: {initial_quality:.2f}
- Final Quality: {final_quality:.2f}
- Total Improvement: {improvement:+.2f}

Quality Progression:
"""
    
    for i, quality in enumerate(state['quality_progression']):
        change = ""
        if i > 0:
            delta = quality - state['quality_progression'][i-1]
            change = f" ({delta:+.2f})"
        report += f"Iteration {i}: {quality:.2f}{change}\n"
    
    if state['refinement_iterations']:
        report += f"""
Refinement History:
"""
        for iteration in state['refinement_iterations']:
            report += f"- Iteration {iteration['iteration']}: Quality {iteration['quality']:.2f}, "
            report += f"Addressed {iteration['improvements_addressed']} improvements\n"
    
    report += f"""
FINAL CONTENT:
{'-' * 50}
{state['current_content']}
{'-' * 50}

Final Content Statistics:
- Words: {len(state['current_content'].split())}
- Sentences: {len([s for s in state['current_content'].split('.') if s.strip()])}
- Paragraphs: {len([p for p in state['current_content'].split('\n\n') if p.strip()])}
- Improvement Rate: {(improvement / max(initial_quality, 0.1)):.1%}
"""
    
    state["final_output"] = state["current_content"]
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_iterative_refinement_graph():
    """Create the iterative refinement workflow graph"""
    workflow = StateGraph(IterativeRefinementState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_refinement)
    workflow.add_node("generate_initial", generate_initial_content)
    workflow.add_node("analyze_content", analyze_content)
    workflow.add_node("identify_improvements", identify_improvements)
    workflow.add_node("check_convergence", check_convergence)
    workflow.add_node("refine_content", refine_content)
    workflow.add_node("generate_report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "generate_initial")
    workflow.add_edge("generate_initial", "analyze_content")
    workflow.add_edge("analyze_content", "identify_improvements")
    workflow.add_edge("identify_improvements", "check_convergence")
    workflow.add_edge("check_convergence", "refine_content")
    
    # Conditional edge - continue refining or finish
    workflow.add_conditional_edges(
        "refine_content",
        decide_next_step,
        {
            "analyze_content": "analyze_content",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample iterative refinement task
    initial_state = {
        "task_description": "Generate refined content about machine learning through iterative improvement",
        "generation_requirements": {
            "topic": "Machine Learning in Healthcare: Transforming Patient Care",
            "purpose": "educational article",
            "min_words": 250,
            "max_words": 400,
            "tone": "professional yet accessible",
            "audience": "healthcare professionals",
            "keywords": ["machine learning", "healthcare", "patient care", "diagnosis", "AI"],
            "max_iterations": 5
        },
        "initial_content": "",
        "refinement_iterations": [],
        "current_iteration": 0,
        "current_content": "",
        "feedback_analysis": {},
        "improvement_areas": [],
        "quality_progression": [],
        "convergence_detected": False,
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_iterative_refinement_graph()
    
    print("Iterative Refinement MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Total Iterations: {result['current_iteration']}")
    print(f"Quality Improvement: {result['quality_progression'][-1] - result['quality_progression'][0]:+.2f}")

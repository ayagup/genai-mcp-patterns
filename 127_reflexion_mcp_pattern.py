"""
Reflexion MCP Pattern

This pattern implements self-reflection and iterative refinement where
the model critiques its own outputs and improves them through multiple iterations.

Key Features:
- Self-critique and reflection
- Iterative refinement
- Error identification and correction
- Learning from mistakes
- Quality improvement over iterations
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ReflexionState(TypedDict):
    """State for reflexion pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    current_attempt: str
    reflections: List[Dict]
    iteration: int
    max_iterations: int
    quality_threshold: float
    current_quality: float
    is_satisfactory: bool
    final_output: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.5)


# Generator Agent
def generator_agent(state: ReflexionState) -> ReflexionState:
    """Generates or refines solution based on reflections"""
    task = state.get("task", "")
    reflections = state.get("reflections", [])
    iteration = state.get("iteration", 0)
    
    # Build context from previous reflections
    reflection_context = ""
    if reflections:
        reflection_context = "\n\nPrevious Reflections:\n"
        for i, refl in enumerate(reflections):
            reflection_context += f"\nIteration {i+1}:\n"
            reflection_context += f"  Issues: {', '.join(refl.get('issues', []))}\n"
            reflection_context += f"  Suggestions: {', '.join(refl.get('suggestions', []))}\n"
    
    system_prompt = """You are a solution generator that improves with feedback.

Generate high-quality solutions, and if given reflections:
- Address identified issues
- Implement suggestions
- Improve quality
- Refine your approach"""
    
    user_prompt = f"""Task: {task}{reflection_context}

Generate {'an improved' if reflections else 'a'} solution:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    current_attempt = response.content
    
    report = f"""
    🎨 Generator Agent (Iteration {iteration + 1}):
    
    Generation Status:
    • Task: {task[:100]}...
    • Current Iteration: {iteration + 1}
    • Previous Reflections: {len(reflections)}
    • Improvement Focus: {', '.join(reflections[-1].get('suggestions', ['Initial generation']) if reflections else ['First attempt'])}
    
    Reflexion Framework:
    
    Core Concept:
    Agents reflect on their own outputs, identify mistakes,
    and iteratively improve through self-critique and refinement.
    
    The Reflexion Loop:
    
    1. Generate:
    • Create initial solution
    • Use best knowledge
    • Apply current understanding
    • Produce output
    
    2. Reflect:
    • Critique own work
    • Identify errors
    • Find weaknesses
    • Analyze failures
    
    3. Learn:
    • Extract lessons
    • Update strategy
    • Internalize feedback
    • Build meta-knowledge
    
    4. Refine:
    • Apply learnings
    • Fix identified issues
    • Improve quality
    • Iterate until satisfactory
    
    Reflexion vs Other Patterns:
    
    Reflexion vs CoT:
    • CoT: Single-pass reasoning
    • Reflexion: Multi-iteration refinement
    • Reflexion: Self-critique
    • Reflexion: Error correction
    
    Reflexion vs ReAct:
    • ReAct: Environment feedback
    • Reflexion: Self-generated feedback
    • ReAct: External observations
    • Reflexion: Internal reflection
    
    Benefits of Reflexion:
    
    Self-Improvement:
    • Learn from mistakes
    • Progressive refinement
    • Adaptive behavior
    • Quality increase
    
    Error Recovery:
    • Identify failures
    • Understand causes
    • Correct mistakes
    • Prevent recurrence
    
    Meta-Learning:
    • Learn how to learn
    • Strategy refinement
    • Approach optimization
    • Transferable skills
    
    Robustness:
    • Handle initial failures
    • Recover from errors
    • Multiple attempts
    • Higher success rate
    
    Types of Reflection:
    
    Error Reflection:
    • What went wrong?
    • Why did it fail?
    • What was incorrect?
    • Root cause analysis
    
    Quality Reflection:
    • How good is output?
    • What's missing?
    • What could improve?
    • Optimization opportunities
    
    Strategy Reflection:
    • Is approach working?
    • Better alternatives?
    • What to change?
    • Process improvement
    
    Learning Reflection:
    • What did I learn?
    • Patterns discovered?
    • Generalizable insights?
    • Future applications?
    
    Research (Shinn et al. 2023):
    
    Performance:
    • AlfWorld: 97% (vs 75% ReAct)
    • HotPotQA: 31% (vs 20% ReAct)
    • Programming: 91% (vs 67% baseline)
    • Significant improvements
    
    Key Insights:
    • Self-reflection is powerful
    • Iterative refinement works
    • Few trials needed (2-3)
    • Generalizes across tasks
    
    Trial-and-Error:
    • Try solution
    • Get feedback (or self-critique)
    • Reflect on errors
    • Try again with improvements
    
    Current Attempt Generated:
    {current_attempt[:300]}...
    
    Reflection Techniques:
    
    Self-Critique Prompting:
    ```
    Review your answer and critique it:
    - What errors exist?
    - What's incomplete?
    - What could be better?
    - How to improve?
    ```
    
    Comparative Reflection:
    ```
    Compare to:
    - Ideal solution
    - Previous attempts
    - Best practices
    - Expert examples
    ```
    
    Socratic Reflection:
    ```
    Ask yourself:
    - Why this approach?
    - What assumptions made?
    - Alternative views?
    - Logical consistency?
    ```
    
    Failure Analysis:
    ```
    If failed:
    - What went wrong?
    - Root cause?
    - What to change?
    - Prevention strategy?
    ```
    """
    
    return {
        "messages": [AIMessage(content=f"🎨 Generator Agent:\n{report}\n\n{response.content}")],
        "current_attempt": current_attempt,
        "iteration": iteration + 1
    }


# Reflector Agent
def reflector_agent(state: ReflexionState) -> ReflexionState:
    """Reflects on current solution and provides critique"""
    task = state.get("task", "")
    current_attempt = state.get("current_attempt", "")
    reflections = state.get("reflections", [])
    quality_threshold = state.get("quality_threshold", 0.8)
    max_iterations = state.get("max_iterations", 3)
    iteration = state.get("iteration", 0)
    
    system_prompt = """You are a critical evaluator that provides constructive feedback.

For each solution:
1. Identify specific issues or errors
2. Assess quality (0-10 scale)
3. Provide concrete improvement suggestions
4. Be constructive and specific

Format:
Issues: [list of problems]
Quality: [score]/10
Suggestions: [specific improvements]
Satisfactory: Yes/No"""
    
    user_prompt = f"""Task: {task}

Current Solution:
{current_attempt}

Provide detailed critique and evaluation:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse reflection
    issues = []
    suggestions = []
    quality = 5.0
    satisfactory = False
    
    for line in content.split("\n"):
        if line.startswith("Issues:"):
            issues_text = line.replace("Issues:", "").strip()
            issues = [i.strip() for i in issues_text.split(",") if i.strip()]
        elif line.startswith("Quality:"):
            quality_text = line.replace("Quality:", "").strip()
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', quality_text)
            if match:
                quality = float(match.group(1))
        elif line.startswith("Suggestions:"):
            sugg_text = line.replace("Suggestions:", "").strip()
            suggestions = [s.strip() for s in sugg_text.split(",") if s.strip()]
        elif line.startswith("Satisfactory:"):
            satisfactory = "yes" in line.lower()
    
    # Normalize quality to 0-1
    quality_normalized = quality / 10.0
    
    # Check if satisfactory
    is_satisfactory = (
        satisfactory or
        quality_normalized >= quality_threshold or
        iteration >= max_iterations
    )
    
    # Store reflection
    reflection_dict = {
        "iteration": iteration,
        "issues": issues,
        "suggestions": suggestions,
        "quality": quality_normalized,
        "satisfactory": is_satisfactory
    }
    
    final_output = current_attempt if is_satisfactory else ""
    
    summary = f"""
    🔍 Reflector Agent:
    
    Reflection Results:
    • Quality Score: {quality:.1f}/10 ({quality_normalized:.0%})
    • Issues Found: {len(issues)}
    • Satisfactory: {is_satisfactory}
    • Continue Iterating: {not is_satisfactory and iteration < max_iterations}
    
    Identified Issues:
    {chr(10).join(f"  • {issue}" for issue in issues) if issues else "  • None - solution looks good!"}
    
    Improvement Suggestions:
    {chr(10).join(f"  • {sugg}" for sugg in suggestions) if suggestions else "  • No major improvements needed"}
    
    Reflexion Implementation Patterns:
    
    Basic Reflexion Loop:
    ```python
    attempt = generate(task)
    for i in range(max_iterations):
        reflection = reflect(attempt)
        if reflection.satisfactory:
            break
        attempt = refine(attempt, reflection)
    return attempt
    ```
    
    Reflexion with Memory:
    ```python
    memory = []
    for trial in trials:
        attempt = generate(task, memory)
        result = execute(attempt)
        reflection = reflect(attempt, result)
        memory.append(reflection)
        if result.success:
            break
    ```
    
    Multi-Aspect Reflexion:
    ```python
    aspects = ['correctness', 'efficiency', 'style']
    for aspect in aspects:
        reflection = reflect_on(attempt, aspect)
        if not reflection.satisfactory:
            attempt = improve(attempt, aspect, reflection)
    ```
    
    Reflexion Strategies:
    
    Immediate Reflexion:
    • Reflect after each action
    • Quick corrections
    • Fine-grained feedback
    • Responsive adaptation
    
    Episodic Reflexion:
    • Reflect after task completion
    • Holistic view
    • Pattern recognition
    • Strategic learning
    
    Comparative Reflexion:
    • Compare multiple attempts
    • Identify best approach
    • Learn from variations
    • Ensemble insights
    
    Guided Reflexion:
    • Use rubrics/criteria
    • Structured evaluation
    • Consistent assessment
    • Objective metrics
    
    Advanced Reflexion Techniques:
    
    Hierarchical Reflexion:
    • Micro-level: individual steps
    • Macro-level: overall strategy
    • Meta-level: learning process
    • Multi-scale feedback
    
    Collaborative Reflexion:
    • Multiple agents reflect
    • Diverse perspectives
    • Cross-validation
    • Collective intelligence
    
    Counterfactual Reflexion:
    • What if I had done X?
    • Alternative scenarios
    • Explore missed opportunities
    • Learn from paths not taken
    
    Predictive Reflexion:
    • Will this approach work?
    • Anticipate issues
    • Proactive adjustment
    • Prevention vs correction
    
    Reflexion Best Practices:
    
    Specific Feedback:
    • Concrete examples
    • Actionable suggestions
    • Clear improvement path
    • Measurable criteria
    
    Balanced Critique:
    • Acknowledge strengths
    • Identify weaknesses
    • Constructive tone
    • Growth mindset
    
    Iteration Management:
    • Set max iterations
    • Quality thresholds
    • Diminishing returns
    • Stop criteria
    
    Learning Retention:
    • Store reflections
    • Build knowledge base
    • Transfer learnings
    • Cumulative improvement
    
    Applications:
    
    Code Generation:
    • Write code
    • Test and debug
    • Reflect on errors
    • Improve implementation
    
    Creative Writing:
    • Draft content
    • Critique style/flow
    • Refine narrative
    • Polish output
    
    Problem Solving:
    • Attempt solution
    • Check correctness
    • Identify mistakes
    • Correct approach
    
    Decision Making:
    • Make decision
    • Evaluate outcome
    • Learn from results
    • Adjust strategy
    
    Quality Metrics:
    
    Improvement Rate:
    • Quality gain per iteration
    • Convergence speed
    • Learning efficiency
    
    Success Rate:
    • Task completion %
    • After N iterations
    • vs baseline
    
    Reflection Quality:
    • Issue detection accuracy
    • Suggestion usefulness
    • Self-awareness level
    
    Current Quality: {quality_normalized:.0%}
    Threshold: {quality_threshold:.0%}
    Decision: {'✅ ACCEPT' if is_satisfactory else '🔄 ITERATE'}
    
    Key Insight:
    Reflexion enables agents to learn from their own mistakes
    through self-critique and iterative refinement, dramatically
    improving performance on complex tasks through trial and error.
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Reflector Agent:\n{summary}")],
        "reflections": reflections + [reflection_dict],
        "current_quality": quality_normalized,
        "is_satisfactory": is_satisfactory,
        "final_output": final_output
    }


# Build the graph
def build_reflexion_graph():
    """Build the reflexion pattern graph"""
    workflow = StateGraph(ReflexionState)
    
    workflow.add_node("generator_agent", generator_agent)
    workflow.add_node("reflector_agent", reflector_agent)
    
    # Conditional routing
    def should_continue(state: ReflexionState) -> str:
        """Determine if we should continue iterating"""
        if state.get("is_satisfactory", False):
            return "end"
        if state.get("iteration", 0) >= state.get("max_iterations", 3):
            return "end"
        return "continue"
    
    workflow.add_edge(START, "generator_agent")
    workflow.add_edge("generator_agent", "reflector_agent")
    
    workflow.add_conditional_edges(
        "reflector_agent",
        should_continue,
        {
            "continue": "generator_agent",
            "end": END
        }
    )
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_reflexion_graph()
    
    print("=== Reflexion MCP Pattern ===\n")
    
    # Test Case: Iterative solution improvement
    print("\n" + "="*70)
    print("TEST CASE: Iterative Refinement with Self-Reflection")
    print("="*70)
    
    state = {
        "messages": [],
        "task": "Write a Python function to find the longest palindromic substring in a string",
        "current_attempt": "",
        "reflections": [],
        "iteration": 0,
        "max_iterations": 3,
        "quality_threshold": 0.8,
        "current_quality": 0.0,
        "is_satisfactory": False,
        "final_output": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 127: Reflexion - COMPLETE")
    print(f"{'='*70}")

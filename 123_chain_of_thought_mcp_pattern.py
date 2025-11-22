"""
Chain-of-Thought MCP Pattern

This pattern implements step-by-step reasoning where the model explicitly
shows its thinking process before arriving at a conclusion.

Key Features:
- Explicit reasoning steps
- Transparent thought process
- Step-by-step problem solving
- Intermediate verification
- Enhanced accuracy for complex tasks
"""

from typing import TypedDict, Sequence, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ChainOfThoughtState(TypedDict):
    """State for chain-of-thought pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    problem: str
    reasoning_steps: List[str]
    final_answer: str
    show_work: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Reasoning Agent
def reasoning_agent(state: ChainOfThoughtState) -> ChainOfThoughtState:
    """Performs step-by-step reasoning"""
    problem = state.get("problem", "")
    show_work = state.get("show_work", True)
    
    # CoT prompt template
    system_prompt = """You are a logical reasoning assistant. For each problem:

1. Break down the problem into steps
2. Think through each step carefully
3. Show your reasoning explicitly
4. Verify your logic
5. Provide the final answer

Always use the format:
Thinking: [your step-by-step reasoning]
Answer: [final answer]"""
    
    user_prompt = f"""Problem: {problem}

Let's think step by step:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Parse reasoning steps from response
    content = response.content
    reasoning_steps = []
    final_answer = ""
    
    if "Thinking:" in content and "Answer:" in content:
        thinking_part = content.split("Answer:")[0].replace("Thinking:", "").strip()
        final_answer = content.split("Answer:")[1].strip()
        reasoning_steps = [step.strip() for step in thinking_part.split("\n") if step.strip()]
    else:
        # Fallback parsing
        lines = content.split("\n")
        for line in lines[:-1]:
            if line.strip():
                reasoning_steps.append(line.strip())
        if lines:
            final_answer = lines[-1].strip()
    
    report = f"""
    🧠 Reasoning Agent:
    
    Problem Analysis:
    • Problem: {problem[:100]}...
    • Reasoning Steps: {len(reasoning_steps)}
    • Show Work: {show_work}
    
    Chain-of-Thought Concepts:
    
    Core Principles:
    
    Explicit Reasoning:
    • Show thinking process
    • Make logic transparent
    • Explain each step
    • Build incrementally
    
    Step-by-Step:
    • Break down complexity
    • One step at a time
    • Sequential logic
    • Progressive refinement
    
    Verification:
    • Check intermediate results
    • Validate assumptions
    • Catch errors early
    • Ensure correctness
    
    Benefits of CoT:
    
    Accuracy:
    • Reduces errors
    • Better for complex tasks
    • Catches mistakes
    • Improves reliability
    
    Interpretability:
    • Understand reasoning
    • Debug issues
    • Trust decisions
    • Learn from process
    
    Generalization:
    • Works across domains
    • Flexible approach
    • Adaptable reasoning
    • Transferable skill
    
    CoT Prompting Techniques:
    
    Basic CoT:
    ```
    Q: [question]
    A: Let's think step by step.
    [reasoning steps]
    Therefore, [answer]
    ```
    
    Zero-Shot CoT:
    ```
    Q: [question]
    A: Let's think step by step.
    ```
    
    Few-Shot CoT:
    ```
    Q: [example question]
    A: Let's think step by step.
    [example reasoning]
    Therefore, [example answer]
    
    Q: [new question]
    A: Let's think step by step.
    ```
    
    Self-Consistency CoT:
    ```
    Generate multiple reasoning paths
    Vote on final answer
    Choose most consistent result
    ```
    
    Reasoning Steps:
    {chr(10).join(f"  {i+1}. {step}" for i, step in enumerate(reasoning_steps))}
    
    CoT Applications:
    
    Math Problems:
    • Arithmetic reasoning
    • Word problems
    • Multi-step calculations
    • Proof verification
    
    Logical Reasoning:
    • Deductive inference
    • Inductive reasoning
    • Syllogisms
    • Puzzle solving
    
    Common Sense:
    • Physical reasoning
    • Social situations
    • Causal relationships
    • Everyday scenarios
    
    Code Generation:
    • Algorithm design
    • Step planning
    • Bug analysis
    • Optimization
    
    Research Insights:
    
    Wei et al. (2022) "Chain-of-Thought Prompting":
    • 8x improvement on math problems
    • Emergent ability in large models
    • Better with scale (>100B params)
    • Works across tasks
    
    Zero-Shot CoT (Kojima et al.):
    • "Let's think step by step" trigger
    • No examples needed
    • Surprisingly effective
    • Simple implementation
    
    Self-Consistency (Wang et al.):
    • Sample multiple paths
    • Majority voting
    • Improves accuracy
    • Reduces errors
    """
    
    return {
        "messages": [AIMessage(content=f"🧠 Reasoning Agent:\n{report}\n\n{response.content}")],
        "reasoning_steps": reasoning_steps,
        "final_answer": final_answer
    }


# Answer Extractor
def answer_extractor(state: ChainOfThoughtState) -> ChainOfThoughtState:
    """Extracts and validates the final answer"""
    reasoning_steps = state.get("reasoning_steps", [])
    final_answer = state.get("final_answer", "")
    problem = state.get("problem", "")
    
    summary = f"""
    📊 CHAIN-OF-THOUGHT COMPLETE
    
    Solution Summary:
    • Problem: {problem[:80]}...
    • Reasoning Steps: {len(reasoning_steps)}
    • Final Answer: {final_answer[:100]}...
    
    CoT Pattern Process:
    1. Reasoning Agent → Step-by-step thinking
    2. Answer Extractor → Extract final result
    
    Advanced CoT Variants:
    
    Least-to-Most Prompting:
    • Decompose into subproblems
    • Solve easiest first
    • Build up to complex
    • Sequential dependency
    
    Example:
    ```
    To solve [complex problem]:
    First, let's solve: [subproblem 1]
    Next, using that: [subproblem 2]
    Finally: [complete solution]
    ```
    
    Program-Aided CoT:
    • Generate code
    • Execute for calculation
    • Combine reasoning + computation
    • Higher accuracy
    
    Example:
    ```python
    # Let's solve this step by step with code
    price = 50
    discount = 0.20
    final_price = price * (1 - discount)
    # Therefore, final price is $40
    ```
    
    Tree-of-Thought:
    • Multiple reasoning branches
    • Explore alternatives
    • Evaluate paths
    • Choose best route
    
    Verification CoT:
    • Generate answer with CoT
    • Verify with reverse reasoning
    • Cross-check steps
    • Confirm correctness
    
    CoT Best Practices:
    
    Prompt Design:
    • Clear step indicators
    • Encourage explicit reasoning
    • Request verification
    • Specify format
    
    Error Handling:
    • Catch logical errors
    • Backtrack when wrong
    • Alternative approaches
    • Self-correction
    
    Optimization:
    • Balance detail vs brevity
    • Focus on critical steps
    • Skip obvious steps
    • Adapt to complexity
    
    Evaluation:
    • Check final answer
    • Validate reasoning
    • Compare approaches
    • Measure accuracy
    
    Implementation Tips:
    
    Temperature Settings:
    • Use lower (0-0.3) for math
    • Higher (0.7-1.0) for creative
    • Adjust for task type
    • Balance exploration
    
    Prompt Engineering:
    • "Let's think step by step"
    • "First, let's break this down"
    • "Walking through this carefully"
    • "Let's solve this systematically"
    
    Quality Metrics:
    • Answer correctness
    • Reasoning validity
    • Step completeness
    • Logic consistency
    
    Key Insight:
    Chain-of-Thought dramatically improves performance on
    complex reasoning tasks by making the thought process
    explicit, enabling error detection and correction.
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Answer Extractor:\n{summary}")]
    }


# Build the graph
def build_cot_graph():
    """Build the chain-of-thought pattern graph"""
    workflow = StateGraph(ChainOfThoughtState)
    
    workflow.add_node("reasoning_agent", reasoning_agent)
    workflow.add_node("answer_extractor", answer_extractor)
    
    workflow.add_edge(START, "reasoning_agent")
    workflow.add_edge("reasoning_agent", "answer_extractor")
    workflow.add_edge("answer_extractor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_cot_graph()
    
    print("=== Chain-of-Thought MCP Pattern ===\n")
    
    # Test Case: Math Word Problem
    print("\n" + "="*70)
    print("TEST CASE: Mathematical Reasoning with CoT")
    print("="*70)
    
    state = {
        "messages": [],
        "problem": "A store sells apples at $3 per pound. If you buy 4 pounds and have a 20% discount coupon, how much do you pay?",
        "reasoning_steps": [],
        "final_answer": "",
        "show_work": True
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 123: Chain-of-Thought - COMPLETE")
    print(f"{'='*70}")

"""
Self-Ask MCP Pattern

This pattern implements question decomposition where the model asks itself
follow-up questions to break down complex queries into manageable sub-questions.

Key Features:
- Question decomposition
- Self-questioning strategy
- Hierarchical query breakdown
- Iterative sub-question answering
- Compositional reasoning
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SelfAskState(TypedDict):
    """State for self-ask pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    main_question: str
    sub_questions: List[Dict]
    answers: Dict[str, str]
    current_depth: int
    max_depth: int
    final_answer: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)


# Question Decomposer
def question_decomposer(state: SelfAskState) -> SelfAskState:
    """Decomposes complex questions into simpler sub-questions"""
    main_question = state.get("main_question", "")
    current_depth = state.get("current_depth", 0)
    answers = state.get("answers", {})
    
    # Build context from answered sub-questions
    context = ""
    if answers:
        context = "\n\nKnown Information:\n"
        for q, a in answers.items():
            context += f"• {q} → {a}\n"
    
    system_prompt = """You are a question decomposition expert. Break down complex questions into simpler sub-questions.

For each complex question:
1. Identify what information is needed
2. Generate specific sub-questions
3. Order them logically
4. Keep them atomic and answerable

Use this format:
Are follow-up questions needed here: Yes/No
Follow-up: [sub-question]
(repeat for each sub-question)"""
    
    user_prompt = f"""Main Question: {main_question}{context}

What follow-up questions do we need to answer this?"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse sub-questions
    sub_questions = []
    needs_followup = "yes" in content.lower().split("\n")[0]
    
    for line in content.split("\n"):
        if line.startswith("Follow-up:"):
            question = line.replace("Follow-up:", "").strip()
            sub_questions.append({
                "question": question,
                "depth": current_depth + 1,
                "answered": False
            })
    
    report = f"""
    ❓ Question Decomposer:
    
    Decomposition Analysis:
    • Main Question: {main_question[:100]}...
    • Needs Follow-up: {needs_followup}
    • Sub-Questions Generated: {len(sub_questions)}
    • Current Depth: {current_depth}
    
    Self-Ask Framework:
    
    Core Concept:
    Break down complex questions into a series of simpler
    sub-questions that can be answered sequentially to
    build toward the final answer.
    
    Decomposition Strategies:
    
    Temporal Decomposition:
    • Break by time periods
    • Sequential events
    • Before/after questions
    • Historical progression
    
    Example:
    "What events led to WW2?"
    → What was the Treaty of Versailles?
    → What was the Great Depression?
    → What was Hitler's rise to power?
    
    Spatial Decomposition:
    • Break by location
    • Geographic components
    • Regional aspects
    • Scale levels
    
    Example:
    "How does climate vary in USA?"
    → What's the climate in West Coast?
    → What's the climate in East Coast?
    → What's the climate in Midwest?
    
    Conceptual Decomposition:
    • Break by concepts
    • Component parts
    • Different aspects
    • Multiple perspectives
    
    Example:
    "How does photosynthesis work?"
    → What is chlorophyll?
    → What is light absorption?
    → What is glucose production?
    
    Computational Decomposition:
    • Break by operations
    • Step-by-step calculation
    • Intermediate values
    • Dependency chain
    
    Example:
    "What's 25% of $240 after 10% discount?"
    → What is 10% of $240?
    → What is $240 minus that amount?
    → What is 25% of the result?
    
    Generated Sub-Questions:
    {chr(10).join(f"  {i+1}. {sq['question']}" for i, sq in enumerate(sub_questions))}
    
    Self-Ask vs Other Patterns:
    
    Self-Ask vs CoT:
    • CoT: Continuous reasoning chain
    • Self-Ask: Discrete question steps
    • Self-Ask: More structured
    • Self-Ask: Explicit information needs
    
    Self-Ask vs ReAct:
    • ReAct: Actions + observations
    • Self-Ask: Questions + answers
    • ReAct: Tool interactions
    • Self-Ask: Pure decomposition
    
    Benefits of Self-Ask:
    
    Clarity:
    • Explicit information needs
    • Clear sub-goals
    • Structured breakdown
    • Transparent process
    
    Modularity:
    • Independent sub-questions
    • Reusable answers
    • Cacheable results
    • Parallel answering possible
    
    Traceability:
    • Track reasoning path
    • See dependencies
    • Understand composition
    • Debug easier
    
    Accuracy:
    • Focused sub-problems
    • Simpler to verify
    • Reduce errors
    • Incremental validation
    
    Question Types:
    
    Factual Questions:
    "What is X?"
    "When did Y happen?"
    "Where is Z located?"
    "Who did A?"
    
    Comparative Questions:
    "How does X compare to Y?"
    → What are properties of X?
    → What are properties of Y?
    → What are differences?
    
    Causal Questions:
    "Why did X happen?"
    → What preceded X?
    → What factors contributed?
    → What were the mechanisms?
    
    Compositional Questions:
    "What is the total/combined/aggregate?"
    → What is component A?
    → What is component B?
    → How do they combine?
    
    Research (Press et al. 2022):
    
    Performance:
    • Outperforms CoT on multi-hop QA
    • 60% → 68% on HotpotQA
    • Better decomposition
    • Clearer reasoning
    
    Interpretability:
    • Explicit question structure
    • Clear information flow
    • Easy to follow
    • Debuggable process
    
    Compositionality:
    • Answers build on each other
    • Modular reasoning
    • Flexible recombination
    • Extensible approach
    
    Question Depth Management:
    
    Depth 1: Direct questions
    • Immediately answerable
    • No further breakdown
    • Factual queries
    • Simple lookups
    
    Depth 2: One level decomposition
    • Break into 2-3 parts
    • Answer sub-questions
    • Combine results
    • Most common case
    
    Depth 3+: Deep decomposition
    • Recursive breakdown
    • Sub-sub-questions
    • Complex reasoning
    • Hierarchical structure
    
    Stopping Criteria:
    • All sub-questions answered
    • Max depth reached
    • No more questions needed
    • Sufficient information
    """
    
    return {
        "messages": [AIMessage(content=f"❓ Question Decomposer:\n{report}\n\n{response.content}")],
        "sub_questions": state.get("sub_questions", []) + sub_questions
    }


# Answer Composer
def answer_composer(state: SelfAskState) -> SelfAskState:
    """Answers sub-questions and composes final answer"""
    main_question = state.get("main_question", "")
    sub_questions = state.get("sub_questions", [])
    answers = state.get("answers", {})
    
    # Answer each sub-question
    for sq in sub_questions:
        if not sq.get("answered", False):
            question = sq["question"]
            
            # Answer the sub-question
            answer_prompt = f"""Answer this specific question concisely:

Question: {question}

Provide a direct, factual answer."""
            
            messages = [HumanMessage(content=answer_prompt)]
            response = llm.invoke(messages)
            
            answers[question] = response.content
            sq["answered"] = True
    
    # Compose final answer from sub-answers
    compose_prompt = f"""Main Question: {main_question}

Sub-Questions and Answers:
{chr(10).join(f"Q: {q}{chr(10)}A: {a}{chr(10)}" for q, a in answers.items())}

Now, synthesize a complete answer to the main question using the information above."""
    
    messages = [HumanMessage(content=compose_prompt)]
    final_response = llm.invoke(messages)
    final_answer = final_response.content
    
    summary = f"""
    ✅ Answer Composer:
    
    Composition Results:
    • Sub-Questions Answered: {len(answers)}
    • Final Answer Generated: Yes
    
    Sub-Answers:
    {chr(10).join(f"  Q: {q[:80]}...{chr(10)}  A: {a[:80]}...{chr(10)}" for q, a in list(answers.items())[:3])}
    
    Answer Composition Strategies:
    
    Sequential Composition:
    • Answer in order
    • Each builds on previous
    • Linear dependency
    • Step-by-step synthesis
    
    Parallel Composition:
    • Answer independently
    • Combine at end
    • No dependencies
    • Faster processing
    
    Hierarchical Composition:
    • Answer sub-sub-questions
    • Roll up to sub-questions
    • Finally main question
    • Tree structure
    
    Aggregation Composition:
    • Collect all answers
    • Aggregate information
    • Synthesize summary
    • Holistic view
    
    Self-Ask Implementation Patterns:
    
    Basic Self-Ask:
    ```
    Q: Main question
    Are follow-ups needed? Yes
    Follow-up: Sub-question 1
    Intermediate Answer: [answer 1]
    Follow-up: Sub-question 2
    Intermediate Answer: [answer 2]
    So the final answer is: [composed answer]
    ```
    
    Recursive Self-Ask:
    ```
    Q: Complex question
    Follow-up: Sub-question
      Follow-up: Sub-sub-question
      Intermediate: [sub-sub-answer]
    Intermediate: [sub-answer]
    Final: [answer]
    ```
    
    Self-Ask with Search:
    ```
    Q: Factual question
    Follow-up: What is X?
    Search: [query X]
    Intermediate: [result]
    Follow-up: What is Y?
    Search: [query Y]
    Intermediate: [result]
    Final: [composed answer]
    ```
    
    Advanced Techniques:
    
    Question Pruning:
    • Skip redundant questions
    • Remove answered questions
    • Optimize query set
    • Reduce overhead
    
    Question Reordering:
    • Optimize answer order
    • Dependencies first
    • Minimize wait time
    • Better flow
    
    Answer Caching:
    • Store sub-answers
    • Reuse across queries
    • Faster responses
    • Memory efficient
    
    Uncertainty Handling:
    • Mark uncertain answers
    • Request clarification
    • Provide confidence
    • Multiple interpretations
    
    Self-Ask Best Practices:
    
    Question Quality:
    • Specific and focused
    • Actually answerable
    • Independent when possible
    • Clear and unambiguous
    
    Decomposition Depth:
    • Balance granularity
    • Avoid over-decomposition
    • Stop when answerable
    • Practical limits
    
    Answer Integration:
    • Check consistency
    • Resolve conflicts
    • Maintain coherence
    • Logical flow
    
    Error Recovery:
    • Handle unanswerable questions
    • Skip if needed
    • Partial answers OK
    • Graceful degradation
    
    Use Cases:
    
    Multi-Hop QA:
    "Who is the spouse of the director of Titanic?"
    → Who directed Titanic? (James Cameron)
    → Who is James Cameron's spouse? (Suzy Amis)
    
    Complex Calculations:
    "If revenue grew 20% to $1.2M, what was original?"
    → What equation relates growth and final value?
    → How to solve for original value?
    → What is the calculation?
    
    Research Questions:
    "What caused the fall of Rome?"
    → What were economic factors?
    → What were military factors?
    → What were political factors?
    → How did these interact?
    
    Final Answer:
    {final_answer[:300]}...
    
    Key Insight:
    Self-Ask excels at complex questions requiring multiple
    pieces of information by explicitly decomposing into
    manageable sub-questions and systematically answering them.
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Answer Composer:\n{summary}")],
        "answers": answers,
        "final_answer": final_answer,
        "sub_questions": sub_questions
    }


# Build the graph
def build_self_ask_graph():
    """Build the self-ask pattern graph"""
    workflow = StateGraph(SelfAskState)
    
    workflow.add_node("question_decomposer", question_decomposer)
    workflow.add_node("answer_composer", answer_composer)
    
    workflow.add_edge(START, "question_decomposer")
    workflow.add_edge("question_decomposer", "answer_composer")
    workflow.add_edge("answer_composer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_self_ask_graph()
    
    print("=== Self-Ask MCP Pattern ===\n")
    
    # Test Case: Multi-hop question
    print("\n" + "="*70)
    print("TEST CASE: Complex Multi-Hop Question with Self-Ask")
    print("="*70)
    
    state = {
        "messages": [],
        "main_question": "What is the population of the birthplace of the author of 'The Great Gatsby'?",
        "sub_questions": [],
        "answers": {},
        "current_depth": 0,
        "max_depth": 3,
        "final_answer": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 126: Self-Ask - COMPLETE")
    print(f"{'='*70}")

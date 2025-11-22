"""
Prompt Chaining MCP Pattern

This pattern implements sequential prompt composition where outputs
from one prompt become inputs to the next, creating a multi-stage pipeline.

Key Features:
- Multi-stage prompt sequences
- Output-to-input forwarding
- Pipeline orchestration
- Intermediate result processing
- Complex workflow composition
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class PromptChainingState(TypedDict):
    """State for prompt chaining pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    initial_input: str
    chain_stages: List[Dict]
    current_stage: int
    intermediate_outputs: List[str]
    final_output: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.4)


# Stage Executor
def stage_executor(state: PromptChainingState) -> PromptChainingState:
    """Executes the current stage in the prompt chain"""
    current_stage = state.get("current_stage", 0)
    chain_stages = state.get("chain_stages", [])
    intermediate_outputs = state.get("intermediate_outputs", [])
    initial_input = state.get("initial_input", "")
    
    if current_stage >= len(chain_stages):
        # All stages complete
        return {
            "messages": [AIMessage(content="All chain stages complete")]
        }
    
    # Get current stage configuration
    stage = chain_stages[current_stage]
    stage_name = stage.get("name", f"Stage {current_stage + 1}")
    stage_prompt_template = stage.get("prompt_template", "")
    stage_description = stage.get("description", "")
    
    # Determine input for this stage
    if current_stage == 0:
        stage_input = initial_input
    else:
        stage_input = intermediate_outputs[-1] if intermediate_outputs else initial_input
    
    # Execute stage
    system_prompt = f"""You are executing stage {current_stage + 1} of a prompt chain.

Stage: {stage_name}
Description: {stage_description}

Process the input and produce the required output."""
    
    user_prompt = stage_prompt_template.format(input=stage_input)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    stage_output = response.content
    
    report = f"""
    🔗 Stage Executor (Stage {current_stage + 1}/{len(chain_stages)}):
    
    Stage Information:
    • Stage Name: {stage_name}
    • Description: {stage_description}
    • Input Length: {len(stage_input)} chars
    • Output Length: {len(stage_output)} chars
    
    Prompt Chaining Concepts:
    
    Core Idea:
    Break complex tasks into sequential stages where each
    stage's output feeds into the next stage's input.
    
    Chain Architecture:
    
    Linear Chain:
    Input → Stage 1 → Stage 2 → Stage 3 → Output
    
    Sequential Processing:
    • Each stage has specific role
    • Output becomes next input
    • Progressive transformation
    • Modular components
    
    Benefits of Prompt Chaining:
    
    Modularity:
    • Separate concerns
    • Reusable stages
    • Independent testing
    • Easy maintenance
    
    Clarity:
    • Clear responsibilities
    • Explicit transformations
    • Traceable flow
    • Understandable process
    
    Flexibility:
    • Add/remove stages
    • Reorder steps
    • Conditional branching
    • Dynamic composition
    
    Quality:
    • Focused prompts
    • Better results per stage
    • Error isolation
    • Incremental improvement
    
    Chain Patterns:
    
    Sequential Chain:
    ```
    result1 = prompt1(input)
    result2 = prompt2(result1)
    result3 = prompt3(result2)
    return result3
    ```
    
    Transform Chain:
    ```
    extract = extraction_prompt(document)
    summarize = summary_prompt(extract)
    format = formatting_prompt(summarize)
    return format
    ```
    
    Refinement Chain:
    ```
    draft = draft_prompt(topic)
    expand = expansion_prompt(draft)
    polish = polish_prompt(expand)
    return polish
    ```
    
    Validation Chain:
    ```
    generate = generation_prompt(query)
    verify = verification_prompt(generate)
    if not verify.valid:
        generate = regeneration_prompt(generate, verify)
    return generate
    ```
    
    Common Chain Stages:
    
    Information Extraction:
    • Input: Raw text/document
    • Process: Extract key information
    • Output: Structured data
    • Example: Pull facts from article
    
    Transformation:
    • Input: Data in format A
    • Process: Convert to format B
    • Output: Data in format B
    • Example: JSON to narrative
    
    Analysis:
    • Input: Data/information
    • Process: Analyze and evaluate
    • Output: Insights/assessment
    • Example: Sentiment analysis
    
    Synthesis:
    • Input: Multiple pieces
    • Process: Combine and summarize
    • Output: Coherent summary
    • Example: Multi-doc summary
    
    Generation:
    • Input: Requirements/context
    • Process: Create content
    • Output: New content
    • Example: Draft article
    
    Validation:
    • Input: Generated content
    • Process: Check quality/correctness
    • Output: Validation result
    • Example: Fact-check response
    
    Current Stage Input:
    {stage_input[:200]}...
    
    Current Stage Output:
    {stage_output[:200]}...
    
    Chain Stage Types:
    
    Map Stage:
    • Apply function to each item
    • Parallel processing possible
    • Independent transformations
    • Aggregate results
    
    Reduce Stage:
    • Combine multiple inputs
    • Aggregation logic
    • Summary generation
    • Consolidation
    
    Filter Stage:
    • Select relevant items
    • Quality checking
    • Criteria-based filtering
    • Data cleaning
    
    Enrich Stage:
    • Add information
    • Lookup additional data
    • Context enhancement
    • Augmentation
    
    Prompt Chaining vs Other Patterns:
    
    Chaining vs Single Prompt:
    • Single: One complex prompt
    • Chain: Multiple focused prompts
    • Chain: Better modularity
    • Chain: Easier debugging
    
    Chaining vs Parallel:
    • Parallel: Independent execution
    • Chain: Sequential dependency
    • Parallel: Faster
    • Chain: Progressive refinement
    
    Implementation Strategies:
    
    Eager Execution:
    • Execute stages immediately
    • No optimization
    • Simpler implementation
    • Immediate results
    
    Lazy Execution:
    • Build chain definition
    • Execute when needed
    • Can optimize
    • Efficient resource use
    
    Cached Execution:
    • Cache intermediate results
    • Reuse when possible
    • Faster re-execution
    • Memory trade-off
    
    Parallel Execution:
    • Run independent stages parallel
    • Faster completion
    • Resource intensive
    • Complex orchestration
    
    Research & Applications:
    
    LangChain Framework:
    • Sequential chains
    • Transform chains
    • Router chains
    • Map-reduce chains
    
    Use Cases:
    • Document processing
    • Content generation
    • Data analysis
    • Multi-step reasoning
    
    Performance:
    • Better accuracy per stage
    • Higher overall quality
    • More reliable results
    • Easier optimization
    """
    
    return {
        "messages": [AIMessage(content=f"🔗 Stage {current_stage + 1} ({stage_name}):\n{report}\n\nOutput:\n{stage_output}")],
        "intermediate_outputs": intermediate_outputs + [stage_output],
        "current_stage": current_stage + 1
    }


# Chain Composer
def chain_composer(state: PromptChainingState) -> PromptChainingState:
    """Composes final output from chain execution"""
    intermediate_outputs = state.get("intermediate_outputs", [])
    chain_stages = state.get("chain_stages", [])
    initial_input = state.get("initial_input", "")
    
    final_output = intermediate_outputs[-1] if intermediate_outputs else ""
    
    summary = f"""
    ✅ Chain Composer:
    
    Chain Execution Summary:
    • Total Stages: {len(chain_stages)}
    • Stages Completed: {len(intermediate_outputs)}
    • Initial Input: {initial_input[:80]}...
    • Final Output Length: {len(final_output)} chars
    
    Chain Execution Flow:
    {chr(10).join(f"  Stage {i+1} ({stage['name']}): {len(output)} chars" for i, (stage, output) in enumerate(zip(chain_stages, intermediate_outputs)))}
    
    Advanced Chaining Patterns:
    
    Conditional Chaining:
    ```python
    result = stage1(input)
    if condition(result):
        result = stage2a(result)
    else:
        result = stage2b(result)
    result = stage3(result)
    ```
    
    Branching Chain:
    ```python
    results = []
    for branch in branches:
        result = execute_branch(input, branch)
        results.append(result)
    final = merge_results(results)
    ```
    
    Iterative Chain:
    ```python
    result = input
    for i in range(max_iterations):
        result = refine_stage(result)
        if quality_check(result):
            break
    return result
    ```
    
    Feedback Chain:
    ```python
    result = generate(input)
    critique = evaluate(result)
    while not critique.satisfactory:
        result = improve(result, critique)
        critique = evaluate(result)
    return result
    ```
    
    Chain Optimization:
    
    Stage Fusion:
    • Combine similar stages
    • Reduce API calls
    • Faster execution
    • Lower cost
    
    Parallel Execution:
    • Identify independent stages
    • Execute simultaneously
    • Merge results
    • Faster completion
    
    Caching:
    • Cache stage outputs
    • Reuse for similar inputs
    • Skip redundant computation
    • Cost savings
    
    Early Stopping:
    • Quality thresholds
    • Skip unnecessary stages
    • Faster when possible
    • Adaptive execution
    
    Chain Monitoring:
    
    Stage Metrics:
    • Execution time per stage
    • Output quality scores
    • Error rates
    • Cost per stage
    
    Chain Metrics:
    • Total execution time
    • End-to-end quality
    • Success rate
    • Total cost
    
    Debugging:
    • Inspect intermediate outputs
    • Identify failing stages
    • Test stages independently
    • Trace data flow
    
    Optimization Insights:
    • Bottleneck identification
    • Cost analysis
    • Quality tracking
    • Performance tuning
    
    Chain Error Handling:
    
    Retry Logic:
    ```python
    for attempt in range(max_retries):
        try:
            result = stage(input)
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            continue
    ```
    
    Fallback Stages:
    ```python
    try:
        result = primary_stage(input)
    except Exception:
        result = fallback_stage(input)
    ```
    
    Partial Success:
    ```python
    results = []
    for stage in stages:
        try:
            result = stage(input)
            results.append(result)
            input = result
        except Exception as e:
            log_error(stage, e)
            # Continue with last successful result
    return results[-1] if results else None
    ```
    
    Chain Design Best Practices:
    
    Stage Granularity:
    • Not too fine-grained
    • Not too coarse
    • Logical boundaries
    • Reusable components
    
    Input/Output Contracts:
    • Clear interfaces
    • Consistent formats
    • Type safety
    • Validation
    
    Error Handling:
    • Graceful degradation
    • Informative errors
    • Recovery strategies
    • Logging
    
    Testing:
    • Unit test stages
    • Integration test chain
    • Edge case handling
    • Performance testing
    
    Real-World Chain Examples:
    
    Document Analysis Chain:
    1. Extract text from PDF
    2. Identify key sections
    3. Summarize each section
    4. Generate overall summary
    5. Format as report
    
    Content Generation Chain:
    1. Research topic
    2. Create outline
    3. Generate draft
    4. Expand sections
    5. Edit and polish
    6. Format output
    
    Data Processing Chain:
    1. Load raw data
    2. Clean and validate
    3. Extract features
    4. Analyze patterns
    5. Generate insights
    6. Create visualization
    
    Customer Support Chain:
    1. Classify query type
    2. Extract key information
    3. Search knowledge base
    4. Generate response
    5. Personalize tone
    6. Add follow-up suggestions
    
    Chain Composition Patterns:
    
    Nested Chains:
    ```python
    def process_document(doc):
        # Outer chain
        sections = extract_sections(doc)
        summaries = []
        for section in sections:
            # Inner chain for each section
            clean = clean_text(section)
            points = extract_points(clean)
            summary = summarize_points(points)
            summaries.append(summary)
        return combine_summaries(summaries)
    ```
    
    Dynamic Chains:
    ```python
    def build_chain(task_type):
        stages = [stage1]  # Always start with stage1
        
        if task_type == "analysis":
            stages.extend([analyze, visualize])
        elif task_type == "generation":
            stages.extend([draft, refine])
        
        stages.append(format)  # Always end with format
        return stages
    ```
    
    Final Output:
    {final_output[:300]}...
    
    Key Insight:
    Prompt Chaining enables complex multi-step workflows by
    breaking them into focused, sequential stages where each
    stage builds on the previous output, creating powerful
    composable AI pipelines.
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Chain Composer:\n{summary}")],
        "final_output": final_output
    }


# Build the graph
def build_prompt_chaining_graph():
    """Build the prompt chaining pattern graph"""
    workflow = StateGraph(PromptChainingState)
    
    workflow.add_node("stage_executor", stage_executor)
    workflow.add_node("chain_composer", chain_composer)
    
    # Conditional routing for multi-stage execution
    def should_continue(state: PromptChainingState) -> str:
        """Determine if more stages to execute"""
        current_stage = state.get("current_stage", 0)
        chain_stages = state.get("chain_stages", [])
        
        if current_stage >= len(chain_stages):
            return "compose"
        return "execute"
    
    workflow.add_edge(START, "stage_executor")
    
    workflow.add_conditional_edges(
        "stage_executor",
        should_continue,
        {
            "execute": "stage_executor",
            "compose": "chain_composer"
        }
    )
    
    workflow.add_edge("chain_composer", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_prompt_chaining_graph()
    
    print("=== Prompt Chaining MCP Pattern ===\n")
    
    # Test Case: Multi-stage content generation
    print("\n" + "="*70)
    print("TEST CASE: Multi-Stage Content Generation Chain")
    print("="*70)
    
    state = {
        "messages": [],
        "initial_input": "artificial intelligence in healthcare",
        "chain_stages": [
            {
                "name": "Research",
                "description": "Research key points about the topic",
                "prompt_template": "Research and list 3-5 key points about: {input}"
            },
            {
                "name": "Outline",
                "description": "Create a structured outline",
                "prompt_template": "Create a clear outline based on these points: {input}"
            },
            {
                "name": "Draft",
                "description": "Generate initial draft",
                "prompt_template": "Write a concise draft following this outline: {input}"
            }
        ],
        "current_stage": 0,
        "intermediate_outputs": [],
        "final_output": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 129: Prompt Chaining - COMPLETE")
    print(f"{'='*70}")

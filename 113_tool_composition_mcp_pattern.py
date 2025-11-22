"""
Tool Composition MCP Pattern

This pattern implements tool composition where multiple tools are
combined to create more complex functionality.

Key Features:
- Tool pipeline composition
- Dynamic workflow building
- Input/output transformation
- Parallel and sequential execution
- Composition optimization
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ToolCompositionState(TypedDict):
    """State for tool composition pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    tools: List[Dict]  # Available tools
    composition_plan: List[Dict]  # [{tool, order, inputs, outputs}]
    execution_results: Dict[str, any]
    final_output: any


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Composition Planner
def composition_planner(state: ToolCompositionState) -> ToolCompositionState:
    """Plans tool composition workflow"""
    tools = state.get("tools", [])
    
    system_message = SystemMessage(content="""You are a composition planner.
    Design optimal tool compositions to achieve complex goals.""")
    
    user_message = HumanMessage(content=f"""Plan tool composition:

Available Tools: {len(tools) if tools else 'Initializing'}

Create composition workflow.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define available tools if not provided
    if not tools:
        tools = [
            {"name": "web_search", "inputs": ["query"], "outputs": ["results"]},
            {"name": "text_extractor", "inputs": ["results"], "outputs": ["text"]},
            {"name": "summarizer", "inputs": ["text"], "outputs": ["summary"]},
            {"name": "translator", "inputs": ["summary"], "outputs": ["translated"]},
            {"name": "formatter", "inputs": ["translated"], "outputs": ["formatted"]}
        ]
    
    # Create composition plan: Search → Extract → Summarize → Translate → Format
    composition_plan = [
        {"tool": "web_search", "order": 1, "inputs": {"query": "AI trends 2024"}, "outputs": ["results"]},
        {"tool": "text_extractor", "order": 2, "inputs": {"results": "results"}, "outputs": ["text"]},
        {"tool": "summarizer", "order": 3, "inputs": {"text": "text"}, "outputs": ["summary"]},
        {"tool": "translator", "order": 4, "inputs": {"summary": "summary"}, "outputs": ["translated"]},
        {"tool": "formatter", "order": 5, "inputs": {"translated": "translated"}, "outputs": ["formatted"]}
    ]
    
    report = f"""
    🔧 Composition Planner:
    
    Composition Plan:
    • Total Tools: {len(tools)}
    • Pipeline Steps: {len(composition_plan)}
    • Execution Mode: Sequential
    
    Tool Composition Patterns:
    
    Sequential Composition:
    ```python
    result = tool3(tool2(tool1(input)))
    ```
    
    Parallel Composition:
    ```python
    results = [tool(input) for tool in parallel_tools]
    final = aggregator(results)
    ```
    
    Conditional Composition:
    ```python
    if condition:
        result = tool_a(input)
    else:
        result = tool_b(input)
    ```
    
    Pipeline Definition:
    {chr(10).join(f"  Step {step['order']}: {step['tool']} - Input: {step['inputs']}" for step in composition_plan)}
    
    Composition Strategies:
    
    Function Composition:
    ```python
    from functools import reduce
    
    def compose(*functions):
        return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)
    
    # Usage
    pipeline = compose(formatter, translator, summarizer, extractor, searcher)
    result = pipeline(query)
    ```
    
    Pipeline Pattern:
    ```python
    class ToolPipeline:
        def __init__(self):
            self.steps = []
        
        def add_step(self, tool, transform=None):
            self.steps.append({{"tool": tool, "transform": transform}})
            return self
        
        def execute(self, input_data):
            data = input_data
            
            for step in self.steps:
                # Execute tool
                result = step["tool"].run(data)
                
                # Apply transformation if provided
                if step["transform"]:
                    data = step["transform"](result)
                else:
                    data = result
            
            return data
    
    # Usage
    pipeline = ToolPipeline()
    pipeline.add_step(search_tool)
    pipeline.add_step(extract_tool, transform=lambda x: x["text"])
    pipeline.add_step(summarize_tool)
    result = pipeline.execute({{"query": "AI trends"}})
    ```
    """
    
    return {
        "messages": [AIMessage(content=f"🔧 Planner:\n{response.content}\n{report}")],
        "tools": tools,
        "composition_plan": composition_plan
    }


# Composition Executor
def composition_executor(state: ToolCompositionState) -> ToolCompositionState:
    """Executes the composition plan"""
    composition_plan = state.get("composition_plan", [])
    
    system_message = SystemMessage(content="""You are a composition executor.
    Execute tool compositions according to the plan.""")
    
    user_message = HumanMessage(content=f"""Execute composition:

Pipeline Steps: {len(composition_plan)}

Run tool pipeline.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate execution
    execution_results = {
        "results": ["AI trends article 1", "AI trends article 2", "AI trends article 3"],
        "text": "Artificial Intelligence continues to evolve with advancements in LLMs, multi-agent systems, and autonomous agents...",
        "summary": "AI in 2024: Major advances in LLMs, agents, and automation",
        "translated": "IA en 2024: Avances importantes en LLMs, agentes y automatización",
        "formatted": "# AI in 2024\n\n**Summary**: Major advances in LLMs, agents, and automation\n\n**Translation**: IA en 2024..."
    }
    
    final_output = execution_results["formatted"]
    
    summary = f"""
    📊 TOOL COMPOSITION COMPLETE
    
    Execution Summary:
    • Steps Executed: {len(composition_plan)}
    • Intermediate Results: {len(execution_results)}
    • Final Output: Generated
    
    Composition Process:
    {chr(10).join(f"  {step['order']}. {step['tool']}: {list(step['inputs'].keys())[0]} → {step['outputs'][0]}" for step in composition_plan)}
    
    Advanced Composition Patterns:
    
    LangChain Expression Language (LCEL):
    ```python
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.prompts import ChatPromptTemplate
    
    # Simple chain
    chain = prompt | llm | output_parser
    
    # Parallel composition
    chain = {{
        "search_results": search_tool,
        "context": context_retriever
    }} | prompt | llm
    
    # Branching
    branch = RunnableBranch(
        (lambda x: x["language"] == "en", english_chain),
        (lambda x: x["language"] == "es", spanish_chain),
        default_chain
    )
    ```
    
    Best Practices:
    • Validate intermediate outputs
    • Handle errors gracefully
    • Log execution steps
    • Optimize data flow
    • Cache expensive operations
    
    Key Insight:
    Tool composition enables building complex
    workflows from simple, reusable tool components.
    """
    
    return {
        "messages": [AIMessage(content=f"⚡ Executor:\n{response.content}\n{summary}")],
        "execution_results": execution_results,
        "final_output": final_output
    }


# Build the graph
def build_tool_composition_graph():
    """Build the tool composition pattern graph"""
    workflow = StateGraph(ToolCompositionState)
    
    workflow.add_node("planner", composition_planner)
    workflow.add_node("executor", composition_executor)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_tool_composition_graph()
    
    print("=== Tool Composition MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "tools": [],
        "composition_plan": [],
        "execution_results": {},
        "final_output": None
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nComposition Results:")
    print(f"Pipeline Steps: {len(result.get('composition_plan', []))}")
    print(f"Output Generated: {'Yes' if result.get('final_output') else 'No'}")

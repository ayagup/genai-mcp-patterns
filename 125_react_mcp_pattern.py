"""
ReAct MCP Pattern (Reasoning + Acting)

This pattern implements the ReAct paradigm where the model alternates between
reasoning about the problem and taking actions (using tools) to gather information.

Key Features:
- Reasoning and acting interleaved
- Tool usage integration
- Observation processing
- Iterative problem solving
- Dynamic action selection
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ReActState(TypedDict):
    """State for ReAct pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    reasoning_trace: List[str]
    actions_taken: List[Dict]
    observations: List[str]
    max_iterations: int
    current_iteration: int
    task_complete: bool
    final_answer: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.2)


# Reasoning Agent
def reasoning_agent(state: ReActState) -> ReActState:
    """Generates reasoning and decides next action"""
    task = state.get("task", "")
    actions_taken = state.get("actions_taken", [])
    observations = state.get("observations", [])
    current_iteration = state.get("current_iteration", 0)
    
    # Build context from previous actions and observations
    context = ""
    if actions_taken and observations:
        for i, (action, obs) in enumerate(zip(actions_taken, observations)):
            context += f"\nStep {i+1}:\n"
            context += f"Thought: {action.get('reasoning', '')}\n"
            context += f"Action: {action.get('action', '')} ({action.get('tool', '')})\n"
            context += f"Observation: {obs}\n"
    
    system_prompt = """You are a ReAct agent that reasons and acts iteratively.

For each step, follow this format:
Thought: [your reasoning about what to do next]
Action: [the action to take]
Tool: [which tool to use: search, calculate, lookup, or final_answer]
Action Input: [input for the tool]

Available Tools:
- search: Search for information online
- calculate: Perform calculations
- lookup: Look up specific facts
- final_answer: Provide final answer when task is complete

Keep iterating until you can provide a final answer."""
    
    user_prompt = f"""Task: {task}

{context}

What should we do next? Provide your Thought, Action, Tool, and Action Input."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse ReAct components
    thought = ""
    action = ""
    tool = ""
    action_input = ""
    
    for line in content.split("\n"):
        if line.startswith("Thought:"):
            thought = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            action = line.replace("Action:", "").strip()
        elif line.startswith("Tool:"):
            tool = line.replace("Tool:", "").strip()
        elif line.startswith("Action Input:"):
            action_input = line.replace("Action Input:", "").strip()
    
    # Record action
    action_dict = {
        "iteration": current_iteration + 1,
        "reasoning": thought,
        "action": action,
        "tool": tool,
        "input": action_input
    }
    
    report = f"""
    🧠 Reasoning Agent (Iteration {current_iteration + 1}):
    
    ReAct Trace:
    • Task: {task[:100]}...
    • Current Iteration: {current_iteration + 1}
    • Actions Taken: {len(actions_taken)}
    
    Current Step:
    Thought: {thought}
    Action: {action}
    Tool: {tool}
    Input: {action_input}
    
    ReAct Framework:
    
    Core Concept:
    Synergize Reasoning and Acting in language models.
    Alternate between thinking and doing.
    
    The ReAct Loop:
    
    1. Thought (Reasoning):
    • Analyze current situation
    • Reason about what's needed
    • Plan next action
    • Reflect on observations
    
    2. Action (Acting):
    • Execute tool/action
    • Gather information
    • Perform computation
    • Interact with environment
    
    3. Observation:
    • Receive action results
    • Process new information
    • Update understanding
    • Inform next reasoning
    
    4. Repeat:
    • Until task complete
    • Or max iterations reached
    • Iterative refinement
    • Progressive solution
    
    ReAct vs Other Patterns:
    
    ReAct vs Chain-of-Thought:
    • CoT: Pure reasoning, no actions
    • ReAct: Reasoning + Actions interleaved
    • ReAct: Can gather new information
    • ReAct: Interactive problem solving
    
    ReAct vs Acting-Only:
    • Acting: Random/scripted actions
    • ReAct: Reasoned, purposeful actions
    • ReAct: Explains why each action
    • ReAct: More interpretable
    
    Benefits of ReAct:
    
    Grounding:
    • Actions based on reasoning
    • Reasoning guided by observations
    • Reduces hallucination
    • Facts from environment
    
    Interpretability:
    • See thought process
    • Understand action choices
    • Track reasoning trace
    • Debug errors
    
    Flexibility:
    • Adapt to observations
    • Change strategy dynamically
    • Handle unexpected results
    • Robust to errors
    
    Reliability:
    • Verify with actions
    • Cross-check reasoning
    • Factual grounding
    • Error recovery
    
    Tool Integration:
    
    Search Tools:
    • Web search
    • Database lookup
    • API calls
    • Information retrieval
    
    Computation Tools:
    • Calculator
    • Code execution
    • Data processing
    • Simulations
    
    Memory Tools:
    • Store information
    • Retrieve context
    • Update knowledge
    • Long-term memory
    
    Communication Tools:
    • Ask clarifying questions
    • Request feedback
    • Collaborate
    • Delegate tasks
    
    Research Insights (Yao et al. 2022):
    
    Performance:
    • Outperforms CoT on QA tasks
    • 27% → 62% on HotpotQA
    • Better factual grounding
    • Reduces errors
    
    Synergy:
    • Reasoning helps action selection
    • Actions inform reasoning
    • Greater than sum of parts
    • Emergent capabilities
    
    Applications:
    • Question answering
    • Fact verification
    • Interactive tasks
    • Multi-step reasoning
    
    Previous Steps Summary:
    {chr(10).join(f"  Step {i+1}: {a.get('tool', '')} - {a.get('action', '')[:60]}..." for i, a in enumerate(actions_taken))}
    """
    
    return {
        "messages": [AIMessage(content=f"🧠 Reasoning Agent:\n{report}\n\n{response.content}")],
        "reasoning_trace": state.get("reasoning_trace", []) + [thought],
        "actions_taken": actions_taken + [action_dict],
        "current_iteration": current_iteration + 1
    }


# Action Executor
def action_executor(state: ReActState) -> ReActState:
    """Simulates tool execution and returns observations"""
    actions_taken = state.get("actions_taken", [])
    observations = state.get("observations", [])
    max_iterations = state.get("max_iterations", 5)
    current_iteration = state.get("current_iteration", 0)
    
    if not actions_taken:
        observation = "No action taken yet."
    else:
        last_action = actions_taken[-1]
        tool = last_action.get("tool", "")
        action_input = last_action.get("input", "")
        
        # Simulate tool execution (in real implementation, call actual tools)
        if tool == "final_answer":
            observation = f"Task complete. Final answer: {action_input}"
            task_complete = True
            final_answer = action_input
        elif tool == "search":
            observation = f"[Simulated search results for: {action_input}] - In a real implementation, this would return actual search results."
            task_complete = False
            final_answer = ""
        elif tool == "calculate":
            observation = f"[Simulated calculation for: {action_input}] - In a real implementation, this would perform the calculation."
            task_complete = False
            final_answer = ""
        elif tool == "lookup":
            observation = f"[Simulated lookup for: {action_input}] - In a real implementation, this would retrieve the information."
            task_complete = False
            final_answer = ""
        else:
            observation = f"Unknown tool: {tool}"
            task_complete = False
            final_answer = ""
    
    # Check if we should continue
    task_complete = (
        state.get("task_complete", False) or
        (actions_taken and actions_taken[-1].get("tool") == "final_answer") or
        current_iteration >= max_iterations
    )
    
    final_answer = ""
    if task_complete and actions_taken:
        final_answer = actions_taken[-1].get("input", "")
    
    report = f"""
    🔧 Action Executor:
    
    Execution Results:
    • Tool Used: {actions_taken[-1].get('tool', 'none') if actions_taken else 'none'}
    • Action Input: {actions_taken[-1].get('input', '')[:100] if actions_taken else ''}...
    • Observation: {observation[:150]}...
    • Task Complete: {task_complete}
    
    ReAct Tool Patterns:
    
    Tool Types:
    
    Information Gathering:
    • search(query) → results
    • lookup(entity) → facts
    • ask(question) → answer
    • retrieve(document) → content
    
    Computation:
    • calculate(expression) → value
    • execute(code) → output
    • simulate(scenario) → outcome
    • process(data) → result
    
    State Modification:
    • store(key, value) → success
    • update(item) → status
    • delete(item) → confirmation
    • modify(object) → new_state
    
    Communication:
    • send_message(recipient, msg) → response
    • request_input(prompt) → user_input
    • notify(event) → acknowledgment
    • collaborate(agent, task) → result
    
    Tool Selection Strategies:
    
    Rule-Based:
    • If-then rules
    • Pattern matching
    • Fixed sequences
    • Deterministic
    
    Learning-Based:
    • Model predicts tool
    • Context-aware selection
    • Adaptive strategy
    • Optimizes over time
    
    Hybrid:
    • Combine rules and learning
    • Fallback mechanisms
    • Best of both worlds
    • Robust and flexible
    
    Error Handling in ReAct:
    
    Tool Failures:
    • Retry with modified input
    • Try alternative tool
    • Ask for clarification
    • Graceful degradation
    
    Invalid Reasoning:
    • Self-correction
    • Re-evaluate assumptions
    • Seek additional info
    • Backtrack if needed
    
    Incomplete Information:
    • Identify gaps
    • Gather more data
    • Make reasonable assumptions
    • State uncertainties
    
    Max Iterations:
    • Provide best effort answer
    • Explain limitations
    • Suggest next steps
    • Partial solutions
    
    Advanced ReAct Techniques:
    
    Self-Ask ReAct:
    • Ask follow-up questions
    • Decompose complex queries
    • Iterative clarification
    • Deeper understanding
    
    Multi-Agent ReAct:
    • Multiple agents collaborate
    • Share observations
    • Parallel exploration
    • Faster convergence
    
    Reflexion ReAct:
    • Reflect on mistakes
    • Learn from errors
    • Improve over trials
    • Meta-learning
    
    Hierarchical ReAct:
    • High-level planning
    • Low-level execution
    • Abstraction layers
    • Scalable reasoning
    
    ReAct Best Practices:
    
    Prompt Engineering:
    • Clear thought/action format
    • List available tools
    • Provide examples
    • Specify constraints
    
    Tool Design:
    • Simple interfaces
    • Clear descriptions
    • Reliable execution
    • Error messages
    
    Iteration Management:
    • Set max iterations
    • Early stopping
    • Progress tracking
    • Timeout handling
    
    Evaluation:
    • Task success rate
    • Steps to solution
    • Tool usage efficiency
    • Error recovery
    
    Implementation Tips:
    
    Temperature:
    • Lower (0.1-0.3) for focused reasoning
    • Higher for creative exploration
    • Adjust per task type
    
    Prompt Format:
    ```
    Thought: [reasoning]
    Action: [what to do]
    Tool: [which tool]
    Action Input: [tool input]
    Observation: [result]
    ... (repeat)
    Thought: I now know the final answer
    Final Answer: [answer]
    ```
    
    Current Observation:
    {observation}
    
    Key Insight:
    ReAct creates a synergistic loop between reasoning and
    acting, enabling language models to interact with external
    tools and environments for grounded problem-solving.
    """
    
    return {
        "messages": [AIMessage(content=f"🔧 Action Executor:\n{report}")],
        "observations": observations + [observation],
        "task_complete": task_complete,
        "final_answer": final_answer
    }


# Build the graph
def build_react_graph():
    """Build the ReAct pattern graph"""
    workflow = StateGraph(ReActState)
    
    workflow.add_node("reasoning_agent", reasoning_agent)
    workflow.add_node("action_executor", action_executor)
    
    # Define conditional routing
    def should_continue(state: ReActState) -> str:
        """Determine if we should continue or end"""
        if state.get("task_complete", False):
            return "end"
        if state.get("current_iteration", 0) >= state.get("max_iterations", 5):
            return "end"
        return "continue"
    
    workflow.add_edge(START, "reasoning_agent")
    workflow.add_edge("reasoning_agent", "action_executor")
    
    # Conditional edge: continue loop or end
    workflow.add_conditional_edges(
        "action_executor",
        should_continue,
        {
            "continue": "reasoning_agent",
            "end": END
        }
    )
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_react_graph()
    
    print("=== ReAct MCP Pattern ===\n")
    
    # Test Case: Multi-step question answering
    print("\n" + "="*70)
    print("TEST CASE: Interactive Question Answering with ReAct")
    print("="*70)
    
    state = {
        "messages": [],
        "task": "What is the population of the capital city of France?",
        "reasoning_trace": [],
        "actions_taken": [],
        "observations": [],
        "max_iterations": 3,
        "current_iteration": 0,
        "task_complete": False,
        "final_answer": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 125: ReAct (Reasoning + Acting) - COMPLETE")
    print(f"{'='*70}")

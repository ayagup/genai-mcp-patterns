"""
Pattern 171: Agent Composition MCP Pattern

This pattern demonstrates how to compose multiple specialized agents into a single
cohesive system using the Composite design pattern. The composition allows treating
individual agents and composite agents uniformly through a common interface.

Key Concepts:
1. Component Interface: Common interface for all agents (leaf and composite)
2. Leaf Agents: Individual specialized agents (e.g., researcher, writer, critic)
3. Composite Agent: Container that holds and coordinates multiple agents
4. Transparent Composition: Clients interact with composite same as individual agents
5. Recursive Structure: Composites can contain other composites
6. Delegation: Composite delegates work to child agents and aggregates results

Agent Composition Patterns:
1. Sequential Composition: Chain agents in order (A -> B -> C)
2. Parallel Composition: Run agents concurrently, aggregate results
3. Hierarchical Composition: Tree structure with parent-child relationships
4. Dynamic Composition: Add/remove agents at runtime
5. Weighted Composition: Combine agent outputs with different weights

Benefits:
- Modularity: Each agent has single responsibility
- Reusability: Compose agents in different configurations
- Flexibility: Easy to add/remove agents
- Scalability: Distribute agents across processes/machines
- Testability: Test individual agents in isolation

Trade-offs:
- Complexity: More moving parts to coordinate
- Overhead: Communication between agents adds latency
- Debugging: Harder to trace through multiple agents
- Resource Usage: Multiple agents consume more resources

Use Cases:
- Content creation: researcher + writer + editor + critic
- Data processing: extractor + transformer + validator + loader
- Customer support: classifier + specialist responders + summarizer
- Code generation: planner + coder + reviewer + tester
"""

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator

# Define the state for agent composition
class CompositeState(TypedDict):
    """State shared across all composed agents"""
    task: str
    research_results: Annotated[List[str], operator.add]
    written_content: Annotated[List[str], operator.add]
    critiques: Annotated[List[str], operator.add]
    final_output: str
    composition_metadata: dict
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# LEAF AGENTS (Individual Specialized Agents)
# ============================================================================

def researcher_agent(state: CompositeState) -> CompositeState:
    """
    Leaf Agent: Research Agent
    Responsibility: Gather information and facts about the topic
    """
    task = state["task"]
    
    prompt = f"""You are a research specialist. Your task is to gather key facts 
    and information about the following topic:
    
    Topic: {task}
    
    Provide 3-5 key research findings or facts that would be useful for writing 
    content about this topic. Be concise and factual."""
    
    response = llm.invoke(prompt)
    research = response.content
    
    return {
        "research_results": [research],
        "messages": [f"[Researcher Agent] Completed research on: {task}"]
    }

def writer_agent(state: CompositeState) -> CompositeState:
    """
    Leaf Agent: Writer Agent
    Responsibility: Create content based on research findings
    """
    task = state["task"]
    research = "\n".join(state.get("research_results", []))
    
    prompt = f"""You are a content writer. Based on the research findings below,
    write a well-structured article or content piece.
    
    Topic: {task}
    
    Research Findings:
    {research}
    
    Write a coherent piece (3-4 paragraphs) that incorporates these findings."""
    
    response = llm.invoke(prompt)
    content = response.content
    
    return {
        "written_content": [content],
        "messages": [f"[Writer Agent] Created content based on research"]
    }

def critic_agent(state: CompositeState) -> CompositeState:
    """
    Leaf Agent: Critic Agent
    Responsibility: Review and critique the written content
    """
    content = "\n".join(state.get("written_content", []))
    
    prompt = f"""You are a content critic and editor. Review the following content
    and provide constructive feedback on:
    1. Clarity and coherence
    2. Factual accuracy
    3. Structure and flow
    4. Suggestions for improvement
    
    Content to review:
    {content}
    
    Provide your critique in a structured format."""
    
    response = llm.invoke(prompt)
    critique = response.content
    
    return {
        "critiques": [critique],
        "messages": [f"[Critic Agent] Provided critique and feedback"]
    }

def editor_agent(state: CompositeState) -> CompositeState:
    """
    Leaf Agent: Editor Agent
    Responsibility: Finalize content based on critiques
    """
    content = "\n".join(state.get("written_content", []))
    critique = "\n".join(state.get("critiques", []))
    
    prompt = f"""You are a final editor. Based on the original content and the 
    critique provided, create a polished final version.
    
    Original Content:
    {content}
    
    Critique:
    {critique}
    
    Produce the final, improved version incorporating the feedback."""
    
    response = llm.invoke(prompt)
    final = response.content
    
    return {
        "final_output": final,
        "messages": [f"[Editor Agent] Finalized content"]
    }

# ============================================================================
# COMPOSITE AGENT (Coordinator)
# ============================================================================

def composite_coordinator(state: CompositeState) -> CompositeState:
    """
    Composite Agent: Coordinates the composition of all child agents
    
    This demonstrates the Composite pattern where the coordinator:
    1. Manages the lifecycle of child agents
    2. Orchestrates execution flow
    3. Aggregates results from multiple agents
    4. Provides unified interface to clients
    """
    
    metadata = {
        "total_agents": 4,
        "agent_types": ["researcher", "writer", "critic", "editor"],
        "composition_type": "sequential",
        "coordinator_version": "1.0"
    }
    
    return {
        "composition_metadata": metadata,
        "messages": [f"[Composite Coordinator] Initialized composition with {metadata['total_agents']} agents"]
    }

# ============================================================================
# ALTERNATIVE COMPOSITION PATTERNS
# ============================================================================

def parallel_composite_example(state: CompositeState) -> CompositeState:
    """
    Example: Parallel Composition
    
    Multiple agents work on the same task simultaneously, and their results
    are aggregated (e.g., ensemble of agents voting or averaging)
    """
    # In a real implementation, you'd use parallel edges in the graph
    # This is a conceptual placeholder
    return {
        "messages": ["[Parallel Composite] Would execute agents concurrently"]
    }

def hierarchical_composite_example(state: CompositeState) -> CompositeState:
    """
    Example: Hierarchical Composition
    
    Tree structure where parent agents delegate to child agents:
    - Level 1: Master coordinator
    - Level 2: Sub-coordinators (research team, writing team, review team)
    - Level 3: Individual specialist agents
    """
    return {
        "messages": ["[Hierarchical Composite] Would organize agents in tree structure"]
    }

def weighted_composite_example(state: CompositeState) -> CompositeState:
    """
    Example: Weighted Composition
    
    Combine outputs from multiple agents with different weights:
    - Expert agent: weight 0.5
    - Generalist agent: weight 0.3
    - Fallback agent: weight 0.2
    
    Final output = weighted average of all agent outputs
    """
    return {
        "messages": ["[Weighted Composite] Would combine agent outputs with weights"]
    }

# ============================================================================
# DYNAMIC COMPOSITION
# ============================================================================

class DynamicComposite:
    """
    Dynamic Composition: Add/remove agents at runtime
    
    This allows for flexible composition where the agent structure can change
    based on runtime conditions (e.g., add expert agent only for complex queries)
    """
    
    def __init__(self):
        self.agents = []
    
    def add_agent(self, agent_name: str, agent_func):
        """Add an agent to the composition"""
        self.agents.append({
            "name": agent_name,
            "function": agent_func,
            "added_at": "runtime"
        })
        return f"Added agent: {agent_name}"
    
    def remove_agent(self, agent_name: str):
        """Remove an agent from the composition"""
        self.agents = [a for a in self.agents if a["name"] != agent_name]
        return f"Removed agent: {agent_name}"
    
    def execute(self, state: CompositeState) -> CompositeState:
        """Execute all agents in the composition"""
        for agent in self.agents:
            state = agent["function"](state)
        return state

# ============================================================================
# COMPOSITE DESIGN PATTERN IMPLEMENTATION
# ============================================================================

class AgentComponent:
    """
    Component: Abstract interface for both leaf agents and composites
    
    This is the classic Composite pattern interface that allows treating
    individual agents and composite agents uniformly.
    """
    
    def execute(self, state: CompositeState) -> CompositeState:
        """All agents (leaf and composite) must implement execute"""
        raise NotImplementedError("Subclasses must implement execute()")
    
    def add(self, component: 'AgentComponent'):
        """Add child component (only relevant for Composite)"""
        raise NotImplementedError("Not supported for leaf agents")
    
    def remove(self, component: 'AgentComponent'):
        """Remove child component (only relevant for Composite)"""
        raise NotImplementedError("Not supported for leaf agents")
    
    def get_children(self) -> List['AgentComponent']:
        """Get child components (only relevant for Composite)"""
        raise NotImplementedError("Not supported for leaf agents")

class LeafAgent(AgentComponent):
    """
    Leaf: Individual agent with no children
    """
    
    def __init__(self, name: str, agent_func):
        self.name = name
        self.agent_func = agent_func
    
    def execute(self, state: CompositeState) -> CompositeState:
        """Execute this individual agent"""
        return self.agent_func(state)
    
    def add(self, component: AgentComponent):
        """Leaf agents don't support adding children"""
        raise Exception(f"Cannot add to leaf agent {self.name}")
    
    def remove(self, component: AgentComponent):
        """Leaf agents don't support removing children"""
        raise Exception(f"Cannot remove from leaf agent {self.name}")
    
    def get_children(self) -> List[AgentComponent]:
        """Leaf agents have no children"""
        return []

class CompositeAgent(AgentComponent):
    """
    Composite: Container that can hold leaf agents and other composites
    """
    
    def __init__(self, name: str):
        self.name = name
        self.children: List[AgentComponent] = []
    
    def execute(self, state: CompositeState) -> CompositeState:
        """Execute all child agents in order"""
        for child in self.children:
            state = child.execute(state)
        return state
    
    def add(self, component: AgentComponent):
        """Add a child component"""
        self.children.append(component)
    
    def remove(self, component: AgentComponent):
        """Remove a child component"""
        self.children.remove(component)
    
    def get_children(self) -> List[AgentComponent]:
        """Get all child components"""
        return self.children

# ============================================================================
# BUILD THE COMPOSITION GRAPH
# ============================================================================

def create_agent_composition_graph():
    """
    Create a StateGraph that demonstrates agent composition.
    
    Flow:
    1. Coordinator initializes the composition
    2. Researcher gathers information
    3. Writer creates content based on research
    4. Critic reviews the content
    5. Editor produces final version
    """
    
    workflow = StateGraph(CompositeState)
    
    # Add all agents to the graph
    workflow.add_node("coordinator", composite_coordinator)
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("critic", critic_agent)
    workflow.add_node("editor", editor_agent)
    
    # Define the composition flow (sequential in this example)
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", "editor")
    workflow.add_edge("editor", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Agent Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Create the composition graph
    composition_graph = create_agent_composition_graph()
    
    # Example 1: Compose agents for content creation
    print("\n" + "=" * 80)
    print("Example 1: Sequential Agent Composition (Content Creation Pipeline)")
    print("=" * 80)
    
    initial_state: CompositeState = {
        "task": "The impact of artificial intelligence on software development",
        "research_results": [],
        "written_content": [],
        "critiques": [],
        "final_output": "",
        "composition_metadata": {},
        "messages": []
    }
    
    result = composition_graph.invoke(initial_state)
    
    print("\nComposition Metadata:")
    print(result["composition_metadata"])
    
    print("\nExecution Flow:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nFinal Output:")
    print(result["final_output"][:500] + "..." if len(result["final_output"]) > 500 else result["final_output"])
    
    # Example 2: Using Composite Pattern Classes
    print("\n" + "=" * 80)
    print("Example 2: Composite Pattern Class-Based Composition")
    print("=" * 80)
    
    # Create leaf agents
    researcher_leaf = LeafAgent("researcher", researcher_agent)
    writer_leaf = LeafAgent("writer", writer_agent)
    critic_leaf = LeafAgent("critic", critic_agent)
    editor_leaf = LeafAgent("editor", editor_agent)
    
    # Create composite for content creation team
    content_team = CompositeAgent("content_creation_team")
    content_team.add(researcher_leaf)
    content_team.add(writer_leaf)
    
    # Create composite for review team
    review_team = CompositeAgent("review_team")
    review_team.add(critic_leaf)
    review_team.add(editor_leaf)
    
    # Create top-level composite that contains both teams
    full_pipeline = CompositeAgent("full_content_pipeline")
    full_pipeline.add(content_team)
    full_pipeline.add(review_team)
    
    print(f"Created composite with {len(full_pipeline.get_children())} top-level components:")
    for child in full_pipeline.get_children():
        if isinstance(child, CompositeAgent):
            print(f"  - {child.name} (composite with {len(child.get_children())} children)")
        else:
            print(f"  - {child.name} (leaf)")
    
    # Execute the full pipeline
    initial_state_2: CompositeState = {
        "task": "Benefits of modular software architecture",
        "research_results": [],
        "written_content": [],
        "critiques": [],
        "final_output": "",
        "composition_metadata": {},
        "messages": []
    }
    
    # Note: In a real implementation, you'd integrate this with the StateGraph
    # This demonstrates the pattern structure
    
    # Example 3: Dynamic Composition
    print("\n" + "=" * 80)
    print("Example 3: Dynamic Composition (Runtime Agent Addition)")
    print("=" * 80)
    
    dynamic_comp = DynamicComposite()
    print(dynamic_comp.add_agent("researcher", researcher_agent))
    print(dynamic_comp.add_agent("writer", writer_agent))
    print(f"Total agents: {len(dynamic_comp.agents)}")
    
    # Add a specialist agent at runtime based on complexity
    task_complexity = "high"
    if task_complexity == "high":
        print(dynamic_comp.add_agent("critic", critic_agent))
        print(dynamic_comp.add_agent("editor", editor_agent))
    
    print(f"Total agents after dynamic addition: {len(dynamic_comp.agents)}")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Agent Composition combines multiple specialized agents into cohesive systems
2. Composite Pattern allows uniform treatment of individual and composite agents
3. Sequential composition chains agents in order (research -> write -> critique -> edit)
4. Parallel composition runs agents concurrently and aggregates results
5. Hierarchical composition organizes agents in tree structures
6. Dynamic composition allows adding/removing agents at runtime
7. Weighted composition combines agent outputs with different importance
8. Benefits: modularity, reusability, flexibility, scalability
9. Trade-offs: complexity, overhead, debugging difficulty
10. Use cases: content creation, data processing, customer support, code generation
    """)

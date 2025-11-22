"""
Pattern 175: Nested Composition MCP Pattern

This pattern demonstrates nested (recursive) composition where components can contain
other components of the same type, creating tree-like structures. Parent components
delegate work to child components, which may themselves contain further nested children.

Key Concepts:
1. Recursive Structure: Components contain components of same type
2. Parent-Child Relationships: Clear hierarchical organization
3. Tree Topology: Root with branches and leaves
4. Delegation: Parents delegate to children
5. Aggregation: Parents aggregate child results
6. Uniform Interface: All components expose same interface
7. Depth: Can nest arbitrarily deep

Nested Composition Patterns:
1. Simple Tree: Single root, multiple children
2. Multi-Level Hierarchy: Department → Team → Individual
3. Composite Pattern: Treat leaf and composite uniformly
4. Fractal Structure: Same pattern repeats at each level
5. Recursive Decomposition: Break complex into simpler nested parts

Parent-Child Dynamics:
- Parent: Coordinator, aggregator, higher-level abstraction
- Child: Executor, specialist, lower-level implementation
- Leaf: Terminal node with no children (actual work happens here)
- Composite: Non-leaf node that delegates to children

Benefits:
- Natural Hierarchy: Models organizational structures
- Recursive Simplicity: Same pattern at all levels
- Flexible Depth: Can nest as deep as needed
- Clear Ownership: Parent owns children
- Encapsulation: Children hidden from outside

Trade-offs:
- Deep Nesting: Can become hard to understand
- Traversal Overhead: Need to walk tree for operations
- Memory: Tree structure has overhead
- Circular References: Must avoid parent ↔ child cycles
- Complexity: More complex than flat structures

Use Cases:
- Organization chart: Company → Division → Department → Team
- File system: Directory → Subdirectory → Files
- UI components: Page → Section → Widget → Elements
- Task breakdown: Project → Phase → Task → Subtask
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from abc import ABC, abstractmethod

# Define the state for nested composition
class NestedCompositionState(TypedDict):
    """State for nested component hierarchy"""
    task: str
    root_result: Optional[Dict[str, Any]]
    hierarchy_trace: Annotated[List[str], operator.add]
    depth_levels: Annotated[List[int], operator.add]
    leaf_results: Annotated[List[Dict[str, Any]], operator.add]
    final_output: str
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# COMPOSITE PATTERN (Classic Implementation)
# ============================================================================

class Component(ABC):
    """
    Component: Abstract base for both Leaf and Composite
    
    This is the classic Composite pattern where:
    - Leaf and Composite both implement Component interface
    - Clients can treat both uniformly
    - Enables recursive composition
    """
    
    def __init__(self, name: str):
        self.name = name
        self.parent: Optional['Component'] = None
    
    @abstractmethod
    def execute(self, task: str) -> Dict[str, Any]:
        """Execute the component's responsibility"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get component description"""
        pass
    
    def get_path(self) -> str:
        """Get full path from root to this component"""
        if self.parent:
            return f"{self.parent.get_path()} → {self.name}"
        return self.name

class Leaf(Component):
    """
    Leaf: Terminal node that actually performs work
    
    Characteristics:
    - Has no children
    - Performs actual work
    - Cannot delegate further
    """
    
    def __init__(self, name: str, capability: str):
        super().__init__(name)
        self.capability = capability
    
    def execute(self, task: str) -> Dict[str, Any]:
        """Leaf nodes do the actual work"""
        return {
            "component": self.name,
            "type": "leaf",
            "capability": self.capability,
            "task": task,
            "result": f"{self.name} processed: {task[:50]}",
            "path": self.get_path()
        }
    
    def get_description(self) -> str:
        return f"Leaf: {self.name} ({self.capability})"

class Composite(Component):
    """
    Composite: Container that holds children and delegates to them
    
    Characteristics:
    - Has children (leaves or other composites)
    - Delegates work to children
    - Aggregates child results
    - Can be nested arbitrarily deep
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self.children: List[Component] = []
    
    def add(self, component: Component):
        """Add a child component"""
        component.parent = self
        self.children.append(component)
    
    def remove(self, component: Component):
        """Remove a child component"""
        component.parent = None
        self.children.remove(component)
    
    def execute(self, task: str) -> Dict[str, Any]:
        """Composite delegates to children and aggregates results"""
        child_results = []
        
        for child in self.children:
            result = child.execute(task)
            child_results.append(result)
        
        return {
            "component": self.name,
            "type": "composite",
            "child_count": len(self.children),
            "task": task,
            "child_results": child_results,
            "aggregated": f"{self.name} coordinated {len(child_results)} children",
            "path": self.get_path()
        }
    
    def get_description(self) -> str:
        return f"Composite: {self.name} ({len(self.children)} children)"
    
    def print_tree(self, indent: int = 0) -> str:
        """Print tree structure"""
        lines = ["  " * indent + f"+ {self.name} (Composite)"]
        for child in self.children:
            if isinstance(child, Composite):
                lines.append(child.print_tree(indent + 1))
            else:
                lines.append("  " * (indent + 1) + f"- {child.name} (Leaf)")
        return "\n".join(lines)

# ============================================================================
# NESTED AGENTS (Multi-Level Hierarchy)
# ============================================================================

def root_coordinator(state: NestedCompositionState) -> NestedCompositionState:
    """
    Root Level: Top-level coordinator
    
    In nested composition, the root:
    - Receives the initial task
    - Breaks it down for child components
    - Coordinates overall execution
    - Aggregates all results
    """
    task = state["task"]
    
    prompt = f"""You are a root coordinator in a nested hierarchy.
    Break down this task for delegation to child coordinators:
    
    Task: {task}
    
    Identify:
    1. Major sub-tasks
    2. Which children should handle each
    3. Coordination strategy"""
    
    response = llm.invoke(prompt)
    
    return {
        "hierarchy_trace": ["Level 0: Root Coordinator"],
        "depth_levels": [0],
        "messages": [f"[Level 0 - Root] Coordinating: {task[:50]}"]
    }

def child_coordinator_1(state: NestedCompositionState) -> NestedCompositionState:
    """
    Level 1: Child Coordinator (can have its own children)
    
    This is both a child (of root) and a parent (of leaves or other composites)
    """
    task = state["task"]
    
    prompt = f"""You are a child coordinator. Process this delegated task
    and further delegate to your children if needed:
    
    Task: {task}
    
    Coordinate your sub-team."""
    
    response = llm.invoke(prompt)
    
    return {
        "hierarchy_trace": ["Level 1: Child Coordinator 1 (under Root)"],
        "depth_levels": [1],
        "messages": [f"[Level 1 - Child Coordinator 1] Processing sub-task"]
    }

def child_coordinator_2(state: NestedCompositionState) -> NestedCompositionState:
    """
    Level 1: Another Child Coordinator
    
    Sibling to child_coordinator_1, both under root
    """
    return {
        "hierarchy_trace": ["Level 1: Child Coordinator 2 (under Root)"],
        "depth_levels": [1],
        "messages": [f"[Level 1 - Child Coordinator 2] Processing sub-task"]
    }

def leaf_worker_1(state: NestedCompositionState) -> NestedCompositionState:
    """
    Level 2: Leaf Worker (terminal node)
    
    This is the deepest level in this example - actual work happens here
    """
    task = state["task"]
    
    prompt = f"""You are a leaf worker. Do the actual work:
    
    Task: {task}
    
    Complete this specific task component."""
    
    response = llm.invoke(prompt)
    
    leaf_result = {
        "worker": "leaf_worker_1",
        "level": 2,
        "result": response.content[:100],
        "completed": True
    }
    
    return {
        "hierarchy_trace": ["Level 2: Leaf Worker 1 (under Child Coordinator 1)"],
        "depth_levels": [2],
        "leaf_results": [leaf_result],
        "messages": [f"[Level 2 - Leaf Worker 1] Completed actual work"]
    }

def leaf_worker_2(state: NestedCompositionState) -> NestedCompositionState:
    """Level 2: Another Leaf Worker"""
    
    leaf_result = {
        "worker": "leaf_worker_2",
        "level": 2,
        "result": "Work completed",
        "completed": True
    }
    
    return {
        "hierarchy_trace": ["Level 2: Leaf Worker 2 (under Child Coordinator 1)"],
        "depth_levels": [2],
        "leaf_results": [leaf_result],
        "messages": [f"[Level 2 - Leaf Worker 2] Completed actual work"]
    }

def leaf_worker_3(state: NestedCompositionState) -> NestedCompositionState:
    """Level 2: Leaf Worker under different parent"""
    
    leaf_result = {
        "worker": "leaf_worker_3",
        "level": 2,
        "result": "Work completed",
        "completed": True
    }
    
    return {
        "hierarchy_trace": ["Level 2: Leaf Worker 3 (under Child Coordinator 2)"],
        "depth_levels": [2],
        "leaf_results": [leaf_result],
        "messages": [f"[Level 2 - Leaf Worker 3] Completed actual work"]
    }

# ============================================================================
# RESULT AGGREGATION (Bottom-Up)
# ============================================================================

def aggregate_level_2(state: NestedCompositionState) -> NestedCompositionState:
    """Aggregate results from level 2 (leaves) to level 1"""
    leaf_results = state.get("leaf_results", [])
    
    return {
        "messages": [f"[Aggregation] Collected {len(leaf_results)} leaf results"]
    }

def aggregate_level_1(state: NestedCompositionState) -> NestedCompositionState:
    """Aggregate results from level 1 to level 0 (root)"""
    return {
        "messages": [f"[Aggregation] Aggregated child coordinator results"]
    }

def final_aggregation(state: NestedCompositionState) -> NestedCompositionState:
    """Final aggregation at root level"""
    hierarchy = state.get("hierarchy_trace", [])
    leaf_results = state.get("leaf_results", [])
    
    final_output = f"""
    Nested Composition Results:
    
    Hierarchy Depth: {max(state.get('depth_levels', [0]))} levels
    Total Components: {len(hierarchy)}
    Leaf Workers: {len(leaf_results)}
    
    Hierarchy Trace:
    {chr(10).join(f"  - {trace}" for trace in hierarchy)}
    
    All nested components completed successfully.
    """
    
    return {
        "final_output": final_output.strip(),
        "messages": ["[Final Aggregation] All nested levels aggregated"]
    }

# ============================================================================
# RECURSIVE TREE TRAVERSAL
# ============================================================================

class TreeTraverser:
    """
    Utility for traversing nested component trees
    
    Traversal Strategies:
    - Depth-First: Go deep before wide
    - Breadth-First: Go wide before deep
    - Pre-Order: Parent before children
    - Post-Order: Children before parent
    """
    
    @staticmethod
    def depth_first(component: Component, visit_fn):
        """Depth-first traversal (pre-order)"""
        # Visit parent first
        visit_fn(component)
        
        # Then visit children
        if isinstance(component, Composite):
            for child in component.children:
                TreeTraverser.depth_first(child, visit_fn)
    
    @staticmethod
    def breadth_first(component: Component, visit_fn):
        """Breadth-first traversal (level-by-level)"""
        queue = [component]
        
        while queue:
            current = queue.pop(0)
            visit_fn(current)
            
            if isinstance(current, Composite):
                queue.extend(current.children)
    
    @staticmethod
    def post_order(component: Component, visit_fn):
        """Post-order: children before parent (for aggregation)"""
        if isinstance(component, Composite):
            for child in component.children:
                TreeTraverser.post_order(child, visit_fn)
        
        # Visit parent after children
        visit_fn(component)

# ============================================================================
# BUILD THE NESTED COMPOSITION GRAPH
# ============================================================================

def create_nested_composition_graph():
    """
    Create a StateGraph demonstrating nested composition.
    
    Hierarchy:
    Level 0: Root Coordinator
      Level 1: Child Coordinator 1
        Level 2: Leaf Worker 1
        Level 2: Leaf Worker 2
      Level 1: Child Coordinator 2
        Level 2: Leaf Worker 3
    
    Flow:
    1. Root delegates to child coordinators
    2. Child coordinators delegate to leaf workers
    3. Leaf workers do actual work
    4. Results aggregate back up the tree
    """
    
    workflow = StateGraph(NestedCompositionState)
    
    # Add all levels
    workflow.add_node("root", root_coordinator)
    workflow.add_node("child_1", child_coordinator_1)
    workflow.add_node("child_2", child_coordinator_2)
    workflow.add_node("leaf_1", leaf_worker_1)
    workflow.add_node("leaf_2", leaf_worker_2)
    workflow.add_node("leaf_3", leaf_worker_3)
    workflow.add_node("aggregator", final_aggregation)
    
    # Build tree structure (parent → children)
    # Root → Children (Level 0 → Level 1)
    workflow.add_edge(START, "root")
    workflow.add_edge("root", "child_1")
    workflow.add_edge("root", "child_2")
    
    # Child 1 → Leaves (Level 1 → Level 2)
    workflow.add_edge("child_1", "leaf_1")
    workflow.add_edge("child_1", "leaf_2")
    
    # Child 2 → Leaf (Level 1 → Level 2)
    workflow.add_edge("child_2", "leaf_3")
    
    # All leaves → Aggregator (Level 2 → Aggregation)
    workflow.add_edge("leaf_1", "aggregator")
    workflow.add_edge("leaf_2", "aggregator")
    workflow.add_edge("leaf_3", "aggregator")
    
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Nested Composition MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Three-Level Nested Hierarchy
    print("\n" + "=" * 80)
    print("Example 1: Three-Level Nested Hierarchy")
    print("=" * 80)
    
    nested_graph = create_nested_composition_graph()
    
    initial_state: NestedCompositionState = {
        "task": "Develop new AI feature with research, implementation, and testing",
        "root_result": None,
        "hierarchy_trace": [],
        "depth_levels": [],
        "leaf_results": [],
        "final_output": "",
        "messages": []
    }
    
    result = nested_graph.invoke(initial_state)
    
    print("\nHierarchy Execution:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nFinal Output:")
    print(result["final_output"])
    
    # Example 2: Composite Pattern Implementation
    print("\n" + "=" * 80)
    print("Example 2: Composite Pattern (Recursive Structure)")
    print("=" * 80)
    
    # Build a nested component tree
    # Root (Composite)
    #   ├─ Development (Composite)
    #   │    ├─ Frontend Developer (Leaf)
    #   │    └─ Backend Developer (Leaf)
    #   ├─ QA (Composite)
    #   │    ├─ Manual Tester (Leaf)
    #   │    └─ Automation Tester (Leaf)
    #   └─ DevOps (Leaf)
    
    root = Composite("Engineering Team")
    
    # Development sub-tree
    development = Composite("Development")
    development.add(Leaf("Frontend Developer", "UI implementation"))
    development.add(Leaf("Backend Developer", "API implementation"))
    root.add(development)
    
    # QA sub-tree
    qa = Composite("QA")
    qa.add(Leaf("Manual Tester", "manual testing"))
    qa.add(Leaf("Automation Tester", "test automation"))
    root.add(qa)
    
    # DevOps (leaf at root level)
    root.add(Leaf("DevOps Engineer", "deployment"))
    
    print("\nComponent Tree Structure:")
    print(root.print_tree())
    
    print("\nExecuting nested task:")
    result = root.execute("Build and deploy new feature")
    
    print(f"\nRoot Result:")
    print(f"  Component: {result['component']}")
    print(f"  Type: {result['type']}")
    print(f"  Children: {result['child_count']}")
    print(f"  Aggregated: {result['aggregated']}")
    
    # Example 3: Tree Traversal
    print("\n" + "=" * 80)
    print("Example 3: Tree Traversal Strategies")
    print("=" * 80)
    
    print("\nDepth-First Traversal (Pre-Order):")
    visited_df = []
    TreeTraverser.depth_first(root, lambda c: visited_df.append(c.name))
    for i, name in enumerate(visited_df):
        print(f"  {i+1}. {name}")
    
    print("\nBreadth-First Traversal (Level-by-Level):")
    visited_bf = []
    TreeTraverser.breadth_first(root, lambda c: visited_bf.append(c.name))
    for i, name in enumerate(visited_bf):
        print(f"  {i+1}. {name}")
    
    print("\nPost-Order Traversal (For Aggregation):")
    visited_po = []
    TreeTraverser.post_order(root, lambda c: visited_po.append(c.name))
    for i, name in enumerate(visited_po):
        print(f"  {i+1}. {name}")
    
    # Example 4: Nested Depth Analysis
    print("\n" + "=" * 80)
    print("Example 4: Analyzing Nested Structure")
    print("=" * 80)
    
    def count_components(component: Component) -> Dict[str, int]:
        """Count components at each level"""
        counts = {"total": 1, "composites": 0, "leaves": 0}
        
        if isinstance(component, Composite):
            counts["composites"] = 1
            for child in component.children:
                child_counts = count_components(child)
                counts["total"] += child_counts["total"]
                counts["composites"] += child_counts["composites"]
                counts["leaves"] += child_counts["leaves"]
        else:
            counts["leaves"] = 1
        
        return counts
    
    stats = count_components(root)
    print(f"\nComponent Statistics:")
    print(f"  Total components: {stats['total']}")
    print(f"  Composite (containers): {stats['composites']}")
    print(f"  Leaves (workers): {stats['leaves']}")
    
    def max_depth(component: Component, current_depth: int = 0) -> int:
        """Calculate maximum depth of tree"""
        if isinstance(component, Composite):
            if not component.children:
                return current_depth
            return max(max_depth(child, current_depth + 1) 
                      for child in component.children)
        return current_depth
    
    depth = max_depth(root)
    print(f"  Maximum depth: {depth} levels")
    
    print("\n" + "=" * 80)
    print("Key Takeaways:")
    print("=" * 80)
    print("""
1. Nested Composition creates recursive tree structures
2. Components contain components of same type (recursive)
3. Composite Pattern: uniform interface for leaf and composite
4. Parent-Child relationships form hierarchy
5. Root: top-level coordinator
6. Composite: container with children (can be nested)
7. Leaf: terminal node, does actual work
8. Delegation: parents delegate to children
9. Aggregation: parents aggregate child results (bottom-up)
10. Traversal: depth-first, breadth-first, pre/post-order
11. Benefits: natural hierarchy, recursive simplicity, flexible depth
12. Trade-offs: deep nesting complexity, traversal overhead
13. Use cases: org charts, file systems, task breakdown, UI components
14. Fractal structure: same pattern repeats at each level
    """)

"""
Synchronous-Asynchronous Hybrid MCP Pattern

This pattern combines synchronous (blocking, sequential) and asynchronous
(non-blocking, concurrent) execution for optimal performance.

Pattern Type: Hybrid
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
import asyncio


# State definition
class SyncAsyncState(TypedDict):
    """State for synchronous-asynchronous hybrid"""
    tasks: List[str]
    sync_results: List[Dict[str, Any]]
    async_results: List[Dict[str, Any]]
    hybrid_execution: Dict[str, Any]
    performance_metrics: Dict[str, float]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class SynchronousExecutor:
    """Execute tasks synchronously"""
    
    def execute_sync(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Synchronous execution (blocking, sequential)"""
        results = []
        
        for i, task in enumerate(tasks):
            # Simulate synchronous processing
            result = {
                "task_id": i,
                "task": task,
                "execution_mode": "synchronous",
                "result": f"Sync result for: {task}",
                "blocking": True,
                "order_preserved": True
            }
            results.append(result)
        
        return results


class AsynchronousExecutor:
    """Execute tasks asynchronously"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def execute_async(self, tasks: List[str]) -> List[Dict[str, Any]]:
        """Asynchronous execution (non-blocking, concurrent)"""
        results = []
        
        # Simulate async execution (in production, use actual async)
        for i, task in enumerate(tasks):
            result = {
                "task_id": i,
                "task": task,
                "execution_mode": "asynchronous",
                "result": f"Async result for: {task}",
                "blocking": False,
                "concurrent": True
            }
            results.append(result)
        
        return results


class HybridScheduler:
    """Schedule tasks using hybrid sync/async approach"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def schedule(self, tasks: List[str]) -> Dict[str, Any]:
        """Determine which tasks should be sync vs async"""
        prompt = f"""Classify tasks as synchronous or asynchronous:

Tasks:
{json.dumps(tasks, indent=2)}

Classify each task:
- Synchronous: Tasks that must be sequential, have dependencies, or require blocking
- Asynchronous: Independent tasks that can run concurrently

Return JSON:
{{
    "sync_tasks": ["task indices that need sync execution"],
    "async_tasks": ["task indices that can be async"],
    "reasoning": "why this classification"
}}"""
        
        messages = [
            SystemMessage(content="You are a task scheduling expert."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "sync_tasks": [],
            "async_tasks": list(range(len(tasks))),
            "reasoning": "Default to async"
        }


class PerformanceAnalyzer:
    """Analyze performance of hybrid execution"""
    
    def analyze(self, sync_results: List[Dict], async_results: List[Dict]) -> Dict[str, float]:
        """Analyze execution performance"""
        return {
            "sync_tasks": len(sync_results),
            "async_tasks": len(async_results),
            "total_tasks": len(sync_results) + len(async_results),
            "sync_ratio": len(sync_results) / (len(sync_results) + len(async_results)) if sync_results or async_results else 0,
            "async_ratio": len(async_results) / (len(sync_results) + len(async_results)) if sync_results or async_results else 0,
            "estimated_speedup": 1.5 if async_results else 1.0  # Simplified
        }


# Agent functions
def initialize_hybrid(state: SyncAsyncState) -> SyncAsyncState:
    """Initialize hybrid execution"""
    state["messages"].append(HumanMessage(
        content=f"Initializing sync-async hybrid for {len(state['tasks'])} tasks"
    ))
    state["current_step"] = "initialized"
    return state


def execute_synchronous(state: SyncAsyncState) -> SyncAsyncState:
    """Execute synchronous tasks"""
    executor = SynchronousExecutor()
    
    # For demo, execute first 2 tasks synchronously
    sync_tasks = state["tasks"][:2]
    state["sync_results"] = executor.execute_sync(sync_tasks)
    
    state["messages"].append(HumanMessage(
        content=f"Executed {len(state['sync_results'])} tasks synchronously"
    ))
    state["current_step"] = "sync_executed"
    return state


def execute_asynchronous(state: SyncAsyncState) -> SyncAsyncState:
    """Execute asynchronous tasks"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    executor = AsynchronousExecutor(llm)
    
    # Execute remaining tasks asynchronously
    async_tasks = state["tasks"][2:]
    state["async_results"] = executor.execute_async(async_tasks)
    
    state["messages"].append(HumanMessage(
        content=f"Executed {len(state['async_results'])} tasks asynchronously"
    ))
    state["current_step"] = "async_executed"
    return state


def analyze_performance(state: SyncAsyncState) -> SyncAsyncState:
    """Analyze hybrid performance"""
    analyzer = PerformanceAnalyzer()
    
    metrics = analyzer.analyze(state["sync_results"], state["async_results"])
    state["performance_metrics"] = metrics
    
    state["messages"].append(HumanMessage(
        content=f"Performance: {metrics['estimated_speedup']:.1f}x speedup with "
                f"{metrics['async_ratio']:.0%} async execution"
    ))
    state["current_step"] = "analyzed"
    return state


def generate_report(state: SyncAsyncState) -> SyncAsyncState:
    """Generate final report"""
    report = f"""
SYNCHRONOUS-ASYNCHRONOUS HYBRID REPORT
======================================

Total Tasks: {len(state['tasks'])}

SYNCHRONOUS EXECUTION ({len(state['sync_results'])} tasks):
"""
    
    for result in state['sync_results']:
        report += f"\n- Task {result['task_id']}: {result['task']}\n"
        report += f"  Mode: {result['execution_mode']}\n"
        report += f"  Blocking: {result['blocking']}\n"
    
    report += f"""
ASYNCHRONOUS EXECUTION ({len(state['async_results'])} tasks):
"""
    
    for result in state['async_results']:
        report += f"\n- Task {result['task_id']}: {result['task']}\n"
        report += f"  Mode: {result['execution_mode']}\n"
        report += f"  Concurrent: {result['concurrent']}\n"
    
    report += f"""
PERFORMANCE METRICS:
--------------------
Sync Tasks: {state['performance_metrics']['sync_tasks']}
Async Tasks: {state['performance_metrics']['async_tasks']}
Async Ratio: {state['performance_metrics']['async_ratio']:.0%}
Estimated Speedup: {state['performance_metrics']['estimated_speedup']:.1f}x

Summary:
Hybrid sync-async execution balances order preservation with concurrency
for optimal performance.
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_sync_async_graph():
    """Create sync-async hybrid workflow"""
    workflow = StateGraph(SyncAsyncState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_hybrid)
    workflow.add_node("sync_exec", execute_synchronous)
    workflow.add_node("async_exec", execute_asynchronous)
    workflow.add_node("analyze", analyze_performance)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "sync_exec")
    workflow.add_edge("sync_exec", "async_exec")
    workflow.add_edge("async_exec", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "tasks": [
            "Initialize database connection",  # Sync - must be first
            "Load configuration",  # Sync - dependency
            "Fetch user data",  # Async - independent
            "Process analytics",  # Async - independent
            "Generate reports"  # Async - independent
        ],
        "sync_results": [],
        "async_results": [],
        "hybrid_execution": {},
        "performance_metrics": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_sync_async_graph()
    
    print("Synchronous-Asynchronous Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")
    
    print(f"\nSpeedup: {result['performance_metrics'].get('estimated_speedup', 0):.1f}x")

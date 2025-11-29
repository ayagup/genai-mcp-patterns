"""
Pattern 243: Context Switching MCP Pattern

This pattern demonstrates context switching - dynamically switching between
different contexts based on task requirements.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextSwitchingState(TypedDict):
    """State for context switching workflow"""
    messages: Annotated[List[str], add]
    current_context: str
    context_history: List[Dict[str, Any]]
    tasks: List[str]


class ContextSwitcher:
    """Manages context switching"""
    
    def __init__(self):
        self.contexts = {
            "coding": {"language": "python", "mode": "development", "tools": ["IDE", "debugger"]},
            "testing": {"framework": "pytest", "mode": "qa", "tools": ["test_runner", "coverage"]},
            "documentation": {"format": "markdown", "mode": "writing", "tools": ["editor", "linter"]},
            "deployment": {"platform": "kubernetes", "mode": "ops", "tools": ["kubectl", "helm"]}
        }
        self.current_context = None
    
    def switch_to(self, context_name: str) -> Dict[str, Any]:
        """Switch to specified context"""
        if context_name in self.contexts:
            self.current_context = context_name
            return self.contexts[context_name]
        return {}


def initialize_tasks_agent(state: ContextSwitchingState) -> ContextSwitchingState:
    """Initialize tasks requiring different contexts"""
    print("\n📋 Initializing Tasks...")
    
    tasks = [
        "Write user authentication function",
        "Test login endpoint",
        "Document API usage",
        "Deploy to production"
    ]
    
    print(f"\n  Tasks to Complete: {len(tasks)}")
    for i, task in enumerate(tasks, 1):
        print(f"    {i}. {task}")
    
    return {
        **state,
        "tasks": tasks,
        "current_context": "",
        "messages": [f"✓ Initialized {len(tasks)} tasks"]
    }


def switch_contexts_agent(state: ContextSwitchingState) -> ContextSwitchingState:
    """Switch contexts for each task"""
    print("\n🔄 Switching Contexts for Tasks...")
    
    switcher = ContextSwitcher()
    context_mapping = {
        "Write user authentication function": "coding",
        "Test login endpoint": "testing",
        "Document API usage": "documentation",
        "Deploy to production": "deployment"
    }
    
    context_history = []
    
    for task in state["tasks"]:
        context_name = context_mapping.get(task, "coding")
        context_data = switcher.switch_to(context_name)
        
        context_history.append({
            "task": task,
            "context": context_name,
            "config": context_data
        })
        
        print(f"\n  Task: {task}")
        print(f"  → Switched to: {context_name}")
        print(f"    Configuration:")
        for key, value in context_data.items():
            print(f"      {key}: {value}")
    
    return {
        **state,
        "current_context": switcher.current_context,
        "context_history": context_history,
        "messages": [f"✓ Switched contexts {len(context_history)} times"]
    }


def generate_context_switching_report_agent(state: ContextSwitchingState) -> ContextSwitchingState:
    """Generate context switching report"""
    print("\n" + "="*70)
    print("CONTEXT SWITCHING REPORT")
    print("="*70)
    
    print(f"\n📊 Context Switches: {len(state['context_history'])}")
    
    print(f"\n🔄 Context History:")
    for i, entry in enumerate(state['context_history'], 1):
        print(f"\n  {i}. {entry['task']}")
        print(f"     Context: {entry['context']}")
        print(f"     Tools: {', '.join(entry['config'].get('tools', []))}")
    
    # Count context usage
    context_counts = {}
    for entry in state['context_history']:
        ctx = entry['context']
        context_counts[ctx] = context_counts.get(ctx, 0) + 1
    
    print(f"\n📈 Context Usage Statistics:")
    for context, count in context_counts.items():
        print(f"  {context}: {count} time(s)")
    
    print("\n💡 Context Switching Benefits:")
    print("  • Task-specific optimization")
    print("  • Efficient resource allocation")
    print("  • Specialized tool usage")
    print("  • Clear separation of concerns")
    print("  • Better focus and productivity")
    
    print("\n="*70)
    print("✅ Context Switching Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_switching_graph():
    workflow = StateGraph(ContextSwitchingState)
    workflow.add_node("initialize", initialize_tasks_agent)
    workflow.add_node("switch", switch_contexts_agent)
    workflow.add_node("report", generate_context_switching_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "switch")
    workflow.add_edge("switch", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 243: Context Switching MCP Pattern")
    print("="*70)
    
    app = create_context_switching_graph()
    final_state = app.invoke({
        "messages": [],
        "current_context": "",
        "context_history": [],
        "tasks": []
    })
    print("\n✅ Context Switching Pattern Complete!")


if __name__ == "__main__":
    main()

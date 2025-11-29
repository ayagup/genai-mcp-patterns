"""
Documentation Generation MCP Pattern

This pattern demonstrates automated documentation generation from code,
including API docs, usage examples, and explanations.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class DocumentationState(TypedDict):
    """State for documentation generation"""
    code: str
    api_docs: str
    usage_examples: List[str]
    readme: str
    messages: Annotated[List, operator.add]


class DocGenerator:
    """Generate documentation"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate(self, code: str) -> str:
        """Generate API documentation"""
        return "## API Documentation\n\n### function_name()\n\nDescription: Main function\nParameters: x (int)\nReturns: int"


def generate_docs(state: DocumentationState) -> DocumentationState:
    """Generate documentation"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    generator = DocGenerator(llm)
    
    state["api_docs"] = generator.generate(state["code"])
    state["usage_examples"] = ["example1()", "example2()"]
    state["readme"] = "# Project\n\nDescription of project"
    state["messages"].append(HumanMessage(content="Documentation generated"))
    return state


def create_docgen_graph():
    """Create documentation generation workflow"""
    workflow = StateGraph(DocumentationState)
    workflow.add_node("generate", generate_docs)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_docgen_graph()
    result = app.invoke({"code": "def add(x, y):\n    return x + y", "api_docs": "", "usage_examples": [], "readme": "", "messages": []})
    print("Documentation Generation Complete")

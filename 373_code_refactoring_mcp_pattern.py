"""
Code Refactoring MCP Pattern

This pattern demonstrates automated code refactoring to improve structure,
readability, and maintainability while preserving functionality.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator


class CodeRefactoringState(TypedDict):
    """State for code refactoring"""
    original_code: str
    language: str
    refactoring_goals: List[str]
    code_smells: List[Dict]
    refactored_code: str
    improvements: List[str]
    messages: Annotated[List, operator.add]


class CodeAnalyzer:
    """Analyze code for refactoring opportunities"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def detect_smells(self, code: str) -> List[Dict]:
        """Detect code smells"""
        prompt = f"""Identify code smells in this code:

{code}

Return JSON array of issues like long methods, duplicated code, etc."""
        
        messages = [SystemMessage(content="You are a code quality expert."), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        return [{"smell": "Long method", "severity": "medium"}]  # Simplified


class Refactorer:
    """Refactor code"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def refactor(self, code: str, smells: List[Dict]) -> str:
        """Refactor code based on smells"""
        prompt = f"""Refactor this code to address these issues:

Code: {code}
Issues: {smells}

Apply refactorings like extract method, rename variables, etc."""
        
        messages = [SystemMessage(content="You are a refactoring expert."), HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        return response.content


def analyze_code(state: CodeRefactoringState) -> CodeRefactoringState:
    """Analyze code"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    analyzer = CodeAnalyzer(llm)
    
    state["code_smells"] = analyzer.detect_smells(state["original_code"])
    state["messages"].append(HumanMessage(content=f"Found {len(state['code_smells'])} code smells"))
    return state


def refactor_code(state: CodeRefactoringState) -> CodeRefactoringState:
    """Refactor code"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    refactorer = Refactorer(llm)
    
    state["refactored_code"] = refactorer.refactor(state["original_code"], state["code_smells"])
    state["improvements"] = ["Extracted methods", "Improved naming", "Reduced complexity"]
    state["messages"].append(HumanMessage(content="Code refactored"))
    return state


def create_refactoring_graph():
    """Create refactoring workflow"""
    workflow = StateGraph(CodeRefactoringState)
    
    workflow.add_node("analyze", analyze_code)
    workflow.add_node("refactor", refactor_code)
    
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "refactor")
    workflow.add_edge("refactor", END)
    
    return workflow.compile()


if __name__ == "__main__":
    app = create_refactoring_graph()
    result = app.invoke({
        "original_code": "def process(data):\n    # Long complex method\n    pass",
        "language": "Python",
        "refactoring_goals": ["Improve readability"],
        "code_smells": [],
        "refactored_code": "",
        "improvements": [],
        "messages": []
    })
    print("Code Refactoring MCP Pattern Complete")

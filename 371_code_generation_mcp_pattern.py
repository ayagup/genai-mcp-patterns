"""
Code Generation MCP Pattern

This pattern demonstrates automated code generation from specifications,
requirements, or natural language descriptions.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class CodeGenerationState(TypedDict):
    """State for code generation"""
    specification: str
    language: str
    requirements: List[str]
    code_structure: Dict[str, Any]
    generated_code: str
    documentation: str
    test_cases: List[str]
    quality_metrics: Dict[str, float]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class SpecificationAnalyzer:
    """Analyze code specifications"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, spec: str, language: str) -> Dict[str, Any]:
        """Analyze specification"""
        prompt = f"""Analyze this code specification:

Specification: {spec}
Target Language: {language}

Extract:
1. Main functionality
2. Input/output requirements
3. Data structures needed
4. Key algorithms
5. Edge cases to handle

Return JSON:
{{
    "functionality": "main purpose",
    "inputs": ["input parameters"],
    "outputs": ["output type/format"],
    "data_structures": ["structures needed"],
    "algorithms": ["algorithms to use"],
    "edge_cases": ["edge cases"]
}}"""
        
        messages = [
            SystemMessage(content="You are a software specification analyst."),
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
            "functionality": "Unable to analyze",
            "inputs": [],
            "outputs": [],
            "data_structures": [],
            "algorithms": [],
            "edge_cases": []
        }


class CodeGenerator:
    """Generate code from structure"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate(self, structure: Dict[str, Any], language: str) -> str:
        """Generate code"""
        prompt = f"""Generate {language} code based on this structure:

Structure:
{json.dumps(structure, indent=2)}

Requirements:
1. Follow best practices
2. Include error handling
3. Add inline comments
4. Make it readable and maintainable

Generate complete, production-ready code."""
        
        messages = [
            SystemMessage(content=f"You are an expert {language} programmer."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        return response.content


class DocumentationGenerator:
    """Generate code documentation"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_docs(self, code: str, language: str) -> str:
        """Generate documentation"""
        prompt = f"""Generate documentation for this {language} code:

Code:
{code[:500]}...

Include:
1. Function/class description
2. Parameters
3. Return values
4. Usage examples
5. Notes/warnings"""
        
        messages = [
            SystemMessage(content="You are a technical documentation writer."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        return response.content


# Agent functions
def initialize(state: CodeGenerationState) -> CodeGenerationState:
    """Initialize"""
    state["messages"].append(HumanMessage(
        content=f"Initializing code generation for {state['language']}"
    ))
    state["current_step"] = "initialized"
    return state


def analyze_spec(state: CodeGenerationState) -> CodeGenerationState:
    """Analyze specification"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    analyzer = SpecificationAnalyzer(llm)
    
    structure = analyzer.analyze(state["specification"], state["language"])
    state["code_structure"] = structure
    
    state["messages"].append(HumanMessage(
        content=f"Specification analyzed: {structure.get('functionality', 'N/A')[:50]}..."
    ))
    state["current_step"] = "spec_analyzed"
    return state


def generate_code(state: CodeGenerationState) -> CodeGenerationState:
    """Generate code"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    generator = CodeGenerator(llm)
    
    code = generator.generate(state["code_structure"], state["language"])
    state["generated_code"] = code
    
    state["messages"].append(HumanMessage(
        content=f"Code generated: {len(code)} characters"
    ))
    state["current_step"] = "code_generated"
    return state


def generate_documentation(state: CodeGenerationState) -> CodeGenerationState:
    """Generate documentation"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    doc_gen = DocumentationGenerator(llm)
    
    docs = doc_gen.generate_docs(state["generated_code"], state["language"])
    state["documentation"] = docs
    
    state["messages"].append(HumanMessage(
        content="Documentation generated"
    ))
    state["current_step"] = "docs_generated"
    return state


def generate_report(state: CodeGenerationState) -> CodeGenerationState:
    """Generate report"""
    report = f"""
CODE GENERATION REPORT
======================

Specification: {state['specification']}
Language: {state['language']}

CODE STRUCTURE:
{json.dumps(state['code_structure'], indent=2)}

GENERATED CODE:
---------------
{state['generated_code'][:500]}...

DOCUMENTATION:
--------------
{state['documentation'][:300]}...

Summary:
Code successfully generated with documentation.
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build graph
def create_codegen_graph():
    """Create code generation workflow"""
    workflow = StateGraph(CodeGenerationState)
    
    workflow.add_node("initialize", initialize)
    workflow.add_node("analyze", analyze_spec)
    workflow.add_node("generate", generate_code)
    workflow.add_node("document", generate_documentation)
    workflow.add_node("report", generate_report)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "document")
    workflow.add_edge("document", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "specification": "Create a function that calculates factorial of a number recursively",
        "language": "Python",
        "requirements": ["Handle edge cases", "Include error handling"],
        "code_structure": {},
        "generated_code": "",
        "documentation": "",
        "test_cases": [],
        "quality_metrics": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_codegen_graph()
    
    print("Code Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")

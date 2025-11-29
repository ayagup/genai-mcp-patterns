"""
Multi-Modal Hybrid MCP Pattern

This pattern combines multiple modalities (text, code, data, etc.)
for comprehensive analysis and generation.

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


# State definition
class MultiModalState(TypedDict):
    """State for multi-modal hybrid"""
    task: str
    modalities: List[str]
    text_analysis: Dict[str, Any]
    code_analysis: Dict[str, Any]
    data_analysis: Dict[str, Any]
    cross_modal_insights: List[Dict[str, Any]]
    integrated_output: Dict[str, Any]
    final_response: str
    messages: Annotated[List, operator.add]
    current_step: str


class TextAnalyzer:
    """Analyze text modality"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, task: str) -> Dict[str, Any]:
        """Analyze text content"""
        prompt = f"""Analyze text aspects of this task:

Task: {task}

Provide text-based analysis:
1. Natural language understanding
2. Semantic meaning
3. Intent and context
4. Key concepts

Return JSON:
{{
    "modality": "text",
    "understanding": "what the text means",
    "key_concepts": ["main concepts"],
    "intent": "user intent",
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are a text analysis expert."),
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
            "modality": "text",
            "understanding": "Unable to analyze",
            "key_concepts": [],
            "intent": "unknown",
            "confidence": 0.0
        }


class CodeAnalyzer:
    """Analyze code modality"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, task: str) -> Dict[str, Any]:
        """Analyze code aspects"""
        prompt = f"""Analyze code/technical aspects of this task:

Task: {task}

Provide code-based analysis:
1. Technical requirements
2. Implementation approach
3. Code patterns needed
4. Technical constraints

Return JSON:
{{
    "modality": "code",
    "technical_requirements": ["requirements"],
    "implementation_approach": "how to implement",
    "patterns": ["relevant code patterns"],
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are a code analysis expert."),
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
            "modality": "code",
            "technical_requirements": [],
            "implementation_approach": "unknown",
            "patterns": [],
            "confidence": 0.0
        }


class DataAnalyzer:
    """Analyze data modality"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze(self, task: str) -> Dict[str, Any]:
        """Analyze data aspects"""
        prompt = f"""Analyze data/structural aspects of this task:

Task: {task}

Provide data-based analysis:
1. Data structures needed
2. Data flow
3. Data transformations
4. Data constraints

Return JSON:
{{
    "modality": "data",
    "structures": ["data structures"],
    "flow": "how data flows",
    "transformations": ["needed transformations"],
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are a data analysis expert."),
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
            "modality": "data",
            "structures": [],
            "flow": "unknown",
            "transformations": [],
            "confidence": 0.0
        }


class CrossModalIntegrator:
    """Integrate insights across modalities"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def integrate(self, text_analysis: Dict, code_analysis: Dict,
                 data_analysis: Dict) -> Dict[str, Any]:
        """Integrate multi-modal insights"""
        prompt = f"""Integrate insights from multiple modalities:

Text Analysis:
{json.dumps(text_analysis, indent=2)}

Code Analysis:
{json.dumps(code_analysis, indent=2)}

Data Analysis:
{json.dumps(data_analysis, indent=2)}

Synthesize:
1. How modalities complement each other
2. Cross-modal patterns
3. Integrated solution
4. Multi-modal confidence

Return JSON:
{{
    "integrated_solution": "complete solution",
    "text_contribution": "how text helped",
    "code_contribution": "how code helped",
    "data_contribution": "how data helped",
    "cross_modal_insights": ["insights from combining modalities"],
    "confidence": 0.0-1.0
}}"""
        
        messages = [
            SystemMessage(content="You are a multi-modal integration expert."),
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
            "integrated_solution": "Unable to integrate",
            "text_contribution": "N/A",
            "code_contribution": "N/A",
            "data_contribution": "N/A",
            "cross_modal_insights": [],
            "confidence": 0.0
        }


# Agent functions
def initialize(state: MultiModalState) -> MultiModalState:
    """Initialize"""
    state["messages"].append(HumanMessage(
        content=f"Initializing multi-modal analysis for: {state['task']}"
    ))
    state["current_step"] = "initialized"
    return state


def analyze_text(state: MultiModalState) -> MultiModalState:
    """Analyze text modality"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = TextAnalyzer(llm)
    
    state["text_analysis"] = analyzer.analyze(state["task"])
    
    state["messages"].append(HumanMessage(
        content=f"Text analysis: {len(state['text_analysis'].get('key_concepts', []))} concepts identified"
    ))
    state["current_step"] = "text_analyzed"
    return state


def analyze_code(state: MultiModalState) -> MultiModalState:
    """Analyze code modality"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = CodeAnalyzer(llm)
    
    state["code_analysis"] = analyzer.analyze(state["task"])
    
    state["messages"].append(HumanMessage(
        content=f"Code analysis: {len(state['code_analysis'].get('patterns', []))} patterns identified"
    ))
    state["current_step"] = "code_analyzed"
    return state


def analyze_data(state: MultiModalState) -> MultiModalState:
    """Analyze data modality"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = DataAnalyzer(llm)
    
    state["data_analysis"] = analyzer.analyze(state["task"])
    
    state["messages"].append(HumanMessage(
        content=f"Data analysis: {len(state['data_analysis'].get('structures', []))} structures identified"
    ))
    state["current_step"] = "data_analyzed"
    return state


def integrate_modalities(state: MultiModalState) -> MultiModalState:
    """Integrate all modalities"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    integrator = CrossModalIntegrator(llm)
    
    integrated = integrator.integrate(
        state["text_analysis"],
        state["code_analysis"],
        state["data_analysis"]
    )
    
    state["integrated_output"] = integrated
    state["cross_modal_insights"] = integrated.get("cross_modal_insights", [])
    
    state["messages"].append(HumanMessage(
        content=f"Integrated {len(state['modalities'])} modalities, "
                f"confidence: {integrated.get('confidence', 0):.2f}"
    ))
    state["current_step"] = "integrated"
    return state


def generate_report(state: MultiModalState) -> MultiModalState:
    """Generate report"""
    report = f"""
MULTI-MODAL HYBRID REPORT
=========================

Task: {state['task']}
Modalities: {', '.join(state['modalities'])}

TEXT MODALITY:
--------------
Understanding: {state['text_analysis'].get('understanding', 'N/A')}
Intent: {state['text_analysis'].get('intent', 'N/A')}
Key Concepts: {', '.join(state['text_analysis'].get('key_concepts', []))}
Confidence: {state['text_analysis'].get('confidence', 0):.2%}

CODE MODALITY:
--------------
Implementation: {state['code_analysis'].get('implementation_approach', 'N/A')}
Patterns: {', '.join(state['code_analysis'].get('patterns', []))}
Confidence: {state['code_analysis'].get('confidence', 0):.2%}

DATA MODALITY:
--------------
Data Flow: {state['data_analysis'].get('flow', 'N/A')}
Structures: {', '.join(state['data_analysis'].get('structures', []))}
Confidence: {state['data_analysis'].get('confidence', 0):.2%}

INTEGRATED SOLUTION:
--------------------
{state['integrated_output'].get('integrated_solution', 'N/A')}

Cross-Modal Insights:
{chr(10).join(f'- {insight}' for insight in state['cross_modal_insights'])}

Contributions:
- Text: {state['integrated_output'].get('text_contribution', 'N/A')}
- Code: {state['integrated_output'].get('code_contribution', 'N/A')}
- Data: {state['integrated_output'].get('data_contribution', 'N/A')}

Overall Confidence: {state['integrated_output'].get('confidence', 0):.2%}

Summary:
Multi-modal approach provides comprehensive understanding by combining
text, code, and data perspectives.
"""
    
    state["final_response"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build graph
def create_multimodal_graph():
    """Create multi-modal workflow"""
    workflow = StateGraph(MultiModalState)
    
    workflow.add_node("initialize", initialize)
    workflow.add_node("text", analyze_text)
    workflow.add_node("code", analyze_code)
    workflow.add_node("data", analyze_data)
    workflow.add_node("integrate", integrate_modalities)
    workflow.add_node("report", generate_report)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "text")
    workflow.add_edge("text", "code")
    workflow.add_edge("code", "data")
    workflow.add_edge("data", "integrate")
    workflow.add_edge("integrate", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    initial_state = {
        "task": "Build a REST API for user management with database persistence",
        "modalities": ["text", "code", "data"],
        "text_analysis": {},
        "code_analysis": {},
        "data_analysis": {},
        "cross_modal_insights": [],
        "integrated_output": {},
        "final_response": "",
        "messages": [],
        "current_step": "pending"
    }
    
    app = create_multimodal_graph()
    
    print("Multi-Modal Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")
    
    print(f"\nIntegrated Confidence: {result['integrated_output'].get('confidence', 0):.2%}")

"""
Retrieval-Generation Hybrid MCP Pattern

This pattern demonstrates combining retrieval (RAG) with generation
for grounded, factual content creation.

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
class RetrievalGenerationState(TypedDict):
    """State for retrieval-generation hybrid"""
    query: str
    knowledge_base: List[Dict[str, str]]
    retrieved_docs: List[Dict[str, Any]]
    relevance_scores: List[float]
    generated_content: str
    grounding_citations: List[str]
    factuality_score: float
    hybrid_output: Dict[str, Any]
    final_response: str
    messages: Annotated[List, operator.add]
    current_step: str


class DocumentRetriever:
    """Retrieve relevant documents"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def retrieve(self, query: str, knowledge_base: List[Dict[str, str]], 
                top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve most relevant documents"""
        prompt = f"""Retrieve most relevant documents for this query:

Query: {query}

Knowledge Base:
{json.dumps(knowledge_base, indent=2)}

Rank documents by relevance and return top {top_k}.

Return JSON array:
[
    {{
        "doc_id": "id",
        "content": "document content",
        "relevance_score": 0.0-1.0,
        "relevance_reason": "why relevant"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are a document retrieval expert."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class GroundedGenerator:
    """Generate content grounded in retrieved documents"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate grounded content"""
        prompt = f"""Generate an answer grounded in the retrieved documents:

Query: {query}

Retrieved Documents:
{json.dumps(retrieved_docs, indent=2)}

Requirements:
1. Answer must be grounded in provided documents
2. Cite sources for claims
3. Don't add information not in documents
4. If insufficient info, say so

Return JSON:
{{
    "answer": "the generated answer",
    "citations": ["doc_id references used"],
    "confidence": 0.0-1.0,
    "coverage": "how well documents cover query"
}}"""
        
        messages = [
            SystemMessage(content="You are a grounded content generator."),
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
            "answer": "Unable to generate",
            "citations": [],
            "confidence": 0.0,
            "coverage": "insufficient"
        }


class FactualityVerifier:
    """Verify factuality of generated content"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def verify(self, generated: str, source_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify factuality against sources"""
        prompt = f"""Verify factuality of generated content:

Generated Content:
{generated}

Source Documents:
{json.dumps(source_docs, indent=2)}

Check:
1. All claims supported by sources
2. No hallucinated information
3. Accurate representation of sources

Return JSON:
{{
    "factuality_score": 0.0-1.0,
    "supported_claims": ["claims backed by sources"],
    "unsupported_claims": ["claims not in sources"],
    "verdict": "factual|partially_factual|hallucinated"
}}"""
        
        messages = [
            SystemMessage(content="You are a factuality verification system."),
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
            "factuality_score": 0.0,
            "supported_claims": [],
            "unsupported_claims": [],
            "verdict": "unknown"
        }


# Agent functions
def initialize_system(state: RetrievalGenerationState) -> RetrievalGenerationState:
    """Initialize retrieval-generation system"""
    state["messages"].append(HumanMessage(
        content=f"Initializing retrieval-generation hybrid for: {state['query']}"
    ))
    state["current_step"] = "initialized"
    return state


def retrieve_documents(state: RetrievalGenerationState) -> RetrievalGenerationState:
    """Retrieve relevant documents"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    retriever = DocumentRetriever(llm)
    
    docs = retriever.retrieve(state["query"], state["knowledge_base"], top_k=3)
    
    state["retrieved_docs"] = docs
    state["relevance_scores"] = [d.get("relevance_score", 0.0) for d in docs]
    
    state["messages"].append(HumanMessage(
        content=f"Retrieved {len(docs)} documents, avg relevance: {sum(state['relevance_scores'])/len(state['relevance_scores']):.2f}"
    ))
    state["current_step"] = "retrieved"
    return state


def generate_content(state: RetrievalGenerationState) -> RetrievalGenerationState:
    """Generate grounded content"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    generator = GroundedGenerator(llm)
    
    result = generator.generate(state["query"], state["retrieved_docs"])
    
    state["generated_content"] = result.get("answer", "")
    state["grounding_citations"] = result.get("citations", [])
    
    state["messages"].append(HumanMessage(
        content=f"Generated content with {len(state['grounding_citations'])} citations"
    ))
    state["current_step"] = "generated"
    return state


def verify_factuality(state: RetrievalGenerationState) -> RetrievalGenerationState:
    """Verify factuality"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    verifier = FactualityVerifier(llm)
    
    verification = verifier.verify(state["generated_content"], state["retrieved_docs"])
    
    state["factuality_score"] = verification.get("factuality_score", 0.0)
    
    state["messages"].append(HumanMessage(
        content=f"Factuality verified: {verification.get('verdict', 'unknown')}, "
                f"score: {state['factuality_score']:.2f}"
    ))
    state["current_step"] = "verified"
    return state


def generate_report(state: RetrievalGenerationState) -> RetrievalGenerationState:
    """Generate final report"""
    report = f"""
RETRIEVAL-GENERATION HYBRID REPORT
===================================

Query: {state['query']}

RETRIEVAL PHASE:
----------------
Knowledge Base Size: {len(state['knowledge_base'])} documents
Retrieved: {len(state['retrieved_docs'])} documents

Retrieved Documents:
"""
    
    for doc in state['retrieved_docs']:
        report += f"\n- Doc {doc.get('doc_id', '?')}: {doc.get('content', 'N/A')[:100]}...\n"
        report += f"  Relevance: {doc.get('relevance_score', 0):.2f}\n"
        report += f"  Reason: {doc.get('relevance_reason', 'N/A')}\n"
    
    report += f"""
GENERATION PHASE:
-----------------
Generated Content:
{state['generated_content']}

Citations: {', '.join(state['grounding_citations'])}

VERIFICATION:
-------------
Factuality Score: {state['factuality_score']:.2%}

HYBRID OUTPUT:
--------------
This response combines:
- Retrieval: {len(state['retrieved_docs'])} relevant documents
- Generation: Synthesized natural language answer
- Grounding: {len(state['grounding_citations'])} source citations
- Factuality: {state['factuality_score']:.2%} verified

Summary:
The retrieval-generation hybrid ensures factual accuracy while
maintaining natural, comprehensive responses.
"""
    
    state["final_response"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_retrieval_generation_graph():
    """Create retrieval-generation workflow"""
    workflow = StateGraph(RetrievalGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_system)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_content)
    workflow.add_node("verify", verify_factuality)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_edge("verify", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample knowledge base
    knowledge_base = [
        {"doc_id": "D1", "content": "Python was created by Guido van Rossum and released in 1991."},
        {"doc_id": "D2", "content": "Python is a high-level, interpreted programming language."},
        {"doc_id": "D3", "content": "Python supports multiple programming paradigms including object-oriented and functional."},
        {"doc_id": "D4", "content": "JavaScript was created by Brendan Eich in 1995."},
        {"doc_id": "D5", "content": "Python's design philosophy emphasizes code readability."}
    ]
    
    initial_state = {
        "query": "When was Python created and by whom?",
        "knowledge_base": knowledge_base,
        "retrieved_docs": [],
        "relevance_scores": [],
        "generated_content": "",
        "grounding_citations": [],
        "factuality_score": 0.0,
        "hybrid_output": {},
        "final_response": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run
    app = create_retrieval_generation_graph()
    
    print("Retrieval-Generation Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")
    
    print(f"\nFactuality: {result['factuality_score']:.2%}")

"""
Abstractive Summarization MCP Pattern

This pattern demonstrates abstractive summarization in an agentic MCP system.
The system generates new text that captures the essence of source documents,
similar to how humans write summaries.

Use cases:
- News article summaries
- Report generation
- Content briefings
- Executive summaries
- Email digest creation
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re


# Define the state for abstractive summarization
class AbstractiveSummarizationState(TypedDict):
    """State for tracking abstractive summarization process"""
    messages: Annotated[List[str], add]
    source_document: Dict[str, Any]
    key_concepts: List[str]
    main_points: List[Dict[str, Any]]
    summary_outline: Dict[str, Any]
    generated_summary: str
    refined_summary: str
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class ConceptExtractor:
    """Extract key concepts from text"""
    
    def extract_concepts(self, text: str) -> List[str]:
        """Extract main concepts"""
        # Simple extraction based on capitalized phrases and important nouns
        # In production, would use NER and topic modeling
        
        concepts = []
        
        # Extract capitalized phrases (potential named entities)
        cap_phrases = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*', text)
        concepts.extend(cap_phrases[:10])
        
        # Extract keywords (simple frequency-based)
        words = re.findall(r'\b\w{5,}\b', text.lower())
        from collections import Counter
        common_words = Counter(words).most_common(10)
        concepts.extend([word for word, _ in common_words])
        
        return list(set(concepts))[:15]


class MainPointIdentifier:
    """Identify main points in document"""
    
    def identify_points(self, text: str, concepts: List[str]) -> List[Dict[str, Any]]:
        """Identify main points"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        points = []
        for i, sentence in enumerate(sentences):
            # Calculate importance based on concept presence
            concept_count = sum(1 for c in concepts if c.lower() in sentence.lower())
            
            if concept_count > 0 or i == 0:  # First sentence or contains concepts
                points.append({
                    'text': sentence.strip(),
                    'concept_count': concept_count,
                    'position': i,
                    'importance': concept_count * 0.5 + (1 if i == 0 else 0)
                })
        
        # Sort by importance
        points.sort(key=lambda x: x['importance'], reverse=True)
        return points[:5]


class SummaryGenerator:
    """Generate abstractive summary using LLM"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def create_outline(self, concepts: List[str], main_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary outline"""
        return {
            'key_themes': concepts[:5],
            'main_arguments': [p['text'] for p in main_points[:3]],
            'target_length': 'concise',  # concise, medium, detailed
            'style': 'informative'  # informative, narrative, technical
        }
    
    def generate_summary(self, document: Dict[str, Any], 
                        outline: Dict[str, Any]) -> str:
        """Generate abstractive summary"""
        
        system_prompt = """You are an expert summarizer. Create a concise, abstractive summary 
        that captures the essence of the content. Use your own words while preserving key information.
        The summary should be coherent, well-structured, and professional."""
        
        user_prompt = f"""
        Summarize the following document:
        
        Title: {document.get('title', 'Untitled')}
        
        Content:
        {document['content'][:2000]}  # Limit for context
        
        Key themes to include: {', '.join(outline['key_themes'])}
        
        Create a {outline['target_length']} summary in a {outline['style']} style.
        Summary should be 3-4 sentences.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class SummaryRefiner:
    """Refine and improve generated summary"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    
    def refine_for_clarity(self, summary: str) -> str:
        """Refine summary for clarity and coherence"""
        
        system_prompt = """You are a professional editor. Refine the given summary to improve 
        clarity, coherence, and readability while maintaining accuracy and conciseness."""
        
        user_prompt = f"""
        Refine this summary:
        
        {summary}
        
        Improve:
        - Clarity and flow
        - Sentence structure
        - Word choice
        - Coherence
        
        Keep the same length and key information.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def check_factual_consistency(self, summary: str, original: str) -> float:
        """Check if summary is factually consistent with original"""
        # Simplified version - in production would use entailment models
        
        # Check if key terms from summary appear in original
        summary_words = set(re.findall(r'\b\w{5,}\b', summary.lower()))
        original_words = set(re.findall(r'\b\w{5,}\b', original.lower()))
        
        if not summary_words:
            return 0.0
        
        overlap = len(summary_words & original_words) / len(summary_words)
        return overlap


class QualityAssessor:
    """Assess quality of abstractive summary"""
    
    def calculate_compression(self, original: str, summary: str) -> float:
        """Calculate compression ratio"""
        orig_words = len(original.split())
        summ_words = len(summary.split())
        return summ_words / orig_words if orig_words > 0 else 0
    
    def assess_readability(self, text: str) -> float:
        """Simple readability assessment"""
        sentences = len(re.split(r'[.!?]', text))
        words = len(text.split())
        
        if sentences == 0:
            return 0
        
        avg_sent_length = words / sentences
        
        # Optimal sentence length is 15-20 words
        if 15 <= avg_sent_length <= 20:
            return 1.0
        elif 10 <= avg_sent_length < 15 or 20 < avg_sent_length <= 25:
            return 0.8
        else:
            return 0.6
    
    def assess_informativeness(self, summary: str, concepts: List[str]) -> float:
        """Assess how informative the summary is"""
        summary_lower = summary.lower()
        concepts_present = sum(1 for c in concepts[:10] if c.lower() in summary_lower)
        return concepts_present / min(len(concepts), 10) if concepts else 0


# Agent functions
def load_document_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Load source document"""
    
    document = {
        'title': 'Climate Change Impact on Global Agriculture',
        'content': """
        Climate change is having profound effects on agricultural systems worldwide. Rising temperatures, 
        changing precipitation patterns, and extreme weather events are disrupting traditional farming practices 
        across all continents. Agricultural productivity in many regions is declining due to heat stress, 
        water scarcity, and increased pest pressure. Smallholder farmers in developing countries are 
        particularly vulnerable to these changes. Crop yields for staple foods like wheat, rice, and maize 
        are projected to decrease in many major producing regions. Livestock farming faces challenges from 
        heat stress and reduced pasture quality. However, some northern regions may see improved growing 
        conditions. Adaptation strategies including drought-resistant crops, improved irrigation systems, 
        and diversified farming practices are being implemented. Agricultural research is focusing on 
        developing climate-resilient varieties and sustainable farming methods. International cooperation 
        and policy interventions are crucial for supporting farmers in adapting to climate change. 
        The agricultural sector must transform to ensure food security for a growing global population 
        while reducing its own contribution to greenhouse gas emissions.
        """.strip(),
        'metadata': {
            'author': 'Global Agriculture Report',
            'date': '2024-11-20',
            'category': 'Environment'
        }
    }
    
    return {
        **state,
        'source_document': document,
        'messages': state['messages'] + [f'Loaded document: "{document["title"]}"']
    }


def extract_key_concepts_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Extract key concepts"""
    
    extractor = ConceptExtractor()
    concepts = extractor.extract_concepts(state['source_document']['content'])
    
    return {
        **state,
        'key_concepts': concepts,
        'messages': state['messages'] + [f'Extracted {len(concepts)} key concepts']
    }


def identify_main_points_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Identify main points"""
    
    identifier = MainPointIdentifier()
    points = identifier.identify_points(
        state['source_document']['content'],
        state['key_concepts']
    )
    
    return {
        **state,
        'main_points': points,
        'messages': state['messages'] + [f'Identified {len(points)} main points']
    }


def create_summary_outline_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Create summary outline"""
    
    generator = SummaryGenerator()
    outline = generator.create_outline(state['key_concepts'], state['main_points'])
    
    return {
        **state,
        'summary_outline': outline,
        'messages': state['messages'] + ['Created summary outline']
    }


def generate_summary_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Generate abstractive summary"""
    
    generator = SummaryGenerator()
    summary = generator.generate_summary(
        state['source_document'],
        state['summary_outline']
    )
    
    return {
        **state,
        'generated_summary': summary,
        'messages': state['messages'] + [f'Generated abstractive summary ({len(summary.split())} words)']
    }


def refine_summary_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Refine generated summary"""
    
    refiner = SummaryRefiner()
    refined = refiner.refine_for_clarity(state['generated_summary'])
    
    # Check factual consistency
    consistency = refiner.check_factual_consistency(
        refined,
        state['source_document']['content']
    )
    
    return {
        **state,
        'refined_summary': refined,
        'messages': state['messages'] + [f'Refined summary (consistency: {consistency:.2f})']
    }


def assess_quality_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Assess summary quality"""
    
    assessor = QualityAssessor()
    refiner = SummaryRefiner()
    
    metrics = {
        'compression_ratio': assessor.calculate_compression(
            state['source_document']['content'],
            state['refined_summary']
        ),
        'readability': assessor.assess_readability(state['refined_summary']),
        'informativeness': assessor.assess_informativeness(
            state['refined_summary'],
            state['key_concepts']
        ),
        'factual_consistency': refiner.check_factual_consistency(
            state['refined_summary'],
            state['source_document']['content']
        ),
        'original_words': len(state['source_document']['content'].split()),
        'summary_words': len(state['refined_summary'].split())
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Assessed quality (readability: {metrics["readability"]:.2f})']
    }


def analyze_abstractive_summarization_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Analyze summarization results"""
    
    analytics = {
        'extraction_stats': {
            'concepts_extracted': len(state['key_concepts']),
            'main_points_identified': len(state['main_points']),
            'outline_themes': len(state['summary_outline'].get('key_themes', []))
        },
        'generation_stats': {
            'initial_summary_words': len(state['generated_summary'].split()),
            'refined_summary_words': len(state['refined_summary'].split()),
            'refinement_delta': len(state['refined_summary'].split()) - len(state['generated_summary'].split())
        },
        'quality_stats': state['quality_metrics']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed abstractive summarization results']
    }


def generate_abstractive_summarization_report_agent(state: AbstractiveSummarizationState) -> AbstractiveSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "ABSTRACTIVE SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "SOURCE DOCUMENT:",
        "-" * 40,
        f"Title: {state['source_document']['title']}",
        f"Length: {state['analytics']['quality_stats']['original_words']} words",
        "",
        "KEY CONCEPTS EXTRACTED:",
        "-" * 40,
        ", ".join(state['key_concepts'][:10]),
        "",
        "MAIN POINTS IDENTIFIED:",
        "-" * 40
    ]
    
    for point in state['main_points'][:3]:
        report_lines.append(f"• {point['text'][:100]}...")
    
    report_lines.extend([
        "",
        "",
        "GENERATED SUMMARY:",
        "-" * 40,
        state['refined_summary'],
        "",
        "",
        "SUMMARY STATISTICS:",
        "-" * 40,
        f"Original: {state['analytics']['quality_stats']['original_words']} words",
        f"Summary: {state['analytics']['quality_stats']['summary_words']} words",
        f"Compression: {state['analytics']['quality_stats']['compression_ratio']:.1%}",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Readability Score: {state['analytics']['quality_stats']['readability']:.2f}/1.0",
        f"Informativeness: {state['analytics']['quality_stats']['informativeness']:.2f}/1.0",
        f"Factual Consistency: {state['analytics']['quality_stats']['factual_consistency']:.2f}/1.0",
        "",
        "GENERATION PROCESS:",
        "-" * 40,
        f"Concepts Extracted: {state['analytics']['extraction_stats']['concepts_extracted']}",
        f"Main Points: {state['analytics']['extraction_stats']['main_points_identified']}",
        f"Initial Generation: {state['analytics']['generation_stats']['initial_summary_words']} words",
        f"After Refinement: {state['analytics']['generation_stats']['refined_summary_words']} words",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Created abstractive summary with {state['analytics']['quality_stats']['compression_ratio']:.0%} compression",
        f"✓ Achieved {state['analytics']['quality_stats']['factual_consistency']:.0%} factual consistency",
        f"✓ Readability score of {state['analytics']['quality_stats']['readability']:.2f}",
        "✓ Summary uses paraphrasing and synthesis, not extraction",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Fine-tune summary length based on audience needs",
        "• Consider multi-step refinement for critical summaries",
        "• Implement human-in-the-loop for verification",
        "• Add domain-specific vocabulary enhancement",
        "• Use retrieval-augmented generation for accuracy",
        "• Implement controlled generation for style consistency",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive report']
    }


# Create the graph
def create_abstractive_summarization_graph():
    """Create the abstractive summarization workflow graph"""
    
    workflow = StateGraph(AbstractiveSummarizationState)
    
    # Add nodes
    workflow.add_node("load_document", load_document_agent)
    workflow.add_node("extract_concepts", extract_key_concepts_agent)
    workflow.add_node("identify_points", identify_main_points_agent)
    workflow.add_node("create_outline", create_summary_outline_agent)
    workflow.add_node("generate_summary", generate_summary_agent)
    workflow.add_node("refine_summary", refine_summary_agent)
    workflow.add_node("assess_quality", assess_quality_agent)
    workflow.add_node("analyze_results", analyze_abstractive_summarization_agent)
    workflow.add_node("generate_report", generate_abstractive_summarization_report_agent)
    
    # Add edges
    workflow.add_edge(START, "load_document")
    workflow.add_edge("load_document", "extract_concepts")
    workflow.add_edge("extract_concepts", "identify_points")
    workflow.add_edge("identify_points", "create_outline")
    workflow.add_edge("create_outline", "generate_summary")
    workflow.add_edge("generate_summary", "refine_summary")
    workflow.add_edge("refine_summary", "assess_quality")
    workflow.add_edge("assess_quality", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the abstractive summarization graph
    app = create_abstractive_summarization_graph()
    
    # Initialize state
    initial_state: AbstractiveSummarizationState = {
        'messages': [],
        'source_document': {},
        'key_concepts': [],
        'main_points': [],
        'summary_outline': {},
        'generated_summary': '',
        'refined_summary': '',
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("ABSTRACTIVE SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nAbstractive summarization pattern execution complete! ✓")

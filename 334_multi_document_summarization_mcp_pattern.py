"""
Multi-Document Summarization MCP Pattern

This pattern demonstrates multi-document summarization in an agentic MCP system.
The system synthesizes information from multiple documents into a coherent summary.

Use cases:
- News aggregation
- Research synthesis
- Comparative analysis
- Information integration
- Multi-source reporting
"""

from typing import TypedDict, Annotated, List, Dict, Any, Set
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import Counter, defaultdict
from datetime import datetime


# Define the state for multi-document summarization
class MultiDocumentSummarizationState(TypedDict):
    """State for tracking multi-document summarization process"""
    messages: Annotated[List[str], add]
    documents: List[Dict[str, Any]]
    topic: str
    document_analysis: Dict[str, Any]
    key_themes: List[Dict[str, Any]]
    consolidated_info: Dict[str, Any]
    cross_doc_relations: List[Dict[str, Any]]
    integrated_summary: str
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class DocumentAnalyzer:
    """Analyze individual documents"""
    
    def extract_main_points(self, document: str) -> List[str]:
        """Extract main points from document"""
        sentences = re.split(r'(?<=[.!?])\s+', document)
        # Simple heuristic: sentences with important keywords
        important = []
        for sentence in sentences:
            words = len(sentence.split())
            if 10 <= words <= 40:  # Good length
                important.append(sentence.strip())
        return important[:3]  # Top 3
    
    def extract_entities(self, document: str) -> Set[str]:
        """Extract named entities (simple capitalized words)"""
        entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', document))
        return entities
    
    def extract_keywords(self, document: str) -> List[str]:
        """Extract keywords using frequency"""
        words = re.findall(r'\b\w+\b', document.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                    'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        filtered = [w for w in words if w not in stopwords and len(w) > 3]
        
        freq = Counter(filtered)
        return [word for word, count in freq.most_common(10)]
    
    def analyze_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single document"""
        content = doc['content']
        
        return {
            'doc_id': doc['id'],
            'title': doc['title'],
            'main_points': self.extract_main_points(content),
            'entities': list(self.extract_entities(content)),
            'keywords': self.extract_keywords(content),
            'length': len(content.split()),
            'source': doc.get('source', 'Unknown')
        }


class ThemeExtractor:
    """Extract common themes across documents"""
    
    def identify_common_keywords(self, doc_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify keywords appearing in multiple documents"""
        keyword_docs = defaultdict(set)
        
        for analysis in doc_analyses:
            for keyword in analysis['keywords']:
                keyword_docs[keyword].add(analysis['doc_id'])
        
        themes = []
        for keyword, doc_ids in keyword_docs.items():
            if len(doc_ids) >= 2:  # Appears in at least 2 docs
                themes.append({
                    'theme': keyword,
                    'document_count': len(doc_ids),
                    'documents': list(doc_ids),
                    'importance': len(doc_ids)
                })
        
        themes.sort(key=lambda x: x['importance'], reverse=True)
        return themes
    
    def identify_entity_overlap(self, doc_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify entities mentioned in multiple documents"""
        entity_docs = defaultdict(set)
        
        for analysis in doc_analyses:
            for entity in analysis['entities']:
                entity_docs[entity].add(analysis['doc_id'])
        
        overlaps = []
        for entity, doc_ids in entity_docs.items():
            if len(doc_ids) >= 2:
                overlaps.append({
                    'entity': entity,
                    'document_count': len(doc_ids),
                    'documents': list(doc_ids),
                    'type': 'entity'
                })
        
        return overlaps


class InformationConsolidator:
    """Consolidate information from multiple documents"""
    
    def group_by_topic(self, themes: List[Dict[str, Any]], 
                       doc_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Group information by topic"""
        
        topics = {}
        
        # Use top themes as topics
        for theme in themes[:5]:
            topic_name = theme['theme']
            relevant_docs = theme['documents']
            
            # Collect relevant content
            content_pieces = []
            for analysis in doc_analyses:
                if analysis['doc_id'] in relevant_docs:
                    for point in analysis['main_points']:
                        if topic_name in point.lower():
                            content_pieces.append({
                                'text': point,
                                'source': analysis['title']
                            })
            
            topics[topic_name] = {
                'document_count': theme['document_count'],
                'content': content_pieces[:3]  # Top 3 pieces
            }
        
        return topics
    
    def identify_agreements(self, doc_analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify common statements across documents"""
        # Simple approach: look for similar main points
        agreements = []
        
        for i, analysis1 in enumerate(doc_analyses):
            for analysis2 in doc_analyses[i+1:]:
                for point1 in analysis1['main_points']:
                    for point2 in analysis2['main_points']:
                        # Check for keyword overlap
                        words1 = set(re.findall(r'\b\w+\b', point1.lower()))
                        words2 = set(re.findall(r'\b\w+\b', point2.lower()))
                        overlap = len(words1 & words2) / len(words1 | words2) if words1 | words2 else 0
                        
                        if overlap > 0.3:  # 30% similarity
                            agreements.append(f"{analysis1['title']} and {analysis2['title']} both discuss: {point1[:80]}...")
                            break
        
        return agreements[:3]  # Top 3
    
    def identify_contradictions(self, doc_analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify potential contradictions"""
        # Simplified: look for opposing keywords
        contradiction_pairs = [
            ('increase', 'decrease'), ('positive', 'negative'),
            ('support', 'oppose'), ('effective', 'ineffective')
        ]
        
        contradictions = []
        for i, analysis1 in enumerate(doc_analyses):
            for analysis2 in doc_analyses[i+1:]:
                doc1_text = ' '.join(analysis1['main_points']).lower()
                doc2_text = ' '.join(analysis2['main_points']).lower()
                
                for word1, word2 in contradiction_pairs:
                    if word1 in doc1_text and word2 in doc2_text:
                        contradictions.append(
                            f"Potential divergence: {analysis1['title']} mentions {word1}, "
                            f"while {analysis2['title']} mentions {word2}"
                        )
                        break
        
        return contradictions[:2]  # Top 2


class CrossDocumentAnalyzer:
    """Analyze relationships between documents"""
    
    def calculate_document_similarity(self, doc1: Dict[str, Any], 
                                      doc2: Dict[str, Any]) -> float:
        """Calculate similarity between two documents"""
        keywords1 = set(doc1['keywords'])
        keywords2 = set(doc2['keywords'])
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0
    
    def build_similarity_matrix(self, doc_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build similarity relationships between documents"""
        
        relations = []
        for i, doc1 in enumerate(doc_analyses):
            for doc2 in doc_analyses[i+1:]:
                similarity = self.calculate_document_similarity(doc1, doc2)
                
                if similarity > 0.2:  # Significant similarity
                    relations.append({
                        'doc1': doc1['title'],
                        'doc2': doc2['title'],
                        'similarity': similarity,
                        'common_keywords': list(set(doc1['keywords']) & set(doc2['keywords']))[:5]
                    })
        
        relations.sort(key=lambda x: x['similarity'], reverse=True)
        return relations


class IntegratedSummarizer:
    """Generate integrated multi-document summary"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def create_context_from_documents(self, doc_analyses: List[Dict[str, Any]], 
                                     consolidated: Dict[str, Any]) -> str:
        """Create context for summarization"""
        
        context_parts = []
        
        # Add document overviews
        context_parts.append("DOCUMENT OVERVIEWS:")
        for analysis in doc_analyses:
            context_parts.append(f"\n{analysis['title']}:")
            context_parts.append(f"  Main points: {' '.join(analysis['main_points'][:2])}")
        
        # Add common themes
        if 'topics' in consolidated:
            context_parts.append("\n\nCOMMON THEMES:")
            for theme, info in list(consolidated['topics'].items())[:3]:
                context_parts.append(f"  - {theme} (mentioned in {info['document_count']} documents)")
        
        # Add agreements
        if 'agreements' in consolidated:
            context_parts.append("\n\nCOMMON FINDINGS:")
            for agreement in consolidated['agreements']:
                context_parts.append(f"  - {agreement}")
        
        return '\n'.join(context_parts)
    
    def generate_integrated_summary(self, topic: str,
                                   doc_analyses: List[Dict[str, Any]],
                                   consolidated: Dict[str, Any]) -> str:
        """Generate integrated summary across documents"""
        
        context = self.create_context_from_documents(doc_analyses, consolidated)
        
        system_prompt = """You are an expert at synthesizing information from multiple sources. 
        Create a coherent summary that integrates key information from all documents, highlighting 
        common themes, agreements, and any notable differences."""
        
        user_prompt = f"""
        Topic: {topic}
        
        {context}
        
        Create an integrated summary (3-4 sentences) that synthesizes information from all documents.
        Focus on common themes and key findings. Mention when multiple sources agree on important points.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class SummaryQualityEvaluator:
    """Evaluate multi-document summary quality"""
    
    def calculate_coverage(self, summary: str, doc_analyses: List[Dict[str, Any]]) -> float:
        """Calculate coverage of key information"""
        
        all_keywords = set()
        for analysis in doc_analyses:
            all_keywords.update(analysis['keywords'][:5])
        
        summary_lower = summary.lower()
        covered = sum(1 for kw in all_keywords if kw in summary_lower)
        
        return covered / len(all_keywords) if all_keywords else 0.0
    
    def calculate_coherence(self, summary: str) -> float:
        """Estimate summary coherence"""
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        if len(sentences) < 2:
            return 1.0
        
        # Check for transition words
        transitions = ['however', 'moreover', 'additionally', 'furthermore', 
                      'therefore', 'consequently', 'similarly', 'in contrast']
        
        transition_count = sum(1 for s in sentences if any(t in s.lower() for t in transitions))
        coherence = min(1.0, transition_count / (len(sentences) - 1) + 0.5)
        
        return coherence
    
    def calculate_informativeness(self, summary: str) -> float:
        """Estimate information density"""
        words = summary.split()
        
        # Simple heuristic: longer summaries with good vocabulary are more informative
        unique_words = len(set(w.lower() for w in words))
        density = unique_words / len(words) if words else 0.0
        
        # Normalize to 0-1 range
        return min(1.0, density * 1.5)


# Agent functions
def initialize_documents_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Initialize documents for summarization"""
    
    topic = "Renewable Energy Adoption"
    
    documents = [
        {
            'id': 'doc1',
            'title': 'Solar Energy Growth Report',
            'source': 'Energy Research Institute',
            'content': """Solar energy capacity has grown exponentially over the past decade. 
            Installation costs have decreased by 70%, making solar more accessible. Government 
            incentives accelerate adoption rates. Grid integration challenges remain significant. 
            Battery storage technology improves solar viability. Efficiency improvements continue 
            with new panel technologies. Large-scale solar farms contribute substantially to 
            energy grids."""
        },
        {
            'id': 'doc2',
            'title': 'Wind Power Developments',
            'source': 'Clean Energy Foundation',
            'content': """Wind power represents a major component of renewable energy expansion. 
            Offshore wind farms deliver higher capacity factors than onshore installations. 
            Turbine technology advances increase energy capture efficiency. Grid integration 
            challenges require infrastructure upgrades. Government incentives support wind 
            development. Environmental concerns about wildlife impact require careful siting. 
            Wind energy costs have decreased substantially."""
        },
        {
            'id': 'doc3',
            'title': 'Policy Impact on Renewables',
            'source': 'Policy Analysis Center',
            'content': """Government incentives play a crucial role in renewable energy adoption. 
            Tax credits reduce installation costs for consumers and businesses. Renewable 
            portfolio standards mandate utility-scale deployment. Grid modernization policies 
            facilitate renewable integration. Regulatory frameworks affect development speed. 
            International climate agreements drive policy changes. Some regions face political 
            opposition to renewable mandates."""
        },
        {
            'id': 'doc4',
            'title': 'Economic Analysis of Renewables',
            'source': 'Economic Research Bureau',
            'content': """Renewable energy costs have declined dramatically, approaching grid 
            parity in many markets. Installation costs vary by technology and geography. 
            Job creation in renewable sectors offsets losses in fossil fuel industries. 
            Energy security benefits from diversified renewable sources. Initial capital 
            requirements remain high despite long-term savings. Market dynamics favor 
            renewables as costs continue falling."""
        }
    ]
    
    return {
        **state,
        'documents': documents,
        'topic': topic,
        'messages': state['messages'] + [f'Initialized {len(documents)} documents on topic: {topic}']
    }


def analyze_documents_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Analyze individual documents"""
    
    analyzer = DocumentAnalyzer()
    analyses = [analyzer.analyze_document(doc) for doc in state['documents']]
    
    return {
        **state,
        'document_analysis': {'analyses': analyses},
        'messages': state['messages'] + [f'Analyzed {len(analyses)} documents']
    }


def extract_themes_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Extract common themes"""
    
    extractor = ThemeExtractor()
    analyses = state['document_analysis']['analyses']
    
    themes = extractor.identify_common_keywords(analyses)
    entity_overlaps = extractor.identify_entity_overlap(analyses)
    
    all_themes = themes + entity_overlaps
    all_themes.sort(key=lambda x: x.get('document_count', x.get('importance', 0)), reverse=True)
    
    return {
        **state,
        'key_themes': all_themes,
        'messages': state['messages'] + [f'Extracted {len(all_themes)} common themes']
    }


def consolidate_information_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Consolidate information from documents"""
    
    consolidator = InformationConsolidator()
    analyses = state['document_analysis']['analyses']
    
    consolidated = {
        'topics': consolidator.group_by_topic(state['key_themes'], analyses),
        'agreements': consolidator.identify_agreements(analyses),
        'contradictions': consolidator.identify_contradictions(analyses),
        'document_count': len(state['documents'])
    }
    
    return {
        **state,
        'consolidated_info': consolidated,
        'messages': state['messages'] + [f'Consolidated information across {len(state["documents"])} documents']
    }


def analyze_cross_document_relations_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Analyze cross-document relationships"""
    
    analyzer = CrossDocumentAnalyzer()
    analyses = state['document_analysis']['analyses']
    
    relations = analyzer.build_similarity_matrix(analyses)
    
    return {
        **state,
        'cross_doc_relations': relations,
        'messages': state['messages'] + [f'Identified {len(relations)} cross-document relationships']
    }


def generate_integrated_summary_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Generate integrated summary"""
    
    summarizer = IntegratedSummarizer()
    summary = summarizer.generate_integrated_summary(
        state['topic'],
        state['document_analysis']['analyses'],
        state['consolidated_info']
    )
    
    return {
        **state,
        'integrated_summary': summary,
        'messages': state['messages'] + [f'Generated integrated summary ({len(summary.split())} words)']
    }


def evaluate_summary_quality_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Evaluate summary quality"""
    
    evaluator = SummaryQualityEvaluator()
    
    metrics = {
        'coverage': evaluator.calculate_coverage(
            state['integrated_summary'],
            state['document_analysis']['analyses']
        ),
        'coherence': evaluator.calculate_coherence(state['integrated_summary']),
        'informativeness': evaluator.calculate_informativeness(state['integrated_summary']),
        'documents_synthesized': len(state['documents']),
        'themes_identified': len(state['key_themes']),
        'summary_length': len(state['integrated_summary'].split())
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Evaluated summary quality (coverage: {metrics["coverage"]:.1%})']
    }


def analyze_multi_doc_summarization_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Analyze multi-document summarization results"""
    
    analytics = {
        'document_stats': {
            'total_documents': len(state['documents']),
            'total_words': sum(len(doc['content'].split()) for doc in state['documents']),
            'avg_doc_length': sum(len(doc['content'].split()) for doc in state['documents']) / len(state['documents'])
        },
        'theme_stats': {
            'themes_found': len(state['key_themes']),
            'multi_doc_themes': sum(1 for t in state['key_themes'] if t.get('document_count', 0) >= 3),
            'top_theme': state['key_themes'][0]['theme'] if state['key_themes'] else None
        },
        'consolidation_stats': {
            'topics_identified': len(state['consolidated_info']['topics']),
            'agreements_found': len(state['consolidated_info']['agreements']),
            'contradictions_found': len(state['consolidated_info']['contradictions'])
        },
        'relation_stats': {
            'document_pairs_analyzed': len(state['cross_doc_relations']),
            'high_similarity_pairs': sum(1 for r in state['cross_doc_relations'] if r['similarity'] > 0.4)
        },
        'quality_stats': state['quality_metrics']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed multi-document summarization results']
    }


def generate_multi_doc_report_agent(state: MultiDocumentSummarizationState) -> MultiDocumentSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "MULTI-DOCUMENT SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "TOPIC:",
        "-" * 40,
        state['topic'],
        "",
        "SOURCE DOCUMENTS:",
        "-" * 40
    ]
    
    for doc in state['documents']:
        report_lines.append(f"• {doc['title']} ({doc['source']})")
    
    report_lines.extend([
        "",
        "INTEGRATED SUMMARY:",
        "-" * 40,
        state['integrated_summary'],
        "",
        "",
        "COMMON THEMES:",
        "-" * 40
    ])
    
    for i, theme in enumerate(state['key_themes'][:5], 1):
        theme_name = theme.get('theme', theme.get('entity', 'Unknown'))
        doc_count = theme.get('document_count', theme.get('importance', 0))
        report_lines.append(f"{i}. {theme_name} (appears in {doc_count} documents)")
    
    report_lines.extend([
        "",
        "CONSOLIDATED FINDINGS:",
        "-" * 40,
        "",
        "Agreements:"
    ])
    
    for agreement in state['consolidated_info']['agreements']:
        report_lines.append(f"  • {agreement}")
    
    if state['consolidated_info']['contradictions']:
        report_lines.append("\nDivergent Views:")
        for contradiction in state['consolidated_info']['contradictions']:
            report_lines.append(f"  • {contradiction}")
    
    report_lines.extend([
        "",
        "",
        "DOCUMENT RELATIONSHIPS:",
        "-" * 40
    ])
    
    for i, relation in enumerate(state['cross_doc_relations'][:3], 1):
        report_lines.append(f"\n{i}. {relation['doc1']} ↔ {relation['doc2']}")
        report_lines.append(f"   Similarity: {relation['similarity']:.1%}")
        report_lines.append(f"   Common keywords: {', '.join(relation['common_keywords'][:3])}")
    
    report_lines.extend([
        "",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Coverage: {state['analytics']['quality_stats']['coverage']:.1%}",
        f"Coherence: {state['analytics']['quality_stats']['coherence']:.1%}",
        f"Informativeness: {state['analytics']['quality_stats']['informativeness']:.1%}",
        f"Documents Synthesized: {state['analytics']['quality_stats']['documents_synthesized']}",
        f"Summary Length: {state['analytics']['quality_stats']['summary_length']} words",
        "",
        "SYNTHESIS STATISTICS:",
        "-" * 40,
        f"Total Input Words: {state['analytics']['document_stats']['total_words']}",
        f"Compression Ratio: {state['analytics']['document_stats']['total_words'] / state['analytics']['quality_stats']['summary_length']:.1f}:1",
        f"Themes Identified: {state['analytics']['theme_stats']['themes_found']}",
        f"Multi-Document Themes: {state['analytics']['theme_stats']['multi_doc_themes']}",
        f"Document Pairs Analyzed: {state['analytics']['relation_stats']['document_pairs_analyzed']}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Successfully synthesized {len(state['documents'])} documents into coherent summary",
        f"✓ Identified {state['analytics']['theme_stats']['multi_doc_themes']} themes across multiple documents",
        f"✓ Found {len(state['consolidated_info']['agreements'])} common findings across sources",
        f"✓ Achieved {state['analytics']['quality_stats']['coverage']:.0%} coverage of key information",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement citation tracking for source attribution",
        "• Add temporal analysis for time-sensitive topics",
        "• Develop conflict resolution for contradictory information",
        "• Enhance entity linking across documents",
        "• Implement hierarchical summarization for large document sets",
        "• Add diversity metrics to ensure balanced coverage",
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
def create_multi_document_summarization_graph():
    """Create the multi-document summarization workflow graph"""
    
    workflow = StateGraph(MultiDocumentSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_documents_agent)
    workflow.add_node("analyze_documents", analyze_documents_agent)
    workflow.add_node("extract_themes", extract_themes_agent)
    workflow.add_node("consolidate_info", consolidate_information_agent)
    workflow.add_node("analyze_relations", analyze_cross_document_relations_agent)
    workflow.add_node("generate_summary", generate_integrated_summary_agent)
    workflow.add_node("evaluate_quality", evaluate_summary_quality_agent)
    workflow.add_node("analyze_results", analyze_multi_doc_summarization_agent)
    workflow.add_node("generate_report", generate_multi_doc_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_documents")
    workflow.add_edge("analyze_documents", "extract_themes")
    workflow.add_edge("extract_themes", "consolidate_info")
    workflow.add_edge("consolidate_info", "analyze_relations")
    workflow.add_edge("analyze_relations", "generate_summary")
    workflow.add_edge("generate_summary", "evaluate_quality")
    workflow.add_edge("evaluate_quality", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the multi-document summarization graph
    app = create_multi_document_summarization_graph()
    
    # Initialize state
    initial_state: MultiDocumentSummarizationState = {
        'messages': [],
        'documents': [],
        'topic': '',
        'document_analysis': {},
        'key_themes': [],
        'consolidated_info': {},
        'cross_doc_relations': [],
        'integrated_summary': '',
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("MULTI-DOCUMENT SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nMulti-document summarization pattern execution complete! ✓")

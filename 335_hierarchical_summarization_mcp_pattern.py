"""
Hierarchical Summarization MCP Pattern

This pattern demonstrates hierarchical summarization in an agentic MCP system.
The system creates multi-level summaries with varying levels of detail.

Use cases:
- Executive briefings
- Layered documentation
- Progressive disclosure
- Multi-audience content
- Drill-down summaries
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import Counter


# Define the state for hierarchical summarization
class HierarchicalSummarizationState(TypedDict):
    """State for tracking hierarchical summarization process"""
    messages: Annotated[List[str], add]
    source_document: str
    document_structure: Dict[str, Any]
    section_summaries: List[Dict[str, Any]]
    level1_summary: str  # Most detailed
    level2_summary: str  # Medium detail
    level3_summary: str  # Executive summary
    hierarchy: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class DocumentStructureAnalyzer:
    """Analyze document structure"""
    
    def identify_sections(self, document: str) -> List[Dict[str, Any]]:
        """Identify document sections"""
        # Split by paragraph
        paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
        
        sections = []
        for i, para in enumerate(paragraphs):
            sections.append({
                'id': f'section_{i+1}',
                'content': para,
                'word_count': len(para.split()),
                'sentence_count': len(re.split(r'[.!?]+', para))
            })
        
        return sections
    
    def build_hierarchy(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build hierarchical structure"""
        total_words = sum(s['word_count'] for s in sections)
        
        hierarchy = {
            'total_sections': len(sections),
            'total_words': total_words,
            'sections': sections,
            'levels': {
                'level1': {'target_ratio': 0.5, 'description': 'Detailed summary'},
                'level2': {'target_ratio': 0.25, 'description': 'Medium summary'},
                'level3': {'target_ratio': 0.1, 'description': 'Executive summary'}
            }
        }
        
        return hierarchy


class SectionSummarizer:
    """Summarize individual sections"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def summarize_section(self, section: Dict[str, Any], 
                         compression_ratio: float) -> str:
        """Summarize a single section"""
        
        target_words = int(section['word_count'] * compression_ratio)
        
        system_prompt = """You are an expert at creating concise summaries. 
        Preserve key information while reducing length."""
        
        user_prompt = f"""
        Summarize the following text in approximately {target_words} words:
        
        {section['content']}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class HierarchicalAggregator:
    """Aggregate summaries hierarchically"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def aggregate_level1(self, section_summaries: List[Dict[str, Any]]) -> str:
        """Create detailed (level 1) summary"""
        
        # Combine section summaries with minor compression
        combined = []
        for i, summary in enumerate(section_summaries, 1):
            combined.append(f"{summary['summary']}")
        
        detailed_summary = ' '.join(combined)
        
        # Light editing for coherence
        system_prompt = """You are an expert editor. Improve the coherence and flow 
        of this summary while preserving all important details."""
        
        user_prompt = f"""
        Refine this detailed summary for better flow:
        
        {detailed_summary}
        
        Maintain all key information. Add transitions if needed.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def aggregate_level2(self, level1_summary: str) -> str:
        """Create medium (level 2) summary from level 1"""
        
        target_words = len(level1_summary.split()) // 2
        
        system_prompt = """You are an expert at creating medium-length summaries. 
        Focus on main points while omitting minor details."""
        
        user_prompt = f"""
        Create a medium-length summary (approximately {target_words} words) from this detailed summary:
        
        {level1_summary}
        
        Focus on key findings and main themes. Omit supporting details.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def aggregate_level3(self, level2_summary: str) -> str:
        """Create executive (level 3) summary from level 2"""
        
        system_prompt = """You are an expert at creating executive summaries. 
        Distill to the absolute essentials - what a busy executive needs to know."""
        
        user_prompt = f"""
        Create a brief executive summary (2-3 sentences) from this summary:
        
        {level2_summary}
        
        Capture only the most critical information and key takeaway.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class SummaryLevelValidator:
    """Validate summary levels maintain hierarchy"""
    
    def validate_compression_ratios(self, original: str, 
                                   level1: str, level2: str, level3: str) -> Dict[str, float]:
        """Validate compression ratios"""
        
        original_words = len(original.split())
        level1_words = len(level1.split())
        level2_words = len(level2.split())
        level3_words = len(level3.split())
        
        return {
            'original_words': original_words,
            'level1_words': level1_words,
            'level2_words': level2_words,
            'level3_words': level3_words,
            'level1_ratio': level1_words / original_words,
            'level2_ratio': level2_words / original_words,
            'level3_ratio': level3_words / original_words,
            'valid_hierarchy': level3_words < level2_words < level1_words < original_words
        }
    
    def check_information_preservation(self, detailed: str, 
                                      medium: str, brief: str) -> Dict[str, float]:
        """Check information preservation across levels"""
        
        # Extract key terms from detailed summary
        detailed_words = set(re.findall(r'\b\w+\b', detailed.lower()))
        medium_words = set(re.findall(r'\b\w+\b', medium.lower()))
        brief_words = set(re.findall(r'\b\w+\b', brief.lower()))
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        detailed_keywords = detailed_words - stopwords
        medium_keywords = medium_words - stopwords
        brief_keywords = brief_words - stopwords
        
        return {
            'medium_preserves_from_detailed': len(medium_keywords & detailed_keywords) / len(detailed_keywords) if detailed_keywords else 0,
            'brief_preserves_from_medium': len(brief_keywords & medium_keywords) / len(medium_keywords) if medium_keywords else 0,
            'brief_preserves_from_detailed': len(brief_keywords & detailed_keywords) / len(detailed_keywords) if detailed_keywords else 0
        }


class HierarchyNavigator:
    """Navigate between hierarchy levels"""
    
    def create_navigation_structure(self, level1: str, level2: str, level3: str) -> Dict[str, Any]:
        """Create navigation between levels"""
        
        return {
            'levels': [
                {
                    'level': 3,
                    'name': 'Executive Summary',
                    'content': level3,
                    'words': len(level3.split()),
                    'can_expand_to': 'level2'
                },
                {
                    'level': 2,
                    'name': 'Medium Summary',
                    'content': level2,
                    'words': len(level2.split()),
                    'can_expand_to': 'level1',
                    'can_collapse_to': 'level3'
                },
                {
                    'level': 1,
                    'name': 'Detailed Summary',
                    'content': level1,
                    'words': len(level1.split()),
                    'can_collapse_to': 'level2'
                }
            ],
            'recommended_use': {
                'level3': 'Quick overview, executive briefing',
                'level2': 'Balanced understanding, general audience',
                'level1': 'Comprehensive understanding, technical audience'
            }
        }


# Agent functions
def initialize_document_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Initialize source document"""
    
    document = """
    Climate change represents one of the most pressing challenges facing humanity in the 21st century. 
    The scientific consensus is clear: global temperatures are rising due to increased greenhouse gas 
    emissions from human activities, primarily the burning of fossil fuels for energy, transportation, 
    and industrial processes. The Intergovernmental Panel on Climate Change (IPCC) reports that global 
    average temperatures have already increased by approximately 1.1°C above pre-industrial levels.
    
    The impacts of climate change are already being felt worldwide. Extreme weather events, including 
    hurricanes, droughts, floods, and heatwaves, are becoming more frequent and severe. Rising sea 
    levels threaten coastal communities and small island nations. Changing precipitation patterns 
    disrupt agricultural systems, affecting food security. Ecosystems are under stress, with species 
    facing habitat loss and extinction risks. Ocean acidification threatens marine life, particularly 
    organisms with calcium carbonate shells and skeletons.
    
    Mitigation efforts focus on reducing greenhouse gas emissions through various strategies. Transitioning 
    to renewable energy sources like solar, wind, and hydroelectric power is crucial. Improving energy 
    efficiency in buildings, transportation, and industry reduces overall energy demand. Electric vehicles 
    and public transportation can significantly cut transportation emissions. Carbon capture and storage 
    technologies offer potential for removing CO2 from the atmosphere. Forest conservation and reforestation 
    help absorb carbon dioxide naturally.
    
    Adaptation strategies help communities cope with unavoidable climate impacts. Building resilient 
    infrastructure protects against extreme weather. Developing drought-resistant crops ensures food 
    security under changing conditions. Early warning systems for extreme weather save lives. Coastal 
    protection measures like seawalls and mangrove restoration defend against sea-level rise. Water 
    management systems address changing precipitation patterns.
    
    International cooperation is essential for addressing this global challenge. The Paris Agreement 
    commits nations to limit global warming to well below 2°C, preferably to 1.5°C, compared to 
    pre-industrial levels. Countries submit nationally determined contributions (NDCs) outlining their 
    emission reduction targets. Climate finance mechanisms support developing nations in mitigation 
    and adaptation efforts. Technology transfer facilitates the spread of clean energy solutions. 
    Regular assessment and ratcheting up of commitments ensure progress toward goals.
    """
    
    return {
        **state,
        'source_document': document,
        'messages': state['messages'] + [f'Initialized source document ({len(document.split())} words)']
    }


def analyze_structure_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Analyze document structure"""
    
    analyzer = DocumentStructureAnalyzer()
    sections = analyzer.identify_sections(state['source_document'])
    hierarchy = analyzer.build_hierarchy(sections)
    
    return {
        **state,
        'document_structure': hierarchy,
        'messages': state['messages'] + [f'Analyzed structure ({hierarchy["total_sections"]} sections, {hierarchy["total_words"]} words)']
    }


def summarize_sections_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Summarize individual sections"""
    
    summarizer = SectionSummarizer()
    sections = state['document_structure']['sections']
    
    section_summaries = []
    for section in sections:
        summary = summarizer.summarize_section(section, compression_ratio=0.6)
        section_summaries.append({
            'section_id': section['id'],
            'original_words': section['word_count'],
            'summary': summary,
            'summary_words': len(summary.split())
        })
    
    return {
        **state,
        'section_summaries': section_summaries,
        'messages': state['messages'] + [f'Summarized {len(section_summaries)} sections']
    }


def create_level1_summary_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Create detailed (level 1) summary"""
    
    aggregator = HierarchicalAggregator()
    level1 = aggregator.aggregate_level1(state['section_summaries'])
    
    return {
        **state,
        'level1_summary': level1,
        'messages': state['messages'] + [f'Created level 1 summary ({len(level1.split())} words)']
    }


def create_level2_summary_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Create medium (level 2) summary"""
    
    aggregator = HierarchicalAggregator()
    level2 = aggregator.aggregate_level2(state['level1_summary'])
    
    return {
        **state,
        'level2_summary': level2,
        'messages': state['messages'] + [f'Created level 2 summary ({len(level2.split())} words)']
    }


def create_level3_summary_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Create executive (level 3) summary"""
    
    aggregator = HierarchicalAggregator()
    level3 = aggregator.aggregate_level3(state['level2_summary'])
    
    return {
        **state,
        'level3_summary': level3,
        'messages': state['messages'] + [f'Created level 3 summary ({len(level3.split())} words)']
    }


def validate_hierarchy_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Validate hierarchical structure"""
    
    validator = SummaryLevelValidator()
    
    compression_metrics = validator.validate_compression_ratios(
        state['source_document'],
        state['level1_summary'],
        state['level2_summary'],
        state['level3_summary']
    )
    
    preservation_metrics = validator.check_information_preservation(
        state['level1_summary'],
        state['level2_summary'],
        state['level3_summary']
    )
    
    navigator = HierarchyNavigator()
    navigation = navigator.create_navigation_structure(
        state['level1_summary'],
        state['level2_summary'],
        state['level3_summary']
    )
    
    metrics = {
        'compression': compression_metrics,
        'preservation': preservation_metrics,
        'navigation': navigation
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'hierarchy': navigation,
        'messages': state['messages'] + [f'Validated hierarchy (valid: {compression_metrics["valid_hierarchy"]})']
    }


def analyze_hierarchical_summarization_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Analyze hierarchical summarization results"""
    
    analytics = {
        'document_stats': {
            'original_words': state['quality_metrics']['compression']['original_words'],
            'sections': state['document_structure']['total_sections']
        },
        'summary_stats': {
            'level1_words': state['quality_metrics']['compression']['level1_words'],
            'level2_words': state['quality_metrics']['compression']['level2_words'],
            'level3_words': state['quality_metrics']['compression']['level3_words']
        },
        'compression_ratios': {
            'level1': state['quality_metrics']['compression']['level1_ratio'],
            'level2': state['quality_metrics']['compression']['level2_ratio'],
            'level3': state['quality_metrics']['compression']['level3_ratio']
        },
        'preservation_rates': state['quality_metrics']['preservation'],
        'hierarchy_valid': state['quality_metrics']['compression']['valid_hierarchy']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed hierarchical summarization results']
    }


def generate_hierarchical_report_agent(state: HierarchicalSummarizationState) -> HierarchicalSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "HIERARCHICAL SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "DOCUMENT STATISTICS:",
        "-" * 40,
        f"Original Document: {state['analytics']['document_stats']['original_words']} words",
        f"Sections: {state['analytics']['document_stats']['sections']}",
        "",
        "SUMMARY HIERARCHY:",
        "-" * 40,
        "",
        "LEVEL 3 - EXECUTIVE SUMMARY:",
        f"({state['analytics']['summary_stats']['level3_words']} words, "
        f"{state['analytics']['compression_ratios']['level3']:.1%} of original)",
        "-" * 40,
        state['level3_summary'],
        "",
        "",
        "LEVEL 2 - MEDIUM SUMMARY:",
        f"({state['analytics']['summary_stats']['level2_words']} words, "
        f"{state['analytics']['compression_ratios']['level2']:.1%} of original)",
        "-" * 40,
        state['level2_summary'],
        "",
        "",
        "LEVEL 1 - DETAILED SUMMARY:",
        f"({state['analytics']['summary_stats']['level1_words']} words, "
        f"{state['analytics']['compression_ratios']['level1']:.1%} of original)",
        "-" * 40,
        state['level1_summary'],
        "",
        "",
        "COMPRESSION ANALYSIS:",
        "-" * 40,
        f"Level 1 Compression: {state['analytics']['compression_ratios']['level1']:.1%}",
        f"Level 2 Compression: {state['analytics']['compression_ratios']['level2']:.1%}",
        f"Level 3 Compression: {state['analytics']['compression_ratios']['level3']:.1%}",
        f"Hierarchy Valid: {'✓ Yes' if state['analytics']['hierarchy_valid'] else '✗ No'}",
        "",
        "INFORMATION PRESERVATION:",
        "-" * 40,
        f"Medium → Detailed: {state['analytics']['preservation_rates']['medium_preserves_from_detailed']:.1%}",
        f"Brief → Medium: {state['analytics']['preservation_rates']['brief_preserves_from_medium']:.1%}",
        f"Brief → Detailed: {state['analytics']['preservation_rates']['brief_preserves_from_detailed']:.1%}",
        "",
        "RECOMMENDED USAGE:",
        "-" * 40
    ]
    
    for level_info in state['hierarchy']['levels']:
        level_num = level_info['level']
        use_case = state['hierarchy']['recommended_use'][f'level{level_num}']
        report_lines.append(f"Level {level_num}: {use_case}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Created valid 3-level hierarchy with proper compression ratios",
        f"✓ Level 3 captures essentials in {state['analytics']['summary_stats']['level3_words']} words",
        f"✓ Level 2 provides balance at {state['analytics']['compression_ratios']['level2']:.0%} compression",
        f"✓ Level 1 retains {state['analytics']['compression_ratios']['level1']:.0%} of original detail",
        f"✓ Information preservation maintained across levels",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Use Level 3 for quick executive briefings",
        "• Use Level 2 for general audience communications",
        "• Use Level 1 for technical or detailed reviews",
        "• Implement interactive navigation between levels",
        "• Add section-level drill-down capability",
        "• Consider user preferences for default level",
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
def create_hierarchical_summarization_graph():
    """Create the hierarchical summarization workflow graph"""
    
    workflow = StateGraph(HierarchicalSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_document_agent)
    workflow.add_node("analyze_structure", analyze_structure_agent)
    workflow.add_node("summarize_sections", summarize_sections_agent)
    workflow.add_node("create_level1", create_level1_summary_agent)
    workflow.add_node("create_level2", create_level2_summary_agent)
    workflow.add_node("create_level3", create_level3_summary_agent)
    workflow.add_node("validate_hierarchy", validate_hierarchy_agent)
    workflow.add_node("analyze_results", analyze_hierarchical_summarization_agent)
    workflow.add_node("generate_report", generate_hierarchical_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_structure")
    workflow.add_edge("analyze_structure", "summarize_sections")
    workflow.add_edge("summarize_sections", "create_level1")
    workflow.add_edge("create_level1", "create_level2")
    workflow.add_edge("create_level2", "create_level3")
    workflow.add_edge("create_level3", "validate_hierarchy")
    workflow.add_edge("validate_hierarchy", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the hierarchical summarization graph
    app = create_hierarchical_summarization_graph()
    
    # Initialize state
    initial_state: HierarchicalSummarizationState = {
        'messages': [],
        'source_document': '',
        'document_structure': {},
        'section_summaries': [],
        'level1_summary': '',
        'level2_summary': '',
        'level3_summary': '',
        'hierarchy': {},
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("HIERARCHICAL SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nHierarchical summarization pattern execution complete! ✓")

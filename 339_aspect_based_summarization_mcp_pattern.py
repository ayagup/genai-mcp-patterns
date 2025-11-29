"""
Aspect-Based Summarization MCP Pattern

This pattern demonstrates aspect-based summarization in an agentic MCP system.
The system creates summaries focused on specific aspects or perspectives.

Use cases:
- Sentiment analysis summaries
- Feature-focused product reviews
- Multi-perspective news analysis
- Stakeholder-specific briefings
- Topic-focused content extraction
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import Counter, defaultdict


# Define the state for aspect-based summarization
class AspectBasedSummarizationState(TypedDict):
    """State for tracking aspect-based summarization process"""
    messages: Annotated[List[str], add]
    source_document: str
    target_aspects: List[str]
    aspect_detection: Dict[str, Any]
    aspect_extractions: Dict[str, List[str]]
    aspect_summaries: Dict[str, str]
    aspect_sentiments: Dict[str, Dict[str, Any]]
    integrated_summary: str
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class AspectDetector:
    """Detect aspects in document"""
    
    def detect_implicit_aspects(self, document: str) -> List[str]:
        """Detect implicit aspects from content"""
        
        # Common aspect categories
        aspect_keywords = {
            'quality': ['quality', 'excellent', 'poor', 'good', 'bad', 'best', 'worst'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value'],
            'performance': ['performance', 'fast', 'slow', 'efficient', 'effective'],
            'design': ['design', 'appearance', 'look', 'style', 'aesthetic'],
            'usability': ['easy', 'difficult', 'simple', 'complex', 'intuitive', 'usable'],
            'reliability': ['reliable', 'stable', 'consistent', 'dependable', 'trustworthy'],
            'features': ['feature', 'capability', 'function', 'functionality'],
            'support': ['support', 'service', 'help', 'assistance', 'customer service']
        }
        
        doc_lower = document.lower()
        detected_aspects = []
        
        for aspect, keywords in aspect_keywords.items():
            if any(keyword in doc_lower for keyword in keywords):
                detected_aspects.append(aspect)
        
        return detected_aspects
    
    def score_aspect_relevance(self, document: str, aspect: str) -> float:
        """Score how relevant an aspect is"""
        
        aspect_keywords = {
            'quality': ['quality', 'excellent', 'superior', 'premium'],
            'price': ['price', 'cost', 'pricing', 'affordable'],
            'performance': ['performance', 'speed', 'efficiency', 'effectiveness'],
            'design': ['design', 'appearance', 'aesthetic', 'style'],
            'usability': ['usability', 'ease', 'user-friendly', 'intuitive'],
            'reliability': ['reliability', 'stable', 'consistent', 'dependable'],
            'features': ['feature', 'capability', 'functionality'],
            'support': ['support', 'service', 'customer service']
        }
        
        keywords = aspect_keywords.get(aspect, [aspect])
        doc_lower = document.lower()
        
        matches = sum(doc_lower.count(keyword) for keyword in keywords)
        total_words = len(document.split())
        
        relevance = min(1.0, (matches / total_words) * 100) if total_words > 0 else 0.0
        return relevance


class AspectExtractor:
    """Extract content related to specific aspects"""
    
    def extract_aspect_sentences(self, document: str, aspect: str) -> List[str]:
        """Extract sentences mentioning aspect"""
        
        aspect_keywords = {
            'quality': ['quality', 'excellent', 'superior', 'premium', 'good', 'bad', 'poor'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'pricing'],
            'performance': ['performance', 'fast', 'slow', 'efficient', 'speed', 'responsive'],
            'design': ['design', 'appearance', 'look', 'aesthetic', 'style', 'beautiful'],
            'usability': ['easy', 'difficult', 'simple', 'complex', 'intuitive', 'user-friendly'],
            'reliability': ['reliable', 'stable', 'consistent', 'dependable', 'crash', 'bug'],
            'features': ['feature', 'capability', 'function', 'functionality', 'includes'],
            'support': ['support', 'service', 'help', 'assistance', 'customer service']
        }
        
        keywords = aspect_keywords.get(aspect, [aspect])
        sentences = re.split(r'(?<=[.!?])\s+', document)
        
        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                relevant_sentences.append(sentence.strip())
        
        return relevant_sentences
    
    def rank_aspect_sentences(self, sentences: List[str], aspect: str) -> List[Dict[str, Any]]:
        """Rank sentences by aspect relevance"""
        
        ranked = []
        for sentence in sentences:
            # Simple relevance scoring
            word_count = len(sentence.split())
            
            # Prefer medium-length sentences
            if 10 <= word_count <= 30:
                length_score = 1.0
            else:
                length_score = 0.7
            
            # Count aspect keyword mentions
            sentence_lower = sentence.lower()
            keyword_count = sentence_lower.count(aspect.lower())
            keyword_score = min(1.0, keyword_count * 0.3)
            
            relevance = (length_score * 0.4 + keyword_score * 0.6)
            
            ranked.append({
                'sentence': sentence,
                'relevance': relevance,
                'word_count': word_count
            })
        
        ranked.sort(key=lambda x: x['relevance'], reverse=True)
        return ranked


class AspectSummarizer:
    """Create aspect-focused summaries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def summarize_aspect(self, aspect: str, relevant_content: List[str]) -> str:
        """Create summary for specific aspect"""
        
        if not relevant_content:
            return f"No information available about {aspect}."
        
        content_text = '\n'.join(relevant_content[:5])  # Top 5 sentences
        
        system_prompt = f"""You are an expert at creating aspect-focused summaries. 
        Focus specifically on the {aspect} aspect."""
        
        user_prompt = f"""
        Create a concise summary (2-3 sentences) about the {aspect} aspect based on:
        
        {content_text}
        
        Focus only on {aspect}-related information. Be specific and factual.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class SentimentAnalyzer:
    """Analyze sentiment for each aspect"""
    
    def analyze_aspect_sentiment(self, aspect_content: List[str]) -> Dict[str, Any]:
        """Analyze sentiment of aspect-related content"""
        
        if not aspect_content:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        # Simple sentiment analysis
        positive_words = ['excellent', 'great', 'good', 'amazing', 'wonderful', 'fantastic',
                         'love', 'best', 'perfect', 'superior', 'impressive']
        negative_words = ['bad', 'poor', 'terrible', 'awful', 'worst', 'disappointing',
                         'hate', 'useless', 'inferior', 'problematic']
        
        combined_text = ' '.join(aspect_content).lower()
        
        positive_count = sum(combined_text.count(word) for word in positive_words)
        negative_count = sum(combined_text.count(word) for word in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        if sentiment_score > 0.3:
            sentiment = 'positive'
        elif sentiment_score < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        confidence = min(1.0, total_sentiment_words / 10)
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': confidence,
            'positive_mentions': positive_count,
            'negative_mentions': negative_count
        }


class AspectIntegrator:
    """Integrate aspect summaries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def integrate_aspects(self, aspect_summaries: Dict[str, str],
                         sentiments: Dict[str, Dict[str, Any]]) -> str:
        """Integrate aspect summaries into cohesive overview"""
        
        aspect_texts = []
        for aspect, summary in aspect_summaries.items():
            sentiment = sentiments.get(aspect, {}).get('sentiment', 'neutral')
            aspect_texts.append(f"{aspect.capitalize()} ({sentiment}): {summary}")
        
        combined = '\n'.join(aspect_texts)
        
        system_prompt = """You are an expert at creating integrated summaries from 
        aspect-based analyses. Create a cohesive overview that synthesizes the aspects."""
        
        user_prompt = f"""
        Create an integrated summary (3-4 sentences) from these aspect summaries:
        
        {combined}
        
        Provide a balanced overview that highlights key points across aspects.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


# Agent functions
def initialize_aspect_document_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Initialize document and target aspects"""
    
    document = """
    The new XPhone Pro delivers impressive performance with its latest processor, 
    achieving benchmark scores that surpass competitors. The device handles demanding 
    applications smoothly and multitasking is effortless. Battery life is excellent, 
    lasting a full day with heavy use.
    
    Design-wise, the XPhone Pro features a sleek aluminum body with a stunning 6.5-inch 
    OLED display. The build quality feels premium and the device is surprisingly lightweight. 
    However, the glass back is prone to fingerprints and scratches without a case.
    
    At $999, the XPhone Pro is positioned as a premium device. While the price is high, 
    the value proposition is strong given the features and performance. Some users may 
    find it expensive compared to mid-range alternatives that offer 80% of the functionality 
    at half the price.
    
    The camera system is a standout feature, with excellent low-light performance and 
    accurate color reproduction. The 108MP main sensor captures stunning detail. Video 
    recording at 8K is smooth, though it drains the battery quickly.
    
    User experience is generally positive, with an intuitive interface and helpful 
    onboarding. However, some users report the learning curve for advanced features 
    is steep. Customer support has been responsive, with most issues resolved within 
    24 hours.
    """
    
    # Define target aspects
    aspects = ['performance', 'design', 'price', 'features', 'usability']
    
    return {
        **state,
        'source_document': document,
        'target_aspects': aspects,
        'messages': state['messages'] + [f'Initialized document with {len(aspects)} target aspects']
    }


def detect_aspects_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Detect aspects in document"""
    
    detector = AspectDetector()
    
    # Detect implicit aspects
    detected = detector.detect_implicit_aspects(state['source_document'])
    
    # Score target aspects
    aspect_scores = {}
    for aspect in state['target_aspects']:
        score = detector.score_aspect_relevance(state['source_document'], aspect)
        aspect_scores[aspect] = score
    
    detection_results = {
        'detected_aspects': detected,
        'target_aspect_scores': aspect_scores,
        'highly_relevant': [a for a, s in aspect_scores.items() if s > 0.5]
    }
    
    return {
        **state,
        'aspect_detection': detection_results,
        'messages': state['messages'] + [f'Detected {len(detected)} implicit aspects, scored {len(aspect_scores)} target aspects']
    }


def extract_aspect_content_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Extract content for each aspect"""
    
    extractor = AspectExtractor()
    extractions = {}
    
    for aspect in state['target_aspects']:
        # Extract relevant sentences
        sentences = extractor.extract_aspect_sentences(state['source_document'], aspect)
        
        # Rank by relevance
        ranked = extractor.rank_aspect_sentences(sentences, aspect)
        
        # Keep top sentences
        top_sentences = [item['sentence'] for item in ranked[:3]]
        extractions[aspect] = top_sentences
    
    return {
        **state,
        'aspect_extractions': extractions,
        'messages': state['messages'] + [f'Extracted content for {len(extractions)} aspects']
    }


def summarize_aspects_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Create summaries for each aspect"""
    
    summarizer = AspectSummarizer()
    summaries = {}
    
    for aspect, content in state['aspect_extractions'].items():
        summary = summarizer.summarize_aspect(aspect, content)
        summaries[aspect] = summary
    
    return {
        **state,
        'aspect_summaries': summaries,
        'messages': state['messages'] + [f'Created summaries for {len(summaries)} aspects']
    }


def analyze_sentiments_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Analyze sentiment for each aspect"""
    
    analyzer = SentimentAnalyzer()
    sentiments = {}
    
    for aspect, content in state['aspect_extractions'].items():
        sentiment = analyzer.analyze_aspect_sentiment(content)
        sentiments[aspect] = sentiment
    
    return {
        **state,
        'aspect_sentiments': sentiments,
        'messages': state['messages'] + [f'Analyzed sentiment for {len(sentiments)} aspects']
    }


def integrate_aspects_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Integrate aspect summaries"""
    
    integrator = AspectIntegrator()
    integrated = integrator.integrate_aspects(
        state['aspect_summaries'],
        state['aspect_sentiments']
    )
    
    return {
        **state,
        'integrated_summary': integrated,
        'messages': state['messages'] + [f'Integrated {len(state["aspect_summaries"])} aspect summaries']
    }


def evaluate_aspect_coverage_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Evaluate aspect coverage"""
    
    metrics = {
        'aspects_targeted': len(state['target_aspects']),
        'aspects_extracted': len([a for a, c in state['aspect_extractions'].items() if c]),
        'aspects_summarized': len(state['aspect_summaries']),
        'coverage': len([a for a, c in state['aspect_extractions'].items() if c]) / len(state['target_aspects']) if state['target_aspects'] else 0,
        'avg_content_per_aspect': sum(len(c) for c in state['aspect_extractions'].values()) / len(state['aspect_extractions']) if state['aspect_extractions'] else 0,
        'sentiment_distribution': {
            'positive': sum(1 for s in state['aspect_sentiments'].values() if s['sentiment'] == 'positive'),
            'negative': sum(1 for s in state['aspect_sentiments'].values() if s['sentiment'] == 'negative'),
            'neutral': sum(1 for s in state['aspect_sentiments'].values() if s['sentiment'] == 'neutral')
        }
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Evaluated coverage ({metrics["coverage"]:.1%} of aspects covered)']
    }


def analyze_aspect_summarization_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Analyze aspect-based summarization results"""
    
    analytics = {
        'aspect_stats': {
            'total_aspects': len(state['target_aspects']),
            'aspects_found': len([a for a, c in state['aspect_extractions'].items() if c]),
            'highly_relevant': state['aspect_detection']['highly_relevant']
        },
        'extraction_stats': {
            aspect: len(content) for aspect, content in state['aspect_extractions'].items()
        },
        'sentiment_stats': state['quality_metrics']['sentiment_distribution'],
        'quality_stats': {
            'coverage': state['quality_metrics']['coverage'],
            'avg_content_per_aspect': state['quality_metrics']['avg_content_per_aspect']
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed aspect-based summarization results']
    }


def generate_aspect_report_agent(state: AspectBasedSummarizationState) -> AspectBasedSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "ASPECT-BASED SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "INTEGRATED SUMMARY:",
        "-" * 40,
        state['integrated_summary'],
        "",
        "",
        "ASPECT SUMMARIES:",
        "-" * 40
    ]
    
    for aspect in state['target_aspects']:
        summary = state['aspect_summaries'].get(aspect, 'N/A')
        sentiment = state['aspect_sentiments'].get(aspect, {})
        
        report_lines.append(f"\n{aspect.upper()}")
        report_lines.append(f"Sentiment: {sentiment.get('sentiment', 'N/A')} "
                          f"(score: {sentiment.get('score', 0):.2f}, "
                          f"confidence: {sentiment.get('confidence', 0):.2f})")
        report_lines.append(f"Summary: {summary}")
        
        # Show extracted content
        content = state['aspect_extractions'].get(aspect, [])
        if content:
            report_lines.append(f"Supporting evidence ({len(content)} sentences):")
            for i, sentence in enumerate(content[:2], 1):
                report_lines.append(f"  {i}. {sentence[:100]}...")
    
    report_lines.extend([
        "",
        "",
        "ASPECT DETECTION:",
        "-" * 40,
        f"Detected Implicit Aspects: {', '.join(state['aspect_detection']['detected_aspects'])}",
        f"Highly Relevant Aspects: {', '.join(state['aspect_detection']['highly_relevant'])}",
        "",
        "Aspect Relevance Scores:"
    ])
    
    for aspect, score in state['aspect_detection']['target_aspect_scores'].items():
        report_lines.append(f"  {aspect}: {score:.3f}")
    
    report_lines.extend([
        "",
        "",
        "SENTIMENT DISTRIBUTION:",
        "-" * 40,
        f"Positive: {state['analytics']['sentiment_stats']['positive']} aspects",
        f"Negative: {state['analytics']['sentiment_stats']['negative']} aspects",
        f"Neutral: {state['analytics']['sentiment_stats']['neutral']} aspects",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Aspect Coverage: {state['analytics']['quality_stats']['coverage']:.1%}",
        f"Average Content per Aspect: {state['analytics']['quality_stats']['avg_content_per_aspect']:.1f} sentences",
        f"Aspects Analyzed: {state['analytics']['aspect_stats']['aspects_found']} / {state['analytics']['aspect_stats']['total_aspects']}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Analyzed {len(state['target_aspects'])} distinct aspects",
        f"✓ Found content for {state['analytics']['aspect_stats']['aspects_found']} aspects",
        f"✓ Overall sentiment distribution: {state['analytics']['sentiment_stats']['positive']} positive, "
        f"{state['analytics']['sentiment_stats']['negative']} negative, {state['analytics']['sentiment_stats']['neutral']} neutral",
        f"✓ Achieved {state['analytics']['quality_stats']['coverage']:.0%} aspect coverage",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Expand aspect detection with domain-specific taxonomies",
        "• Implement aspect relationship modeling",
        "• Add comparative aspect analysis across documents",
        "• Enable user-defined custom aspects",
        "• Implement aspect importance weighting",
        "• Add temporal aspect tracking for evolving content",
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
def create_aspect_based_summarization_graph():
    """Create the aspect-based summarization workflow graph"""
    
    workflow = StateGraph(AspectBasedSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_aspect_document_agent)
    workflow.add_node("detect_aspects", detect_aspects_agent)
    workflow.add_node("extract_content", extract_aspect_content_agent)
    workflow.add_node("summarize_aspects", summarize_aspects_agent)
    workflow.add_node("analyze_sentiments", analyze_sentiments_agent)
    workflow.add_node("integrate_aspects", integrate_aspects_agent)
    workflow.add_node("evaluate_coverage", evaluate_aspect_coverage_agent)
    workflow.add_node("analyze_results", analyze_aspect_summarization_agent)
    workflow.add_node("generate_report", generate_aspect_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "detect_aspects")
    workflow.add_edge("detect_aspects", "extract_content")
    workflow.add_edge("extract_content", "summarize_aspects")
    workflow.add_edge("summarize_aspects", "analyze_sentiments")
    workflow.add_edge("analyze_sentiments", "integrate_aspects")
    workflow.add_edge("integrate_aspects", "evaluate_coverage")
    workflow.add_edge("evaluate_coverage", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the aspect-based summarization graph
    app = create_aspect_based_summarization_graph()
    
    # Initialize state
    initial_state: AspectBasedSummarizationState = {
        'messages': [],
        'source_document': '',
        'target_aspects': [],
        'aspect_detection': {},
        'aspect_extractions': {},
        'aspect_summaries': {},
        'aspect_sentiments': {},
        'integrated_summary': '',
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("ASPECT-BASED SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nAspect-based summarization pattern execution complete! ✓")

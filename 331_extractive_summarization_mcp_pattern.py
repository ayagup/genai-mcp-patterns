"""
Extractive Summarization MCP Pattern

This pattern demonstrates extractive summarization in an agentic MCP system.
The system identifies and extracts the most important sentences/passages
from source documents to create summaries.

Use cases:
- Document summarization
- News article summaries
- Research paper abstracts
- Meeting notes extraction
- Content highlights
"""

from typing import TypedDict, Annotated, List, Dict, Any, Tuple
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from collections import Counter
import re
import math


# Define the state for extractive summarization
class ExtractiveSummarizationState(TypedDict):
    """State for tracking extractive summarization process"""
    messages: Annotated[List[str], add]
    source_document: Dict[str, Any]
    sentences: List[Dict[str, Any]]
    sentence_scores: List[Dict[str, Any]]
    selected_sentences: List[Dict[str, Any]]
    summary: str
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class TextPreprocessor:
    """Preprocess text for extractive summarization"""
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence tokenization
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_words(self, text: str) -> List[str]:
        """Split text into words"""
        # Simple word tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def remove_stopwords(self, words: List[str]) -> List[str]:
        """Remove common stopwords"""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'it', 'its', 'they', 'their', 'them'
        }
        return [w for w in words if w not in stopwords]


class SentenceScorer:
    """Score sentences for importance"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def calculate_tf_scores(self, sentences: List[str]) -> Dict[str, float]:
        """Calculate term frequency scores"""
        # Combine all sentences
        all_words = []
        for sentence in sentences:
            words = self.preprocessor.tokenize_words(sentence)
            words = self.preprocessor.remove_stopwords(words)
            all_words.extend(words)
        
        # Calculate TF
        word_freq = Counter(all_words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        tf_scores = {word: freq / max_freq for word, freq in word_freq.items()}
        return tf_scores
    
    def score_by_position(self, index: int, total: int) -> float:
        """Score based on position in document"""
        # First and last sentences often more important
        if index == 0:
            return 1.0
        elif index == total - 1:
            return 0.8
        elif index < 3:
            return 0.7
        else:
            return 0.5
    
    def score_by_length(self, sentence: str) -> float:
        """Score based on sentence length"""
        words = len(self.preprocessor.tokenize_words(sentence))
        
        # Prefer medium-length sentences
        if 10 <= words <= 30:
            return 1.0
        elif 5 <= words < 10 or 30 < words <= 40:
            return 0.7
        else:
            return 0.3
    
    def score_by_tf(self, sentence: str, tf_scores: Dict[str, float]) -> float:
        """Score based on term frequency"""
        words = self.preprocessor.tokenize_words(sentence)
        words = self.preprocessor.remove_stopwords(words)
        
        if not words:
            return 0.0
        
        total_score = sum(tf_scores.get(word, 0) for word in words)
        return total_score / len(words)
    
    def score_by_title_similarity(self, sentence: str, title: str) -> float:
        """Score based on similarity to title"""
        if not title:
            return 0.0
        
        sentence_words = set(self.preprocessor.tokenize_words(sentence))
        sentence_words = set(self.preprocessor.remove_stopwords(list(sentence_words)))
        
        title_words = set(self.preprocessor.tokenize_words(title))
        title_words = set(self.preprocessor.remove_stopwords(list(title_words)))
        
        if not sentence_words or not title_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(sentence_words & title_words)
        union = len(sentence_words | title_words)
        
        return intersection / union if union > 0 else 0.0
    
    def score_by_numeric_data(self, sentence: str) -> float:
        """Score sentences containing numeric data"""
        # Count numbers in sentence
        numbers = re.findall(r'\d+', sentence)
        return min(len(numbers) * 0.2, 1.0)
    
    def score_by_proper_nouns(self, sentence: str) -> float:
        """Score sentences with proper nouns (capitalized words)"""
        # Simple heuristic: count capitalized words (not at start)
        words = sentence.split()[1:]  # Skip first word
        capitalized = sum(1 for w in words if w and w[0].isupper())
        return min(capitalized * 0.15, 1.0)
    
    def calculate_composite_score(self, sentence: str, index: int, total: int,
                                  tf_scores: Dict[str, float], title: str = "") -> float:
        """Calculate composite score from multiple factors"""
        scores = {
            'position': self.score_by_position(index, total),
            'length': self.score_by_length(sentence),
            'tf': self.score_by_tf(sentence, tf_scores),
            'title_similarity': self.score_by_title_similarity(sentence, title),
            'numeric': self.score_by_numeric_data(sentence),
            'proper_nouns': self.score_by_proper_nouns(sentence)
        }
        
        # Weighted combination
        weights = {
            'position': 0.15,
            'length': 0.10,
            'tf': 0.35,
            'title_similarity': 0.20,
            'numeric': 0.10,
            'proper_nouns': 0.10
        }
        
        composite = sum(scores[k] * weights[k] for k in scores)
        return composite


class SentenceSelector:
    """Select sentences for summary"""
    
    def select_top_k(self, scored_sentences: List[Dict[str, Any]], 
                     k: int = 5) -> List[Dict[str, Any]]:
        """Select top k sentences by score"""
        # Sort by score
        sorted_sentences = sorted(scored_sentences, 
                                 key=lambda x: x['total_score'], 
                                 reverse=True)
        
        return sorted_sentences[:k]
    
    def select_by_threshold(self, scored_sentences: List[Dict[str, Any]], 
                           threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Select sentences above score threshold"""
        return [s for s in scored_sentences if s['total_score'] >= threshold]
    
    def select_diverse(self, scored_sentences: List[Dict[str, Any]], 
                      k: int = 5, diversity_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Select diverse sentences to avoid redundancy"""
        selected = []
        candidates = sorted(scored_sentences, 
                          key=lambda x: x['total_score'], 
                          reverse=True)
        
        for candidate in candidates:
            if len(selected) >= k:
                break
            
            # Check similarity with already selected
            if not selected:
                selected.append(candidate)
            else:
                # Simple diversity check: word overlap
                candidate_words = set(candidate['text'].lower().split())
                max_overlap = 0
                
                for sel in selected:
                    sel_words = set(sel['text'].lower().split())
                    overlap = len(candidate_words & sel_words) / len(candidate_words | sel_words)
                    max_overlap = max(max_overlap, overlap)
                
                # Add if diverse enough
                if max_overlap < diversity_weight:
                    selected.append(candidate)
        
        return selected


class SummaryAssembler:
    """Assemble selected sentences into coherent summary"""
    
    def assemble_by_position(self, sentences: List[Dict[str, Any]]) -> str:
        """Assemble sentences in original document order"""
        # Sort by original position
        sorted_sentences = sorted(sentences, key=lambda x: x['index'])
        return ' '.join(s['text'] for s in sorted_sentences)
    
    def assemble_by_score(self, sentences: List[Dict[str, Any]]) -> str:
        """Assemble sentences by importance score"""
        sorted_sentences = sorted(sentences, 
                                 key=lambda x: x['total_score'], 
                                 reverse=True)
        return ' '.join(s['text'] for s in sorted_sentences)
    
    def add_transitions(self, summary: str) -> str:
        """Add simple transition words between sentences"""
        sentences = summary.split('. ')
        
        if len(sentences) <= 1:
            return summary
        
        transitions = ['Additionally,', 'Furthermore,', 'Moreover,', 'Also,']
        result = [sentences[0]]
        
        for i, sentence in enumerate(sentences[1:], 1):
            if i < len(transitions) and sentence:
                result.append(f"{transitions[i-1]} {sentence}")
            else:
                result.append(sentence)
        
        return '. '.join(result)


class QualityEvaluator:
    """Evaluate summary quality"""
    
    def calculate_compression_ratio(self, original: str, summary: str) -> float:
        """Calculate compression ratio"""
        original_words = len(original.split())
        summary_words = len(summary.split())
        
        return summary_words / original_words if original_words > 0 else 0
    
    def calculate_coverage(self, original: str, summary: str) -> float:
        """Calculate topic coverage"""
        preprocessor = TextPreprocessor()
        
        orig_words = set(preprocessor.remove_stopwords(
            preprocessor.tokenize_words(original)))
        summ_words = set(preprocessor.remove_stopwords(
            preprocessor.tokenize_words(summary)))
        
        if not orig_words:
            return 0
        
        return len(orig_words & summ_words) / len(orig_words)
    
    def calculate_coherence(self, summary: str) -> float:
        """Estimate summary coherence (simple heuristic)"""
        sentences = summary.split('. ')
        
        if len(sentences) <= 1:
            return 1.0
        
        # Check for transition words
        transition_words = {'additionally', 'furthermore', 'moreover', 'also', 
                          'however', 'therefore', 'thus', 'consequently'}
        
        has_transitions = sum(1 for s in sentences 
                            if any(tw in s.lower() for tw in transition_words))
        
        return min(has_transitions / (len(sentences) - 1), 1.0)


# Agent functions
def load_source_document_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Load source document for summarization"""
    
    # Sample document
    document = {
        'title': 'Artificial Intelligence in Healthcare',
        'content': """
        Artificial intelligence is transforming the healthcare industry in unprecedented ways. 
        Machine learning algorithms can now analyze medical images with accuracy comparable to human radiologists. 
        AI-powered diagnostic tools are helping doctors identify diseases earlier and more accurately than ever before. 
        Natural language processing enables automated analysis of medical records and research papers. 
        Predictive analytics are being used to forecast patient outcomes and optimize treatment plans. 
        Healthcare providers are implementing AI chatbots to improve patient engagement and provide 24/7 support. 
        Deep learning models can detect patterns in genomic data that humans might miss. 
        The integration of AI in healthcare is expected to save billions in healthcare costs annually. 
        However, challenges remain including data privacy concerns and the need for regulatory frameworks. 
        Despite these challenges, the potential benefits of AI in healthcare are enormous and continue to grow.
        """.strip(),
        'metadata': {
            'author': 'Healthcare Technology Review',
            'date': '2024-11-15',
            'category': 'Technology'
        }
    }
    
    return {
        **state,
        'source_document': document,
        'messages': state['messages'] + [f'Loaded document: "{document["title"]}" ({len(document["content"].split())} words)']
    }


def preprocess_text_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Preprocess text into sentences"""
    
    preprocessor = TextPreprocessor()
    content = state['source_document']['content']
    
    # Tokenize into sentences
    sentence_list = preprocessor.tokenize_sentences(content)
    
    # Create sentence objects
    sentences = []
    for i, text in enumerate(sentence_list):
        sentences.append({
            'index': i,
            'text': text,
            'word_count': len(preprocessor.tokenize_words(text))
        })
    
    return {
        **state,
        'sentences': sentences,
        'messages': state['messages'] + [f'Preprocessed into {len(sentences)} sentences']
    }


def score_sentences_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Score sentences for importance"""
    
    scorer = SentenceScorer()
    
    # Extract sentence texts
    sentence_texts = [s['text'] for s in state['sentences']]
    title = state['source_document'].get('title', '')
    
    # Calculate TF scores
    tf_scores = scorer.calculate_tf_scores(sentence_texts)
    
    # Score each sentence
    scored_sentences = []
    for sentence in state['sentences']:
        scores = {
            'position': scorer.score_by_position(sentence['index'], len(state['sentences'])),
            'length': scorer.score_by_length(sentence['text']),
            'tf': scorer.score_by_tf(sentence['text'], tf_scores),
            'title_similarity': scorer.score_by_title_similarity(sentence['text'], title),
            'numeric': scorer.score_by_numeric_data(sentence['text']),
            'proper_nouns': scorer.score_by_proper_nouns(sentence['text'])
        }
        
        total_score = scorer.calculate_composite_score(
            sentence['text'],
            sentence['index'],
            len(state['sentences']),
            tf_scores,
            title
        )
        
        scored_sentences.append({
            **sentence,
            'scores': scores,
            'total_score': total_score
        })
    
    return {
        **state,
        'sentence_scores': scored_sentences,
        'messages': state['messages'] + [f'Scored {len(scored_sentences)} sentences']
    }


def select_sentences_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Select most important sentences"""
    
    selector = SentenceSelector()
    
    # Select diverse sentences
    selected = selector.select_diverse(
        state['sentence_scores'],
        k=5,
        diversity_weight=0.4
    )
    
    return {
        **state,
        'selected_sentences': selected,
        'messages': state['messages'] + [f'Selected {len(selected)} sentences for summary']
    }


def assemble_summary_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Assemble selected sentences into summary"""
    
    assembler = SummaryAssembler()
    
    # Assemble in document order
    summary = assembler.assemble_by_position(state['selected_sentences'])
    
    return {
        **state,
        'summary': summary,
        'messages': state['messages'] + [f'Assembled summary ({len(summary.split())} words)']
    }


def evaluate_quality_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Evaluate summary quality"""
    
    evaluator = QualityEvaluator()
    
    original = state['source_document']['content']
    summary = state['summary']
    
    metrics = {
        'compression_ratio': evaluator.calculate_compression_ratio(original, summary),
        'coverage': evaluator.calculate_coverage(original, summary),
        'coherence': evaluator.calculate_coherence(summary),
        'original_sentences': len(state['sentences']),
        'summary_sentences': len(state['selected_sentences']),
        'original_words': len(original.split()),
        'summary_words': len(summary.split())
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Evaluated quality (compression: {metrics["compression_ratio"]:.1%}, coverage: {metrics["coverage"]:.1%})']
    }


def analyze_extractive_summarization_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Analyze extractive summarization results"""
    
    # Analyze score distribution
    scores = [s['total_score'] for s in state['sentence_scores']]
    avg_score = sum(scores) / len(scores) if scores else 0
    selected_avg = sum(s['total_score'] for s in state['selected_sentences']) / len(state['selected_sentences']) if state['selected_sentences'] else 0
    
    analytics = {
        'sentence_stats': {
            'total_sentences': len(state['sentences']),
            'selected_sentences': len(state['selected_sentences']),
            'selection_rate': len(state['selected_sentences']) / len(state['sentences']) if state['sentences'] else 0
        },
        'score_stats': {
            'avg_all_scores': avg_score,
            'avg_selected_scores': selected_avg,
            'min_selected_score': min(s['total_score'] for s in state['selected_sentences']) if state['selected_sentences'] else 0,
            'max_selected_score': max(s['total_score'] for s in state['selected_sentences']) if state['selected_sentences'] else 0
        },
        'quality_stats': state['quality_metrics']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed extractive summarization results']
    }


def generate_extractive_summarization_report_agent(state: ExtractiveSummarizationState) -> ExtractiveSummarizationState:
    """Generate comprehensive extractive summarization report"""
    
    report_lines = [
        "=" * 80,
        "EXTRACTIVE SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "SOURCE DOCUMENT:",
        "-" * 40,
        f"Title: {state['source_document']['title']}",
        f"Original Length: {state['analytics']['quality_stats']['original_words']} words",
        f"Original Sentences: {state['analytics']['sentence_stats']['total_sentences']}",
        "",
        "EXTRACTIVE SUMMARY:",
        "-" * 40,
        state['summary'],
        "",
        "",
        "SUMMARY STATISTICS:",
        "-" * 40,
        f"Summary Length: {state['analytics']['quality_stats']['summary_words']} words",
        f"Summary Sentences: {state['analytics']['sentence_stats']['selected_sentences']}",
        f"Compression Ratio: {state['analytics']['quality_stats']['compression_ratio']:.1%}",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Topic Coverage: {state['analytics']['quality_stats']['coverage']:.1%}",
        f"Coherence Score: {state['analytics']['quality_stats']['coherence']:.2f}",
        f"Selection Rate: {state['analytics']['sentence_stats']['selection_rate']:.1%}",
        "",
        "SELECTED SENTENCES (by score):",
        "-" * 40
    ]
    
    for sent in sorted(state['selected_sentences'], key=lambda x: x['total_score'], reverse=True):
        report_lines.append(f"\n[Score: {sent['total_score']:.3f}] {sent['text'][:100]}...")
        report_lines.append(f"  Position: #{sent['index']+1} | TF: {sent['scores']['tf']:.2f} | Title Sim: {sent['scores']['title_similarity']:.2f}")
    
    report_lines.extend([
        "",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Reduced document to {state['analytics']['quality_stats']['compression_ratio']:.0%} of original length",
        f"✓ Selected {len(state['selected_sentences'])} most important sentences",
        f"✓ Achieved {state['analytics']['quality_stats']['coverage']:.0%} topic coverage",
        f"✓ Average selected sentence score: {state['analytics']['score_stats']['avg_selected_scores']:.3f}",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Adjust compression ratio based on use case requirements",
        "• Consider adding sentence reordering for better flow",
        "• Implement coreference resolution for clarity",
        "• Add sentence fusion to reduce redundancy",
        "• Use domain-specific scoring weights",
        "• Implement multi-document extractive summarization",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive extractive summarization report']
    }


# Create the graph
def create_extractive_summarization_graph():
    """Create the extractive summarization workflow graph"""
    
    workflow = StateGraph(ExtractiveSummarizationState)
    
    # Add nodes
    workflow.add_node("load_document", load_source_document_agent)
    workflow.add_node("preprocess", preprocess_text_agent)
    workflow.add_node("score_sentences", score_sentences_agent)
    workflow.add_node("select_sentences", select_sentences_agent)
    workflow.add_node("assemble_summary", assemble_summary_agent)
    workflow.add_node("evaluate_quality", evaluate_quality_agent)
    workflow.add_node("analyze_results", analyze_extractive_summarization_agent)
    workflow.add_node("generate_report", generate_extractive_summarization_report_agent)
    
    # Add edges
    workflow.add_edge(START, "load_document")
    workflow.add_edge("load_document", "preprocess")
    workflow.add_edge("preprocess", "score_sentences")
    workflow.add_edge("score_sentences", "select_sentences")
    workflow.add_edge("select_sentences", "assemble_summary")
    workflow.add_edge("assemble_summary", "evaluate_quality")
    workflow.add_edge("evaluate_quality", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the extractive summarization graph
    app = create_extractive_summarization_graph()
    
    # Initialize state
    initial_state: ExtractiveSummarizationState = {
        'messages': [],
        'source_document': {},
        'sentences': [],
        'sentence_scores': [],
        'selected_sentences': [],
        'summary': '',
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("EXTRACTIVE SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nExtractive summarization pattern execution complete! ✓")

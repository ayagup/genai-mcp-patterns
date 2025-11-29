"""
Incremental Summarization MCP Pattern

This pattern demonstrates incremental summarization in an agentic MCP system.
The system updates summaries in real-time as new content becomes available.

Use cases:
- Live event coverage
- Streaming data summarization
- Real-time news aggregation
- Continuous document updates
- Dynamic content synthesis
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import deque
from datetime import datetime


# Define the state for incremental summarization
class IncrementalSummarizationState(TypedDict):
    """State for tracking incremental summarization process"""
    messages: Annotated[List[str], add]
    content_stream: List[Dict[str, Any]]
    current_summary: str
    summary_history: List[Dict[str, Any]]
    update_triggers: List[str]
    content_buffer: List[str]
    incremental_updates: List[Dict[str, Any]]
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class ContentStreamProcessor:
    """Process streaming content"""
    
    def process_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming content chunk"""
        
        return {
            'chunk_id': chunk['id'],
            'content': chunk['content'],
            'timestamp': chunk.get('timestamp', datetime.now().isoformat()),
            'word_count': len(chunk['content'].split()),
            'importance': chunk.get('importance', 0.5)
        }
    
    def assess_significance(self, chunk: Dict[str, Any], 
                           current_summary: str) -> float:
        """Assess significance of new chunk"""
        
        # Check novelty
        chunk_words = set(re.findall(r'\b\w+\b', chunk['content'].lower()))
        summary_words = set(re.findall(r'\b\w+\b', current_summary.lower())) if current_summary else set()
        
        if not chunk_words:
            return 0.0
        
        new_words = chunk_words - summary_words
        novelty = len(new_words) / len(chunk_words) if chunk_words else 0.0
        
        # Combine with importance
        significance = novelty * 0.6 + chunk.get('importance', 0.5) * 0.4
        
        return significance


class UpdateTriggerManager:
    """Manage summary update triggers"""
    
    def __init__(self):
        self.triggers = {
            'word_threshold': 100,  # Update every 100 words
            'time_threshold': 60,   # Update every 60 seconds
            'significance_threshold': 0.7,  # High significance content
            'buffer_size': 5        # Update every 5 chunks
        }
    
    def check_word_threshold(self, buffer: List[str]) -> bool:
        """Check if word threshold reached"""
        total_words = sum(len(chunk.split()) for chunk in buffer)
        return total_words >= self.triggers['word_threshold']
    
    def check_buffer_size(self, buffer: List[str]) -> bool:
        """Check if buffer size threshold reached"""
        return len(buffer) >= self.triggers['buffer_size']
    
    def check_significance(self, significance: float) -> bool:
        """Check if significance threshold reached"""
        return significance >= self.triggers['significance_threshold']
    
    def should_update(self, buffer: List[str], 
                     latest_significance: float) -> tuple[bool, List[str]]:
        """Determine if summary should be updated"""
        
        reasons = []
        
        if self.check_buffer_size(buffer):
            reasons.append('buffer_size')
        
        if self.check_word_threshold(buffer):
            reasons.append('word_threshold')
        
        if self.check_significance(latest_significance):
            reasons.append('high_significance')
        
        should_update = len(reasons) > 0
        return should_update, reasons


class IncrementalSummarizer:
    """Generate incremental summaries"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def create_initial_summary(self, content: str) -> str:
        """Create initial summary"""
        
        system_prompt = """You are an expert at creating concise summaries. 
        Create a brief initial summary that can be expanded later."""
        
        user_prompt = f"""
        Create a brief summary (2-3 sentences) of this content:
        
        {content}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def update_summary(self, current_summary: str, 
                      new_content: str, update_strategy: str = 'merge') -> str:
        """Update existing summary with new content"""
        
        if update_strategy == 'merge':
            system_prompt = """You are an expert at updating summaries. 
            Merge the new information into the existing summary smoothly."""
            
            user_prompt = f"""
            Current summary:
            {current_summary}
            
            New content to incorporate:
            {new_content}
            
            Update the summary to include the new information. Maintain brevity 
            while ensuring completeness. Remove redundancies.
            """
        elif update_strategy == 'append':
            system_prompt = """You are an expert at extending summaries. 
            Append new information to the existing summary."""
            
            user_prompt = f"""
            Current summary:
            {current_summary}
            
            New content:
            {new_content}
            
            Extend the summary with the new information. Add transitional phrases.
            """
        else:  # regenerate
            system_prompt = """You are an expert at comprehensive summarization. 
            Create a fresh summary considering all information."""
            
            user_prompt = f"""
            Previous summary:
            {current_summary}
            
            Additional content:
            {new_content}
            
            Create a new comprehensive summary incorporating all information.
            """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class SummaryEvolutionTracker:
    """Track how summary evolves"""
    
    def calculate_change_magnitude(self, old_summary: str, new_summary: str) -> float:
        """Calculate magnitude of change"""
        
        old_words = set(re.findall(r'\b\w+\b', old_summary.lower()))
        new_words = set(re.findall(r'\b\w+\b', new_summary.lower()))
        
        if not old_words:
            return 1.0
        
        added = new_words - old_words
        removed = old_words - new_words
        
        change_ratio = (len(added) + len(removed)) / (len(old_words) + len(new_words))
        return min(1.0, change_ratio)
    
    def identify_additions(self, old_summary: str, new_summary: str) -> List[str]:
        """Identify what was added"""
        
        old_sentences = set(re.split(r'[.!?]+', old_summary))
        new_sentences = set(re.split(r'[.!?]+', new_summary))
        
        additions = [s.strip() for s in (new_sentences - old_sentences) if s.strip()]
        return additions
    
    def track_evolution(self, summary_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track overall evolution"""
        
        if len(summary_history) < 2:
            return {'evolution_score': 0.0, 'stability': 1.0}
        
        changes = []
        for i in range(1, len(summary_history)):
            old = summary_history[i-1]['summary']
            new = summary_history[i]['summary']
            change = self.calculate_change_magnitude(old, new)
            changes.append(change)
        
        avg_change = sum(changes) / len(changes)
        stability = 1.0 - avg_change
        
        return {
            'evolution_score': avg_change,
            'stability': stability,
            'total_updates': len(summary_history) - 1,
            'avg_change_per_update': avg_change
        }


class QualityMonitor:
    """Monitor summary quality over time"""
    
    def assess_completeness(self, summary: str, all_content: List[str]) -> float:
        """Assess completeness"""
        
        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        all_words = set()
        for content in all_content:
            all_words.update(re.findall(r'\b\w+\b', content.lower()))
        
        if not all_words:
            return 0.0
        
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        summary_keywords = summary_words - stopwords
        all_keywords = all_words - stopwords
        
        if not all_keywords:
            return 0.0
        
        coverage = len(summary_keywords & all_keywords) / len(all_keywords)
        return coverage
    
    def assess_conciseness(self, summary: str, content_count: int) -> float:
        """Assess conciseness"""
        
        summary_words = len(summary.split())
        ideal_words = 50 + content_count * 10  # Ideal grows with content
        
        if summary_words <= ideal_words:
            return 1.0
        else:
            # Penalty for exceeding ideal length
            excess_ratio = (summary_words - ideal_words) / ideal_words
            return max(0.0, 1.0 - excess_ratio * 0.5)


# Agent functions
def initialize_stream_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Initialize content stream"""
    
    # Simulate streaming content chunks
    content_stream = [
        {
            'id': 1,
            'content': 'Breaking: Major technology company announces new AI research breakthrough.',
            'importance': 0.9,
            'timestamp': '2025-11-29T10:00:00'
        },
        {
            'id': 2,
            'content': 'The breakthrough involves a new neural architecture that significantly improves efficiency.',
            'importance': 0.8,
            'timestamp': '2025-11-29T10:02:00'
        },
        {
            'id': 3,
            'content': 'Researchers claim the new model achieves 10x better performance with 50% less computational resources.',
            'importance': 0.9,
            'timestamp': '2025-11-29T10:05:00'
        },
        {
            'id': 4,
            'content': 'The architecture introduces a novel attention mechanism called "Sparse Dynamic Attention".',
            'importance': 0.7,
            'timestamp': '2025-11-29T10:08:00'
        },
        {
            'id': 5,
            'content': 'Initial tests show remarkable results on language understanding and generation tasks.',
            'importance': 0.6,
            'timestamp': '2025-11-29T10:10:00'
        },
        {
            'id': 6,
            'content': 'The company plans to open-source the architecture within six months.',
            'importance': 0.8,
            'timestamp': '2025-11-29T10:12:00'
        },
        {
            'id': 7,
            'content': 'Industry experts are calling this a potential paradigm shift in AI development.',
            'importance': 0.7,
            'timestamp': '2025-11-29T10:15:00'
        }
    ]
    
    return {
        **state,
        'content_stream': content_stream,
        'current_summary': '',
        'content_buffer': [],
        'summary_history': [],
        'incremental_updates': [],
        'messages': state['messages'] + [f'Initialized content stream ({len(content_stream)} chunks)']
    }


def process_content_chunk_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Process incoming content chunk"""
    
    processor = ContentStreamProcessor()
    
    # Get next chunk (simulate streaming)
    processed_count = len(state.get('incremental_updates', []))
    
    if processed_count >= len(state['content_stream']):
        return {
            **state,
            'messages': state['messages'] + ['All chunks processed']
        }
    
    chunk = state['content_stream'][processed_count]
    processed_chunk = processor.process_chunk(chunk)
    
    # Assess significance
    significance = processor.assess_significance(processed_chunk, state['current_summary'])
    processed_chunk['significance'] = significance
    
    # Add to buffer
    new_buffer = state['content_buffer'] + [processed_chunk['content']]
    
    return {
        **state,
        'content_buffer': new_buffer,
        'messages': state['messages'] + [f'Processed chunk {chunk["id"]} (significance: {significance:.2f})']
    }


def check_update_trigger_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Check if summary update should be triggered"""
    
    trigger_manager = UpdateTriggerManager()
    
    # Get latest significance
    processed_count = len(state.get('incremental_updates', []))
    latest_significance = 0.5
    
    if processed_count < len(state['content_stream']):
        chunk = state['content_stream'][processed_count]
        processor = ContentStreamProcessor()
        processed = processor.process_chunk(chunk)
        latest_significance = processor.assess_significance(processed, state['current_summary'])
    
    should_update, reasons = trigger_manager.should_update(
        state['content_buffer'],
        latest_significance
    )
    
    return {
        **state,
        'update_triggers': reasons,
        'messages': state['messages'] + [f'Update check: {should_update} (reasons: {", ".join(reasons) if reasons else "none"})']
    }


def update_summary_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Update summary incrementally"""
    
    if not state['update_triggers']:
        return state
    
    summarizer = IncrementalSummarizer()
    new_content = '\n'.join(state['content_buffer'])
    
    if not state['current_summary']:
        # Create initial summary
        updated_summary = summarizer.create_initial_summary(new_content)
        strategy = 'initial'
    else:
        # Determine strategy based on trigger
        if 'high_significance' in state['update_triggers']:
            strategy = 'merge'
        elif len(state['summary_history']) % 3 == 0:
            strategy = 'regenerate'
        else:
            strategy = 'append'
        
        updated_summary = summarizer.update_summary(
            state['current_summary'],
            new_content,
            strategy
        )
    
    # Track this update
    tracker = SummaryEvolutionTracker()
    change_magnitude = tracker.calculate_change_magnitude(
        state['current_summary'],
        updated_summary
    ) if state['current_summary'] else 1.0
    
    update_record = {
        'update_number': len(state['summary_history']) + 1,
        'summary': updated_summary,
        'strategy': strategy,
        'triggers': state['update_triggers'],
        'chunks_processed': len(state['incremental_updates']) + 1,
        'change_magnitude': change_magnitude,
        'timestamp': datetime.now().isoformat()
    }
    
    new_history = state['summary_history'] + [update_record]
    
    # Record incremental update
    increment_record = {
        'chunk_id': len(state['incremental_updates']) + 1,
        'content_added': new_content,
        'summary_version': len(new_history)
    }
    
    new_increments = state.get('incremental_updates', []) + [increment_record]
    
    return {
        **state,
        'current_summary': updated_summary,
        'summary_history': new_history,
        'incremental_updates': new_increments,
        'content_buffer': [],  # Clear buffer
        'update_triggers': [],
        'messages': state['messages'] + [f'Updated summary (strategy: {strategy}, change: {change_magnitude:.2f})']
    }


def evaluate_quality_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Evaluate summary quality"""
    
    monitor = QualityMonitor()
    tracker = SummaryEvolutionTracker()
    
    # Collect all content
    all_content = [chunk['content'] for chunk in state['content_stream'][:len(state['incremental_updates'])]]
    
    completeness = monitor.assess_completeness(state['current_summary'], all_content)
    conciseness = monitor.assess_conciseness(state['current_summary'], len(all_content))
    evolution = tracker.track_evolution(state['summary_history'])
    
    metrics = {
        'completeness': completeness,
        'conciseness': conciseness,
        'evolution': evolution,
        'total_updates': len(state['summary_history']),
        'chunks_processed': len(state['incremental_updates']),
        'current_summary_words': len(state['current_summary'].split())
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Evaluated quality (completeness: {completeness:.1%}, conciseness: {conciseness:.1%})']
    }


def analyze_incremental_summarization_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Analyze incremental summarization results"""
    
    analytics = {
        'stream_stats': {
            'total_chunks': len(state['content_stream']),
            'chunks_processed': len(state['incremental_updates']),
            'total_words_processed': sum(len(chunk['content'].split()) for chunk in state['content_stream'][:len(state['incremental_updates'])])
        },
        'update_stats': {
            'total_updates': len(state['summary_history']),
            'update_strategies': [h['strategy'] for h in state['summary_history']],
            'avg_change_magnitude': sum(h['change_magnitude'] for h in state['summary_history']) / len(state['summary_history']) if state['summary_history'] else 0
        },
        'quality_stats': state['quality_metrics'],
        'evolution_metrics': state['quality_metrics']['evolution']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed incremental summarization results']
    }


def generate_incremental_report_agent(state: IncrementalSummarizationState) -> IncrementalSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "INCREMENTAL SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "FINAL SUMMARY:",
        "-" * 40,
        state['current_summary'],
        "",
        "",
        "SUMMARY EVOLUTION:",
        "-" * 40
    ]
    
    for i, history in enumerate(state['summary_history'], 1):
        report_lines.append(f"\nUpdate {i} (Strategy: {history['strategy']}, Change: {history['change_magnitude']:.2f}):")
        report_lines.append(f"Triggers: {', '.join(history['triggers'])}")
        report_lines.append(f"Summary: {history['summary'][:100]}...")
    
    report_lines.extend([
        "",
        "",
        "CONTENT STREAM PROCESSED:",
        "-" * 40
    ])
    
    for i, chunk in enumerate(state['content_stream'][:len(state['incremental_updates'])], 1):
        report_lines.append(f"{i}. [{chunk.get('timestamp', 'N/A')}] {chunk['content']}")
    
    report_lines.extend([
        "",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Completeness: {state['analytics']['quality_stats']['completeness']:.1%}",
        f"Conciseness: {state['analytics']['quality_stats']['conciseness']:.1%}",
        f"Stability: {state['analytics']['evolution_metrics']['stability']:.1%}",
        f"Average Change per Update: {state['analytics']['evolution_metrics']['avg_change_per_update']:.2f}",
        "",
        "PROCESSING STATISTICS:",
        "-" * 40,
        f"Total Chunks: {state['analytics']['stream_stats']['total_chunks']}",
        f"Chunks Processed: {state['analytics']['stream_stats']['chunks_processed']}",
        f"Total Words Processed: {state['analytics']['stream_stats']['total_words_processed']}",
        f"Summary Updates: {state['analytics']['update_stats']['total_updates']}",
        f"Current Summary Length: {state['quality_metrics']['current_summary_words']} words",
        "",
        "UPDATE STRATEGIES USED:",
        "-" * 40
    ])
    
    strategy_counts = {}
    for strategy in state['analytics']['update_stats']['update_strategies']:
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in strategy_counts.items():
        report_lines.append(f"  {strategy}: {count} times")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Processed {state['analytics']['stream_stats']['chunks_processed']} content chunks incrementally",
        f"✓ Performed {state['analytics']['update_stats']['total_updates']} summary updates",
        f"✓ Maintained {state['analytics']['evolution_metrics']['stability']:.0%} stability across updates",
        f"✓ Achieved {state['analytics']['quality_stats']['completeness']:.0%} content coverage",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Optimize update frequency based on content velocity",
        "• Implement adaptive trigger thresholds",
        "• Add summary compression for long streams",
        "• Enable summary rollback for quality issues",
        "• Implement parallel processing for high-velocity streams",
        "• Add user-configurable update preferences",
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
def create_incremental_summarization_graph():
    """Create the incremental summarization workflow graph"""
    
    workflow = StateGraph(IncrementalSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_stream_agent)
    workflow.add_node("process_chunk", process_content_chunk_agent)
    workflow.add_node("check_trigger", check_update_trigger_agent)
    workflow.add_node("update_summary", update_summary_agent)
    workflow.add_node("evaluate_quality", evaluate_quality_agent)
    workflow.add_node("analyze_results", analyze_incremental_summarization_agent)
    workflow.add_node("generate_report", generate_incremental_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "process_chunk")
    workflow.add_edge("process_chunk", "check_trigger")
    workflow.add_edge("check_trigger", "update_summary")
    
    # Conditional: continue processing or finish
    def should_continue_processing(state: IncrementalSummarizationState) -> str:
        chunks_processed = len(state.get('incremental_updates', []))
        total_chunks = len(state['content_stream'])
        
        if chunks_processed < total_chunks:
            return "process_chunk"
        else:
            return "evaluate_quality"
    
    workflow.add_conditional_edges(
        "update_summary",
        should_continue_processing,
        {
            "process_chunk": "process_chunk",
            "evaluate_quality": "evaluate_quality"
        }
    )
    
    workflow.add_edge("evaluate_quality", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the incremental summarization graph
    app = create_incremental_summarization_graph()
    
    # Initialize state
    initial_state: IncrementalSummarizationState = {
        'messages': [],
        'content_stream': [],
        'current_summary': '',
        'summary_history': [],
        'update_triggers': [],
        'content_buffer': [],
        'incremental_updates': [],
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("INCREMENTAL SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nIncremental summarization pattern execution complete! ✓")

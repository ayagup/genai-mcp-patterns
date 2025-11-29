"""
Timeline Summarization MCP Pattern

This pattern demonstrates timeline summarization in an agentic MCP system.
The system creates chronological summaries of events and developments.

Use cases:
- News timeline generation
- Project progress tracking
- Historical event summarization
- Development changelog summaries
- Event sequence analysis
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from datetime import datetime
from collections import defaultdict


# Define the state for timeline summarization
class TimelineSummarizationState(TypedDict):
    """State for tracking timeline summarization process"""
    messages: Annotated[List[str], add]
    source_documents: List[Dict[str, Any]]
    extracted_events: List[Dict[str, Any]]
    temporal_clusters: List[Dict[str, Any]]
    event_sequences: List[Dict[str, Any]]
    timeline_summary: str
    milestone_highlights: List[str]
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class TemporalExtractor:
    """Extract temporal information from text"""
    
    def extract_events(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract events with temporal markers"""
        
        content = document['content']
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Temporal markers
        temporal_patterns = [
            r'\b(\d{4})\b',  # Year
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})',  # Month Day
            r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',  # Day of week
            r'\b(today|yesterday|tomorrow|recently|currently|previously|later|earlier)\b',  # Relative time
            r'\b(in|on|at|during|after|before)\s+\d{4}\b',  # Temporal prepositions
        ]
        
        events = []
        for i, sentence in enumerate(sentences):
            # Check for temporal markers
            has_temporal = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in temporal_patterns)
            
            if has_temporal:
                # Extract year if present
                year_match = re.search(r'\b(19\d{2}|20\d{2})\b', sentence)
                year = int(year_match.group(1)) if year_match else None
                
                # Extract month if present
                month_match = re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', sentence, re.IGNORECASE)
                month = month_match.group(1) if month_match else None
                
                events.append({
                    'text': sentence.strip(),
                    'year': year,
                    'month': month,
                    'position': i,
                    'source': document.get('source', 'Unknown'),
                    'timestamp': document.get('timestamp', None)
                })
        
        return events
    
    def normalize_temporal_references(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize relative temporal references"""
        
        normalized = []
        for event in events:
            normalized_event = event.copy()
            
            # Convert relative references
            text_lower = event['text'].lower()
            if 'recently' in text_lower or 'currently' in text_lower:
                normalized_event['temporal_type'] = 'recent'
            elif 'previously' in text_lower or 'earlier' in text_lower:
                normalized_event['temporal_type'] = 'past'
            elif 'later' in text_lower or 'future' in text_lower:
                normalized_event['temporal_type'] = 'future'
            else:
                normalized_event['temporal_type'] = 'specific'
            
            normalized.append(normalized_event)
        
        return normalized


class EventSequencer:
    """Sequence events chronologically"""
    
    def sort_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort events chronologically"""
        
        def event_sort_key(event):
            # Primary sort by year
            year = event.get('year', 9999)
            
            # Secondary sort by month
            month_order = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4,
                'May': 5, 'June': 6, 'July': 7, 'August': 8,
                'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            month = month_order.get(event.get('month', ''), 13)
            
            # Tertiary sort by position in document
            position = event.get('position', 999)
            
            return (year, month, position)
        
        return sorted(events, key=event_sort_key)
    
    def identify_sequences(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify event sequences"""
        
        sequences = []
        current_sequence = []
        last_year = None
        
        for event in events:
            year = event.get('year')
            
            if year is None:
                continue
            
            if last_year is None or abs(year - last_year) <= 2:
                # Continue sequence
                current_sequence.append(event)
                last_year = year
            else:
                # Start new sequence
                if current_sequence:
                    sequences.append({
                        'events': current_sequence,
                        'start_year': current_sequence[0].get('year'),
                        'end_year': current_sequence[-1].get('year'),
                        'span': len(current_sequence)
                    })
                current_sequence = [event]
                last_year = year
        
        # Add final sequence
        if current_sequence:
            sequences.append({
                'events': current_sequence,
                'start_year': current_sequence[0].get('year'),
                'end_year': current_sequence[-1].get('year'),
                'span': len(current_sequence)
            })
        
        return sequences


class TemporalClusterer:
    """Cluster events by time period"""
    
    def cluster_by_period(self, events: List[Dict[str, Any]], 
                         period_type: str = 'year') -> List[Dict[str, Any]]:
        """Cluster events by time period"""
        
        clusters = defaultdict(list)
        
        for event in events:
            if period_type == 'year':
                key = event.get('year', 'unknown')
            elif period_type == 'month':
                year = event.get('year', 'unknown')
                month = event.get('month', 'unknown')
                key = f"{year}-{month}"
            else:
                key = event.get('temporal_type', 'unknown')
            
            clusters[key].append(event)
        
        # Convert to list of cluster objects
        cluster_list = []
        for period, events in sorted(clusters.items()):
            cluster_list.append({
                'period': period,
                'event_count': len(events),
                'events': events
            })
        
        return cluster_list


class MilestoneIdentifier:
    """Identify key milestones"""
    
    def identify_milestones(self, events: List[Dict[str, Any]]) -> List[str]:
        """Identify key milestone events"""
        
        # Keywords indicating importance
        milestone_keywords = [
            'breakthrough', 'achievement', 'launched', 'announced', 'released',
            'discovered', 'invented', 'founded', 'established', 'milestone',
            'historic', 'first', 'major', 'significant', 'revolutionary'
        ]
        
        milestones = []
        for event in events:
            text_lower = event['text'].lower()
            
            # Check for milestone keywords
            if any(keyword in text_lower for keyword in milestone_keywords):
                year = event.get('year', 'Unknown year')
                milestones.append(f"[{year}] {event['text']}")
        
        return milestones


class TimelineSummarizer:
    """Create timeline summary"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def create_chronological_summary(self, sequences: List[Dict[str, Any]],
                                    clusters: List[Dict[str, Any]]) -> str:
        """Create chronological summary"""
        
        # Prepare timeline context
        timeline_parts = []
        
        for cluster in clusters[:5]:  # Top 5 periods
            period = cluster['period']
            events = cluster['events']
            
            event_texts = [e['text'] for e in events[:3]]  # Top 3 events
            timeline_parts.append(f"{period}: {' '.join(event_texts)}")
        
        context = '\n'.join(timeline_parts)
        
        system_prompt = """You are an expert at creating chronological summaries. 
        Create a timeline summary that captures the progression of events over time."""
        
        user_prompt = f"""
        Create a chronological summary (3-4 sentences) from this timeline:
        
        {context}
        
        Focus on the temporal progression and key developments over time.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


# Agent functions
def initialize_timeline_documents_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Initialize timeline documents"""
    
    documents = [
        {
            'source': 'Tech History Archive',
            'content': """The personal computer revolution began in 1975 when the Altair 8800 was introduced. 
            In 1976, Apple Computer was founded by Steve Jobs and Steve Wozniak. The Apple II, released in 1977, 
            became one of the first successful mass-produced personal computers. IBM entered the market in 1981 
            with the IBM PC, establishing the platform that would dominate business computing."""
        },
        {
            'source': 'Digital Era Timeline',
            'content': """The World Wide Web was invented by Tim Berners-Lee in 1989 while working at CERN. 
            The first website went live in 1991, marking the beginning of the internet age. Netscape Navigator 
            launched in 1994, making web browsing accessible to the general public. Google was founded in 1998, 
            revolutionizing internet search."""
        },
        {
            'source': 'Mobile Computing History',
            'content': """The smartphone era began in 2007 with Apple's introduction of the iPhone. The App Store 
            opened in 2008, creating a new software distribution model. Android smartphones gained market share 
            rapidly after 2010. By 2015, smartphones had become ubiquitous, fundamentally changing how people 
            access information and communicate."""
        },
        {
            'source': 'AI Development Chronicle',
            'content': """Deep learning breakthroughs occurred in 2012 with AlexNet winning ImageNet. Google's 
            AlphaGo defeated the world Go champion in 2016, demonstrating AI capabilities in complex strategy. 
            GPT-3 was released in 2020, showcasing advanced natural language processing. ChatGPT launched in 
            November 2022, bringing AI to mainstream attention."""
        }
    ]
    
    return {
        **state,
        'source_documents': documents,
        'messages': state['messages'] + [f'Initialized {len(documents)} timeline documents']
    }


def extract_events_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Extract temporal events"""
    
    extractor = TemporalExtractor()
    all_events = []
    
    for doc in state['source_documents']:
        events = extractor.extract_events(doc)
        all_events.extend(events)
    
    # Normalize temporal references
    normalized_events = extractor.normalize_temporal_references(all_events)
    
    return {
        **state,
        'extracted_events': normalized_events,
        'messages': state['messages'] + [f'Extracted {len(normalized_events)} temporal events']
    }


def sequence_events_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Sequence events chronologically"""
    
    sequencer = EventSequencer()
    
    # Sort events
    sorted_events = sequencer.sort_events(state['extracted_events'])
    
    # Identify sequences
    sequences = sequencer.identify_sequences(sorted_events)
    
    return {
        **state,
        'event_sequences': sequences,
        'messages': state['messages'] + [f'Sequenced events into {len(sequences)} temporal sequences']
    }


def cluster_temporal_events_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Cluster events by time period"""
    
    clusterer = TemporalClusterer()
    
    # Cluster by year
    clusters = clusterer.cluster_by_period(state['extracted_events'], 'year')
    
    return {
        **state,
        'temporal_clusters': clusters,
        'messages': state['messages'] + [f'Clustered events into {len(clusters)} temporal periods']
    }


def identify_milestones_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Identify key milestones"""
    
    identifier = MilestoneIdentifier()
    milestones = identifier.identify_milestones(state['extracted_events'])
    
    return {
        **state,
        'milestone_highlights': milestones,
        'messages': state['messages'] + [f'Identified {len(milestones)} key milestones']
    }


def create_timeline_summary_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Create timeline summary"""
    
    summarizer = TimelineSummarizer()
    summary = summarizer.create_chronological_summary(
        state['event_sequences'],
        state['temporal_clusters']
    )
    
    return {
        **state,
        'timeline_summary': summary,
        'messages': state['messages'] + [f'Created timeline summary ({len(summary.split())} words)']
    }


def evaluate_timeline_quality_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Evaluate timeline quality"""
    
    # Calculate metrics
    total_events = len(state['extracted_events'])
    events_with_year = sum(1 for e in state['extracted_events'] if e.get('year'))
    
    # Time span coverage
    years = [e.get('year') for e in state['extracted_events'] if e.get('year')]
    time_span = max(years) - min(years) if years else 0
    
    # Milestone coverage
    milestone_ratio = len(state['milestone_highlights']) / total_events if total_events > 0 else 0
    
    metrics = {
        'total_events': total_events,
        'temporal_precision': events_with_year / total_events if total_events > 0 else 0,
        'time_span_years': time_span,
        'sequence_count': len(state['event_sequences']),
        'cluster_count': len(state['temporal_clusters']),
        'milestone_count': len(state['milestone_highlights']),
        'milestone_ratio': milestone_ratio,
        'avg_events_per_cluster': total_events / len(state['temporal_clusters']) if state['temporal_clusters'] else 0
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Evaluated quality (temporal precision: {metrics["temporal_precision"]:.1%})']
    }


def analyze_timeline_summarization_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Analyze timeline summarization results"""
    
    analytics = {
        'timeline_stats': {
            'total_events': state['quality_metrics']['total_events'],
            'time_span': state['quality_metrics']['time_span_years'],
            'earliest_year': min([e.get('year', 9999) for e in state['extracted_events']]),
            'latest_year': max([e.get('year', 0) for e in state['extracted_events']])
        },
        'sequence_stats': {
            'sequence_count': state['quality_metrics']['sequence_count'],
            'avg_sequence_length': sum(s['span'] for s in state['event_sequences']) / len(state['event_sequences']) if state['event_sequences'] else 0
        },
        'cluster_stats': {
            'cluster_count': state['quality_metrics']['cluster_count'],
            'avg_events_per_cluster': state['quality_metrics']['avg_events_per_cluster']
        },
        'milestone_stats': {
            'milestone_count': state['quality_metrics']['milestone_count'],
            'milestone_ratio': state['quality_metrics']['milestone_ratio']
        },
        'quality_stats': {
            'temporal_precision': state['quality_metrics']['temporal_precision']
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed timeline summarization results']
    }


def generate_timeline_report_agent(state: TimelineSummarizationState) -> TimelineSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "TIMELINE SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "CHRONOLOGICAL SUMMARY:",
        "-" * 40,
        state['timeline_summary'],
        "",
        "",
        "KEY MILESTONES:",
        "-" * 40
    ]
    
    for milestone in state['milestone_highlights']:
        report_lines.append(f"• {milestone}")
    
    report_lines.extend([
        "",
        "",
        "TEMPORAL CLUSTERS:",
        "-" * 40
    ])
    
    for cluster in state['temporal_clusters']:
        report_lines.append(f"\n{cluster['period']} ({cluster['event_count']} events):")
        for event in cluster['events'][:2]:  # Show first 2 events
            report_lines.append(f"  - {event['text'][:80]}...")
    
    report_lines.extend([
        "",
        "",
        "EVENT SEQUENCES:",
        "-" * 40
    ])
    
    for i, sequence in enumerate(state['event_sequences'], 1):
        report_lines.append(f"\nSequence {i}: {sequence['start_year']} - {sequence['end_year']} ({sequence['span']} events)")
        for event in sequence['events'][:2]:
            report_lines.append(f"  • {event['text'][:80]}...")
    
    report_lines.extend([
        "",
        "",
        "TIMELINE STATISTICS:",
        "-" * 40,
        f"Time Span: {state['analytics']['timeline_stats']['time_span']} years "
        f"({state['analytics']['timeline_stats']['earliest_year']} - {state['analytics']['timeline_stats']['latest_year']})",
        f"Total Events: {state['analytics']['timeline_stats']['total_events']}",
        f"Event Sequences: {state['analytics']['sequence_stats']['sequence_count']}",
        f"Temporal Clusters: {state['analytics']['cluster_stats']['cluster_count']}",
        f"Key Milestones: {state['analytics']['milestone_stats']['milestone_count']}",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Temporal Precision: {state['analytics']['quality_stats']['temporal_precision']:.1%}",
        f"Average Events per Cluster: {state['analytics']['cluster_stats']['avg_events_per_cluster']:.1f}",
        f"Average Sequence Length: {state['analytics']['sequence_stats']['avg_sequence_length']:.1f} events",
        f"Milestone Ratio: {state['analytics']['milestone_stats']['milestone_ratio']:.1%}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Timeline spans {state['analytics']['timeline_stats']['time_span']} years of development",
        f"✓ Identified {state['analytics']['milestone_stats']['milestone_count']} key milestone events",
        f"✓ Organized {state['analytics']['timeline_stats']['total_events']} events into chronological structure",
        f"✓ Achieved {state['analytics']['quality_stats']['temporal_precision']:.0%} temporal precision",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Add date normalization for improved precision",
        "• Implement event causality analysis",
        "• Enable interactive timeline visualization",
        "• Add temporal uncertainty handling",
        "• Implement multi-granularity timeline views",
        "• Enable parallel timeline comparison",
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
def create_timeline_summarization_graph():
    """Create the timeline summarization workflow graph"""
    
    workflow = StateGraph(TimelineSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_timeline_documents_agent)
    workflow.add_node("extract_events", extract_events_agent)
    workflow.add_node("sequence_events", sequence_events_agent)
    workflow.add_node("cluster_events", cluster_temporal_events_agent)
    workflow.add_node("identify_milestones", identify_milestones_agent)
    workflow.add_node("create_summary", create_timeline_summary_agent)
    workflow.add_node("evaluate_quality", evaluate_timeline_quality_agent)
    workflow.add_node("analyze_results", analyze_timeline_summarization_agent)
    workflow.add_node("generate_report", generate_timeline_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "extract_events")
    workflow.add_edge("extract_events", "sequence_events")
    workflow.add_edge("sequence_events", "cluster_events")
    workflow.add_edge("cluster_events", "identify_milestones")
    workflow.add_edge("identify_milestones", "create_summary")
    workflow.add_edge("create_summary", "evaluate_quality")
    workflow.add_edge("evaluate_quality", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the timeline summarization graph
    app = create_timeline_summarization_graph()
    
    # Initialize state
    initial_state: TimelineSummarizationState = {
        'messages': [],
        'source_documents': [],
        'extracted_events': [],
        'temporal_clusters': [],
        'event_sequences': [],
        'timeline_summary': '',
        'milestone_highlights': [],
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("TIMELINE SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nTimeline summarization pattern execution complete! ✓")

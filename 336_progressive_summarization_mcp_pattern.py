"""
Progressive Summarization MCP Pattern

This pattern demonstrates progressive summarization in an agentic MCP system.
The system reveals information incrementally based on user interest and engagement.

Use cases:
- Interactive content exploration
- Adaptive learning systems
- Progressive disclosure interfaces
- Interest-based content delivery
- Engagement-driven summarization
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import deque


# Define the state for progressive summarization
class ProgressiveSummarizationState(TypedDict):
    """State for tracking progressive summarization process"""
    messages: Annotated[List[str], add]
    source_document: str
    content_hierarchy: Dict[str, Any]
    interest_signals: List[Dict[str, Any]]
    progressive_layers: List[Dict[str, Any]]
    current_layer: int
    user_engagement: Dict[str, Any]
    revealed_content: List[str]
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class ContentLayerBuilder:
    """Build progressive content layers"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def create_initial_hook(self, document: str) -> str:
        """Create engaging initial hook"""
        
        system_prompt = """You are an expert at creating engaging hooks. 
        Create a compelling one-sentence hook that captures the essence and 
        sparks curiosity."""
        
        user_prompt = f"""
        Create an engaging hook (1 sentence) for this content:
        
        {document[:300]}...
        
        Make it intriguing and encourage further reading.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def create_layer(self, document: str, previous_content: str, 
                    layer_number: int, detail_level: float) -> str:
        """Create a progressive layer"""
        
        target_words = int(len(document.split()) * detail_level)
        
        system_prompt = f"""You are an expert at progressive content disclosure. 
        Create layer {layer_number} that builds on previous information and adds 
        new details progressively."""
        
        user_prompt = f"""
        Previous content revealed:
        {previous_content}
        
        Full document (for context):
        {document[:500]}...
        
        Create the next layer (approximately {target_words} words) that:
        1. Expands on the previous layer
        2. Adds new important details
        3. Maintains reader engagement
        4. Leaves some details for potential next layer
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class InterestSignalDetector:
    """Detect user interest signals"""
    
    def simulate_engagement_signals(self, layer_number: int) -> Dict[str, Any]:
        """Simulate user engagement signals"""
        
        # Simulate different engagement patterns
        engagement_levels = {
            1: {'read_time': 15, 'scroll_depth': 1.0, 'interactions': 2},
            2: {'read_time': 25, 'scroll_depth': 1.0, 'interactions': 3},
            3: {'read_time': 35, 'scroll_depth': 0.9, 'interactions': 2},
            4: {'read_time': 20, 'scroll_depth': 0.7, 'interactions': 1}
        }
        
        signals = engagement_levels.get(layer_number, {'read_time': 10, 'scroll_depth': 0.5, 'interactions': 0})
        
        # Calculate interest score
        interest_score = (
            (signals['read_time'] / 40) * 0.4 +
            signals['scroll_depth'] * 0.3 +
            (signals['interactions'] / 5) * 0.3
        )
        
        return {
            'layer': layer_number,
            'read_time_seconds': signals['read_time'],
            'scroll_depth': signals['scroll_depth'],
            'interactions': signals['interactions'],
            'interest_score': min(1.0, interest_score),
            'continue_signal': interest_score > 0.5
        }
    
    def aggregate_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate engagement signals"""
        
        if not signals:
            return {'overall_engagement': 0.0, 'trend': 'neutral'}
        
        avg_interest = sum(s['interest_score'] for s in signals) / len(signals)
        
        # Detect trend
        if len(signals) >= 2:
            recent_avg = sum(s['interest_score'] for s in signals[-2:]) / 2
            earlier_avg = sum(s['interest_score'] for s in signals[:-2]) / max(1, len(signals) - 2)
            
            if recent_avg > earlier_avg + 0.1:
                trend = 'increasing'
            elif recent_avg < earlier_avg - 0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'initial'
        
        return {
            'overall_engagement': avg_interest,
            'trend': trend,
            'total_layers_viewed': len(signals),
            'recommend_continue': avg_interest > 0.6
        }


class AdaptiveContentController:
    """Control adaptive content revelation"""
    
    def determine_next_action(self, engagement: Dict[str, Any], 
                              current_layer: int, max_layers: int) -> str:
        """Determine next action based on engagement"""
        
        if current_layer >= max_layers:
            return 'complete'
        
        if engagement['overall_engagement'] < 0.3:
            return 'stop'  # User not interested
        elif engagement['overall_engagement'] > 0.7:
            return 'continue_detailed'  # High interest, more detail
        elif engagement['overall_engagement'] > 0.5:
            return 'continue_moderate'  # Moderate interest
        else:
            return 'stop'
    
    def adjust_detail_level(self, base_level: float, engagement: Dict[str, Any]) -> float:
        """Adjust detail level based on engagement"""
        
        if engagement['trend'] == 'increasing':
            return min(1.0, base_level * 1.2)  # Increase detail
        elif engagement['trend'] == 'decreasing':
            return max(0.1, base_level * 0.8)  # Decrease detail
        else:
            return base_level


class ProgressTracker:
    """Track progression through content"""
    
    def calculate_coverage(self, revealed: List[str], full_document: str) -> float:
        """Calculate content coverage"""
        
        revealed_text = ' '.join(revealed)
        revealed_words = set(re.findall(r'\b\w+\b', revealed_text.lower()))
        full_words = set(re.findall(r'\b\w+\b', full_document.lower()))
        
        if not full_words:
            return 0.0
        
        return len(revealed_words & full_words) / len(full_words)
    
    def estimate_remaining_content(self, coverage: float) -> Dict[str, Any]:
        """Estimate remaining content"""
        
        remaining = 1.0 - coverage
        
        if remaining < 0.1:
            status = 'nearly_complete'
            estimated_layers = 0
        elif remaining < 0.3:
            status = 'mostly_revealed'
            estimated_layers = 1
        elif remaining < 0.6:
            status = 'partially_revealed'
            estimated_layers = 2
        else:
            status = 'early_stage'
            estimated_layers = 3
        
        return {
            'coverage': coverage,
            'remaining': remaining,
            'status': status,
            'estimated_layers_remaining': estimated_layers
        }


# Agent functions
def initialize_progressive_content_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Initialize progressive content"""
    
    document = """
    Artificial intelligence has evolved dramatically over the past decade, transforming from a specialized 
    research field into a technology that impacts nearly every aspect of modern life. Machine learning 
    algorithms now power recommendation systems, autonomous vehicles, medical diagnostics, and countless 
    other applications. The rapid advancement of deep learning, particularly with transformer architectures, 
    has enabled unprecedented capabilities in natural language processing, computer vision, and generative AI.
    
    The current wave of AI development centers on large language models and multimodal systems that can 
    understand and generate human-like text, images, and even video. These models, trained on vast amounts 
    of data, demonstrate emergent capabilities that weren't explicitly programmed. They can reason, solve 
    problems, write code, and engage in sophisticated conversations. However, they also present challenges 
    related to bias, hallucination, and the potential for misuse.
    
    Looking forward, AI development faces both tremendous opportunities and significant challenges. Researchers 
    are working on making AI systems more reliable, interpretable, and aligned with human values. The field 
    is exploring new architectures, training methods, and approaches to artificial general intelligence (AGI). 
    Simultaneously, society grapples with questions about AI governance, ethics, job displacement, and the 
    concentration of AI capabilities in the hands of a few organizations. The next decade will likely determine 
    whether AI becomes a broadly beneficial technology or whether its risks outweigh its rewards.
    """
    
    return {
        **state,
        'source_document': document,
        'current_layer': 0,
        'revealed_content': [],
        'interest_signals': [],
        'messages': state['messages'] + [f'Initialized progressive content ({len(document.split())} words)']
    }


def build_content_hierarchy_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Build content hierarchy"""
    
    hierarchy = {
        'total_words': len(state['source_document'].split()),
        'max_layers': 5,
        'layer_config': [
            {'layer': 0, 'type': 'hook', 'detail_level': 0.05},
            {'layer': 1, 'type': 'overview', 'detail_level': 0.15},
            {'layer': 2, 'type': 'key_points', 'detail_level': 0.30},
            {'layer': 3, 'type': 'detailed', 'detail_level': 0.50},
            {'layer': 4, 'type': 'comprehensive', 'detail_level': 0.75}
        ]
    }
    
    return {
        **state,
        'content_hierarchy': hierarchy,
        'messages': state['messages'] + [f'Built content hierarchy ({hierarchy["max_layers"]} layers)']
    }


def reveal_layer_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Reveal next progressive layer"""
    
    builder = ContentLayerBuilder()
    layer_num = state['current_layer']
    config = state['content_hierarchy']['layer_config'][layer_num]
    
    if layer_num == 0:
        # Create initial hook
        layer_content = builder.create_initial_hook(state['source_document'])
    else:
        # Create progressive layer
        previous_content = '\n\n'.join(state['revealed_content'])
        layer_content = builder.create_layer(
            state['source_document'],
            previous_content,
            layer_num,
            config['detail_level']
        )
    
    # Add to revealed content
    new_revealed = state['revealed_content'] + [layer_content]
    
    # Create layer record
    layer_record = {
        'layer_number': layer_num,
        'type': config['type'],
        'content': layer_content,
        'word_count': len(layer_content.split()),
        'detail_level': config['detail_level']
    }
    
    new_layers = state.get('progressive_layers', []) + [layer_record]
    
    return {
        **state,
        'progressive_layers': new_layers,
        'revealed_content': new_revealed,
        'current_layer': layer_num + 1,
        'messages': state['messages'] + [f'Revealed layer {layer_num} ({config["type"]}, {len(layer_content.split())} words)']
    }


def capture_engagement_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Capture user engagement signals"""
    
    detector = InterestSignalDetector()
    current_layer = state['current_layer'] - 1  # Just revealed layer
    
    signals = detector.simulate_engagement_signals(current_layer)
    
    new_signals = state['interest_signals'] + [signals]
    aggregated = detector.aggregate_signals(new_signals)
    
    return {
        **state,
        'interest_signals': new_signals,
        'user_engagement': aggregated,
        'messages': state['messages'] + [f'Captured engagement (interest: {signals["interest_score"]:.2f})']
    }


def decide_continuation_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Decide whether to continue revealing"""
    
    controller = AdaptiveContentController()
    tracker = ProgressTracker()
    
    # Calculate coverage
    coverage = tracker.calculate_coverage(state['revealed_content'], state['source_document'])
    progress = tracker.estimate_remaining_content(coverage)
    
    # Determine next action
    action = controller.determine_next_action(
        state['user_engagement'],
        state['current_layer'],
        state['content_hierarchy']['max_layers']
    )
    
    # Prepare metrics
    metrics = {
        'coverage': coverage,
        'progress': progress,
        'action': action,
        'engagement': state['user_engagement'],
        'layers_revealed': len(state['progressive_layers']),
        'should_continue': action.startswith('continue') and state['current_layer'] < state['content_hierarchy']['max_layers']
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Decided action: {action} (coverage: {coverage:.1%})']
    }


def analyze_progressive_summarization_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Analyze progressive summarization results"""
    
    analytics = {
        'content_stats': {
            'total_words': state['content_hierarchy']['total_words'],
            'layers_revealed': len(state['progressive_layers']),
            'max_layers': state['content_hierarchy']['max_layers'],
            'completion_rate': len(state['progressive_layers']) / state['content_hierarchy']['max_layers']
        },
        'engagement_stats': {
            'overall_engagement': state['user_engagement']['overall_engagement'],
            'trend': state['user_engagement']['trend'],
            'avg_interest': sum(s['interest_score'] for s in state['interest_signals']) / len(state['interest_signals']) if state['interest_signals'] else 0,
            'total_interactions': sum(s['interactions'] for s in state['interest_signals'])
        },
        'coverage_stats': {
            'content_coverage': state['quality_metrics']['coverage'],
            'status': state['quality_metrics']['progress']['status'],
            'remaining': state['quality_metrics']['progress']['remaining']
        },
        'layer_details': [
            {
                'layer': layer['layer_number'],
                'type': layer['type'],
                'words': layer['word_count'],
                'detail_level': layer['detail_level']
            }
            for layer in state['progressive_layers']
        ]
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed progressive summarization results']
    }


def generate_progressive_report_agent(state: ProgressiveSummarizationState) -> ProgressiveSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "PROGRESSIVE SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "CONTENT PROGRESSION:",
        "-" * 40
    ]
    
    for i, layer in enumerate(state['progressive_layers']):
        report_lines.append(f"\nLAYER {layer['layer_number']} - {layer['type'].upper()}:")
        report_lines.append(f"({layer['word_count']} words, {layer['detail_level']:.0%} detail level)")
        report_lines.append("-" * 40)
        report_lines.append(layer['content'])
        
        if i < len(state['interest_signals']):
            signal = state['interest_signals'][i]
            report_lines.append(f"\nEngagement: Interest Score {signal['interest_score']:.2f}, "
                              f"Read Time {signal['read_time_seconds']}s, "
                              f"Interactions: {signal['interactions']}")
    
    report_lines.extend([
        "",
        "",
        "ENGAGEMENT ANALYSIS:",
        "-" * 40,
        f"Overall Engagement: {state['analytics']['engagement_stats']['overall_engagement']:.1%}",
        f"Engagement Trend: {state['analytics']['engagement_stats']['trend']}",
        f"Average Interest: {state['analytics']['engagement_stats']['avg_interest']:.1%}",
        f"Total Interactions: {state['analytics']['engagement_stats']['total_interactions']}",
        "",
        "CONTENT COVERAGE:",
        "-" * 40,
        f"Coverage Achieved: {state['analytics']['coverage_stats']['content_coverage']:.1%}",
        f"Status: {state['analytics']['coverage_stats']['status']}",
        f"Remaining Content: {state['analytics']['coverage_stats']['remaining']:.1%}",
        "",
        "PROGRESSION STATISTICS:",
        "-" * 40,
        f"Layers Revealed: {state['analytics']['content_stats']['layers_revealed']} / {state['analytics']['content_stats']['max_layers']}",
        f"Completion Rate: {state['analytics']['content_stats']['completion_rate']:.1%}",
        f"Final Action: {state['quality_metrics']['action']}",
        "",
        "LAYER BREAKDOWN:",
        "-" * 40
    ])
    
    for layer_info in state['analytics']['layer_details']:
        report_lines.append(f"Layer {layer_info['layer']} ({layer_info['type']}): {layer_info['words']} words at {layer_info['detail_level']:.0%} detail")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Revealed {len(state['progressive_layers'])} progressive layers",
        f"✓ Maintained {state['analytics']['engagement_stats']['overall_engagement']:.0%} overall engagement",
        f"✓ Achieved {state['analytics']['coverage_stats']['content_coverage']:.0%} content coverage",
        f"✓ Engagement trend: {state['analytics']['engagement_stats']['trend']}",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Continue revealing if engagement remains high",
        "• Adjust detail level based on engagement trends",
        "• Provide skip-ahead option for highly engaged users",
        "• Allow backtracking to previous layers",
        "• Implement personalized layer pacing",
        "• Add interactive elements to boost engagement",
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
def create_progressive_summarization_graph():
    """Create the progressive summarization workflow graph"""
    
    workflow = StateGraph(ProgressiveSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_progressive_content_agent)
    workflow.add_node("build_hierarchy", build_content_hierarchy_agent)
    workflow.add_node("reveal_layer", reveal_layer_agent)
    workflow.add_node("capture_engagement", capture_engagement_agent)
    workflow.add_node("decide_continuation", decide_continuation_agent)
    workflow.add_node("analyze_results", analyze_progressive_summarization_agent)
    workflow.add_node("generate_report", generate_progressive_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "build_hierarchy")
    workflow.add_edge("build_hierarchy", "reveal_layer")
    workflow.add_edge("reveal_layer", "capture_engagement")
    workflow.add_edge("capture_engagement", "decide_continuation")
    
    # Conditional: continue or finish
    def should_continue(state: ProgressiveSummarizationState) -> str:
        if state['quality_metrics'].get('should_continue', False):
            return "reveal_layer"
        else:
            return "analyze_results"
    
    workflow.add_conditional_edges(
        "decide_continuation",
        should_continue,
        {
            "reveal_layer": "reveal_layer",
            "analyze_results": "analyze_results"
        }
    )
    
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the progressive summarization graph
    app = create_progressive_summarization_graph()
    
    # Initialize state
    initial_state: ProgressiveSummarizationState = {
        'messages': [],
        'source_document': '',
        'content_hierarchy': {},
        'interest_signals': [],
        'progressive_layers': [],
        'current_layer': 0,
        'user_engagement': {},
        'revealed_content': [],
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("PROGRESSIVE SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nProgressive summarization pattern execution complete! ✓")

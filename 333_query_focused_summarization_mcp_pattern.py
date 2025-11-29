"""
Query-Focused Summarization MCP Pattern

This pattern demonstrates query-focused summarization in an agentic MCP system.
The system generates summaries that specifically address user queries or
information needs.

Use cases:
- Question answering
- Information retrieval
- Search result summarization
- Topic-specific briefings
- Targeted content extraction
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import Counter


# Define the state for query-focused summarization
class QueryFocusedSummarizationState(TypedDict):
    """State for tracking query-focused summarization process"""
    messages: Annotated[List[str], add]
    query: str
    source_documents: List[Dict[str, Any]]
    query_analysis: Dict[str, Any]
    relevant_passages: List[Dict[str, Any]]
    ranked_passages: List[Dict[str, Any]]
    focused_summary: str
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class QueryAnalyzer:
    """Analyze user query"""
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and extract key terms"""
        
        # Extract keywords
        words = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'what', 'how', 'when', 'where', 'why', 'who', 'which', 
                    'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Determine query type
        query_lower = query.lower()
        if query_lower.startswith('what'):
            query_type = 'definition'
        elif query_lower.startswith('how'):
            query_type = 'process'
        elif query_lower.startswith('why'):
            query_type = 'causation'
        elif query_lower.startswith('when'):
            query_type = 'temporal'
        elif query_lower.startswith('where'):
            query_type = 'location'
        else:
            query_type = 'general'
        
        # Extract entities (simple capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', query)
        
        return {
            'original_query': query,
            'keywords': keywords,
            'query_type': query_type,
            'entities': entities,
            'focus_terms': keywords[:5]  # Top 5 keywords
        }


class PassageRetriever:
    """Retrieve relevant passages"""
    
    def retrieve_relevant_passages(self, documents: List[Dict[str, Any]], 
                                   query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve passages relevant to query"""
        
        keywords = set(query_analysis['keywords'])
        passages = []
        
        for doc in documents:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', doc['content'])
            
            for i, sentence in enumerate(sentences):
                # Calculate relevance
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                keyword_matches = len(keywords & sentence_words)
                
                if keyword_matches > 0:
                    passages.append({
                        'text': sentence.strip(),
                        'source': doc['title'],
                        'keyword_matches': keyword_matches,
                        'position': i,
                        'relevance_score': keyword_matches / len(keywords) if keywords else 0
                    })
        
        return passages


class PassageRanker:
    """Rank passages by query relevance"""
    
    def calculate_query_similarity(self, passage: str, query_terms: List[str]) -> float:
        """Calculate similarity to query"""
        passage_words = set(re.findall(r'\b\w+\b', passage.lower()))
        query_words = set(query_terms)
        
        if not passage_words or not query_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(passage_words & query_words)
        union = len(passage_words | query_words)
        
        return intersection / union if union > 0 else 0.0
    
    def score_passage(self, passage: Dict[str, Any], 
                     query_analysis: Dict[str, Any]) -> float:
        """Score passage for ranking"""
        
        base_score = passage['relevance_score']
        
        # Boost for entity matches
        entities_in_passage = sum(
            1 for entity in query_analysis['entities'] 
            if entity.lower() in passage['text'].lower()
        )
        entity_bonus = entities_in_passage * 0.2
        
        # Boost for position (earlier passages slightly preferred)
        position_score = 1.0 / (1 + passage['position'] * 0.01)
        
        # Query similarity
        similarity = self.calculate_query_similarity(
            passage['text'], 
            query_analysis['focus_terms']
        )
        
        # Composite score
        total_score = (
            base_score * 0.4 +
            similarity * 0.3 +
            entity_bonus +
            position_score * 0.1
        )
        
        return total_score
    
    def rank_passages(self, passages: List[Dict[str, Any]], 
                     query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank passages by relevance"""
        
        ranked = []
        for passage in passages:
            score = self.score_passage(passage, query_analysis)
            ranked.append({
                **passage,
                'final_score': score
            })
        
        ranked.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked


class FocusedSummarizer:
    """Generate query-focused summary"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def generate_focused_summary(self, query: str, 
                                 top_passages: List[Dict[str, Any]],
                                 query_type: str) -> str:
        """Generate summary focused on query"""
        
        system_prompt = f"""You are an expert at creating query-focused summaries. 
        Answer the user's query based on the provided passages. Focus specifically on 
        information relevant to the query. This is a {query_type} type query."""
        
        passages_text = "\n\n".join([
            f"Passage {i+1} (from {p['source']}): {p['text']}"
            for i, p in enumerate(top_passages[:5])
        ])
        
        user_prompt = f"""
        Query: {query}
        
        Relevant Passages:
        {passages_text}
        
        Create a focused summary that directly answers the query using information 
        from the passages. Be concise (2-3 sentences) and specific to the query.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class QueryCoverageEvaluator:
    """Evaluate how well summary covers query"""
    
    def calculate_query_coverage(self, summary: str, 
                                 query_analysis: Dict[str, Any]) -> float:
        """Calculate coverage of query terms"""
        
        summary_lower = summary.lower()
        keywords = query_analysis['keywords']
        
        if not keywords:
            return 0.0
        
        covered = sum(1 for kw in keywords if kw in summary_lower)
        return covered / len(keywords)
    
    def calculate_relevance(self, summary: str, query: str) -> float:
        """Calculate relevance to original query"""
        
        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        
        if not summary_words or not query_words:
            return 0.0
        
        overlap = len(summary_words & query_words)
        return overlap / len(query_words)
    
    def assess_directness(self, summary: str, query_type: str) -> float:
        """Assess how directly summary answers query"""
        
        # Simple heuristic based on query type
        if query_type == 'definition' and ('is ' in summary.lower() or 'refers to' in summary.lower()):
            return 0.9
        elif query_type == 'process' and ('first' in summary.lower() or 'then' in summary.lower()):
            return 0.9
        elif query_type == 'causation' and ('because' in summary.lower() or 'due to' in summary.lower()):
            return 0.9
        else:
            return 0.7  # Default


# Agent functions
def initialize_query_and_documents_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Initialize query and source documents"""
    
    query = "What are the main impacts of climate change on agriculture?"
    
    documents = [
        {
            'title': 'Climate Change and Food Security',
            'content': """Climate change significantly affects agricultural productivity worldwide. 
            Rising temperatures reduce crop yields, especially for heat-sensitive crops like wheat and rice. 
            Changing rainfall patterns cause droughts in some regions and floods in others, disrupting 
            planting and harvesting schedules. Extreme weather events damage crops and infrastructure. 
            Increased pest pressure due to warmer temperatures leads to higher crop losses. Water scarcity 
            is becoming a major constraint for irrigation-dependent agriculture."""
        },
        {
            'title': 'Agricultural Adaptation Strategies',
            'content': """Farmers are implementing various adaptation strategies to cope with climate change. 
            Drought-resistant crop varieties are being developed and deployed. Improved irrigation efficiency 
            helps conserve water resources. Diversification of crops reduces risk from climate variability. 
            Changes in planting dates accommodate shifting seasons. Soil conservation practices improve 
            resilience to extreme weather."""
        },
        {
            'title': 'Economic Impact on Farming',
            'content': """Climate change creates substantial economic challenges for the agricultural sector. 
            Reduced productivity leads to lower farm incomes. Increased costs for pest management and 
            irrigation strain farm budgets. Market volatility from supply disruptions affects pricing. 
            Small-scale farmers face particular hardship adapting to climate impacts."""
        }
    ]
    
    return {
        **state,
        'query': query,
        'source_documents': documents,
        'messages': state['messages'] + [f'Initialized query and {len(documents)} source documents']
    }


def analyze_query_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Analyze user query"""
    
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze_query(state['query'])
    
    return {
        **state,
        'query_analysis': analysis,
        'messages': state['messages'] + [f'Analyzed query (type: {analysis["query_type"]}, {len(analysis["keywords"])} keywords)']
    }


def retrieve_passages_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Retrieve relevant passages"""
    
    retriever = PassageRetriever()
    passages = retriever.retrieve_relevant_passages(
        state['source_documents'],
        state['query_analysis']
    )
    
    return {
        **state,
        'relevant_passages': passages,
        'messages': state['messages'] + [f'Retrieved {len(passages)} relevant passages']
    }


def rank_passages_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Rank passages by relevance"""
    
    ranker = PassageRanker()
    ranked = ranker.rank_passages(
        state['relevant_passages'],
        state['query_analysis']
    )
    
    return {
        **state,
        'ranked_passages': ranked,
        'messages': state['messages'] + [f'Ranked {len(ranked)} passages']
    }


def generate_focused_summary_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Generate query-focused summary"""
    
    summarizer = FocusedSummarizer()
    summary = summarizer.generate_focused_summary(
        state['query'],
        state['ranked_passages'],
        state['query_analysis']['query_type']
    )
    
    return {
        **state,
        'focused_summary': summary,
        'messages': state['messages'] + [f'Generated focused summary ({len(summary.split())} words)']
    }


def evaluate_coverage_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Evaluate query coverage"""
    
    evaluator = QueryCoverageEvaluator()
    
    metrics = {
        'query_coverage': evaluator.calculate_query_coverage(
            state['focused_summary'],
            state['query_analysis']
        ),
        'relevance': evaluator.calculate_relevance(
            state['focused_summary'],
            state['query']
        ),
        'directness': evaluator.assess_directness(
            state['focused_summary'],
            state['query_analysis']['query_type']
        ),
        'passages_used': len(state['ranked_passages'][:5]),
        'total_passages': len(state['ranked_passages']),
        'summary_words': len(state['focused_summary'].split())
    }
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Evaluated coverage (query coverage: {metrics["query_coverage"]:.1%})']
    }


def analyze_query_focused_summarization_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Analyze summarization results"""
    
    analytics = {
        'query_stats': {
            'query_type': state['query_analysis']['query_type'],
            'keywords': len(state['query_analysis']['keywords']),
            'entities': len(state['query_analysis']['entities'])
        },
        'retrieval_stats': {
            'passages_retrieved': len(state['relevant_passages']),
            'passages_ranked': len(state['ranked_passages']),
            'passages_used': state['quality_metrics']['passages_used'],
            'avg_passage_score': sum(p['final_score'] for p in state['ranked_passages'][:5]) / 5 if state['ranked_passages'] else 0
        },
        'quality_stats': state['quality_metrics']
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed query-focused summarization results']
    }


def generate_query_focused_report_agent(state: QueryFocusedSummarizationState) -> QueryFocusedSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "QUERY-FOCUSED SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "USER QUERY:",
        "-" * 40,
        state['query'],
        "",
        "QUERY ANALYSIS:",
        "-" * 40,
        f"Query Type: {state['query_analysis']['query_type']}",
        f"Keywords: {', '.join(state['query_analysis']['keywords'])}",
        f"Entities: {', '.join(state['query_analysis']['entities']) if state['query_analysis']['entities'] else 'None'}",
        "",
        "FOCUSED SUMMARY:",
        "-" * 40,
        state['focused_summary'],
        "",
        "",
        "TOP RANKED PASSAGES:",
        "-" * 40
    ]
    
    for i, passage in enumerate(state['ranked_passages'][:5], 1):
        report_lines.append(f"\n{i}. [Score: {passage['final_score']:.3f}] {passage['text'][:100]}...")
        report_lines.append(f"   Source: {passage['source']} | Keyword Matches: {passage['keyword_matches']}")
    
    report_lines.extend([
        "",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Query Coverage: {state['analytics']['quality_stats']['query_coverage']:.1%}",
        f"Relevance Score: {state['analytics']['quality_stats']['relevance']:.1%}",
        f"Directness: {state['analytics']['quality_stats']['directness']:.1%}",
        f"Summary Length: {state['analytics']['quality_stats']['summary_words']} words",
        "",
        "RETRIEVAL STATISTICS:",
        "-" * 40,
        f"Passages Retrieved: {state['analytics']['retrieval_stats']['passages_retrieved']}",
        f"Passages Used: {state['analytics']['retrieval_stats']['passages_used']}",
        f"Avg Passage Score: {state['analytics']['retrieval_stats']['avg_passage_score']:.3f}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Query type '{state['query_analysis']['query_type']}' successfully identified",
        f"✓ {state['analytics']['quality_stats']['query_coverage']:.0%} of query terms covered in summary",
        f"✓ {state['analytics']['retrieval_stats']['passages_used']} most relevant passages synthesized",
        f"✓ Achieved {state['analytics']['quality_stats']['directness']:.0%} directness in answering",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Expand passage retrieval for comprehensive coverage",
        "• Implement query expansion for better recall",
        "• Add citation tracking for source attribution",
        "• Use neural retrieval models for better ranking",
        "• Implement query reformulation for ambiguous queries",
        "• Add multi-hop reasoning for complex queries",
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
def create_query_focused_summarization_graph():
    """Create the query-focused summarization workflow graph"""
    
    workflow = StateGraph(QueryFocusedSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_query_and_documents_agent)
    workflow.add_node("analyze_query", analyze_query_agent)
    workflow.add_node("retrieve_passages", retrieve_passages_agent)
    workflow.add_node("rank_passages", rank_passages_agent)
    workflow.add_node("generate_summary", generate_focused_summary_agent)
    workflow.add_node("evaluate_coverage", evaluate_coverage_agent)
    workflow.add_node("analyze_results", analyze_query_focused_summarization_agent)
    workflow.add_node("generate_report", generate_query_focused_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_query")
    workflow.add_edge("analyze_query", "retrieve_passages")
    workflow.add_edge("retrieve_passages", "rank_passages")
    workflow.add_edge("rank_passages", "generate_summary")
    workflow.add_edge("generate_summary", "evaluate_coverage")
    workflow.add_edge("evaluate_coverage", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the query-focused summarization graph
    app = create_query_focused_summarization_graph()
    
    # Initialize state
    initial_state: QueryFocusedSummarizationState = {
        'messages': [],
        'query': '',
        'source_documents': [],
        'query_analysis': {},
        'relevant_passages': [],
        'ranked_passages': [],
        'focused_summary': '',
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("QUERY-FOCUSED SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nQuery-focused summarization pattern execution complete! ✓")

"""
Summarization MCP Pattern

This pattern implements summarization techniques for condensing
information while preserving key content.

Key Features:
- Extractive summarization
- Abstractive summarization
- Key point extraction
- Length control
- Multi-document synthesis
"""

from typing import TypedDict, Sequence, Annotated, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class SummarizationState(TypedDict):
    """State for summarization pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    source_text: str
    summary_type: str
    target_length: int
    key_points: List[str]
    summary: str


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def summarization_agent(state: SummarizationState) -> SummarizationState:
    """Generates summaries of text"""
    source = state.get("source_text", "")[:200]  # Preview
    summary_type = state.get("summary_type", "abstractive")
    
    system_prompt = """You are a summarization expert.

Summarization Goals:
• Preserve key information
• Reduce length
• Maintain coherence
• Remove redundancy
• Adapt to purpose

Concise yet complete."""
    
    user_prompt = f"""Source (preview): {source}...
Type: {summary_type}

Design summarization system.
Show summarization strategies."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    📝 Summarization Agent:
    
    Summarization System:
    ```python
    class SummarizationSystem:
        def __init__(self):
            self.extractor = ExtractiveSummarizer()
            self.abstractor = AbstractiveSummarizer()
            self.ranker = SentenceRanker()
        
        def summarize(self, text, method='abstractive', length=100):
            '''Generate summary'''
            if method == 'extractive':
                return self.extractive_summary(text, length)
            elif method == 'abstractive':
                return self.abstractive_summary(text, length)
            elif method == 'hybrid':
                return self.hybrid_summary(text, length)
    ```
    
    Extractive Summarization:
    
    Sentence Ranking:
    ```python
    def extractive_summary(text, target_length):
        '''Select important sentences'''
        # Split into sentences
        sentences = split_sentences(text)
        
        # Score each sentence
        scores = []
        for sentence in sentences:
            score = calculate_importance(sentence, text)
            scores.append((sentence, score))
        
        # Sort by score
        ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Select top sentences until length reached
        summary_sentences = []
        current_length = 0
        
        for sentence, score in ranked:
            if current_length + len(sentence) <= target_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
        
        # Order by appearance in original
        summary = reorder_by_position(summary_sentences, sentences)
        
        return ' '.join(summary)
    ```
    
    Importance Scoring:
    ```python
    def calculate_importance(sentence, document):
        '''Score sentence importance'''
        score = 0
        
        # Position (first/last paragraphs important)
        score += position_score(sentence, document) * 0.2
        
        # Centrality (similar to other sentences)
        score += centrality_score(sentence, document) * 0.3
        
        # Keyword frequency
        score += keyword_score(sentence, document) * 0.2
        
        # Title words
        score += title_overlap_score(sentence, document) * 0.15
        
        # Length (prefer medium length)
        score += length_score(sentence) * 0.1
        
        # Numerical data
        score += numerical_score(sentence) * 0.05
        
        return score
    ```
    
    TextRank Algorithm:
    ```python
    def textrank_summarize(text):
        '''Graph-based ranking'''
        sentences = split_sentences(text)
        
        # Build similarity graph
        graph = build_similarity_graph(sentences)
        
        # Run PageRank
        scores = pagerank(graph)
        
        # Select top-ranked sentences
        ranked = sorted(zip(sentences, scores), 
                       key=lambda x: x[1], 
                       reverse=True)
        
        top_sentences = [s for s, score in ranked[:5]]
        
        return ' '.join(top_sentences)
    ```
    
    Abstractive Summarization:
    
    LLM-based Summarization:
    ```python
    def abstractive_summary(text, target_length):
        '''Generate new text'''
        prompt = f'''
        Summarize the following text in {target_length} words.
        Focus on the main points and key information.
        
        Text: {text}
        
        Summary:
        '''
        
        summary = llm.generate(prompt)
        
        return summary
    ```
    
    Controlled Generation:
    ```python
    def controlled_abstractive(text, constraints):
        '''Summarize with constraints'''
        prompt = f'''
        Summarize with these requirements:
        - Length: {constraints['length']} words
        - Style: {constraints['style']}
        - Focus: {constraints['focus']}
        - Include: {constraints['must_include']}
        
        Text: {text}
        '''
        
        summary = llm.generate(prompt)
        
        # Verify constraints
        if not meets_constraints(summary, constraints):
            summary = regenerate_with_feedback(summary, constraints)
        
        return summary
    ```
    
    Key Point Extraction:
    
    Main Ideas:
    ```python
    def extract_key_points(text):
        '''Identify main ideas'''
        # Method 1: Topic sentences
        paragraphs = split_paragraphs(text)
        key_points = [first_sentence(p) for p in paragraphs]
        
        # Method 2: Clustering
        sentences = split_sentences(text)
        clusters = cluster_sentences(sentences)
        key_points = [select_representative(c) for c in clusters]
        
        # Method 3: LLM extraction
        prompt = f"Extract 5 key points from: {text}"
        key_points = llm.generate(prompt)
        
        return key_points
    ```
    
    Bullet Point Summary:
    ```python
    def bullet_point_summary(text):
        '''Concise bullet points'''
        key_points = extract_key_points(text)
        
        bullets = []
        for point in key_points:
            # Condense to single phrase
            bullet = condense_to_phrase(point)
            bullets.append(f"• {bullet}")
        
        return '\\n'.join(bullets)
    ```
    
    Hierarchical Summarization:
    
    Multi-level Summary:
    ```python
    def hierarchical_summary(text):
        '''Multiple granularity levels'''
        summaries = {{
            'tweet': summarize(text, length=280),      # Ultra-short
            'abstract': summarize(text, length=200),    # Short
            'executive': summarize(text, length=500),   # Medium
            'detailed': summarize(text, length=1000)    # Long
        }}
        
        return summaries
    ```
    
    Progressive Summarization:
    ```python
    def progressive_summary(long_document):
        '''Summarize in stages'''
        # Stage 1: Summarize sections
        sections = split_into_sections(long_document)
        section_summaries = [summarize(s) for s in sections]
        
        # Stage 2: Summarize section summaries
        combined = '\\n'.join(section_summaries)
        final_summary = summarize(combined)
        
        return final_summary
    ```
    
    Multi-Document Summarization:
    
    Cross-document Synthesis:
    ```python
    def multi_doc_summary(documents):
        '''Summarize multiple documents'''
        # Extract information from each
        info_items = []
        for doc in documents:
            items = extract_key_info(doc)
            info_items.extend(items)
        
        # Remove duplicates
        unique_items = remove_duplicates(info_items)
        
        # Cluster similar information
        clusters = cluster_information(unique_items)
        
        # Generate summary from clusters
        summary_parts = []
        for cluster in clusters:
            part = synthesize_cluster(cluster)
            summary_parts.append(part)
        
        # Combine into coherent summary
        summary = combine_coherently(summary_parts)
        
        return summary
    ```
    
    Query-Focused Summarization:
    
    Relevance-based Summary:
    ```python
    def query_focused_summary(text, query):
        '''Summarize relevant to query'''
        sentences = split_sentences(text)
        
        # Score relevance to query
        scored = []
        for sentence in sentences:
            relevance = calculate_relevance(sentence, query)
            importance = calculate_importance(sentence, text)
            
            # Combine scores
            score = 0.6 * relevance + 0.4 * importance
            scored.append((sentence, score))
        
        # Select top relevant sentences
        top_sentences = sorted(scored, key=lambda x: x[1], reverse=True)[:5]
        
        return ' '.join([s for s, score in top_sentences])
    ```
    
    Aspect-based Summarization:
    
    Multi-aspect Summary:
    ```python
    def aspect_based_summary(text, aspects):
        '''Summarize for each aspect'''
        summaries = {{}}
        
        for aspect in aspects:
            # Extract sentences about aspect
            relevant = filter_by_aspect(text, aspect)
            
            # Summarize aspect
            summaries[aspect] = summarize(relevant)
        
        # Example aspects for product review:
        # {'quality': '...', 'price': '...', 'service': '...'}
        
        return summaries
    ```
    
    Update Summarization:
    
    Incremental Summary:
    ```python
    def update_summary(existing_summary, new_content):
        '''Update summary with new info'''
        # Extract new information
        new_info = extract_novel_info(new_content, existing_summary)
        
        if not new_info:
            return existing_summary  # No update needed
        
        # Integrate new information
        updated = f"{existing_summary} {new_info}"
        
        # Recompress if too long
        if len(updated) > max_length:
            updated = recompress(updated, max_length)
        
        return updated
    ```
    
    Compression Techniques:
    
    Sentence Compression:
    ```python
    def compress_sentence(sentence):
        '''Remove non-essential words'''
        # Parse sentence
        parse = parse_syntax(sentence)
        
        # Keep: subject, verb, object
        # Remove: adjectives, adverbs, prepositional phrases
        
        compressed = extract_core(parse)
        
        # Example:
        # Original: "The very large red car drove quickly"
        # Compressed: "Car drove"
        
        return compressed
    ```
    
    Paraphrasing for Brevity:
    ```python
    def paraphrase_shorter(text):
        '''Rephrase more concisely'''
        prompt = f'''
        Rephrase more concisely: {text}
        
        Remove redundancy and wordiness while preserving meaning.
        '''
        
        compressed = llm.generate(prompt)
        
        return compressed
    ```
    
    Quality Metrics:
    
    ROUGE Score:
    ```python
    def evaluate_summary(summary, reference):
        '''Compare to reference summary'''
        # ROUGE-N: n-gram overlap
        rouge_1 = ngram_overlap(summary, reference, n=1)
        rouge_2 = ngram_overlap(summary, reference, n=2)
        
        # ROUGE-L: longest common subsequence
        rouge_l = lcs_score(summary, reference)
        
        return {{'rouge-1': rouge_1, 'rouge-2': rouge_2, 'rouge-l': rouge_l}}
    ```
    
    Content Coverage:
    ```python
    def check_coverage(summary, original):
        '''Does summary cover main points?'''
        # Extract topics from original
        original_topics = extract_topics(original)
        
        # Check which are in summary
        covered = [t for t in original_topics if mentioned_in(t, summary)]
        
        coverage = len(covered) / len(original_topics)
        
        return coverage
    ```
    
    Best Practices:
    ✓ Identify purpose (why summarize?)
    ✓ Know audience (what do they need?)
    ✓ Preserve key information
    ✓ Maintain coherence
    ✓ Control length appropriately
    ✓ Verify coverage
    ✓ Remove redundancy
    
    Key Insight:
    Effective summarization balances brevity with
    completeness, preserving what matters most.
    """
    
    return {
        "messages": [AIMessage(content=f"📝 Summarization Agent:\n{report}\n\n{response.content}")]
    }


def build_summarization_graph():
    workflow = StateGraph(SummarizationState)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_edge(START, "summarization_agent")
    workflow.add_edge("summarization_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_summarization_graph()
    
    print("=== Summarization MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "source_text": "Long article about AI developments...",
        "summary_type": "abstractive",
        "target_length": 100,
        "key_points": [],
        "summary": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 158: Summarization - COMPLETE")
    print(f"{'='*70}")

"""
Pattern 248: Context Compression MCP Pattern

This pattern demonstrates context compression - reducing context size while
preserving essential information to fit within token limits.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# State definition
class ContextCompressionState(TypedDict):
    """State for context compression workflow"""
    messages: Annotated[List[str], add]
    original_context: str
    compressed_context: str
    compression_ratio: float


class ContextCompressor:
    """Compresses context"""
    
    def remove_redundancy(self, text: str) -> str:
        """Remove redundant information"""
        lines = text.split('\n')
        unique_lines = []
        seen = set()
        
        for line in lines:
            normalized = line.strip().lower()
            if normalized and normalized not in seen:
                unique_lines.append(line)
                seen.add(normalized)
        
        return '\n'.join(unique_lines)
    
    def extract_key_points(self, text: str) -> str:
        """Extract key points"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        # Keep sentences with important keywords
        important_keywords = ['important', 'critical', 'key', 'must', 'required', 'essential']
        
        key_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in important_keywords):
                key_sentences.append(sentence)
        
        # If no important sentences, take first few
        if not key_sentences:
            key_sentences = sentences[:3]
        
        return '. '.join(key_sentences) + '.'
    
    def summarize(self, text: str, max_length: int = 500) -> str:
        """Summarize to fit max length"""
        if len(text) <= max_length:
            return text
        
        # Simple truncation with ellipsis
        return text[:max_length-3] + '...'
    
    def compress(self, text: str) -> str:
        """Apply all compression techniques"""
        # Step 1: Remove redundancy
        compressed = self.remove_redundancy(text)
        
        # Step 2: Extract key points
        compressed = self.extract_key_points(compressed)
        
        # Step 3: Summarize if still too long
        compressed = self.summarize(compressed, max_length=500)
        
        return compressed


def load_original_context_agent(state: ContextCompressionState) -> ContextCompressionState:
    """Load original context"""
    print("\n📥 Loading Original Context...")
    
    original = """
    The system is experiencing high load. The system is experiencing high load.
    Performance metrics show critical issues. Performance is degraded.
    Database response time is slow. Query optimization is required.
    Important: User authentication is affected by the performance issues.
    Critical: Payment processing delays are occurring.
    The load balancer configuration must be reviewed.
    Cache hit rate is low. Cache must be optimized.
    Essential: Security patches need to be applied immediately.
    Monitoring alerts have been triggered. The system requires immediate attention.
    The infrastructure scaling is required. Additional resources are needed.
    Key stakeholders have been notified about the issues.
    """
    
    print(f"\n  Original Size: {len(original)} characters")
    print(f"  Preview: {original[:150]}...")
    
    return {
        **state,
        "original_context": original,
        "messages": [f"✓ Loaded context ({len(original)} chars)"]
    }


def compress_context_agent(state: ContextCompressionState) -> ContextCompressionState:
    """Compress context"""
    print("\n🗜️ Compressing Context...")
    
    compressor = ContextCompressor()
    original = state["original_context"]
    
    # Step-by-step compression
    print("\n  Step 1: Remove redundancy")
    step1 = compressor.remove_redundancy(original)
    print(f"    Size: {len(original)} → {len(step1)} chars ({len(original)-len(step1)} removed)")
    
    print("\n  Step 2: Extract key points")
    step2 = compressor.extract_key_points(step1)
    print(f"    Size: {len(step1)} → {len(step2)} chars ({len(step1)-len(step2)} removed)")
    
    print("\n  Step 3: Final summarization")
    compressed = compressor.summarize(step2, max_length=500)
    print(f"    Size: {len(step2)} → {len(compressed)} chars")
    
    compression_ratio = (1 - len(compressed) / len(original)) * 100
    
    print(f"\n  Total Compression: {compression_ratio:.1f}%")
    print(f"  Original: {len(original)} chars → Compressed: {len(compressed)} chars")
    
    return {
        **state,
        "compressed_context": compressed,
        "compression_ratio": compression_ratio,
        "messages": [f"✓ Compressed by {compression_ratio:.1f}%"]
    }


def generate_context_compression_report_agent(state: ContextCompressionState) -> ContextCompressionState:
    """Generate context compression report"""
    print("\n" + "="*70)
    print("CONTEXT COMPRESSION REPORT")
    print("="*70)
    
    print(f"\n📊 Compression Statistics:")
    print(f"  Original Size: {len(state['original_context'])} characters")
    print(f"  Compressed Size: {len(state['compressed_context'])} characters")
    print(f"  Compression Ratio: {state['compression_ratio']:.1f}%")
    print(f"  Space Saved: {len(state['original_context']) - len(state['compressed_context'])} characters")
    
    print(f"\n📝 Original Context:")
    print(f"  {state['original_context'][:200]}...")
    
    print(f"\n🗜️ Compressed Context:")
    print(f"  {state['compressed_context']}")
    
    print("\n💡 Context Compression Benefits:")
    print("  • Reduced token usage")
    print("  • Faster processing")
    print("  • Lower costs")
    print("  • Preserved essential information")
    print("  • Better focus on key points")
    print("  • Fits within token limits")
    
    print("\n="*70)
    print("✅ Context Compression Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_compression_graph():
    workflow = StateGraph(ContextCompressionState)
    workflow.add_node("load", load_original_context_agent)
    workflow.add_node("compress", compress_context_agent)
    workflow.add_node("report", generate_context_compression_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "compress")
    workflow.add_edge("compress", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 248: Context Compression MCP Pattern")
    print("="*70)
    
    app = create_context_compression_graph()
    final_state = app.invoke({
        "messages": [],
        "original_context": "",
        "compressed_context": "",
        "compression_ratio": 0.0
    })
    print("\n✅ Context Compression Pattern Complete!")


if __name__ == "__main__":
    main()

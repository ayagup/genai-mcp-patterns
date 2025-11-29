"""
Pattern 242: Context Extraction MCP Pattern

This pattern demonstrates context extraction - extracting relevant context
from various sources to build a comprehensive understanding.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import re


# State definition
class ContextExtractionState(TypedDict):
    """State for context extraction workflow"""
    messages: Annotated[List[str], add]
    raw_data: str
    extracted_contexts: Dict[str, Any]


class ContextExtractor:
    """Extracts context from raw data"""
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities"""
        # Simple pattern matching for demo
        words = text.split()
        entities = [w for w in words if w[0].isupper() and len(w) > 1]
        return entities
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates"""
        date_pattern = r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
        return re.findall(date_pattern, text)
    
    def extract_numbers(self, text: str) -> List[str]:
        """Extract numbers"""
        return re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        # Simple keyword extraction based on length and frequency
        words = text.lower().split()
        keywords = [w for w in words if len(w) > 4]
        return list(set(keywords))[:10]  # Top 10 unique keywords


def load_raw_data_agent(state: ContextExtractionState) -> ContextExtractionState:
    """Load raw data for extraction"""
    print("\n📥 Loading Raw Data...")
    
    raw_data = """
    Project Alpha launched on 2024-01-15 with budget of 500000 dollars.
    Team lead John Smith reported 95% completion rate.
    Next milestone scheduled for 2024-06-30.
    Performance metrics show 1200 requests per second with 99.9% uptime.
    Customer satisfaction rating increased to 4.8 out of 5.
    """
    
    print(f"\n  Raw Data Length: {len(raw_data)} characters")
    print(f"  Preview: {raw_data[:100]}...")
    
    return {
        **state,
        "raw_data": raw_data,
        "messages": ["✓ Raw data loaded"]
    }


def extract_context_agent(state: ContextExtractionState) -> ContextExtractionState:
    """Extract context from raw data"""
    print("\n🔍 Extracting Context...")
    
    extractor = ContextExtractor()
    extracted = {}
    
    # Extract different types of context
    entities = extractor.extract_entities(state["raw_data"])
    dates = extractor.extract_dates(state["raw_data"])
    numbers = extractor.extract_numbers(state["raw_data"])
    keywords = extractor.extract_keywords(state["raw_data"])
    
    extracted["entities"] = entities
    extracted["dates"] = dates
    extracted["numbers"] = numbers
    extracted["keywords"] = keywords
    
    print(f"\n  ✓ Extracted {len(entities)} entities")
    print(f"  ✓ Extracted {len(dates)} dates")
    print(f"  ✓ Extracted {len(numbers)} numbers")
    print(f"  ✓ Extracted {len(keywords)} keywords")
    
    return {
        **state,
        "extracted_contexts": extracted,
        "messages": ["✓ Context extracted"]
    }


def generate_context_extraction_report_agent(state: ContextExtractionState) -> ContextExtractionState:
    """Generate context extraction report"""
    print("\n" + "="*70)
    print("CONTEXT EXTRACTION REPORT")
    print("="*70)
    
    print(f"\n📥 Raw Data:")
    print(f"  {state['raw_data'][:200]}...")
    
    print(f"\n🔍 Extracted Context:")
    
    print(f"\n  Entities ({len(state['extracted_contexts']['entities'])}):")
    for entity in state['extracted_contexts']['entities'][:5]:
        print(f"    • {entity}")
    
    print(f"\n  Dates ({len(state['extracted_contexts']['dates'])}):")
    for date in state['extracted_contexts']['dates']:
        print(f"    • {date}")
    
    print(f"\n  Numbers ({len(state['extracted_contexts']['numbers'])}):")
    for num in state['extracted_contexts']['numbers'][:5]:
        print(f"    • {num}")
    
    print(f"\n  Keywords ({len(state['extracted_contexts']['keywords'])}):")
    for keyword in state['extracted_contexts']['keywords'][:8]:
        print(f"    • {keyword}")
    
    print("\n💡 Context Extraction Benefits:")
    print("  • Structured understanding")
    print("  • Entity recognition")
    print("  • Temporal awareness")
    print("  • Quantitative insights")
    print("  • Key concept identification")
    
    print("\n="*70)
    print("✅ Context Extraction Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_context_extraction_graph():
    workflow = StateGraph(ContextExtractionState)
    workflow.add_node("load", load_raw_data_agent)
    workflow.add_node("extract", extract_context_agent)
    workflow.add_node("report", generate_context_extraction_report_agent)
    workflow.add_edge(START, "load")
    workflow.add_edge("load", "extract")
    workflow.add_edge("extract", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 242: Context Extraction MCP Pattern")
    print("="*70)
    
    app = create_context_extraction_graph()
    final_state = app.invoke({
        "messages": [],
        "raw_data": "",
        "extracted_contexts": {}
    })
    print("\n✅ Context Extraction Pattern Complete!")


if __name__ == "__main__":
    main()

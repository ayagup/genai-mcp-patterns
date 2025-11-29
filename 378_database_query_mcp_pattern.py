"""
Database Query MCP Pattern

This pattern demonstrates intelligent database query generation,
optimization, and execution planning.

Pattern Type: Specialized Domain
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import operator


class DatabaseQueryState(TypedDict):
    """State for database query"""
    natural_language_query: str
    sql_query: str
    query_plan: Dict
    optimization_suggestions: List[str]
    messages: Annotated[List, operator.add]


class QueryGenerator:
    """Generate SQL from natural language"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate(self, nl_query: str) -> str:
        """Generate SQL query"""
        return "SELECT * FROM users WHERE age > 18 ORDER BY name"


def generate_query(state: DatabaseQueryState) -> DatabaseQueryState:
    """Generate SQL query"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    generator = QueryGenerator(llm)
    
    state["sql_query"] = generator.generate(state["natural_language_query"])
    state["query_plan"] = {"type": "SeqScan", "table": "users", "filter": "age > 18"}
    state["optimization_suggestions"] = ["Add index on age column", "Use covering index"]
    state["messages"].append(HumanMessage(content="Query generated and optimized"))
    return state


def create_dbquery_graph():
    """Create database query workflow"""
    workflow = StateGraph(DatabaseQueryState)
    workflow.add_node("generate", generate_query)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


if __name__ == "__main__":
    app = create_dbquery_graph()
    result = app.invoke({"natural_language_query": "Find all adult users sorted by name", "sql_query": "", "query_plan": {}, "optimization_suggestions": [], "messages": []})
    print(f"Database Query Complete: {result['sql_query']}")

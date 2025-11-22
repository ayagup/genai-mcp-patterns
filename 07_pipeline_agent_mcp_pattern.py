"""
Pipeline Agent MCP Pattern
===========================
Sequential processing where each agent performs a specific transformation stage.
Data flows through the pipeline from one agent to the next in a linear fashion.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
import operator


# Define the state
class PipelineState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    raw_data: str
    cleaned_data: str
    transformed_data: str
    enriched_data: str
    final_output: str
    pipeline_stage: str


# Stage 1: Ingestion Agent
def ingestion_agent(state: PipelineState):
    """Pipeline stage 1: Data ingestion and initial processing."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    system_msg = SystemMessage(
        content="You are the Ingestion Agent (Pipeline Stage 1). "
                "Extract and collect raw data from the input. "
                "Focus on gathering all relevant information."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Stage 1 - Ingestion: {response.content}")],
        "raw_data": response.content,
        "pipeline_stage": "ingestion_complete"
    }


# Stage 2: Cleaning Agent
def cleaning_agent(state: PipelineState):
    """Pipeline stage 2: Data cleaning and validation."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    raw_data = state.get("raw_data", "No data")
    
    system_msg = SystemMessage(
        content=f"You are the Cleaning Agent (Pipeline Stage 2). "
                f"Raw data: {raw_data}\n"
                "Clean, validate, and standardize the data. Remove noise and errors."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Stage 2 - Cleaning: {response.content}")],
        "cleaned_data": response.content,
        "pipeline_stage": "cleaning_complete"
    }


# Stage 3: Transformation Agent
def transformation_agent(state: PipelineState):
    """Pipeline stage 3: Data transformation and processing."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    cleaned_data = state.get("cleaned_data", "No data")
    
    system_msg = SystemMessage(
        content=f"You are the Transformation Agent (Pipeline Stage 3). "
                f"Cleaned data: {cleaned_data}\n"
                "Transform and process the data into the desired format."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Stage 3 - Transformation: {response.content}")],
        "transformed_data": response.content,
        "pipeline_stage": "transformation_complete"
    }


# Stage 4: Enrichment Agent
def enrichment_agent(state: PipelineState):
    """Pipeline stage 4: Data enrichment with additional context."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    transformed_data = state.get("transformed_data", "No data")
    
    system_msg = SystemMessage(
        content=f"You are the Enrichment Agent (Pipeline Stage 4). "
                f"Transformed data: {transformed_data}\n"
                "Enrich the data with additional context, insights, and metadata."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Stage 4 - Enrichment: {response.content}")],
        "enriched_data": response.content,
        "pipeline_stage": "enrichment_complete"
    }


# Stage 5: Output Agent
def output_agent(state: PipelineState):
    """Pipeline stage 5: Final output generation."""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    
    enriched_data = state.get("enriched_data", "No data")
    
    system_msg = SystemMessage(
        content=f"You are the Output Agent (Pipeline Stage 5). "
                f"Enriched data: {enriched_data}\n"
                "Generate the final output in the desired format. "
                "Ensure quality and completeness."
    )
    
    response = llm.invoke([system_msg] + list(state["messages"]))
    
    return {
        "messages": [AIMessage(content=f"Stage 5 - Output: {response.content}")],
        "final_output": response.content,
        "pipeline_stage": "pipeline_complete"
    }


# Build the pipeline graph
def create_pipeline_agent_graph():
    """Create a sequential pipeline workflow graph."""
    workflow = StateGraph(PipelineState)
    
    # Add pipeline stage nodes in sequence
    workflow.add_node("ingestion", ingestion_agent)
    workflow.add_node("cleaning", cleaning_agent)
    workflow.add_node("transformation", transformation_agent)
    workflow.add_node("enrichment", enrichment_agent)
    workflow.add_node("output", output_agent)
    
    # Create linear pipeline: START -> Stage1 -> Stage2 -> ... -> Stage5 -> END
    workflow.add_edge(START, "ingestion")
    workflow.add_edge("ingestion", "cleaning")
    workflow.add_edge("cleaning", "transformation")
    workflow.add_edge("transformation", "enrichment")
    workflow.add_edge("enrichment", "output")
    workflow.add_edge("output", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the pipeline agent system
    graph = create_pipeline_agent_graph()
    
    print("=" * 60)
    print("PIPELINE AGENT MCP PATTERN DEMO")
    print("=" * 60)
    
    # Example: Data processing pipeline
    print("\n[Task: Process customer feedback data]")
    result = graph.invoke({
        "messages": [HumanMessage(content="Process customer feedback: 'Great product but shipping was slow. Customer service was helpful.'")],
        "raw_data": "",
        "cleaned_data": "",
        "transformed_data": "",
        "enriched_data": "",
        "final_output": "",
        "pipeline_stage": "start"
    })
    
    print("\n--- Pipeline Execution Flow ---")
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")
    
    print(f"\n--- Pipeline Stages Completed ---")
    print(f"1. Raw Data: {bool(result.get('raw_data'))}")
    print(f"2. Cleaned Data: {bool(result.get('cleaned_data'))}")
    print(f"3. Transformed Data: {bool(result.get('transformed_data'))}")
    print(f"4. Enriched Data: {bool(result.get('enriched_data'))}")
    print(f"5. Final Output: {bool(result.get('final_output'))}")
    
    print(f"\n--- Final Output ---")
    print(f"{result.get('final_output', 'N/A')[:300]}...")
    
    print("\n" + "=" * 60)

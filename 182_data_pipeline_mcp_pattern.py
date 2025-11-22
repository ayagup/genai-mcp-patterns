"""
Pattern 182: Data Pipeline MCP Pattern

This pattern demonstrates a data pipeline where data flows through multiple
sequential stages, with each stage performing specific processing operations.
Pipelines are fundamental for data engineering and workflow orchestration.

Key Concepts:
1. Stages: Sequential processing steps (ingest → process → output)
2. Data Flow: Data moves from stage to stage
3. Transformations: Each stage transforms data
4. Orchestration: Coordinate stage execution
5. Error Handling: Handle failures at each stage
6. Monitoring: Track pipeline health and performance
7. Checkpointing: Save intermediate results

Pipeline Characteristics:
- Sequential Execution: Stages run in order
- Stage Independence: Each stage has single responsibility
- Data Lineage: Track data provenance through pipeline
- Idempotency: Re-running produces same results
- Scalability: Scale individual stages independently
- Fault Tolerance: Handle stage failures gracefully

Pipeline Stages:
1. Ingestion: Read data from sources
2. Validation: Check data quality
3. Transformation: Apply business logic
4. Enrichment: Add external data
5. Aggregation: Summarize data
6. Output: Write to destinations

Pipeline Patterns:
- Linear Pipeline: A → B → C → D (sequential)
- Branching Pipeline: A → [B, C] → D (parallel branches)
- Conditional Pipeline: A → (condition) → B or C
- Looping Pipeline: A → B → (condition) → A (iterate)

Benefits:
- Modularity: Each stage is independent
- Reusability: Stages can be reused in different pipelines
- Testability: Test each stage separately
- Maintainability: Easy to modify individual stages
- Observability: Monitor each stage independently
- Scalability: Scale bottleneck stages

Trade-offs:
- Latency: Sequential stages add delay
- Complexity: Many stages increase complexity
- Debugging: Harder to debug multi-stage failures
- Resource Usage: Each stage consumes resources

Use Cases:
- Data Engineering: Process data at scale
- ML Pipelines: Feature engineering → Training → Deployment
- ETL Workflows: Extract → Transform → Load
- CI/CD Pipelines: Build → Test → Deploy
- Event Processing: Ingest → Filter → Aggregate → Alert
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Define the state for data pipeline
class DataPipelineState(TypedDict):
    """State for data pipeline"""
    input_data: List[Dict[str, Any]]
    stage_1_output: Optional[List[Dict[str, Any]]]
    stage_2_output: Optional[List[Dict[str, Any]]]
    stage_3_output: Optional[List[Dict[str, Any]]]
    stage_4_output: Optional[List[Dict[str, Any]]]
    final_output: Optional[List[Dict[str, Any]]]
    pipeline_metadata: Dict[str, Any]
    stage_metrics: Annotated[List[Dict[str, Any]], operator.add]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# PIPELINE STAGE ABSTRACTION
# ============================================================================

class StageStatus(Enum):
    """Status of pipeline stage"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class StageMetrics:
    """Metrics for pipeline stage"""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    input_records: int
    output_records: int
    records_filtered: int
    status: StageStatus
    error_message: Optional[str] = None

class PipelineStage:
    """
    Abstract Pipeline Stage
    
    Each stage:
    - Takes input data
    - Performs transformation
    - Returns output data
    - Tracks metrics
    """
    
    def __init__(self, stage_name: str):
        self.stage_name = stage_name
    
    def execute(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute stage and return results with metrics"""
        start_time = datetime.now()
        
        try:
            # Process data (implemented by subclasses)
            output_data = self.process(input_data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Calculate metrics
            metrics = StageMetrics(
                stage_name=self.stage_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_records=len(input_data),
                output_records=len(output_data),
                records_filtered=len(input_data) - len(output_data),
                status=StageStatus.COMPLETED
            )
            
            return {
                "output": output_data,
                "metrics": metrics,
                "status": "success"
            }
        
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            metrics = StageMetrics(
                stage_name=self.stage_name,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                input_records=len(input_data),
                output_records=0,
                records_filtered=0,
                status=StageStatus.FAILED,
                error_message=str(e)
            )
            
            return {
                "output": [],
                "metrics": metrics,
                "status": "failed",
                "error": str(e)
            }
    
    def process(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process data (to be implemented by subclasses)"""
        raise NotImplementedError("Subclasses must implement process()")

# ============================================================================
# CONCRETE PIPELINE STAGES
# ============================================================================

class IngestionStage(PipelineStage):
    """
    Stage 1: Data Ingestion
    
    Responsibilities:
    - Read data from sources
    - Parse and structure data
    - Initial validation
    """
    
    def __init__(self):
        super().__init__("Ingestion")
    
    def process(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ingest and parse data"""
        ingested = []
        
        for record in input_data:
            # Parse and structure data
            ingested_record = {
                "id": record.get("id"),
                "raw_data": record.get("data", ""),
                "source": record.get("source", "unknown"),
                "ingested_at": datetime.now().isoformat()
            }
            ingested.append(ingested_record)
        
        return ingested

class ValidationStage(PipelineStage):
    """
    Stage 2: Data Validation
    
    Responsibilities:
    - Validate data quality
    - Check schema compliance
    - Filter invalid records
    """
    
    def __init__(self):
        super().__init__("Validation")
    
    def process(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate data and filter invalid records"""
        validated = []
        
        for record in input_data:
            # Validation rules
            is_valid = True
            
            # Check required fields
            if not record.get("id"):
                is_valid = False
            
            # Check data not empty
            if not record.get("raw_data"):
                is_valid = False
            
            # Only include valid records
            if is_valid:
                record["validation_status"] = "VALID"
                validated.append(record)
        
        return validated

class TransformationStage(PipelineStage):
    """
    Stage 3: Data Transformation
    
    Responsibilities:
    - Apply business logic
    - Convert formats
    - Calculate derived fields
    """
    
    def __init__(self):
        super().__init__("Transformation")
    
    def process(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform data according to business rules"""
        transformed = []
        
        for record in input_data:
            # Transform raw_data (uppercase)
            raw = record.get("raw_data", "")
            
            transformed_record = record.copy()
            transformed_record["processed_data"] = raw.upper()
            transformed_record["data_length"] = len(raw)
            transformed_record["transformed_at"] = datetime.now().isoformat()
            
            # Add derived field
            if transformed_record["data_length"] > 10:
                transformed_record["data_category"] = "LONG"
            else:
                transformed_record["data_category"] = "SHORT"
            
            transformed.append(transformed_record)
        
        return transformed

class EnrichmentStage(PipelineStage):
    """
    Stage 4: Data Enrichment
    
    Responsibilities:
    - Add external data
    - Lookup reference data
    - Augment with context
    """
    
    def __init__(self):
        super().__init__("Enrichment")
        # Simulated reference data
        self.reference_data = {
            "source_1": {"location": "US", "priority": "HIGH"},
            "source_2": {"location": "EU", "priority": "MEDIUM"},
            "unknown": {"location": "UNKNOWN", "priority": "LOW"}
        }
    
    def process(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich data with external information"""
        enriched = []
        
        for record in input_data:
            enriched_record = record.copy()
            
            # Lookup enrichment data
            source = record.get("source", "unknown")
            ref_data = self.reference_data.get(source, self.reference_data["unknown"])
            
            # Add enrichment fields
            enriched_record["location"] = ref_data["location"]
            enriched_record["priority"] = ref_data["priority"]
            enriched_record["enriched_at"] = datetime.now().isoformat()
            
            enriched.append(enriched_record)
        
        return enriched

class AggregationStage(PipelineStage):
    """
    Stage 5: Data Aggregation
    
    Responsibilities:
    - Group data
    - Calculate aggregates
    - Summarize metrics
    """
    
    def __init__(self):
        super().__init__("Aggregation")
    
    def process(self, input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate data by category"""
        # Group by data_category
        aggregations = {}
        
        for record in input_data:
            category = record.get("data_category", "UNKNOWN")
            
            if category not in aggregations:
                aggregations[category] = {
                    "category": category,
                    "count": 0,
                    "total_length": 0,
                    "records": []
                }
            
            aggregations[category]["count"] += 1
            aggregations[category]["total_length"] += record.get("data_length", 0)
            aggregations[category]["records"].append(record["id"])
        
        # Convert to list
        aggregated = []
        for category, agg_data in aggregations.items():
            agg_data["average_length"] = (
                agg_data["total_length"] / agg_data["count"]
                if agg_data["count"] > 0 else 0
            )
            aggregated.append(agg_data)
        
        return aggregated

# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class PipelineOrchestrator:
    """
    Pipeline Orchestrator: Manages pipeline execution
    
    Responsibilities:
    - Execute stages in order
    - Handle errors
    - Collect metrics
    - Checkpoint progress
    """
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self.checkpoint = {}
    
    def execute_pipeline(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute all pipeline stages"""
        current_data = input_data
        all_metrics = []
        stage_outputs = {}
        
        for i, stage in enumerate(self.stages):
            print(f"Executing stage {i+1}/{len(self.stages)}: {stage.stage_name}")
            
            # Execute stage
            result = stage.execute(current_data)
            
            # Collect metrics
            all_metrics.append(result["metrics"])
            
            # Save checkpoint
            self.checkpoint[stage.stage_name] = result["output"]
            stage_outputs[f"stage_{i+1}_output"] = result["output"]
            
            # Check for failure
            if result["status"] == "failed":
                return {
                    "status": "pipeline_failed",
                    "failed_stage": stage.stage_name,
                    "error": result.get("error"),
                    "metrics": all_metrics,
                    "stage_outputs": stage_outputs
                }
            
            # Pass output to next stage
            current_data = result["output"]
        
        return {
            "status": "pipeline_completed",
            "final_output": current_data,
            "metrics": all_metrics,
            "stage_outputs": stage_outputs
        }
    
    def get_checkpoint(self, stage_name: str) -> List[Dict[str, Any]]:
        """Get checkpointed data from stage"""
        return self.checkpoint.get(stage_name, [])

# Create pipeline stages
pipeline_stages = [
    IngestionStage(),
    ValidationStage(),
    TransformationStage(),
    EnrichmentStage(),
    AggregationStage()
]

orchestrator = PipelineOrchestrator(pipeline_stages)

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def ingestion_stage_agent(state: DataPipelineState) -> DataPipelineState:
    """Stage 1: Ingestion"""
    input_data = state["input_data"]
    
    stage = IngestionStage()
    result = stage.execute(input_data)
    
    return {
        "stage_1_output": result["output"],
        "stage_metrics": [{
            "stage": "Ingestion",
            "records_in": result["metrics"].input_records,
            "records_out": result["metrics"].output_records,
            "duration": result["metrics"].duration_seconds
        }],
        "messages": [f"[Stage 1: Ingestion] Processed {len(result['output'])} records"]
    }

def validation_stage_agent(state: DataPipelineState) -> DataPipelineState:
    """Stage 2: Validation"""
    input_data = state.get("stage_1_output", [])
    
    stage = ValidationStage()
    result = stage.execute(input_data)
    
    return {
        "stage_2_output": result["output"],
        "stage_metrics": [{
            "stage": "Validation",
            "records_in": result["metrics"].input_records,
            "records_out": result["metrics"].output_records,
            "filtered": result["metrics"].records_filtered,
            "duration": result["metrics"].duration_seconds
        }],
        "messages": [f"[Stage 2: Validation] Validated {len(result['output'])} records, filtered {result['metrics'].records_filtered}"]
    }

def transformation_stage_agent(state: DataPipelineState) -> DataPipelineState:
    """Stage 3: Transformation"""
    input_data = state.get("stage_2_output", [])
    
    stage = TransformationStage()
    result = stage.execute(input_data)
    
    return {
        "stage_3_output": result["output"],
        "stage_metrics": [{
            "stage": "Transformation",
            "records_in": result["metrics"].input_records,
            "records_out": result["metrics"].output_records,
            "duration": result["metrics"].duration_seconds
        }],
        "messages": [f"[Stage 3: Transformation] Transformed {len(result['output'])} records"]
    }

def enrichment_stage_agent(state: DataPipelineState) -> DataPipelineState:
    """Stage 4: Enrichment"""
    input_data = state.get("stage_3_output", [])
    
    stage = EnrichmentStage()
    result = stage.execute(input_data)
    
    return {
        "stage_4_output": result["output"],
        "stage_metrics": [{
            "stage": "Enrichment",
            "records_in": result["metrics"].input_records,
            "records_out": result["metrics"].output_records,
            "duration": result["metrics"].duration_seconds
        }],
        "messages": [f"[Stage 4: Enrichment] Enriched {len(result['output'])} records"]
    }

def aggregation_stage_agent(state: DataPipelineState) -> DataPipelineState:
    """Stage 5: Aggregation"""
    input_data = state.get("stage_4_output", [])
    
    stage = AggregationStage()
    result = stage.execute(input_data)
    
    return {
        "final_output": result["output"],
        "stage_metrics": [{
            "stage": "Aggregation",
            "records_in": result["metrics"].input_records,
            "records_out": result["metrics"].output_records,
            "duration": result["metrics"].duration_seconds
        }],
        "messages": [f"[Stage 5: Aggregation] Generated {len(result['output'])} aggregates"]
    }

# ============================================================================
# BUILD THE PIPELINE GRAPH
# ============================================================================

def create_pipeline_graph():
    """
    Create a StateGraph demonstrating data pipeline pattern.
    
    Pipeline Flow:
    1. Ingestion: Read and parse data
    2. Validation: Check quality and filter
    3. Transformation: Apply business logic
    4. Enrichment: Add external data
    5. Aggregation: Summarize results
    """
    
    workflow = StateGraph(DataPipelineState)
    
    # Add pipeline stages
    workflow.add_node("ingestion", ingestion_stage_agent)
    workflow.add_node("validation", validation_stage_agent)
    workflow.add_node("transformation", transformation_stage_agent)
    workflow.add_node("enrichment", enrichment_stage_agent)
    workflow.add_node("aggregation", aggregation_stage_agent)
    
    # Define linear pipeline flow
    workflow.add_edge(START, "ingestion")
    workflow.add_edge("ingestion", "validation")
    workflow.add_edge("validation", "transformation")
    workflow.add_edge("transformation", "enrichment")
    workflow.add_edge("enrichment", "aggregation")
    workflow.add_edge("aggregation", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Pipeline MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Linear Data Pipeline
    print("\n" + "=" * 80)
    print("Example 1: 5-Stage Linear Pipeline")
    print("=" * 80)
    
    pipeline_graph = create_pipeline_graph()
    
    initial_state: DataPipelineState = {
        "input_data": [
            {"id": 1, "data": "hello world", "source": "source_1"},
            {"id": 2, "data": "data pipeline", "source": "source_2"},
            {"id": 3, "data": "test", "source": "source_1"},
            {"id": 4, "data": "", "source": "unknown"},  # Invalid (will be filtered)
            {"id": 5, "data": "langchain langgraph", "source": "source_2"},
        ],
        "stage_1_output": None,
        "stage_2_output": None,
        "stage_3_output": None,
        "stage_4_output": None,
        "final_output": None,
        "pipeline_metadata": {},
        "stage_metrics": [],
        "messages": []
    }
    
    result = pipeline_graph.invoke(initial_state)
    
    print("\nPipeline Execution Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nStage Metrics:")
    for metric in result["stage_metrics"]:
        print(f"  {metric['stage']}: {metric['records_in']} → {metric['records_out']} records ({metric['duration']:.3f}s)")
        if "filtered" in metric:
            print(f"    Filtered: {metric['filtered']} records")
    
    print("\nFinal Output (Aggregated):")
    for agg in result["final_output"]:
        print(f"  Category: {agg['category']}, Count: {agg['count']}, Avg Length: {agg['average_length']:.1f}")
    
    # Example 2: Pipeline with Orchestrator
    print("\n" + "=" * 80)
    print("Example 2: Pipeline Orchestrator with Checkpointing")
    print("=" * 80)
    
    input_data = [
        {"id": 10, "data": "orchestrated pipeline", "source": "source_1"},
        {"id": 11, "data": "with checkpoints", "source": "source_2"},
    ]
    
    result = orchestrator.execute_pipeline(input_data)
    
    print(f"\nPipeline Status: {result['status']}")
    print("\nCheckpoints available for:")
    for stage in pipeline_stages:
        checkpoint_data = orchestrator.get_checkpoint(stage.stage_name)
        print(f"  - {stage.stage_name}: {len(checkpoint_data)} records")
    
    # Example 3: Pipeline Patterns
    print("\n" + "=" * 80)
    print("Example 3: Pipeline Patterns")
    print("=" * 80)
    
    print("\n1. Linear Pipeline (Current Implementation):")
    print("   A → B → C → D → E")
    print("   Each stage processes sequentially")
    
    print("\n2. Branching Pipeline:")
    print("   A → [B, C] → D")
    print("   Stage A outputs to both B and C in parallel")
    
    print("\n3. Conditional Pipeline:")
    print("   A → (if condition) → B else → C")
    print("   Route based on data characteristics")
    
    print("\n4. Fan-Out/Fan-In Pipeline:")
    print("   A → [B1, B2, B3] → C")
    print("   Parallel processing, then merge results")
    
    print("\n" + "=" * 80)
    print("Data Pipeline Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Pipelines organize data processing into sequential stages
  • Each stage has single responsibility (SRP)
  • Stages are reusable and testable independently
  • Orchestrator manages execution and error handling
  • Checkpointing enables recovery from failures
  • Metrics provide visibility into pipeline performance
  • Common patterns: Linear, Branching, Conditional, Fan-Out/Fan-In
  • Tools: Apache Airflow, Luigi, Prefect, Dagster, Kubeflow
    """)

"""
Pattern 184: Batch Processing MCP Pattern

This pattern demonstrates processing large volumes of data in batches at scheduled
intervals. Unlike stream processing which handles continuous data, batch processing
works with bounded datasets and optimizes for throughput over latency.

Key Concepts:
1. Bounded Data: Finite dataset with known start and end
2. Scheduled Execution: Run at specific times (hourly, daily, weekly)
3. Batch Size: Process data in chunks for efficiency
4. Fault Tolerance: Retry failed batches
5. Checkpointing: Track progress through dataset
6. Parallel Processing: Process multiple batches concurrently
7. Resource Optimization: Maximize throughput and resource utilization

Batch Characteristics:
- Bounded Dataset: Fixed amount of data to process
- High Latency: Minutes to hours (acceptable delay)
- High Throughput: Process millions of records efficiently
- Scheduled: Runs periodically (nightly, hourly)
- Resource Intensive: Can use significant compute resources
- Idempotent: Re-running produces same results

Batch Processing Steps:
1. Read: Load data from source in batches
2. Process: Transform data in parallel
3. Write: Store results to destination
4. Checkpoint: Track progress for recovery

Batch Strategies:
- Full Batch: Process entire dataset
- Mini-Batch: Process in smaller chunks
- Micro-Batch: Very small batches (near real-time)
- Incremental: Process only new/changed data

Benefits:
- Throughput: Process large volumes efficiently
- Resource Utilization: Optimize for batch workloads
- Simplicity: Easier than stream processing
- Cost Effective: Can use spot instances, off-peak hours
- Fault Tolerance: Easy to retry failed batches

Trade-offs:
- Latency: Not suitable for real-time requirements
- Resource Spikes: Heavy resource usage during batch runs
- Complexity: Large batches require careful memory management
- Freshness: Data may be stale between batch runs

Use Cases:
- ETL/Data Warehousing: Nightly data loads
- Report Generation: Daily/monthly reports
- Machine Learning: Training on historical data
- Data Migration: Move data between systems
- Log Processing: Analyze daily logs
- Billing: Calculate monthly bills
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import math

# Define the state for batch processing
class BatchProcessingState(TypedDict):
    """State for batch processing"""
    dataset: List[Dict[str, Any]]
    batch_size: int
    batches_processed: Annotated[List[Dict[str, Any]], operator.add]
    processing_results: Optional[List[Dict[str, Any]]]
    batch_metadata: Dict[str, Any]
    checkpoint: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# BATCH JOB MODEL
# ============================================================================

class BatchStatus(Enum):
    """Status of batch job"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass
class BatchJob:
    """Batch job configuration"""
    job_id: str
    job_name: str
    schedule: str  # cron expression or interval
    batch_size: int
    max_retries: int
    timeout_seconds: int
    
@dataclass
class BatchResult:
    """Result of processing a batch"""
    batch_id: str
    batch_index: int
    records_processed: int
    records_failed: int
    start_time: datetime
    end_time: datetime
    status: BatchStatus
    error_message: Optional[str] = None
    
    def get_duration_seconds(self) -> float:
        """Get batch processing duration"""
        return (self.end_time - self.start_time).total_seconds()

# ============================================================================
# BATCH PROCESSOR
# ============================================================================

class BatchReader:
    """
    Batch Reader: Read data in batches from source
    
    Strategies:
    - Sequential: Read batches one by one
    - Parallel: Read multiple batches concurrently
    - Partitioned: Read from partitioned data source
    """
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    def create_batches(self, dataset: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Split dataset into batches"""
        batches = []
        
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i:i + self.batch_size]
            batches.append(batch)
        
        return batches
    
    def get_batch_metadata(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get metadata about batches"""
        total_records = len(dataset)
        num_batches = math.ceil(total_records / self.batch_size)
        
        return {
            "total_records": total_records,
            "batch_size": self.batch_size,
            "num_batches": num_batches,
            "avg_records_per_batch": total_records / num_batches if num_batches > 0 else 0
        }

class BatchProcessor:
    """
    Batch Processor: Process data in batches
    
    Features:
    - Parallel processing of batches
    - Error handling and retry logic
    - Progress tracking
    - Resource management
    """
    
    def __init__(self):
        self.checkpoint = {}
    
    def process_batch(self, batch: List[Dict[str, Any]], batch_index: int) -> BatchResult:
        """Process a single batch"""
        start_time = datetime.now()
        
        try:
            processed_records = []
            failed_records = 0
            
            for record in batch:
                try:
                    # Simulate processing (transformation)
                    processed_record = self._process_record(record)
                    processed_records.append(processed_record)
                except Exception as e:
                    failed_records += 1
            
            end_time = datetime.now()
            
            # Save checkpoint
            self.checkpoint[f"batch_{batch_index}"] = {
                "completed": True,
                "timestamp": end_time.isoformat()
            }
            
            return BatchResult(
                batch_id=f"batch_{batch_index}",
                batch_index=batch_index,
                records_processed=len(processed_records),
                records_failed=failed_records,
                start_time=start_time,
                end_time=end_time,
                status=BatchStatus.COMPLETED
            )
        
        except Exception as e:
            end_time = datetime.now()
            
            return BatchResult(
                batch_id=f"batch_{batch_index}",
                batch_index=batch_index,
                records_processed=0,
                records_failed=len(batch),
                start_time=start_time,
                end_time=end_time,
                status=BatchStatus.FAILED,
                error_message=str(e)
            )
    
    def _process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual record (transformation logic)"""
        processed = record.copy()
        
        # Example transformations
        if "value" in processed:
            processed["value_doubled"] = processed["value"] * 2
        
        if "name" in processed:
            processed["name_upper"] = processed["name"].upper()
        
        processed["processed_at"] = datetime.now().isoformat()
        processed["batch_processed"] = True
        
        return processed
    
    def process_all_batches(self, batches: List[List[Dict[str, Any]]]) -> List[BatchResult]:
        """Process all batches sequentially"""
        results = []
        
        for i, batch in enumerate(batches):
            result = self.process_batch(batch, i)
            results.append(result)
        
        return results
    
    def get_checkpoint(self) -> Dict[str, Any]:
        """Get current checkpoint state"""
        return self.checkpoint.copy()
    
    def resume_from_checkpoint(self, batches: List[List[Dict[str, Any]]], 
                               checkpoint: Dict[str, Any]) -> List[BatchResult]:
        """Resume processing from checkpoint"""
        results = []
        
        for i, batch in enumerate(batches):
            batch_key = f"batch_{i}"
            
            # Skip already processed batches
            if checkpoint.get(batch_key, {}).get("completed"):
                print(f"Skipping batch {i} (already completed)")
                continue
            
            # Process remaining batches
            result = self.process_batch(batch, i)
            results.append(result)
        
        return results

class BatchWriter:
    """
    Batch Writer: Write processed data to destination
    
    Write Strategies:
    - Bulk Insert: Write entire batch at once
    - Streaming Write: Write records one by one
    - Partitioned Write: Write to partitioned destination
    """
    
    def __init__(self):
        self.output_data = []
    
    def write_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Write batch to destination"""
        # Simulate writing to database/file
        self.output_data.extend(batch)
        
        return {
            "records_written": len(batch),
            "total_records": len(self.output_data),
            "write_timestamp": datetime.now().isoformat()
        }
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Get summary of written data"""
        return {
            "total_records": len(self.output_data),
            "sample_records": self.output_data[:3] if self.output_data else []
        }

# ============================================================================
# BATCH SCHEDULER
# ============================================================================

class BatchScheduler:
    """
    Batch Scheduler: Schedule and orchestrate batch jobs
    
    Scheduling:
    - Cron: Run at specific times (0 0 * * * = daily at midnight)
    - Interval: Run every N minutes/hours
    - Event-Driven: Trigger on file arrival, etc.
    """
    
    def __init__(self):
        self.jobs: Dict[str, BatchJob] = {}
        self.job_history: List[Dict[str, Any]] = []
    
    def register_job(self, job: BatchJob):
        """Register a batch job"""
        self.jobs[job.job_id] = job
    
    def execute_job(self, job_id: str, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a batch job"""
        if job_id not in self.jobs:
            return {"error": f"Job {job_id} not found"}
        
        job = self.jobs[job_id]
        start_time = datetime.now()
        
        # Create batches
        reader = BatchReader(job.batch_size)
        batches = reader.create_batches(dataset)
        
        # Process batches
        processor = BatchProcessor()
        results = processor.process_all_batches(batches)
        
        # Calculate statistics
        total_processed = sum(r.records_processed for r in results)
        total_failed = sum(r.records_failed for r in results)
        total_duration = sum(r.get_duration_seconds() for r in results)
        
        end_time = datetime.now()
        
        execution_result = {
            "job_id": job_id,
            "job_name": job.job_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "batches_processed": len(results),
            "total_records_processed": total_processed,
            "total_records_failed": total_failed,
            "average_batch_duration": total_duration / len(results) if results else 0,
            "status": "COMPLETED" if total_failed == 0 else "COMPLETED_WITH_ERRORS"
        }
        
        # Save to history
        self.job_history.append(execution_result)
        
        return execution_result

# Create instances
reader = BatchReader(batch_size=100)
processor = BatchProcessor()
writer = BatchWriter()
scheduler = BatchScheduler()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def read_batches(state: BatchProcessingState) -> BatchProcessingState:
    """Read and partition data into batches"""
    dataset = state["dataset"]
    batch_size = state["batch_size"]
    
    # Create batches
    batch_reader = BatchReader(batch_size)
    batches = batch_reader.create_batches(dataset)
    metadata = batch_reader.get_batch_metadata(dataset)
    
    return {
        "batch_metadata": metadata,
        "messages": [f"[Read] Created {len(batches)} batches of size {batch_size}"]
    }

def process_batches(state: BatchProcessingState) -> BatchProcessingState:
    """Process all batches"""
    dataset = state["dataset"]
    batch_size = state["batch_size"]
    
    # Create and process batches
    batch_reader = BatchReader(batch_size)
    batches = batch_reader.create_batches(dataset)
    
    batch_processor = BatchProcessor()
    results = batch_processor.process_all_batches(batches)
    
    # Collect processed data
    processed_data = []
    for i, batch in enumerate(batches):
        for record in batch:
            processed = batch_processor._process_record(record)
            processed_data.append(processed)
    
    # Convert results to dicts
    batch_results = []
    for result in results:
        batch_results.append({
            "batch_id": result.batch_id,
            "records_processed": result.records_processed,
            "records_failed": result.records_failed,
            "duration_seconds": result.get_duration_seconds(),
            "status": result.status.value
        })
    
    return {
        "batches_processed": batch_results,
        "processing_results": processed_data,
        "checkpoint": batch_processor.get_checkpoint(),
        "messages": [f"[Process] Processed {len(batches)} batches, {len(processed_data)} total records"]
    }

def write_results(state: BatchProcessingState) -> BatchProcessingState:
    """Write processed results to destination"""
    processed_data = state.get("processing_results", [])
    
    # Write in batches
    batch_writer = BatchWriter()
    write_result = batch_writer.write_batch(processed_data)
    
    return {
        "messages": [f"[Write] Wrote {write_result['records_written']} records to destination"]
    }

# ============================================================================
# BUILD THE BATCH PROCESSING GRAPH
# ============================================================================

def create_batch_processing_graph():
    """
    Create a StateGraph demonstrating batch processing pattern.
    
    Flow:
    1. Read: Partition data into batches
    2. Process: Process each batch
    3. Write: Write results to destination
    """
    
    workflow = StateGraph(BatchProcessingState)
    
    # Add batch processing stages
    workflow.add_node("read", read_batches)
    workflow.add_node("process", process_batches)
    workflow.add_node("write", write_results)
    
    # Define batch flow
    workflow.add_edge(START, "read")
    workflow.add_edge("read", "process")
    workflow.add_edge("process", "write")
    workflow.add_edge("write", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Batch Processing MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic Batch Processing
    print("\n" + "=" * 80)
    print("Example 1: Batch Processing with Checkpointing")
    print("=" * 80)
    
    batch_graph = create_batch_processing_graph()
    
    # Create dataset
    dataset = [{"id": i, "name": f"Record_{i}", "value": i * 10} for i in range(250)]
    
    initial_state: BatchProcessingState = {
        "dataset": dataset,
        "batch_size": 50,
        "batches_processed": [],
        "processing_results": None,
        "batch_metadata": {},
        "checkpoint": {},
        "messages": []
    }
    
    result = batch_graph.invoke(initial_state)
    
    print("\nBatch Processing Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nBatch Metadata:")
    metadata = result["batch_metadata"]
    print(f"  Total Records: {metadata['total_records']}")
    print(f"  Batch Size: {metadata['batch_size']}")
    print(f"  Number of Batches: {metadata['num_batches']}")
    print(f"  Avg Records/Batch: {metadata['avg_records_per_batch']:.1f}")
    
    print("\nBatch Results:")
    for batch_result in result["batches_processed"]:
        print(f"  {batch_result['batch_id']}: {batch_result['records_processed']} records, "
              f"{batch_result['duration_seconds']:.3f}s, status={batch_result['status']}")
    
    print(f"\nCheckpoint: {len(result['checkpoint'])} batches checkpointed")
    
    # Example 2: Scheduled Batch Job
    print("\n" + "=" * 80)
    print("Example 2: Scheduled Batch Job")
    print("=" * 80)
    
    # Register batch job
    nightly_job = BatchJob(
        job_id="nightly_etl",
        job_name="Nightly ETL Job",
        schedule="0 0 * * *",  # Daily at midnight
        batch_size=100,
        max_retries=3,
        timeout_seconds=3600
    )
    
    scheduler.register_job(nightly_job)
    
    # Execute job
    job_result = scheduler.execute_job("nightly_etl", dataset)
    
    print(f"\nJob: {job_result['job_name']}")
    print(f"Status: {job_result['status']}")
    print(f"Duration: {job_result['duration_seconds']:.2f}s")
    print(f"Batches: {job_result['batches_processed']}")
    print(f"Records Processed: {job_result['total_records_processed']}")
    print(f"Records Failed: {job_result['total_records_failed']}")
    print(f"Avg Batch Duration: {job_result['average_batch_duration']:.3f}s")
    
    # Example 3: Batch vs Stream Processing
    print("\n" + "=" * 80)
    print("Example 3: Batch vs Stream Processing Comparison")
    print("=" * 80)
    
    print("\nBatch Processing:")
    print("  Data Type: Bounded (finite dataset)")
    print("  Latency: Minutes to hours (high latency acceptable)")
    print("  Throughput: Very high (millions of records)")
    print("  Schedule: Periodic (hourly, daily, weekly)")
    print("  Resource Usage: High during batch run, idle otherwise")
    print("  Use Cases: ETL, reporting, data warehousing, ML training")
    print("  Tools: Apache Spark, Hadoop MapReduce, AWS Batch")
    print("  Cost: Can use spot instances, off-peak pricing")
    
    print("\nStream Processing:")
    print("  Data Type: Unbounded (continuous stream)")
    print("  Latency: Milliseconds to seconds (low latency required)")
    print("  Throughput: High (thousands to millions of events/sec)")
    print("  Schedule: Continuous (24/7)")
    print("  Resource Usage: Consistent resource usage")
    print("  Use Cases: Real-time analytics, fraud detection, monitoring")
    print("  Tools: Apache Flink, Kafka Streams, Apache Storm")
    print("  Cost: Always running, may cost more")
    
    # Example 4: Batch Optimization Strategies
    print("\n" + "=" * 80)
    print("Example 4: Batch Processing Optimization")
    print("=" * 80)
    
    print("\n1. Batch Size Tuning:")
    print("   Small batches (10-100): Lower memory, more overhead")
    print("   Medium batches (1000-10000): Balanced")
    print("   Large batches (100000+): High throughput, high memory")
    
    print("\n2. Parallel Processing:")
    print("   Process multiple batches concurrently")
    print("   Use thread pools or distributed workers")
    
    print("\n3. Checkpointing:")
    print("   Save progress after each batch")
    print("   Resume from last checkpoint on failure")
    
    print("\n4. Resource Management:")
    print("   Batch during off-peak hours")
    print("   Use auto-scaling for variable workloads")
    print("   Leverage spot instances for cost savings")
    
    print("\n5. Incremental Processing:")
    print("   Process only new/changed data")
    print("   Use watermarks or timestamps")
    
    print("\n" + "=" * 80)
    print("Batch Processing Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Batch processing handles bounded datasets at scheduled intervals
  • Optimize for throughput over latency
  • Process data in chunks (batches) for efficiency
  • Use checkpointing for fault tolerance
  • Schedule during off-peak hours for cost savings
  • Common batch sizes: 100-10,000 records
  • Tools: Apache Spark, Hadoop, AWS Batch, Azure Batch
  • Use for: ETL, reporting, data warehousing, ML training
  • Choose batch when: latency > 1 minute acceptable, periodic processing
    """)

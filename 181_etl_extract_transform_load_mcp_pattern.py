"""
Pattern 181: Extract-Transform-Load (ETL) MCP Pattern

This pattern demonstrates the ETL process where data is extracted from sources,
transformed according to business rules, and loaded into a target system (typically
a data warehouse). ETL is fundamental for data integration and analytics.

Key Concepts:
1. Extract: Read data from source systems (databases, APIs, files)
2. Transform: Clean, validate, aggregate, and restructure data
3. Load: Write transformed data to target destination
4. Data Quality: Ensure data accuracy and consistency
5. Incremental Loading: Load only new/changed data
6. Error Handling: Handle extraction/transformation failures
7. Logging: Track ETL process execution and data lineage

ETL Stages:
- Stage 1 - Extract: Pull data from multiple sources
- Stage 2 - Transform: Apply business rules and transformations
- Stage 3 - Load: Insert data into target warehouse

ETL Characteristics:
- Batch Processing: Typically runs on schedule (nightly, hourly)
- Data Integration: Combines data from multiple sources
- Data Warehouse: Optimized for analytics (OLAP)
- Historical Data: Maintains history for trend analysis
- Data Quality: Validates and cleanses data
- Idempotent: Same input produces same output

Transform Operations:
1. Cleansing: Remove duplicates, fix errors, standardize
2. Validation: Check data quality and constraints
3. Enrichment: Add derived fields, lookups
4. Aggregation: Summarize data (daily totals, averages)
5. Normalization: Convert to standard formats
6. Join: Combine data from multiple sources

Benefits:
- Data Consolidation: Single source of truth
- Analytics-Ready: Optimized for queries
- Data Quality: Cleaned and validated
- Historical Analysis: Time-series data
- Business Intelligence: Support reporting/dashboards

Trade-offs:
- Latency: Not real-time (batch delays)
- Complexity: Complex transformations difficult to maintain
- Resource Intensive: Large batch jobs consume resources
- Schema Dependencies: Changes in source/target schemas

Use Cases:
- Data Warehousing: Populate analytics databases
- Business Intelligence: Feed BI tools (Tableau, Power BI)
- Data Migration: Move data between systems
- Reporting: Generate periodic reports
- Data Lake: Centralize data from multiple sources
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
import json

# Define the state for ETL process
class ETLState(TypedDict):
    """State for ETL pipeline"""
    source_config: Dict[str, Any]
    extracted_data: Optional[List[Dict[str, Any]]]
    transformed_data: Optional[List[Dict[str, Any]]]
    loaded_data: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    etl_metadata: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# DATA SOURCES (Extract Stage)
# ============================================================================

@dataclass
class DataSource:
    """Configuration for data source"""
    source_id: str
    source_type: str  # database, api, file, stream
    connection_string: str
    query_config: Dict[str, Any]

class ExtractorEngine:
    """
    Extractor: Reads data from various sources
    
    Extraction Types:
    - Full Extract: All data
    - Incremental Extract: Only new/changed data (delta)
    - Change Data Capture (CDC): Track changes in real-time
    """
    
    def extract_from_database(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from database"""
        # Simulated database extraction
        print(f"Extracting from database: {source.source_id}")
        
        # Simulate reading from database
        extracted = [
            {"id": 1, "customer_name": "John Doe", "order_total": 150.50, "order_date": "2024-01-15"},
            {"id": 2, "customer_name": "Jane Smith", "order_total": 200.75, "order_date": "2024-01-16"},
            {"id": 3, "customer_name": "Bob Johnson", "order_total": 99.99, "order_date": "2024-01-16"},
            {"id": 4, "customer_name": "Alice Brown", "order_total": 300.00, "order_date": "2024-01-17"},
        ]
        
        return extracted
    
    def extract_from_api(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from REST API"""
        # Simulated API extraction
        print(f"Extracting from API: {source.source_id}")
        
        # Simulate API response
        extracted = [
            {"user_id": 101, "email": "john@example.com", "signup_date": "2024-01-10"},
            {"user_id": 102, "email": "jane@example.com", "signup_date": "2024-01-12"},
        ]
        
        return extracted
    
    def extract_from_file(self, source: DataSource) -> List[Dict[str, Any]]:
        """Extract data from file (CSV, JSON, Parquet)"""
        # Simulated file extraction
        print(f"Extracting from file: {source.source_id}")
        
        # Simulate reading CSV file
        extracted = [
            {"product_id": "P001", "product_name": "Widget", "price": 19.99},
            {"product_id": "P002", "product_name": "Gadget", "price": 29.99},
        ]
        
        return extracted
    
    def extract_incremental(self, source: DataSource, last_extract_time: datetime) -> List[Dict[str, Any]]:
        """
        Incremental Extract: Only new/changed data since last extraction
        
        Uses watermark (timestamp, sequence number) to track progress
        """
        # Simulated incremental extraction
        print(f"Incremental extract from {source.source_id} since {last_extract_time}")
        
        # Only extract records after last_extract_time
        extracted = [
            {"id": 5, "customer_name": "Charlie Davis", "order_total": 125.00, "order_date": "2024-01-18"},
        ]
        
        return extracted

# ============================================================================
# TRANSFORMATION ENGINE (Transform Stage)
# ============================================================================

class TransformationEngine:
    """
    Transformer: Applies business rules and data transformations
    
    Transformation Types:
    1. Cleansing: Remove nulls, duplicates, fix errors
    2. Standardization: Convert to standard formats
    3. Validation: Check data quality
    4. Enrichment: Add derived fields
    5. Aggregation: Summarize data
    6. Join: Combine multiple sources
    """
    
    def cleanse_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Data Cleansing: Remove duplicates, handle nulls, fix errors
        """
        cleansed = []
        
        for record in data:
            # Remove duplicates (based on ID)
            if record not in cleansed:
                # Handle missing values
                cleaned_record = {
                    k: (v if v is not None else "UNKNOWN")
                    for k, v in record.items()
                }
                
                # Standardize text (uppercase names)
                if "customer_name" in cleaned_record:
                    cleaned_record["customer_name"] = cleaned_record["customer_name"].upper()
                
                cleansed.append(cleaned_record)
        
        return cleansed
    
    def validate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Data Validation: Check quality and constraints
        """
        validation_results = {
            "total_records": len(data),
            "valid_records": 0,
            "invalid_records": 0,
            "validation_errors": []
        }
        
        for record in data:
            is_valid = True
            
            # Validate order_total is positive
            if "order_total" in record:
                if float(record["order_total"]) <= 0:
                    is_valid = False
                    validation_results["validation_errors"].append(
                        f"Invalid order_total for record {record.get('id', 'unknown')}"
                    )
            
            # Validate required fields
            required_fields = ["id", "customer_name"]
            for field in required_fields:
                if field not in record or record[field] == "UNKNOWN":
                    is_valid = False
                    validation_results["validation_errors"].append(
                        f"Missing required field '{field}' in record {record.get('id', 'unknown')}"
                    )
            
            if is_valid:
                validation_results["valid_records"] += 1
            else:
                validation_results["invalid_records"] += 1
        
        return validation_results
    
    def enrich_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Data Enrichment: Add derived fields and lookups
        """
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            # Add derived field: order_category based on total
            if "order_total" in record:
                total = float(record["order_total"])
                if total < 100:
                    enriched_record["order_category"] = "SMALL"
                elif total < 200:
                    enriched_record["order_category"] = "MEDIUM"
                else:
                    enriched_record["order_category"] = "LARGE"
            
            # Add metadata
            enriched_record["processed_date"] = datetime.now().isoformat()
            enriched_record["etl_batch_id"] = "BATCH_001"
            
            enriched.append(enriched_record)
        
        return enriched
    
    def aggregate_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Data Aggregation: Summarize data for analytics
        """
        aggregations = {
            "total_orders": len(data),
            "total_revenue": sum(float(r.get("order_total", 0)) for r in data),
            "average_order_value": 0,
            "orders_by_category": {}
        }
        
        if aggregations["total_orders"] > 0:
            aggregations["average_order_value"] = (
                aggregations["total_revenue"] / aggregations["total_orders"]
            )
        
        # Group by category
        for record in data:
            category = record.get("order_category", "UNKNOWN")
            aggregations["orders_by_category"][category] = (
                aggregations["orders_by_category"].get(category, 0) + 1
            )
        
        return aggregations

# ============================================================================
# LOADER ENGINE (Load Stage)
# ============================================================================

class LoaderEngine:
    """
    Loader: Writes transformed data to target destination
    
    Loading Strategies:
    1. Full Load: Truncate and reload all data
    2. Incremental Load: Append new records
    3. Upsert: Insert new, update existing (merge)
    4. SCD Type 2: Maintain history (slowly changing dimensions)
    """
    
    def __init__(self):
        self.data_warehouse = {
            "fact_orders": [],
            "dim_customers": [],
            "agg_daily_sales": []
        }
    
    def load_full(self, table: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Full Load: Replace all data in target table
        """
        # Truncate existing data
        self.data_warehouse[table] = []
        
        # Load all records
        self.data_warehouse[table] = data.copy()
        
        return {
            "load_type": "FULL",
            "table": table,
            "records_loaded": len(data),
            "status": "SUCCESS"
        }
    
    def load_incremental(self, table: str, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Incremental Load: Append new records to existing data
        """
        # Append new records
        self.data_warehouse[table].extend(data)
        
        return {
            "load_type": "INCREMENTAL",
            "table": table,
            "records_loaded": len(data),
            "total_records": len(self.data_warehouse[table]),
            "status": "SUCCESS"
        }
    
    def load_upsert(self, table: str, data: List[Dict[str, Any]], key_field: str) -> Dict[str, Any]:
        """
        Upsert Load: Insert new records, update existing (merge)
        """
        inserted = 0
        updated = 0
        
        for new_record in data:
            # Find existing record by key
            existing_index = None
            for i, existing_record in enumerate(self.data_warehouse[table]):
                if existing_record.get(key_field) == new_record.get(key_field):
                    existing_index = i
                    break
            
            if existing_index is not None:
                # Update existing record
                self.data_warehouse[table][existing_index] = new_record
                updated += 1
            else:
                # Insert new record
                self.data_warehouse[table].append(new_record)
                inserted += 1
        
        return {
            "load_type": "UPSERT",
            "table": table,
            "records_inserted": inserted,
            "records_updated": updated,
            "status": "SUCCESS"
        }
    
    def get_warehouse_summary(self) -> Dict[str, Any]:
        """Get summary of data warehouse"""
        return {
            table: len(records)
            for table, records in self.data_warehouse.items()
        }

# Create shared instances
extractor = ExtractorEngine()
transformer = TransformationEngine()
loader = LoaderEngine()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def extract_stage(state: ETLState) -> ETLState:
    """Stage 1: Extract data from sources"""
    source_config = state["source_config"]
    
    # Create data source
    source = DataSource(
        source_id=source_config.get("source_id", "orders_db"),
        source_type=source_config.get("source_type", "database"),
        connection_string=source_config.get("connection", "postgresql://..."),
        query_config=source_config.get("query", {})
    )
    
    # Extract data based on source type
    if source.source_type == "database":
        extracted_data = extractor.extract_from_database(source)
    elif source.source_type == "api":
        extracted_data = extractor.extract_from_api(source)
    elif source.source_type == "file":
        extracted_data = extractor.extract_from_file(source)
    else:
        extracted_data = []
    
    return {
        "extracted_data": extracted_data,
        "etl_metadata": {
            "extract_timestamp": datetime.now().isoformat(),
            "records_extracted": len(extracted_data)
        },
        "messages": [f"[EXTRACT] Extracted {len(extracted_data)} records from {source.source_id}"]
    }

def transform_stage(state: ETLState) -> ETLState:
    """Stage 2: Transform data according to business rules"""
    extracted_data = state.get("extracted_data", [])
    
    # Step 1: Cleanse data
    cleansed_data = transformer.cleanse_data(extracted_data)
    
    # Step 2: Validate data
    validation_results = transformer.validate_data(cleansed_data)
    
    # Step 3: Enrich data (add derived fields)
    enriched_data = transformer.enrich_data(cleansed_data)
    
    # Step 4: Aggregate data (for reporting)
    aggregations = transformer.aggregate_data(enriched_data)
    
    return {
        "transformed_data": enriched_data,
        "validation_results": validation_results,
        "etl_metadata": {
            "transform_timestamp": datetime.now().isoformat(),
            "records_transformed": len(enriched_data),
            "aggregations": aggregations
        },
        "messages": [
            f"[TRANSFORM] Cleansed {len(cleansed_data)} records",
            f"[TRANSFORM] Validated: {validation_results['valid_records']} valid, {validation_results['invalid_records']} invalid",
            f"[TRANSFORM] Enriched {len(enriched_data)} records with derived fields"
        ]
    }

def load_stage(state: ETLState) -> ETLState:
    """Stage 3: Load transformed data into target warehouse"""
    transformed_data = state.get("transformed_data", [])
    
    # Load fact table (incremental)
    load_result = loader.load_incremental("fact_orders", transformed_data)
    
    # Get warehouse summary
    warehouse_summary = loader.get_warehouse_summary()
    
    return {
        "loaded_data": load_result,
        "etl_metadata": {
            "load_timestamp": datetime.now().isoformat(),
            "records_loaded": load_result["records_loaded"],
            "warehouse_summary": warehouse_summary
        },
        "messages": [
            f"[LOAD] Loaded {load_result['records_loaded']} records into {load_result['table']}",
            f"[LOAD] Warehouse summary: {warehouse_summary}"
        ]
    }

# ============================================================================
# BUILD THE ETL GRAPH
# ============================================================================

def create_etl_graph():
    """
    Create a StateGraph demonstrating ETL pattern.
    
    Flow:
    1. Extract: Pull data from sources
    2. Transform: Cleanse, validate, enrich
    3. Load: Write to data warehouse
    """
    
    workflow = StateGraph(ETLState)
    
    # Add ETL stages
    workflow.add_node("extract", extract_stage)
    workflow.add_node("transform", transform_stage)
    workflow.add_node("load", load_stage)
    
    # Define ETL pipeline flow
    workflow.add_edge(START, "extract")
    workflow.add_edge("extract", "transform")
    workflow.add_edge("transform", "load")
    workflow.add_edge("load", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ETL (Extract-Transform-Load) MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Basic ETL Pipeline
    print("\n" + "=" * 80)
    print("Example 1: Basic ETL Pipeline (Database → Warehouse)")
    print("=" * 80)
    
    etl_graph = create_etl_graph()
    
    initial_state: ETLState = {
        "source_config": {
            "source_id": "orders_database",
            "source_type": "database",
            "connection": "postgresql://localhost:5432/orders",
            "query": {"table": "orders", "where": "order_date >= '2024-01-15'"}
        },
        "extracted_data": None,
        "transformed_data": None,
        "loaded_data": None,
        "validation_results": None,
        "etl_metadata": {},
        "messages": []
    }
    
    result = etl_graph.invoke(initial_state)
    
    print("\nETL Execution Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nExtracted Data Sample:")
    for record in result["extracted_data"][:2]:
        print(f"  {record}")
    
    print("\nTransformed Data Sample:")
    for record in result["transformed_data"][:2]:
        print(f"  {record}")
    
    print("\nValidation Results:")
    print(f"  Valid: {result['validation_results']['valid_records']}")
    print(f"  Invalid: {result['validation_results']['invalid_records']}")
    
    print("\nLoad Results:")
    print(f"  Table: {result['loaded_data']['table']}")
    print(f"  Records Loaded: {result['loaded_data']['records_loaded']}")
    
    # Example 2: Data Transformations
    print("\n" + "=" * 80)
    print("Example 2: Transformation Details")
    print("=" * 80)
    
    print("\nTransformation Steps:")
    print("  1. Cleansing: Remove duplicates, handle nulls, standardize formats")
    print("  2. Validation: Check data quality and business rules")
    print("  3. Enrichment: Add derived fields (order_category)")
    print("  4. Aggregation: Summarize for analytics")
    
    print("\nAggregations:")
    agg = result["etl_metadata"]["aggregations"]
    print(f"  Total Orders: {agg['total_orders']}")
    print(f"  Total Revenue: ${agg['total_revenue']:.2f}")
    print(f"  Average Order Value: ${agg['average_order_value']:.2f}")
    print(f"  Orders by Category: {agg['orders_by_category']}")
    
    # Example 3: ETL vs ELT
    print("\n" + "=" * 80)
    print("Example 3: ETL vs ELT Comparison")
    print("=" * 80)
    
    print("\nETL (Extract-Transform-Load):")
    print("  Flow: Source → Transform → Warehouse")
    print("  Transform: Before loading (in ETL server)")
    print("  Best for: Complex transformations, data quality")
    print("  Tools: Informatica, Talend, SSIS, Apache Airflow")
    print("  Use case: Traditional data warehousing")
    
    print("\nELT (Extract-Load-Transform):")
    print("  Flow: Source → Warehouse → Transform")
    print("  Transform: After loading (in warehouse using SQL)")
    print("  Best for: Cloud warehouses (Snowflake, BigQuery)")
    print("  Tools: dbt, Matillion, Fivetran")
    print("  Use case: Modern cloud data platforms")
    
    # Example 4: Loading Strategies
    print("\n" + "=" * 80)
    print("Example 4: Loading Strategies")
    print("=" * 80)
    
    print("\n1. Full Load:")
    full_result = loader.load_full("dim_customers", [
        {"customer_id": 1, "name": "John Doe"},
        {"customer_id": 2, "name": "Jane Smith"}
    ])
    print(f"   {full_result}")
    
    print("\n2. Incremental Load:")
    inc_result = loader.load_incremental("dim_customers", [
        {"customer_id": 3, "name": "Bob Johnson"}
    ])
    print(f"   {inc_result}")
    
    print("\n3. Upsert Load:")
    upsert_result = loader.load_upsert("dim_customers", [
        {"customer_id": 1, "name": "John Doe Updated"},
        {"customer_id": 4, "name": "Alice Brown"}
    ], "customer_id")
    print(f"   {upsert_result}")
    
    print("\nWarehouse Summary:")
    print(f"   {loader.get_warehouse_summary()}")
    
    print("\n" + "=" * 80)
    print("ETL Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • ETL is fundamental for data warehousing and analytics
  • Extract: Pull data from multiple sources (databases, APIs, files)
  • Transform: Cleanse, validate, enrich, aggregate data
  • Load: Write to target warehouse (full, incremental, upsert)
  • Data Quality: Validation ensures accurate analytics
  • Batch Processing: Typically scheduled (nightly, hourly)
  • Use ETL for traditional warehouses, ELT for cloud platforms
    """)

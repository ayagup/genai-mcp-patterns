"""
Pattern 186: Data Aggregation MCP Pattern

This pattern demonstrates aggregating data by grouping, summarizing, and computing
metrics across datasets. Aggregation is fundamental for analytics, reporting, and
deriving insights from large volumes of data.

Key Concepts:
1. Group By: Partition data into groups
2. Aggregate Functions: SUM, COUNT, AVG, MIN, MAX
3. Multi-Level Aggregation: Hierarchical grouping
4. Rollup: Aggregate at multiple levels
5. Cube: All possible combinations of dimensions
6. Window Functions: Aggregations over sliding windows
7. Pre-Aggregation: Compute and store aggregates

Aggregation Types:
- Simple Aggregation: COUNT(*), SUM(amount)
- Grouped Aggregation: GROUP BY category
- Multi-Dimensional: GROUP BY dim1, dim2, dim3
- Time-Based: GROUP BY date, hour
- Nested Aggregation: Aggregate of aggregates
- Incremental: Update aggregates as data arrives

Common Aggregate Functions:
- COUNT: Number of records
- SUM: Total of values
- AVG: Average value
- MIN/MAX: Minimum/maximum value
- STDDEV: Standard deviation
- PERCENTILE: 50th, 95th, 99th percentiles
- DISTINCT: Unique values

Benefits:
- Insights: Derive meaningful metrics from raw data
- Performance: Pre-aggregated data faster to query
- Data Reduction: Summarize large datasets
- Reporting: Generate dashboards and reports
- Trend Analysis: Identify patterns over time

Trade-offs:
- Memory Usage: Aggregations can consume significant memory
- Accuracy: Approximate aggregations trade precision for speed
- Complexity: Multi-dimensional aggregations complex
- Staleness: Pre-aggregated data may be outdated

Use Cases:
- Analytics Dashboards: Sales by region, revenue by product
- Metrics Collection: Request counts, error rates
- Business Intelligence: KPIs, trends, comparisons
- Time-Series Analysis: Daily/hourly metrics
- OLAP: Multi-dimensional analytics
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from enum import Enum
import statistics

# Define the state for data aggregation
class DataAggregationState(TypedDict):
    """State for data aggregation"""
    raw_data: List[Dict[str, Any]]
    simple_aggregates: Optional[Dict[str, Any]]
    grouped_aggregates: Optional[List[Dict[str, Any]]]
    multi_dim_aggregates: Optional[List[Dict[str, Any]]]
    time_series_aggregates: Optional[List[Dict[str, Any]]]
    aggregation_metadata: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# AGGREGATION FUNCTIONS
# ============================================================================

class AggregateFunction(Enum):
    """Common aggregate functions"""
    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    MEDIAN = "median"
    DISTINCT_COUNT = "distinct_count"

class SimpleAggregator:
    """
    Simple Aggregation: Compute basic statistics over entire dataset
    
    Functions: COUNT, SUM, AVG, MIN, MAX, STDDEV
    """
    
    def aggregate(self, data: List[Dict[str, Any]], value_field: str) -> Dict[str, Any]:
        """Compute simple aggregates"""
        if not data:
            return {"error": "No data"}
        
        values = [record.get(value_field, 0) for record in data]
        
        aggregates = {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values) if values else 0,
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
        }
        
        # Standard deviation
        if len(values) > 1:
            aggregates["stddev"] = statistics.stdev(values)
        else:
            aggregates["stddev"] = 0
        
        # Median
        aggregates["median"] = statistics.median(values) if values else 0
        
        return aggregates

class GroupedAggregator:
    """
    Grouped Aggregation: Group by dimension and aggregate
    
    Example: GROUP BY category, compute SUM(sales) for each category
    """
    
    def group_by(self, data: List[Dict[str, Any]], 
                 group_field: str, 
                 value_field: str) -> List[Dict[str, Any]]:
        """Group data and compute aggregates"""
        # Group data
        groups = defaultdict(list)
        
        for record in data:
            group_key = record.get(group_field, "UNKNOWN")
            groups[group_key].append(record.get(value_field, 0))
        
        # Compute aggregates for each group
        result = []
        
        for group_key, values in groups.items():
            aggregate = {
                "group": group_key,
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            }
            result.append(aggregate)
        
        # Sort by sum descending
        result.sort(key=lambda x: x["sum"], reverse=True)
        
        return result

class MultiDimensionalAggregator:
    """
    Multi-Dimensional Aggregation: Group by multiple dimensions
    
    Example: GROUP BY region, category, month
    Produces aggregates for all combinations
    """
    
    def aggregate_multi_dim(self, data: List[Dict[str, Any]], 
                           dimensions: List[str],
                           value_field: str) -> List[Dict[str, Any]]:
        """Aggregate across multiple dimensions"""
        # Group by combination of dimensions
        groups = defaultdict(list)
        
        for record in data:
            # Create composite key from all dimensions
            key_parts = [str(record.get(dim, "UNKNOWN")) for dim in dimensions]
            group_key = "|".join(key_parts)
            groups[group_key].append(record.get(value_field, 0))
        
        # Compute aggregates
        result = []
        
        for group_key, values in groups.items():
            # Split key back into dimensions
            dim_values = group_key.split("|")
            
            aggregate = {
                **{dim: val for dim, val in zip(dimensions, dim_values)},
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values) if values else 0
            }
            result.append(aggregate)
        
        return result
    
    def rollup(self, data: List[Dict[str, Any]], 
               dimensions: List[str],
               value_field: str) -> List[Dict[str, Any]]:
        """
        ROLLUP: Hierarchical aggregation
        
        Example: ROLLUP(region, category, product)
        Produces:
        - (region, category, product)
        - (region, category)
        - (region)
        - ()  # Grand total
        """
        all_aggregates = []
        
        # Aggregate at each level of hierarchy
        for level in range(len(dimensions) + 1):
            dims_at_level = dimensions[:level] if level > 0 else []
            
            if not dims_at_level:
                # Grand total
                values = [record.get(value_field, 0) for record in data]
                all_aggregates.append({
                    "level": "TOTAL",
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values) if values else 0
                })
            else:
                # Aggregate at this level
                level_aggs = self.aggregate_multi_dim(data, dims_at_level, value_field)
                for agg in level_aggs:
                    agg["level"] = f"Level {level}"
                    all_aggregates.append(agg)
        
        return all_aggregates

class TimeSeriesAggregator:
    """
    Time-Series Aggregation: Aggregate over time periods
    
    Time Buckets:
    - Hourly: GROUP BY date_trunc('hour', timestamp)
    - Daily: GROUP BY date_trunc('day', timestamp)
    - Monthly: GROUP BY date_trunc('month', timestamp)
    """
    
    def aggregate_by_time(self, data: List[Dict[str, Any]], 
                         time_field: str,
                         value_field: str,
                         bucket_size: str = "day") -> List[Dict[str, Any]]:
        """
        Aggregate data into time buckets
        
        bucket_size: 'hour', 'day', 'month'
        """
        # Group by time bucket
        groups = defaultdict(list)
        
        for record in data:
            timestamp_str = record.get(time_field, "")
            
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    
                    # Determine bucket
                    if bucket_size == "hour":
                        bucket = timestamp.strftime("%Y-%m-%d %H:00")
                    elif bucket_size == "day":
                        bucket = timestamp.strftime("%Y-%m-%d")
                    elif bucket_size == "month":
                        bucket = timestamp.strftime("%Y-%m")
                    else:
                        bucket = timestamp.strftime("%Y-%m-%d")
                    
                    groups[bucket].append(record.get(value_field, 0))
                except:
                    pass
        
        # Compute aggregates for each bucket
        result = []
        
        for bucket, values in sorted(groups.items()):
            aggregate = {
                "time_bucket": bucket,
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values) if values else 0,
                "min": min(values) if values else 0,
                "max": max(values) if values else 0
            }
            result.append(aggregate)
        
        return result

class IncrementalAggregator:
    """
    Incremental Aggregation: Update aggregates as new data arrives
    
    Useful for streaming data and real-time dashboards
    """
    
    def __init__(self):
        self.aggregates = {
            "count": 0,
            "sum": 0,
            "min": float('inf'),
            "max": float('-inf')
        }
    
    def update(self, value: float):
        """Update aggregates with new value"""
        self.aggregates["count"] += 1
        self.aggregates["sum"] += value
        self.aggregates["min"] = min(self.aggregates["min"], value)
        self.aggregates["max"] = max(self.aggregates["max"], value)
    
    def get_aggregates(self) -> Dict[str, Any]:
        """Get current aggregates"""
        agg = self.aggregates.copy()
        
        if agg["count"] > 0:
            agg["avg"] = agg["sum"] / agg["count"]
        else:
            agg["avg"] = 0
        
        return agg
    
    def reset(self):
        """Reset aggregates"""
        self.aggregates = {
            "count": 0,
            "sum": 0,
            "min": float('inf'),
            "max": float('-inf')
        }

# Create aggregator instances
simple_agg = SimpleAggregator()
grouped_agg = GroupedAggregator()
multi_dim_agg = MultiDimensionalAggregator()
time_series_agg = TimeSeriesAggregator()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def compute_simple_aggregates(state: DataAggregationState) -> DataAggregationState:
    """Compute simple aggregates over entire dataset"""
    raw_data = state["raw_data"]
    
    aggregates = simple_agg.aggregate(raw_data, "amount")
    
    return {
        "simple_aggregates": aggregates,
        "messages": [f"[Simple Agg] count={aggregates['count']}, sum={aggregates['sum']}, avg={aggregates['avg']:.2f}"]
    }

def compute_grouped_aggregates(state: DataAggregationState) -> DataAggregationState:
    """Compute grouped aggregates (GROUP BY)"""
    raw_data = state["raw_data"]
    
    # Group by category
    grouped = grouped_agg.group_by(raw_data, "category", "amount")
    
    return {
        "grouped_aggregates": grouped,
        "messages": [f"[Grouped Agg] {len(grouped)} groups created"]
    }

def compute_multidim_aggregates(state: DataAggregationState) -> DataAggregationState:
    """Compute multi-dimensional aggregates"""
    raw_data = state["raw_data"]
    
    # Aggregate by multiple dimensions
    multi_dim = multi_dim_agg.aggregate_multi_dim(
        raw_data, 
        ["region", "category"], 
        "amount"
    )
    
    return {
        "multi_dim_aggregates": multi_dim,
        "messages": [f"[Multi-Dim Agg] {len(multi_dim)} dimension combinations"]
    }

def compute_timeseries_aggregates(state: DataAggregationState) -> DataAggregationState:
    """Compute time-series aggregates"""
    raw_data = state["raw_data"]
    
    # Aggregate by day
    time_series = time_series_agg.aggregate_by_time(
        raw_data,
        "timestamp",
        "amount",
        bucket_size="day"
    )
    
    return {
        "time_series_aggregates": time_series,
        "messages": [f"[Time-Series Agg] {len(time_series)} time buckets"]
    }

# ============================================================================
# BUILD THE AGGREGATION GRAPH
# ============================================================================

def create_aggregation_graph():
    """
    Create a StateGraph demonstrating data aggregation pattern.
    
    Flow:
    1. Simple Aggregates: COUNT, SUM, AVG over entire dataset
    2. Grouped Aggregates: GROUP BY category
    3. Multi-Dim Aggregates: GROUP BY region, category
    4. Time-Series Aggregates: GROUP BY day
    """
    
    workflow = StateGraph(DataAggregationState)
    
    # Add aggregation nodes
    workflow.add_node("simple", compute_simple_aggregates)
    workflow.add_node("grouped", compute_grouped_aggregates)
    workflow.add_node("multidim", compute_multidim_aggregates)
    workflow.add_node("timeseries", compute_timeseries_aggregates)
    
    # Define aggregation flow
    workflow.add_edge(START, "simple")
    workflow.add_edge("simple", "grouped")
    workflow.add_edge("grouped", "multidim")
    workflow.add_edge("multidim", "timeseries")
    workflow.add_edge("timeseries", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Aggregation MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Comprehensive Data Aggregation
    print("\n" + "=" * 80)
    print("Example 1: Multi-Level Data Aggregation")
    print("=" * 80)
    
    aggregation_graph = create_aggregation_graph()
    
    # Sample sales data
    sales_data = [
        {"id": 1, "category": "Electronics", "region": "North", "amount": 1000, "timestamp": "2024-01-15T10:00:00"},
        {"id": 2, "category": "Electronics", "region": "South", "amount": 1500, "timestamp": "2024-01-15T11:00:00"},
        {"id": 3, "category": "Clothing", "region": "North", "amount": 500, "timestamp": "2024-01-15T12:00:00"},
        {"id": 4, "category": "Electronics", "region": "North", "amount": 2000, "timestamp": "2024-01-16T10:00:00"},
        {"id": 5, "category": "Clothing", "region": "South", "amount": 700, "timestamp": "2024-01-16T11:00:00"},
        {"id": 6, "category": "Food", "region": "North", "amount": 300, "timestamp": "2024-01-16T12:00:00"},
        {"id": 7, "category": "Food", "region": "South", "amount": 400, "timestamp": "2024-01-17T10:00:00"},
    ]
    
    initial_state: DataAggregationState = {
        "raw_data": sales_data,
        "simple_aggregates": None,
        "grouped_aggregates": None,
        "multi_dim_aggregates": None,
        "time_series_aggregates": None,
        "aggregation_metadata": {},
        "messages": []
    }
    
    result = aggregation_graph.invoke(initial_state)
    
    print("\nAggregation Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\n1. Simple Aggregates (Entire Dataset):")
    simple = result["simple_aggregates"]
    print(f"  Count: {simple['count']}")
    print(f"  Sum: ${simple['sum']}")
    print(f"  Average: ${simple['avg']:.2f}")
    print(f"  Min: ${simple['min']}, Max: ${simple['max']}")
    print(f"  Std Dev: ${simple['stddev']:.2f}")
    
    print("\n2. Grouped Aggregates (By Category):")
    for group in result["grouped_aggregates"]:
        print(f"  {group['group']}: count={group['count']}, sum=${group['sum']}, avg=${group['avg']:.2f}")
    
    print("\n3. Multi-Dimensional Aggregates (By Region + Category):")
    for agg in result["multi_dim_aggregates"]:
        print(f"  {agg['region']} + {agg['category']}: count={agg['count']}, sum=${agg['sum']}, avg=${agg['avg']:.2f}")
    
    print("\n4. Time-Series Aggregates (By Day):")
    for ts in result["time_series_aggregates"]:
        print(f"  {ts['time_bucket']}: count={ts['count']}, sum=${ts['sum']}, avg=${ts['avg']:.2f}")
    
    # Example 2: ROLLUP Aggregation
    print("\n" + "=" * 80)
    print("Example 2: ROLLUP Hierarchical Aggregation")
    print("=" * 80)
    
    rollup_result = multi_dim_agg.rollup(sales_data, ["region", "category"], "amount")
    
    print("\nROLLUP Results (Hierarchical Subtotals):")
    for agg in rollup_result:
        if agg["level"] == "TOTAL":
            print(f"\n{agg['level']}: sum=${agg['sum']}, avg=${agg['avg']:.2f}")
        elif "region" in agg and "category" in agg:
            print(f"  {agg['level']} ({agg['region']} + {agg['category']}): sum=${agg['sum']}")
        elif "region" in agg:
            print(f"  {agg['level']} ({agg['region']}): sum=${agg['sum']}")
    
    # Example 3: Incremental Aggregation
    print("\n" + "=" * 80)
    print("Example 3: Incremental Aggregation (Streaming)")
    print("=" * 80)
    
    incremental = IncrementalAggregator()
    
    print("\nAdding values incrementally:")
    for i, record in enumerate(sales_data[:5], 1):
        incremental.update(record["amount"])
        current_agg = incremental.get_aggregates()
        print(f"  After {i} values: count={current_agg['count']}, sum=${current_agg['sum']}, avg=${current_agg['avg']:.2f}")
    
    # Example 4: Aggregation Use Cases
    print("\n" + "=" * 80)
    print("Example 4: Aggregation Use Cases")
    print("=" * 80)
    
    print("\n1. Analytics Dashboards:")
    print("   SELECT region, SUM(sales) FROM orders GROUP BY region")
    print("   Show total sales by region on dashboard")
    
    print("\n2. Business Intelligence:")
    print("   SELECT category, AVG(price) FROM products GROUP BY category")
    print("   Compare average prices across categories")
    
    print("\n3. Metrics Collection:")
    print("   SELECT date, COUNT(*) FROM requests GROUP BY date")
    print("   Track daily request counts")
    
    print("\n4. Time-Series Analysis:")
    print("   SELECT hour, SUM(revenue) FROM transactions GROUP BY hour")
    print("   Identify peak sales hours")
    
    print("\n5. OLAP Cubes:")
    print("   SELECT region, category, product, SUM(sales)")
    print("   FROM orders GROUP BY CUBE(region, category, product)")
    print("   Multi-dimensional analysis")
    
    print("\n" + "=" * 80)
    print("Data Aggregation Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Aggregation summarizes large datasets into insights
  • Common functions: COUNT, SUM, AVG, MIN, MAX, STDDEV
  • GROUP BY: Partition data and aggregate each group
  • Multi-dimensional: Aggregate across multiple dimensions
  • ROLLUP: Hierarchical subtotals
  • Time-series: Aggregate into time buckets (hourly, daily)
  • Incremental: Update aggregates as data streams in
  • Use for: dashboards, BI, metrics, analytics, reporting
    """)

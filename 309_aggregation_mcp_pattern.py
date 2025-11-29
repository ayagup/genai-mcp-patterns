"""
Aggregation MCP Pattern

This pattern demonstrates data aggregation in an agentic MCP system.
Aggregation combines data from multiple sources or records into summary information,
computing statistics, totals, averages, and other aggregate values.

Use cases:
- Analytics and reporting
- Dashboard creation
- Data summarization
- Statistical analysis
- Business intelligence
- Time-series aggregation
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime


# Define the state for aggregation
class AggregationState(TypedDict):
    """State for tracking aggregation process"""
    messages: Annotated[List[str], add]
    raw_data: List[Dict[str, Any]]
    aggregation_rules: List[Dict[str, Any]]
    aggregated_data: Dict[str, Any]
    time_series_aggregation: Dict[str, Any]
    hierarchical_aggregation: Dict[str, Any]
    statistical_summary: Dict[str, Any]
    report: str


class DataAggregator:
    """Class to handle various aggregation operations"""
    
    def __init__(self):
        self.aggregation_functions = {
            'sum': self._sum,
            'avg': self._avg,
            'count': self._count,
            'min': self._min,
            'max': self._max,
            'distinct_count': self._distinct_count,
            'median': self._median,
            'stddev': self._stddev
        }
    
    def _sum(self, values: List[float]) -> float:
        """Calculate sum"""
        return sum(values) if values else 0
    
    def _avg(self, values: List[float]) -> float:
        """Calculate average"""
        return sum(values) / len(values) if values else 0
    
    def _count(self, values: List[Any]) -> int:
        """Count values"""
        return len(values)
    
    def _min(self, values: List[float]) -> float:
        """Find minimum"""
        return min(values) if values else 0
    
    def _max(self, values: List[float]) -> float:
        """Find maximum"""
        return max(values) if values else 0
    
    def _distinct_count(self, values: List[Any]) -> int:
        """Count distinct values"""
        return len(set(str(v) for v in values))
    
    def _median(self, values: List[float]) -> float:
        """Calculate median"""
        if not values:
            return 0
        sorted_values = sorted(values)
        n = len(sorted_values)
        if n % 2 == 0:
            return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
        else:
            return sorted_values[n//2]
    
    def _stddev(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values or len(values) < 2:
            return 0
        mean = self._avg(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def aggregate_by_group(self, data: List[Dict], group_by: str, 
                          aggregations: List[Dict]) -> Dict[str, Any]:
        """Aggregate data by grouping"""
        groups = {}
        
        # Group data
        for record in data:
            group_key = record.get(group_by, 'unknown')
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(record)
        
        # Aggregate each group
        result = {}
        for group_key, group_records in groups.items():
            result[group_key] = {}
            
            for agg in aggregations:
                field = agg['field']
                function = agg['function']
                
                values = [record.get(field) for record in group_records 
                         if record.get(field) is not None]
                
                if function in self.aggregation_functions:
                    result[group_key][f"{function}_{field}"] = \
                        self.aggregation_functions[function](values)
        
        return result


class TimeSeriesAggregator:
    """Aggregate time-series data"""
    
    def aggregate_by_period(self, data: List[Dict], date_field: str, 
                           period: str, aggregations: List[Dict]) -> Dict[str, Any]:
        """Aggregate time-series data by period (day, week, month)"""
        periods = {}
        
        for record in data:
            date_str = record.get(date_field, '')
            if not date_str:
                continue
            
            # Parse date and determine period
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                
                if period == 'day':
                    period_key = date_obj.strftime('%Y-%m-%d')
                elif period == 'week':
                    week_start = date_obj - datetime.timedelta(days=date_obj.weekday())
                    period_key = week_start.strftime('%Y-W%U')
                elif period == 'month':
                    period_key = date_obj.strftime('%Y-%m')
                elif period == 'year':
                    period_key = date_obj.strftime('%Y')
                else:
                    period_key = date_str
                
                if period_key not in periods:
                    periods[period_key] = []
                periods[period_key].append(record)
            except:
                continue
        
        # Aggregate each period
        aggregator = DataAggregator()
        result = {}
        
        for period_key, period_records in periods.items():
            result[period_key] = {'record_count': len(period_records)}
            
            for agg in aggregations:
                field = agg['field']
                function = agg['function']
                
                values = [record.get(field) for record in period_records 
                         if record.get(field) is not None]
                
                if function in aggregator.aggregation_functions:
                    result[period_key][f"{function}_{field}"] = \
                        aggregator.aggregation_functions[function](values)
        
        return result


class HierarchicalAggregator:
    """Aggregate data hierarchically"""
    
    def aggregate_hierarchy(self, data: List[Dict], hierarchy: List[str], 
                           aggregations: List[Dict]) -> Dict[str, Any]:
        """Aggregate data across multiple hierarchical levels"""
        aggregator = DataAggregator()
        result = {}
        
        def build_hierarchy(records, level_idx):
            if level_idx >= len(hierarchy):
                return None
            
            level_field = hierarchy[level_idx]
            groups = {}
            
            for record in records:
                group_key = record.get(level_field, 'unknown')
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(record)
            
            level_result = {}
            for group_key, group_records in groups.items():
                level_result[group_key] = {
                    'count': len(group_records)
                }
                
                # Aggregate at this level
                for agg in aggregations:
                    field = agg['field']
                    function = agg['function']
                    
                    values = [record.get(field) for record in group_records 
                             if record.get(field) is not None]
                    
                    if function in aggregator.aggregation_functions:
                        level_result[group_key][f"{function}_{field}"] = \
                            aggregator.aggregation_functions[function](values)
                
                # Recurse to next level
                if level_idx + 1 < len(hierarchy):
                    level_result[group_key]['children'] = \
                        build_hierarchy(group_records, level_idx + 1)
            
            return level_result
        
        result = build_hierarchy(data, 0)
        return result


# Agent functions
def initialize_aggregation_agent(state: AggregationState) -> AggregationState:
    """Initialize aggregation with sample data"""
    
    # Sample sales transaction data
    raw_data = [
        {'transaction_id': 'T001', 'date': '2024-01-05', 'customer_id': 'C001', 'product': 'Laptop', 
         'category': 'Electronics', 'region': 'North', 'amount': 1200.00, 'quantity': 1},
        {'transaction_id': 'T002', 'date': '2024-01-07', 'customer_id': 'C002', 'product': 'Mouse', 
         'category': 'Electronics', 'region': 'South', 'amount': 25.00, 'quantity': 2},
        {'transaction_id': 'T003', 'date': '2024-01-10', 'customer_id': 'C001', 'product': 'Desk', 
         'category': 'Furniture', 'region': 'North', 'amount': 350.00, 'quantity': 1},
        {'transaction_id': 'T004', 'date': '2024-01-12', 'customer_id': 'C003', 'product': 'Chair', 
         'category': 'Furniture', 'region': 'East', 'amount': 200.00, 'quantity': 4},
        {'transaction_id': 'T005', 'date': '2024-01-15', 'customer_id': 'C002', 'product': 'Monitor', 
         'category': 'Electronics', 'region': 'South', 'amount': 400.00, 'quantity': 1},
        {'transaction_id': 'T006', 'date': '2024-02-03', 'customer_id': 'C004', 'product': 'Keyboard', 
         'category': 'Electronics', 'region': 'West', 'amount': 75.00, 'quantity': 3},
        {'transaction_id': 'T007', 'date': '2024-02-08', 'customer_id': 'C001', 'product': 'Tablet', 
         'category': 'Electronics', 'region': 'North', 'amount': 600.00, 'quantity': 1},
        {'transaction_id': 'T008', 'date': '2024-02-12', 'customer_id': 'C005', 'product': 'Bookshelf', 
         'category': 'Furniture', 'region': 'South', 'amount': 180.00, 'quantity': 2},
        {'transaction_id': 'T009', 'date': '2024-02-18', 'customer_id': 'C003', 'product': 'Lamp', 
         'category': 'Furniture', 'region': 'East', 'amount': 45.00, 'quantity': 5},
        {'transaction_id': 'T010', 'date': '2024-02-25', 'customer_id': 'C002', 'product': 'Headphones', 
         'category': 'Electronics', 'region': 'South', 'amount': 150.00, 'quantity': 2},
        {'transaction_id': 'T011', 'date': '2024-03-05', 'customer_id': 'C006', 'product': 'Webcam', 
         'category': 'Electronics', 'region': 'North', 'amount': 90.00, 'quantity': 1},
        {'transaction_id': 'T012', 'date': '2024-03-10', 'customer_id': 'C004', 'product': 'Sofa', 
         'category': 'Furniture', 'region': 'West', 'amount': 800.00, 'quantity': 1}
    ]
    
    return {
        **state,
        'raw_data': raw_data,
        'messages': state['messages'] + [f'Initialized aggregation with {len(raw_data)} transaction records']
    }


def define_aggregation_rules_agent(state: AggregationState) -> AggregationState:
    """Define aggregation rules and operations"""
    
    aggregation_rules = [
        {
            'name': 'Sales by Category',
            'group_by': 'category',
            'aggregations': [
                {'field': 'amount', 'function': 'sum'},
                {'field': 'amount', 'function': 'avg'},
                {'field': 'quantity', 'function': 'sum'},
                {'field': 'transaction_id', 'function': 'count'}
            ]
        },
        {
            'name': 'Sales by Region',
            'group_by': 'region',
            'aggregations': [
                {'field': 'amount', 'function': 'sum'},
                {'field': 'amount', 'function': 'min'},
                {'field': 'amount', 'function': 'max'},
                {'field': 'customer_id', 'function': 'distinct_count'}
            ]
        },
        {
            'name': 'Customer Summary',
            'group_by': 'customer_id',
            'aggregations': [
                {'field': 'amount', 'function': 'sum'},
                {'field': 'transaction_id', 'function': 'count'},
                {'field': 'amount', 'function': 'avg'}
            ]
        }
    ]
    
    return {
        **state,
        'aggregation_rules': aggregation_rules,
        'messages': state['messages'] + [f'Defined {len(aggregation_rules)} aggregation rules']
    }


def apply_basic_aggregation_agent(state: AggregationState) -> AggregationState:
    """Apply basic aggregation rules"""
    
    aggregator = DataAggregator()
    aggregated_data = {}
    
    for rule in state['aggregation_rules']:
        aggregated_data[rule['name']] = aggregator.aggregate_by_group(
            state['raw_data'],
            rule['group_by'],
            rule['aggregations']
        )
    
    return {
        **state,
        'aggregated_data': aggregated_data,
        'messages': state['messages'] + [f'Applied {len(state["aggregation_rules"])} aggregation rules']
    }


def apply_time_series_aggregation_agent(state: AggregationState) -> AggregationState:
    """Apply time-series aggregation"""
    
    ts_aggregator = TimeSeriesAggregator()
    
    # Aggregate by month
    monthly_aggregation = ts_aggregator.aggregate_by_period(
        state['raw_data'],
        'date',
        'month',
        [
            {'field': 'amount', 'function': 'sum'},
            {'field': 'amount', 'function': 'avg'},
            {'field': 'quantity', 'function': 'sum'}
        ]
    )
    
    time_series_aggregation = {
        'monthly': monthly_aggregation,
        'period_count': len(monthly_aggregation)
    }
    
    return {
        **state,
        'time_series_aggregation': time_series_aggregation,
        'messages': state['messages'] + [
            f'Applied time-series aggregation: {len(monthly_aggregation)} monthly periods'
        ]
    }


def apply_hierarchical_aggregation_agent(state: AggregationState) -> AggregationState:
    """Apply hierarchical aggregation"""
    
    hierarchical_aggregator = HierarchicalAggregator()
    
    # Aggregate by category -> region hierarchy
    hierarchy_result = hierarchical_aggregator.aggregate_hierarchy(
        state['raw_data'],
        ['category', 'region'],
        [
            {'field': 'amount', 'function': 'sum'},
            {'field': 'quantity', 'function': 'sum'}
        ]
    )
    
    hierarchical_aggregation = {
        'hierarchy': ['category', 'region'],
        'data': hierarchy_result
    }
    
    return {
        **state,
        'hierarchical_aggregation': hierarchical_aggregation,
        'messages': state['messages'] + ['Applied hierarchical aggregation (category -> region)']
    }


def calculate_statistical_summary_agent(state: AggregationState) -> AggregationState:
    """Calculate statistical summary"""
    
    aggregator = DataAggregator()
    amounts = [record['amount'] for record in state['raw_data']]
    quantities = [record['quantity'] for record in state['raw_data']]
    
    statistical_summary = {
        'transaction_count': len(state['raw_data']),
        'amount_statistics': {
            'sum': aggregator._sum(amounts),
            'avg': aggregator._avg(amounts),
            'min': aggregator._min(amounts),
            'max': aggregator._max(amounts),
            'median': aggregator._median(amounts),
            'stddev': aggregator._stddev(amounts)
        },
        'quantity_statistics': {
            'sum': aggregator._sum(quantities),
            'avg': aggregator._avg(quantities),
            'min': aggregator._min(quantities),
            'max': aggregator._max(quantities)
        },
        'unique_customers': aggregator._distinct_count(
            [record['customer_id'] for record in state['raw_data']]
        ),
        'unique_products': aggregator._distinct_count(
            [record['product'] for record in state['raw_data']]
        ),
        'unique_categories': aggregator._distinct_count(
            [record['category'] for record in state['raw_data']]
        ),
        'unique_regions': aggregator._distinct_count(
            [record['region'] for record in state['raw_data']]
        )
    }
    
    return {
        **state,
        'statistical_summary': statistical_summary,
        'messages': state['messages'] + ['Calculated comprehensive statistical summary']
    }


def generate_aggregation_report_agent(state: AggregationState) -> AggregationState:
    """Generate comprehensive aggregation report"""
    
    report_lines = [
        "=" * 80,
        "AGGREGATION REPORT",
        "=" * 80,
        "",
        "DATA OVERVIEW:",
        "-" * 40,
        f"Total transactions: {len(state['raw_data'])}",
        f"Date range: {min(r['date'] for r in state['raw_data'])} to {max(r['date'] for r in state['raw_data'])}",
        "",
        "STATISTICAL SUMMARY:",
        "-" * 40,
        f"Total revenue: ${state['statistical_summary']['amount_statistics']['sum']:,.2f}",
        f"Average transaction: ${state['statistical_summary']['amount_statistics']['avg']:,.2f}",
        f"Min transaction: ${state['statistical_summary']['amount_statistics']['min']:,.2f}",
        f"Max transaction: ${state['statistical_summary']['amount_statistics']['max']:,.2f}",
        f"Median transaction: ${state['statistical_summary']['amount_statistics']['median']:,.2f}",
        f"Std deviation: ${state['statistical_summary']['amount_statistics']['stddev']:,.2f}",
        "",
        f"Total items sold: {state['statistical_summary']['quantity_statistics']['sum']}",
        f"Average quantity per transaction: {state['statistical_summary']['quantity_statistics']['avg']:.2f}",
        "",
        f"Unique customers: {state['statistical_summary']['unique_customers']}",
        f"Unique products: {state['statistical_summary']['unique_products']}",
        f"Unique categories: {state['statistical_summary']['unique_categories']}",
        f"Unique regions: {state['statistical_summary']['unique_regions']}",
        ""
    ]
    
    # Basic aggregations
    for rule_name, aggregation in state['aggregated_data'].items():
        report_lines.extend([
            f"{rule_name.upper()}:",
            "-" * 40
        ])
        for group, metrics in aggregation.items():
            report_lines.append(f"\n{group}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric_name}: {value:,.2f}")
                else:
                    report_lines.append(f"  {metric_name}: {value}")
        report_lines.append("")
    
    # Time-series aggregation
    report_lines.extend([
        "TIME-SERIES AGGREGATION (MONTHLY):",
        "-" * 40
    ])
    for period, metrics in state['time_series_aggregation']['monthly'].items():
        report_lines.append(f"\n{period}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                report_lines.append(f"  {metric_name}: {value:,.2f}")
            else:
                report_lines.append(f"  {metric_name}: {value}")
    report_lines.append("")
    
    # Hierarchical aggregation
    report_lines.extend([
        "HIERARCHICAL AGGREGATION (Category -> Region):",
        "-" * 40
    ])
    
    def print_hierarchy(data, indent=0):
        lines = []
        for key, value in data.items():
            prefix = "  " * indent
            lines.append(f"{prefix}{key}:")
            
            for k, v in value.items():
                if k == 'children' and v:
                    lines.extend(print_hierarchy(v, indent + 1))
                elif k != 'children':
                    if isinstance(v, float):
                        lines.append(f"{prefix}  {k}: {v:,.2f}")
                    else:
                        lines.append(f"{prefix}  {k}: {v}")
        return lines
    
    report_lines.extend(print_hierarchy(state['hierarchical_aggregation']['data']))
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive aggregation report']
    }


# Create the graph
def create_aggregation_graph():
    """Create the aggregation workflow graph"""
    
    workflow = StateGraph(AggregationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_aggregation_agent)
    workflow.add_node("define_rules", define_aggregation_rules_agent)
    workflow.add_node("apply_basic", apply_basic_aggregation_agent)
    workflow.add_node("apply_timeseries", apply_time_series_aggregation_agent)
    workflow.add_node("apply_hierarchical", apply_hierarchical_aggregation_agent)
    workflow.add_node("calculate_stats", calculate_statistical_summary_agent)
    workflow.add_node("generate_report", generate_aggregation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "apply_basic")
    workflow.add_edge("apply_basic", "apply_timeseries")
    workflow.add_edge("apply_timeseries", "apply_hierarchical")
    workflow.add_edge("apply_hierarchical", "calculate_stats")
    workflow.add_edge("calculate_stats", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the aggregation graph
    app = create_aggregation_graph()
    
    # Initialize state
    initial_state: AggregationState = {
        'messages': [],
        'raw_data': [],
        'aggregation_rules': [],
        'aggregated_data': {},
        'time_series_aggregation': {},
        'hierarchical_aggregation': {},
        'statistical_summary': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("AGGREGATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nAggregation pattern execution complete! ✓")

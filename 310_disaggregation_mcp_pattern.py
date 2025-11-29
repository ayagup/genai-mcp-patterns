"""
Disaggregation MCP Pattern

This pattern demonstrates data disaggregation in an agentic MCP system.
Disaggregation breaks down aggregated or summarized data into detailed records,
expanding collapsed data into its constituent parts.

Use cases:
- Drilling down from summary to detail
- Expanding aggregated reports
- Data enrichment from summaries
- Reversing aggregation for analysis
- Creating detail records from totals
- Time-series expansion
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta


# Define the state for disaggregation
class DisaggregationState(TypedDict):
    """State for tracking disaggregation process"""
    messages: Annotated[List[str], add]
    aggregated_data: Dict[str, Any]
    disaggregation_rules: List[Dict[str, Any]]
    disaggregated_data: List[Dict[str, Any]]
    allocation_strategy: str
    detail_level: str
    validation_results: Dict[str, Any]
    report: str


class DisaggregationStrategy:
    """Class to handle different disaggregation strategies"""
    
    def __init__(self):
        self.strategies = {
            'equal_distribution': self.equal_distribution,
            'proportional_allocation': self.proportional_allocation,
            'pattern_based': self.pattern_based,
            'time_series_expansion': self.time_series_expansion
        }
    
    def equal_distribution(self, aggregated: Dict[str, Any], detail_count: int) -> List[Dict[str, Any]]:
        """Distribute aggregated value equally across detail records"""
        total_amount = aggregated.get('total_amount', 0)
        per_item_amount = total_amount / detail_count if detail_count > 0 else 0
        
        details = []
        for i in range(detail_count):
            details.append({
                'detail_id': f"{aggregated.get('group_id', 'GRP')}_{i+1:03d}",
                'group_id': aggregated.get('group_id', ''),
                'amount': round(per_item_amount, 2),
                'allocation_method': 'equal_distribution'
            })
        
        return details
    
    def proportional_allocation(self, aggregated: Dict[str, Any], 
                                weights: List[float]) -> List[Dict[str, Any]]:
        """Allocate based on proportional weights"""
        total_amount = aggregated.get('total_amount', 0)
        total_weight = sum(weights)
        
        details = []
        for i, weight in enumerate(weights):
            proportion = weight / total_weight if total_weight > 0 else 0
            details.append({
                'detail_id': f"{aggregated.get('group_id', 'GRP')}_{i+1:03d}",
                'group_id': aggregated.get('group_id', ''),
                'amount': round(total_amount * proportion, 2),
                'weight': weight,
                'proportion': round(proportion * 100, 2),
                'allocation_method': 'proportional'
            })
        
        return details
    
    def pattern_based(self, aggregated: Dict[str, Any], 
                     pattern: List[float]) -> List[Dict[str, Any]]:
        """Disaggregate based on historical pattern"""
        total_amount = aggregated.get('total_amount', 0)
        pattern_sum = sum(pattern)
        
        details = []
        for i, pattern_value in enumerate(pattern):
            ratio = pattern_value / pattern_sum if pattern_sum > 0 else 0
            details.append({
                'detail_id': f"{aggregated.get('group_id', 'GRP')}_{i+1:03d}",
                'group_id': aggregated.get('group_id', ''),
                'amount': round(total_amount * ratio, 2),
                'pattern_value': pattern_value,
                'allocation_method': 'pattern_based'
            })
        
        return details
    
    def time_series_expansion(self, aggregated: Dict[str, Any], 
                             start_date: str, periods: int, 
                             frequency: str = 'daily') -> List[Dict[str, Any]]:
        """Expand aggregated time period into detailed periods"""
        total_amount = aggregated.get('total_amount', 0)
        per_period = total_amount / periods if periods > 0 else 0
        
        details = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        for i in range(periods):
            if frequency == 'daily':
                period_date = current_date + timedelta(days=i)
            elif frequency == 'weekly':
                period_date = current_date + timedelta(weeks=i)
            elif frequency == 'monthly':
                # Simplified monthly calculation
                month_offset = i
                period_date = current_date.replace(
                    month=((current_date.month - 1 + month_offset) % 12) + 1,
                    year=current_date.year + ((current_date.month - 1 + month_offset) // 12)
                )
            else:
                period_date = current_date + timedelta(days=i)
            
            details.append({
                'detail_id': f"{aggregated.get('group_id', 'GRP')}_{period_date.strftime('%Y%m%d')}",
                'group_id': aggregated.get('group_id', ''),
                'date': period_date.strftime('%Y-%m-%d'),
                'amount': round(per_period, 2),
                'period': i + 1,
                'allocation_method': f'time_series_{frequency}'
            })
        
        return details


class DisaggregationValidator:
    """Validate disaggregation results"""
    
    def validate(self, aggregated: Dict[str, Any], 
                disaggregated: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that disaggregation preserves totals"""
        
        aggregated_total = aggregated.get('total_amount', 0)
        disaggregated_total = sum(item.get('amount', 0) for item in disaggregated)
        
        difference = abs(aggregated_total - disaggregated_total)
        tolerance = 0.01  # Allow 1 cent difference due to rounding
        
        is_valid = difference <= tolerance
        
        return {
            'is_valid': is_valid,
            'aggregated_total': aggregated_total,
            'disaggregated_total': disaggregated_total,
            'difference': round(difference, 2),
            'record_count': len(disaggregated),
            'average_per_record': round(disaggregated_total / len(disaggregated), 2) 
                                 if disaggregated else 0
        }


# Agent functions
def initialize_disaggregation_agent(state: DisaggregationState) -> DisaggregationState:
    """Initialize disaggregation with aggregated data"""
    
    # Sample aggregated data (monthly sales summaries)
    aggregated_data = {
        'monthly_summaries': [
            {
                'group_id': 'SALES_2024_01',
                'period': '2024-01',
                'category': 'Electronics',
                'total_amount': 3500.00,
                'transaction_count': 15,
                'start_date': '2024-01-01'
            },
            {
                'group_id': 'SALES_2024_02',
                'period': '2024-02',
                'category': 'Electronics',
                'total_amount': 4200.00,
                'transaction_count': 18,
                'start_date': '2024-02-01'
            },
            {
                'group_id': 'SALES_2024_03',
                'period': '2024-03',
                'category': 'Electronics',
                'total_amount': 3800.00,
                'transaction_count': 16,
                'start_date': '2024-03-01'
            }
        ],
        'regional_summaries': [
            {
                'group_id': 'REGION_NORTH',
                'region': 'North',
                'total_amount': 5000.00,
                'store_count': 5
            },
            {
                'group_id': 'REGION_SOUTH',
                'region': 'South',
                'total_amount': 3500.00,
                'store_count': 4
            },
            {
                'group_id': 'REGION_EAST',
                'region': 'East',
                'total_amount': 4500.00,
                'store_count': 6
            }
        ]
    }
    
    return {
        **state,
        'aggregated_data': aggregated_data,
        'messages': state['messages'] + [
            f'Initialized with {len(aggregated_data["monthly_summaries"])} monthly and '
            f'{len(aggregated_data["regional_summaries"])} regional summaries'
        ]
    }


def define_disaggregation_rules_agent(state: DisaggregationState) -> DisaggregationState:
    """Define disaggregation rules and strategies"""
    
    disaggregation_rules = [
        {
            'name': 'Monthly to Daily - Equal Distribution',
            'source': 'monthly_summaries',
            'strategy': 'time_series_expansion',
            'target_frequency': 'daily',
            'target_periods': 30  # Days in month (simplified)
        },
        {
            'name': 'Regional to Store - Proportional',
            'source': 'regional_summaries',
            'strategy': 'proportional_allocation',
            'use_weights': True
        },
        {
            'name': 'Monthly to Weekly - Pattern Based',
            'source': 'monthly_summaries',
            'strategy': 'pattern_based',
            'pattern': [0.2, 0.3, 0.25, 0.25]  # 4 weeks pattern
        }
    ]
    
    return {
        **state,
        'disaggregation_rules': disaggregation_rules,
        'messages': state['messages'] + [f'Defined {len(disaggregation_rules)} disaggregation rules']
    }


def apply_time_series_disaggregation_agent(state: DisaggregationState) -> DisaggregationState:
    """Apply time-series disaggregation (monthly to daily)"""
    
    disaggregator = DisaggregationStrategy()
    disaggregated_records = []
    
    # Disaggregate first monthly summary to daily
    if state['aggregated_data']['monthly_summaries']:
        monthly_summary = state['aggregated_data']['monthly_summaries'][0]
        
        daily_records = disaggregator.time_series_expansion(
            monthly_summary,
            monthly_summary['start_date'],
            30,  # 30 days
            'daily'
        )
        
        disaggregated_records.extend(daily_records)
    
    return {
        **state,
        'disaggregated_data': disaggregated_records,
        'allocation_strategy': 'time_series_expansion',
        'detail_level': 'daily',
        'messages': state['messages'] + [
            f'Disaggregated monthly data to {len(disaggregated_records)} daily records'
        ]
    }


def apply_proportional_disaggregation_agent(state: DisaggregationState) -> DisaggregationState:
    """Apply proportional disaggregation (regional to stores)"""
    
    disaggregator = DisaggregationStrategy()
    all_disaggregated = state['disaggregated_data'].copy()
    
    # Disaggregate first regional summary to stores
    if state['aggregated_data']['regional_summaries']:
        regional_summary = state['aggregated_data']['regional_summaries'][0]
        store_count = regional_summary['store_count']
        
        # Create weights (e.g., different store sizes)
        weights = [1.5, 1.2, 1.0, 0.8, 0.5][:store_count]
        
        store_records = disaggregator.proportional_allocation(
            regional_summary,
            weights
        )
        
        # Add metadata
        for i, record in enumerate(store_records):
            record['store_id'] = f"STORE_{regional_summary['region'][:3].upper()}_{i+1:02d}"
            record['region'] = regional_summary['region']
        
        all_disaggregated.extend(store_records)
    
    return {
        **state,
        'disaggregated_data': all_disaggregated,
        'messages': state['messages'] + [
            f'Added {len(store_records) if "store_records" in locals() else 0} store-level records via proportional allocation'
        ]
    }


def apply_pattern_based_disaggregation_agent(state: DisaggregationState) -> DisaggregationState:
    """Apply pattern-based disaggregation (monthly to weekly)"""
    
    disaggregator = DisaggregationStrategy()
    all_disaggregated = state['disaggregated_data'].copy()
    
    # Disaggregate second monthly summary using weekly pattern
    if len(state['aggregated_data']['monthly_summaries']) > 1:
        monthly_summary = state['aggregated_data']['monthly_summaries'][1]
        
        # Weekly pattern (4 weeks)
        pattern = [0.2, 0.3, 0.25, 0.25]
        
        weekly_records = disaggregator.pattern_based(
            monthly_summary,
            pattern
        )
        
        # Add metadata
        for i, record in enumerate(weekly_records):
            record['week'] = i + 1
            record['period'] = monthly_summary['period']
            record['category'] = monthly_summary['category']
        
        all_disaggregated.extend(weekly_records)
    
    return {
        **state,
        'disaggregated_data': all_disaggregated,
        'messages': state['messages'] + [
            f'Added {len(weekly_records) if "weekly_records" in locals() else 0} weekly records via pattern-based allocation'
        ]
    }


def apply_equal_distribution_agent(state: DisaggregationState) -> DisaggregationState:
    """Apply equal distribution disaggregation"""
    
    disaggregator = DisaggregationStrategy()
    all_disaggregated = state['disaggregated_data'].copy()
    
    # Disaggregate third monthly summary equally
    if len(state['aggregated_data']['monthly_summaries']) > 2:
        monthly_summary = state['aggregated_data']['monthly_summaries'][2]
        transaction_count = monthly_summary['transaction_count']
        
        equal_records = disaggregator.equal_distribution(
            monthly_summary,
            transaction_count
        )
        
        # Add metadata
        for record in equal_records:
            record['period'] = monthly_summary['period']
            record['category'] = monthly_summary['category']
        
        all_disaggregated.extend(equal_records)
    
    return {
        **state,
        'disaggregated_data': all_disaggregated,
        'messages': state['messages'] + [
            f'Added {len(equal_records) if "equal_records" in locals() else 0} transaction records via equal distribution'
        ]
    }


def validate_disaggregation_agent(state: DisaggregationState) -> DisaggregationState:
    """Validate disaggregation results"""
    
    validator = DisaggregationValidator()
    validation_results = {}
    
    # Validate monthly to daily (first 30 records)
    daily_records = [r for r in state['disaggregated_data'] 
                    if r.get('allocation_method') == 'time_series_daily'][:30]
    if daily_records:
        monthly_source = state['aggregated_data']['monthly_summaries'][0]
        validation_results['monthly_to_daily'] = validator.validate(
            monthly_source,
            daily_records
        )
    
    # Validate regional to stores
    store_records = [r for r in state['disaggregated_data'] 
                    if 'store_id' in r and 'NORTH' in r.get('store_id', '')]
    if store_records:
        regional_source = state['aggregated_data']['regional_summaries'][0]
        validation_results['regional_to_stores'] = validator.validate(
            regional_source,
            store_records
        )
    
    # Overall validation summary
    validation_results['summary'] = {
        'total_disaggregated_records': len(state['disaggregated_data']),
        'validations_performed': len([v for v in validation_results.values() 
                                     if isinstance(v, dict) and 'is_valid' in v]),
        'all_valid': all(v.get('is_valid', False) for v in validation_results.values() 
                        if isinstance(v, dict) and 'is_valid' in v)
    }
    
    return {
        **state,
        'validation_results': validation_results,
        'messages': state['messages'] + [
            f'Validated disaggregation: {validation_results["summary"]["total_disaggregated_records"]} records created'
        ]
    }


def generate_disaggregation_report_agent(state: DisaggregationState) -> DisaggregationState:
    """Generate comprehensive disaggregation report"""
    
    report_lines = [
        "=" * 80,
        "DISAGGREGATION REPORT",
        "=" * 80,
        "",
        "AGGREGATED DATA SUMMARY:",
        "-" * 40,
        f"Monthly summaries: {len(state['aggregated_data']['monthly_summaries'])}",
        f"Regional summaries: {len(state['aggregated_data']['regional_summaries'])}",
        ""
    ]
    
    # Show sample aggregated data
    report_lines.extend([
        "SAMPLE AGGREGATED DATA:",
        "-" * 40
    ])
    
    for summary in state['aggregated_data']['monthly_summaries'][:2]:
        report_lines.extend([
            f"\n{summary['group_id']}:",
            f"  Period: {summary['period']}",
            f"  Category: {summary['category']}",
            f"  Total Amount: ${summary['total_amount']:,.2f}",
            f"  Transaction Count: {summary['transaction_count']}"
        ])
    
    report_lines.extend([
        "",
        "DISAGGREGATION STRATEGIES APPLIED:",
        "-" * 40
    ])
    
    for rule in state['disaggregation_rules']:
        report_lines.extend([
            f"\n{rule['name']}:",
            f"  Source: {rule['source']}",
            f"  Strategy: {rule['strategy']}"
        ])
    
    report_lines.extend([
        "",
        "DISAGGREGATED DATA:",
        "-" * 40,
        f"Total records created: {len(state['disaggregated_data'])}",
        ""
    ])
    
    # Group by allocation method
    by_method = {}
    for record in state['disaggregated_data']:
        method = record.get('allocation_method', 'unknown')
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(record)
    
    report_lines.append("Records by allocation method:")
    for method, records in by_method.items():
        total = sum(r.get('amount', 0) for r in records)
        report_lines.append(f"  {method}: {len(records)} records (${total:,.2f})")
    
    report_lines.extend([
        "",
        "SAMPLE DISAGGREGATED RECORDS:",
        "-" * 40
    ])
    
    # Show samples from each method
    for method, records in list(by_method.items())[:3]:
        report_lines.append(f"\n{method} (showing first 3):")
        for record in records[:3]:
            report_lines.append(f"  ID: {record.get('detail_id', 'N/A')}")
            report_lines.append(f"    Amount: ${record.get('amount', 0):,.2f}")
            if 'date' in record:
                report_lines.append(f"    Date: {record['date']}")
            if 'proportion' in record:
                report_lines.append(f"    Proportion: {record['proportion']}%")
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for validation_name, validation in state['validation_results'].items():
        if isinstance(validation, dict) and 'is_valid' in validation:
            status = "✓ VALID" if validation['is_valid'] else "✗ INVALID"
            report_lines.extend([
                f"\n{validation_name}: {status}",
                f"  Aggregated total: ${validation['aggregated_total']:,.2f}",
                f"  Disaggregated total: ${validation['disaggregated_total']:,.2f}",
                f"  Difference: ${validation['difference']:,.2f}",
                f"  Record count: {validation['record_count']}",
                f"  Average per record: ${validation['average_per_record']:,.2f}"
            ])
    
    summary = state['validation_results'].get('summary', {})
    if summary:
        overall_status = "✓ ALL VALID" if summary.get('all_valid') else "✗ SOME INVALID"
        report_lines.extend([
            "",
            f"Overall Status: {overall_status}",
            f"Total records created: {summary.get('total_disaggregated_records', 0)}",
            f"Validations performed: {summary.get('validations_performed', 0)}"
        ])
    
    report_lines.extend([
        "",
        "USE CASES:",
        "-" * 40,
        "✓ Drill-down analysis from summaries",
        "✓ Creating detail records for simulation",
        "✓ Expanding compressed data",
        "✓ Time-series forecasting and planning",
        "✓ Budget allocation across departments",
        "✓ Sales target distribution",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive disaggregation report']
    }


# Create the graph
def create_disaggregation_graph():
    """Create the disaggregation workflow graph"""
    
    workflow = StateGraph(DisaggregationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_disaggregation_agent)
    workflow.add_node("define_rules", define_disaggregation_rules_agent)
    workflow.add_node("apply_timeseries", apply_time_series_disaggregation_agent)
    workflow.add_node("apply_proportional", apply_proportional_disaggregation_agent)
    workflow.add_node("apply_pattern", apply_pattern_based_disaggregation_agent)
    workflow.add_node("apply_equal", apply_equal_distribution_agent)
    workflow.add_node("validate", validate_disaggregation_agent)
    workflow.add_node("generate_report", generate_disaggregation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "apply_timeseries")
    workflow.add_edge("apply_timeseries", "apply_proportional")
    workflow.add_edge("apply_proportional", "apply_pattern")
    workflow.add_edge("apply_pattern", "apply_equal")
    workflow.add_edge("apply_equal", "validate")
    workflow.add_edge("validate", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the disaggregation graph
    app = create_disaggregation_graph()
    
    # Initialize state
    initial_state: DisaggregationState = {
        'messages': [],
        'aggregated_data': {},
        'disaggregation_rules': [],
        'disaggregated_data': [],
        'allocation_strategy': '',
        'detail_level': '',
        'validation_results': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("DISAGGREGATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nDisaggregation pattern execution complete! ✓")

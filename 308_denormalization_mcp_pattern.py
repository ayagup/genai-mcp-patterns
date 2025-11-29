"""
Denormalization MCP Pattern

This pattern demonstrates denormalization in an agentic MCP system.
Denormalization is the process of combining normalized tables to optimize read performance,
reduce join operations, and improve query speed at the cost of some data redundancy.

Use cases:
- Read-heavy applications
- Reporting and analytics systems
- Data warehousing
- Query performance optimization
- Caching strategies
- NoSQL database design
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI


# Define the state for denormalization
class DenormalizationState(TypedDict):
    """State for tracking denormalization process"""
    messages: Annotated[List[str], add]
    normalized_data: Dict[str, List[Dict[str, Any]]]
    denormalized_data: List[Dict[str, Any]]
    denormalization_strategy: str
    query_patterns: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    redundancy_analysis: Dict[str, Any]
    report: str


class DenormalizationStrategy:
    """Class to handle different denormalization strategies"""
    
    def __init__(self):
        self.strategies = {
            'full_denormalization': self.full_denormalization,
            'partial_denormalization': self.partial_denormalization,
            'selective_denormalization': self.selective_denormalization,
            'materialized_view': self.materialized_view
        }
    
    def full_denormalization(self, normalized_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Completely denormalize all tables into a single flat structure"""
        customers = {c['customer_id']: c for c in normalized_data.get('customers', [])}
        addresses = {a['address_id']: a for a in normalized_data.get('addresses', [])}
        states = {s['state_id']: s for s in normalized_data.get('states', [])}
        orders = normalized_data.get('orders', [])
        
        denormalized = []
        for order in orders:
            customer = customers.get(order['customer_id'], {})
            address_id = customer.get('address_id')
            address = addresses.get(address_id, {})
            state_id = address.get('state_id')
            state = states.get(state_id, {})
            
            denormalized.append({
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'customer_name': customer.get('name', ''),
                'customer_email': customer.get('email', ''),
                'customer_phone': customer.get('phone', ''),
                'street': address.get('street', ''),
                'city': address.get('city', ''),
                'state_code': state.get('state_code', ''),
                'state_name': state.get('state_name', ''),
                'zip_code': address.get('zip_code', ''),
                'order_total': order.get('total', 0),
                'order_date': order.get('order_date', '')
            })
        
        return denormalized
    
    def partial_denormalization(self, normalized_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Denormalize frequently joined tables only"""
        customers = {c['customer_id']: c for c in normalized_data.get('customers', [])}
        orders = normalized_data.get('orders', [])
        
        denormalized = []
        for order in orders:
            customer = customers.get(order['customer_id'], {})
            
            denormalized.append({
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'customer_name': customer.get('name', ''),
                'customer_email': customer.get('email', ''),
                'order_total': order.get('total', 0),
                'order_date': order.get('order_date', ''),
                'address_id': customer.get('address_id')  # Keep reference
            })
        
        return denormalized
    
    def selective_denormalization(self, normalized_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Denormalize only specific columns based on access patterns"""
        customers = {c['customer_id']: c for c in normalized_data.get('customers', [])}
        orders = normalized_data.get('orders', [])
        
        denormalized = []
        for order in orders:
            customer = customers.get(order['customer_id'], {})
            
            # Only denormalize customer name for display purposes
            denormalized.append({
                'order_id': order['order_id'],
                'customer_id': order['customer_id'],
                'customer_name': customer.get('name', ''),  # Denormalized
                'order_total': order.get('total', 0),
                'order_date': order.get('order_date', '')
            })
        
        return denormalized
    
    def materialized_view(self, normalized_data: Dict[str, List[Dict]]) -> List[Dict]:
        """Create a materialized view for common queries"""
        customers = {c['customer_id']: c for c in normalized_data.get('customers', [])}
        addresses = {a['address_id']: a for a in normalized_data.get('addresses', [])}
        states = {s['state_id']: s for s in normalized_data.get('states', [])}
        orders = normalized_data.get('orders', [])
        
        # Aggregate data for reporting
        denormalized = []
        customer_orders = {}
        
        for order in orders:
            customer_id = order['customer_id']
            if customer_id not in customer_orders:
                customer = customers.get(customer_id, {})
                address_id = customer.get('address_id')
                address = addresses.get(address_id, {})
                state_id = address.get('state_id')
                state = states.get(state_id, {})
                
                customer_orders[customer_id] = {
                    'customer_id': customer_id,
                    'customer_name': customer.get('name', ''),
                    'customer_email': customer.get('email', ''),
                    'state_name': state.get('state_name', ''),
                    'order_count': 0,
                    'total_spent': 0,
                    'orders': []
                }
            
            customer_orders[customer_id]['order_count'] += 1
            customer_orders[customer_id]['total_spent'] += order.get('total', 0)
            customer_orders[customer_id]['orders'].append({
                'order_id': order['order_id'],
                'order_date': order.get('order_date', ''),
                'total': order.get('total', 0)
            })
        
        return list(customer_orders.values())


class PerformanceAnalyzer:
    """Analyze performance benefits of denormalization"""
    
    def analyze_query_performance(self, normalized_data: Dict, denormalized_data: List[Dict], 
                                  query_patterns: List[Dict]) -> Dict[str, Any]:
        """Analyze performance improvements"""
        metrics = {
            'normalized_joins': 0,
            'denormalized_joins': 0,
            'read_performance_gain': 0,
            'storage_overhead': 0,
            'query_improvements': []
        }
        
        # Calculate joins required
        for pattern in query_patterns:
            query_type = pattern.get('type', '')
            
            if query_type == 'order_with_customer':
                metrics['normalized_joins'] += 1  # Join customers table
                metrics['denormalized_joins'] += 0  # No join needed
            
            elif query_type == 'order_with_full_address':
                metrics['normalized_joins'] += 3  # Join customers, addresses, states
                metrics['denormalized_joins'] += 0  # No join needed
            
            elif query_type == 'customer_order_summary':
                metrics['normalized_joins'] += 2  # Join customers and aggregate orders
                metrics['denormalized_joins'] += 0  # Pre-aggregated
        
        # Calculate performance gain (simplified)
        if metrics['normalized_joins'] > 0:
            metrics['read_performance_gain'] = (
                (metrics['normalized_joins'] - metrics['denormalized_joins']) / 
                metrics['normalized_joins'] * 100
            )
        
        # Calculate storage overhead
        normalized_size = sum(len(table) for table in normalized_data.values())
        denormalized_size = len(denormalized_data)
        metrics['storage_overhead'] = (
            (denormalized_size - normalized_size) / normalized_size * 100
            if normalized_size > 0 else 0
        )
        
        # Query-specific improvements
        for pattern in query_patterns:
            metrics['query_improvements'].append({
                'query_type': pattern.get('type', ''),
                'normalized_steps': pattern.get('normalized_steps', 0),
                'denormalized_steps': pattern.get('denormalized_steps', 0),
                'improvement': pattern.get('improvement_percent', 0)
            })
        
        return metrics


# Agent functions
def initialize_denormalization_agent(state: DenormalizationState) -> DenormalizationState:
    """Initialize denormalization with normalized data"""
    
    # Sample normalized data (3NF structure from pattern 307)
    normalized_data = {
        'customers': [
            {'customer_id': 1, 'name': 'John Doe', 'email': 'john.doe@example.com', 
             'phone': '(555) 123-4567', 'address_id': 1},
            {'customer_id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com', 
             'phone': '(555) 456-7890', 'address_id': 2},
            {'customer_id': 3, 'name': 'Bob Johnson', 'email': 'bob.j@sample.org', 
             'phone': '(555) 123-4567', 'address_id': 3}
        ],
        'addresses': [
            {'address_id': 1, 'street': '123 Main St', 'city': 'New York', 'state_id': 1, 'zip_code': '10001'},
            {'address_id': 2, 'street': '456 Oak Ave', 'city': 'Los Angeles', 'state_id': 2, 'zip_code': '90001'},
            {'address_id': 3, 'street': '789 Pine Rd', 'city': 'Chicago', 'state_id': 3, 'zip_code': '60601'}
        ],
        'states': [
            {'state_id': 1, 'state_code': 'NY', 'state_name': 'New York'},
            {'state_id': 2, 'state_code': 'CA', 'state_name': 'California'},
            {'state_id': 3, 'state_code': 'IL', 'state_name': 'Illinois'}
        ],
        'orders': [
            {'order_id': 'ORD001', 'customer_id': 1, 'total': 150.00, 'order_date': '2024-01-15'},
            {'order_id': 'ORD002', 'customer_id': 1, 'total': 200.00, 'order_date': '2024-02-20'},
            {'order_id': 'ORD003', 'customer_id': 2, 'total': 300.00, 'order_date': '2024-01-25'},
            {'order_id': 'ORD004', 'customer_id': 3, 'total': 100.00, 'order_date': '2024-03-10'},
            {'order_id': 'ORD005', 'customer_id': 3, 'total': 250.00, 'order_date': '2024-03-15'},
            {'order_id': 'ORD006', 'customer_id': 3, 'total': 175.00, 'order_date': '2024-04-01'}
        ]
    }
    
    return {
        **state,
        'normalized_data': normalized_data,
        'messages': state['messages'] + ['Initialized denormalization with normalized data (4 tables)']
    }


def analyze_query_patterns_agent(state: DenormalizationState) -> DenormalizationState:
    """Analyze common query patterns to determine denormalization strategy"""
    
    # Define common query patterns
    query_patterns = [
        {
            'type': 'order_with_customer',
            'description': 'Get order details with customer name',
            'frequency': 'very_high',
            'normalized_steps': 2,  # Join customers table
            'denormalized_steps': 1,  # Direct access
            'improvement_percent': 50
        },
        {
            'type': 'order_with_full_address',
            'description': 'Get order with complete customer address',
            'frequency': 'high',
            'normalized_steps': 4,  # Join customers, addresses, states
            'denormalized_steps': 1,  # Direct access
            'improvement_percent': 75
        },
        {
            'type': 'customer_order_summary',
            'description': 'Get customer order count and total spent',
            'frequency': 'medium',
            'normalized_steps': 3,  # Join customers, aggregate orders
            'denormalized_steps': 1,  # Pre-aggregated view
            'improvement_percent': 66
        }
    ]
    
    return {
        **state,
        'query_patterns': query_patterns,
        'messages': state['messages'] + [f'Analyzed {len(query_patterns)} common query patterns']
    }


def select_strategy_agent(state: DenormalizationState) -> DenormalizationState:
    """Select denormalization strategy based on query patterns"""
    
    # For this example, use full denormalization for maximum read performance
    strategy = 'full_denormalization'
    
    return {
        **state,
        'denormalization_strategy': strategy,
        'messages': state['messages'] + [f'Selected strategy: {strategy}']
    }


def apply_denormalization_agent(state: DenormalizationState) -> DenormalizationState:
    """Apply the selected denormalization strategy"""
    
    denormalizer = DenormalizationStrategy()
    strategy_func = denormalizer.strategies.get(state['denormalization_strategy'])
    
    if strategy_func:
        denormalized_data = strategy_func(state['normalized_data'])
    else:
        denormalized_data = []
    
    return {
        **state,
        'denormalized_data': denormalized_data,
        'messages': state['messages'] + [
            f'Applied {state["denormalization_strategy"]} to {len(denormalized_data)} records'
        ]
    }


def analyze_performance_agent(state: DenormalizationState) -> DenormalizationState:
    """Analyze performance improvements from denormalization"""
    
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_query_performance(
        state['normalized_data'],
        state['denormalized_data'],
        state['query_patterns']
    )
    
    return {
        **state,
        'performance_metrics': metrics,
        'messages': state['messages'] + [
            f'Performance analysis complete: {metrics["read_performance_gain"]:.1f}% read improvement'
        ]
    }


def analyze_redundancy_agent(state: DenormalizationState) -> DenormalizationState:
    """Analyze data redundancy introduced by denormalization"""
    
    normalized_data = state['normalized_data']
    denormalized_data = state['denormalized_data']
    
    # Calculate redundancy
    normalized_fields = sum(
        len(record) * len(table)
        for table in normalized_data.values()
        for record in table
    )
    
    denormalized_fields = sum(len(record) for record in denormalized_data)
    
    redundancy_analysis = {
        'normalized_field_count': normalized_fields,
        'denormalized_field_count': denormalized_fields,
        'redundancy_factor': denormalized_fields / normalized_fields if normalized_fields > 0 else 0,
        'duplicate_data_examples': [
            {
                'field': 'customer_name',
                'appears_in_orders': sum(1 for order in denormalized_data if 'customer_name' in order),
                'unique_values': len(set(order.get('customer_name', '') for order in denormalized_data))
            },
            {
                'field': 'customer_email',
                'appears_in_orders': sum(1 for order in denormalized_data if 'customer_email' in order),
                'unique_values': len(set(order.get('customer_email', '') for order in denormalized_data))
            }
        ],
        'update_complexity': 'Higher - must update multiple denormalized records when customer info changes',
        'storage_cost': 'Higher - duplicated data across records',
        'read_benefit': 'Lower latency - no joins required'
    }
    
    return {
        **state,
        'redundancy_analysis': redundancy_analysis,
        'messages': state['messages'] + [
            f'Redundancy factor: {redundancy_analysis["redundancy_factor"]:.2f}x'
        ]
    }


def generate_report_agent(state: DenormalizationState) -> DenormalizationState:
    """Generate comprehensive denormalization report"""
    
    report_lines = [
        "=" * 80,
        "DENORMALIZATION REPORT",
        "=" * 80,
        "",
        "NORMALIZED DATA STRUCTURE:",
        "-" * 40,
        f"Tables: {len(state['normalized_data'])}",
        f"  - Customers: {len(state['normalized_data']['customers'])} records",
        f"  - Addresses: {len(state['normalized_data']['addresses'])} records",
        f"  - States: {len(state['normalized_data']['states'])} records",
        f"  - Orders: {len(state['normalized_data']['orders'])} records",
        "",
        "DENORMALIZATION STRATEGY:",
        "-" * 40,
        f"Strategy: {state['denormalization_strategy']}",
        f"Denormalized records: {len(state['denormalized_data'])}",
        "",
        "SAMPLE DENORMALIZED RECORD:",
        "-" * 40
    ]
    
    if state['denormalized_data']:
        sample = state['denormalized_data'][0]
        for key, value in sample.items():
            report_lines.append(f"  {key}: {value}")
    
    report_lines.extend([
        "",
        "PERFORMANCE METRICS:",
        "-" * 40,
        f"Normalized joins required: {state['performance_metrics']['normalized_joins']}",
        f"Denormalized joins required: {state['performance_metrics']['denormalized_joins']}",
        f"Read performance gain: {state['performance_metrics']['read_performance_gain']:.1f}%",
        f"Storage overhead: {state['performance_metrics']['storage_overhead']:.1f}%",
        "",
        "QUERY IMPROVEMENTS:",
        "-" * 40
    ])
    
    for improvement in state['performance_metrics']['query_improvements']:
        report_lines.extend([
            f"Query: {improvement['query_type']}",
            f"  Normalized steps: {improvement['normalized_steps']}",
            f"  Denormalized steps: {improvement['denormalized_steps']}",
            f"  Improvement: {improvement['improvement_percent']}%",
            ""
        ])
    
    report_lines.extend([
        "REDUNDANCY ANALYSIS:",
        "-" * 40,
        f"Redundancy factor: {state['redundancy_analysis']['redundancy_factor']:.2f}x",
        f"Update complexity: {state['redundancy_analysis']['update_complexity']}",
        f"Storage cost: {state['redundancy_analysis']['storage_cost']}",
        f"Read benefit: {state['redundancy_analysis']['read_benefit']}",
        "",
        "DUPLICATE DATA EXAMPLES:",
        "-" * 40
    ])
    
    for example in state['redundancy_analysis']['duplicate_data_examples']:
        report_lines.extend([
            f"Field: {example['field']}",
            f"  Appears in {example['appears_in_orders']} order records",
            f"  Unique values: {example['unique_values']}",
            ""
        ])
    
    report_lines.extend([
        "TRADE-OFFS:",
        "-" * 40,
        "Benefits:",
        "  ✓ Faster read queries (no joins)",
        "  ✓ Simpler query logic",
        "  ✓ Better for read-heavy workloads",
        "  ✓ Reduced database load",
        "",
        "Costs:",
        "  ✗ Data redundancy",
        "  ✗ Higher storage requirements",
        "  ✗ More complex updates",
        "  ✗ Risk of data inconsistency",
        "  ✗ Higher write costs",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive denormalization report']
    }


# Create the graph
def create_denormalization_graph():
    """Create the denormalization workflow graph"""
    
    workflow = StateGraph(DenormalizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_denormalization_agent)
    workflow.add_node("analyze_queries", analyze_query_patterns_agent)
    workflow.add_node("select_strategy", select_strategy_agent)
    workflow.add_node("apply_denormalization", apply_denormalization_agent)
    workflow.add_node("analyze_performance", analyze_performance_agent)
    workflow.add_node("analyze_redundancy", analyze_redundancy_agent)
    workflow.add_node("generate_report", generate_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_queries")
    workflow.add_edge("analyze_queries", "select_strategy")
    workflow.add_edge("select_strategy", "apply_denormalization")
    workflow.add_edge("apply_denormalization", "analyze_performance")
    workflow.add_edge("analyze_performance", "analyze_redundancy")
    workflow.add_edge("analyze_redundancy", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the denormalization graph
    app = create_denormalization_graph()
    
    # Initialize state
    initial_state: DenormalizationState = {
        'messages': [],
        'normalized_data': {},
        'denormalized_data': [],
        'denormalization_strategy': '',
        'query_patterns': [],
        'performance_metrics': {},
        'redundancy_analysis': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("DENORMALIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nDenormalization pattern execution complete! ✓")

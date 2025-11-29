"""
Business Rule Validation MCP Pattern

This pattern demonstrates business rule validation in an agentic MCP system.
Business rule validation ensures data and operations comply with domain-specific
rules, policies, and business logic beyond basic data type validation.

Use cases:
- Order processing validation
- Financial transaction rules
- Inventory management rules
- Pricing and discount logic
- Compliance checking
- Policy enforcement
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional, Callable
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta


# Define the state for business rule validation
class BusinessRuleValidationState(TypedDict):
    """State for tracking business rule validation"""
    messages: Annotated[List[str], add]
    transactions: List[Dict[str, Any]]
    business_rules: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    rule_violations: List[Dict[str, Any]]
    compliance_score: Dict[str, Any]
    report: str


class BusinessRule:
    """Base class for business rules"""
    
    def __init__(self, rule_id: str, name: str, description: str, severity: str = 'ERROR'):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity  # ERROR, WARNING, INFO
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate data against rule"""
        raise NotImplementedError


class OrderValueRule(BusinessRule):
    """Rule: Order value must be within allowed range"""
    
    def __init__(self):
        super().__init__(
            'BR001',
            'Order Value Range',
            'Order total must be between minimum and maximum allowed values',
            'ERROR'
        )
        self.min_value = 10.00
        self.max_value = 50000.00
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        total = data.get('total', 0)
        
        if total < self.min_value:
            return False, f"Order total ${total:.2f} below minimum ${self.min_value:.2f}"
        if total > self.max_value:
            return False, f"Order total ${total:.2f} exceeds maximum ${self.max_value:.2f}"
        
        return True, None


class DiscountLimitRule(BusinessRule):
    """Rule: Discount cannot exceed maximum percentage"""
    
    def __init__(self):
        super().__init__(
            'BR002',
            'Discount Limit',
            'Discount percentage cannot exceed maximum allowed',
            'ERROR'
        )
        self.max_discount_percent = 50
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        discount = data.get('discount_percent', 0)
        
        if discount > self.max_discount_percent:
            return False, f"Discount {discount}% exceeds maximum {self.max_discount_percent}%"
        if discount < 0:
            return False, "Discount cannot be negative"
        
        return True, None


class InventoryAvailabilityRule(BusinessRule):
    """Rule: Product must be in stock"""
    
    def __init__(self):
        super().__init__(
            'BR003',
            'Inventory Availability',
            'Ordered quantity must not exceed available stock',
            'ERROR'
        )
        # Mock inventory
        self.inventory = {
            'PROD001': 100,
            'PROD002': 50,
            'PROD003': 0,  # Out of stock
            'PROD004': 25
        }
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        items = data.get('items', [])
        
        for item in items:
            product_id = item.get('product_id')
            quantity = item.get('quantity', 0)
            
            available = self.inventory.get(product_id, 0)
            
            if available == 0:
                return False, f"Product {product_id} is out of stock"
            if quantity > available:
                return False, f"Product {product_id}: requested {quantity}, available {available}"
        
        return True, None


class CustomerCreditLimitRule(BusinessRule):
    """Rule: Customer order must not exceed credit limit"""
    
    def __init__(self):
        super().__init__(
            'BR004',
            'Customer Credit Limit',
            'Order total must not exceed customer credit limit',
            'WARNING'
        )
        # Mock customer credit limits
        self.credit_limits = {
            'CUST001': 10000.00,
            'CUST002': 5000.00,
            'CUST003': 1000.00,
            'CUST004': 15000.00
        }
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        customer_id = data.get('customer_id')
        total = data.get('total', 0)
        
        credit_limit = self.credit_limits.get(customer_id, 0)
        
        if total > credit_limit:
            return False, f"Order total ${total:.2f} exceeds credit limit ${credit_limit:.2f}"
        
        return True, None


class BusinessHoursRule(BusinessRule):
    """Rule: Orders can only be placed during business hours"""
    
    def __init__(self):
        super().__init__(
            'BR005',
            'Business Hours',
            'Orders must be placed during business hours (9 AM - 6 PM)',
            'INFO'
        )
        self.start_hour = 9
        self.end_hour = 18
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        order_time_str = data.get('order_time', '')
        
        try:
            order_time = datetime.fromisoformat(order_time_str)
            hour = order_time.hour
            
            if hour < self.start_hour or hour >= self.end_hour:
                return False, f"Order placed at {hour}:00, outside business hours ({self.start_hour}:00-{self.end_hour}:00)"
        except:
            pass  # If parsing fails, assume valid
        
        return True, None


class MinimumOrderQuantityRule(BusinessRule):
    """Rule: Minimum order quantity per item"""
    
    def __init__(self):
        super().__init__(
            'BR006',
            'Minimum Order Quantity',
            'Each item must meet minimum order quantity',
            'WARNING'
        )
        self.min_quantities = {
            'PROD001': 5,
            'PROD002': 1,
            'PROD003': 10,
            'PROD004': 1
        }
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        items = data.get('items', [])
        
        for item in items:
            product_id = item.get('product_id')
            quantity = item.get('quantity', 0)
            
            min_qty = self.min_quantities.get(product_id, 1)
            
            if quantity < min_qty:
                return False, f"Product {product_id}: quantity {quantity} below minimum {min_qty}"
        
        return True, None


class PaymentMethodRule(BusinessRule):
    """Rule: Payment method must be valid for order amount"""
    
    def __init__(self):
        super().__init__(
            'BR007',
            'Payment Method Validation',
            'Payment method must be appropriate for order amount',
            'ERROR'
        )
        self.cash_limit = 1000.00
    
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        payment_method = data.get('payment_method', '')
        total = data.get('total', 0)
        
        allowed_methods = ['credit_card', 'debit_card', 'cash', 'wire_transfer']
        
        if payment_method not in allowed_methods:
            return False, f"Invalid payment method: {payment_method}"
        
        if payment_method == 'cash' and total > self.cash_limit:
            return False, f"Cash payment not allowed for orders over ${self.cash_limit:.2f}"
        
        return True, None


class BusinessRuleEngine:
    """Engine to execute business rules"""
    
    def __init__(self):
        self.rules = [
            OrderValueRule(),
            DiscountLimitRule(),
            InventoryAvailabilityRule(),
            CustomerCreditLimitRule(),
            BusinessHoursRule(),
            MinimumOrderQuantityRule(),
            PaymentMethodRule()
        ]
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against all rules"""
        results = {
            'is_valid': True,
            'violations': [],
            'warnings': [],
            'info': [],
            'rules_checked': 0,
            'rules_passed': 0,
            'rules_failed': 0
        }
        
        for rule in self.rules:
            results['rules_checked'] += 1
            is_valid, message = rule.validate(data)
            
            if not is_valid:
                violation = {
                    'rule_id': rule.rule_id,
                    'rule_name': rule.name,
                    'severity': rule.severity,
                    'message': message
                }
                
                if rule.severity == 'ERROR':
                    results['violations'].append(violation)
                    results['is_valid'] = False
                    results['rules_failed'] += 1
                elif rule.severity == 'WARNING':
                    results['warnings'].append(violation)
                    results['rules_failed'] += 1
                else:  # INFO
                    results['info'].append(violation)
                    results['rules_failed'] += 1
            else:
                results['rules_passed'] += 1
        
        return results


# Agent functions
def initialize_validation_agent(state: BusinessRuleValidationState) -> BusinessRuleValidationState:
    """Initialize with sample transactions"""
    
    transactions = [
        {
            'order_id': 'ORD001',
            'customer_id': 'CUST001',
            'total': 5000.00,
            'discount_percent': 10,
            'payment_method': 'credit_card',
            'order_time': '2024-03-15T10:30:00',
            'items': [
                {'product_id': 'PROD001', 'quantity': 10, 'price': 400.00},
                {'product_id': 'PROD002', 'quantity': 20, 'price': 50.00}
            ]
        },
        {
            'order_id': 'ORD002',
            'customer_id': 'CUST002',
            'total': 75000.00,  # Exceeds max order value
            'discount_percent': 60,  # Exceeds max discount
            'payment_method': 'cash',  # Not allowed for this amount
            'order_time': '2024-03-15T20:00:00',  # Outside business hours
            'items': [
                {'product_id': 'PROD003', 'quantity': 5, 'price': 15000.00}  # Out of stock
            ]
        },
        {
            'order_id': 'ORD003',
            'customer_id': 'CUST003',
            'total': 1500.00,  # Exceeds customer credit limit
            'discount_percent': 5,
            'payment_method': 'debit_card',
            'order_time': '2024-03-15T14:00:00',
            'items': [
                {'product_id': 'PROD001', 'quantity': 2, 'price': 750.00}  # Below minimum quantity
            ]
        },
        {
            'order_id': 'ORD004',
            'customer_id': 'CUST004',
            'total': 2000.00,
            'discount_percent': 15,
            'payment_method': 'credit_card',
            'order_time': '2024-03-15T11:00:00',
            'items': [
                {'product_id': 'PROD002', 'quantity': 30, 'price': 50.00},
                {'product_id': 'PROD004', 'quantity': 10, 'price': 50.00}
            ]
        }
    ]
    
    return {
        **state,
        'transactions': transactions,
        'messages': state['messages'] + [f'Initialized with {len(transactions)} transactions']
    }


def define_business_rules_agent(state: BusinessRuleValidationState) -> BusinessRuleValidationState:
    """Define business rules"""
    
    business_rules = [
        {'rule_id': 'BR001', 'name': 'Order Value Range', 'severity': 'ERROR'},
        {'rule_id': 'BR002', 'name': 'Discount Limit', 'severity': 'ERROR'},
        {'rule_id': 'BR003', 'name': 'Inventory Availability', 'severity': 'ERROR'},
        {'rule_id': 'BR004', 'name': 'Customer Credit Limit', 'severity': 'WARNING'},
        {'rule_id': 'BR005', 'name': 'Business Hours', 'severity': 'INFO'},
        {'rule_id': 'BR006', 'name': 'Minimum Order Quantity', 'severity': 'WARNING'},
        {'rule_id': 'BR007', 'name': 'Payment Method Validation', 'severity': 'ERROR'}
    ]
    
    return {
        **state,
        'business_rules': business_rules,
        'messages': state['messages'] + [f'Defined {len(business_rules)} business rules']
    }


def validate_transactions_agent(state: BusinessRuleValidationState) -> BusinessRuleValidationState:
    """Validate transactions against business rules"""
    
    engine = BusinessRuleEngine()
    validation_results = []
    all_violations = []
    
    for transaction in state['transactions']:
        result = engine.validate(transaction)
        
        validation_results.append({
            'order_id': transaction['order_id'],
            'is_valid': result['is_valid'],
            'rules_checked': result['rules_checked'],
            'rules_passed': result['rules_passed'],
            'rules_failed': result['rules_failed'],
            'error_count': len(result['violations']),
            'warning_count': len(result['warnings']),
            'info_count': len(result['info'])
        })
        
        # Collect all violations
        for violation in result['violations'] + result['warnings'] + result['info']:
            all_violations.append({
                'order_id': transaction['order_id'],
                **violation
            })
    
    valid_count = sum(1 for r in validation_results if r['is_valid'])
    
    return {
        **state,
        'validation_results': validation_results,
        'rule_violations': all_violations,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} transactions: {valid_count} valid, '
            f'{len(validation_results) - valid_count} invalid, {len(all_violations)} total violations'
        ]
    }


def calculate_compliance_score_agent(state: BusinessRuleValidationState) -> BusinessRuleValidationState:
    """Calculate compliance score"""
    
    total_transactions = len(state['validation_results'])
    valid_transactions = sum(1 for r in state['validation_results'] if r['is_valid'])
    
    total_rules_checked = sum(r['rules_checked'] for r in state['validation_results'])
    total_rules_passed = sum(r['rules_passed'] for r in state['validation_results'])
    
    compliance_score = {
        'transaction_compliance_rate': (valid_transactions / total_transactions * 100) if total_transactions > 0 else 0,
        'rule_compliance_rate': (total_rules_passed / total_rules_checked * 100) if total_rules_checked > 0 else 0,
        'total_transactions': total_transactions,
        'valid_transactions': valid_transactions,
        'invalid_transactions': total_transactions - valid_transactions,
        'total_violations': len(state['rule_violations']),
        'violations_by_severity': {
            'ERROR': len([v for v in state['rule_violations'] if v['severity'] == 'ERROR']),
            'WARNING': len([v for v in state['rule_violations'] if v['severity'] == 'WARNING']),
            'INFO': len([v for v in state['rule_violations'] if v['severity'] == 'INFO'])
        }
    }
    
    return {
        **state,
        'compliance_score': compliance_score,
        'messages': state['messages'] + [
            f'Calculated compliance: {compliance_score["transaction_compliance_rate"]:.1f}% transaction compliance, '
            f'{compliance_score["rule_compliance_rate"]:.1f}% rule compliance'
        ]
    }


def generate_validation_report_agent(state: BusinessRuleValidationState) -> BusinessRuleValidationState:
    """Generate comprehensive validation report"""
    
    report_lines = [
        "=" * 80,
        "BUSINESS RULE VALIDATION REPORT",
        "=" * 80,
        "",
        "COMPLIANCE SUMMARY:",
        "-" * 40,
        f"Transaction compliance rate: {state['compliance_score']['transaction_compliance_rate']:.1f}%",
        f"Rule compliance rate: {state['compliance_score']['rule_compliance_rate']:.1f}%",
        f"Total transactions: {state['compliance_score']['total_transactions']}",
        f"Valid transactions: {state['compliance_score']['valid_transactions']}",
        f"Invalid transactions: {state['compliance_score']['invalid_transactions']}",
        f"Total violations: {state['compliance_score']['total_violations']}",
        "",
        "VIOLATIONS BY SEVERITY:",
        "-" * 40,
        f"  ERRORS: {state['compliance_score']['violations_by_severity']['ERROR']}",
        f"  WARNINGS: {state['compliance_score']['violations_by_severity']['WARNING']}",
        f"  INFO: {state['compliance_score']['violations_by_severity']['INFO']}",
        "",
        "BUSINESS RULES DEFINED:",
        "-" * 40
    ]
    
    for rule in state['business_rules']:
        report_lines.append(f"  {rule['rule_id']}: {rule['name']} [{rule['severity']}]")
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        report_lines.append(f"\n{result['order_id']}: {status}")
        report_lines.append(f"  Rules: {result['rules_passed']}/{result['rules_checked']} passed")
        
        if result['error_count'] > 0:
            report_lines.append(f"  Errors: {result['error_count']}")
        if result['warning_count'] > 0:
            report_lines.append(f"  Warnings: {result['warning_count']}")
        if result['info_count'] > 0:
            report_lines.append(f"  Info: {result['info_count']}")
    
    report_lines.extend([
        "",
        "DETAILED VIOLATIONS:",
        "-" * 40
    ])
    
    # Group violations by order
    violations_by_order = {}
    for violation in state['rule_violations']:
        order_id = violation['order_id']
        if order_id not in violations_by_order:
            violations_by_order[order_id] = []
        violations_by_order[order_id].append(violation)
    
    for order_id, violations in violations_by_order.items():
        report_lines.append(f"\n{order_id}:")
        for v in violations:
            severity_icon = "✗" if v['severity'] == 'ERROR' else ("⚠" if v['severity'] == 'WARNING' else "ℹ")
            report_lines.append(f"  {severity_icon} [{v['severity']}] {v['rule_name']}: {v['message']}")
    
    report_lines.extend([
        "",
        "RECOMMENDATIONS:",
        "-" * 40
    ])
    
    if state['compliance_score']['violations_by_severity']['ERROR'] > 0:
        report_lines.append("⚠ Critical errors detected:")
        report_lines.append("  • Review order validation before processing")
        report_lines.append("  • Implement pre-submission validation")
        report_lines.append("  • Add business rule checks in UI")
    
    if state['compliance_score']['transaction_compliance_rate'] < 75:
        report_lines.append("⚠ Low compliance rate:")
        report_lines.append("  • Provide better guidance to users")
        report_lines.append("  • Add real-time validation feedback")
        report_lines.append("  • Review and update business rules")
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive validation report']
    }


# Create the graph
def create_business_rule_validation_graph():
    """Create the business rule validation workflow graph"""
    
    workflow = StateGraph(BusinessRuleValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_validation_agent)
    workflow.add_node("define_rules", define_business_rules_agent)
    workflow.add_node("validate", validate_transactions_agent)
    workflow.add_node("calculate_compliance", calculate_compliance_score_agent)
    workflow.add_node("generate_report", generate_validation_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "validate")
    workflow.add_edge("validate", "calculate_compliance")
    workflow.add_edge("calculate_compliance", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the business rule validation graph
    app = create_business_rule_validation_graph()
    
    # Initialize state
    initial_state: BusinessRuleValidationState = {
        'messages': [],
        'transactions': [],
        'business_rules': [],
        'validation_results': [],
        'rule_violations': [],
        'compliance_score': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("BUSINESS RULE VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nBusiness rule validation pattern execution complete! ✓")

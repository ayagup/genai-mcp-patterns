"""
Semantic Validation MCP Pattern

This pattern demonstrates semantic validation in an agentic MCP system.
Semantic validation ensures data makes logical sense in context,
going beyond syntax to validate meaning and relationships.

Use cases:
- Logical consistency validation
- Contextual appropriateness
- Temporal relationship validation
- Geographic coherence
- Domain-specific meaning validation
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from datetime import datetime, timedelta


# Define the state for semantic validation
class SemanticValidationState(TypedDict):
    """State for tracking semantic validation process"""
    messages: Annotated[List[str], add]
    data_samples: List[Dict[str, Any]]
    semantic_rules: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    semantic_violations: List[Dict[str, Any]]
    context_data: Dict[str, Any]
    statistics: Dict[str, Any]
    report: str


class SemanticValidator:
    """Validator for semantic and logical consistency"""
    
    def __init__(self, context: Dict[str, Any]):
        self.context = context
    
    def validate_temporal_logic(self, start_date: Any, end_date: Any,
                                field_names: tuple = ('start', 'end')) -> tuple[bool, Optional[str]]:
        """Validate temporal relationships"""
        try:
            if isinstance(start_date, str):
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            else:
                start_dt = start_date
            
            if isinstance(end_date, str):
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            else:
                end_dt = end_date
            
            if start_dt >= end_dt:
                return False, f"{field_names[0]} date must be before {field_names[1]} date"
            
            # Check if duration is reasonable (e.g., not more than 100 years)
            duration = end_dt - start_dt
            if duration.days > 36500:  # 100 years
                return False, f"Duration of {duration.days} days seems unreasonable"
            
            return True, None
        except (ValueError, AttributeError) as e:
            return False, f"Invalid date format: {str(e)}"
    
    def validate_age_birthdate_consistency(self, age: int, birthdate: Any) -> tuple[bool, Optional[str]]:
        """Validate age matches birthdate"""
        try:
            if isinstance(birthdate, str):
                birth_dt = datetime.fromisoformat(birthdate.replace('Z', '+00:00'))
            else:
                birth_dt = birthdate
            
            today = datetime.now()
            calculated_age = (today - birth_dt).days // 365
            
            # Allow 1 year tolerance for birthday timing
            if abs(calculated_age - age) > 1:
                return False, f"Age {age} doesn't match birthdate {birth_dt.date()} (calculated age: {calculated_age})"
            
            return True, None
        except (ValueError, AttributeError) as e:
            return False, f"Invalid birthdate: {str(e)}"
    
    def validate_quantity_price_total(self, quantity: float, unit_price: float,
                                     total: float, tolerance: float = 0.01) -> tuple[bool, Optional[str]]:
        """Validate quantity * price = total"""
        calculated_total = quantity * unit_price
        difference = abs(calculated_total - total)
        
        if difference > tolerance:
            return False, f"Total {total} doesn't match quantity({quantity}) * price({unit_price}) = {calculated_total}"
        
        return True, None
    
    def validate_percentage_parts_sum(self, parts: List[float],
                                     expected_total: float = 100.0,
                                     tolerance: float = 0.1) -> tuple[bool, Optional[str]]:
        """Validate percentage parts sum to expected total"""
        actual_sum = sum(parts)
        difference = abs(actual_sum - expected_total)
        
        if difference > tolerance:
            return False, f"Parts sum to {actual_sum} instead of expected {expected_total}"
        
        return True, None
    
    def validate_geographic_coherence(self, city: str, state: str, country: str) -> tuple[bool, Optional[str]]:
        """Validate geographic relationships"""
        # Simplified validation - in real world, use a comprehensive database
        city_state_map = self.context.get('city_state_map', {})
        state_country_map = self.context.get('state_country_map', {})
        
        # Check if city belongs to state
        if city in city_state_map:
            expected_state = city_state_map[city]
            if expected_state != state:
                return False, f"City {city} is in {expected_state}, not {state}"
        
        # Check if state belongs to country
        if state in state_country_map:
            expected_country = state_country_map[state]
            if expected_country != country:
                return False, f"State {state} is in {expected_country}, not {country}"
        
        return True, None
    
    def validate_business_logic(self, status: str, allowed_transitions: Dict[str, List[str]],
                                previous_status: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Validate status transitions make business sense"""
        if previous_status is None:
            # First status should be a valid initial state
            initial_states = self.context.get('initial_states', [])
            if initial_states and status not in initial_states:
                return False, f"Status {status} is not a valid initial state"
            return True, None
        
        # Check if transition is allowed
        allowed = allowed_transitions.get(previous_status, [])
        if status not in allowed:
            return False, f"Transition from {previous_status} to {status} is not allowed"
        
        return True, None
    
    def validate_currency_amount_consistency(self, amount: float, currency: str) -> tuple[bool, Optional[str]]:
        """Validate currency precision matches currency standards"""
        currency_decimals = {
            'USD': 2,
            'EUR': 2,
            'JPY': 0,  # Yen doesn't use decimals
            'BTC': 8   # Bitcoin uses 8 decimals
        }
        
        expected_decimals = currency_decimals.get(currency)
        if expected_decimals is None:
            return True, None  # Unknown currency, can't validate
        
        # Check decimal places
        amount_str = f"{amount:.10f}".rstrip('0')
        if '.' in amount_str:
            actual_decimals = len(amount_str.split('.')[1])
            if actual_decimals > expected_decimals:
                return False, f"Currency {currency} should have max {expected_decimals} decimals, found {actual_decimals}"
        
        return True, None
    
    def validate_contextual_range(self, value: float, field: str,
                                  context_field: str, context_value: Any) -> tuple[bool, Optional[str]]:
        """Validate value is appropriate for context"""
        # Example: Validate discount based on customer tier
        if field == 'discount' and context_field == 'customer_tier':
            max_discounts = {'bronze': 10, 'silver': 20, 'gold': 30, 'platinum': 50}
            max_discount = max_discounts.get(context_value, 0)
            
            if value > max_discount:
                return False, f"Discount {value}% exceeds max {max_discount}% for {context_value} tier"
        
        # Example: Validate credit limit based on credit score
        elif field == 'credit_limit' and context_field == 'credit_score':
            if context_value < 600 and value > 5000:
                return False, f"Credit limit ${value} too high for credit score {context_value}"
            elif context_value < 700 and value > 15000:
                return False, f"Credit limit ${value} too high for credit score {context_value}"
        
        return True, None


# Agent functions
def initialize_semantic_validation_agent(state: SemanticValidationState) -> SemanticValidationState:
    """Initialize with sample data and context"""
    
    today = datetime.now()
    
    data_samples = [
        {
            'id': 1,
            'customer_name': 'John Doe',
            'age': 30,
            'birthdate': (today - timedelta(days=30*365)).isoformat(),  # ~30 years ago
            'order_date': (today - timedelta(days=5)).isoformat(),
            'ship_date': (today + timedelta(days=2)).isoformat(),
            'quantity': 5,
            'unit_price': 19.99,
            'total': 99.95,
            'city': 'New York',
            'state': 'NY',
            'country': 'USA',
            'status': 'pending',
            'previous_status': None,
            'discount': 10,
            'customer_tier': 'gold',
            'currency': 'USD',
            'amount': 99.95
        },
        {
            'id': 2,
            'customer_name': 'Jane Smith',
            'age': 25,
            'birthdate': (today - timedelta(days=40*365)).isoformat(),  # ~40 years ago - MISMATCH
            'order_date': (today + timedelta(days=1)).isoformat(),  # Future order date
            'ship_date': (today - timedelta(days=1)).isoformat(),  # Ship before order - INVALID
            'quantity': 3,
            'unit_price': 10.00,
            'total': 50.00,  # Wrong total: should be 30.00
            'city': 'Los Angeles',
            'state': 'TX',  # Wrong state for LA
            'country': 'USA',
            'status': 'delivered',  # Can't skip from None to delivered
            'previous_status': None,
            'discount': 40,  # Exceeds silver tier max
            'customer_tier': 'silver',
            'currency': 'JPY',
            'amount': 1000.50  # JPY shouldn't have decimals
        },
        {
            'id': 3,
            'customer_name': 'Bob Johnson',
            'age': 45,
            'birthdate': (today - timedelta(days=45*365 + 100)).isoformat(),  # ~45 years
            'order_date': (today - timedelta(days=10)).isoformat(),
            'ship_date': (today - timedelta(days=8)).isoformat(),
            'quantity': 2,
            'unit_price': 25.50,
            'total': 51.00,
            'city': 'Chicago',
            'state': 'IL',
            'country': 'USA',
            'status': 'shipped',
            'previous_status': 'pending',
            'discount': 15,
            'customer_tier': 'bronze',  # Discount exceeds bronze max
            'currency': 'EUR',
            'amount': 45.99
        },
        {
            'id': 4,
            'customer_name': 'Alice Williams',
            'age': 28,
            'birthdate': (today - timedelta(days=28*365 + 50)).isoformat(),
            'order_date': (today - timedelta(days=3)).isoformat(),
            'ship_date': (today + timedelta(days=1)).isoformat(),
            'quantity': 10,
            'unit_price': 5.99,
            'total': 59.90,
            'city': 'Seattle',
            'state': 'WA',
            'country': 'USA',
            'status': 'pending',
            'previous_status': None,
            'discount': 5,
            'customer_tier': 'bronze',
            'currency': 'USD',
            'amount': 59.90
        }
    ]
    
    # Context data for validation
    context_data = {
        'city_state_map': {
            'New York': 'NY',
            'Los Angeles': 'CA',
            'Chicago': 'IL',
            'Houston': 'TX',
            'Seattle': 'WA'
        },
        'state_country_map': {
            'NY': 'USA',
            'CA': 'USA',
            'IL': 'USA',
            'TX': 'USA',
            'WA': 'USA'
        },
        'initial_states': ['pending', 'draft'],
        'status_transitions': {
            'pending': ['processing', 'cancelled'],
            'processing': ['shipped', 'cancelled'],
            'shipped': ['delivered', 'returned'],
            'delivered': ['returned'],
            'cancelled': [],
            'returned': []
        }
    }
    
    return {
        **state,
        'data_samples': data_samples,
        'context_data': context_data,
        'messages': state['messages'] + [f'Initialized with {len(data_samples)} samples and context data']
    }


def define_semantic_rules_agent(state: SemanticValidationState) -> SemanticValidationState:
    """Define semantic validation rules"""
    
    semantic_rules = [
        {
            'rule_id': 'SR001',
            'name': 'Temporal Order',
            'description': 'Order date must be before ship date',
            'type': 'temporal_logic',
            'fields': ['order_date', 'ship_date']
        },
        {
            'rule_id': 'SR002',
            'name': 'Age-Birthdate Consistency',
            'description': 'Age must match birthdate',
            'type': 'age_birthdate',
            'fields': ['age', 'birthdate']
        },
        {
            'rule_id': 'SR003',
            'name': 'Order Total Calculation',
            'description': 'Total must equal quantity * unit_price',
            'type': 'quantity_price_total',
            'fields': ['quantity', 'unit_price', 'total']
        },
        {
            'rule_id': 'SR004',
            'name': 'Geographic Coherence',
            'description': 'City, state, and country must be consistent',
            'type': 'geographic',
            'fields': ['city', 'state', 'country']
        },
        {
            'rule_id': 'SR005',
            'name': 'Status Transition',
            'description': 'Status transitions must follow business rules',
            'type': 'business_logic',
            'fields': ['status', 'previous_status']
        },
        {
            'rule_id': 'SR006',
            'name': 'Contextual Discount',
            'description': 'Discount must be appropriate for customer tier',
            'type': 'contextual_range',
            'fields': ['discount', 'customer_tier']
        },
        {
            'rule_id': 'SR007',
            'name': 'Currency Precision',
            'description': 'Amount precision must match currency standards',
            'type': 'currency_precision',
            'fields': ['amount', 'currency']
        }
    ]
    
    return {
        **state,
        'semantic_rules': semantic_rules,
        'messages': state['messages'] + [f'Defined {len(semantic_rules)} semantic rules']
    }


def validate_semantics_agent(state: SemanticValidationState) -> SemanticValidationState:
    """Validate semantic rules"""
    
    validator = SemanticValidator(state['context_data'])
    validation_results = []
    all_violations = []
    
    for sample in state['data_samples']:
        sample_id = sample.get('id')
        violations = []
        
        for rule in state['semantic_rules']:
            is_valid = False
            error = None
            
            if rule['type'] == 'temporal_logic':
                is_valid, error = validator.validate_temporal_logic(
                    sample.get('order_date'),
                    sample.get('ship_date'),
                    ('order', 'ship')
                )
            elif rule['type'] == 'age_birthdate':
                is_valid, error = validator.validate_age_birthdate_consistency(
                    sample.get('age'),
                    sample.get('birthdate')
                )
            elif rule['type'] == 'quantity_price_total':
                is_valid, error = validator.validate_quantity_price_total(
                    sample.get('quantity'),
                    sample.get('unit_price'),
                    sample.get('total')
                )
            elif rule['type'] == 'geographic':
                is_valid, error = validator.validate_geographic_coherence(
                    sample.get('city'),
                    sample.get('state'),
                    sample.get('country')
                )
            elif rule['type'] == 'business_logic':
                is_valid, error = validator.validate_business_logic(
                    sample.get('status'),
                    state['context_data']['status_transitions'],
                    sample.get('previous_status')
                )
            elif rule['type'] == 'contextual_range':
                is_valid, error = validator.validate_contextual_range(
                    sample.get('discount'),
                    'discount',
                    'customer_tier',
                    sample.get('customer_tier')
                )
            elif rule['type'] == 'currency_precision':
                is_valid, error = validator.validate_currency_amount_consistency(
                    sample.get('amount'),
                    sample.get('currency')
                )
            
            if not is_valid:
                violations.append({
                    'rule_id': rule['rule_id'],
                    'rule_name': rule['name'],
                    'description': rule['description'],
                    'error': error
                })
        
        validation_results.append({
            'sample_id': sample_id,
            'is_valid': len(violations) == 0,
            'rules_checked': len(state['semantic_rules']),
            'violations_count': len(violations)
        })
        
        for violation in violations:
            all_violations.append({
                'sample_id': sample_id,
                **violation
            })
    
    valid_count = sum(1 for r in validation_results if r['is_valid'])
    
    return {
        **state,
        'validation_results': validation_results,
        'semantic_violations': all_violations,
        'messages': state['messages'] + [
            f'Validated {len(validation_results)} samples: {valid_count} valid, '
            f'{len(validation_results) - valid_count} with semantic violations'
        ]
    }


def analyze_semantic_violations_agent(state: SemanticValidationState) -> SemanticValidationState:
    """Analyze semantic violations"""
    
    # Violations by rule
    violations_by_rule = {}
    for violation in state['semantic_violations']:
        rule_id = violation['rule_id']
        violations_by_rule[rule_id] = violations_by_rule.get(rule_id, 0) + 1
    
    # Violations by category
    violations_by_category = {}
    for violation in state['semantic_violations']:
        # Extract category from rule
        rule = next((r for r in state['semantic_rules'] if r['rule_id'] == violation['rule_id']), None)
        if rule:
            category = rule['type']
            violations_by_category[category] = violations_by_category.get(category, 0) + 1
    
    statistics = {
        'total_samples': len(state['data_samples']),
        'valid_samples': sum(1 for r in state['validation_results'] if r['is_valid']),
        'invalid_samples': sum(1 for r in state['validation_results'] if not r['is_valid']),
        'total_violations': len(state['semantic_violations']),
        'violations_by_rule': violations_by_rule,
        'violations_by_category': violations_by_category,
        'most_violated_rule': max(violations_by_rule.items(), key=lambda x: x[1])[0] if violations_by_rule else None,
        'pass_rate': (sum(1 for r in state['validation_results'] if r['is_valid']) / 
                     len(state['data_samples']) * 100) if state['data_samples'] else 0
    }
    
    return {
        **state,
        'statistics': statistics,
        'messages': state['messages'] + ['Analyzed semantic violations']
    }


def generate_semantic_report_agent(state: SemanticValidationState) -> SemanticValidationState:
    """Generate comprehensive semantic validation report"""
    
    report_lines = [
        "=" * 80,
        "SEMANTIC VALIDATION REPORT",
        "=" * 80,
        "",
        "VALIDATION SUMMARY:",
        "-" * 40,
        f"Total samples: {state['statistics']['total_samples']}",
        f"Valid samples: {state['statistics']['valid_samples']}",
        f"Invalid samples: {state['statistics']['invalid_samples']}",
        f"Total violations: {state['statistics']['total_violations']}",
        f"Pass rate: {state['statistics']['pass_rate']:.1f}%",
        "",
        "SEMANTIC RULES:",
        "-" * 40
    ]
    
    for rule in state['semantic_rules']:
        report_lines.append(f"• {rule['rule_id']}: {rule['name']}")
        report_lines.append(f"  {rule['description']}")
        report_lines.append(f"  Fields: {', '.join(rule['fields'])}")
    
    report_lines.extend([
        "",
        "VALIDATION RESULTS:",
        "-" * 40
    ])
    
    for result in state['validation_results']:
        status = "✓ VALID" if result['is_valid'] else "✗ INVALID"
        report_lines.append(f"Sample #{result['sample_id']}: {status}")
        if not result['is_valid']:
            report_lines.append(f"  Violations: {result['violations_count']}/{result['rules_checked']}")
    
    report_lines.extend([
        "",
        "VIOLATIONS BY RULE:",
        "-" * 40
    ])
    
    for rule_id, count in sorted(state['statistics']['violations_by_rule'].items(),
                                 key=lambda x: x[1], reverse=True):
        rule = next((r for r in state['semantic_rules'] if r['rule_id'] == rule_id), None)
        rule_name = rule['name'] if rule else rule_id
        report_lines.append(f"  {rule_id} ({rule_name}): {count} violations")
    
    report_lines.extend([
        "",
        "VIOLATIONS BY CATEGORY:",
        "-" * 40
    ])
    
    for category, count in sorted(state['statistics']['violations_by_category'].items(),
                                  key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {category}: {count} violations")
    
    report_lines.extend([
        "",
        "DETAILED VIOLATIONS:",
        "-" * 40
    ])
    
    # Group by sample
    violations_by_sample = {}
    for violation in state['semantic_violations']:
        sample_id = violation['sample_id']
        if sample_id not in violations_by_sample:
            violations_by_sample[sample_id] = []
        violations_by_sample[sample_id].append(violation)
    
    for sample_id, violations in sorted(violations_by_sample.items()):
        report_lines.append(f"\nSample #{sample_id}:")
        for violation in violations:
            report_lines.append(f"  ✗ {violation['rule_id']}: {violation['rule_name']}")
            report_lines.append(f"     Error: {violation['error']}")
    
    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Most violated rule: {state['statistics']['most_violated_rule']}",
        f"✓ Overall pass rate: {state['statistics']['pass_rate']:.1f}%",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement cross-field validation before data submission",
        "• Add business logic validation at service layer",
        "• Use domain-driven design to encode business rules",
        "• Implement contextual validation based on state/tier",
        "• Add semantic validation to integration tests",
        "• Document business rules and constraints clearly",
        "• Consider state machines for status transitions",
        "• Validate geographic data against authoritative sources",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive semantic validation report']
    }


# Create the graph
def create_semantic_validation_graph():
    """Create the semantic validation workflow graph"""
    
    workflow = StateGraph(SemanticValidationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_semantic_validation_agent)
    workflow.add_node("define_rules", define_semantic_rules_agent)
    workflow.add_node("validate", validate_semantics_agent)
    workflow.add_node("analyze", analyze_semantic_violations_agent)
    workflow.add_node("generate_report", generate_semantic_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_rules")
    workflow.add_edge("define_rules", "validate")
    workflow.add_edge("validate", "analyze")
    workflow.add_edge("analyze", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the semantic validation graph
    app = create_semantic_validation_graph()
    
    # Initialize state
    initial_state: SemanticValidationState = {
        'messages': [],
        'data_samples': [],
        'semantic_rules': [],
        'validation_results': [],
        'semantic_violations': [],
        'context_data': {},
        'statistics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("SEMANTIC VALIDATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nSemantic validation pattern execution complete! ✓")

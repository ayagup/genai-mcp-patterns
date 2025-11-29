"""
Pattern 251: Rule-Based Decision MCP Pattern

This pattern demonstrates rule-based decision making - using predefined rules
to make deterministic decisions based on conditions.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class RuleBasedDecisionState(TypedDict):
    """State for rule-based decision workflow"""
    messages: Annotated[List[str], add]
    input_data: Dict[str, Any]
    rules_applied: List[Dict[str, Any]]
    decision: str


class Rule:
    """Represents a decision rule"""
    
    def __init__(self, name: str, condition, action: str):
        self.name = name
        self.condition = condition
        self.action = action
    
    def evaluate(self, data: Dict[str, Any]) -> bool:
        """Evaluate rule condition"""
        try:
            return self.condition(data)
        except:
            return False


class RuleEngine:
    """Rule-based decision engine"""
    
    def __init__(self):
        self.rules = []
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup decision rules"""
        self.rules = [
            Rule("High Priority", lambda d: d.get("priority", 0) >= 8, "ESCALATE_IMMEDIATELY"),
            Rule("Security Issue", lambda d: "security" in str(d.get("type", "")).lower(), "ALERT_SECURITY_TEAM"),
            Rule("Performance Issue", lambda d: d.get("response_time", 0) > 1000, "OPTIMIZE_PERFORMANCE"),
            Rule("Low Priority", lambda d: d.get("priority", 0) <= 3, "SCHEDULE_FOR_LATER"),
            Rule("Normal Processing", lambda d: True, "PROCESS_NORMALLY")
        ]
    
    def decide(self, data: Dict[str, Any]) -> tuple:
        """Make decision based on rules"""
        applied_rules = []
        
        for rule in self.rules:
            if rule.evaluate(data):
                applied_rules.append({
                    "rule": rule.name,
                    "action": rule.action
                })
                return rule.action, applied_rules
        
        return "NO_ACTION", applied_rules


def receive_input_agent(state: RuleBasedDecisionState) -> RuleBasedDecisionState:
    """Receive input data"""
    print("\n📥 Receiving Input Data...")
    
    input_data = {
        "type": "security_vulnerability",
        "priority": 9,
        "response_time": 1500,
        "source": "automated_scan"
    }
    
    print(f"\n  Input Data:")
    for key, value in input_data.items():
        print(f"    {key}: {value}")
    
    return {**state, "input_data": input_data, "messages": ["✓ Input received"]}


def apply_rules_agent(state: RuleBasedDecisionState) -> RuleBasedDecisionState:
    """Apply rules and make decision"""
    print("\n⚙️ Applying Rules...")
    
    engine = RuleEngine()
    decision, applied_rules = engine.decide(state["input_data"])
    
    print(f"\n  Rules Evaluated:")
    for rule_info in applied_rules:
        print(f"    ✓ {rule_info['rule']} → {rule_info['action']}")
    
    print(f"\n  Final Decision: {decision}")
    
    return {
        **state,
        "rules_applied": applied_rules,
        "decision": decision,
        "messages": [f"✓ Decision: {decision}"]
    }


def generate_report_agent(state: RuleBasedDecisionState) -> RuleBasedDecisionState:
    """Generate decision report"""
    print("\n" + "="*70)
    print("RULE-BASED DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Input: {state['input_data']}")
    print(f"\n⚙️ Rules Applied: {len(state['rules_applied'])}")
    for rule in state["rules_applied"]:
        print(f"  • {rule['rule']}: {rule['action']}")
    
    print(f"\n✅ Decision: {state['decision']}")
    
    print("\n💡 Benefits:")
    print("  • Deterministic & predictable")
    print("  • Transparent reasoning")
    print("  • Easy to audit")
    print("  • Consistent decisions")
    
    print("\n="*70)
    return {**state, "messages": ["✓ Report generated"]}


def create_rule_based_decision_graph():
    workflow = StateGraph(RuleBasedDecisionState)
    workflow.add_node("receive", receive_input_agent)
    workflow.add_node("apply_rules", apply_rules_agent)
    workflow.add_node("report", generate_report_agent)
    workflow.add_edge(START, "receive")
    workflow.add_edge("receive", "apply_rules")
    workflow.add_edge("apply_rules", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 251: Rule-Based Decision MCP Pattern")
    print("="*70)
    
    app = create_rule_based_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "input_data": {},
        "rules_applied": [],
        "decision": ""
    })
    print("\n✅ Rule-Based Decision Pattern Complete!")


if __name__ == "__main__":
    main()

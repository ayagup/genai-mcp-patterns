"""
Pattern 255: Threshold-Based Decision MCP Pattern

This pattern demonstrates threshold-based decision making - making decisions
when values cross predefined thresholds or boundaries.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class ThresholdBasedDecisionState(TypedDict):
    """State for threshold-based decision workflow"""
    messages: Annotated[List[str], add]
    metrics: Dict[str, float]
    thresholds: Dict[str, Dict[str, Any]]
    threshold_violations: List[Dict[str, Any]]
    actions: List[str]


class ThresholdMonitor:
    """Monitors thresholds and triggers actions"""
    
    def __init__(self):
        self.thresholds = {
            "cpu_usage": {
                "critical": 90,
                "warning": 75,
                "normal": 50,
                "action_map": {
                    "critical": "SCALE_OUT_IMMEDIATELY",
                    "warning": "ALERT_OPERATIONS",
                    "normal": "MONITOR"
                }
            },
            "error_rate": {
                "critical": 5.0,
                "warning": 2.0,
                "normal": 0.5,
                "action_map": {
                    "critical": "ROLLBACK_DEPLOYMENT",
                    "warning": "INVESTIGATE",
                    "normal": "CONTINUE"
                }
            },
            "response_time": {
                "critical": 2000,
                "warning": 1000,
                "normal": 500,
                "action_map": {
                    "critical": "ENABLE_CACHING",
                    "warning": "OPTIMIZE_QUERIES",
                    "normal": "MAINTAIN"
                }
            },
            "memory_usage": {
                "critical": 95,
                "warning": 80,
                "normal": 60,
                "action_map": {
                    "critical": "RESTART_SERVICES",
                    "warning": "CLEAR_CACHE",
                    "normal": "MONITOR"
                }
            }
        }
    
    def check_threshold(self, metric_name: str, value: float) -> tuple:
        """Check if value crosses thresholds"""
        if metric_name not in self.thresholds:
            return "unknown", "NO_ACTION"
        
        threshold_config = self.thresholds[metric_name]
        
        if value >= threshold_config["critical"]:
            level = "critical"
        elif value >= threshold_config["warning"]:
            level = "warning"
        elif value <= threshold_config["normal"]:
            level = "normal"
        else:
            level = "elevated"
        
        action = threshold_config["action_map"].get(level, "MONITOR")
        
        return level, action
    
    def evaluate_all(self, metrics: Dict[str, float]) -> tuple:
        """Evaluate all metrics against thresholds"""
        violations = []
        actions = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                level, action = self.check_threshold(metric_name, value)
                
                threshold_config = self.thresholds[metric_name]
                
                violation = {
                    "metric": metric_name,
                    "value": value,
                    "level": level,
                    "action": action,
                    "thresholds": {
                        "critical": threshold_config["critical"],
                        "warning": threshold_config["warning"],
                        "normal": threshold_config["normal"]
                    }
                }
                violations.append(violation)
                
                if action != "MONITOR" and action != "MAINTAIN" and action != "CONTINUE":
                    actions.append(action)
        
        return violations, actions


def collect_metrics_agent(state: ThresholdBasedDecisionState) -> ThresholdBasedDecisionState:
    """Collect system metrics"""
    print("\n📊 Collecting System Metrics...")
    
    # Simulated metrics
    metrics = {
        "cpu_usage": 82.5,        # Warning level
        "error_rate": 6.2,        # Critical level
        "response_time": 1200,    # Warning level
        "memory_usage": 65.0      # Normal level
    }
    
    print(f"\n  Current Metrics:")
    for metric, value in metrics.items():
        print(f"    {metric}: {value}")
    
    return {
        **state,
        "metrics": metrics,
        "messages": [f"✓ Collected {len(metrics)} metrics"]
    }


def evaluate_thresholds_agent(state: ThresholdBasedDecisionState) -> ThresholdBasedDecisionState:
    """Evaluate metrics against thresholds"""
    print("\n⚠️ Evaluating Thresholds...")
    
    monitor = ThresholdMonitor()
    violations, actions = monitor.evaluate_all(state["metrics"])
    
    print(f"\n  Threshold Evaluation Results:")
    for violation in violations:
        symbol = "🔴" if violation["level"] == "critical" else \
                "🟡" if violation["level"] == "warning" else \
                "🟢" if violation["level"] == "normal" else "⚪"
        
        print(f"\n    {symbol} {violation['metric']}: {violation['value']}")
        print(f"      Level: {violation['level'].upper()}")
        print(f"      Thresholds: Normal≤{violation['thresholds']['normal']}, "
              f"Warning≥{violation['thresholds']['warning']}, "
              f"Critical≥{violation['thresholds']['critical']}")
        print(f"      Action: {violation['action']}")
    
    if actions:
        print(f"\n  Required Actions: {len(actions)}")
        for action in actions:
            print(f"    • {action}")
    else:
        print(f"\n  No immediate actions required")
    
    return {
        **state,
        "thresholds": monitor.thresholds,
        "threshold_violations": violations,
        "actions": actions,
        "messages": [f"✓ Evaluated {len(violations)} metrics"]
    }


def generate_threshold_report_agent(state: ThresholdBasedDecisionState) -> ThresholdBasedDecisionState:
    """Generate threshold-based decision report"""
    print("\n" + "="*70)
    print("THRESHOLD-BASED DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Metrics Monitored: {len(state['metrics'])}")
    
    print(f"\n⚠️ Threshold Status:")
    
    # Group by severity
    critical = [v for v in state["threshold_violations"] if v["level"] == "critical"]
    warning = [v for v in state["threshold_violations"] if v["level"] == "warning"]
    normal = [v for v in state["threshold_violations"] if v["level"] == "normal"]
    
    if critical:
        print(f"\n  🔴 CRITICAL ({len(critical)}):")
        for v in critical:
            print(f"    • {v['metric']}: {v['value']} → {v['action']}")
    
    if warning:
        print(f"\n  🟡 WARNING ({len(warning)}):")
        for v in warning:
            print(f"    • {v['metric']}: {v['value']} → {v['action']}")
    
    if normal:
        print(f"\n  🟢 NORMAL ({len(normal)}):")
        for v in normal:
            print(f"    • {v['metric']}: {v['value']}")
    
    print(f"\n🎯 Actions Required:")
    if state["actions"]:
        for i, action in enumerate(state["actions"], 1):
            print(f"  {i}. {action}")
    else:
        print(f"  None - All metrics within acceptable limits")
    
    print(f"\n📋 Threshold Configuration:")
    for metric, config in state["thresholds"].items():
        if metric in state["metrics"]:
            print(f"\n  {metric}:")
            print(f"    Normal: ≤ {config['normal']}")
            print(f"    Warning: ≥ {config['warning']}")
            print(f"    Critical: ≥ {config['critical']}")
    
    print("\n💡 Threshold-Based Decision Benefits:")
    print("  • Automated monitoring")
    print("  • Immediate response to issues")
    print("  • Clear decision boundaries")
    print("  • Proactive problem detection")
    print("  • Consistent trigger points")
    print("  • Reduced manual intervention")
    
    print("\n="*70)
    print("✅ Threshold-Based Decision Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_threshold_based_decision_graph():
    workflow = StateGraph(ThresholdBasedDecisionState)
    workflow.add_node("collect", collect_metrics_agent)
    workflow.add_node("evaluate", evaluate_thresholds_agent)
    workflow.add_node("report", generate_threshold_report_agent)
    workflow.add_edge(START, "collect")
    workflow.add_edge("collect", "evaluate")
    workflow.add_edge("evaluate", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 255: Threshold-Based Decision MCP Pattern")
    print("="*70)
    
    app = create_threshold_based_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "metrics": {},
        "thresholds": {},
        "threshold_violations": [],
        "actions": []
    })
    print("\n✅ Threshold-Based Decision Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Self-Healing MCP Pattern

This pattern automatically detects, diagnoses, and recovers from failures
without human intervention, combining monitoring, analysis, and remediation.

Key Features:
- Automatic problem detection
- Root cause analysis
- Automated remediation
- Recovery verification
- Learning from failures
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SelfHealingState(TypedDict):
    """State for self-healing pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    system_name: str
    detected_issues: list[dict[str, str]]  # [{issue, severity, component}]
    root_cause: str
    remediation_plan: list[str]
    remediation_actions: list[dict[str, str]]  # [{action, status, result}]
    recovery_successful: bool
    verification_status: str
    lessons_learned: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Problem Detector
def problem_detector(state: SelfHealingState) -> SelfHealingState:
    """Detects problems automatically through monitoring"""
    system_name = state.get("system_name", "")
    
    system_message = SystemMessage(content="""You are a problem detector. 
    Monitor system health and automatically detect problems.""")
    
    user_message = HumanMessage(content=f"""Detect problems:

System: {system_name}

Monitor and identify any issues in the system.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate detected issues
    detected_issues = [
        {
            "issue": "Database connection pool exhausted",
            "severity": "critical",
            "component": "database",
            "symptoms": "Timeouts, failed queries, connection errors"
        },
        {
            "issue": "Memory leak in API service",
            "severity": "high",
            "component": "api_service",
            "symptoms": "Increasing memory usage, eventual OOM errors"
        },
        {
            "issue": "Disk space low on logs partition",
            "severity": "medium",
            "component": "storage",
            "symptoms": "96% disk usage, log rotation failing"
        }
    ]
    
    detection_report = f"""
    🔍 Problem Detection Results:
    
    Issues Detected: {len(detected_issues)}
    
    Details:
{chr(10).join(f'''    {{"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}}.get(issue["severity"], "⚪") {issue["severity"].upper()}: {issue["issue"]}
       Component: {issue["component"]}
       Symptoms: {issue["symptoms"]}''' for issue in detected_issues)}
    
    Detection Methods:
    • Health check failures
    • Performance degradation
    • Error rate spikes
    • Resource exhaustion
    • Anomaly detection
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Problem Detector:\n{response.content}\n{detection_report}")],
        "detected_issues": detected_issues
    }


# Root Cause Analyzer
def root_cause_analyzer(state: SelfHealingState) -> SelfHealingState:
    """Analyzes detected problems to identify root cause"""
    system_name = state.get("system_name", "")
    detected_issues = state.get("detected_issues", [])
    
    system_message = SystemMessage(content="""You are a root cause analyzer. 
    Analyze detected issues to identify the underlying root cause.""")
    
    issues_summary = "\n".join([
        f"- {issue['issue']} ({issue['severity']}) in {issue['component']}"
        for issue in detected_issues
    ])
    
    user_message = HumanMessage(content=f"""Analyze root cause:

System: {system_name}

Detected Issues:
{issues_summary}

Identify the root cause of these issues.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Analyze and determine root cause
    root_cause = "Database connection pool misconfiguration: max_connections=10 too low for current load (100+ concurrent requests)"
    
    analysis = f"""
    🔬 Root Cause Analysis:
    
    Primary Root Cause:
    {root_cause}
    
    Analysis Process:
    1. Correlation Analysis → Related issues share database component
    2. Timeline Analysis → Issues started after traffic spike
    3. Log Analysis → Connection pool exhaustion messages
    4. Metric Analysis → Connection wait times increased 500%
    5. Configuration Review → Pool size unchanged since deployment
    
    Contributing Factors:
    • Traffic increased 10x over past month
    • Connection pool not auto-scaled
    • No alerts on pool saturation
    • Long-running queries holding connections
    
    Impact Chain:
    Database Pool Exhaustion → API Timeouts → Request Backlog →
    Memory Accumulation → OOM Risk → Service Degradation
    """
    
    return {
        "messages": [AIMessage(content=f"🔬 Root Cause Analyzer:\n{response.content}\n{analysis}")],
        "root_cause": root_cause
    }


# Remediation Planner
def remediation_planner(state: SelfHealingState) -> SelfHealingState:
    """Creates automated remediation plan"""
    root_cause = state.get("root_cause", "")
    detected_issues = state.get("detected_issues", [])
    
    system_message = SystemMessage(content="""You are a remediation planner. 
    Create actionable remediation plans to fix identified problems.""")
    
    user_message = HumanMessage(content=f"""Create remediation plan:

Root Cause: {root_cause}
Issues to Fix: {len(detected_issues)}

Develop step-by-step remediation plan.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create remediation plan
    remediation_plan = [
        "1. Immediate: Increase database connection pool size (10 → 50)",
        "2. Immediate: Restart API service to clear memory leak",
        "3. Immediate: Clean up old logs to free disk space",
        "4. Short-term: Implement connection pool monitoring",
        "5. Short-term: Add auto-scaling for connection pool",
        "6. Long-term: Fix memory leak in API service code",
        "7. Long-term: Implement log rotation and archival"
    ]
    
    plan_report = f"""
    📋 Remediation Plan:
    
{chr(10).join(remediation_plan)}
    
    Plan Priority:
    • Immediate actions: Restore service (steps 1-3)
    • Short-term actions: Prevent recurrence (steps 4-5)
    • Long-term actions: Fix underlying issues (steps 6-7)
    
    Estimated Recovery Time: 5 minutes (immediate actions)
    Risk Level: LOW (standard operations, tested procedures)
    """
    
    return {
        "messages": [AIMessage(content=f"📋 Remediation Planner:\n{response.content}\n{plan_report}")],
        "remediation_plan": remediation_plan
    }


# Auto Remediator
def auto_remediator(state: SelfHealingState) -> SelfHealingState:
    """Executes automated remediation actions"""
    remediation_plan = state.get("remediation_plan", [])
    
    system_message = SystemMessage(content="""You are an auto remediator. 
    Execute remediation actions safely and automatically.""")
    
    user_message = HumanMessage(content=f"""Execute remediation:

Remediation Plan: {len(remediation_plan)} steps

Execute immediate remediation actions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Execute remediation actions
    remediation_actions = [
        {
            "action": "Increase DB connection pool (10→50)",
            "status": "completed",
            "result": "✅ Pool size updated, connections available",
            "duration": "2s"
        },
        {
            "action": "Restart API service",
            "status": "completed",
            "result": "✅ Service restarted, memory usage normalized",
            "duration": "15s"
        },
        {
            "action": "Clean up old logs",
            "status": "completed",
            "result": "✅ 50GB freed, disk usage: 96%→45%",
            "duration": "8s"
        }
    ]
    
    actions_report = f"""
    ⚡ Remediation Execution:
    
    Actions Executed: {len(remediation_actions)}
    
{chr(10).join(f'''    {action["status"] == "completed" and "✅" or "❌"} {action["action"]}
       Status: {action["status"].upper()}
       Result: {action["result"]}
       Duration: {action["duration"]}''' for action in remediation_actions)}
    
    Total Remediation Time: 25 seconds
    
    Safety Measures:
    • Pre-execution validation
    • Rollback capability enabled
    • Change logging active
    • Health monitoring during execution
    """
    
    recovery_successful = all(action["status"] == "completed" for action in remediation_actions)
    
    return {
        "messages": [AIMessage(content=f"⚡ Auto Remediator:\n{response.content}\n{actions_report}")],
        "remediation_actions": remediation_actions,
        "recovery_successful": recovery_successful
    }


# Recovery Verifier
def recovery_verifier(state: SelfHealingState) -> SelfHealingState:
    """Verifies that recovery was successful"""
    system_name = state.get("system_name", "")
    recovery_successful = state.get("recovery_successful", False)
    detected_issues = state.get("detected_issues", [])
    
    system_message = SystemMessage(content="""You are a recovery verifier. 
    Verify that remediation actions successfully resolved the issues.""")
    
    user_message = HumanMessage(content=f"""Verify recovery:

System: {system_name}
Remediation Complete: {recovery_successful}
Original Issues: {len(detected_issues)}

Verify system is healthy and issues are resolved.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Verify recovery
    verification_checks = [
        {
            "check": "Database connection pool health",
            "status": "✅ PASS",
            "detail": "Pool utilization: 35/50 (70%), no timeouts"
        },
        {
            "check": "API service health",
            "status": "✅ PASS",
            "detail": "Memory: 512MB stable, no leaks detected"
        },
        {
            "check": "Disk space availability",
            "status": "✅ PASS",
            "detail": "Disk usage: 45%, plenty of headroom"
        },
        {
            "check": "Error rate",
            "status": "✅ PASS",
            "detail": "Error rate: 0.1% (normal baseline)"
        },
        {
            "check": "Response time",
            "status": "✅ PASS",
            "detail": "P95 latency: 120ms (within SLA)"
        }
    ]
    
    all_passed = all(check["status"] == "✅ PASS" for check in verification_checks)
    verification_status = "SUCCESS" if all_passed else "PARTIAL"
    
    verification_report = f"""
    ✅ Recovery Verification:
    
    Verification Status: {verification_status}
    
    Verification Checks:
{chr(10).join(f'''    {check["status"]} {check["check"]}
       {check["detail"]}''' for check in verification_checks)}
    
    Overall: {'✅ System fully recovered' if all_passed else '⚠️ Partial recovery - monitoring required'}
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Recovery Verifier:\n{response.content}\n{verification_report}")],
        "verification_status": verification_status
    }


# Learning System
def learning_system(state: SelfHealingState) -> SelfHealingState:
    """Learns from the incident to improve future responses"""
    detected_issues = state.get("detected_issues", [])
    root_cause = state.get("root_cause", "")
    remediation_plan = state.get("remediation_plan", [])
    recovery_successful = state.get("recovery_successful", False)
    
    system_message = SystemMessage(content="""You are a learning system. 
    Analyze incidents to extract lessons and improve future self-healing.""")
    
    user_message = HumanMessage(content=f"""Learn from incident:

Root Cause: {root_cause}
Recovery: {'Successful' if recovery_successful else 'Failed'}

Extract lessons learned and improvement opportunities.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Extract lessons learned
    lessons_learned = [
        "Monitor connection pool metrics proactively (not just errors)",
        "Implement auto-scaling for connection pools based on load",
        "Set alerts for 80% pool utilization (before exhaustion)",
        "Add circuit breakers to prevent cascade failures",
        "Regular capacity planning reviews (monthly)",
        "Memory leak detection in CI/CD pipeline",
        "Automated log rotation and archival",
        "Document successful remediation patterns"
    ]
    
    learning_report = f"""
    📚 Learning System:
    
    Incident Summary:
    • Root Cause: Connection pool misconfiguration
    • Resolution Time: 25 seconds (automated)
    • Recovery: ✅ Successful
    
    Lessons Learned:
{chr(10).join(f'    • {lesson}' for lesson in lessons_learned)}
    
    Improvements to Self-Healing:
    • Add connection pool remediation to playbook
    • Tune detection thresholds for earlier warning
    • Automate capacity scaling decisions
    • Expand monitoring coverage
    
    Knowledge Base Updates:
    • New remediation pattern: "Database pool exhaustion"
    • Verified fix: Increase pool size + service restart
    • Prevention: Auto-scaling + proactive alerts
    
    Self-Healing Pattern Complete:
    ✅ Detected → ✅ Analyzed → ✅ Planned → ✅ Remediated → ✅ Verified → ✅ Learned
    """
    
    return {
        "messages": [AIMessage(content=f"📚 Learning System:\n{response.content}\n{learning_report}")],
        "lessons_learned": lessons_learned
    }


# Self-Healing Monitor
def self_healing_monitor(state: SelfHealingState) -> SelfHealingState:
    """Monitors and reports on self-healing process"""
    system_name = state.get("system_name", "")
    detected_issues = state.get("detected_issues", [])
    root_cause = state.get("root_cause", "")
    remediation_plan = state.get("remediation_plan", [])
    remediation_actions = state.get("remediation_actions", [])
    recovery_successful = state.get("recovery_successful", False)
    verification_status = state.get("verification_status", "")
    lessons_learned = state.get("lessons_learned", [])
    
    outcome_icon = "✅" if recovery_successful else "❌"
    
    summary = f"""
    {outcome_icon} SELF-HEALING PATTERN COMPLETE
    
    System: {system_name}
    Recovery Status: {'✅ SUCCESSFUL' if recovery_successful else '❌ FAILED'}
    
    Self-Healing Process Summary:
    
    1️⃣ Detection Phase:
    • Issues Detected: {len(detected_issues)}
    • Critical: {sum(1 for i in detected_issues if i.get("severity") == "critical")}
    • High: {sum(1 for i in detected_issues if i.get("severity") == "high")}
    • Medium: {sum(1 for i in detected_issues if i.get("severity") == "medium")}
    
    2️⃣ Analysis Phase:
    • Root Cause: {root_cause[:80]}...
    • Analysis Method: Correlation + Timeline + Logs
    
    3️⃣ Planning Phase:
    • Remediation Steps: {len(remediation_plan)}
    • Immediate Actions: 3
    • Short-term Actions: 2
    • Long-term Actions: 2
    
    4️⃣ Remediation Phase:
    • Actions Executed: {len(remediation_actions)}
    • Successful: {sum(1 for a in remediation_actions if a.get("status") == "completed")}
    • Total Time: 25 seconds
    
    5️⃣ Verification Phase:
    • Status: {verification_status}
    • System Health: ✅ Normal
    • All Checks: Passed
    
    6️⃣ Learning Phase:
    • Lessons Learned: {len(lessons_learned)}
    • Playbook Updated: ✅
    • Monitoring Enhanced: ✅
    
    Self-Healing Capabilities:
    
    Automatic Detection:
    • Health monitoring
    • Anomaly detection
    • Error pattern recognition
    • Performance degradation tracking
    • Resource utilization monitoring
    
    Intelligent Analysis:
    • Root cause analysis
    • Impact assessment
    • Correlation analysis
    • Timeline reconstruction
    • Pattern matching
    
    Automated Remediation:
    • Pre-approved actions
    • Safe execution
    • Rollback capability
    • Parallel execution
    • Progress tracking
    
    Self-Healing Pattern Benefits:
    • Zero-touch recovery (no human intervention)
    • Reduced MTTR (Mean Time To Recovery)
    • 24/7 availability
    • Consistent responses
    • Learning and improvement
    • Cost reduction
    • Better reliability
    
    Self-Healing Maturity Levels:
    
    Level 0: Manual
    • Human detection and remediation
    
    Level 1: Assisted
    • Automated detection
    • Manual remediation
    
    Level 2: Guided
    • Automated detection
    • Recommended remediation
    • Human approval required
    
    Level 3: Automated
    • Automated detection
    • Automated remediation
    • Human notification
    
    Level 4: Autonomous
    • Predictive prevention
    • Self-optimization
    • Continuous learning
    
    Common Self-Healing Actions:
    • Service restart
    • Resource scaling
    • Configuration adjustment
    • Traffic redirection
    • Cache clearing
    • Connection pool resize
    • Circuit breaker activation
    • Failover trigger
    • Log cleanup
    • Memory release
    
    Self-Healing Best Practices:
    • Start with safe, tested actions
    • Implement rollback capability
    • Monitor during remediation
    • Log all actions
    • Verify recovery
    • Learn from incidents
    • Gradual automation
    • Human oversight initially
    • Clear escalation path
    • Regular testing
    
    Safety Considerations:
    ⚠️ Avoid destructive actions without verification
    ⚠️ Implement rate limiting on remediation
    ⚠️ Require human approval for high-risk actions
    ⚠️ Test remediation in staging first
    ⚠️ Monitor for oscillation/flapping
    ⚠️ Set remediation timeouts
    
    Key Metrics:
    • MTTR: 25 seconds (target: < 5 min)
    • Detection Time: 5 seconds
    • Analysis Time: 8 seconds
    • Remediation Time: 25 seconds
    • Verification Time: 3 seconds
    • Success Rate: 100%
    • Auto-Resolution Rate: 95%
    
    Integration with Other Patterns:
    • Health Check: Continuous monitoring
    • Circuit Breaker: Prevent cascades
    • Failover: Automatic switching
    • Retry: Transient failures
    • Graceful Degradation: Partial service
    
    Key Insight:
    Self-healing pattern combines detection, analysis, and automated remediation
    to recover from failures without human intervention. Essential for highly
    available, resilient systems. Reduces operational burden and improves MTTR.
    The holy grail of operational excellence.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Self-Healing Monitor:\n{summary}")]
    }


# Build the graph
def build_self_healing_graph():
    """Build the self-healing pattern graph"""
    workflow = StateGraph(SelfHealingState)
    
    workflow.add_node("detector", problem_detector)
    workflow.add_node("analyzer", root_cause_analyzer)
    workflow.add_node("planner", remediation_planner)
    workflow.add_node("remediator", auto_remediator)
    workflow.add_node("verifier", recovery_verifier)
    workflow.add_node("learner", learning_system)
    workflow.add_node("monitor", self_healing_monitor)
    
    workflow.add_edge(START, "detector")
    workflow.add_edge("detector", "analyzer")
    workflow.add_edge("analyzer", "planner")
    workflow.add_edge("planner", "remediator")
    workflow.add_edge("remediator", "verifier")
    workflow.add_edge("verifier", "learner")
    workflow.add_edge("learner", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_self_healing_graph()
    
    print("=== Self-Healing MCP Pattern ===\n")
    print("Demonstrating end-to-end self-healing process...\n")
    
    state = {
        "messages": [],
        "system_name": "E-Commerce Platform",
        "detected_issues": [],
        "root_cause": "",
        "remediation_plan": [],
        "remediation_actions": [],
        "recovery_successful": False,
        "verification_status": "",
        "lessons_learned": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("SELF-HEALING DEMONSTRATION COMPLETE")
    print("="*70)
    print(f"\nSystem: {state['system_name']}")
    print(f"Issues Detected: {len(result.get('detected_issues', []))}")
    print(f"Recovery: {'✅ Successful' if result.get('recovery_successful', False) else '❌ Failed'}")
    print(f"Resolution Time: 25 seconds")
    print(f"Lessons Learned: {len(result.get('lessons_learned', []))}")
    print(f"\nSelf-Healing Status: ✅ FULLY OPERATIONAL")

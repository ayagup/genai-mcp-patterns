"""
Zero-Trust MCP Pattern

This pattern implements the Zero-Trust security model: "Never trust, always verify".
All access requests are authenticated, authorized, and encrypted regardless of source.

Key Features:
- Continuous verification
- Least privilege access
- Micro-segmentation
- Device trust assessment
- Assume breach mentality
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ZeroTrustState(TypedDict):
    """State for zero-trust pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_id: str
    device_id: str
    resource_id: str
    requested_action: str
    source_location: str  # IP address or network location
    source_network: str  # "internal", "external", "unknown"
    device_trust_score: float  # 0.0 to 1.0
    user_risk_score: float  # 0.0 to 1.0
    context_risk_score: float  # 0.0 to 1.0
    overall_trust_score: float  # 0.0 to 1.0
    verification_results: Dict[str, bool]
    access_granted: bool
    security_controls: List[str]
    continuous_monitoring: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Identity Verifier
def identity_verifier(state: ZeroTrustState) -> ZeroTrustState:
    """Continuously verifies identity"""
    user_id = state.get("user_id", "")
    device_id = state.get("device_id", "")
    source_location = state.get("source_location", "")
    
    system_message = SystemMessage(content="""You are an identity verifier in a zero-trust architecture. 
    Continuously verify identity, never assume trust based on network location.""")
    
    user_message = HumanMessage(content=f"""Verify identity:

User: {user_id}
Device: {device_id}
Location: {source_location}

Apply continuous verification (never trust, always verify).""")
    
    response = llm.invoke([system_message, user_message])
    
    # Multi-factor verification
    verification_checks = {
        "user_authenticated": True,  # Primary authentication
        "mfa_verified": True,  # Multi-factor authentication
        "device_registered": True,  # Device is known
        "certificate_valid": True,  # Device certificate
        "biometric_match": True,  # Biometric verification (if available)
        "behavioral_normal": True  # Behavioral analytics
    }
    
    # Calculate user risk score based on verifications
    passed_checks = sum(verification_checks.values())
    total_checks = len(verification_checks)
    user_risk_score = 1.0 - (passed_checks / total_checks)  # Lower is better
    
    verification_report = f"""
    🔐 Identity Verification:
    
    User Information:
    • User ID: {user_id}
    • Device ID: {device_id}
    • Source: {source_location}
    
    Verification Checks:
    """
    
    for check, status in verification_checks.items():
        icon = "✅" if status else "❌"
        verification_report += f"\n  {icon} {check.replace('_', ' ').title()}"
    
    verification_report += f"""
    
    User Risk Score: {user_risk_score:.2f} {'🟢 LOW' if user_risk_score < 0.3 else '🟡 MEDIUM' if user_risk_score < 0.7 else '🔴 HIGH'}
    
    Zero-Trust Identity Principles:
    
    1. Never Trust, Always Verify:
       • No implicit trust
       • Verify every request
       • Continuous authentication
       • No perimeter assumption
    
    2. Multi-Factor Authentication:
       • Something you know (password)
       • Something you have (token, phone)
       • Something you are (biometric)
       • Contextual factors (location, behavior)
    
    3. Least Privilege:
       • Minimum necessary access
       • Just-in-time access
       • Time-limited permissions
       • Regularly reviewed
    
    4. Continuous Monitoring:
       • Real-time verification
       • Anomaly detection
       • Behavioral analytics
       • Adaptive authentication
    
    Identity Verification Methods:
    • Password + MFA
    • Certificate-based authentication
    • Biometric verification
    • Behavioral biometrics
    • Risk-based authentication
    • Continuous authentication
    
    Verification Frequency:
    • Initial authentication
    • Periodic re-authentication
    • On sensitive operations
    • On risk score changes
    • On context changes
    """
    
    return {
        "messages": [AIMessage(content=f"🔐 Identity Verifier:\n{response.content}\n{verification_report}")],
        "user_risk_score": user_risk_score,
        "verification_results": verification_checks
    }


# Device Trust Assessor
def device_trust_assessor(state: ZeroTrustState) -> ZeroTrustState:
    """Assesses device trustworthiness"""
    device_id = state.get("device_id", "")
    user_id = state.get("user_id", "")
    
    system_message = SystemMessage(content="""You are a device trust assessor. 
    Evaluate device security posture before granting access.""")
    
    user_message = HumanMessage(content=f"""Assess device trust:

Device: {device_id}
User: {user_id}

Evaluate device security posture.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Device security checks
    device_checks = {
        "os_updated": True,  # Operating system up to date
        "antivirus_active": True,  # Antivirus running
        "firewall_enabled": True,  # Firewall active
        "disk_encrypted": True,  # Full disk encryption
        "screen_lock": True,  # Screen lock enabled
        "no_jailbreak": True,  # Not jailbroken/rooted
        "mdm_compliant": True,  # MDM policy compliant
        "app_whitelist": True  # Only approved apps
    }
    
    # Calculate device trust score
    passed_checks = sum(device_checks.values())
    total_checks = len(device_checks)
    device_trust_score = passed_checks / total_checks
    
    # Device risk based on trust score
    if device_trust_score >= 0.9:
        risk_level = "🟢 LOW RISK"
    elif device_trust_score >= 0.7:
        risk_level = "🟡 MEDIUM RISK"
    else:
        risk_level = "🔴 HIGH RISK"
    
    device_report = f"""
    📱 Device Trust Assessment:
    
    Device: {device_id}
    
    Security Posture Checks:
    """
    
    for check, status in device_checks.items():
        icon = "✅" if status else "❌"
        device_report += f"\n  {icon} {check.replace('_', ' ').title()}"
    
    device_report += f"""
    
    Device Trust Score: {device_trust_score:.2f} {risk_level}
    Compliance: {int(device_trust_score * 100)}%
    
    Device Security Requirements:
    
    Baseline Security:
    • Updated operating system
    • Active antivirus/EDR
    • Enabled firewall
    • Full disk encryption
    • Screen lock (< 5 min timeout)
    • Strong password/PIN
    
    Advanced Security:
    • TPM/Secure Enclave
    • Verified boot
    • Application control
    • Network isolation
    • DLP (Data Loss Prevention)
    • Mobile Device Management
    
    Zero-Trust Device Posture:
    
    Device Health:
    • OS patch level
    • Security software status
    • Configuration compliance
    • Vulnerability status
    • Threat detection
    
    Device Identity:
    • Unique device ID
    • Hardware attestation
    • Certificate-based auth
    • Device fingerprinting
    
    Device Context:
    • Managed vs. BYOD
    • Corporate vs. personal
    • On-premises vs. remote
    • Trusted vs. untrusted network
    
    Enforcement Actions:
    
    Compliant Device:
    • Full access granted
    • Standard monitoring
    • Regular re-assessment
    
    Non-Compliant Device:
    • Limited access
    • Remediation required
    • Quarantine if critical
    • Notify user and admin
    
    Compromised Device:
    • Access denied
    • Immediate quarantine
    • Revoke certificates
    • Alert security team
    • Incident investigation
    
    Device Management:
    • MDM/EMM solutions
    • Unified Endpoint Management
    • Conditional access policies
    • Compliance policies
    • Remote wipe capability
    """
    
    return {
        "messages": [AIMessage(content=f"📱 Device Trust Assessor:\n{response.content}\n{device_report}")],
        "device_trust_score": device_trust_score
    }


# Context Analyzer
def context_analyzer(state: ZeroTrustState) -> ZeroTrustState:
    """Analyzes contextual risk factors"""
    user_id = state.get("user_id", "")
    source_location = state.get("source_location", "")
    source_network = state.get("source_network", "external")
    resource_id = state.get("resource_id", "")
    requested_action = state.get("requested_action", "")
    
    system_message = SystemMessage(content="""You are a context analyzer. 
    Assess risk based on context: location, time, resource sensitivity, behavior.""")
    
    user_message = HumanMessage(content=f"""Analyze context:

User: {user_id}
Location: {source_location}
Network: {source_network}
Resource: {resource_id}
Action: {requested_action}

Assess contextual risk.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Contextual risk factors
    current_hour = time.localtime().tm_hour
    is_business_hours = 9 <= current_hour <= 17
    
    risk_factors = {
        "external_network": source_network == "external",  # Higher risk
        "after_hours": not is_business_hours,  # Higher risk
        "high_value_resource": "critical" in resource_id.lower() or "admin" in resource_id.lower(),
        "unusual_location": "unknown" in source_location.lower(),
        "sensitive_action": requested_action in ["delete", "export", "modify"],
        "unusual_behavior": False  # Would be calculated from behavioral analytics
    }
    
    # Calculate context risk score
    risk_count = sum(risk_factors.values())
    total_factors = len(risk_factors)
    context_risk_score = risk_count / total_factors
    
    context_report = f"""
    🔍 Context Analysis:
    
    Access Context:
    • User: {user_id}
    • Source Location: {source_location}
    • Network: {source_network.upper()}
    • Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
    • Business Hours: {'Yes ✅' if is_business_hours else 'No ⚠️'}
    
    Risk Factors:
    """
    
    for factor, is_risk in risk_factors.items():
        icon = "⚠️" if is_risk else "✅"
        status = "RISK" if is_risk else "OK"
        context_report += f"\n  {icon} {factor.replace('_', ' ').title()}: {status}"
    
    context_report += f"""
    
    Context Risk Score: {context_risk_score:.2f} {'🟢 LOW' if context_risk_score < 0.3 else '🟡 MEDIUM' if context_risk_score < 0.6 else '🔴 HIGH'}
    Active Risk Factors: {risk_count}/{total_factors}
    
    Contextual Risk Factors:
    
    Network Context:
    • Internal network (lower risk)
    • External network (higher risk)
    • Unknown network (highest risk)
    • VPN connection
    • Network segmentation
    
    Location Context:
    • Known location (lower risk)
    • New location (medium risk)
    • Impossible travel (high risk)
    • Geofencing violations
    • Country-based risk
    
    Temporal Context:
    • Business hours (lower risk)
    • After hours (medium risk)
    • Unusual time patterns
    • Rapid succession access
    • Time-based policies
    
    Resource Context:
    • Public resources (lower risk)
    • Confidential data (medium risk)
    • Critical systems (high risk)
    • Data classification
    • Sensitivity labeling
    
    Behavioral Context:
    • Normal access patterns
    • Unusual behavior
    • Anomaly detection
    • Peer group comparison
    • Machine learning models
    
    Action Context:
    • Read operations (lower risk)
    • Write operations (medium risk)
    • Delete operations (high risk)
    • Administrative actions
    • Bulk operations
    
    Context-Aware Policies:
    • Step-up authentication for high risk
    • Additional verification required
    • Restricted access hours
    • Geofencing enforcement
    • Adaptive access controls
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 Context Analyzer:\n{response.content}\n{context_report}")],
        "context_risk_score": context_risk_score
    }


# Access Decision Engine
def access_decision_engine(state: ZeroTrustState) -> ZeroTrustState:
    """Makes zero-trust access decision"""
    user_id = state.get("user_id", "")
    resource_id = state.get("resource_id", "")
    requested_action = state.get("requested_action", "")
    user_risk_score = state.get("user_risk_score", 0.5)
    device_trust_score = state.get("device_trust_score", 0.5)
    context_risk_score = state.get("context_risk_score", 0.5)
    
    system_message = SystemMessage(content="""You are a zero-trust access decision engine. 
    Grant access only when all conditions are met, apply least privilege.""")
    
    user_message = HumanMessage(content=f"""Make access decision:

User: {user_id}
Resource: {resource_id}
Action: {requested_action}

Risk Scores:
• User Risk: {user_risk_score:.2f}
• Device Trust: {device_trust_score:.2f}
• Context Risk: {context_risk_score:.2f}

Apply zero-trust decision logic.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Calculate overall trust score
    # Lower user risk + higher device trust - context risk
    overall_trust_score = ((1.0 - user_risk_score) + device_trust_score - context_risk_score) / 2.0
    overall_trust_score = max(0.0, min(1.0, overall_trust_score))  # Clamp to 0-1
    
    # Decision thresholds
    TRUST_THRESHOLD = 0.7  # Require 70% trust score
    
    # Access decision
    access_granted = overall_trust_score >= TRUST_THRESHOLD
    
    # Security controls to apply
    security_controls = [
        "encryption_required",
        "audit_logging",
        "session_timeout_15min"
    ]
    
    if context_risk_score > 0.5:
        security_controls.append("step_up_authentication")
        security_controls.append("additional_monitoring")
    
    if device_trust_score < 0.8:
        security_controls.append("restricted_access")
        security_controls.append("data_loss_prevention")
    
    if not access_granted:
        security_controls = ["access_denied", "alert_security_team", "log_attempt"]
    
    decision_report = f"""
    ⚖️ Zero-Trust Access Decision:
    
    Request Summary:
    • User: {user_id}
    • Resource: {resource_id}
    • Action: {requested_action}
    
    Trust Calculation:
    • User Risk Score: {user_risk_score:.2f} (lower is better)
    • Device Trust Score: {device_trust_score:.2f} (higher is better)
    • Context Risk Score: {context_risk_score:.2f} (lower is better)
    • Overall Trust Score: {overall_trust_score:.2f}
    
    Decision: {('✅ ACCESS GRANTED' if access_granted else '❌ ACCESS DENIED')}
    Trust Threshold: {TRUST_THRESHOLD:.2f}
    {'Trust score meets requirements' if access_granted else f'Trust score below threshold (need {TRUST_THRESHOLD:.2f}, got {overall_trust_score:.2f})'}
    
    Security Controls Applied:
    {chr(10).join(['  • ' + ctrl.replace('_', ' ').title() for ctrl in security_controls])}
    
    Zero-Trust Decision Framework:
    
    Trust Calculation:
    • Combine multiple signals
    • Weight by importance
    • Real-time evaluation
    • Continuous re-assessment
    
    Decision Factors:
    • Identity verification (WHO)
    • Device posture (WHAT device)
    • Context analysis (WHEN, WHERE, WHY)
    • Resource sensitivity (WHICH resource)
    • Action risk (WHAT action)
    
    Access Policies:
    
    Grant Access:
    • All verifications pass
    • Trust score above threshold
    • No active security alerts
    • Compliance met
    • Policy permits action
    
    Deny Access:
    • Any critical check fails
    • Trust score too low
    • Active security incident
    • Non-compliant device
    • Policy violation
    
    Conditional Access:
    • Step-up authentication
    • Additional verification
    • Limited scope access
    • Enhanced monitoring
    • Time-limited access
    
    Least Privilege Enforcement:
    • Minimum necessary permissions
    • Time-limited access
    • Scope-limited access
    • Just-in-time access
    • Regular access reviews
    
    Micro-Segmentation:
    • Network segmentation
    • Application segmentation
    • Data segmentation
    • User segmentation
    • Device segmentation
    
    Continuous Monitoring:
    • Real-time verification
    • Anomaly detection
    • Threat intelligence
    • Behavioral analytics
    • Automated response
    """
    
    return {
        "messages": [AIMessage(content=f"⚖️ Access Decision Engine:\n{response.content}\n{decision_report}")],
        "overall_trust_score": overall_trust_score,
        "access_granted": access_granted,
        "security_controls": security_controls,
        "continuous_monitoring": True
    }


# Zero-Trust Monitor
def zero_trust_monitor(state: ZeroTrustState) -> ZeroTrustState:
    """Monitors zero-trust implementation"""
    user_id = state.get("user_id", "")
    device_id = state.get("device_id", "")
    resource_id = state.get("resource_id", "")
    requested_action = state.get("requested_action", "")
    access_granted = state.get("access_granted", False)
    overall_trust_score = state.get("overall_trust_score", 0.0)
    security_controls = state.get("security_controls", [])
    user_risk_score = state.get("user_risk_score", 0.0)
    device_trust_score = state.get("device_trust_score", 0.0)
    context_risk_score = state.get("context_risk_score", 0.0)
    
    summary = f"""
    🛡️ ZERO-TRUST SECURITY COMPLETE
    
    Access Request Summary:
    • User: {user_id}
    • Device: {device_id}
    • Resource: {resource_id}
    • Action: {requested_action}
    • Decision: {('✅ GRANTED' if access_granted else '❌ DENIED')}
    
    Trust Scores:
    • User Risk: {user_risk_score:.2f}
    • Device Trust: {device_trust_score:.2f}
    • Context Risk: {context_risk_score:.2f}
    • Overall Trust: {overall_trust_score:.2f}
    
    Security Controls:
    {chr(10).join(['  • ' + ctrl.replace('_', ' ').title() for ctrl in security_controls])}
    
    Zero-Trust Pattern Process:
    1. Identity Verifier → Continuously verify user identity
    2. Device Trust Assessor → Assess device security posture
    3. Context Analyzer → Analyze contextual risk factors
    4. Access Decision Engine → Grant/deny based on trust score
    5. Monitor → Continuous monitoring and re-assessment
    
    Zero-Trust Architecture:
    
    Core Principles:
    
    1. Never Trust, Always Verify:
       • No implicit trust
       • Verify every request
       • No trust based on location
       • Continuous verification
       • Assume breach
    
    2. Least Privilege Access:
       • Minimum necessary permissions
       • Just-in-time access
       • Time-limited permissions
       • Scope-limited access
       • Regular reviews
    
    3. Micro-Segmentation:
       • Network segmentation
       • Application isolation
       • Workload separation
       • Data segmentation
       • Lateral movement prevention
    
    4. Continuous Monitoring:
       • Real-time verification
       • Anomaly detection
       • Threat intelligence
       • Behavioral analytics
       • Automated response
    
    5. Assume Breach:
       • Prepare for compromise
       • Limit blast radius
       • Quick detection
       • Rapid response
       • Recovery readiness
    
    Zero-Trust Components:
    
    1. Policy Engine:
       • Central decision point
       • Policy evaluation
       • Risk calculation
       • Access decisions
       • Dynamic policies
    
    2. Policy Administrator:
       • Enforce decisions
       • Grant/revoke access
       • Session management
       • Security controls
       • Logging and audit
    
    3. Policy Enforcement Point:
       • Proxy connections
       • Apply controls
       • Monitor traffic
       • Block threats
       • Endpoint protection
    
    4. Identity Provider:
       • User authentication
       • MFA enforcement
       • Identity lifecycle
       • SSO integration
       • Federation
    
    5. Device Security:
       • Posture assessment
       • Compliance checking
       • Health verification
       • Certificate management
       • EDR integration
    
    6. Data Protection:
       • Classification
       • Encryption
       • DLP (Data Loss Prevention)
       • Access controls
       • Audit trails
    
    7. Network Security:
       • Micro-segmentation
       • Software-defined perimeter
       • Encrypted tunnels
       • Traffic inspection
       • Threat prevention
    
    8. Analytics and Monitoring:
       • SIEM integration
       • Behavioral analytics
       • Threat intelligence
       • Incident response
       • Reporting and compliance
    
    Implementation Phases:
    
    Phase 1: Visibility
    • Discover all assets
    • Map data flows
    • Identify users and devices
    • Understand dependencies
    • Baseline normal behavior
    
    Phase 2: Micro-Segmentation:
       • Segment network
       • Isolate workloads
       • Define zones
       • Implement policies
       • Test segmentation
    
    Phase 3: Least Privilege:
       • Review permissions
       • Remove excessive access
       • Implement RBAC
       • JIT access
       • Regular reviews
    
    Phase 4: Continuous Verification:
       • Implement MFA
       • Device posture checking
       • Context-aware policies
       • Behavioral analytics
       • Real-time monitoring
    
    Phase 5: Automation:
       • Automated response
       • Policy automation
       • Orchestration
       • Self-healing
       • Continuous improvement
    
    Zero-Trust Use Cases:
    
    Remote Workforce:
    • Secure remote access
    • BYOD support
    • Anywhere access
    • Cloud applications
    • No VPN bottleneck
    
    Cloud Migration:
    • Multi-cloud security
    • Hybrid environments
    • API security
    • Container security
    • Serverless protection
    
    Insider Threat:
    • Lateral movement prevention
    • Privilege monitoring
    • Anomaly detection
    • Data exfiltration prevention
    • Audit trails
    
    Third-Party Access:
    • Vendor access control
    • Limited scope
    • Time-limited access
    • Activity monitoring
    • Compliance enforcement
    
    IoT Security:
    • Device authentication
    • Segmentation
    • Limited communication
    • Monitoring
    • Patching enforcement
    
    Benefits of Zero-Trust:
    
    Security:
    • Reduced attack surface
    • Limited lateral movement
    • Faster threat detection
    • Breach containment
    • Improved compliance
    
    Business:
    • Support remote work
    • Enable cloud adoption
    • Protect sensitive data
    • Reduce complexity
    • Cost optimization
    
    Operations:
    • Centralized policy
    • Automated enforcement
    • Better visibility
    • Simplified management
    • Faster incident response
    
    Challenges:
    
    Technical:
    • Complexity
    • Legacy systems
    • Performance impact
    • Integration requirements
    • Skill requirements
    
    Organizational:
    • Culture change
    • User experience
    • Implementation time
    • Initial costs
    • Change management
    
    Best Practices:
    
    Start Small:
    • Pilot project
    • High-value assets first
    • Learn and iterate
    • Gradual expansion
    • Measure success
    
    Identity-Centric:
    • Strong authentication
    • MFA everywhere
    • Identity lifecycle
    • Privileged access management
    • Service accounts
    
    Data-Centric:
    • Know your data
    • Classify data
    • Protect at source
    • Encrypt everywhere
    • Monitor access
    
    Automation:
    • Automate policies
    • Automated response
    • Orchestration
    • Self-service
    • Continuous monitoring
    
    Measurement:
    • Define metrics
    • Track progress
    • Measure effectiveness
    • Report to stakeholders
    • Continuous improvement
    
    Zero-Trust vs Traditional Security:
    
    Traditional Perimeter:
    • Trust inside network
    • Perimeter defense
    • Castle-and-moat
    • VPN for remote access
    • Network-based controls
    
    Zero-Trust:
    • Never trust, verify always
    • No trusted network
    • Identity-centric
    • Micro-segmentation
    • Least privilege
    • Continuous monitoring
    
    Frameworks and Standards:
    
    NIST SP 800-207:
    • Zero Trust Architecture
    • Reference architecture
    • Implementation guidance
    • Best practices
    
    Google BeyondCorp:
    • Pioneering ZT model
    • User and device trust
    • Context-aware access
    • No VPN
    
    Forrester Zero Trust:
    • eXtended (ZTX) model
    • Data, networks, workloads
    • People, devices, automation
    
    Key Insight:
    Zero-Trust fundamentally changes security from perimeter-based
    to identity-centric. "Never trust, always verify" applies to
    every access request, regardless of source. Essential for
    modern environments with cloud, mobile, and remote work.
    Reduces breach impact through micro-segmentation and least
    privilege. Requires cultural shift and continuous commitment.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Zero-Trust Monitor:\n{summary}")]
    }


# Build the graph
def build_zero_trust_graph():
    """Build the zero-trust pattern graph"""
    workflow = StateGraph(ZeroTrustState)
    
    workflow.add_node("identity", identity_verifier)
    workflow.add_node("device", device_trust_assessor)
    workflow.add_node("context", context_analyzer)
    workflow.add_node("decision", access_decision_engine)
    workflow.add_node("monitor", zero_trust_monitor)
    
    workflow.add_edge(START, "identity")
    workflow.add_edge("identity", "device")
    workflow.add_edge("device", "context")
    workflow.add_edge("context", "decision")
    workflow.add_edge("decision", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_zero_trust_graph()
    
    print("=== Zero-Trust MCP Pattern ===\n")
    
    # Test Case 1: Trusted user, compliant device, normal context
    print("\n" + "="*70)
    print("TEST CASE 1: Low Risk Access (Should Grant)")
    print("="*70)
    
    state1 = {
        "messages": [],
        "user_id": "john.doe",
        "device_id": "device_12345",
        "resource_id": "/api/reports/sales",
        "requested_action": "read",
        "source_location": "192.168.1.100",
        "source_network": "internal",
        "device_trust_score": 0.0,
        "user_risk_score": 0.0,
        "context_risk_score": 0.0,
        "overall_trust_score": 0.0,
        "verification_results": {},
        "access_granted": False,
        "security_controls": [],
        "continuous_monitoring": False
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nDecision: {'✅ ACCESS GRANTED' if result1.get('access_granted') else '❌ ACCESS DENIED'}")
    print(f"Trust Score: {result1.get('overall_trust_score', 0):.2f}")
    
    # Test Case 2: External access, sensitive resource, after hours
    print("\n\n" + "="*70)
    print("TEST CASE 2: High Risk Access (May Deny or Require Step-Up)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "user_id": "contractor_99",
        "device_id": "device_unknown",
        "resource_id": "/admin/critical_config",
        "requested_action": "delete",
        "source_location": "203.0.113.45",
        "source_network": "external",
        "device_trust_score": 0.0,
        "user_risk_score": 0.0,
        "context_risk_score": 0.0,
        "overall_trust_score": 0.0,
        "verification_results": {},
        "access_granted": False,
        "security_controls": [],
        "continuous_monitoring": False
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nUser: {state2['user_id']}")
    print(f"Resource: {state2['resource_id']}")
    print(f"Action: {state2['requested_action']}")
    print(f"Network: {state2['source_network'].upper()}")
    print(f"Decision: {'GRANTED ✅' if result2.get('access_granted') else 'DENIED ❌'}")
    print(f"Trust Score: {result2.get('overall_trust_score', 0):.2f}")
    print(f"Security Controls: {len(result2.get('security_controls', []))}")

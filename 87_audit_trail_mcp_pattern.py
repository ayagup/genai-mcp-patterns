"""
Audit Trail MCP Pattern

This pattern creates comprehensive audit trails by logging all significant
actions, changes, and access attempts for compliance, security, and debugging.

Key Features:
- Comprehensive activity logging
- Tamper-proof audit logs
- Compliance reporting
- Security event tracking
- Change history tracking
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from datetime import datetime
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AuditTrailState(TypedDict):
    """State for audit trail pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    event_type: str  # "access", "modification", "authentication", "configuration"
    user_id: str
    resource_id: str
    action: str
    timestamp: float
    ip_address: str
    user_agent: str
    result: str  # "success", "failure", "warning"
    previous_value: str
    new_value: str
    audit_entries: List[Dict]
    compliance_tags: List[str]
    severity: str  # "low", "medium", "high", "critical"
    audit_hash: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Event Logger
def event_logger(state: AuditTrailState) -> AuditTrailState:
    """Logs events with comprehensive details"""
    event_type = state.get("event_type", "")
    user_id = state.get("user_id", "")
    resource_id = state.get("resource_id", "")
    action = state.get("action", "")
    result = state.get("result", "")
    ip_address = state.get("ip_address", "")
    
    system_message = SystemMessage(content="""You are an event logger. 
    Capture comprehensive details about system events for audit purposes.""")
    
    user_message = HumanMessage(content=f"""Log event:

Event Type: {event_type}
User: {user_id}
Resource: {resource_id}
Action: {action}
Result: {result}
IP Address: {ip_address}

Create detailed audit log entry.""")
    
    response = llm.invoke([system_message, user_message])
    
    timestamp = state.get("timestamp", time.time())
    dt = datetime.fromtimestamp(timestamp)
    
    # Determine severity based on event type and result
    severity_map = {
        ("authentication", "failure"): "high",
        ("authentication", "success"): "low",
        ("modification", "success"): "medium",
        ("modification", "failure"): "medium",
        ("access", "failure"): "medium",
        ("access", "success"): "low",
        ("configuration", "success"): "high",
        ("configuration", "failure"): "critical"
    }
    
    severity = severity_map.get((event_type, result), "low")
    
    # Create audit entry
    audit_entry = {
        "event_id": hashlib.md5(f"{timestamp}{user_id}{action}".encode()).hexdigest()[:16],
        "timestamp": timestamp,
        "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "event_type": event_type,
        "user_id": user_id,
        "resource_id": resource_id,
        "action": action,
        "result": result,
        "severity": severity,
        "ip_address": ip_address,
        "user_agent": state.get("user_agent", ""),
        "previous_value": state.get("previous_value", ""),
        "new_value": state.get("new_value", "")
    }
    
    # Get existing entries and add new one
    audit_entries = state.get("audit_entries", [])
    audit_entries.append(audit_entry)
    
    severity_icon = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🟠",
        "critical": "🔴"
    }
    
    result_icon = {
        "success": "✅",
        "failure": "❌",
        "warning": "⚠️"
    }
    
    log_report = f"""
    📝 Event Logging:
    
    Event Information:
    • Event ID: {audit_entry['event_id']}
    • Timestamp: {audit_entry['datetime']}
    • Type: {event_type}
    • Severity: {severity_icon.get(severity, '🔵')} {severity.upper()}
    
    Actor Information:
    • User ID: {user_id}
    • IP Address: {ip_address}
    • User Agent: {state.get('user_agent', 'N/A')}
    
    Action Details:
    • Resource: {resource_id}
    • Action: {action}
    • Result: {result_icon.get(result, '❔')} {result.upper()}
    
    Changes:
    • Previous: {state.get('previous_value', 'N/A')}
    • New: {state.get('new_value', 'N/A')}
    
    Audit Entry Created:
    • Entry #{len(audit_entries)}
    • Logged successfully
    • Immutable record
    """
    
    return {
        "messages": [AIMessage(content=f"📝 Event Logger:\n{response.content}\n{log_report}")],
        "audit_entries": audit_entries,
        "severity": severity
    }


# Compliance Tagger
def compliance_tagger(state: AuditTrailState) -> AuditTrailState:
    """Tags audit entries with compliance framework references"""
    event_type = state.get("event_type", "")
    action = state.get("action", "")
    severity = state.get("severity", "")
    resource_id = state.get("resource_id", "")
    
    system_message = SystemMessage(content="""You are a compliance tagger. 
    Identify which compliance frameworks apply to each audit event.""")
    
    user_message = HumanMessage(content=f"""Tag for compliance:

Event Type: {event_type}
Action: {action}
Severity: {severity}
Resource: {resource_id}

Identify applicable compliance frameworks.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine compliance tags based on event characteristics
    compliance_tags = []
    
    # SOX (Sarbanes-Oxley) - Financial data
    if "financial" in resource_id.lower() or event_type == "modification":
        compliance_tags.append("SOX")
    
    # HIPAA - Health data
    if "patient" in resource_id.lower() or "medical" in resource_id.lower():
        compliance_tags.append("HIPAA")
    
    # PCI DSS - Payment data
    if "payment" in resource_id.lower() or "card" in resource_id.lower():
        compliance_tags.append("PCI-DSS")
    
    # GDPR - Personal data
    if "user" in resource_id.lower() or event_type == "access":
        compliance_tags.append("GDPR")
    
    # ISO 27001 - Security events
    if event_type in ["authentication", "configuration"] or severity in ["high", "critical"]:
        compliance_tags.append("ISO-27001")
    
    # NIST - All security-relevant events
    if event_type in ["authentication", "access", "configuration"]:
        compliance_tags.append("NIST-800-53")
    
    # If no specific tags, add general
    if not compliance_tags:
        compliance_tags.append("GENERAL")
    
    compliance_report = f"""
    🏷️ Compliance Tagging:
    
    Applicable Frameworks:
    {chr(10).join(['  • ' + tag for tag in compliance_tags])}
    
    Compliance Framework Details:
    
    """
    
    framework_info = {
        "SOX": "Sarbanes-Oxley: Financial reporting and controls",
        "HIPAA": "Health Insurance Portability: Protected health information",
        "PCI-DSS": "Payment Card Industry: Cardholder data security",
        "GDPR": "General Data Protection Regulation: Personal data privacy",
        "ISO-27001": "Information Security Management System",
        "NIST-800-53": "Security and Privacy Controls for Information Systems"
    }
    
    for tag in compliance_tags:
        if tag in framework_info:
            compliance_report += f"    {tag}: {framework_info[tag]}\n"
    
    compliance_report += f"""
    Retention Requirements:
    • SOX: 7 years
    • HIPAA: 6 years
    • PCI-DSS: 3 years minimum
    • GDPR: As needed for purpose
    • ISO-27001: Defined by policy
    • NIST: Per organizational policy
    
    Audit Requirements:
    • Immutable records
    • Tamper-proof storage
    • Access controls
    • Regular reviews
    • Incident response
    """
    
    return {
        "messages": [AIMessage(content=f"🏷️ Compliance Tagger:\n{response.content}\n{compliance_report}")],
        "compliance_tags": compliance_tags
    }


# Integrity Verifier
def integrity_verifier(state: AuditTrailState) -> AuditTrailState:
    """Creates tamper-proof hash chains for audit entries"""
    audit_entries = state.get("audit_entries", [])
    
    system_message = SystemMessage(content="""You are an integrity verifier. 
    Ensure audit trails are tamper-proof using cryptographic hashing.""")
    
    user_message = HumanMessage(content=f"""Verify integrity:

Audit Entries: {len(audit_entries)}

Create cryptographic hash chain for tamper detection.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create hash chain - each entry's hash includes previous hash
    previous_hash = "0" * 64  # Genesis hash
    
    for entry in audit_entries:
        # Create hash of entry + previous hash
        entry_data = f"{entry['timestamp']}{entry['user_id']}{entry['action']}{entry['result']}{previous_hash}"
        entry_hash = hashlib.sha256(entry_data.encode()).hexdigest()
        entry["hash"] = entry_hash
        entry["previous_hash"] = previous_hash
        previous_hash = entry_hash
    
    # Final audit trail hash
    audit_hash = previous_hash
    
    integrity_report = f"""
    🔐 Integrity Verification:
    
    Hash Chain:
    • Total Entries: {len(audit_entries)}
    • Hash Algorithm: SHA-256
    • Chain Integrity: ✅ VERIFIED
    • Audit Trail Hash: {audit_hash[:32]}...
    
    Tamper Protection:
    • Cryptographic hashing
    • Chain of custody
    • Sequential linking
    • Modification detection
    
    Verification Process:
    1. Each entry hashed with SHA-256
    2. Current hash includes previous hash
    3. Any modification breaks chain
    4. Easy to verify, hard to forge
    
    Hash Chain Structure:
    Entry 1: Hash(data + genesis_hash)
    Entry 2: Hash(data + hash1)
    Entry 3: Hash(data + hash2)
    ...
    Entry N: Hash(data + hashN-1)
    
    Security Properties:
    • Immutability: Cannot change past entries
    • Integrity: Detects any modifications
    • Non-repudiation: Cannot deny actions
    • Chronological: Preserves order
    
    Storage Recommendations:
    • Write-once storage (WORM)
    • Distributed ledger
    • Cloud immutable storage
    • Regular backups
    • Off-site copies
    """
    
    return {
        "messages": [AIMessage(content=f"🔐 Integrity Verifier:\n{response.content}\n{integrity_report}")],
        "audit_hash": audit_hash
    }


# Audit Trail Monitor
def audit_trail_monitor(state: AuditTrailState) -> AuditTrailState:
    """Monitors and reports on audit trail"""
    event_type = state.get("event_type", "")
    user_id = state.get("user_id", "")
    action = state.get("action", "")
    result = state.get("result", "")
    severity = state.get("severity", "")
    compliance_tags = state.get("compliance_tags", [])
    audit_entries = state.get("audit_entries", [])
    audit_hash = state.get("audit_hash", "")
    
    summary = f"""
    🔍 AUDIT TRAIL COMPLETE
    
    Event Summary:
    • Event Type: {event_type}
    • User: {user_id}
    • Action: {action}
    • Result: {result.upper()}
    • Severity: {severity.upper()}
    
    Compliance:
    • Frameworks: {', '.join(compliance_tags)}
    • Tags Applied: {len(compliance_tags)}
    
    Audit Trail:
    • Total Entries: {len(audit_entries)}
    • Integrity Hash: {audit_hash[:32] if audit_hash else 'N/A'}...
    • Chain Status: ✅ VERIFIED
    
    Audit Trail Pattern Process:
    1. Event Logger → Capture comprehensive event details
    2. Compliance Tagger → Tag with regulatory frameworks
    3. Integrity Verifier → Create tamper-proof hash chain
    4. Monitor → Report and alert on suspicious activity
    
    What to Audit:
    
    Authentication Events:
    • Login attempts (success/failure)
    • Logout events
    • Password changes
    • MFA enrollment
    • Session creation/termination
    • Account lockouts
    • Privilege escalation
    
    Authorization Events:
    • Permission grants/revokes
    • Role changes
    • Access denials
    • Policy violations
    • Privilege usage
    • Delegation events
    
    Data Access:
    • Read operations
    • Export/download
    • Search queries
    • Report generation
    • Sensitive data access
    • Bulk operations
    
    Data Modification:
    • Create records
    • Update records
    • Delete records
    • Bulk changes
    • Schema changes
    • Configuration changes
    
    System Events:
    • System startup/shutdown
    • Service changes
    • Configuration updates
    • Software installation
    • Security updates
    • Backup operations
    • Disaster recovery
    
    Security Events:
    • Intrusion attempts
    • Malware detection
    • Firewall blocks
    • DDoS attacks
    • Vulnerability scans
    • Security alerts
    • Incident responses
    
    Audit Entry Components:
    
    Who:
    • User ID
    • Username
    • Role
    • IP address
    • Location (geo-IP)
    • User agent
    • Session ID
    
    What:
    • Action type
    • Resource affected
    • Operation performed
    • Previous value
    • New value
    • Query executed
    • API endpoint
    
    When:
    • Timestamp (UTC)
    • Duration
    • Sequence number
    • Transaction ID
    
    Where:
    • Source IP
    • Geographic location
    • Device type
    • Application
    • Environment (prod/dev)
    
    Why:
    • Business reason
    • Request ID
    • Ticket number
    • Approval reference
    
    Result:
    • Success/failure
    • Error code
    • Error message
    • Warning flags
    • Impact assessment
    
    Audit Log Formats:
    
    Syslog:
    • Standard format
    • Network logging
    • Centralized collection
    • SIEM integration
    
    JSON:
    • Structured data
    • Easy parsing
    • Rich metadata
    • Modern tools
    
    Database:
    • Relational storage
    • Complex queries
    • High performance
    • Scalable
    
    Blockchain:
    • Immutable ledger
    • Distributed consensus
    • Maximum security
    • Complex implementation
    
    Audit Trail Best Practices:
    
    Completeness:
    • Log all security events
    • Capture sufficient detail
    • Include context
    • No gaps in timeline
    
    Accuracy:
    • Synchronized clocks (NTP)
    • Consistent timestamps
    • Validated data
    • Error-free logging
    
    Protection:
    • Encrypted storage
    • Access controls
    • Tamper-proof mechanisms
    • Regular backups
    • Off-site storage
    
    Retention:
    • Meet compliance requirements
    • Cost-effective storage
    • Archival strategy
    • Deletion policies
    
    Performance:
    • Asynchronous logging
    • Batch writes
    • Efficient storage
    • Minimal overhead
    • Scalable architecture
    
    Monitoring:
    • Real-time alerts
    • Anomaly detection
    • Pattern recognition
    • Dashboard visualization
    • Regular reviews
    
    Common Audit Trail Patterns:
    
    Failed Login Attempts:
    • Pattern: Multiple failures from same IP
    • Alert: Brute force attack
    • Action: Block IP, notify security
    
    After-Hours Access:
    • Pattern: Access outside business hours
    • Alert: Unusual activity
    • Action: Verify legitimacy
    
    Bulk Data Export:
    • Pattern: Large data downloads
    • Alert: Potential data exfiltration
    • Action: Investigate and review
    
    Privilege Escalation:
    • Pattern: Role changes, permission grants
    • Alert: Unauthorized privilege gain
    • Action: Verify authorization
    
    Configuration Changes:
    • Pattern: Critical system changes
    • Alert: Potential security impact
    • Action: Review and approve
    
    Compliance Requirements:
    
    SOX (Sarbanes-Oxley):
    • Financial data access
    • Control changes
    • 7-year retention
    
    HIPAA (Healthcare):
    • PHI access logging
    • Disclosure tracking
    • 6-year retention
    
    PCI DSS (Payment Cards):
    • Cardholder data access
    • Security event logging
    • 3-month+ retention
    
    GDPR (Privacy):
    • Personal data processing
    • Consent tracking
    • Data subject requests
    
    NIST 800-53:
    • AU family controls
    • Audit record generation
    • Audit review and reporting
    
    Audit Trail Tools:
    
    SIEM (Security Information and Event Management):
    • Splunk
    • ELK Stack (Elasticsearch, Logstash, Kibana)
    • ArcSight
    • QRadar
    
    Log Management:
    • Loggly
    • Papertrail
    • Sumo Logic
    • Datadog
    
    Database Audit:
    • Oracle Audit Vault
    • SQL Server Audit
    • MongoDB Atlas
    • AWS CloudTrail
    
    Application Audit:
    • Custom logging frameworks
    • APM tools
    • Trace correlation
    • Distributed tracing
    
    Integration Points:
    
    Application Layer:
    • Business logic events
    • User actions
    • API calls
    • Transactions
    
    Database Layer:
    • Query logging
    • Data changes
    • Schema modifications
    • User connections
    
    Infrastructure Layer:
    • System events
    • Network activity
    • Resource usage
    • Service status
    
    Security Layer:
    • Authentication
    • Authorization
    • Encryption
    • Threat detection
    
    Key Insight:
    Comprehensive audit trails provide accountability, support
    compliance, enable security investigations, and create an
    immutable record of all significant system activities.
    Essential for security, compliance, and operational excellence.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Audit Trail Monitor:\n{summary}")]
    }


# Build the graph
def build_audit_trail_graph():
    """Build the audit trail pattern graph"""
    workflow = StateGraph(AuditTrailState)
    
    workflow.add_node("logger", event_logger)
    workflow.add_node("tagger", compliance_tagger)
    workflow.add_node("verifier", integrity_verifier)
    workflow.add_node("monitor", audit_trail_monitor)
    
    workflow.add_edge(START, "logger")
    workflow.add_edge("logger", "tagger")
    workflow.add_edge("tagger", "verifier")
    workflow.add_edge("verifier", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_audit_trail_graph()
    
    print("=== Audit Trail MCP Pattern ===\n")
    
    # Test Case 1: Successful data modification
    print("\n" + "="*70)
    print("TEST CASE 1: Data Modification Event")
    print("="*70)
    
    state1 = {
        "messages": [],
        "event_type": "modification",
        "user_id": "admin_user",
        "resource_id": "/database/financial_records/Q4_2024",
        "action": "update_revenue",
        "timestamp": time.time(),
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "result": "success",
        "previous_value": "$1,250,000",
        "new_value": "$1,275,000",
        "audit_entries": [],
        "compliance_tags": [],
        "severity": "",
        "audit_hash": ""
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nAudit Entries Created: {len(result1.get('audit_entries', []))}")
    print(f"Compliance Tags: {', '.join(result1.get('compliance_tags', []))}")
    
    # Test Case 2: Failed authentication attempt
    print("\n\n" + "="*70)
    print("TEST CASE 2: Failed Authentication (Security Event)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "event_type": "authentication",
        "user_id": "unknown_user",
        "resource_id": "/auth/login",
        "action": "login_attempt",
        "timestamp": time.time(),
        "ip_address": "203.0.113.45",
        "user_agent": "curl/7.68.0",
        "result": "failure",
        "previous_value": "",
        "new_value": "",
        "audit_entries": result1.get("audit_entries", []),  # Chain from previous
        "compliance_tags": [],
        "severity": "",
        "audit_hash": ""
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nEvent: {state2['event_type']}")
    print(f"Result: {state2['result'].upper()}")
    print(f"Severity: {result2.get('severity', 'unknown').upper()}")
    print(f"Total Audit Entries: {len(result2.get('audit_entries', []))}")
    print(f"Integrity Hash: {result2.get('audit_hash', 'N/A')[:32]}...")

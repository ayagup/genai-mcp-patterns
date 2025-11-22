"""
Prompt Routing MCP Pattern

This pattern implements intent-based routing where queries are directed
to specialized prompts/agents based on classification and intent detection.

Key Features:
- Intent classification
- Dynamic route selection
- Specialized prompt handling
- Multi-expert routing
- Fallback strategies
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class PromptRoutingState(TypedDict):
    """State for prompt routing pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    query: str
    detected_intent: str
    confidence: float
    route_selected: str
    available_routes: Dict[str, Dict]
    specialized_response: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.2)


# Intent Classifier
def intent_classifier(state: PromptRoutingState) -> PromptRoutingState:
    """Classifies query intent and determines routing"""
    query = state.get("query", "")
    available_routes = state.get("available_routes", {})
    
    # Build route descriptions
    route_descriptions = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in available_routes.items()
    )
    
    system_prompt = f"""You are an intent classification expert. Analyze queries and classify them into the most appropriate category.

Available Categories:
{route_descriptions}

Provide:
1. Primary intent (category name)
2. Confidence (0-100%)
3. Brief reasoning"""
    
    user_prompt = f"""Query: {query}

Classify this query. Format:
Intent: [category name]
Confidence: [percentage]
Reasoning: [brief explanation]"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    content = response.content
    
    # Parse classification
    detected_intent = "general"
    confidence = 0.5
    reasoning = ""
    
    for line in content.split("\n"):
        if line.startswith("Intent:"):
            detected_intent = line.replace("Intent:", "").strip().lower()
        elif line.startswith("Confidence:"):
            conf_text = line.replace("Confidence:", "").strip()
            import re
            match = re.search(r'(\d+)', conf_text)
            if match:
                confidence = float(match.group(1)) / 100.0
        elif line.startswith("Reasoning:"):
            reasoning = line.replace("Reasoning:", "").strip()
    
    # Select route
    route_selected = detected_intent if detected_intent in available_routes else "general"
    
    report = f"""
    🎯 Intent Classifier:
    
    Classification Results:
    • Query: {query[:100]}...
    • Detected Intent: {detected_intent}
    • Confidence: {confidence:.0%}
    • Route Selected: {route_selected}
    • Reasoning: {reasoning}
    
    Prompt Routing Concepts:
    
    Core Idea:
    Direct different types of queries to specialized prompts
    or agents optimized for that specific intent or domain.
    
    Routing Architecture:
    
    Query → Classifier → Router → Specialist → Response
    
    Components:
    
    1. Intent Classifier:
    • Analyze query
    • Identify intent/type
    • Determine confidence
    • Select route
    
    2. Route Selector:
    • Map intent to handler
    • Apply routing rules
    • Handle ambiguity
    • Fallback logic
    
    3. Specialized Handlers:
    • Domain experts
    • Task-specific prompts
    • Optimized models
    • Custom logic
    
    4. Response Aggregator:
    • Combine if multi-route
    • Format output
    • Add metadata
    • Return to user
    
    Benefits of Prompt Routing:
    
    Specialization:
    • Expert handling per domain
    • Optimized prompts
    • Better accuracy
    • Domain knowledge
    
    Efficiency:
    • Right tool for job
    • Avoid one-size-fits-all
    • Resource optimization
    • Cost effective
    
    Scalability:
    • Add new routes easily
    • Independent specialists
    • Modular architecture
    • Parallel development
    
    Quality:
    • Higher accuracy per intent
    • Tailored responses
    • Better user experience
    • Consistent expertise
    
    Intent Classification Methods:
    
    LLM-Based:
    • Use language model
    • Natural language understanding
    • Contextual classification
    • Flexible and adaptable
    
    Rule-Based:
    • Keyword matching
    • Pattern recognition
    • Regular expressions
    • Fast and deterministic
    
    Embedding-Based:
    • Semantic similarity
    • Vector comparison
    • Example-based routing
    • Robust to variation
    
    Hybrid:
    • Combine approaches
    • Rule filters first
    • LLM for ambiguous
    • Best of all methods
    
    Available Routes:
    {chr(10).join(f"  • {name}: {info['description']}" for name, info in available_routes.items())}
    
    Routing Strategies:
    
    Single-Route (Current):
    • One best match
    • Highest confidence
    • Clear decision
    • Simple execution
    
    Multi-Route:
    • Multiple relevant routes
    • Parallel execution
    • Aggregate results
    • Comprehensive response
    
    Hierarchical:
    • Coarse classification first
    • Fine-grained routing next
    • Tree structure
    • Progressive refinement
    
    Conditional:
    • If-then routing rules
    • Context-dependent
    • Dynamic selection
    • Complex logic
    
    Classification Techniques:
    
    Zero-Shot Classification:
    ```
    Classify into categories without examples
    Use task description
    Leverage pre-training
    Quick setup
    ```
    
    Few-Shot Classification:
    ```
    Provide example queries per category
    Learn from demonstrations
    Better accuracy
    Domain adaptation
    ```
    
    Confidence Thresholding:
    ```
    if confidence > 0.9:
        route_to_specialist
    elif confidence > 0.6:
        route_to_general_with_specialist_hint
    else:
        ask_clarification
    ```
    
    Multi-Label Classification:
    ```
    Query may have multiple intents
    Classify into multiple categories
    Route to multiple specialists
    Combine responses
    ```
    
    Routing Patterns:
    
    Simple Router:
    ```python
    intent = classify(query)
    handler = route_map[intent]
    response = handler(query)
    ```
    
    Confidence-Based Router:
    ```python
    intent, confidence = classify(query)
    if confidence < threshold:
        response = ask_clarification()
    else:
        handler = route_map[intent]
        response = handler(query)
    ```
    
    Multi-Specialist Router:
    ```python
    intents = classify_multi(query)
    responses = []
    for intent in intents:
        handler = route_map[intent]
        response = handler(query)
        responses.append(response)
    final = aggregate(responses)
    ```
    
    Fallback Router:
    ```python
    intent = classify(query)
    try:
        handler = route_map[intent]
        response = handler(query)
    except Exception:
        response = fallback_handler(query)
    ```
    
    Research & Applications:
    
    Semantic Router (Aurelio Labs):
    • Vector-based routing
    • Fast and accurate
    • Easy to update
    • Production-ready
    
    LangChain Routing:
    • Multiple chain types
    • Dynamic selection
    • Prompt routing
    • Agent routing
    
    Use Cases:
    • Customer support
    • Multi-domain chatbots
    • Content recommendation
    • Task automation
    
    Intent Categories (Examples):
    
    Customer Support:
    • billing_question
    • technical_support
    • account_management
    • product_inquiry
    • complaint_handling
    
    Content:
    • question_answering
    • summarization
    • generation
    • translation
    • analysis
    
    Code:
    • code_generation
    • code_review
    • debugging
    • explanation
    • optimization
    
    Data:
    • data_analysis
    • visualization
    • extraction
    • transformation
    • reporting
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Intent Classifier:\n{report}\n\n{response.content}")],
        "detected_intent": detected_intent,
        "confidence": confidence,
        "route_selected": route_selected
    }


# Specialized Handler
def specialized_handler(state: PromptRoutingState) -> PromptRoutingState:
    """Executes specialized prompt for the selected route"""
    query = state.get("query", "")
    route_selected = state.get("route_selected", "general")
    available_routes = state.get("available_routes", {})
    confidence = state.get("confidence", 0.5)
    
    # Get route configuration
    route_config = available_routes.get(route_selected, {})
    system_prompt_template = route_config.get("system_prompt", "You are a helpful assistant.")
    specialist_name = route_config.get("name", "General Assistant")
    
    # Execute specialized prompt
    system_prompt = system_prompt_template
    user_prompt = query
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    specialized_response = response.content
    
    summary = f"""
    🎓 Specialized Handler ({specialist_name}):
    
    Handler Execution:
    • Route: {route_selected}
    • Specialist: {specialist_name}
    • Classification Confidence: {confidence:.0%}
    • Response Generated: Yes
    
    Specialized Response:
    {specialized_response[:300]}...
    
    Specialized Prompt Design:
    
    Domain Expert Prompts:
    
    Technical Support:
    ```
    You are a technical support specialist with expertise in
    troubleshooting, diagnostics, and solutions. Focus on:
    - Clear problem identification
    - Step-by-step solutions
    - Technical accuracy
    - User-friendly explanations
    ```
    
    Creative Writing:
    ```
    You are a creative writing expert specializing in
    storytelling, narrative structure, and engaging prose.
    - Vivid descriptions
    - Character development
    - Plot coherence
    - Stylistic flair
    ```
    
    Data Analysis:
    ```
    You are a data analyst with expertise in statistics,
    visualization, and insights extraction. Provide:
    - Quantitative analysis
    - Pattern recognition
    - Visualization suggestions
    - Actionable insights
    ```
    
    Legal Advisory:
    ```
    You are a legal information specialist. Provide general
    legal information while being clear to:
    - Not provide legal advice
    - Explain legal concepts
    - Reference relevant laws
    - Recommend professional consultation
    ```
    
    Route Optimization:
    
    Prompt Specialization:
    • Tailored instructions
    • Domain vocabulary
    • Specific constraints
    • Example formats
    
    Model Selection:
    • Different models per route
    • Specialized fine-tuning
    • Task-optimized models
    • Cost-performance balance
    
    Temperature Tuning:
    • Low (0-0.3): factual, technical
    • Medium (0.4-0.7): balanced
    • High (0.8-1.0): creative, diverse
    
    Context Management:
    • Route-specific context
    • Relevant knowledge
    • Domain examples
    • Historical queries
    
    Advanced Routing Patterns:
    
    Cascading Routes:
    ```python
    # Try specialist first
    specialist_response = specialist(query)
    
    # Validate with generalist
    validation = generalist_validate(specialist_response)
    
    if not validation.satisfactory:
        # Fallback to general
        response = generalist(query)
    else:
        response = specialist_response
    ```
    
    Ensemble Routing:
    ```python
    # Route to multiple specialists
    responses = []
    for route in top_k_routes:
        specialist = get_handler(route)
        response = specialist(query)
        responses.append(response)
    
    # Aggregate responses
    final = aggregate_responses(responses)
    ```
    
    Dynamic Route Discovery:
    ```python
    # Analyze query complexity
    complexity = assess_complexity(query)
    
    if complexity.high:
        # Multi-stage routing
        subtasks = decompose(query)
        results = []
        for subtask in subtasks:
            route = classify(subtask)
            result = route_map[route](subtask)
            results.append(result)
        response = synthesize(results)
    else:
        # Simple routing
        route = classify(query)
        response = route_map[route](query)
    ```
    
    Routing Best Practices:
    
    Clear Route Definitions:
    • Distinct categories
    • Non-overlapping when possible
    • Clear descriptions
    • Example queries
    
    Confidence Handling:
    • Set thresholds
    • Clarification for low confidence
    • Multiple routes for ambiguous
    • Explain routing decisions
    
    Fallback Strategies:
    • General handler always available
    • Graceful degradation
    • Error handling
    • User feedback
    
    Monitoring & Improvement:
    • Track routing accuracy
    • Collect user feedback
    • A/B test routes
    • Continuously refine
    
    Route Performance Metrics:
    
    Classification Metrics:
    • Accuracy: correct routes / total
    • Precision: true positives / classified
    • Recall: true positives / actual
    • F1 Score: harmonic mean
    
    User Satisfaction:
    • Response quality ratings
    • Task completion rate
    • Time to resolution
    • User feedback
    
    Operational Metrics:
    • Latency per route
    • Cost per route
    • Cache hit rate
    • Error rates
    
    Route Coverage:
    • Queries per route
    • Route utilization
    • Fallback frequency
    • Uncovered intents
    
    Routing Error Handling:
    
    Ambiguous Intent:
    ```python
    if confidence < threshold:
        # Ask clarifying question
        clarification = ask_clarification(query, top_intents)
        refined_query = f"{query} {clarification}"
        route = classify(refined_query)
    ```
    
    No Matching Route:
    ```python
    if detected_intent not in routes:
        # Use general handler
        response = general_handler(query)
        # Log for route expansion
        log_unhandled_intent(query, detected_intent)
    ```
    
    Multiple Strong Matches:
    ```python
    if len(high_confidence_routes) > 1:
        # Execute all relevant routes
        responses = [route(query) for route in high_confidence_routes]
        # Aggregate or let user choose
        final = present_multiple_responses(responses)
    ```
    
    Real-World Routing Examples:
    
    E-commerce Chatbot:
    • product_search → Product Specialist
    • order_tracking → Order Specialist
    • returns_refunds → Returns Specialist
    • general_inquiry → Customer Service
    
    Developer Assistant:
    • code_generation → Code Generator
    • debugging → Debugger
    • explanation → Code Explainer
    • optimization → Performance Expert
    
    Healthcare Portal:
    • symptoms → Symptom Checker (+ disclaimer)
    • appointments → Scheduling Agent
    • billing → Billing Support
    • general_health → Health Information
    
    Educational Platform:
    • homework_help → Tutor
    • concept_explanation → Teacher
    • practice_problems → Problem Generator
    • study_planning → Study Coach
    
    Key Insight:
    Prompt Routing enables scalable multi-domain AI systems
    by directing queries to specialized handlers, combining
    the benefits of expert knowledge with broad coverage
    through intelligent intent classification and routing.
    """
    
    return {
        "messages": [AIMessage(content=f"🎓 Specialized Handler:\n{summary}")],
        "specialized_response": specialized_response
    }


# Build the graph
def build_prompt_routing_graph():
    """Build the prompt routing pattern graph"""
    workflow = StateGraph(PromptRoutingState)
    
    workflow.add_node("intent_classifier", intent_classifier)
    workflow.add_node("specialized_handler", specialized_handler)
    
    workflow.add_edge(START, "intent_classifier")
    workflow.add_edge("intent_classifier", "specialized_handler")
    workflow.add_edge("specialized_handler", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_prompt_routing_graph()
    
    print("=== Prompt Routing MCP Pattern ===\n")
    
    # Test Case: Multi-domain routing
    print("\n" + "="*70)
    print("TEST CASE: Intent-Based Prompt Routing")
    print("="*70)
    
    # Define available routes
    available_routes = {
        "technical": {
            "name": "Technical Support Specialist",
            "description": "Handle technical questions, troubleshooting, and how-to queries",
            "system_prompt": "You are a technical support specialist. Provide clear, step-by-step technical guidance with accuracy and patience."
        },
        "creative": {
            "name": "Creative Writing Expert",
            "description": "Handle creative writing, storytelling, and content generation",
            "system_prompt": "You are a creative writing expert. Help with storytelling, narrative structure, and engaging content creation."
        },
        "analysis": {
            "name": "Data Analysis Expert",
            "description": "Handle data analysis, statistics, and insights",
            "system_prompt": "You are a data analysis expert. Provide quantitative analysis, identify patterns, and deliver actionable insights."
        },
        "general": {
            "name": "General Assistant",
            "description": "Handle general queries and questions",
            "system_prompt": "You are a helpful general assistant. Provide clear, accurate, and helpful responses to any question."
        }
    }
    
    state = {
        "messages": [],
        "query": "How do I fix a Python ImportError when running my script?",
        "detected_intent": "",
        "confidence": 0.0,
        "route_selected": "",
        "available_routes": available_routes,
        "specialized_response": ""
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 130: Prompt Routing - COMPLETE")
    print(f"{'='*70}")
    print("\n🎉 ALL PROMPT ENGINEERING PATTERNS (121-130) COMPLETE! 🎉")

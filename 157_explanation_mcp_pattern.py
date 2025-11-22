"""
Explanation MCP Pattern

This pattern implements explanation generation for making
AI reasoning transparent and understandable.

Key Features:
- Explanation types
- Reasoning traces
- Simplification strategies
- Visual aids
- Verification support
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ExplanationState(TypedDict):
    """State for explanation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    decision: str
    reasoning_trace: List[str]
    explanation_type: str
    target_audience: str
    complexity_level: str


llm = ChatOpenAI(model="gpt-4", temperature=0.4)


def explanation_agent(state: ExplanationState) -> ExplanationState:
    """Generates explanations for decisions"""
    decision = state.get("decision", "")
    audience = state.get("target_audience", "general")
    
    system_prompt = """You are an explanation expert.

Explanation Goals:
• Transparency
• Understanding
• Trust building
• Learning support
• Verification

Make AI interpretable."""
    
    user_prompt = f"""Decision: {decision}
Audience: {audience}

Design explanation system.
Show explanation strategies."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    💡 Explanation Agent:
    
    Explanation System:
    ```python
    class ExplanationSystem:
        def __init__(self):
            self.tracer = ReasoningTracer()
            self.simplifier = ComplexityAdapter()
            self.visualizer = ExplanationVisualizer()
        
        def explain(self, decision, audience):
            '''Generate explanation'''
            # Capture reasoning
            trace = self.tracer.capture_reasoning(decision)
            
            # Adapt to audience
            explanation = self.simplifier.adapt(trace, audience)
            
            # Add visuals
            enhanced = self.visualizer.add_visuals(explanation)
            
            return enhanced
    ```
    
    Explanation Types:
    
    Why Explanation:
    ```python
    def explain_why(decision):
        '''Why this decision?'''
        # Show reasoning chain
        reasoning = [
            "1. Observed: Customer has premium account",
            "2. Rule: Premium customers get priority",
            "3. Therefore: Route to priority queue"
        ]
        
        return format_chain(reasoning)
    ```
    
    Why Not Explanation:
    ```python
    def explain_why_not(decision, alternative):
        '''Why not alternative?'''
        # Compare options
        comparison = {{
            'chosen': {{
                'option': decision,
                'score': 0.85,
                'pros': ['Faster', 'Cheaper'],
                'cons': ['Riskier']
            }},
            'alternative': {{
                'option': alternative,
                'score': 0.72,
                'pros': ['Safer'],
                'cons': ['Slower', 'More expensive']
            }}
        }}
        
        # Explain difference
        return f"Chose {{decision}} because score higher (0.85 vs 0.72)"
    ```
    
    How Explanation:
    ```python
    def explain_how(process):
        '''How was this done?'''
        # Step-by-step process
        steps = [
            "Step 1: Collect data from sensors",
            "Step 2: Normalize values",
            "Step 3: Apply model",
            "Step 4: Threshold result",
            "Step 5: Generate alert"
        ]
        
        return format_steps(steps)
    ```
    
    What-If Explanation:
    ```python
    def explain_what_if(decision, change):
        '''What if conditions were different?'''
        # Counterfactual
        original = evaluate(current_conditions)
        modified = evaluate(change_conditions(change))
        
        explanation = f'''
        Current decision: {{original.decision}}
        
        If {{change}}, then: {{modified.decision}}
        
        Because: {{modified.reason}}
        '''
        
        return explanation
    ```
    
    Reasoning Traces:
    
    Forward Chaining:
    ```python
    def capture_forward_trace(facts, rules):
        '''Show forward reasoning'''
        trace = []
        
        current_facts = facts.copy()
        
        while rules_apply:
            # Find applicable rule
            rule = find_applicable_rule(current_facts, rules)
            
            # Apply rule
            new_fact = apply_rule(rule, current_facts)
            
            # Record
            trace.append({{
                'step': len(trace) + 1,
                'rule': rule.name,
                'input': current_facts,
                'output': new_fact
            }})
            
            current_facts.add(new_fact)
        
        return trace
    ```
    
    Backward Chaining:
    ```python
    def capture_backward_trace(goal, rules):
        '''Show backward reasoning'''
        trace = []
        
        def prove(goal):
            # Find rule that concludes goal
            rule = find_rule_for(goal)
            
            trace.append(f"To prove {{goal}}, need to prove {{rule.conditions}}")
            
            # Recursively prove conditions
            for condition in rule.conditions:
                if not is_fact(condition):
                    prove(condition)
        
        prove(goal)
        
        return reverse(trace)  # Show top-down
    ```
    
    Feature Attribution:
    
    LIME (Local Interpretable Model-agnostic Explanations):
    ```python
    def explain_with_lime(model, instance):
        '''Which features mattered?'''
        # Perturb instance
        perturbed = generate_perturbations(instance)
        
        # Get predictions
        predictions = [model.predict(p) for p in perturbed]
        
        # Fit simple model
        simple_model = fit_linear_model(perturbed, predictions)
        
        # Extract feature importance
        importance = simple_model.coefficients
        
        explanation = format_feature_importance(importance)
        
        return explanation
    ```
    
    SHAP (SHapley Additive exPlanations):
    ```python
    def explain_with_shap(model, instance):
        '''Contribution of each feature'''
        # Calculate Shapley values
        shapley_values = calculate_shapley(model, instance)
        
        # Base prediction
        base = model.predict(baseline)
        
        # Show contribution
        explanation = f'''
        Base prediction: {{base}}
        
        Feature contributions:
        {{format_contributions(shapley_values)}}
        
        Final prediction: {{base + sum(shapley_values)}}
        '''
        
        return explanation
    ```
    
    Simplification Strategies:
    
    Audience Adaptation:
    ```python
    def adapt_to_audience(explanation, audience):
        '''Adjust complexity'''
        if audience == 'expert':
            return technical_explanation(explanation)
        
        elif audience == 'general':
            # Remove jargon
            simplified = remove_jargon(explanation)
            
            # Add analogies
            with_analogies = add_analogies(simplified)
            
            # Use examples
            with_examples = add_examples(with_analogies)
            
            return with_examples
        
        elif audience == 'child':
            return ultra_simple_explanation(explanation)
    ```
    
    Progressive Disclosure:
    ```python
    def progressive_explanation(full_explanation):
        '''Start simple, add detail on request'''
        levels = {{
            'summary': "Decision: A. Main reason: X",
            'basic': "Decision: A. Reasons: X, Y, Z",
            'detailed': "Decision: A. Analysis: [full trace]",
            'expert': "Decision: A. Technical: [all details]"
        }}
        
        # Start with summary
        show(levels['summary'])
        
        # Provide "Learn more" option
        if user_requests_more:
            show(levels['basic'])
    ```
    
    Analogies and Examples:
    
    Analogy Generation:
    ```python
    def generate_analogy(concept):
        '''Explain via comparison'''
        analogies = {{
            'neural_network': 'Like brain neurons',
            'gradient_descent': 'Like walking downhill',
            'overfitting': 'Like memorizing vs understanding'
        }}
        
        analogy = analogies.get(concept)
        
        explanation = f'''
        {{concept}} is like {{analogy}}.
        
        [Detailed comparison]
        '''
        
        return explanation
    ```
    
    Concrete Examples:
    ```python
    def add_examples(abstract_explanation):
        '''Illustrate with examples'''
        # Abstract: "Model classifies based on patterns"
        
        example = '''
        For example, to classify emails:
        - Spam emails often have "FREE" in subject
        - Spam emails often have many exclamation marks
        - Model learned these patterns from training data
        '''
        
        return abstract_explanation + example
    ```
    
    Visual Explanations:
    
    Decision Trees:
    ```python
    def visualize_decision_tree(tree):
        '''Show decision path'''
        visualization = '''
        Start
          ├─ Age > 30?
          │   ├─ Yes → Income > 50k?
          │   │   ├─ Yes → Approve
          │   │   └─ No → Deny
          │   └─ No → Deny
        '''
        
        return visualization
    ```
    
    Saliency Maps:
    ```python
    def show_saliency_map(image, model):
        '''Highlight important regions'''
        # Compute gradients
        gradients = compute_gradients(model, image)
        
        # Create heatmap
        heatmap = generate_heatmap(gradients)
        
        # Overlay on image
        visualization = overlay(image, heatmap)
        
        return visualization
    ```
    
    Attention Visualization:
    ```python
    def visualize_attention(text, model):
        '''Show which words mattered'''
        # Get attention weights
        attention = model.get_attention_weights(text)
        
        # Highlight words
        highlighted = highlight_words(text, attention)
        
        # Example output:
        # "The **movie** was **great**"
        # (bold = high attention)
        
        return highlighted
    ```
    
    Contrastive Explanations:
    
    Minimal Changes:
    ```python
    def contrastive_explanation(instance, model):
        '''Smallest change to flip decision'''
        current_prediction = model.predict(instance)
        
        # Find minimal perturbation
        for feature in instance.features:
            modified = instance.copy()
            modified[feature] = find_threshold(feature)
            
            if model.predict(modified) != current_prediction:
                return f"If {{feature}} was {{modified[feature]}}, decision would flip"
    ```
    
    Verification Support:
    
    Explanation Evaluation:
    ```python
    def evaluate_explanation(explanation):
        '''Check quality'''
        criteria = {{
            'completeness': 'All key factors covered?',
            'correctness': 'Accurately reflects model?',
            'consistency': 'Same explanation for similar cases?',
            'comprehensibility': 'Audience can understand?',
            'actionability': 'Can user act on it?'
        }}
        
        scores = {{criterion: assess(explanation, criterion) 
                   for criterion in criteria}}
        
        return scores
    ```
    
    Interactive Explanation:
    
    Q&A Interface:
    ```python
    def interactive_explanation(decision):
        '''Answer follow-up questions'''
        explanation = generate_base_explanation(decision)
        
        while True:
            question = get_user_question()
            
            if question.startswith("why"):
                answer = explain_why(decision, question)
            elif question.startswith("what if"):
                answer = explain_what_if(decision, question)
            elif question.startswith("how"):
                answer = explain_how(decision, question)
            
            show(answer)
            
            if user_satisfied:
                break
    ```
    
    Best Practices:
    ✓ Tailor to audience
    ✓ Start simple, allow drill-down
    ✓ Use multiple modalities (text, visual)
    ✓ Provide examples
    ✓ Show reasoning chain
    ✓ Enable verification
    ✓ Be honest about limitations
    
    Key Insight:
    Effective explanations build trust by making
    AI reasoning transparent and understandable.
    """
    
    return {
        "messages": [AIMessage(content=f"💡 Explanation Agent:\n{report}\n\n{response.content}")]
    }


def build_explanation_graph():
    workflow = StateGraph(ExplanationState)
    workflow.add_node("explanation_agent", explanation_agent)
    workflow.add_edge(START, "explanation_agent")
    workflow.add_edge("explanation_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_explanation_graph()
    
    print("=== Explanation MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "decision": "Loan application denied",
        "reasoning_trace": [
            "Credit score below threshold",
            "Income insufficient for requested amount",
            "High debt-to-income ratio"
        ],
        "explanation_type": "why",
        "target_audience": "applicant",
        "complexity_level": "simple"
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 157: Explanation - COMPLETE")
    print(f"{'='*70}")

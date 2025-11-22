"""
Adaptation MCP Pattern

This pattern implements dynamic adaptation to user
context, preferences, and changing conditions.

Key Features:
- User profiling
- Context awareness
- Preference learning
- Dynamic adjustment
- Personalization
"""

from typing import TypedDict, Sequence, Annotated, Dict, List
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class AdaptationState(TypedDict):
    """State for adaptation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    user_profile: Dict
    context: Dict
    adaptations: List[str]
    performance_metrics: Dict


llm = ChatOpenAI(model="gpt-4", temperature=0.5)


def adaptation_agent(state: AdaptationState) -> AdaptationState:
    """Manages adaptive behavior"""
    profile = state.get("user_profile", {})
    context = state.get("context", {})
    
    system_prompt = """You are an adaptation expert.

Adaptation Framework:
• Learn preferences
• Detect context
• Adjust behavior
• Track performance
• Continuous improvement

Personalized experience."""
    
    user_prompt = f"""User Profile: {profile}
Context: {context}

Design adaptation system.
Show adaptation strategies."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🎨 Adaptation Agent:
    
    Adaptation System:
    ```python
    class AdaptationSystem:
        def __init__(self):
            self.profiler = UserProfiler()
            self.context_detector = ContextDetector()
            self.behavior_adjuster = BehaviorAdjuster()
            self.performance_tracker = PerformanceTracker()
        
        def adapt(self, user, context):
            '''Adapt to user and context'''
            # Build profile
            profile = self.profiler.build_profile(user)
            
            # Detect context
            current_context = self.context_detector.detect(context)
            
            # Adjust behavior
            adaptations = self.behavior_adjuster.adjust(profile, current_context)
            
            # Track effectiveness
            self.performance_tracker.track(adaptations)
            
            return adaptations
    ```
    
    User Profiling:
    
    Profile Building:
    ```python
    class UserProfile:
        def __init__(self):
            self.demographics = {{}}
            self.preferences = {{}}
            self.behavior_patterns = {{}}
            self.expertise_level = 'beginner'
            self.goals = []
            self.constraints = {{}}
        
        def update_from_interaction(self, interaction):
            '''Learn from each interaction'''
            # Implicit signals
            if interaction.completed_quickly:
                self.expertise_level = increase_level(self.expertise_level)
            
            # Explicit feedback
            if interaction.has_feedback:
                self.update_preferences(interaction.feedback)
            
            # Behavioral patterns
            self.behavior_patterns = detect_patterns(interaction)
    ```
    
    Preference Learning:
    ```python
    def learn_preferences(user_actions):
        '''Infer from behavior'''
        preferences = {{}}
        
        # From choices
        if user.frequently_chooses('dark_mode'):
            preferences['theme'] = 'dark'
        
        # From time patterns
        if user.active_during('evening'):
            preferences['notification_time'] = 'evening'
        
        # From interaction style
        if user.prefers('detailed_explanations'):
            preferences['explanation_level'] = 'detailed'
        
        return preferences
    ```
    
    Context Detection:
    
    Environmental Context:
    ```python
    def detect_environment():
        '''Sense environmental factors'''
        context = {{
            'device': detect_device(),           # Mobile, desktop, tablet
            'location': get_location(),          # Home, work, public
            'time': get_time_context(),          # Morning, afternoon, evening
            'network': check_network(),          # WiFi, cellular, offline
            'battery': get_battery_level()       # High, medium, low
        }}
        
        return context
    ```
    
    Task Context:
    ```python
    def detect_task_context(current_activity):
        '''Understand what user is doing'''
        context = {{
            'task_type': classify_task(current_activity),
            'urgency': assess_urgency(current_activity),
            'complexity': assess_complexity(current_activity),
            'progress': track_progress(current_activity)
        }}
        
        return context
    ```
    
    Social Context:
    ```python
    def detect_social_context():
        '''Understand social situation'''
        context = {{
            'alone_or_with_others': detect_presence(),
            'formality': assess_formality(),
            'privacy_needed': assess_privacy_needs()
        }}
        
        return context
    ```
    
    Adaptive Behaviors:
    
    Interface Adaptation:
    ```python
    def adapt_interface(user_profile, context):
        '''Customize UI/UX'''
        adaptations = []
        
        # Expertise-based
        if user_profile.expertise == 'expert':
            adaptations.append('show_advanced_features')
            adaptations.append('reduce_help_text')
        else:
            adaptations.append('show_tutorials')
            adaptations.append('provide_hints')
        
        # Context-based
        if context.device == 'mobile':
            adaptations.append('enlarge_touch_targets')
            adaptations.append('simplify_layout')
        
        if context.battery == 'low':
            adaptations.append('reduce_animations')
            adaptations.append('optimize_performance')
        
        return apply_adaptations(adaptations)
    ```
    
    Content Adaptation:
    ```python
    def adapt_content(content, user_profile):
        '''Personalize content'''
        adapted = content.copy()
        
        # Language level
        if user_profile.reading_level == 'advanced':
            # Keep complex vocabulary
            pass
        else:
            adapted = simplify_language(adapted)
        
        # Detail level
        if user_profile.prefers == 'concise':
            adapted = summarize(adapted)
        else:
            adapted = add_details(adapted)
        
        # Examples
        if user_profile.domain == 'healthcare':
            adapted = add_healthcare_examples(adapted)
        
        return adapted
    ```
    
    Interaction Adaptation:
    
    Response Style:
    ```python
    def adapt_response_style(user_profile):
        '''Adjust communication style'''
        style = {{}}
        
        # Formality
        if user_profile.prefers_formal:
            style['formality'] = 'formal'
            style['contractions'] = False
        else:
            style['formality'] = 'casual'
            style['contractions'] = True
        
        # Verbosity
        if user_profile.expertise == 'expert':
            style['verbosity'] = 'concise'
        else:
            style['verbosity'] = 'detailed'
        
        # Tone
        style['tone'] = user_profile.preferred_tone or 'friendly'
        
        return style
    ```
    
    Proactivity Level:
    ```python
    def adapt_proactivity(user_profile, context):
        '''Decide when to offer help'''
        if user_profile.values_autonomy:
            # Wait for explicit request
            proactivity = 'low'
        
        elif context.user_struggling:
            # Offer help proactively
            proactivity = 'high'
        
        else:
            # Moderate suggestions
            proactivity = 'medium'
        
        return proactivity
    ```
    
    Learning and Improvement:
    
    Reinforcement Learning:
    ```python
    def reinforce_from_feedback(action, feedback):
        '''Learn from outcomes'''
        if feedback == 'positive':
            # Increase probability of similar actions
            increase_weight(action)
        
        elif feedback == 'negative':
            # Decrease probability
            decrease_weight(action)
        
        # Update model
        retrain_adaptation_model()
    ```
    
    A/B Testing:
    ```python
    def test_adaptations(user_group):
        '''Experiment with different approaches'''
        # Split users
        group_a = user_group[:len(user_group)//2]
        group_b = user_group[len(user_group)//2:]
        
        # Apply different adaptations
        apply_adaptation_a(group_a)
        apply_adaptation_b(group_b)
        
        # Measure performance
        performance_a = measure_performance(group_a)
        performance_b = measure_performance(group_b)
        
        # Choose better approach
        if performance_a > performance_b:
            adopt_adaptation_a()
        else:
            adopt_adaptation_b()
    ```
    
    Contextual Bandits:
    
    Multi-Armed Bandit:
    ```python
    def select_adaptation(context, available_adaptations):
        '''Choose best adaptation for context'''
        # Exploration vs Exploitation
        if random() < epsilon:
            # Explore: try random adaptation
            chosen = random.choice(available_adaptations)
        else:
            # Exploit: use best known adaptation
            scores = [score_adaptation(a, context) 
                     for a in available_adaptations]
            chosen = available_adaptations[argmax(scores)]
        
        return chosen
    ```
    
    Personalization Strategies:
    
    Collaborative Filtering:
    ```python
    def recommend_based_on_similar_users(user, all_users):
        '''Find similar users and their preferences'''
        # Find similar users
        similar_users = find_similar(user, all_users)
        
        # Aggregate their preferences
        recommended_preferences = aggregate_preferences(similar_users)
        
        # Suggest to current user
        return recommended_preferences
    ```
    
    Content-Based Filtering:
    ```python
    def recommend_based_on_history(user):
        '''Recommend similar to what user liked'''
        # Analyze past interactions
        liked_items = user.get_liked_items()
        
        # Find similar items
        features = extract_features(liked_items)
        similar_items = find_items_with_features(features)
        
        return similar_items
    ```
    
    Graceful Degradation:
    
    Handle Uncertainty:
    ```python
    def adapt_with_uncertainty(user_profile):
        '''Handle incomplete information'''
        if user_profile.confidence < threshold:
            # Use safe defaults
            adaptation = apply_defaults()
        else:
            # Use learned preferences
            adaptation = apply_preferences(user_profile)
        
        # Allow easy override
        show_customization_options(adaptation)
        
        return adaptation
    ```
    
    Privacy-Preserving Adaptation:
    
    Federated Learning:
    ```python
    def learn_without_central_data():
        '''Adapt without sharing private data'''
        # Learn locally on device
        local_model = train_on_device(user_data)
        
        # Share only model updates (not data)
        model_update = compute_gradient(local_model)
        
        # Aggregate updates from all users
        global_model = aggregate_updates([model_update, ...])
        
        return global_model
    ```
    
    Differential Privacy:
    ```python
    def learn_with_privacy(user_data):
        '''Add noise to protect privacy'''
        # Add calibrated noise
        noisy_data = add_noise(user_data, epsilon=0.1)
        
        # Learn from noisy data
        model = train(noisy_data)
        
        # Still useful while protecting privacy
        return model
    ```
    
    Performance Tracking:
    
    Metrics:
    ```python
    def track_adaptation_performance():
        '''Measure effectiveness'''
        metrics = {{
            'user_satisfaction': measure_satisfaction(),
            'task_completion': measure_completion_rate(),
            'time_saved': measure_efficiency(),
            'engagement': measure_engagement(),
            'error_reduction': measure_accuracy()
        }}
        
        return metrics
    ```
    
    Continuous Monitoring:
    ```python
    def monitor_and_adjust():
        '''Ongoing improvement'''
        while True:
            # Collect metrics
            metrics = collect_performance_metrics()
            
            # Detect degradation
            if metrics['satisfaction'] < threshold:
                # Adjust adaptations
                tune_adaptation_parameters()
            
            # Periodic retraining
            if time_since_last_update > interval:
                retrain_models()
    ```
    
    Best Practices:
    ✓ Start with defaults, adapt gradually
    ✓ Learn from implicit and explicit signals
    ✓ Respect user privacy
    ✓ Allow user control/override
    ✓ Explain adaptations
    ✓ Track performance
    ✓ Handle uncertainty gracefully
    
    Key Insight:
    Effective adaptation learns from user behavior
    and context to provide personalized experiences
    while respecting privacy and maintaining control.
    """
    
    return {
        "messages": [AIMessage(content=f"🎨 Adaptation Agent:\n{report}\n\n{response.content}")]
    }


def build_adaptation_graph():
    workflow = StateGraph(AdaptationState)
    workflow.add_node("adaptation_agent", adaptation_agent)
    workflow.add_edge(START, "adaptation_agent")
    workflow.add_edge("adaptation_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_adaptation_graph()
    
    print("=== Adaptation MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "user_profile": {
            "expertise": "intermediate",
            "preferences": {"theme": "dark", "verbosity": "concise"},
            "goals": ["learn_efficiently", "save_time"]
        },
        "context": {
            "device": "mobile",
            "time": "evening",
            "task": "learning"
        },
        "adaptations": [],
        "performance_metrics": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 160: Adaptation - COMPLETE")
    print(f"{'='*70}")
    print(f"\n🎉 Interaction Patterns (151-160) - ALL COMPLETE! 🎉")
    print(f"Progress: 160/400 patterns (40%)")

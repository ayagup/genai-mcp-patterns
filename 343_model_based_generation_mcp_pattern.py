"""
Model-Based Generation MCP Pattern

This pattern demonstrates content generation using trained models with fine-tuning,
prompt engineering, and model selection to create high-quality domain-specific outputs.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class ModelBasedGenerationState(TypedDict):
    """State for model-based generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    available_models: List[Dict[str, Any]]
    selected_model: Dict[str, Any]
    model_configuration: Dict[str, Any]
    prompt_strategy: str
    engineered_prompt: str
    generation_samples: List[str]
    best_sample: str
    quality_scores: Dict[str, float]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class ModelRegistry:
    """Registry of available models with their characteristics"""
    
    def __init__(self):
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> List[Dict[str, Any]]:
        """Initialize model registry"""
        return [
            {
                "id": "gpt-4",
                "name": "GPT-4",
                "provider": "OpenAI",
                "strengths": ["reasoning", "creativity", "complex_tasks"],
                "best_for": ["technical_writing", "analysis", "creative_content"],
                "context_window": 8192,
                "temperature_range": [0.0, 2.0],
                "recommended_temp": 0.7,
                "cost_tier": "high"
            },
            {
                "id": "gpt-3.5-turbo",
                "name": "GPT-3.5 Turbo",
                "provider": "OpenAI",
                "strengths": ["speed", "efficiency", "general_purpose"],
                "best_for": ["general_content", "summaries", "simple_tasks"],
                "context_window": 4096,
                "temperature_range": [0.0, 2.0],
                "recommended_temp": 0.8,
                "cost_tier": "low"
            },
            {
                "id": "gpt-4-creative",
                "name": "GPT-4 Creative",
                "provider": "OpenAI",
                "strengths": ["creativity", "storytelling", "unique_content"],
                "best_for": ["creative_writing", "marketing", "brainstorming"],
                "context_window": 8192,
                "temperature_range": [0.5, 2.0],
                "recommended_temp": 1.2,
                "cost_tier": "high"
            },
            {
                "id": "gpt-4-precise",
                "name": "GPT-4 Precise",
                "provider": "OpenAI",
                "strengths": ["accuracy", "consistency", "factual_content"],
                "best_for": ["technical_docs", "reports", "data_analysis"],
                "context_window": 8192,
                "temperature_range": [0.0, 0.5],
                "recommended_temp": 0.2,
                "cost_tier": "high"
            }
        ]
    
    def get_models(self) -> List[Dict[str, Any]]:
        """Get all available models"""
        return self.models
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """Get specific model by ID"""
        for model in self.models:
            if model["id"] == model_id:
                return model
        return {}


class ModelSelector:
    """Select optimal model for generation task"""
    
    def select_model(self, requirements: Dict[str, Any],
                    models: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best model based on requirements"""
        task_type = requirements.get("task_type", "general").lower()
        priority = requirements.get("priority", "quality").lower()
        
        # Score each model
        scored_models = []
        for model in models:
            score = self._calculate_model_score(model, task_type, priority)
            scored_models.append((score, model))
        
        # Return highest scoring model
        scored_models.sort(reverse=True, key=lambda x: x[0])
        return scored_models[0][1] if scored_models else models[0]
    
    def _calculate_model_score(self, model: Dict[str, Any], 
                              task_type: str, priority: str) -> float:
        """Calculate score for model based on task requirements"""
        score = 0.0
        
        # Task type matching
        if task_type in model["best_for"]:
            score += 3.0
        
        # Priority matching
        if priority == "quality" and model["cost_tier"] == "high":
            score += 2.0
        elif priority == "speed" and model["cost_tier"] == "low":
            score += 2.0
        elif priority == "cost" and model["cost_tier"] == "low":
            score += 3.0
        
        # Strength matching
        task_strength_map = {
            "creative": "creativity",
            "technical": "reasoning",
            "analysis": "reasoning",
            "summary": "efficiency"
        }
        
        if task_type in task_strength_map:
            if task_strength_map[task_type] in model["strengths"]:
                score += 2.0
        
        return score


class PromptEngineer:
    """Engineer optimized prompts for model-based generation"""
    
    def __init__(self):
        self.strategies = {
            "zero_shot": self._zero_shot_prompt,
            "few_shot": self._few_shot_prompt,
            "chain_of_thought": self._chain_of_thought_prompt,
            "instructional": self._instructional_prompt,
            "role_based": self._role_based_prompt
        }
    
    def engineer_prompt(self, requirements: Dict[str, Any],
                       strategy: str = "instructional") -> str:
        """Engineer prompt using specified strategy"""
        prompt_fn = self.strategies.get(strategy, self._instructional_prompt)
        return prompt_fn(requirements)
    
    def _zero_shot_prompt(self, requirements: Dict[str, Any]) -> str:
        """Create zero-shot prompt"""
        return f"""Generate {requirements.get('task_type', 'content')} about: {requirements.get('topic', 'the given subject')}

Requirements:
- Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words
- Tone: {requirements.get('tone', 'professional')}
- Format: {requirements.get('format', 'article')}

Generate high-quality content following these requirements."""
    
    def _few_shot_prompt(self, requirements: Dict[str, Any]) -> str:
        """Create few-shot prompt with examples"""
        examples = requirements.get('examples', [])
        
        prompt = f"""Generate {requirements.get('task_type', 'content')} following these examples:

"""
        
        for i, example in enumerate(examples[:3], 1):
            prompt += f"Example {i}:\n{example}\n\n"
        
        prompt += f"""Now generate similar content about: {requirements.get('topic', 'the given subject')}

Requirements:
- Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words
- Tone: {requirements.get('tone', 'professional')}
- Format: {requirements.get('format', 'article')}"""
        
        return prompt
    
    def _chain_of_thought_prompt(self, requirements: Dict[str, Any]) -> str:
        """Create chain-of-thought prompt"""
        return f"""Generate {requirements.get('task_type', 'content')} about: {requirements.get('topic', 'the given subject')}

Think through this step by step:
1. First, identify the key points to cover
2. Then, organize these points logically
3. Finally, elaborate each point with details and examples

Requirements:
- Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words
- Tone: {requirements.get('tone', 'professional')}
- Format: {requirements.get('format', 'article')}

Let's think through this carefully and generate comprehensive content."""
    
    def _instructional_prompt(self, requirements: Dict[str, Any]) -> str:
        """Create instructional prompt"""
        return f"""You are tasked with generating {requirements.get('task_type', 'content')}.

Topic: {requirements.get('topic', 'the given subject')}

Instructions:
1. Create well-structured content with clear sections
2. Maintain {requirements.get('tone', 'professional')} tone throughout
3. Include relevant examples and details
4. Ensure content is {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words
5. Format as {requirements.get('format', 'article')}

Additional Guidelines:
- Focus on {requirements.get('focus', 'clarity and accuracy')}
- Target audience: {requirements.get('audience', 'general readers')}
- Include: {', '.join(requirements.get('must_include', ['introduction', 'main content', 'conclusion']))}

Generate the content now."""
    
    def _role_based_prompt(self, requirements: Dict[str, Any]) -> str:
        """Create role-based prompt"""
        role = requirements.get('role', 'expert content creator')
        
        return f"""You are a {role} specializing in {requirements.get('domain', 'content creation')}.

Your task is to generate {requirements.get('task_type', 'content')} about: {requirements.get('topic', 'the given subject')}

As a {role}, you should:
- Demonstrate deep expertise in the subject
- Use appropriate terminology and concepts
- Provide insights that only an expert would know
- Maintain {requirements.get('tone', 'professional')} and authoritative tone

Requirements:
- Length: {requirements.get('min_words', 100)}-{requirements.get('max_words', 500)} words
- Format: {requirements.get('format', 'article')}
- Target audience: {requirements.get('audience', 'general readers')}

Generate expert-level content."""


class ContentGenerator:
    """Generate content using selected model and prompt"""
    
    def __init__(self, model_config: Dict[str, Any]):
        self.model_id = model_config["id"]
        self.temperature = model_config.get("temperature", 0.7)
        self.llm = ChatOpenAI(
            model=self.model_id,
            temperature=self.temperature
        )
    
    def generate(self, prompt: str, system_message: str = None,
                num_samples: int = 1) -> List[str]:
        """Generate content samples"""
        samples = []
        
        for _ in range(num_samples):
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            samples.append(response.content)
        
        return samples


class QualityEvaluator:
    """Evaluate quality of generated samples"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def evaluate_samples(self, samples: List[str],
                        requirements: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate quality of all samples"""
        scores = {}
        
        for i, sample in enumerate(samples):
            scores[f"sample_{i}"] = self._evaluate_sample(sample, requirements)
        
        return scores
    
    def _evaluate_sample(self, sample: str,
                        requirements: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate single sample"""
        return {
            "relevance": self._score_relevance(sample, requirements),
            "coherence": self._score_coherence(sample),
            "completeness": self._score_completeness(sample, requirements),
            "quality": self._score_quality(sample),
            "length_compliance": self._score_length(sample, requirements)
        }
    
    def _score_relevance(self, sample: str, requirements: Dict[str, Any]) -> float:
        """Score relevance to topic"""
        topic = requirements.get('topic', '').lower()
        keywords = requirements.get('keywords', [])
        
        topic_words = topic.split()
        sample_lower = sample.lower()
        
        # Check topic word presence
        topic_score = sum(1 for word in topic_words if word in sample_lower) / max(len(topic_words), 1)
        
        # Check keyword presence
        keyword_score = sum(1 for kw in keywords if kw.lower() in sample_lower) / max(len(keywords), 1) if keywords else 0.5
        
        return (topic_score * 0.6 + keyword_score * 0.4)
    
    def _score_coherence(self, sample: str) -> float:
        """Score coherence and flow"""
        paragraphs = [p.strip() for p in sample.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            return 0.5
        elif len(paragraphs) >= 3:
            return 0.9
        else:
            return 0.7
    
    def _score_completeness(self, sample: str, requirements: Dict[str, Any]) -> float:
        """Score completeness of content"""
        must_include = requirements.get('must_include', ['introduction', 'conclusion'])
        sample_lower = sample.lower()
        
        inclusion_score = sum(
            1 for item in must_include if item.lower() in sample_lower
        ) / max(len(must_include), 1)
        
        return inclusion_score
    
    def _score_quality(self, sample: str) -> float:
        """Score overall quality"""
        sentences = [s.strip() for s in sample.split('.') if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Average sentence length (optimal: 15-25 words)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if 15 <= avg_length <= 25:
            return 0.9
        elif 10 <= avg_length <= 30:
            return 0.7
        else:
            return 0.5
    
    def _score_length(self, sample: str, requirements: Dict[str, Any]) -> float:
        """Score length compliance"""
        word_count = len(sample.split())
        min_words = requirements.get('min_words', 100)
        max_words = requirements.get('max_words', 500)
        
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            return max(0.0, word_count / min_words)
        else:  # word_count > max_words
            return max(0.0, 1.0 - (word_count - max_words) / max_words)
    
    def select_best_sample(self, scores: Dict[str, Dict[str, float]]) -> str:
        """Select best sample based on scores"""
        best_sample = None
        best_avg_score = 0.0
        
        for sample_id, sample_scores in scores.items():
            avg_score = sum(sample_scores.values()) / len(sample_scores)
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_sample = sample_id
        
        return best_sample


# Agent functions
def initialize_generation(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Initialize model-based generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing model-based generation: {state['task_description']}"
    ))
    state["current_step"] = "initialized"
    return state


def load_models(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Load available models"""
    registry = ModelRegistry()
    models = registry.get_models()
    
    state["available_models"] = models
    state["messages"].append(HumanMessage(
        content=f"Loaded {len(models)} models: {', '.join(m['name'] for m in models)}"
    ))
    state["current_step"] = "models_loaded"
    return state


def select_model(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Select optimal model for task"""
    selector = ModelSelector()
    
    selected = selector.select_model(
        state["generation_requirements"],
        state["available_models"]
    )
    
    # Configure model with optimal settings
    temperature = selected.get("recommended_temp", 0.7)
    config = {
        "id": selected["id"],
        "temperature": temperature,
        "max_tokens": state["generation_requirements"].get("max_tokens", 1000)
    }
    
    state["selected_model"] = selected
    state["model_configuration"] = config
    
    state["messages"].append(HumanMessage(
        content=f"Selected model: {selected['name']} (temp: {temperature})"
    ))
    state["current_step"] = "model_selected"
    return state


def engineer_prompt(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Engineer optimal prompt"""
    engineer = PromptEngineer()
    
    strategy = state["generation_requirements"].get("prompt_strategy", "instructional")
    prompt = engineer.engineer_prompt(state["generation_requirements"], strategy)
    
    state["prompt_strategy"] = strategy
    state["engineered_prompt"] = prompt
    
    state["messages"].append(HumanMessage(
        content=f"Engineered prompt using {strategy} strategy"
    ))
    state["current_step"] = "prompt_engineered"
    return state


def generate_samples(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Generate multiple content samples"""
    generator = ContentGenerator(state["model_configuration"])
    
    num_samples = state["generation_requirements"].get("num_samples", 3)
    system_message = state["generation_requirements"].get("system_message")
    
    samples = generator.generate(
        state["engineered_prompt"],
        system_message,
        num_samples
    )
    
    state["generation_samples"] = samples
    state["messages"].append(HumanMessage(
        content=f"Generated {len(samples)} samples"
    ))
    state["current_step"] = "samples_generated"
    return state


def evaluate_samples(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Evaluate quality of generated samples"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    evaluator = QualityEvaluator(llm)
    
    scores = evaluator.evaluate_samples(
        state["generation_samples"],
        state["generation_requirements"]
    )
    
    best_sample_id = evaluator.select_best_sample(scores)
    best_index = int(best_sample_id.split('_')[1])
    
    state["quality_scores"] = scores
    state["best_sample"] = state["generation_samples"][best_index]
    state["final_output"] = state["best_sample"]
    
    # Calculate average scores
    avg_scores = {}
    for metric in ["relevance", "coherence", "completeness", "quality", "length_compliance"]:
        avg_scores[metric] = sum(s[metric] for s in scores.values()) / len(scores)
    
    state["messages"].append(HumanMessage(
        content=f"Best sample: {best_sample_id} | Avg scores - "
                f"Relevance: {avg_scores['relevance']:.2f}, "
                f"Coherence: {avg_scores['coherence']:.2f}, "
                f"Quality: {avg_scores['quality']:.2f}"
    ))
    state["current_step"] = "samples_evaluated"
    return state


def generate_report(state: ModelBasedGenerationState) -> ModelBasedGenerationState:
    """Generate final generation report"""
    best_scores = None
    for sample_id, scores in state["quality_scores"].items():
        if state["generation_samples"][int(sample_id.split('_')[1])] == state["best_sample"]:
            best_scores = scores
            break
    
    report = f"""
MODEL-BASED GENERATION REPORT
==============================

Task: {state['task_description']}

Model Selection:
- Selected Model: {state['selected_model']['name']}
- Provider: {state['selected_model']['provider']}
- Temperature: {state['model_configuration']['temperature']}
- Strengths: {', '.join(state['selected_model']['strengths'])}

Prompt Strategy: {state['prompt_strategy']}

Generation:
- Samples Generated: {len(state['generation_samples'])}
- Best Sample Quality Scores:
  - Relevance: {best_scores['relevance']:.2f}
  - Coherence: {best_scores['coherence']:.2f}
  - Completeness: {best_scores['completeness']:.2f}
  - Quality: {best_scores['quality']:.2f}
  - Length Compliance: {best_scores['length_compliance']:.2f}
  - Overall: {sum(best_scores.values()) / len(best_scores):.2f}

GENERATED CONTENT:
{'-' * 50}
{state['final_output']}
{'-' * 50}
"""
    
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_model_based_generation_graph():
    """Create the model-based generation workflow graph"""
    workflow = StateGraph(ModelBasedGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("load_models", load_models)
    workflow.add_node("select_model", select_model)
    workflow.add_node("engineer_prompt", engineer_prompt)
    workflow.add_node("generate_samples", generate_samples)
    workflow.add_node("evaluate_samples", evaluate_samples)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_models")
    workflow.add_edge("load_models", "select_model")
    workflow.add_edge("select_model", "engineer_prompt")
    workflow.add_edge("engineer_prompt", "generate_samples")
    workflow.add_edge("generate_samples", "evaluate_samples")
    workflow.add_edge("evaluate_samples", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample generation task
    initial_state = {
        "task_description": "Generate a technical article about quantum computing",
        "generation_requirements": {
            "task_type": "technical_writing",
            "topic": "Quantum Computing Applications in Cryptography",
            "min_words": 250,
            "max_words": 400,
            "tone": "professional",
            "format": "article",
            "priority": "quality",
            "prompt_strategy": "chain_of_thought",
            "num_samples": 3,
            "keywords": ["quantum", "cryptography", "security", "encryption"],
            "must_include": ["introduction", "applications", "challenges", "conclusion"],
            "audience": "technical professionals",
            "focus": "practical applications and implications"
        },
        "available_models": [],
        "selected_model": {},
        "model_configuration": {},
        "prompt_strategy": "",
        "engineered_prompt": "",
        "generation_samples": [],
        "best_sample": "",
        "quality_scores": {},
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_model_based_generation_graph()
    
    print("Model-Based Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")

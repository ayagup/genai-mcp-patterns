"""
Few-Shot Prompt MCP Pattern

This pattern implements few-shot learning through example-based prompting,
where the model learns from a small number of demonstrations.

Key Features:
- Example-based learning
- Dynamic example selection
- Multi-task adaptation
- Quality example curation
- Context-aware demonstrations
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class FewShotState(TypedDict):
    """State for few-shot prompt pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task_type: str  # "classification", "generation", "extraction", "qa"
    examples: List[Dict]  # [{input, output, explanation}]
    num_shots: int  # 0, 1, 3, 5 (zero-shot, one-shot, few-shot)
    query: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)


# Example Builder
def example_builder(state: FewShotState) -> FewShotState:
    """Builds few-shot examples for the task"""
    task_type = state.get("task_type", "classification")
    num_shots = state.get("num_shots", 3)
    
    # Predefined example sets for different tasks
    example_library = {
        "classification": [
            {
                "input": "The movie was absolutely fantastic! Great acting and plot.",
                "output": "Positive",
                "explanation": "Enthusiastic language and praise indicate positive sentiment"
            },
            {
                "input": "Terrible experience. Would not recommend to anyone.",
                "output": "Negative",
                "explanation": "Strong negative words indicate negative sentiment"
            },
            {
                "input": "The product works as expected. Nothing special.",
                "output": "Neutral",
                "explanation": "Lack of strong emotion indicates neutral sentiment"
            },
            {
                "input": "Best purchase I've ever made! Exceeded all expectations.",
                "output": "Positive",
                "explanation": "Superlatives and excitement indicate strong positive sentiment"
            },
            {
                "input": "Completely disappointed. Waste of money.",
                "output": "Negative",
                "explanation": "Disappointment and regret indicate negative sentiment"
            }
        ],
        "extraction": [
            {
                "input": "John Smith works at Acme Corp in New York and can be reached at john@acme.com",
                "output": '{"name": "John Smith", "company": "Acme Corp", "location": "New York", "email": "john@acme.com"}',
                "explanation": "Extracted all structured entities from unstructured text"
            },
            {
                "input": "Meeting scheduled with Sarah Johnson from TechStart on Monday at 2pm",
                "output": '{"name": "Sarah Johnson", "company": "TechStart", "date": "Monday", "time": "2pm"}',
                "explanation": "Identified person, organization, and temporal information"
            },
            {
                "input": "Contact Dr. Michael Chen at michael.chen@hospital.org for appointment",
                "output": '{"name": "Dr. Michael Chen", "title": "Dr.", "email": "michael.chen@hospital.org", "purpose": "appointment"}',
                "explanation": "Extracted professional title, name, and contact information"
            }
        ],
        "generation": [
            {
                "input": "Topic: Climate Change",
                "output": "Climate change represents one of the most pressing challenges of our time, affecting ecosystems, economies, and communities worldwide.",
                "explanation": "Generated engaging introduction for the topic"
            },
            {
                "input": "Topic: Artificial Intelligence",
                "output": "Artificial intelligence is revolutionizing industries by enabling machines to learn, adapt, and make decisions with increasing sophistication.",
                "explanation": "Created compelling opening statement about AI"
            },
            {
                "input": "Topic: Space Exploration",
                "output": "Humanity's quest to explore space continues to push the boundaries of science, technology, and our understanding of the universe.",
                "explanation": "Crafted inspirational introduction to space exploration"
            }
        ],
        "qa": [
            {
                "input": "Question: What is the capital of France?",
                "output": "The capital of France is Paris.",
                "explanation": "Direct factual answer to geography question"
            },
            {
                "input": "Question: Why is the sky blue?",
                "output": "The sky appears blue because molecules in Earth's atmosphere scatter blue light from the sun more than other colors.",
                "explanation": "Scientific explanation with cause and effect"
            },
            {
                "input": "Question: How do you make coffee?",
                "output": "To make coffee: 1) Heat water to 195-205°F, 2) Add ground coffee to filter, 3) Pour water over grounds, 4) Let it brew for 4-5 minutes.",
                "explanation": "Step-by-step procedural answer"
            }
        ]
    }
    
    # Select examples for the task
    available_examples = example_library.get(task_type, example_library["classification"])
    selected_examples = available_examples[:num_shots]
    
    report = f"""
    📚 Example Builder:
    
    Configuration:
    • Task Type: {task_type.capitalize()}
    • Number of Shots: {num_shots}
    • Examples Selected: {len(selected_examples)}
    
    Few-Shot Learning Concepts:
    
    Shot Types:
    
    Zero-Shot (0 examples):
    • No demonstrations
    • Rely on pre-training
    • Task described in prompt
    • General capabilities
    
    One-Shot (1 example):
    • Single demonstration
    • Shows desired format
    • Quick adaptation
    • Minimal context
    
    Few-Shot (3-5 examples):
    • Multiple demonstrations
    • Pattern recognition
    • Better consistency
    • Task clarification
    
    Many-Shot (10+ examples):
    • Extensive examples
    • Fine-grained patterns
    • Higher accuracy
    • Context window limits
    
    Example Quality Criteria:
    
    Diversity:
    • Cover edge cases
    • Various input types
    • Different scenarios
    • Representative samples
    
    Clarity:
    • Clear input-output pairs
    • Unambiguous labels
    • Consistent format
    • Easy to understand
    
    Relevance:
    • Similar to target task
    • Same domain
    • Matching complexity
    • Appropriate difficulty
    
    Balance:
    • Equal class distribution
    • Varied complexity levels
    • Different lengths
    • Comprehensive coverage
    
    Few-Shot Prompting Strategies:
    
    Direct Examples:
    ```
    Example 1:
    Input: [text]
    Output: [label]
    
    Example 2:
    Input: [text]
    Output: [label]
    
    Now classify:
    Input: [new text]
    Output:
    ```
    
    With Explanations:
    ```
    Example 1:
    Input: [text]
    Output: [label]
    Reasoning: [why]
    
    Example 2:
    Input: [text]
    Output: [label]
    Reasoning: [why]
    
    Now classify with reasoning:
    Input: [new text]
    ```
    
    Structured Format:
    ```json
    {{
      "examples": [
        {{"input": "...", "output": "...", "explanation": "..."}},
        {{"input": "...", "output": "...", "explanation": "..."}}
      ],
      "task": "...",
      "query": "..."
    }}
    ```
    
    Example Selection Strategies:
    
    Random Selection:
    • Simple approach
    • Quick setup
    • May miss patterns
    • Baseline performance
    
    Similarity-Based:
    • Select similar examples
    • Embedding similarity
    • Relevant demonstrations
    • Better performance
    
    Diverse Selection:
    • Maximize coverage
    • Different categories
    • Comprehensive view
    • Robust learning
    
    Active Learning:
    • Select informative examples
    • Uncertainty sampling
    • Query by committee
    • Iterative improvement
    
    Selected Examples:
    {chr(10).join([f"  {i+1}. Input: {ex['input'][:60]}..." + f"{chr(10)}     Output: {ex['output']}" for i, ex in enumerate(selected_examples)])}
    """
    
    return {
        "messages": [AIMessage(content=f"📚 Example Builder:\n{report}")],
        "examples": selected_examples
    }


# Few-Shot Generator
def few_shot_generator(state: FewShotState) -> FewShotState:
    """Generates response using few-shot examples"""
    examples = state.get("examples", [])
    query = state.get("query", "Analyze: This product is okay, nothing special.")
    task_type = state.get("task_type", "classification")
    
    # Build few-shot prompt
    system_prompt = f"You are performing {task_type}. Learn from the examples below and apply the same pattern to the new input."
    
    # Format examples
    example_text = []
    for i, ex in enumerate(examples, 1):
        example_text.append(f"Example {i}:")
        example_text.append(f"Input: {ex['input']}")
        example_text.append(f"Output: {ex['output']}")
        if ex.get("explanation"):
            example_text.append(f"Explanation: {ex['explanation']}")
        example_text.append("")
    
    # Add the query
    example_text.append("Now apply the same pattern:")
    example_text.append(f"Input: {query}")
    example_text.append("Output:")
    
    prompt = "\n".join(example_text)
    
    # Generate response
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    summary = f"""
    📊 FEW-SHOT PATTERN COMPLETE
    
    Pattern Summary:
    • Task: {task_type.capitalize()}
    • Shots Used: {len(examples)}
    • Query: {query[:100]}...
    
    Few-Shot Pattern Process:
    1. Example Builder → Select relevant demonstrations
    2. Few-Shot Generator → Apply pattern to new input
    
    Advanced Few-Shot Techniques:
    
    Chain-of-Thought Few-Shot:
    • Include reasoning steps
    • Show thought process
    • Explicit logic
    • Better accuracy
    
    Multi-Task Few-Shot:
    • Examples from multiple tasks
    • Transfer learning
    • Generalization
    • Versatility
    
    Meta-Learning Few-Shot:
    • Learn to learn
    • Quick adaptation
    • Few examples needed
    • Strong generalization
    
    Contrastive Examples:
    • Show what to do
    • Show what not to do
    • Positive and negative
    • Clear boundaries
    
    Few-Shot Best Practices:
    
    Example Ordering:
    • Easy to hard
    • Random shuffle
    • By similarity
    • Chronological
    
    Example Quality:
    • Verify correctness
    • Remove ambiguity
    • Check consistency
    • Test effectiveness
    
    Context Management:
    • Balance detail vs. brevity
    • Stay within token limits
    • Prioritize relevant examples
    • Compress when needed
    
    Performance Optimization:
    • A/B test examples
    • Monitor accuracy
    • Refine selection
    • Update regularly
    
    Research Findings:
    
    GPT-3 Paper (Brown et al.):
    • Few-shot > zero-shot
    • More shots → better (up to limit)
    • Example quality matters
    • Task-dependent effectiveness
    
    In-Context Learning:
    • Models learn from context
    • No parameter updates
    • Rapid adaptation
    • Emergent capability
    
    Prompt Sensitivity:
    • Example order affects results
    • Format consistency important
    • Label distribution matters
    • Random seed variation
    
    Generated Result:
    {'-' * 60}
    Input: {query}
    Output: {response.content}
    {'-' * 60}
    
    Key Insight:
    Few-shot learning enables rapid task adaptation through
    demonstration-based learning, achieving strong performance
    with minimal examples and no fine-tuning.
    """
    
    return {
        "messages": [AIMessage(content=f"✅ Few-Shot Generator:\n{summary}")]
    }


# Build the graph
def build_few_shot_graph():
    """Build the few-shot prompt pattern graph"""
    workflow = StateGraph(FewShotState)
    
    workflow.add_node("example_builder", example_builder)
    workflow.add_node("few_shot_generator", few_shot_generator)
    
    workflow.add_edge(START, "example_builder")
    workflow.add_edge("example_builder", "few_shot_generator")
    workflow.add_edge("few_shot_generator", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    graph = build_few_shot_graph()
    
    print("=== Few-Shot Prompt MCP Pattern ===\n")
    
    # Test Case: Sentiment Classification
    print("\n" + "="*70)
    print("TEST CASE: Few-Shot Sentiment Classification")
    print("="*70)
    
    state = {
        "messages": [],
        "task_type": "classification",
        "examples": [],
        "num_shots": 3,
        "query": "The service was decent but the food could be better."
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 122: Few-Shot Prompt - COMPLETE")
    print(f"{'='*70}")

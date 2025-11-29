"""
Constrained Generation MCP Pattern

This pattern demonstrates content generation with hard constraints including length limits,
vocabulary restrictions, format requirements, and structural constraints.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Intermediate
"""

from typing import TypedDict, List, Dict, Annotated, Any, Set
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
import re


# State definition
class ConstrainedGenerationState(TypedDict):
    """State for constrained generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    constraint_validation: Dict[str, Any]
    generated_content: str
    violations: List[Dict[str, Any]]
    repair_attempts: int
    constraint_satisfaction_score: float
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class ConstraintDefinition:
    """Define and manage generation constraints"""
    
    def __init__(self):
        self.constraint_types = {
            "length": self._length_constraints,
            "vocabulary": self._vocabulary_constraints,
            "format": self._format_constraints,
            "structure": self._structure_constraints,
            "content": self._content_constraints
        }
    
    def _length_constraints(self) -> List[Dict[str, Any]]:
        """Define length-based constraints"""
        return [
            {
                "id": "exact_word_count",
                "type": "length",
                "description": "Exactly N words",
                "validator": lambda content, params: len(content.split()) == params.get("exact_words"),
                "strict": True
            },
            {
                "id": "word_range",
                "type": "length",
                "description": "Word count within range",
                "validator": lambda content, params: params.get("min_words", 0) <= len(content.split()) <= params.get("max_words", float('inf')),
                "strict": True
            },
            {
                "id": "sentence_count",
                "type": "length",
                "description": "Specific number of sentences",
                "validator": lambda content, params: len([s for s in re.split(r'[.!?]', content) if s.strip()]) == params.get("num_sentences"),
                "strict": False
            },
            {
                "id": "paragraph_count",
                "type": "length",
                "description": "Specific number of paragraphs",
                "validator": lambda content, params: len([p for p in content.split('\n\n') if p.strip()]) == params.get("num_paragraphs"),
                "strict": False
            }
        ]
    
    def _vocabulary_constraints(self) -> List[Dict[str, Any]]:
        """Define vocabulary-based constraints"""
        return [
            {
                "id": "allowed_words",
                "type": "vocabulary",
                "description": "Only use words from allowed list",
                "validator": lambda content, params: all(
                    word.lower() in params.get("allowed_words", set())
                    for word in re.findall(r'\b\w+\b', content)
                ),
                "strict": True
            },
            {
                "id": "forbidden_words",
                "type": "vocabulary",
                "description": "Must not use forbidden words",
                "validator": lambda content, params: not any(
                    word.lower() in content.lower()
                    for word in params.get("forbidden_words", [])
                ),
                "strict": True
            },
            {
                "id": "required_keywords",
                "type": "vocabulary",
                "description": "Must include required keywords",
                "validator": lambda content, params: all(
                    kw.lower() in content.lower()
                    for kw in params.get("required_keywords", [])
                ),
                "strict": True
            },
            {
                "id": "max_word_length",
                "type": "vocabulary",
                "description": "No word exceeds max length",
                "validator": lambda content, params: all(
                    len(word) <= params.get("max_word_len", 20)
                    for word in re.findall(r'\b\w+\b', content)
                ),
                "strict": False
            }
        ]
    
    def _format_constraints(self) -> List[Dict[str, Any]]:
        """Define format-based constraints"""
        return [
            {
                "id": "markdown_format",
                "type": "format",
                "description": "Must be valid Markdown",
                "validator": lambda content, params: bool(re.search(r'^#|\*\*|__|\[.+\]\(.+\)', content, re.M)),
                "strict": False
            },
            {
                "id": "bullet_points",
                "type": "format",
                "description": "Must use bullet points",
                "validator": lambda content, params: len(re.findall(r'^\s*[-*•]', content, re.M)) >= params.get("min_bullets", 3),
                "strict": False
            },
            {
                "id": "numbered_list",
                "type": "format",
                "description": "Must use numbered list",
                "validator": lambda content, params: len(re.findall(r'^\s*\d+\.', content, re.M)) >= params.get("min_items", 3),
                "strict": False
            },
            {
                "id": "no_special_chars",
                "type": "format",
                "description": "No special characters except punctuation",
                "validator": lambda content, params: not bool(re.search(r'[^a-zA-Z0-9\s.,!?;:\-\'\"()]', content)),
                "strict": False
            }
        ]
    
    def _structure_constraints(self) -> List[Dict[str, Any]]:
        """Define structure-based constraints"""
        return [
            {
                "id": "has_title",
                "type": "structure",
                "description": "Must have a title",
                "validator": lambda content, params: bool(re.match(r'^#\s+.+', content, re.M)),
                "strict": False
            },
            {
                "id": "has_sections",
                "type": "structure",
                "description": "Must have specific sections",
                "validator": lambda content, params: all(
                    section.lower() in content.lower()
                    for section in params.get("required_sections", [])
                ),
                "strict": True
            },
            {
                "id": "alphabetical_order",
                "type": "structure",
                "description": "Content in alphabetical order",
                "validator": lambda content, params: self._check_alphabetical(content),
                "strict": False
            },
            {
                "id": "hierarchical_structure",
                "type": "structure",
                "description": "Proper heading hierarchy",
                "validator": lambda content, params: self._check_heading_hierarchy(content),
                "strict": False
            }
        ]
    
    def _content_constraints(self) -> List[Dict[str, Any]]:
        """Define content-based constraints"""
        return [
            {
                "id": "factual_only",
                "type": "content",
                "description": "No opinions or subjective statements",
                "validator": lambda content, params: not bool(re.search(r'\b(I think|I believe|in my opinion|personally)\b', content, re.I)),
                "strict": False
            },
            {
                "id": "no_questions",
                "type": "content",
                "description": "No question marks allowed",
                "validator": lambda content, params: '?' not in content,
                "strict": True
            },
            {
                "id": "positive_sentiment",
                "type": "content",
                "description": "Must maintain positive sentiment",
                "validator": lambda content, params: self._check_positive_sentiment(content),
                "strict": False
            },
            {
                "id": "technical_terms",
                "type": "content",
                "description": "Must include technical terminology",
                "validator": lambda content, params: sum(
                    1 for term in params.get("technical_terms", [])
                    if term.lower() in content.lower()
                ) >= params.get("min_technical_terms", 3),
                "strict": False
            }
        ]
    
    def _check_alphabetical(self, content: str) -> bool:
        """Check if lines are in alphabetical order"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if len(lines) < 2:
            return True
        return lines == sorted(lines)
    
    def _check_heading_hierarchy(self, content: str) -> bool:
        """Check proper heading hierarchy (# before ##, etc.)"""
        headings = re.findall(r'^(#{1,6})\s', content, re.M)
        if not headings:
            return True
        
        levels = [len(h) for h in headings]
        # Check no skips in hierarchy
        for i in range(1, len(levels)):
            if levels[i] > levels[i-1] + 1:
                return False
        return True
    
    def _check_positive_sentiment(self, content: str) -> bool:
        """Simple positive sentiment check"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'beneficial', 'successful']
        negative_words = ['bad', 'poor', 'terrible', 'negative', 'harmful', 'failed']
        
        content_lower = content.lower()
        pos_count = sum(1 for word in positive_words if word in content_lower)
        neg_count = sum(1 for word in negative_words if word in content_lower)
        
        return pos_count >= neg_count
    
    def get_constraints(self, constraint_types: List[str]) -> List[Dict[str, Any]]:
        """Get constraints of specified types"""
        constraints = []
        for ctype in constraint_types:
            if ctype in self.constraint_types:
                constraints.extend(self.constraint_types[ctype]())
        return constraints


class ConstraintValidator:
    """Validate content against constraints"""
    
    def validate(self, content: str, constraints: List[Dict[str, Any]],
                params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content against all constraints"""
        results = []
        violations = []
        
        for constraint in constraints:
            try:
                is_valid = constraint["validator"](content, params)
                
                result = {
                    "constraint_id": constraint["id"],
                    "type": constraint["type"],
                    "description": constraint["description"],
                    "valid": is_valid,
                    "strict": constraint["strict"]
                }
                
                results.append(result)
                
                if not is_valid:
                    violations.append({
                        "constraint_id": constraint["id"],
                        "description": constraint["description"],
                        "strict": constraint["strict"],
                        "type": constraint["type"]
                    })
            
            except Exception as e:
                results.append({
                    "constraint_id": constraint["id"],
                    "valid": False,
                    "error": str(e)
                })
        
        # Calculate satisfaction score
        strict_constraints = [c for c in constraints if c["strict"]]
        strict_results = [r for r in results if r.get("strict", False)]
        
        strict_satisfaction = (
            sum(1 for r in strict_results if r["valid"]) / len(strict_results)
            if strict_results else 1.0
        )
        
        total_satisfaction = (
            sum(1 for r in results if r["valid"]) / len(results)
            if results else 0.0
        )
        
        return {
            "results": results,
            "violations": violations,
            "strict_violations": [v for v in violations if v["strict"]],
            "strict_satisfaction": strict_satisfaction,
            "total_satisfaction": total_satisfaction,
            "is_valid": len([v for v in violations if v["strict"]]) == 0
        }


class ConstrainedContentGenerator:
    """Generate content with constraint awareness"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate(self, requirements: Dict[str, Any],
                constraints: List[Dict[str, Any]],
                constraint_params: Dict[str, Any]) -> str:
        """Generate content with constraint awareness"""
        # Build constraint description
        constraint_text = self._build_constraint_text(constraints, constraint_params)
        
        prompt = f"""Generate content following these STRICT constraints:

{constraint_text}

Requirements:
- Topic: {requirements.get('topic', 'general')}
- Purpose: {requirements.get('purpose', 'informative')}
- Audience: {requirements.get('audience', 'general')}

CRITICAL: You MUST satisfy ALL constraints listed above. Do not deviate from any constraint."""
        
        messages = [
            SystemMessage(content="You are a precision content generator that STRICTLY follows all constraints without exception."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content
    
    def _build_constraint_text(self, constraints: List[Dict[str, Any]],
                              params: Dict[str, Any]) -> str:
        """Build human-readable constraint text"""
        constraint_lines = []
        
        for constraint in constraints:
            line = f"- {constraint['description']}"
            
            # Add specific parameters
            if constraint['id'] == 'word_range':
                line += f" ({params.get('min_words', 0)}-{params.get('max_words', 1000)} words)"
            elif constraint['id'] == 'exact_word_count':
                line += f" ({params.get('exact_words')} words)"
            elif constraint['id'] == 'required_keywords':
                line += f" ({', '.join(params.get('required_keywords', []))})"
            elif constraint['id'] == 'forbidden_words':
                line += f" (avoid: {', '.join(params.get('forbidden_words', []))})"
            
            if constraint['strict']:
                line += " [STRICT]"
            
            constraint_lines.append(line)
        
        return '\n'.join(constraint_lines)


class ContentRepairer:
    """Repair content to satisfy violated constraints"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def repair(self, content: str, violations: List[Dict[str, Any]],
              params: Dict[str, Any]) -> str:
        """Repair content to fix violations"""
        if not violations:
            return content
        
        violation_text = "\n".join([
            f"- {v['description']} [{'STRICT' if v['strict'] else 'SOFT'}]"
            for v in violations
        ])
        
        prompt = f"""Fix the following content to satisfy ALL violated constraints:

Current Content:
{content}

Violated Constraints:
{violation_text}

Parameters: {json.dumps(params, indent=2)}

Generate the corrected version that satisfies ALL constraints while maintaining the core message."""
        
        messages = [
            SystemMessage(content="You are an expert content editor who fixes constraint violations precisely."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


# Agent functions
def initialize_generation(state: ConstrainedGenerationState) -> ConstrainedGenerationState:
    """Initialize constrained generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing constrained generation: {state['task_description']}"
    ))
    state["repair_attempts"] = 0
    state["current_step"] = "initialized"
    return state


def define_constraints(state: ConstrainedGenerationState) -> ConstrainedGenerationState:
    """Define and load constraints"""
    definition = ConstraintDefinition()
    
    constraint_types = state["generation_requirements"].get(
        "constraint_types",
        ["length", "vocabulary", "format"]
    )
    
    constraints = definition.get_constraints(constraint_types)
    
    # Filter to active constraints based on IDs
    active_constraint_ids = state["generation_requirements"].get("active_constraints", [])
    if active_constraint_ids:
        constraints = [c for c in constraints if c["id"] in active_constraint_ids]
    
    state["constraints"] = {
        "constraint_list": constraints,
        "params": state["generation_requirements"].get("constraint_params", {})
    }
    
    state["messages"].append(HumanMessage(
        content=f"Defined {len(constraints)} constraints across {len(constraint_types)} types"
    ))
    state["current_step"] = "constraints_defined"
    return state


def generate_content(state: ConstrainedGenerationState) -> ConstrainedGenerationState:
    """Generate content with constraints"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    generator = ConstrainedContentGenerator(llm)
    
    content = generator.generate(
        state["generation_requirements"],
        state["constraints"]["constraint_list"],
        state["constraints"]["params"]
    )
    
    state["generated_content"] = content
    state["messages"].append(HumanMessage(
        content=f"Generated content ({len(content.split())} words)"
    ))
    state["current_step"] = "content_generated"
    return state


def validate_constraints(state: ConstrainedGenerationState) -> ConstrainedGenerationState:
    """Validate content against constraints"""
    validator = ConstraintValidator()
    
    validation = validator.validate(
        state["generated_content"],
        state["constraints"]["constraint_list"],
        state["constraints"]["params"]
    )
    
    state["constraint_validation"] = validation
    state["violations"] = validation["violations"]
    state["constraint_satisfaction_score"] = validation["total_satisfaction"]
    
    state["messages"].append(HumanMessage(
        content=f"Validation: {validation['total_satisfaction']:.1%} satisfaction, "
                f"{len(validation['strict_violations'])} strict violations, "
                f"{len(validation['violations']) - len(validation['strict_violations'])} soft violations"
    ))
    state["current_step"] = "constraints_validated"
    return state


def repair_content(state: ConstrainedGenerationState) -> ConstrainedGenerationState:
    """Repair content if needed"""
    # Check if repair is needed
    if state["constraint_validation"]["is_valid"] or state["repair_attempts"] >= 3:
        state["final_output"] = state["generated_content"]
        state["current_step"] = "repair_complete"
        return state
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    repairer = ContentRepairer(llm)
    
    # Prioritize strict violations
    violations_to_fix = state["constraint_validation"]["strict_violations"]
    if not violations_to_fix:
        violations_to_fix = state["violations"][:3]  # Fix top 3 soft violations
    
    repaired = repairer.repair(
        state["generated_content"],
        violations_to_fix,
        state["constraints"]["params"]
    )
    
    state["generated_content"] = repaired
    state["repair_attempts"] += 1
    
    state["messages"].append(HumanMessage(
        content=f"Content repaired (attempt {state['repair_attempts']})"
    ))
    state["current_step"] = "content_repaired"
    return state


def check_repair_needed(state: ConstrainedGenerationState) -> str:
    """Check if more repair is needed"""
    if state["constraint_validation"]["is_valid"]:
        return "generate_report"
    elif state["repair_attempts"] >= 3:
        return "generate_report"
    else:
        return "validate_constraints"


def generate_report(state: ConstrainedGenerationState) -> ConstrainedGenerationState:
    """Generate final report"""
    validation = state["constraint_validation"]
    
    report = f"""
CONSTRAINED GENERATION REPORT
==============================

Task: {state['task_description']}

Constraints Applied: {len(state['constraints']['constraint_list'])}
Repair Attempts: {state['repair_attempts']}

Constraint Satisfaction:
- Total Satisfaction: {validation['total_satisfaction']:.1%}
- Strict Satisfaction: {validation['strict_satisfaction']:.1%}
- Validation Status: {'✅ PASSED' if validation['is_valid'] else '❌ FAILED'}

Constraint Results:
"""
    
    for result in validation['results']:
        status = "✅" if result['valid'] else "❌"
        strict = "[STRICT]" if result.get('strict', False) else "[SOFT]"
        report += f"{status} {result['description']} {strict}\n"
    
    if validation['violations']:
        report += f"\nRemaining Violations ({len(validation['violations'])}):\n"
        for v in validation['violations']:
            report += f"  - {v['description']} [{'STRICT' if v['strict'] else 'SOFT'}]\n"
    
    report += f"""
GENERATED CONTENT:
{'-' * 50}
{state['generated_content']}
{'-' * 50}

Content Stats:
- Words: {len(state['generated_content'].split())}
- Sentences: {len([s for s in re.split(r'[.!?]', state['generated_content']) if s.strip()])}
- Paragraphs: {len([p for p in state['generated_content'].split('\n\n') if p.strip()])}
"""
    
    state["final_output"] = state["generated_content"]
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_constrained_generation_graph():
    """Create the constrained generation workflow graph"""
    workflow = StateGraph(ConstrainedGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("define_constraints", define_constraints)
    workflow.add_node("generate_content", generate_content)
    workflow.add_node("validate_constraints", validate_constraints)
    workflow.add_node("repair_content", repair_content)
    workflow.add_node("generate_report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "define_constraints")
    workflow.add_edge("define_constraints", "generate_content")
    workflow.add_edge("generate_content", "validate_constraints")
    
    # Conditional edge for repair loop
    workflow.add_conditional_edges(
        "repair_content",
        check_repair_needed,
        {
            "validate_constraints": "validate_constraints",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("validate_constraints", "repair_content")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample constrained generation task
    initial_state = {
        "task_description": "Generate a constrained product description",
        "generation_requirements": {
            "topic": "Eco-Friendly Water Bottle",
            "purpose": "product marketing",
            "audience": "environmentally conscious consumers",
            "constraint_types": ["length", "vocabulary", "format", "structure"],
            "active_constraints": [
                "word_range",
                "required_keywords",
                "forbidden_words",
                "bullet_points",
                "has_sections"
            ],
            "constraint_params": {
                "min_words": 150,
                "max_words": 250,
                "required_keywords": ["sustainable", "eco-friendly", "recyclable", "durable"],
                "forbidden_words": ["cheap", "disposable", "plastic"],
                "min_bullets": 3,
                "required_sections": ["Features", "Benefits", "Specifications"]
            }
        },
        "constraints": {},
        "constraint_validation": {},
        "generated_content": "",
        "violations": [],
        "repair_attempts": 0,
        "constraint_satisfaction_score": 0.0,
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_constrained_generation_graph()
    
    print("Constrained Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Satisfaction Score: {result['constraint_satisfaction_score']:.1%}")

"""
Rule-Based Generation MCP Pattern

This pattern demonstrates content generation using explicit rules, constraints, and
conditional logic to create structured, compliant outputs based on defined rulesets.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Intermediate
"""

from typing import TypedDict, List, Dict, Annotated, Any, Callable
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
import re


# State definition
class RuleBasedGenerationState(TypedDict):
    """State for rule-based generation workflow"""
    task_description: str
    generation_context: Dict[str, Any]
    rule_sets: List[Dict[str, Any]]
    active_rules: List[Dict[str, Any]]
    rule_evaluation_results: Dict[str, Any]
    generated_content: str
    rule_violations: List[Dict[str, Any]]
    compliance_score: float
    refinement_iterations: int
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class RuleEngine:
    """Engine for managing and evaluating generation rules"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize predefined rule sets"""
        return {
            "content_length": [
                {
                    "id": "min_length",
                    "description": "Content must meet minimum length requirement",
                    "type": "constraint",
                    "check": lambda content, params: len(content.split()) >= params.get("min_words", 100),
                    "severity": "error",
                    "message": "Content is too short"
                },
                {
                    "id": "max_length",
                    "description": "Content must not exceed maximum length",
                    "type": "constraint",
                    "check": lambda content, params: len(content.split()) <= params.get("max_words", 1000),
                    "severity": "error",
                    "message": "Content is too long"
                }
            ],
            "structure": [
                {
                    "id": "has_introduction",
                    "description": "Content must have an introduction",
                    "type": "structural",
                    "check": lambda content, params: bool(re.search(r'(introduction|overview|about)', content, re.I)),
                    "severity": "warning",
                    "message": "Missing introduction section"
                },
                {
                    "id": "has_sections",
                    "description": "Content must have multiple sections",
                    "type": "structural",
                    "check": lambda content, params: len(re.findall(r'\n\n', content)) >= params.get("min_sections", 3),
                    "severity": "warning",
                    "message": "Insufficient content sections"
                },
                {
                    "id": "has_conclusion",
                    "description": "Content must have a conclusion",
                    "type": "structural",
                    "check": lambda content, params: bool(re.search(r'(conclusion|summary|finally)', content, re.I)),
                    "severity": "warning",
                    "message": "Missing conclusion section"
                }
            ],
            "tone": [
                {
                    "id": "professional_tone",
                    "description": "Content must maintain professional tone",
                    "type": "quality",
                    "check": lambda content, params: not any(word in content.lower() for word in ['gonna', 'wanna', 'yeah']),
                    "severity": "error",
                    "message": "Unprofessional language detected"
                },
                {
                    "id": "formal_language",
                    "description": "Content must use formal language",
                    "type": "quality",
                    "check": lambda content, params: not bool(re.search(r'[!]{2,}', content)),
                    "severity": "warning",
                    "message": "Overly casual punctuation"
                }
            ],
            "formatting": [
                {
                    "id": "proper_capitalization",
                    "description": "Sentences must start with capital letters",
                    "type": "formatting",
                    "check": lambda content, params: all(
                        s[0].isupper() for s in re.split(r'[.!?]\s+', content) if s.strip()
                    ),
                    "severity": "warning",
                    "message": "Improper capitalization"
                },
                {
                    "id": "no_excessive_whitespace",
                    "description": "No excessive whitespace",
                    "type": "formatting",
                    "check": lambda content, params: not bool(re.search(r'\s{3,}', content)),
                    "severity": "info",
                    "message": "Excessive whitespace detected"
                }
            ],
            "compliance": [
                {
                    "id": "no_prohibited_terms",
                    "description": "Must not contain prohibited terms",
                    "type": "compliance",
                    "check": lambda content, params: not any(
                        term in content.lower() for term in params.get("prohibited_terms", [])
                    ),
                    "severity": "error",
                    "message": "Prohibited terms detected"
                },
                {
                    "id": "required_keywords",
                    "description": "Must contain required keywords",
                    "type": "compliance",
                    "check": lambda content, params: all(
                        kw.lower() in content.lower() for kw in params.get("required_keywords", [])
                    ),
                    "severity": "warning",
                    "message": "Missing required keywords"
                }
            ]
        }
    
    def get_rules(self, rule_categories: List[str]) -> List[Dict[str, Any]]:
        """Get rules from specified categories"""
        rules = []
        for category in rule_categories:
            rules.extend(self.rules.get(category, []))
        return rules
    
    def evaluate_rule(self, rule: Dict[str, Any], content: str, 
                     params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single rule"""
        try:
            passed = rule["check"](content, params)
            return {
                "rule_id": rule["id"],
                "passed": passed,
                "severity": rule["severity"],
                "message": rule["message"] if not passed else "Rule satisfied"
            }
        except Exception as e:
            return {
                "rule_id": rule["id"],
                "passed": False,
                "severity": "error",
                "message": f"Rule evaluation failed: {str(e)}"
            }


class ContentGenerator:
    """Generate content following specified rules"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def generate_content(self, context: Dict[str, Any], 
                        rules: List[Dict[str, Any]]) -> str:
        """Generate content with rule awareness"""
        # Create rule-aware prompt
        rule_descriptions = "\n".join([
            f"- {rule['description']}" for rule in rules
        ])
        
        prompt = f"""Generate content based on the following requirements and rules:

Context: {json.dumps(context, indent=2)}

Rules to Follow:
{rule_descriptions}

Additional Instructions:
- Topic: {context.get('topic', 'general')}
- Target Length: {context.get('min_words', 100)}-{context.get('max_words', 1000)} words
- Tone: {context.get('tone', 'professional')}
- Format: {context.get('format', 'article')}

Generate comprehensive, well-structured content that strictly adheres to all rules."""
        
        messages = [
            SystemMessage(content="You are a content generation expert who strictly follows rules and guidelines."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class RuleValidator:
    """Validate generated content against rules"""
    
    def __init__(self, rule_engine: RuleEngine):
        self.rule_engine = rule_engine
    
    def validate_content(self, content: str, rules: List[Dict[str, Any]],
                        params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate content against all rules"""
        results = []
        violations = []
        
        for rule in rules:
            result = self.rule_engine.evaluate_rule(rule, content, params)
            results.append(result)
            
            if not result["passed"]:
                violations.append({
                    "rule_id": result["rule_id"],
                    "severity": result["severity"],
                    "message": result["message"]
                })
        
        # Calculate compliance score
        passed_count = sum(1 for r in results if r["passed"])
        total_count = len(results)
        compliance_score = passed_count / total_count if total_count > 0 else 0.0
        
        # Separate by severity
        errors = [v for v in violations if v["severity"] == "error"]
        warnings = [v for v in violations if v["severity"] == "warning"]
        info = [v for v in violations if v["severity"] == "info"]
        
        return {
            "results": results,
            "violations": violations,
            "compliance_score": compliance_score,
            "errors": errors,
            "warnings": warnings,
            "info": info,
            "is_valid": len(errors) == 0
        }


class ContentRefiner:
    """Refine content to fix rule violations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def refine_content(self, content: str, violations: List[Dict[str, Any]],
                      rules: List[Dict[str, Any]]) -> str:
        """Refine content to address violations"""
        if not violations:
            return content
        
        violation_details = "\n".join([
            f"- {v['severity'].upper()}: {v['message']}"
            for v in violations
        ])
        
        prompt = f"""Refine the following content to fix the identified violations while maintaining its core message and quality.

Current Content:
{content}

Violations to Fix:
{violation_details}

Rules to Follow:
{chr(10).join([f"- {rule['description']}" for rule in rules])}

Generate the refined version that addresses all violations."""
        
        messages = [
            SystemMessage(content="You are an expert content editor who fixes rule violations while preserving content quality."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content


class ComplianceReporter:
    """Generate compliance reports"""
    
    def generate_report(self, content: str, validation_results: Dict[str, Any],
                       iterations: int) -> str:
        """Generate detailed compliance report"""
        report = f"""
RULE-BASED GENERATION COMPLIANCE REPORT
=======================================

Content Length: {len(content.split())} words
Refinement Iterations: {iterations}

Compliance Summary:
- Overall Score: {validation_results['compliance_score']:.1%}
- Rules Passed: {sum(1 for r in validation_results['results'] if r['passed'])}/{len(validation_results['results'])}
- Errors: {len(validation_results['errors'])}
- Warnings: {len(validation_results['warnings'])}
- Info: {len(validation_results['info'])}

"""
        
        if validation_results['errors']:
            report += "\nERRORS:\n"
            for error in validation_results['errors']:
                report += f"  ❌ {error['message']}\n"
        
        if validation_results['warnings']:
            report += "\nWARNINGS:\n"
            for warning in validation_results['warnings']:
                report += f"  ⚠️  {warning['message']}\n"
        
        if validation_results['info']:
            report += "\nINFO:\n"
            for info_item in validation_results['info']:
                report += f"  ℹ️  {info_item['message']}\n"
        
        report += f"\nValidation Status: {'✅ PASSED' if validation_results['is_valid'] else '❌ FAILED'}\n"
        
        return report


# Agent functions
def initialize_generation(state: RuleBasedGenerationState) -> RuleBasedGenerationState:
    """Initialize rule-based generation"""
    state["messages"].append(HumanMessage(
        content=f"Initializing rule-based generation: {state['task_description']}"
    ))
    state["refinement_iterations"] = 0
    state["current_step"] = "initialized"
    return state


def load_rules(state: RuleBasedGenerationState) -> RuleBasedGenerationState:
    """Load applicable rules for generation"""
    engine = RuleEngine()
    
    # Get rule categories from context
    rule_categories = state["generation_context"].get(
        "rule_categories", 
        ["content_length", "structure", "tone", "formatting"]
    )
    
    rules = engine.get_rules(rule_categories)
    state["active_rules"] = rules
    
    state["messages"].append(HumanMessage(
        content=f"Loaded {len(rules)} rules across {len(rule_categories)} categories"
    ))
    state["current_step"] = "rules_loaded"
    return state


def generate_initial_content(state: RuleBasedGenerationState) -> RuleBasedGenerationState:
    """Generate initial content following rules"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    generator = ContentGenerator(llm)
    
    content = generator.generate_content(
        state["generation_context"],
        state["active_rules"]
    )
    
    state["generated_content"] = content
    state["messages"].append(HumanMessage(
        content=f"Generated initial content ({len(content.split())} words)"
    ))
    state["current_step"] = "content_generated"
    return state


def validate_rules(state: RuleBasedGenerationState) -> RuleBasedGenerationState:
    """Validate content against rules"""
    engine = RuleEngine()
    validator = RuleValidator(engine)
    
    validation_results = validator.validate_content(
        state["generated_content"],
        state["active_rules"],
        state["generation_context"]
    )
    
    state["rule_evaluation_results"] = validation_results
    state["rule_violations"] = validation_results["violations"]
    state["compliance_score"] = validation_results["compliance_score"]
    
    state["messages"].append(HumanMessage(
        content=f"Validation complete: {validation_results['compliance_score']:.1%} compliance, "
                f"{len(validation_results['errors'])} errors, "
                f"{len(validation_results['warnings'])} warnings"
    ))
    state["current_step"] = "rules_validated"
    return state


def refine_content(state: RuleBasedGenerationState) -> RuleBasedGenerationState:
    """Refine content to fix violations"""
    # Check if refinement is needed
    if state["compliance_score"] >= 1.0 or state["refinement_iterations"] >= 3:
        state["final_output"] = state["generated_content"]
        state["messages"].append(HumanMessage(
            content="No refinement needed or max iterations reached"
        ))
        state["current_step"] = "refinement_complete"
        return state
    
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    refiner = ContentRefiner(llm)
    
    # Focus on errors first, then warnings
    violations_to_fix = (
        state["rule_evaluation_results"]["errors"] +
        state["rule_evaluation_results"]["warnings"][:3]  # Limit warnings
    )
    
    refined_content = refiner.refine_content(
        state["generated_content"],
        violations_to_fix,
        state["active_rules"]
    )
    
    state["generated_content"] = refined_content
    state["refinement_iterations"] += 1
    
    state["messages"].append(HumanMessage(
        content=f"Content refined (iteration {state['refinement_iterations']})"
    ))
    state["current_step"] = "content_refined"
    return state


def check_refinement_needed(state: RuleBasedGenerationState) -> str:
    """Check if more refinement is needed"""
    if state["compliance_score"] >= 1.0:
        return "generate_report"
    elif state["refinement_iterations"] >= 3:
        return "generate_report"
    else:
        return "validate_rules"


def generate_compliance_report(state: RuleBasedGenerationState) -> RuleBasedGenerationState:
    """Generate final compliance report"""
    reporter = ComplianceReporter()
    
    report = reporter.generate_report(
        state["generated_content"],
        state["rule_evaluation_results"],
        state["refinement_iterations"]
    )
    
    final_report = f"""
{report}

GENERATED CONTENT:
{'-' * 50}
{state['generated_content']}
{'-' * 50}
"""
    
    state["final_output"] = state["generated_content"]
    state["messages"].append(HumanMessage(content=final_report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_rule_based_generation_graph():
    """Create the rule-based generation workflow graph"""
    workflow = StateGraph(RuleBasedGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("load_rules", load_rules)
    workflow.add_node("generate_content", generate_initial_content)
    workflow.add_node("validate_rules", validate_rules)
    workflow.add_node("refine_content", refine_content)
    workflow.add_node("generate_report", generate_compliance_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_rules")
    workflow.add_edge("load_rules", "generate_content")
    workflow.add_edge("generate_content", "validate_rules")
    
    # Conditional edge for refinement loop
    workflow.add_conditional_edges(
        "refine_content",
        check_refinement_needed,
        {
            "validate_rules": "validate_rules",
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_edge("validate_rules", "refine_content")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample generation task
    initial_state = {
        "task_description": "Generate a professional blog post about sustainable technology",
        "generation_context": {
            "topic": "Sustainable Technology and Green Innovation",
            "min_words": 200,
            "max_words": 500,
            "tone": "professional",
            "format": "blog_post",
            "rule_categories": ["content_length", "structure", "tone", "formatting", "compliance"],
            "required_keywords": ["sustainability", "technology", "innovation"],
            "prohibited_terms": ["guaranteed", "miracle", "secret"]
        },
        "rule_sets": [],
        "active_rules": [],
        "rule_evaluation_results": {},
        "generated_content": "",
        "rule_violations": [],
        "compliance_score": 0.0,
        "refinement_iterations": 0,
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_rule_based_generation_graph()
    
    print("Rule-Based Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Final Compliance Score: {result['compliance_score']:.1%}")

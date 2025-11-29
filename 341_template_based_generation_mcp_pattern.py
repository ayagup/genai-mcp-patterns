"""
Template-Based Generation MCP Pattern

This pattern demonstrates content generation using predefined templates with dynamic
variable substitution, template selection, and validation to create structured outputs.

Pattern Type: Generation
Category: Agentic MCP Pattern
Complexity: Intermediate
"""

from typing import TypedDict, List, Dict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
import re


# State definition
class TemplateGenerationState(TypedDict):
    """State for template-based generation workflow"""
    task_description: str
    generation_requirements: Dict[str, Any]
    available_templates: List[Dict[str, Any]]
    selected_template: Dict[str, Any]
    template_variables: Dict[str, Any]
    filled_template: str
    validation_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    generated_content: str
    messages: Annotated[List, operator.add]
    current_step: str


class TemplateLibrary:
    """Library of predefined templates for various content types"""
    
    def __init__(self):
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize template library with various content types"""
        return {
            "email": {
                "name": "Professional Email",
                "category": "communication",
                "template": """Subject: {subject}

Dear {recipient_name},

{opening_paragraph}

{main_content}

{closing_paragraph}

{sign_off},
{sender_name}
{sender_title}""",
                "variables": ["subject", "recipient_name", "opening_paragraph", 
                            "main_content", "closing_paragraph", "sign_off", 
                            "sender_name", "sender_title"],
                "validation_rules": {
                    "subject": {"max_length": 100, "required": True},
                    "recipient_name": {"required": True},
                    "main_content": {"min_length": 50, "required": True}
                }
            },
            "blog_post": {
                "name": "Blog Post Article",
                "category": "content",
                "template": """# {title}

**Published:** {publish_date}
**Author:** {author_name}
**Category:** {category}

## Introduction

{introduction}

## {section1_title}

{section1_content}

## {section2_title}

{section2_content}

## {section3_title}

{section3_content}

## Conclusion

{conclusion}

---
**Tags:** {tags}""",
                "variables": ["title", "publish_date", "author_name", "category",
                            "introduction", "section1_title", "section1_content",
                            "section2_title", "section2_content", "section3_title",
                            "section3_content", "conclusion", "tags"],
                "validation_rules": {
                    "title": {"max_length": 150, "required": True},
                    "introduction": {"min_length": 100, "max_length": 500, "required": True},
                    "conclusion": {"min_length": 50, "required": True}
                }
            },
            "product_description": {
                "name": "Product Description",
                "category": "marketing",
                "template": """**{product_name}**

{tagline}

**Price:** ${price}
**Availability:** {availability}

### Overview

{overview}

### Key Features

{features}

### Specifications

{specifications}

### What's Included

{whats_included}

### Customer Reviews

{customer_reviews}

**Rating:** {rating}/5.0 ({review_count} reviews)""",
                "variables": ["product_name", "tagline", "price", "availability",
                            "overview", "features", "specifications", "whats_included",
                            "customer_reviews", "rating", "review_count"],
                "validation_rules": {
                    "product_name": {"max_length": 100, "required": True},
                    "price": {"type": "numeric", "required": True},
                    "overview": {"min_length": 100, "required": True},
                    "features": {"min_length": 50, "required": True}
                }
            },
            "report": {
                "name": "Business Report",
                "category": "business",
                "template": """# {report_title}

**Report Type:** {report_type}
**Date:** {report_date}
**Prepared by:** {author}
**Department:** {department}

## Executive Summary

{executive_summary}

## Background

{background}

## Findings

{findings}

## Analysis

{analysis}

## Recommendations

{recommendations}

## Conclusion

{conclusion}

---
**Confidentiality:** {confidentiality_level}""",
                "variables": ["report_title", "report_type", "report_date", "author",
                            "department", "executive_summary", "background", "findings",
                            "analysis", "recommendations", "conclusion", "confidentiality_level"],
                "validation_rules": {
                    "report_title": {"max_length": 200, "required": True},
                    "executive_summary": {"min_length": 100, "max_length": 500, "required": True},
                    "findings": {"min_length": 200, "required": True}
                }
            },
            "job_posting": {
                "name": "Job Posting",
                "category": "recruitment",
                "template": """# {job_title}

**Company:** {company_name}
**Location:** {location}
**Job Type:** {job_type}
**Experience Level:** {experience_level}
**Salary Range:** {salary_range}

## About the Company

{company_description}

## Job Description

{job_description}

## Responsibilities

{responsibilities}

## Requirements

{requirements}

## Benefits

{benefits}

## How to Apply

{application_instructions}

**Application Deadline:** {deadline}
**Contact:** {contact_email}""",
                "variables": ["job_title", "company_name", "location", "job_type",
                            "experience_level", "salary_range", "company_description",
                            "job_description", "responsibilities", "requirements",
                            "benefits", "application_instructions", "deadline", "contact_email"],
                "validation_rules": {
                    "job_title": {"max_length": 100, "required": True},
                    "job_description": {"min_length": 100, "required": True},
                    "responsibilities": {"min_length": 100, "required": True},
                    "requirements": {"min_length": 50, "required": True}
                }
            }
        }
    
    def get_template(self, template_type: str) -> Dict[str, Any]:
        """Retrieve a specific template"""
        return self.templates.get(template_type, {})
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates"""
        return [
            {
                "type": key,
                "name": value["name"],
                "category": value["category"],
                "variables": value["variables"]
            }
            for key, value in self.templates.items()
        ]


class TemplateSelector:
    """Select appropriate template based on requirements"""
    
    def select_template(self, requirements: Dict[str, Any], 
                       templates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best matching template"""
        content_type = requirements.get("content_type", "").lower()
        category = requirements.get("category", "").lower()
        
        # Direct match by content type
        for template in templates:
            if template["type"] == content_type:
                return template
        
        # Match by category
        for template in templates:
            if template.get("category", "").lower() == category:
                return template
        
        # Default to first template
        return templates[0] if templates else {}


class VariableExtractor:
    """Extract and generate values for template variables"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def extract_variables(self, template: Dict[str, Any], 
                         requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Extract variable values from requirements or generate them"""
        variables = {}
        provided_data = requirements.get("data", {})
        
        # First, use any provided data
        for var in template["variables"]:
            if var in provided_data:
                variables[var] = provided_data[var]
        
        # Generate missing variables using LLM
        missing_vars = [v for v in template["variables"] if v not in variables]
        if missing_vars:
            generated = self._generate_missing_variables(
                template, missing_vars, requirements, variables
            )
            variables.update(generated)
        
        return variables
    
    def _generate_missing_variables(self, template: Dict[str, Any],
                                   missing_vars: List[str],
                                   requirements: Dict[str, Any],
                                   existing_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Generate values for missing variables"""
        prompt = f"""Generate appropriate content for the following template variables.

Template Type: {template['name']}
Requirements: {json.dumps(requirements, indent=2)}
Existing Variables: {json.dumps(existing_vars, indent=2)}

Missing Variables: {', '.join(missing_vars)}

For each missing variable, generate appropriate, realistic content that fits the template context.
Return as JSON with variable names as keys.

Example format:
{{
    "variable1": "Generated content for variable 1",
    "variable2": "Generated content for variable 2"
}}"""
        
        messages = [
            SystemMessage(content="You are a content generation expert."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            # Extract JSON from response
            content = response.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
            return {}
        except:
            return {}


class TemplateFiller:
    """Fill template with variable values"""
    
    def fill_template(self, template_string: str, 
                     variables: Dict[str, Any]) -> str:
        """Replace template placeholders with variable values"""
        filled = template_string
        
        for var, value in variables.items():
            placeholder = "{" + var + "}"
            filled = filled.replace(placeholder, str(value))
        
        return filled
    
    def validate_fill_completeness(self, filled_template: str) -> Dict[str, Any]:
        """Check if all placeholders were filled"""
        unfilled = re.findall(r'\{(\w+)\}', filled_template)
        
        return {
            "complete": len(unfilled) == 0,
            "unfilled_variables": unfilled,
            "fill_rate": 1.0 if len(unfilled) == 0 else 0.0
        }


class TemplateValidator:
    """Validate filled template against rules"""
    
    def validate_template(self, filled_template: str, 
                         variables: Dict[str, Any],
                         validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template content against rules"""
        issues = []
        
        for var, rules in validation_rules.items():
            if var not in variables:
                if rules.get("required", False):
                    issues.append(f"Required variable '{var}' is missing")
                continue
            
            value = str(variables[var])
            
            # Check length constraints
            if "min_length" in rules and len(value) < rules["min_length"]:
                issues.append(f"Variable '{var}' is too short (min: {rules['min_length']})")
            
            if "max_length" in rules and len(value) > rules["max_length"]:
                issues.append(f"Variable '{var}' is too long (max: {rules['max_length']})")
            
            # Check type constraints
            if rules.get("type") == "numeric":
                try:
                    float(value.replace('$', '').replace(',', ''))
                except ValueError:
                    issues.append(f"Variable '{var}' must be numeric")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "validation_score": 0.0 if issues else 1.0
        }


class QualityAssessor:
    """Assess quality of generated content"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def assess_quality(self, content: str, template_type: str) -> Dict[str, float]:
        """Assess various quality metrics"""
        return {
            "readability": self._assess_readability(content),
            "completeness": self._assess_completeness(content),
            "coherence": self._assess_coherence(content),
            "professionalism": self._assess_professionalism(content, template_type)
        }
    
    def _assess_readability(self, content: str) -> float:
        """Assess readability based on sentence structure"""
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.0
        
        # Average sentence length (optimal: 15-20 words)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
        
        if 15 <= avg_length <= 20:
            return 1.0
        elif 10 <= avg_length <= 25:
            return 0.8
        else:
            return 0.6
    
    def _assess_completeness(self, content: str) -> float:
        """Assess content completeness"""
        # Check for essential sections
        has_intro = bool(re.search(r'(introduction|overview|about)', content, re.I))
        has_body = len(content) > 200
        has_conclusion = bool(re.search(r'(conclusion|summary|contact)', content, re.I))
        
        score = 0.0
        if has_intro:
            score += 0.3
        if has_body:
            score += 0.4
        if has_conclusion:
            score += 0.3
        
        return score
    
    def _assess_coherence(self, content: str) -> float:
        """Assess content coherence"""
        # Simple coherence check based on structure
        sections = re.split(r'\n\n+', content)
        
        if len(sections) < 3:
            return 0.5
        elif len(sections) >= 5:
            return 0.9
        else:
            return 0.7
    
    def _assess_professionalism(self, content: str, template_type: str) -> float:
        """Assess professionalism based on template type"""
        # Check for informal language
        informal_words = ['gonna', 'wanna', 'yeah', 'cool', 'awesome', 'stuff']
        informal_count = sum(1 for word in informal_words if word in content.lower())
        
        if informal_count > 0:
            return 0.6
        
        # Business templates should have formal tone
        if template_type in ['report', 'email', 'job_posting']:
            has_formal_language = bool(re.search(
                r'(please|kindly|sincerely|respectfully)', content, re.I
            ))
            return 0.9 if has_formal_language else 0.7
        
        return 0.8


# Agent functions
def initialize_generation(state: TemplateGenerationState) -> TemplateGenerationState:
    """Initialize template generation process"""
    state["messages"].append(HumanMessage(
        content=f"Initializing template-based generation for: {state['task_description']}"
    ))
    state["current_step"] = "initialized"
    return state


def load_templates(state: TemplateGenerationState) -> TemplateGenerationState:
    """Load available templates from library"""
    library = TemplateLibrary()
    templates = library.list_templates()
    
    # Add full template data
    state["available_templates"] = []
    for template_info in templates:
        full_template = library.get_template(template_info["type"])
        state["available_templates"].append({
            "type": template_info["type"],
            **full_template
        })
    
    state["messages"].append(HumanMessage(
        content=f"Loaded {len(templates)} templates: {', '.join(t['name'] for t in templates)}"
    ))
    state["current_step"] = "templates_loaded"
    return state


def select_template(state: TemplateGenerationState) -> TemplateGenerationState:
    """Select appropriate template for generation"""
    selector = TemplateSelector()
    selected = selector.select_template(
        state["generation_requirements"],
        state["available_templates"]
    )
    
    state["selected_template"] = selected
    state["messages"].append(HumanMessage(
        content=f"Selected template: {selected.get('name', 'Unknown')}"
    ))
    state["current_step"] = "template_selected"
    return state


def extract_variables(state: TemplateGenerationState) -> TemplateGenerationState:
    """Extract and generate template variable values"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.7)
    extractor = VariableExtractor(llm)
    
    variables = extractor.extract_variables(
        state["selected_template"],
        state["generation_requirements"]
    )
    
    state["template_variables"] = variables
    state["messages"].append(HumanMessage(
        content=f"Extracted {len(variables)} variables"
    ))
    state["current_step"] = "variables_extracted"
    return state


def fill_template(state: TemplateGenerationState) -> TemplateGenerationState:
    """Fill template with variable values"""
    filler = TemplateFiller()
    
    filled = filler.fill_template(
        state["selected_template"]["template"],
        state["template_variables"]
    )
    
    completeness = filler.validate_fill_completeness(filled)
    
    state["filled_template"] = filled
    state["messages"].append(HumanMessage(
        content=f"Template filled (completeness: {completeness['fill_rate']:.1%})"
    ))
    state["current_step"] = "template_filled"
    return state


def validate_template(state: TemplateGenerationState) -> TemplateGenerationState:
    """Validate filled template"""
    validator = TemplateValidator()
    
    validation_results = validator.validate_template(
        state["filled_template"],
        state["template_variables"],
        state["selected_template"].get("validation_rules", {})
    )
    
    state["validation_results"] = validation_results
    state["messages"].append(HumanMessage(
        content=f"Validation: {'Passed' if validation_results['valid'] else 'Failed'} "
                f"(score: {validation_results['validation_score']:.2f})"
    ))
    state["current_step"] = "template_validated"
    return state


def assess_quality(state: TemplateGenerationState) -> TemplateGenerationState:
    """Assess quality of generated content"""
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    assessor = QualityAssessor(llm)
    
    metrics = assessor.assess_quality(
        state["filled_template"],
        state["selected_template"]["type"]
    )
    
    state["quality_metrics"] = metrics
    state["generated_content"] = state["filled_template"]
    
    avg_quality = sum(metrics.values()) / len(metrics)
    state["messages"].append(HumanMessage(
        content=f"Quality assessment complete (average: {avg_quality:.2f})"
    ))
    state["current_step"] = "quality_assessed"
    return state


def generate_report(state: TemplateGenerationState) -> TemplateGenerationState:
    """Generate final generation report"""
    report = f"""
TEMPLATE-BASED GENERATION REPORT
================================

Task: {state['task_description']}

Selected Template: {state['selected_template']['name']}
Template Category: {state['selected_template']['category']}
Variables Used: {len(state['template_variables'])}

Validation Results:
- Valid: {state['validation_results']['valid']}
- Validation Score: {state['validation_results']['validation_score']:.2f}
- Issues: {len(state['validation_results'].get('issues', []))}

Quality Metrics:
- Readability: {state['quality_metrics']['readability']:.2f}
- Completeness: {state['quality_metrics']['completeness']:.2f}
- Coherence: {state['quality_metrics']['coherence']:.2f}
- Professionalism: {state['quality_metrics']['professionalism']:.2f}
- Overall Quality: {sum(state['quality_metrics'].values()) / len(state['quality_metrics']):.2f}

Generated Content Length: {len(state['generated_content'])} characters

GENERATED CONTENT:
{'-' * 50}
{state['generated_content']}
{'-' * 50}
"""
    
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_template_generation_graph():
    """Create the template-based generation workflow graph"""
    workflow = StateGraph(TemplateGenerationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_generation)
    workflow.add_node("load_templates", load_templates)
    workflow.add_node("select_template", select_template)
    workflow.add_node("extract_variables", extract_variables)
    workflow.add_node("fill_template", fill_template)
    workflow.add_node("validate", validate_template)
    workflow.add_node("assess_quality", assess_quality)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_templates")
    workflow.add_edge("load_templates", "select_template")
    workflow.add_edge("select_template", "extract_variables")
    workflow.add_edge("extract_variables", "fill_template")
    workflow.add_edge("fill_template", "validate")
    workflow.add_edge("validate", "assess_quality")
    workflow.add_edge("assess_quality", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample generation task
    initial_state = {
        "task_description": "Generate a professional product description for a new smartphone",
        "generation_requirements": {
            "content_type": "product_description",
            "category": "marketing",
            "data": {
                "product_name": "TechPro X5 Smartphone",
                "price": "799.99",
                "availability": "In Stock",
                "rating": "4.7",
                "review_count": "1,247"
            }
        },
        "available_templates": [],
        "selected_template": {},
        "template_variables": {},
        "filled_template": "",
        "validation_results": {},
        "quality_metrics": {},
        "generated_content": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_template_generation_graph()
    
    print("Template-Based Generation MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")

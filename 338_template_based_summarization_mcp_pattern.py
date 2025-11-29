"""
Template-Based Summarization MCP Pattern

This pattern demonstrates template-based summarization in an agentic MCP system.
The system uses predefined templates to create structured, domain-specific summaries.

Use cases:
- Medical report summarization
- Legal document summaries
- Financial briefings
- Research paper abstracts
- Product specifications
"""

from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from operator import add
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import re
from collections import defaultdict


# Define the state for template-based summarization
class TemplateBasedSummarizationState(TypedDict):
    """State for tracking template-based summarization process"""
    messages: Annotated[List[str], add]
    source_document: str
    document_type: str
    available_templates: List[Dict[str, Any]]
    selected_template: Dict[str, Any]
    extracted_fields: Dict[str, Any]
    filled_template: str
    validation_results: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    analytics: Dict[str, Any]
    report: str


class TemplateRegistry:
    """Registry of summarization templates"""
    
    def get_templates(self) -> List[Dict[str, Any]]:
        """Get available templates"""
        
        return [
            {
                'id': 'medical_report',
                'name': 'Medical Report Summary',
                'domain': 'healthcare',
                'fields': [
                    'patient_info', 'chief_complaint', 'diagnosis',
                    'treatment', 'prognosis', 'recommendations'
                ],
                'template': """
MEDICAL REPORT SUMMARY
Patient: {patient_info}
Chief Complaint: {chief_complaint}
Diagnosis: {diagnosis}
Treatment Plan: {treatment}
Prognosis: {prognosis}
Recommendations: {recommendations}
                """.strip()
            },
            {
                'id': 'research_abstract',
                'name': 'Research Paper Abstract',
                'domain': 'academic',
                'fields': [
                    'background', 'objective', 'methods',
                    'results', 'conclusions', 'implications'
                ],
                'template': """
ABSTRACT
Background: {background}
Objective: {objective}
Methods: {methods}
Results: {results}
Conclusions: {conclusions}
Implications: {implications}
                """.strip()
            },
            {
                'id': 'financial_brief',
                'name': 'Financial Briefing',
                'domain': 'finance',
                'fields': [
                    'company', 'period', 'revenue', 'expenses',
                    'profit', 'key_metrics', 'outlook'
                ],
                'template': """
FINANCIAL BRIEFING
Company: {company}
Period: {period}
Revenue: {revenue}
Expenses: {expenses}
Net Profit: {profit}
Key Metrics: {key_metrics}
Outlook: {outlook}
                """.strip()
            },
            {
                'id': 'meeting_minutes',
                'name': 'Meeting Minutes Summary',
                'domain': 'business',
                'fields': [
                    'date', 'attendees', 'agenda', 'discussions',
                    'decisions', 'action_items', 'next_meeting'
                ],
                'template': """
MEETING MINUTES
Date: {date}
Attendees: {attendees}
Agenda: {agenda}
Key Discussions: {discussions}
Decisions Made: {decisions}
Action Items: {action_items}
Next Meeting: {next_meeting}
                """.strip()
            },
            {
                'id': 'product_spec',
                'name': 'Product Specification',
                'domain': 'product',
                'fields': [
                    'product_name', 'category', 'features',
                    'specifications', 'benefits', 'pricing'
                ],
                'template': """
PRODUCT SPECIFICATION
Product: {product_name}
Category: {category}
Key Features: {features}
Technical Specifications: {specifications}
Benefits: {benefits}
Pricing: {pricing}
                """.strip()
            }
        ]
    
    def select_template(self, document_type: str) -> Dict[str, Any]:
        """Select appropriate template"""
        
        templates = self.get_templates()
        
        # Simple matching by domain
        for template in templates:
            if template['domain'] == document_type or template['id'] == document_type:
                return template
        
        # Default to first template
        return templates[0]


class FieldExtractor:
    """Extract fields from document"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def extract_field(self, document: str, field_name: str, 
                     field_description: str = '') -> str:
        """Extract specific field from document"""
        
        system_prompt = f"""You are an expert at extracting specific information from documents.
        Extract the {field_name} from the document. Be concise and accurate."""
        
        description = field_description or f"the {field_name.replace('_', ' ')}"
        
        user_prompt = f"""
        Extract {description} from this document:
        
        {document}
        
        Provide only the extracted information, nothing else. If not found, write "Not specified".
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        return response.content.strip()
    
    def extract_all_fields(self, document: str, 
                          template: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all required fields"""
        
        field_descriptions = {
            # Medical
            'patient_info': 'patient demographic information',
            'chief_complaint': 'main reason for visit',
            'diagnosis': 'medical diagnosis',
            'treatment': 'treatment plan',
            'prognosis': 'expected outcome',
            'recommendations': 'follow-up recommendations',
            
            # Research
            'background': 'research background and context',
            'objective': 'research objective or hypothesis',
            'methods': 'research methodology',
            'results': 'key findings and results',
            'conclusions': 'main conclusions',
            'implications': 'practical implications',
            
            # Financial
            'company': 'company name',
            'period': 'reporting period',
            'revenue': 'total revenue',
            'expenses': 'total expenses',
            'profit': 'net profit',
            'key_metrics': 'important financial metrics',
            'outlook': 'future outlook',
            
            # Meeting
            'date': 'meeting date',
            'attendees': 'list of attendees',
            'agenda': 'meeting agenda',
            'discussions': 'key discussion points',
            'decisions': 'decisions made',
            'action_items': 'action items assigned',
            'next_meeting': 'next meeting details',
            
            # Product
            'product_name': 'product name',
            'category': 'product category',
            'features': 'key features',
            'specifications': 'technical specifications',
            'benefits': 'customer benefits',
            'pricing': 'pricing information'
        }
        
        extracted = {}
        for field in template['fields']:
            description = field_descriptions.get(field, f"the {field}")
            extracted[field] = self.extract_field(document, field, description)
        
        return extracted


class TemplateFiller:
    """Fill template with extracted data"""
    
    def fill_template(self, template: Dict[str, Any], 
                     fields: Dict[str, Any]) -> str:
        """Fill template with field values"""
        
        filled = template['template']
        
        for field_name, field_value in fields.items():
            placeholder = '{' + field_name + '}'
            filled = filled.replace(placeholder, field_value)
        
        return filled
    
    def post_process(self, filled_template: str) -> str:
        """Post-process filled template"""
        
        # Remove any unfilled placeholders
        filled_template = re.sub(r'\{[^}]+\}', 'Not specified', filled_template)
        
        # Clean up extra whitespace
        lines = filled_template.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines)


class TemplateValidator:
    """Validate filled template"""
    
    def check_completeness(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Check if all fields are filled"""
        
        filled = {}
        missing = []
        
        for field_name, field_value in fields.items():
            if field_value and field_value.lower() not in ['not specified', 'n/a', 'none']:
                filled[field_name] = True
            else:
                filled[field_name] = False
                missing.append(field_name)
        
        completeness = sum(filled.values()) / len(filled) if filled else 0.0
        
        return {
            'completeness': completeness,
            'filled_fields': [k for k, v in filled.items() if v],
            'missing_fields': missing
        }
    
    def check_consistency(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """Check internal consistency"""
        
        # Simple checks for consistency
        issues = []
        
        # Check for contradictions (simplified)
        field_values = ' '.join(str(v) for v in fields.values()).lower()
        
        if 'positive' in field_values and 'negative' in field_values:
            issues.append('Potential contradiction between positive and negative statements')
        
        # Check for reasonable lengths
        for field_name, field_value in fields.items():
            if len(str(field_value).split()) > 50:
                issues.append(f'{field_name} may be too verbose')
        
        consistency_score = max(0.0, 1.0 - len(issues) * 0.2)
        
        return {
            'consistency_score': consistency_score,
            'issues': issues
        }


class QualityAssessor:
    """Assess template-based summary quality"""
    
    def assess_structure(self, filled_template: str, template: Dict[str, Any]) -> float:
        """Assess structural quality"""
        
        # Check if all section headers are present
        expected_sections = len(template['fields'])
        present_sections = sum(1 for field in template['fields'] if field.replace('_', ' ') in filled_template.lower())
        
        structure_score = present_sections / expected_sections if expected_sections > 0 else 0.0
        return structure_score
    
    def assess_readability(self, filled_template: str) -> float:
        """Assess readability"""
        
        # Simple readability metrics
        words = filled_template.split()
        sentences = re.split(r'[.!?]+', filled_template)
        
        if not sentences or not words:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Optimal range: 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            readability = 1.0
        elif avg_words_per_sentence < 15:
            readability = 0.8
        else:
            readability = max(0.5, 1.0 - (avg_words_per_sentence - 20) * 0.02)
        
        return readability


# Agent functions
def initialize_document_and_template_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Initialize document and template"""
    
    # Sample research document
    document = """
    Recent advances in quantum computing have opened new possibilities for solving 
    previously intractable computational problems. This study investigates the application 
    of quantum annealing algorithms to optimization problems in logistics and supply chain 
    management. The objective was to determine whether quantum approaches could provide 
    significant speedups over classical algorithms for real-world routing problems.
    
    We employed a hybrid quantum-classical approach, using D-Wave's quantum annealer for 
    the optimization core while classical preprocessing and postprocessing handled problem 
    encoding and solution validation. The methodology involved testing on benchmark routing 
    datasets with varying complexity levels, from 50 to 500 nodes.
    
    Results demonstrated that quantum annealing achieved 3-5x speedup on medium-complexity 
    problems (100-200 nodes) compared to state-of-the-art classical solvers. However, for 
    very large problems (>300 nodes), problem decomposition overhead reduced the advantage. 
    Solution quality was comparable to classical methods, with 95% optimality on average.
    
    We conclude that quantum annealing shows promise for mid-scale logistics optimization, 
    particularly in time-critical scenarios where faster-than-classical solutions provide 
    business value. The findings suggest that hybrid quantum-classical approaches represent 
    a practical path to quantum advantage in optimization. Further research should focus on 
    improving problem decomposition strategies and exploring industry-specific applications.
    """
    
    document_type = 'research_abstract'
    
    return {
        **state,
        'source_document': document,
        'document_type': document_type,
        'messages': state['messages'] + [f'Initialized document (type: {document_type}, {len(document.split())} words)']
    }


def load_templates_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Load available templates"""
    
    registry = TemplateRegistry()
    templates = registry.get_templates()
    
    return {
        **state,
        'available_templates': templates,
        'messages': state['messages'] + [f'Loaded {len(templates)} templates']
    }


def select_template_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Select appropriate template"""
    
    registry = TemplateRegistry()
    template = registry.select_template(state['document_type'])
    
    return {
        **state,
        'selected_template': template,
        'messages': state['messages'] + [f'Selected template: {template["name"]} ({len(template["fields"])} fields)']
    }


def extract_fields_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Extract fields from document"""
    
    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(
        state['source_document'],
        state['selected_template']
    )
    
    return {
        **state,
        'extracted_fields': fields,
        'messages': state['messages'] + [f'Extracted {len(fields)} fields']
    }


def fill_template_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Fill template with extracted fields"""
    
    filler = TemplateFiller()
    filled = filler.fill_template(state['selected_template'], state['extracted_fields'])
    filled = filler.post_process(filled)
    
    return {
        **state,
        'filled_template': filled,
        'messages': state['messages'] + [f'Filled template ({len(filled.split())} words)']
    }


def validate_template_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Validate filled template"""
    
    validator = TemplateValidator()
    
    completeness_results = validator.check_completeness(state['extracted_fields'])
    consistency_results = validator.check_consistency(state['extracted_fields'])
    
    validation = {
        'completeness': completeness_results,
        'consistency': consistency_results
    }
    
    return {
        **state,
        'validation_results': validation,
        'messages': state['messages'] + [f'Validated template (completeness: {completeness_results["completeness"]:.1%})']
    }


def assess_quality_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Assess summary quality"""
    
    assessor = QualityAssessor()
    
    metrics = {
        'structure': assessor.assess_structure(state['filled_template'], state['selected_template']),
        'readability': assessor.assess_readability(state['filled_template']),
        'completeness': state['validation_results']['completeness']['completeness'],
        'consistency': state['validation_results']['consistency']['consistency_score'],
        'template_conformance': 1.0 if state['filled_template'] else 0.0
    }
    
    # Overall score
    metrics['overall_quality'] = sum(metrics.values()) / len(metrics)
    
    return {
        **state,
        'quality_metrics': metrics,
        'messages': state['messages'] + [f'Assessed quality (overall: {metrics["overall_quality"]:.1%})']
    }


def analyze_template_summarization_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Analyze template-based summarization results"""
    
    analytics = {
        'template_info': {
            'template_id': state['selected_template']['id'],
            'template_name': state['selected_template']['name'],
            'domain': state['selected_template']['domain'],
            'fields_count': len(state['selected_template']['fields'])
        },
        'extraction_stats': {
            'fields_extracted': len(state['extracted_fields']),
            'fields_filled': len(state['validation_results']['completeness']['filled_fields']),
            'fields_missing': len(state['validation_results']['completeness']['missing_fields'])
        },
        'quality_stats': state['quality_metrics'],
        'validation_stats': {
            'completeness': state['validation_results']['completeness']['completeness'],
            'consistency_score': state['validation_results']['consistency']['consistency_score'],
            'issues_found': len(state['validation_results']['consistency']['issues'])
        }
    }
    
    return {
        **state,
        'analytics': analytics,
        'messages': state['messages'] + ['Analyzed template-based summarization results']
    }


def generate_template_report_agent(state: TemplateBasedSummarizationState) -> TemplateBasedSummarizationState:
    """Generate comprehensive report"""
    
    report_lines = [
        "=" * 80,
        "TEMPLATE-BASED SUMMARIZATION REPORT",
        "=" * 80,
        "",
        "TEMPLATE USED:",
        "-" * 40,
        f"Name: {state['selected_template']['name']}",
        f"Domain: {state['selected_template']['domain']}",
        f"Fields: {', '.join(state['selected_template']['fields'])}",
        "",
        "FILLED SUMMARY:",
        "-" * 40,
        state['filled_template'],
        "",
        "",
        "FIELD EXTRACTION DETAILS:",
        "-" * 40
    ]
    
    for field_name, field_value in state['extracted_fields'].items():
        status = "✓" if field_name in state['validation_results']['completeness']['filled_fields'] else "✗"
        report_lines.append(f"{status} {field_name}: {field_value[:80]}{'...' if len(field_value) > 80 else ''}")
    
    report_lines.extend([
        "",
        "",
        "VALIDATION RESULTS:",
        "-" * 40,
        f"Completeness: {state['validation_results']['completeness']['completeness']:.1%}",
        f"Filled Fields: {len(state['validation_results']['completeness']['filled_fields'])}",
        f"Missing Fields: {', '.join(state['validation_results']['completeness']['missing_fields']) if state['validation_results']['completeness']['missing_fields'] else 'None'}",
        f"Consistency Score: {state['validation_results']['consistency']['consistency_score']:.1%}",
        ""
    ])
    
    if state['validation_results']['consistency']['issues']:
        report_lines.append("Consistency Issues:")
        for issue in state['validation_results']['consistency']['issues']:
            report_lines.append(f"  • {issue}")
    else:
        report_lines.append("No consistency issues found")
    
    report_lines.extend([
        "",
        "",
        "QUALITY METRICS:",
        "-" * 40,
        f"Overall Quality: {state['analytics']['quality_stats']['overall_quality']:.1%}",
        f"Structure: {state['analytics']['quality_stats']['structure']:.1%}",
        f"Readability: {state['analytics']['quality_stats']['readability']:.1%}",
        f"Completeness: {state['analytics']['quality_stats']['completeness']:.1%}",
        f"Consistency: {state['analytics']['quality_stats']['consistency']:.1%}",
        "",
        "STATISTICS:",
        "-" * 40,
        f"Template: {state['analytics']['template_info']['template_name']}",
        f"Domain: {state['analytics']['template_info']['domain']}",
        f"Total Fields: {state['analytics']['template_info']['fields_count']}",
        f"Fields Filled: {state['analytics']['extraction_stats']['fields_filled']} / {state['analytics']['extraction_stats']['fields_extracted']}",
        f"Success Rate: {state['analytics']['extraction_stats']['fields_filled'] / state['analytics']['extraction_stats']['fields_extracted']:.1%}",
        "",
        "INSIGHTS:",
        "-" * 40,
        f"✓ Successfully applied {state['selected_template']['name']} template",
        f"✓ Extracted {state['analytics']['extraction_stats']['fields_filled']} of {state['analytics']['extraction_stats']['fields_count']} fields",
        f"✓ Achieved {state['analytics']['quality_stats']['overall_quality']:.0%} overall quality",
        f"✓ Maintained {state['analytics']['quality_stats']['structure']:.0%} structural conformance",
        "",
        "RECOMMENDATIONS:",
        "-" * 40,
        "• Implement field validation rules for critical fields",
        "• Add domain-specific templates for better coverage",
        "• Enable template customization for specific use cases",
        "• Implement automatic template selection based on content analysis",
        "• Add multi-language template support",
        "• Enable template versioning and evolution tracking",
        "",
        "=" * 80
    ])
    
    report = "\n".join(report_lines)
    
    return {
        **state,
        'report': report,
        'messages': state['messages'] + ['Generated comprehensive report']
    }


# Create the graph
def create_template_based_summarization_graph():
    """Create the template-based summarization workflow graph"""
    
    workflow = StateGraph(TemplateBasedSummarizationState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_document_and_template_agent)
    workflow.add_node("load_templates", load_templates_agent)
    workflow.add_node("select_template", select_template_agent)
    workflow.add_node("extract_fields", extract_fields_agent)
    workflow.add_node("fill_template", fill_template_agent)
    workflow.add_node("validate_template", validate_template_agent)
    workflow.add_node("assess_quality", assess_quality_agent)
    workflow.add_node("analyze_results", analyze_template_summarization_agent)
    workflow.add_node("generate_report", generate_template_report_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_templates")
    workflow.add_edge("load_templates", "select_template")
    workflow.add_edge("select_template", "extract_fields")
    workflow.add_edge("extract_fields", "fill_template")
    workflow.add_edge("fill_template", "validate_template")
    workflow.add_edge("validate_template", "assess_quality")
    workflow.add_edge("assess_quality", "analyze_results")
    workflow.add_edge("analyze_results", "generate_report")
    workflow.add_edge("generate_report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Create and run the template-based summarization graph
    app = create_template_based_summarization_graph()
    
    # Initialize state
    initial_state: TemplateBasedSummarizationState = {
        'messages': [],
        'source_document': '',
        'document_type': '',
        'available_templates': [],
        'selected_template': {},
        'extracted_fields': {},
        'filled_template': '',
        'validation_results': {},
        'quality_metrics': {},
        'analytics': {},
        'report': ''
    }
    
    # Run the workflow
    result = app.invoke(initial_state)
    
    # Print results
    print("\n" + "=" * 80)
    print("TEMPLATE-BASED SUMMARIZATION MCP PATTERN - EXECUTION RESULTS")
    print("=" * 80 + "\n")
    
    print("Workflow Steps:")
    print("-" * 40)
    for i, message in enumerate(result['messages'], 1):
        print(f"{i}. {message}")
    
    print("\n" + result['report'])
    
    print("\nTemplate-based summarization pattern execution complete! ✓")

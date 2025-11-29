"""
Analogical Reasoning MCP Pattern

This pattern demonstrates reasoning by analogy, finding similarities between
different domains to transfer knowledge and solve problems through comparison.

Pattern Type: Advanced Reasoning
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any, Tuple
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json


# State definition
class AnalogicalReasoningState(TypedDict):
    """State for analogical reasoning workflow"""
    source_domain: str
    target_domain: str
    problem_description: str
    source_analysis: Dict[str, Any]
    target_analysis: Dict[str, Any]
    analogical_mappings: List[Dict[str, Any]]
    structural_similarities: List[Dict[str, Any]]
    transferred_knowledge: Dict[str, Any]
    analogical_solution: str
    validity_assessment: Dict[str, Any]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class DomainAnalyzer:
    """Analyze domains to extract structural features"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_domain(self, domain: str, context: str) -> Dict[str, Any]:
        """Analyze domain structure and features"""
        prompt = f"""Analyze this domain for analogical reasoning:

Domain: {domain}
Context: {context}

Extract the following structural elements:
1. Key entities and their roles
2. Relationships between entities
3. Processes or mechanisms
4. Principles or rules
5. Common patterns
6. Causal structures

Provide a detailed analysis in JSON format:
{{
    "domain_name": "{domain}",
    "entities": [
        {{"name": "entity1", "role": "description", "properties": ["prop1", "prop2"]}}
    ],
    "relationships": [
        {{"type": "relationship_type", "from": "entity1", "to": "entity2", "nature": "description"}}
    ],
    "processes": [
        {{"name": "process", "steps": ["step1", "step2"], "purpose": "description"}}
    ],
    "principles": [
        {{"principle": "statement", "applicability": "where it applies"}}
    ],
    "patterns": [
        {{"pattern": "pattern_name", "description": "what it describes"}}
    ]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing domains for structural features."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON from response
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except:
                pass
        
        # Fallback structure
        return {
            "domain_name": domain,
            "entities": [],
            "relationships": [],
            "processes": [],
            "principles": [],
            "patterns": []
        }


class AnalogicalMapper:
    """Map analogies between source and target domains"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def find_mappings(self, source_analysis: Dict[str, Any],
                     target_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find analogical mappings between domains"""
        mappings = []
        
        # Map entities
        entity_mappings = self._map_entities(
            source_analysis.get("entities", []),
            target_analysis.get("entities", [])
        )
        mappings.extend(entity_mappings)
        
        # Map relationships
        relationship_mappings = self._map_relationships(
            source_analysis.get("relationships", []),
            target_analysis.get("relationships", [])
        )
        mappings.extend(relationship_mappings)
        
        # Map processes
        process_mappings = self._map_processes(
            source_analysis.get("processes", []),
            target_analysis.get("processes", [])
        )
        mappings.extend(process_mappings)
        
        return mappings
    
    def _map_entities(self, source_entities: List[Dict], 
                     target_entities: List[Dict]) -> List[Dict[str, Any]]:
        """Map entities based on roles and properties"""
        mappings = []
        
        for s_entity in source_entities:
            for t_entity in target_entities:
                similarity = self._calculate_entity_similarity(s_entity, t_entity)
                
                if similarity > 0.5:  # Threshold for mapping
                    mappings.append({
                        "type": "entity",
                        "source": s_entity.get("name", "unknown"),
                        "target": t_entity.get("name", "unknown"),
                        "similarity": similarity,
                        "mapping_basis": f"Role similarity: {s_entity.get('role', 'N/A')} → {t_entity.get('role', 'N/A')}"
                    })
        
        return mappings
    
    def _map_relationships(self, source_rels: List[Dict],
                          target_rels: List[Dict]) -> List[Dict[str, Any]]:
        """Map relationships based on structure"""
        mappings = []
        
        for s_rel in source_rels:
            for t_rel in target_rels:
                if s_rel.get("type") == t_rel.get("type"):
                    mappings.append({
                        "type": "relationship",
                        "source": f"{s_rel.get('from', '')} -{s_rel.get('type', '')}- {s_rel.get('to', '')}",
                        "target": f"{t_rel.get('from', '')} -{t_rel.get('type', '')}- {t_rel.get('to', '')}",
                        "similarity": 0.8,
                        "mapping_basis": f"Structural relationship: {s_rel.get('type', 'N/A')}"
                    })
        
        return mappings
    
    def _map_processes(self, source_procs: List[Dict],
                      target_procs: List[Dict]) -> List[Dict[str, Any]]:
        """Map processes based on purpose"""
        mappings = []
        
        for s_proc in source_procs:
            for t_proc in target_procs:
                # Simple keyword matching for demonstration
                s_purpose = s_proc.get("purpose", "").lower()
                t_purpose = t_proc.get("purpose", "").lower()
                
                # Count common words
                s_words = set(s_purpose.split())
                t_words = set(t_purpose.split())
                common = s_words & t_words
                
                if len(common) > 2 or len(s_proc.get("steps", [])) == len(t_proc.get("steps", [])):
                    mappings.append({
                        "type": "process",
                        "source": s_proc.get("name", "unknown"),
                        "target": t_proc.get("name", "unknown"),
                        "similarity": min(0.9, len(common) / 5),
                        "mapping_basis": f"Process similarity: {len(s_proc.get('steps', []))} steps"
                    })
        
        return mappings
    
    def _calculate_entity_similarity(self, entity1: Dict, entity2: Dict) -> float:
        """Calculate similarity between entities"""
        role1 = entity1.get("role", "").lower()
        role2 = entity2.get("role", "").lower()
        
        # Simple word overlap similarity
        words1 = set(role1.split())
        words2 = set(role2.split())
        
        if not words1 or not words2:
            return 0.0
        
        overlap = words1 & words2
        union = words1 | words2
        
        return len(overlap) / len(union) if union else 0.0


class StructuralAnalyzer:
    """Analyze structural similarities between domains"""
    
    def find_structural_similarities(self, source_analysis: Dict[str, Any],
                                    target_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find deep structural similarities"""
        similarities = []
        
        # Analyze entity count similarity
        if source_analysis.get("entities") and target_analysis.get("entities"):
            similarities.append({
                "aspect": "entity_count",
                "source_value": len(source_analysis["entities"]),
                "target_value": len(target_analysis["entities"]),
                "similarity": "similar" if abs(len(source_analysis["entities"]) - len(target_analysis["entities"])) <= 2 else "different"
            })
        
        # Analyze relationship complexity
        if source_analysis.get("relationships") and target_analysis.get("relationships"):
            s_rel_types = set(r.get("type") for r in source_analysis["relationships"])
            t_rel_types = set(r.get("type") for r in target_analysis["relationships"])
            
            common_types = s_rel_types & t_rel_types
            
            similarities.append({
                "aspect": "relationship_types",
                "source_value": list(s_rel_types),
                "target_value": list(t_rel_types),
                "similarity": f"{len(common_types)} common relationship types"
            })
        
        # Analyze process complexity
        if source_analysis.get("processes") and target_analysis.get("processes"):
            avg_source_steps = sum(len(p.get("steps", [])) for p in source_analysis["processes"]) / len(source_analysis["processes"])
            avg_target_steps = sum(len(p.get("steps", [])) for p in target_analysis["processes"]) / len(target_analysis["processes"])
            
            similarities.append({
                "aspect": "process_complexity",
                "source_value": f"{avg_source_steps:.1f} avg steps",
                "target_value": f"{avg_target_steps:.1f} avg steps",
                "similarity": "similar" if abs(avg_source_steps - avg_target_steps) <= 2 else "different"
            })
        
        return similarities


class KnowledgeTransfer:
    """Transfer knowledge from source to target domain"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def transfer_knowledge(self, problem: str, source_domain: str,
                          target_domain: str, mappings: List[Dict[str, Any]],
                          source_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge through analogy"""
        # Build mapping context
        mapping_context = self._build_mapping_context(mappings)
        
        prompt = f"""Transfer knowledge using analogical reasoning:

Problem in Target Domain ({target_domain}):
{problem}

Source Domain ({source_domain}) Analysis:
Principles: {json.dumps(source_analysis.get('principles', []), indent=2)}
Patterns: {json.dumps(source_analysis.get('patterns', []), indent=2)}

Analogical Mappings:
{mapping_context}

Based on these analogies, what insights from {source_domain} can be transferred to solve the problem in {target_domain}?

Provide:
1. Key insights from source domain
2. How they map to target domain
3. Adapted solutions for target problem
4. Potential limitations of the analogy

Return as JSON:
{{
    "key_insights": ["insight1", "insight2"],
    "mappings_applied": ["mapping1", "mapping2"],
    "adapted_solutions": ["solution1", "solution2"],
    "limitations": ["limitation1", "limitation2"]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at analogical reasoning and knowledge transfer."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            try:
                return json.loads(json_str)
            except:
                pass
        
        return {
            "key_insights": [],
            "mappings_applied": [],
            "adapted_solutions": [],
            "limitations": []
        }
    
    def _build_mapping_context(self, mappings: List[Dict[str, Any]]) -> str:
        """Build readable mapping context"""
        context = ""
        
        for mapping in mappings[:10]:  # Limit to top 10
            context += f"- {mapping['type'].capitalize()}: {mapping['source']} → {mapping['target']} "
            context += f"(similarity: {mapping['similarity']:.2f})\n"
        
        return context if context else "No mappings found"


class AnalogyValidator:
    """Validate the quality and applicability of analogies"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def validate_analogy(self, source_domain: str, target_domain: str,
                        mappings: List[Dict[str, Any]],
                        transferred_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the analogy"""
        # Calculate mapping strength
        avg_similarity = sum(m.get("similarity", 0) for m in mappings) / len(mappings) if mappings else 0
        
        # Count different mapping types
        mapping_types = set(m.get("type") for m in mappings)
        
        # Assess coverage
        coverage = {
            "entity_mappings": sum(1 for m in mappings if m.get("type") == "entity"),
            "relationship_mappings": sum(1 for m in mappings if m.get("type") == "relationship"),
            "process_mappings": sum(1 for m in mappings if m.get("type") == "process")
        }
        
        # Overall validity score
        validity_score = (
            (avg_similarity * 0.4) +  # Mapping quality
            (min(len(mapping_types) / 3, 1.0) * 0.3) +  # Mapping diversity
            (min(len(mappings) / 10, 1.0) * 0.3)  # Mapping quantity
        )
        
        return {
            "validity_score": validity_score,
            "average_similarity": avg_similarity,
            "mapping_count": len(mappings),
            "mapping_types": list(mapping_types),
            "coverage": coverage,
            "assessment": self._assess_validity(validity_score),
            "limitations_count": len(transferred_knowledge.get("limitations", []))
        }
    
    def _assess_validity(self, score: float) -> str:
        """Assess validity based on score"""
        if score >= 0.8:
            return "Strong analogy - high confidence in transfer"
        elif score >= 0.6:
            return "Moderate analogy - reasonable confidence"
        elif score >= 0.4:
            return "Weak analogy - use with caution"
        else:
            return "Poor analogy - limited applicability"


# Agent functions
def initialize_reasoning(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Initialize analogical reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing analogical reasoning: {state['source_domain']} → {state['target_domain']}"
    ))
    state["analogical_mappings"] = []
    state["structural_similarities"] = []
    state["current_step"] = "initialized"
    return state


def analyze_source_domain(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Analyze source domain structure"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = DomainAnalyzer(llm)
    
    source_analysis = analyzer.analyze_domain(
        state["source_domain"],
        "Source domain for analogical reasoning"
    )
    
    state["source_analysis"] = source_analysis
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed source domain: {len(source_analysis.get('entities', []))} entities, "
                f"{len(source_analysis.get('relationships', []))} relationships"
    ))
    state["current_step"] = "source_analyzed"
    return state


def analyze_target_domain(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Analyze target domain structure"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = DomainAnalyzer(llm)
    
    target_analysis = analyzer.analyze_domain(
        state["target_domain"],
        state["problem_description"]
    )
    
    state["target_analysis"] = target_analysis
    
    state["messages"].append(HumanMessage(
        content=f"Analyzed target domain: {len(target_analysis.get('entities', []))} entities, "
                f"{len(target_analysis.get('relationships', []))} relationships"
    ))
    state["current_step"] = "target_analyzed"
    return state


def find_analogical_mappings(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Find mappings between domains"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    mapper = AnalogicalMapper(llm)
    
    mappings = mapper.find_mappings(
        state["source_analysis"],
        state["target_analysis"]
    )
    
    state["analogical_mappings"] = mappings
    
    state["messages"].append(HumanMessage(
        content=f"Found {len(mappings)} analogical mappings between domains"
    ))
    state["current_step"] = "mappings_found"
    return state


def analyze_structural_similarities(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Analyze structural similarities"""
    analyzer = StructuralAnalyzer()
    
    similarities = analyzer.find_structural_similarities(
        state["source_analysis"],
        state["target_analysis"]
    )
    
    state["structural_similarities"] = similarities
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(similarities)} structural similarities"
    ))
    state["current_step"] = "similarities_analyzed"
    return state


def transfer_knowledge(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Transfer knowledge through analogy"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.6)
    transferer = KnowledgeTransfer(llm)
    
    transferred = transferer.transfer_knowledge(
        state["problem_description"],
        state["source_domain"],
        state["target_domain"],
        state["analogical_mappings"],
        state["source_analysis"]
    )
    
    state["transferred_knowledge"] = transferred
    
    state["messages"].append(HumanMessage(
        content=f"Transferred {len(transferred.get('key_insights', []))} insights, "
                f"{len(transferred.get('adapted_solutions', []))} solutions"
    ))
    state["current_step"] = "knowledge_transferred"
    return state


def validate_analogy(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Validate the analogy"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    validator = AnalogyValidator(llm)
    
    validation = validator.validate_analogy(
        state["source_domain"],
        state["target_domain"],
        state["analogical_mappings"],
        state["transferred_knowledge"]
    )
    
    state["validity_assessment"] = validation
    
    state["messages"].append(HumanMessage(
        content=f"Analogy validity: {validation['validity_score']:.2f} - {validation['assessment']}"
    ))
    state["current_step"] = "analogy_validated"
    return state


def synthesize_solution(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Synthesize final analogical solution"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    # Build synthesis prompt
    insights = state["transferred_knowledge"].get("key_insights", [])
    solutions = state["transferred_knowledge"].get("adapted_solutions", [])
    limitations = state["transferred_knowledge"].get("limitations", [])
    
    prompt = f"""Synthesize an analogical solution:

Problem: {state['problem_description']}
Target Domain: {state['target_domain']}
Source Analogy: {state['source_domain']}

Transferred Insights:
{chr(10).join(f'- {insight}' for insight in insights)}

Adapted Solutions:
{chr(10).join(f'- {solution}' for solution in solutions)}

Limitations:
{chr(10).join(f'- {limitation}' for limitation in limitations)}

Validity: {state['validity_assessment']['assessment']}

Create a comprehensive solution that:
1. Applies the analogical insights
2. Adapts them to the target domain
3. Acknowledges limitations
4. Provides actionable recommendations"""
    
    messages = [
        SystemMessage(content="You are an expert at synthesizing analogical solutions."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    state["analogical_solution"] = response.content
    
    state["messages"].append(HumanMessage(
        content=f"Synthesized analogical solution ({len(response.content.split())} words)"
    ))
    state["current_step"] = "solution_synthesized"
    return state


def generate_report(state: AnalogicalReasoningState) -> AnalogicalReasoningState:
    """Generate final analogical reasoning report"""
    report = f"""
ANALOGICAL REASONING REPORT
===========================

Analogy: {state['source_domain']} → {state['target_domain']}

Problem: {state['problem_description']}

Domain Analysis:

Source Domain ({state['source_domain']}):
- Entities: {len(state['source_analysis'].get('entities', []))}
- Relationships: {len(state['source_analysis'].get('relationships', []))}
- Processes: {len(state['source_analysis'].get('processes', []))}
- Principles: {len(state['source_analysis'].get('principles', []))}

Target Domain ({state['target_domain']}):
- Entities: {len(state['target_analysis'].get('entities', []))}
- Relationships: {len(state['target_analysis'].get('relationships', []))}
- Processes: {len(state['target_analysis'].get('processes', []))}

Analogical Mappings ({len(state['analogical_mappings'])}):
"""
    
    # Show top 5 mappings
    for i, mapping in enumerate(state['analogical_mappings'][:5], 1):
        report += f"{i}. {mapping['type'].capitalize()}: {mapping['source']} → {mapping['target']} "
        report += f"(similarity: {mapping['similarity']:.2f})\n"
    
    if len(state['analogical_mappings']) > 5:
        report += f"... and {len(state['analogical_mappings']) - 5} more\n"
    
    report += f"""
Structural Similarities:
"""
    
    for similarity in state['structural_similarities']:
        report += f"- {similarity['aspect']}: {similarity['similarity']}\n"
    
    report += f"""
Knowledge Transfer:

Key Insights:
"""
    
    for insight in state['transferred_knowledge'].get('key_insights', []):
        report += f"• {insight}\n"
    
    report += f"""
Adapted Solutions:
"""
    
    for solution in state['transferred_knowledge'].get('adapted_solutions', []):
        report += f"• {solution}\n"
    
    report += f"""
Validity Assessment:
- Score: {state['validity_assessment']['validity_score']:.2f}
- Assessment: {state['validity_assessment']['assessment']}
- Mapping Quality: {state['validity_assessment']['average_similarity']:.2f}
- Coverage: {state['validity_assessment']['mapping_count']} mappings

Limitations:
"""
    
    for limitation in state['transferred_knowledge'].get('limitations', []):
        report += f"• {limitation}\n"
    
    report += f"""
ANALOGICAL SOLUTION:
{'-' * 50}
{state['analogical_solution']}
{'-' * 50}

Reasoning Summary:
- Analogy Strength: {state['validity_assessment']['validity_score']:.2%}
- Mappings Found: {len(state['analogical_mappings'])}
- Insights Transferred: {len(state['transferred_knowledge'].get('key_insights', []))}
- Solutions Generated: {len(state['transferred_knowledge'].get('adapted_solutions', []))}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_analogical_reasoning_graph():
    """Create the analogical reasoning workflow graph"""
    workflow = StateGraph(AnalogicalReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_reasoning)
    workflow.add_node("analyze_source", analyze_source_domain)
    workflow.add_node("analyze_target", analyze_target_domain)
    workflow.add_node("find_mappings", find_analogical_mappings)
    workflow.add_node("analyze_similarities", analyze_structural_similarities)
    workflow.add_node("transfer_knowledge", transfer_knowledge)
    workflow.add_node("validate", validate_analogy)
    workflow.add_node("synthesize", synthesize_solution)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "analyze_source")
    workflow.add_edge("analyze_source", "analyze_target")
    workflow.add_edge("analyze_target", "find_mappings")
    workflow.add_edge("find_mappings", "analyze_similarities")
    workflow.add_edge("analyze_similarities", "transfer_knowledge")
    workflow.add_edge("transfer_knowledge", "validate")
    workflow.add_edge("validate", "synthesize")
    workflow.add_edge("synthesize", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample analogical reasoning task
    initial_state = {
        "source_domain": "Immune System",
        "target_domain": "Cybersecurity",
        "problem_description": "How to design an adaptive defense system that learns from attacks and improves over time, similar to how the immune system adapts to pathogens",
        "source_analysis": {},
        "target_analysis": {},
        "analogical_mappings": [],
        "structural_similarities": [],
        "transferred_knowledge": {},
        "analogical_solution": "",
        "validity_assessment": {},
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_analogical_reasoning_graph()
    
    print("Analogical Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Analogy Strength: {result['validity_assessment']['validity_score']:.2%}")

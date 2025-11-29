"""
Spatial Reasoning MCP Pattern

This pattern demonstrates reasoning about space, locations, distances,
directions, and spatial relationships.

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
import math


# State definition
class SpatialReasoningState(TypedDict):
    """State for spatial reasoning workflow"""
    spatial_scenario: str
    spatial_question: str
    entities: List[Dict[str, Any]]
    spatial_relations: List[Dict[str, str]]
    spatial_map: Dict[str, Any]
    distance_calculations: List[Dict[str, Any]]
    path_analysis: Dict[str, Any]
    spatial_inferences: List[Dict[str, Any]]
    answer: Dict[str, Any]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class SpatialEntityExtractor:
    """Extract spatial entities and their locations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def extract_entities(self, scenario: str) -> List[Dict[str, Any]]:
        """Extract spatial entities"""
        prompt = f"""Extract spatial entities and their locations:

Scenario: {scenario}

Extract all entities (objects, people, places) with spatial information:
- What/who it is
- Where it is located
- Absolute or relative position
- Size/dimensions if mentioned

Return JSON array:
[
    {{
        "id": 1,
        "entity": "name/description",
        "location": "where it is",
        "location_type": "absolute|relative",
        "reference_point": "what it's relative to (if relative)",
        "dimensions": "size if mentioned"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at extracting spatial information."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class SpatialRelationIdentifier:
    """Identify spatial relations"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def identify_relations(self, entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Identify spatial relations between entities"""
        prompt = f"""Identify spatial relations between entities:

Entities:
{json.dumps(entities, indent=2)}

Identify relations like:
- above/below
- left/right
- near/far
- inside/outside
- north/south/east/west
- in front of/behind
- adjacent to

Return JSON array:
[
    {{
        "entity1_id": 1,
        "relation": "above|below|left|right|near|far|inside|outside|etc",
        "entity2_id": 2,
        "distance": "approximate distance if known",
        "confidence": 0.0-1.0
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying spatial relations."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class SpatialMapBuilder:
    """Build spatial map/representation"""
    
    def build_map(self, entities: List[Dict[str, Any]],
                  relations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build spatial map"""
        # Attempt to assign relative coordinates
        entity_positions = {}
        
        # Start with entities that have absolute positions
        for entity in entities:
            if entity.get("location_type") == "absolute":
                entity_positions[entity.get("id")] = {
                    "x": 0,  # Placeholder - would need parsing
                    "y": 0,
                    "z": 0,
                    "type": "absolute"
                }
        
        # Use relations to infer relative positions
        for rel in relations:
            entity1_id = rel.get("entity1_id")
            entity2_id = rel.get("entity2_id")
            relation = rel.get("relation", "")
            
            # Simple heuristic positioning
            if entity1_id in entity_positions and entity2_id not in entity_positions:
                base_pos = entity_positions[entity1_id]
                
                # Infer position based on relation
                new_pos = self._infer_position(base_pos, relation)
                entity_positions[entity2_id] = new_pos
        
        spatial_map = {
            "entity_positions": entity_positions,
            "total_entities": len(entities),
            "positioned_entities": len(entity_positions),
            "spatial_dimensions": self._infer_dimensions(relations),
            "bounds": self._calculate_bounds(entity_positions)
        }
        
        return spatial_map
    
    def _infer_position(self, base_pos: Dict[str, Any], relation: str) -> Dict[str, Any]:
        """Infer position from relation"""
        x, y, z = base_pos.get("x", 0), base_pos.get("y", 0), base_pos.get("z", 0)
        
        if "above" in relation:
            z += 1
        elif "below" in relation:
            z -= 1
        elif "left" in relation or "west" in relation:
            x -= 1
        elif "right" in relation or "east" in relation:
            x += 1
        elif "north" in relation:
            y += 1
        elif "south" in relation:
            y -= 1
        elif "near" in relation:
            x += 0.5  # Slight offset
        
        return {"x": x, "y": y, "z": z, "type": "inferred"}
    
    def _infer_dimensions(self, relations: List[Dict[str, str]]) -> str:
        """Infer if 2D or 3D"""
        has_vertical = any("above" in r.get("relation", "") or "below" in r.get("relation", "") 
                          for r in relations)
        
        return "3D" if has_vertical else "2D"
    
    def _calculate_bounds(self, positions: Dict) -> Dict[str, float]:
        """Calculate spatial bounds"""
        if not positions:
            return {"min_x": 0, "max_x": 0, "min_y": 0, "max_y": 0, "min_z": 0, "max_z": 0}
        
        xs = [pos.get("x", 0) for pos in positions.values()]
        ys = [pos.get("y", 0) for pos in positions.values()]
        zs = [pos.get("z", 0) for pos in positions.values()]
        
        return {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(ys),
            "max_y": max(ys),
            "min_z": min(zs),
            "max_z": max(zs)
        }


class DistanceCalculator:
    """Calculate distances"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def calculate_distances(self, entities: List[Dict[str, Any]],
                           spatial_map: Dict[str, Any],
                           question: str) -> List[Dict[str, Any]]:
        """Calculate relevant distances"""
        prompt = f"""Calculate distances between entities:

Question: {question}

Entities:
{json.dumps(entities, indent=2)}

Spatial Map:
{json.dumps(spatial_map, indent=2)}

Calculate:
1. Direct distances between relevant entities
2. Path distances if applicable
3. Relative distances

Return JSON array:
[
    {{
        "from_entity": 1,
        "to_entity": 2,
        "distance": "value with units",
        "distance_type": "direct|manhattan|path",
        "calculation_method": "how calculated"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at spatial distance calculation."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


class PathAnalyzer:
    """Analyze paths and routes"""
    
    def analyze_paths(self, entities: List[Dict[str, Any]],
                     relations: List[Dict[str, str]],
                     spatial_map: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze possible paths"""
        # Find connected entities (adjacency)
        adjacency = {}
        
        for rel in relations:
            entity1 = rel.get("entity1_id")
            entity2 = rel.get("entity2_id")
            
            # Consider certain relations as creating paths
            relation = rel.get("relation", "")
            if any(r in relation for r in ["adjacent", "near", "connected"]):
                if entity1 not in adjacency:
                    adjacency[entity1] = []
                if entity2 not in adjacency:
                    adjacency[entity2] = []
                
                adjacency[entity1].append(entity2)
                adjacency[entity2].append(entity1)
        
        return {
            "connectivity_graph": adjacency,
            "connected_pairs": len(adjacency),
            "isolated_entities": len(entities) - len(adjacency),
            "avg_connections": sum(len(v) for v in adjacency.values()) / len(adjacency) if adjacency else 0
        }


class SpatialInferenceEngine:
    """Make spatial inferences"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def make_inferences(self, entities: List[Dict[str, Any]],
                       relations: List[Dict[str, str]],
                       spatial_map: Dict[str, Any],
                       question: str) -> List[Dict[str, Any]]:
        """Make spatial inferences"""
        prompt = f"""Make spatial inferences to answer the question:

Question: {question}

Entities:
{json.dumps(entities, indent=2)}

Spatial Relations:
{json.dumps(relations, indent=2)}

Spatial Map:
{json.dumps(spatial_map, indent=2)}

Make inferences about:
1. Implicit spatial relationships
2. Transitive relations (if A is left of B, B is left of C, then A is left of C)
3. Spatial constraints
4. Impossibilities/conflicts

Return JSON array:
[
    {{
        "inference": "what can be inferred",
        "based_on": ["relation/entity ids"],
        "inference_type": "transitive|implicit|constraint|conflict",
        "confidence": 0.0-1.0
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at spatial inference."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return []


# Agent functions
def initialize_spatial_reasoning(state: SpatialReasoningState) -> SpatialReasoningState:
    """Initialize spatial reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing spatial reasoning: {state['spatial_question']}"
    ))
    state["entities"] = []
    state["spatial_relations"] = []
    state["current_step"] = "initialized"
    return state


def extract_entities(state: SpatialReasoningState) -> SpatialReasoningState:
    """Extract spatial entities"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    extractor = SpatialEntityExtractor(llm)
    
    entities = extractor.extract_entities(state["spatial_scenario"])
    
    state["entities"] = entities
    
    state["messages"].append(HumanMessage(
        content=f"Extracted {len(entities)} spatial entities"
    ))
    state["current_step"] = "entities_extracted"
    return state


def identify_relations(state: SpatialReasoningState) -> SpatialReasoningState:
    """Identify spatial relations"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    identifier = SpatialRelationIdentifier(llm)
    
    relations = identifier.identify_relations(state["entities"])
    
    state["spatial_relations"] = relations
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(relations)} spatial relations"
    ))
    state["current_step"] = "relations_identified"
    return state


def build_spatial_map(state: SpatialReasoningState) -> SpatialReasoningState:
    """Build spatial map"""
    builder = SpatialMapBuilder()
    
    spatial_map = builder.build_map(
        state["entities"],
        state["spatial_relations"]
    )
    
    state["spatial_map"] = spatial_map
    
    state["messages"].append(HumanMessage(
        content=f"Built spatial map: {spatial_map['spatial_dimensions']}, "
                f"{spatial_map['positioned_entities']}/{spatial_map['total_entities']} entities positioned"
    ))
    state["current_step"] = "map_built"
    return state


def calculate_distances(state: SpatialReasoningState) -> SpatialReasoningState:
    """Calculate distances"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    calculator = DistanceCalculator(llm)
    
    distances = calculator.calculate_distances(
        state["entities"],
        state["spatial_map"],
        state["spatial_question"]
    )
    
    state["distance_calculations"] = distances
    
    state["messages"].append(HumanMessage(
        content=f"Calculated {len(distances)} distances"
    ))
    state["current_step"] = "distances_calculated"
    return state


def analyze_paths(state: SpatialReasoningState) -> SpatialReasoningState:
    """Analyze paths"""
    analyzer = PathAnalyzer()
    
    path_analysis = analyzer.analyze_paths(
        state["entities"],
        state["spatial_relations"],
        state["spatial_map"]
    )
    
    state["path_analysis"] = path_analysis
    
    state["messages"].append(HumanMessage(
        content=f"Path analysis: {path_analysis['connected_pairs']} connected pairs"
    ))
    state["current_step"] = "paths_analyzed"
    return state


def make_inferences(state: SpatialReasoningState) -> SpatialReasoningState:
    """Make spatial inferences"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    engine = SpatialInferenceEngine(llm)
    
    inferences = engine.make_inferences(
        state["entities"],
        state["spatial_relations"],
        state["spatial_map"],
        state["spatial_question"]
    )
    
    state["spatial_inferences"] = inferences
    
    state["messages"].append(HumanMessage(
        content=f"Made {len(inferences)} spatial inferences"
    ))
    state["current_step"] = "inferences_made"
    return state


def synthesize_answer(state: SpatialReasoningState) -> SpatialReasoningState:
    """Synthesize spatial answer"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    prompt = f"""Answer this spatial question:

Question: {state['spatial_question']}

Spatial Map:
{json.dumps(state['spatial_map'], indent=2)}

Distance Calculations:
{json.dumps(state['distance_calculations'], indent=2)}

Spatial Inferences:
{json.dumps(state['spatial_inferences'], indent=2)}

Provide a clear answer based on spatial reasoning.

Return JSON:
{{
    "answer": "the answer",
    "spatial_reasoning": "explanation of spatial logic",
    "key_locations": ["critical locations in reasoning"],
    "confidence": 0.0-1.0
}}"""
    
    messages = [
        SystemMessage(content="You are an expert at spatial reasoning."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    # Extract JSON
    content = response.content
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    
    if json_start != -1 and json_end > json_start:
        try:
            answer = json.loads(content[json_start:json_end])
        except:
            answer = {
                "answer": "Unable to determine",
                "spatial_reasoning": "Parsing error",
                "key_locations": [],
                "confidence": 0.0
            }
    else:
        answer = {
            "answer": "Unable to determine",
            "spatial_reasoning": "No response",
            "key_locations": [],
            "confidence": 0.0
        }
    
    state["answer"] = answer
    
    state["messages"].append(HumanMessage(
        content=f"Answer: {answer.get('answer', 'Unknown')[:100]}..."
    ))
    state["current_step"] = "answer_synthesized"
    return state


def generate_report(state: SpatialReasoningState) -> SpatialReasoningState:
    """Generate final report"""
    report = f"""
SPATIAL REASONING REPORT
========================

Scenario: {state['spatial_scenario']}
Question: {state['spatial_question']}

Spatial Entities ({len(state['entities'])}):
"""
    
    for entity in state['entities']:
        report += f"\n{entity.get('id', 0)}. {entity.get('entity', 'Unknown')}\n"
        report += f"   Location: {entity.get('location', 'unknown')}\n"
        report += f"   Type: {entity.get('location_type', 'unknown')}\n"
    
    report += f"""
Spatial Relations ({len(state['spatial_relations'])}):
"""
    
    for rel in state['spatial_relations']:
        report += f"- Entity {rel.get('entity1_id', 0)} is {rel.get('relation', 'unknown')} Entity {rel.get('entity2_id', 0)}\n"
        if rel.get('distance'):
            report += f"  Distance: {rel['distance']}\n"
    
    report += f"""
Spatial Map:
- Dimensions: {state['spatial_map']['spatial_dimensions']}
- Positioned Entities: {state['spatial_map']['positioned_entities']}/{state['spatial_map']['total_entities']}
- Bounds: {state['spatial_map']['bounds']}

Distance Calculations ({len(state['distance_calculations'])}):
"""
    
    for dist in state['distance_calculations']:
        report += f"- Entity {dist.get('from_entity', 0)} to Entity {dist.get('to_entity', 0)}: {dist.get('distance', 'unknown')}\n"
        report += f"  Type: {dist.get('distance_type', 'unknown')}\n"
    
    report += f"""
Path Analysis:
- Connected Pairs: {state['path_analysis']['connected_pairs']}
- Isolated Entities: {state['path_analysis']['isolated_entities']}
- Avg Connections: {state['path_analysis']['avg_connections']:.2f}

Spatial Inferences ({len(state['spatial_inferences'])}):
"""
    
    for inf in state['spatial_inferences']:
        report += f"- {inf.get('inference', 'Unknown')} ({inf.get('inference_type', 'unknown')})\n"
        report += f"  Confidence: {inf.get('confidence', 0):.2f}\n"
    
    report += f"""
ANSWER:
{state['answer'].get('answer', 'No answer')}

Spatial Reasoning:
{state['answer'].get('spatial_reasoning', 'N/A')}

Key Locations:
{chr(10).join(f'- {loc}' for loc in state['answer'].get('key_locations', []))}

Confidence: {state['answer'].get('confidence', 0):.2f}

Reasoning Summary:
- Entities: {len(state['entities'])}
- Relations: {len(state['spatial_relations'])}
- Distances: {len(state['distance_calculations'])}
- Inferences: {len(state['spatial_inferences'])}
- Answer Confidence: {state['answer'].get('confidence', 0):.2%}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_spatial_reasoning_graph():
    """Create the spatial reasoning workflow graph"""
    workflow = StateGraph(SpatialReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_spatial_reasoning)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("identify_relations", identify_relations)
    workflow.add_node("build_map", build_spatial_map)
    workflow.add_node("calculate_distances", calculate_distances)
    workflow.add_node("analyze_paths", analyze_paths)
    workflow.add_node("make_inferences", make_inferences)
    workflow.add_node("synthesize_answer", synthesize_answer)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "extract_entities")
    workflow.add_edge("extract_entities", "identify_relations")
    workflow.add_edge("identify_relations", "build_map")
    workflow.add_edge("build_map", "calculate_distances")
    workflow.add_edge("calculate_distances", "analyze_paths")
    workflow.add_edge("analyze_paths", "make_inferences")
    workflow.add_edge("make_inferences", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample spatial reasoning task
    initial_state = {
        "spatial_scenario": "The library is north of the cafeteria. The gym is east of the library. The parking lot is south of the cafeteria. The main building is west of the cafeteria.",
        "spatial_question": "What is the shortest path from the gym to the parking lot?",
        "entities": [],
        "spatial_relations": [],
        "spatial_map": {},
        "distance_calculations": [],
        "path_analysis": {},
        "spatial_inferences": [],
        "answer": {},
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_spatial_reasoning_graph()
    
    print("Spatial Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Answer: {result['answer'].get('answer', 'Unknown')}")

"""
Temporal Reasoning MCP Pattern

This pattern demonstrates reasoning about time, sequences, durations,
temporal relationships, and time-based constraints.

Pattern Type: Advanced Reasoning
Category: Agentic MCP Pattern
Complexity: Advanced
"""

from typing import TypedDict, List, Dict, Annotated, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import operator
import json
from datetime import datetime, timedelta


# State definition
class TemporalReasoningState(TypedDict):
    """State for temporal reasoning workflow"""
    temporal_scenario: str
    temporal_question: str
    events: List[Dict[str, Any]]
    temporal_relations: List[Dict[str, str]]
    timeline: Dict[str, Any]
    duration_analysis: Dict[str, Any]
    temporal_constraints: List[Dict[str, str]]
    temporal_inferences: List[Dict[str, Any]]
    answer: Dict[str, Any]
    final_reasoning: str
    messages: Annotated[List, operator.add]
    current_step: str


class EventExtractor:
    """Extract events and their temporal information"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def extract_events(self, scenario: str) -> List[Dict[str, Any]]:
        """Extract temporal events from scenario"""
        prompt = f"""Extract temporal events from this scenario:

Scenario: {scenario}

Extract all events with their temporal information:
- What happened
- When it happened (absolute time or relative)
- Duration (if mentioned)
- Any temporal indicators

Return JSON array:
[
    {{
        "id": 1,
        "event": "description of what happened",
        "time_reference": "when it happened",
        "time_type": "absolute|relative|duration",
        "duration": "how long if applicable",
        "temporal_markers": ["before", "after", "during", etc]
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at extracting temporal information from text."),
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


class TemporalRelationIdentifier:
    """Identify temporal relations between events"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def identify_relations(self, events: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Identify temporal relations"""
        prompt = f"""Identify temporal relations between these events:

Events:
{json.dumps(events, indent=2)}

Identify relations like:
- before/after
- during/overlaps
- starts/finishes
- meets (one ends when another starts)
- simultaneous

Return JSON array:
[
    {{
        "event1_id": 1,
        "relation": "before|after|during|overlaps|starts|finishes|meets|simultaneous",
        "event2_id": 2,
        "confidence": 0.0-1.0,
        "evidence": "why we know this"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at identifying temporal relations."),
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


class TimelineBuilder:
    """Build temporal timeline"""
    
    def build_timeline(self, events: List[Dict[str, Any]],
                      relations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build timeline from events and relations"""
        # Sort events by temporal order
        ordered_events = self._order_events(events, relations)
        
        # Identify earliest and latest events
        timeline = {
            "ordered_events": ordered_events,
            "start_event": ordered_events[0] if ordered_events else None,
            "end_event": ordered_events[-1] if ordered_events else None,
            "event_count": len(ordered_events),
            "has_absolute_times": any(e.get("time_type") == "absolute" for e in events),
            "has_durations": any(e.get("duration") for e in events)
        }
        
        return timeline
    
    def _order_events(self, events: List[Dict[str, Any]],
                     relations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Order events temporally"""
        # Build adjacency based on "before" relations
        before_map = {}
        
        for rel in relations:
            if rel.get("relation") == "before":
                event1 = rel.get("event1_id")
                event2 = rel.get("event2_id")
                
                if event1 not in before_map:
                    before_map[event1] = []
                before_map[event1].append(event2)
        
        # Simple topological sort
        ordered = []
        visited = set()
        
        def visit(event_id):
            if event_id in visited:
                return
            visited.add(event_id)
            
            # Visit events that come after this one
            for next_id in before_map.get(event_id, []):
                visit(next_id)
            
            # Find event object
            event = next((e for e in events if e.get("id") == event_id), None)
            if event:
                ordered.insert(0, event)
        
        # Visit all events
        for event in events:
            visit(event.get("id"))
        
        # Add any events not in relations
        for event in events:
            if event not in ordered:
                ordered.append(event)
        
        return ordered


class DurationAnalyzer:
    """Analyze durations and time spans"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def analyze_durations(self, events: List[Dict[str, Any]],
                         timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze durations"""
        prompt = f"""Analyze temporal durations:

Events:
{json.dumps(events, indent=2)}

Timeline:
{json.dumps(timeline, indent=2)}

Analyze:
1. Individual event durations
2. Time spans between events
3. Total timeline duration
4. Overlapping periods

Return JSON:
{{
    "individual_durations": [{{"event_id": 1, "duration": "X hours", "estimated": true|false}}],
    "inter_event_gaps": [{{"from": 1, "to": 2, "gap": "X hours"}}],
    "total_span": "overall duration",
    "overlaps": [{{"events": [1, 2], "overlap_period": "duration"}}]
}}"""
        
        messages = [
            SystemMessage(content="You are an expert at analyzing temporal durations."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        # Extract JSON
        content = response.content
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            try:
                return json.loads(content[json_start:json_end])
            except:
                pass
        
        return {
            "individual_durations": [],
            "inter_event_gaps": [],
            "total_span": "unknown",
            "overlaps": []
        }


class TemporalConstraintIdentifier:
    """Identify temporal constraints"""
    
    def identify_constraints(self, events: List[Dict[str, Any]],
                            question: str) -> List[Dict[str, str]]:
        """Identify temporal constraints"""
        constraints = []
        
        # Look for deadline constraints
        for event in events:
            markers = event.get("temporal_markers", [])
            
            if any(m in ["deadline", "by", "before"] for m in markers):
                constraints.append({
                    "type": "deadline",
                    "event_id": event.get("id", 0),
                    "constraint": f"Must occur before specified time",
                    "criticality": "high"
                })
            
            if any(m in ["after", "following"] for m in markers):
                constraints.append({
                    "type": "precedence",
                    "event_id": event.get("id", 0),
                    "constraint": f"Must occur after another event",
                    "criticality": "medium"
                })
            
            if event.get("duration"):
                constraints.append({
                    "type": "duration",
                    "event_id": event.get("id", 0),
                    "constraint": f"Has duration: {event.get('duration')}",
                    "criticality": "low"
                })
        
        return constraints


class TemporalInferenceEngine:
    """Make temporal inferences"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def make_inferences(self, events: List[Dict[str, Any]],
                       relations: List[Dict[str, str]],
                       timeline: Dict[str, Any],
                       question: str) -> List[Dict[str, Any]]:
        """Make temporal inferences"""
        prompt = f"""Make temporal inferences to answer the question:

Question: {question}

Events:
{json.dumps(events, indent=2)}

Relations:
{json.dumps(relations, indent=2)}

Timeline:
{json.dumps(timeline, indent=2)}

Make inferences about:
1. Event ordering
2. Time calculations
3. Temporal impossibilities/conflicts
4. Derived temporal facts

Return JSON array:
[
    {{
        "inference": "what can be inferred",
        "based_on": ["event/relation ids"],
        "inference_type": "ordering|calculation|conflict|derivation",
        "confidence": 0.0-1.0
    }}
]"""
        
        messages = [
            SystemMessage(content="You are an expert at temporal inference."),
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
def initialize_temporal_reasoning(state: TemporalReasoningState) -> TemporalReasoningState:
    """Initialize temporal reasoning"""
    state["messages"].append(HumanMessage(
        content=f"Initializing temporal reasoning: {state['temporal_question']}"
    ))
    state["events"] = []
    state["temporal_relations"] = []
    state["current_step"] = "initialized"
    return state


def extract_events(state: TemporalReasoningState) -> TemporalReasoningState:
    """Extract temporal events"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    extractor = EventExtractor(llm)
    
    events = extractor.extract_events(state["temporal_scenario"])
    
    state["events"] = events
    
    state["messages"].append(HumanMessage(
        content=f"Extracted {len(events)} temporal events"
    ))
    state["current_step"] = "events_extracted"
    return state


def identify_relations(state: TemporalReasoningState) -> TemporalReasoningState:
    """Identify temporal relations"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    identifier = TemporalRelationIdentifier(llm)
    
    relations = identifier.identify_relations(state["events"])
    
    state["temporal_relations"] = relations
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(relations)} temporal relations"
    ))
    state["current_step"] = "relations_identified"
    return state


def build_timeline(state: TemporalReasoningState) -> TemporalReasoningState:
    """Build timeline"""
    builder = TimelineBuilder()
    
    timeline = builder.build_timeline(
        state["events"],
        state["temporal_relations"]
    )
    
    state["timeline"] = timeline
    
    state["messages"].append(HumanMessage(
        content=f"Built timeline with {timeline['event_count']} ordered events"
    ))
    state["current_step"] = "timeline_built"
    return state


def analyze_durations(state: TemporalReasoningState) -> TemporalReasoningState:
    """Analyze durations"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    analyzer = DurationAnalyzer(llm)
    
    duration_analysis = analyzer.analyze_durations(
        state["events"],
        state["timeline"]
    )
    
    state["duration_analysis"] = duration_analysis
    
    state["messages"].append(HumanMessage(
        content=f"Duration analysis: total span {duration_analysis.get('total_span', 'unknown')}"
    ))
    state["current_step"] = "durations_analyzed"
    return state


def identify_constraints(state: TemporalReasoningState) -> TemporalReasoningState:
    """Identify temporal constraints"""
    identifier = TemporalConstraintIdentifier()
    
    constraints = identifier.identify_constraints(
        state["events"],
        state["temporal_question"]
    )
    
    state["temporal_constraints"] = constraints
    
    state["messages"].append(HumanMessage(
        content=f"Identified {len(constraints)} temporal constraints"
    ))
    state["current_step"] = "constraints_identified"
    return state


def make_inferences(state: TemporalReasoningState) -> TemporalReasoningState:
    """Make temporal inferences"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    engine = TemporalInferenceEngine(llm)
    
    inferences = engine.make_inferences(
        state["events"],
        state["temporal_relations"],
        state["timeline"],
        state["temporal_question"]
    )
    
    state["temporal_inferences"] = inferences
    
    state["messages"].append(HumanMessage(
        content=f"Made {len(inferences)} temporal inferences"
    ))
    state["current_step"] = "inferences_made"
    return state


def synthesize_answer(state: TemporalReasoningState) -> TemporalReasoningState:
    """Synthesize temporal answer"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    
    prompt = f"""Answer this temporal question:

Question: {state['temporal_question']}

Timeline:
{json.dumps(state['timeline'], indent=2)}

Duration Analysis:
{json.dumps(state['duration_analysis'], indent=2)}

Temporal Inferences:
{json.dumps(state['temporal_inferences'], indent=2)}

Provide a clear answer based on temporal reasoning.

Return JSON:
{{
    "answer": "the answer",
    "temporal_reasoning": "explanation of temporal logic",
    "key_time_points": ["critical time points in reasoning"],
    "confidence": 0.0-1.0
}}"""
    
    messages = [
        SystemMessage(content="You are an expert at temporal reasoning."),
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
                "temporal_reasoning": "Parsing error",
                "key_time_points": [],
                "confidence": 0.0
            }
    else:
        answer = {
            "answer": "Unable to determine",
            "temporal_reasoning": "No response",
            "key_time_points": [],
            "confidence": 0.0
        }
    
    state["answer"] = answer
    
    state["messages"].append(HumanMessage(
        content=f"Answer: {answer.get('answer', 'Unknown')[:100]}..."
    ))
    state["current_step"] = "answer_synthesized"
    return state


def generate_report(state: TemporalReasoningState) -> TemporalReasoningState:
    """Generate final report"""
    report = f"""
TEMPORAL REASONING REPORT
=========================

Scenario: {state['temporal_scenario']}
Question: {state['temporal_question']}

Temporal Events ({len(state['events'])}):
"""
    
    for event in state['events']:
        report += f"\n{event.get('id', 0)}. {event.get('event', 'Unknown')}\n"
        report += f"   Time: {event.get('time_reference', 'unknown')}\n"
        report += f"   Type: {event.get('time_type', 'unknown')}\n"
        if event.get('duration'):
            report += f"   Duration: {event['duration']}\n"
    
    report += f"""
Temporal Relations ({len(state['temporal_relations'])}):
"""
    
    for rel in state['temporal_relations']:
        report += f"- Event {rel.get('event1_id', 0)} {rel.get('relation', 'unknown')} Event {rel.get('event2_id', 0)}\n"
        report += f"  Evidence: {rel.get('evidence', 'N/A')}\n"
    
    report += f"""
Timeline:
- Event Count: {state['timeline']['event_count']}
- Start Event: {state['timeline']['start_event'].get('event', 'Unknown') if state['timeline']['start_event'] else 'Unknown'}
- End Event: {state['timeline']['end_event'].get('event', 'Unknown') if state['timeline']['end_event'] else 'Unknown'}
- Has Absolute Times: {state['timeline']['has_absolute_times']}
- Has Durations: {state['timeline']['has_durations']}

Ordered Events:
"""
    
    for i, event in enumerate(state['timeline']['ordered_events'], 1):
        report += f"{i}. {event.get('event', 'Unknown')}\n"
    
    report += f"""
Duration Analysis:
- Total Span: {state['duration_analysis'].get('total_span', 'unknown')}
- Individual Durations: {len(state['duration_analysis'].get('individual_durations', []))}
- Inter-Event Gaps: {len(state['duration_analysis'].get('inter_event_gaps', []))}
- Overlaps: {len(state['duration_analysis'].get('overlaps', []))}

Temporal Constraints ({len(state['temporal_constraints'])}):
"""
    
    for constraint in state['temporal_constraints']:
        report += f"- {constraint.get('type', 'unknown')}: {constraint.get('constraint', 'Unknown')}\n"
        report += f"  Criticality: {constraint.get('criticality', 'unknown')}\n"
    
    report += f"""
Temporal Inferences ({len(state['temporal_inferences'])}):
"""
    
    for inf in state['temporal_inferences']:
        report += f"- {inf.get('inference', 'Unknown')} ({inf.get('inference_type', 'unknown')})\n"
        report += f"  Confidence: {inf.get('confidence', 0):.2f}\n"
    
    report += f"""
ANSWER:
{state['answer'].get('answer', 'No answer')}

Temporal Reasoning:
{state['answer'].get('temporal_reasoning', 'N/A')}

Key Time Points:
{chr(10).join(f'- {tp}' for tp in state['answer'].get('key_time_points', []))}

Confidence: {state['answer'].get('confidence', 0):.2f}

Reasoning Summary:
- Events: {len(state['events'])}
- Relations: {len(state['temporal_relations'])}
- Constraints: {len(state['temporal_constraints'])}
- Inferences: {len(state['temporal_inferences'])}
- Answer Confidence: {state['answer'].get('confidence', 0):.2%}
"""
    
    state["final_reasoning"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_temporal_reasoning_graph():
    """Create the temporal reasoning workflow graph"""
    workflow = StateGraph(TemporalReasoningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_temporal_reasoning)
    workflow.add_node("extract_events", extract_events)
    workflow.add_node("identify_relations", identify_relations)
    workflow.add_node("build_timeline", build_timeline)
    workflow.add_node("analyze_durations", analyze_durations)
    workflow.add_node("identify_constraints", identify_constraints)
    workflow.add_node("make_inferences", make_inferences)
    workflow.add_node("synthesize_answer", synthesize_answer)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "extract_events")
    workflow.add_edge("extract_events", "identify_relations")
    workflow.add_edge("identify_relations", "build_timeline")
    workflow.add_edge("build_timeline", "analyze_durations")
    workflow.add_edge("analyze_durations", "identify_constraints")
    workflow.add_edge("identify_constraints", "make_inferences")
    workflow.add_edge("make_inferences", "synthesize_answer")
    workflow.add_edge("synthesize_answer", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample temporal reasoning task
    initial_state = {
        "temporal_scenario": "The meeting started at 2 PM and lasted 90 minutes. John arrived 15 minutes late. The presentation began after everyone arrived and took 45 minutes. There was a 10-minute break in the middle of the meeting.",
        "temporal_question": "What time did the presentation end?",
        "events": [],
        "temporal_relations": [],
        "timeline": {},
        "duration_analysis": {},
        "temporal_constraints": [],
        "temporal_inferences": [],
        "answer": {},
        "final_reasoning": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run the graph
    app = create_temporal_reasoning_graph()
    
    print("Temporal Reasoning MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content}")
    
    print(f"\nFinal Status: {result['current_step']}")
    print(f"Answer: {result['answer'].get('answer', 'Unknown')}")

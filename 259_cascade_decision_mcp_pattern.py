"""
Pattern 259: Cascade Decision MCP Pattern

This pattern demonstrates cascade decision making - sequential multi-stage
decision process where each stage filters or refines options.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class CascadeDecisionState(TypedDict):
    """State for cascade decision workflow"""
    messages: Annotated[List[str], add]
    initial_candidates: List[Dict[str, Any]]
    stage_results: List[Dict[str, Any]]
    final_candidates: List[Dict[str, Any]]
    decision: str


class CascadeStage:
    """Represents one stage in the cascade"""
    
    def __init__(self, name: str, criteria: str, threshold: float):
        self.name = name
        self.criteria = criteria
        self.threshold = threshold
    
    def filter(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates based on stage criteria"""
        passed = []
        failed = []
        
        for candidate in candidates:
            score = candidate.get(self.criteria, 0)
            if score >= self.threshold:
                passed.append(candidate)
            else:
                failed.append(candidate)
        
        return passed, failed


class CascadeDecisionMaker:
    """Makes decisions through cascading stages"""
    
    def __init__(self):
        self.stages = [
            CascadeStage("Initial Screening", "basic_requirements", 0.70),
            CascadeStage("Technical Assessment", "technical_score", 0.75),
            CascadeStage("Cultural Fit", "culture_score", 0.80),
            CascadeStage("Leadership Evaluation", "leadership_score", 0.70),
            CascadeStage("Final Review", "overall_score", 0.85)
        ]
    
    def process_cascade(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process candidates through all cascade stages"""
        results = []
        current_candidates = candidates.copy()
        
        for stage in self.stages:
            passed, failed = stage.filter(current_candidates)
            
            stage_result = {
                "stage": stage.name,
                "criteria": stage.criteria,
                "threshold": stage.threshold,
                "input_count": len(current_candidates),
                "passed_count": len(passed),
                "failed_count": len(failed),
                "passed": [c["name"] for c in passed],
                "failed": [c["name"] for c in failed],
                "pass_rate": len(passed) / len(current_candidates) if current_candidates else 0
            }
            results.append(stage_result)
            
            # Cascade: passed candidates become input for next stage
            current_candidates = passed
            
            # Stop if no candidates remain
            if not current_candidates:
                break
        
        return results, current_candidates


def initialize_candidates_agent(state: CascadeDecisionState) -> CascadeDecisionState:
    """Initialize candidates for cascade"""
    print("\n👥 Initializing Candidates...")
    
    candidates = [
        {
            "name": "Candidate A",
            "basic_requirements": 0.85,
            "technical_score": 0.88,
            "culture_score": 0.82,
            "leadership_score": 0.75,
            "overall_score": 0.87
        },
        {
            "name": "Candidate B",
            "basic_requirements": 0.92,
            "technical_score": 0.65,  # Will fail technical
            "culture_score": 0.88,
            "leadership_score": 0.82,
            "overall_score": 0.78
        },
        {
            "name": "Candidate C",
            "basic_requirements": 0.78,
            "technical_score": 0.92,
            "culture_score": 0.85,
            "leadership_score": 0.88,
            "overall_score": 0.90
        },
        {
            "name": "Candidate D",
            "basic_requirements": 0.65,  # Will fail initial
            "technical_score": 0.85,
            "culture_score": 0.80,
            "leadership_score": 0.78,
            "overall_score": 0.75
        },
        {
            "name": "Candidate E",
            "basic_requirements": 0.88,
            "technical_score": 0.85,
            "culture_score": 0.75,  # Will fail culture
            "leadership_score": 0.82,
            "overall_score": 0.82
        },
        {
            "name": "Candidate F",
            "basic_requirements": 0.90,
            "technical_score": 0.95,
            "culture_score": 0.92,
            "leadership_score": 0.85,
            "overall_score": 0.95
        }
    ]
    
    print(f"\n  Total Candidates: {len(candidates)}")
    for candidate in candidates:
        print(f"    • {candidate['name']}")
    
    return {
        **state,
        "initial_candidates": candidates,
        "messages": [f"✓ Initialized {len(candidates)} candidates"]
    }


def process_cascade_agent(state: CascadeDecisionState) -> CascadeDecisionState:
    """Process candidates through cascade stages"""
    print("\n🔄 Processing Cascade Stages...")
    
    decision_maker = CascadeDecisionMaker()
    stage_results, final_candidates = decision_maker.process_cascade(state["initial_candidates"])
    
    print(f"\n  Cascade Stages:")
    for result in stage_results:
        print(f"\n  Stage: {result['stage']}")
        print(f"    Criteria: {result['criteria']} >= {result['threshold']}")
        print(f"    Input: {result['input_count']} candidates")
        print(f"    Passed: {result['passed_count']} ({result['pass_rate']:.0%})")
        print(f"    Failed: {result['failed_count']}")
    
    print(f"\n  Final Candidates: {len(final_candidates)}")
    for candidate in final_candidates:
        print(f"    • {candidate['name']}")
    
    return {
        **state,
        "stage_results": stage_results,
        "final_candidates": final_candidates,
        "messages": [f"✓ Cascade complete: {len(final_candidates)} candidates passed"]
    }


def generate_cascade_report_agent(state: CascadeDecisionState) -> CascadeDecisionState:
    """Generate cascade decision report"""
    print("\n" + "="*70)
    print("CASCADE DECISION REPORT")
    print("="*70)
    
    print(f"\n📊 Initial Pool:")
    print(f"  Total Candidates: {len(state['initial_candidates'])}")
    for candidate in state["initial_candidates"]:
        print(f"    • {candidate['name']}")
    
    print(f"\n🔄 Cascade Process:")
    for i, result in enumerate(state["stage_results"], 1):
        print(f"\n  Stage {i}: {result['stage']}")
        print(f"    Criteria: {result['criteria']} >= {result['threshold']}")
        print(f"    Input: {result['input_count']} candidates")
        print(f"    ✅ Passed: {result['passed_count']} ({result['pass_rate']:.0%})")
        if result['passed']:
            for name in result['passed']:
                print(f"       • {name}")
        print(f"    ❌ Failed: {result['failed_count']}")
        if result['failed']:
            for name in result['failed']:
                print(f"       • {name}")
    
    print(f"\n✅ Final Results:")
    print(f"  Candidates Passed All Stages: {len(state['final_candidates'])}")
    if state["final_candidates"]:
        for candidate in state["final_candidates"]:
            print(f"\n    • {candidate['name']}")
            print(f"      Basic Requirements: {candidate['basic_requirements']:.0%}")
            print(f"      Technical Score: {candidate['technical_score']:.0%}")
            print(f"      Culture Score: {candidate['culture_score']:.0%}")
            print(f"      Leadership Score: {candidate['leadership_score']:.0%}")
            print(f"      Overall Score: {candidate['overall_score']:.0%}")
    else:
        print("    No candidates passed all stages")
    
    # Funnel analysis
    print(f"\n📉 Cascade Funnel Analysis:")
    initial_count = len(state["initial_candidates"])
    print(f"  Initial: {initial_count} (100%)")
    
    remaining = initial_count
    for result in state["stage_results"]:
        remaining = result["passed_count"]
        retention = (remaining / initial_count * 100) if initial_count > 0 else 0
        print(f"  After {result['stage']}: {remaining} ({retention:.0%})")
    
    print("\n💡 Cascade Decision Benefits:")
    print("  • Progressive filtering")
    print("  • Resource efficiency")
    print("  • Early elimination of weak candidates")
    print("  • Multi-dimensional evaluation")
    print("  • Transparent process")
    print("  • Quality assurance")
    
    print("\n="*70)
    print("✅ Cascade Decision Complete!")
    print("="*70)
    
    decision = f"Selected {len(state['final_candidates'])} candidate(s)"
    
    return {**state, "decision": decision, "messages": ["✓ Report generated"]}


def create_cascade_decision_graph():
    workflow = StateGraph(CascadeDecisionState)
    workflow.add_node("initialize", initialize_candidates_agent)
    workflow.add_node("process", process_cascade_agent)
    workflow.add_node("report", generate_cascade_report_agent)
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "process")
    workflow.add_edge("process", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 259: Cascade Decision MCP Pattern")
    print("="*70)
    
    app = create_cascade_decision_graph()
    final_state = app.invoke({
        "messages": [],
        "initial_candidates": [],
        "stage_results": [],
        "final_candidates": [],
        "decision": ""
    })
    print("\n✅ Cascade Decision Pattern Complete!")


if __name__ == "__main__":
    main()

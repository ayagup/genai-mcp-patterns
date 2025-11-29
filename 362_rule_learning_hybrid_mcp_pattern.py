"""
Rule-Learning Hybrid MCP Pattern

This pattern demonstrates combining predefined rules with learned patterns
from data/experience for adaptive reasoning.

Pattern Type: Hybrid
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
class RuleLearningState(TypedDict):
    """State for rule-learning hybrid"""
    task: str
    domain: str
    predefined_rules: List[Dict[str, Any]]
    training_examples: List[Dict[str, Any]]
    learned_patterns: List[Dict[str, Any]]
    rule_predictions: List[Dict[str, Any]]
    learned_predictions: List[Dict[str, Any]]
    hybrid_decision: Dict[str, Any]
    performance_metrics: Dict[str, float]
    adaptation_log: List[str]
    final_output: str
    messages: Annotated[List, operator.add]
    current_step: str


class RuleBasedSystem:
    """Traditional rule-based system"""
    
    def __init__(self):
        self.rules = self._load_predefined_rules()
    
    def _load_predefined_rules(self) -> List[Dict[str, Any]]:
        """Load predefined expert rules"""
        return [
            {
                "rule_id": "R1",
                "condition": "age < 18",
                "action": "classify as minor",
                "priority": 1,
                "certainty": 1.0
            },
            {
                "rule_id": "R2",
                "condition": "age >= 18 AND age < 65",
                "action": "classify as adult",
                "priority": 1,
                "certainty": 1.0
            },
            {
                "rule_id": "R3",
                "condition": "score > 90",
                "action": "assign grade A",
                "priority": 2,
                "certainty": 1.0
            },
            {
                "rule_id": "R4",
                "condition": "temperature > 100",
                "action": "alert high temperature",
                "priority": 3,
                "certainty": 0.95
            },
            {
                "rule_id": "R5",
                "condition": "contains 'urgent' OR contains 'emergency'",
                "action": "prioritize high",
                "priority": 1,
                "certainty": 0.9
            }
        ]
    
    def apply_rules(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply rules to input"""
        predictions = []
        
        for rule in self.rules:
            # Simplified rule matching (in production, use proper rule engine)
            if self._evaluate_condition(rule["condition"], input_data):
                predictions.append({
                    "rule_id": rule["rule_id"],
                    "action": rule["action"],
                    "certainty": rule["certainty"],
                    "source": "rule-based"
                })
        
        return predictions
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Simple condition evaluation"""
        # Simplified evaluation for demonstration
        condition_lower = condition.lower()
        
        # Check for age conditions
        if "age" in condition_lower:
            age = data.get("age", 0)
            if "<" in condition and "18" in condition:
                return age < 18
            elif ">=" in condition and "18" in condition:
                return age >= 18
        
        # Check for score conditions
        if "score" in condition_lower:
            score = data.get("score", 0)
            if ">" in condition and "90" in condition:
                return score > 90
        
        # Check for temperature
        if "temperature" in condition_lower:
            temp = data.get("temperature", 0)
            if ">" in condition and "100" in condition:
                return temp > 100
        
        # Check for text content
        if "contains" in condition_lower:
            text = data.get("text", "").lower()
            if "urgent" in condition_lower:
                return "urgent" in text or "emergency" in text
        
        return False


class PatternLearner:
    """Learn patterns from examples"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def learn_from_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Learn patterns from training examples"""
        prompt = f"""Learn patterns from these examples:

Training Examples:
{json.dumps(examples, indent=2)}

Identify:
1. Common patterns in the data
2. Relationships between inputs and outputs
3. Exceptions to general rules
4. Confidence in each pattern

Return JSON array:
[
    {{
        "pattern_id": "P1",
        "description": "pattern description",
        "conditions": ["when this pattern applies"],
        "prediction": "what it predicts",
        "confidence": 0.0-1.0,
        "support": "number of examples supporting this"
    }}
]"""
        
        messages = [
            SystemMessage(content="You are a pattern learning system."),
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
    
    def make_prediction(self, patterns: List[Dict[str, Any]], 
                       input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Make prediction based on learned patterns"""
        predictions = []
        
        for pattern in patterns:
            # Simple matching (in production, use ML model)
            conditions = pattern.get("conditions", [])
            
            # Check if any condition matches
            match_score = 0.5  # Simplified
            
            predictions.append({
                "pattern_id": pattern["pattern_id"],
                "prediction": pattern["prediction"],
                "confidence": pattern["confidence"] * match_score,
                "source": "learned"
            })
        
        return predictions


class HybridDecisionMaker:
    """Combine rule-based and learned predictions"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def combine_predictions(self, rule_predictions: List[Dict[str, Any]],
                          learned_predictions: List[Dict[str, Any]],
                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine rule and learned predictions"""
        prompt = f"""Combine rule-based and learned predictions:

Input: {json.dumps(input_data, indent=2)}

Rule-Based Predictions:
{json.dumps(rule_predictions, indent=2)}

Learned Predictions:
{json.dumps(learned_predictions, indent=2)}

Decide:
1. Which predictions to trust more
2. How to resolve conflicts
3. Final decision with reasoning

Return JSON:
{{
    "decision": "final decision",
    "primary_source": "rules|learned|hybrid",
    "rule_weight": 0.0-1.0,
    "learned_weight": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "why this decision",
    "conflicts_resolved": ["any conflicts and resolutions"]
}}"""
        
        messages = [
            SystemMessage(content="You are a hybrid decision maker."),
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
            "decision": "Unable to decide",
            "primary_source": "unknown",
            "rule_weight": 0.5,
            "learned_weight": 0.5,
            "confidence": 0.0,
            "reasoning": "Parsing error",
            "conflicts_resolved": []
        }


class AdaptiveSystem:
    """Adapt rules based on learning"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def adapt_rules(self, existing_rules: List[Dict[str, Any]],
                   learned_patterns: List[Dict[str, Any]],
                   performance: Dict[str, float]) -> List[str]:
        """Suggest rule adaptations"""
        prompt = f"""Suggest how to adapt rules based on learned patterns:

Existing Rules:
{json.dumps(existing_rules, indent=2)}

Learned Patterns:
{json.dumps(learned_patterns, indent=2)}

Performance Metrics:
{json.dumps(performance, indent=2)}

Suggest:
1. New rules to add
2. Rules to modify
3. Rules to deprecate
4. Weight adjustments

Return JSON array of adaptation suggestions:
[
    "suggestion 1",
    "suggestion 2"
]"""
        
        messages = [
            SystemMessage(content="You are a rule adaptation engine."),
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
        
        return ["No adaptations suggested"]


# Agent functions
def initialize_hybrid_system(state: RuleLearningState) -> RuleLearningState:
    """Initialize rule-learning hybrid"""
    state["messages"].append(HumanMessage(
        content=f"Initializing rule-learning hybrid for: {state['task']}"
    ))
    state["adaptation_log"] = []
    state["current_step"] = "initialized"
    return state


def load_rules(state: RuleLearningState) -> RuleLearningState:
    """Load predefined rules"""
    system = RuleBasedSystem()
    
    state["predefined_rules"] = system.rules
    
    state["messages"].append(HumanMessage(
        content=f"Loaded {len(system.rules)} predefined rules"
    ))
    state["current_step"] = "rules_loaded"
    return state


def learn_patterns(state: RuleLearningState) -> RuleLearningState:
    """Learn patterns from examples"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    learner = PatternLearner(llm)
    
    patterns = learner.learn_from_examples(state["training_examples"])
    
    state["learned_patterns"] = patterns
    
    state["messages"].append(HumanMessage(
        content=f"Learned {len(patterns)} patterns from {len(state['training_examples'])} examples"
    ))
    state["current_step"] = "patterns_learned"
    return state


def apply_rules_agent(state: RuleLearningState) -> RuleLearningState:
    """Apply rule-based predictions"""
    system = RuleBasedSystem()
    
    # Simulate input (would come from task in production)
    input_data = {
        "age": 25,
        "score": 95,
        "temperature": 98.6,
        "text": "This is an urgent request"
    }
    
    predictions = system.apply_rules(input_data)
    
    state["rule_predictions"] = predictions
    
    state["messages"].append(HumanMessage(
        content=f"Rule-based system made {len(predictions)} predictions"
    ))
    state["current_step"] = "rules_applied"
    return state


def apply_learned_patterns(state: RuleLearningState) -> RuleLearningState:
    """Apply learned pattern predictions"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    learner = PatternLearner(llm)
    
    # Simulate input
    input_data = {
        "age": 25,
        "score": 95,
        "temperature": 98.6,
        "text": "This is an urgent request"
    }
    
    predictions = learner.make_prediction(state["learned_patterns"], input_data)
    
    state["learned_predictions"] = predictions
    
    state["messages"].append(HumanMessage(
        content=f"Learned patterns made {len(predictions)} predictions"
    ))
    state["current_step"] = "learned_applied"
    return state


def make_hybrid_decision(state: RuleLearningState) -> RuleLearningState:
    """Make hybrid decision"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.4)
    decision_maker = HybridDecisionMaker(llm)
    
    input_data = {"age": 25, "score": 95, "text": "urgent"}
    
    decision = decision_maker.combine_predictions(
        state["rule_predictions"],
        state["learned_predictions"],
        input_data
    )
    
    state["hybrid_decision"] = decision
    
    state["messages"].append(HumanMessage(
        content=f"Hybrid decision: {decision.get('decision', 'Unknown')}, "
                f"source: {decision.get('primary_source', 'unknown')}"
    ))
    state["current_step"] = "decision_made"
    return state


def evaluate_and_adapt(state: RuleLearningState) -> RuleLearningState:
    """Evaluate performance and adapt"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.5)
    adaptive_system = AdaptiveSystem(llm)
    
    # Simulate performance metrics
    performance = {
        "rule_accuracy": 0.85,
        "learned_accuracy": 0.78,
        "hybrid_accuracy": 0.92
    }
    
    state["performance_metrics"] = performance
    
    # Get adaptation suggestions
    adaptations = adaptive_system.adapt_rules(
        state["predefined_rules"],
        state["learned_patterns"],
        performance
    )
    
    state["adaptation_log"] = adaptations
    
    state["messages"].append(HumanMessage(
        content=f"Performance: Hybrid {performance['hybrid_accuracy']:.2%}, "
                f"{len(adaptations)} adaptations suggested"
    ))
    state["current_step"] = "adapted"
    return state


def generate_report(state: RuleLearningState) -> RuleLearningState:
    """Generate final report"""
    report = f"""
RULE-LEARNING HYBRID REPORT
============================

Task: {state['task']}
Domain: {state['domain']}

PREDEFINED RULES ({len(state['predefined_rules'])}):
"""
    
    for rule in state['predefined_rules']:
        report += f"\n{rule['rule_id']}: {rule['condition']} → {rule['action']}\n"
        report += f"  Priority: {rule['priority']}, Certainty: {rule['certainty']:.2f}\n"
    
    report += f"""
LEARNED PATTERNS ({len(state['learned_patterns'])}):
"""
    
    for pattern in state['learned_patterns']:
        report += f"\n{pattern.get('pattern_id', 'P?')}: {pattern.get('description', 'N/A')}\n"
        report += f"  Prediction: {pattern.get('prediction', 'N/A')}\n"
        report += f"  Confidence: {pattern.get('confidence', 0):.2f}\n"
    
    report += f"""
RULE-BASED PREDICTIONS ({len(state['rule_predictions'])}):
"""
    
    for pred in state['rule_predictions']:
        report += f"- {pred.get('action', 'N/A')} (certainty: {pred.get('certainty', 0):.2f})\n"
    
    report += f"""
LEARNED PREDICTIONS ({len(state['learned_predictions'])}):
"""
    
    for pred in state['learned_predictions']:
        report += f"- {pred.get('prediction', 'N/A')} (confidence: {pred.get('confidence', 0):.2f})\n"
    
    report += f"""
HYBRID DECISION:
----------------
Decision: {state['hybrid_decision'].get('decision', 'N/A')}
Primary Source: {state['hybrid_decision'].get('primary_source', 'unknown')}
Rule Weight: {state['hybrid_decision'].get('rule_weight', 0):.2f}
Learned Weight: {state['hybrid_decision'].get('learned_weight', 0):.2f}
Confidence: {state['hybrid_decision'].get('confidence', 0):.2%}

Reasoning: {state['hybrid_decision'].get('reasoning', 'N/A')}

PERFORMANCE METRICS:
--------------------
Rule Accuracy: {state['performance_metrics'].get('rule_accuracy', 0):.2%}
Learned Accuracy: {state['performance_metrics'].get('learned_accuracy', 0):.2%}
Hybrid Accuracy: {state['performance_metrics'].get('hybrid_accuracy', 0):.2%}

ADAPTATION SUGGESTIONS:
-----------------------
{chr(10).join(f'- {a}' for a in state['adaptation_log'])}

Summary:
This hybrid approach combines the reliability of expert rules with
the flexibility of learned patterns for optimal performance.
"""
    
    state["final_output"] = report
    state["messages"].append(HumanMessage(content=report))
    state["current_step"] = "completed"
    return state


# Build the graph
def create_rule_learning_graph():
    """Create rule-learning hybrid workflow"""
    workflow = StateGraph(RuleLearningState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_hybrid_system)
    workflow.add_node("load_rules", load_rules)
    workflow.add_node("learn_patterns", learn_patterns)
    workflow.add_node("apply_rules", apply_rules_agent)
    workflow.add_node("apply_learned", apply_learned_patterns)
    workflow.add_node("decide", make_hybrid_decision)
    workflow.add_node("adapt", evaluate_and_adapt)
    workflow.add_node("report", generate_report)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "load_rules")
    workflow.add_edge("load_rules", "learn_patterns")
    workflow.add_edge("learn_patterns", "apply_rules")
    workflow.add_edge("apply_rules", "apply_learned")
    workflow.add_edge("apply_learned", "decide")
    workflow.add_edge("decide", "adapt")
    workflow.add_edge("adapt", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


# Main execution
if __name__ == "__main__":
    # Sample task
    initial_state = {
        "task": "Classify and prioritize customer requests",
        "domain": "customer_service",
        "predefined_rules": [],
        "training_examples": [
            {"age": 17, "output": "minor"},
            {"age": 30, "output": "adult"},
            {"score": 95, "output": "grade A"},
            {"text": "urgent help needed", "output": "high priority"}
        ],
        "learned_patterns": [],
        "rule_predictions": [],
        "learned_predictions": [],
        "hybrid_decision": {},
        "performance_metrics": {},
        "adaptation_log": [],
        "final_output": "",
        "messages": [],
        "current_step": "pending"
    }
    
    # Create and run
    app = create_rule_learning_graph()
    
    print("Rule-Learning Hybrid MCP Pattern")
    print("=" * 50)
    
    result = app.invoke(initial_state)
    
    print("\nExecution trace:")
    for msg in result["messages"]:
        print(f"- {msg.content[:100]}...")
    
    print(f"\nHybrid Accuracy: {result['performance_metrics'].get('hybrid_accuracy', 0):.2%}")

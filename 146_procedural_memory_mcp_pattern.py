"""
Procedural Memory MCP Pattern

This pattern implements procedural memory for storing and executing
learned skills, procedures, and automated behaviors.

Key Features:
- Skill storage
- Procedural knowledge
- Automated execution
- Practice improvement
- Habit formation
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class ProceduralMemoryState(TypedDict):
    """State for procedural memory pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    skill_name: str
    procedures: List[Dict]
    proficiency_level: str
    execution_log: List[Dict]


llm = ChatOpenAI(model="gpt-4", temperature=0.2)


def procedural_memory_agent(state: ProceduralMemoryState) -> ProceduralMemoryState:
    """Manages procedural memory operations"""
    skill_name = state.get("skill_name", "")
    proficiency = state.get("proficiency_level", "novice")
    
    system_prompt = """You are a procedural memory expert.

Procedural Memory:
• "Knowing how" vs "knowing that"
• Skills and procedures
• Automated behaviors
• Motor programs
• Difficult to verbalize

Learned through practice and repetition."""
    
    user_prompt = f"""Skill: {skill_name}
Proficiency: {proficiency}

Design procedural memory system.
Show skill acquisition and execution."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🎯 Procedural Memory Agent:
    
    Skill Management:
    • Skill: {skill_name}
    • Proficiency: {proficiency}
    • Type: Procedural knowledge
    
    Procedural Memory Implementation:
    ```python
    class ProceduralMemory:
        '''Store and execute learned procedures'''
        
        def __init__(self):
            self.procedures = {{}}
            self.motor_programs = {{}}
            self.skills = {{}}
        
        def learn_procedure(self, name, steps, practice_trials=0):
            '''Acquire new procedure'''
            procedure = {{
                'name': name,
                'steps': steps,
                'proficiency': 'novice',
                'practice_count': practice_trials,
                'automation_level': 0.0,
                'execution_time': self.estimate_time(steps),
                'error_rate': 0.3,  # High initially
                'chunks': self.chunk_steps(steps)
            }}
            
            self.procedures[name] = procedure
            return procedure
        
        def execute_procedure(self, name, context):
            '''Execute learned procedure'''
            procedure = self.procedures.get(name)
            
            if not procedure:
                raise ValueError(f"Procedure {{name}} not learned")
            
            # Automatic execution if well-practiced
            if procedure['automation_level'] > 0.8:
                return self.automatic_execute(procedure, context)
            else:
                return self.controlled_execute(procedure, context)
        
        def practice(self, name, trials):
            '''Improve through practice'''
            procedure = self.procedures[name]
            
            for _ in range(trials):
                # Execute
                result = self.execute_procedure(name, {{}})
                
                # Update based on outcome
                procedure['practice_count'] += 1
                
                # Power law of practice
                procedure['execution_time'] *= 0.95
                procedure['error_rate'] *= 0.9
                procedure['automation_level'] = min(
                    1.0,
                    procedure['automation_level'] + 0.05
                )
            
            # Update proficiency
            procedure['proficiency'] = self.calculate_proficiency(procedure)
            
            return procedure
        
        def automatic_execute(self, procedure, context):
            '''Fast, automatic execution'''
            # Minimal cognitive load
            # Direct stimulus-response
            # Hard to interrupt mid-execution
            
            result = []
            for chunk in procedure['chunks']:
                result.append(self.execute_chunk(chunk, context))
            
            return result
        
        def controlled_execute(self, procedure, context):
            '''Slow, deliberate execution'''
            # High cognitive load
            # Step-by-step processing
            # Can be interrupted
            
            result = []
            for step in procedure['steps']:
                # Conscious attention to each step
                step_result = self.execute_with_monitoring(step, context)
                result.append(step_result)
                
                # Check for errors
                if not self.verify_step(step_result):
                    return self.handle_error(step)
            
            return result
    ```
    
    Skill Acquisition Stages:
    
    Fitts & Posner Model:
    ```python
    class SkillAcquisition:
        '''Three-stage skill learning model'''
        
        def cognitive_stage(self, skill):
            '''Stage 1: Understanding'''
            return {{
                'characteristics': [
                    'High cognitive load',
                    'Frequent errors',
                    'Slow execution',
                    'Deliberate attention',
                    'Verbal mediation'
                ],
                'learning_strategy': [
                    'Explicit instruction',
                    'Demonstration',
                    'Declarative knowledge',
                    'Step-by-step guidance'
                ]
            }}
        
        def associative_stage(self, skill):
            '''Stage 2: Practice'''
            return {{
                'characteristics': [
                    'Reduced errors',
                    'Faster execution',
                    'Pattern detection',
                    'Less verbal mediation',
                    'Chunking begins'
                ],
                'learning_strategy': [
                    'Repetition',
                    'Feedback',
                    'Error correction',
                    'Gradual refinement'
                ]
            }}
        
        def autonomous_stage(self, skill):
            '''Stage 3: Automatization'''
            return {{
                'characteristics': [
                    'Minimal errors',
                    'Fast execution',
                    'Automatic',
                    'Low cognitive load',
                    'Difficult to verbalize'
                ],
                'learning_strategy': [
                    'Continued practice',
                    'Varied conditions',
                    'Fine-tuning',
                    'Maintains automation'
                ]
            }}
    ```
    
    Procedure Types:
    
    Motor Procedures:
    ```python
    typing_procedure = {{
        'type': 'motor',
        'name': 'touch_typing',
        'effectors': ['fingers', 'hands'],
        'sequence': [
            'position_hands_on_home_row',
            'recognize_target_letter',
            'activate_correct_finger',
            'press_key',
            'return_to_home_position'
        ],
        'timing': 'milliseconds',
        'feedback': 'kinesthetic'
    }}
    ```
    
    Cognitive Procedures:
    ```python
    debugging_procedure = {{
        'type': 'cognitive',
        'name': 'debug_code',
        'steps': [
            'reproduce_error',
            'read_error_message',
            'locate_error_line',
            'form_hypothesis',
            'test_hypothesis',
            'fix_if_confirmed',
            'verify_fix'
        ],
        'chunked_version': [
            'identify_error_type',  # Chunks steps 1-3
            'diagnose_cause',        # Chunks steps 4-5
            'implement_fix'          # Chunks steps 6-7
        ]
    }}
    ```
    
    Perceptual Procedures:
    ```python
    code_review_procedure = {{
        'type': 'perceptual',
        'name': 'spot_code_issues',
        'pattern_recognition': [
            'identify_code_smells',
            'detect_antipatterns',
            'notice_security_issues',
            'recognize_performance_problems'
        ],
        'learned_through': 'exposure',
        'automaticity': 'high'
    }}
    ```
    
    Chunking:
    
    Hierarchical Organization:
    ```python
    def chunk_procedure(steps):
        '''Organize into hierarchical chunks'''
        # Low level (individual actions)
        low_level = steps
        
        # Mid level (action sequences)
        mid_level = [
            steps[0:3],  # Initialization chunk
            steps[3:6],  # Processing chunk
            steps[6:9]   # Finalization chunk
        ]
        
        # High level (goal-oriented)
        high_level = [
            {{'goal': 'prepare', 'steps': mid_level[0]}},
            {{'goal': 'execute', 'steps': mid_level[1]}},
            {{'goal': 'complete', 'steps': mid_level[2]}}
        ]
        
        return high_level
    ```
    
    Practice Effects:
    
    Power Law of Practice:
    ```python
    def calculate_execution_time(practice_count, initial_time):
        '''T = a * N^(-b)'''
        a = initial_time
        b = 0.4  # Learning rate
        
        return a * (practice_count ** (-b))
    
    # Example
    times = [calculate_execution_time(n, 100) 
             for n in [1, 10, 100, 1000]]
    # [100, 39.8, 15.8, 6.3] seconds
    ```
    
    Automation:
    ```python
    def calculate_automation(practice_count):
        '''Automation increases with practice'''
        # Logistic function
        max_automation = 1.0
        growth_rate = 0.01
        midpoint = 100
        
        automation = max_automation / (
            1 + np.exp(-growth_rate * (practice_count - midpoint))
        )
        
        return automation
    ```
    
    Procedural vs Declarative:
    
    Comparison:
    ```
    Declarative (Knowing That):
    • Facts and events
    • Explicit
    • Easy to verbalize
    • Conscious access
    • Example: "Decorators use @ syntax"
    
    Procedural (Knowing How):
    • Skills and procedures
    • Implicit
    • Hard to verbalize
    • Automatic execution
    • Example: Writing decorators fluently
    ```
    
    Interaction:
    ```python
    def declarative_to_procedural(declarative_knowledge):
        '''Proceduralization through practice'''
        # Start with explicit rules (declarative)
        rules = declarative_knowledge.get_rules()
        
        # Practice applying rules
        for _ in range(1000):
            apply_rules(rules)
        
        # Rules become automated (procedural)
        procedure = compile_to_procedure(rules)
        
        return procedure
    ```
    
    Habit Formation:
    
    Habit Loop:
    ```python
    class Habit:
        '''Automated behavior pattern'''
        
        def __init__(self, cue, routine, reward):
            self.cue = cue
            self.routine = routine
            self.reward = reward
            self.strength = 0.0
        
        def execute(self, context):
            '''Execute habit when cued'''
            if self.detect_cue(context):
                # Automatic execution
                result = self.routine()
                
                # Reward reinforces
                if self.reward():
                    self.strength += 0.1
                
                return result
    ```
    
    Applications:
    
    Code Generation:
    ```python
    # Novice: Looks up syntax
    procedure_novice = [
        'search_decorator_syntax',
        'read_documentation',
        'copy_template',
        'modify_for_use_case',
        'test'
    ]
    
    # Expert: Automatic writing
    procedure_expert = [
        'recognize_need_for_decorator',
        'automatically_write_pattern',  # Single chunk
        'move_to_next_task'
    ]
    ```
    
    Error Recovery:
    ```python
    def handle_execution_error(procedure, error):
        '''Recover from procedural error'''
        if procedure['automation_level'] > 0.8:
            # Automatic: hard to interrupt
            # Complete current chunk, then diagnose
            complete_current_chunk()
        
        # Revert to controlled execution
        procedure['automation_level'] *= 0.5
        
        # Analyze error
        correction = diagnose_and_correct(error)
        
        # Update procedure
        update_procedure(procedure, correction)
    ```
    
    Best Practices:
    ✓ Start with explicit instruction
    ✓ Practice for automation
    ✓ Chunk related steps
    ✓ Monitor performance
    ✓ Provide immediate feedback
    
    Key Insight:
    Procedural memory enables skilled performance
    through practice-based automation - from
    conscious steps to fluid expertise.
    """
    
    return {
        "messages": [AIMessage(content=f"🎯 Procedural Memory Agent:\n{report}\n\n{response.content}")],
        "execution_log": [{"step": "execute", "result": "success"}]
    }


def build_procedural_memory_graph():
    workflow = StateGraph(ProceduralMemoryState)
    workflow.add_node("procedural_memory_agent", procedural_memory_agent)
    workflow.add_edge(START, "procedural_memory_agent")
    workflow.add_edge("procedural_memory_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_procedural_memory_graph()
    
    print("=== Procedural Memory MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "skill_name": "Writing Python decorators",
        "procedures": [],
        "proficiency_level": "intermediate",
        "execution_log": []
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 146: Procedural Memory - COMPLETE")
    print(f"{'='*70}")

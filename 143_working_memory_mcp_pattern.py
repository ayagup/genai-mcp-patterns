"""
Working Memory MCP Pattern

This pattern implements working memory for active manipulation
and processing of information during complex tasks.

Key Features:
- Active processing
- Limited capacity
- Task-focused
- Manipulation buffer
- Central executive
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class WorkingMemoryState(TypedDict):
    """State for working memory pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    task: str
    phonological_loop: List[str]
    visuospatial_sketchpad: List[Dict]
    episodic_buffer: List[Dict]
    central_executive: Dict


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def working_memory_agent(state: WorkingMemoryState) -> WorkingMemoryState:
    """Manages working memory operations"""
    task = state.get("task", "")
    
    system_prompt = """You are a working memory expert.

Working Memory (Baddeley Model):
• Central Executive: Control and coordination
• Phonological Loop: Verbal/acoustic info
• Visuospatial Sketchpad: Visual/spatial info
• Episodic Buffer: Integrates information

Active processing workspace for complex tasks."""
    
    user_prompt = f"""Task: {task}

Design working memory system.
Show how different components handle the task."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🔧 Working Memory Agent:
    
    Task Processing:
    • Task: {task[:100]}...
    • Model: Baddeley's Multi-Component Model
    
    Working Memory Architecture:
    ```python
    class WorkingMemory:
        '''Baddeley's Working Memory Model'''
        
        def __init__(self):
            # Central control system
            self.central_executive = CentralExecutive()
            
            # Subsystems
            self.phonological_loop = PhonologicalLoop(capacity=2)  # seconds
            self.visuospatial_sketchpad = VisuospatialSketchpad()
            self.episodic_buffer = EpisodicBuffer(capacity=4)  # chunks
            
            # Long-term memory interface
            self.ltm_interface = LTMInterface()
        
        def process_task(self, task):
            # Central executive coordinates
            plan = self.central_executive.create_plan(task)
            
            for step in plan:
                # Route to appropriate subsystem
                if step.type == 'verbal':
                    self.phonological_loop.process(step)
                elif step.type == 'visual':
                    self.visuospatial_sketchpad.process(step)
                else:
                    self.episodic_buffer.integrate(step)
            
            # Combine results
            result = self.episodic_buffer.synthesize()
            return result
    ```
    
    Component Details:
    
    1. Central Executive:
    ```python
    class CentralExecutive:
        '''Attention control and coordination'''
        
        def __init__(self):
            self.attention_focus = None
            self.goal_stack = []
            self.inhibition_control = []
        
        def allocate_attention(self, tasks):
            '''Direct attention to most important task'''
            priorities = [self.calculate_priority(t) for t in tasks]
            self.attention_focus = tasks[np.argmax(priorities)]
        
        def switch_task(self, new_task):
            '''Task switching with cost'''
            # Save current state
            self.save_context()
            
            # Switch (incurs cost)
            self.attention_focus = new_task
            
            # Load new context
            self.load_context(new_task)
        
        def update_goals(self, goal):
            '''Maintain goal hierarchy'''
            self.goal_stack.append(goal)
        
        def inhibit(self, distractor):
            '''Suppress irrelevant information'''
            self.inhibition_control.append(distractor)
    ```
    
    2. Phonological Loop:
    ```python
    class PhonologicalLoop:
        '''Verbal and acoustic information'''
        
        def __init__(self, capacity_seconds=2):
            self.phonological_store = []  # Passive storage
            self.articulatory_rehearsal = []  # Active rehearsal
            self.capacity = capacity_seconds
        
        def store_verbal(self, words):
            '''Store verbal information'''
            # Automatic phonological coding
            phonological_codes = [self.encode(w) for w in words]
            
            # Limited by time-based capacity
            self.phonological_store = phonological_codes[-self.capacity:]
        
        def rehearse(self):
            '''Subvocal rehearsal to maintain info'''
            # Rehearsal loop (~1.5-2 sec for ~7 items)
            for item in self.articulatory_rehearsal:
                self.refresh(item)
        
        def encode(self, word):
            '''Phonological encoding'''
            return {{
                'word': word,
                'phonemes': self.to_phonemes(word),
                'timestamp': time.time()
            }}
    ```
    
    3. Visuospatial Sketchpad:
    ```python
    class VisuospatialSketchpad:
        '''Visual and spatial information'''
        
        def __init__(self):
            self.visual_cache = []  # Visual form/color
            self.inner_scribe = []  # Spatial/movement
        
        def store_visual(self, image):
            '''Store visual information'''
            self.visual_cache.append({{
                'image': image,
                'features': extract_features(image),
                'colors': extract_colors(image)
            }})
        
        def store_spatial(self, location, movement):
            '''Store spatial information'''
            self.inner_scribe.append({{
                'location': location,
                'path': movement,
                'relations': calculate_relations(location)
            }})
        
        def mental_rotation(self, image, degrees):
            '''Mental imagery operations'''
            rotated = rotate(image, degrees)
            self.visual_cache.append(rotated)
            return rotated
    ```
    
    4. Episodic Buffer:
    ```python
    class EpisodicBuffer:
        '''Integration of multimodal information'''
        
        def __init__(self, capacity=4):
            self.buffer = []
            self.capacity = capacity  # chunks
        
        def integrate(self, verbal, visual, ltm_info):
            '''Combine different sources'''
            episode = {{
                'verbal': verbal,
                'visual': visual,
                'ltm_context': ltm_info,
                'timestamp': time.time(),
                'coherence': self.calculate_coherence(verbal, visual, ltm_info)
            }}
            
            self.buffer.append(episode)
            
            # Maintain capacity
            if len(self.buffer) > self.capacity:
                self.buffer.pop(0)
        
        def synthesize(self):
            '''Create coherent representation'''
            return {{
                'integrated_representation': self.combine_episodes(),
                'narrative': self.create_narrative()
            }}
    ```
    
    Working Memory in Action:
    
    Math Problem Example:
    ```python
    task = "Calculate: (23 + 47) × 3"
    
    # Central Executive: Plan
    plan = [
        'Add 23 and 47',
        'Multiply result by 3'
    ]
    
    # Phonological Loop: Hold numbers
    wm.phonological_loop.store(['23', '47', '3'])
    wm.phonological_loop.rehearse()  # Keep active
    
    # Episodic Buffer: Hold intermediate
    sum_result = 23 + 47  # = 70
    wm.episodic_buffer.store({{'step': 1, 'result': 70}})
    
    # Retrieve and continue
    prev = wm.episodic_buffer.retrieve('step 1')
    final = prev.result * 3  # = 210
    ```
    
    Reading Comprehension:
    ```python
    task = "Read and understand paragraph"
    
    # Phonological Loop: Current sentence
    wm.phonological_loop.store(current_sentence)
    
    # Visuospatial: Mental model
    wm.visuospatial.build_scene(described_scene)
    
    # Episodic Buffer: Integrate
    wm.episodic_buffer.integrate(
        verbal=current_sentence,
        visual=mental_model,
        ltm=background_knowledge
    )
    
    # Central Executive: Monitor comprehension
    if wm.central_executive.check_understanding():
        continue_reading()
    else:
        re_read()
    ```
    
    Capacity Limits:
    
    Individual Differences:
    ```python
    # Working memory capacity varies
    wm_capacity = {{
        'low': 2-3,      # Difficulty with complex tasks
        'average': 4-5,  # Typical adult
        'high': 6-7      # Enhanced performance
    }}
    
    def measure_capacity(person):
        '''Operation Span Task'''
        max_items = 0
        for set_size in range(2, 8):
            if can_recall_all(set_size):
                max_items = set_size
            else:
                break
        return max_items
    ```
    
    Cognitive Load:
    ```python
    class CognitiveLoad:
        def __init__(self):
            self.intrinsic = 0   # Task complexity
            self.extraneous = 0  # Poor design
            self.germane = 0     # Learning effort
        
        def total_load(self):
            return self.intrinsic + self.extraneous + self.germane
        
        def optimize(self):
            '''Reduce load to fit WM capacity'''
            # Reduce extraneous
            self.extraneous = minimize_distractions()
            
            # Chunk to reduce intrinsic
            self.intrinsic = chunk_information()
            
            # Ensure germane load productive
            self.germane = focus_on_learning()
    ```
    
    Optimization Strategies:
    
    Chunking:
    ```python
    def chunk_for_wm(items):
        '''Group items to fit WM capacity'''
        # Phone: 5551234567 → 555-123-4567 (3 chunks)
        # Text: Individual words → Meaningful phrases
        
        chunks = []
        current = []
        
        for item in items:
            current.append(item)
            if self.is_meaningful_chunk(current):
                chunks.append(current)
                current = []
        
        return chunks[:4]  # WM capacity
    ```
    
    Offloading:
    ```python
    def offload_to_external(info):
        '''Use external aids to reduce WM load'''
        # Write down intermediate results
        write_to_paper(info)
        
        # Use calculator for complex math
        # Use diagrams for spatial reasoning
        # Use notes for multi-step tasks
    ```
    
    Best Practices:
    ✓ Monitor WM load
    ✓ Break complex tasks into steps
    ✓ Use external aids when possible
    ✓ Minimize task switching
    ✓ Chunk related information
    
    Key Insight:
    Working memory is the cognitive workspace where
    information is actively processed and manipulated
    during complex reasoning tasks.
    """
    
    return {
        "messages": [AIMessage(content=f"🔧 Working Memory Agent:\n{report}\n\n{response.content}")],
        "central_executive": {"status": "coordinating", "attention": task}
    }


def build_working_memory_graph():
    workflow = StateGraph(WorkingMemoryState)
    workflow.add_node("working_memory_agent", working_memory_agent)
    workflow.add_edge(START, "working_memory_agent")
    workflow.add_edge("working_memory_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_working_memory_graph()
    
    print("=== Working Memory MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "task": "Solve multi-step reasoning problem with multiple constraints",
        "phonological_loop": [],
        "visuospatial_sketchpad": [],
        "episodic_buffer": [],
        "central_executive": {}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 143: Working Memory - COMPLETE")
    print(f"{'='*70}")

"""
Pattern 264: Swarm Intelligence MCP Pattern

This pattern demonstrates swarm intelligence - simple agents following
local rules that lead to emergent intelligent collective behavior.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import random


class SwarmIntelligenceState(TypedDict):
    """State for swarm intelligence workflow"""
    messages: Annotated[List[str], add]
    problem: Dict[str, Any]
    swarm: List[Dict[str, Any]]
    iterations: List[Dict[str, Any]]
    best_solution: Dict[str, Any]


class SwarmAgent:
    """Individual agent in the swarm"""
    
    def __init__(self, agent_id: str, position: List[float]):
        self.agent_id = agent_id
        self.position = position.copy()
        self.velocity = [random.uniform(-1, 1) for _ in position]
        self.best_position = position.copy()
        self.best_fitness = float('-inf')
    
    def update_velocity(self, global_best: List[float], w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        """Update velocity based on personal and global best"""
        for i in range(len(self.velocity)):
            r1 = random.random()
            r2 = random.random()
            
            # Velocity update: inertia + cognitive + social
            cognitive = c1 * r1 * (self.best_position[i] - self.position[i])
            social = c2 * r2 * (global_best[i] - self.position[i])
            
            self.velocity[i] = w * self.velocity[i] + cognitive + social
            
            # Limit velocity
            self.velocity[i] = max(-2, min(2, self.velocity[i]))
    
    def update_position(self):
        """Update position based on velocity"""
        for i in range(len(self.position)):
            self.position[i] += self.velocity[i]
            # Keep within bounds [0, 10]
            self.position[i] = max(0, min(10, self.position[i]))
    
    def evaluate_fitness(self, objective_function) -> float:
        """Evaluate fitness at current position"""
        return objective_function(self.position)


class SwarmOptimizer:
    """Particle Swarm Optimization algorithm"""
    
    def __init__(self, num_agents: int, dimensions: int, objective_function):
        self.num_agents = num_agents
        self.dimensions = dimensions
        self.objective_function = objective_function
        
        # Initialize swarm
        self.swarm = []
        for i in range(num_agents):
            position = [random.uniform(0, 10) for _ in range(dimensions)]
            agent = SwarmAgent(f"Agent_{i+1}", position)
            self.swarm.append(agent)
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
    
    def optimize(self, iterations: int) -> List[Dict[str, Any]]:
        """Run swarm optimization"""
        history = []
        
        for iteration in range(iterations):
            # Evaluate all agents
            for agent in self.swarm:
                fitness = agent.evaluate_fitness(self.objective_function)
                
                # Update personal best
                if fitness > agent.best_fitness:
                    agent.best_fitness = fitness
                    agent.best_position = agent.position.copy()
                
                # Update global best
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = agent.position.copy()
            
            # Update velocities and positions
            for agent in self.swarm:
                agent.update_velocity(self.global_best_position)
                agent.update_position()
            
            # Record iteration
            avg_fitness = sum(agent.best_fitness for agent in self.swarm) / len(self.swarm)
            history.append({
                "iteration": iteration + 1,
                "global_best_fitness": self.global_best_fitness,
                "global_best_position": self.global_best_position.copy(),
                "average_fitness": avg_fitness,
                "diversity": self._calculate_diversity()
            })
        
        return history
    
    def _calculate_diversity(self) -> float:
        """Calculate swarm diversity"""
        if not self.swarm:
            return 0.0
        
        # Calculate average distance from centroid
        centroid = [0] * self.dimensions
        for agent in self.swarm:
            for i in range(self.dimensions):
                centroid[i] += agent.position[i]
        centroid = [c / len(self.swarm) for c in centroid]
        
        total_distance = 0
        for agent in self.swarm:
            distance = sum((agent.position[i] - centroid[i])**2 for i in range(self.dimensions))**0.5
            total_distance += distance
        
        return total_distance / len(self.swarm)


def define_optimization_problem_agent(state: SwarmIntelligenceState) -> SwarmIntelligenceState:
    """Define optimization problem"""
    print("\n📋 Defining Optimization Problem...")
    
    # Example: Maximize a multi-modal function
    # f(x, y) = -((x-5)^2 + (y-5)^2) + 20 (peak at x=5, y=5)
    def objective_function(position):
        x, y = position[0], position[1]
        return -((x - 5)**2 + (y - 5)**2) + 20
    
    problem = {
        "name": "2D Peak Finding",
        "description": "Find maximum of f(x,y) = -((x-5)^2 + (y-5)^2) + 20",
        "dimensions": 2,
        "search_space": [0, 10],
        "optimal_solution": [5.0, 5.0],
        "optimal_value": 20.0,
        "objective": objective_function
    }
    
    print(f"\n  Problem: {problem['name']}")
    print(f"  Description: {problem['description']}")
    print(f"  Dimensions: {problem['dimensions']}")
    print(f"  Search Space: {problem['search_space']}")
    print(f"  Known Optimum: {problem['optimal_solution']} -> {problem['optimal_value']}")
    
    return {
        **state,
        "problem": problem,
        "messages": ["✓ Problem defined"]
    }


def run_swarm_optimization_agent(state: SwarmIntelligenceState) -> SwarmIntelligenceState:
    """Run swarm optimization"""
    print("\n🐝 Running Swarm Optimization...")
    
    # Initialize swarm
    swarm_size = 20
    iterations = 30
    
    optimizer = SwarmOptimizer(
        num_agents=swarm_size,
        dimensions=state["problem"]["dimensions"],
        objective_function=state["problem"]["objective"]
    )
    
    print(f"\n  Swarm Size: {swarm_size}")
    print(f"  Iterations: {iterations}")
    
    # Run optimization
    history = optimizer.optimize(iterations)
    
    # Get swarm state
    swarm_state = []
    for agent in optimizer.swarm:
        swarm_state.append({
            "id": agent.agent_id,
            "position": agent.position.copy(),
            "best_position": agent.best_position.copy(),
            "best_fitness": agent.best_fitness
        })
    
    best_solution = {
        "position": optimizer.global_best_position,
        "fitness": optimizer.global_best_fitness
    }
    
    print(f"\n  Optimization Progress:")
    # Show select iterations
    show_iterations = [0, 9, 19, 29]
    for idx in show_iterations:
        if idx < len(history):
            h = history[idx]
            print(f"\n    Iteration {h['iteration']}:")
            print(f"      Best Fitness: {h['global_best_fitness']:.4f}")
            print(f"      Best Position: [{h['global_best_position'][0]:.3f}, {h['global_best_position'][1]:.3f}]")
            print(f"      Avg Fitness: {h['average_fitness']:.4f}")
            print(f"      Diversity: {h['diversity']:.3f}")
    
    print(f"\n  🎯 Final Solution:")
    print(f"     Position: [{best_solution['position'][0]:.4f}, {best_solution['position'][1]:.4f}]")
    print(f"     Fitness: {best_solution['fitness']:.4f}")
    print(f"     Error: {abs(best_solution['fitness'] - state['problem']['optimal_value']):.4f}")
    
    return {
        **state,
        "swarm": swarm_state,
        "iterations": history,
        "best_solution": best_solution,
        "messages": [f"✓ Swarm optimization complete: {iterations} iterations"]
    }


def generate_swarm_report_agent(state: SwarmIntelligenceState) -> SwarmIntelligenceState:
    """Generate swarm intelligence report"""
    print("\n" + "="*70)
    print("SWARM INTELLIGENCE REPORT")
    print("="*70)
    
    print(f"\n📋 Optimization Problem:")
    print(f"  Name: {state['problem']['name']}")
    print(f"  Description: {state['problem']['description']}")
    print(f"  Dimensions: {state['problem']['dimensions']}")
    print(f"  Known Optimum: {state['problem']['optimal_solution']} -> {state['problem']['optimal_value']}")
    
    print(f"\n🐝 Swarm Configuration:")
    print(f"  Swarm Size: {len(state['swarm'])}")
    print(f"  Iterations: {len(state['iterations'])}")
    
    print(f"\n📈 Optimization Progress:")
    milestones = [0, len(state['iterations'])//3, 2*len(state['iterations'])//3, len(state['iterations'])-1]
    for idx in milestones:
        if idx < len(state['iterations']):
            h = state['iterations'][idx]
            print(f"\n  Iteration {h['iteration']}:")
            print(f"    Best Fitness: {h['global_best_fitness']:.4f}")
            print(f"    Position: [{h['global_best_position'][0]:.3f}, {h['global_best_position'][1]:.3f}]")
            print(f"    Average Fitness: {h['average_fitness']:.4f}")
            print(f"    Swarm Diversity: {h['diversity']:.3f}")
    
    print(f"\n🎯 Final Solution:")
    best = state["best_solution"]
    optimal = state["problem"]["optimal_value"]
    error = abs(best["fitness"] - optimal)
    accuracy = (1 - error/optimal) * 100 if optimal != 0 else 0
    
    print(f"  Position: [{best['position'][0]:.4f}, {best['position'][1]:.4f}]")
    print(f"  Fitness: {best['fitness']:.4f}")
    print(f"  Known Optimum: {optimal:.4f}")
    print(f"  Error: {error:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    print(f"\n🐝 Top 5 Swarm Agents:")
    sorted_swarm = sorted(state["swarm"], key=lambda x: x["best_fitness"], reverse=True)
    for i, agent in enumerate(sorted_swarm[:5], 1):
        print(f"\n  {i}. {agent['id']}")
        print(f"     Best Fitness: {agent['best_fitness']:.4f}")
        print(f"     Best Position: [{agent['best_position'][0]:.3f}, {agent['best_position'][1]:.3f}]")
        print(f"     Current Position: [{agent['position'][0]:.3f}, {agent['position'][1]:.3f}]")
    
    print(f"\n💡 Swarm Intelligence Benefits:")
    print("  • Simple individual rules")
    print("  • Emergent collective intelligence")
    print("  • Robust to local optima")
    print("  • Parallel exploration")
    print("  • Self-organizing behavior")
    print("  • Scalable approach")
    print("  • No central coordination needed")
    
    print(f"\n📊 Convergence Analysis:")
    first_fitness = state['iterations'][0]['global_best_fitness']
    final_fitness = state['iterations'][-1]['global_best_fitness']
    improvement = final_fitness - first_fitness
    print(f"  Initial Best: {first_fitness:.4f}")
    print(f"  Final Best: {final_fitness:.4f}")
    print(f"  Improvement: {improvement:.4f}")
    print(f"  Improvement Rate: {(improvement/abs(first_fitness)*100):.2f}%" if first_fitness != 0 else "  N/A")
    
    print("\n="*70)
    print("✅ Swarm Intelligence Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_swarm_intelligence_graph():
    workflow = StateGraph(SwarmIntelligenceState)
    workflow.add_node("define", define_optimization_problem_agent)
    workflow.add_node("optimize", run_swarm_optimization_agent)
    workflow.add_node("report", generate_swarm_report_agent)
    workflow.add_edge(START, "define")
    workflow.add_edge("define", "optimize")
    workflow.add_edge("optimize", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 264: Swarm Intelligence MCP Pattern")
    print("="*70)
    
    app = create_swarm_intelligence_graph()
    final_state = app.invoke({
        "messages": [],
        "problem": {},
        "swarm": [],
        "iterations": [],
        "best_solution": {}
    })
    print("\n✅ Swarm Intelligence Pattern Complete!")


if __name__ == "__main__":
    main()

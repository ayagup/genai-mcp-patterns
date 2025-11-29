"""
Pattern 292: Agent Discovery MCP Pattern

This pattern demonstrates how agents discover and locate other agents
in a multi-agent system for collaboration and task delegation.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class AgentDiscoveryPattern(TypedDict):
    """State for agent discovery"""
    messages: Annotated[List[str], add]
    agent_directory: Dict[str, Any]
    discovered_agents: List[Dict[str, Any]]
    capability_matches: Dict[str, List[str]]
    collaboration_requests: List[Dict[str, Any]]


class AgentProfile:
    """Represents an agent's profile"""
    
    def __init__(self, agent_id: str, name: str, agent_type: str):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.capabilities = []
        self.specializations = []
        self.status = "available"
        self.load = 0.0
        self.location = None
        self.registered_at = time.time()
    
    def add_capability(self, capability: str):
        """Add capability to agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def add_specialization(self, specialization: str):
        """Add specialization"""
        if specialization not in self.specializations:
            self.specializations.append(specialization)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "specializations": self.specializations,
            "status": self.status,
            "load": self.load,
            "location": self.location,
            "registered_at": self.registered_at
        }


class AgentDirectory:
    """Directory for agent discovery"""
    
    def __init__(self):
        self.agents = {}
        self.capability_index = {}
        self.type_index = {}
    
    def register_agent(self, profile: AgentProfile):
        """Register an agent"""
        self.agents[profile.agent_id] = profile
        
        # Index by type
        if profile.agent_type not in self.type_index:
            self.type_index[profile.agent_type] = []
        self.type_index[profile.agent_type].append(profile.agent_id)
        
        # Index by capabilities
        for capability in profile.capabilities:
            if capability not in self.capability_index:
                self.capability_index[capability] = []
            self.capability_index[capability].append(profile.agent_id)
    
    def discover_by_capability(self, capability: str):
        """Discover agents with specific capability"""
        agent_ids = self.capability_index.get(capability, [])
        return [self.agents[aid].to_dict() for aid in agent_ids if aid in self.agents]
    
    def discover_by_type(self, agent_type: str):
        """Discover agents by type"""
        agent_ids = self.type_index.get(agent_type, [])
        return [self.agents[aid].to_dict() for aid in agent_ids if aid in self.agents]
    
    def discover_by_specialization(self, specialization: str):
        """Discover agents with specialization"""
        results = []
        for agent in self.agents.values():
            if specialization in agent.specializations:
                results.append(agent.to_dict())
        return results
    
    def discover_available(self, capability: str = None):
        """Discover available agents"""
        results = []
        for agent in self.agents.values():
            if agent.status == "available":
                if capability is None or capability in agent.capabilities:
                    results.append(agent.to_dict())
        return results
    
    def find_best_match(self, required_capabilities: List[str]):
        """Find best agent matching multiple capabilities"""
        scores = {}
        
        for agent_id, agent in self.agents.items():
            if agent.status != "available":
                continue
            
            # Calculate match score
            matched = sum(1 for cap in required_capabilities if cap in agent.capabilities)
            
            if matched > 0:
                # Score based on matches and load
                score = matched / len(required_capabilities)
                load_penalty = agent.load * 0.3
                scores[agent_id] = score - load_penalty
        
        if scores:
            best_agent_id = max(scores, key=scores.get)
            return self.agents[best_agent_id].to_dict(), scores[best_agent_id]
        
        return None, 0


def initialize_agent_directory_agent(state: AgentDiscoveryPattern) -> AgentDiscoveryPattern:
    """Initialize agent directory"""
    print("\n📁 Initializing Agent Directory...")
    
    directory = AgentDirectory()
    
    print(f"  Directory: Ready")
    print(f"  Features:")
    print(f"    • Agent registration")
    print(f"    • Capability indexing")
    print(f"    • Type-based discovery")
    print(f"    • Load-aware matching")
    
    return {
        **state,
        "agent_directory": {},
        "discovered_agents": [],
        "capability_matches": {},
        "collaboration_requests": [],
        "messages": ["✓ Agent directory initialized"]
    }


def register_agents_agent(state: AgentDiscoveryPattern) -> AgentDiscoveryPattern:
    """Register multiple agents"""
    print("\n📝 Registering Agents...")
    
    directory = AgentDirectory()
    
    # Create and register various agents
    agents_config = [
        {
            "agent_id": "agent_nlp_001",
            "name": "NLP Specialist",
            "type": "specialist",
            "capabilities": ["text_analysis", "sentiment_analysis", "entity_extraction"],
            "specializations": ["natural_language", "linguistics"]
        },
        {
            "agent_id": "agent_vision_001",
            "name": "Vision Agent",
            "type": "specialist",
            "capabilities": ["image_recognition", "object_detection", "ocr"],
            "specializations": ["computer_vision", "image_processing"]
        },
        {
            "agent_id": "agent_data_001",
            "name": "Data Analyst",
            "type": "analyst",
            "capabilities": ["data_analysis", "statistical_modeling", "visualization"],
            "specializations": ["statistics", "data_science"]
        },
        {
            "agent_id": "agent_coord_001",
            "name": "Coordinator",
            "type": "coordinator",
            "capabilities": ["task_coordination", "workflow_management", "resource_allocation"],
            "specializations": ["orchestration", "planning"]
        },
        {
            "agent_id": "agent_code_001",
            "name": "Code Assistant",
            "type": "specialist",
            "capabilities": ["code_generation", "code_review", "debugging"],
            "specializations": ["programming", "software_engineering"]
        },
        {
            "agent_id": "agent_search_001",
            "name": "Search Agent",
            "type": "retrieval",
            "capabilities": ["information_retrieval", "web_search", "document_search"],
            "specializations": ["search", "indexing"]
        }
    ]
    
    for config in agents_config:
        profile = AgentProfile(
            config["agent_id"],
            config["name"],
            config["type"]
        )
        
        for cap in config["capabilities"]:
            profile.add_capability(cap)
        
        for spec in config["specializations"]:
            profile.add_specialization(spec)
        
        # Set random load
        profile.load = (hash(config["agent_id"]) % 50) / 100.0
        
        directory.register_agent(profile)
        
        print(f"  ✓ Registered: {profile.name}")
        print(f"    ID: {profile.agent_id}")
        print(f"    Type: {profile.agent_type}")
        print(f"    Capabilities: {len(profile.capabilities)}")
    
    print(f"\n  Total Agents: {len(directory.agents)}")
    print(f"  Indexed Capabilities: {len(directory.capability_index)}")
    
    agent_dict = {aid: agent.to_dict() for aid, agent in directory.agents.items()}
    
    return {
        **state,
        "agent_directory": agent_dict,
        "messages": [f"✓ Registered {len(agents_config)} agents"]
    }


def discover_agents_by_capability_agent(state: AgentDiscoveryPattern) -> AgentDiscoveryPattern:
    """Discover agents by capabilities"""
    print("\n🔍 Discovering Agents by Capability...")
    
    directory = AgentDirectory()
    
    # Recreate directory from state
    for agent_id, agent_data in state["agent_directory"].items():
        profile = AgentProfile(
            agent_data["agent_id"],
            agent_data["name"],
            agent_data["agent_type"]
        )
        profile.capabilities = agent_data["capabilities"]
        profile.specializations = agent_data["specializations"]
        profile.status = agent_data["status"]
        profile.load = agent_data["load"]
        directory.register_agent(profile)
    
    # Discover agents with specific capabilities
    capability_queries = [
        "text_analysis",
        "code_generation",
        "data_analysis",
        "task_coordination",
        "image_recognition"
    ]
    
    capability_matches = {}
    all_discovered = []
    
    for capability in capability_queries:
        agents = directory.discover_by_capability(capability)
        capability_matches[capability] = [a["agent_id"] for a in agents]
        all_discovered.extend(agents)
        
        print(f"\n  Capability: {capability}")
        print(f"  Found: {len(agents)} agent(s)")
        for agent in agents:
            print(f"    • {agent['name']} (Load: {agent['load']:.2f})")
    
    return {
        **state,
        "discovered_agents": all_discovered,
        "capability_matches": capability_matches,
        "messages": [f"✓ Queried {len(capability_queries)} capabilities"]
    }


def find_best_match_agent(state: AgentDiscoveryPattern) -> AgentDiscoveryPattern:
    """Find best agent match for complex tasks"""
    print("\n🎯 Finding Best Agent Matches...")
    
    directory = AgentDirectory()
    
    # Recreate directory
    for agent_id, agent_data in state["agent_directory"].items():
        profile = AgentProfile(
            agent_data["agent_id"],
            agent_data["name"],
            agent_data["agent_type"]
        )
        profile.capabilities = agent_data["capabilities"]
        profile.status = agent_data["status"]
        profile.load = agent_data["load"]
        directory.register_agent(profile)
    
    # Complex tasks requiring multiple capabilities
    task_requirements = [
        {
            "task": "Analyze document sentiment",
            "capabilities": ["text_analysis", "sentiment_analysis"]
        },
        {
            "task": "Generate and review code",
            "capabilities": ["code_generation", "code_review"]
        },
        {
            "task": "Process and analyze data",
            "capabilities": ["data_analysis", "statistical_modeling"]
        }
    ]
    
    collaboration_requests = []
    
    for task_req in task_requirements:
        best_agent, score = directory.find_best_match(task_req["capabilities"])
        
        print(f"\n  Task: {task_req['task']}")
        print(f"  Required: {', '.join(task_req['capabilities'])}")
        
        if best_agent:
            print(f"  Best Match: {best_agent['name']}")
            print(f"  Match Score: {score:.2f}")
            print(f"  Agent Load: {best_agent['load']:.2f}")
            
            collaboration_requests.append({
                "task": task_req["task"],
                "agent_id": best_agent["agent_id"],
                "agent_name": best_agent["name"],
                "score": score
            })
        else:
            print(f"  No suitable agent found")
    
    return {
        **state,
        "collaboration_requests": collaboration_requests,
        "messages": [f"✓ Matched {len(collaboration_requests)} tasks"]
    }


def generate_agent_discovery_report_agent(state: AgentDiscoveryPattern) -> AgentDiscoveryPattern:
    """Generate agent discovery report"""
    print("\n" + "="*70)
    print("AGENT DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n📁 Agent Directory:")
    print(f"  Total Agents: {len(state['agent_directory'])}")
    
    # Group by type
    by_type = {}
    for agent_data in state['agent_directory'].values():
        atype = agent_data['agent_type']
        by_type[atype] = by_type.get(atype, 0) + 1
    
    print(f"\n  Agents by Type:")
    for atype, count in by_type.items():
        print(f"    {atype}: {count}")
    
    print(f"\n🤖 Registered Agents:")
    for agent_data in list(state['agent_directory'].values())[:5]:
        print(f"\n  • {agent_data['name']}:")
        print(f"      ID: {agent_data['agent_id']}")
        print(f"      Type: {agent_data['agent_type']}")
        print(f"      Capabilities: {', '.join(agent_data['capabilities'])}")
        print(f"      Status: {agent_data['status']}")
        print(f"      Load: {agent_data['load']:.2f}")
    
    print(f"\n🔍 Capability Matches:")
    for capability, agent_ids in state['capability_matches'].items():
        print(f"  {capability}: {len(agent_ids)} agent(s)")
    
    print(f"\n🎯 Collaboration Requests:")
    for req in state['collaboration_requests']:
        print(f"\n  Task: {req['task']}")
        print(f"    Assigned to: {req['agent_name']}")
        print(f"    Match Score: {req['score']:.2f}")
    
    print(f"\n💡 Agent Discovery Benefits:")
    print("  ✓ Dynamic agent location")
    print("  ✓ Capability-based matching")
    print("  ✓ Load-aware assignment")
    print("  ✓ Specialization discovery")
    print("  ✓ Multi-agent collaboration")
    print("  ✓ Resource optimization")
    
    print(f"\n🔧 Discovery Strategies:")
    print("  • Capability-based")
    print("  • Type-based")
    print("  • Specialization-based")
    print("  • Load-aware")
    print("  • Multi-criteria matching")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Multi-agent systems")
    print("  • Task delegation")
    print("  • Workload distribution")
    print("  • Agent collaboration")
    print("  • Dynamic teaming")
    print("  • Resource allocation")
    
    print("\n" + "="*70)
    print("✅ Agent Discovery Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_agent_discovery_graph():
    """Create agent discovery workflow"""
    workflow = StateGraph(AgentDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_agent_directory_agent)
    workflow.add_node("register", register_agents_agent)
    workflow.add_node("discover", discover_agents_by_capability_agent)
    workflow.add_node("match", find_best_match_agent)
    workflow.add_node("report", generate_agent_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "register")
    workflow.add_edge("register", "discover")
    workflow.add_edge("discover", "match")
    workflow.add_edge("match", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 292: Agent Discovery MCP Pattern")
    print("="*70)
    
    app = create_agent_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "agent_directory": {},
        "discovered_agents": [],
        "capability_matches": {},
        "collaboration_requests": []
    })
    
    print("\n✅ Agent Discovery Pattern Complete!")


if __name__ == "__main__":
    main()

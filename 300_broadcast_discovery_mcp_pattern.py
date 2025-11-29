"""
Pattern 300: Broadcast Discovery MCP Pattern

This pattern demonstrates service discovery using broadcast messages
where services announce their presence by broadcasting to all nodes
in the network.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class BroadcastDiscoveryPattern(TypedDict):
    """State for broadcast-based discovery"""
    messages: Annotated[List[str], add]
    network_nodes: Dict[str, Any]
    broadcast_messages: List[Dict[str, Any]]
    discovered_services: Dict[str, Any]
    network_statistics: Dict[str, Any]


class NetworkNode:
    """Network node that can send/receive broadcasts"""
    
    def __init__(self, node_id: str, ip_address: str):
        self.node_id = node_id
        self.ip_address = ip_address
        self.services = []
        self.discovered_peers = {}
        self.broadcast_count = 0
        self.received_count = 0
    
    def add_service(self, service_name: str, port: int, metadata: dict = None):
        """Add a service to this node"""
        service = {
            "name": service_name,
            "port": port,
            "node_id": self.node_id,
            "endpoint": f"{self.ip_address}:{port}",
            "metadata": metadata or {},
            "announced_at": time.time()
        }
        self.services.append(service)
        return service
    
    def create_broadcast(self):
        """Create broadcast announcement"""
        self.broadcast_count += 1
        
        return {
            "type": "service_announcement",
            "from_node": self.node_id,
            "from_ip": self.ip_address,
            "services": self.services,
            "timestamp": time.time(),
            "broadcast_id": f"{self.node_id}_{self.broadcast_count}"
        }
    
    def receive_broadcast(self, broadcast: dict):
        """Receive and process broadcast"""
        self.received_count += 1
        
        # Don't process own broadcasts
        if broadcast["from_node"] == self.node_id:
            return
        
        # Store discovered peer
        peer_id = broadcast["from_node"]
        self.discovered_peers[peer_id] = {
            "node_id": peer_id,
            "ip": broadcast["from_ip"],
            "services": broadcast["services"],
            "last_seen": broadcast["timestamp"]
        }
    
    def get_all_discovered_services(self):
        """Get all discovered services from peers"""
        all_services = []
        
        for peer in self.discovered_peers.values():
            for service in peer["services"]:
                all_services.append({
                    **service,
                    "discovered_from": peer["node_id"]
                })
        
        return all_services
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "ip_address": self.ip_address,
            "services_count": len(self.services),
            "discovered_peers": len(self.discovered_peers),
            "broadcast_count": self.broadcast_count,
            "received_count": self.received_count
        }


class BroadcastNetwork:
    """Network that handles broadcast discovery"""
    
    def __init__(self):
        self.nodes = {}
        self.broadcast_history = []
    
    def add_node(self, node: NetworkNode):
        """Add a node to the network"""
        self.nodes[node.node_id] = node
    
    def broadcast_from_node(self, node_id: str):
        """Broadcast from a specific node"""
        if node_id not in self.nodes:
            return None
        
        sender = self.nodes[node_id]
        broadcast = sender.create_broadcast()
        
        # Deliver to all other nodes
        for node in self.nodes.values():
            node.receive_broadcast(broadcast)
        
        self.broadcast_history.append(broadcast)
        return broadcast
    
    def broadcast_all(self):
        """All nodes broadcast their services"""
        broadcasts = []
        
        for node_id in list(self.nodes.keys()):
            broadcast = self.broadcast_from_node(node_id)
            if broadcast:
                broadcasts.append(broadcast)
        
        return broadcasts
    
    def get_network_topology(self):
        """Get network topology view"""
        topology = {}
        
        for node_id, node in self.nodes.items():
            topology[node_id] = {
                "node": node.to_dict(),
                "peers": list(node.discovered_peers.keys()),
                "services": node.services
            }
        
        return topology


def initialize_broadcast_network_agent(state: BroadcastDiscoveryPattern) -> BroadcastDiscoveryPattern:
    """Initialize broadcast network"""
    print("\n📡 Initializing Broadcast Network...")
    
    print(f"  Network Type: Broadcast")
    print(f"  Discovery Method: Service Announcement")
    
    print(f"\n  Features:")
    print(f"    • Peer-to-peer discovery")
    print(f"    • No central registry")
    print(f"    • Automatic service announcement")
    print(f"    • Network-wide visibility")
    
    return {
        **state,
        "network_nodes": {},
        "broadcast_messages": [],
        "discovered_services": {},
        "network_statistics": {},
        "messages": ["✓ Broadcast network initialized"]
    }


def setup_network_nodes_agent(state: BroadcastDiscoveryPattern) -> BroadcastDiscoveryPattern:
    """Setup network nodes with services"""
    print("\n🖥️ Setting Up Network Nodes...")
    
    network = BroadcastNetwork()
    
    # Create nodes with services
    nodes_config = [
        ("node_1", "192.168.1.10", [
            ("api-service", 8080, {"version": "1.0"}),
            ("metrics", 9090, {"type": "prometheus"})
        ]),
        ("node_2", "192.168.1.11", [
            ("api-service", 8080, {"version": "1.0"}),
            ("database", 5432, {"type": "postgresql"})
        ]),
        ("node_3", "192.168.1.12", [
            ("cache", 6379, {"type": "redis"}),
            ("queue", 5672, {"type": "rabbitmq"})
        ]),
        ("node_4", "192.168.1.13", [
            ("web-server", 80, {"type": "nginx"}),
            ("storage", 9000, {"type": "minio"})
        ])
    ]
    
    for node_id, ip, services in nodes_config:
        node = NetworkNode(node_id, ip)
        
        print(f"\n  Node: {node_id} ({ip})")
        
        for service_name, port, metadata in services:
            service = node.add_service(service_name, port, metadata)
            print(f"    ✓ Service: {service_name}")
            print(f"      Port: {port}")
            print(f"      Endpoint: {service['endpoint']}")
        
        network.add_node(node)
    
    print(f"\n  Total Nodes: {len(network.nodes)}")
    
    # Convert to state
    nodes_dict = {nid: node.to_dict() for nid, node in network.nodes.items()}
    
    return {
        **state,
        "network_nodes": nodes_dict,
        "messages": [f"✓ Setup {len(network.nodes)} nodes"]
    }


def broadcast_service_announcements_agent(state: BroadcastDiscoveryPattern) -> BroadcastDiscoveryPattern:
    """Broadcast service announcements"""
    print("\n📢 Broadcasting Service Announcements...")
    
    network = BroadcastNetwork()
    
    # Recreate network from state
    for node_id, node_data in state["network_nodes"].items():
        # We need to recreate nodes with their services
        # This is simplified - in real scenario we'd store more state
        node = NetworkNode(node_id, "192.168.1." + node_id.split("_")[1])
        network.add_node(node)
    
    # Simulate nodes with services (recreate based on config)
    nodes_config = {
        "node_1": [("api-service", 8080), ("metrics", 9090)],
        "node_2": [("api-service", 8080), ("database", 5432)],
        "node_3": [("cache", 6379), ("queue", 5672)],
        "node_4": [("web-server", 80), ("storage", 9000)]
    }
    
    for node_id, services in nodes_config.items():
        if node_id in network.nodes:
            for service_name, port in services:
                network.nodes[node_id].add_service(service_name, port)
    
    # Perform broadcast
    print(f"\n  Broadcasting from all nodes...")
    broadcasts = network.broadcast_all()
    
    for broadcast in broadcasts:
        print(f"\n  📡 Broadcast from {broadcast['from_node']}:")
        print(f"     Services: {len(broadcast['services'])}")
        for service in broadcast['services']:
            print(f"       • {service['name']} @ {service['endpoint']}")
    
    print(f"\n  Total Broadcasts: {len(broadcasts)}")
    
    return {
        **state,
        "broadcast_messages": broadcasts,
        "messages": [f"✓ Broadcasted {len(broadcasts)} announcements"]
    }


def discover_services_agent(state: BroadcastDiscoveryPattern) -> BroadcastDiscoveryPattern:
    """Discover services from broadcasts"""
    print("\n🔍 Discovering Services from Broadcasts...")
    
    network = BroadcastNetwork()
    
    # Recreate network and process broadcasts
    for node_id in state["network_nodes"].keys():
        node = NetworkNode(node_id, "192.168.1." + node_id.split("_")[1])
        network.add_node(node)
    
    # Process all broadcast messages
    for broadcast in state["broadcast_messages"]:
        for node in network.nodes.values():
            node.receive_broadcast(broadcast)
    
    # Collect discovered services from each node
    discovered_services = {}
    
    for node_id, node in network.nodes.items():
        discovered = node.get_all_discovered_services()
        discovered_services[node_id] = discovered
        
        print(f"\n  Node {node_id} discovered:")
        print(f"    Peers: {len(node.discovered_peers)}")
        print(f"    Services: {len(discovered)}")
        
        # Show sample discoveries
        for service in discovered[:3]:
            print(f"      • {service['name']} @ {service['endpoint']}")
            print(f"        From: {service['discovered_from']}")
    
    total_discoveries = sum(len(services) for services in discovered_services.values())
    print(f"\n  Total Service Discoveries: {total_discoveries}")
    
    return {
        **state,
        "discovered_services": discovered_services,
        "messages": [f"✓ Discovered services across {len(network.nodes)} nodes"]
    }


def analyze_network_topology_agent(state: BroadcastDiscoveryPattern) -> BroadcastDiscoveryPattern:
    """Analyze network topology"""
    print("\n🗺️ Analyzing Network Topology...")
    
    network = BroadcastNetwork()
    
    # Recreate network
    for node_id in state["network_nodes"].keys():
        node = NetworkNode(node_id, "192.168.1." + node_id.split("_")[1])
        network.add_node(node)
    
    # Process broadcasts to rebuild peer relationships
    for broadcast in state["broadcast_messages"]:
        for node in network.nodes.values():
            node.receive_broadcast(broadcast)
    
    # Get topology
    topology = network.get_network_topology()
    
    # Calculate statistics
    total_nodes = len(network.nodes)
    total_broadcasts = len(state["broadcast_messages"])
    total_services = sum(
        len(services) for services in state["discovered_services"].values()
    )
    
    # Calculate connectivity
    total_possible_connections = total_nodes * (total_nodes - 1)
    actual_connections = sum(
        len(node.discovered_peers) for node in network.nodes.values()
    )
    
    connectivity = actual_connections / max(total_possible_connections, 1)
    
    statistics = {
        "total_nodes": total_nodes,
        "total_broadcasts": total_broadcasts,
        "total_service_discoveries": total_services,
        "connectivity": connectivity,
        "avg_peers_per_node": actual_connections / max(total_nodes, 1),
        "broadcasts_per_node": total_broadcasts / max(total_nodes, 1)
    }
    
    print(f"  Total Nodes: {statistics['total_nodes']}")
    print(f"  Total Broadcasts: {statistics['total_broadcasts']}")
    print(f"  Service Discoveries: {statistics['total_service_discoveries']}")
    print(f"  Network Connectivity: {statistics['connectivity']:.1%}")
    print(f"  Avg Peers/Node: {statistics['avg_peers_per_node']:.1f}")
    
    return {
        **state,
        "network_statistics": statistics,
        "messages": ["✓ Network topology analyzed"]
    }


def generate_broadcast_discovery_report_agent(state: BroadcastDiscoveryPattern) -> BroadcastDiscoveryPattern:
    """Generate broadcast discovery report"""
    print("\n" + "="*70)
    print("BROADCAST DISCOVERY REPORT")
    print("="*70)
    
    print(f"\n🖥️ Network Nodes:")
    for node_id, node_data in state["network_nodes"].items():
        print(f"\n  Node: {node_id}")
        print(f"    IP: {node_data.get('ip_address', 'N/A')}")
        print(f"    Services: {node_data['services_count']}")
        print(f"    Discovered Peers: {node_data['discovered_peers']}")
        print(f"    Broadcasts Sent: {node_data['broadcast_count']}")
        print(f"    Broadcasts Received: {node_data['received_count']}")
    
    print(f"\n📡 Broadcast Messages:")
    for i, broadcast in enumerate(state['broadcast_messages'], 1):
        print(f"\n  Broadcast {i}:")
        print(f"    From: {broadcast['from_node']} ({broadcast['from_ip']})")
        print(f"    Services Announced: {len(broadcast['services'])}")
        print(f"    ID: {broadcast['broadcast_id']}")
    
    print(f"\n🔍 Discovered Services:")
    total_discovered = 0
    for node_id, services in state['discovered_services'].items():
        if services:
            print(f"\n  Node {node_id}:")
            for service in services[:3]:
                print(f"    • {service['name']} @ {service['endpoint']}")
            total_discovered += len(services)
    print(f"\n  Total Discovered: {total_discovered} service instances")
    
    print(f"\n📊 Network Statistics:")
    stats = state['network_statistics']
    if stats:
        print(f"  Total Nodes: {stats['total_nodes']}")
        print(f"  Total Broadcasts: {stats['total_broadcasts']}")
        print(f"  Service Discoveries: {stats['total_service_discoveries']}")
        print(f"  Network Connectivity: {stats['connectivity']:.1%}")
        print(f"  Avg Peers/Node: {stats['avg_peers_per_node']:.1f}")
        print(f"  Broadcasts/Node: {stats['broadcasts_per_node']:.1f}")
    
    print(f"\n💡 Broadcast Discovery Benefits:")
    print("  ✓ No central point of failure")
    print("  ✓ Automatic peer discovery")
    print("  ✓ Simple implementation")
    print("  ✓ Self-organizing network")
    print("  ✓ Fault tolerant")
    print("  ✓ Zero configuration")
    
    print(f"\n🔧 Broadcast Mechanism:")
    print("  • Service announcement messages")
    print("  • Network-wide distribution")
    print("  • Peer list maintenance")
    print("  • Periodic re-announcement")
    print("  • Local service cache")
    
    print(f"\n⚙️ Use Cases:")
    print("  • IoT device discovery")
    print("  • Local network services")
    print("  • Peer-to-peer networks")
    print("  • Zero-config networking")
    print("  • Ad-hoc networks")
    print("  • Service mesh bootstrap")
    
    print(f"\n🎯 Popular Protocols:")
    print("  • mDNS (Multicast DNS)")
    print("  • SSDP (Simple Service Discovery)")
    print("  • UPnP")
    print("  • Bonjour")
    print("  • Avahi")
    
    print(f"\n⚠️ Limitations:")
    print("  • Network broadcast overhead")
    print("  • Limited to local network")
    print("  • Scalability challenges")
    print("  • Security considerations")
    
    print("\n" + "="*70)
    print("✅ Broadcast Discovery Pattern Complete!")
    print("="*70)
    print("\n🎉 DISCOVERY PATTERNS (291-300) COMPLETE!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_broadcast_discovery_graph():
    """Create broadcast discovery workflow"""
    workflow = StateGraph(BroadcastDiscoveryPattern)
    
    workflow.add_node("initialize", initialize_broadcast_network_agent)
    workflow.add_node("setup", setup_network_nodes_agent)
    workflow.add_node("broadcast", broadcast_service_announcements_agent)
    workflow.add_node("discover", discover_services_agent)
    workflow.add_node("analyze", analyze_network_topology_agent)
    workflow.add_node("report", generate_broadcast_discovery_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "setup")
    workflow.add_edge("setup", "broadcast")
    workflow.add_edge("broadcast", "discover")
    workflow.add_edge("discover", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 300: Broadcast Discovery MCP Pattern")
    print("="*70)
    
    app = create_broadcast_discovery_graph()
    final_state = app.invoke({
        "messages": [],
        "network_nodes": {},
        "broadcast_messages": [],
        "discovered_services": {},
        "network_statistics": {}
    })
    
    print("\n✅ Broadcast Discovery Pattern Complete!")


if __name__ == "__main__":
    main()

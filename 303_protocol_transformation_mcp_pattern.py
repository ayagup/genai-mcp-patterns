"""
Pattern 303: Protocol Transformation MCP Pattern

This pattern demonstrates transforming communication protocols
(HTTP, gRPC, WebSocket, MQTT, etc.) while maintaining message semantics.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json
import time


class ProtocolTransformationPattern(TypedDict):
    """State for protocol transformation"""
    messages: Annotated[List[str], add]
    source_protocol: str
    target_protocol: str
    protocol_adapters: Dict[str, Any]
    transformation_log: List[Dict[str, Any]]
    protocol_statistics: Dict[str, Any]


class ProtocolMessage:
    """Generic protocol message"""
    
    def __init__(self, protocol: str, headers: dict, body: Any, metadata: dict = None):
        self.protocol = protocol
        self.headers = headers
        self.body = body
        self.metadata = metadata or {}
        self.timestamp = time.time()
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "protocol": self.protocol,
            "headers": self.headers,
            "body": self.body,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ProtocolAdapter:
    """Adapt between different protocols"""
    
    def __init__(self):
        self.adapters = {}
    
    def http_to_grpc(self, http_msg: ProtocolMessage) -> ProtocolMessage:
        """Convert HTTP message to gRPC"""
        
        # Map HTTP headers to gRPC metadata
        grpc_metadata = {}
        for key, value in http_msg.headers.items():
            # gRPC metadata keys are lowercase
            grpc_key = key.lower().replace("-", "_")
            grpc_metadata[grpc_key] = value
        
        # Extract method and path
        method = http_msg.metadata.get("method", "GET")
        path = http_msg.metadata.get("path", "/")
        
        # Map to gRPC service call
        grpc_service = path.split("/")[1] if "/" in path else "DefaultService"
        grpc_method = method.lower()
        
        grpc_msg = ProtocolMessage(
            protocol="grpc",
            headers=grpc_metadata,
            body=http_msg.body,
            metadata={
                "service": grpc_service,
                "method": grpc_method,
                "content_type": "application/grpc"
            }
        )
        
        return grpc_msg
    
    def grpc_to_http(self, grpc_msg: ProtocolMessage) -> ProtocolMessage:
        """Convert gRPC message to HTTP"""
        
        # Map gRPC metadata to HTTP headers
        http_headers = {}
        for key, value in grpc_msg.headers.items():
            # HTTP headers use kebab-case
            http_key = key.replace("_", "-").title()
            http_headers[http_key] = value
        
        # Build HTTP path
        service = grpc_msg.metadata.get("service", "api")
        method = grpc_msg.metadata.get("method", "call")
        path = f"/{service}/{method}"
        
        http_msg = ProtocolMessage(
            protocol="http",
            headers=http_headers,
            body=grpc_msg.body,
            metadata={
                "method": "POST",
                "path": path,
                "status": 200
            }
        )
        
        return http_msg
    
    def http_to_websocket(self, http_msg: ProtocolMessage) -> ProtocolMessage:
        """Convert HTTP to WebSocket message"""
        
        # WebSocket frame
        ws_msg = ProtocolMessage(
            protocol="websocket",
            headers={
                "Upgrade": "websocket",
                "Connection": "Upgrade",
                "Sec-WebSocket-Version": "13"
            },
            body=http_msg.body,
            metadata={
                "opcode": "text",
                "fin": True,
                "masked": False
            }
        )
        
        return ws_msg
    
    def http_to_mqtt(self, http_msg: ProtocolMessage) -> ProtocolMessage:
        """Convert HTTP to MQTT message"""
        
        # Extract topic from path
        path = http_msg.metadata.get("path", "/")
        topic = path.lstrip("/").replace("/", ".")
        
        mqtt_msg = ProtocolMessage(
            protocol="mqtt",
            headers={},
            body=http_msg.body,
            metadata={
                "topic": topic,
                "qos": 1,
                "retain": False
            }
        )
        
        return mqtt_msg
    
    def transform(self, source_msg: ProtocolMessage, target_protocol: str) -> ProtocolMessage:
        """Transform message to target protocol"""
        
        source_protocol = source_msg.protocol
        transform_key = f"{source_protocol}_to_{target_protocol}"
        
        transformers = {
            "http_to_grpc": self.http_to_grpc,
            "grpc_to_http": self.grpc_to_http,
            "http_to_websocket": self.http_to_websocket,
            "http_to_mqtt": self.http_to_mqtt
        }
        
        if transform_key in transformers:
            return transformers[transform_key](source_msg)
        
        # Default: copy message with new protocol
        return ProtocolMessage(
            protocol=target_protocol,
            headers=source_msg.headers,
            body=source_msg.body,
            metadata=source_msg.metadata
        )


def initialize_protocol_adapter_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Initialize protocol adapter"""
    print("\n🔌 Initializing Protocol Adapter...")
    
    print(f"  Supported Protocols:")
    print(f"    • HTTP/HTTPS")
    print(f"    • gRPC")
    print(f"    • WebSocket")
    print(f"    • MQTT")
    print(f"    • REST")
    
    print(f"\n  Transformation Capabilities:")
    print(f"    • HTTP ↔ gRPC")
    print(f"    • HTTP → WebSocket")
    print(f"    • HTTP → MQTT")
    print(f"    • Protocol header mapping")
    print(f"    • Message format conversion")
    
    return {
        **state,
        "source_protocol": "",
        "target_protocol": "",
        "protocol_adapters": {},
        "transformation_log": [],
        "protocol_statistics": {},
        "messages": ["✓ Protocol adapter initialized"]
    }


def transform_http_to_grpc_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Transform HTTP to gRPC"""
    print("\n📡 Transforming HTTP → gRPC...")
    
    adapter = ProtocolAdapter()
    
    # Create HTTP message
    http_msg = ProtocolMessage(
        protocol="http",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
            "User-Agent": "MyClient/1.0"
        },
        body={"user_id": 123, "action": "login"},
        metadata={
            "method": "POST",
            "path": "/users/authenticate",
            "status": 200
        }
    )
    
    print(f"  Source: HTTP POST /users/authenticate")
    print(f"  Headers: {len(http_msg.headers)}")
    print(f"  Body: {http_msg.body}")
    
    # Transform
    grpc_msg = adapter.transform(http_msg, "grpc")
    
    print(f"\n  Target: gRPC")
    print(f"  Service: {grpc_msg.metadata['service']}")
    print(f"  Method: {grpc_msg.metadata['method']}")
    print(f"  Metadata: {list(grpc_msg.headers.keys())}")
    print(f"  Body: {grpc_msg.body}")
    
    transformation_log = [{
        "from_protocol": "http",
        "to_protocol": "grpc",
        "source_message": http_msg.to_dict(),
        "target_message": grpc_msg.to_dict(),
        "timestamp": time.time()
    }]
    
    return {
        **state,
        "source_protocol": "http",
        "target_protocol": "grpc",
        "transformation_log": transformation_log,
        "messages": ["✓ HTTP → gRPC transformation complete"]
    }


def transform_grpc_to_http_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Transform gRPC to HTTP"""
    print("\n📡 Transforming gRPC → HTTP...")
    
    adapter = ProtocolAdapter()
    
    # Create gRPC message
    grpc_msg = ProtocolMessage(
        protocol="grpc",
        headers={
            "authorization": "Bearer token456",
            "content_type": "application/grpc"
        },
        body={"order_id": 789, "status": "confirmed"},
        metadata={
            "service": "orders",
            "method": "update",
            "content_type": "application/grpc"
        }
    )
    
    print(f"  Source: gRPC")
    print(f"  Service: {grpc_msg.metadata['service']}")
    print(f"  Method: {grpc_msg.metadata['method']}")
    print(f"  Body: {grpc_msg.body}")
    
    # Transform
    http_msg = adapter.transform(grpc_msg, "http")
    
    print(f"\n  Target: HTTP {http_msg.metadata['method']} {http_msg.metadata['path']}")
    print(f"  Headers: {list(http_msg.headers.keys())}")
    print(f"  Status: {http_msg.metadata['status']}")
    print(f"  Body: {http_msg.body}")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "from_protocol": "grpc",
        "to_protocol": "http",
        "source_message": grpc_msg.to_dict(),
        "target_message": http_msg.to_dict(),
        "timestamp": time.time()
    })
    
    return {
        **state,
        "transformation_log": transformation_log,
        "messages": ["✓ gRPC → HTTP transformation complete"]
    }


def transform_http_to_websocket_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Transform HTTP to WebSocket"""
    print("\n📡 Transforming HTTP → WebSocket...")
    
    adapter = ProtocolAdapter()
    
    # Create HTTP message
    http_msg = ProtocolMessage(
        protocol="http",
        headers={
            "Content-Type": "application/json"
        },
        body={"event": "message", "data": "Hello WebSocket!"},
        metadata={
            "method": "POST",
            "path": "/chat/send"
        }
    )
    
    print(f"  Source: HTTP POST /chat/send")
    print(f"  Body: {http_msg.body}")
    
    # Transform
    ws_msg = adapter.transform(http_msg, "websocket")
    
    print(f"\n  Target: WebSocket")
    print(f"  Opcode: {ws_msg.metadata['opcode']}")
    print(f"  FIN: {ws_msg.metadata['fin']}")
    print(f"  Body: {ws_msg.body}")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "from_protocol": "http",
        "to_protocol": "websocket",
        "source_message": http_msg.to_dict(),
        "target_message": ws_msg.to_dict(),
        "timestamp": time.time()
    })
    
    return {
        **state,
        "transformation_log": transformation_log,
        "messages": ["✓ HTTP → WebSocket transformation complete"]
    }


def transform_http_to_mqtt_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Transform HTTP to MQTT"""
    print("\n📡 Transforming HTTP → MQTT...")
    
    adapter = ProtocolAdapter()
    
    # Create HTTP message
    http_msg = ProtocolMessage(
        protocol="http",
        headers={},
        body={"temperature": 22.5, "humidity": 65},
        metadata={
            "method": "POST",
            "path": "/sensors/temperature"
        }
    )
    
    print(f"  Source: HTTP POST /sensors/temperature")
    print(f"  Body: {http_msg.body}")
    
    # Transform
    mqtt_msg = adapter.transform(http_msg, "mqtt")
    
    print(f"\n  Target: MQTT")
    print(f"  Topic: {mqtt_msg.metadata['topic']}")
    print(f"  QoS: {mqtt_msg.metadata['qos']}")
    print(f"  Body: {mqtt_msg.body}")
    
    transformation_log = state["transformation_log"].copy()
    transformation_log.append({
        "from_protocol": "http",
        "to_protocol": "mqtt",
        "source_message": http_msg.to_dict(),
        "target_message": mqtt_msg.to_dict(),
        "timestamp": time.time()
    })
    
    return {
        **state,
        "transformation_log": transformation_log,
        "messages": ["✓ HTTP → MQTT transformation complete"]
    }


def analyze_protocol_transformations_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Analyze protocol transformations"""
    print("\n📊 Analyzing Protocol Transformations...")
    
    transformations = state["transformation_log"]
    
    # Analyze transformations
    total_transformations = len(transformations)
    protocols_involved = set()
    
    for trans in transformations:
        protocols_involved.add(trans["from_protocol"])
        protocols_involved.add(trans["to_protocol"])
    
    # Count transformation types
    transformation_types = {}
    for trans in transformations:
        key = f"{trans['from_protocol']}_to_{trans['to_protocol']}"
        transformation_types[key] = transformation_types.get(key, 0) + 1
    
    statistics = {
        "total_transformations": total_transformations,
        "protocols_involved": list(protocols_involved),
        "transformation_types": transformation_types,
        "unique_transformations": len(transformation_types)
    }
    
    print(f"  Total Transformations: {statistics['total_transformations']}")
    print(f"  Protocols Involved: {', '.join(statistics['protocols_involved'])}")
    print(f"  Unique Transformation Types: {statistics['unique_transformations']}")
    
    print(f"\n  Transformation Distribution:")
    for trans_type, count in transformation_types.items():
        print(f"    {trans_type}: {count}")
    
    return {
        **state,
        "protocol_statistics": statistics,
        "messages": ["✓ Protocol transformations analyzed"]
    }


def generate_protocol_transformation_report_agent(state: ProtocolTransformationPattern) -> ProtocolTransformationPattern:
    """Generate protocol transformation report"""
    print("\n" + "="*70)
    print("PROTOCOL TRANSFORMATION REPORT")
    print("="*70)
    
    print(f"\n🔄 Transformations:")
    for i, trans in enumerate(state["transformation_log"], 1):
        print(f"\n  {i}. {trans['from_protocol'].upper()} → {trans['to_protocol'].upper()}")
        
        print(f"     Source Message:")
        src = trans["source_message"]
        print(f"       Protocol: {src['protocol']}")
        print(f"       Headers: {list(src['headers'].keys())}")
        print(f"       Body: {src['body']}")
        
        print(f"     Target Message:")
        tgt = trans["target_message"]
        print(f"       Protocol: {tgt['protocol']}")
        print(f"       Headers: {list(tgt['headers'].keys())}")
        print(f"       Body: {tgt['body']}")
    
    print(f"\n📊 Statistics:")
    stats = state["protocol_statistics"]
    if stats:
        print(f"  Total Transformations: {stats['total_transformations']}")
        print(f"  Protocols: {', '.join(stats['protocols_involved'])}")
        print(f"  Unique Transformations: {stats['unique_transformations']}")
    
    print(f"\n💡 Protocol Transformation Benefits:")
    print("  ✓ System interoperability")
    print("  ✓ Legacy integration")
    print("  ✓ Microservice communication")
    print("  ✓ API gateway functionality")
    print("  ✓ Protocol migration")
    print("  ✓ Multi-protocol support")
    
    print(f"\n🔧 Protocol Features:")
    print("  • HTTP: RESTful, stateless, wide support")
    print("  • gRPC: High performance, streaming, type-safe")
    print("  • WebSocket: Full-duplex, real-time, persistent")
    print("  • MQTT: Lightweight, pub/sub, IoT-focused")
    
    print(f"\n⚙️ Use Cases:")
    print("  • API gateway")
    print("  • Service mesh")
    print("  • Legacy modernization")
    print("  • IoT integration")
    print("  • Real-time systems")
    print("  • Microservices")
    
    print(f"\n🎯 Common Transformations:")
    print("  • REST → gRPC (performance)")
    print("  • HTTP → WebSocket (real-time)")
    print("  • HTTP → MQTT (IoT)")
    print("  • SOAP → REST (modernization)")
    
    print("\n" + "="*70)
    print("✅ Protocol Transformation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_protocol_transformation_graph():
    """Create protocol transformation workflow"""
    workflow = StateGraph(ProtocolTransformationPattern)
    
    workflow.add_node("initialize", initialize_protocol_adapter_agent)
    workflow.add_node("http_to_grpc", transform_http_to_grpc_agent)
    workflow.add_node("grpc_to_http", transform_grpc_to_http_agent)
    workflow.add_node("http_to_ws", transform_http_to_websocket_agent)
    workflow.add_node("http_to_mqtt", transform_http_to_mqtt_agent)
    workflow.add_node("analyze", analyze_protocol_transformations_agent)
    workflow.add_node("report", generate_protocol_transformation_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "http_to_grpc")
    workflow.add_edge("http_to_grpc", "grpc_to_http")
    workflow.add_edge("grpc_to_http", "http_to_ws")
    workflow.add_edge("http_to_ws", "http_to_mqtt")
    workflow.add_edge("http_to_mqtt", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 303: Protocol Transformation MCP Pattern")
    print("="*70)
    
    app = create_protocol_transformation_graph()
    final_state = app.invoke({
        "messages": [],
        "source_protocol": "",
        "target_protocol": "",
        "protocol_adapters": {},
        "transformation_log": [],
        "protocol_statistics": {}
    })
    
    print("\n✅ Protocol Transformation Pattern Complete!")


if __name__ == "__main__":
    main()

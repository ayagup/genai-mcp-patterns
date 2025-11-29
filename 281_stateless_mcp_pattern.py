"""
Pattern 281: Stateless MCP Pattern

This pattern demonstrates stateless agent architecture where each request
is independent and contains all necessary information for processing.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class StatelessState(TypedDict):
    """State for stateless workflow - each request is independent"""
    messages: Annotated[List[str], add]
    request: Dict[str, Any]
    response: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class StatelessProcessor:
    """Processes requests without maintaining state between calls"""
    
    def __init__(self):
        self.supported_operations = ["analyze", "transform", "validate", "compute"]
    
    def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process request independently - no shared state"""
        operation = request.get("operation", "unknown")
        data = request.get("data", {})
        
        if operation not in self.supported_operations:
            return {
                "status": "error",
                "message": f"Unsupported operation: {operation}",
                "result": None
            }
        
        # Process based on operation type
        if operation == "analyze":
            result = self._analyze(data)
        elif operation == "transform":
            result = self._transform(data)
        elif operation == "validate":
            result = self._validate(data)
        else:  # compute
            result = self._compute(data)
        
        return {
            "status": "success",
            "operation": operation,
            "result": result,
            "request_id": request.get("request_id", "unknown")
        }
    
    def _analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze data without maintaining state"""
        text = data.get("text", "")
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "has_numbers": any(c.isdigit() for c in text),
            "has_special_chars": any(not c.isalnum() and not c.isspace() for c in text)
        }
    
    def _transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data independently"""
        text = data.get("text", "")
        operation = data.get("transform_type", "uppercase")
        
        if operation == "uppercase":
            transformed = text.upper()
        elif operation == "lowercase":
            transformed = text.lower()
        elif operation == "reverse":
            transformed = text[::-1]
        else:
            transformed = text
        
        return {"transformed": transformed}
    
    def _validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data without state"""
        rules = data.get("rules", {})
        value = data.get("value", "")
        
        errors = []
        if rules.get("min_length") and len(value) < rules["min_length"]:
            errors.append(f"Too short (min: {rules['min_length']})")
        
        if rules.get("max_length") and len(value) > rules["max_length"]:
            errors.append(f"Too long (max: {rules['max_length']})")
        
        if rules.get("required") and not value:
            errors.append("Value is required")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    def _compute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute result from input data only"""
        numbers = data.get("numbers", [])
        operation = data.get("compute_type", "sum")
        
        if not numbers:
            return {"result": 0}
        
        if operation == "sum":
            result = sum(numbers)
        elif operation == "average":
            result = sum(numbers) / len(numbers)
        elif operation == "max":
            result = max(numbers)
        elif operation == "min":
            result = min(numbers)
        else:
            result = 0
        
        return {"result": result}


def receive_request_agent(state: StatelessState) -> StatelessState:
    """Receive and validate incoming request"""
    print("\n📥 Receiving Request...")
    
    # Simulate incoming request (no previous state needed)
    request = {
        "request_id": "REQ-001",
        "operation": "analyze",
        "data": {
            "text": "Hello World! This is a stateless pattern example with 123 numbers."
        },
        "timestamp": "2025-11-29T10:00:00Z"
    }
    
    print(f"  Request ID: {request['request_id']}")
    print(f"  Operation: {request['operation']}")
    print(f"  Timestamp: {request['timestamp']}")
    
    return {
        **state,
        "request": request,
        "messages": [f"✓ Received request {request['request_id']}"]
    }


def process_request_agent(state: StatelessState) -> StatelessState:
    """Process request without maintaining state"""
    print("\n⚙️ Processing Request (Stateless)...")
    
    processor = StatelessProcessor()
    response = processor.process_request(state["request"])
    
    print(f"  Status: {response['status']}")
    print(f"  Operation: {response['operation']}")
    
    if response["status"] == "success":
        print(f"  Result: {response['result']}")
    
    return {
        **state,
        "response": response,
        "messages": [f"✓ Processed {response['operation']} operation"]
    }


def add_metadata_agent(state: StatelessState) -> StatelessState:
    """Add processing metadata (derived from current request only)"""
    print("\n📝 Adding Metadata...")
    
    import time
    
    metadata = {
        "processing_time_ms": 45,  # Simulated
        "processor_id": "PROC-STATELESS-01",
        "request_id": state["request"]["request_id"],
        "timestamp": time.time(),
        "stateless": True,
        "no_session": True
    }
    
    print(f"  Processor: {metadata['processor_id']}")
    print(f"  Processing Time: {metadata['processing_time_ms']}ms")
    print(f"  Stateless: {metadata['stateless']}")
    
    return {
        **state,
        "processing_metadata": metadata,
        "messages": ["✓ Metadata added"]
    }


def generate_stateless_report_agent(state: StatelessState) -> StatelessState:
    """Generate report for stateless processing"""
    print("\n" + "="*70)
    print("STATELESS PROCESSING REPORT")
    print("="*70)
    
    request = state["request"]
    response = state["response"]
    metadata = state["processing_metadata"]
    
    print(f"\n📥 Request Information:")
    print(f"  Request ID: {request['request_id']}")
    print(f"  Operation: {request['operation']}")
    print(f"  Timestamp: {request['timestamp']}")
    
    print(f"\n⚙️ Processing Details:")
    print(f"  Status: {response['status']}")
    print(f"  Processor: {metadata['processor_id']}")
    print(f"  Processing Time: {metadata['processing_time_ms']}ms")
    print(f"  Stateless: {metadata['stateless']}")
    
    print(f"\n📊 Request Data:")
    for key, value in request.get("data", {}).items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ Processing Result:")
    if response["status"] == "success":
        result = response["result"]
        for key, value in result.items():
            print(f"  {key}: {value}")
    else:
        print(f"  Error: {response.get('message', 'Unknown error')}")
    
    print(f"\n💡 Stateless Architecture Benefits:")
    print("  ✓ No session management required")
    print("  ✓ Each request is independent")
    print("  ✓ Easy horizontal scaling")
    print("  ✓ No state synchronization needed")
    print("  ✓ Simple error recovery")
    print("  ✓ Predictable behavior")
    print("  ✓ High availability")
    print("  ✓ Load balancer friendly")
    
    print(f"\n📋 Key Characteristics:")
    print("  • No memory of previous requests")
    print("  • All context in request")
    print("  • Idempotent operations possible")
    print("  • Stateless scalability")
    print("  • Request-response pattern")
    
    # Demonstrate multiple independent requests
    print(f"\n🔄 Processing Multiple Independent Requests:")
    
    processor = StatelessProcessor()
    test_requests = [
        {"request_id": "REQ-002", "operation": "transform", 
         "data": {"text": "hello", "transform_type": "uppercase"}},
        {"request_id": "REQ-003", "operation": "compute", 
         "data": {"numbers": [1, 2, 3, 4, 5], "compute_type": "average"}},
        {"request_id": "REQ-004", "operation": "validate", 
         "data": {"value": "test", "rules": {"min_length": 3, "max_length": 10}}}
    ]
    
    for req in test_requests:
        result = processor.process_request(req)
        print(f"\n  Request {req['request_id']} ({req['operation']}):")
        print(f"    Status: {result['status']}")
        print(f"    Result: {result['result']}")
    
    print("\n" + "="*70)
    print("✅ Stateless Processing Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_stateless_graph():
    """Create stateless processing workflow"""
    workflow = StateGraph(StatelessState)
    
    workflow.add_node("receive", receive_request_agent)
    workflow.add_node("process", process_request_agent)
    workflow.add_node("metadata", add_metadata_agent)
    workflow.add_node("report", generate_stateless_report_agent)
    
    workflow.add_edge(START, "receive")
    workflow.add_edge("receive", "process")
    workflow.add_edge("process", "metadata")
    workflow.add_edge("metadata", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 281: Stateless MCP Pattern")
    print("="*70)
    print("\n💡 Demonstrating stateless architecture:")
    print("  • No state maintained between requests")
    print("  • Each request is self-contained")
    print("  • Horizontally scalable")
    
    app = create_stateless_graph()
    final_state = app.invoke({
        "messages": [],
        "request": {},
        "response": {},
        "processing_metadata": {}
    })
    
    print("\n✅ Stateless Pattern Complete!")


if __name__ == "__main__":
    main()

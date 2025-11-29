"""
Pattern 290: CQRS MCP Pattern

This pattern demonstrates Command Query Responsibility Segregation (CQRS)
where read and write operations use separate models and data stores.
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time
from datetime import datetime


class CQRSPattern(TypedDict):
    """State for CQRS pattern"""
    messages: Annotated[List[str], add]
    write_model: Dict[str, Any]
    read_models: Dict[str, Any]
    commands: List[Dict[str, Any]]
    queries: List[Dict[str, Any]]
    performance_metrics: Dict[str, Any]


class Command:
    """Represents a command (write operation)"""
    
    def __init__(self, command_type: str, data: Dict[str, Any]):
        self.command_id = f"cmd_{int(time.time() * 1000)}"
        self.command_type = command_type
        self.data = data
        self.timestamp = time.time()
        self.executed = False
    
    def to_dict(self):
        return {
            "command_id": self.command_id,
            "command_type": self.command_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "executed": self.executed
        }


class Query:
    """Represents a query (read operation)"""
    
    def __init__(self, query_type: str, parameters: Dict[str, Any]):
        self.query_id = f"qry_{int(time.time() * 1000)}"
        self.query_type = query_type
        self.parameters = parameters
        self.timestamp = time.time()
        self.result = None
    
    def to_dict(self):
        return {
            "query_id": self.query_id,
            "query_type": self.query_type,
            "parameters": self.parameters,
            "timestamp": self.timestamp,
            "result": self.result
        }


class WriteModel:
    """Write model - optimized for commands"""
    
    def __init__(self):
        self.data = {}
        self.version = 0
        self.command_history = []
    
    def execute_command(self, command: Command):
        """Execute write command"""
        self.version += 1
        
        if command.command_type == "CreateOrder":
            order_id = command.data["order_id"]
            self.data[order_id] = {
                "order_id": order_id,
                "customer_id": command.data["customer_id"],
                "items": command.data["items"],
                "total": command.data["total"],
                "status": "pending",
                "created_at": time.time(),
                "version": self.version
            }
        
        elif command.command_type == "UpdateOrderStatus":
            order_id = command.data["order_id"]
            if order_id in self.data:
                self.data[order_id]["status"] = command.data["status"]
                self.data[order_id]["updated_at"] = time.time()
                self.data[order_id]["version"] = self.version
        
        elif command.command_type == "AddOrderItem":
            order_id = command.data["order_id"]
            if order_id in self.data:
                self.data[order_id]["items"].append(command.data["item"])
                self.data[order_id]["total"] += command.data["item"]["price"]
                self.data[order_id]["version"] = self.version
        
        command.executed = True
        self.command_history.append(command)


class ReadModel:
    """Read model - optimized for queries"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.data = {}
        self.indexes = {}
    
    def build_from_write_model(self, write_data: Dict[str, Any]):
        """Build optimized read model from write model"""
        if self.model_name == "order_summary":
            # Optimized for order summaries
            self.data = {
                order_id: {
                    "order_id": order["order_id"],
                    "customer_id": order["customer_id"],
                    "item_count": len(order["items"]),
                    "total": order["total"],
                    "status": order["status"]
                }
                for order_id, order in write_data.items()
            }
        
        elif self.model_name == "customer_orders":
            # Indexed by customer
            self.indexes = {}
            for order_id, order in write_data.items():
                customer_id = order["customer_id"]
                if customer_id not in self.indexes:
                    self.indexes[customer_id] = []
                self.indexes[customer_id].append({
                    "order_id": order_id,
                    "total": order["total"],
                    "status": order["status"]
                })
        
        elif self.model_name == "order_analytics":
            # Aggregated data
            total_orders = len(write_data)
            total_revenue = sum(order["total"] for order in write_data.values())
            status_counts = {}
            
            for order in write_data.values():
                status = order["status"]
                status_counts[status] = status_counts.get(status, 0) + 1
            
            self.data = {
                "total_orders": total_orders,
                "total_revenue": total_revenue,
                "average_order_value": total_revenue / max(total_orders, 1),
                "status_distribution": status_counts
            }
    
    def execute_query(self, query: Query):
        """Execute read query"""
        if query.query_type == "GetOrderSummary":
            order_id = query.parameters.get("order_id")
            query.result = self.data.get(order_id)
        
        elif query.query_type == "GetCustomerOrders":
            customer_id = query.parameters.get("customer_id")
            query.result = self.indexes.get(customer_id, [])
        
        elif query.query_type == "GetAnalytics":
            query.result = self.data
        
        return query.result


def initialize_cqrs_agent(state: CQRSPattern) -> CQRSPattern:
    """Initialize CQRS components"""
    print("\n🔀 Initializing CQRS Pattern...")
    
    write_model = WriteModel()
    
    print(f"  Write Model: Initialized")
    print(f"  Read Models: Creating...")
    print(f"    • order_summary (optimized for lookups)")
    print(f"    • customer_orders (indexed by customer)")
    print(f"    • order_analytics (aggregated data)")
    
    print(f"\n  Features:")
    print(f"    • Separate read/write models")
    print(f"    • Optimized queries")
    print(f"    • Scalable reads")
    print(f"    • Eventual consistency")
    
    return {
        **state,
        "write_model": {},
        "read_models": {},
        "commands": [],
        "queries": [],
        "performance_metrics": {},
        "messages": ["✓ CQRS initialized"]
    }


def execute_commands_agent(state: CQRSPattern) -> CQRSPattern:
    """Execute write commands"""
    print("\n✍️ Executing Commands (Writes)...")
    
    write_model = WriteModel()
    
    # Create commands
    commands_to_execute = [
        Command("CreateOrder", {
            "order_id": "ORD001",
            "customer_id": "CUST123",
            "items": [
                {"product": "Laptop", "quantity": 1, "price": 1200}
            ],
            "total": 1200
        }),
        Command("CreateOrder", {
            "order_id": "ORD002",
            "customer_id": "CUST123",
            "items": [
                {"product": "Mouse", "quantity": 2, "price": 25}
            ],
            "total": 50
        }),
        Command("CreateOrder", {
            "order_id": "ORD003",
            "customer_id": "CUST456",
            "items": [
                {"product": "Monitor", "quantity": 1, "price": 300}
            ],
            "total": 300
        }),
        Command("UpdateOrderStatus", {
            "order_id": "ORD001",
            "status": "processing"
        }),
        Command("AddOrderItem", {
            "order_id": "ORD001",
            "item": {"product": "Keyboard", "quantity": 1, "price": 80}
        }),
        Command("UpdateOrderStatus", {
            "order_id": "ORD002",
            "status": "shipped"
        })
    ]
    
    start_time = time.time()
    
    for command in commands_to_execute:
        time.sleep(0.01)
        write_model.execute_command(command)
        print(f"  ✓ {command.command_type}: {command.data.get('order_id', 'N/A')}")
    
    write_duration = time.time() - start_time
    
    print(f"\n  Commands Executed: {len(commands_to_execute)}")
    print(f"  Write Model Version: {write_model.version}")
    print(f"  Execution Time: {write_duration:.3f}s")
    
    command_dicts = [cmd.to_dict() for cmd in write_model.command_history]
    
    return {
        **state,
        "write_model": write_model.data,
        "commands": command_dicts,
        "performance_metrics": {"write_duration": write_duration},
        "messages": [f"✓ Executed {len(commands_to_execute)} commands"]
    }


def build_read_models_agent(state: CQRSPattern) -> CQRSPattern:
    """Build optimized read models"""
    print("\n🔨 Building Read Models...")
    
    write_data = state["write_model"]
    
    # Create specialized read models
    read_models = {}
    
    # Model 1: Order Summary
    print(f"\n  Building 'order_summary' model...")
    order_summary = ReadModel("order_summary")
    order_summary.build_from_write_model(write_data)
    read_models["order_summary"] = order_summary.data
    print(f"    Records: {len(order_summary.data)}")
    
    # Model 2: Customer Orders Index
    print(f"\n  Building 'customer_orders' model...")
    customer_orders = ReadModel("customer_orders")
    customer_orders.build_from_write_model(write_data)
    read_models["customer_orders"] = customer_orders.indexes
    print(f"    Customers indexed: {len(customer_orders.indexes)}")
    
    # Model 3: Analytics
    print(f"\n  Building 'order_analytics' model...")
    analytics = ReadModel("order_analytics")
    analytics.build_from_write_model(write_data)
    read_models["order_analytics"] = analytics.data
    print(f"    Analytics computed: {len(analytics.data)} metrics")
    
    print(f"\n  Total Read Models: {len(read_models)}")
    
    return {
        **state,
        "read_models": read_models,
        "messages": [f"✓ Built {len(read_models)} read models"]
    }


def execute_queries_agent(state: CQRSPattern) -> CQRSPattern:
    """Execute read queries"""
    print("\n📖 Executing Queries (Reads)...")
    
    # Recreate read models
    order_summary = ReadModel("order_summary")
    order_summary.data = state["read_models"]["order_summary"]
    
    customer_orders = ReadModel("customer_orders")
    customer_orders.indexes = state["read_models"]["customer_orders"]
    
    analytics = ReadModel("order_analytics")
    analytics.data = state["read_models"]["order_analytics"]
    
    # Create queries
    queries_to_execute = [
        (Query("GetOrderSummary", {"order_id": "ORD001"}), order_summary),
        (Query("GetCustomerOrders", {"customer_id": "CUST123"}), customer_orders),
        (Query("GetAnalytics", {}), analytics),
        (Query("GetOrderSummary", {"order_id": "ORD002"}), order_summary)
    ]
    
    start_time = time.time()
    executed_queries = []
    
    for query, model in queries_to_execute:
        time.sleep(0.01)
        result = model.execute_query(query)
        executed_queries.append(query.to_dict())
        print(f"  ✓ {query.query_type}: {query.parameters}")
        if result:
            print(f"    Result: {result}")
    
    read_duration = time.time() - start_time
    
    print(f"\n  Queries Executed: {len(queries_to_execute)}")
    print(f"  Read Duration: {read_duration:.3f}s")
    
    metrics = state["performance_metrics"].copy()
    metrics["read_duration"] = read_duration
    metrics["read_write_ratio"] = len(queries_to_execute) / max(len(state["commands"]), 1)
    
    return {
        **state,
        "queries": executed_queries,
        "performance_metrics": metrics,
        "messages": [f"✓ Executed {len(queries_to_execute)} queries"]
    }


def generate_cqrs_report_agent(state: CQRSPattern) -> CQRSPattern:
    """Generate CQRS report"""
    print("\n" + "="*70)
    print("CQRS PATTERN REPORT")
    print("="*70)
    
    print(f"\n📝 Write Model:")
    print(f"  Orders: {len(state['write_model'])}")
    print(f"  Commands Executed: {len(state['commands'])}")
    
    for order_id, order in list(state['write_model'].items())[:3]:
        print(f"\n  Order {order_id}:")
        print(f"    Customer: {order['customer_id']}")
        print(f"    Items: {len(order['items'])}")
        print(f"    Total: ${order['total']}")
        print(f"    Status: {order['status']}")
    
    print(f"\n📖 Read Models:")
    print(f"  Total Models: {len(state['read_models'])}")
    
    for model_name, model_data in state['read_models'].items():
        print(f"\n  {model_name}:")
        if isinstance(model_data, dict):
            if model_name == "order_analytics":
                print(f"    Total Orders: {model_data.get('total_orders', 0)}")
                print(f"    Total Revenue: ${model_data.get('total_revenue', 0)}")
                print(f"    Avg Order Value: ${model_data.get('average_order_value', 0):.2f}")
                print(f"    Status Distribution: {model_data.get('status_distribution', {})}")
            else:
                print(f"    Records: {len(model_data)}")
    
    print(f"\n⚡ Performance Metrics:")
    metrics = state['performance_metrics']
    print(f"  Write Duration: {metrics.get('write_duration', 0):.3f}s")
    print(f"  Read Duration: {metrics.get('read_duration', 0):.3f}s")
    print(f"  Read/Write Ratio: {metrics.get('read_write_ratio', 0):.2f}")
    
    print(f"\n📋 Command Log:")
    print(f"  Total Commands: {len(state['commands'])}")
    for cmd in state['commands'][:5]:
        print(f"  • {cmd['command_type']}: {cmd['data'].get('order_id', 'N/A')}")
    
    print(f"\n🔍 Query Log:")
    print(f"  Total Queries: {len(state['queries'])}")
    for query in state['queries'][:5]:
        print(f"  • {query['query_type']}: {query['parameters']}")
    
    print(f"\n💡 CQRS Benefits:")
    print("  ✓ Optimized read operations")
    print("  ✓ Scalable queries")
    print("  ✓ Independent scaling")
    print("  ✓ Specialized models")
    print("  ✓ Performance optimization")
    print("  ✓ Flexibility")
    
    print(f"\n🔧 Pattern Components:")
    print("  • Command handlers")
    print("  • Write model (normalized)")
    print("  • Read models (denormalized)")
    print("  • Query handlers")
    print("  • Model synchronization")
    
    print(f"\n⚙️ Use Cases:")
    print("  • High-read systems")
    print("  • Complex queries")
    print("  • Reporting systems")
    print("  • Analytics platforms")
    print("  • E-commerce")
    print("  • Content management")
    
    print(f"\n🎯 Read Model Types:")
    print("  • Summary views")
    print("  • Indexed lookups")
    print("  • Aggregated data")
    print("  • Cached results")
    print("  • Materialized views")
    
    print(f"\n⚠️ Considerations:")
    print("  • Eventual consistency")
    print("  • Model synchronization")
    print("  • Increased complexity")
    print("  • Data duplication")
    print("  • Sync delays")
    
    print("\n" + "="*70)
    print("✅ CQRS Pattern Complete!")
    print("="*70)
    print("\n🎉 STATE MANAGEMENT CATEGORY COMPLETE! (281-290)")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_cqrs_graph():
    """Create CQRS workflow"""
    workflow = StateGraph(CQRSPattern)
    
    workflow.add_node("initialize", initialize_cqrs_agent)
    workflow.add_node("commands", execute_commands_agent)
    workflow.add_node("build_reads", build_read_models_agent)
    workflow.add_node("queries", execute_queries_agent)
    workflow.add_node("report", generate_cqrs_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "commands")
    workflow.add_edge("commands", "build_reads")
    workflow.add_edge("build_reads", "queries")
    workflow.add_edge("queries", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 290: CQRS MCP Pattern")
    print("="*70)
    
    app = create_cqrs_graph()
    final_state = app.invoke({
        "messages": [],
        "write_model": {},
        "read_models": {},
        "commands": [],
        "queries": [],
        "performance_metrics": {}
    })
    
    print("\n✅ CQRS Pattern Complete!")


if __name__ == "__main__":
    main()

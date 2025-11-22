"""
Saga MCP Pattern

This pattern demonstrates long-running distributed transactions with compensating 
actions for rollback when failures occur.

Key Features:
- Sequential execution of distributed operations
- Compensating transactions for rollback
- Forward recovery or backward compensation
- Maintains eventual consistency across services
"""

from typing import TypedDict, Sequence, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


# Define the state
class SagaState(TypedDict):
    """State for saga pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    saga_name: str
    steps_completed: list[str]  # List of successfully completed steps
    current_step: int
    total_steps: int
    compensation_needed: bool
    compensations_completed: list[str]
    saga_status: str  # "in_progress", "completed", "compensating", "failed"
    final_result: str


# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)


# Saga Coordinator
def saga_coordinator_start(state: SagaState) -> SagaState:
    """Coordinator initiates the saga"""
    saga_name = state["saga_name"]
    
    system_message = SystemMessage(content="""You are a saga coordinator managing a long-running 
    distributed transaction. Explain the saga steps and initiate execution.""")
    
    user_message = HumanMessage(content=f"""Initiating saga: {saga_name}
    
    This saga consists of multiple distributed operations. Each step must succeed, or 
    compensating transactions will roll back completed steps.
    
    Begin the saga execution.""")
    
    response = llm.invoke([system_message, user_message])
    
    return {
        "messages": [AIMessage(content=f"🎯 Saga Coordinator: {response.content}")],
        "saga_status": "in_progress",
        "current_step": 1
    }


# Step 1: Reserve Inventory
def reserve_inventory_step(state: SagaState) -> SagaState:
    """Step 1: Reserve inventory in warehouse"""
    saga_status = state.get("saga_status", "")
    
    if saga_status == "compensating":
        # Compensating transaction: Release inventory
        system_message = SystemMessage(content="""You are the inventory service executing a 
        COMPENSATING transaction. Release the reserved inventory and restore availability.""")
        
        user_message = HumanMessage(content="""COMPENSATE: Release reserved inventory
        
        Undo the inventory reservation and make items available again.""")
        
        response = llm.invoke([system_message, user_message])
        
        compensations = state.get("compensations_completed", [])
        compensations.append("inventory_released")
        
        return {
            "messages": [AIMessage(content=f"🔄 Inventory Service (COMPENSATE): {response.content}")],
            "compensations_completed": compensations
        }
    
    else:
        # Forward transaction: Reserve inventory
        system_message = SystemMessage(content="""You are the inventory service. Reserve the 
        requested items for the order. Check availability and lock inventory.""")
        
        user_message = HumanMessage(content="""EXECUTE: Reserve inventory for order
        
        Items: 2x Widget A, 1x Widget B
        Check availability and reserve items.""")
        
        response = llm.invoke([system_message, user_message])
        
        # Simulate success
        steps_completed = state.get("steps_completed", [])
        steps_completed.append("inventory_reserved")
        
        return {
            "messages": [AIMessage(content=f"📦 Inventory Service: {response.content}")],
            "steps_completed": steps_completed,
            "current_step": 2
        }


# Step 2: Process Payment
def process_payment_step(state: SagaState) -> SagaState:
    """Step 2: Process payment"""
    saga_status = state.get("saga_status", "")
    
    if saga_status == "compensating":
        # Compensating transaction: Refund payment
        system_message = SystemMessage(content="""You are the payment service executing a 
        COMPENSATING transaction. Issue a full refund to the customer.""")
        
        user_message = HumanMessage(content="""COMPENSATE: Refund payment
        
        Issue full refund for the failed transaction.""")
        
        response = llm.invoke([system_message, user_message])
        
        compensations = state.get("compensations_completed", [])
        compensations.append("payment_refunded")
        
        return {
            "messages": [AIMessage(content=f"🔄 Payment Service (COMPENSATE): {response.content}")],
            "compensations_completed": compensations
        }
    
    else:
        # Forward transaction: Process payment
        system_message = SystemMessage(content="""You are the payment service. Process the 
        customer's payment. Validate payment method and charge the amount.""")
        
        user_message = HumanMessage(content="""EXECUTE: Process payment
        
        Amount: $150.00
        Payment Method: Credit Card ending in 1234
        Process the payment.""")
        
        response = llm.invoke([system_message, user_message])
        
        # Simulate success
        steps_completed = state.get("steps_completed", [])
        steps_completed.append("payment_processed")
        
        return {
            "messages": [AIMessage(content=f"💳 Payment Service: {response.content}")],
            "steps_completed": steps_completed,
            "current_step": 3
        }


# Step 3: Create Shipment
def create_shipment_step(state: SagaState) -> SagaState:
    """Step 3: Create shipment"""
    saga_status = state.get("saga_status", "")
    
    if saga_status == "compensating":
        # Compensating transaction: Cancel shipment
        system_message = SystemMessage(content="""You are the shipping service executing a 
        COMPENSATING transaction. Cancel the shipment and notify the carrier.""")
        
        user_message = HumanMessage(content="""COMPENSATE: Cancel shipment
        
        Cancel the shipping label and notify carrier of cancellation.""")
        
        response = llm.invoke([system_message, user_message])
        
        compensations = state.get("compensations_completed", [])
        compensations.append("shipment_cancelled")
        
        return {
            "messages": [AIMessage(content=f"🔄 Shipping Service (COMPENSATE): {response.content}")],
            "compensations_completed": compensations
        }
    
    else:
        # Forward transaction: Create shipment
        system_message = SystemMessage(content="""You are the shipping service. Create a 
        shipment for the order. Generate shipping label and schedule pickup.""")
        
        user_message = HumanMessage(content="""EXECUTE: Create shipment
        
        Destination: 123 Main St, Anytown, USA
        Items: 2x Widget A, 1x Widget B
        Create shipment and generate tracking number.""")
        
        response = llm.invoke([system_message, user_message])
        
        # Simulate potential failure here (for demo purposes)
        # In real scenario, this could fail due to invalid address, etc.
        import random
        success = random.random() > 0.3  # 70% success rate
        
        if success:
            steps_completed = state.get("steps_completed", [])
            steps_completed.append("shipment_created")
            
            return {
                "messages": [AIMessage(content=f"🚚 Shipping Service: {response.content}")],
                "steps_completed": steps_completed,
                "current_step": 4
            }
        else:
            # Failure! Need to compensate
            return {
                "messages": [AIMessage(content=f"🚚 Shipping Service: ❌ FAILED - Invalid shipping address. Cannot create shipment.")],
                "compensation_needed": True,
                "saga_status": "compensating"
            }


# Step 4: Send Confirmation
def send_confirmation_step(state: SagaState) -> SagaState:
    """Step 4: Send order confirmation"""
    saga_status = state.get("saga_status", "")
    
    if saga_status == "compensating":
        # Compensating transaction: Send cancellation email
        system_message = SystemMessage(content="""You are the notification service executing a 
        COMPENSATING transaction. Send order cancellation notification to customer.""")
        
        user_message = HumanMessage(content="""COMPENSATE: Send cancellation notification
        
        Inform customer that order was cancelled and refund is being processed.""")
        
        response = llm.invoke([system_message, user_message])
        
        compensations = state.get("compensations_completed", [])
        compensations.append("cancellation_sent")
        
        return {
            "messages": [AIMessage(content=f"🔄 Notification Service (COMPENSATE): {response.content}")],
            "compensations_completed": compensations
        }
    
    else:
        # Forward transaction: Send confirmation
        system_message = SystemMessage(content="""You are the notification service. Send order 
        confirmation email to customer with order details and tracking information.""")
        
        user_message = HumanMessage(content="""EXECUTE: Send order confirmation
        
        Customer: customer@example.com
        Order ID: ORD-2024-12345
        Send confirmation with order details.""")
        
        response = llm.invoke([system_message, user_message])
        
        steps_completed = state.get("steps_completed", [])
        steps_completed.append("confirmation_sent")
        
        return {
            "messages": [AIMessage(content=f"📧 Notification Service: {response.content}")],
            "steps_completed": steps_completed,
            "current_step": 5,
            "saga_status": "completed"
        }


# Saga Finalizer
def saga_finalizer(state: SagaState) -> SagaState:
    """Finalize the saga with success or failure report"""
    saga_status = state.get("saga_status", "")
    steps_completed = state.get("steps_completed", [])
    compensations = state.get("compensations_completed", [])
    
    if saga_status == "completed":
        result = "SUCCESS"
        summary = f"""
        ✅ SAGA COMPLETED SUCCESSFULLY
        
        Saga: {state['saga_name']}
        
        Steps Completed:
        {chr(10).join([f'  ✓ {step}' for step in steps_completed])}
        
        Total Steps: {len(steps_completed)}/{state['total_steps']}
        
        Order has been successfully processed and customer notified.
        """
    else:
        result = "COMPENSATED"
        summary = f"""
        🔄 SAGA COMPENSATED (ROLLED BACK)
        
        Saga: {state['saga_name']}
        
        Steps Completed Before Failure:
        {chr(10).join([f'  ✓ {step}' for step in steps_completed])}
        
        Compensating Transactions Executed:
        {chr(10).join([f'  ↩️  {comp}' for comp in compensations])}
        
        System returned to consistent state. All changes have been rolled back.
        Customer will be notified of cancellation and refund.
        """
    
    return {
        "messages": [AIMessage(content=f"🎯 Saga Coordinator (FINAL):\n{summary}")],
        "final_result": result
    }


# Routing logic
def should_compensate(state: SagaState) -> str:
    """Determine if compensation is needed"""
    return "compensate" if state.get("compensation_needed", False) else "continue"


def check_saga_complete(state: SagaState) -> str:
    """Check if saga is complete"""
    saga_status = state.get("saga_status", "")
    if saga_status == "completed":
        return "finalize"
    elif saga_status == "compensating":
        compensations = state.get("compensations_completed", [])
        steps_completed = state.get("steps_completed", [])
        # All completed steps have been compensated
        if len(compensations) >= len(steps_completed):
            return "finalize"
        else:
            return "compensate_more"
    return "continue"


# Build the graph
def build_saga_graph():
    """Build the saga MCP pattern graph"""
    workflow = StateGraph(SagaState)
    
    # Add nodes
    workflow.add_node("coordinator", saga_coordinator_start)
    workflow.add_node("inventory", reserve_inventory_step)
    workflow.add_node("payment", process_payment_step)
    workflow.add_node("shipping", create_shipment_step)
    workflow.add_node("notification", send_confirmation_step)
    workflow.add_node("finalize", saga_finalizer)
    
    # Forward flow
    workflow.add_edge(START, "coordinator")
    workflow.add_edge("coordinator", "inventory")
    workflow.add_edge("inventory", "payment")
    workflow.add_edge("payment", "shipping")
    
    # After shipping, check if compensation needed
    workflow.add_conditional_edges(
        "shipping",
        should_compensate,
        {
            "continue": "notification",
            "compensate": "notification"  # Start compensation from last step
        }
    )
    
    workflow.add_conditional_edges(
        "notification",
        check_saga_complete,
        {
            "finalize": "finalize",
            "continue": "finalize",
            "compensate_more": "shipping"  # Continue backward compensation
        }
    )
    
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = build_saga_graph()
    
    print("=== E-Commerce Order Saga Pattern ===\n")
    print("This saga demonstrates a distributed order processing workflow.")
    print("If any step fails, compensating transactions will roll back all changes.\n")
    
    initial_state = {
        "messages": [],
        "saga_name": "E-Commerce Order Processing",
        "steps_completed": [],
        "current_step": 0,
        "total_steps": 4,
        "compensation_needed": False,
        "compensations_completed": [],
        "saga_status": "",
        "final_result": ""
    }
    
    # Run the saga
    result = graph.invoke(initial_state)
    
    print("\n=== Saga Execution Log ===")
    for msg in result["messages"]:
        print(f"\n{msg.content}")
    
    print(f"\n\n{'='*70}")
    print(f"Final Saga Result: {result['final_result']}")
    print(f"Steps Completed: {len(result['steps_completed'])}/{result['total_steps']}")
    if result['compensations_completed']:
        print(f"Compensations Executed: {len(result['compensations_completed'])}")

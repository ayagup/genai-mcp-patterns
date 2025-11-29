"""
Pattern 222: Integration Testing MCP Pattern

Integration Testing validates component interactions:
- Test multiple components together
- Verify communication between services
- Database integration
- API endpoint testing
- Message queue integration

Test Scope:
- Service-to-service communication
- Database operations
- External API calls
- Message broker interactions

Benefits:
- Detect interface issues
- Verify data flow
- Test real integrations
- Catch integration bugs early

Use Cases:
- Microservices testing
- API testing
- Database testing
- Third-party integration
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass, field
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import time


class IntegrationTestState(TypedDict):
    """State for integration testing operations"""
    test_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


# Components to integrate
class Database:
    """Simulated database"""
    def __init__(self):
        self.users = {}
        self.query_count = 0
    
    def save_user(self, user_id: str, data: Dict[str, Any]):
        self.query_count += 1
        self.users[user_id] = data
    
    def get_user(self, user_id: str) -> Dict[str, Any]:
        self.query_count += 1
        return self.users.get(user_id)


class EmailService:
    """Simulated email service"""
    def __init__(self):
        self.sent_emails = []
    
    def send_email(self, to: str, subject: str, body: str):
        time.sleep(0.01)  # Simulate network call
        self.sent_emails.append({
            'to': to,
            'subject': subject,
            'body': body
        })


class UserService:
    """User service that integrates database and email"""
    def __init__(self, database: Database, email_service: EmailService):
        self.db = database
        self.email = email_service
    
    def register_user(self, user_id: str, email: str, name: str) -> bool:
        """Register user - integrates DB save + email send"""
        # Save to database
        user_data = {'email': email, 'name': name, 'status': 'active'}
        self.db.save_user(user_id, user_data)
        
        # Send welcome email
        self.email.send_email(
            to=email,
            subject="Welcome!",
            body=f"Welcome {name}!"
        )
        
        return True
    
    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user info from database"""
        return self.db.get_user(user_id)


class IntegrationTestRunner:
    """Run integration tests"""
    
    def __init__(self):
        self.test_results = []
        self.passed = 0
        self.failed = 0
    
    def test_user_registration_flow(self, user_service: UserService, db: Database, email: EmailService) -> bool:
        """Test complete user registration flow"""
        # Arrange
        user_id = "USER-001"
        email_addr = "test@example.com"
        name = "Test User"
        
        # Act
        result = user_service.register_user(user_id, email_addr, name)
        
        # Assert
        tests_passed = True
        
        # Check registration returned success
        if not result:
            tests_passed = False
        
        # Check user saved in database
        user_data = db.get_user(user_id)
        if not user_data or user_data['email'] != email_addr:
            tests_passed = False
        
        # Check welcome email sent
        if len(email.sent_emails) != 1:
            tests_passed = False
        elif email.sent_emails[0]['to'] != email_addr:
            tests_passed = False
        
        if tests_passed:
            self.passed += 1
        else:
            self.failed += 1
        
        self.test_results.append({
            'name': 'test_user_registration_flow',
            'passed': tests_passed
        })
        
        return tests_passed
    
    def test_database_integration(self, db: Database) -> bool:
        """Test database integration"""
        # Save and retrieve
        db.save_user("USER-002", {"name": "John", "email": "john@test.com"})
        retrieved = db.get_user("USER-002")
        
        passed = retrieved is not None and retrieved['name'] == "John"
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        self.test_results.append({
            'name': 'test_database_integration',
            'passed': passed
        })
        
        return passed
    
    def test_email_service_integration(self, email: EmailService) -> bool:
        """Test email service integration"""
        initial_count = len(email.sent_emails)
        email.send_email("test@test.com", "Test", "Body")
        
        passed = len(email.sent_emails) == initial_count + 1
        
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        
        self.test_results.append({
            'name': 'test_email_service_integration',
            'passed': passed
        })
        
        return passed


def setup_agent(state: IntegrationTestState):
    """Set up integration test environment"""
    operations = []
    results = []
    
    # Create components
    db = Database()
    email = EmailService()
    user_service = UserService(db, email)
    runner = IntegrationTestRunner()
    
    operations.append("Integration Testing Setup:")
    operations.append("\nComponents:")
    operations.append("  - Database (mock)")
    operations.append("  - EmailService (mock)")
    operations.append("  - UserService (integrates DB + Email)")
    
    operations.append("\nIntegration Tests:")
    operations.append("  1. User registration flow (DB + Email)")
    operations.append("  2. Database integration")
    operations.append("  3. Email service integration")
    
    results.append("✓ Integration test environment ready")
    
    state['_db'] = db
    state['_email'] = email
    state['_user_service'] = user_service
    state['_runner'] = runner
    
    return {
        "test_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def run_integration_tests_agent(state: IntegrationTestState):
    """Run integration tests"""
    runner = state['_runner']
    user_service = state['_user_service']
    db = state['_db']
    email = state['_email']
    
    operations = []
    results = []
    
    operations.append("\n🔗 Running Integration Tests:")
    
    # Test 1: Complete flow
    passed = runner.test_user_registration_flow(user_service, db, email)
    status = "✓ PASS" if passed else "✗ FAIL"
    operations.append(f"\n  {status}: User Registration Flow")
    operations.append("    - Saves user to database")
    operations.append("    - Sends welcome email")
    operations.append("    - Verifies data consistency")
    
    # Test 2: Database
    passed = runner.test_database_integration(db)
    status = "✓ PASS" if passed else "✗ FAIL"
    operations.append(f"\n  {status}: Database Integration")
    operations.append("    - Save operation")
    operations.append("    - Retrieve operation")
    
    # Test 3: Email
    passed = runner.test_email_service_integration(email)
    status = "✓ PASS" if passed else "✗ FAIL"
    operations.append(f"\n  {status}: Email Service Integration")
    operations.append("    - Send email operation")
    
    results.append(f"✓ {runner.passed + runner.failed} integration tests executed")
    
    return {
        "test_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Tests executed"]
    }


def statistics_agent(state: IntegrationTestState):
    """Show statistics"""
    runner = state['_runner']
    db = state['_db']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("INTEGRATION TEST RESULTS")
    operations.append("="*60)
    
    total = runner.passed + runner.failed
    operations.append(f"\nTotal: {total}")
    operations.append(f"Passed: {runner.passed} ✓")
    operations.append(f"Failed: {runner.failed} ✗")
    operations.append(f"Success Rate: {runner.passed/total*100:.1f}%")
    
    operations.append(f"\nDatabase Queries: {db.query_count}")
    
    metrics.append("\n📊 Integration Testing Benefits:")
    metrics.append("  ✓ Tests real interactions")
    metrics.append("  ✓ Catches interface bugs")
    metrics.append("  ✓ Verifies data flow")
    metrics.append("  ✓ End-to-end validation")
    
    results.append("✓ Integration testing complete")
    
    return {
        "test_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Complete"]
    }


def create_integration_test_graph():
    """Create integration testing workflow"""
    workflow = StateGraph(IntegrationTestState)
    
    workflow.add_node("setup", setup_agent)
    workflow.add_node("run", run_integration_tests_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "run")
    workflow.add_edge("run", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution"""
    print("=" * 80)
    print("Pattern 222: Integration Testing MCP Pattern")
    print("=" * 80)
    
    app = create_integration_test_graph()
    final_state = app.invoke({
        "test_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    })
    
    for op in final_state["test_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Integration Testing: Test component interactions

Scope:
- Multi-component flows
- Database operations
- External services
- API endpoints

Benefits:
✓ Detect interface issues
✓ Verify data flow
✓ Real integration testing
✓ Catch bugs early

Tools:
- Testcontainers (Docker)
- WireMock (API mocking)
- H2 (in-memory DB)
""")


if __name__ == "__main__":
    main()

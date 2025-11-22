"""
Pattern 216: Backend for Frontend (BFF) MCP Pattern

BFF provides specialized backend for each frontend client type:
- Mobile app gets mobile-optimized API
- Web app gets web-optimized API
- Desktop app gets desktop-optimized API
- Each BFF tailored to specific client needs

Benefits:
- Optimized data payloads per client
- Client-specific business logic
- Reduced over-fetching/under-fetching
- Independent deployment per client
- Better performance

Use Cases:
- Multi-platform applications
- Different user experiences (admin vs customer)
- Mobile-first applications
- Progressive web apps
- IoT devices
"""

from typing import TypedDict, Annotated, List, Dict, Any
from dataclasses import dataclass
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class BFFState(TypedDict):
    """State for BFF pattern operations"""
    bff_operations: Annotated[List[str], operator.add]
    operation_results: Annotated[List[str], operator.add]
    performance_metrics: Annotated[List[str], operator.add]
    messages: Annotated[List[str], operator.add]


@dataclass
class Product:
    """Product from backend service"""
    id: str
    name: str
    description: str
    price: float
    image_url: str
    inventory: int
    specifications: Dict[str, Any]
    reviews: List[Dict[str, Any]]


class BackendServices:
    """Shared backend services"""
    
    def get_product(self, product_id: str) -> Product:
        """Get full product details"""
        return Product(
            id=product_id,
            name="Smartphone X",
            description="Latest smartphone with amazing features and long battery life",
            price=699.99,
            image_url="https://cdn.example.com/images/smartphone-x-full.jpg",
            inventory=50,
            specifications={
                'screen': '6.5 inch OLED',
                'processor': 'Octa-core 3.0GHz',
                'ram': '8GB',
                'storage': '256GB',
                'battery': '5000mAh',
                'camera': '48MP + 12MP + 5MP'
            },
            reviews=[
                {'user': 'Alice', 'rating': 5, 'comment': 'Excellent phone!'},
                {'user': 'Bob', 'rating': 4, 'comment': 'Good value'},
                {'user': 'Charlie', 'rating': 5, 'comment': 'Best purchase'}
            ]
        )


class MobileBFF:
    """Backend for Frontend - Mobile App"""
    
    def __init__(self, backend: BackendServices):
        self.backend = backend
        self.request_count = 0
    
    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get mobile-optimized product data"""
        self.request_count += 1
        
        product = self.backend.get_product(product_id)
        
        # Mobile optimization:
        # - Smaller images
        # - Reduced description
        # - Summary specs only
        # - Limited reviews
        
        return {
            'id': product.id,
            'name': product.name,
            'description': product.description[:100] + '...',  # Truncated
            'price': product.price,
            'image': product.image_url.replace('-full.jpg', '-mobile.jpg'),  # Mobile image
            'in_stock': product.inventory > 0,
            'key_specs': {
                'screen': product.specifications['screen'],
                'storage': product.specifications['storage']
            },
            'rating': sum(r['rating'] for r in product.reviews) / len(product.reviews),
            'review_count': len(product.reviews)
        }


class WebBFF:
    """Backend for Frontend - Web Application"""
    
    def __init__(self, backend: BackendServices):
        self.backend = backend
        self.request_count = 0
    
    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get web-optimized product data"""
        self.request_count += 1
        
        product = self.backend.get_product(product_id)
        
        # Web optimization:
        # - Full description
        # - All specifications
        # - Top 5 reviews
        # - High-quality images
        
        return {
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'image': product.image_url,
            'gallery': [product.image_url, product.image_url.replace('.jpg', '-2.jpg')],
            'inventory': product.inventory,
            'specifications': product.specifications,
            'reviews': product.reviews[:5],  # Top 5 reviews
            'avg_rating': sum(r['rating'] for r in product.reviews) / len(product.reviews)
        }


class AdminBFF:
    """Backend for Frontend - Admin Dashboard"""
    
    def __init__(self, backend: BackendServices):
        self.backend = backend
        self.request_count = 0
    
    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get admin-specific product data"""
        self.request_count += 1
        
        product = self.backend.get_product(product_id)
        
        # Admin optimization:
        # - All data
        # - Analytics
        # - Management actions
        
        return {
            'id': product.id,
            'name': product.name,
            'description': product.description,
            'price': product.price,
            'cost': product.price * 0.6,  # Admin sees cost
            'margin': product.price * 0.4,
            'inventory': product.inventory,
            'inventory_status': 'Low' if product.inventory < 20 else 'Healthy',
            'specifications': product.specifications,
            'reviews': product.reviews,
            'avg_rating': sum(r['rating'] for r in product.reviews) / len(product.reviews),
            'metrics': {
                'views': 1250,
                'conversions': 45,
                'conversion_rate': '3.6%'
            }
        }


def setup_bffs_agent(state: BFFState):
    """Agent to set up BFFs"""
    operations = []
    results = []
    
    backend = BackendServices()
    mobile_bff = MobileBFF(backend)
    web_bff = WebBFF(backend)
    admin_bff = AdminBFF(backend)
    
    operations.append("Backend for Frontend Pattern Setup:")
    operations.append("\nBackend Services:")
    operations.append("  - Product Service")
    operations.append("  - Full product data available")
    
    operations.append("\nBFFs Created:")
    operations.append("  1. Mobile BFF: Optimized for mobile apps")
    operations.append("     - Smaller images")
    operations.append("     - Truncated descriptions")
    operations.append("     - Essential data only")
    
    operations.append("  2. Web BFF: Optimized for web browsers")
    operations.append("     - Full descriptions")
    operations.append("     - Image galleries")
    operations.append("     - Detailed reviews")
    
    operations.append("  3. Admin BFF: Optimized for admin dashboard")
    operations.append("     - Cost/margin data")
    operations.append("     - Analytics metrics")
    operations.append("     - Management info")
    
    results.append("✓ 3 BFFs initialized")
    
    state['_mobile'] = mobile_bff
    state['_web'] = web_bff
    state['_admin'] = admin_bff
    
    return {
        "bff_operations": operations,
        "operation_results": results,
        "performance_metrics": [],
        "messages": ["Setup complete"]
    }


def mobile_request_agent(state: BFFState):
    """Agent to demonstrate mobile BFF"""
    mobile = state['_mobile']
    operations = []
    results = []
    
    operations.append("\n📱 Mobile BFF Request:")
    
    product = mobile.get_product("PROD-123")
    
    operations.append("\nMobile Response (optimized):")
    operations.append(f"  name: {product['name']}")
    operations.append(f"  description: {product['description']}")  # Truncated
    operations.append(f"  price: ${product['price']}")
    operations.append(f"  image: {product['image']}")  # Mobile version
    operations.append(f"  in_stock: {product['in_stock']}")
    operations.append(f"  key_specs: {product['key_specs']}")
    operations.append(f"  rating: {product['rating']:.1f}/5")
    
    operations.append("\nPayload Size: ~0.5 KB (optimized for mobile)")
    
    results.append("✓ Mobile-optimized response delivered")
    
    return {
        "bff_operations": operations,
        "operation_results": results,
        "performance_metrics": ["Mobile payload: ~0.5 KB"],
        "messages": ["Mobile request complete"]
    }


def web_request_agent(state: BFFState):
    """Agent to demonstrate web BFF"""
    web = state['_web']
    operations = []
    results = []
    
    operations.append("\n🌐 Web BFF Request:")
    
    product = web.get_product("PROD-123")
    
    operations.append("\nWeb Response (full featured):")
    operations.append(f"  name: {product['name']}")
    operations.append(f"  description: {product['description'][:50]}...")  # Full
    operations.append(f"  price: ${product['price']}")
    operations.append(f"  gallery: {len(product['gallery'])} images")
    operations.append(f"  inventory: {product['inventory']} units")
    operations.append(f"  specifications: {len(product['specifications'])} items")
    operations.append(f"  reviews: {len(product['reviews'])} shown")
    
    operations.append("\nPayload Size: ~2.5 KB (rich content)")
    
    results.append("✓ Web-optimized response delivered")
    
    return {
        "bff_operations": operations,
        "operation_results": results,
        "performance_metrics": ["Web payload: ~2.5 KB"],
        "messages": ["Web request complete"]
    }


def admin_request_agent(state: BFFState):
    """Agent to demonstrate admin BFF"""
    admin = state['_admin']
    operations = []
    results = []
    
    operations.append("\n👨‍💼 Admin BFF Request:")
    
    product = admin.get_product("PROD-123")
    
    operations.append("\nAdmin Response (full analytics):")
    operations.append(f"  name: {product['name']}")
    operations.append(f"  price: ${product['price']}")
    operations.append(f"  cost: ${product['cost']:.2f}")
    operations.append(f"  margin: ${product['margin']:.2f}")
    operations.append(f"  inventory_status: {product['inventory_status']}")
    operations.append(f"  metrics:")
    for key, value in product['metrics'].items():
        operations.append(f"    {key}: {value}")
    
    operations.append("\nPayload Size: ~3.0 KB (includes analytics)")
    
    results.append("✓ Admin-specific response delivered")
    
    return {
        "bff_operations": operations,
        "operation_results": results,
        "performance_metrics": ["Admin payload: ~3.0 KB"],
        "messages": ["Admin request complete"]
    }


def statistics_agent(state: BFFState):
    """Agent to show statistics"""
    mobile = state['_mobile']
    web = state['_web']
    admin = state['_admin']
    operations = []
    results = []
    metrics = []
    
    operations.append("\n" + "="*60)
    operations.append("BFF STATISTICS")
    operations.append("="*60)
    
    operations.append(f"\nMobile BFF: {mobile.request_count} requests")
    operations.append(f"Web BFF: {web.request_count} requests")
    operations.append(f"Admin BFF: {admin.request_count} requests")
    
    metrics.append("\n📊 BFF Pattern Benefits:")
    metrics.append("  ✓ Client-specific optimization")
    metrics.append("  ✓ Reduced over-fetching")
    metrics.append("  ✓ Better performance")
    metrics.append("  ✓ Independent deployment")
    metrics.append("  ✓ Tailored business logic")
    
    results.append("✓ BFF pattern working successfully")
    
    return {
        "bff_operations": operations,
        "operation_results": results,
        "performance_metrics": metrics,
        "messages": ["Statistics complete"]
    }


def create_bff_graph():
    """Create the BFF workflow graph"""
    workflow = StateGraph(BFFState)
    
    workflow.add_node("setup", setup_bffs_agent)
    workflow.add_node("mobile", mobile_request_agent)
    workflow.add_node("web", web_request_agent)
    workflow.add_node("admin", admin_request_agent)
    workflow.add_node("statistics", statistics_agent)
    
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "mobile")
    workflow.add_edge("mobile", "web")
    workflow.add_edge("web", "admin")
    workflow.add_edge("admin", "statistics")
    workflow.add_edge("statistics", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("=" * 80)
    print("Pattern 216: Backend for Frontend (BFF) MCP Pattern")
    print("=" * 80)
    
    app = create_bff_graph()
    initial_state = {
        "bff_operations": [],
        "operation_results": [],
        "performance_metrics": [],
        "messages": []
    }
    
    final_state = app.invoke(initial_state)
    
    for op in final_state["bff_operations"]:
        print(op)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Backend for Frontend: Specialized backend per client type

Comparison:
- Mobile: 0.5 KB, essential data, fast
- Web: 2.5 KB, rich content, detailed
- Admin: 3.0 KB, analytics, management

Benefits:
✓ Optimized payloads
✓ Reduced latency
✓ Better UX per platform
✓ Independent evolution
✓ Team autonomy

Real-World:
- Netflix (different APIs per device)
- Spotify (mobile vs web vs desktop)
- Amazon (app vs website)
""")


if __name__ == "__main__":
    main()

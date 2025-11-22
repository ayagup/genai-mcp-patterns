"""
Pattern 190: Data Enrichment MCP Pattern

This pattern demonstrates augmenting existing data with additional information from
external sources. Enrichment adds context, improves data quality, and enables
deeper insights by combining multiple data sources.

Key Concepts:
1. Lookup Enrichment: Add reference data (product details, user profiles)
2. API Enrichment: Call external APIs (geolocation, weather, pricing)
3. Calculated Enrichment: Derive fields from existing data
4. Time-based Enrichment: Add temporal context (day of week, season)
5. Geospatial Enrichment: Add location data (lat/lon, timezone)
6. Categorical Enrichment: Add classifications (product category, risk level)
7. Statistical Enrichment: Add aggregates (running totals, percentiles)

Enrichment Sources:
- Internal Databases: Reference tables, master data
- External APIs: Third-party services (geocoding, pricing)
- Static Files: Lookup tables, mappings
- Computed: Derived from existing fields
- Real-time: Live data (stock prices, weather)
- Historical: Time-series, trends

Enrichment Strategies:
- Batch: Enrich all records at once
- Streaming: Enrich as data arrives
- On-demand: Enrich when accessed
- Cached: Store enriched data
- Incremental: Only enrich new/changed records
- Fallback: Use default when source unavailable

Benefits:
- Context: Add meaning to raw data
- Insights: Enable deeper analysis
- Quality: Fill missing information
- Integration: Combine disparate sources
- Value: Increase data utility

Trade-offs:
- Latency: External calls add delay
- Cost: API calls may have charges
- Reliability: Depends on external sources
- Complexity: Managing multiple sources
- Staleness: Enriched data can become outdated
- Privacy: External sharing concerns

Use Cases:
- Customer Data: Add demographics, preferences
- Transaction Data: Add product details, pricing
- Log Data: Add user info, session context
- Sensor Data: Add device metadata, location
- Click Data: Add user profile, A/B test variant
- Order Data: Add shipping address, tax rate
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
import operator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Define the state for data enrichment
class DataEnrichmentState(TypedDict):
    """State for data enrichment"""
    input_data: List[Dict[str, Any]]
    lookup_enriched: Optional[List[Dict[str, Any]]]
    api_enriched: Optional[List[Dict[str, Any]]]
    calculated_enriched: Optional[List[Dict[str, Any]]]
    temporal_enriched: Optional[List[Dict[str, Any]]]
    enrichment_metadata: Dict[str, Any]
    messages: Annotated[List[str], operator.add]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# ============================================================================
# ENRICHMENT ENGINES
# ============================================================================

class LookupEnricher:
    """
    Lookup Enrichment: Add reference data from lookup tables
    
    Use Cases:
    - Add product details from product catalog
    - Add user profile from user database
    - Add category from classification table
    """
    
    def __init__(self, lookup_table: Dict[Any, Dict[str, Any]]):
        """
        lookup_table: {key: {enrichment_fields}}
        
        Example:
        {
            "PROD001": {"name": "Laptop", "category": "Electronics", "price": 999},
            "PROD002": {"name": "Mouse", "category": "Electronics", "price": 29}
        }
        """
        self.lookup_table = lookup_table
    
    def enrich(self, data: List[Dict[str, Any]], 
              lookup_key: str,
              prefix: str = "") -> List[Dict[str, Any]]:
        """
        Enrich data by looking up reference information
        
        Args:
            data: Records to enrich
            lookup_key: Field to use for lookup
            prefix: Prefix for enriched fields (e.g., "product_")
        """
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            key_value = record.get(lookup_key)
            if key_value in self.lookup_table:
                lookup_data = self.lookup_table[key_value]
                
                # Add lookup data with optional prefix
                for field, value in lookup_data.items():
                    enriched_field = f"{prefix}{field}" if prefix else field
                    enriched_record[enriched_field] = value
            
            enriched.append(enriched_record)
        
        return enriched

class APIEnricher:
    """
    API Enrichment: Call external APIs to add data
    
    Use Cases:
    - Geocode address to lat/lon
    - Get weather for location
    - Validate email/phone
    - Get current stock price
    """
    
    def __init__(self, mock_mode: bool = True):
        """Initialize with mock data for demonstration"""
        self.mock_mode = mock_mode
        
        # Mock API responses
        self.mock_geocoding = {
            "New York": {"lat": 40.7128, "lon": -74.0060, "timezone": "America/New_York"},
            "London": {"lat": 51.5074, "lon": -0.1278, "timezone": "Europe/London"},
            "Tokyo": {"lat": 35.6762, "lon": 139.6503, "timezone": "Asia/Tokyo"}
        }
        
        self.mock_weather = {
            "New York": {"temp": 72, "condition": "Sunny", "humidity": 65},
            "London": {"temp": 60, "condition": "Cloudy", "humidity": 75},
            "Tokyo": {"temp": 68, "condition": "Rainy", "humidity": 80}
        }
    
    def geocode(self, data: List[Dict[str, Any]], 
               location_field: str) -> List[Dict[str, Any]]:
        """Enrich with geocoding data (lat/lon)"""
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            location = record.get(location_field)
            if location and location in self.mock_geocoding:
                geo_data = self.mock_geocoding[location]
                enriched_record["latitude"] = geo_data["lat"]
                enriched_record["longitude"] = geo_data["lon"]
                enriched_record["timezone"] = geo_data["timezone"]
            
            enriched.append(enriched_record)
        
        return enriched
    
    def add_weather(self, data: List[Dict[str, Any]], 
                   location_field: str) -> List[Dict[str, Any]]:
        """Enrich with weather data"""
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            location = record.get(location_field)
            if location and location in self.mock_weather:
                weather_data = self.mock_weather[location]
                enriched_record["temperature"] = weather_data["temp"]
                enriched_record["weather_condition"] = weather_data["condition"]
                enriched_record["humidity"] = weather_data["humidity"]
            
            enriched.append(enriched_record)
        
        return enriched

class CalculatedEnricher:
    """
    Calculated Enrichment: Derive fields from existing data
    
    Use Cases:
    - Calculate total from price * quantity
    - Derive age from birth date
    - Compute duration from start/end time
    - Calculate category from value
    """
    
    def add_calculated_fields(self, data: List[Dict[str, Any]],
                             calculations: Dict[str, callable]) -> List[Dict[str, Any]]:
        """
        Add calculated fields
        
        calculations: {field_name: lambda record: calculation}
        """
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            for field, calc_fn in calculations.items():
                try:
                    enriched_record[field] = calc_fn(record)
                except:
                    # Skip on error
                    pass
            
            enriched.append(enriched_record)
        
        return enriched
    
    def categorize(self, data: List[Dict[str, Any]],
                  value_field: str,
                  category_field: str,
                  ranges: List[tuple]) -> List[Dict[str, Any]]:
        """
        Categorize based on value ranges
        
        ranges: [(max_value, category), ...]
        Example: [(100, "Low"), (500, "Medium"), (float('inf'), "High")]
        """
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            value = record.get(value_field, 0)
            
            for max_val, category in ranges:
                if value <= max_val:
                    enriched_record[category_field] = category
                    break
            
            enriched.append(enriched_record)
        
        return enriched

class TemporalEnricher:
    """
    Temporal Enrichment: Add time-based context
    
    Use Cases:
    - Add day of week, month, quarter
    - Add is_weekend, is_business_hours
    - Add season, holiday indicator
    - Add time since event
    """
    
    def enrich_datetime(self, data: List[Dict[str, Any]],
                       datetime_field: str) -> List[Dict[str, Any]]:
        """Add temporal fields from datetime"""
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            dt_value = record.get(datetime_field)
            
            if dt_value:
                # Parse if string
                if isinstance(dt_value, str):
                    try:
                        dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                    except:
                        dt = None
                else:
                    dt = dt_value
                
                if dt:
                    enriched_record["day_of_week"] = dt.strftime("%A")
                    enriched_record["month"] = dt.strftime("%B")
                    enriched_record["year"] = dt.year
                    enriched_record["quarter"] = (dt.month - 1) // 3 + 1
                    enriched_record["is_weekend"] = dt.weekday() >= 5
                    enriched_record["hour"] = dt.hour
                    enriched_record["is_business_hours"] = 9 <= dt.hour < 17
            
            enriched.append(enriched_record)
        
        return enriched
    
    def add_time_since(self, data: List[Dict[str, Any]],
                      datetime_field: str,
                      reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Add time elapsed since event"""
        if reference_time is None:
            reference_time = datetime.now()
        
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            dt_value = record.get(datetime_field)
            
            if dt_value:
                if isinstance(dt_value, str):
                    try:
                        dt = datetime.fromisoformat(dt_value.replace('Z', '+00:00'))
                    except:
                        dt = None
                else:
                    dt = dt_value
                
                if dt:
                    delta = reference_time - dt
                    enriched_record["days_since"] = delta.days
                    enriched_record["hours_since"] = delta.total_seconds() / 3600
            
            enriched.append(enriched_record)
        
        return enriched

class StatisticalEnricher:
    """
    Statistical Enrichment: Add aggregates and statistics
    
    Use Cases:
    - Add running total
    - Add percentile rank
    - Add moving average
    - Add z-score (standard deviations from mean)
    """
    
    def add_running_total(self, data: List[Dict[str, Any]],
                         value_field: str) -> List[Dict[str, Any]]:
        """Add running total"""
        enriched = []
        running_total = 0
        
        for record in data:
            enriched_record = record.copy()
            
            value = record.get(value_field, 0)
            running_total += value
            enriched_record["running_total"] = running_total
            
            enriched.append(enriched_record)
        
        return enriched
    
    def add_rank(self, data: List[Dict[str, Any]],
                value_field: str,
                descending: bool = True) -> List[Dict[str, Any]]:
        """Add rank based on value"""
        # Sort by value
        sorted_data = sorted(data, key=lambda x: x.get(value_field, 0), reverse=descending)
        
        # Add rank
        enriched = []
        for rank, record in enumerate(sorted_data, 1):
            enriched_record = record.copy()
            enriched_record["rank"] = rank
            enriched.append(enriched_record)
        
        return enriched
    
    def add_percentile(self, data: List[Dict[str, Any]],
                      value_field: str) -> List[Dict[str, Any]]:
        """Add percentile"""
        values = sorted([r.get(value_field, 0) for r in data])
        n = len(values)
        
        enriched = []
        
        for record in data:
            enriched_record = record.copy()
            
            value = record.get(value_field, 0)
            
            # Count values less than current
            count_below = sum(1 for v in values if v < value)
            percentile = (count_below / n) * 100 if n > 0 else 0
            
            enriched_record["percentile"] = round(percentile, 2)
            
            enriched.append(enriched_record)
        
        return enriched

# Create enricher instances
calculated_enricher = CalculatedEnricher()
temporal_enricher = TemporalEnricher()
statistical_enricher = StatisticalEnricher()

# ============================================================================
# LANGGRAPH INTEGRATION
# ============================================================================

def enrich_lookup(state: DataEnrichmentState) -> DataEnrichmentState:
    """Enrich with lookup data"""
    input_data = state["input_data"]
    
    # Create product lookup table
    product_lookup = {
        "PROD001": {"product_name": "Laptop", "category": "Electronics", "price": 999},
        "PROD002": {"product_name": "Mouse", "category": "Electronics", "price": 29},
        "PROD003": {"product_name": "Desk", "category": "Furniture", "price": 299}
    }
    
    enricher = LookupEnricher(product_lookup)
    enriched = enricher.enrich(input_data, "product_id", prefix="")
    
    return {
        "lookup_enriched": enriched,
        "messages": [f"[Lookup] Enriched with product catalog data"]
    }

def enrich_api(state: DataEnrichmentState) -> DataEnrichmentState:
    """Enrich with API data"""
    lookup_enriched = state.get("lookup_enriched", [])
    
    # Add geocoding
    api_enricher = APIEnricher(mock_mode=True)
    enriched = api_enricher.geocode(lookup_enriched, "location")
    
    # Add weather
    enriched = api_enricher.add_weather(enriched, "location")
    
    return {
        "api_enriched": enriched,
        "messages": [f"[API] Enriched with geocoding and weather data"]
    }

def enrich_calculated(state: DataEnrichmentState) -> DataEnrichmentState:
    """Enrich with calculated fields"""
    api_enriched = state.get("api_enriched", [])
    
    # Define calculations
    calculations = {
        "total_price": lambda r: r.get("price", 0) * r.get("quantity", 1),
        "discount": lambda r: r.get("price", 0) * 0.1 if r.get("price", 0) > 500 else 0,
        "final_price": lambda r: (r.get("price", 0) * r.get("quantity", 1)) - 
                                (r.get("price", 0) * 0.1 if r.get("price", 0) > 500 else 0)
    }
    
    enriched = calculated_enricher.add_calculated_fields(api_enriched, calculations)
    
    # Add price category
    enriched = calculated_enricher.categorize(
        enriched, "price", "price_category",
        [(100, "Budget"), (500, "Mid-range"), (float('inf'), "Premium")]
    )
    
    return {
        "calculated_enriched": enriched,
        "messages": [f"[Calculated] Added derived fields (total, discount, category)"]
    }

def enrich_temporal(state: DataEnrichmentState) -> DataEnrichmentState:
    """Enrich with temporal data"""
    calculated_enriched = state.get("calculated_enriched", [])
    
    # Add datetime fields
    enriched = temporal_enricher.enrich_datetime(calculated_enriched, "order_date")
    
    return {
        "temporal_enriched": enriched,
        "enrichment_metadata": {
            "total_records": len(enriched),
            "enrichments_applied": ["lookup", "api", "calculated", "temporal"]
        },
        "messages": [f"[Temporal] Added time-based context (day, month, quarter)"]
    }

# ============================================================================
# BUILD THE ENRICHMENT GRAPH
# ============================================================================

def create_enrichment_graph():
    """
    Create a StateGraph demonstrating data enrichment pattern.
    
    Flow:
    1. Lookup Enrichment: Add reference data
    2. API Enrichment: Call external APIs
    3. Calculated Enrichment: Derive fields
    4. Temporal Enrichment: Add time context
    """
    
    workflow = StateGraph(DataEnrichmentState)
    
    # Add enrichment nodes
    workflow.add_node("lookup", enrich_lookup)
    workflow.add_node("api", enrich_api)
    workflow.add_node("calculated", enrich_calculated)
    workflow.add_node("temporal", enrich_temporal)
    
    # Define enrichment flow
    workflow.add_edge(START, "lookup")
    workflow.add_edge("lookup", "api")
    workflow.add_edge("api", "calculated")
    workflow.add_edge("calculated", "temporal")
    workflow.add_edge("temporal", END)
    
    return workflow.compile()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Data Enrichment MCP Pattern - LangGraph Implementation")
    print("=" * 80)
    
    # Example 1: Multi-Source Enrichment
    print("\n" + "=" * 80)
    print("Example 1: Multi-Source Data Enrichment Pipeline")
    print("=" * 80)
    
    enrichment_graph = create_enrichment_graph()
    
    # Sample data (minimal)
    input_data = [
        {"order_id": 1, "product_id": "PROD001", "quantity": 1, "location": "New York", "order_date": "2024-01-15T10:30:00"},
        {"order_id": 2, "product_id": "PROD002", "quantity": 2, "location": "London", "order_date": "2024-01-16T14:45:00"},
        {"order_id": 3, "product_id": "PROD003", "quantity": 1, "location": "Tokyo", "order_date": "2024-01-17T09:15:00"}
    ]
    
    initial_state: DataEnrichmentState = {
        "input_data": input_data,
        "lookup_enriched": None,
        "api_enriched": None,
        "calculated_enriched": None,
        "temporal_enriched": None,
        "enrichment_metadata": {},
        "messages": []
    }
    
    result = enrichment_graph.invoke(initial_state)
    
    print("\nEnrichment Log:")
    for msg in result["messages"]:
        print(f"  {msg}")
    
    print("\nBefore Enrichment (Order 1):")
    print(f"  Fields: {list(input_data[0].keys())}")
    print(f"  Data: {input_data[0]}")
    
    print("\nAfter Enrichment (Order 1):")
    enriched = result["temporal_enriched"][0]
    print(f"  Fields: {list(enriched.keys())}")
    print(f"  Added Fields: product_name, category, price, latitude, longitude, timezone,")
    print(f"                temperature, weather_condition, total_price, discount,")
    print(f"                final_price, price_category, day_of_week, month, quarter")
    
    # Example 2: Individual Enrichers
    print("\n" + "=" * 80)
    print("Example 2: Individual Enrichment Types")
    print("=" * 80)
    
    print("\n1. Lookup Enrichment:")
    user_lookup = {
        "U001": {"name": "John Doe", "tier": "Gold", "credit_limit": 10000},
        "U002": {"name": "Jane Smith", "tier": "Silver", "credit_limit": 5000}
    }
    user_data = [{"user_id": "U001", "purchase": 500}, {"user_id": "U002", "purchase": 300}]
    enricher = LookupEnricher(user_lookup)
    enriched = enricher.enrich(user_data, "user_id", prefix="")
    print(f"   Before: {list(user_data[0].keys())}")
    print(f"   After: {list(enriched[0].keys())}")
    print(f"   Added: name, tier, credit_limit")
    
    print("\n2. Calculated Enrichment:")
    sales_data = [{"price": 100, "quantity": 3}]
    calculations = {
        "total": lambda r: r["price"] * r["quantity"],
        "tax": lambda r: r["price"] * r["quantity"] * 0.08
    }
    enriched = calculated_enricher.add_calculated_fields(sales_data, calculations)
    print(f"   Before: {sales_data[0]}")
    print(f"   After: {enriched[0]}")
    
    print("\n3. Temporal Enrichment:")
    event_data = [{"event_id": 1, "timestamp": "2024-01-15T14:30:00"}]
    enriched = temporal_enricher.enrich_datetime(event_data, "timestamp")
    print(f"   Original: timestamp=2024-01-15T14:30:00")
    print(f"   Added: day_of_week={enriched[0].get('day_of_week')}, month={enriched[0].get('month')}")
    print(f"          quarter={enriched[0].get('quarter')}, is_weekend={enriched[0].get('is_weekend')}")
    print(f"          is_business_hours={enriched[0].get('is_business_hours')}")
    
    print("\n4. Statistical Enrichment:")
    scores = [{"id": 1, "score": 85}, {"id": 2, "score": 92}, {"id": 3, "score": 78}]
    enriched = statistical_enricher.add_running_total(scores, "score")
    print("   Running Total:")
    for r in enriched:
        print(f"     ID {r['id']}: score={r['score']}, running_total={r['running_total']}")
    
    enriched = statistical_enricher.add_rank(scores, "score", descending=True)
    print("\n   Rank (descending):")
    for r in enriched:
        print(f"     ID {r['id']}: score={r['score']}, rank={r['rank']}")
    
    # Example 3: Categorization
    print("\n" + "=" * 80)
    print("Example 3: Categorization Enrichment")
    print("=" * 80)
    
    transactions = [
        {"id": 1, "amount": 50},
        {"id": 2, "amount": 250},
        {"id": 3, "amount": 750}
    ]
    
    enriched = calculated_enricher.categorize(
        transactions, "amount", "risk_level",
        [(100, "Low"), (500, "Medium"), (float('inf'), "High")]
    )
    
    print("\nRisk Categorization:")
    for r in enriched:
        print(f"  Transaction {r['id']}: ${r['amount']} → {r['risk_level']} Risk")
    
    print("\n" + "=" * 80)
    print("Data Enrichment Pattern Complete!")
    print("=" * 80)
    print("""
Key Takeaways:
  • Enrichment adds context and value to raw data
  • Lookup enrichment: Join with reference tables (product catalog, user profiles)
  • API enrichment: Call external services (geocoding, weather, pricing)
  • Calculated enrichment: Derive fields from existing data (totals, categories)
  • Temporal enrichment: Add time context (day of week, quarter, business hours)
  • Statistical enrichment: Add aggregates (running total, rank, percentile)
  • Multi-source: Combine multiple enrichment types in pipeline
  • Strategies: Batch (all at once), streaming (as data arrives), cached (store results)
  • Tools: DBT, Talend, Apache NiFi, custom scripts
  • Best practices: Cache results, handle missing data, monitor API costs
  • Use cases: Customer 360, fraud detection, personalization, analytics
    """)

"""
Pattern 305: Semantic Transformation MCP Pattern

This pattern demonstrates transforming data while preserving or enriching
its semantic meaning, including entity resolution, synonym mapping, and
concept normalization.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class SemanticTransformationPattern(TypedDict):
    """State for semantic transformation"""
    messages: Annotated[List[str], add]
    source_data: Dict[str, Any]
    semantic_mappings: Dict[str, Any]
    ontology: Dict[str, Any]
    transformed_data: Dict[str, Any]
    semantic_statistics: Dict[str, Any]


class SemanticMapper:
    """Map concepts to standardized semantic representations"""
    
    def __init__(self):
        self.synonym_map = {}
        self.concept_hierarchy = {}
        self.entity_mappings = {}
    
    def add_synonym_mapping(self, canonical: str, synonyms: List[str]):
        """Add synonym mapping"""
        for synonym in synonyms:
            self.synonym_map[synonym.lower()] = canonical
    
    def add_concept_hierarchy(self, parent: str, children: List[str]):
        """Add concept hierarchy"""
        self.concept_hierarchy[parent] = children
    
    def normalize_term(self, term: str) -> str:
        """Normalize term to canonical form"""
        term_lower = term.lower()
        return self.synonym_map.get(term_lower, term)
    
    def get_concept_parent(self, concept: str) -> str:
        """Get parent concept"""
        for parent, children in self.concept_hierarchy.items():
            if concept in children:
                return parent
        return concept
    
    def resolve_entity(self, entity: str, context: str = None) -> dict:
        """Resolve entity to standard representation"""
        normalized = self.normalize_term(entity)
        parent = self.get_concept_parent(normalized)
        
        return {
            "original": entity,
            "normalized": normalized,
            "category": parent,
            "context": context
        }


class OntologyMapper:
    """Map data to ontology concepts"""
    
    def __init__(self):
        self.ontology = {}
        self.mappings = {}
    
    def define_concept(self, concept: str, properties: dict):
        """Define an ontology concept"""
        self.ontology[concept] = properties
    
    def map_field_to_concept(self, field: str, concept: str, property_name: str = None):
        """Map field to ontology concept"""
        self.mappings[field] = {
            "concept": concept,
            "property": property_name or field
        }
    
    def transform_to_ontology(self, data: dict) -> dict:
        """Transform data to ontology representation"""
        result = {}
        
        for field, value in data.items():
            if field in self.mappings:
                mapping = self.mappings[field]
                concept = mapping["concept"]
                property_name = mapping["property"]
                
                if concept not in result:
                    result[concept] = {}
                
                result[concept][property_name] = value
        
        return result


def initialize_semantic_mapper_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Initialize semantic transformation system"""
    print("\n🧠 Initializing Semantic Transformation System...")
    
    # Source data with varied terminology
    source_data = {
        "customer_name": "John Smith",
        "client_email": "john@example.com",
        "phone_no": "555-0123",
        "addr_line1": "123 Main St",
        "addr_line2": "Apt 4B",
        "city_name": "New York",
        "state_code": "NY",
        "postal_code": "10001",
        "product_purchased": "laptop computer",
        "product_category": "electronics",
        "order_amount": 1299.99,
        "payment_method": "credit card",
        "delivery_preference": "express shipping"
    }
    
    print(f"  Source Data Fields: {len(source_data)}")
    print(f"\n  Sample Fields:")
    for key in list(source_data.keys())[:5]:
        print(f"    {key}: {source_data[key]}")
    
    print(f"\n  Semantic Challenges:")
    print(f"    • Inconsistent terminology (customer vs client)")
    print(f"    • Abbreviations (addr, no)")
    print(f"    • Implicit relationships (address components)")
    print(f"    • Product categorization")
    
    return {
        **state,
        "source_data": source_data,
        "semantic_mappings": {},
        "ontology": {},
        "transformed_data": {},
        "semantic_statistics": {},
        "messages": ["✓ Semantic mapper initialized"]
    }


def build_semantic_mappings_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Build semantic mappings"""
    print("\n📚 Building Semantic Mappings...")
    
    mapper = SemanticMapper()
    
    # Synonym mappings
    synonym_mappings = [
        ("customer", ["customer", "client", "buyer", "purchaser"]),
        ("email", ["email", "e-mail", "email_address", "electronic_mail"]),
        ("phone", ["phone", "telephone", "phone_number", "tel", "mobile"]),
        ("address", ["address", "addr", "location", "street_address"]),
        ("product", ["product", "item", "merchandise", "goods"]),
        ("payment", ["payment", "pay", "transaction", "remittance"]),
        ("shipping", ["shipping", "delivery", "shipment", "freight"])
    ]
    
    for canonical, synonyms in synonym_mappings:
        mapper.add_synonym_mapping(canonical, synonyms)
        print(f"  ✓ Mapped: {canonical} ← {', '.join(synonyms[:3])}")
    
    # Concept hierarchy
    mapper.add_concept_hierarchy("contact_info", ["email", "phone", "address"])
    mapper.add_concept_hierarchy("location", ["address", "city", "state", "postal_code"])
    mapper.add_concept_hierarchy("order", ["product", "payment", "shipping"])
    
    print(f"\n  Concept Hierarchies:")
    for parent, children in mapper.concept_hierarchy.items():
        print(f"    {parent}: {', '.join(children)}")
    
    # Entity resolution examples
    entities = [
        ("customer_name", "person"),
        ("laptop computer", "product"),
        ("credit card", "payment"),
        ("express shipping", "shipping")
    ]
    
    resolved = []
    for entity, context in entities:
        resolution = mapper.resolve_entity(entity, context)
        resolved.append(resolution)
    
    print(f"\n  Entity Resolutions: {len(resolved)}")
    
    semantic_mappings = {
        "synonyms": mapper.synonym_map,
        "hierarchies": mapper.concept_hierarchy,
        "entity_resolutions": resolved
    }
    
    return {
        **state,
        "semantic_mappings": semantic_mappings,
        "messages": [f"✓ Built {len(synonym_mappings)} semantic mappings"]
    }


def build_ontology_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Build domain ontology"""
    print("\n🏗️ Building Domain Ontology...")
    
    ontology_mapper = OntologyMapper()
    
    # Define ontology concepts
    concepts = [
        ("Person", {
            "properties": ["name", "email", "phone"],
            "description": "Individual or entity"
        }),
        ("Address", {
            "properties": ["line1", "line2", "city", "state", "postal_code"],
            "description": "Physical location"
        }),
        ("Product", {
            "properties": ["name", "category", "price"],
            "description": "Item for sale"
        }),
        ("Order", {
            "properties": ["product", "amount", "payment", "shipping"],
            "description": "Purchase transaction"
        })
    ]
    
    for concept, properties in concepts:
        ontology_mapper.define_concept(concept, properties)
        print(f"  ✓ Defined concept: {concept}")
        print(f"    Properties: {', '.join(properties['properties'])}")
    
    # Map fields to ontology
    field_mappings = [
        ("customer_name", "Person", "name"),
        ("client_email", "Person", "email"),
        ("phone_no", "Person", "phone"),
        ("addr_line1", "Address", "line1"),
        ("addr_line2", "Address", "line2"),
        ("city_name", "Address", "city"),
        ("state_code", "Address", "state"),
        ("postal_code", "Address", "postal_code"),
        ("product_purchased", "Product", "name"),
        ("product_category", "Product", "category"),
        ("order_amount", "Order", "amount"),
        ("payment_method", "Order", "payment"),
        ("delivery_preference", "Order", "shipping")
    ]
    
    for field, concept, prop in field_mappings:
        ontology_mapper.map_field_to_concept(field, concept, prop)
    
    print(f"\n  Field Mappings: {len(field_mappings)}")
    
    return {
        **state,
        "ontology": {
            "concepts": ontology_mapper.ontology,
            "mappings": ontology_mapper.mappings
        },
        "messages": [f"✓ Built ontology with {len(concepts)} concepts"]
    }


def transform_to_semantic_model_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Transform data to semantic model"""
    print("\n🔄 Transforming to Semantic Model...")
    
    # Rebuild ontology mapper
    ontology_mapper = OntologyMapper()
    
    for concept, properties in state["ontology"]["concepts"].items():
        ontology_mapper.define_concept(concept, properties)
    
    for field, mapping in state["ontology"]["mappings"].items():
        ontology_mapper.map_field_to_concept(
            field,
            mapping["concept"],
            mapping["property"]
        )
    
    # Transform source data
    source_data = state["source_data"]
    transformed = ontology_mapper.transform_to_ontology(source_data)
    
    print(f"  Source Fields: {len(source_data)}")
    print(f"  Semantic Concepts: {len(transformed)}")
    
    print(f"\n  Transformed Data:")
    for concept, properties in transformed.items():
        print(f"\n    {concept}:")
        for prop, value in properties.items():
            print(f"      {prop}: {value}")
    
    return {
        **state,
        "transformed_data": transformed,
        "messages": ["✓ Data transformed to semantic model"]
    }


def enrich_semantic_data_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Enrich data with semantic information"""
    print("\n✨ Enriching with Semantic Information...")
    
    transformed = state["transformed_data"].copy()
    
    # Add inferred relationships
    enrichments = []
    
    # Infer customer type
    if "Order" in transformed and "amount" in transformed["Order"]:
        amount = transformed["Order"]["amount"]
        customer_type = "premium" if amount > 1000 else "standard"
        
        if "Person" not in transformed:
            transformed["Person"] = {}
        transformed["Person"]["customer_type"] = customer_type
        
        enrichments.append({
            "type": "inference",
            "concept": "Person",
            "property": "customer_type",
            "value": customer_type,
            "reason": f"Based on order amount: ${amount}"
        })
        print(f"  ✓ Inferred customer_type: {customer_type}")
    
    # Add product category hierarchy
    if "Product" in transformed and "category" in transformed["Product"]:
        category = transformed["Product"]["category"]
        transformed["Product"]["category_hierarchy"] = ["retail", category]
        
        enrichments.append({
            "type": "hierarchy",
            "concept": "Product",
            "property": "category_hierarchy",
            "value": ["retail", category]
        })
        print(f"  ✓ Added category hierarchy")
    
    # Add shipping metadata
    if "Order" in transformed and "shipping" in transformed["Order"]:
        shipping = transformed["Order"]["shipping"]
        is_express = "express" in shipping.lower()
        transformed["Order"]["is_express"] = is_express
        
        enrichments.append({
            "type": "extraction",
            "concept": "Order",
            "property": "is_express",
            "value": is_express,
            "source": shipping
        })
        print(f"  ✓ Extracted is_express: {is_express}")
    
    # Add location metadata
    if "Address" in transformed and "state" in transformed["Address"]:
        state_code = transformed["Address"]["state"]
        region = self._get_region(state_code)
        transformed["Address"]["region"] = region
        
        enrichments.append({
            "type": "lookup",
            "concept": "Address",
            "property": "region",
            "value": region
        })
        print(f"  ✓ Looked up region: {region}")
    
    print(f"\n  Total Enrichments: {len(enrichments)}")
    
    return {
        **state,
        "transformed_data": transformed,
        "messages": [f"✓ Added {len(enrichments)} semantic enrichments"]
    }

def _get_region(state_code: str) -> str:
    """Get region from state code"""
    regions = {
        "NY": "Northeast", "MA": "Northeast", "CT": "Northeast",
        "CA": "West", "OR": "West", "WA": "West",
        "TX": "South", "FL": "South", "GA": "South",
        "IL": "Midwest", "OH": "Midwest", "MI": "Midwest"
    }
    return regions.get(state_code, "Unknown")


def analyze_semantic_transformation_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Analyze semantic transformation"""
    print("\n📊 Analyzing Semantic Transformation...")
    
    source_fields = len(state["source_data"])
    semantic_concepts = len(state["transformed_data"])
    
    # Count mappings
    total_mappings = len(state["semantic_mappings"].get("synonyms", {}))
    hierarchies = len(state["semantic_mappings"].get("hierarchies", {}))
    
    # Count properties per concept
    properties_count = {}
    for concept, properties in state["transformed_data"].items():
        properties_count[concept] = len(properties)
    
    total_properties = sum(properties_count.values())
    
    statistics = {
        "source_fields": source_fields,
        "semantic_concepts": semantic_concepts,
        "total_properties": total_properties,
        "synonym_mappings": total_mappings,
        "concept_hierarchies": hierarchies,
        "avg_properties_per_concept": total_properties / max(semantic_concepts, 1),
        "properties_by_concept": properties_count
    }
    
    print(f"  Source Fields: {statistics['source_fields']}")
    print(f"  Semantic Concepts: {statistics['semantic_concepts']}")
    print(f"  Total Properties: {statistics['total_properties']}")
    print(f"  Synonym Mappings: {statistics['synonym_mappings']}")
    print(f"  Concept Hierarchies: {statistics['concept_hierarchies']}")
    print(f"  Avg Properties/Concept: {statistics['avg_properties_per_concept']:.1f}")
    
    print(f"\n  Properties by Concept:")
    for concept, count in properties_count.items():
        print(f"    {concept}: {count}")
    
    return {
        **state,
        "semantic_statistics": statistics,
        "messages": ["✓ Semantic transformation analyzed"]
    }


def generate_semantic_transformation_report_agent(state: SemanticTransformationPattern) -> SemanticTransformationPattern:
    """Generate semantic transformation report"""
    print("\n" + "="*70)
    print("SEMANTIC TRANSFORMATION REPORT")
    print("="*70)
    
    print(f"\n📥 Source Data:")
    for key, value in list(state["source_data"].items())[:8]:
        print(f"  {key}: {value}")
    print(f"  ... ({len(state['source_data'])} total fields)")
    
    print(f"\n📚 Semantic Mappings:")
    mappings = state["semantic_mappings"]
    if mappings.get("synonyms"):
        print(f"  Synonym Mappings: {len(mappings['synonyms'])}")
        for canonical, synonym_list in list(mappings.get("synonyms", {}).items())[:3]:
            print(f"    {synonym_list} → {canonical}")
    
    if mappings.get("hierarchies"):
        print(f"\n  Concept Hierarchies:")
        for parent, children in mappings["hierarchies"].items():
            print(f"    {parent}: {', '.join(children)}")
    
    print(f"\n🏗️ Ontology:")
    ontology = state["ontology"]
    if ontology.get("concepts"):
        print(f"  Concepts: {len(ontology['concepts'])}")
        for concept, props in ontology["concepts"].items():
            print(f"    {concept}: {', '.join(props['properties'])}")
    
    print(f"\n📤 Transformed Data (Semantic Model):")
    for concept, properties in state["transformed_data"].items():
        print(f"\n  {concept}:")
        for prop, value in properties.items():
            value_str = str(value)[:50]
            print(f"    {prop}: {value_str}")
    
    print(f"\n📊 Statistics:")
    stats = state["semantic_statistics"]
    if stats:
        print(f"  Source Fields: {stats['source_fields']}")
        print(f"  Semantic Concepts: {stats['semantic_concepts']}")
        print(f"  Total Properties: {stats['total_properties']}")
        print(f"  Synonym Mappings: {stats['synonym_mappings']}")
        print(f"  Avg Properties/Concept: {stats['avg_properties_per_concept']:.1f}")
    
    print(f"\n💡 Semantic Transformation Benefits:")
    print("  ✓ Consistent terminology")
    print("  ✓ Entity resolution")
    print("  ✓ Concept normalization")
    print("  ✓ Semantic enrichment")
    print("  ✓ Ontology alignment")
    print("  ✓ Knowledge representation")
    
    print(f"\n🔧 Semantic Operations:")
    print("  • Synonym mapping")
    print("  • Entity resolution")
    print("  • Concept hierarchies")
    print("  • Ontology mapping")
    print("  • Semantic enrichment")
    print("  • Relationship inference")
    
    print(f"\n⚙️ Use Cases:")
    print("  • Data integration")
    print("  • Knowledge graphs")
    print("  • Semantic search")
    print("  • Natural language processing")
    print("  • Entity recognition")
    print("  • Data standardization")
    
    print(f"\n🎯 Semantic Technologies:")
    print("  • RDF (Resource Description Framework)")
    print("  • OWL (Web Ontology Language)")
    print("  • SKOS (Simple Knowledge Organization)")
    print("  • Schema.org")
    print("  • Knowledge graphs")
    
    print("\n" + "="*70)
    print("✅ Semantic Transformation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_semantic_transformation_graph():
    """Create semantic transformation workflow"""
    workflow = StateGraph(SemanticTransformationPattern)
    
    workflow.add_node("initialize", initialize_semantic_mapper_agent)
    workflow.add_node("build_mappings", build_semantic_mappings_agent)
    workflow.add_node("build_ontology", build_ontology_agent)
    workflow.add_node("transform", transform_to_semantic_model_agent)
    workflow.add_node("enrich", enrich_semantic_data_agent)
    workflow.add_node("analyze", analyze_semantic_transformation_agent)
    workflow.add_node("report", generate_semantic_transformation_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "build_mappings")
    workflow.add_edge("build_mappings", "build_ontology")
    workflow.add_edge("build_ontology", "transform")
    workflow.add_edge("transform", "enrich")
    workflow.add_edge("enrich", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 305: Semantic Transformation MCP Pattern")
    print("="*70)
    
    app = create_semantic_transformation_graph()
    final_state = app.invoke({
        "messages": [],
        "source_data": {},
        "semantic_mappings": {},
        "ontology": {},
        "transformed_data": {},
        "semantic_statistics": {}
    })
    
    print("\n✅ Semantic Transformation Pattern Complete!")


if __name__ == "__main__":
    main()

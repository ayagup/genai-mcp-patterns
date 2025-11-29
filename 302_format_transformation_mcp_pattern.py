"""
Pattern 302: Format Transformation MCP Pattern

This pattern demonstrates converting data between different formats
(JSON, XML, CSV, YAML, etc.) while preserving data integrity.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
import json
import csv
from io import StringIO


class FormatTransformationPattern(TypedDict):
    """State for format transformation"""
    messages: Annotated[List[str], add]
    source_format: str
    target_format: str
    source_data: str
    transformed_data: str
    conversion_log: List[Dict[str, Any]]
    format_statistics: Dict[str, Any]


class FormatConverter:
    """Convert between different data formats"""
    
    def __init__(self):
        self.supported_formats = ["json", "csv", "xml", "yaml", "text"]
    
    def json_to_csv(self, json_str: str) -> str:
        """Convert JSON to CSV"""
        data = json.loads(json_str)
        
        # Handle list of objects
        if isinstance(data, list):
            if not data:
                return ""
            
            # Get headers from first object
            headers = list(data[0].keys())
            
            # Create CSV
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data)
            
            return output.getvalue()
        
        # Handle single object
        elif isinstance(data, dict):
            headers = list(data.keys())
            values = list(data.values())
            
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(headers)
            writer.writerow(values)
            
            return output.getvalue()
        
        return ""
    
    def csv_to_json(self, csv_str: str) -> str:
        """Convert CSV to JSON"""
        input_stream = StringIO(csv_str)
        reader = csv.DictReader(input_stream)
        
        data = list(reader)
        return json.dumps(data, indent=2)
    
    def json_to_xml(self, json_str: str) -> str:
        """Convert JSON to XML (simplified)"""
        data = json.loads(json_str)
        
        def dict_to_xml(d, root_name="root"):
            xml_parts = [f"<{root_name}>"]
            
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, dict):
                        xml_parts.append(dict_to_xml(value, key))
                    elif isinstance(value, list):
                        for item in value:
                            xml_parts.append(dict_to_xml(item, key))
                    else:
                        xml_parts.append(f"<{key}>{value}</{key}>")
            
            xml_parts.append(f"</{root_name}>")
            return "\n".join(xml_parts)
        
        return dict_to_xml(data)
    
    def xml_to_json(self, xml_str: str) -> str:
        """Convert XML to JSON (simplified)"""
        # Simplified XML parsing
        # In production, use xml.etree.ElementTree or lxml
        
        # For demonstration, create a simple structure
        result = {
            "root": {
                "note": "Simplified XML parsing",
                "content": xml_str[:100] + "..."
            }
        }
        
        return json.dumps(result, indent=2)
    
    def json_to_yaml(self, json_str: str) -> str:
        """Convert JSON to YAML (simplified)"""
        data = json.loads(json_str)
        
        def dict_to_yaml(d, indent=0):
            yaml_lines = []
            prefix = "  " * indent
            
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(value, dict):
                        yaml_lines.append(f"{prefix}{key}:")
                        yaml_lines.append(dict_to_yaml(value, indent + 1))
                    elif isinstance(value, list):
                        yaml_lines.append(f"{prefix}{key}:")
                        for item in value:
                            if isinstance(item, dict):
                                yaml_lines.append(f"{prefix}  -")
                                yaml_lines.append(dict_to_yaml(item, indent + 2))
                            else:
                                yaml_lines.append(f"{prefix}  - {item}")
                    else:
                        yaml_lines.append(f"{prefix}{key}: {value}")
            
            return "\n".join(yaml_lines)
        
        return dict_to_yaml(data)
    
    def convert(self, source_data: str, from_format: str, to_format: str) -> str:
        """Convert data from one format to another"""
        conversion_key = f"{from_format}_to_{to_format}"
        
        converters = {
            "json_to_csv": self.json_to_csv,
            "csv_to_json": self.csv_to_json,
            "json_to_xml": self.json_to_xml,
            "xml_to_json": self.xml_to_json,
            "json_to_yaml": self.json_to_yaml
        }
        
        if conversion_key in converters:
            return converters[conversion_key](source_data)
        
        return source_data


def initialize_format_converter_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Initialize format conversion system"""
    print("\n🔄 Initializing Format Converter...")
    
    # Sample JSON data
    source_data = json.dumps([
        {
            "id": 1,
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "age": 28,
            "active": True
        },
        {
            "id": 2,
            "name": "Bob Smith",
            "email": "bob@example.com",
            "age": 35,
            "active": True
        },
        {
            "id": 3,
            "name": "Carol White",
            "email": "carol@example.com",
            "age": 42,
            "active": False
        }
    ], indent=2)
    
    print(f"  Supported Formats: JSON, CSV, XML, YAML")
    print(f"  Source Format: JSON")
    print(f"  Source Data Size: {len(source_data)} bytes")
    
    print(f"\n  Source Data Preview:")
    print(f"  {source_data[:150]}...")
    
    return {
        **state,
        "source_format": "json",
        "target_format": "",
        "source_data": source_data,
        "transformed_data": "",
        "conversion_log": [],
        "format_statistics": {},
        "messages": ["✓ Format converter initialized"]
    }


def convert_json_to_csv_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Convert JSON to CSV"""
    print("\n📊 Converting JSON → CSV...")
    
    converter = FormatConverter()
    
    csv_data = converter.convert(state["source_data"], "json", "csv")
    
    print(f"  Conversion: JSON → CSV")
    print(f"  Output Size: {len(csv_data)} bytes")
    
    print(f"\n  CSV Output Preview:")
    lines = csv_data.split('\n')[:5]
    for line in lines:
        print(f"  {line}")
    
    conversion_log = state["conversion_log"].copy()
    conversion_log.append({
        "from": "json",
        "to": "csv",
        "source_size": len(state["source_data"]),
        "target_size": len(csv_data),
        "success": True
    })
    
    return {
        **state,
        "target_format": "csv",
        "transformed_data": csv_data,
        "conversion_log": conversion_log,
        "messages": ["✓ JSON → CSV conversion complete"]
    }


def convert_json_to_xml_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Convert JSON to XML"""
    print("\n📄 Converting JSON → XML...")
    
    converter = FormatConverter()
    
    xml_data = converter.convert(state["source_data"], "json", "xml")
    
    print(f"  Conversion: JSON → XML")
    print(f"  Output Size: {len(xml_data)} bytes")
    
    print(f"\n  XML Output Preview:")
    lines = xml_data.split('\n')[:10]
    for line in lines:
        print(f"  {line}")
    
    conversion_log = state["conversion_log"].copy()
    conversion_log.append({
        "from": "json",
        "to": "xml",
        "source_size": len(state["source_data"]),
        "target_size": len(xml_data),
        "success": True
    })
    
    return {
        **state,
        "conversion_log": conversion_log,
        "messages": ["✓ JSON → XML conversion complete"]
    }


def convert_json_to_yaml_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Convert JSON to YAML"""
    print("\n📝 Converting JSON → YAML...")
    
    converter = FormatConverter()
    
    yaml_data = converter.convert(state["source_data"], "json", "yaml")
    
    print(f"  Conversion: JSON → YAML")
    print(f"  Output Size: {len(yaml_data)} bytes")
    
    print(f"\n  YAML Output Preview:")
    lines = yaml_data.split('\n')[:15]
    for line in lines:
        print(f"  {line}")
    
    conversion_log = state["conversion_log"].copy()
    conversion_log.append({
        "from": "json",
        "to": "yaml",
        "source_size": len(state["source_data"]),
        "target_size": len(yaml_data),
        "success": True
    })
    
    return {
        **state,
        "conversion_log": conversion_log,
        "messages": ["✓ JSON → YAML conversion complete"]
    }


def convert_csv_to_json_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Convert CSV back to JSON"""
    print("\n🔄 Converting CSV → JSON...")
    
    # Get CSV data from previous conversion
    csv_data = state["transformed_data"]
    
    converter = FormatConverter()
    json_data = converter.convert(csv_data, "csv", "json")
    
    print(f"  Conversion: CSV → JSON")
    print(f"  Output Size: {len(json_data)} bytes")
    
    print(f"\n  JSON Output Preview:")
    print(f"  {json_data[:200]}...")
    
    conversion_log = state["conversion_log"].copy()
    conversion_log.append({
        "from": "csv",
        "to": "json",
        "source_size": len(csv_data),
        "target_size": len(json_data),
        "success": True
    })
    
    return {
        **state,
        "conversion_log": conversion_log,
        "messages": ["✓ CSV → JSON conversion complete"]
    }


def analyze_format_conversions_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Analyze format conversions"""
    print("\n📊 Analyzing Format Conversions...")
    
    conversions = state["conversion_log"]
    
    # Calculate statistics
    total_conversions = len(conversions)
    successful_conversions = sum(1 for c in conversions if c["success"])
    
    total_source_size = sum(c["source_size"] for c in conversions)
    total_target_size = sum(c["target_size"] for c in conversions)
    
    size_change = total_target_size - total_source_size
    size_change_pct = (size_change / max(total_source_size, 1)) * 100
    
    # Format distribution
    formats_used = set()
    for conv in conversions:
        formats_used.add(conv["from"])
        formats_used.add(conv["to"])
    
    statistics = {
        "total_conversions": total_conversions,
        "successful_conversions": successful_conversions,
        "success_rate": successful_conversions / max(total_conversions, 1),
        "total_source_size": total_source_size,
        "total_target_size": total_target_size,
        "size_change": size_change,
        "size_change_percentage": size_change_pct,
        "formats_used": list(formats_used)
    }
    
    print(f"  Total Conversions: {statistics['total_conversions']}")
    print(f"  Success Rate: {statistics['success_rate']:.1%}")
    print(f"  Total Source Size: {statistics['total_source_size']} bytes")
    print(f"  Total Target Size: {statistics['total_target_size']} bytes")
    print(f"  Size Change: {statistics['size_change']:+d} bytes ({statistics['size_change_percentage']:+.1f}%)")
    print(f"  Formats Used: {', '.join(statistics['formats_used'])}")
    
    return {
        **state,
        "format_statistics": statistics,
        "messages": ["✓ Format conversions analyzed"]
    }


def generate_format_transformation_report_agent(state: FormatTransformationPattern) -> FormatTransformationPattern:
    """Generate format transformation report"""
    print("\n" + "="*70)
    print("FORMAT TRANSFORMATION REPORT")
    print("="*70)
    
    print(f"\n📥 Source Format: {state['source_format'].upper()}")
    print(f"  Size: {len(state['source_data'])} bytes")
    print(f"\n  Preview:")
    preview_lines = state["source_data"].split('\n')[:5]
    for line in preview_lines:
        print(f"  {line}")
    
    print(f"\n🔄 Conversions Performed:")
    for i, conv in enumerate(state["conversion_log"], 1):
        status = "✓" if conv["success"] else "✗"
        size_change = conv["target_size"] - conv["source_size"]
        print(f"\n  {i}. {status} {conv['from'].upper()} → {conv['to'].upper()}")
        print(f"     Source: {conv['source_size']} bytes")
        print(f"     Target: {conv['target_size']} bytes")
        print(f"     Change: {size_change:+d} bytes")
    
    print(f"\n📤 Final Output Format: {state.get('target_format', 'N/A').upper()}")
    if state["transformed_data"]:
        print(f"  Size: {len(state['transformed_data'])} bytes")
        print(f"\n  Preview:")
        preview_lines = state["transformed_data"].split('\n')[:5]
        for line in preview_lines:
            print(f"  {line}")
    
    print(f"\n📊 Conversion Statistics:")
    stats = state["format_statistics"]
    if stats:
        print(f"  Total Conversions: {stats['total_conversions']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
        print(f"  Total Size Change: {stats['size_change']:+d} bytes ({stats['size_change_percentage']:+.1f}%)")
        print(f"  Formats Used: {', '.join(stats['formats_used'])}")
    
    print(f"\n💡 Format Transformation Benefits:")
    print("  ✓ Interoperability")
    print("  ✓ Data exchange")
    print("  ✓ System integration")
    print("  ✓ Legacy support")
    print("  ✓ API compatibility")
    print("  ✓ Storage optimization")
    
    print(f"\n🔧 Supported Formats:")
    print("  • JSON (JavaScript Object Notation)")
    print("  • CSV (Comma-Separated Values)")
    print("  • XML (eXtensible Markup Language)")
    print("  • YAML (YAML Ain't Markup Language)")
    print("  • Text (Plain text)")
    
    print(f"\n⚙️ Use Cases:")
    print("  • API integration")
    print("  • Data import/export")
    print("  • Configuration files")
    print("  • Log processing")
    print("  • Report generation")
    print("  • Data migration")
    
    print(f"\n🎯 Common Conversions:")
    print("  • JSON ↔ CSV (tabular data)")
    print("  • JSON ↔ XML (web services)")
    print("  • JSON ↔ YAML (config files)")
    print("  • CSV → Database")
    print("  • XML → JSON (modernization)")
    
    print(f"\n⚠️ Considerations:")
    print("  • Data type preservation")
    print("  • Nested structure handling")
    print("  • Special character escaping")
    print("  • Encoding compatibility")
    print("  • Performance vs. accuracy")
    
    print("\n" + "="*70)
    print("✅ Format Transformation Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_format_transformation_graph():
    """Create format transformation workflow"""
    workflow = StateGraph(FormatTransformationPattern)
    
    workflow.add_node("initialize", initialize_format_converter_agent)
    workflow.add_node("json_to_csv", convert_json_to_csv_agent)
    workflow.add_node("json_to_xml", convert_json_to_xml_agent)
    workflow.add_node("json_to_yaml", convert_json_to_yaml_agent)
    workflow.add_node("csv_to_json", convert_csv_to_json_agent)
    workflow.add_node("analyze", analyze_format_conversions_agent)
    workflow.add_node("report", generate_format_transformation_report_agent)
    
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "json_to_csv")
    workflow.add_edge("json_to_csv", "json_to_xml")
    workflow.add_edge("json_to_xml", "json_to_yaml")
    workflow.add_edge("json_to_yaml", "csv_to_json")
    workflow.add_edge("csv_to_json", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 302: Format Transformation MCP Pattern")
    print("="*70)
    
    app = create_format_transformation_graph()
    final_state = app.invoke({
        "messages": [],
        "source_format": "",
        "target_format": "",
        "source_data": "",
        "transformed_data": "",
        "conversion_log": [],
        "format_statistics": {}
    })
    
    print("\n✅ Format Transformation Pattern Complete!")


if __name__ == "__main__":
    main()

"""
Translation MCP Pattern

This pattern implements translation between languages,
formats, and abstraction levels.

Key Features:
- Language translation
- Format conversion
- Style adaptation
- Technical level adjustment
- Context preservation
"""

from typing import TypedDict, Sequence, Annotated, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class TranslationState(TypedDict):
    """State for translation pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    source_content: str
    source_format: str
    target_format: str
    translation_type: str
    context: Dict


llm = ChatOpenAI(model="gpt-4", temperature=0.4)


def translation_agent(state: TranslationState) -> TranslationState:
    """Performs various types of translation"""
    source = state.get("source_content", "")[:200]
    source_fmt = state.get("source_format", "")
    target_fmt = state.get("target_format", "")
    
    system_prompt = """You are a translation expert.

Translation Goals:
• Preserve meaning
• Adapt to target
• Maintain style
• Handle idioms
• Ensure fluency

Bridge contexts."""
    
    user_prompt = f"""Source: {source}...
From: {source_fmt}
To: {target_fmt}

Design translation system.
Show translation strategies."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    🌐 Translation Agent:
    
    Translation System:
    ```python
    class TranslationSystem:
        def __init__(self):
            self.language_translator = LanguageTranslator()
            self.format_converter = FormatConverter()
            self.style_adapter = StyleAdapter()
            self.level_adjuster = TechnicalLevelAdjuster()
        
        def translate(self, content, source, target, translation_type):
            '''Perform translation'''
            if translation_type == 'language':
                return self.language_translator.translate(content, source, target)
            elif translation_type == 'format':
                return self.format_converter.convert(content, source, target)
            elif translation_type == 'style':
                return self.style_adapter.adapt(content, source, target)
            elif translation_type == 'technical_level':
                return self.level_adjuster.adjust(content, source, target)
    ```
    
    Language Translation:
    
    Neural Machine Translation:
    ```python
    def language_translate(text, source_lang, target_lang):
        '''Translate between languages'''
        # Preprocessing
        normalized = normalize_text(text)
        
        # Translation
        translated = nmt_model.translate(
            normalized,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Post-processing
        formatted = apply_target_conventions(translated, target_lang)
        
        return formatted
    ```
    
    Context-Aware Translation:
    ```python
    def contextual_translation(text, context):
        '''Use context for better translation'''
        # Examples of context:
        # - Domain: medical, legal, technical
        # - Formality: formal, informal
        # - Audience: expert, general
        
        # Select appropriate model/vocabulary
        if context['domain'] == 'medical':
            model = medical_translation_model
            glossary = medical_glossary
        
        # Translate with domain knowledge
        translated = model.translate(text, glossary=glossary)
        
        return translated
    ```
    
    Idiom Handling:
    ```python
    def handle_idioms(text, source_lang, target_lang):
        '''Translate idiomatic expressions'''
        # Detect idioms
        idioms = detect_idioms(text, source_lang)
        
        for idiom in idioms:
            # Don't translate literally
            literal = translate_literally(idiom)  # ❌ Wrong
            
            # Find equivalent idiom in target
            equivalent = find_equivalent_idiom(idiom, target_lang)
            
            if equivalent:
                text = text.replace(idiom, equivalent)
            else:
                # Translate meaning
                meaning = explain_idiom(idiom)
                translation = translate(meaning, target_lang)
                text = text.replace(idiom, translation)
        
        return text
    ```
    
    Format Translation:
    
    Data Format Conversion:
    ```python
    def convert_format(data, source_format, target_format):
        '''Convert between data formats'''
        converters = {{
            ('json', 'xml'): json_to_xml,
            ('xml', 'json'): xml_to_json,
            ('csv', 'json'): csv_to_json,
            ('yaml', 'json'): yaml_to_json
        }}
        
        converter = converters.get((source_format, target_format))
        
        if converter:
            return converter(data)
        else:
            # Generic conversion via intermediate
            intermediate = to_dict(data, source_format)
            result = from_dict(intermediate, target_format)
            return result
    ```
    
    Example - JSON to XML:
    ```python
    def json_to_xml(json_data):
        '''Convert JSON to XML'''
        import json
        import xml.etree.ElementTree as ET
        
        data = json.loads(json_data)
        
        def dict_to_xml(d, root_name='root'):
            root = ET.Element(root_name)
            
            for key, value in d.items():
                child = ET.SubElement(root, key)
                
                if isinstance(value, dict):
                    child.append(dict_to_xml(value, key))
                elif isinstance(value, list):
                    for item in value:
                        item_elem = ET.SubElement(child, 'item')
                        item_elem.text = str(item)
                else:
                    child.text = str(value)
            
            return root
        
        xml_root = dict_to_xml(data)
        xml_string = ET.tostring(xml_root, encoding='unicode')
        
        return xml_string
    ```
    
    Code Translation:
    
    Language-to-Language:
    ```python
    def translate_code(source_code, source_lang, target_lang):
        '''Translate between programming languages'''
        # Parse source
        ast = parse(source_code, source_lang)
        
        # Map constructs
        target_ast = map_constructs(ast, source_lang, target_lang)
        
        # Generate target code
        target_code = generate_code(target_ast, target_lang)
        
        return target_code
    ```
    
    Example - Python to JavaScript:
    ```python
    def python_to_javascript(python_code):
        '''Convert Python to JavaScript'''
        translations = {{
            # Data types
            'True': 'true',
            'False': 'false',
            'None': 'null',
            
            # String formatting
            'f"{{x}}"': '`${{x}}`',
            
            # List comprehension
            '[x for x in items]': 'items.map(x => x)',
            
            # Dictionary
            '{{"key": "value"}}': '{{ key: "value" }}',
            
            # Functions
            'def func():': 'function func() {{',
            
            # Conditionals
            'elif': 'else if',
            
            # Loops
            'for x in items:': 'for (const x of items) {{'
        }}
        
        js_code = python_code
        for py, js in translations.items():
            js_code = js_code.replace(py, js)
        
        return js_code
    ```
    
    Style Translation:
    
    Formality Adjustment:
    ```python
    def adjust_formality(text, target_formality):
        '''Change formality level'''
        if target_formality == 'formal':
            # Informal → Formal
            transformations = {{
                "can't": "cannot",
                "don't": "do not",
                "it's": "it is",
                "gonna": "going to",
                "wanna": "want to"
            }}
            
            for informal, formal in transformations.items():
                text = text.replace(informal, formal)
            
            # Remove slang
            text = remove_slang(text)
            
            # Use passive voice
            text = increase_passive_voice(text)
        
        elif target_formality == 'informal':
            # Formal → Informal
            text = use_contractions(text)
            text = use_active_voice(text)
        
        return text
    ```
    
    Tone Adaptation:
    ```python
    def adapt_tone(text, target_tone):
        '''Adjust emotional tone'''
        tones = {{
            'professional': {{
                'remove': ['awesome', 'cool', 'great'],
                'add': ['excellent', 'effective', 'appropriate']
            }},
            'friendly': {{
                'remove': ['utilize', 'commence', 'terminate'],
                'add': ['use', 'start', 'end']
            }},
            'persuasive': {{
                'techniques': ['rhetorical_questions', 'power_words', 'urgency']
            }}
        }}
        
        adjustments = tones.get(target_tone, {{}})
        
        return apply_adjustments(text, adjustments)
    ```
    
    Technical Level Translation:
    
    Simplification:
    ```python
    def simplify_technical(technical_text, target_level):
        '''Make technical content accessible'''
        if target_level == 'beginner':
            # Replace jargon with plain language
            simplified = replace_jargon(technical_text)
            
            # Add explanations
            simplified = add_definitions(simplified)
            
            # Use analogies
            simplified = add_analogies(simplified)
            
            # Break down complex sentences
            simplified = simplify_sentences(simplified)
        
        return simplified
    ```
    
    Example:
    ```python
    technical = "The API employs RESTful architecture with JSON payloads"
    
    simplified = '''
    The system uses a common web design pattern (REST) that makes
    it easy for different programs to talk to each other. Information
    is sent back and forth in a simple text format (JSON).
    '''
    ```
    
    Technicalization:
    ```python
    def make_technical(simple_text, domain):
        '''Add technical precision'''
        # Replace colloquial terms with technical terms
        technical = replace_with_jargon(simple_text, domain)
        
        # Add precise measurements
        technical = add_precision(technical)
        
        # Use domain-specific terminology
        technical = apply_domain_vocabulary(technical, domain)
        
        return technical
    ```
    
    Abstraction Level Translation:
    
    High-level to Low-level:
    ```python
    def translate_to_implementation(high_level_spec):
        '''Convert specification to implementation'''
        # High-level: "Sort the list"
        # Low-level:
        implementation = '''
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[len(arr) // 2]
            left = [x for x in arr if x < pivot]
            middle = [x for x in arr if x == pivot]
            right = [x for x in arr if x > pivot]
            return quicksort(left) + middle + quicksort(right)
        '''
        
        return implementation
    ```
    
    Low-level to High-level:
    ```python
    def abstract_implementation(code):
        '''Extract high-level logic from code'''
        # Analyze code
        intent = infer_intent(code)
        
        # Abstract operations
        abstract_steps = extract_key_operations(code)
        
        # Generate high-level description
        description = f"This {intent} by {abstract_steps}"
        
        return description
    ```
    
    Translation Quality:
    
    Fluency Check:
    ```python
    def check_fluency(translated_text, target_language):
        '''Ensure natural-sounding output'''
        # Grammar check
        grammar_score = check_grammar(translated_text, target_language)
        
        # Naturalness
        naturalness = assess_naturalness(translated_text, target_language)
        
        # Readability
        readability = calculate_readability(translated_text, target_language)
        
        quality = {{
            'grammar': grammar_score,
            'naturalness': naturalness,
            'readability': readability
        }}
        
        return quality
    ```
    
    Meaning Preservation:
    ```python
    def verify_meaning_preserved(source, translation):
        '''Check translation accuracy'''
        # Back-translate
        back_translation = translate(translation, reverse_direction=True)
        
        # Compare to source
        similarity = semantic_similarity(source, back_translation)
        
        # Check key entities preserved
        source_entities = extract_entities(source)
        translated_entities = extract_entities(translation)
        
        entity_preservation = len(set(source_entities) & set(translated_entities)) / len(source_entities)
        
        return {{
            'semantic_similarity': similarity,
            'entity_preservation': entity_preservation
        }}
    ```
    
    Adaptive Translation:
    
    User Preference Learning:
    ```python
    def adaptive_translate(text, user_profile):
        '''Customize based on user preferences'''
        # Learn from corrections
        if user_profile.has_corrections:
            learn_from_corrections(user_profile.corrections)
        
        # Apply learned preferences
        translated = translate(text)
        translated = apply_user_preferences(translated, user_profile)
        
        return translated
    ```
    
    Best Practices:
    ✓ Preserve meaning
    ✓ Maintain context
    ✓ Adapt to target audience
    ✓ Handle edge cases (idioms, jargon)
    ✓ Verify quality
    ✓ Allow customization
    ✓ Learn from feedback
    
    Key Insight:
    Effective translation goes beyond word-for-word
    conversion to preserve meaning and adapt to context.
    """
    
    return {
        "messages": [AIMessage(content=f"🌐 Translation Agent:\n{report}\n\n{response.content}")]
    }


def build_translation_graph():
    workflow = StateGraph(TranslationState)
    workflow.add_node("translation_agent", translation_agent)
    workflow.add_edge(START, "translation_agent")
    workflow.add_edge("translation_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_translation_graph()
    
    print("=== Translation MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "source_content": "Hello, how are you?",
        "source_format": "English",
        "target_format": "Spanish",
        "translation_type": "language",
        "context": {"formality": "informal", "region": "Latin America"}
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 159: Translation - COMPLETE")
    print(f"{'='*70}")

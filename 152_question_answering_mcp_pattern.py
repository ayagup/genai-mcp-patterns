"""
Question-Answering MCP Pattern

This pattern implements question-answering systems with
question understanding, answer retrieval, and generation.

Key Features:
- Question classification
- Answer retrieval
- Knowledge synthesis
- Evidence-based responses
- Confidence scoring
"""

from typing import TypedDict, Sequence, Annotated, List, Dict
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END


class QuestionAnsweringState(TypedDict):
    """State for question-answering pattern"""
    messages: Annotated[Sequence[HumanMessage | SystemMessage | AIMessage], operator.add]
    question: str
    question_type: str
    retrieved_evidence: List[Dict]
    answer: str
    confidence: float


llm = ChatOpenAI(model="gpt-4", temperature=0.3)


def qa_agent(state: QuestionAnsweringState) -> QuestionAnsweringState:
    """Manages question-answering operations"""
    question = state.get("question", "")
    
    system_prompt = """You are a question-answering expert.

Question-Answering System:
• Understand question intent
• Retrieve relevant knowledge
• Synthesize accurate answer
• Provide evidence
• Assess confidence

Deliver precise, evidence-based answers."""
    
    user_prompt = f"""Question: {question}

Design QA system.
Show question analysis and answer generation."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    response = llm.invoke(messages)
    
    report = f"""
    ❓ Question-Answering Agent:
    
    QA Task:
    • Question: {question[:100]}...
    • Goal: Provide accurate, evidence-based answer
    
    QA System Implementation:
    ```python
    class QuestionAnsweringSystem:
        '''End-to-end QA pipeline'''
        
        def __init__(self, knowledge_base):
            self.knowledge_base = knowledge_base
            self.question_analyzer = QuestionAnalyzer()
            self.retriever = EvidenceRetriever()
            self.answer_generator = AnswerGenerator()
        
        def answer_question(self, question):
            '''Main QA pipeline'''
            # 1. Analyze question
            analysis = self.question_analyzer.analyze(question)
            
            # 2. Retrieve evidence
            evidence = self.retriever.retrieve(
                query=analysis['reformulated'],
                question_type=analysis['type'],
                top_k=5
            )
            
            # 3. Generate answer
            answer = self.answer_generator.generate(
                question=question,
                evidence=evidence,
                answer_type=analysis['answer_type']
            )
            
            # 4. Verify and score
            verified = self.verify_answer(answer, evidence)
            
            return {{
                'question': question,
                'answer': verified['answer'],
                'evidence': evidence,
                'confidence': verified['confidence'],
                'answer_type': analysis['answer_type']
            }}
    ```
    
    Question Analysis:
    
    Question Classification:
    ```python
    QUESTION_TYPES = {{
        'factoid': {{
            'patterns': ['who', 'what', 'when', 'where'],
            'answer_type': 'entity',
            'example': "Who invented Python?"
        }},
        'definition': {{
            'patterns': ['what is', 'define', 'meaning of'],
            'answer_type': 'explanation',
            'example': "What is a decorator?"
        }},
        'how_to': {{
            'patterns': ['how to', 'how do I', 'steps to'],
            'answer_type': 'procedure',
            'example': "How to write a decorator?"
        }},
        'why': {{
            'patterns': ['why', 'reason for', 'purpose of'],
            'answer_type': 'explanation',
            'example': "Why use decorators?"
        }},
        'comparison': {{
            'patterns': ['difference between', 'versus', 'compare'],
            'answer_type': 'analysis',
            'example': "Difference between list and tuple?"
        }},
        'yes_no': {{
            'patterns': ['is', 'are', 'can', 'does'],
            'answer_type': 'boolean',
            'example': "Is Python object-oriented?"
        }}
    }}
    
    def classify_question(question):
        '''Determine question type'''
        q_lower = question.lower()
        
        for qtype, info in QUESTION_TYPES.items():
            for pattern in info['patterns']:
                if q_lower.startswith(pattern):
                    return {{
                        'type': qtype,
                        'answer_type': info['answer_type']
                    }}
        
        return {{'type': 'open_ended', 'answer_type': 'explanation'}}
    ```
    
    Question Reformulation:
    ```python
    def reformulate_question(question):
        '''Improve question for retrieval'''
        # Remove question words for keyword search
        keywords = extract_keywords(question)
        
        # Expand with synonyms
        expanded = expand_with_synonyms(keywords)
        
        # Multiple formulations
        formulations = [
            question,  # Original
            ' '.join(keywords),  # Keywords only
            generate_declarative(question),  # As statement
            ' '.join(expanded)  # With synonyms
        ]
        
        return formulations
    ```
    
    Evidence Retrieval:
    
    Multi-Source Retrieval:
    ```python
    class EvidenceRetriever:
        '''Retrieve relevant information'''
        
        def __init__(self, knowledge_base):
            self.kb = knowledge_base
            self.vector_search = VectorSearch()
            self.keyword_search = KeywordSearch()
        
        def retrieve(self, query, question_type, top_k=5):
            '''Retrieve evidence from KB'''
            # Hybrid search
            vector_results = self.vector_search.search(query, top_k)
            keyword_results = self.keyword_search.search(query, top_k)
            
            # Combine and rerank
            combined = self.combine_results(vector_results, keyword_results)
            reranked = self.rerank(combined, query, question_type)
            
            return reranked[:top_k]
        
        def rerank(self, results, query, question_type):
            '''Rerank by relevance'''
            scored = []
            
            for result in results:
                score = (
                    self.semantic_similarity(query, result) * 0.5 +
                    self.answer_type_match(result, question_type) * 0.3 +
                    self.recency(result) * 0.1 +
                    self.source_quality(result) * 0.1
                )
                scored.append((result, score))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            return [r for r, s in scored]
    ```
    
    Answer Generation:
    
    Extractive QA:
    ```python
    def extractive_answer(question, passages):
        '''Extract answer span from text'''
        # Use reading comprehension model
        # Find exact answer in passages
        
        best_answer = None
        best_score = 0
        
        for passage in passages:
            # Score each span
            for span in extract_spans(passage):
                score = score_answer_span(question, span, passage)
                
                if score > best_score:
                    best_answer = span
                    best_score = score
        
        return {{
            'answer': best_answer,
            'confidence': best_score,
            'source': passage
        }}
    ```
    
    Abstractive QA:
    ```python
    def abstractive_answer(question, evidence):
        '''Generate answer from evidence'''
        # Synthesize from multiple sources
        
        prompt = f'''
        Question: {{question}}
        
        Evidence:
        {{format_evidence(evidence)}}
        
        Based on the evidence above, provide a concise,
        accurate answer to the question.
        '''
        
        answer = llm.generate(prompt)
        
        return answer
    ```
    
    Hybrid QA:
    ```python
    def hybrid_answer(question, evidence):
        '''Combine extraction and generation'''
        # Extract key facts
        facts = [extract_key_fact(e) for e in evidence]
        
        # Generate coherent answer from facts
        answer = synthesize_from_facts(question, facts)
        
        # Add citations
        cited_answer = add_citations(answer, evidence)
        
        return cited_answer
    ```
    
    Answer Types:
    
    Factoid Answers:
    ```python
    def answer_factoid(question, evidence):
        '''Short, specific answer'''
        # "Who invented Python?" → "Guido van Rossum"
        
        # Extract entity
        entity = extract_entity(evidence, question)
        
        return {{
            'answer': entity,
            'type': 'factoid',
            'format': 'short'
        }}
    ```
    
    Explanation Answers:
    ```python
    def answer_explanation(question, evidence):
        '''Detailed explanation'''
        # "What is a decorator?" → [paragraph explanation]
        
        # Gather relevant info
        definition = extract_definition(evidence)
        examples = extract_examples(evidence)
        use_cases = extract_use_cases(evidence)
        
        # Structure explanation
        explanation = f'''
        {{definition}}
        
        Key characteristics:
        {{format_characteristics()}}
        
        Example:
        {{examples[0]}}
        
        Common uses:
        {{format_uses(use_cases)}}
        '''
        
        return explanation
    ```
    
    Procedural Answers:
    ```python
    def answer_how_to(question, evidence):
        '''Step-by-step procedure'''
        # "How to write decorator?" → [steps]
        
        steps = extract_steps(evidence)
        
        formatted = []
        for i, step in enumerate(steps, 1):
            formatted.append(f"{{i}}. {{step}}")
        
        return '\\n'.join(formatted)
    ```
    
    Confidence Scoring:
    
    Answer Confidence:
    ```python
    def calculate_confidence(answer, question, evidence):
        '''Assess answer reliability'''
        confidence = 0
        
        # Evidence quality
        confidence += evidence_quality_score(evidence) * 0.3
        
        # Answer-question alignment
        confidence += alignment_score(answer, question) * 0.3
        
        # Source agreement (multiple sources say same thing)
        confidence += source_agreement(answer, evidence) * 0.2
        
        # Model confidence
        confidence += model_confidence_score(answer) * 0.2
        
        return confidence
    
    def should_answer(confidence, threshold=0.7):
        '''Decide whether to answer'''
        if confidence > threshold:
            return True
        else:
            return False  # Say "I don't know"
    ```
    
    Answer Verification:
    
    Fact Checking:
    ```python
    def verify_answer(answer, evidence):
        '''Check answer accuracy'''
        # Does evidence support answer?
        support = []
        
        for fact in extract_facts(answer):
            supported = any(
                fact_in_evidence(fact, e) 
                for e in evidence
            )
            support.append(supported)
        
        # All facts must be supported
        verified = all(support)
        
        return verified
    ```
    
    Consistency Check:
    ```python
    def check_consistency(answer, knowledge_base):
        '''Ensure no contradictions'''
        # Check against known facts
        contradictions = []
        
        for known_fact in knowledge_base.facts:
            if contradicts(answer, known_fact):
                contradictions.append(known_fact)
        
        if contradictions:
            # Revise answer or lower confidence
            return False, contradictions
        
        return True, []
    ```
    
    Multi-Hop QA:
    
    Reasoning Chains:
    ```python
    def multi_hop_reasoning(question):
        '''Answer requiring multiple steps'''
        # "What language was used to create Django's ORM?"
        # → Need: Django → Django uses Python → Answer: Python
        
        # Decompose into sub-questions
        sub_questions = [
            "What is Django?",
            "What language is Django written in?"
        ]
        
        # Answer each
        answers = []
        for sq in sub_questions:
            ans = answer_question(sq)
            answers.append(ans)
        
        # Synthesize final answer
        final = synthesize_multi_hop(answers, question)
        
        return final
    ```
    
    Response Formatting:
    
    Structured Answers:
    ```python
    def format_answer(answer, question_type):
        '''Format appropriately'''
        if question_type == 'yes_no':
            return f"{{answer['decision']}}. {{answer['explanation']}}"
        
        elif question_type == 'factoid':
            return f"{{answer['entity']}}"
        
        elif question_type == 'how_to':
            return format_steps(answer['steps'])
        
        elif question_type == 'comparison':
            return format_comparison_table(answer)
        
        else:
            return answer
    ```
    
    Applications:
    
    FAQ System:
    ```python
    faq_qa = QuestionAnsweringSystem(
        knowledge_base=faq_database,
        retriever='semantic_search'
    )
    ```
    
    Document QA:
    ```python
    doc_qa = QuestionAnsweringSystem(
        knowledge_base=document_collection,
        retriever='dense_retrieval'
    )
    ```
    
    Best Practices:
    ✓ Classify question type first
    ✓ Use hybrid retrieval
    ✓ Verify answers with evidence
    ✓ Calculate confidence scores
    ✓ Say "I don't know" when uncertain
    ✓ Cite sources
    
    Key Insight:
    Effective QA requires understanding questions,
    retrieving relevant evidence, and synthesizing
    accurate, well-supported answers.
    """
    
    return {
        "messages": [AIMessage(content=f"❓ QA Agent:\n{report}\n\n{response.content}")],
        "answer": "Generated answer based on evidence",
        "confidence": 0.85
    }


def build_qa_graph():
    workflow = StateGraph(QuestionAnsweringState)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_edge(START, "qa_agent")
    workflow.add_edge("qa_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_qa_graph()
    
    print("=== Question-Answering MCP Pattern ===\n")
    
    state = {
        "messages": [],
        "question": "What is a Python decorator and how does it work?",
        "question_type": "",
        "retrieved_evidence": [],
        "answer": "",
        "confidence": 0.0
    }
    
    result = graph.invoke(state)
    
    for msg in result["messages"]:
        print(f"\n{msg.content}")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("Pattern 152: Question-Answering - COMPLETE")
    print(f"{'='*70}")

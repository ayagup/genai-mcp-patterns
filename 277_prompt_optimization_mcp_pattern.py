"""
Pattern 277: Prompt Optimization MCP Pattern

This pattern demonstrates LLM prompt optimization including prompt engineering,
token efficiency, and quality improvement strategies.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class PromptOptimizationState(TypedDict):
    """State for prompt optimization workflow"""
    messages: Annotated[List[str], add]
    prompt_analysis: Dict[str, Any]
    optimization_opportunities: List[Dict[str, Any]]
    optimized_prompts: List[Dict[str, Any]]
    improvement_metrics: Dict[str, Any]


class PromptAnalyzer:
    """Analyzes prompts for optimization opportunities"""
    
    def __init__(self):
        self.best_practices = {
            "clarity": ["Be specific", "Avoid ambiguity", "Use clear instructions"],
            "structure": ["Use sections", "Number steps", "Format with examples"],
            "context": ["Provide necessary context", "Avoid overloading", "Focus on relevant information"],
            "efficiency": ["Remove redundancy", "Use concise language", "Optimize token usage"]
        }
    
    def analyze_prompt(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single prompt"""
        text = prompt["text"]
        tokens = len(text.split())  # Simple token estimation
        
        issues = []
        
        # Check length
        if tokens > 500:
            issues.append({"type": "length", "severity": "high", "issue": "Prompt too long (>500 tokens)"})
        elif tokens > 300:
            issues.append({"type": "length", "severity": "medium", "issue": "Prompt lengthy (>300 tokens)"})
        
        # Check clarity
        if "unclear" in text.lower() or len(text.split('.')) > 10:
            issues.append({"type": "clarity", "severity": "medium", "issue": "May lack clarity"})
        
        # Check for redundancy
        words = text.lower().split()
        if len(words) != len(set(words)):
            redundancy = 1 - (len(set(words)) / len(words))
            if redundancy > 0.3:
                issues.append({"type": "redundancy", "severity": "medium", "issue": f"High redundancy ({redundancy:.1%})"})
        
        # Check structure
        has_examples = "example:" in text.lower() or "for instance" in text.lower()
        has_steps = any(str(i) + "." in text or str(i) + ")" in text for i in range(1, 6))
        
        if not has_examples and tokens > 100:
            issues.append({"type": "structure", "severity": "low", "issue": "Missing examples"})
        
        if not has_steps and "step" in text.lower():
            issues.append({"type": "structure", "severity": "low", "issue": "Steps not numbered"})
        
        # Check context
        if tokens < 20:
            issues.append({"type": "context", "severity": "high", "issue": "Insufficient context"})
        
        return {
            "prompt_id": prompt["id"],
            "token_count": tokens,
            "issues": issues,
            "quality_score": self._calculate_quality_score(issues, tokens)
        }
    
    def _calculate_quality_score(self, issues: List[Dict], tokens: int) -> float:
        """Calculate prompt quality score (0-100)"""
        score = 100.0
        
        for issue in issues:
            if issue["severity"] == "high":
                score -= 20
            elif issue["severity"] == "medium":
                score -= 10
            else:
                score -= 5
        
        # Token efficiency bonus/penalty
        if 50 <= tokens <= 200:
            score += 10  # Optimal length
        elif tokens > 500:
            score -= 15  # Too long
        
        return max(0, min(100, score))


class PromptOptimizer:
    """Generates optimized prompts"""
    
    def __init__(self):
        self.optimization_strategies = {
            "length": {
                "high": "Significantly reduce prompt length by removing unnecessary details",
                "medium": "Condense prompt while maintaining key information",
                "low": "Minor length adjustments"
            },
            "clarity": {
                "high": "Rewrite for maximum clarity with specific instructions",
                "medium": "Improve clarity by restructuring sentences",
                "low": "Minor clarity improvements"
            },
            "redundancy": {
                "high": "Remove duplicate information and repetitive phrases",
                "medium": "Eliminate redundant content",
                "low": "Clean up minor repetitions"
            },
            "structure": {
                "high": "Add clear structure with sections and numbering",
                "medium": "Improve organization with better formatting",
                "low": "Minor structural enhancements"
            },
            "context": {
                "high": "Add essential context for understanding",
                "medium": "Supplement with relevant background",
                "low": "Refine contextual information"
            }
        }
    
    def optimize_prompt(self, prompt: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimized version of prompt"""
        optimizations = []
        expected_token_reduction = 0
        expected_quality_improvement = 0
        
        for issue in analysis["issues"]:
            issue_type = issue["type"]
            severity = issue["severity"]
            
            strategy = self.optimization_strategies.get(issue_type, {}).get(severity, "General optimization")
            optimizations.append({
                "issue": issue["issue"],
                "strategy": strategy,
                "severity": severity
            })
            
            # Estimate improvements
            if issue_type == "length":
                if severity == "high":
                    expected_token_reduction += analysis["token_count"] * 0.40
                    expected_quality_improvement += 15
                elif severity == "medium":
                    expected_token_reduction += analysis["token_count"] * 0.20
                    expected_quality_improvement += 8
            elif issue_type == "redundancy":
                expected_token_reduction += analysis["token_count"] * 0.15
                expected_quality_improvement += 10
            elif severity == "high":
                expected_quality_improvement += 20
            elif severity == "medium":
                expected_quality_improvement += 10
            else:
                expected_quality_improvement += 5
        
        # Generate optimized prompt (simplified example)
        original_text = prompt["text"]
        optimized_text = self._apply_optimizations(original_text, optimizations)
        optimized_tokens = len(optimized_text.split())
        
        return {
            "prompt_id": prompt["id"],
            "original_text": original_text,
            "optimized_text": optimized_text,
            "original_tokens": analysis["token_count"],
            "optimized_tokens": optimized_tokens,
            "token_reduction": analysis["token_count"] - optimized_tokens,
            "token_savings_percentage": ((analysis["token_count"] - optimized_tokens) / analysis["token_count"] * 100) if analysis["token_count"] > 0 else 0,
            "original_quality": analysis["quality_score"],
            "expected_quality": min(100, analysis["quality_score"] + expected_quality_improvement),
            "optimizations": optimizations
        }
    
    def _apply_optimizations(self, text: str, optimizations: List[Dict]) -> str:
        """Apply optimizations to text (simplified)"""
        # This is a simplified example - real implementation would use LLM
        optimized = text
        
        # Remove redundancy
        if any(o["issue"].startswith("High redundancy") for o in optimizations):
            words = optimized.split()
            seen = set()
            filtered = []
            for word in words:
                if word.lower() not in seen or word in ["the", "a", "an", "and", "or", "but"]:
                    filtered.append(word)
                    seen.add(word.lower())
            optimized = " ".join(filtered)
        
        # Condense length
        if any("too long" in o["issue"].lower() for o in optimizations):
            # Simulate condensing
            optimized = optimized[:int(len(optimized) * 0.7)]
        
        return optimized


def collect_prompts_agent(state: PromptOptimizationState) -> PromptOptimizationState:
    """Collect prompts for analysis"""
    print("\n📝 Collecting Prompts for Analysis...")
    
    prompts = [
        {
            "id": "P1",
            "text": "Analyze the data and provide insights. Look at the data carefully and analyze it thoroughly to provide comprehensive insights about the data patterns.",
            "use_case": "Data Analysis"
        },
        {
            "id": "P2",
            "text": "You are a helpful assistant. Help me write code. Write good code that works well and is efficient and clean and follows best practices and is well documented.",
            "use_case": "Code Generation"
        },
        {
            "id": "P3",
            "text": "Summarize this article in 3 bullet points:\n\n1. Focus on main ideas\n2. Be concise\n3. Maintain accuracy\n\nArticle: [content]",
            "use_case": "Summarization"
        },
        {
            "id": "P4",
            "text": "Tell me about AI.",
            "use_case": "Information"
        },
        {
            "id": "P5",
            "text": "Given the following customer feedback, extract: 1) Overall sentiment (positive/negative/neutral), 2) Key issues mentioned, 3) Suggested improvements. Format as JSON. Example: {\"sentiment\": \"positive\", \"issues\": [...], \"improvements\": [...]}",
            "use_case": "Sentiment Analysis"
        }
    ]
    
    print(f"\n  Prompts Collected: {len(prompts)}")
    print(f"\n  Sample Prompts:")
    for prompt in prompts[:3]:
        print(f"    • {prompt['id']}: {prompt['use_case']} ({len(prompt['text'].split())} tokens)")
    
    return {
        **state,
        "prompt_analysis": {"prompts": prompts},
        "messages": [f"✓ Collected {len(prompts)} prompts"]
    }


def analyze_prompts_agent(state: PromptOptimizationState) -> PromptOptimizationState:
    """Analyze prompts"""
    print("\n🔍 Analyzing Prompts...")
    
    analyzer = PromptAnalyzer()
    prompts = state["prompt_analysis"]["prompts"]
    
    analyses = []
    opportunities = []
    
    for prompt in prompts:
        analysis = analyzer.analyze_prompt(prompt)
        analyses.append(analysis)
        
        if analysis["issues"]:
            opportunities.append({
                **analysis,
                "use_case": prompt["use_case"]
            })
    
    avg_quality = sum(a["quality_score"] for a in analyses) / len(analyses) if analyses else 0
    avg_tokens = sum(a["token_count"] for a in analyses) / len(analyses) if analyses else 0
    
    print(f"\n  Average Quality Score: {avg_quality:.1f}/100")
    print(f"  Average Token Count: {avg_tokens:.1f}")
    print(f"  Optimization Opportunities: {len(opportunities)}")
    
    print(f"\n  Issues Found:")
    issue_types = {}
    for analysis in analyses:
        for issue in analysis["issues"]:
            issue_type = issue["type"]
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    for issue_type, count in issue_types.items():
        print(f"    • {issue_type}: {count}")
    
    return {
        **state,
        "prompt_analysis": {
            **state["prompt_analysis"],
            "analyses": analyses,
            "avg_quality": avg_quality,
            "avg_tokens": avg_tokens
        },
        "optimization_opportunities": opportunities,
        "messages": [f"✓ Analyzed {len(prompts)} prompts, found {len(opportunities)} optimization opportunities"]
    }


def optimize_prompts_agent(state: PromptOptimizationState) -> PromptOptimizationState:
    """Optimize prompts"""
    print("\n💡 Optimizing Prompts...")
    
    optimizer = PromptOptimizer()
    optimized = []
    
    for opp in state["optimization_opportunities"]:
        # Find original prompt
        original_prompt = next((p for p in state["prompt_analysis"]["prompts"] if p["id"] == opp["prompt_id"]), None)
        if original_prompt:
            optimized_prompt = optimizer.optimize_prompt(original_prompt, opp)
            optimized.append(optimized_prompt)
    
    # Calculate overall metrics
    total_token_reduction = sum(o["token_reduction"] for o in optimized)
    avg_quality_improvement = sum(o["expected_quality"] - o["original_quality"] for o in optimized) / len(optimized) if optimized else 0
    
    metrics = {
        "prompts_optimized": len(optimized),
        "total_token_reduction": total_token_reduction,
        "avg_quality_improvement": avg_quality_improvement,
        "avg_token_savings": sum(o["token_savings_percentage"] for o in optimized) / len(optimized) if optimized else 0
    }
    
    print(f"\n  Prompts Optimized: {len(optimized)}")
    print(f"  Total Token Reduction: {total_token_reduction}")
    print(f"  Average Quality Improvement: {avg_quality_improvement:.1f} points")
    
    return {
        **state,
        "optimized_prompts": optimized,
        "improvement_metrics": metrics,
        "messages": [f"✓ Optimized {len(optimized)} prompts"]
    }


def generate_prompt_report_agent(state: PromptOptimizationState) -> PromptOptimizationState:
    """Generate prompt optimization report"""
    print("\n" + "="*70)
    print("PROMPT OPTIMIZATION REPORT")
    print("="*70)
    
    analysis = state["prompt_analysis"]
    print(f"\n📊 Prompt Analysis Summary:")
    print(f"  Total Prompts Analyzed: {len(analysis['prompts'])}")
    print(f"  Average Quality Score: {analysis['avg_quality']:.1f}/100")
    print(f"  Average Token Count: {analysis['avg_tokens']:.1f}")
    print(f"  Optimization Opportunities: {len(state['optimization_opportunities'])}")
    
    print(f"\n🔍 Common Issues:")
    issue_counts = {}
    for opp in state["optimization_opportunities"]:
        for issue in opp["issues"]:
            issue_type = issue["type"]
            severity = issue["severity"]
            key = f"{issue_type} ({severity})"
            issue_counts[key] = issue_counts.get(key, 0) + 1
    
    for issue_key, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"    • {issue_key}: {count} occurrences")
    
    print(f"\n💡 Optimized Prompts:")
    for i, opt in enumerate(state["optimized_prompts"], 1):
        print(f"\n  {i}. Prompt {opt['prompt_id']}:")
        print(f"      Original Tokens: {opt['original_tokens']}")
        print(f"      Optimized Tokens: {opt['optimized_tokens']}")
        print(f"      Token Reduction: {opt['token_reduction']} ({opt['token_savings_percentage']:.1f}%)")
        print(f"      Quality: {opt['original_quality']:.1f} → {opt['expected_quality']:.1f}")
        
        print(f"\n      Original:")
        print(f"      \"{opt['original_text'][:100]}...\"" if len(opt['original_text']) > 100 else f"      \"{opt['original_text']}\"")
        
        print(f"\n      Optimized:")
        print(f"      \"{opt['optimized_text'][:100]}...\"" if len(opt['optimized_text']) > 100 else f"      \"{opt['optimized_text']}\"")
        
        print(f"\n      Optimizations Applied:")
        for optimization in opt["optimizations"][:3]:
            print(f"        • {optimization['strategy']}")
    
    print(f"\n📈 Overall Improvement Metrics:")
    metrics = state["improvement_metrics"]
    print(f"  Prompts Optimized: {metrics['prompts_optimized']}")
    print(f"  Total Token Reduction: {metrics['total_token_reduction']}")
    print(f"  Average Token Savings: {metrics['avg_token_savings']:.1f}%")
    print(f"  Average Quality Improvement: {metrics['avg_quality_improvement']:.1f} points")
    
    # Calculate cost savings (example: $0.002 per 1K tokens)
    cost_per_1k_tokens = 0.002
    monthly_requests = 100000
    monthly_savings = (metrics['total_token_reduction'] / 1000) * cost_per_1k_tokens * monthly_requests
    
    print(f"\n💰 Estimated Cost Savings:")
    print(f"  Cost per 1K tokens: ${cost_per_1k_tokens}")
    print(f"  Monthly requests: {monthly_requests:,}")
    print(f"  Monthly savings: ${monthly_savings:.2f}")
    print(f"  Annual savings: ${monthly_savings * 12:.2f}")
    
    print(f"\n💡 Prompt Optimization Benefits:")
    print("  • Reduced API costs")
    print("  • Faster response times")
    print("  • Improved output quality")
    print("  • Better user experience")
    print("  • More efficient token usage")
    print("  • Clearer instructions")
    
    print("\n="*70)
    print("✅ Prompt Optimization Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_prompt_optimization_graph():
    workflow = StateGraph(PromptOptimizationState)
    workflow.add_node("collect_prompts", collect_prompts_agent)
    workflow.add_node("analyze_prompts", analyze_prompts_agent)
    workflow.add_node("optimize_prompts", optimize_prompts_agent)
    workflow.add_node("generate_report", generate_prompt_report_agent)
    workflow.add_edge(START, "collect_prompts")
    workflow.add_edge("collect_prompts", "analyze_prompts")
    workflow.add_edge("analyze_prompts", "optimize_prompts")
    workflow.add_edge("optimize_prompts", "generate_report")
    workflow.add_edge("generate_report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 277: Prompt Optimization MCP Pattern")
    print("="*70)
    
    app = create_prompt_optimization_graph()
    final_state = app.invoke({
        "messages": [],
        "prompt_analysis": {},
        "optimization_opportunities": [],
        "optimized_prompts": [],
        "improvement_metrics": {}
    })
    print("\n✅ Prompt Optimization Pattern Complete!")


if __name__ == "__main__":
    main()

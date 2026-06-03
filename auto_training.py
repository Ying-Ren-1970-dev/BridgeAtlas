"""Automatic training and evaluation system for RAG search."""
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import openai

import config


class AutoTrainer:
    """Automatic training and evaluation for RAG search."""
    
    def __init__(self):
        """Initialize the auto trainer."""
        self.api_url = "http://localhost:8000/search"
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.results = []
        
    def generate_search_scenarios(self) -> List[Dict]:
        """
        Generate 40 realistic search scenarios for structural engineering documents.
        
        Returns:
            List of search scenarios with query, expected_type, and difficulty
        """
        scenarios = [
            # Foundation & Deep Foundation (10 scenarios)
            {"query": "Foundation Plan", "category": "plan_type", "difficulty": "easy", "expected": "foundation plans"},
            {"query": "CIDH pile details", "category": "structural_element", "difficulty": "medium", "expected": "CIDH pile drawings"},
            {"query": "drilled shaft specifications", "category": "structural_element", "difficulty": "medium", "expected": "drilled shaft details"},
            {"query": "24 inch CIDH pile", "category": "specific_detail", "difficulty": "medium", "expected": "24-inch pile specifications"},
            {"query": "pile cap reinforcement", "category": "structural_element", "difficulty": "medium", "expected": "pile cap rebar details"},
            {"query": "foundation bearing capacity", "category": "technical_spec", "difficulty": "hard", "expected": "bearing capacity calculations"},
            {"query": "deep foundation design", "category": "plan_type", "difficulty": "easy", "expected": "deep foundation plans"},
            {"query": "soil bearing pressure", "category": "technical_spec", "difficulty": "hard", "expected": "soil bearing data"},
            {"query": "pile layout", "category": "plan_type", "difficulty": "easy", "expected": "pile layout plans"},
            {"query": "footing dimensions", "category": "specific_detail", "difficulty": "medium", "expected": "footing size details"},
            
            # Bridge & Superstructure (10 scenarios)
            {"query": "abutment details", "category": "structural_element", "difficulty": "medium", "expected": "abutment drawings"},
            {"query": "girder layout", "category": "plan_type", "difficulty": "easy", "expected": "girder layout plans"},
            {"query": "bridge bent details", "category": "structural_element", "difficulty": "medium", "expected": "bent drawings"},
            {"query": "prestressed girder", "category": "structural_element", "difficulty": "medium", "expected": "prestressed girder details"},
            {"query": "expansion joint details", "category": "specific_detail", "difficulty": "medium", "expected": "expansion joint drawings"},
            {"query": "deck reinforcement", "category": "structural_element", "difficulty": "medium", "expected": "deck rebar plans"},
            {"query": "bearing pad specifications", "category": "specific_detail", "difficulty": "medium", "expected": "bearing pad details"},
            {"query": "wingwall reinforcement", "category": "structural_element", "difficulty": "medium", "expected": "wingwall rebar details"},
            {"query": "diaphragm details", "category": "structural_element", "difficulty": "medium", "expected": "diaphragm drawings"},
            {"query": "soffit details", "category": "specific_detail", "difficulty": "medium", "expected": "soffit drawings"},
            
            # Retaining Walls & Earth Retention (5 scenarios)
            {"query": "retaining wall reinforcement", "category": "structural_element", "difficulty": "medium", "expected": "retaining wall rebar"},
            {"query": "shear key details", "category": "specific_detail", "difficulty": "medium", "expected": "shear key drawings"},
            {"query": "cantilever wall", "category": "structural_element", "difficulty": "medium", "expected": "cantilever wall details"},
            {"query": "drainage details behind wall", "category": "specific_detail", "difficulty": "hard", "expected": "wall drainage systems"},
            {"query": "wall footing design", "category": "structural_element", "difficulty": "medium", "expected": "wall footing details"},
            
            # Structural Details (10 scenarios)
            {"query": "pilaster details", "category": "specific_detail", "difficulty": "medium", "expected": "pilaster drawings"},
            {"query": "column reinforcement schedule", "category": "structural_element", "difficulty": "medium", "expected": "column rebar schedule"},
            {"query": "beam splice details", "category": "specific_detail", "difficulty": "medium", "expected": "beam splice drawings"},
            {"query": "rebar lap splice length", "category": "technical_spec", "difficulty": "hard", "expected": "splice length specifications"},
            {"query": "concrete cover requirements", "category": "technical_spec", "difficulty": "hard", "expected": "cover specifications"},
            {"query": "joint filler material", "category": "technical_spec", "difficulty": "hard", "expected": "joint material specs"},
            {"query": "anchor bolt layout", "category": "specific_detail", "difficulty": "medium", "expected": "anchor bolt plans"},
            {"query": "dowel bar placement", "category": "specific_detail", "difficulty": "medium", "expected": "dowel bar details"},
            {"query": "construction joint location", "category": "specific_detail", "difficulty": "hard", "expected": "construction joint plans"},
            {"query": "post-tensioning details", "category": "structural_element", "difficulty": "hard", "expected": "PT details"},
            
            # Project-Specific & General Queries (5 scenarios)
            {"query": "Mar Vista project girders", "category": "project_specific", "difficulty": "easy", "expected": "Mar Vista girder details"},
            {"query": "Sports Park structural details", "category": "project_specific", "difficulty": "easy", "expected": "Sports Park structure"},
            {"query": "general notes", "category": "plan_type", "difficulty": "easy", "expected": "general notes sheets"},
            {"query": "typical sections", "category": "plan_type", "difficulty": "easy", "expected": "typical section drawings"},
            {"query": "cross section details", "category": "plan_type", "difficulty": "easy", "expected": "cross section drawings"},
        ]
        
        return scenarios
    
    def run_search(self, query: str, k: int = 10) -> Dict:
        """
        Execute a search query against the API.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            API response dictionary
        """
        try:
            response = requests.post(
                self.api_url,
                json={"query": query, "k": k},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "results": []}
    
    def evaluate_results(self, scenario: Dict, search_results: Dict) -> Dict:
        """
        Use GPT-4 to evaluate if search results are relevant to the query.
        
        Args:
            scenario: Search scenario with query and expected results
            search_results: API response with search results
            
        Returns:
            Evaluation results with pass/fail and reasoning
        """
        results = search_results.get('results', [])
        
        if not results:
            return {
                "pass": False,
                "score": 0.0,
                "reasoning": "No results returned",
                "result_count": 0
            }
        
        # Prepare result summary for GPT-4
        result_summary = []
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            result_summary.append({
                "rank": i,
                "file": result.get('pdf_file_name', 'Unknown'),
                "page": result.get('page_number', 0),
                "relevance_score": result.get('relevance_score', 0),
                "content_preview": result.get('content_sample', '')[:200]
            })
        
        # Construct evaluation prompt
        prompt = f"""Evaluate if the search results are relevant to the query.

Query: "{scenario['query']}"
Expected Type: {scenario['expected']}
Category: {scenario['category']}
Difficulty: {scenario['difficulty']}

Top 5 Results:
{json.dumps(result_summary, indent=2)}

Evaluation Criteria:
1. Do the results contain information relevant to the query?
2. Are the file names and page numbers appropriate for the query type?
3. Does the content preview match what would be expected?
4. For specific queries (e.g., "Mar Vista"), do results come from the correct project?

Respond in JSON format:
{{
    "pass": true/false,
    "score": 0.0-1.0 (relevance score),
    "reasoning": "Brief explanation of why results pass/fail",
    "top_result_relevance": "High/Medium/Low",
    "issues": ["list", "of", "any", "issues"]
}}"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for structural engineering document search systems. Evaluate search result relevance objectively."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            evaluation = json.loads(response.choices[0].message.content)
            evaluation['result_count'] = len(results)
            return evaluation
            
        except Exception as e:
            return {
                "pass": False,
                "score": 0.0,
                "reasoning": f"Evaluation error: {str(e)}",
                "result_count": len(results)
            }
    
    def run_full_evaluation(self) -> Dict:
        """
        Run complete evaluation: generate scenarios, search, evaluate.
        
        Returns:
            Complete evaluation report
        """
        print("=" * 80)
        print("AUTOMATIC RAG TRAINING & EVALUATION")
        print("=" * 80)
        
        # Generate scenarios
        print("\n[1/4] Generating search scenarios...")
        scenarios = self.generate_search_scenarios()
        print(f"✓ Generated {len(scenarios)} search scenarios")
        
        # Run searches
        print("\n[2/4] Running searches...")
        self.results = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"  [{i}/{len(scenarios)}] Searching: {scenario['query']}")
            search_results = self.run_search(scenario['query'])
            
            result = {
                "scenario": scenario,
                "search_results": search_results,
                "timestamp": datetime.now().isoformat()
            }
            self.results.append(result)
        
        print(f"✓ Completed {len(self.results)} searches")
        
        # Evaluate results
        print("\n[3/4] Evaluating results with GPT-4...")
        evaluations = []
        for i, result in enumerate(self.results, 1):
            print(f"  [{i}/{len(self.results)}] Evaluating: {result['scenario']['query']}")
            evaluation = self.evaluate_results(
                result['scenario'],
                result['search_results']
            )
            result['evaluation'] = evaluation
            evaluations.append(evaluation)
        
        print(f"✓ Completed {len(evaluations)} evaluations")
        
        # Calculate metrics
        print("\n[4/4] Calculating metrics...")
        report = self.generate_report(self.results)
        
        return report
    
    def generate_report(self, results: List[Dict]) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Report dictionary with metrics and recommendations
        """
        total = len(results)
        passed = sum(1 for r in results if r['evaluation'].get('pass', False))
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        # Calculate average scores by category
        category_stats = {}
        difficulty_stats = {}
        
        for result in results:
            scenario = result['scenario']
            evaluation = result['evaluation']
            
            # Category stats
            category = scenario['category']
            if category not in category_stats:
                category_stats[category] = {"passed": 0, "total": 0, "scores": []}
            category_stats[category]['total'] += 1
            category_stats[category]['scores'].append(evaluation.get('score', 0))
            if evaluation.get('pass', False):
                category_stats[category]['passed'] += 1
            
            # Difficulty stats
            difficulty = scenario['difficulty']
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {"passed": 0, "total": 0, "scores": []}
            difficulty_stats[difficulty]['total'] += 1
            difficulty_stats[difficulty]['scores'].append(evaluation.get('score', 0))
            if evaluation.get('pass', False):
                difficulty_stats[difficulty]['passed'] += 1
        
        # Calculate averages
        for cat, stats in category_stats.items():
            stats['avg_score'] = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            stats['pass_rate'] = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        for diff, stats in difficulty_stats.items():
            stats['avg_score'] = sum(stats['scores']) / len(stats['scores']) if stats['scores'] else 0
            stats['pass_rate'] = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Identify failures
        failures = [
            {
                "query": r['scenario']['query'],
                "category": r['scenario']['category'],
                "difficulty": r['scenario']['difficulty'],
                "reasoning": r['evaluation'].get('reasoning', 'Unknown'),
                "issues": r['evaluation'].get('issues', []),
                "result_count": r['evaluation'].get('result_count', 0)
            }
            for r in results if not r['evaluation'].get('pass', False)
        ]
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            category_stats,
            difficulty_stats,
            failures
        )
        
        report = {
            "summary": {
                "total_scenarios": total,
                "passed": passed,
                "failed": total - passed,
                "pass_rate": round(pass_rate, 2),
                "timestamp": datetime.now().isoformat()
            },
            "category_performance": category_stats,
            "difficulty_performance": difficulty_stats,
            "failures": failures,
            "recommendations": recommendations,
            "detailed_results": results
        }
        
        return report
    
    def generate_recommendations(
        self,
        category_stats: Dict,
        difficulty_stats: Dict,
        failures: List[Dict]
    ) -> List[Dict]:
        """
        Generate recommendations for improvement based on results.
        
        Args:
            category_stats: Performance by category
            difficulty_stats: Performance by difficulty
            failures: List of failed scenarios
            
        Returns:
            List of recommendations with priority and actions
        """
        recommendations = []
        
        # Identify weak categories
        weak_categories = [
            (cat, stats) for cat, stats in category_stats.items()
            if stats['pass_rate'] < 70
        ]
        
        if weak_categories:
            for cat, stats in weak_categories:
                recommendations.append({
                    "priority": "HIGH",
                    "category": cat,
                    "issue": f"Low pass rate ({stats['pass_rate']:.1f}%) for {cat} queries",
                    "actions": [
                        f"Review enrichment data quality for {cat}",
                        f"Add more training data for {cat} scenarios",
                        f"Verify vector embeddings capture {cat} semantics",
                        "Consider adding category-specific metadata fields"
                    ]
                })
        
        # Check difficulty issues
        hard_queries = difficulty_stats.get('hard', {})
        if hard_queries.get('pass_rate', 100) < 60:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "hard_queries",
                "issue": f"Difficulty with complex queries ({hard_queries['pass_rate']:.1f}% pass rate)",
                "actions": [
                    "Improve chunk size/overlap for better context",
                    "Add technical specifications to enrichment",
                    "Consider hybrid search (keyword + semantic)",
                    "Increase k value for hard queries to improve recall"
                ]
            })
        
        # Analyze common failure patterns
        zero_result_failures = [f for f in failures if f['result_count'] == 0]
        if len(zero_result_failures) > 3:
            recommendations.append({
                "priority": "HIGH",
                "category": "zero_results",
                "issue": f"{len(zero_result_failures)} queries returned no results",
                "queries": [f['query'] for f in zero_result_failures],
                "actions": [
                    "Verify these terms exist in source documents",
                    "Check if enrichment captured these concepts",
                    "Consider adding synonyms/aliases",
                    "Review PDF text extraction quality"
                ]
            })
        
        # Project-specific failures
        project_failures = [
            f for f in failures
            if f['category'] == 'project_specific'
        ]
        if project_failures:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "project_specific",
                "issue": f"{len(project_failures)} project-specific queries failed",
                "actions": [
                    "Verify project name extraction is accurate",
                    "Ensure project metadata is properly indexed",
                    "Consider adding project aliases",
                    "Check if file names are included in searchable text"
                ]
            })
        
        # General recommendations
        overall_pass_rate = sum(
            stats['passed'] for stats in category_stats.values()
        ) / sum(
            stats['total'] for stats in category_stats.values()
        ) * 100
        
        if overall_pass_rate < 80:
            recommendations.append({
                "priority": "HIGH",
                "category": "overall",
                "issue": f"Overall pass rate below target ({overall_pass_rate:.1f}% < 80%)",
                "actions": [
                    "Run enrichment on remaining un-enriched documents",
                    "Increase chunk overlap from current settings",
                    "Add more context in enrichment prompts",
                    "Consider using larger embedding model",
                    "Implement re-ranking with cross-encoder"
                ]
            })
        
        return recommendations
    
    def save_report(self, report: Dict, output_file: str = "training_report.json"):
        """
        Save evaluation report to file.
        
        Args:
            report: Report dictionary
            output_file: Output file path
        """
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to: {output_path}")
    
    def print_report(self, report: Dict):
        """
        Print evaluation report to console.
        
        Args:
            report: Report dictionary
        """
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)
        
        summary = report['summary']
        print(f"\nOverall Performance:")
        print(f"  Total Scenarios: {summary['total_scenarios']}")
        print(f"  Passed: {summary['passed']} ✓")
        print(f"  Failed: {summary['failed']} ✗")
        print(f"  Pass Rate: {summary['pass_rate']}%")
        
        print(f"\nPerformance by Category:")
        for cat, stats in report['category_performance'].items():
            status = "✓" if stats['pass_rate'] >= 70 else "✗"
            print(f"  {status} {cat:25s}: {stats['pass_rate']:5.1f}% ({stats['passed']}/{stats['total']}) | Avg Score: {stats['avg_score']:.2f}")
        
        print(f"\nPerformance by Difficulty:")
        for diff, stats in report['difficulty_performance'].items():
            status = "✓" if stats['pass_rate'] >= 70 else "✗"
            print(f"  {status} {diff:10s}: {stats['pass_rate']:5.1f}% ({stats['passed']}/{stats['total']}) | Avg Score: {stats['avg_score']:.2f}")
        
        if report['failures']:
            print(f"\nFailed Scenarios ({len(report['failures'])}):")
            for i, failure in enumerate(report['failures'][:10], 1):  # Show first 10
                print(f"\n  [{i}] {failure['query']}")
                print(f"      Category: {failure['category']} | Difficulty: {failure['difficulty']}")
                print(f"      Results: {failure['result_count']}")
                print(f"      Reason: {failure['reasoning']}")
        
        print(f"\n" + "=" * 80)
        print(f"RECOMMENDATIONS ({len(report['recommendations'])})")
        print("=" * 80)
        
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"\n[{rec['priority']}] {rec['issue']}")
            print(f"    Actions:")
            for action in rec['actions']:
                print(f"      • {action}")


def main():
    """Main entry point."""
    trainer = AutoTrainer()
    
    print("Starting automatic training evaluation...")
    print("This will take several minutes (API calls to GPT-4).\n")
    
    # Run evaluation
    report = trainer.run_full_evaluation()
    
    # Print report
    trainer.print_report(report)
    
    # Save report
    trainer.save_report(report)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

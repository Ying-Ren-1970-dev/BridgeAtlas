"""Investigate specific failures from training report."""
import requests
import json
from typing import Dict, List

API_URL = "http://localhost:8000/search"

# Failed queries from training report
FAILED_QUERIES = [
    "Foundation Plan",
    "pile cap reinforcement",
    "foundation bearing capacity",
    "soil bearing pressure",
    "footing dimensions",
    "diaphragm details",
    "soffit details",
    "wall footing design",
    "pilaster details",
    "beam splice details",
    "rebar lap splice length",
    "concrete cover requirements",
    "joint filler material",
    "anchor bolt layout",
    "dowel bar placement",
    "construction joint location"
]


def test_query(query: str, k: int = 10) -> Dict:
    """Test a single query and return results."""
    try:
        response = requests.post(
            API_URL,
            json={"query": query, "k": k},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return {
            "query": query,
            "success": True,
            "result_count": len(data.get('results', [])),
            "results": data.get('results', [])[:3]  # Top 3 results
        }
    except Exception as e:
        return {
            "query": query,
            "success": False,
            "error": str(e),
            "result_count": 0
        }


def analyze_results(test_results: List[Dict]) -> Dict:
    """Analyze test results and identify issues."""
    analysis = {
        "total_tested": len(test_results),
        "now_working": 0,
        "still_failing": 0,
        "zero_results": [],
        "low_results": [],  # < 5 results
        "working_queries": [],
        "issues": []
    }
    
    for result in test_results:
        if not result['success']:
            analysis['still_failing'] += 1
            analysis['issues'].append({
                "query": result['query'],
                "issue": "API Error",
                "details": result.get('error', 'Unknown error')
            })
        elif result['result_count'] == 0:
            analysis['zero_results'].append(result['query'])
        elif result['result_count'] < 5:
            analysis['low_results'].append({
                "query": result['query'],
                "count": result['result_count']
            })
        else:
            analysis['now_working'] += 1
            analysis['working_queries'].append({
                "query": result['query'],
                "count": result['result_count']
            })
    
    return analysis


def main():
    """Run failure investigation."""
    print("=" * 80)
    print("INVESTIGATING FAILED QUERIES FROM TRAINING")
    print("=" * 80)
    print(f"\nTesting {len(FAILED_QUERIES)} queries that failed during training...\n")
    
    results = []
    for i, query in enumerate(FAILED_QUERIES, 1):
        print(f"[{i}/{len(FAILED_QUERIES)}] Testing: {query[:50]}...", end=" ")
        result = test_query(query)
        results.append(result)
        
        if result['success']:
            if result['result_count'] == 0:
                print(f"❌ 0 results")
            elif result['result_count'] < 5:
                print(f"⚠️  {result['result_count']} results (low)")
            else:
                print(f"✓ {result['result_count']} results")
        else:
            print(f"❌ ERROR: {result.get('error', 'Unknown')[:50]}")
    
    # Analyze
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    analysis = analyze_results(results)
    
    print(f"\nTotal Tested: {analysis['total_tested']}")
    print(f"Now Working: {analysis['now_working']} ✓")
    print(f"Still Failing: {analysis['still_failing']} ❌")
    
    if analysis['zero_results']:
        print(f"\nQueries with ZERO results ({len(analysis['zero_results'])}):")
        for query in analysis['zero_results']:
            print(f"  • {query}")
    
    if analysis['low_results']:
        print(f"\nQueries with LOW results (<5) ({len(analysis['low_results'])}):")
        for item in analysis['low_results']:
            print(f"  • {item['query']}: {item['count']} results")
    
    if analysis['working_queries']:
        print(f"\nQueries NOW WORKING ({len(analysis['working_queries'])}):")
        for item in analysis['working_queries'][:5]:
            print(f"  ✓ {item['query']}: {item['count']} results")
    
    # Root cause analysis
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    zero_count = len(analysis['zero_results'])
    low_count = len(analysis['low_results'])
    
    if zero_count > 0:
        print(f"\n🔍 {zero_count} queries return NO results:")
        print("   Possible causes:")
        print("   1. Terms not present in indexed documents")
        print("   2. Enrichment didn't capture these concepts")
        print("   3. Text extraction quality issues")
        print("   4. Chunking split relevant content")
        
    if low_count > 0:
        print(f"\n🔍 {low_count} queries return FEW results (<5):")
        print("   Possible causes:")
        print("   1. Limited coverage in document set")
        print("   2. Embedding model doesn't capture semantic similarity well")
        print("   3. Queries too specific for available content")
    
    if analysis['now_working'] > analysis['total_tested'] * 0.6:
        print(f"\n✓ Good news! {analysis['now_working']}/{analysis['total_tested']} queries working")
        print("  This suggests training failures may have been temporary API issues.")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("SPECIFIC RECOMMENDATIONS")
    print("=" * 80)
    
    if zero_count > 0:
        print(f"\n[HIGH PRIORITY] Fix zero-result queries:")
        for query in analysis['zero_results'][:3]:
            print(f"  1. Search source PDFs manually for '{query}'")
            print(f"  2. If found, check why it wasn't indexed")
            print(f"  3. If not found, consider removing from test suite")
    
    if low_count > 5:
        print(f"\n[MEDIUM PRIORITY] Improve low-result queries:")
        print("  1. Increase chunk overlap (current → +50 tokens)")
        print("  2. Add more contextual metadata to enrichment")
        print("  3. Consider hybrid search (semantic + keyword)")
    
    # Save detailed results
    output = {
        "test_results": results,
        "analysis": analysis
    }
    
    with open("failure_investigation.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("\n✓ Detailed results saved to: failure_investigation.json")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

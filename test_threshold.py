"""Test camber diagram search with relevance threshold filtering."""
import requests
import json

API_URL = "http://localhost:8000/search"

def test_with_threshold(threshold):
    """Test search with specific relevance threshold."""
    
    payload = {
        "query": "camber diagram",
        "k": 20,
        "relevance_threshold": threshold
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data['results']
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD: {threshold} - '{payload['query']}'")
        print(f"{'='*80}")
        print(f"Results returned: {len(results)}")
        
        if results:
            best_score = min(r['relevance_score'] for r in results)
            worst_score = max(r['relevance_score'] for r in results)
            print(f"Score range: {best_score:.4f} to {worst_score:.4f}")
            print(f"Score spread: {worst_score - best_score:.4f}")
            
            print("\nPages returned:")
            for i, result in enumerate(results, 1):
                print(f"  [{i}] Page {result['page_number']}: {result['relevance_score']:.4f}")
                print(f"      Content: {result['content_sample'][:80]}...")
        else:
            print("No results returned")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("="*80)
    print("TESTING RELEVANCE THRESHOLD FILTERING")
    print("="*80)
    
    # Test with different thresholds
    thresholds = [0.0, 0.10, 0.15, 0.20, 0.30, 1.0]
    
    for threshold in thresholds:
        test_with_threshold(threshold)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print("threshold=0.15 (default) should return only the most relevant pages")
    print("threshold=0.0 returns only the single best match")
    print("threshold=1.0 returns all matches")

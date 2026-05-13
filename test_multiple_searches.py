"""Test multiple searches to verify scoring works correctly."""
import requests
import json

API_URL = "http://localhost:8000/search"

def test_search(query):
    """Test a search query and show results."""
    
    payload = {"query": query, "k": 20}
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data['results']
        
        print("\n" + "="*80)
        print(f"SEARCH: '{query}'")
        print("="*80)
        print(f"Total results: {len(results)}")
        
        if results:
            scores = [r['relevance_score'] for r in results]
            print(f"Score range: {min(scores):.4f} to {max(scores):.4f}")
            
            # Check for duplicates
            unique_scores = len(set(scores))
            if unique_scores == 1 and len(results) > 1:
                print("⚠️  WARNING: All scores identical!")
            else:
                print(f"✓ {unique_scores} unique scores across {len(results)} pages")
            
            # Show top 3 results
            print("\nTop 3 pages:")
            for i, result in enumerate(results[:3], 1):
                print(f"  [{i}] Page {result['page_number']}: {result['relevance_score']:.4f}")
                print(f"      Topology: {', '.join(result['topology_elements'][:3])}")
                print(f"      Content: {result['content_sample'][:80]}...")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Test various searches
    searches = [
        "camber diagram",
        "pipe pin",
        "CIDH pile",
        "bearing pad",
        "abutment details",
        "steel shear key"
    ]
    
    for query in searches:
        test_search(query)
    
    print("\n" + "="*80)
    print("✓ All searches complete")
    print("="*80)

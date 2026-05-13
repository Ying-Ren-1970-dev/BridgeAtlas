"""Test different searches with default threshold."""
import requests
import json

API_URL = "http://localhost:8000/search"

def test_search(query):
    """Test a search query with default threshold."""
    
    payload = {
        "query": query,
        "k": 20
        # Using default relevance_threshold=0.15
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        results = data['results']
        
        print(f"\n{'='*80}")
        print(f"SEARCH: '{query}' (threshold=0.15 default)")
        print(f"{'='*80}")
        print(f"Results: {len(results)} pages")
        
        if results:
            best_score = min(r['relevance_score'] for r in results)
            worst_score = max(r['relevance_score'] for r in results)
            print(f"Score range: {best_score:.4f} to {worst_score:.4f}")
            
            print("\nTop pages:")
            for i, result in enumerate(results[:5], 1):
                print(f"  [{i}] Page {result['page_number']}: {result['relevance_score']:.4f}")
                topology = ', '.join(result['topology_elements'][:3]) if result['topology_elements'] else 'None'
                print(f"      Topology: {topology}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("="*80)
    print("TESTING DEFAULT THRESHOLD (0.15) WITH VARIOUS SEARCHES")
    print("="*80)
    
    searches = [
        "camber diagram",
        "pipe pin",
        "CIDH pile foundation",
        "bearing pad abutment",
        "steel shear key",
        "girder spacing"
    ]
    
    for query in searches:
        test_search(query)
    
    print("\n" + "="*80)
    print("✓ All searches complete with threshold filtering")
    print("="*80)

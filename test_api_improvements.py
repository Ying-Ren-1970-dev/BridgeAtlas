"""
Test API with improved query expansion.
"""
import requests
import json

API_URL = "http://localhost:8000/search"

def test_search(query, k=50):
    """Test a search query."""
    print(f"\n{'='*80}")
    print(f"Testing query: '{query}' (k={k})")
    print('='*80)
    
    response = requests.post(
        API_URL,
        json={"query": query, "k": k}
    )
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Results returned: {len(results)}")
        
        if results:
            # Show unique pages
            pages = sorted(set(r['page_number'] for r in results))
            print(f"✓ Unique pages: {pages}")
            print(f"✓ Total pages: {len(pages)}")
            
            # Show top 3 results
            print(f"\nTop 3 results:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. Page {result['page_number']}, Score: {result['relevance_score']:.3f}")
                print(f"     Content: {result['content_sample'][:100]}...")
        else:
            print("⚠ No results returned")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")

# Test queries that previously failed
test_queries = [
    ("drilled shaft", "Should match CIDH pile via synonym expansion"),
    ("cast-in-drilled-hole pile", "Another CIDH synonym"),
    ("steel shear key", "Should match pipe pin"),
    ("rebar", "Should match reinforcing bars"),
    ("bent", "Should differentiate from abutment"),
]

print("\n" + "="*80)
print("TESTING API WITH QUERY EXPANSION")
print("="*80)

for query, description in test_queries:
    print(f"\n💡 {description}")
    test_search(query)

print("\n" + "="*80)
print("✓ API testing complete!")
print("="*80)

"""Test camber diagram search via API to check relevance scores."""
import requests
import json

API_URL = "http://localhost:8000/search"

def test_camber_search():
    """Search for camber diagram and display results with scores."""
    
    payload = {
        "query": "camber diagram",
        "k": 20
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        print("="*80)
        print("CAMBER DIAGRAM SEARCH - API RESULTS")
        print("="*80)
        print(f"\nTotal results: {len(data['results'])}")
        print(f"\nSearch Summary:\n{data['search_summary']}\n")
        
        print("-"*80)
        print("INDIVIDUAL PAGE RESULTS:")
        print("-"*80)
        
        for i, result in enumerate(data['results'], 1):
            print(f"\n[{i}] Page {result['page_number']}")
            print(f"    Relevance Score: {result['relevance_score']:.4f}")
            print(f"    Topology: {', '.join(result['topology_elements']) if result['topology_elements'] else 'None'}")
            print(f"    Has Vision: {result['has_vision_analysis']}")
            print(f"    Content Preview: {result['content_sample'][:150]}...")
        
        # Check if all scores are the same
        scores = [r['relevance_score'] for r in data['results']]
        if len(set(scores)) == 1:
            print("\n" + "!"*80)
            print("⚠️  WARNING: All pages have identical relevance scores!")
            print(f"    Score: {scores[0]:.4f}")
            print("!"*80)
        else:
            print(f"\n✓ Score range: {min(scores):.4f} to {max(scores):.4f}")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API server at http://localhost:8000")
        print("   Make sure the API server is running: python api.py")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_camber_search()

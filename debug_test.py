import requests

# Test the exact scenario from test_scenarios.json
query = "R/C Box Girder"
expected_pages = [1]

response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": query,
        "k": 20,
        "relevance_threshold": 0.15
    },
    timeout=30
)

print(f"Status code: {response.status_code}")
print(f"\nResponse keys: {response.json().keys()}")

data = response.json()
print(f"\nTotal results: {data.get('total_results', 0)}")
print(f"Number of result items: {len(data.get('results', []))}")

if data.get('results'):
    print(f"\nFirst result keys: {data['results'][0].keys()}")
    print(f"\nAll pages returned:")
    for r in data['results']:
        print(f"  - Page {r['page_number']}: score {r['relevance_score']}")
    
    actual_pages = set(r['page_number'] for r in data['results'])
    print(f"\nActual pages set: {actual_pages}")
    print(f"Expected pages: {set(expected_pages)}")
    print(f"Match: {set(expected_pages).issubset(actual_pages)}")

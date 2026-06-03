"""
Test the Librarian API with sample requests.
"""
import requests
import json
from time import sleep

# Wait for server to fully initialize
sleep(2)

API_URL = "http://localhost:8000"

print("="*80)
print("TESTING LIBRARIAN API")
print("="*80)

# Test 1: Health Check
print("\n1. Health Check")
print("-" * 40)
try:
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"❌ Error: {e}")

# Test 2: Search for "steel shear key"
print("\n\n2. Search: 'steel shear key'")
print("-" * 40)
try:
    response = requests.post(
        f"{API_URL}/search",
        json={"query": "steel shear key", "k": 10}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Query: {data['query']}")
        print(f"✓ Total Results: {data['total_results']}")
        print(f"✓ Projects Found: {data['projects_found']}")
        print(f"\n📊 Top 5 Results:")
        
        for idx, result in enumerate(data['results'][:5], 1):
            print(f"\n  {idx}. {result['project_name']}")
            print(f"     File: {result['pdf_file_name']}")
            print(f"     Page: {result['page_number']}")
            print(f"     Score: {result['relevance_score']}")
            print(f"     Topology: {', '.join(result['topology_elements'][:3])}")
            print(f"     Vision: {'✓' if result['has_vision_analysis'] else '✗'}")
        
        print(f"\n📋 Project Summaries:")
        for proj in data['project_summaries']:
            print(f"\n  • {proj['project_name']}")
            print(f"    Pages: {proj['relevant_pages'][:10]}")
            print(f"    Categories: {', '.join(proj['categories'])}")
        
        print(f"\n💡 AI Summary:")
        print(f"  {data['search_summary'][:300]}...")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 3: List all projects
print("\n\n3. List All Projects")
print("-" * 40)
try:
    response = requests.get(f"{API_URL}/projects")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Total Projects: {data['total_projects']}")
        
        for idx, proj in enumerate(data['projects'][:5], 1):
            print(f"\n  {idx}. {proj['project_name']}")
            print(f"     File: {proj['pdf_file_name']}")
            print(f"     Pages: {proj['total_pages']}")
            print(f"     Categories: {', '.join(proj['categories'])}")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 4: Search for "pipe pin"
print("\n\n4. Search: 'pipe pin'")
print("-" * 40)
try:
    response = requests.post(
        f"{API_URL}/search",
        json={"query": "pipe pin", "k": 5}
    )
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Total Results: {data['total_results']}")
        print(f"✓ Projects: {data['projects_found']}")
        
        for idx, result in enumerate(data['results'], 1):
            print(f"\n  {idx}. Page {result['page_number']} - Score: {result['relevance_score']}")
            topology = [t for t in result['topology_elements'] if 'pipe pin' in t.lower()]
            if topology:
                print(f"     ✓ Found: {', '.join(topology)}")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

# Test 5: Get categories
print("\n\n5. List Categories")
print("-" * 40)
try:
    response = requests.get(f"{API_URL}/categories")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Total Categories: {data['total_categories']}")
        
        for category, info in list(data['categories'].items())[:5]:
            print(f"\n  • {category}: {info['project_count']} projects")
    else:
        print(json.dumps(response.json(), indent=2))
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("✓ API TESTING COMPLETE")
print("="*80)
print("\n📖 View interactive docs at: http://localhost:8000/docs")
print("📋 View OpenAPI schema at: http://localhost:8000/openapi.json")
print("\n")

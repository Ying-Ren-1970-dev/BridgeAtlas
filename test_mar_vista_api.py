"""
Test the Mar Vista filtered API search.
"""
import requests
import json
from time import sleep

# Wait for server to be ready
sleep(1)

API_URL = "http://localhost:8000"

print("="*80)
print("TESTING MAR VISTA FILTERED API SEARCH")
print("="*80)

# Test searches on Mar Vista project
test_queries = [
    "steel shear key",
    "pipe pin",
    "CIDH pile",
    "bearing pad",
    "abutment details"
]

for query in test_queries:
    print(f"\n{'='*80}")
    print(f"SEARCH: '{query}'")
    print("="*80)
    
    try:
        response = requests.post(
            f"{API_URL}/search",
            json={"query": query, "k": 10}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"\n✓ Query: {data['query']}")
            print(f"✓ Total Results: {data['total_results']}")
            print(f"✓ Projects Found: {data['projects_found']}")
            
            if data['total_results'] > 0:
                print(f"\n📄 Results from Mar Vista Project:")
                print(f"   File: {data['results'][0]['pdf_file_name']}")
                
                # Extract unique page numbers
                pages = sorted(set([r['page_number'] for r in data['results']]))
                print(f"\n📍 Page Numbers with '{query}':")
                print(f"   {pages}")
                
                print(f"\n🔍 Top 5 Results:")
                for idx, result in enumerate(data['results'][:5], 1):
                    print(f"\n   {idx}. Page {result['page_number']} (Score: {result['relevance_score']:.3f})")
                    
                    # Show topology if present
                    if result['topology_elements']:
                        print(f"      Topology: {', '.join(result['topology_elements'][:3])}")
                    
                    # Show vision indicator
                    vision = "✓ Vision" if result['has_vision_analysis'] else "✗ No Vision"
                    print(f"      Analysis: {vision}")
                    
                    # Show brief content sample
                    sample = result['content_sample'][:150].replace('\n', ' ')
                    print(f"      Content: {sample}...")
                
                # Show project summary
                if data['project_summaries']:
                    summary = data['project_summaries'][0]
                    print(f"\n📋 Project Summary:")
                    print(f"   Project: {summary['project_name']}")
                    print(f"   Total Pages: {summary['total_pages']}")
                    print(f"   Relevant Pages: {summary['relevant_pages']}")
                    print(f"   Categories: {', '.join(summary['categories'])}")
            else:
                print("\n⚠ No results found for this query")
                
        else:
            print(f"❌ Error: Status {response.status_code}")
            print(json.dumps(response.json(), indent=2))
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Summary test - search for multiple terms
print(f"\n\n{'='*80}")
print("COMPREHENSIVE SEARCH: 'pipe pin steel shear key'")
print("="*80)

try:
    response = requests.post(
        f"{API_URL}/search",
        json={"query": "pipe pin steel shear key", "k": 20}
    )
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"\n✓ Total Results: {data['total_results']}")
        
        # Get all unique pages
        all_pages = sorted(set([r['page_number'] for r in data['results']]))
        
        print(f"\n📍 ALL PAGES with Pipe Pin or Steel Shear Key content:")
        print(f"   Pages: {all_pages}")
        print(f"   Total: {len(all_pages)} pages")
        
        # Group by topology
        topology_pages = {}
        for result in data['results']:
            for topo in result['topology_elements']:
                if 'pipe pin' in topo.lower() or 'shear key' in topo.lower():
                    if topo not in topology_pages:
                        topology_pages[topo] = set()
                    topology_pages[topo].add(result['page_number'])
        
        print(f"\n🏗️ Topology Distribution:")
        for topo, pages in topology_pages.items():
            print(f"   • {topo}: pages {sorted(pages)}")
        
        print(f"\n💡 AI Summary:")
        print(f"   {data['search_summary'][:400]}...")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*80)
print("✓ MAR VISTA FILTERED SEARCH TEST COMPLETE")
print("="*80)
print("\n📖 API Documentation: http://localhost:8000/docs")
print("🔍 The API now only searches the Mar Vista POC project")
print("📄 Page numbers are returned for all search results\n")

"""Test if enriched metadata is searchable in vector store."""
import sys
from vector_store import VectorStore

# Initialize vector store
vs = VectorStore()
vs.initialize_vectorstore()

# Search for enriched metadata terms
test_queries = [
    "Sports Park",
    "pilaster details",
    "GIRDER LAYOUT",
    "Sheet Title",
    "ENRICHED METADATA"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    results = vs.similarity_search(query, k=3)
    
    if results:
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            metadata = result.get('metadata', {})
            print(f"File: {metadata.get('file_name', 'N/A')}")
            print(f"Page: {metadata.get('page', 'N/A')}")
            
            # Show enriched metadata fields if present
            enriched_fields = {k: v for k, v in metadata.items() if k.startswith('enriched_') or k.startswith('plan_') or k in ['page_type', 'structural_elements', 'detail_types']}
            if enriched_fields:
                print(f"\n✓ Enriched metadata fields found:")
                for key, value in enriched_fields.items():
                    print(f"  - {key}: {value}")
            
            print(f"\nContent preview (first 300 chars):")
            content = result.get('content', '')
            print(content[:300])
            
            # Check if enriched metadata marker exists
            if '[ENRICHED METADATA]' in content:
                print("\n✓ Contains enriched metadata!")
                # Show the enriched part
                enriched_start = content.find('[ENRICHED METADATA]')
                enriched_section = content[enriched_start:enriched_start+500]
                print(f"\nEnriched section:\n{enriched_section}")
    else:
        print("No results found")

print("\n" + "="*60)
print("Test complete")

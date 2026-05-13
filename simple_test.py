"""Simple check for Sports Park in vector store."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

print("Searching for 'Sports Park'...")
results = vs.similarity_search("Sports Park", k=5)

print(f"\nFound {len(results)} results\n")

if results:
    for i, result in enumerate(results[:3], 1):
        metadata = result.get('metadata', {})
        content = result.get('content', '')
        print(f"Result {i}:")
        print(f"  File: {metadata.get('file_name')}")
        print(f"  Page: {metadata.get('page')}")
        print(f"  Has enriched marker: {'[ENRICHED METADATA]' in content}")
        print(f"  Page type: {metadata.get('page_type', 'N/A')}")
        print(f"  Plan primary type: {metadata.get('plan_primary_type', 'N/A')}")
        print()

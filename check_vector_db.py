from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

print("Testing vector store search...\n")

# Test a few queries
queries = ["CIDH pile", "retaining wall", "R/C Box Girder", "bent spacing"]

for query in queries:
    results = vs.vectorstore.similarity_search_with_score(query, k=3)
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results")
    
    for i, (doc, score) in enumerate(results[:2], 1):
        print(f"\n  Result {i} (score: {score:.3f}):")
        print(f"  - Page: {doc.metadata.get('page', 'unknown')}")
        print(f"  - Project: {doc.metadata.get('project_name', 'unknown')}")
        print(f"  - Content preview: {doc.page_content[:150]}...")

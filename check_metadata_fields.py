from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

# Search for Mar Vista specifically
results = vs.vectorstore.similarity_search_with_score("CIDH pile", k=3)

print("Checking metadata fields for Mar Vista results:\n")
for i, (doc, score) in enumerate(results, 1):
    if "Mar Vista" in doc.metadata.get('project_name', ''):
        print(f"Result {i}:")
        print(f"  Metadata keys: {doc.metadata.keys()}")
        print(f"  file_name: '{doc.metadata.get('file_name', 'NOT FOUND')}'")
        print(f"  project_name: '{doc.metadata.get('project_name', 'NOT FOUND')}'")
        print(f"  page: {doc.metadata.get('page')}")
        print()
        break

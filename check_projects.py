from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

# Get all documents to see what projects are stored
collection = vs.client.get_or_create_collection(name="librarian_documents")
results = collection.get(limit=10, include=["metadatas"])

print(f"Total documents in collection: {collection.count()}\n")
print("Sample project names in vector DB:")
print("="*60)

seen_projects = set()
for metadata in results['metadatas']:
    project = metadata.get('project_name', 'NO PROJECT NAME')
    if project not in seen_projects:
        print(f"- '{project}'")
        seen_projects.add(project)
        
print(f"\n Total unique projects found: {len(seen_projects)}")

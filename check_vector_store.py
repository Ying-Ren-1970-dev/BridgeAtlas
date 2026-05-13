"""Check what's actually in the vector store."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

# Get collection stats
collection = vs.client.get_collection("librarian_documents")
count = collection.count()

print(f"Total documents in vector store: {count}")

# Sample a few documents
if count > 0:
    results = collection.get(limit=10, include=['metadatas'])
    print(f"\nSample of {len(results['ids'])} documents:")
    for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
        print(f"\n{i+1}. ID: {doc_id[:50]}...")
        print(f"   File: {metadata.get('file_name', 'N/A')}")
        print(f"   Page: {metadata.get('page', 'N/A')}")
        print(f"   Page type: {metadata.get('page_type', 'N/A')}")
else:
    print("Vector store is empty!")

import chromadb
from pathlib import Path

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection("librarian_docs")

# Query for Mar Vista documents
results = collection.get(
    where={"file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"},
    include=["metadatas", "documents"]
)

print(f"Found {len(results['ids'])} chunks for Mar Vista POC")
print("\nChecking for CIDH mentions:")

for i, (doc_id, metadata, document) in enumerate(zip(results['ids'], results['metadatas'], results['documents'])):
    if 'CIDH' in document.upper():
        print(f"\n--- Chunk {i+1} ---")
        print(f"Page: {metadata.get('page', 'Unknown')}")
        print(f"Content snippet: {document[:200]}...")

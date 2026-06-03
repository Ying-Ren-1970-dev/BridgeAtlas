"""Investigate duplicate chunks in the vector database."""
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="librarian_documents")

# Get all documents for Mar Vista
results = collection.get(
    where={"file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"},
    include=["documents", "metadatas"]
)

print(f"Total chunks for Mar Vista: {len(results['documents'])}\n")

# Find chunks with "camber diagram" or similar content
camber_chunks = []
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
    if 'camber' in doc.lower():
        camber_chunks.append({
            'index': i,
            'page': meta.get('page_number', 'Unknown'),
            'content': doc[:200],
            'full_content': doc
        })

print(f"Chunks mentioning 'camber': {len(camber_chunks)}\n")
print("="*80)

# Group by content to find duplicates
from collections import defaultdict
content_groups = defaultdict(list)
for chunk in camber_chunks:
    # Use first 150 chars as key to group similar chunks
    key = chunk['content'][:150]
    content_groups[key].append(chunk['page'])

print("\nDUPLICATE CONTENT ANALYSIS:")
print("="*80)

for i, (content_key, pages) in enumerate(content_groups.items(), 1):
    if len(pages) > 1:
        print(f"\n[{i}] DUPLICATE found on {len(pages)} pages: {sorted(set(pages))}")
        print(f"Content preview: {content_key}...")
        print(f"Pages: {pages}")

# Show unique content chunks
print("\n\nUNIQUE CAMBER CHUNKS:")
print("="*80)
for content_key, pages in content_groups.items():
    if len(pages) == 1:
        print(f"\nPage {pages[0]}:")
        print(f"  {content_key}...")

import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection("librarian_documents")

# Get all Mar Vista documents
results = collection.get(
    where={"file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"},
    include=["metadatas", "documents"]
)

print(f"Total chunks for Mar Vista: {len(results['ids'])}")

# Find ALL chunks with CIDH and collect unique page numbers
cidh_pages = set()
for metadata, document in zip(results['metadatas'], results['documents']):
    if 'CIDH' in document.upper():
        cidh_pages.add(metadata.get('page'))

cidh_pages = sorted(list(cidh_pages))

print(f"\nFound CIDH content on {len(cidh_pages)} page(s)")
print(f"PDF Page numbers: {cidh_pages}")

# Show snippet from each page
print(f"\n{'='*80}")
print("Content preview from each CIDH page:")
print('='*80)
for page_num in cidh_pages:
    # Get one chunk from this page
    for metadata, document in zip(results['metadatas'], results['documents']):
        if metadata.get('page') == page_num and 'CIDH' in document.upper():
            print(f"\nPDF Page {page_num}:")
            print(f"  {document[:150]}...")
            break

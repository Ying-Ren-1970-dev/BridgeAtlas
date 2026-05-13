"""
Check why page 16 (CIDH detail sheet) isn't appearing in search results.
"""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

# Test different queries
queries = [
    "24 inch CIDH pile",
    "24\" CIDH pile detail",
    "CIDH pile detail sheet",
    "cast-in-drilled-hole pile 24 inch"
]

print("="*80)
print("INVESTIGATING PAGE 16 CIDH DETAIL SHEET")
print("="*80)

for query in queries:
    print(f"\nQuery: '{query}'")
    results = vs.vectorstore.similarity_search_with_score(
        query, 
        k=20,
        filter={'file_name': 'Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf'}
    )
    
    pages_found = {}
    for doc, score in results:
        page = doc.metadata.get('page')
        if page not in pages_found:
            pages_found[page] = score
    
    print(f"  Pages found: {sorted(pages_found.keys())}")
    
    # Check if page 16 is in results
    if 16 in pages_found:
        print(f"  ✓ Page 16 found! Score: {pages_found[16]:.3f}")
    else:
        print(f"  ✗ Page 16 NOT in top 20 results")

# Now check what content is actually on page 16
print("\n" + "="*80)
print("CHECKING PAGE 16 CONTENT IN VECTOR DB")
print("="*80)

all_docs = vs.vectorstore.get()
page_16_chunks = [
    (i, doc) for i, doc in enumerate(all_docs['documents']) 
    if all_docs['metadatas'][i].get('page') == 16 
    and all_docs['metadatas'][i].get('file_name') == 'Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf'
]

print(f"\nTotal chunks from page 16: {len(page_16_chunks)}")
if page_16_chunks:
    print("\nSample content from page 16:")
    for i, (idx, content) in enumerate(page_16_chunks[:2]):
        print(f"\nChunk {i+1}:")
        print(content[:300])
else:
    print("⚠ WARNING: No chunks found from page 16 in vector DB!")

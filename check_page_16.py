"""Check page 16 for CIDH content."""
from vector_store import VectorStore

# Initialize
vector_store = VectorStore()
vector_store.initialize_vectorstore()

filename = "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
page_num = 16

print("="*80)
print(f"CHECKING PAGE {page_num} FOR CIDH CONTENT")
print("="*80)

# Search for chunks from page 16
results = vector_store.similarity_search_with_scores(
    query="CIDH pile foundation structural",
    k=100
)

# Filter for page 16
page_chunks = []
for doc, score in results:
    if (doc.metadata.get('file_name') == filename and 
        doc.metadata.get('page') == page_num):
        page_chunks.append((doc, score))

if page_chunks:
    print(f"\n✓ Found {len(page_chunks)} chunk(s) indexed from page {page_num}")
    print(f"Relevance scores: {[round(score, 3) for _, score in page_chunks]}")
    print(f"\n--- CONTENT FROM PAGE {page_num} ---\n")
    
    for idx, (doc, score) in enumerate(page_chunks, 1):
        print(f"[Chunk {idx}, Relevance Score: {score:.3f}]")
        
        content = doc.page_content
        
        # Check type of content
        if "[DRAWING ANALYSIS]" in content:
            print("🎯 VISION-ANALYZED CAD DRAWING")
        elif "[TEXT CONTENT]" in content:
            print("📝 TEXT + VISION ANALYSIS")
        else:
            print("📄 TEXT ONLY")
        
        print("\n" + content)
        print("\n" + "-"*80 + "\n")
else:
    print(f"\n✗ No chunks found indexed for page {page_num}")

# Also do a direct CIDH search to see if page 16 appears
print("\n" + "="*80)
print("DIRECT 'CIDH' SEARCH - CHECKING IF PAGE 16 APPEARS")
print("="*80)

cidh_results = vector_store.similarity_search_with_scores(
    query="CIDH",
    k=50
)

page_16_in_results = False
for doc, score in cidh_results:
    if (doc.metadata.get('file_name') == filename and 
        doc.metadata.get('page') == page_num):
        page_16_in_results = True
        print(f"\n✓ Page {page_num} appears in top 50 CIDH search results")
        print(f"   Relevance score: {score:.3f}")
        print(f"   Content preview: {doc.page_content[:200]}...")
        break

if not page_16_in_results:
    print(f"\n⚠ Page {page_num} NOT in top 50 CIDH search results")
    print(f"   This means other pages have stronger CIDH relevance")

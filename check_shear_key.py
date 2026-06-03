"""Check Mar Vista page 18 for shear key details."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

filename = "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
page_num = 18

print("="*80)
print(f"MAR VISTA PROJECT - PAGE {page_num} (SHEAR KEY DETAILS)")
print("="*80)

# Search for chunks from page 18
results = vs.similarity_search_with_scores(
    query="shear key details structural",
    k=50
)

# Filter for page 18
page_chunks = []
for doc, score in results:
    if (doc.metadata.get('file_name') == filename and 
        doc.metadata.get('page') == page_num):
        page_chunks.append((doc, score))

if page_chunks:
    print(f"\n✓ Found {len(page_chunks)} chunk(s) from page {page_num}")
    print("\n" + "="*80)
    print("FULL CONTENT")
    print("="*80)
    
    for idx, (doc, score) in enumerate(page_chunks, 1):
        content = doc.page_content
        
        if "[DRAWING ANALYSIS]" in content:
            print(f"\n🎯 VISION ANALYSIS (Relevance: {score:.3f})")
        elif "[TEXT CONTENT]" in content:
            print(f"\n📝 TEXT + VISION (Relevance: {score:.3f})")
        else:
            print(f"\n📄 TEXT ONLY (Relevance: {score:.3f})")
        
        print("\n" + content)
        print("\n" + "-"*80)
else:
    print(f"\n⚠ No chunks found for page {page_num}")

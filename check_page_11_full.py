"""Check all content from Mar Vista page 11."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

filename = "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
page_num = 11

print("="*80)
print(f"CHECKING ALL CONTENT FROM MAR VISTA PAGE {page_num}")
print("="*80)

# Get all chunks - broad search
results = vs.similarity_search_with_scores(
    query="structural engineering bridge foundation",
    k=200
)

# Filter for page 11
page_11_chunks = []
for doc, score in results:
    if (doc.metadata.get('file_name') == filename and 
        doc.metadata.get('page') == page_num):
        page_11_chunks.append((doc, score))

if page_11_chunks:
    print(f"\n✓ Page {page_num} IS indexed - found {len(page_11_chunks)} chunk(s)")
    print("\n" + "="*80)
    print("CONTENT FROM PAGE 11:")
    print("="*80)
    
    for idx, (doc, score) in enumerate(page_11_chunks, 1):
        content = doc.page_content
        
        if "[DRAWING ANALYSIS]" in content:
            print(f"\n🎯 VISION ANALYSIS")
        elif "[TEXT CONTENT]" in content:
            print(f"\n📝 TEXT + VISION")
        else:
            print(f"\n📄 TEXT ONLY")
        
        print("\n" + content)
        print("\n" + "-"*80)
else:
    print(f"\n⚠ Page {page_num} NOT indexed or no content extracted")
    print("\nThis could mean:")
    print("- Page had no extractable text")
    print("- Vision analysis failed or skipped")
    print("- Page was outside MAX_PAGES_PER_PDF limit")
    
    # Check which Mar Vista pages ARE indexed
    print("\n" + "="*80)
    print("CHECKING WHICH PAGES ARE INDEXED")
    print("="*80)
    
    all_results = vs.similarity_search_with_scores(
        query="bridge structural",
        k=200
    )
    
    mar_vista_pages = set()
    for doc, score in all_results:
        if doc.metadata.get('file_name') == filename:
            page = doc.metadata.get('page')
            if page:
                mar_vista_pages.add(page)
    
    print(f"\nMar Vista pages indexed: {sorted(mar_vista_pages)}")
    print(f"Total: {len(mar_vista_pages)} pages")
    
    if page_num not in mar_vista_pages:
        print(f"\n⚠ Page {page_num} is MISSING from index")

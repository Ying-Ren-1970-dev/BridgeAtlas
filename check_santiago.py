"""Check Santiago Rd pages 1-5 for retaining wall content."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

filename = "Santiago Rd OC RW 100% Signed.pdf"
pages = [1, 2, 3, 4, 5, 10, 11]

print("="*80)
print("SANTIAGO RD PROJECT - PAGE CONTENT VERIFICATION")
print("="*80)
print(f"\nFile: {filename}")
print("Checking pages 1-5 (retaining wall plans) and 10-11 (boring logs)\n")

for page_num in pages:
    print(f"\n{'='*80}")
    print(f"PAGE {page_num}")
    print(f"{'='*80}")
    
    # Search for chunks from this specific page
    results = vs.similarity_search_with_scores(
        query="retaining wall structural foundation",
        k=100
    )
    
    # Filter for this page
    page_chunks = []
    for doc, score in results:
        if (doc.metadata.get('file_name') == filename and 
            doc.metadata.get('page') == page_num):
            page_chunks.append((doc, score))
    
    if page_chunks:
        print(f"✓ Found {len(page_chunks)} chunk(s) indexed")
        for idx, (doc, score) in enumerate(page_chunks, 1):
            content = doc.page_content
            
            # Check type
            if "[DRAWING ANALYSIS]" in content:
                print(f"\n[Chunk {idx}] 🎯 VISION ANALYSIS")
            elif "[TEXT CONTENT]" in content:
                print(f"\n[Chunk {idx}] 📝 TEXT + VISION")
            else:
                print(f"\n[Chunk {idx}] 📄 TEXT ONLY")
            
            print(f"Relevance: {score:.3f}")
            print(f"\nContent preview (first 500 chars):")
            print(content[:500] + "..." if len(content) > 500 else content)
    else:
        print("⚠ NO CONTENT INDEXED for this page")
        print("   This page may have been skipped or failed to extract")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)
print("\nIf pages 1-5 show 'NO CONTENT', the retaining wall plans weren't indexed.")
print("This would explain why only boring log pages (10-11) appeared in search.")

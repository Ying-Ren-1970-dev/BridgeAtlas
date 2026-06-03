"""Query vector store to show indexed content from specific pages."""
from vector_store import VectorStore
from metadata_manager import MetadataManager

# Initialize
vector_store = VectorStore()
vector_store.initialize_vectorstore()
metadata_manager = MetadataManager()

filename = "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
pages_to_check = [8, 9, 10, 21]

print("="*80)
print("CIDH PILE INFORMATION - INDEXED CONTENT VERIFICATION")
print("="*80)
print(f"\nFile: {filename}")
print(f"Pages to verify: {pages_to_check}")

for page_num in pages_to_check:
    print(f"\n{'='*80}")
    print(f"PAGE {page_num}")
    print(f"{'='*80}")
    
    # Search for chunks from this specific page
    # We'll search with a broad query and filter by metadata
    results = vector_store.similarity_search_with_scores(
        query="CIDH pile foundation structural",
        k=100  # Get many results to find all from this page
    )
    
    # Filter for this page
    page_chunks = []
    for doc, score in results:
        if (doc.metadata.get('file_name') == filename and 
            doc.metadata.get('page') == page_num):
            page_chunks.append((doc, score))
    
    if page_chunks:
        print(f"✓ Found {len(page_chunks)} chunk(s) indexed from this page")
        print(f"\n--- CONTENT FROM PAGE {page_num} ---\n")
        
        # Show all chunks from this page
        for idx, (doc, score) in enumerate(page_chunks, 1):
            print(f"[Chunk {idx}, Relevance: {score:.3f}]")
            content = doc.page_content
            
            # Highlight if it's vision analysis
            if "[DRAWING ANALYSIS]" in content:
                print("🎯 VISION-ANALYZED CAD DRAWING")
            elif "[TEXT CONTENT]" in content:
                print("📝 TEXT + VISION ANALYSIS")
            else:
                print("📄 TEXT ONLY")
            
            print("\n" + content)
            print("\n" + "-"*40 + "\n")
    else:
        print("✗ No chunks found indexed for this page")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

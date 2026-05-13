"""Check what was extracted from Mar Vista page 11 with pipe pins."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

filename = "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
page_num = 11

print("="*80)
print(f"MAR VISTA PAGE {page_num} - PIPE PIN (STEEL SHEAR KEY) CONTENT")
print("="*80)

# Search for pipe pin content
results = vs.similarity_search_with_scores(
    query="pipe pin",
    k=100
)

# Filter for page 11
page_11_chunks = []
for doc, score in results:
    if (doc.metadata.get('file_name') == filename and 
        doc.metadata.get('page') == page_num):
        page_11_chunks.append((doc, score))

if page_11_chunks:
    print(f"\n✓ Found {len(page_11_chunks)} chunk(s) from page {page_num}")
    print("\n" + "="*80)
    print("FULL CONTENT FROM PAGE 11:")
    print("="*80)
    
    for idx, (doc, score) in enumerate(page_11_chunks, 1):
        content = doc.page_content
        
        if "[DRAWING ANALYSIS]" in content:
            print(f"\n🎯 VISION ANALYSIS (Relevance: {score:.3f})")
        elif "[TEXT CONTENT]" in content:
            print(f"\n📝 TEXT + VISION (Relevance: {score:.3f})")
        else:
            print(f"\n📄 TEXT ONLY (Relevance: {score:.3f})")
        
        print("\n" + content)
        print("\n" + "-"*80)
        
    # Check if "steel shear key" is mentioned
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    combined_text = " ".join([doc.page_content.lower() for doc, _ in page_11_chunks])
    
    has_pipe_pin = "pipe pin" in combined_text
    has_shear_key = "shear key" in combined_text
    has_steel = "steel" in combined_text
    
    print(f"\nContent includes:")
    print(f"  - 'pipe pin': {has_pipe_pin}")
    print(f"  - 'shear key': {has_shear_key}")
    print(f"  - 'steel': {has_steel}")
    
    if has_pipe_pin and not (has_shear_key or "steel shear key" in combined_text):
        print("\n⚠ ISSUE: Vision extracted 'pipe pin' but didn't classify it as 'steel shear key'")
        print("   The enhanced prompt should classify pipe pins as steel shear keys")
else:
    print(f"\n⚠ No chunks found for page {page_num}")

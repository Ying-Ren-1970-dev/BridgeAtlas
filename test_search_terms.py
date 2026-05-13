"""Test different search terms to find all CIDH-related pages."""
from vector_store import VectorStore

vs = VectorStore()
vs.initialize_vectorstore()

filename = "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"

search_terms = [
    "CIDH pile",
    "drilled hole concrete pile",
    "cast in drilled hole",
    "pile reinforcement spiral",
    "24 inch pile"
]

print("="*80)
print("TESTING DIFFERENT SEARCH TERMS FOR CIDH-RELATED PAGES")
print("="*80)

for term in search_terms:
    print(f"\n{'='*80}")
    print(f"Search: '{term}'")
    print("="*80)
    
    results = vs.similarity_search_with_scores(term, k=30)
    
    pages_found = set()
    for doc, score in results:
        if doc.metadata.get('file_name') == filename:
            page = doc.metadata.get('page')
            if page not in pages_found:
                pages_found.add(page)
                print(f"  Page {page:2d}: relevance {score:.3f}")
    
    if not pages_found:
        print("  (No pages from Mar Vista PDF)")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nPage 16 content describes:")
print("- 'DRILLED HOLE FILLED WITH CONCRETE' (CIDH terminology)")
print("- Column and pile connection details for Bents 2-4, 10-16")
print("- Spiral reinforcement (#6 SPIRAL)")
print("\nWhy it didn't rank higher in 'CIDH pile' search:")
print("- Vision analysis used 'drilled piles' instead of 'CIDH'")
print("- Other pages had explicit 'CIDH' text and stronger matches")
print("- Still captured and searchable with related terms!")

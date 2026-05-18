import chromadb

client = chromadb.PersistentClient(path='./vector_db')
col = client.get_collection('librarian_documents')

# Check all unique file names first
res0 = col.get(include=['metadatas'], limit=1000)
files = {}
for m in res0['metadatas']:
    fn = m.get('file_name', 'NONE')
    if fn not in files:
        files[fn] = []
    files[fn].append(m.get('page_number'))
print("=== All files in librarian_documents ===")
for fn, pages in sorted(files.items()):
    print(f"  {fn[:70]}")
print()

for exact_fname, pages_of_interest in [
    ('Laguna Creek Plan.pdf', [9, 14, 15]),
    ('Austin Rd OH - Plans - Final Submittal_.pdf', [11, 20, 21, 23]),
]:
    res = col.get(where={'file_name': {'$eq': exact_fname}}, include=['metadatas', 'documents'])
    metas = res['metadatas']
    docs = res['documents']
    print(f"=== {exact_fname} ===")
    for m, d in zip(metas, docs):
        pg = m.get('page')
        if pg in pages_of_interest:
            print(f"  p{pg} sheet_type={m.get('plan_sheet_type')} structural_elements={m.get('structural_elements')}")
            print(f"       detail_intents={m.get('detail_intents')}")
            print(f"       sheet_title={m.get('plan_sheet_title')}")
            # Show first 200 chars of document text
            print(f"       text[:250]={d[:250]}")
            print()
    print()

# Check Mar Vista pages NOT in results (19, 20, 23, 26, 27) vs ones that ARE (15, 28, 30)
res2 = col.get(where={'file_name': {'$eq': 'Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf'}}, include=['metadatas', 'documents'])
metas2 = res2['metadatas']
docs2 = res2['documents']
print("=== Mar Vista - key pages ===")
for m, d in zip(metas2, docs2):
    pg = m.get('page')
    if pg in [15, 19, 20, 23, 26, 27, 28, 30]:
        print(f"  p{pg} sheet_type={m.get('plan_sheet_type')} sheet_title={m.get('plan_sheet_title')}")
        print(f"       structural_elements={m.get('structural_elements')}")
        print(f"       detail_intents={m.get('detail_intents')}")
        print()

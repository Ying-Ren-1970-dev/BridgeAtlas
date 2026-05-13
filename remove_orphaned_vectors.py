import chromadb
from pathlib import Path

# Initialize ChromaDB client
vector_db_path = "vector_db"
client = chromadb.PersistentClient(path=vector_db_path)

# Get the collection
try:
    collection = client.get_collection("librarian_documents")
    
    # Check how many documents exist
    total_count = collection.count()
    print(f"=== Vector Database Status ===")
    print(f"Total vectors: {total_count}\n")
    
    # Get all documents with IDC YR.pdf
    orphaned_file = "IDC YR.pdf"
    print(f"Searching for vectors from '{orphaned_file}'...")
    
    # Query for IDC YR vectors
    results = collection.get(
        where={"file_name": orphaned_file},
        include=['metadatas']
    )
    
    if results and results['ids']:
        orphan_count = len(results['ids'])
        print(f"Found {orphan_count} orphaned vectors\n")
        print(f"Deleting vectors from '{orphaned_file}'...")
        
        # Delete by IDs
        collection.delete(ids=results['ids'])
        
        # Verify deletion
        new_count = collection.count()
        deleted = total_count - new_count
        print(f"✓ Deleted {deleted} vectors")
        print(f"✓ Remaining vectors: {new_count}")
    else:
        print(f"✗ No vectors found for '{orphaned_file}'")
        print("Vector database is already clean.")
    
    # Show remaining files
    print("\n=== Verifying Remaining Files ===")
    all_results = collection.get(limit=10000, include=['metadatas'])
    if all_results and all_results.get('metadatas'):
        file_names = set()
        for metadata in all_results['metadatas']:
            if 'file_name' in metadata:
                file_names.add(metadata['file_name'])
        
        print(f"Files in vector database: {len(file_names)}")
        for filename in sorted(file_names):
            file_path = Path("Projects") / filename
            exists = "✓" if file_path.exists() else "✗ MISSING"
            print(f"  {exists} {filename}")
    
except Exception as e:
    print(f"Error: {e}")

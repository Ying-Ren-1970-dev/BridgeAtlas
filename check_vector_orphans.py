from vector_store import VectorStore
from pathlib import Path
import json

# Initialize vector store
vs = VectorStore()

# Get all unique file names in vector database
try:
    # Get a sample of documents to see structure
    all_docs = vs.db.get(limit=10000, include=['metadatas'])
    
    if all_docs and all_docs.get('metadatas'):
        # Collect all unique file names
        file_names = set()
        for metadata in all_docs['metadatas']:
            if 'file_name' in metadata:
                file_names.add(metadata['file_name'])
        
        print(f"=== Vector Database Contents ===")
        print(f"Total vectors: {len(all_docs['metadatas'])}")
        print(f"Unique files: {len(file_names)}\n")
        
        # Check which files exist
        projects_dir = Path("Projects")
        orphaned_files = []
        
        for filename in sorted(file_names):
            file_path = projects_dir / filename
            exists = "✓" if file_path.exists() else "✗ ORPHANED"
            print(f"  {exists} {filename}")
            if not file_path.exists():
                orphaned_files.append(filename)
        
        if orphaned_files:
            print(f"\n⚠ Found {len(orphaned_files)} orphaned file(s) in vector database:")
            for filename in orphaned_files:
                # Count vectors for this file
                file_docs = [m for m in all_docs['metadatas'] if m.get('file_name') == filename]
                print(f"  - {filename}: {len(file_docs)} vectors")
            
            print("\nThese vectors should be removed. Use main.py to rebuild or manually delete.")
        else:
            print("\n✓ All files in vector database exist in Projects folder")
    else:
        print("No documents found in vector database")
        
except Exception as e:
    print(f"Error checking vector database: {e}")

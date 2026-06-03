#!/usr/bin/env python3
"""
Remove a specific document from the Librarian database.
Usage: python remove_document.py "document_name.pdf"
"""

import sys
from pathlib import Path
from vector_store import VectorStore
from metadata_manager import MetadataManager

def remove_document(file_name: str):
    """Remove a document from the database."""
    print(f"\n{'='*60}")
    print(f"REMOVING DOCUMENT: {file_name}")
    print('='*60)
    
    # Initialize components
    vector_store = VectorStore()
    metadata_manager = MetadataManager()
    
    # Initialize vector store to connect to it
    vector_store.initialize_vectorstore()
    
    try:
        # Step 1: Delete from vector store
        print(f"\n1. Deleting from vector store...")
        vector_store.delete_by_filename(file_name)
        
        # Step 2: Delete from metadata
        print(f"2. Deleting from metadata...")
        metadata_manager.delete_project(file_name)
        metadata_manager.save()
        
        print(f"\n✓ Successfully removed '{file_name}' from the database")
        print('='*60)
        
    except Exception as e:
        print(f"\n✗ Error removing document: {str(e)}")
        print('='*60)
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_document.py <document_filename>")
        print("\nExample:")
        print('  python remove_document.py "Alta Mesa Type Selection Report.pdf"')
        sys.exit(1)
    
    file_name = ' '.join(sys.argv[1:])
    remove_document(file_name)

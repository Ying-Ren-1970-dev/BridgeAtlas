"""Vector store module for embedding and semantic search."""
from typing import List, Dict, Optional
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

import config
from enriched_metadata_loader import EnrichedMetadataLoader


class VectorStore:
    """Manages vector embeddings and semantic search using ChromaDB."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        self.vector_db_path = str(config.VECTOR_DB_PATH)
        self.collection_name = "librarian_documents"
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.vector_db_path)
        
        self.vectorstore = None
    
    def initialize_vectorstore(self):
        """Initialize or load existing vectorstore."""
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.vector_db_path,
            )
            print(f"Vector store initialized at {self.vector_db_path}")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, processed_data: List[Dict]) -> int:
        """
        Add processed PDF data to the vector store with enriched metadata.
        
        Args:
            processed_data: List of processed PDF data from PDFProcessor
            
        Returns:
            Number of chunks added
        """
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        texts = []
        metadatas = []
        enriched_count = 0
        
        for pdf_data in processed_data:
            file_name = pdf_data['file_name']
            file_path = pdf_data['file_path']
            pdf_metadata = pdf_data['metadata']
            
            # Load enriched metadata for this PDF
            enriched_metadata = EnrichedMetadataLoader.load_enriched_metadata(file_name)
            if enriched_metadata:
                enriched_count += 1
                print(f"  ✓ Loaded enriched metadata for {file_name} ({len(enriched_metadata)} pages)")
            
            for chunk in pdf_data['chunks']:
                chunk_text = chunk['text']
                page_num = chunk['page']
                
                # Create base metadata for chunk
                chunk_metadata = {
                    'file_name': file_name,
                    'file_path': file_path,
                    'page': page_num,
                    'chunk_id': chunk['chunk_id'],
                    'project_name': pdf_metadata.get('project_name', ''),
                    'phase': pdf_metadata.get('phase', ''),
                    'engineer_of_record': pdf_metadata.get('engineer_of_record', ''),
                    'date': pdf_metadata.get('date', ''),
                    'total_pages': pdf_data['total_pages'],
                }
                
                # Merge enriched metadata if available for this page
                if enriched_metadata and page_num in enriched_metadata:
                    page_enriched = enriched_metadata[page_num]
                    
                    # Get enriched metadata fields
                    enriched_fields = EnrichedMetadataLoader.get_metadata_fields(page_enriched)
                    chunk_metadata.update(enriched_fields)
                    
                    # Append searchable text from enriched metadata
                    enriched_text = EnrichedMetadataLoader.extract_searchable_text(page_enriched)
                    if enriched_text:
                        chunk_text = f"{chunk_text}\n\n[ENRICHED METADATA]\n{enriched_text}"
                
                texts.append(chunk_text)
                metadatas.append(chunk_metadata)
        
        # Add to vector store in batches
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            try:
                self.vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_texts)
                print(f"Added batch {i//batch_size + 1}: {len(batch_texts)} chunks")
            except Exception as e:
                print(f"Error adding batch {i//batch_size + 1}: {str(e)}")
        
        print(f"\nTotal chunks added to vector store: {total_added}")
        if enriched_count > 0:
            print(f"✓ {enriched_count} files enriched with metadata")
        return total_added
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of search results with content and metadata
        """
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            # Perform similarity search
            if filter_dict:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
            
            # Format results
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def similarity_search_with_scores(
        self, 
        query: str, 
        k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of tuples (document, score)
        """
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            if filter_dict:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter_dict
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            return results
            
        except Exception as e:
            print(f"Error during similarity search: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'path': self.vector_db_path,
            }
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            return {}
    
    def delete_by_filename(self, file_name: str):
        """Delete all documents for a specific file."""
        try:
            if self.vectorstore:
                # Delete documents with matching file_name metadata
                self.vectorstore.delete(where={"file_name": file_name})
                print(f"Deleted documents for: {file_name}")
        except Exception as e:
            print(f"Error deleting documents for {file_name}: {str(e)}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' cleared")
            self.vectorstore = None
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
    
    def enrich_existing_documents(self, file_name: str) -> int:
        """
        Add enrichment metadata to existing documents without re-processing PDFs.
        Preserves existing Vision analysis while adding enrichment data.
        
        Args:
            file_name: Name of the PDF file to enrich
            
        Returns:
            Number of documents updated
        """
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            # Load enriched metadata from JSON
            enriched_metadata = EnrichedMetadataLoader.load_enriched_metadata(file_name)
            if not enriched_metadata:
                print(f"  No enrichment data found for {file_name}")
                return 0
            
            print(f"  ✓ Loaded enrichment data ({len(enriched_metadata)} pages)")
            
            # Get all existing documents for this file
            collection = self.vectorstore._collection
            results = collection.get(
                where={"file_name": file_name},
                include=["metadatas", "documents"]
            )
            
            if not results['ids']:
                print(f"  No existing documents found for {file_name}")
                return 0
            
            print(f"  Found {len(results['ids'])} existing chunks")
            
            # Prepare updated data
            updated_texts = []
            updated_metadatas = []
            ids_to_update = []
            enriched_count = 0
            
            for i, (doc_id, doc_text, doc_metadata) in enumerate(
                zip(results['ids'], results['documents'], results['metadatas'])
            ):
                page_num = doc_metadata.get('page')
                
                # Check if text already has enrichment (skip if already enriched)
                if "[ENRICHED METADATA]" in doc_text:
                    continue
                
                # Create updated metadata dict
                updated_metadata = doc_metadata.copy()
                updated_text = doc_text
                
                # Merge enrichment data if available for this page
                if page_num and page_num in enriched_metadata:
                    page_enriched = enriched_metadata[page_num]
                    
                    # Add enriched metadata fields
                    enriched_fields = EnrichedMetadataLoader.get_metadata_fields(page_enriched)
                    updated_metadata.update(enriched_fields)
                    
                    # Append searchable text from enriched metadata
                    enriched_text = EnrichedMetadataLoader.extract_searchable_text(page_enriched)
                    if enriched_text:
                        updated_text = f"{doc_text}\n\n[ENRICHED METADATA]\n{enriched_text}"
                        enriched_count += 1
                
                updated_texts.append(updated_text)
                updated_metadatas.append(updated_metadata)
                ids_to_update.append(doc_id)
            
            if not ids_to_update:
                print(f"  All chunks already enriched or no matching pages")
                return 0
            
            # Delete old documents
            collection.delete(ids=ids_to_update)
            print(f"  Deleted {len(ids_to_update)} old chunks")
            
            # Re-add with enriched data (in batches)
            batch_size = 100
            total_added = 0
            
            for i in range(0, len(updated_texts), batch_size):
                batch_texts = updated_texts[i:i+batch_size]
                batch_metadatas = updated_metadatas[i:i+batch_size]
                
                self.vectorstore.add_texts(
                    texts=batch_texts,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_texts)
            
            print(f"  ✓ Updated {total_added} chunks with enrichment ({enriched_count} chunks enriched)")
            return total_added
            
        except Exception as e:
            print(f"  Error enriching documents for {file_name}: {str(e)}")
            raise

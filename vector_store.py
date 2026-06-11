"""Vector store module for embedding and semantic search."""
from typing import List, Dict, Optional
import json
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import config
from enriched_metadata_loader import EnrichedMetadataLoader
from enhanced_topology_loader import EnhancedTopologyLoader
from title_block_catalog import TitleBlockCatalog


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
        
        self.vectorstore = None
    
    def initialize_vectorstore(self):
        """Initialize or load existing vectorstore."""
        try:
            import os
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Ensure we're in embedded mode, not client-server
            os.environ.pop('CHROMA_HOST', None)
            os.environ.pop('CHROMA_PORT', None) 
            os.environ.pop('CHROMA_SERVER_HOST', None)
            
            # Create a PersistentClient with explicit local/embedded settings
            chroma_client = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                    chroma_server_host=None,  # Disable server mode
                    chroma_server_http_port=None,  # Disable HTTP
                )
            )
            
            # Use langchain-chroma with explicit client
            self.vectorstore = Chroma(
                client=chroma_client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
            print(f"Vector store initialized at {self.vector_db_path}")
        except Exception as e:
            print(f"Error initializing vector store: {str(e)}")
            import traceback
            traceback.print_exc()
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

            sheet_catalog = TitleBlockCatalog.build_for_project(file_name)
            
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
                elif sheet_catalog and page_num in sheet_catalog:
                    sheet_record = sheet_catalog[page_num]
                    chunk_metadata.update(TitleBlockCatalog.get_metadata_fields(sheet_record))
                    category_text = TitleBlockCatalog.extract_searchable_text(sheet_record)
                    if category_text:
                        chunk_text = f"{chunk_text}\n\n[SHEET CATEGORY]\n{category_text}"
                
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
            # Check collection count first
            try:
                collection = self.vectorstore._client.get_collection(self.collection_name)
                count = collection.count()
                print(f"Collection has {count} documents")
            except Exception as ce:
                print(f"Error checking collection: {str(ce)}")
            
            # Perform similarity search
            if filter_dict:
                try:
                    results = self.vectorstore.similarity_search(
                        query=query,
                        k=k,
                        filter=filter_dict
                    )
                except Exception as filtered_err:
                    # Chroma can intermittently fail filtered queries with "Error finding id".
                    # Fallback to unfiltered retrieval and apply metadata filter in-memory.
                    print(f"Filtered similarity_search failed, falling back to in-memory filter: {filtered_err}")
                    fallback_k = max(k * 10, 200)
                    raw_results = self.vectorstore.similarity_search(
                        query=query,
                        k=fallback_k
                    )
                    results = [
                        doc for doc in raw_results
                        if self._metadata_matches_filter(doc.metadata or {}, filter_dict)
                    ][:k]
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
            import traceback
            import sys
            error_msg = f"=== SIMILARITY SEARCH ERROR ===\nException type: {type(e).__name__}\nException message: {str(e)}\n"
            print(error_msg, file=sys.stderr)
            print(f"Full traceback:\n{traceback.format_exc()}", file=sys.stderr)
            print(error_msg)  # Also to stdout
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
                try:
                    results = self.vectorstore.similarity_search_with_score(
                        query=query,
                        k=k,
                        filter=filter_dict
                    )
                except Exception as filtered_err:
                    # Chroma can intermittently fail filtered queries with "Error finding id".
                    # Fallback to unfiltered retrieval and apply metadata filter in-memory.
                    print(f"Filtered similarity_search_with_scores failed, falling back to in-memory filter: {filtered_err}")
                    fallback_k = max(k * 10, 200)
                    raw_results = self.vectorstore.similarity_search_with_score(
                        query=query,
                        k=fallback_k
                    )
                    results = [
                        (doc, score) for (doc, score) in raw_results
                        if self._metadata_matches_filter(doc.metadata or {}, filter_dict)
                    ][:k]
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            return results
            
        except Exception as e:
            import traceback
            import sys
            error_msg = f"\n=== SIMILARITY SEARCH WITH SCORES ERROR ===\nException type: {type(e).__name__}\nException message: {str(e)}\n"
            print(error_msg, file=sys.stderr)
            print(f"Full traceback:\n{traceback.format_exc()}", file=sys.stderr)
            print(error_msg)  # Also to stdout
            print(f"ChromaDB exception details: {repr(e)}")
            return []

    def _metadata_matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Apply a small subset of Chroma where filtering semantics in Python."""
        for key, expected in (filter_dict or {}).items():
            actual = metadata.get(key)

            if isinstance(expected, dict):
                if "$in" in expected:
                    allowed = expected.get("$in") or []
                    if actual not in allowed:
                        return False
                elif "$eq" in expected:
                    if actual != expected.get("$eq"):
                        return False
                else:
                    # Unsupported operator for fallback path.
                    return False
            else:
                if actual != expected:
                    return False

        return True

    def get_keyword_search_candidates(
        self,
        filter_dict: Optional[Dict] = None,
        limit: int = 4000,
    ) -> List[Dict]:
        """
        Retrieve raw chunk documents for local keyword ranking.

        Args:
            filter_dict: Optional metadata filter to pre-limit candidates
            limit: Maximum number of chunks to fetch

        Returns:
            List of dictionaries with id, content, and metadata
        """
        if not self.vectorstore:
            self.initialize_vectorstore()

        try:
            collection = self.vectorstore._collection
            get_kwargs = {
                "include": ["documents", "metadatas"],
                "limit": limit,
            }

            if filter_dict:
                get_kwargs["where"] = filter_dict

            results = collection.get(**get_kwargs)

            ids = results.get("ids", [])
            docs = results.get("documents", [])
            metas = results.get("metadatas", [])

            candidates = []
            for doc_id, doc_text, doc_meta in zip(ids, docs, metas):
                if not doc_text:
                    continue
                candidates.append(
                    {
                        "id": doc_id,
                        "content": doc_text,
                        "metadata": doc_meta or {},
                    }
                )

            return candidates
        except Exception as e:
            print(f"Error getting keyword candidates: {str(e)}")
            return []

    def get_page_chunk(self, file_name: str, page_number: int) -> Optional[Dict]:
        """Fetch a single indexed chunk for an exact file/page pair."""
        if not self.vectorstore:
            self.initialize_vectorstore()

        try:
            collection = self.vectorstore._collection
            results = collection.get(
                where={
                    "$and": [
                        {"file_name": file_name},
                        {"page": int(page_number)},
                    ]
                },
                include=["documents", "metadatas"],
                limit=1,
            )
            ids = results.get("ids") or []
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            if not ids or not docs:
                return None

            return {
                "id": ids[0],
                "content": docs[0],
                "metadata": metas[0] or {},
            }
        except Exception as e:
            print(f"Error getting page chunk for {file_name} p{page_number}: {str(e)}")
            return None
    
    def get_collection_info(self) -> Dict:
        """Get information about the current collection."""
        if not self.vectorstore:
            self.initialize_vectorstore()
        
        try:
            collection = self.vectorstore._client.get_collection(self.collection_name)
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

    def delete_by_file_names(self, file_names: List[str]) -> int:
        """Delete documents for multiple files by their file_name metadata."""
        deleted = 0
        if not self.vectorstore or not file_names:
            return deleted

        for file_name in file_names:
            try:
                self.delete_by_filename(file_name)
                deleted += 1
            except Exception:
                continue
        return deleted
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            if self.vectorstore:
                self.vectorstore._client.delete_collection(self.collection_name)
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

    def merge_deep_vision_topology(self, file_name: str) -> int:
        """
        Merge deep-vision enhanced topology into existing vector chunks.

        Appends a [DEEP VISION TOPOLOGY] block and topology metadata fields
        without removing existing PDF text or enriched metadata.
        """
        if not self.vectorstore:
            self.initialize_vectorstore()

        try:
            topology_data = EnhancedTopologyLoader.load_topology(
                file_name, require_deep_vision=True
            )
            if not topology_data:
                print(f"  No deep vision topology found for {file_name}")
                return 0

            page_map = EnhancedTopologyLoader.get_page_data(topology_data)
            if not page_map:
                print(f"  Enhanced topology has no page data for {file_name}")
                return 0

            print(
                f"  Loaded deep vision topology "
                f"({topology_data.get('pages_analyzed', len(page_map))} pages)"
            )

            collection = self.vectorstore._collection
            results = collection.get(
                where={"file_name": file_name},
                include=["metadatas", "documents"],
            )

            if not results["ids"]:
                print(f"  No existing documents found for {file_name}")
                return 0

            print(f"  Found {len(results['ids'])} existing chunks")

            updated_texts = []
            updated_metadatas = []
            ids_to_update = []
            merged_count = 0

            sheet_catalog = TitleBlockCatalog.build_for_project(file_name)

            for doc_id, doc_text, doc_metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                if (
                    doc_metadata.get("layered_topology_version")
                    == EnhancedTopologyLoader.LAYERED_TOPOLOGY_VERSION
                ):
                    continue

                page_num = doc_metadata.get("page")
                if not page_num or int(page_num) not in page_map:
                    continue

                page_topology = page_map[int(page_num)]
                sheet_record = sheet_catalog.get(int(page_num))
                topology_text = EnhancedTopologyLoader.extract_layered_topology_text(
                    topology_data,
                    int(page_num),
                    page_topology,
                    sheet_record=sheet_record,
                )
                if not topology_text:
                    continue

                updated_metadata = doc_metadata.copy()
                topology_fields = EnhancedTopologyLoader.get_metadata_fields(
                    page_topology,
                    topology_data=topology_data,
                    page_num=int(page_num),
                    sheet_record=sheet_record,
                )
                updated_metadata.update(topology_fields)
                base_text = EnhancedTopologyLoader.strip_topology_blocks(doc_text)
                updated_text = f"{base_text}\n\n{topology_text}"

                updated_texts.append(updated_text)
                updated_metadatas.append(updated_metadata)
                ids_to_update.append(doc_id)
                merged_count += 1

            if not ids_to_update:
                print("  All chunks already merged or no matching pages")
                return 0

            batch_size = 100
            total_updated = 0
            for i in range(0, len(ids_to_update), batch_size):
                batch_ids = ids_to_update[i : i + batch_size]
                batch_texts = updated_texts[i : i + batch_size]
                batch_metadatas = updated_metadatas[i : i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                collection.update(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
                total_updated += len(batch_ids)

            print(
                f"  Updated {total_updated} chunks in place with deep vision topology "
                f"({merged_count} chunks with topology text)"
            )
            return total_updated

        except Exception as e:
            print(f"  Error merging deep vision topology for {file_name}: {str(e)}")
            raise

    def merge_project_profile(self, file_name: str, profile: Dict) -> bool:
        """
        Add or replace a single project-overview chunk for cross-project search.
        """
        if not self.vectorstore:
            self.initialize_vectorstore()

        from project_profile_builder import ProjectProfileBuilder

        try:
            collection = self.vectorstore._collection
            existing = collection.get(
                where={
                    "$and": [
                        {"file_name": file_name},
                        {"index_level": "project_overview"},
                    ]
                },
                include=["metadatas"],
            )
            if existing.get("ids"):
                collection.delete(ids=existing["ids"])

            structured = profile.get("structured") or {}
            project_meta = profile.get("project_name", "")
            profile_text = ProjectProfileBuilder.profile_to_search_text(profile)

            metadata = {
                "file_name": file_name,
                "project_name": project_meta,
                "page": 0,
                "chunk_id": 0,
                "index_level": "project_overview",
                "profile_version": str(profile.get("version", 1)),
                "span_count": structured.get("span_count"),
                "bridge_type": structured.get("bridge_type") or "",
                "structure_type": structured.get("structure_type") or "",
                "concrete_box_girder": str(bool(structured.get("concrete_box_girder"))).lower(),
                "plan_sheet_type": "project_overview",
                "plan_primary_type": "Project Overview",
                "topology_source": "project_profile",
            }

            self.vectorstore.add_texts(texts=[profile_text], metadatas=[metadata])
            return True
        except Exception as e:
            print(f"  Error merging project profile for {file_name}: {str(e)}")
            return False

    def merge_all_deep_vision_topology(self) -> int:
        """Merge deep-vision topology for every enhanced_topology_*.json file."""
        topology_files = EnhancedTopologyLoader.list_deep_vision_topology_files()
        if not topology_files:
            print("No deep vision topology files found in data/")
            return 0

        total_updated = 0
        for topology_file in topology_files:
            pdf_stem = topology_file.stem.replace("enhanced_topology_", "", 1)
            file_name = f"{pdf_stem}.pdf"
            print(f"\nMerging deep vision topology for {file_name}...")
            total_updated += self.merge_deep_vision_topology(file_name)

        print(f"\nTotal chunks updated across all projects: {total_updated}")
        return total_updated

    def merge_sheet_categories(self, file_name: str) -> int:
        """
        Merge per-sheet title block categories into existing vector chunks.

        Uses enriched title blocks when available and deep-vision topology for
        all remaining pages so every sheet has a category.
        """
        if not self.vectorstore:
            self.initialize_vectorstore()

        try:
            catalog = TitleBlockCatalog.build_for_project(file_name)
            if not catalog:
                print(f"  No title block catalog available for {file_name}")
                return 0

            print(f"  Loaded sheet category catalog ({len(catalog)} pages)")

            collection = self.vectorstore._collection
            results = collection.get(
                where={"file_name": file_name},
                include=["metadatas", "documents"],
            )

            if not results["ids"]:
                print(f"  No existing documents found for {file_name}")
                return 0

            updated_texts = []
            updated_metadatas = []
            ids_to_update = []
            merged_count = 0

            for doc_id, doc_text, doc_metadata in zip(
                results["ids"], results["documents"], results["metadatas"]
            ):
                page_num = doc_metadata.get("page")
                if not page_num or int(page_num) not in catalog:
                    continue

                record = catalog[int(page_num)]
                needs_pdf_notes = bool(record.get("pdf_text_excerpt")) and "[GENERAL NOTES TEXT]" not in doc_text
                if doc_metadata.get("sheet_category_merged") == "true" and not needs_pdf_notes:
                    continue

                category_text = TitleBlockCatalog.extract_searchable_text(record)
                if not category_text:
                    continue

                updated_metadata = doc_metadata.copy()
                updated_metadata.update(TitleBlockCatalog.get_metadata_fields(record))
                if record.get("pdf_text_excerpt"):
                    updated_metadata["general_notes_text_merged"] = "true"
                updated_text = doc_text
                has_enriched_block = "[ENRICHED METADATA]" in doc_text
                if "[SHEET CATEGORY]" not in doc_text and not has_enriched_block:
                    updated_text = f"{doc_text}\n\n[SHEET CATEGORY]\n{category_text}"
                elif needs_pdf_notes and "[SHEET CATEGORY]" in doc_text:
                    updated_text = f"{doc_text}\n\n[SHEET CATEGORY]\n{category_text}"

                if needs_pdf_notes and record.get("pdf_text_excerpt"):
                    updated_text = (
                        f"{updated_text}\n\n[GENERAL NOTES TEXT]\n{record['pdf_text_excerpt']}"
                    )

                updated_texts.append(updated_text)
                updated_metadatas.append(updated_metadata)
                ids_to_update.append(doc_id)
                if updated_text != doc_text or needs_pdf_notes:
                    merged_count += 1

            if not ids_to_update:
                print("  All chunks already have sheet categories")
                return 0

            batch_size = 100
            total_updated = 0
            for i in range(0, len(ids_to_update), batch_size):
                batch_ids = ids_to_update[i : i + batch_size]
                batch_texts = updated_texts[i : i + batch_size]
                batch_metadatas = updated_metadatas[i : i + batch_size]
                batch_embeddings = self.embeddings.embed_documents(batch_texts)
                collection.update(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                )
                total_updated += len(batch_ids)

            print(
                f"  Updated {total_updated} chunks with sheet categories "
                f"({merged_count} categorized chunks)"
            )
            return total_updated

        except Exception as e:
            print(f"  Error merging sheet categories for {file_name}: {str(e)}")
            raise

    def merge_all_sheet_categories(self) -> int:
        """Merge sheet categories for every deep-vision topology project."""
        topology_files = EnhancedTopologyLoader.list_deep_vision_topology_files()
        if not topology_files:
            print("No deep vision topology files found in data/")
            return 0

        total_updated = 0
        for topology_file in topology_files:
            pdf_stem = topology_file.stem.replace("enhanced_topology_", "", 1)
            file_name = f"{pdf_stem}.pdf"
            print(f"\nMerging sheet categories for {file_name}...")
            total_updated += self.merge_sheet_categories(file_name)

        print(f"\nTotal chunks updated across all projects: {total_updated}")
        return total_updated

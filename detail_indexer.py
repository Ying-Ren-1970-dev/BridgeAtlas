"""
Detail-level indexing for structural engineering documents.

Extracts individual details from graph nodes and creates separate embeddings
for improved search coverage and cross-project detail discovery.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings

import config
from classification_training import SearchClassificationTrainer


@dataclass
class DetailNode:
    """Represents a structural detail from a graph."""
    detail_id: str              # e.g., "Page2_Detail0"
    page_number: int
    title: str                  # e.g., "SECTION A-A"
    element_type: str           # "section", "elevation", "plan", "detail"
    text_content: List[str]     # Full text content of detail
    file_name: str
    project_name: str
    source: str = "graph"       # Always "graph" for details


class DetailIndexer:
    """Manages detail-level indexing from graph structures."""
    
    def __init__(self):
        """Initialize the detail indexer."""
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        self.vector_db_path = str(config.VECTOR_DB_PATH)
        self.detail_collection_name = "librarian_details"
        self.graphs_dir = os.path.join(os.path.dirname(__file__), "data", "graphs")
        self.classification_trainer = SearchClassificationTrainer()
        
        self.chroma_client = None
        self.detail_collection = None
    
    def initialize(self):
        """Initialize ChromaDB client and detail collection."""
        try:
            # Create ChromaDB persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=self.vector_db_path,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True,
                    chroma_server_host=None,
                    chroma_server_http_port=None,
                )
            )
            
            # Get or create detail collection
            self.detail_collection = self.chroma_client.get_or_create_collection(
                name=self.detail_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            print(f"Detail indexer initialized at {self.vector_db_path}")
            return True
        except Exception as e:
            print(f"Error initializing detail indexer: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _ensure_collection_ready(self) -> bool:
        """Ensure detail collection handle is valid, reinitializing if needed."""
        if self.detail_collection is None:
            return self.initialize()

        try:
            # Lightweight call to validate the collection handle.
            self.detail_collection.count()
            return True
        except Exception:
            return self.initialize()
    
    def extract_details_from_graph(self, graph_file: str) -> List[DetailNode]:
        """
        Extract detail nodes from a graph JSON file.
        
        Args:
            graph_file: Path to graph JSON file
            
        Returns:
            List of DetailNode objects
        """
        details = []
        
        try:
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            file_name = graph_data.get('file_name', '')
            project_name = graph_data.get('project_name', '')
            nodes = graph_data.get('nodes', {})
            
            # Nodes are stored as a dict with node_id as key
            if isinstance(nodes, dict):
                for node_id, node in nodes.items():
                    detail_id = node.get('detail_id', node_id)
                    page_number = node.get('page_number', 0)
                    title = node.get('title', '')
                    element_type = node.get('element_type', 'detail')
                    text_content = node.get('text_content', [])
                    
                    # Skip empty or invalid nodes
                    if not detail_id or not title:
                        continue
                    
                    node_file_name = node.get('file_name', '')
                    node_project_name = node.get('project_name', '')

                    detail = DetailNode(
                        detail_id=detail_id,
                        page_number=page_number,
                        title=title,
                        element_type=element_type,
                        text_content=text_content,
                        file_name=node_file_name or file_name,
                        project_name=node_project_name or project_name,
                        source="graph"
                    )
                    details.append(detail)
            
            print(f"  Extracted {len(details)} details from {Path(graph_file).name}")
            return details
        
        except Exception as e:
            print(f"Error extracting details from {graph_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def build_detail_text(self, detail: DetailNode) -> str:
        """
        Build searchable text from a detail node.
        
        Combines title with text content for better semantic matching.
        """
        # Start with title
        text_parts = [detail.title]
        
        # Add all text content
        text_parts.extend(detail.text_content)
        
        # Join and clean
        full_text = " ".join(str(p).strip() for p in text_parts if p)
        
        # Add element type context
        full_text = f"{detail.element_type.upper()}: {full_text}"
        
        return full_text
    
    def index_detail(self, detail: DetailNode) -> bool:
        """
        Index a single detail node.
        
        Args:
            detail: DetailNode to index
            
        Returns:
            True if successful
        """
        if not self.detail_collection:
            return False
        
        try:
            # Build searchable text
            detail_text = self.build_detail_text(detail)
            detail_embedding = self.embeddings.embed_query(detail_text)

            project_context = f"{detail.project_name} | {detail.file_name}"
            page_context = f"{detail.title} | {detail.element_type} | page {detail.page_number}"
            project_level_labels = self.classification_trainer.classify_project(project_context)
            page_level_labels = self.classification_trainer.classify_page(page_context)
            detail_level = self.classification_trainer.classify_detail(detail_text)

            if not project_level_labels:
                project_level_labels = ["Unclassified > project"]
            if not page_level_labels:
                page_level_labels = ["Unclassified > page"]
            if not detail_level.labels:
                detail_level.labels = ["Unclassified > detail"]
            
            # Create metadata
            metadata = {
                'detail_id': detail.detail_id,
                'page_number': detail.page_number,
                'title': detail.title,
                'element_type': detail.element_type,
                'file_name': detail.file_name,
                'project_name': detail.project_name,
                'source': detail.source,
                'index_level': 'detail_chunk',
                'project_level_labels': ', '.join(project_level_labels),
                'page_level_labels': ', '.join(page_level_labels),
                'detail_level_labels': ', '.join(detail_level.labels),
                'referenced_sheet_ids': ', '.join(detail_level.referenced_sheet_ids),
                'referenced_detail_ids': ', '.join(detail_level.referenced_detail_ids),
            }
            
            # Add to collection
            self.detail_collection.add(
                ids=[detail.detail_id],
                documents=[detail_text],
                metadatas=[metadata],
                embeddings=[detail_embedding],
            )
            
            return True
        except Exception as e:
            print(f"Error indexing detail {detail.detail_id}: {str(e)}")
            return False
    
    def index_all_graphs(self) -> int:
        """
        Index all graphs in data/graphs directory.
        
        Returns:
            Total number of details indexed
        """
        if not self.detail_collection:
            if not self.initialize():
                return 0
        
        total_indexed = 0
        
        # Find all graph files
        if not os.path.exists(self.graphs_dir):
            print(f"Graphs directory not found: {self.graphs_dir}")
            return 0
        
        graph_files = sorted(Path(self.graphs_dir).glob("*_detail_graph.json"))
        
        if not graph_files:
            print(f"No graph files found in {self.graphs_dir}")
            return 0
        
        print(f"\nIndexing details from {len(graph_files)} graphs...")
        
        for graph_file in graph_files:
            project_name = Path(graph_file).stem.replace("_detail_graph", "")
            print(f"\nProcessing: {project_name}")
            
            # Extract details from graph
            details = self.extract_details_from_graph(str(graph_file))
            
            # Index each detail
            indexed = 0
            for detail in details:
                if self.index_detail(detail):
                    indexed += 1
            
            total_indexed += indexed
            print(f"  Indexed {indexed}/{len(details)} details")
        
        print(f"\n✓ Total details indexed: {total_indexed}")
        return total_indexed
    
    def search_details(
        self,
        query: str,
        k: int = 20,
        element_type_filter: Optional[str] = None,
        project_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Search detail nodes using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            element_type_filter: Optional filter (e.g., "section", "elevation")
            project_filter: Optional project scope filter
            
        Returns:
            List of matched details with scores
        """
        if not self._ensure_collection_ready():
            return []
        
        try:
            query_embedding = self.embeddings.embed_query(query)

            # Build where filter
            where_filter = None
            if element_type_filter or project_filter:
                filters = []
                if element_type_filter:
                    filters.append({"element_type": {"$eq": element_type_filter}})
                if project_filter:
                    filters.append({"project_name": {"$eq": project_filter}})
                
                if len(filters) == 1:
                    where_filter = filters[0]
                elif len(filters) > 1:
                    where_filter = {"$and": filters}
            
            # Query collection
            results = self.detail_collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Transform results to list of dicts
            output = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i, detail_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    document = results['documents'][0][i] if results['documents'] else ""
                    
                    # Convert distance to similarity score (0-1, higher is better)
                    # Cosine distance ranges 0-2, convert to similarity
                    similarity_score = 1 - (distance / 2) if distance else 1.0
                    
                    output.append({
                        'detail_id': detail_id,
                        'title': metadata.get('title', ''),
                        'element_type': metadata.get('element_type', ''),
                        'page_number': metadata.get('page_number', 0),
                        'file_name': metadata.get('file_name', ''),
                        'project_name': metadata.get('project_name', ''),
                        'content_sample': document[:200],  # First 200 chars
                        'similarity_score': similarity_score,
                        'source': 'detail_graph'
                    })
            
            return output
        
        except Exception as e:
            if "does not exist" in str(e).lower():
                if self.initialize():
                    return self.search_details(
                        query=query,
                        k=k,
                        element_type_filter=element_type_filter,
                        project_filter=project_filter,
                    )
            print(f"Error searching details: {str(e)}")
            return []
    
    def get_related_details(
        self,
        detail_id: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Find details semantically related to a given detail.
        
        Args:
            detail_id: The detail to find relations for
            k: Number of related details to return
            
        Returns:
            List of related details
        """
        if not self._ensure_collection_ready():
            return []
        
        try:
            # Get the detail's metadata and document
            results = self.detail_collection.get(
                ids=[detail_id],
                include=["documents", "embeddings"]
            )
            
            if not results or not results['documents'] or len(results['documents']) == 0:
                return []
            
            detail_text = results['documents'][0]
            detail_embedding = None
            if results.get('embeddings') and len(results['embeddings']) > 0:
                detail_embedding = results['embeddings'][0]
            if detail_embedding is None:
                detail_embedding = self.embeddings.embed_query(detail_text)
            
            # Search for similar details (exclude the detail itself)
            similar = self.detail_collection.query(
                query_embeddings=[detail_embedding],
                n_results=k + 1,  # +1 because result will include itself
                include=["metadatas", "distances"]
            )
            
            # Filter out the original detail and build results
            output = []
            if similar and similar['ids'] and len(similar['ids']) > 0:
                for i, found_id in enumerate(similar['ids'][0]):
                    if found_id == detail_id:  # Skip the original
                        continue
                    
                    distance = similar['distances'][0][i] if similar['distances'] else 0
                    metadata = similar['metadatas'][0][i] if similar['metadatas'] else {}
                    
                    similarity_score = 1 - (distance / 2) if distance else 1.0
                    
                    output.append({
                        'detail_id': found_id,
                        'title': metadata.get('title', ''),
                        'element_type': metadata.get('element_type', ''),
                        'project_name': metadata.get('project_name', ''),
                        'similarity_score': similarity_score,
                    })
                    
                    if len(output) >= k:
                        break
            
            return output
        
        except Exception as e:
            if "does not exist" in str(e).lower():
                if self.initialize():
                    return self.get_related_details(detail_id=detail_id, k=k)
            print(f"Error finding related details: {str(e)}")
            return []
    
    def clear_collection(self):
        """Clear all indexed details."""
        if not self.chroma_client:
            return False
        
        try:
            self.chroma_client.delete_collection(name=self.detail_collection_name)
            self.detail_collection = None
            print("Cleared detail collection")
            return True
        except Exception as e:
            print(f"Error clearing collection: {str(e)}")
            return False

"""
Integration module for building detail graphs from enriched PDF metadata.
Connects the geometric analysis with the enriched metadata layer to create
a comprehensive detail connectivity graph.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import logging

from detail_graph_builder import DetailNode, DetailEdge, DetailGraph

logger = logging.getLogger(__name__)


class EnrichedMetadataGraphBuilder:
    """
    Build detail graphs from enriched PDF metadata.
    Uses the structural_elements, detail_types, and detail_intents from enriched metadata
    to construct meaningful detail connectivity relationships.
    """
    
    # Mapping from detail types to standardized categories
    DETAIL_TYPE_MAPPING = {
        'section': 'section',
        'sections': 'section',
        'elevation': 'elevation',
        'elevations': 'elevation',
        'detail': 'detail',
        'details': 'detail',
        'schedule': 'schedule',
        'plan': 'plan',
        'layout': 'layout',
        'diagram': 'diagram',
        'sketch': 'detail',
        'view': 'view'
    }
    
    # Keywords that indicate relationships between elements
    RELATIONSHIP_KEYWORDS = {
        'coincident': 'same_element',
        'aligned': 'aligned_with',
        'perpendicular': 'perpendicular_to',
        'parallel': 'parallel_to',
        'on': 'resting_on',
        'above': 'above',
        'below': 'below',
        'attached': 'attached_to',
        'connected': 'connected_to',
        'detailed': 'details_of',
        'shown': 'shown_in',
        'reference': 'references'
    }
    
    def __init__(self):
        pass
    
    def build_graph_from_enriched_metadata(
        self,
        enriched_metadata: Dict[int, Dict[str, Any]],
        pdf_file_name: str,
        project_name: str,
        geometry_data: Optional[Dict] = None
    ) -> DetailGraph:
        """
        Build a detail graph from enriched PDF metadata.
        
        Args:
            enriched_metadata: Output from enriched_metadata_loader.load_enriched_metadata()
                              Dict mapping page_num → metadata dict with title_block, page_type, etc.
            pdf_file_name: Name of source PDF
            project_name: Name of project
            geometry_data: Optional geometry data from PDFGeometryParser for enhanced relationships
            
        Returns:
            DetailGraph with nodes and edges
        """
        graph = DetailGraph(project_name=project_name)
        
        # Phase 1: Extract detail nodes from enriched metadata
        page_details: Dict[int, Dict[str, DetailNode]] = {}  # page_num → {detail_id → node}
        
        for page_num, metadata in enriched_metadata.items():
            page_details[page_num] = {}
            
            # Get detail types from this page
            title_block = metadata.get('title_block') or {}
            plan_contents = title_block.get('plan_contents') or {}
            detail_types = plan_contents.get('detail_types') or []
            structural_elements = plan_contents.get('structural_elements') or []
            
            # Get page type
            page_type = metadata.get('page_type', 'unknown')
            
            # Create nodes for each detail type mentioned
            detail_idx = 0
            for detail_str in detail_types:
                detail_title = str(detail_str).strip()
                if not detail_title:
                    continue
                
                # Classify detail type
                element_type = self._classify_detail_type(detail_title)
                
                # Create node
                detail_id = f"Page{page_num}_Detail{detail_idx}"
                node = DetailNode(
                    detail_id=detail_id,
                    page_number=page_num,
                    title=detail_title,
                    element_type=element_type,
                    bbox={
                        'x0': 0, 'y0': 0, 'x1': 100, 'y1': 100  # Placeholder
                    },
                    text_content=[detail_title] + structural_elements[:3],
                    file_name=pdf_file_name,
                    project_name=project_name
                )
                graph.add_node(node)
                page_details[page_num][detail_id] = node
                detail_idx += 1
        
        # Phase 2: Build relationships within and across pages
        self._build_element_based_relationships(graph, page_details, enriched_metadata)
        self._build_intent_based_relationships(graph, enriched_metadata, page_details)
        self._build_page_hierarchy_relationships(graph, enriched_metadata, page_details)
        
        logger.info(f"Built graph with {len(graph.nodes)} detail nodes and "
                   f"{sum(len(e) for e in graph.edges.values())} relationships")
        
        return graph
    
    def _build_element_based_relationships(
        self,
        graph: DetailGraph,
        page_details: Dict[int, Dict[str, DetailNode]],
        enriched_metadata: Dict[int, Dict[str, Any]]
    ) -> None:
        """
        Build relationships based on shared structural elements.
        Details that reference the same elements are connected.
        """
        # Build a map of element → [detail_ids that mention it]
        element_to_details: Dict[str, List[str]] = defaultdict(list)
        
        for page_num, details_on_page in page_details.items():
            metadata = enriched_metadata.get(page_num) or {}
            title_block = metadata.get('title_block') or {}
            plan_contents = title_block.get('plan_contents') or {}
            structural_elements = plan_contents.get('structural_elements') or []
            
            for element in structural_elements:
                element_lower = str(element).lower()
                for detail_id in details_on_page.keys():
                    element_to_details[element_lower].append(detail_id)
        
        # Create edges for details sharing elements
        for element, detail_ids in element_to_details.items():
            detail_ids = list(set(detail_ids))  # Remove duplicates
            
            # Connect all pairs of details sharing this element
            for i, detail_a in enumerate(detail_ids):
                for detail_b in detail_ids[i+1:]:
                    if detail_a != detail_b and detail_a in graph.nodes and detail_b in graph.nodes:
                        edge = DetailEdge(
                            source_id=detail_a,
                            target_id=detail_b,
                            relationship_type='shared_element',
                            strength=0.7,
                            reason=f"Both reference: {element[:40]}"
                        )
                        graph.add_edge(edge)
    
    def _build_intent_based_relationships(
        self,
        graph: DetailGraph,
        enriched_metadata: Dict[int, Dict[str, Any]],
        page_details: Dict[int, Dict[str, DetailNode]]
    ) -> None:
        """
        Build relationships based on detail intents (e.g., "detail of girder", "reinforcement detail").
        """
        # Extract all intents and map to details
        for page_num, metadata in enriched_metadata.items():
            metadata = metadata or {}
            details_on_page = page_details.get(page_num, {})
            detail_intents = metadata.get('detail_intents') or []
            
            for intent in detail_intents:
                intent_str = str(intent).lower()
                
                # Find which detail this intent belongs to
                for detail_id, node in details_on_page.items():
                    node_title_lower = node.title.lower()
                    
                    # Check for relationships in the intent string
                    for keyword, rel_type in self.RELATIONSHIP_KEYWORDS.items():
                        if keyword in intent_str:
                            # Look for the target element in the intent
                            words = intent_str.split()
                            for word in words:
                                # Check if this word matches another detail
                                for other_page_num, other_details in page_details.items():
                                    for other_detail_id, other_node in other_details.items():
                                        if detail_id != other_detail_id:
                                            # Check for semantic relationship
                                            if word in other_node.title.lower():
                                                edge = DetailEdge(
                                                    source_id=detail_id,
                                                    target_id=other_detail_id,
                                                    relationship_type=rel_type,
                                                    strength=0.6,
                                                    reason=f"Intent: {intent_str[:50]}"
                                                )
                                                graph.add_edge(edge)
    
    def _build_page_hierarchy_relationships(
        self,
        graph: DetailGraph,
        enriched_metadata: Dict[int, Dict[str, Any]],
        page_details: Dict[int, Dict[str, DetailNode]]
    ) -> None:
        """
        Build hierarchical relationships based on page types and ordering.
        General plans typically reference detail pages.
        """
        # Categorize pages by type
        general_pages = []
        detail_pages = []
        section_pages = []
        
        for page_num, metadata in enriched_metadata.items():
            page_type = metadata.get('page_type', '').lower()
            
            if 'general' in page_type or 'plan' in page_type:
                general_pages.append(page_num)
            elif 'detail' in page_type:
                detail_pages.append(page_num)
            elif 'section' in page_type:
                section_pages.append(page_num)
        
        # Connect general pages to detail pages
        for gen_page in general_pages:
            for det_page in detail_pages:
                gen_details = page_details.get(gen_page, {})
                det_details = page_details.get(det_page, {})
                
                # General page details reference detail page details
                for gen_detail_id in gen_details.keys():
                    for det_detail_id in det_details.keys():
                        edge = DetailEdge(
                            source_id=gen_detail_id,
                            target_id=det_detail_id,
                            relationship_type='references_detail',
                            strength=0.5,
                            reason=f"General plan (page {gen_page}) references detail (page {det_page})"
                        )
                        graph.add_edge(edge)
    
    def _classify_detail_type(self, detail_title: str) -> str:
        """Classify detail type from its title."""
        title_lower = detail_title.lower()
        
        for keyword, detail_type in self.DETAIL_TYPE_MAPPING.items():
            if keyword in title_lower:
                return detail_type
        
        return 'detail'  # Default


class GraphIntegrationManager:
    """
    Manages integration of detail graphs with the knowledge base system.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path('./data/graphs')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.builder = EnrichedMetadataGraphBuilder()
        self.graphs: Dict[str, DetailGraph] = {}  # project_name → graph
    
    def build_and_store_graph(
        self,
        enriched_metadata: Dict[int, Dict[str, Any]],
        pdf_file_name: str,
        project_name: str
    ) -> DetailGraph:
        """
        Build a detail graph from enriched metadata and store it.
        
        Returns:
            The constructed DetailGraph
        """
        graph = self.builder.build_graph_from_enriched_metadata(
            enriched_metadata=enriched_metadata,
            pdf_file_name=pdf_file_name,
            project_name=project_name
        )
        
        # Store in memory
        self.graphs[project_name] = graph
        
        # Persist to file
        self._save_graph(graph)
        
        return graph
    
    def get_graph(self, project_name: str) -> Optional[DetailGraph]:
        """Get a graph, loading from file if not in memory."""
        if project_name in self.graphs:
            return self.graphs[project_name]
        
        graph = self._load_graph(project_name)
        if graph:
            self.graphs[project_name] = graph
        
        return graph
    
    def find_related_details(
        self,
        project_name: str,
        query_detail_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Find all details related to a query detail.
        
        Returns:
            Dict with:
            - 'query_detail': DetailNode
            - 'connected': Set of connected detail IDs
            - 'related_nodes': List of related DetailNode objects
            - 'relationships': List of DetailEdge objects
        """
        graph = self.get_graph(project_name)
        if not graph:
            return {'error': f'Graph not found for project: {project_name}'}
        
        # Find connected components
        connected = graph.get_connected_components(query_detail_id, max_depth=max_depth)
        
        # Get related nodes and their relationships
        related_nodes = []
        relationships = []
        
        for detail_id in connected:
            if detail_id != query_detail_id and detail_id in graph.nodes:
                related_nodes.append(graph.nodes[detail_id])
        
        # Collect all edges connected to query detail
        for edge in graph.edges.get(query_detail_id, []):
            relationships.append(edge)
        for edge in graph.reverse_edges.get(query_detail_id, []):
            relationships.append(edge)
        
        return {
            'query_detail': graph.nodes.get(query_detail_id),
            'connected': connected,
            'related_nodes': related_nodes,
            'relationships': relationships,
            'total_connected': len(connected)
        }

    def _ensure_all_graphs_loaded(self) -> None:
        """Load all persisted project graphs into memory for global graph search."""
        graph_files = sorted(self.data_dir.glob('*_detail_graph.json'))

        for graph_file in graph_files:
            project_name = graph_file.stem.removesuffix('_detail_graph')
            if project_name in self.graphs:
                continue

            graph = self._load_graph(project_name)
            if graph:
                self.graphs[project_name] = graph
    
    def find_related_documents(
        self,
        query: str,
        max_hops: int = 2,
        limit: int = 20
    ) -> List[Dict]:
        """
        Find documents related to a query by searching graph structures.
        
        Args:
            query: Search query string
            max_hops: Maximum hops to traverse in graphs
            limit: Maximum number of related documents to return
            
        Returns:
            List of related document dicts with 'id', 'content', and 'metadata'
        """
        # Ensure queries cover all projects, not only graphs built during this process.
        self._ensure_all_graphs_loaded()

        related_docs = []
        
        # Extract keywords from query
        query_lower = query.lower()
        keywords = query_lower.split()
        
        # Search all available graphs
        for project_name, graph in self.graphs.items():
            if not graph:
                continue
            
            # Search nodes for matching details
            for node_id, node in graph.nodes.items():
                if not node:
                    continue
                
                node_title = str(getattr(node, 'title', '')).lower()
                
                # Check if any keyword matches the node title
                if any(keyword in node_title for keyword in keywords):
                    # Found a matching detail, get related docs
                    # For now, return the detail itself
                    doc_dict = {
                        'id': node_id,
                        'content': node_title,
                        'metadata': {
                            'project': project_name,
                            'detail_type': getattr(node, 'detail_type', 'unknown'),
                            'page': getattr(node, 'page_num', 0),
                            'source': 'graph'
                        }
                    }
                    related_docs.append(doc_dict)
                    
                    if len(related_docs) >= limit:
                        return related_docs
        
        return related_docs
    
    def _save_graph(self, graph: DetailGraph) -> None:
        """Save graph to JSON file."""
        output_path = self.data_dir / f"{graph.project_name}_detail_graph.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(graph.to_dict(), f, indent=2)
            logger.info(f"Saved detail graph to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
    
    def _load_graph(self, project_name: str) -> Optional[DetailGraph]:
        """Load graph from JSON file."""
        graph_path = self.data_dir / f"{project_name}_detail_graph.json"
        
        if not graph_path.exists():
            return None
        
        try:
            with open(graph_path, 'r') as f:
                data = json.load(f)
            return DetailGraph.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return None

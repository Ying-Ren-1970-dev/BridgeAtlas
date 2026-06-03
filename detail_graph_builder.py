"""
Detail graph builder for constructing connectivity graphs from parsed PDF geometry.
Stores graph as JSON-based adjacency structure for efficient querying.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DetailNode:
    """Represents a detail/callout as a node in the graph."""
    detail_id: str
    page_number: int
    title: str
    element_type: str  # detail, section, elevation, schedule, etc.
    bbox: Dict[str, float]  # {x0, y0, x1, y1}
    text_content: List[str]
    file_name: str  # Source PDF file
    project_name: str  # Project name
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DetailEdge:
    """Represents a relationship between two details."""
    source_id: str
    target_id: str
    relationship_type: str  # connects_to, references, calls_out, intersects_with, same_page
    strength: float = 1.0  # Confidence/weight of relationship (0-1)
    reason: str = ""  # Human-readable reason for relationship
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DetailGraph:
    """
    Graph representation of details and their relationships.
    Stored as JSON-serializable adjacency list.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.nodes: Dict[str, DetailNode] = {}  # detail_id → DetailNode
        self.edges: Dict[str, List[DetailEdge]] = defaultdict(list)  # source_id → [edges]
        self.reverse_edges: Dict[str, List[DetailEdge]] = defaultdict(list)  # target_id → [edges]
    
    def add_node(self, node: DetailNode) -> None:
        """Add a detail node to the graph."""
        self.nodes[node.detail_id] = node
    
    def add_edge(self, edge: DetailEdge) -> None:
        """Add an edge between two details."""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.warning(f"Cannot add edge: nodes not found ({edge.source_id}, {edge.target_id})")
            return
        
        self.edges[edge.source_id].append(edge)
        self.reverse_edges[edge.target_id].append(edge)
    
    def get_connected_components(self, detail_id: str, max_depth: int = 2) -> Set[str]:
        """
        Get all details connected to a given detail up to max_depth hops.
        
        Args:
            detail_id: Starting detail ID
            max_depth: Maximum number of hops to traverse
            
        Returns:
            Set of connected detail IDs
        """
        if detail_id not in self.nodes:
            return set()
        
        visited = set()
        queue = [(detail_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Add outgoing edges
            for edge in self.edges.get(current_id, []):
                if edge.target_id not in visited:
                    queue.append((edge.target_id, depth + 1))
            
            # Add incoming edges
            for edge in self.reverse_edges.get(current_id, []):
                if edge.source_id not in visited:
                    queue.append((edge.source_id, depth + 1))
        
        return visited
    
    def get_node_details(self, detail_id: str) -> Optional[DetailNode]:
        """Get a detail node by ID."""
        return self.nodes.get(detail_id)
    
    def get_related_nodes(self, detail_id: str, relationship_type: Optional[str] = None) -> List[DetailNode]:
        """
        Get details related to the given detail.
        
        Args:
            detail_id: Detail ID to query
            relationship_type: Filter by relationship type (optional)
            
        Returns:
            List of related DetailNode objects
        """
        related = []
        
        # Outgoing edges
        for edge in self.edges.get(detail_id, []):
            if relationship_type is None or edge.relationship_type == relationship_type:
                if edge.target_id in self.nodes:
                    related.append(self.nodes[edge.target_id])
        
        # Incoming edges
        for edge in self.reverse_edges.get(detail_id, []):
            if relationship_type is None or edge.relationship_type == relationship_type:
                if edge.source_id in self.nodes:
                    related.append(self.nodes[edge.source_id])
        
        return related
    
    def to_dict(self) -> Dict:
        """Convert graph to dictionary format."""
        return {
            'project_name': self.project_name,
            'nodes': {nid: node.to_dict() for nid, node in self.nodes.items()},
            'edges': {nid: [edge.to_dict() for edge in edges] 
                     for nid, edges in self.edges.items()},
            'node_count': len(self.nodes),
            'edge_count': sum(len(edges) for edges in self.edges.values())
        }
    
    @staticmethod
    def from_dict(data: Dict) -> 'DetailGraph':
        """Create graph from dictionary."""
        graph = DetailGraph(project_name=data.get('project_name', 'default'))
        
        # Add nodes
        for detail_id, node_data in data.get('nodes', {}).items():
            node = DetailNode(**node_data)
            graph.add_node(node)
        
        # Add edges
        for source_id, edge_list in data.get('edges', {}).items():
            for edge_data in edge_list:
                edge = DetailEdge(**edge_data)
                graph.add_edge(edge)
        
        return graph


class DetailGraphBuilder:
    """
    Builder for constructing detail graphs from parsed PDF geometry data.
    """
    
    # Keywords indicating spatial references/connections
    REFERENCE_KEYWORDS = [
        'see', 'refer', 'shown', 'match', 'align', 'connects',
        'detail', 'section', 'notes', 'reference', 'for details',
        'call out', 'denote', 'indicates'
    ]
    
    # Keywords indicating same-element references
    ELEMENT_KEYWORDS = [
        'girder', 'column', 'pier', 'abutment', 'pile', 'cap',
        'slab', 'deck', 'bearing', 'expansion', 'joint',
        'reinforcement', 'connection', 'splice', 'pin', 'bolt'
    ]
    
    def __init__(self):
        self.current_graph: Optional[DetailGraph] = None
    
    def build_graph_from_geometry(
        self,
        geometry_data: Dict,
        pdf_file_name: str,
        project_name: str
    ) -> DetailGraph:
        """
        Build detail graph from parsed geometry data.
        
        Args:
            geometry_data: Output from PDFGeometryParser.parse_pdf()
            pdf_file_name: Name of source PDF file
            project_name: Name of project
            
        Returns:
            DetailGraph object
        """
        graph = DetailGraph(project_name=project_name)
        
        # Phase 1: Add all detail nodes
        page_details: Dict[int, List[str]] = defaultdict(list)  # page_num → [detail_ids]
        
        for page_num_str, page_data in geometry_data.items():
            page_num = int(page_num_str)
            
            for callout_data in page_data.get('detail_callouts', []):
                # Create node for this detail callout
                node = DetailNode(
                    detail_id=callout_data['detail_id'],
                    page_number=callout_data['page_number'],
                    title=callout_data['title'],
                    element_type=callout_data['element_type'],
                    bbox=callout_data['bbox'],
                    text_content=callout_data['text_content'],
                    file_name=pdf_file_name,
                    project_name=project_name
                )
                graph.add_node(node)
                page_details[page_num].append(callout_data['detail_id'])
        
        # Phase 2: Build relationships between details
        self._build_text_based_relationships(graph, geometry_data)
        self._build_spatial_relationships(graph, geometry_data)
        self._build_same_page_relationships(graph, page_details)
        
        self.current_graph = graph
        return graph
    
    def _build_text_based_relationships(self, graph: DetailGraph, geometry_data: Dict) -> None:
        """
        Build relationships based on text references between details.
        E.g., if Detail A's text says "See Detail B", create an edge.
        """
        all_details = list(graph.nodes.keys())
        
        for source_detail_id in all_details:
            source_node = graph.nodes[source_detail_id]
            source_text_lower = ' '.join(source_node.text_content).lower()
            
            # Look for references to other details
            for target_detail_id in all_details:
                if source_detail_id == target_detail_id:
                    continue
                
                target_node = graph.nodes[target_detail_id]
                target_text_lower = target_node.title.lower()
                
                # Check if source text references target
                if target_text_lower in source_text_lower:
                    # Found a reference
                    edge = DetailEdge(
                        source_id=source_detail_id,
                        target_id=target_detail_id,
                        relationship_type='references',
                        strength=0.8,
                        reason=f"Text reference: '{target_node.title}' found in {source_detail_id}"
                    )
                    graph.add_edge(edge)
    
    def _build_spatial_relationships(self, graph: DetailGraph, geometry_data: Dict) -> None:
        """
        Build relationships based on spatial proximity and element sharing.
        E.g., if two details reference the same structural element, connect them.
        """
        all_details = list(graph.nodes.keys())
        
        for source_detail_id in all_details:
            source_node = graph.nodes[source_detail_id]
            source_page = source_node.page_number
            
            for target_detail_id in all_details:
                if source_detail_id >= target_detail_id:  # Avoid duplicates
                    continue
                
                target_node = graph.nodes[target_detail_id]
                
                # Connect details on same page that mention same elements
                if source_page == target_node.page_number:
                    # Check for shared element keywords
                    source_elements = self._extract_element_keywords(source_node.text_content)
                    target_elements = self._extract_element_keywords(target_node.text_content)
                    
                    shared_elements = source_elements & target_elements
                    
                    if shared_elements:
                        edge = DetailEdge(
                            source_id=source_detail_id,
                            target_id=target_detail_id,
                            relationship_type='intersects_with',
                            strength=len(shared_elements) / max(len(source_elements), len(target_elements)),
                            reason=f"Shared elements: {', '.join(list(shared_elements)[:3])}"
                        )
                        graph.add_edge(edge)
    
    def _build_same_page_relationships(
        self,
        graph: DetailGraph,
        page_details: Dict[int, List[str]]
    ) -> None:
        """
        Create lightweight relationships for details on the same page.
        """
        for page_num, detail_ids in page_details.items():
            # Connect all details on same page (transitive)
            for i, detail_id_a in enumerate(detail_ids):
                for detail_id_b in detail_ids[i+1:]:
                    edge = DetailEdge(
                        source_id=detail_id_a,
                        target_id=detail_id_b,
                        relationship_type='same_page',
                        strength=0.5,
                        reason=f"Both on page {page_num}"
                    )
                    graph.add_edge(edge)
    
    def _extract_element_keywords(self, text_list: List[str]) -> Set[str]:
        """Extract structural element keywords from text."""
        text_lower = ' '.join(text_list).lower()
        found_elements = set()
        
        for keyword in self.ELEMENT_KEYWORDS:
            if keyword in text_lower:
                found_elements.add(keyword)
        
        return found_elements


class GraphStore:
    """
    Manages persistence and retrieval of detail graphs.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize graph store.
        
        Args:
            data_dir: Directory to store graph JSON files
        """
        self.data_dir = data_dir or Path('./data/graphs')
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graphs: Dict[str, DetailGraph] = {}  # project_name → graph
    
    def save_graph(self, graph: DetailGraph, graph_name: Optional[str] = None) -> Path:
        """
        Save a detail graph to JSON file.
        
        Args:
            graph: DetailGraph to save
            graph_name: Name for the graph file (defaults to project_name)
            
        Returns:
            Path to saved file
        """
        name = graph_name or graph.project_name
        output_path = self.data_dir / f"{name}_detail_graph.json"
        
        try:
            with open(output_path, 'w') as f:
                json.dump(graph.to_dict(), f, indent=2)
            
            logger.info(f"Saved detail graph to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise
    
    def load_graph(self, graph_name: str) -> Optional[DetailGraph]:
        """
        Load a detail graph from JSON file.
        
        Args:
            graph_name: Name of graph to load
            
        Returns:
            DetailGraph or None if not found
        """
        graph_path = self.data_dir / f"{graph_name}_detail_graph.json"
        
        if not graph_path.exists():
            logger.warning(f"Graph file not found: {graph_path}")
            return None
        
        try:
            with open(graph_path, 'r') as f:
                data = json.load(f)
            
            graph = DetailGraph.from_dict(data)
            logger.info(f"Loaded detail graph from {graph_path}")
            return graph
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return None
    
    def list_graphs(self) -> List[str]:
        """List all available graphs."""
        graphs = []
        for path in self.data_dir.glob("*_detail_graph.json"):
            name = path.stem.replace("_detail_graph", "")
            graphs.append(name)
        return sorted(graphs)

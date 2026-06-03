"""
Test script to validate PDF geometry parser and detail graph builder.
"""

import json
import logging
from pathlib import Path
from pdf_geometry_parser import PDFGeometryParser
from detail_graph_builder import DetailGraphBuilder, GraphStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_geometry_parser():
    """Test geometry extraction on a sample PDF."""
    
    # Find a sample PDF
    projects_dir = Path('./Projects')
    pdfs = list(projects_dir.rglob('*.pdf'))
    
    if not pdfs:
        logger.error("No PDFs found in Projects directory")
        return
    
    # Use Mar Vista POC if available, otherwise first PDF
    sample_pdf = None
    for pdf in pdfs:
        if 'Mar Vista' in pdf.name:
            sample_pdf = pdf
            break
    
    if not sample_pdf:
        sample_pdf = pdfs[0]
    
    logger.info(f"Testing geometry parser on: {sample_pdf.name}")
    
    # Parse geometry
    parser = PDFGeometryParser(max_pages=5)  # Limit to 5 pages for testing
    geometry_data = parser.parse_pdf(sample_pdf)
    
    if not geometry_data:
        logger.error("No geometry data extracted")
        return
    
    logger.info(f"Extracted geometry from {len(geometry_data)} pages")
    
    # Print summary
    for page_num in sorted(geometry_data.keys())[:3]:  # Show first 3 pages
        page_data = geometry_data[page_num]
        logger.info(f"\nPage {page_num}:")
        logger.info(f"  - Dimensions: {page_data['width']:.1f} x {page_data['height']:.1f}")
        logger.info(f"  - Geometric elements: {len(page_data['elements'])}")
        logger.info(f"  - Text labels: {len(page_data['text_labels'])}")
        logger.info(f"  - Detail callouts: {len(page_data['detail_callouts'])}")
        
        # Show detail callouts
        for callout in page_data['detail_callouts'][:3]:
            logger.info(f"    * {callout['detail_id']}: {callout['title']} ({callout['element_type']})")
            if callout['text_content']:
                logger.info(f"      Text: {', '.join(callout['text_content'][:2])}")
    
    return sample_pdf, geometry_data


def test_graph_builder(pdf_path: Path, geometry_data: dict):
    """Test graph construction from geometry data."""
    
    logger.info(f"\nBuilding detail graph...")
    
    builder = DetailGraphBuilder()
    graph = builder.build_graph_from_geometry(
        geometry_data=geometry_data,
        pdf_file_name=pdf_path.name,
        project_name=pdf_path.stem
    )
    
    logger.info(f"Graph constructed:")
    logger.info(f"  - Nodes (details): {len(graph.nodes)}")
    logger.info(f"  - Edges (relationships): {sum(len(edges) for edges in graph.edges.values())}")
    
    # Show some nodes and their relationships
    for detail_id in list(graph.nodes.keys())[:5]:
        node = graph.nodes[detail_id]
        outgoing = len(graph.edges.get(detail_id, []))
        incoming = len(graph.reverse_edges.get(detail_id, []))
        logger.info(f"  * {detail_id}: '{node.title}' (outgoing: {outgoing}, incoming: {incoming})")
    
    # Show some edge examples
    for source_id in list(graph.edges.keys())[:3]:
        edges = graph.edges[source_id]
        for edge in edges[:2]:
            logger.info(f"    → {edge.source_id} --[{edge.relationship_type}]--> {edge.target_id}")
            logger.info(f"       Reason: {edge.reason}")
    
    return graph


def test_graph_store(graph):
    """Test graph persistence."""
    
    logger.info(f"\nTesting graph persistence...")
    
    store = GraphStore()
    
    # Save graph
    saved_path = store.save_graph(graph)
    logger.info(f"Saved graph to: {saved_path}")
    
    # Load graph back
    loaded_graph = store.load_graph(graph.project_name)
    if loaded_graph:
        logger.info(f"Successfully loaded graph with {len(loaded_graph.nodes)} nodes")
    else:
        logger.error("Failed to load graph")
        return False
    
    # Verify graph properties
    assert len(loaded_graph.nodes) == len(graph.nodes), "Node count mismatch"
    total_edges_original = sum(len(edges) for edges in graph.edges.values())
    total_edges_loaded = sum(len(edges) for edges in loaded_graph.edges.values())
    assert total_edges_loaded == total_edges_original, "Edge count mismatch"
    
    logger.info("Graph persistence verification passed")
    
    # Test graph queries
    if loaded_graph.nodes:
        test_detail_id = list(loaded_graph.nodes.keys())[0]
        connected = loaded_graph.get_connected_components(test_detail_id, max_depth=2)
        logger.info(f"Connected components from {test_detail_id}: {len(connected)} details")
        
        related = loaded_graph.get_related_nodes(test_detail_id)
        logger.info(f"Directly related details: {len(related)}")
    
    return True


def main():
    """Run all tests."""
    logger.info("="*60)
    logger.info("Testing Algorithmic Graph Approach for CAD PDF Parsing")
    logger.info("="*60)
    
    # Test 1: Geometry parsing
    result = test_geometry_parser()
    if not result:
        logger.error("Geometry parsing test failed")
        return
    
    pdf_path, geometry_data = result
    
    # Test 2: Graph building
    graph = test_graph_builder(pdf_path, geometry_data)
    
    # Test 3: Graph persistence
    success = test_graph_store(graph)
    
    if success:
        logger.info("\n" + "="*60)
        logger.info("All tests passed!")
        logger.info("="*60)
    else:
        logger.error("\nSome tests failed")


if __name__ == '__main__':
    main()

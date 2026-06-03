"""
Test script for building detail graphs from enriched metadata.
"""

import logging
from pathlib import Path
import config
from enriched_metadata_loader import EnrichedMetadataLoader
from enriched_metadata_graph_builder import GraphIntegrationManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_graph_building():
    """Test graph building from enriched metadata."""
    
    print("\n" + "=" * 60)
    print("Testing Detail Graph Building")
    print("=" * 60)
    
    # Find enriched files
    enriched_files = sorted(list(config.DATA_FOLDER.glob('enriched_*.json')))
    print(f"\nFound {len(enriched_files)} enriched metadata files")
    
    if not enriched_files:
        print("No enriched files found!")
        return
    
    # Test with first file
    enriched_file = enriched_files[0]
    project_name = enriched_file.stem.replace('enriched_', '')
    pdf_file_name = f"{project_name}.pdf"
    
    print(f"\nTesting with: {pdf_file_name}")
    
    # Load enriched metadata
    print("\nLoading enriched metadata...")
    loader = EnrichedMetadataLoader()
    enriched_metadata = loader.load_enriched_metadata(pdf_file_name)
    
    if not enriched_metadata:
        print(f"Failed to load enriched metadata for {pdf_file_name}")
        return
    
    print(f"Loaded metadata for {len(enriched_metadata)} pages")
    
    # Show sample page data
    sample_page_num = list(enriched_metadata.keys())[0]
    sample_page = enriched_metadata[sample_page_num]
    print(f"\nSample page {sample_page_num}:")
    print(f"  - Page type: {sample_page.get('page_type')}")
    if 'title_block' in sample_page:
        title_block = sample_page['title_block']
        plan_contents = title_block.get('plan_contents', {})
        print(f"  - Detail types: {plan_contents.get('detail_types', [])[:3]}")
        print(f"  - Structural elements: {plan_contents.get('structural_elements', [])[:3]}")
    
    # Build graph
    print("\nBuilding detail graph...")
    manager = GraphIntegrationManager()
    
    try:
        graph = manager.build_and_store_graph(
            enriched_metadata=enriched_metadata,
            pdf_file_name=pdf_file_name,
            project_name=project_name
        )
        
        print(f"\n✓ Graph built successfully!")
        print(f"  - Nodes (details): {len(graph.nodes)}")
        print(f"  - Edges (relationships): {sum(len(e) for e in graph.edges.values())}")
        
        if graph.nodes:
            # Show first few nodes
            print(f"\nFirst 5 nodes:")
            for detail_id in list(graph.nodes.keys())[:5]:
                node = graph.nodes[detail_id]
                print(f"  - {detail_id}: {node.title} (page {node.page_number})")
        
        # Show saved path
        graph_file = manager.data_dir / f"{project_name}_detail_graph.json"
        print(f"\n✓ Graph saved to: {graph_file}")
        
        return graph
        
    except Exception as e:
        print(f"\n❌ Error building graph: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    test_graph_building()

"""
Graph-Enabled Search Integration Guide
======================================

This guide explains how to integrate the detail graphs with the search system
to enable graph-grounded AI queries.

Key Concept:
When a user searches for "CIDH pile detail", the system should:
1. Query the detail graph to find all details matching "CIDH pile"
2. Expand to connected details (e.g., reinforcement details, connection details)
3. Retrieve those specific details from the knowledge base
4. Feed them to the LLM for a comprehensive, connected answer

This provides much better results than returning entire pages containing unrelated details.
"""

# ============================================================================
# STEP 1: PREPARE GRAPHS FOR SEARCH
# ============================================================================

# Run once to build graphs for all projects:
# python main.py build-graphs

# This creates JSON files in data/graphs/ directory:
# - data/graphs/Mar Vista POC 100%_CheckPrint_20211118 Complete_detail_graph.json
# - data/graphs/30%_Elk Grove Station Structure Plan Set_detail_graph.json
# - etc.


# ============================================================================
# STEP 2: INTEGRATE GRAPHS INTO SEARCH (Code Changes)
# ============================================================================

"""
File: search_agent.py

Add to imports:
    from enriched_metadata_graph_builder import GraphIntegrationManager
    from typing import Set

Add to SearchAgent.__init__():
    self.graph_manager = GraphIntegrationManager()

Add new method:

    def find_related_details_from_graph(
        self,
        query: str,
        project_name: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        \"\"\"
        Query the detail graph to find related details.
        
        Args:
            query: Search query (e.g., "CIDH pile detail")
            project_name: Project to search in
            max_depth: How many hops to traverse in the graph
            
        Returns:
            Dict with matched details and connected details
        \"\"\"
        graph = self.graph_manager.get_graph(project_name)
        if not graph:
            return {'matched': [], 'connected': []}
        
        # Find details that match the query keywords
        query_lower = query.lower()
        matched_detail_ids = set()
        
        for detail_id, node in graph.nodes.items():
            # Check if query matches this detail's title or text content
            title_lower = node.title.lower()
            text_lower = ' '.join([t.lower() for t in node.text_content])
            
            # Simple keyword matching (could use semantic similarity)
            query_words = query_lower.split()
            if any(word in title_lower or word in text_lower for word in query_words):
                matched_detail_ids.add(detail_id)
        
        # For each matched detail, find connected components
        all_connected = set()
        for detail_id in matched_detail_ids:
            connected = graph.get_connected_components(detail_id, max_depth=max_depth)
            all_connected.update(connected)
        
        # Get detail information
        matched_nodes = [graph.nodes[did] for did in matched_detail_ids if did in graph.nodes]
        connected_nodes = [graph.nodes[did] for did in all_connected if did in graph.nodes]
        
        return {
            'matched': matched_nodes,
            'connected': connected_nodes,
            'total_details': len(matched_detail_ids | all_connected)
        }


Modify search() method to use graphs:

    def search(self, query: str, k: int = 10, use_graph: bool = True) -> Dict[str, Any]:
        \"\"\"
        Search with optional graph grounding.
        
        If use_graph=True, first queries the detail graph to find connected details,
        then searches vector store for those specific details.
        \"\"\"
        results = []
        
        # Step 1: Query graphs to find related details (optional but recommended)
        if use_graph:
            print(f"\\nQuerying detail graphs...")
            
            # Try each project's graph
            for project_name in self.graph_manager.graphs.keys():
                graph_results = self.find_related_details_from_graph(
                    query=query,
                    project_name=project_name,
                    max_depth=2
                )
                
                if graph_results['matched'] or graph_results['connected']:
                    print(f"  - {project_name}: Found {graph_results['total_details']} related details")
                    
                    # Could enhance vector search by boosting these details
                    # For now, just note them for context
        
        # Step 2: Standard vector search
        vector_results = self.vector_store.similarity_search(
            query=query,
            k=k,
            use_hybrid_search=True,
            relevance_threshold=0.5
        )
        
        # Step 3: Optionally add graph context to results
        for result in vector_results:
            # Could add fields like:
            # result['graph_context'] = {
            #     'related_details': [...],
            #     'detail_type': 'section' | 'detail' | 'elevation'
            # }
            results.append(result)
        
        return {'results': results[:k], 'count': len(results)}
"""


# ============================================================================
# STEP 3: UPDATE VECTOR STORE FOR DETAIL-LEVEL INDEXING
# ============================================================================

"""
File: vector_store.py

Modify add_documents() to optionally break pages into details:

    def add_documents(self, pdf_data_list, use_detail_level=False):
        \"\"\"
        Add documents to vector store.
        
        If use_detail_level=True, extract individual details from each page
        using the detail graph, creating separate vectors for each detail.
        \"\"\"
        
        for pdf_data in pdf_data_list:
            chunks = pdf_data['chunks']
            file_name = pdf_data['file_name']
            project_name = pdf_data.get('project_name', '')
            
            if use_detail_level and project_name:
                # Load detail graph for this project
                graph = self.graph_manager.get_graph(project_name)
                
                if graph:
                    # Break chunks by detail
                    detail_chunks = self._break_chunks_by_detail(chunks, graph, file_name)
                    self.vector_store.add(
                        ids=[f"detail-{i}" for i in range(len(detail_chunks))],
                        metadatas=[chunk['metadata'] for chunk in detail_chunks],
                        documents=[chunk['text'] for chunk in detail_chunks]
                    )
                else:
                    # Fall back to regular chunking
                    self.vector_store.add(chunks)
            else:
                # Regular page-level indexing
                self.vector_store.add(chunks)

    def _break_chunks_by_detail(self, chunks, graph, file_name):
        \"\"\"Break page-level chunks into detail-level chunks.\"\"\"
        detail_chunks = []
        
        for chunk in chunks:
            page_num = chunk['metadata'].get('page')
            
            # Find details on this page
            details_on_page = [
                (did, node) for did, node in graph.nodes.items()
                if node.page_number == page_num
            ]
            
            if not details_on_page:
                # No details found, use chunk as-is
                detail_chunks.append(chunk)
            else:
                # Create separate chunks for each detail
                for detail_id, detail_node in details_on_page:
                    metadata = chunk['metadata'].copy()
                    metadata['detail_id'] = detail_id
                    metadata['detail_title'] = detail_node.title
                    metadata['element_type'] = detail_node.element_type
                    
                    # Extract relevant text for this detail
                    detail_text = f"{detail_node.title}\\n"
                    detail_text += f"Type: {detail_node.element_type}\\n"
                    detail_text += f"Page: {detail_node.page_number}\\n"
                    detail_text += f"Content: {' '.join(detail_node.text_content[:5])}\\n"
                    detail_text += f"\\n{chunk['text']}"
                    
                    detail_chunks.append({
                        'text': detail_text,
                        'metadata': metadata
                    })
        
        return detail_chunks
"""


# ============================================================================
# STEP 4: TEST GRAPH-ENABLED SEARCH
# ============================================================================

"""
Create a test script (test_graph_search.py):

import json
from search_agent import SearchAgent

# Initialize search agent
search = SearchAgent()

# Test queries
test_queries = [
    "CIDH pile detail",
    "pipe pin connection",
    "bent cap reinforcement",
    "girder section",
    "abutment details"
]

print("Testing Graph-Enabled Search\\n" + "="*60)

for query in test_queries:
    print(f"\\nQuery: {query}")
    
    # Option 1: Standard search
    results = search.search(query, k=5, use_graph=False)
    print(f"  Standard results: {results['count']} hits")
    if results['results']:
        print(f"    - Top: {results['results'][0].get('pdf_file_name', 'unknown')}")
    
    # Option 2: Graph-enabled search
    results_with_graph = search.search(query, k=5, use_graph=True)
    print(f"  Graph-enabled results: {results_with_graph['count']} hits")
    if results_with_graph['results']:
        print(f"    - Top: {results_with_graph['results'][0].get('pdf_file_name', 'unknown')}")
        if 'detail_id' in results_with_graph['results'][0].get('metadata', {}):
            print(f"    - Detail: {results_with_graph['results'][0]['metadata']['detail_title']}")
"""


# ============================================================================
# STEP 5: EXPECTED IMPROVEMENTS
# ============================================================================

"""
Before Graph Integration:
Query: "pipe pin connection detail"
Results:
  - Page 21 (entire page with multiple details)
  - Returns "Pipe Pin Detail" but also irrelevant details on same page
  - User has to manually find the relevant callout

After Graph Integration:
Query: "pipe pin connection detail"
Results:
  - Pipe Pin Detail callout (specific detail, not entire page)
  - Connected details: Reinforcement Detail, Connection Assembly
  - User gets exactly what they need + related details

Key Benefits:
✓ Detail-level granularity (not page-level)
✓ Related details automatically surfaced
✓ No irrelevant details from same page
✓ Better semantic grounding for LLM
✓ Faster, more focused answers
"""


# ============================================================================
# GRAPH QUERY EXAMPLES
# ============================================================================

"""
Once graphs are built and loaded:

from enriched_metadata_graph_builder import GraphIntegrationManager

manager = GraphIntegrationManager()
graph = manager.get_graph("Mar Vista POC 100%_CheckPrint_20211118 Complete")

# Find all details matching a keyword
cidh_details = [
    (did, node) for did, node in graph.nodes.items()
    if "cidh" in node.title.lower()
]
print(f"Found {len(cidh_details)} CIDH-related details")

# Find details connected to a specific detail
if cidh_details:
    detail_id = cidh_details[0][0]
    connected = graph.get_connected_components(detail_id, max_depth=2)
    print(f"Connected to {detail_id}: {len(connected)} details")
    
    # Get relationship info
    related = graph.get_related_nodes(detail_id)
    for related_node in related:
        print(f"  → {related_node.title} ({related_node.element_type})")
"""


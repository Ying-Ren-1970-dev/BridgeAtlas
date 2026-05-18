#!/usr/bin/env python3
"""
Test script to verify graph-search integration.
Tests that GraphIntegrationManager is properly integrated into SearchAgent.
"""

import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from search_agent import SearchAgent
    print("✓ SearchAgent imported successfully")
except Exception as e:
    print(f"✗ Failed to import SearchAgent: {e}")
    sys.exit(1)

try:
    from enriched_metadata_graph_builder import GraphIntegrationManager
    print("✓ GraphIntegrationManager imported successfully")
except Exception as e:
    print(f"✗ Failed to import GraphIntegrationManager: {e}")
    sys.exit(1)

# Test GraphIntegrationManager instantiation
print("\nTesting GraphIntegrationManager instantiation...")
try:
    graph_manager = GraphIntegrationManager()
    print("✓ GraphIntegrationManager instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate GraphIntegrationManager: {e}")
    sys.exit(1)

# Test find_related_documents method exists and is callable
print("\nTesting find_related_documents method...")
try:
    if hasattr(graph_manager, 'find_related_documents'):
        print("✓ find_related_documents method exists")
        # Test calling it with empty graphs (should return empty list)
        result = graph_manager.find_related_documents("test query", max_hops=2, limit=20)
        print(f"✓ find_related_documents is callable, returned: {type(result)} with {len(result)} items")
    else:
        print("✗ find_related_documents method does not exist")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error calling find_related_documents: {e}")
    sys.exit(1)

# Test SearchAgent initialization with graph integration
print("\nTesting SearchAgent initialization...")
try:
    search_agent = SearchAgent()
    print("✓ SearchAgent instantiated successfully")
except Exception as e:
    print(f"✗ Failed to instantiate SearchAgent: {e}")
    sys.exit(1)

# Test that SearchAgent has graph_manager
print("\nTesting SearchAgent graph integration...")
try:
    if hasattr(search_agent, 'graph_manager'):
        print("✓ SearchAgent has graph_manager attribute")
    else:
        print("✗ SearchAgent does not have graph_manager attribute")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking graph_manager: {e}")
    sys.exit(1)

# Test that SearchAgent has the new methods
print("\nTesting SearchAgent new methods...")
try:
    if hasattr(search_agent, '_query_graph_for_related_docs'):
        print("✓ SearchAgent has _query_graph_for_related_docs method")
    else:
        print("✗ SearchAgent does not have _query_graph_for_related_docs method")
        sys.exit(1)
    
    if hasattr(search_agent, '_augment_search_with_graph_context'):
        print("✓ SearchAgent has _augment_search_with_graph_context method")
    else:
        print("✗ SearchAgent does not have _augment_search_with_graph_context method")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking methods: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All integration tests passed!")
print("="*60)
print("\nThe graph integration is ready for use in search operations.")

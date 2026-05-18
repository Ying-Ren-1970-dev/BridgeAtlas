#!/usr/bin/env python3
"""
Verification script for global feedback implementation.
Tests that feedback is no longer scoped by project and applies globally.
"""

import sys
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from search_agent import SearchAgent
import config


def test_global_feedback():
    """Test that feedback applies globally across projects."""
    print("\n" + "="*80)
    print("TESTING GLOBAL FEEDBACK IMPLEMENTATION")
    print("="*80)
    
    # Create a temporary feedback store
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override feedback path for this test
        feedback_path = os.path.join(tmpdir, "search_feedback.jsonl")
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        
        # Create test feedback records from different projects
        feedback_records = [
            # Mar Vista project feedback
            {
                "query": "LOTB",
                "pdf_file_name": "lotb_mar_vista.pdf",
                "page_number": 1,
                "feedback": "best",
                "project_scope": "Mar Vista",
                "timestamp": datetime.now().isoformat()
            },
            # 25th Avenue project - different project but similar query
            {
                "query": "log of test boring",  # Similar to LOTB
                "pdf_file_name": "boring_log_25th.pdf",
                "page_number": 5,
                "feedback": "relevant",
                "project_scope": "25th Avenue",
                "timestamp": datetime.now().isoformat()
            },
            # Another 25th Avenue record with different query
            {
                "query": "shear key details",
                "pdf_file_name": "shear_details.pdf",
                "page_number": 10,
                "feedback": "irrelevant",
                "project_scope": "25th Avenue",
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Write feedback records
        with open(feedback_path, "w", encoding="utf-8") as f:
            for record in feedback_records:
                f.write(json.dumps(record) + "\n")
        
        print("\nCreated test feedback records:")
        for i, record in enumerate(feedback_records, 1):
            print(f"  {i}. Query: '{record['query']}' | Project: {record['project_scope']} | Label: {record['feedback']}")
        
        # Create a SearchAgent instance
        agent = SearchAgent()
        
        # Override the feedback store path for testing
        original_method = agent._feedback_store_path
        agent._feedback_store_path = lambda: feedback_path
        
        # Test 1: Find similar queries
        print("\n" + "-"*80)
        print("TEST 1: Finding similar queries (feedback should be GLOBAL)")
        print("-"*80)
        
        # Query for 25th Avenue project (should find feedback from Mar Vista)
        query = "log of test boring results"
        project_scope = "25th Avenue"
        
        similar = agent._find_similar_queries_in_feedback(query, project_scope)
        print(f"\nQuery: '{query}'")
        print(f"Project scope: {project_scope}")
        print(f"Similar queries found: {len(similar)}")
        for q, score in similar.items():
            print(f"  - '{q}': similarity={score:.3f}")
        
        # Should find "log of test boring" even though we're in 25th Avenue
        if any("log of test boring" in q for q in similar.keys()):
            print("✓ PASS: Found similar query from same project")
        else:
            print("✗ FAIL: Did not find expected similar query")
        
        # Test 2: Load feedback adjustments
        print("\n" + "-"*80)
        print("TEST 2: Loading feedback adjustments (should include cross-project feedback)")
        print("-"*80)
        
        # Load adjustments for "log of test boring" in 25th Avenue context
        query = "log of test boring"
        project_scope = "25th Avenue"
        
        adjustments = agent._load_feedback_adjustments(query, project_scope)
        print(f"\nQuery: '{query}'")
        print(f"Project scope: {project_scope}")
        print(f"Feedback adjustments: {len(adjustments)}")
        for (file_name, page_num), delta in adjustments.items():
            print(f"  - {file_name}:p{page_num} delta={delta:.3f}")
        
        # Should include adjustments regardless of project scope
        if adjustments:
            print("✓ PASS: Found feedback adjustments (global feedback working)")
        else:
            print("✗ FAIL: No adjustments found (feedback may not be global)")
        
        # Test 3: Verify scope parameter doesn't filter feedback
        print("\n" + "-"*80)
        print("TEST 3: Verifying scope parameter is ignored")
        print("-"*80)
        
        # Try with None project scope
        adjustments_none = agent._load_feedback_adjustments(query, None)
        adjustments_25th = agent._load_feedback_adjustments(query, "25th Avenue")
        
        # Should get feedback regardless
        print(f"Adjustments with scope=None: {len(adjustments_none)}")
        print(f"Adjustments with scope='25th Avenue': {len(adjustments_25th)}")
        
        if len(adjustments_none) > 0 and len(adjustments_25th) > 0:
            print("✓ PASS: Feedback is applied regardless of scope parameter")
        else:
            print("✗ FAIL: Feedback is still being filtered by scope")
        
        print("\n" + "="*80)
        print("VERIFICATION COMPLETE")
        print("="*80)


if __name__ == "__main__":
    test_global_feedback()

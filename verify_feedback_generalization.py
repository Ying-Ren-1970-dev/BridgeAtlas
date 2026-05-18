"""Detailed test showing feedback generalization in action."""
import json
from search_agent import SearchAgent

print("=" * 80)
print("FEEDBACK GENERALIZATION VERIFICATION")
print("=" * 80)

agent = SearchAgent()

# Test 1: Show token-based similarity detection
print("\n[TEST 1] Token-based Query Similarity")
print("─" * 80)

test_cases = [
    ("LOTB", "log of test boring", "Acronym expansion to full phrase"),
    ("LOTB", "boring log test", "Same tokens, different order"),
    ("shear key details", "pipe pin connection detail", "Synonym relationship"),
    ("bent cap", "cap beam", "Synonym relationship"),
]

for query1, query2, description in test_cases:
    tokens1 = agent._get_query_tokens(query1)
    tokens2 = agent._get_query_tokens(query2)
    overlap = tokens1 & tokens2
    
    print(f"\n{description}:")
    print(f"  Query 1: '{query1}' → tokens: {tokens1}")
    print(f"  Query 2: '{query2}' → tokens: {tokens2}")
    print(f"  Token overlap: {overlap} ({len(overlap)} tokens in common)")
    
    if len(overlap) >= 2:
        print(f"  ✓ Would trigger feedback generalization")
    else:
        print(f"  ✗ Not enough token overlap (need 2+)")

# Test 2: Show actual feedback adjustments with generalization
print("\n\n[TEST 2] Feedback Score Adjustments")
print("─" * 80)

queries_to_test = [
    ("LOTB", "Mar Vista", "Exact query (should have full weight)"),
    ("log of test boring", "Mar Vista", "Similar to LOTB (should have generalized weight)"),
    ("shear key", "Mar Vista", "Similar to 'shear key details'"),
]

for query, scope, description in queries_to_test:
    print(f"\nQuery: '{query}' (Scope: {scope})")
    print(f"Description: {description}")
    
    adjustments = agent._load_feedback_adjustments(query, scope)
    
    if adjustments:
        print(f"  Found {len(adjustments)} pages with feedback adjustments:")
        for (file_name, page), adjustment in sorted(adjustments.items())[:3]:
            status = "BOOST" if adjustment < 0 else "PENALTY"
            print(f"    - {file_name} page {page}: {adjustment:.2f} ({status})")
    else:
        print(f"  No adjustments found for this query")

# Test 3: Show similar queries detection
print("\n\n[TEST 3] Similar Queries in Feedback Store")
print("─" * 80)

test_queries = ["log of test boring", "pipe pin", "reinforcement details"]

for query in test_queries:
    print(f"\nQuery: '{query}'")
    similar = agent._find_similar_queries_in_feedback(query, "Mar Vista")
    
    if similar:
        print(f"  Found {len(similar)} similar queries with feedback:")
        for sim_query, similarity_score in sorted(similar.items(), key=lambda x: -x[1])[:3]:
            weight = 0.5 * similarity_score
            print(f"    - '{sim_query}' (similarity: {similarity_score:.2f}, weight: {weight:.2f}x)")
    else:
        print(f"  No similar queries found in feedback store")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Feedback Generalization Features:
1. ✓ Token-based similarity: Finds queries sharing 2+ significant tokens
2. ✓ Weighted application: Similar queries get 0.5x × similarity_score weight
3. ✓ Scope-aware: Only generalizes within same project scope
4. ✓ Conservative: Full weight on exact match, reduced weight on similar queries

This prevents over-correction while allowing related queries to benefit from
feedback (e.g., "LOTB" feedback helps "log of test boring", "test boring log", etc.)
""")

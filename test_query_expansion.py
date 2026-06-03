"""
Test query expansion with engineering terminology.
"""
from engineering_terminology import expand_query, get_all_synonyms, get_primary_term


def test_query_expansion():
    """Test query expansion functionality."""
    
    print("=" * 80)
    print("TESTING QUERY EXPANSION")
    print("=" * 80)
    
    # Test cases from failed tests
    test_queries = [
        "drilled shaft",
        "CIDH pile",
        "steel shear key",
        "pipe pin",
        "rebar",
        "#4 bars",
        "cast-in-drilled-hole pile",
        "bearing pad",
        "abutment",
        "bent",
        "reinforcing bars",
        "tamper proof screws",
        "chain link fence",
        "R/C box girder",
    ]
    
    for query in test_queries:
        expanded = expand_query(query, max_expansions=3)
        primary = get_primary_term(query)
        
        print(f"\n🔍 Query: '{query}'")
        print(f"   Primary term: {primary}")
        print(f"   Expanded to ({len(expanded)} terms):")
        for i, term in enumerate(expanded, 1):
            marker = "✓" if term == query else "+"
            print(f"      {marker} {term}")
    
    print("\n" + "=" * 80)
    print("TESTING SYNONYM LOOKUP")
    print("=" * 80)
    
    # Test synonym lookup
    test_terms = ["CIDH pile", "drilled shaft", "pipe pin"]
    
    for term in test_terms:
        synonyms = get_all_synonyms(term)
        print(f"\n📚 All synonyms for '{term}':")
        for syn in synonyms:
            print(f"   • {syn}")
    
    print("\n" + "=" * 80)
    print("✓ Query expansion testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_query_expansion()

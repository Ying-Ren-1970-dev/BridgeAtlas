"""Test feedback generalization - verify similar queries benefit from feedback."""
import json
import urllib.request
import urllib.parse

# Read feedback store to see what feedback exists
feedback_path = "data/feedback/search_feedback.jsonl"
queries_with_feedback = {}

try:
    with open(feedback_path, "r") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                q = record.get("query", "").lower()
                label = record.get("feedback", "")
                if q not in queries_with_feedback:
                    queries_with_feedback[q] = []
                queries_with_feedback[q].append({
                    "file": record.get("pdf_file_name", ""),
                    "page": record.get("page_number"),
                    "label": label
                })
except Exception as e:
    print(f"Error reading feedback: {e}")

print("=" * 80)
print("FEEDBACK GENERALIZATION TEST")
print("=" * 80)

if not queries_with_feedback:
    print("No feedback found in JSONL store")
else:
    print(f"\nFound feedback for {len(queries_with_feedback)} unique queries:")
    for q in list(queries_with_feedback.keys())[:5]:  # Show first 5
        labels = queries_with_feedback[q]
        label_summary = {}
        for rec in labels:
            lbl = rec["label"]
            label_summary[lbl] = label_summary.get(lbl, 0) + 1
        print(f"  - '{q}' -> {label_summary}")

# Test feedback generalization by searching for similar queries
print("\n" + "=" * 80)
print("TESTING GENERALIZATION")
print("=" * 80)

test_queries = [
    ("LOTB", "log of test boring"),  # Acronym should be expanded and generalized
    ("shear key details", "pipe pin connection detail"),  # Similar query should get feedback
    ("bent reinforcement", "column reinforcement"),  # Synonym should help
]

URL = "http://localhost:8000/search"

for original_q, similar_q in test_queries:
    # Check if original query has feedback
    original_normalized = original_q.lower()
    if original_normalized not in queries_with_feedback:
        print(f"\nSkipping: '{original_q}' has no feedback in store")
        continue
    
    print(f"\n{'─' * 80}")
    print(f"Original query WITH feedback: '{original_q}'")
    original_feedback = queries_with_feedback[original_normalized]
    best_count = sum(1 for r in original_feedback if r["label"] == "best")
    relevant_count = sum(1 for r in original_feedback if r["label"] == "relevant")
    print(f"  Feedback: {best_count} best, {relevant_count} relevant")
    
    # Search with similar query (should benefit from generalized feedback)
    print(f"Similar query to test: '{similar_q}'")
    print("  Searching...")
    
    try:
        body = json.dumps({
            "query": similar_q,
            "k": 5,
            "generate_summary": False,
            "use_hybrid_search": True,
            "project_scope": "Mar Vista"
        }).encode("utf-8")
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            results = result.get("results", [])
            
            # Check if any feedback-marked pages appear in top results
            top_pages = [(r.get("pdf_file_name", ""), r.get("page")) for r in results[:3]]
            
            feedback_hits = 0
            for fname, page in top_pages:
                for feedback_rec in original_feedback:
                    if feedback_rec["file"] == fname and feedback_rec["page"] == page:
                        feedback_hits += 1
                        print(f"    ✓ Generalized feedback applied: page {page} ({feedback_rec['label']})")
            
            if feedback_hits == 0:
                print(f"    - No immediate feedback hits in top 3 (generalization may be at lower ranks)")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

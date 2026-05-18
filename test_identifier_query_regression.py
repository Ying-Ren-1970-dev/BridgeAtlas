"""Regression test for identifier queries (project numbers) in local API search."""

import sys
import requests

API_URL = "http://localhost:8000/search"


def run_identifier_regression() -> int:
    """
    Validate that project-number lookup finds Mar Vista in top results.

    This locks in the fix for queries like "project number 05-1C720" where
    generic "project number" text in unrelated docs previously outranked the
    exact identifier match.
    """
    payload = {
        "query": "project number 05-1C720",
        "k": 10,
        "generate_summary": False,
        "use_hybrid_search": True,
        "relevance_threshold": 2.0,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=45)
        response.raise_for_status()
    except Exception as exc:
        print(f"FAIL: API request failed: {exc}")
        return 1

    data = response.json()
    results = data.get("results", [])

    if not results:
        print("FAIL: API returned zero results for identifier query")
        return 1

    mar_vista_rank = None
    for idx, result in enumerate(results, start=1):
        file_name = str(result.get("pdf_file_name", ""))
        project_name = str(result.get("project_name", ""))
        haystack = (file_name + " " + project_name).lower()
        if "mar vista" in haystack:
            mar_vista_rank = idx
            break

    if mar_vista_rank is None:
        top_preview = [
            (r.get("project_name", "?"), r.get("pdf_file_name", "?"))
            for r in results[:3]
        ]
        print("FAIL: Mar Vista not found in returned results")
        print(f"Top results: {top_preview}")
        return 1

    if mar_vista_rank != 1:
        print(f"FAIL: Mar Vista expected at rank 1, got rank {mar_vista_rank}")
        return 1

    print("PASS: Identifier query regression test")
    print("- Query: project number 05-1C720")
    print("- Mar Vista rank: 1")
    print(f"- Total results: {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(run_identifier_regression())

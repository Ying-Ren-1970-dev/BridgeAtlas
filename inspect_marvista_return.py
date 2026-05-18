import json
import urllib.request

queries = [
    "pipe pin connection detail",
    "bent cap details",
    "CIDH piles diameter and length",
    "shear key dimensions",
    "wing wall reinforcement",
    "GENERAL PLAN No. 1",
    "project number 05-1C720",
]

for query in queries:
    body = json.dumps({
        "query": query,
        "k": 10,
        "generate_summary": False,
        "use_hybrid_search": True,
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    print(f"\nQ: {query}")
    for i, result in enumerate(data.get("results", [])[:5], 1):
        file_name = result.get("pdf_file_name", "")
        page = result.get("page_number")
        score = result.get("relevance_score")
        print(f" {i}. {file_name[:60]:60} p{page} s{score}")

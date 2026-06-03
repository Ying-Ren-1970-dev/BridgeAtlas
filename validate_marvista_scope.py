import json
import urllib.request

queries = [
    "GENERAL PLAN No. 1",
    "project number 05-1C720",
    "pipe pin connection detail",
    "bent cap details",
    "CIDH piles diameter and length",
]


def run_search(query: str, scoped: bool):
    body = {
        "query": query,
        "k": 10,
        "generate_summary": False,
        "use_hybrid_search": True,
    }
    if scoped:
        body["project_scope"] = "mar vista"

    req = urllib.request.Request(
        "http://localhost:8000/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


for query in queries:
    for scoped in [False, True]:
        data = run_search(query, scoped)
        first = data.get("results", [{}])[0] if data.get("results") else {}
        print(
            "Q={} | scoped={} | top={} p{} s{} | total={}".format(
                query,
                scoped,
                first.get("pdf_file_name"),
                first.get("page_number"),
                first.get("relevance_score"),
                data.get("total_results"),
            )
        )
    print("-" * 120)

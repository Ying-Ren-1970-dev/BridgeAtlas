import json
import urllib.request

queries = [
    "concrete box girder",
    "r/c box girder",
    "abutment footing reinforcement detail",
    "general notes",
    "rebar detail abutment",
]

for query in queries:
    body = {
        "query": query,
        "k": 10,
        "project_scope": "mar vista",
        "use_hybrid_search": True,
        "generate_summary": False,
    }
    req = urllib.request.Request(
        "http://localhost:8000/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=40) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    print(f"\nQ: {query}  total={data.get('total_results')}")
    for i, result in enumerate(data.get("results", [])[:5], 1):
        sample = (result.get("content_sample") or "").replace("\n", " ")[:120]
        print(
            " {}. p{} s{} | {}".format(
                i,
                result.get("page_number"),
                result.get("relevance_score"),
                sample,
            )
        )

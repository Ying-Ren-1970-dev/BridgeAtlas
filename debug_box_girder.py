import json
import urllib.request

query = "concrete box girder"

for scoped in [False, True]:
    body = {
        "query": query,
        "k": 20,
        "use_hybrid_search": True,
        "generate_summary": False,
    }
    if scoped:
        body["project_scope"] = "mar vista"

    req = urllib.request.Request(
        "http://localhost:8000/search",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    print("\n=== scoped={} total={} ===".format(scoped, data.get("total_results")))
    for i, r in enumerate(data.get("results", [])[:10], 1):
        sample = (r.get("content_sample") or "").replace("\n", " ")[:160]
        print("{}: {} p{} s{} | {}".format(i, r.get("pdf_file_name"), r.get("page_number"), r.get("relevance_score"), sample))

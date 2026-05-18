import urllib.request, json

queries = [
    "CIDH piles diameter and length",
    "pipe pin connection detail",
    "shear key dimensions",
    "bearing pad specifications",
    "abutment footing reinforcement",
    "bent cap details",
    "pile cap reinforcement",
    "seismic design criteria",
    "expansion joint detail",
    "wing wall reinforcement",
]

def search(q, hybrid):
    body = json.dumps({"query": q, "k": 5, "generate_summary": False, "use_hybrid_search": hybrid}).encode()
    req = urllib.request.Request("http://localhost:8000/search", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def fmt_results(data):
    mv = [r for r in data["results"] if "Mar Vista" in r.get("pdf_file_name", "")]
    if not mv:
        return "(no MV)"
    return " ".join("p{}({})".format(r["page_number"], round(r["relevance_score"], 2)) for r in mv[:3])

print("{:<40} {:<28} {:<28}".format("Query", "HYBRID (pages/score)", "VECTOR-ONLY (pages/score)"))
print("-" * 96)
for q in queries:
    rh = search(q, True)
    rv = search(q, False)
    h = fmt_results(rh)
    v = fmt_results(rv)
    marker = " <-- DIFF" if h != v else ""
    print("{:<40} {:<28} {:<28}{}".format(q[:39], h, v, marker))

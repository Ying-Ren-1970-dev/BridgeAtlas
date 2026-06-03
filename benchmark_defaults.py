import json
import urllib.request

URL = "http://localhost:8000/search"
queries = [
    "Bent 1 location",
    "Abutment 17",
    "R/C box girder",
    "GENERAL PLAN No. 1",
    "plan scale 1\" = 30'",
    "project number 05-1C720",
    "ERIC FREDRICKSON",
    "MAR VISTA DRIVE POC",
    "pipe pin connection detail",
    "bent cap details",
    "concrete cover requirements",
    "moment slab thickness",
    "CIDH piles diameter and length",
    "shear key dimensions",
    "wing wall reinforcement",
]


def call_search(q, hybrid):
    body = json.dumps({
        "query": q,
        "k": 10,
        "generate_summary": False,
        "use_hybrid_search": hybrid,
    }).encode("utf-8")
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def first_mv_rank(results):
    for i, r in enumerate(results, 1):
        combined = (str(r.get("pdf_file_name", "")) + " " + str(r.get("project_name", ""))).lower()
        if "mar vista" in combined:
            return i
    return None


for hybrid, label in [(True, "Hybrid default threshold"), (False, "Vector default threshold")]:
    ranks = []
    for q in queries:
        rr = first_mv_rank(call_search(q, hybrid).get("results", []))
        if rr is not None:
            ranks.append(rr)

    total = len(queries)
    hit = len(ranks)
    avg = round(sum(ranks) / hit, 2) if hit else None
    mrr = round(sum(1.0 / r for r in ranks) / total, 3) if total else 0.0
    print(f"{label}: hit@10={hit}/{total}, avg_first_rank={avg}, MRR={mrr}")

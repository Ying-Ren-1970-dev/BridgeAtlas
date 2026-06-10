"""Compare local vs cloud search results for the same queries."""
import json
import requests

LOCAL = "http://localhost:8000"
CLOUD = "https://librarian-716133050580.us-central1.run.app"
QUERIES = ["CIDH pile", "abutment", "ARS Curve", "CIP Box Girder"]
PAYLOAD = {"k": 50, "use_hybrid_search": True}


def result_key(item):
    return (
        item.get("pdf_file_name"),
        item.get("page_number"),
        round(float(item.get("relevance_score") or 0), 3),
    )


def main():
    for query in QUERIES:
        body = {**PAYLOAD, "query": query}
        local = requests.post(f"{LOCAL}/search", json=body, timeout=120).json()
        cloud = requests.post(f"{CLOUD}/search", json=body, timeout=120).json()

        local_set = {result_key(r) for r in local.get("results", [])}
        cloud_set = {result_key(r) for r in cloud.get("results", [])}

        print(f"=== {query} ===")
        print(f"local total={local.get('total_results')} cloud total={cloud.get('total_results')}")
        print(f"only local ({len(local_set - cloud_set)}): {sorted(local_set - cloud_set)[:8]}")
        print(f"only cloud ({len(cloud_set - local_set)}): {sorted(cloud_set - local_set)[:8]}")
        print()


if __name__ == "__main__":
    main()

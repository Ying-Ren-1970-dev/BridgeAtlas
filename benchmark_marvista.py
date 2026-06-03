import json
import urllib.request

URL = 'http://localhost:8000/search'

queries = [
    'Bent 1 location',
    'Abutment 17',
    'R/C box girder',
    'GENERAL PLAN No. 1',
    'plan scale 1" = 30\'',
    'project number 05-1C720',
    'ERIC FREDRICKSON',
    'MAR VISTA DRIVE POC',
    'pipe pin connection detail',
    'bent cap details',
    'concrete cover requirements',
    'moment slab thickness',
    'CIDH piles diameter and length',
    'shear key dimensions',
    'wing wall reinforcement',
]


def call_search(q, hybrid, threshold):
    body = json.dumps({
        'query': q,
        'k': 10,
        'generate_summary': False,
        'use_hybrid_search': hybrid,
        'relevance_threshold': threshold,
    }).encode('utf-8')
    req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode('utf-8'))


def first_mv_rank(results):
    for i, r in enumerate(results, 1):
        name = str(r.get('pdf_file_name', ''))
        proj = str(r.get('project_name', ''))
        if 'mar vista' in (name + ' ' + proj).lower():
            return i
    return None


def run(threshold):
    rows = []
    for q in queries:
        h = call_search(q, True, threshold)
        v = call_search(q, False, threshold)
        hr = first_mv_rank(h.get('results', []))
        vr = first_mv_rank(v.get('results', []))
        rows.append((q, hr, vr))

    def metrics(idx):
        ranks = [r[idx] for r in rows if r[idx] is not None]
        hit = len(ranks)
        total = len(rows)
        avg_rank = round(sum(ranks) / len(ranks), 2) if ranks else None
        mrr = round(sum(1.0 / r for r in ranks) / total, 3) if total else 0.0
        return hit, total, avg_rank, mrr

    h_hit, total, h_avg, h_mrr = metrics(1)
    v_hit, _, v_avg, v_mrr = metrics(2)

    print(f'\nTHRESHOLD={threshold}')
    print(f'Hybrid:      hit@10={h_hit}/{total}, avg_first_rank={h_avg}, MRR={h_mrr}')
    print(f'Vector-only: hit@10={v_hit}/{total}, avg_first_rank={v_avg}, MRR={v_mrr}')

run(0.5)
run(2.0)

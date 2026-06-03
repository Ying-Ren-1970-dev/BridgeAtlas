import json
import urllib.request

URL = 'http://localhost:8000/search'

queries = [
    'Bent 1 location',
    'Abutment 17',
    'R/C box girder',
    'GENERAL PLAN No. 1',
    "plan scale 1 inch equals 30 feet",
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
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def first_mv_rank(results):
    for i, r in enumerate(results, 1):
        name = str(r.get('pdf_file_name', ''))
        proj = str(r.get('project_name', ''))
        if 'mar vista' in (name + ' ' + proj).lower():
            return i
    return None


print('--- MISS ANALYSIS (hybrid, threshold=2.0) ---')
missed = []
for q in queries:
    try:
        res = call_search(q, True, 2.0)
        rank = first_mv_rank(res.get('results', []))
        if rank is None:
            missed.append(q)
            top = [(r.get('project_name', '?'), r.get('pdf_file_name', '?')) for r in res.get('results', [])[:3]]
            print('MISS: ' + repr(q))
            for j, (pn, fn) in enumerate(top, 1):
                print('  #' + str(j) + ' ' + pn + ' / ' + fn)
            print()
        else:
            print('HIT rank=' + str(rank) + ': ' + repr(q))
    except Exception as e:
        print('ERROR for ' + repr(q) + ': ' + str(e))

if not missed:
    print('\nAll 15/15 hit!')
else:
    print('\nTotal misses: ' + str(len(missed)) + '/' + str(len(queries)))

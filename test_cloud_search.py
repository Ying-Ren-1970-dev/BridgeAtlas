import urllib.request
import json

# Test search endpoint on cloud
url = 'https://librarian-czwgfoksfa-uc.a.run.app/search'
body = json.dumps({'query': 'CIDH pile detail', 'k': 5}).encode()
req = urllib.request.Request(url, data=body, headers={'Content-Type': 'application/json'})

try:
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
        print('Search Results:', len(data.get('results', [])))
        if data.get('results'):
            print('First result:', data['results'][0].get('pdf_file_name'))
        else:
            print('NO RESULTS - Full response:')
            print(json.dumps(data, indent=2))
except Exception as e:
    print('Error:', type(e).__name__)
    print('Details:', str(e)[:300])

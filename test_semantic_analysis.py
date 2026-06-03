#!/usr/bin/env python
"""Test the semantic analysis engine."""
import urllib.request
import json

URL = 'http://localhost:8000/analyze'
body = json.dumps({
    'query': 'CIDH pile detail',
    'k': 10,
    'use_hybrid_search': True,
}).encode('utf-8')

req = urllib.request.Request(URL, data=body, headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(req, timeout=30) as resp:
    result = json.loads(resp.read().decode('utf-8'))
    print('Query:', result['query'])
    print('\nInferred Intent:')
    intent = result['inferred_intent']
    print(f"  Intent types: {intent['intent_types']}")
    print(f"  Components: {intent['structural_components']}")
    print(f'\nIntent Confidence: {result["intent_confidence"]}')
    print(f'\nAnswer:\n{result["answer"]}')
    
    if result['semantic_evidence']:
        print('\n' + '='*60)
        print('Top 3 Results:')
        for i, ev in enumerate(result['semantic_evidence'][:3], 1):
            print(f'\n{i}. {ev["pdf_file_name"]} (Project: {ev["project_name"]}, Page {ev["page_number"]})')
            print(f'   Relevance: {ev["relevance_score"]:.3f} | Semantic: {ev["semantic_match_score"]:.3f}')
            print(f'   Components: {", ".join(ev["component_matches"]) if ev["component_matches"] else "None"}')
            print(f'   Why matched: {ev["why_matched"]}')

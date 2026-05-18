from search_agent import SearchAgent
import json

sa = SearchAgent()
r = sa.search(query='cip box girder reinforcement detail', k=30, generate_summary=False, use_hybrid_search=True)

results = r.get('results', [])
print('Total projects:', r.get('total_projects'))
for proj in results:
    print(f"  {str(proj.get('pdf_file_name',''))[:60]}")
    print(f"    relevant_pages={proj.get('relevant_pages')}")
    print(f"    project_name={proj.get('project_name','')[:40]}")
from search_agent import SearchAgent

sa = SearchAgent()
r = sa.search(query='cip box girder reinforcement detail', k=30, generate_summary=False, use_hybrid_search=True)

results = r.get('results', [])
print('Total projects:', r.get('total_projects'))
for proj in results:
    print(f"  {str(proj.get('pdf_file_name') or proj.get('project_name',''))[:60]}")
    print(f"    relevant_pages={proj.get('relevant_pages')}")

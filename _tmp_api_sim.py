from search_agent import SearchAgent

agent = SearchAgent()
out = agent.search("abutment pile detail", k=50)
threshold = 0.15

all_pages = []
for proj in out.get("results", []):
    fn = proj.get("file_name", "")
    for ch in proj.get("chunks", []):
        all_pages.append((ch.get("score", 9), fn, ch.get("page"), ch.get("plan_sheet_title", "")))

all_pages.sort()
best = all_pages[0][0] if all_pages else 9
cutoff = best + threshold
print(f"best={best:.4f} cutoff={cutoff:.4f}")
print("--- all pages (sorted) ---")
for score, fn, page, title in all_pages[:30]:
    flag = "HIGH" if score <= cutoff else "low"
    print(f"  {flag} {score:.4f}  {fn} p{page}  {str(title)[:45]}")

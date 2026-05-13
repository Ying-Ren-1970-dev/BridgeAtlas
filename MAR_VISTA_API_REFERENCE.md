# Mar Vista API - Quick Reference

## Summary of Changes

✅ **API now filters to Mar Vista project only**
- Only searches: "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
- Returns page numbers for all matching results
- Includes topology elements and relevance scores

## API Endpoint

**POST** `http://localhost:8000/search`

### Request
```json
{
  "query": "pipe pin",
  "k": 20
}
```

### Response
```json
{
  "query": "pipe pin",
  "total_results": 7,
  "projects_found": 1,
  "results": [
    {
      "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
      "project_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete",
      "page_number": 5,
      "relevance_score": 1.112,
      "topology_elements": ["Pipe Pin (Steel Shear Key)", "Bent"],
      "content_sample": "Material Specifications: Pipe: Standard and galvanized...",
      "has_vision_analysis": false
    },
    ...
  ],
  "project_summaries": [
    {
      "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
      "project_name": "Mar Vista POC",
      "relevant_pages": [5, 10, 11, 18, 19, 21, 27],
      "total_pages": 42,
      "categories": ["Bridges", "Retaining Walls", "General Structural"]
    }
  ],
  "search_summary": "AI-generated summary...",
  "timestamp": "2026-05-10T22:00:00Z"
}
```

## Example Searches & Results

### 🔍 "steel shear key"
**Pages Found:** 18, 19, 30
**Topology:** Pipe Pin (Steel Shear Key), Bearing Pad, Abutment

### 🔍 "pipe pin"
**Pages Found:** 5, 10, 11, 18, 19, 21, 27
**Topology:** Pipe Pin (Steel Shear Key), Bent

### 🔍 "CIDH pile"
**Pages Found:** 6, 8, 9, 16, 21
**Topology:** CIDH Piles/Shafts, Foundation, Reinforcement
**Vision Analysis:** ✓ Yes

### 🔍 "bearing pad"
**Pages Found:** 10, 11, 18, 30, 34
**Topology:** Pipe Pin (Steel Shear Key), Bearing Pad, Abutment

### 🔍 "abutment details"
**Pages Found:** 6, 10, 11, 13, 18, 39
**Topology:** Bearing Pad, Abutment, Bent

### 🔍 "pipe pin steel shear key" (comprehensive)
**Pages Found:** 5, 9, 11, 16, 18, 19, 21, 25, 27, 30, 39, 40
**Total:** 12 pages with pipe pin or steel shear key content

## Usage Examples

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "pipe pin", "k": 20}
)

data = response.json()

# Get all page numbers
pages = [r['page_number'] for r in data['results']]
print(f"Found on pages: {sorted(set(pages))}")

# Get topology elements
for result in data['results']:
    print(f"Page {result['page_number']}: {', '.join(result['topology_elements'])}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "pipe pin", "k": 20}'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'pipe pin', k: 20 })
});

const data = await response.json();
const pages = data.results.map(r => r.page_number);
console.log('Found on pages:', [...new Set(pages)].sort());
```

## Response Fields

| Field | Description |
|-------|-------------|
| `pdf_file_name` | Always "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf" |
| `project_name` | Human-readable project name |
| `page_number` | **Page number where content was found** (1-42) |
| `relevance_score` | Relevance score (lower = better match) |
| `topology_elements` | List of structural elements identified |
| `content_sample` | First 500 chars of relevant content |
| `has_vision_analysis` | Whether GPT-4 Vision analyzed the page |

## Topology Elements Recognized

The API automatically identifies 12 structural element types:

1. **Pipe Pin (Steel Shear Key)** - Steel shear keys using pipe pins
2. **Shear Key** - General shear keys
3. **CIDH Piles/Shafts** - Cast-In-Drilled-Hole foundations
4. **Bearing Pad** - Load distribution pads
5. **Abutment** - Bridge end supports
6. **Bent** - Bridge support piers
7. **End Diaphragm** - Structural closure elements
8. **Retaining Wall** - Earth retention systems
9. **Foundation** - Base structural support
10. **Reinforcement** - Rebar, steel reinforcement
11. **Bridge Girder** - Horizontal spanning members
12. **Column** - Vertical load-bearing members

## Start the Server

```powershell
cd C:\Users\alexl\Desktop\xttribute\Librarian
python api.py
```

Server runs on: http://localhost:8000
Documentation: http://localhost:8000/docs

## Test the API

```powershell
python test_mar_vista_api.py
```

This runs comprehensive tests on common search terms and shows page numbers for each.

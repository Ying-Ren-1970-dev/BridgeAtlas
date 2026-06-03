# Multi-Project Feedback Generalization

## Overview

The Librarian system now supports feedback generalization across **all projects**. When you provide feedback on one query in a project, similar queries automatically benefit from that feedback—even as you add more projects to the system.

**Key Features:**
- ✅ Feedback isolated per project (Mar Vista feedback doesn't leak to other projects)
- ✅ Token-based similarity detection (finds related queries in same project)
- ✅ Conservative weighting (similar queries get reduced weight vs exact matches)
- ✅ Easy project addition with no code changes needed

---

## How It Works

### 1. **Feedback Storage (Project-Scoped)**
```
data/feedback/search_feedback.jsonl

{
  "timestamp": "2026-05-14T12:30:00",
  "query": "LOTB",
  "project_scope": "mar vista",        # ← Isolates feedback by project
  "pdf_file_name": "Mar Vista POC...",
  "page_number": 42,
  "feedback": "best",
  "note": ""
}
```

### 2. **Generalization Algorithm**
When you search for "log of test boring" in Mar Vista:
1. System looks for exact match: "log of test boring" → finds nothing
2. System finds similar queries: "LOTB" (shares tokens: "log", "test", "boring")
3. Applies generalized feedback with reduced weight (0.5x × similarity)
4. Pages marked "best" on "LOTB" get boosted for "log of test boring"

**Weight Formula:**
- Exact match: **1.0x** (full adjustment)
- Similar query: **0.5x × similarity_score** (conservative)

---

## Adding a New Project

### Step 1: Update Project Configuration (UI)

Edit `search_ui.html` at the top of the `<script>` section:

```javascript
const AVAILABLE_PROJECTS = [
    { id: 'mar vista', name: 'Mar Vista POC', enabled: true },
    { id: '25th ave', name: '25th Avenue', enabled: false },
    { id: 'golden gate', name: 'Golden Gate Bridge', enabled: false },  // ← Add new project
    // Add more as needed
];
```

**Parameters:**
- `id`: Unique identifier (lowercase, used in feedback records)
- `name`: Display name in UI
- `enabled`: Set to `true` to activate, `false` to hide

### Step 2: Enable the Project When Data is Ready

Once you've uploaded the project's PDFs and built the vector index:
```javascript
{ id: '25th ave', name: '25th Avenue', enabled: true },  // ← Change to true
```

### Step 3: Start Collecting Feedback

1. Switch to the project using the dropdown selector (new UI control)
2. Perform searches and provide feedback using the Best/Relevant/Irrelevant buttons
3. Feedback is automatically stored with `project_scope: "25th ave"`
4. Feedback generalization activates automatically

---

## Implementation Details

### Backend Infrastructure (Already in Place)

**API Endpoints Accept `project_scope`:**
```python
POST /search
{
    "query": "shear key details",
    "project_scope": "mar vista",  # ← Filters retrieval + feedback
    "k": 20
}

GET /feedback/labels?query=LOTB&project_scope=mar vista
→ Returns feedback labels for this query in this project only
```

**Search Agent Filters by Scope:**
```python
def _load_feedback_adjustments(query, project_scope):
    # Loads JSONL
    # Filters by normalized query AND project_scope
    # Only applies feedback if scopes match
    # Finds similar queries only within same project
```

### Frontend UI Changes

1. **Project Selector Dropdown** - Added above results section
   - Shows all configured projects
   - Saves selection to localStorage
   - Updates project info title dynamically

2. **Active Project Indicator** - Project title in results section
   - Shows which project feedback is scoped to
   - Updates when switching projects

3. **Notification System** - Toast appears when changing projects
   - Confirms the switch
   - Explains feedback generalization scope

---

## Multi-Project Workflows

### Workflow 1: Sequential Project Refinement
```
1. Collect feedback on Mar Vista (20+ queries)
   → Feedback generalization active within Mar Vista
   
2. Switch to 25th Avenue when uploaded
   → Start fresh feedback collection
   → Generalization works within 25th Avenue scope
   
3. Both projects maintain independent feedback
   → Can later compare which has better quality
```

### Workflow 2: Cross-Project Learning (Future)
```
Optional: Implement cross-project feedback at reduced weight
- Pages marked "best" in Mar Vista for "shear key details"
- Could benefit "shear key details" search in 25th Avenue
- Weight: 0.25x (conservative to prevent incorrect generalizations)
```

---

## Testing Multi-Project Feedback

### Local Verification Script

Run this to verify feedback generalization is working:

```bash
python verify_feedback_generalization.py
```

Output shows:
- Token-based similarity detection
- Exact vs. generalized weight differences
- Similar queries found in feedback store

---

## FAQ

**Q: Will feedback from Mar Vista affect 25th Avenue searches?**
A: No. Each project is isolated. Feedback for "LOTB" in Mar Vista only affects Mar Vista queries.

**Q: Can I disable a project without losing feedback?**
A: Yes. Set `enabled: false` in config. Feedback remains in JSONL. Re-enable anytime to reactivate.

**Q: How do I migrate feedback between projects?**
A: Edit the JSONL file's `project_scope` field. Recommend scripting this for large datasets.

**Q: Will adding projects slow down searches?**
A: No. Filtering by project_scope is fast. Only relevant feedback is processed.

**Q: Can I copy feedback from Mar Vista to 25th Avenue?**
A: Yes. Write a script to read JSONL, modify `project_scope` and `query` fields, append to store. Useful for similar query types across projects.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   Search Interface                        │
│  [Project Selector] Mar Vista | 25th Avenue | ... (✓)   │
└─────────────────┬───────────────────────────────────────┘
                  │ project_scope="mar vista"
                  ↓
        ┌─────────────────────┐
        │   Search Agent      │
        │  (search_agent.py)  │
        └─────────┬───────────┘
                  │ 
        ┌─────────┴──────────────────────────────────┐
        │                                             │
        ↓                                             ↓
    [1] Hybrid Search                        [2] Feedback Generalization
    (BM25 + Vector)                          - Load JSONL
                                             - Filter by project_scope
    ↓                                        - Find similar queries (token overlap)
    Raw Results                              - Apply weighted adjustments
        │                                    
        └──────────────────┬──────────────────┘
                           ↓
                  [Re-ranked Results]
                  (Best pages first)
                           ↓
                  ┌─────────────────────┐
                  │  Display in UI      │
                  │ [Feedback Buttons]  │
                  └──────────┬──────────┘
                             │ User clicks: Best/Relevant/Irrelevant
                             ↓
                  Append to search_feedback.jsonl
                  { query, project_scope, page, feedback }
                             ↓
                  Next search in same project
                  uses this feedback (generalized)
```

---

## Deployment Notes

### Cloud Run Deployment
No code changes needed! The system automatically handles multiple projects:

```bash
gcloud run deploy librarian \
  --source . \
  --set-env-vars "ENVIRONMENT=production,USE_FIRESTORE=false,..."
```

The feedback JSONL grows with each project's feedback. Ensure adequate storage.

### Scaling Considerations
- **Feedback JSONL Size**: ~100 bytes per feedback record. 1000 records = 100KB
- **Generalization Lookup**: O(records) to scan JSONL. Optimize with future indexing if needed
- **Project Count**: No limit. UI shows all configured projects.

---

## Next Steps

1. **Add 25th Avenue** - Upload PDFs, build index, enable in config
2. **Collect Feedback** - Use UI dropdown to switch between projects
3. **Monitor Effectiveness** - Run benchmark_marvista.py for each project
4. **Expand to More Projects** - Repeat steps 1-3 as needed
5. **Optional: Cross-Project Learning** - Implement if projects have common elements (e.g., bridge design standards)

---

**Summary**: Feedback generalization is now **project-aware, scalable, and ready for multi-project deployment**. Add new projects by updating the configuration list. The system handles the rest automatically.

# Drawing-Region Chunking Guide

How to train, apply, and validate **per-drawing** chunks (one chunk per labeled plan region) so search matches engineer-annotated sheets instead of whole-page text flow.

Use this workflow when onboarding a **new structural plan PDF** into the Librarian vector index.

---

## What it does

Traditional indexing splits pages by text flow. Drawing-region chunking:

1. **Detects** separate drawing regions on each plan sheet (vision + cached JSON).
2. **Builds** one searchable chunk per region (label, view type, leader text, isolated OCR).
3. **Classifies** each chunk (`plan`, `elevation`, `layout`, `detail`, `section`, `diagram`, `as_built`, …).
4. **Merges** CAD cross-reference metadata for detail↔section search expansion.

Red-box boundaries in training examples are approximate; leader-line text belongs where the arrowhead points.

---

## Key files

| File | Role |
|------|------|
| `drawing_chunk_examples.py` | Annotated training pages (expected labels per sheet) |
| `drawing_chunk_training.py` | Trainer; writes `data/training/drawing_chunk_model.json` |
| `drawing_region_chunker.py` | Vision detection, spatial text isolation, applicator |
| `chunk_inspector.py` | `inspect-chunks` pass/fail reports |
| `cad_drawing_knowledge.py` | Per-chunk view classification and cross-refs |
| `data/drawing_chunks/{pdf_stem}.json` | Cached vision detections per page |

---

## Prerequisites

- PDF indexed in Chroma (`python main.py index` or existing project in `metadata.json`).
- OpenAI API key (`gpt-4o` for region detection).
- `pdfplumber` for spatial OCR isolation (in `requirements.txt`).
- Active project listed in `metadata_manager` / `MetadataManager.get_active_pdf_file_names()`.

---

For the full multi-layer ingest workflow (enrich, rechunk, CAD, topology, cloud save), see [PDF_PROCESSING_PIPELINE.md](PDF_PROCESSING_PIPELINE.md).

Quick one-shot (auto-trains vision, rechunks, validates training pages):

```bash
python main.py process-pipeline --no-ingest --no-enrich --file MyProject --sync-cloud
python main.py auto-train-rechunk --file MyProject --sync-cloud
```

Auto-train module: `drawing_chunk_auto_train.py`  
Validation report: `data/training/rechunk_validation_report.json`

---

## Workflow for a new project

### 1. Add training examples (recommended)

Edit `drawing_chunk_examples.py` and add 1–2 annotated pages with exact visible titles:

```python
{
    "id": "myproject_bent_layout_1",
    "file_name": "MyProject.pdf",
    "page": 12,
    "sheet_title": "BENT LAYOUT No. 1",
    "sheet_group": "bent",
    "expected_chunks": [
        {"label": "PLAN", "view_type": "plan"},
        {"label": "ELEVATION", "view_type": "elevation"},
        {"label": "SECTION A-A", "view_type": "section"},
    ],
}
```

Add matching entries to `chunk_inspector.KNOWN_PAGE_EXPECTATIONS` for `inspect-chunks` pass/fail.

### 2. Train on examples

```bash
python main.py train-drawing-chunks
```

- Runs vision detection on all examples in `TRAINING_EXAMPLES`.
- Validates detected regions vs expected labels.
- Saves model to `data/training/drawing_chunk_model.json`.
- **Apply threshold:** at least **4/6** examples pass (`ready_for_apply`).

Re-train after prompt or example changes:

```bash
python main.py train-drawing-chunks --force-refresh
```

### 3. Apply to one PDF or all active PDFs

**Full project** (every plan page):

```bash
python main.py apply-drawing-chunks --file MyProject
```

**All active PDFs:**

```bash
python main.py apply-drawing-chunks
```

**Training pages only** (fast iteration; keeps other pages unchanged):

```bash
python main.py apply-drawing-chunks --file MyProject --training-only
```

**Re-detect regions** (ignore vision cache):

```bash
python main.py apply-drawing-chunks --file MyProject --force-refresh
```

`apply-drawing-chunks` automatically runs `merge-cad-knowledge` when finished.

### 4. Validate

```bash
python main.py inspect-chunks --file MyProject --page 12
python main.py inspect-chunks --file MyProject --page 12 --labels "PLAN,ELEVATION,SECTION A-A" --verbose
```

**PASS** requires:

- Chunk count = expected region count
- Zero mixed chunks (no two expected labels in one chunk)
- Every expected label matched by exactly one chunk

### 5. Optional manual CAD refresh

If you only updated classification rules without re-chunking:

```bash
python main.py merge-cad-knowledge
```

---

## How text isolation works

Each region chunk is built in `drawing_region_chunker.py`:

1. **Vision** returns region list: label, view type, scale, description, leader text, cross-refs.
2. **Spatial OCR** (`pdfplumber` words): label bounding box expanded with midpoint clips between neighboring labels on the same page.
3. **Bounded fallback**: if spatial text is empty, slice raw PDF text between sibling label positions (not a fixed 1800-char window).
4. **Scrub**: remove sibling label phrases from OCR context; drop cross-refs that mention other regions on the sheet.
5. **Index hygiene** (`vector_store.add_documents`): drawing-region chunks do **not** get `[ENRICHED METADATA]` appended (that caused label bleed across chunks). Explicit `drawing_label` is preserved through CAD merge.

Chunk text structure:

```
[DRAWING REGION CHUNK]
Sheet: {file} page {n}
Drawing Label: SECTION A-A
Drawing View: section
Description: ...
Leader Text: ...
[LABEL CONTEXT]
{isolated OCR for this region only}

[CAD DRAWING KNOWLEDGE]
{per-chunk classification; no page-wide section-cut lists}
```

---

## View types and search

| View | Typical labels |
|------|----------------|
| `plan` | PLAN, PLAN STAGE 1 |
| `elevation` | ELEVATION, 24" ø STEEL PIPE PILE ELEVATION |
| `layout` | layout sheets with staged plan/elevation |
| `detail` | DETAIL 1, JOINT PROTECTION DETAIL, TYPICAL DETAILS |
| `section` | SECTION A-A, TYPICAL SECTION |
| `diagram` | BENT CAP CAMBER DIAGRAM |
| `as_built` | AS BUILT, AS BUILT SECTION Y-Y |

Detail queries expand to matching section chunks on the same component (`cad_drawing_knowledge.py`, `search_agent.py`).

---

## Active library (current)

| PDF | Training pages | Notes |
|-----|----------------|-------|
| Goldenwest | 17, 21, 61 | Abutment + railing details |
| Westminster | 11 | Abutment layout |
| Magnolia | 10 | Bent layout — vision needs `--force-refresh` |
| Fresno | 12 | Holdout / needs region refresh |
| Edwards | — | Full apply only |

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| All chunks **MIXED** | Enriched metadata or page OCR in every chunk | Re-apply after `add_documents` region-chunk fix; verify no `[ENRICHED METADATA]` in chunk body |
| Wrong chunk **count** | Stale or bad vision cache | `--force-refresh` on that PDF |
| **Missing** expected labels | Vision didn't detect region | Add to training examples; re-train; refresh cache |
| `DETAIL 1` in every chunk | Sheet note OCR bleed | Tighten spatial bbox or scrub; check raw PDF text quality |
| Apply hangs on Edwards | Stale Chroma lock or slow pdfplumber reopen | Stop stale Python jobs; apply uses indexed text + cached PDF handle |
| `UnicodeEncodeError` on Windows | Checkmark in console output | Fixed in `vector_store.add_documents` (ASCII-only prints) |
| Apply hangs on Edwards | Old `--training-only` loaded all pages | Use per-file apply; training-only now skips non-training PDFs |
| `429` on merge | Embedding rate limit | Retry built into `vector_store._embed_documents_with_retry` |

---

## Command reference

```bash
# Train
python main.py train-drawing-chunks [--force-refresh]

# Apply
python main.py apply-drawing-chunks [--file <name>] [--training-only] [--force-refresh]

# Inspect
python main.py inspect-chunks --file <name> --page <n> [--labels "A,B"] [--verbose] [--show-text]

# CAD metadata only
python main.py merge-cad-knowledge
```

---

## Checklist: new PDF end-to-end

- [ ] PDF in active `metadata.json` and vector index
- [ ] 1–2 pages added to `drawing_chunk_examples.py` + `KNOWN_PAGE_EXPECTATIONS`
- [ ] `train-drawing-chunks` → ≥ 4/6 pass (or add more examples)
- [ ] `apply-drawing-chunks --file <pdf>`
- [ ] `inspect-chunks` on training pages → PASS
- [ ] Spot-check search: e.g. "shear key detail" returns detail + section chunks

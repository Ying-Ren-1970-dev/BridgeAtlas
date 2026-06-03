# Librarian Topology Spec (One Page)

## 1) Purpose
Define a practical, engineer-facing topology model for retrieval across structural plan sets so the system can return the right project, sheet, and detail for design and constructability questions.

## 2) Scope
- Domain: civil/structural transportation plan sets (bridges, walls, foundations, stations).
- Input artifacts: PDF plan sheets + extracted metadata + analyzed detail text.
- Retrieval target: project-level relevance, page-level relevance, and detail-level evidence.

## 3) Core Entity Model
- Project
  - Represents one plan set PDF and project context.
  - Key fields: file_name, project_name, enriched_project_number, categories, phase, total_pages.
- Page (Sheet)
  - Represents one sheet within a project.
  - Key fields: page, plan_sheet_title, plan_sheet_number, plan_sheet_type, plan_primary_type.
- Detail Chunk
  - Represents a semantically meaningful subsection of a sheet.
  - Key fields: chunk_id, detail_types, detail_intents, structural_elements, page/detail labels.
- Term/Synonym
  - Canonical engineering term and known variants/acronyms.
  - Examples: CIDH <-> cast-in-drilled-hole, shear key <-> pipe pin, rebar <-> reinforcement.
- Graph Detail Node
  - Cross-sheet detail abstraction from enriched metadata graph.
  - Holds references to related sheets/details for multi-hop retrieval.

## 4) Relationship Model
- Project CONTAINS Page (1:N)
- Page CONTAINS Detail Chunk (1:N)
- Detail Chunk REFERENCES Sheet ID (N:M)
- Detail Chunk REFERENCES Detail ID (N:M)
- Detail Chunk MENTIONS Structural Element (N:M)
- Term EXPANDS_TO Synonym Variant (1:N)
- Graph Detail Node LINKS_TO Related Node (N:M)

## 5) Structural Topology Taxonomy
- Foundation system: CIDH pile, drilled shaft, footing, pile cap.
- Substructure: abutment, bent, column, cap beam.
- Superstructure: girder, R/C box girder, deck slab, diaphragm.
- Connection/detail components: shear key, pipe pin, bearing pad, expansion joint.
- Materials/reinforcement: rebar, reinforcing steel, concrete, structural steel.
- Retaining/safety ancillaries: retaining wall, anchors/tie-backs, railing/fence.

## 6) Sheet Semantics
- structure_plan: context/navigation sheets; broad system intent.
- details/rebar: execution-critical sheets; high value for reinforcement/detail queries.
- section/elevation/detail signals: strengthen detail relevance and cross-reference behavior.

## 7) Query Intent Types
- Identifier intent
  - Example: project number, sheet number, tagged IDs.
  - Rule: strict lexical containment required.
- Component intent
  - Example: shear key, CIDH, pipe pin, diaphragm.
  - Rule: synonym expansion allowed, precision guardrails active.
- Detail intent
  - Example: reinforcement details, typical section, rebar layout.
  - Rule: prioritize details/rebar sheets; avoid over-filtering by exact phrase on every chunk.
- Scoped intent
  - Example: queries limited to one project/file.
  - Rule: larger retrieval pool within scope, then rank by evidence density.

## 8) Ranking Policy (High Level)
Final score is a fused relevance score from multiple evidence channels:

- Semantic vector similarity (broad conceptual match)
- BM25-style lexical relevance (literal engineering terms)
- Metadata boosts (sheet type, detail intents, structural elements, identifiers)
- Phrase boosts (exact multi-word technical phrases)
- Graph/detail support (related detail nodes and references)
- Feedback adjustments (best/relevant/irrelevant labels)

Practical bias:
- For detail+reinforcement queries: boost details/rebar sheets, reduce generic structure_plan dominance.
- For identifier queries: prioritize exact identifier presence over semantic similarity.

## 9) Retrieval Pipeline
1. Normalize query and detect intent.
2. Expand terms with engineering synonyms (except strict identifier mode).
3. Retrieve vector candidates.
4. Retrieve keyword/BM25 candidates.
5. Fuse rankings (hybrid).
6. Apply intent guardrails and project-aware filtering.
7. Add graph/detail evidence and rerank.
8. Group by project and return pages with topology elements.

## 10) Quality Gates
- Precision gate: wrong structural system projects (e.g., I-girder for box-girder query) should be filtered.
- Recall gate: key detail sheets (typical section/rebar detail) must remain discoverable.
- Explainability gate: every returned page should include structural topology elements and evidence text.
- Regression gate: known identifier and project-scope benchmarks remain stable after ranking changes.

## 11) Known Limits
- OCR/vision quality can weaken sheet_type and structural_elements extraction.
- Some projects require project-level context to infer system type from sparse detail text.
- Cross-project false positives can occur when generic reinforcement language dominates.

## 12) Future Enhancements
- Explicit project-level structural system label (box girder vs I-girder) in metadata.
- Confidence-calibrated intent filter with fallback tiers.
- Learned re-ranking model from engineer feedback signals.
- Deterministic topology validation checks during indexing.

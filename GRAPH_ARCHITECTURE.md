# Algorithmic Graph Approach - Architecture & Implementation

## Overview

This document describes the implementation of the **Algorithmic Graph Approach** for CAD PDF indexing, which solves the problem of search results being "off" by moving from **page-centric** to **detail-centric** knowledge base architecture.

## Problem Statement

**Original Architecture (Page-Centric)**:
- Each PDF page indexed as single vector chunk
- Multiple drawings/details on one page collapse into one vector representation
- Search cannot distinguish "Detail 3/Section A-A" from "General Plan" on same page
- Result: Broad, low-relevance search results

**User Symptom**: "Search results are still off. Can you check if within the page of pdf, each detail/drawing is identified in the current knowledge base?"

**Root Cause Identified**: Knowledge base stores entire pages as single chunks; no detail-level identifiers or connectivity information.

## Solution: Algorithmic Graph Approach

Three phases:

1. **Deterministic Parsing (No ML)**: Extract geometric relationships from PDFs using `pdfplumber` and `shapely`
2. **Graph Storage**: Save connectivity relationships in local JSON/graph structure
3. **Grounding the General AI**: When searching, query graph first to find connected details, then feed to GPT

## Architecture

### Phase 1: Geometry Extraction

**File**: `pdf_geometry_parser.py`

**Purpose**: Extract raw geometric elements and detail callouts from CAD PDFs without ML.

**Key Components**:
- `BoundingBox`: Rectangle representation with geometric operations
- `TextLabel`: Text with position, font size, styling
- `GeometricElement`: Lines, rectangles, curves from PDF vector data
- `DetailCallout`: Identified detail with associated elements
- `PDFGeometryParser`: Main extraction engine

**Process**:
```
PDF → pdfplumber → {lines, rectangles, curves} → shapely → geometric relationships
     → text extraction → clustering → detail callouts
```

**Output**: Dictionary mapping page_num → {elements, text_labels, detail_callouts}

**Usage**:
```python
from pdf_geometry_parser import PDFGeometryParser

parser = PDFGeometryParser(max_pages=50)
geometry_data = parser.parse_pdf(Path("plan.pdf"))
# geometry_data[1] = {
#     'page_number': 1,
#     'width': 1224.0, 'height': 792.0,
#     'elements': [...],  # 20,000+ geometric elements
#     'detail_callouts': [...],
#     'text_labels': [...]
# }
```

### Phase 2: Detail Graph Construction

**Files**: 
- `detail_graph_builder.py` - Graph structure and algorithms
- `enriched_metadata_graph_builder.py` - Integration with enriched metadata (primary approach)

**Purpose**: Build connectivity graph from geometry or enriched metadata.

**Key Components**:

**DetailNode**:
```python
@dataclass
class DetailNode:
    detail_id: str              # "Page5_Detail2"
    page_number: int
    title: str                  # "Girder Detail"
    element_type: str           # "detail" | "section" | "elevation" | "schedule"
    bbox: Dict                  # {x0, y0, x1, y1}
    text_content: List[str]     # ["Girder Detail", "1:10 scale", ...]
    file_name: str
    project_name: str
```

**DetailEdge**:
```python
@dataclass
class DetailEdge:
    source_id: str              # "Page5_Detail2"
    target_id: str              # "Page6_Detail1"
    relationship_type: str      # "references" | "intersects_with" | "same_page"
    strength: float             # 0.0-1.0 (confidence)
    reason: str                 # "Both reference: CIDH Pile"
```

**DetailGraph**:
```python
class DetailGraph:
    nodes: Dict[str, DetailNode]        # detail_id → node
    edges: Dict[str, List[DetailEdge]]  # source_id → [edges]
    reverse_edges: Dict[...]             # target_id → [edges]
    
    def get_connected_components(detail_id: str, max_depth: int) → Set[str]
    def get_related_nodes(detail_id: str, relationship_type: str = None) → List[DetailNode]
```

**Two Approaches**:

**Approach 1: Geometric Parsing** (Deterministic, detailed)
```python
from pdf_geometry_parser import PDFGeometryParser
from detail_graph_builder import DetailGraphBuilder

parser = PDFGeometryParser()
geometry_data = parser.parse_pdf("plan.pdf")

builder = DetailGraphBuilder()
graph = builder.build_graph_from_geometry(geometry_data, "plan.pdf", "MyProject")
```

**Approach 2: Enriched Metadata** (Faster, uses existing data)
```python
from enriched_metadata_loader import EnrichedMetadataLoader
from enriched_metadata_graph_builder import EnrichedMetadataGraphBuilder

loader = EnrichedMetadataLoader()
enriched = loader.load_enriched_metadata("plan.pdf")

builder = EnrichedMetadataGraphBuilder()
graph = builder.build_graph_from_enriched_metadata(
    enriched_metadata=enriched,
    pdf_file_name="plan.pdf",
    project_name="MyProject"
)
```

**Relationship Types**:
- `shared_element`: Details reference same structural element (girder, column, etc.)
- `references`: Textual reference from one detail to another
- `intersects_with`: Details share bounding box/spatial area
- `references_detail`: General plan references detail page
- `same_page`: Details on same page (lightweight connection)

### Phase 3: Graph Storage & Integration

**File**: `enriched_metadata_graph_builder.py`

**GraphIntegrationManager**:
```python
class GraphIntegrationManager:
    def build_and_store_graph(enriched_metadata, pdf_file_name, project_name) → DetailGraph
    def get_graph(project_name: str) → DetailGraph
    def find_related_details(project_name, detail_id, max_depth) → Dict
```

**Storage Format**: JSON
```json
{
  "project_name": "Mar Vista POC 100%",
  "nodes": {
    "Page5_Detail0": {
      "detail_id": "Page5_Detail0",
      "page_number": 5,
      "title": "Girder Detail",
      "element_type": "detail",
      "bbox": {"x0": 0, "y0": 0, "x1": 100, "y1": 100},
      "text_content": ["Girder Detail", "Detail calls to Pile Cap"],
      "file_name": "plan.pdf",
      "project_name": "Mar Vista POC 100%"
    }
  },
  "edges": {
    "Page5_Detail0": [
      {
        "source_id": "Page5_Detail0",
        "target_id": "Page6_Detail1",
        "relationship_type": "references",
        "strength": 0.8,
        "reason": "Text reference found"
      }
    ]
  }
}
```

**Location**: `data/graphs/{project_name}_detail_graph.json`

### Phase 4: Search Integration

**Integration Point**: `search_agent.py`

**Search Flow with Graphs**:
```
User Query
    ↓
[1] Parse semantically → identify search intent
    ↓
[2] Query detail graphs → find matching details
    ↓
[3] Expand via graph → find connected details (depth ≤ 2)
    ↓
[4] Retrieve vectors → for matched + connected details
    ↓
[5] Rank & score → using semantic analysis
    ↓
[6] Generate answer → pass to LLM with context
    ↓
Grounded, Detail-Specific Answer
```

**Example**:
```
Query: "CIDH pile detail"

Graph Query Results:
  Matched: [Page8_Detail3 "CIDH Pile Detail"]
  Connected: [
    Page9_Detail1 "Pile Reinforcement",
    Page10_Detail0 "Connection to Pile Cap",
    Page8_Detail4 "Pile Layout"
  ]

Vector Search:
  - Detail vectors instead of page vectors
  - Higher relevance for "Pile Detail" + "Reinforcement"
  - Eliminates unrelated details on same page

Answer:
  "CIDH piles on this project are [detail info] with reinforcement
   requirements [from Detail 1] and connections [from Detail 0]..."
```

## Implementation Status

### ✅ Completed
- [x] `pdf_geometry_parser.py` - Geometry extraction engine
- [x] `detail_graph_builder.py` - Graph construction from geometry
- [x] `enriched_metadata_graph_builder.py` - Graph construction from enriched metadata
- [x] `GraphIntegrationManager` - High-level API for graph management
- [x] `main.py build-graphs` command - CLI integration
- [x] Graph storage and serialization
- [x] Relationship building (element-based, intent-based, hierarchical)
- [x] Test suites (`test_graph_approach.py`, `test_graph_building.py`)

### ✅ Tested
- Sample graph built successfully for "30%_Elk Grove Station Structure Plan Set"
- 12 detail nodes, 198 relationships extracted
- JSON serialization working
- Graph queries (connected components, related nodes) operational

### 🔄 In Progress
- Integration with search engine (search_agent.py updates)
- Detail-level vector indexing (vector_store.py updates)
- End-to-end testing with benchmark queries

### ⏳ Not Started
- Deployment with graph-enabled search
- UI updates for detail-centric display
- Performance optimization for large graphs

## Usage Guide

### Build Graphs for All Projects
```bash
python main.py enrich-kb      # First ensure enrichment is done
python main.py build-graphs   # Build detail graphs for all projects
```

### Query a Graph Programmatically
```python
from enriched_metadata_graph_builder import GraphIntegrationManager

manager = GraphIntegrationManager()

# Load graph
graph = manager.get_graph("Mar Vista POC 100%_CheckPrint_20211118 Complete")

# Find matching details
cidh_details = [
    (did, node) for did, node in graph.nodes.items()
    if "cidh" in node.title.lower()
]
print(f"Found {len(cidh_details)} CIDH details")

# Expand to connected details
if cidh_details:
    detail_id = cidh_details[0][0]
    connected = graph.get_connected_components(detail_id, max_depth=2)
    print(f"Total connected: {len(connected)} details")
    
    related = graph.get_related_nodes(detail_id)
    for node in related:
        print(f"  - {node.title}")
```

### Inspect Graph Structure
```python
# Graph statistics
print(f"Nodes: {len(graph.nodes)}")
print(f"Edges: {sum(len(e) for e in graph.edges.values())}")

# Node details
for detail_id in list(graph.nodes.keys())[:5]:
    node = graph.nodes[detail_id]
    print(f"{detail_id}: {node.title} (page {node.page_number}, {node.element_type})")

# Edge details
for source_id in list(graph.edges.keys())[:3]:
    edges = graph.edges[source_id]
    for edge in edges[:2]:
        print(f"{edge.source_id} --[{edge.relationship_type}]--> {edge.target_id}")
        print(f"  Reason: {edge.reason}")
```

## Key Design Decisions

1. **Deterministic Over ML**: Uses geometric relationships and text matching, not ML models
   - Pro: Reproducible, explainable, no API calls needed
   - Pro: Works offline
   - Con: May miss semantic relationships not captured by rules

2. **Enriched Metadata First**: Primary approach uses enriched metadata, not raw geometry parsing
   - Pro: Fast (no PDF re-parsing)
   - Pro: Leverages existing structure extraction
   - Pro: Integrates naturally with existing KB pipeline
   - Con: Limited to relationships in enriched data

3. **JSON-Based Storage**: Graph persisted as JSON, not Neo4j
   - Pro: Simple, no database setup
   - Pro: Easy to version control and backup
   - Pro: Queryable with Python dict operations
   - Con: Slower for very large graphs (1000s+ details)

4. **Multiple Relationship Types**: Different edge types for different query scenarios
   - Enables flexible, nuanced traversal
   - Allows querying with relationship type filters
   - Supports future ML ranking of relationships

5. **Bounded Traversal**: `max_depth` parameter limits graph expansion
   - Prevents returning everything
   - Keeps results focused and relevant
   - Typical: max_depth=2 (your detail + neighboring details)

## Performance Characteristics

**Graph Building**:
- 10 projects with enriched metadata: ~2-5 seconds total
- Graph size: 10-50 detail nodes per project
- Relationship edges: 50-200 edges per project

**Graph Queries**:
- Find matching details: O(n) where n = node count (~50 nodes) = < 1ms
- Get connected components: O(n+m) where m = edge count (~200) = < 5ms
- Total graph query overhead: negligible (<10ms)

**Memory**:
- Loaded graph in memory: ~1MB per project
- All 10 projects: ~10MB total

## Future Enhancements

1. **Semantic Similarity**: Use embeddings to find conceptually related details beyond keywords
2. **Visualization**: D3.js graph visualization of detail connectivity
3. **Neo4j Backend**: For very large graphs (1000s+ details per project)
4. **ML Relationship Ranking**: Train model to weight edges by relevance to queries
5. **Cross-Project Relationships**: Link details across projects that share structural patterns
6. **Detail Hierarchy**: Express parent-child relationships (e.g., Girder Detail contains Sub-Details)

## Integration with Existing Systems

**Knowledge Base Pipeline**:
```
PDF Files
  ↓
pdf_processor.py (extract text + vision)
  ↓
enriched_metadata_loader.py (add structure metadata)
  ↓
NEW: enriched_metadata_graph_builder.py (build detail graphs)
  ↓
vector_store.py (updated for detail-level indexing)
  ↓
search_agent.py (query graphs before vector search)
  ↓
api.py (expose graph-enhanced search)
```

**API Endpoints** (to be added):
- `GET /graph/{project_name}/details` - List all details in project
- `GET /graph/{project_name}/detail/{detail_id}` - Get detail info + connections
- `GET /graph/{project_name}/search/{query}` - Search graph for details
- `POST /search?use_graph=true` - Search with graph grounding

## References

- Graph code: `detail_graph_builder.py`, `enriched_metadata_graph_builder.py`
- Integration guide: `GRAPH_INTEGRATION_GUIDE.md`
- Test scripts: `test_graph_approach.py`, `test_graph_building.py`
- Sample graph: `data/graphs/*.json`

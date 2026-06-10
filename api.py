"""
OpenAPI REST API for Librarian Search System
Provides search functionality over structural engineering PDF documents.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
import uvicorn
from datetime import datetime
import fitz  # PyMuPDF
from io import BytesIO
import os
import json
import re
from collections import Counter

from search_agent import SearchAgent
from vector_store import VectorStore
from metadata_manager import MetadataManager
from storage_adapter import storage
from engineering_terminology import EngineeringTerminology
import config
from analysis_engine import AnalysisEngine

# Initialize FastAPI app
app = FastAPI(
    title="Structural Engineering Librarian API",
    description="Search and retrieve information from structural engineering plan documents using RAG-powered semantic search",
    version="1.0.0",
    contact={
        "name": "Librarian API",
        "email": "support@example.com"
    }
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
search_agent = None
vector_store = None
metadata_manager = None
analysis_engine = None

# Feedback storage
FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "data", "feedback")
SEARCH_FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, "search_feedback.jsonl")


def _ensure_feedback_store():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if not os.path.exists(SEARCH_FEEDBACK_FILE):
        with open(SEARCH_FEEDBACK_FILE, "w", encoding="utf-8"):
            pass


def _append_feedback_record(record: dict):
    _ensure_feedback_store()
    with open(SEARCH_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _read_feedback_records(limit: Optional[int] = None) -> List[dict]:
    _ensure_feedback_store()
    records: List[dict] = []
    with open(SEARCH_FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    if limit is not None and limit > 0:
        return records[-limit:]
    return records


def _normalize_text(value: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", _normalize_text(value))
    slug = slug.strip("_")
    return slug or "query"


def _latest_feedback_labels(query: str, project_scope: Optional[str] = None) -> Dict[Tuple[str, int], dict]:
    """Return latest feedback label per (pdf_file_name, page_number) for query/scope."""
    query_key = _normalize_text(query)
    scope_key = _normalize_text(project_scope)

    latest: Dict[Tuple[str, int], dict] = {}
    for item in _read_feedback_records(limit=None):
        if _normalize_text(item.get("query")) != query_key:
            continue

        item_scope = _normalize_text(item.get("project_scope"))
        if scope_key and item_scope and item_scope != scope_key:
            continue

        file_name = str(item.get("pdf_file_name", "")).strip()
        page_number = item.get("page_number")
        label = str(item.get("feedback", "")).strip().lower()

        if not file_name or page_number is None:
            continue

        try:
            page_number = int(page_number)
        except Exception:
            continue

        latest[(file_name, page_number)] = {
            "feedback": label,
            "timestamp": item.get("timestamp"),
            "note": item.get("note"),
        }

    return latest


def _build_feedback_pages_pdf(
    query: str,
    project_scope: Optional[str],
    accepted_labels: set,
    empty_detail: str,
    output_prefix: str,
):
    """Build a merged PDF for pages whose latest feedback is in accepted_labels."""
    latest = _latest_feedback_labels(query=query, project_scope=project_scope)
    selected_pages = sorted(
        [
            (file_name, page_number)
            for (file_name, page_number), meta in latest.items()
            if meta.get("feedback") in accepted_labels
        ],
        key=lambda x: (x[0].lower(), x[1]),
    )

    if not selected_pages:
        raise HTTPException(status_code=404, detail=empty_detail)

    merged = fitz.open()
    added = 0

    try:
        for file_name, page_number in selected_pages:
            try:
                pdf_path = storage.get_pdf_temp_path(file_name)
                src = fitz.open(pdf_path)
                try:
                    if 1 <= page_number <= len(src):
                        merged.insert_pdf(src, from_page=page_number - 1, to_page=page_number - 1)
                        added += 1
                finally:
                    src.close()
            except Exception:
                continue

        if added == 0:
            raise HTTPException(status_code=404, detail="No valid pages could be exported")

        output = BytesIO(merged.tobytes())
        output.seek(0)
    finally:
        merged.close()

    filename = f"{output_prefix}_{_slugify(query)}.pdf"
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(output, media_type="application/pdf", headers=headers)


@app.on_event("startup")
async def startup_event():
    """Initialize search components on startup."""
    global search_agent, vector_store, metadata_manager, analysis_engine
    
    # Sync vector database from cloud storage in production
    storage.sync_vector_db_from_cloud()
    
    search_agent = SearchAgent()
    vector_store = search_agent.vector_store
    metadata_manager = MetadataManager()
    analysis_engine = AnalysisEngine()
    kb_stats = _knowledge_base_stats()
    print(
        "[OK] Librarian API initialized successfully "
        f"(chunks={kb_stats.get('chunk_count', 0)}, "
        f"deep_vision={kb_stats.get('deep_vision_chunks', 0)})"
    )


# Request/Response Models
class SearchRequest(BaseModel):
    """Search query request model."""
    query: str = Field(..., description="Search query text (keywords, phrases, or questions)", example="steel shear key")
    k: int = Field(default=50, ge=1, le=100, description="Number of results to return (1-100)")
    relevance_threshold: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Optional max score difference from best result. If omitted, mode-specific defaults are used (hybrid=2.0, vector-only=0.5)")
    project_scope: Optional[str] = Field(default=None, description="Optional project/file scope, e.g. 'mar vista', to restrict retrieval")
    generate_summary: bool = Field(default=False, description="Whether to generate an LLM summary for this request")
    use_hybrid_search: bool = Field(default=True, description="Whether to use hybrid retrieval (vector + keyword BM25-style fusion)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "steel shear key",
                "k": 50,
                "relevance_threshold": 2.0,
                "project_scope": "mar vista",
                "generate_summary": False,
                "use_hybrid_search": True
            }
        }


class TopologyDetail(BaseModel):
    """Structural topology details extracted from content."""
    element_type: str = Field(..., description="Type of structural element")
    mentioned: bool = Field(..., description="Whether this element was mentioned in the content")
    
    class Config:
        json_schema_extra = {
            "example": {
                "element_type": "Pipe Pin (Steel Shear Key)",
                "mentioned": True
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    pdf_file_name: str = Field(..., description="Name of the PDF file")
    project_name: str = Field(..., description="Project name")
    page_number: int = Field(..., description="Page number within the PDF")
    relevance_score: float = Field(..., description="Relevance score (lower is better, 0.0-2.0 typical range)")
    topology_elements: List[str] = Field(..., description="Structural topology elements found in this content")
    content_sample: str = Field(..., description="Sample of the relevant content")
    has_vision_analysis: bool = Field(..., description="Whether this page was analyzed using GPT-4 Vision")
    
    class Config:
        json_schema_extra = {
            "example": {
                "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
                "project_name": "Mar Vista POC",
                "page_number": 11,
                "relevance_score": 0.85,
                "topology_elements": ["Pipe Pin (Steel Shear Key)", "Abutment", "Bearing Pad"],
                "content_sample": "Pipe Pin Detail: Includes cover pipe, standard pipe...",
                "has_vision_analysis": True
            }
        }


class ProjectSummary(BaseModel):
    """Summary of a project with relevant pages."""
    pdf_file_name: str
    project_name: str
    phase: Optional[str]
    engineer_of_record: Optional[str]
    date: Optional[str]
    categories: List[str]
    relevant_pages: List[int]
    total_pages: int
    
    class Config:
        json_schema_extra = {
            "example": {
                "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
                "project_name": "Mar Vista POC",
                "phase": "100% Final",
                "engineer_of_record": "Example Engineering Inc.",
                "date": "2021-11-18",
                "categories": ["Bridges", "Retaining Walls"],
                "relevant_pages": [5, 10, 11, 18],
                "total_pages": 42
            }
        }


class SearchResponse(BaseModel):
    """Complete search response."""
    query: str = Field(..., description="The search query that was executed")
    total_results: int = Field(..., description="Total number of results found")
    projects_found: int = Field(..., description="Number of projects with relevant content")
    results: List[SearchResult] = Field(..., description="Detailed search results ranked by relevance")
    project_summaries: List[ProjectSummary] = Field(..., description="Summary by project")
    search_summary: str = Field(..., description="AI-generated summary of findings")
    timestamp: str = Field(..., description="Search timestamp (ISO format)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "steel shear key",
                "total_results": 15,
                "projects_found": 3,
                "results": [],
                "project_summaries": [],
                "search_summary": "Found steel shear key details in Mar Vista project...",
                "timestamp": "2026-05-10T22:00:00Z"
            }
        }


class TerminologyFeedbackRequest(BaseModel):
    """User-confirmed terminology mapping request."""
    acronym: str = Field(..., description="Acronym to learn, e.g., LOTB")
    expansion: str = Field(..., description="Expanded phrase, e.g., log of test boring")


class SearchResultFeedbackRequest(BaseModel):
    """Engineer feedback for a specific search result."""
    query: str = Field(..., description="Original user query")
    pdf_file_name: str = Field(..., description="Matched PDF file name")
    page_number: int = Field(..., ge=1, description="Matched page number")
    feedback: str = Field(..., description="Feedback label: relevant, irrelevant, or best")
    note: Optional[str] = Field(default=None, description="Optional reason/comment")
    project_scope: Optional[str] = Field(default=None, description="Optional project scope used during search")


class SearchFeedbackStatsResponse(BaseModel):
    total_feedback: int
    by_feedback: dict
    top_queries: List[dict]


class AnalyzeRequest(BaseModel):
    """Request model for analysis-style multi-evidence answers."""
    query: str = Field(..., description="Question or task to analyze")
    k: int = Field(default=40, ge=5, le=100, description="Initial retrieval depth before analysis")
    relevance_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Optional score threshold offset from best result"
    )
    project_scope: Optional[str] = Field(default=None, description="Optional project/file scope")
    use_hybrid_search: bool = Field(default=True, description="Use hybrid retrieval (vector + keyword)")


class AnalysisEvidence(BaseModel):
    pdf_file_name: str
    project_name: str
    page_number: int
    relevance_score: float
    semantic_match_score: float
    component_matches: List[str]
    snippet: str
    why_matched: str


class InferredIntent(BaseModel):
    query: str
    intent_types: Dict[str, bool]
    structural_components: List[str]
    normalized: str


class AnalyzeResponse(BaseModel):
    query: str
    answer: str
    inferred_intent: InferredIntent
    intent_confidence: float
    semantic_evidence: List[AnalysisEvidence]
    terminology_variants: Dict[str, List[str]]
    search_summary: Optional[str] = None
    timestamp: str


# API Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Serve the search UI interface."""
    search_ui_path = os.path.join(os.path.dirname(__file__), "search_ui.html")
    if os.path.exists(search_ui_path):
        return FileResponse(search_ui_path, media_type="text/html")
    return {
        "name": "Structural Engineering Librarian API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "search_ui": "/ui"
    }


@app.get("/ui", tags=["UI"])
async def serve_search_ui():
    """Serve the search UI interface."""
    search_ui_path = os.path.join(os.path.dirname(__file__), "search_ui.html")
    if not os.path.exists(search_ui_path):
        raise HTTPException(status_code=404, detail="Search UI not found")
    return FileResponse(search_ui_path, media_type="text/html")


@app.get("/dashboard", tags=["UI"])
async def serve_dashboard():
    """Serve the training dashboard interface."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "training_dashboard.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(dashboard_path, media_type="text/html")


def _knowledge_base_stats() -> Dict:
    """Return lightweight KB diagnostics for local/cloud parity checks."""
    stats = {
        "use_cloud_storage": config.USE_CLOUD_STORAGE,
        "vector_db_path": str(config.VECTOR_DB_PATH),
    }
    if not search_agent or not search_agent.vector_store or not search_agent.vector_store.vectorstore:
        stats["status"] = "not_initialized"
        return stats

    collection = search_agent.vector_store.vectorstore._collection
    all_docs = collection.get(include=[])
    stats["chunk_count"] = len(all_docs.get("ids", []))
    merged = collection.get(where={"deep_vision_merged": "true"}, include=[])
    stats["deep_vision_chunks"] = len(merged.get("ids", []))
    stats["status"] = "ready"
    return stats


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "vector_store": "operational" if vector_store else "not initialized",
        "search_agent": "operational" if search_agent else "not initialized",
        "knowledge_base": _knowledge_base_stats(),
    }


@app.get("/test-openai", tags=["Health"])
async def test_openai():
    """Test OpenAI API connectivity and embedding generation."""
    try:
        from langchain_openai import OpenAIEmbeddings
        import config
        import time
        
        start = time.time()
        embeddings = OpenAIEmbeddings(
            openai_api_key=config.OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        
        # Try to generate a simple embedding
        result = embeddings.embed_query("test")
        elapsed = time.time() - start
        
        return {
            "status": "success",
            "message": "OpenAI API connection working",
            "embedding_dimensions": len(result),
            "response_time_ms": int(elapsed * 1000),
            "api_key_present": bool(config.OPENAI_API_KEY),
            "api_key_prefix": config.OPENAI_API_KEY[:10] + "..." if config.OPENAI_API_KEY else None
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "message": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "api_key_present": bool(config.OPENAI_API_KEY) if 'config' in locals() else False
        }


@app.post("/terminology/feedback", tags=["Terminology"])
async def terminology_feedback(request: TerminologyFeedbackRequest):
    """Save a user-confirmed acronym expansion for future automatic query expansion."""
    acronym = request.acronym.strip()
    expansion = request.expansion.strip()
    if not acronym or not expansion:
        raise HTTPException(status_code=400, detail="Both acronym and expansion are required")

    EngineeringTerminology.add_feedback_mapping(acronym=acronym, expansion=expansion)
    return {
        "status": "ok",
        "message": "Terminology mapping saved",
        "acronym": acronym.upper(),
        "expansion": expansion,
    }


@app.get("/terminology/learned", tags=["Terminology"])
async def terminology_learned():
    """List persisted learned terminology mappings."""
    terms = EngineeringTerminology._combined_terms()
    return {
        "total_terms": len(terms),
        "terms": terms,
    }


@app.post("/feedback/search-result", tags=["Feedback"])
async def submit_search_result_feedback(request: SearchResultFeedbackRequest):
    """Save engineer feedback for a retrieved page result."""
    allowed = {"relevant", "irrelevant", "best"}
    label = request.feedback.strip().lower()
    if label not in allowed:
        raise HTTPException(status_code=400, detail=f"feedback must be one of: {sorted(allowed)}")

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": request.query.strip(),
        "project_scope": (request.project_scope or "").strip() or None,
        "pdf_file_name": request.pdf_file_name.strip(),
        "page_number": request.page_number,
        "feedback": label,
        "note": (request.note or "").strip() or None,
    }
    _append_feedback_record(record)
    return {"status": "ok", "saved": record}


@app.get("/feedback/search-result", tags=["Feedback"])
async def list_search_result_feedback(limit: int = Query(default=200, ge=1, le=5000)):
    """List recent search-result feedback entries."""
    items = _read_feedback_records(limit=limit)
    return {"count": len(items), "items": items}


@app.get("/feedback/stats", response_model=SearchFeedbackStatsResponse, tags=["Feedback"])
async def search_feedback_stats(limit: int = Query(default=2000, ge=1, le=100000)):
    """Return aggregate feedback stats for evaluation and tuning."""
    items = _read_feedback_records(limit=limit)
    feedback_counter = Counter(item.get("feedback", "unknown") for item in items)
    query_counter = Counter(item.get("query", "") for item in items if item.get("query"))

    top_queries = [
        {"query": q, "count": c}
        for q, c in query_counter.most_common(20)
    ]

    return SearchFeedbackStatsResponse(
        total_feedback=len(items),
        by_feedback=dict(feedback_counter),
        top_queries=top_queries,
    )


@app.get("/feedback/labels", tags=["Feedback"])
async def latest_feedback_labels(
    query: str = Query(..., description="Search query to resolve latest labels for"),
    project_scope: Optional[str] = Query(default=None, description="Optional project scope used during search"),
):
    """Get latest feedback label per page for a query/scope."""
    latest = _latest_feedback_labels(query=query, project_scope=project_scope)
    items = [
        {
            "pdf_file_name": file_name,
            "page_number": page_number,
            "feedback": meta.get("feedback"),
            "timestamp": meta.get("timestamp"),
            "note": meta.get("note"),
        }
        for (file_name, page_number), meta in sorted(latest.items(), key=lambda x: (x[0][0].lower(), x[0][1]))
    ]
    return {
        "count": len(items),
        "query": query,
        "project_scope": project_scope,
        "items": items,
    }


@app.get("/feedback/export/best-pdf", tags=["Feedback"])
async def export_best_feedback_pdf(
    query: str = Query(..., description="Search query used to collect feedback"),
    project_scope: Optional[str] = Query(default=None, description="Optional project scope used during search"),
):
    """Download merged PDF containing pages currently labeled as 'best'."""
    return _build_feedback_pages_pdf(
        query=query,
        project_scope=project_scope,
        accepted_labels={"best"},
        empty_detail="No pages labeled 'best' for this query/scope",
        output_prefix="best_pages",
    )


@app.get("/feedback/export/relevant-best-pdf", tags=["Feedback"])
async def export_relevant_best_feedback_pdf(
    query: str = Query(..., description="Search query used to collect feedback"),
    project_scope: Optional[str] = Query(default=None, description="Optional project scope used during search"),
):
    """Download merged PDF containing pages labeled as 'relevant' or 'best'."""
    return _build_feedback_pages_pdf(
        query=query,
        project_scope=project_scope,
        accepted_labels={"best", "relevant"},
        empty_detail="No pages labeled 'relevant' or 'best' for this query/scope",
        output_prefix="relevant_best_pages",
    )


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search for structural engineering content in the Mar Vista POC project.
    
    **Project Filter:** Results are limited to "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
    
    **Parameters:**
    - **query**: Search text (keywords, phrases, technical terms, or natural language questions)
    - **k**: Number of results to return (default: 20, max: 100)
    
    **Returns:**
    - Ranked search results with project details, **page numbers**, and topology elements
    - Project summaries with list of relevant pages
    - AI-generated summary of findings
    
    **Example Queries:**
    - "steel shear key"
    - "pipe pin"
    - "CIDH pile foundations"
    - "bearing pad specifications"
    """
    if not search_agent:
        raise HTTPException(status_code=503, detail="Search agent not initialized")
    
    try:
        # Execute search
        search_results = search_agent.search(
            query=request.query,
            k=request.k,
            generate_summary=request.generate_summary,
            use_hybrid_search=request.use_hybrid_search,
            project_scope=request.project_scope,
        )
        
        # Extract topology elements from content
        topology_keywords = {
            "Pipe Pin (Steel Shear Key)": ["pipe pin", "steel shear key"],
            "Shear Key": ["shear key"],
            "CIDH Piles/Shafts": ["cidh", "cast-in-drilled-hole", "drilled shaft"],
            "Bearing Pad": ["bearing pad"],
            "Abutment": ["abutment"],
            "Bent": ["bent"],
            "End Diaphragm": ["end diaphragm", "diaphragm"],
            "Retaining Wall": ["retaining wall", "shga", "ground anchor"],
            "Foundation": ["foundation", "footing"],
            "Reinforcement": ["reinforcement", "rebar"],
            "Bridge Girder": ["girder", "beam"],
            "Column": ["column"],
        }
        
        def extract_topology(content: str) -> List[str]:
            """Extract topology elements from content."""
            content_lower = content.lower()
            found_elements = []
            for element, keywords in topology_keywords.items():
                if any(kw in content_lower for kw in keywords):
                    found_elements.append(element)
            return found_elements
        
        # Build detailed results - convert project results to page-level results
        results = []
        projects_dict = {}
        
        for project_result in search_results.get('results', []):
            file_name = project_result.get('file_name', '')
            
            # Get project metadata
            metadata = metadata_manager.get_project_metadata(file_name)
            project_name = metadata.get('project_name', file_name.replace('.pdf', '')) if metadata else file_name.replace('.pdf', '')
            
            # Get chunks with page-specific content and scores
            chunks = project_result.get('chunks', [])
            
            # Group chunks by page and keep the best score per page
            page_data = {}
            for chunk in chunks:
                page_num = chunk.get('page')
                if page_num is None:
                    continue
                    
                content = chunk.get('content', '')
                score = chunk.get('score', 1.0)
                
                # Keep the best (lowest) score and longest content for each page
                if page_num not in page_data or score < page_data[page_num]['score']:
                    page_data[page_num] = {
                        'content': content,
                        'score': score,
                        'has_vision': '[DRAWING ANALYSIS]' in content or '[TEXT CONTENT]' in content,
                        'topology': extract_topology(content)
                    }
            
            # Create a result entry for EACH page with its specific data
            for page_num, data in page_data.items():
                result_entry = SearchResult(
                    pdf_file_name=file_name,
                    project_name=project_name,
                    page_number=page_num,
                    relevance_score=round(data['score'], 3),
                    topology_elements=data['topology'],
                    content_sample=data['content'][:500].replace('[TEXT CONTENT]', '').replace('[DRAWING ANALYSIS]', '').strip(),
                    has_vision_analysis=data['has_vision']
                )
                results.append(result_entry)
                
                # Build project summary tracking
                if file_name not in projects_dict:
                    projects_dict[file_name] = {
                        'metadata': metadata,
                        'pages': set()
                    }
                projects_dict[file_name]['pages'].add(page_num)
        
        # Apply relevance threshold filter (adaptive default if not explicitly provided)
        if results:
            effective_threshold = request.relevance_threshold
            if effective_threshold is None:
                effective_threshold = (
                    config.HYBRID_DEFAULT_RELEVANCE_THRESHOLD
                    if request.use_hybrid_search
                    else config.VECTOR_DEFAULT_RELEVANCE_THRESHOLD
                )

            best_score = min(r.relevance_score for r in results)
            threshold = best_score + effective_threshold
            results = [r for r in results if r.relevance_score <= threshold]
        
        # Build project summaries
        project_summaries = []
        for file_name, data in projects_dict.items():
            metadata = data['metadata']
            if metadata:
                summary = ProjectSummary(
                    pdf_file_name=file_name,
                    project_name=metadata.get('project_name', file_name.replace('.pdf', '')),
                    phase=metadata.get('phase'),
                    engineer_of_record=metadata.get('engineer_of_record'),
                    date=metadata.get('date'),
                    categories=metadata.get('categories', []),
                    relevant_pages=sorted(list(data['pages'])),
                    total_pages=metadata.get('total_pages', 0)
                )
                project_summaries.append(summary)
        
        # Build response
        response = SearchResponse(
            query=request.query,
            total_results=len(results),
            projects_found=len(project_summaries),
            results=results,
            project_summaries=project_summaries,
            search_summary=search_results.get('summary', 'No summary available'),
            timestamp=datetime.utcnow().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(request: AnalyzeRequest):
    """
    Analyze a structural question with semantic query understanding.

    This endpoint performs semantic analysis to:
    - Infer what the user is really asking for (intent extraction)
    - Find structural components and terminology synonyms
    - Rank results by semantic relevance to the inferred intent
    - Explain why each result matches the query
    """
    if not search_agent or not analysis_engine:
        raise HTTPException(status_code=503, detail="Analysis components not initialized")

    try:
        search_response = await search(
            SearchRequest(
                query=request.query,
                k=request.k,
                relevance_threshold=request.relevance_threshold,
                project_scope=request.project_scope,
                generate_summary=True,
                use_hybrid_search=request.use_hybrid_search,
            )
        )

        normalized_results = [item.model_dump() for item in search_response.results]
        analysis = analysis_engine.build_analysis(
            query=request.query,
            normalized_results=normalized_results,
            search_summary=search_response.search_summary,
        )

        inferred_intent = analysis["inferred_intent"]
        semantic_evidence_data = analysis["semantic_evidence"]

        return AnalyzeResponse(
            query=request.query,
            answer=analysis["answer"],
            inferred_intent=InferredIntent(**inferred_intent),
            intent_confidence=analysis["intent_confidence"],
            semantic_evidence=[AnalysisEvidence(**ev) for ev in semantic_evidence_data],
            terminology_variants=analysis["terminology_variants"],
            search_summary=analysis.get("search_summary"),
            timestamp=datetime.utcnow().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/projects", tags=["Projects"])
async def list_projects():
    """
    List all indexed projects in the knowledge base.
    
    Returns basic information about each project including file name, categories, and page count.
    """
    if not metadata_manager:
        raise HTTPException(status_code=503, detail="Metadata manager not initialized")
    
    try:
        all_projects = metadata_manager.get_all_projects()
        return {
            "total_projects": len(all_projects),
            "projects": [
                {
                    "pdf_file_name": proj['file_name'],
                    "project_name": proj['project_name'],
                    "phase": proj.get('phase'),
                    "engineer_of_record": proj.get('engineer_of_record'),
                    "categories": proj.get('categories', []),
                    "total_pages": proj['total_pages'],
                    "indexed_at": proj['indexed_at']
                }
                for proj in all_projects.values()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {str(e)}")


@app.get("/projects/{file_name}", tags=["Projects"])
async def get_project_details(file_name: str):
    """
    Get detailed information about a specific project.
    
    **Parameters:**
    - **file_name**: The PDF file name (e.g., "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf")
    """
    if not metadata_manager:
        raise HTTPException(status_code=503, detail="Metadata manager not initialized")
    
    try:
        metadata = metadata_manager.get_project_metadata(file_name)
        if not metadata:
            raise HTTPException(status_code=404, detail=f"Project not found: {file_name}")
        
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get project details: {str(e)}")


@app.get("/categories", tags=["Categories"])
async def list_categories():
    """
    List all structural engineering categories and the projects in each category.
    """
    if not metadata_manager:
        raise HTTPException(status_code=503, detail="Metadata manager not initialized")
    
    try:
        from config import STRUCTURAL_CATEGORIES
        
        categories_data = {}
        for category in STRUCTURAL_CATEGORIES:
            projects = metadata_manager.search_projects_by_category(category)
            categories_data[category] = {
                "project_count": len(projects),
                "projects": [
                    {
                        "file_name": proj['file_name'],
                        "project_name": proj['project_name'],
                        "total_pages": proj['total_pages']
                    }
                    for proj in projects
                ]
            }
        
        return {
            "total_categories": len(STRUCTURAL_CATEGORIES),
            "categories": categories_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list categories: {str(e)}")


@app.get("/pdf-page/{file_name}/{page_number}", tags=["PDF Pages"])
async def get_pdf_page_image(
    file_name: str,
    page_number: int,
    size: str = Query(default="medium", regex="^(thumbnail|medium|full)$")
):
    """
    Render a specific PDF page as a PNG image.
    
    - **file_name**: Name of the PDF file
    - **page_number**: Page number (1-indexed)
    - **size**: Image size - 'thumbnail' (200px width), 'medium' (600px width), 'full' (original resolution)
    """
    try:
        # Check if PDF exists
        if not storage.pdf_exists(file_name):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {file_name}")
        
        # Get PDF path (downloads from GCS if in production)
        pdf_path = storage.get_pdf_temp_path(file_name)
        
        # Open PDF and get page
        doc = fitz.open(pdf_path)
        if page_number < 1 or page_number > len(doc):
            doc.close()
            raise HTTPException(status_code=400, detail=f"Invalid page number. PDF has {len(doc)} pages.")
        
        # Get page (convert from 1-indexed to 0-indexed)
        page = doc[page_number - 1]
        
        # Determine zoom level based on size
        if size == "thumbnail":
            zoom = 200 / page.rect.width  # 200px width
        elif size == "medium":
            zoom = 600 / page.rect.width  # 600px width
        else:  # full
            zoom = 2.0  # 2x for high quality
        
        # Render page to pixmap
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PNG bytes
        img_bytes = pix.tobytes("png")
        doc.close()
        
        # Return as streaming response
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF page: {str(e)}")


# Run server
if __name__ == "__main__":
    print("="*80)
    print("STRUCTURAL ENGINEERING LIBRARIAN API")
    print("="*80)
    print("\nStarting API server...")
    print("[DOC] API Documentation: http://localhost:8000/docs")
    print("[SCHEMA] OpenAPI Schema: http://localhost:8000/openapi.json")
    print("[SEARCH] Try searching: POST http://localhost:8000/search")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

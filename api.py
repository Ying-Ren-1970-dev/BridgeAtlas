"""
OpenAPI REST API for Librarian Search System
Provides search functionality over structural engineering PDF documents.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from datetime import datetime
import fitz  # PyMuPDF
from io import BytesIO
import os

from search_agent import SearchAgent
from vector_store import VectorStore
from metadata_manager import MetadataManager

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


@app.on_event("startup")
async def startup_event():
    """Initialize search components on startup."""
    global search_agent, vector_store, metadata_manager
    search_agent = SearchAgent()
    vector_store = VectorStore()
    vector_store.initialize_vectorstore()
    metadata_manager = MetadataManager()
    print("✓ Librarian API initialized successfully")


# Request/Response Models
class SearchRequest(BaseModel):
    """Search query request model."""
    query: str = Field(..., description="Search query text (keywords, phrases, or questions)", example="steel shear key")
    k: int = Field(default=50, ge=1, le=100, description="Number of results to return (1-100)")
    relevance_threshold: float = Field(default=0.5, ge=0.0, le=2.0, description="Maximum score difference from best result (0.0 = only best, 2.0 = very permissive)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "steel shear key",
                "k": 50,
                "relevance_threshold": 0.5
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


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "vector_store": "operational" if vector_store else "not initialized",
        "search_agent": "operational" if search_agent else "not initialized"
    }


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
        search_results = search_agent.search(query=request.query, k=request.k)
        
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
        
        # Apply relevance threshold filter
        if results:
            best_score = min(r.relevance_score for r in results)
            threshold = best_score + request.relevance_threshold
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
        # Find PDF file in Projects/ directory
        pdf_path = os.path.join("Projects", file_name)
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {file_name}")
        
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
    print("📖 API Documentation: http://localhost:8000/docs")
    print("📋 OpenAPI Schema: http://localhost:8000/openapi.json")
    print("🔍 Try searching: POST http://localhost:8000/search")
    print("\nPress CTRL+C to stop the server")
    print("="*80 + "\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Structural Engineering Librarian API

OpenAPI REST API for searching structural engineering PDF documents using RAG-powered semantic search.

## Features

- **Semantic Search**: Natural language queries over structural engineering documents
- **Topology Recognition**: Automatically identifies structural elements (pipe pins, shear keys, CIDH piles, etc.)
- **Vision-Enhanced Results**: Indicates which pages were analyzed using GPT-4 Vision
- **Project Management**: List and filter projects by category, engineer, or phase
- **Interactive Documentation**: Auto-generated OpenAPI/Swagger docs

## Installation

### 1. Install API Dependencies

```powershell
pip install -r requirements_api.txt
```

This installs:
- `fastapi` - Modern web framework
- `uvicorn` - ASGI server
- `pydantic` - Data validation
- `python-multipart` - Form data support

### 2. Ensure Main Dependencies Are Installed

```powershell
pip install -r requirements.txt
```

## Running the API

### Start the Server

```powershell
python api.py
```

The API will start on `http://localhost:8000`

### Access Documentation

- **Interactive API Docs**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 🔍 Search

**POST** `/search`

Search for content across all indexed PDF documents.

**Request Body:**
```json
{
  "query": "steel shear key",
  "k": 20
}
```

**Response:**
```json
{
  "query": "steel shear key",
  "total_results": 15,
  "projects_found": 3,
  "results": [
    {
      "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
      "project_name": "Mar Vista POC",
      "page_number": 11,
      "relevance_score": 0.850,
      "topology_elements": [
        "Pipe Pin (Steel Shear Key)",
        "Abutment",
        "Bearing Pad"
      ],
      "content_sample": "Pipe Pin Detail: Includes cover pipe...",
      "has_vision_analysis": true
    }
  ],
  "project_summaries": [
    {
      "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
      "project_name": "Mar Vista POC",
      "phase": null,
      "engineer_of_record": null,
      "categories": ["Bridges", "Retaining Walls"],
      "relevant_pages": [5, 10, 11, 18, 19],
      "total_pages": 42
    }
  ],
  "search_summary": "Found steel shear key details in 3 projects...",
  "timestamp": "2026-05-10T22:00:00Z"
}
```

**Parameters:**
- `query` (required): Search text - keywords, phrases, or natural language questions
- `k` (optional): Number of results to return (1-100, default: 20)

**Example Queries:**
- `"steel shear key"`
- `"CIDH pile foundations"`
- `"retaining wall with ground anchors"`
- `"What are the bearing pad specifications?"`

### 📚 Projects

**GET** `/projects`

List all indexed projects.

**Response:**
```json
{
  "total_projects": 8,
  "projects": [
    {
      "pdf_file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
      "project_name": "Mar Vista POC",
      "phase": null,
      "engineer_of_record": null,
      "categories": ["Bridges", "Retaining Walls", "General Structural"],
      "total_pages": 42,
      "indexed_at": "2026-05-10T21:32:38.634952"
    }
  ]
}
```

**GET** `/projects/{file_name}`

Get detailed information about a specific project.

**Response:**
```json
{
  "file_name": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
  "project_name": "Mar Vista POC",
  "phase": null,
  "engineer_of_record": null,
  "date": null,
  "categories": ["Bridges", "Retaining Walls", "General Structural"],
  "total_pages": 42,
  "indexed_at": "2026-05-10T21:32:38.634952"
}
```

### 🏗️ Categories

**GET** `/categories`

List all structural engineering categories with projects.

**Response:**
```json
{
  "total_categories": 10,
  "categories": {
    "Bridges": {
      "project_count": 3,
      "projects": [...]
    },
    "Retaining Walls": {
      "project_count": 2,
      "projects": [...]
    }
  }
}
```

### ❤️ Health

**GET** `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-05-10T22:00:00Z",
  "vector_store": "operational",
  "search_agent": "operational"
}
```

## Usage Examples

### cURL

```bash
# Search for steel shear keys
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "steel shear key", "k": 20}'

# List all projects
curl "http://localhost:8000/projects"

# Get project details
curl "http://localhost:8000/projects/Mar%20Vista%20POC%20100%25_CheckPrint_20211118%20Complete.pdf"
```

### Python

```python
import requests

# Search
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "steel shear key", "k": 20}
)
results = response.json()

# Print results
for result in results['results']:
    print(f"{result['project_name']} - Page {result['page_number']}")
    print(f"  Topology: {', '.join(result['topology_elements'])}")
    print(f"  Score: {result['relevance_score']}")
```

### JavaScript/TypeScript

```javascript
// Search
const response = await fetch('http://localhost:8000/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query: 'steel shear key', k: 20 })
});

const data = await response.json();
console.log(`Found ${data.total_results} results in ${data.projects_found} projects`);
```

## Response Fields

### SearchResult

- **pdf_file_name**: Name of the PDF file
- **project_name**: Human-readable project name
- **page_number**: Page number within the PDF (1-indexed)
- **relevance_score**: Relevance score (lower is better, typically 0.0-2.0)
- **topology_elements**: List of structural elements found (e.g., "Pipe Pin (Steel Shear Key)")
- **content_sample**: First 500 characters of relevant content
- **has_vision_analysis**: Whether GPT-4 Vision analyzed this page

### Topology Elements

The API automatically identifies these structural elements:

- Pipe Pin (Steel Shear Key)
- Shear Key
- CIDH Piles/Shafts
- Bearing Pad
- Abutment
- Bent
- End Diaphragm
- Retaining Wall
- Foundation
- Reinforcement
- Bridge Girder
- Column

## Development

### Auto-Reload Mode

The server runs in auto-reload mode by default, automatically restarting when code changes.

### Custom Host/Port

Edit `api.py`:

```python
uvicorn.run(
    "api:app",
    host="0.0.0.0",  # Change to "127.0.0.1" for localhost only
    port=8000,       # Change port number
    reload=True
)
```

### Production Deployment

For production, use a production ASGI server:

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## CORS Configuration

The API allows cross-origin requests from any origin by default. For production, update the CORS settings in `api.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-domain.com"],  # Specific origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Error Handling

The API returns standard HTTP status codes:

- **200 OK**: Successful request
- **404 Not Found**: Project not found
- **500 Internal Server Error**: Search or processing error
- **503 Service Unavailable**: Components not initialized

Error responses include a detail message:

```json
{
  "detail": "Project not found: NonExistent.pdf"
}
```

## Performance

- Initial startup: ~2-5 seconds (loads vector store)
- Search queries: ~1-3 seconds depending on result count
- Concurrent requests: Supported via async/await

## Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/REST
       │
┌──────▼──────┐
│  FastAPI    │
│   Server    │
└──────┬──────┘
       │
    ┌──┴───────────────┐
    │                  │
┌───▼────┐      ┌──────▼────────┐
│ Search │      │   Metadata    │
│ Agent  │      │   Manager     │
└───┬────┘      └───────────────┘
    │
┌───▼─────────┐
│   Vector    │
│   Store     │
│  (ChromaDB) │
└─────────────┘
```

## Next Steps

1. **Test the API**: Open http://localhost:8000/docs and try the interactive interface
2. **Integrate**: Use the API in your web applications or scripts
3. **Customize**: Modify search parameters or add new endpoints as needed

## Support

For issues or questions:
- Check the interactive docs at `/docs`
- Review the OpenAPI schema at `/openapi.json`
- Check the health endpoint at `/health`

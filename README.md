# Librarian - Structural Engineering Project Search System

A RAG (Retrieval Augmented Generation) powered knowledge base system for searching and analyzing structural engineering project PDF documents using OpenAI GPT-4.

## Features

- **PDF Processing**: Automatically extracts text and metadata from structural engineering plan sets
- **Semantic Search**: AI-powered search using vector embeddings for finding relevant content
- **Auto-Categorization**: Automatically categorizes projects using structural engineering taxonomy
- **Metadata Extraction**: Extracts project name, phase, engineer of record, dates, and more
- **Page-Level Results**: Returns specific page numbers within PDFs that are relevant to searches
- **Drawing-Region Chunking**: One vector chunk per labeled CAD drawing (plan, section, detail, etc.) — see [DRAWING_REGION_CHUNKING.md](DRAWING_REGION_CHUNKING.md)
- **Interactive CLI**: User-friendly command-line interface for searching and managing the knowledge base

## Project Structure

```
Librarian/
├── Projects/                    # Your PDF files go here
│   ├── 25th ave/
│   │   └── *.pdf
│   └── *.pdf
├── config.py                    # Configuration settings
├── pdf_processor.py             # PDF text extraction and processing
├── vector_store.py              # Vector database and embeddings
├── categorizer.py               # Project categorization
├── metadata_manager.py          # Metadata storage and retrieval
├── search_agent.py              # Intelligent search agent
├── main.py                      # Main application logic
├── cli.py                       # Interactive CLI
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Setup Steps

1. **Clone or navigate to the project directory**:
   ```powershell
   cd "C:\Users\alexl\Desktop\xttribute\Librarian"
   ```

2. **Create a virtual environment** (recommended):
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   - Copy `.env.example` to `.env`:
     ```powershell
     Copy-Item .env.example .env
     ```
   - Edit `.env` and add your OpenAI API key:
     ```
     OPENAI_API_KEY=sk-your-api-key-here
     OPENAI_MODEL=gpt-4
     ```

5. **Verify Projects folder**:
   - Ensure your PDF files are in `C:\Users\alexl\Desktop\xttribute\Librarian\Projects`
   - The system will process all PDF files in this folder and its subfolders

## Usage

### Interactive CLI (Recommended)

Launch the interactive command-line interface:

```powershell
python cli.py
```

#### Available Commands

**Build Knowledge Base:**
```
Librarian> build
Librarian> build --clear   # Clear existing data first
```

**Search for Projects:**
```
Librarian> search retaining wall foundation design
Librarian> search tunnel boring seismic
Librarian> search bridge girder connections
```

**Advanced Search with Filters:**
```
Librarian> filter bridge design --category=Bridges
Librarian> filter seismic --phase=100% Final
Librarian> filter foundation --engineer=Smith
```

**View Information:**
```
Librarian> list                  # List all projects
Librarian> stats                 # Show statistics
Librarian> details <filename>    # Show project details
Librarian> categories            # List all categories
Librarian> phases                # List all phases
```

### Command Line Interface

You can also use direct commands:

**Build the knowledge base:**
```powershell
python main.py build
python main.py build --clear  # Clear existing data first
```

**Search:**
```powershell
python main.py search "retaining wall foundation"
python main.py search "tunnel seismic design"
```

**List all projects:**
```powershell
python main.py list
```

**Show statistics:**
```powershell
python main.py stats
```

**Get project details:**
```powershell
python main.py details "30%_Elk Grove Station Structure Plan Set.pdf"
```

## How It Works

### 1. Building the Knowledge Base

When you run `build`, the system:

1. **Scans** all PDF files in the Projects folder
2. **Extracts** text from each page using advanced PDF processing
3. **Chunks** the text into manageable segments
4. **Generates** vector embeddings using OpenAI's embedding model
5. **Categorizes** each project using AI (Bridges, Tunnels, Stations, etc.)
6. **Extracts** metadata (project name, phase, engineer, dates)
7. **Stores** everything in a searchable vector database

### 2. Searching

When you search:

1. **Query Embedding**: Your search text is converted to a vector embedding
2. **Semantic Search**: The system finds the most similar chunks across all documents
3. **Result Aggregation**: Results are organized by project and page
4. **Metadata Enrichment**: Each result includes full project metadata
5. **AI Summary**: GPT-4 generates a natural language summary of findings

### 3. Search Results

Each search result includes:

- **Project Name**: Extracted from filename or document
- **Phase**: 30% Design, 100% Final, IFB, etc.
- **Engineer of Record**: Name of engineering firm
- **File Name**: Original PDF filename
- **Relevant Pages**: Specific page numbers containing your search terms
- **Date**: Project date
- **Categories**: Structural engineering categories

## Categories

Projects are automatically categorized into:

- Bridges
- Tunnels
- Stations
- Retaining Walls
- Pedestrian Structures
- Overhead Structures
- Foundations
- Seismic Design
- Structural Analysis
- General Structural

## Supported Plan Phases

- Conceptual
- Preliminary
- 30% Design
- 60% Design
- 90% Design
- 100% Final
- IFB (Invitation for Bid)
- As-Built
- Markup

## Configuration

Edit `config.py` or `.env` to customize:

- **OPENAI_API_KEY**: Your OpenAI API key
- **OPENAI_MODEL**: LLM model to use (default: gpt-4)
- **PROJECTS_FOLDER**: Path to your PDF files
- **CHUNK_SIZE**: Text chunk size for embeddings (default: 1000)
- **CHUNK_OVERLAP**: Overlap between chunks (default: 200)

## Example Searches

```
search foundation design seismic
search retaining wall soil pressure
search bridge deck reinforcement
search tunnel boring groundwater
search station platform structural support
search pedestrian ramp ADA compliance
```

## Troubleshooting

### No results found

- Ensure the knowledge base has been built (`build` command)
- Try broader search terms
- Check that PDF files are in the Projects folder

### OpenAI API errors

- Verify your API key in `.env` file
- Check your OpenAI account has sufficient credits
- Ensure you have access to GPT-4 model

### PDF processing issues

- Some PDFs may be scanned images without text - consider OCR preprocessing
- Very large PDFs may take longer to process
- Check console output for specific error messages

## Performance Tips

- **Initial Build**: First-time building may take several minutes depending on the number of PDFs
- **Incremental Updates**: Use `build` without `--clear` to add new documents
- **Search Speed**: Searches are typically very fast (< 2 seconds)
- **API Costs**: Building the knowledge base makes multiple API calls; monitor your OpenAI usage

## Technology Stack

- **OpenAI GPT-4**: Large language model for categorization and summaries
- **OpenAI Embeddings**: text-embedding-3-small for vector embeddings
- **ChromaDB**: Vector database for semantic search
- **LangChain**: Framework for RAG implementation
- **PyPDF2/pdfplumber**: PDF text extraction
- **Python 3.8+**: Core programming language

## Future Enhancements

- OCR support for scanned documents
- Web-based user interface
- Multi-user support
- Document comparison features
- Export search results to Excel/PDF
- Advanced filtering and sorting options

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review console output for error messages
3. Verify all dependencies are installed correctly
4. Ensure OpenAI API key is valid

## License

This project is for internal use in structural engineering project management.

---

**Version**: 1.0  
**Last Updated**: May 2026  
**Author**: Librarian Development Team

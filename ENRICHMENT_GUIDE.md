# Knowledge Base Enhancement - Page Classification & Title Block Extraction

## Overview

This enhancement adds intelligent page classification and title block extraction to the Librarian system using GPT-4 Vision.

## Features

### 1. Page Classification
- Automatically classifies each page as:
  - **report**: Text-heavy reports, calculations, specifications
  - **plan**: CAD drawings, structural plans, foundation plans, details

### 2. Title Block Extraction (for Plans)
Extracts structured information from title blocks:

**Project Information:**
- Project name, number, location
- Client name
- Engineer on record
- Engineering firm
- Project year

**Plan Type:**
- Primary type (Foundation Plan, Structural Plan, etc.)
- Sheet title and number
- Scale, revision, date

**Plan Contents:**
- Tables and diagrams detection
- Structural elements (piles, footings, columns, etc.)
- Detail types
- Grid references (bent numbers, stations, etc.)

## Usage

### Enrich a Single PDF

```powershell
# Enrich metadata for a specific file
python main.py enrich "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"
```

### Enrich All Projects

```powershell
# Get list of all PDFs in Projects folder
$pdfs = Get-ChildItem -Path "Projects" -Filter "*.pdf" -Recurse

# Enrich each PDF
foreach ($pdf in $pdfs) {
    python main.py enrich $pdf.FullName
}
```

### Output

The enriched metadata is saved to: `data/enriched_<filename>.json`

Example output:
```json
[
  {
    "page_type": "plan",
    "confidence": 0.95,
    "page_number": 5,
    "filename": "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf",
    "title_block": {
      "project_info": {
        "project_name": "MAR VISTA PEDESTRIAN OVERCROSSING",
        "project_number": "RH-2021-001",
        "client": "City of Los Angeles",
        "engineer_on_record": "John Smith, PE",
        "engineering_firm": "ABC Engineering",
        "project_year": "2021"
      },
      "plan_type": {
        "primary_type": "Foundation Plan",
        "sheet_title": "FOUNDATION PLAN - BENT 5",
        "sheet_number": "S-5",
        "scale": "1/4\" = 1'-0\"",
        "revision": "2",
        "date": "11/18/2021"
      },
      "plan_contents": {
        "has_tables": true,
        "has_diagrams": false,
        "structural_elements": [
          "CIDH pile",
          "pile cap",
          "column",
          "footing",
          "shear key"
        ],
        "detail_types": [
          "pile cap detail",
          "column connection detail"
        ],
        "grid_references": [
          "Bent 5",
          "Station 10+25"
        ]
      }
    }
  }
]
```

## API Usage

### Programmatic Access

```python
from pathlib import Path
from page_classifier import PageClassifier, classify_page_type

# Initialize classifier
classifier = PageClassifier()

# Classify a single page
result = classify_page_type(
    pdf_path=Path("Projects/project.pdf"),
    page_num=10
)

print(f"Page Type: {result['page_type']}")
print(f"Confidence: {result['confidence']}")

# Batch classify multiple pages
results = classifier.batch_classify_pdf(
    pdf_path=Path("Projects/project.pdf"),
    sample_pages=[1, 5, 10, 15]  # Specific pages, or None for auto-sample
)

for result in results:
    if result['page_type'] == 'plan':
        title_block = result['title_block']
        print(f"Page {result['page_number']}: {title_block['plan_type']['sheet_title']}")
```

## Cost Efficiency

The system intelligently samples pages to minimize API costs:

- **Small PDFs (≤15 pages):** Analyzes all pages
- **Large PDFs (>15 pages):** Samples first 5, middle 5, and last 5 pages
- **Custom sampling:** Specify exact pages to analyze

## Integration with Vector Store

To integrate this enriched metadata into your vector store, you can modify `vector_store.py` to include the page classification and title block data in chunk metadata:

```python
chunk_metadata = {
    'file_name': file_name,
    'page': chunk['page'],
    'page_type': page_classification['page_type'],  # NEW
    'sheet_number': title_block['plan_type']['sheet_number'],  # NEW
    'structural_elements': title_block['plan_contents']['structural_elements'],  # NEW
    # ... existing metadata
}
```

This enables searching by:
- Page type (plans only, reports only)
- Specific sheet numbers
- Structural elements present on the page
- Project information from title blocks

## Example Queries with Enhanced Metadata

Once integrated:

```python
# Find all foundation plan pages
results = vector_store.similarity_search(
    query="foundation design",
    filter_dict={'page_type': 'plan', 'sheet_number': 'S-5'}
)

# Find pages with CIDH piles
results = vector_store.similarity_search(
    query="pile design",
    filter_dict={'structural_elements': {'$contains': 'CIDH pile'}}
)
```

## Next Steps

1. **Run enrichment** on your existing projects
2. **Review** the extracted data in the JSON files
3. **Integrate** into vector store metadata (optional)
4. **Enhance search** with type and content filters (optional)

## Notes

- Uses GPT-4o Vision API (charges per image)
- Processes ~1-2 pages per second
- Saves progress incrementally
- JSON output is human-readable and machine-parsable

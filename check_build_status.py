import json
from pathlib import Path

# Check metadata_db.json
metadata_file = Path("metadata_db.json")
if metadata_file.exists():
    with open(metadata_file) as f:
        data = json.load(f)
    
    projects = data.get('projects', {})
    print(f"=== Build Progress ===\n")
    print(f"Projects indexed: {len(projects)} of 11")
    print(f"Last updated: {data.get('last_updated', 'Unknown')}\n")
    
    if projects:
        print("Projects completed:")
        for i, (name, info) in enumerate(sorted(projects.items()), 1):
            pages = info.get('total_pages', '?')
            indexed_at = info.get('indexed_at', 'Unknown')
            print(f"  {i}. {name} ({pages} pages) - {indexed_at}")
    
    # Check expected PDFs
    projects_dir = Path("Projects/25th ave")
    if projects_dir.exists():
        pdf_files = list(projects_dir.glob("*.pdf"))
        print(f"\n\nTotal PDFs in folder: {len(pdf_files)}")
        print(f"Still to process: {len(pdf_files) - len(projects)}")
else:
    print("metadata_db.json not found!")

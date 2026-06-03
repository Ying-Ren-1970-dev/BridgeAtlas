import json
from pathlib import Path

# Load metadata database
metadata_file = Path("metadata_db.json")
with open(metadata_file) as f:
    data = json.load(f)

# Check for orphaned projects (files that don't exist)
projects = data.get('projects', {})
projects_dir = Path("Projects")

orphaned = []
for filename, info in list(projects.items()):
    file_path = projects_dir / filename
    if not file_path.exists():
        orphaned.append(filename)
        print(f"Found orphaned project: {filename}")
        print(f"  Expected path: {file_path}")
        print(f"  Indexed at: {info.get('indexed_at', 'Unknown')}")

if orphaned:
    print(f"\nRemoving {len(orphaned)} orphaned project(s) from metadata...")
    for filename in orphaned:
        del projects[filename]
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Metadata cleaned. {len(projects)} projects remaining.")
else:
    print("\n✓ No orphaned projects found. Metadata is clean.")

# Show current projects
print(f"\nCurrent projects ({len(projects)}):")
for i, name in enumerate(sorted(projects.keys()), 1):
    print(f"  {i}. {name}")

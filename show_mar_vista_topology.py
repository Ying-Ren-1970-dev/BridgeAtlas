"""Display Mar Vista project topology structure and categorization."""
import json
from vector_store import VectorStore

# Load metadata
with open('metadata_db.json', 'r') as f:
    metadata = json.load(f)

# Find Mar Vista project
mar_vista = None
for file_name, project in metadata['projects'].items():
    if 'Mar Vista' in file_name:
        mar_vista = project
        break

if not mar_vista:
    print("Mar Vista project not found!")
    exit(1)

print("="*80)
print("MAR VISTA PROJECT - TOPOLOGY & CATEGORIZATION")
print("="*80)

print("\n📋 PROJECT METADATA:")
print(f"  File Name:          {mar_vista['file_name']}")
print(f"  Project Name:       {mar_vista['project_name']}")
print(f"  Phase:              {mar_vista.get('phase', 'None')}")
print(f"  Engineer:           {mar_vista.get('engineer_of_record', 'None')}")
print(f"  Date:               {mar_vista.get('date', 'None')}")
print(f"  Total Pages:        {mar_vista['total_pages']}")
print(f"  Indexed:            {mar_vista['indexed_at']}")

print("\n🏗️ STRUCTURAL CATEGORIES:")
for category in mar_vista['categories']:
    print(f"  ✓ {category}")

print("\n" + "="*80)
print("EXTRACTED STRUCTURAL TOPOLOGY - PAGE BY PAGE")
print("="*80)

# Initialize vector store to get page content
vs = VectorStore()
vs.initialize_vectorstore()

# Get all chunks for Mar Vista
results = vs.similarity_search_with_scores(
    query="structural elements pipe pin shear key CIDH foundation",
    k=200
)

# Organize by page and extract structural elements
pages_data = {}
for doc, score in results:
    if doc.metadata.get('file_name') == mar_vista['file_name']:
        page_num = doc.metadata.get('page', 0)
        content = doc.page_content
        
        if page_num not in pages_data:
            pages_data[page_num] = {
                'elements': set(),
                'has_vision': False,
                'has_text': False
            }
        
        # Check for vision analysis
        if '[DRAWING ANALYSIS]' in content:
            pages_data[page_num]['has_vision'] = True
        elif '[TEXT CONTENT]' in content:
            pages_data[page_num]['has_text'] = True
        else:
            pages_data[page_num]['has_text'] = True
        
        # Extract structural elements from content
        content_lower = content.lower()
        
        # Key structural elements to look for
        element_keywords = {
            'Pipe Pin (Steel Shear Key)': ['pipe pin', 'steel shear key'],
            'Shear Key': ['shear key'],
            'CIDH Piles/Shafts': ['cidh', 'cast-in-drilled-hole', 'drilled shaft'],
            'Bearing Pad': ['bearing pad'],
            'Abutment': ['abutment'],
            'Bent': ['bent'],
            'End Diaphragm': ['end diaphragm', 'diaphragm'],
            'Retaining Wall': ['retaining wall', 'shga', 'ground anchor'],
            'Foundation': ['foundation', 'footing'],
            'Reinforcement': ['reinforcement', 'rebar'],
            'Bridge Girder': ['girder', 'beam'],
            'Column': ['column'],
        }
        
        for element, keywords in element_keywords.items():
            if any(kw in content_lower for kw in keywords):
                pages_data[page_num]['elements'].add(element)

# Display by page
print("\nKey pages with structural topology:")
print()

important_pages = sorted([p for p in pages_data.keys() if pages_data[p]['elements']])[:15]

for page_num in important_pages:
    data = pages_data[page_num]
    elements = sorted(data['elements'])
    
    if not elements:
        continue
    
    analysis_type = []
    if data['has_vision']:
        analysis_type.append("🎯 Vision")
    if data['has_text']:
        analysis_type.append("📄 Text")
    
    print(f"Page {page_num:2d} [{' + '.join(analysis_type)}]")
    for element in elements:
        print(f"  • {element}")
    print()

# Summary statistics
print("="*80)
print("TOPOLOGY SUMMARY")
print("="*80)

all_elements = set()
for data in pages_data.values():
    all_elements.update(data['elements'])

print(f"\nTotal unique structural element types identified: {len(all_elements)}")
print("\nStructural Elements Found:")
for element in sorted(all_elements):
    pages_with_element = [p for p in pages_data if element in pages_data[p]['elements']]
    print(f"  • {element:30s} (found on {len(pages_with_element)} pages)")

print(f"\nTotal pages with extracted content: {len(pages_data)}")
vision_pages = len([p for p in pages_data.values() if p['has_vision']])
print(f"Pages with vision analysis: {vision_pages}")

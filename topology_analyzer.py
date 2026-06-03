"""
Deep Topology Analyzer - Enhanced structural element extraction using GPT-4 Vision and Text Analysis.

This module performs comprehensive analysis of structural engineering PDFs to extract:
- Detailed structural elements and their properties
- Relationships between elements
- Spatial information and locations
- Material specifications
- Dimensional data
- Connection details
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import openai
from collections import defaultdict
from tqdm import tqdm

import config
from vision_analyzer import VisionAnalyzer
from pdf_processor import PDFProcessor


class DeepTopologyAnalyzer:
    """Enhanced topology extraction using GPT-4 Vision and comprehensive text analysis."""
    
    def __init__(self):
        """Initialize the deep topology analyzer."""
        self.client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.vision_analyzer = VisionAnalyzer()
        self.pdf_processor = PDFProcessor(use_vision=True)
        
        # Enhanced topology categories
        self.topology_categories = {
            "structural_elements": [],
            "foundation_systems": [],
            "connection_types": [],
            "materials": [],
            "dimensions": [],
            "locations": [],
            "specifications": [],
            "details": []
        }
    
    def analyze_project_comprehensive(self, pdf_path: Path, output_path: str = "enhanced_topology.json") -> Dict:
        """
        Perform comprehensive topology analysis on a project.
        
        Args:
            pdf_path: Path to the PDF file
            output_path: Path to save the enhanced topology JSON
            
        Returns:
            Dictionary containing enhanced topology
        """
        print(f"\n{'='*80}")
        print(f"DEEP TOPOLOGY ANALYSIS: {pdf_path.name}")
        print(f"{'='*80}\n")
        
        # Extract all text and vision analysis
        print("📄 Extracting text and analyzing drawings...")
        pages_data, total_pages = self.pdf_processor.extract_text_from_pdf(pdf_path)
        
        # Initialize topology storage
        enhanced_topology = {
            "project_name": pdf_path.stem,
            "total_pages": total_pages,
            "pages_analyzed": len(pages_data),
            "topology": defaultdict(lambda: defaultdict(list)),
            "page_index": {},  # Maps element to pages where it appears
            "relationships": [],  # Element relationships
            "summary": {}
        }
        
        print(f"\n🔍 Analyzing {len(pages_data)} pages for topology...")
        
        # Analyze each page
        for page_num, content in tqdm(pages_data.items(), desc="Extracting topology"):
            page_topology = self._extract_page_topology(content, page_num)
            
            # Merge into enhanced topology
            for category, elements in page_topology.items():
                if not isinstance(elements, list):
                    continue  # Skip if elements is not a list
                    
                for element in elements:
                    # Handle cases where element might be a string or invalid type
                    if isinstance(element, str):
                        element_key = element if element else 'unknown'
                        element_data = {'value': element}
                    elif isinstance(element, dict):
                        element_key = element.get('name') or element.get('type') or element.get('item') or element.get('material') or element.get('element') or 'unknown'
                        element_data = element
                    else:
                        continue  # Skip invalid elements
                    
                    # Skip if element_key is still None or empty
                    if not element_key:
                        element_key = 'unknown'
                    
                    enhanced_topology['topology'][category][element_key].append({
                        'page': page_num,
                        'details': element_data
                    })
                    
                    # Update page index
                    if element_key not in enhanced_topology['page_index']:
                        enhanced_topology['page_index'][element_key] = set()
                    enhanced_topology['page_index'][element_key].add(page_num)
        
        # Convert sets to lists for JSON serialization
        for key in enhanced_topology['page_index']:
            enhanced_topology['page_index'][key] = sorted(list(enhanced_topology['page_index'][key]))
        
        # Generate relationships
        print("\n🔗 Analyzing element relationships...")
        enhanced_topology['relationships'] = self._analyze_relationships(enhanced_topology)
        
        # Generate comprehensive summary
        print("\n📊 Generating topology summary...")
        enhanced_topology['summary'] = self._generate_topology_summary(enhanced_topology)
        
        # Convert defaultdict to regular dict for JSON
        enhanced_topology['topology'] = {
            k: dict(v) for k, v in enhanced_topology['topology'].items()
        }
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_topology, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Enhanced topology saved to: {output_path}")
        self._print_summary(enhanced_topology['summary'])
        
        return enhanced_topology
    
    def _extract_page_topology(self, content: str, page_num: int) -> Dict:
        """
        Extract topology from a single page using GPT-4.
        
        Args:
            content: Page content (text + vision analysis)
            page_num: Page number
            
        Returns:
            Dictionary of topology elements by category
        """
        prompt = f"""Analyze this structural engineering page content and extract ALL structural elements, specifications, and details.

Page {page_num} Content:
{content[:4000]}

Extract and categorize into JSON format:
{{
  "structural_elements": [
    {{"type": "element type", "name": "specific name", "quantity": "number", "location": "where"}},
    ...
  ],
  "foundation_systems": [
    {{"type": "foundation type", "size": "dimensions", "depth": "depth", "reinforcement": "details"}},
    ...
  ],
  "connection_types": [
    {{"type": "connection type", "components": ["list"], "specifications": "details"}},
    ...
  ],
  "materials": [
    {{"material": "material name", "grade": "grade", "application": "where used"}},
    ...
  ],
  "dimensions": [
    {{"element": "what", "dimension": "value", "unit": "unit"}},
    ...
  ],
  "specifications": [
    {{"item": "what", "specification": "requirement", "standard": "code reference"}},
    ...
  ],
  "details": [
    {{"detail_name": "name", "description": "what it shows", "references": ["related items"]}},
    ...
  ]
}}

Be comprehensive - extract every specific element, dimension, material, and specification mentioned.
Return ONLY valid JSON, no additional text."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert structural engineering analyst extracting detailed topology from technical documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            topology = json.loads(result_text)
            return topology
            
        except Exception as e:
            print(f"    ⚠️  Error extracting topology from page {page_num}: {e}")
            return {}
    
    def _analyze_relationships(self, topology_data: Dict) -> List[Dict]:
        """
        Analyze relationships between structural elements.
        
        Args:
            topology_data: Enhanced topology dictionary
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Common structural relationships
        relationship_patterns = [
            ("CIDH", "Foundation", "supports"),
            ("Column", "CIDH", "rests_on"),
            ("Bent Cap", "Column", "spans_between"),
            ("Girder", "Bent", "supported_by"),
            ("Bearing Pad", "Girder", "cushions"),
            ("Shear Key", "Abutment", "lateral_restraint"),
            ("Pipe Pin", "Bearing", "transfers_load"),
            ("Reinforcement", "Concrete", "strengthens"),
        ]
        
        # Check for co-occurrence on same pages
        page_index = topology_data.get('page_index', {})
        
        for elem1_key, pages1 in page_index.items():
            # Skip if key is None
            if elem1_key is None:
                continue
                
            for elem2_key, pages2 in page_index.items():
                if elem1_key == elem2_key:
                    continue
                
                # Skip if either key is None
                if elem1_key is None or elem2_key is None:
                    continue
                
                # Find common pages
                common_pages = set(pages1) & set(pages2)
                
                if common_pages:
                    # Check if there's a known relationship pattern
                    for pattern1, pattern2, rel_type in relationship_patterns:
                        if pattern1.lower() in elem1_key.lower() and pattern2.lower() in elem2_key.lower():
                            relationships.append({
                                "element1": elem1_key,
                                "element2": elem2_key,
                                "relationship": rel_type,
                                "pages": sorted(list(common_pages))
                            })
                            break
        
        return relationships
    
    def _generate_topology_summary(self, topology_data: Dict) -> Dict:
        """
        Generate a comprehensive summary of the topology.
        
        Args:
            topology_data: Enhanced topology dictionary
            
        Returns:
            Summary dictionary
        """
        summary = {
            "total_elements": 0,
            "categories": {},
            "most_common_elements": [],
            "unique_elements": len(topology_data.get('page_index', {})),
            "total_relationships": len(topology_data.get('relationships', [])),
            "coverage_by_page": {}
        }
        
        # Count elements by category
        for category, elements in topology_data.get('topology', {}).items():
            count = sum(len(occurrences) for occurrences in elements.values())
            summary['categories'][category] = {
                "unique_items": len(elements),
                "total_occurrences": count
            }
            summary['total_elements'] += count
        
        # Find most common elements
        element_counts = []
        for element, pages in topology_data.get('page_index', {}).items():
            element_counts.append({
                "element": element,
                "page_count": len(pages),
                "pages": pages
            })
        
        element_counts.sort(key=lambda x: x['page_count'], reverse=True)
        summary['most_common_elements'] = element_counts[:20]
        
        # Coverage by page
        all_pages = set()
        for pages in topology_data.get('page_index', {}).values():
            all_pages.update(pages)
        
        for page in sorted(all_pages):
            elements_on_page = [
                elem for elem, pages in topology_data.get('page_index', {}).items()
                if page in pages
            ]
            summary['coverage_by_page'][page] = {
                "element_count": len(elements_on_page),
                "elements": elements_on_page[:10]  # Top 10
            }
        
        return summary
    
    def _print_summary(self, summary: Dict):
        """Print topology summary to console."""
        print(f"\n{'='*80}")
        print("TOPOLOGY ANALYSIS SUMMARY")
        print(f"{'='*80}")
        print(f"\n📊 Total Elements: {summary['total_elements']}")
        print(f"🔢 Unique Elements: {summary['unique_elements']}")
        print(f"🔗 Relationships: {summary['total_relationships']}")
        
        print(f"\n📂 Categories:")
        for category, stats in summary['categories'].items():
            print(f"  • {category}: {stats['unique_items']} unique ({stats['total_occurrences']} occurrences)")
        
        print(f"\n🏆 Most Common Elements:")
        for i, elem in enumerate(summary['most_common_elements'][:10], 1):
            print(f"  {i}. {elem['element']}: {elem['page_count']} pages")
        
        print(f"\n{'='*80}\n")


def main():
    """Main function to run deep topology analysis."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python topology_analyzer.py <pdf_filename>")
        print("\nExample:")
        print('  python topology_analyzer.py "Mar Vista POC 100%_CheckPrint_20211118 Complete.pdf"')
        return
    
    filename = sys.argv[1]
    pdf_path = Path(config.PROJECTS_FOLDER) / filename
    
    if not pdf_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return
    
    analyzer = DeepTopologyAnalyzer()
    
    print("\n" + "="*80)
    print("DEEP TOPOLOGY ANALYZER")
    print("="*80)
    print(f"\nThis will perform comprehensive analysis of: {filename}")
    print("\n⚠️  HUMAN-IN-THE-LOOP CHECKPOINT #1")
    print("This process will:")
    print("  1. Analyze all pages with GPT-4 for detailed topology extraction")
    print("  2. Extract relationships between structural elements")
    print("  3. Generate comprehensive topology database")
    print(f"  4. Cost estimate: ~$2-5 for {filename}")
    print("\nContinue? (yes/no): ", end="")
    
    response = input().strip().lower()
    if response != 'yes':
        print("❌ Analysis cancelled by user")
        return
    
    # Run analysis
    output_path = "enhanced_topology.json"
    enhanced_topology = analyzer.analyze_project_comprehensive(pdf_path, output_path)
    
    print("\n" + "="*80)
    print("✓ DEEP TOPOLOGY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 Results saved to: {output_path}")
    print("\n🔍 Review the topology file to verify accuracy before proceeding to scenario generation.")
    print("\n💡 Next step: python scenario_generator.py")


if __name__ == "__main__":
    main()

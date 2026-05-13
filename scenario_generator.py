"""
Scenario Generator - Creates comprehensive search test scenarios from enhanced topology.

Generates diverse search scenarios that cover:
- Direct element searches
- Compound queries (multiple elements)
- Specification searches
- Dimensional queries
- Location-based searches
- Relationship queries
"""

import json
import random
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


class ScenarioGenerator:
    """Generates test search scenarios from enhanced topology."""
    
    def __init__(self, topology_path: str = "enhanced_topology.json"):
        """
        Initialize scenario generator.
        
        Args:
            topology_path: Path to enhanced topology JSON file
        """
        with open(topology_path, 'r', encoding='utf-8') as f:
            self.topology = json.load(f)
        
        self.scenarios = []
        self.coverage_target = 0.90  # 90% coverage goal
    
    def generate_scenarios(self, output_path: str = "test_scenarios.json") -> List[Dict]:
        """
        Generate comprehensive test scenarios.
        
        Args:
            output_path: Path to save scenarios JSON
            
        Returns:
            List of scenario dictionaries
        """
        print(f"\n{'='*80}")
        print("SCENARIO GENERATION")
        print(f"{'='*80}\n")
        
        print("🎯 Generating search scenarios to achieve 90% coverage...")
        
        # Generate different types of scenarios
        self.scenarios = []
        
        # 1. Direct element searches
        print("\n1️⃣  Generating direct element searches...")
        self._generate_direct_element_scenarios()
        
        # 2. Compound queries (multiple elements)
        print("2️⃣  Generating compound queries...")
        self._generate_compound_scenarios()
        
        # 3. Specification and detail searches
        print("3️⃣  Generating specification searches...")
        self._generate_specification_scenarios()
        
        # 4. Dimensional queries
        print("4️⃣  Generating dimensional queries...")
        self._generate_dimensional_scenarios()
        
        # 5. Relationship queries
        print("5️⃣  Generating relationship queries...")
        self._generate_relationship_scenarios()
        
        # 6. Variation queries (synonyms, technical terms)
        print("6️⃣  Generating variation queries...")
        self._generate_variation_scenarios()
        
        # Calculate coverage
        coverage = self._calculate_coverage()
        
        print(f"\n📊 Scenario Generation Summary:")
        print(f"  • Total scenarios: {len(self.scenarios)}")
        print(f"  • Knowledge base coverage: {coverage*100:.1f}%")
        print(f"  • Target coverage: {self.coverage_target*100:.0f}%")
        
        if coverage < self.coverage_target:
            print(f"\n⚠️  Coverage below target. Generating additional scenarios...")
            self._generate_additional_scenarios(self.coverage_target - coverage)
            coverage = self._calculate_coverage()
            print(f"  • Updated coverage: {coverage*100:.1f}%")
        
        # Save scenarios
        scenario_data = {
            "project_name": self.topology['project_name'],
            "total_scenarios": len(self.scenarios),
            "coverage": coverage,
            "scenarios": self.scenarios
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scenario_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Scenarios saved to: {output_path}")
        
        return self.scenarios
    
    def _generate_direct_element_scenarios(self):
        """Generate scenarios for direct element searches."""
        page_index = self.topology.get('page_index', {})
        
        for element, pages in page_index.items():
            self.scenarios.append({
                "id": len(self.scenarios) + 1,
                "type": "direct_element",
                "query": element,
                "expected_pages": pages,
                "expected_element": element,
                "min_results": 1,
                "description": f"Search for {element}"
            })
    
    def _generate_compound_scenarios(self):
        """Generate scenarios combining multiple elements."""
        relationships = self.topology.get('relationships', [])
        
        for rel in relationships[:20]:  # Top 20 relationships
            elem1 = rel['element1']
            elem2 = rel['element2']
            pages = rel['pages']
            
            # Create compound query
            query = f"{elem1} {elem2}"
            
            self.scenarios.append({
                "id": len(self.scenarios) + 1,
                "type": "compound",
                "query": query,
                "expected_pages": pages,
                "expected_elements": [elem1, elem2],
                "relationship": rel['relationship'],
                "min_results": 1,
                "description": f"Search for related elements: {elem1} and {elem2}"
            })
    
    def _generate_specification_scenarios(self):
        """Generate scenarios for specifications and details."""
        topology = self.topology.get('topology', {})
        
        # Extract specifications
        specs = topology.get('specifications', {})
        for spec_name, occurrences in list(specs.items())[:10]:
            pages = [occ['page'] for occ in occurrences]
            
            self.scenarios.append({
                "id": len(self.scenarios) + 1,
                "type": "specification",
                "query": spec_name,
                "expected_pages": list(set(pages)),
                "min_results": 1,
                "description": f"Search for specification: {spec_name}"
            })
        
        # Extract details
        details = topology.get('details', {})
        for detail_name, occurrences in list(details.items())[:10]:
            pages = [occ['page'] for occ in occurrences]
            
            self.scenarios.append({
                "id": len(self.scenarios) + 1,
                "type": "detail",
                "query": detail_name,
                "expected_pages": list(set(pages)),
                "min_results": 1,
                "description": f"Search for detail: {detail_name}"
            })
    
    def _generate_dimensional_scenarios(self):
        """Generate scenarios for dimensional searches."""
        topology = self.topology.get('topology', {})
        dimensions = topology.get('dimensions', {})
        
        for dim_key, occurrences in list(dimensions.items())[:15]:
            pages = [occ['page'] for occ in occurrences]
            
            # Extract the element being dimensioned
            sample = occurrences[0]['details'] if occurrences else {}
            element = sample.get('element', dim_key)
            
            self.scenarios.append({
                "id": len(self.scenarios) + 1,
                "type": "dimensional",
                "query": f"{element} dimensions",
                "expected_pages": list(set(pages)),
                "min_results": 1,
                "description": f"Search for dimensions of {element}"
            })
    
    def _generate_relationship_scenarios(self):
        """Generate scenarios based on structural relationships."""
        relationships = self.topology.get('relationships', [])
        
        # Create natural language queries about relationships
        relationship_templates = {
            "supports": "{elem1} supporting {elem2}",
            "rests_on": "{elem2} foundation for {elem1}",
            "spans_between": "{elem1} span details",
            "supported_by": "{elem1} support system",
            "cushions": "{elem1} {elem2} connection",
            "lateral_restraint": "{elem1} {elem2} restraint",
            "transfers_load": "{elem1} load transfer",
            "strengthens": "{elem2} reinforcement"
        }
        
        for rel in relationships[:15]:
            rel_type = rel['relationship']
            if rel_type in relationship_templates:
                template = relationship_templates[rel_type]
                query = template.format(elem1=rel['element1'], elem2=rel['element2'])
                
                self.scenarios.append({
                    "id": len(self.scenarios) + 1,
                    "type": "relationship",
                    "query": query,
                    "expected_pages": rel['pages'],
                    "expected_elements": [rel['element1'], rel['element2']],
                    "min_results": 1,
                    "description": f"Search for relationship: {rel_type}"
                })
    
    def _generate_variation_scenarios(self):
        """Generate scenarios with terminology variations."""
        # Common variations and synonyms in structural engineering
        variations = {
            "CIDH pile": ["drilled shaft", "cast-in-drilled-hole pile", "CIDH foundation"],
            "Pipe Pin": ["steel shear key", "pin connection"],
            "Bent": ["pier", "support bent", "column bent"],
            "Abutment": ["end support", "abutment structure"],
            "Girder": ["beam", "bridge girder", "main beam"],
            "Bearing Pad": ["elastomeric bearing", "bearing support"],
            "Reinforcement": ["rebar", "reinforcing steel", "reinforcing bars"]
        }
        
        page_index = self.topology.get('page_index', {})
        
        for element, variants in variations.items():
            # Find if this element exists in topology
            matching_keys = [key for key in page_index.keys() if element.lower() in key.lower()]
            
            if matching_keys:
                pages = page_index[matching_keys[0]]
                
                for variant in variants:
                    self.scenarios.append({
                        "id": len(self.scenarios) + 1,
                        "type": "variation",
                        "query": variant,
                        "expected_pages": pages,
                        "expected_element": matching_keys[0],
                        "variation_of": element,
                        "min_results": 1,
                        "description": f"Search using variation: {variant} (for {element})"
                    })
    
    def _generate_additional_scenarios(self, coverage_gap: float):
        """Generate additional scenarios to reach coverage target."""
        # Find elements with low coverage
        page_index = self.topology.get('page_index', {})
        covered_elements = set()
        
        for scenario in self.scenarios:
            if 'expected_element' in scenario:
                covered_elements.add(scenario['expected_element'])
            if 'expected_elements' in scenario:
                covered_elements.update(scenario['expected_elements'])
        
        uncovered_elements = [elem for elem in page_index.keys() if elem not in covered_elements]
        
        # Generate scenarios for uncovered elements
        for element in uncovered_elements[:int(coverage_gap * 100)]:
            pages = page_index[element]
            
            self.scenarios.append({
                "id": len(self.scenarios) + 1,
                "type": "coverage_fill",
                "query": element,
                "expected_pages": pages,
                "expected_element": element,
                "min_results": 1,
                "description": f"Additional coverage for: {element}"
            })
    
    def _calculate_coverage(self) -> float:
        """
        Calculate what percentage of the knowledge base is covered by scenarios.
        
        Returns:
            Coverage percentage (0.0 to 1.0)
        """
        page_index = self.topology.get('page_index', {})
        total_elements = len(page_index)
        
        if total_elements == 0:
            return 0.0
        
        covered_elements = set()
        for scenario in self.scenarios:
            if 'expected_element' in scenario:
                covered_elements.add(scenario['expected_element'])
            if 'expected_elements' in scenario:
                covered_elements.update(scenario['expected_elements'])
        
        coverage = len(covered_elements) / total_elements
        return coverage


def main():
    """Main function to generate test scenarios."""
    print("\n" + "="*80)
    print("TEST SCENARIO GENERATOR")
    print("="*80)
    
    topology_file = "enhanced_topology.json"
    
    if not Path(topology_file).exists():
        print(f"\n❌ Error: {topology_file} not found")
        print("\n💡 Run topology_analyzer.py first to generate enhanced topology")
        return
    
    print("\n⚠️  HUMAN-IN-THE-LOOP CHECKPOINT #2")
    print("\nThis will generate comprehensive search scenarios including:")
    print("  • Direct element searches")
    print("  • Compound queries (multiple elements)")
    print("  • Specification and detail searches")
    print("  • Dimensional queries")
    print("  • Relationship queries")
    print("  • Terminology variations")
    print("\nTarget: 90% knowledge base coverage")
    print("\nContinue? (yes/no): ", end="")
    
    response = input().strip().lower()
    if response != 'yes':
        print("❌ Scenario generation cancelled by user")
        return
    
    generator = ScenarioGenerator(topology_file)
    scenarios = generator.generate_scenarios()
    
    print("\n" + "="*80)
    print("✓ SCENARIO GENERATION COMPLETE")
    print("="*80)
    print(f"\n✅ Generated {len(scenarios)} test scenarios")
    print(f"📁 Results saved to: test_scenarios.json")
    print("\n🔍 Review scenarios before running the simulator")
    print("\n💡 Next step: python test_simulator.py")


if __name__ == "__main__":
    main()

"""Build enhanced topology JSON files from enriched metadata (no API calls)."""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import config
from enriched_metadata_loader import EnrichedMetadataLoader
from topology_analyzer import DeepTopologyAnalyzer


class EnhancedTopologyBuilder:
    """Convert enriched page metadata into enhanced_topology.json format."""

    def __init__(self):
        self.loader = EnrichedMetadataLoader()
        self.analyzer = DeepTopologyAnalyzer()

    def build_from_enriched_file(self, enriched_file: Path) -> Dict:
        """Build enhanced topology for one enriched metadata file."""
        with open(enriched_file, 'r', encoding='utf-8') as f:
            pages = json.load(f)

        if not pages:
            raise ValueError(f"No pages found in {enriched_file}")

        pdf_file_name = pages[0].get('filename') or f"{enriched_file.stem.replace('enriched_', '')}.pdf"
        project_name = enriched_file.stem.replace('enriched_', '')
        page_numbers = [p.get('page_number', 0) for p in pages if p.get('page_number')]
        total_pages = max(page_numbers) if page_numbers else len(pages)

        enhanced_topology = {
            "project_name": project_name,
            "pdf_file_name": pdf_file_name,
            "total_pages": total_pages,
            "pages_analyzed": len(pages),
            "source": "enriched_metadata",
            "topology": defaultdict(lambda: defaultdict(list)),
            "page_index": {},
            "relationships": [],
            "summary": {},
        }

        for page in pages:
            page_num = page.get('page_number')
            if not page_num:
                continue

            title_block = page.get('title_block') or {}
            plan_contents = title_block.get('plan_contents') or {}
            plan_type = title_block.get('plan_type') or {}
            project_info = title_block.get('project_info') or {}

            entries = []

            for element in self.loader._to_list(plan_contents.get('structural_elements')):
                entries.append(
                    ("structural_elements", element, {"type": element, "name": element})
                )

            for detail in self.loader._to_list(plan_contents.get('detail_types')):
                entries.append(
                    ("details", detail, {"detail_name": detail, "description": detail, "references": []})
                )

            sheet_title = (plan_type.get('sheet_title') or '').strip()
            primary_type = (plan_type.get('primary_type') or '').strip()
            if sheet_title:
                entries.append(
                    ("details", sheet_title, {"detail_name": sheet_title, "description": primary_type or sheet_title, "references": []})
                )
            if primary_type and primary_type != sheet_title:
                entries.append(
                    ("structural_elements", primary_type, {"type": primary_type, "name": primary_type})
                )

            for field, label in (
                (project_info.get('project_name'), 'project_name'),
                (project_info.get('project_number'), 'project_number'),
                (project_info.get('engineer_on_record'), 'engineer_on_record'),
            ):
                if field:
                    entries.append(
                        ("specifications", str(field), {"item": label, "specification": str(field), "standard": ""})
                    )

            for category, element_key, details in entries:
                element_key = str(element_key).strip()
                if not element_key:
                    continue

                enhanced_topology['topology'][category][element_key].append({
                    "page": page_num,
                    "details": details,
                })

                if element_key not in enhanced_topology['page_index']:
                    enhanced_topology['page_index'][element_key] = set()
                enhanced_topology['page_index'][element_key].add(page_num)

        for key in enhanced_topology['page_index']:
            enhanced_topology['page_index'][key] = sorted(enhanced_topology['page_index'][key])

        enhanced_topology['relationships'] = self.analyzer._analyze_relationships(enhanced_topology)
        enhanced_topology['summary'] = self.analyzer._generate_topology_summary(enhanced_topology)
        enhanced_topology['topology'] = {
            category: dict(elements)
            for category, elements in enhanced_topology['topology'].items()
        }

        return enhanced_topology

    def build_all(
        self,
        output_dir: Optional[Path] = None,
        write_manifest: bool = True,
    ) -> List[Path]:
        """Build enhanced topology files for every enriched_*.json in data/."""
        output_dir = output_dir or config.DATA_FOLDER
        output_dir.mkdir(parents=True, exist_ok=True)

        enriched_files = sorted(config.DATA_FOLDER.glob('enriched_*.json'))
        if not enriched_files:
            raise FileNotFoundError("No enriched metadata files found in data/")

        written_files: List[Path] = []
        manifest = {"projects": []}

        print("=" * 60)
        print("BUILDING ENHANCED TOPOLOGY FROM ENRICHED METADATA")
        print("=" * 60)
        print(f"Found {len(enriched_files)} enriched metadata files")
        print("(No API calls)")

        for enriched_file in enriched_files:
            project_name = enriched_file.stem.replace('enriched_', '')
            output_path = output_dir / f"enhanced_topology_{project_name}.json"
            print(f"\nBuilding: {project_name}")

            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                    if existing.get('source') == 'deep_vision':
                        print(
                            f"  Skipped: deep vision topology preserved at {output_path.name}"
                        )
                        continue
                except Exception:
                    pass

            try:
                topology = self.build_from_enriched_file(enriched_file)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(topology, f, indent=2, ensure_ascii=False)

                summary = topology.get('summary', {})
                print(
                    f"  Saved {output_path.name} "
                    f"({summary.get('unique_elements', 0)} elements, "
                    f"{topology.get('pages_analyzed', 0)} pages)"
                )
                written_files.append(output_path)
                manifest['projects'].append({
                    "project_name": project_name,
                    "pdf_file_name": topology.get('pdf_file_name'),
                    "topology_file": output_path.name,
                    "unique_elements": summary.get('unique_elements', 0),
                    "total_relationships": summary.get('total_relationships', 0),
                })
            except Exception as exc:
                print(f"  Error: {exc}")

        if write_manifest and manifest['projects']:
            manifest_path = output_dir / 'enhanced_topology_manifest.json'
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        existing_manifest = json.load(f)
                    if existing_manifest.get('source') == 'deep_vision':
                        print(
                            "\nManifest not overwritten "
                            f"(deep vision manifest preserved at {manifest_path.name})"
                        )
                        write_manifest = False
                except Exception:
                    pass

            if write_manifest:
                manifest['source'] = 'enriched_metadata'
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                print(f"\nManifest saved to: {manifest_path}")

        print("\n" + "=" * 60)
        print("ENHANCED TOPOLOGY BUILD COMPLETE")
        print("=" * 60)
        print(f"Built {len(written_files)}/{len(enriched_files)} topology files")

        return written_files


def main():
    builder = EnhancedTopologyBuilder()
    builder.build_all()


if __name__ == "__main__":
    main()

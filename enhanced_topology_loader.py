"""Loader for deep-vision enhanced topology JSON files."""
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import config


class EnhancedTopologyLoader:
    """Load and format per-page data from enhanced_topology_*.json files."""

    TOPOLOGY_CATEGORIES = [
        "structural_elements",
        "foundation_systems",
        "connection_types",
        "materials",
        "dimensions",
        "specifications",
        "details",
    ]

    @staticmethod
    def load_topology(
        file_name: str,
        require_deep_vision: bool = False,
    ) -> Optional[Dict]:
        """Load enhanced topology for a PDF file."""
        pdf_stem = Path(file_name).stem
        topology_file = config.DATA_FOLDER / f"enhanced_topology_{pdf_stem}.json"

        if not topology_file.exists():
            return None

        try:
            with open(topology_file, "r", encoding="utf-8") as f:
                topology_data = json.load(f)
        except Exception as exc:
            print(f"Warning: Could not load enhanced topology for {file_name}: {exc}")
            return None

        if require_deep_vision and topology_data.get("source") != "deep_vision":
            return None

        return topology_data

    @staticmethod
    def list_deep_vision_topology_files() -> List[Path]:
        """Return enhanced topology JSON files produced by deep vision."""
        files = []
        for topology_file in sorted(config.DATA_FOLDER.glob("enhanced_topology_*.json")):
            if topology_file.name == "enhanced_topology_manifest.json":
                continue
            try:
                with open(topology_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("source") == "deep_vision":
                    files.append(topology_file)
            except Exception:
                continue
        return files

    @staticmethod
    def build_page_map(topology_data: Dict) -> Dict[int, Dict]:
        """Aggregate topology elements by page number."""
        pages: Dict[int, Dict] = defaultdict(
            lambda: {category: [] for category in EnhancedTopologyLoader.TOPOLOGY_CATEGORIES}
        )

        for category, elements in topology_data.get("topology", {}).items():
            if not isinstance(elements, dict):
                continue
            for element_name, occurrences in elements.items():
                if not isinstance(occurrences, list):
                    continue
                for occurrence in occurrences:
                    page_num = occurrence.get("page")
                    if not page_num:
                        continue
                    pages[int(page_num)][category].append(
                        {
                            "name": str(element_name),
                            "details": occurrence.get("details", {}),
                        }
                    )

        return dict(pages)

    @staticmethod
    def _format_detail(details: Dict) -> str:
        if not isinstance(details, dict):
            return str(details)

        parts = []
        for key in ("type", "name", "detail_name", "description", "material", "element", "specification"):
            value = details.get(key)
            if value:
                parts.append(str(value))
        return " - ".join(parts) if parts else ""

    @classmethod
    def extract_searchable_text(cls, page_data: Dict, max_items_per_category: int = 12) -> str:
        """Build searchable text for one page from topology elements."""
        parts = []

        for category in cls.TOPOLOGY_CATEGORIES:
            entries = page_data.get(category, [])
            if not entries:
                continue

            labels = []
            for entry in entries[:max_items_per_category]:
                name = entry.get("name", "")
                detail_text = cls._format_detail(entry.get("details", {}))
                if detail_text and detail_text != name:
                    labels.append(f"{name} ({detail_text})")
                elif name:
                    labels.append(name)

            if labels:
                category_label = category.replace("_", " ").title()
                parts.append(f"{category_label}: {', '.join(labels)}")

            remaining = len(entries) - max_items_per_category
            if remaining > 0:
                parts.append(f"{category.replace('_', ' ').title()} (+{remaining} more)")

        return " | ".join(parts)

    @classmethod
    def get_metadata_fields(cls, page_data: Dict, max_elements: int = 40) -> Dict[str, str]:
        """Extract flat metadata fields for vector-store filtering."""
        all_elements = []
        categories_present = []

        for category in cls.TOPOLOGY_CATEGORIES:
            entries = page_data.get(category, [])
            if not entries:
                continue
            categories_present.append(category)
            for entry in entries:
                name = str(entry.get("name", "")).strip()
                if name and name.lower() != "unknown":
                    all_elements.append(name)

        # Preserve order while deduplicating
        seen = set()
        unique_elements = []
        for element in all_elements:
            key = element.lower()
            if key in seen:
                continue
            seen.add(key)
            unique_elements.append(element)

        fields = {
            "deep_vision_merged": "true",
            "topology_source": "deep_vision",
        }

        if categories_present:
            fields["topology_categories"] = ", ".join(categories_present)
        if unique_elements:
            fields["topology_elements"] = ", ".join(unique_elements[:max_elements])
            fields["topology_element_count"] = str(len(unique_elements))

        return fields

    @classmethod
    def get_page_data(cls, topology_data: Dict) -> Dict[int, Dict]:
        """Return per-page topology payloads keyed by page number."""
        return cls.build_page_map(topology_data)

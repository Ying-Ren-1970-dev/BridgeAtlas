"""Loader for deep-vision enhanced topology JSON files."""
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

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

    LAYERED_TOPOLOGY_VERSION = "6"

    SUPERSTRUCTURE_PATTERNS = (
        r"\bbox girder\b",
        r"\bcip\b",
        r"\bps conc\b",
        r"\bprestressed\b",
        r"\btruss\b",
        r"\bi girder\b",
        r"\bplate girder\b",
        r"\bgirder\b",
        r"\bcable stayed\b",
        r"\barch\b",
        r"\bconcrete slab\b",
    )
    FOUNDATION_PATTERNS = (
        r"\bcidh\b",
        r"\bdrilled shaft\b",
        r"\bpile\b",
        r"\bfooting\b",
        r"\bfoundation\b",
        r"\bshaft\b",
    )
    SUBSTRUCTURE_PATTERNS = (
        r"\babutment\b",
        r"\bbent\b",
        r"\bcolumn\b",
        r"\bpier\b",
        r"\bcap beam\b",
        r"\bpedestal\b",
    )

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
    def _matches_any_pattern(cls, blob: str, patterns: tuple) -> bool:
        return any(re.search(pattern, blob) for pattern in patterns)

    @classmethod
    def _classify_element_bucket(cls, name: str, details: Dict) -> str:
        blob = f"{name} {cls._format_detail(details)}".lower()
        if cls._matches_any_pattern(blob, cls.FOUNDATION_PATTERNS):
            return "foundation"
        if cls._matches_any_pattern(blob, cls.SUPERSTRUCTURE_PATTERNS):
            return "superstructure"
        if cls._matches_any_pattern(blob, cls.SUBSTRUCTURE_PATTERNS):
            return "substructure"
        if re.search(r"\bbridge\b", blob):
            return "bridge"
        return "other"

    @classmethod
    @lru_cache(maxsize=8)
    def _classification_trainer(cls):
        from classification_training import SearchClassificationTrainer

        return SearchClassificationTrainer()

    @classmethod
    def _heuristic_project_topology(cls, topology_data: Dict) -> Dict[str, str]:
        """Infer raw element buckets from topology when taxonomy labels are sparse."""
        superstructure: Set[str] = set()
        substructure: Set[str] = set()
        foundations: Set[str] = set()
        bridge_types: Set[str] = set()
        structure_types: Set[str] = set()

        for _category, elements in topology_data.get("topology", {}).items():
            if not isinstance(elements, dict):
                continue
            for element_name, occurrences in elements.items():
                if not isinstance(occurrences, list):
                    continue
                for occurrence in occurrences:
                    details = occurrence.get("details", {}) or {}
                    name = str(element_name or details.get("name") or "").strip()
                    if not name or name.lower() == "unknown":
                        continue

                    blob = f"{name} {cls._format_detail(details)}".lower()
                    bucket = cls._classify_element_bucket(name, details)
                    if bucket == "superstructure":
                        superstructure.add(name)
                        if cls._matches_any_pattern(
                            blob, (r"\bbox girder\b", r"\bcip\b", r"\btruss\b", r"\bgirder\b")
                        ):
                            bridge_types.add(name)
                    elif bucket == "substructure":
                        substructure.add(name)
                    elif bucket == "foundation":
                        foundations.add(name)
                    elif bucket == "bridge":
                        structure_types.add("bridge")

                    detail_type = str(details.get("type") or "").strip()
                    if detail_type.lower() == "bridge":
                        structure_types.add("bridge")

        if superstructure:
            structure_types.add("bridge")
        if not structure_types and (substructure or foundations):
            structure_types.add("bridge")

        if not bridge_types and superstructure:
            bridge_types = set(list(superstructure)[:6])

        return {
            "structure_type": ", ".join(sorted(structure_types)) or "structural",
            "bridge_type": ", ".join(sorted(bridge_types)[:8]),
            "superstructure_type": ", ".join(sorted(superstructure)[:8]),
            "substructure_type": ", ".join(sorted(substructure)[:8]),
            "foundation_type": ", ".join(sorted(foundations)[:8]),
        }

    @classmethod
    def infer_project_topology(cls, topology_data: Dict) -> Dict[str, str]:
        """
        Infer project-level structure and bridge type using search classifications.md,
        with heuristic element buckets as fallback context.
        """
        heuristic = cls._heuristic_project_topology(topology_data)
        try:
            classified = cls._classification_trainer().classify_topology_project(topology_data)
        except Exception:
            classified = {}

        merged = heuristic.copy()
        for key in (
            "structure_type",
            "bridge_type",
            "bridge_superstructure",
            "bridge_substructure",
            "bridge_foundation",
            "project_level_labels",
        ):
            if classified.get(key):
                merged[key] = classified[key]

        if classified.get("bridge_superstructure"):
            merged["superstructure_type"] = classified["bridge_superstructure"]
        if classified.get("bridge_substructure"):
            merged["substructure_type"] = classified["bridge_substructure"]
        if classified.get("bridge_foundation"):
            merged["foundation_type"] = classified["bridge_foundation"]

        return merged

    @classmethod
    def extract_search_classification_text(
        cls,
        topology_data: Dict,
        page_num: int,
        page_data: Dict,
        sheet_record: Optional[Dict] = None,
    ) -> str:
        """Build searchable classification block from search classifications.md taxonomy."""
        try:
            trainer = cls._classification_trainer()
            project = trainer.classify_topology_project(topology_data)
            page = trainer.classify_topology_page(page_data, sheet_record, page_num)
        except Exception:
            return ""

        parts = []
        for key, label in (
            ("structure_type", "Structure Type"),
            ("bridge_type", "Bridge Type"),
            ("bridge_superstructure", "Superstructure"),
            ("bridge_substructure", "Substructure"),
            ("bridge_foundation", "Foundation"),
            ("project_level_labels", "Project Labels"),
            ("page_level_labels", "Page Labels"),
            ("detail_level_labels", "Detail Labels"),
            ("referenced_sheet_ids", "Referenced Sheets"),
            ("referenced_detail_ids", "Referenced Details"),
        ):
            value = project.get(key) or page.get(key)
            if value:
                parts.append(f"{label}: {value}")

        if not parts:
            return ""
        return "[SEARCH CLASSIFICATION]\n" + " | ".join(parts)

    @classmethod
    def _page_relationships(cls, topology_data: Dict, page_num: int) -> List[str]:
        labels = []
        for rel in topology_data.get("relationships", []) or []:
            pages = rel.get("pages") or []
            if int(page_num) not in {int(p) for p in pages}:
                continue
            element1 = str(rel.get("element1") or "").strip()
            element2 = str(rel.get("element2") or "").strip()
            relationship = str(rel.get("relationship") or "").strip()
            if element1 and element2 and relationship:
                labels.append(f"{element1} {relationship} {element2}")
        return labels[:20]

    @classmethod
    def _page_detail_cross_references(cls, page_data: Dict) -> List[str]:
        labels = []
        for entry in page_data.get("details", []) or []:
            details = entry.get("details", {}) or {}
            detail_name = str(details.get("detail_name") or entry.get("name") or "").strip()
            refs = details.get("references") or []
            clean_refs = [
                str(ref).strip()
                for ref in refs
                if str(ref).strip() and str(ref).strip().lower() not in {"not specified", "unknown", "n/a"}
            ]
            if detail_name and clean_refs:
                labels.append(f"{detail_name} -> {', '.join(clean_refs[:6])}")
            elif detail_name and str(details.get("description") or "").strip():
                labels.append(f"{detail_name}: {details.get('description')}")
        return labels[:20]

    @classmethod
    def extract_searchable_text(cls, page_data: Dict, max_items_per_category: int = 12) -> str:
        """Build flat searchable text for one page from topology elements."""
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
    def extract_layered_topology_text(
        cls,
        topology_data: Dict,
        page_num: int,
        page_data: Dict,
        sheet_record: Optional[Dict] = None,
    ) -> str:
        """
        Build layered topology text:
        project -> sheet -> detail -> cross references.
        """
        blocks: List[str] = []
        project = cls.infer_project_topology(topology_data)
        project_parts = [
            f"Project: {topology_data.get('project_name', '')}",
            f"Structure Type: {project.get('structure_type', '')}",
            f"Bridge Type: {project.get('bridge_type', '')}",
            f"Superstructure: {project.get('superstructure_type', '')}",
            f"Substructure: {project.get('substructure_type', '')}",
            f"Foundation: {project.get('foundation_type', '')}",
        ]
        blocks.append("[TOPOLOGY PROJECT]\n" + " | ".join(p for p in project_parts if p.split(": ", 1)[-1]))

        sheet_parts = [f"Page: {page_num}"]
        if sheet_record:
            for key, label in (
                ("sheet_category", "Sheet Category"),
                ("sheet_title", "Sheet Title"),
                ("primary_type", "Plan Type"),
                ("sheet_number", "Sheet Number"),
            ):
                if sheet_record.get(key):
                    sheet_parts.append(f"{label}: {sheet_record[key]}")
        blocks.append("[TOPOLOGY SHEET]\n" + " | ".join(sheet_parts))

        detail_text = cls.extract_searchable_text(page_data)
        if detail_text:
            blocks.append(f"[TOPOLOGY DETAIL]\n{detail_text}")

        cross_refs = cls._page_detail_cross_references(page_data)
        cross_refs.extend(cls._page_relationships(topology_data, page_num))
        if cross_refs:
            blocks.append("[TOPOLOGY CROSS REFERENCES]\n" + " | ".join(cross_refs))

        classification_text = cls.extract_search_classification_text(
            topology_data, page_num, page_data, sheet_record=sheet_record
        )
        if classification_text:
            blocks.append(classification_text)

        return "\n\n".join(blocks)

    @classmethod
    def strip_topology_blocks(cls, doc_text: str) -> str:
        """Remove previously merged topology blocks before re-merging."""
        markers = [
            "[DEEP VISION TOPOLOGY]",
            "[TOPOLOGY PROJECT]",
            "[TOPOLOGY SHEET]",
            "[TOPOLOGY DETAIL]",
            "[TOPOLOGY CROSS REFERENCES]",
            "[SEARCH CLASSIFICATION]",
        ]
        text = doc_text or ""
        cut_at = len(text)
        for marker in markers:
            idx = text.find(marker)
            if idx != -1:
                cut_at = min(cut_at, idx)
        return text[:cut_at].rstrip()

    @classmethod
    def get_metadata_fields(
        cls,
        page_data: Dict,
        topology_data: Optional[Dict] = None,
        page_num: Optional[int] = None,
        sheet_record: Optional[Dict] = None,
        max_elements: int = 40,
    ) -> Dict[str, str]:
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
            "layered_topology_version": cls.LAYERED_TOPOLOGY_VERSION,
        }

        if categories_present:
            fields["topology_categories"] = ", ".join(categories_present)
        if unique_elements:
            fields["topology_elements"] = ", ".join(unique_elements[:max_elements])
            fields["topology_element_count"] = str(len(unique_elements))

        if topology_data:
            project = cls.infer_project_topology(topology_data)
            if project.get("structure_type"):
                fields["topology_structure_type"] = project["structure_type"]
            if project.get("bridge_type"):
                fields["topology_bridge_type"] = project["bridge_type"]
            if project.get("superstructure_type"):
                fields["topology_superstructure_type"] = project["superstructure_type"]
            if project.get("foundation_type"):
                fields["topology_foundation_type"] = project["foundation_type"]
            if project.get("project_level_labels"):
                fields["search_classification_project_labels"] = project["project_level_labels"]

        if topology_data and page_num is not None:
            try:
                page_classification = cls._classification_trainer().classify_topology_page(
                    page_data, sheet_record, page_num
                )
                if page_classification.get("page_level_labels"):
                    fields["search_classification_page_labels"] = page_classification["page_level_labels"]
                if page_classification.get("detail_level_labels"):
                    fields["search_classification_detail_labels"] = page_classification["detail_level_labels"]
                if page_classification.get("referenced_sheet_ids"):
                    fields["search_classification_referenced_sheets"] = page_classification["referenced_sheet_ids"]
            except Exception:
                pass

        if sheet_record:
            if sheet_record.get("sheet_category"):
                fields["topology_sheet_category"] = str(sheet_record["sheet_category"])
            if sheet_record.get("sheet_title"):
                fields["topology_sheet_title"] = str(sheet_record["sheet_title"])

        if topology_data and page_num is not None:
            cross_refs = cls._page_detail_cross_references(page_data)
            cross_refs.extend(cls._page_relationships(topology_data, page_num))
            if cross_refs:
                fields["topology_cross_references"] = " | ".join(cross_refs[:12])

        return fields

    @classmethod
    def get_page_data(cls, topology_data: Dict) -> Dict[int, Dict]:
        """Return per-page topology payloads keyed by page number."""
        return cls.build_page_map(topology_data)

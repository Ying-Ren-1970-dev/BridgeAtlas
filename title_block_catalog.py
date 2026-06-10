"""Build per-sheet title block categories for every page in a project."""
import re
from typing import Dict, List, Optional, Tuple

from enriched_metadata_loader import EnrichedMetadataLoader
from enhanced_topology_loader import EnhancedTopologyLoader


class TitleBlockCatalog:
    """Resolve sheet title/category for every page using enriched + deep vision data."""

    ADMIN_DETAIL_TOKENS = (
        "quantities",
        "specifications",
        "checked",
        "design oversight",
        "project engineer",
        "post miles",
        "layout",
        "file",
        "monument",
        "architectural treatment",
        "design detail sheet",
        "bridge no.",
    )

    PRIMARY_TYPE_PATTERNS: List[Tuple[str, str]] = [
        ("general plan", "General Plan"),
        ("foundation plan", "Foundation Plan"),
        ("bridge removal plan", "Bridge Removal Plan"),
        ("removal plan", "Removal Plan"),
        ("deck plan", "Deck Plan"),
        ("footing plan", "Footing Plan"),
        ("elevation", "Elevation"),
        ("section", "Section"),
        ("typical section", "Typical Section"),
        ("detail", "Detail"),
        ("general note", "General Notes"),
        ("general notes", "General Notes"),
        ("schedule", "Schedule"),
        ("profile", "Profile"),
    ]

    @classmethod
    def build_for_project(cls, file_name: str) -> Dict[int, Dict]:
        """
        Return title-block records keyed by page number for all known pages.

        Enriched metadata wins when present; remaining pages are inferred from
        deep-vision topology detail names.
        """
        catalog: Dict[int, Dict] = {}

        enriched = EnrichedMetadataLoader.load_enriched_metadata(file_name) or {}
        for page_num, page_data in enriched.items():
            record = cls._from_enriched(page_data)
            if record:
                catalog[int(page_num)] = record

        topology = EnhancedTopologyLoader.load_topology(file_name, require_deep_vision=True)
        if not topology:
            return catalog

        total_pages = int(topology.get("total_pages") or topology.get("pages_analyzed") or 0)
        page_map = EnhancedTopologyLoader.get_page_data(topology)

        for page_num in range(1, total_pages + 1):
            if page_num in catalog:
                continue
            page_data = page_map.get(page_num)
            if page_data:
                record = cls._from_topology_page(page_data)
                if record:
                    catalog[page_num] = record
                    continue
            record = cls._from_pdf_text_page(file_name, page_num)
            if record:
                catalog[page_num] = record

        for page_num, record in list(catalog.items()):
            if record.get("sheet_category") != "general_note":
                continue
            excerpt = cls._extract_pdf_page_text(file_name, page_num)
            if excerpt:
                record["pdf_text_excerpt"] = excerpt

        return catalog

    @classmethod
    def _from_enriched(cls, page_data: Dict) -> Optional[Dict]:
        title_block = page_data.get("title_block") or {}
        if not title_block:
            return None

        plan_type = title_block.get("plan_type") or {}
        plan_contents = title_block.get("plan_contents") or {}
        sheet_title = str(plan_type.get("sheet_title") or "").strip()
        primary_type = str(plan_type.get("primary_type") or "").strip()
        detail_types = EnrichedMetadataLoader._to_list(plan_contents.get("detail_types"))

        if not sheet_title and not primary_type:
            return None

        sheet_category = EnrichedMetadataLoader.classify_sheet_type(
            sheet_title, detail_types, primary_type=primary_type
        )
        return {
            "source": "enriched_metadata",
            "page_type": str(page_data.get("page_type") or "plan"),
            "sheet_title": sheet_title,
            "primary_type": primary_type,
            "sheet_number": str(plan_type.get("sheet_number") or "").strip(),
            "sheet_category": sheet_category,
            "detail_types": ", ".join(detail_types),
        }

    @classmethod
    def _extract_pdf_page_text(cls, file_name: str, page_num: int) -> str:
        try:
            import fitz
            from storage_adapter import StorageAdapter

            pdf_path = StorageAdapter().get_pdf_temp_path(file_name)
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return ""
            text = doc[page_num - 1].get_text("text")
            doc.close()
        except Exception:
            return ""
        return re.sub(r"\s+", " ", text).strip()[:4000]

    @classmethod
    def _from_pdf_text_page(cls, file_name: str, page_num: int) -> Optional[Dict]:
        """Infer sheet category from native PDF text when enriched/topology data is missing."""
        text = cls._extract_pdf_page_text(file_name, page_num)
        if not text:
            return None

        text_lower = text.lower()
        if "index to plans" in text_lower:
            sheet_title = "INDEX TO PLANS"
            primary_type = "Index to Plans"
            detail_types = ["general notes"] if "general notes" in text_lower else []
        elif "general notes" in text_lower:
            sheet_title = "GENERAL NOTES"
            primary_type = "General Notes"
            detail_types = ["general notes"]
        else:
            return None

        sheet_category = EnrichedMetadataLoader.classify_sheet_type(
            sheet_title, detail_types, primary_type=primary_type
        )
        return {
            "source": "pdf_text",
            "page_type": "plan",
            "sheet_title": sheet_title,
            "primary_type": primary_type,
            "sheet_number": "",
            "sheet_category": sheet_category,
            "detail_types": ", ".join(detail_types),
            "pdf_text_excerpt": text,
        }

    @classmethod
    def _from_topology_page(cls, page_data: Dict) -> Optional[Dict]:
        details = page_data.get("details") or []
        if not details:
            return None

        best_title = ""
        best_score = -1
        for entry in details:
            details_blob = entry.get("details") or {}
            candidate = str(details_blob.get("detail_name") or entry.get("name") or "").strip()
            if not candidate:
                continue
            score = cls._score_sheet_title(candidate)
            if score > best_score:
                best_score = score
                best_title = candidate

        if best_score < 1 or not best_title:
            return None

        primary_type = cls._infer_primary_type(best_title)
        sheet_category = EnrichedMetadataLoader.classify_sheet_type(
            best_title, [], primary_type=primary_type
        )
        return {
            "source": "deep_vision",
            "page_type": "plan",
            "sheet_title": best_title,
            "primary_type": primary_type,
            "sheet_number": "",
            "sheet_category": sheet_category,
            "detail_types": "",
        }

    @classmethod
    def _score_sheet_title(cls, detail_name: str) -> int:
        name = detail_name.lower().strip()
        if not name or name in {"n/a", "unknown", "not specified"}:
            return -1
        if any(token in name for token in cls.ADMIN_DETAIL_TOKENS):
            return -5

        score = 0
        if "plan" in name:
            score += 6
        if "general" in name:
            score += 4
        if "foundation" in name:
            score += 4
        if any(token in name for token in ("elevation", "section", "detail", "layout", "schedule", "profile", "typical")):
            score += 3
        if any(token in name for token in ("abutment", "girder", "pile", "bent", "bridge", "removal", "deck", "footing")):
            score += 2
        if len(name) >= 8:
            score += 1
        return score

    @classmethod
    def _infer_primary_type(cls, sheet_title: str) -> str:
        title_lower = sheet_title.lower()
        for pattern, label in cls.PRIMARY_TYPE_PATTERNS:
            if pattern in title_lower:
                return label
        if "plan" in title_lower:
            return "Plan"
        return "Drawing Sheet"

    @classmethod
    def extract_searchable_text(cls, record: Dict) -> str:
        parts = []
        if record.get("sheet_category"):
            parts.append(f"Sheet Category: {record['sheet_category']}")
        if record.get("primary_type"):
            parts.append(f"Plan Type: {record['primary_type']}")
        if record.get("sheet_title"):
            parts.append(f"Sheet Title: {record['sheet_title']}")
        if record.get("sheet_number"):
            parts.append(f"Sheet Number: {record['sheet_number']}")
        if record.get("page_type"):
            parts.append(f"Page Type: {record['page_type']}")
        if record.get("detail_types"):
            parts.append(f"Detail Types: {record['detail_types']}")
        if record.get("pdf_text_excerpt"):
            parts.append(f"General Notes Text: {record['pdf_text_excerpt']}")
        return " | ".join(parts)

    @classmethod
    def get_metadata_fields(cls, record: Dict) -> Dict[str, str]:
        fields = {
            "sheet_category_merged": "true",
            "title_block_source": str(record.get("source") or ""),
            "page_type": str(record.get("page_type") or ""),
            "plan_sheet_type": str(record.get("sheet_category") or ""),
        }
        if record.get("sheet_title"):
            fields["plan_sheet_title"] = str(record["sheet_title"])
        if record.get("primary_type"):
            fields["plan_primary_type"] = str(record["primary_type"])
        if record.get("sheet_number"):
            fields["plan_sheet_number"] = str(record["sheet_number"])
        if record.get("detail_types"):
            fields["detail_types"] = str(record["detail_types"])

        detail_intents = set(
            EnrichedMetadataLoader.derive_detail_intents(
                [],
                EnrichedMetadataLoader._to_list(record.get("detail_types")),
                str(record.get("sheet_title") or ""),
                primary_type=str(record.get("primary_type") or ""),
            )
        )
        excerpt = str(record.get("pdf_text_excerpt") or "").lower()
        if "ars" in excerpt and "curve" in excerpt:
            detail_intents.add("ars curve")
        if detail_intents:
            fields["detail_intents"] = ", ".join(sorted(detail_intents))
        return fields

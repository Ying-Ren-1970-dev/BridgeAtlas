"""
CAD drawing view classification, sheet grouping, and cross-reference extraction.

Each drawing chunk is classified primarily from its visible label (the yellow
title on the sheet). Common types are plan, elevation, layout, detail, and
section. View is uncommon and only used when explicitly labeled (VIEW A-A).
Additional categories (diagram, as_built, schedule, etc.) are inferred from the
label text when present.

Imprecise labels such as TYPICAL SECTION or TYPICAL DETAILS fall back to section
or detail without a numbered/name precision.

Section cuts (A-A, B-B), views (View A-A), and detail callouts (Detail 1, Detail 2)
link drawings together. Notes on a sheet often reference other sheets when the
target section/view/detail is not on the same page.

Plan sheets are organized into groups (abutment, bent, girder, etc.). Within a
group, layout sheets may contain plan/elevation/layout views; detail sheets may
contain detail/section/view. Section letters and detail numbers typically restart
per group (A-A or Detail 1 at the start of each group).

Leader-line text belongs to the drawing chunk where the leader arrowhead points,
not where the text sits. Chunk boundaries are approximate guides; attribution
follows leader targets.

For search, detail and section are equivalent for component intent: a query for
"shear key detail" should also match section chunks showing the same component.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Core searchable view types (section and detail are most common).
CAD_DRAWING_CORE_TYPES: Tuple[str, ...] = (
    "plan",
    "elevation",
    "layout",
    "detail",
    "section",
)

# Less common or label-derived types.
CAD_DRAWING_EXTENDED_TYPES: Tuple[str, ...] = (
    "view",
    "diagram",
    "as_built",
    "schedule",
)

CAD_DRAWING_VIEWS: Tuple[str, ...] = CAD_DRAWING_CORE_TYPES + CAD_DRAWING_EXTENDED_TYPES

# Detail and section are categorized separately on drawings but equivalent for search.
DETAIL_SECTION_EQUIVALENT_TYPES: Set[str] = {"detail", "section"}

LEADER_ATTRIBUTION_RULE = (
    "Leader text belongs to the drawing chunk where the leader arrowhead points. "
    "Chunk boundaries are approximate layout guides, not strict clip boxes."
)

DETAIL_SECTION_SEARCH_RULE = (
    "Search equivalence: detail intent also matches section chunks for the same "
    "component (e.g. 'shear key detail' matches shear key sections)."
)

# Ordered label rules: (regex, view_type, precise). First match wins per label.
LABEL_CLASSIFICATION_RULES: Tuple[Tuple[str, str, bool], ...] = (
    (r"\bSECTION\s+([A-Z])\s*[-–]\s*\1\b", "section", True),
    (r"\bDETAIL\s+(?:NO\.?\s*)?(\d+)\b", "detail", True),
    (r"\bVIEW\s+([A-Z])\s*[-–]\s*\1\b", "view", True),
    (r"\bTYPICAL\s+SECTION\b", "section", False),
    (r"\bTYPICAL\s+DETAILS?\b", "detail", False),
    (r"\b(PLAN(?:\s+STAGE\s+\d+)?)\b", "plan", True),
    (r"\b(ELEVATION(?:\s+STAGE\s+\d+)?)\b", "elevation", True),
    (r"\b([A-Z0-9\"ø/\.%\-]+\s+[A-Z0-9\"ø/\.%\-]+\s+ELEVATION)\b", "elevation", True),
    (r"\b([A-Z][A-Z0-9\s\-]+\s+DIAGRAM)\b", "diagram", True),
    (r"\bAS\s+BUILT(?:\s+SECTION\s+[A-Z]\s*[-–]\s*[A-Z])?\b", "as_built", False),
    (r"\b([A-Z][A-Z0-9\s\-]+\s+DETAILS?)\b", "detail", False),
    (r"\bDIAGRAM\b", "diagram", False),
)

# Keyword suffixes used when no explicit rule matches.
_CATEGORY_KEYWORDS: Tuple[Tuple[str, str], ...] = (
    ("diagram", "diagram"),
    ("schedule", "schedule"),
    ("elevation", "elevation"),
    ("section", "section"),
    ("detail", "detail"),
    ("layout", "layout"),
    ("plan", "plan"),
)

SHEET_GROUP_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("abutment", r"abutment"),
    ("bent", r"\bbent\b"),
    ("pier", r"\bpier\b"),
    ("girder", r"girder|box girder|superstructure"),
    ("cap", r"\bcap\b"),
    ("column", r"\bcolumn\b"),
    ("footing", r"footing|foundation"),
    ("wingwall", r"wing\s*wall|wingwall"),
    ("retaining_wall", r"retaining wall|shga"),
    ("barrier", r"barrier|railing"),
    ("diaphragm", r"diaphragm"),
    ("pile", r"\bpile\b|cidh|drilled shaft"),
    ("deck", r"\bdeck\b"),
    ("approach", r"approach slab|approach"),
)

SHEET_ROLE_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("layout", r"layout"),
    ("details", r"details?"),
    ("plan", r"\bplan\b"),
    ("elevation", r"elevation"),
    ("section", r"sections?"),
    ("general_plan", r"general plan"),
    ("general_note", r"general note|notes"),
)

SECTION_CUT_RE = re.compile(
    r"\bsection\s+([A-Z])\s*[-–]\s*\1\b",
    re.IGNORECASE,
)
VIEW_REF_RE = re.compile(
    r"\bview\s+([A-Z])\s*[-–]\s*\1\b",
    re.IGNORECASE,
)
DETAIL_CALLOUT_RE = re.compile(
    r"\bdetail\s+(?:no\.?\s*)?(\d+)\b",
    re.IGNORECASE,
)
DETAIL_SHEET_RE = re.compile(
    r"\bdetails?\s+(?:no\.?\s*)?(\d+)\b",
    re.IGNORECASE,
)
SHEET_REF_RE = re.compile(
    r"\b(?:see|refer to|on|sheet|page)\s+(?:sheet\s+)?(?:no\.?\s*)?([A-Z0-9][-A-Z0-9./]*)\b",
    re.IGNORECASE,
)
NOTE_CROSS_REF_RE = re.compile(
    r"\b(?:see|refer to|refer|shown on|details? on|section on|view on)\s+"
    r"(?:sheet|page|detail|section|view)\s+[^.;,\n]{2,60}",
    re.IGNORECASE,
)
SAME_SHEET_NOTE_RE = re.compile(
    r"\b(?:see|refer to)\s+(?:detail|section|view)\s+[^.;,\n]{2,40}",
    re.IGNORECASE,
)


@dataclass
class SheetGroupInfo:
    """Structural sheet group parsed from title block text."""

    group_name: str = ""
    group_sheet_index: Optional[int] = None
    group_sheet_role: str = ""
    sheet_title: str = ""

    def as_dict(self) -> Dict[str, str]:
        payload = {"sheet_title": self.sheet_title}
        if self.group_name:
            payload["group_name"] = self.group_name
        if self.group_sheet_index is not None:
            payload["group_sheet_index"] = str(self.group_sheet_index)
        if self.group_sheet_role:
            payload["group_sheet_role"] = self.group_sheet_role
        return payload


@dataclass
class DrawingClassification:
    """Label-driven classification for one drawing chunk."""

    view_type: str
    label: str = ""
    is_precise: bool = False
    category: str = ""

    def __post_init__(self) -> None:
        if not self.category:
            self.category = self.view_type


@dataclass
class CrossReferenceInfo:
    """Cross-reference links extracted from visible sheet text."""

    section_cuts: List[str] = field(default_factory=list)
    view_refs: List[str] = field(default_factory=list)
    detail_callouts: List[str] = field(default_factory=list)
    sheet_refs: List[str] = field(default_factory=list)
    note_refs: List[str] = field(default_factory=list)

    def all_refs(self) -> List[str]:
        refs: List[str] = []
        refs.extend(self.section_cuts)
        refs.extend(self.view_refs)
        refs.extend(self.detail_callouts)
        refs.extend(self.sheet_refs)
        refs.extend(self.note_refs)
        return refs


class CadDrawingKnowledge:
    """Classify CAD chunks and extract cross-reference topology from sheet text."""

    _QUERY_STOPWORDS: Set[str] = {
        "detail",
        "details",
        "section",
        "sections",
        "drawing",
        "drawings",
        "sheet",
        "sheets",
        "plan",
        "plans",
        "show",
        "find",
        "the",
        "a",
        "an",
        "of",
        "for",
        "on",
    }

    @classmethod
    def search_view_types(cls, view_type: str) -> List[str]:
        """Return view types that should match the same search intent."""
        normalized = (view_type or "").strip().lower()
        if normalized in DETAIL_SECTION_EQUIVALENT_TYPES:
            return ["detail", "section"]
        return [normalized] if normalized else []

    @classmethod
    def expand_detail_section_query(cls, query: str) -> str:
        """
        When a query asks for a component detail, also include the section variant
        so vector retrieval surfaces matching section chunks.
        """
        if not query:
            return query
        lowered = query.lower()
        has_detail = bool(re.search(r"\bdetails?\b", lowered))
        has_section = bool(re.search(r"\bsections?\b", lowered))
        if not has_detail or has_section:
            return query

        component_tokens = [
            token
            for token in re.findall(r"[a-z0-9]+", lowered)
            if token not in cls._QUERY_STOPWORDS and len(token) > 2
        ]
        if not component_tokens:
            return f"{query} section"

        component_phrase = " ".join(component_tokens[:4])
        return f"{query} {component_phrase} section"

    @staticmethod
    def _normalize_text(*parts: str) -> str:
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip().lower()

    @classmethod
    def infer_sheet_group(
        cls,
        sheet_title: str = "",
        primary_type: str = "",
    ) -> SheetGroupInfo:
        """Parse abutment/bent/girder group and sheet number within the group."""
        combined = cls._normalize_text(sheet_title, primary_type)
        info = SheetGroupInfo(sheet_title=sheet_title or primary_type)

        group_name = ""
        for name, pattern in SHEET_GROUP_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                group_name = name
                break
        info.group_name = group_name

        for role, pattern in SHEET_ROLE_PATTERNS:
            if re.search(pattern, combined, re.IGNORECASE):
                info.group_sheet_role = role
                break

        number_match = re.search(
            r"(?:no\.?|number|sheet)\s*(\d+)",
            combined,
            re.IGNORECASE,
        )
        if not number_match:
            number_match = re.search(r"\bno\.?\s*(\d+)\b", combined, re.IGNORECASE)
        if number_match:
            try:
                info.group_sheet_index = int(number_match.group(1))
            except Exception:
                pass

        return info

    @classmethod
    def classify_sheet_views(
        cls,
        sheet_title: str = "",
        primary_type: str = "",
        detail_types: Optional[List[str]] = None,
    ) -> List[str]:
        """Primary CAD view types present on a sheet (from title block)."""
        detail_types = detail_types or []
        combined = cls._normalize_text(
            sheet_title,
            primary_type,
            " ".join(detail_types),
        )
        views: Set[str] = set()

        if "general plan" in combined or re.search(r"\bplan\b", combined):
            views.add("plan")
        if "elevation" in combined:
            views.add("elevation")
        if "layout" in combined:
            views.add("layout")
        if re.search(r"\bdetails?\b", combined):
            views.add("detail")
        if re.search(r"\bsections?\b", combined) or SECTION_CUT_RE.search(combined):
            views.add("section")
        if re.search(r"\bview\b", combined) or VIEW_REF_RE.search(combined):
            views.add("view")

        if not views and "typical section" in combined:
            views.add("section")

        return sorted(views)

    @classmethod
    def _slugify_label(cls, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return slug or "drawing"

    @classmethod
    def infer_category_from_label(cls, label: str) -> str:
        """
        Derive a searchable category slug from label text.
        Uses known keywords first; otherwise slugifies the label.
        """
        normalized = cls._normalize_text(label)
        if not normalized:
            return "drawing"

        if VIEW_REF_RE.search(label):
            return "view"

        for keyword, category in _CATEGORY_KEYWORDS:
            if keyword in normalized:
                return category

        return cls._slugify_label(label)

    @classmethod
    def classify_label(cls, label: str) -> DrawingClassification:
        """
        Classify a drawing from its visible label text.
        Precise labels: SECTION A-A, DETAIL 1, PLAN STAGE 1, BENT CAP CAMBER DIAGRAM.
        Imprecise labels: TYPICAL SECTION, TYPICAL DETAILS, POST ANCHORAGE DETAIL.
        """
        cleaned = re.sub(r"\s+", " ", label).strip()
        if not cleaned:
            return DrawingClassification(view_type="detail", label="", is_precise=False)

        for pattern, view_type, is_precise in LABEL_CLASSIFICATION_RULES:
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                matched_label = match.group(0).strip()
                return DrawingClassification(
                    view_type=view_type,
                    label=matched_label.upper() if view_type in {"section", "detail", "view", "plan", "elevation"} else matched_label,
                    is_precise=is_precise,
                    category=view_type,
                )

        category = cls.infer_category_from_label(cleaned)
        fallback_type = category if category in CAD_DRAWING_VIEWS else "detail"
        return DrawingClassification(
            view_type=fallback_type,
            label=cleaned,
            is_precise=False,
            category=category,
        )

    @classmethod
    def extract_drawing_labels(cls, text: str) -> List[str]:
        """Extract candidate drawing labels from chunk text, most specific first."""
        if not text:
            return []

        labels: List[str] = []
        seen: Set[str] = set()

        def add_label(raw: str) -> None:
            cleaned = re.sub(r"\s+", " ", raw).strip(" .,-")
            key = cleaned.lower()
            if cleaned and key not in seen and len(cleaned) >= 4:
                seen.add(key)
                labels.append(cleaned)

        for pattern, _, _ in LABEL_CLASSIFICATION_RULES:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                add_label(match.group(0))

        return labels

    @classmethod
    def classify_chunk(
        cls,
        chunk_text: str,
        page_views: Optional[List[str]] = None,
        explicit_label: Optional[str] = None,
    ) -> DrawingClassification:
        """
        Classify a chunk using its drawing label when available.
        explicit_label is set when spatial/vision chunking provides the title.
        """
        if explicit_label:
            return cls.classify_label(explicit_label)

        labels = cls.extract_drawing_labels(chunk_text)
        if labels:
            classifications = [cls.classify_label(label) for label in labels]
            precise = [c for c in classifications if c.is_precise]
            if len(precise) == 1:
                return precise[0]
            if precise:
                return precise[0]
            return classifications[0]

        text = cls._normalize_text(chunk_text)
        scores: Dict[str, int] = {view: 0 for view in CAD_DRAWING_VIEWS}

        if re.search(r"\bsection\s+[a-z]\s*[-–]\s*[a-z]\b", text, re.IGNORECASE):
            scores["section"] += 8
        if DETAIL_CALLOUT_RE.search(text):
            scores["detail"] += 7
        if "diagram" in text:
            scores["diagram"] += 7
        if "elevation" in text:
            scores["elevation"] += 6
        if "layout" in text:
            scores["layout"] += 6
        if re.search(r"\bplan\b", text):
            scores["plan"] += 5
        if re.search(r"\bdetail\b", text):
            scores["detail"] += 4
        if "typical section" in text:
            scores["section"] += 5
        if VIEW_REF_RE.search(chunk_text):
            scores["view"] += 3

        best_view = max(scores.items(), key=lambda item: item[1])
        if best_view[1] > 0:
            return DrawingClassification(
                view_type=best_view[0],
                label="",
                is_precise=False,
                category=best_view[0],
            )

        if page_views:
            return DrawingClassification(
                view_type=page_views[0],
                label="",
                is_precise=False,
                category=page_views[0],
            )
        return DrawingClassification(view_type="detail", label="", is_precise=False, category="detail")

    @classmethod
    def classify_chunk_view(
        cls,
        chunk_text: str,
        page_views: Optional[List[str]] = None,
    ) -> str:
        """Backward-compatible helper returning only the view type string."""
        return cls.classify_chunk(chunk_text, page_views).view_type

    @classmethod
    def extract_cross_references(cls, text: str) -> CrossReferenceInfo:
        """Extract section cuts, views, detail callouts, and note-based sheet links."""
        info = CrossReferenceInfo()
        if not text:
            return info

        for match in SECTION_CUT_RE.finditer(text):
            label = match.group(0).strip()
            if label not in info.section_cuts:
                info.section_cuts.append(label)

        for match in VIEW_REF_RE.finditer(text):
            label = match.group(0).strip()
            if label not in info.view_refs:
                info.view_refs.append(label)

        for match in DETAIL_CALLOUT_RE.finditer(text):
            label = match.group(0).strip()
            if label not in info.detail_callouts:
                info.detail_callouts.append(label)

        for match in DETAIL_SHEET_RE.finditer(text):
            label = match.group(0).strip()
            if label not in info.detail_callouts:
                info.detail_callouts.append(label)

        for match in NOTE_CROSS_REF_RE.finditer(text):
            label = re.sub(r"\s+", " ", match.group(0)).strip()
            if label not in info.note_refs:
                info.note_refs.append(label)

        for match in SAME_SHEET_NOTE_RE.finditer(text):
            label = re.sub(r"\s+", " ", match.group(0)).strip()
            if label not in info.note_refs:
                info.note_refs.append(label)

        for match in SHEET_REF_RE.finditer(text):
            label = match.group(0).strip()
            if label.lower().startswith(("see sheet", "refer to sheet", "on sheet")):
                if label not in info.sheet_refs:
                    info.sheet_refs.append(label)

        return info

    @classmethod
    def format_cross_reference_summary(cls, refs: CrossReferenceInfo) -> List[str]:
        """Human-readable cross-reference lines for embedding."""
        lines: List[str] = []
        if refs.section_cuts:
            lines.append(f"Section cuts: {', '.join(refs.section_cuts[:12])}")
        if refs.view_refs:
            lines.append(f"Views: {', '.join(refs.view_refs[:12])}")
        if refs.detail_callouts:
            lines.append(f"Detail callouts: {', '.join(refs.detail_callouts[:12])}")
        if refs.sheet_refs:
            lines.append(f"Referenced sheets: {', '.join(refs.sheet_refs[:8])}")
        if refs.note_refs:
            lines.append(f"Note cross-references: {', '.join(refs.note_refs[:8])}")
        return lines

    @classmethod
    def build_page_context(
        cls,
        page_metadata: Optional[Dict] = None,
        page_text: str = "",
    ) -> Dict[str, str]:
        """
        Build CAD knowledge metadata fields for a page (applied to each chunk on the page).
        """
        page_metadata = page_metadata or {}
        title_block = page_metadata.get("title_block") or {}
        plan_type = title_block.get("plan_type") or {}
        contents = title_block.get("plan_contents") or {}

        sheet_title = str(plan_type.get("sheet_title") or "")
        primary_type = str(plan_type.get("primary_type") or "")
        detail_types = contents.get("detail_types") or []
        if isinstance(detail_types, str):
            detail_types = [detail_types]

        cad_block = title_block.get("cad_drawing") or {}

        group = cls.infer_sheet_group(sheet_title, primary_type)
        if cad_block.get("sheet_group"):
            group.group_name = str(cad_block["sheet_group"]).strip().lower().replace(" ", "_")
        if cad_block.get("sheet_group_role"):
            group.group_sheet_role = str(cad_block["sheet_group_role"]).strip().lower()
        if cad_block.get("sheet_group_number"):
            try:
                group.group_sheet_index = int(str(cad_block["sheet_group_number"]).strip())
            except Exception:
                pass

        sheet_views = cls.classify_sheet_views(sheet_title, primary_type, detail_types)
        vision_views = cad_block.get("sheet_views") or []
        if isinstance(vision_views, list):
            sheet_views = sorted(set(sheet_views) | {str(v).strip().lower() for v in vision_views if v})

        cross_refs = cls.extract_cross_references(page_text)
        for key, target in (
            ("section_cuts", cross_refs.section_cuts),
            ("view_refs", cross_refs.view_refs),
            ("detail_callouts", cross_refs.detail_callouts),
            ("cross_reference_notes", cross_refs.note_refs),
        ):
            values = cad_block.get(key) or []
            if isinstance(values, list):
                for value in values:
                    label = str(value).strip()
                    if label and label not in target:
                        target.append(label)

        fields: Dict[str, str] = {}
        if sheet_views:
            fields["cad_sheet_views"] = ", ".join(sheet_views)
        if group.group_name:
            fields["cad_sheet_group"] = group.group_name
        if group.group_sheet_role:
            fields["cad_sheet_group_role"] = group.group_sheet_role
        if group.group_sheet_index is not None:
            fields["cad_sheet_group_index"] = str(group.group_sheet_index)
        if cross_refs.section_cuts:
            fields["cad_section_cuts"] = ", ".join(cross_refs.section_cuts[:20])
        if cross_refs.view_refs:
            fields["cad_view_refs"] = ", ".join(cross_refs.view_refs[:20])
        if cross_refs.detail_callouts:
            fields["cad_detail_callouts"] = ", ".join(cross_refs.detail_callouts[:20])
        if cross_refs.sheet_refs:
            fields["cad_sheet_refs"] = ", ".join(cross_refs.sheet_refs[:12])
        if cross_refs.note_refs:
            fields["cad_note_refs"] = ", ".join(cross_refs.note_refs[:12])

        xref_summary = cls.format_cross_reference_summary(cross_refs)
        if xref_summary:
            fields["cad_cross_reference_summary"] = " | ".join(xref_summary)

        return fields

    @classmethod
    def _is_region_chunk(cls, chunk_text: str, explicit_label: Optional[str] = None) -> bool:
        return bool(explicit_label) or "[DRAWING REGION CHUNK]" in (chunk_text or "")

    @classmethod
    def build_chunk_fields(
        cls,
        chunk_text: str,
        page_metadata: Optional[Dict] = None,
        explicit_label: Optional[str] = None,
    ) -> Tuple[Dict[str, str], str]:
        """
        Build per-chunk CAD metadata and searchable text block.

        Returns:
            (metadata_fields, searchable_text_block)
        """
        region_chunk = cls._is_region_chunk(chunk_text, explicit_label)
        region_body = cls.strip_cad_knowledge_block(chunk_text or "")
        page_fields = cls.build_page_context(
            page_metadata,
            "" if region_chunk else chunk_text,
        )
        if region_chunk:
            for key in (
                "cad_section_cuts",
                "cad_view_refs",
                "cad_detail_callouts",
                "cad_sheet_refs",
                "cad_note_refs",
                "cad_cross_reference_summary",
            ):
                page_fields.pop(key, None)

        sheet_views = [
            view.strip()
            for view in str(page_fields.get("cad_sheet_views", "")).split(",")
            if view.strip()
        ]
        if explicit_label:
            classification = cls.classify_label(explicit_label)
        else:
            classification = cls.classify_chunk(chunk_text, sheet_views)
        chunk_xrefs = cls.extract_cross_references(region_body)

        fields = dict(page_fields)
        fields["cad_drawing_view"] = classification.view_type
        if classification.label:
            fields["cad_drawing_label"] = classification.label
        fields["cad_drawing_category"] = classification.category
        fields["cad_drawing_precise"] = "true" if classification.is_precise else "false"
        search_view_types = cls.search_view_types(classification.view_type)
        if search_view_types:
            fields["cad_search_view_types"] = ", ".join(search_view_types)
        fields["cad_leader_attribution"] = "leader_target"

        lines = [
            f"CAD Drawing View: {classification.view_type}",
        ]
        if classification.label:
            precision = "precise" if classification.is_precise else "general"
            lines.append(f"Drawing Label: {classification.label} ({precision})")
        if classification.category and classification.category != classification.view_type:
            lines.append(f"Drawing Category: {classification.category}")
        if sheet_views:
            lines.append(f"Sheet Views: {', '.join(sheet_views)}")
        if page_fields.get("cad_sheet_group"):
            group_line = f"Sheet Group: {page_fields['cad_sheet_group']}"
            if page_fields.get("cad_sheet_group_role"):
                group_line += f" ({page_fields['cad_sheet_group_role']})"
            if page_fields.get("cad_sheet_group_index"):
                group_line += f" sheet {page_fields['cad_sheet_group_index']}"
            lines.append(group_line)

        chunk_xref_lines = cls.format_cross_reference_summary(chunk_xrefs)
        if chunk_xref_lines:
            lines.extend(chunk_xref_lines)
        elif not region_chunk and page_fields.get("cad_cross_reference_summary"):
            lines.append(page_fields["cad_cross_reference_summary"])

        lines.append(
            "Cross-reference rules: section/view/detail on a sheet reference plan, "
            "elevation, layout, detail, section, or view on the same or another sheet "
            "via section cut (A-A), view (View B-B), or detail callout (Detail 1). "
            "Off-sheet targets are usually noted in sheet notes."
        )
        lines.append(LEADER_ATTRIBUTION_RULE)
        if classification.view_type in DETAIL_SECTION_EQUIVALENT_TYPES:
            lines.append(DETAIL_SECTION_SEARCH_RULE)
            lines.append(
                f"Search view aliases: {', '.join(search_view_types)}"
            )

        return fields, "\n".join(lines)

    @classmethod
    def strip_cad_knowledge_block(cls, chunk_text: str) -> str:
        marker = "[CAD DRAWING KNOWLEDGE]"
        if marker not in chunk_text:
            return chunk_text
        return chunk_text.split(marker, 1)[0].rstrip()

    @classmethod
    def append_to_chunk_text(cls, chunk_text: str, knowledge_block: str) -> str:
        if not knowledge_block:
            return chunk_text
        base_text = cls.strip_cad_knowledge_block(chunk_text)
        return f"{base_text}\n\n[CAD DRAWING KNOWLEDGE]\n{knowledge_block}"

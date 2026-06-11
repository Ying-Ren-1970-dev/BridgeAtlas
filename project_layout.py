"""Project-level layout signals from General Plan sheets and early index pages."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

PROJECT_DESCRIPTOR_EXCLUDED_SHEET_TYPES: Set[str] = {
    "foundation_plan",
    "bridge_removal",
    "rebar",
}


def span_count_from_bent_number_lines(content: str) -> Optional[int]:
    """Parse ramp-style 'Bent numbers: Bent 1, Bent 2, ...' layout lists."""
    best_count = None
    for match in re.findall(r"Bent numbers:\s*([^|\n]+)", content, re.IGNORECASE):
        bents = [
            int(value)
            for value in re.findall(r"\bBent\s*(\d+)\b", match, re.IGNORECASE)
            if 1 <= int(value) <= 12
        ]
        if len(bents) < 2 or 1 not in bents:
            continue
        candidate = len(bents)
        if best_count is None or candidate > best_count:
            best_count = candidate
    return best_count


def infer_span_count_from_general_plan_items(gp_items: List[Dict]) -> Optional[int]:
    """
    Infer span count from General Plan layout only.

    One interior bent (Bent 2) between abutments => 2-span bridge.
    Bent 2 and Bent 3 on GP => 3-span bridge.
    """
    interior_bents: Set[int] = set()

    for item in gp_items:
        metadata = item.get("metadata", {}) or {}
        try:
            page_number = int(metadata.get("page") or 0)
        except Exception:
            continue
        if page_number > 2:
            continue

        content = item.get("content", "") or ""
        visible = content
        for marker in ("[TOPOLOGY PROJECT]", "[topology project]"):
            if marker in content:
                visible = content.split(marker, 1)[0]
                break

        for grid in re.findall(r"Grid References:\s*([^|\n\]]+)", visible, re.IGNORECASE):
            for bent in re.findall(r"\bBent\s*(\d+)\b", grid, re.IGNORECASE):
                bent_number = int(bent)
                if bent_number >= 2:
                    interior_bents.add(bent_number)

        if "[TOPOLOGY DETAIL]" in content:
            detail = content.split("[TOPOLOGY DETAIL]", 1)[1][:900]
            for bent in re.findall(r"\bBent\s*(\d+)\b", detail, re.IGNORECASE):
                bent_number = int(bent)
                if bent_number >= 2:
                    interior_bents.add(bent_number)

    if not interior_bents:
        return None
    return len(interior_bents) + 1


def collect_layout_signals(
    file_name: str,
    vector_store,
    keyword_candidates: Optional[List[Dict]] = None,
) -> Dict:
    """Collect deterministic layout/material signals for one project."""
    items_by_file: List[Dict] = []
    if keyword_candidates:
        for item in keyword_candidates:
            metadata = item.get("metadata", {}) or {}
            if str(metadata.get("file_name") or "") != file_name:
                continue
            page_number = metadata.get("page")
            try:
                page_number = int(page_number)
            except Exception:
                continue
            if page_number > 7:
                continue
            items_by_file.append(item)

    concrete_bridge = False
    is_bridge = False
    gp_page = None
    gp_items: List[Dict] = []
    ramp_span = None
    gp_titles: List[Dict] = []

    for item in items_by_file:
        content = item.get("content", "") or ""
        metadata = item.get("metadata", {}) or {}
        content_lower = content.lower()
        sheet_type = str(metadata.get("plan_sheet_type") or "")

        if sheet_type == "general_plan":
            gp_items.append(item)
            page_number = int(metadata.get("page") or 0)
            gp_titles.append(
                {
                    "page": page_number,
                    "title": metadata.get("plan_sheet_title") or metadata.get("topology_sheet_title"),
                }
            )
            if gp_page is None:
                gp_page = page_number

        if (
            "structure type: bridge" in content_lower
            or "structure type > bridge" in content_lower
            or "general plan > structure type > bridge" in content_lower
        ):
            is_bridge = True
        if "cip concrete box girder" in content_lower or "concrete box girder" in content_lower:
            concrete_bridge = True

        try:
            page_number = int(metadata.get("page") or 0)
        except Exception:
            page_number = 0
        if page_number <= 7:
            bent_line_span = span_count_from_bent_number_lines(content)
            if bent_line_span is not None:
                ramp_span = max(ramp_span or 0, bent_line_span)

    if not gp_items:
        for page_number in range(1, 4):
            chunk = vector_store.get_page_chunk(file_name, page_number)
            if not chunk:
                continue
            metadata = chunk.get("metadata", {}) or {}
            if str(metadata.get("plan_sheet_type") or "") != "general_plan":
                continue
            gp_page = page_number
            gp_items.append(
                {
                    "content": chunk.get("content", ""),
                    "metadata": metadata,
                }
            )
            gp_titles.append(
                {
                    "page": page_number,
                    "title": metadata.get("plan_sheet_title") or metadata.get("topology_sheet_title"),
                }
            )
            break

    if concrete_bridge:
        span_count = infer_span_count_from_general_plan_items(gp_items)
    else:
        span_count = ramp_span or infer_span_count_from_general_plan_items(gp_items)

    interior_bents: Set[int] = set()
    for item in gp_items:
        content = item.get("content", "") or ""
        if "[TOPOLOGY DETAIL]" in content:
            detail = content.split("[TOPOLOGY DETAIL]", 1)[1][:900]
            for bent in re.findall(r"\bBent\s*(\d+)\b", detail, re.IGNORECASE):
                n = int(bent)
                if n >= 2:
                    interior_bents.add(n)

    return {
        "span_count": span_count,
        "concrete": concrete_bridge,
        "bridge": is_bridge,
        "gp_page": gp_page,
        "gp_items": gp_items,
        "gp_titles": gp_titles,
        "interior_bents": sorted(interior_bents),
    }

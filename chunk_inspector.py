"""Inspect indexed chunks for a PDF page and compare to expected drawing labels."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from cad_drawing_knowledge import CadDrawingKnowledge
from metadata_manager import MetadataManager

# Known annotated example pages from training review.
KNOWN_PAGE_EXPECTATIONS: Dict[Tuple[str, int], List[str]] = {
    ("55-1119_GoldenwestOc_As_BuiltWM.pdf", 17): [
        "SECTION B-B",
        "JOINT PROTECTION DETAIL",
        '24" ø STEEL PIPE PILE ELEVATION',
        "SECTION Y-Y",
        "AS BUILT SECTION Y-Y",
    ],
    ("55-1119_GoldenwestOc_As_BuiltWM.pdf", 21): [
        "DETAIL 1",
        "DETAIL 2",
        "DETAIL 3",
        "SECTION A-A",
        "SECTION B-B",
        "SECTION C-C",
        "DETAIL 5",
    ],
    ("55-1127_WestminsterOC_As_Built.pdf", 11): [
        "AS BUILT",
        "PLAN",
        "ELEVATION",
    ],
    ("55-1121_MagnoliaOc_As_BuiltWM.pdf", 10): [
        "PLAN STAGE 1",
        "ELEVATION STAGE 1",
        "SECTION B-B",
        "SECTION A-A",
    ],
    ("55-1119_GoldenwestOc_As_BuiltWM.pdf", 61): [
        "TYPICAL SECTION",
        "ELEVATION",
        "TYPICAL DETAILS",
        "TOP CABLE ANCHORAGE DETAIL",
        "POST ANCHORAGE DETAIL",
        "DETAIL 1",
    ],
    ("Fresno 180 Ramp.pdf", 12): [
        "SECTION A-A",
        "SECTION B-B",
        "BENT CAP CAMBER DIAGRAM",
        "PLAN",
        "ELEVATION",
    ],
}


@dataclass
class ChunkInspection:
    chunk_id: int
    view_type: str = ""
    label: str = ""
    category: str = ""
    precise: str = ""
    search_view_types: str = ""
    leader_attribution: str = ""
    char_length: int = 0
    matched_labels: List[str] = field(default_factory=list)
    is_mixed: bool = False
    text_preview: str = ""


@dataclass
class PageInspectionReport:
    file_name: str
    page: int
    chunk_count: int
    expected_labels: List[str]
    chunks: List[ChunkInspection]
    missing_labels: List[str]
    extra_labels: List[str]
    mixed_chunks: int
    passed: bool
    notes: List[str] = field(default_factory=list)


def resolve_file_name(file_input: str) -> Optional[str]:
    """Resolve a full or partial PDF file name against indexed projects."""
    if not file_input:
        return None

    candidate = file_input.strip()
    if not candidate.lower().endswith(".pdf"):
        candidate = f"{candidate}.pdf"

    active_files = sorted(MetadataManager().get_active_pdf_file_names())
    if candidate in active_files:
        return candidate

    lowered = candidate.lower()
    partial_matches = [
        name for name in active_files if lowered.replace(".pdf", "") in name.lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        raise ValueError(
            "Multiple PDFs match "
            f"'{file_input}': {', '.join(partial_matches)}. Use the full file name."
        )
    return None


def _label_in_text(label: str, text: str) -> bool:
    pattern = rf"(?<!\w){re.escape(label).replace(r'\ ', r'\s+')}(?!\w)"
    return bool(re.search(pattern, text, re.IGNORECASE))


def _find_matched_labels(
    text: str,
    expected_labels: Sequence[str],
    primary_label: str = "",
) -> List[str]:
    primary = re.sub(r"\s+", " ", primary_label).strip().upper()
    matched: List[str] = []
    primary_norm = re.sub(r"\s+", " ", primary_label).strip().upper()
    for label in expected_labels:
        normalized = re.sub(r"\s+", " ", label).strip().upper()
        if primary_norm and normalized != primary_norm and normalized in primary_norm:
            # e.g. SECTION Y-Y inside AS BUILT SECTION Y-Y is not a separate match.
            continue
        if _label_in_text(label, text):
            matched.append(label)
            continue
        if primary_norm and (
            normalized == primary_norm
            or primary_norm.startswith(f"{normalized} ")
            or primary_norm.startswith(f"{normalized}-")
        ):
            matched.append(label)
    return matched


def parse_labels_arg(labels_arg: str) -> List[str]:
    if not labels_arg:
        return []
    parts = re.split(r"[|,]", labels_arg)
    return [part.strip() for part in parts if part.strip()]


def inspect_page_chunks(
    file_name: str,
    page: int,
    expected_labels: Optional[Sequence[str]] = None,
    verbose: bool = False,
    show_text: bool = False,
    collection=None,
) -> PageInspectionReport:
    """Load chunks for one page and compare them to expected drawing labels."""
    resolved_name = resolve_file_name(file_name)
    if not resolved_name:
        raise ValueError(f"No indexed PDF matches '{file_name}'")

    expected = list(expected_labels or KNOWN_PAGE_EXPECTATIONS.get((resolved_name, page), []))
    notes: List[str] = []
    if not expected:
        notes.append(
            "No built-in expected labels for this page. "
            "Pass --labels \"SECTION A-A,DETAIL 1,...\" to enable pass/fail checks."
        )

    if collection is None:
        from vector_store import VectorStore

        vector_store = VectorStore()
        vector_store.initialize_vectorstore()
        collection = vector_store.vectorstore._collection

    results = collection.get(
        where={"file_name": resolved_name},
        include=["metadatas", "documents"],
    )

    rows: List[Tuple[dict, str]] = []
    for metadata, document in zip(results["metadatas"], results["documents"]):
        if metadata.get("page") == page:
            rows.append((metadata, document))

    rows.sort(key=lambda item: int(item[0].get("chunk_id") or 0))
    chunk_reports: List[ChunkInspection] = []
    found_labels: List[str] = []
    mixed_chunks = 0

    for metadata, document in rows:
        primary_label = str(metadata.get("cad_drawing_label") or "")
        matched = (
            _find_matched_labels(document, expected, primary_label)
            if expected
            else []
        )
        if matched:
            found_labels.extend(matched)
        is_mixed = len(matched) > 1
        if is_mixed:
            mixed_chunks += 1

        preview = re.sub(r"\s+", " ", document).strip()
        if len(preview) > 220:
            preview = preview[:220] + "..."

        chunk_reports.append(
            ChunkInspection(
                chunk_id=int(metadata.get("chunk_id") or 0),
                view_type=str(metadata.get("cad_drawing_view") or ""),
                label=str(metadata.get("cad_drawing_label") or ""),
                category=str(metadata.get("cad_drawing_category") or ""),
                precise=str(metadata.get("cad_drawing_precise") or ""),
                search_view_types=str(metadata.get("cad_search_view_types") or ""),
                leader_attribution=str(metadata.get("cad_leader_attribution") or ""),
                char_length=len(document),
                matched_labels=matched,
                is_mixed=is_mixed,
                text_preview=preview if verbose or show_text else "",
            )
        )

    missing_labels = []
    extra_labels = []
    if expected:
        missing_labels = [label for label in expected if label not in found_labels]
        extra_labels = sorted(set(found_labels) - set(expected))

    passed = True
    if expected:
        passed = (
            len(rows) == len(expected)
            and mixed_chunks == 0
            and not missing_labels
            and all(len(chunk.matched_labels) <= 1 for chunk in chunk_reports)
            and sum(1 for chunk in chunk_reports if chunk.matched_labels) == len(expected)
        )

    return PageInspectionReport(
        file_name=resolved_name,
        page=page,
        chunk_count=len(rows),
        expected_labels=list(expected),
        chunks=chunk_reports,
        missing_labels=missing_labels,
        extra_labels=extra_labels,
        mixed_chunks=mixed_chunks,
        passed=passed,
        notes=notes,
    )


def format_report(report: PageInspectionReport, show_text: bool = False) -> str:
    """Render a human-readable inspection report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("CHUNK INSPECTION REPORT")
    lines.append("=" * 72)
    lines.append(f"File:   {report.file_name}")
    lines.append(f"Page:   {report.page}")
    lines.append(f"Chunks: {report.chunk_count}")

    if report.expected_labels:
        lines.append(f"Expected drawing chunks: {len(report.expected_labels)}")
        for idx, label in enumerate(report.expected_labels, 1):
            lines.append(f"  {idx}. {label}")
    else:
        lines.append("Expected drawing chunks: (not provided)")

    lines.append("-" * 72)
    if not report.chunks:
        lines.append("No chunks indexed for this page.")
    else:
        for chunk in report.chunks:
            status = "MIXED" if chunk.is_mixed else "OK"
            if report.expected_labels and not chunk.matched_labels:
                status = "UNMATCHED"
            lines.append(
                f"chunk {chunk.chunk_id}: view={chunk.view_type or '-'} "
                f"label={chunk.label or '-'} precise={chunk.precise or '-'} "
                f"[{status}]"
            )
            if chunk.category:
                lines.append(f"  category: {chunk.category}")
            if chunk.search_view_types:
                lines.append(f"  search_view_types: {chunk.search_view_types}")
            if chunk.leader_attribution:
                lines.append(f"  leader_attribution: {chunk.leader_attribution}")
            lines.append(f"  size: {chunk.char_length} chars")
            if chunk.matched_labels:
                lines.append(f"  matched expected labels: {', '.join(chunk.matched_labels)}")
            elif report.expected_labels:
                lines.append("  matched expected labels: none")
            if chunk.label:
                classified = CadDrawingKnowledge.classify_label(chunk.label)
                if classified.view_type and classified.view_type != chunk.view_type:
                    lines.append(
                        f"  label reclassify hint: {classified.view_type} "
                        f"(precise={classified.is_precise})"
                    )
            if show_text and chunk.text_preview:
                lines.append(f"  preview: {chunk.text_preview}")

    lines.append("-" * 72)
    if report.expected_labels:
        lines.append(f"Mixed chunks: {report.mixed_chunks}")
        if report.missing_labels:
            lines.append(f"Missing labels: {', '.join(report.missing_labels)}")
        else:
            lines.append("Missing labels: none")
        if report.extra_labels:
            lines.append(f"Extra matched labels: {', '.join(report.extra_labels)}")
        else:
            lines.append("Extra matched labels: none")

    for note in report.notes:
        lines.append(f"Note: {note}")

    lines.append("=" * 72)
    if report.expected_labels:
        lines.append("RESULT: PASS" if report.passed else "RESULT: FAIL")
        if not report.passed:
            lines.append(
                "Typical causes: text-flow chunking instead of drawing-region chunks, "
                "or multiple drawing labels inside one chunk."
            )
    else:
        lines.append("RESULT: INFO ONLY (no expected labels to compare)")
    lines.append("=" * 72)
    return "\n".join(lines)


def parse_inspect_chunks_args(argv: Sequence[str]) -> dict:
    """Parse inspect-chunks CLI flags from argv."""
    args = {
        "file_name": None,
        "page": None,
        "labels": None,
        "verbose": False,
        "show_text": False,
    }
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--file" and index + 1 < len(argv):
            args["file_name"] = argv[index + 1]
            index += 2
            continue
        if token == "--page" and index + 1 < len(argv):
            args["page"] = int(argv[index + 1])
            index += 2
            continue
        if token == "--labels" and index + 1 < len(argv):
            args["labels"] = argv[index + 1]
            index += 2
            continue
        if token == "--verbose":
            args["verbose"] = True
            index += 1
            continue
        if token == "--show-text":
            args["show_text"] = True
            index += 1
            continue
        index += 1
    return args

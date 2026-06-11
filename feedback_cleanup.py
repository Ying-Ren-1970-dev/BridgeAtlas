"""Utilities for keeping search feedback aligned with active indexed projects."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional, Set

from metadata_manager import MetadataManager


def active_pdf_file_names(metadata_manager: Optional[MetadataManager] = None) -> Set[str]:
    """Return PDF file names that are currently indexed in metadata."""
    manager = metadata_manager or MetadataManager()
    return set(manager.get_all_projects().keys())


def is_active_feedback_file(file_name: str, active_files: Optional[Set[str]] = None) -> bool:
    """Return True when feedback for this PDF should be honored."""
    if not file_name:
        return False
    allowed = active_files if active_files is not None else active_pdf_file_names()
    return file_name in allowed


def purge_legacy_feedback(
    feedback_path: Optional[str] = None,
    metadata_manager: Optional[MetadataManager] = None,
) -> Dict[str, int]:
    """
    Remove feedback rows for PDFs that are no longer in the active project library.

    Returns counts: kept, removed, active_projects.
    """
    root = os.path.dirname(__file__)
    path = feedback_path or os.path.join(root, "data", "feedback", "search_feedback.jsonl")
    active_files = active_pdf_file_names(metadata_manager)

    if not os.path.exists(path):
        return {"kept": 0, "removed": 0, "active_projects": len(active_files)}

    kept_lines: list[str] = []
    removed = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except Exception:
                removed += 1
                continue

            file_name = str(record.get("pdf_file_name", "")).strip()
            if file_name not in active_files:
                removed += 1
                continue

            kept_lines.append(raw)

    with open(path, "w", encoding="utf-8") as handle:
        if kept_lines:
            handle.write("\n".join(kept_lines) + "\n")

    return {
        "kept": len(kept_lines),
        "removed": removed,
        "active_projects": len(active_files),
    }

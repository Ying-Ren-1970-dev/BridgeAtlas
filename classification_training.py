"""Training-driven classification for project/page/detail indexing levels.

This module learns a lightweight taxonomy from
`Training Materials/search classifications.md` and applies it to:
- project (file) level
- page (sheet) level
- detail (chunk) level
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
from collections import defaultdict

import config


def _normalize(value: str) -> str:
    text = (value or "").lower()
    text = re.sub(r"&#x20;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9/+.-]*", _normalize(value))


@dataclass
class ClassifiedDetail:
    labels: List[str]
    referenced_sheet_ids: List[str]
    referenced_detail_ids: List[str]


class SearchClassificationTrainer:
    """Parses training markdown and classifies text across three index levels."""

    def __init__(self, training_file: Path | None = None):
        self.training_file = training_file or (config.BASE_DIR / "Training Materials" / "search classifications.md")
        self.cache_file = config.DATA_FOLDER / "search_classification_training.json"
        self.taxonomy_paths: Dict[str, List[List[str]]] = {
            "project": [],
            "page": [],
        }
        self.lexicon: Dict[str, Dict[str, List[str]]] = {
            "project": {},
            "page": {},
        }
        self._loaded = False

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_or_build()
        self._loaded = True

    def _load_or_build(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.taxonomy_paths = data.get("taxonomy_paths", self.taxonomy_paths)
                self.lexicon = data.get("lexicon", self.lexicon)
                return
            except Exception:
                pass

        self._parse_training_markdown()
        self._build_lexicon()
        self.build_training_artifact()

    def _parse_training_markdown(self) -> None:
        if not self.training_file.exists():
            return

        lines = self.training_file.read_text(encoding="utf-8").splitlines()

        current_scope = None
        stack: List[Tuple[int, str]] = []

        for raw in lines:
            line = raw.rstrip()
            if not line.strip():
                continue

            normalized_line = _normalize(line)
            if "project level information" in normalized_line:
                current_scope = "project"
                stack = []
                continue
            if "sheet/page level information" in normalized_line:
                current_scope = "page"
                stack = []
                continue

            if current_scope not in {"project", "page"}:
                continue

            bullet = re.match(r"^(\s*)[*-]\s+(.*)$", line)
            if not bullet:
                continue

            indent = len(bullet.group(1))
            text = bullet.group(2).strip()
            text = re.sub(r"\s+", " ", text)
            text = text.strip(":")
            if not text:
                continue

            while stack and stack[-1][0] >= indent:
                stack.pop()
            stack.append((indent, text))

            path = [p[1] for p in stack]
            if len(path) >= 2:
                self.taxonomy_paths[current_scope].append(path)

    def _build_lexicon(self) -> None:
        for scope in ("project", "page"):
            score_terms: Dict[str, set] = defaultdict(set)
            for path in self.taxonomy_paths.get(scope, []):
                label = " > ".join(path)
                for segment in path:
                    seg_norm = _normalize(segment)
                    if not seg_norm:
                        continue
                    score_terms[label].add(seg_norm)
                    for token in _tokenize(seg_norm):
                        if len(token) >= 4:
                            score_terms[label].add(token)
            self.lexicon[scope] = {k: sorted(v) for k, v in score_terms.items()}

    def build_training_artifact(self) -> Dict:
        """Build and persist parsed taxonomy/lexicon artifact."""
        config.DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        if not self.taxonomy_paths["project"] and not self.taxonomy_paths["page"]:
            self._parse_training_markdown()
            self._build_lexicon()

        artifact = {
            "training_file": str(self.training_file),
            "taxonomy_paths": self.taxonomy_paths,
            "lexicon": self.lexicon,
            "stats": {
                "project_paths": len(self.taxonomy_paths.get("project", [])),
                "page_paths": len(self.taxonomy_paths.get("page", [])),
                "project_labels": len(self.lexicon.get("project", {})),
                "page_labels": len(self.lexicon.get("page", {})),
            },
        }

        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2, ensure_ascii=False)

        return artifact

    def _score_scope(self, scope: str, text: str, top_k: int = 6) -> List[str]:
        text_norm = _normalize(text)
        if not text_norm:
            return []

        scores: List[Tuple[str, float]] = []
        for label, terms in self.lexicon.get(scope, {}).items():
            score = 0.0
            for term in terms:
                if len(term) <= 2:
                    continue
                if term in text_norm:
                    score += 2.0 if " " in term else 1.0
            if score > 0:
                scores.append((label, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [label for label, _ in scores[:top_k]]

    def classify_project(self, text: str) -> List[str]:
        self.ensure_loaded()
        return self._score_scope("project", text, top_k=8)

    def classify_page(self, text: str) -> List[str]:
        self.ensure_loaded()
        return self._score_scope("page", text, top_k=8)

    CANONICAL_SUPERSTRUCTURE_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"cip\s*/?\s*ps\s*/?\s*pt.*box\s*girder", re.I), "CIP PS/PT Box Girder"),
        (re.compile(r"cip.*box\s*girder|\bbox\s*girder\b", re.I), "CIP Concrete Box Girder"),
        (re.compile(r"cip.*concrete\s*slab", re.I), "CIP concrete slab"),
        (re.compile(r"precast.*u\s*tub", re.I), "Precast U tub"),
        (re.compile(r"precast.*i\s*girder", re.I), "Precast I girder"),
        (re.compile(r"precast.*box", re.I), "precast Concrete Box"),
        (re.compile(r"\btruss\b", re.I), "Truss"),
        (re.compile(r"plate\s*girder", re.I), "Plate girder"),
        (re.compile(r"steel\s*i\s*girder", re.I), "Steel I girder"),
        (re.compile(r"cable\s*stayed", re.I), "Cable stayed"),
    ]

    CANONICAL_FOUNDATION_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"\bcidh\b|cast[- ]in[- ]drilled", re.I), "CIDH pile - drilled shaft"),
        (re.compile(r"drilled\s*shaft", re.I), "Large diameter drilled shaft (5' or larger CIDH)"),
        (re.compile(r"steel\s*pipe\s*pile|pipe\s*pile", re.I), "Steel pipe"),
        (re.compile(r"spread\s*footing", re.I), "spread footing"),
        (re.compile(r"pile\s*cap", re.I), "pile group with pile cap"),
    ]

    def build_topology_project_context(self, topology_data: Dict) -> str:
        """Aggregate topology evidence for project-level taxonomy matching."""
        parts = [str(topology_data.get("project_name") or "")]
        for _category, elements in (topology_data.get("topology") or {}).items():
            if not isinstance(elements, dict):
                continue
            for element_name, occurrences in elements.items():
                parts.append(str(element_name))
                if not isinstance(occurrences, list):
                    continue
                for occurrence in occurrences[:3]:
                    details = occurrence.get("details") or {}
                    for key in ("type", "name", "detail_name", "description", "material", "specification"):
                        value = details.get(key)
                        if value:
                            parts.append(str(value))
        return " | ".join(p for p in parts if p)

    def _match_canonical_patterns(self, text: str, patterns: List[Tuple[re.Pattern, str]]) -> List[str]:
        matches = []
        for pattern, label in patterns:
            if pattern.search(text or ""):
                matches.append(label)
        deduped = []
        seen = set()
        for label in matches:
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(label)
        return deduped

    def summarize_project_labels(self, labels: List[str]) -> Dict[str, str]:
        """Collapse taxonomy paths into engineer-facing project categories."""
        summary: Dict[str, str] = {}
        for label in labels:
            parts = [p.strip() for p in label.split(">")]
            lower_parts = [_normalize(p) for p in parts]

            if "structure type" in lower_parts:
                for candidate in ("bridge", "tunnel", "retaining wall", "culvert", "station", "building"):
                    if candidate in " ".join(lower_parts):
                        summary["structure_type"] = candidate
                        break

            if "superstructure" in " ".join(lower_parts):
                summary["bridge_superstructure"] = parts[-1]
            if "foundation" in " ".join(lower_parts):
                summary["bridge_foundation"] = parts[-1]
            if "substructure" in " ".join(lower_parts):
                summary["bridge_substructure"] = parts[-1]
            if "general note sheet" in " ".join(lower_parts):
                summary["sheet_family"] = "General Note sheet"

        return summary

    def classify_topology_project(self, topology_data: Dict) -> Dict[str, str]:
        """
        Classify project-level structure/bridge type using search classifications.md.
        """
        self.ensure_loaded()
        context = self.build_topology_project_context(topology_data)
        labels = self.classify_project(context)
        summary = self.summarize_project_labels(labels)

        canonical_super = self._match_canonical_patterns(
            context, self.CANONICAL_SUPERSTRUCTURE_PATTERNS
        )
        canonical_foundation = self._match_canonical_patterns(
            context, self.CANONICAL_FOUNDATION_PATTERNS
        )

        if canonical_super:
            summary["bridge_type"] = canonical_super[0]
            summary["bridge_superstructure"] = canonical_super[0]
            summary["structure_type"] = "bridge"
        elif summary.get("bridge_superstructure"):
            summary["bridge_type"] = summary["bridge_superstructure"]
            if re.search(r"\bbridge\b", _normalize(context)):
                summary["structure_type"] = "bridge"

        if canonical_foundation:
            summary["bridge_foundation"] = canonical_foundation[0]

        if summary.get("bridge_type") or summary.get("bridge_superstructure"):
            summary["structure_type"] = "bridge"
        elif not summary.get("structure_type"):
            if re.search(r"\bbridge\b", _normalize(context)):
                summary["structure_type"] = "bridge"
            else:
                summary["structure_type"] = "structural"

        bridge_labels = [
            label
            for label in labels
            if "structure type" in _normalize(label) and "bridge" in _normalize(label)
        ]
        canonical_labels: List[str] = []
        if summary.get("structure_type") == "bridge":
            canonical_labels.append("General Plan > structure type > bridge")
        if summary.get("bridge_superstructure"):
            canonical_labels.append(
                "General Plan > structure type > bridge > Superstructure > "
                f"{summary['bridge_superstructure']}"
            )
        if summary.get("bridge_foundation"):
            canonical_labels.append(
                "General Plan > structure type > bridge > Foundation > "
                f"{summary['bridge_foundation']}"
            )
        if summary.get("bridge_substructure"):
            canonical_labels.append(
                "General Plan > structure type > bridge > Substructure > "
                f"{summary['bridge_substructure']}"
            )

        merged_labels = canonical_labels + bridge_labels + labels
        deduped_labels = []
        seen = set()
        for label in merged_labels:
            key = _normalize(label)
            if key in seen:
                continue
            seen.add(key)
            deduped_labels.append(label)
        summary["project_level_labels"] = " ; ".join(deduped_labels[:6])
        return summary

    def _sheet_category_taxonomy_hints(self, sheet_record: Optional[Dict]) -> List[str]:
        """Map title-block sheet records to paths from search classifications.md."""
        if not sheet_record:
            return []

        hints: List[str] = []
        category = _normalize(str(sheet_record.get("sheet_category") or ""))
        title = _normalize(str(sheet_record.get("sheet_title") or ""))
        primary = _normalize(str(sheet_record.get("primary_type") or ""))
        detail_types = _normalize(str(sheet_record.get("detail_types") or ""))
        excerpt = _normalize(str(sheet_record.get("pdf_text_excerpt") or ""))
        combined = " ".join((category, title, primary, detail_types, excerpt))

        if (
            category in {"general_note", "general_notes"}
            or "general note" in combined
            or "index to plans" in combined
        ):
            hints.append("General Note sheet")
            if "caltrans" in combined:
                hints.append("General Note sheet > Design code > Caltrans")
                if "sdc" in combined or "seismic design criteria" in combined:
                    hints.append(
                        "General Note sheet > Design code > Caltrans > "
                        "Seismic Design Criteria (SDC)"
                    )
            if "aashto" in combined:
                hints.append("General Note sheet > Design code > AASHTO > LRFD BDS")
            if "ars" in combined or "spectral acceleration" in combined:
                hints.append("General Note sheet > Seismic")
            if "standard plan" in combined:
                hints.append("General Note sheet > Index to plans")
            if "index to plans" in combined:
                hints.append("General Note sheet > Index to plans")

        if category == "general_plan" or "general plan" in combined:
            hints.append("General Plan")

        if "title block" in combined or sheet_record.get("sheet_title"):
            hints.append("title block > sheet name")

        deduped = []
        seen = set()
        for hint in hints:
            key = _normalize(hint)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(hint)
        return deduped

    def classify_topology_page(
        self,
        page_data: Dict,
        sheet_record: Optional[Dict] = None,
        page_num: Optional[int] = None,
    ) -> Dict[str, str]:
        """Classify sheet/page level labels from title block + page topology."""
        self.ensure_loaded()
        parts = []
        if sheet_record:
            for key in ("sheet_title", "primary_type", "sheet_category", "detail_types", "pdf_text_excerpt"):
                if sheet_record.get(key):
                    parts.append(str(sheet_record[key]))
        parts.append(self._format_topology_page_context(page_data))
        if page_num is not None:
            parts.append(f"page {page_num}")

        context = " | ".join(p for p in parts if p)
        page_labels = self.classify_page(context)
        taxonomy_hints = self._sheet_category_taxonomy_hints(sheet_record)
        merged_page_labels = taxonomy_hints + page_labels
        detail_level = self.classify_detail(context)

        result = {
            "page_level_labels": " ; ".join(merged_page_labels[:8]),
            "detail_level_labels": " ; ".join(detail_level.labels[:6]),
        }
        if detail_level.referenced_sheet_ids:
            result["referenced_sheet_ids"] = ", ".join(detail_level.referenced_sheet_ids[:12])
        if detail_level.referenced_detail_ids:
            result["referenced_detail_ids"] = ", ".join(detail_level.referenced_detail_ids[:12])
        return result

    @staticmethod
    def _format_topology_page_context(page_data: Dict) -> str:
        parts = []
        for category, entries in (page_data or {}).items():
            if not isinstance(entries, list):
                continue
            for entry in entries[:8]:
                name = str(entry.get("name") or "")
                if name:
                    parts.append(name)
                details = entry.get("details") or {}
                for key in ("detail_name", "description", "type"):
                    if details.get(key):
                        parts.append(str(details[key]))
        return " | ".join(parts)

    def classify_detail(self, text: str) -> ClassifiedDetail:
        self.ensure_loaded()

        detail_labels = self._score_scope("page", text, top_k=6)

        text_norm = _normalize(text)
        referenced_sheet_ids = sorted(set(re.findall(r"\b[A-Z]{1,3}-\d{1,4}[A-Z]?\b", text_norm.upper())))
        referenced_detail_ids = sorted(set(re.findall(r"\b\d+/[A-Z]{1,3}-\d{1,4}[A-Z]?\b", text_norm.upper())))

        if "detail" in text_norm:
            detail_labels.insert(0, "Detail-level > detail")
        if "section" in text_norm:
            detail_labels.insert(0, "Detail-level > section")
        if "elevation" in text_norm:
            detail_labels.insert(0, "Detail-level > elevation")

        # Deduplicate while preserving order.
        seen = set()
        deduped_labels = []
        for label in detail_labels:
            if label not in seen:
                seen.add(label)
                deduped_labels.append(label)

        return ClassifiedDetail(
            labels=deduped_labels[:8],
            referenced_sheet_ids=referenced_sheet_ids[:20],
            referenced_detail_ids=referenced_detail_ids[:20],
        )

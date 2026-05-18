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
from typing import Dict, List, Tuple
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

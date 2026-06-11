"""Build a searchable feedback model from engineer search-result labels."""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from engineering_terminology import EngineeringTerminology
from feedback_cleanup import active_pdf_file_names

DEFAULT_ACRONYM_EXPANSIONS = {
    "LOTB": "log of test boring",
    "P JACK": "post tension jack",
    "PJACK": "post tension jack",
}


class SearchFeedbackLearner:
    """Aggregate JSONL feedback into a model used for ranking and query generalization."""

    def __init__(self, base_dir: Optional[str] = None):
        root = base_dir or os.path.dirname(__file__)
        self.feedback_path = os.path.join(root, "data", "feedback", "search_feedback.jsonl")
        self.model_path = os.path.join(root, "data", "feedback", "search_feedback_model.json")

    @staticmethod
    def _normalize_query(query: str) -> str:
        q = re.sub(r"\s+", " ", str(query or "").strip().lower())
        q = re.sub(r"\bdetails\b", "detail", q)
        return re.sub(r"\s+", " ", q).strip()

    @staticmethod
    def _query_tokens(query: str) -> Set[str]:
        stop_words = {"and", "the", "for", "with", "all", "are", "one", "two", "pin", "no"}
        tokens = set()
        for token in SearchFeedbackLearner._normalize_query(query).split():
            if len(token) >= 3 and token not in stop_words:
                tokens.add(token)
        return tokens

    def _read_latest_labels(self) -> Tuple[Dict[str, Dict[str, List[dict]]], int]:
        """
        Return latest label per (normalized_query, file, page).
        Structure: queries[normalized_query][label] -> [{file, page, note, timestamp}]
        """
        latest: Dict[Tuple[str, str, int], dict] = {}
        raw_count = 0
        active_files = active_pdf_file_names()

        if not os.path.exists(self.feedback_path):
            return {}, 0

        with open(self.feedback_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw_count += 1
                try:
                    record = json.loads(line)
                except Exception:
                    continue

                query_key = self._normalize_query(record.get("query", ""))
                file_name = str(record.get("pdf_file_name", "")).strip()
                label = str(record.get("feedback", "")).strip().lower()
                page_number = record.get("page_number")

                if file_name not in active_files:
                    continue

                if not query_key or not file_name or page_number is None or label not in {
                    "best",
                    "relevant",
                    "irrelevant",
                }:
                    continue

                try:
                    page_number = int(page_number)
                except Exception:
                    continue

                latest[(query_key, file_name, page_number)] = {
                    "file": file_name,
                    "page": page_number,
                    "label": label,
                    "note": record.get("note"),
                    "timestamp": record.get("timestamp"),
                    "project_scope": record.get("project_scope"),
                }

        grouped: Dict[str, Dict[str, List[dict]]] = defaultdict(
            lambda: {"best": [], "relevant": [], "irrelevant": []}
        )
        for (query_key, _file_name, _page), meta in latest.items():
            grouped[query_key][meta["label"]].append(
                {
                    "file": meta["file"],
                    "page": meta["page"],
                    "note": meta.get("note"),
                    "timestamp": meta.get("timestamp"),
                }
            )

        return dict(grouped), raw_count

    def _build_related_queries(
        self, queries: Dict[str, Dict[str, List[dict]]]
    ) -> Dict[str, Dict[str, float]]:
        """Link queries that share labeled pages or have strong lexical overlap."""
        related: Dict[str, Dict[str, float]] = defaultdict(dict)
        query_keys = list(queries.keys())

        best_pages_by_query: Dict[str, Set[Tuple[str, int]]] = {}
        for query_key, labels in queries.items():
            pages = {
                (item["file"], int(item["page"]))
                for item in labels.get("best", []) + labels.get("relevant", [])
            }
            best_pages_by_query[query_key] = pages

        for i, query_a in enumerate(query_keys):
            tokens_a = self._query_tokens(query_a)
            pages_a = best_pages_by_query.get(query_a, set())
            for query_b in query_keys[i + 1 :]:
                if query_a == query_b:
                    continue

                tokens_b = self._query_tokens(query_b)
                overlap = len(tokens_a & tokens_b)
                union = len(tokens_a | tokens_b) or 1
                token_similarity = overlap / union

                pages_b = best_pages_by_query.get(query_b, set())
                shared_pages = pages_a & pages_b
                page_similarity = 0.0
                if pages_a and pages_b and shared_pages:
                    page_similarity = len(shared_pages) / len(pages_a | pages_b)

                contains = query_a in query_b or query_b in query_a
                min_overlap = 1 if min(len(tokens_a), len(tokens_b)) <= 3 else 2
                score = 0.0
                if page_similarity >= 0.5:
                    score = max(score, 0.55 + 0.35 * page_similarity)
                if contains and (overlap >= 1 or shared_pages):
                    score = max(score, 0.75)
                if overlap >= min_overlap:
                    score = max(score, 0.45 + 0.4 * token_similarity)

                if score >= 0.45:
                    related[query_a][query_b] = max(related[query_a].get(query_b, 0.0), score)
                    related[query_b][query_a] = max(related[query_b].get(query_a, 0.0), score)

        return {key: dict(value) for key, value in related.items()}

    def build_model(self) -> dict:
        queries, raw_count = self._read_latest_labels()
        related_queries = self._build_related_queries(queries)

        label_counts = Counter()
        for labels in queries.values():
            for label_name, pages in labels.items():
                label_counts[label_name] += len(pages)

        acronym_queries = []
        for query_key in queries:
            compact = query_key.replace(" ", "")
            if len(compact) <= 6 and (compact.isupper() or len(query_key.split()) <= 2):
                acronym_queries.append(query_key)

        return {
            "version": 1,
            "built_at": datetime.utcnow().isoformat(),
            "source_records": raw_count,
            "query_count": len(queries),
            "label_counts": dict(label_counts),
            "queries": queries,
            "related_queries": related_queries,
            "acronym_queries": sorted(set(acronym_queries)),
        }

    def save_model(self, model: dict) -> str:
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "w", encoding="utf-8") as handle:
            json.dump(model, handle, indent=2)
        return self.model_path

    def sync_terminology(self, model: dict) -> List[str]:
        """Persist acronym expansions implied by short feedback queries."""
        synced = []
        for query_key in model.get("acronym_queries", []):
            expansion = DEFAULT_ACRONYM_EXPANSIONS.get(query_key.upper().replace(" ", ""))
            if not expansion:
                expansion = DEFAULT_ACRONYM_EXPANSIONS.get(query_key.upper())
            if not expansion:
                continue

            acronym = query_key.strip().upper()
            EngineeringTerminology.add_feedback_mapping(acronym=acronym, expansion=expansion)
            synced.append(f"{acronym} -> {expansion}")

        return synced

    def run(self, sync_terms: bool = True) -> dict:
        model = self.build_model()
        output_path = self.save_model(model)
        synced = self.sync_terminology(model) if sync_terms else []

        return {
            "model_path": output_path,
            "query_count": model.get("query_count", 0),
            "source_records": model.get("source_records", 0),
            "label_counts": model.get("label_counts", {}),
            "related_query_links": sum(len(v) for v in model.get("related_queries", {}).values()),
            "terminology_synced": synced,
        }


def learn_from_feedback(sync_terms: bool = True) -> dict:
    return SearchFeedbackLearner().run(sync_terms=sync_terms)

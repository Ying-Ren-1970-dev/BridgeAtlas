"""Search agent module for intelligent querying of the knowledge base."""
from typing import List, Dict, Optional, Set
from collections import defaultdict
from collections import Counter
import math
import re
import json
import os
import logging
from openai import OpenAI
from langchain_core.documents import Document

import config
from vector_store import VectorStore
from metadata_manager import MetadataManager
from engineering_terminology import expand_query
from cad_drawing_knowledge import CadDrawingKnowledge
from project_layout import collect_layout_signals
from project_profile_builder import ProjectProfileBuilder

logger = logging.getLogger(__name__)


class SearchAgent:
    """Intelligent search agent using RAG and OpenAI."""
    
    def __init__(self):
        """Initialize the search agent."""
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_store = VectorStore()
        self.metadata_manager = MetadataManager()
        
        # Initialize vector store
        self.vector_store.initialize_vectorstore()
    
    def search(
        self,
        query: str,
        k: int = 50,
        category_filter: Optional[str] = None,
        phase_filter: Optional[str] = None,
        use_query_expansion: bool = True,
        generate_summary: Optional[bool] = None,
        use_hybrid_search: Optional[bool] = None,
        project_scope: Optional[str] = None,
    ) -> Dict:
        """
        Perform semantic search across the knowledge base.
        
        Args:
            query: Search query or keywords
            k: Number of results to retrieve (default: 50)
            category_filter: Optional category filter
            phase_filter: Optional phase filter
            use_query_expansion: Whether to expand query with engineering synonyms (default: True)
            generate_summary: Override for summary generation (default from config)
            use_hybrid_search: Override for hybrid retrieval (default from config)
            project_scope: Optional project/file substring scope (e.g., "goldenwest")
            
        Returns:
            Dictionary containing search results and metadata
        """
        print(f"\nSearching for: '{query}'")

        normalized_query = self._normalize_query_text(query)
        project_descriptor = self._parse_project_descriptor_query(query)

        # Identifier-like queries (project numbers, sheet IDs) are brittle; avoid broad synonym drift.
        looks_like_identifier_query = bool(re.search(r"\d", normalized_query) and ("-" in normalized_query or "/" in normalized_query))
        token_count = len(self._tokenize_for_keyword_search(normalized_query))
        has_specific_phrase = any(
            phrase in normalized_query
            for phrase in [
                "box girder",
                "pipe pin",
                "shear key",
                "truss bridge",
                "steel truss",
                "abutment footing",
                "abutment pile detail",
                "abutment pile",
                "pile detail",
                "abutment detail",
                "p jack",
                "plastic hinge",
                "general plan",
                "rebar detail",
                "caltrans standard plan",
                "standard plan reference",
                "general note",
                "general notes",
            ]
        )
        feedback_model = self._load_feedback_model()
        learned_acronym_queries = {
            self._normalize_query_text(item)
            for item in feedback_model.get("acronym_queries", [])
        }
        avoid_expansion = (
            looks_like_identifier_query
            or (has_specific_phrase and token_count <= 6)
            or (token_count == 1)
            or normalized_query in learned_acronym_queries
            or project_descriptor is not None
        )
        
        # Expand query with engineering terminology synonyms
        expanded_query = normalized_query
        if self._is_detail_intent_query(normalized_query):
            detail_section_query = CadDrawingKnowledge.expand_detail_section_query(
                normalized_query
            )
            if detail_section_query != normalized_query:
                expanded_query = detail_section_query
                print(f"Detail/section expanded query: '{expanded_query}'")
        if use_query_expansion and not avoid_expansion:
            expanded_terms = expand_query(expanded_query, max_expansions=3)
            if len(expanded_terms) > 1:
                # Combine terms for semantic search (embedding will handle similarity)
                expanded_query = " ".join(expanded_terms)
                print(f"Expanded query: '{expanded_query}'")
        
        # Build filter dictionary
        filter_dict = {}
        if category_filter:
            # Note: We'd need to enhance vector_store metadata to include categories
            pass
        if phase_filter:
            filter_dict['phase'] = phase_filter

        scoped_file_names = self._resolve_project_scope_files(project_scope)
        if project_scope and not scoped_file_names:
            return {
                'query': query,
                'summary': "No matching projects found for the provided project scope.",
                'results': [],
                'total_projects': 0,
            }

        if scoped_file_names:
            # Chroma where filter supports $in for exact metadata matches.
            filter_dict['file_name'] = {'$in': sorted(scoped_file_names)}
        
        active_filter = filter_dict if filter_dict else None
        use_hybrid = config.ENABLE_HYBRID_SEARCH if use_hybrid_search is None else use_hybrid_search

        query_tokens = self._tokenize_for_keyword_search(normalized_query)
        single_token_query = len(query_tokens) == 1 and len(query_tokens[0]) >= 4

        # Always get vector results; optionally fuse with keyword BM25 ranking.
        vector_results = self.vector_store.similarity_search_with_scores(
            query=expanded_query,
            k=k,
            filter_dict=active_filter,
        )

        keyword_candidates = self.vector_store.get_keyword_search_candidates(
            filter_dict=active_filter,
            limit=config.HYBRID_KEYWORD_CANDIDATE_LIMIT,
        )

        if single_token_query:
            # Single-term intent should be literal by default to avoid semantic drift.
            keyword_results = self._keyword_search_bm25(normalized_query, keyword_candidates, top_k=k)
            search_results = keyword_results
        elif use_hybrid:
            keyword_results = self._keyword_search_bm25(normalized_query, keyword_candidates, top_k=k)
            search_results = self._hybrid_fuse_results(vector_results, keyword_results, k=k)
        else:
            search_results = vector_results

        # Single-token queries are often broad; require literal token presence for precision.
        if single_token_query:
            strict_token = query_tokens[0]
            search_results = [
                (doc, score)
                for doc, score in search_results
                if self._doc_contains_query_token(doc, strict_token)
            ]

        # Precision guardrails: apply intent-aware lexical checks with synonym variants
        # so behavior applies across related queries, not just one exact phrase.
        query_token_set = set(query_tokens)
        intent_rules = [
            {
                "name": "truss_bridge",
                "trigger_groups": [["truss"], ["bridge", "bridges"]],
                "required_groups": [["truss"], ["bridge", "bridges"]],
            },
            {
                "name": "shear_key",
                "trigger_groups": [["shear key", "steel shear key", "pipe pin"]],
                "required_groups": [["shear key", "steel shear key", "pipe pin"]],
            },
            {
                "name": "box_girder",
                "trigger_groups": [["box girder", "r/c box girder", "concrete box girder"]],
                "required_groups": [["box girder", "r/c box girder", "concrete box girder"]],
            },
            {
                "name": "cidh_foundation",
                "trigger_groups": [
                    ["cidh", "cast drilled hole", "drilled shaft"],
                    ["pile", "piles", "shaft", "shafts"],
                ],
                "required_groups": [
                    ["cidh", "cast drilled hole", "drilled shaft"],
                    ["pile", "piles", "shaft", "shafts"],
                ],
            },
            {
                "name": "general_plan",
                "trigger_groups": [["general plan", "general plans"]],
                "required_groups": [["general plan", "general plans"]],
            },
        ]

        search_results = self._apply_title_block_boost(search_results, normalized_query)
        search_results = self._apply_detail_intent_boost(search_results, normalized_query)
        search_results = self._apply_general_notes_reference_boost(
            search_results, normalized_query
        )
        search_results = self._apply_topology_layer_boost(search_results, normalized_query)
        search_results = self._ensure_per_project_general_notes_coverage(
            search_results,
            normalized_query,
            keyword_candidates,
            scoped_file_names,
        )
        search_results = self._ensure_per_project_detail_intent_coverage(
            search_results,
            normalized_query,
            keyword_candidates,
            scoped_file_names,
        )
        if self._is_detail_intent_query(normalized_query):
            page_strength = self._page_detail_intent_strength(search_results, normalized_query)
            search_results = [
                (doc, score)
                for doc, score in search_results
                if page_strength.get(
                    (
                        str(doc.metadata.get("file_name") or ""),
                        int(doc.metadata.get("page") or 0),
                    ),
                    0,
                )
                > -6
            ]

        for rule in intent_rules:
            is_triggered = all(
                any(self._query_matches_variant(query_token_set, variant) for variant in group)
                for group in rule["trigger_groups"]
            )
            if not is_triggered:
                continue

            search_results = [
                (doc, score)
                for doc, score in search_results
                if all(
                    any(self._doc_matches_variant(doc, variant) for variant in group)
                    for group in rule["required_groups"]
                )
            ]

        # Apply engineer-in-the-loop feedback boosts/penalties for this exact query/scope.
        feedback_adjustments = self._load_feedback_adjustments(query, project_scope)
        if feedback_adjustments:
            search_results = self._filter_strong_irrelevant_feedback(
                search_results, feedback_adjustments
            )
            search_results = self._apply_feedback_adjustments(search_results, feedback_adjustments)

        positive_feedback_pages = self._load_positive_feedback_pages(query, project_scope)
        gp_only_descriptor = (
            project_descriptor is not None
            and not project_descriptor.get("sheet_intents")
        )
        if positive_feedback_pages and not gp_only_descriptor:
            search_results = self._inject_feedback_labeled_pages(
                search_results,
                positive_feedback_pages,
                keyword_candidates,
                scoped_file_names,
            )

        search_results = self._apply_exact_feedback_filters(search_results, query)
        search_results = self._prioritize_exact_feedback_best_pages(search_results, query)
        search_results = self._filter_weak_unlabeled_detail_results(search_results, query)
        search_results = self._ensure_missing_feedback_best_pages(
            search_results,
            query,
            keyword_candidates,
            scoped_file_names,
        )

        sheet_level_query = self._parse_sheet_level_query(query)

        if project_descriptor:
            search_results = self._apply_project_descriptor_ranking(
                search_results,
                project_descriptor,
                keyword_candidates,
                scoped_file_names,
                query=query,
            )
        elif sheet_level_query:
            search_results = self._apply_sheet_intent_ranking(
                search_results,
                sheet_level_query,
                keyword_candidates,
                scoped_file_names,
                query=query,
            )

        search_results = self._filter_project_overview_pages(search_results, query)
        
        # Organize results by project
        projects_data = self._organize_results_by_project(search_results)
        
        # Enrich with metadata
        enriched_results = self._enrich_with_metadata(projects_data)
        
        should_generate_summary = config.GENERATE_SEARCH_SUMMARY if generate_summary is None else generate_summary
        if (
            not enriched_results
            and project_descriptor
            and project_descriptor.get("sheet_intents")
        ):
            span_count = project_descriptor.get("span_count")
            sheet_label = ", ".join(
                intent.replace("_", " ") for intent in project_descriptor.get("sheet_intents", [])
            )
            summary = (
                f"No {span_count}-span bridge projects with matching {sheet_label} sheets "
                "were found in the library."
            )
        elif should_generate_summary:
            summary = self._generate_search_summary(query, enriched_results)
        else:
            summary = f"Found {len(enriched_results)} relevant project(s)."
        
        return {
            'query': query,
            'summary': summary,
            'results': enriched_results,
            'total_projects': len(enriched_results),
        }

    def _resolve_project_scope_files(self, project_scope: Optional[str]) -> Set[str]:
        """Resolve a user-provided project scope string to matching file names."""
        if not project_scope:
            return set()

        needle = project_scope.lower().strip()
        if not needle:
            return set()

        matches: Set[str] = set()
        for file_name, project in self.metadata_manager.get_all_projects().items():
            haystack = " ".join(
                [
                    str(file_name),
                    str(project.get('project_name', '')),
                    str(project.get('file_name', '')),
                ]
            ).lower()
            if needle in haystack:
                matches.add(file_name)

        return matches

    def _doc_key(self, metadata: Dict) -> str:
        """Create a stable key for deduplicating chunk results across rankers."""
        return "|".join(
            [
                str(metadata.get("file_name", "")),
                str(metadata.get("page", "")),
                str(metadata.get("chunk_id", "")),
            ]
        )

    def _tokenize_for_keyword_search(self, text: str) -> List[str]:
        """Tokenize while preserving engineering identifiers like S-502 or 4/S-502."""
        if not text:
            return []
        return re.findall(r"[A-Za-z0-9_./-]+", text.lower())

    def _normalize_query_text(self, query: str) -> str:
        """Normalize common engineering wording/abbreviations before retrieval."""
        q = (query or "").lower()
        replacements = [
            (r"\blotb\b", " log of test boring lotb "),
            (r"\breinforc(e|ement|ing)\b", " rebar "),
            (r"\bconcrete\b", " r/c "),
            (r"\brc\b", " r/c "),
            (r"\babut\.\b", " abutment "),
        ]
        for pattern, replacement in replacements:
            q = re.sub(pattern, replacement, q)
        return re.sub(r"\s+", " ", q).strip()

    def _canonical_feedback_query(self, query: str) -> str:
        """Canonical form for feedback lookup (detail/details, normalized casing)."""
        q = self._normalize_query_text(query)
        q = re.sub(r"\bdetails\b", "detail", q)
        return re.sub(r"\s+", " ", q).strip()

    def _extract_detail_titles(self, text: str) -> List[str]:
        """Extract likely detail/sheet titles from vision analysis text blocks."""
        if not text:
            return []

        titles: List[str] = []
        patterns = [
            r"\*\*Drawing Type\*\*:\s*([^\n]+)",
            r"\bDrawing Type\s*:\s*([^\n]+)",
            r"\bSheet Title\s*:\s*([^\n|]+)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                value = match.group(1).strip(" -*\t")
                if 2 < len(value) <= 120:
                    titles.append(value)

        seen = set()
        deduped = []
        for t in titles:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        return deduped[:8]

    def _doc_contains_query_token(self, doc: Document, token: str) -> bool:
        """Check whether token appears in chunk text or key metadata fields."""
        token = (token or "").strip().lower()
        if not token:
            return False

        pattern = rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])"

        content = str(getattr(doc, "page_content", "") or "").lower()
        if re.search(pattern, content):
            return True

        metadata_blob = self._metadata_search_blob(getattr(doc, "metadata", {}) or {})
        return bool(re.search(pattern, metadata_blob))

    def _query_matches_variant(self, query_tokens: Set[str], variant: str) -> bool:
        """Return True if all tokens from variant text are present in query token set."""
        variant_tokens = self._tokenize_for_keyword_search(variant)
        if not variant_tokens:
            return False
        return all(token in query_tokens for token in variant_tokens)

    def _doc_matches_variant(self, doc: Document, variant: str) -> bool:
        """Return True if all tokens from variant text are present in doc text/metadata."""
        variant_tokens = self._tokenize_for_keyword_search(variant)
        if not variant_tokens:
            return False
        return all(self._doc_contains_query_token(doc, token) for token in variant_tokens)

    def _metadata_search_blob(self, metadata: Dict) -> str:
        """Flatten searchable metadata fields for title-block and intent matching."""
        return " ".join(
            str(metadata.get(field, ""))
            for field in [
                "file_name",
                "project_name",
                "enriched_project_name",
                "plan_primary_type",
                "plan_sheet_title",
                "plan_sheet_number",
                "plan_sheet_type",
                "cad_drawing_view",
                "cad_drawing_label",
                "cad_drawing_category",
                "cad_drawing_precise",
                "cad_search_view_types",
                "cad_leader_attribution",
                "cad_sheet_views",
                "cad_sheet_group",
                "cad_sheet_group_role",
                "cad_sheet_group_index",
                "cad_section_cuts",
                "cad_detail_callouts",
                "cad_view_refs",
                "cad_sheet_refs",
                "cad_note_refs",
                "cad_cross_reference_summary",
                "detail_intents",
                "detail_types",
                "structural_elements",
                "topology_structure_type",
                "topology_bridge_type",
                "topology_superstructure_type",
                "topology_foundation_type",
                "topology_sheet_category",
                "topology_sheet_title",
                "topology_cross_references",
                "topology_elements",
                "search_classification_project_labels",
                "search_classification_page_labels",
                "search_classification_detail_labels",
                "search_classification_referenced_sheets",
            ]
        ).lower()

    PILE_DETAIL_POSITIVE_PHRASES: List[str] = [
        "pile (type a) detail",
        "steel pipe pile detail",
        "steel pipe pile (type a) detail",
        '24" dia steel pipe pile',
        '24"o steel pipe pile',
        "driven steel pipe pile",
    ]

    PILE_DETAIL_NEGATIVE_PHRASES: List[str] = [
        "footing plan",
        "section k-k",
        "section l-l",
    ]

    ABUTMENT_DETAIL_SHEET_PHRASES: List[str] = [
        "abutment details no",
        "abutment detail no",
        "abutment details no.",
    ]

    SHEAR_KEY_POSITIVE_PHRASES: List[str] = [
        "shear key detail",
        "steel shear key",
        "pipe pin",
        "shear key (",
    ]

    SHEAR_KEY_NEGATIVE_PHRASES: List[str] = [
        "shear key reinforcement not shown",
        "shear key not shown",
        "shear key - see",
        "shear key, see",
    ]

    def _strip_search_blocks_for_intent(self, content: str) -> str:
        """Prefer visible sheet text over merged topology blocks for intent matching."""
        text = content or ""
        lower = text.lower()
        cut_at = len(text)
        for marker in (
            "[topology project]",
            "[topology sheet]",
            "[topology detail]",
            "[topology cross references]",
            "[search classification]",
            "[deep vision topology]",
        ):
            idx = lower.find(marker)
            if idx != -1:
                cut_at = min(cut_at, idx)
        return text[:cut_at]

    def _visible_sheet_text(self, doc: Document) -> str:
        return self._strip_search_blocks_for_intent(
            str(getattr(doc, "page_content", "") or "")
        ).lower()

    def _is_detail_intent_query(self, query: str) -> bool:
        query_tokens = self._query_token_set(query)
        if not query_tokens:
            return False
        has_detail = bool(query_tokens & {"detail", "details"})
        if has_detail and {"shear", "key"} <= query_tokens:
            return True
        structural_tokens = {
            "abutment",
            "pile",
            "piles",
            "footing",
            "girder",
            "cidh",
            "bent",
            "pier",
            "cap",
            "rebar",
            "reinforcement",
            "shaft",
        }
        return has_detail and bool(query_tokens & structural_tokens)

    def _detail_intent_match_strength(self, doc: Document, query: str) -> int:
        """Score how strongly a chunk matches a detail-sheet lookup query."""
        query_lower = (query or "").lower()
        query_tokens = self._query_token_set(query)
        visible = self._visible_sheet_text(doc)
        metadata = getattr(doc, "metadata", {}) or {}
        sheet_title = str(
            metadata.get("topology_sheet_title")
            or metadata.get("plan_sheet_title")
            or ""
        ).lower()
        plan_type = str(
            metadata.get("plan_primary_type")
            or metadata.get("topology_sheet_category")
            or ""
        ).lower()

        strength = 0
        has_explicit_pile_detail = any(
            phrase in visible for phrase in self.PILE_DETAIL_POSITIVE_PHRASES
        )

        if {"shear", "key"} <= query_tokens or "shear key" in query_lower:
            has_dedicated_shear_key_sheet = any(
                phrase in visible for phrase in self.SHEAR_KEY_POSITIVE_PHRASES
            )
            if has_dedicated_shear_key_sheet:
                strength += 14
            if "shear key" in sheet_title:
                strength += 10
            elif "shear key" in visible and "detail" in visible and not has_dedicated_shear_key_sheet:
                strength += 2
            for phrase in self.SHEAR_KEY_NEGATIVE_PHRASES:
                if phrase in visible:
                    strength -= 12
            if (
                "abutment" in sheet_title
                and "shear key" not in sheet_title
                and not has_dedicated_shear_key_sheet
            ):
                strength -= 8
            if "general plan" in sheet_title or "index to plans" in sheet_title:
                strength -= 10

        if {"abutment", "pile"} & query_tokens or "abutment pile" in query_lower:
            if has_explicit_pile_detail:
                strength += 12

            for phrase in self.PILE_DETAIL_NEGATIVE_PHRASES:
                if phrase in visible:
                    strength -= 8

            if "abutment" in query_tokens:
                if re.search(r"abutment details no\.?\s*\d", visible):
                    strength += 7
                for phrase in self.ABUTMENT_DETAIL_SHEET_PHRASES:
                    if phrase in visible:
                        strength += 3

        if {"pile", "detail"} <= query_tokens or {"pile", "details"} <= query_tokens:
            if "footing plan" in visible and not has_explicit_pile_detail:
                strength -= 12
            if "abutment detail" in sheet_title and "footing plan" in visible:
                strength -= 10
            if ("abutment detail" in sheet_title or "abutment details" in sheet_title):
                if not has_explicit_pile_detail:
                    strength -= 10

        component_tokens = {
            token
            for token in query_tokens
            if token not in CadDrawingKnowledge._QUERY_STOPWORDS
            and len(token) > 2
        }
        cad_view = str(metadata.get("cad_drawing_view") or "").lower()
        cad_search_views = {
            part.strip().lower()
            for part in str(metadata.get("cad_search_view_types") or "").split(",")
            if part.strip()
        }

        if "detail" in query_tokens and component_tokens:
            component_hits = sum(1 for token in component_tokens if token in visible)
            if cad_view == "section" or "section" in cad_search_views:
                strength += 6
                strength += min(component_hits * 2, 8)
            elif cad_view == "detail" or "detail" in cad_search_views:
                strength += min(component_hits, 4)

        if "pile" in query_tokens:
            if "steel pipe pile" in visible:
                strength += 2
            if "footing plan" in visible and not has_explicit_pile_detail:
                strength -= 8

        key_tokens = [
            token
            for token in query_tokens
            if token in {"abutment", "pile", "piles", "detail", "details", "footing", "steel", "pipe"}
        ]
        strength += sum(1 for token in key_tokens if token in visible)

        return strength

    def _page_detail_intent_strength(self, search_results: List[tuple], query: str) -> Dict[tuple, int]:
        """Aggregate chunk signals to a page-level detail intent score."""
        page_positive: Dict[tuple, int] = {}
        page_negative: Dict[tuple, int] = {}

        for doc, _ in search_results:
            metadata = getattr(doc, "metadata", {}) or {}
            file_name = metadata.get("file_name")
            page_num = metadata.get("page")
            if file_name is None or page_num is None:
                continue

            key = (str(file_name), int(page_num))
            strength = self._detail_intent_match_strength(doc, query)
            if strength >= 0:
                page_positive[key] = max(page_positive.get(key, 0), strength)
            else:
                page_negative[key] = min(page_negative.get(key, 0), strength)

        combined: Dict[tuple, int] = {}
        keys = set(page_positive) | set(page_negative)
        for key in keys:
            combined[key] = page_positive.get(key, 0) + page_negative.get(key, 0)
        return combined

    def _apply_detail_intent_boost(self, search_results: List[tuple], query: str) -> List[tuple]:
        """Boost dedicated detail sheets and demote section/footing pages for detail queries."""
        if not self._is_detail_intent_query(query):
            return search_results

        page_strength = self._page_detail_intent_strength(search_results, query)

        boosted = []
        for doc, score in search_results:
            adjusted = score
            metadata = getattr(doc, "metadata", {}) or {}
            key = (str(metadata.get("file_name") or ""), int(metadata.get("page") or 0))
            strength = page_strength.get(key, self._detail_intent_match_strength(doc, query))

            if strength >= 14:
                adjusted = min(adjusted, max(0.0, score - 0.95))
            elif strength >= 9:
                adjusted = min(adjusted, max(0.0, score - 0.88))
            elif strength >= 5:
                adjusted = min(adjusted, max(0.0, score - 0.78))
            elif strength >= 2:
                adjusted = min(adjusted, max(0.0, score - 0.55))
            elif strength <= -6:
                adjusted = max(adjusted, score + 0.65)

            boosted.append((doc, adjusted))

        boosted.sort(key=lambda item: item[1])
        return boosted

    def _ensure_per_project_detail_intent_coverage(
        self,
        search_results: List[tuple],
        query: str,
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
    ) -> List[tuple]:
        """Backfill the strongest detail-sheet hit for each project on detail-intent queries."""
        if not self._is_detail_intent_query(query):
            return search_results

        target_projects = scoped_file_names or set(self.metadata_manager.get_all_projects().keys())
        if not target_projects:
            return search_results

        present_pages = {
            (
                str(doc.metadata.get("file_name") or ""),
                int(doc.metadata.get("page") or 0),
            )
            for doc, _ in search_results
            if doc.metadata.get("file_name") and doc.metadata.get("page") is not None
        }
        anchor_score = min(score for _, score in search_results) if search_results else 0.0

        page_positive: Dict[tuple, int] = {}
        page_negative: Dict[tuple, int] = {}
        page_docs: Dict[tuple, Document] = {}
        for item in keyword_candidates:
            metadata = item.get("metadata", {}) or {}
            file_name = str(metadata.get("file_name") or "")
            page_num = metadata.get("page")
            if file_name not in target_projects or page_num is None:
                continue

            doc = Document(page_content=item.get("content", ""), metadata=metadata)
            page_key = (file_name, int(page_num))
            page_docs[page_key] = doc
            strength = self._detail_intent_match_strength(doc, query)
            if strength >= 0:
                page_positive[page_key] = max(page_positive.get(page_key, 0), strength)
            else:
                page_negative[page_key] = min(page_negative.get(page_key, 0), strength)

        exact_labels = self._load_exact_feedback_labels(query)
        has_exact_best_labels = any(
            label == "best" for label, _weight in exact_labels.values()
        )
        min_backfill_strength = 10 if has_exact_best_labels else 5

        best_by_page: Dict[tuple, tuple] = {}
        for page_key, doc in page_docs.items():
            strength = page_positive.get(page_key, 0) + page_negative.get(page_key, 0)
            if strength < min_backfill_strength:
                continue
            best_by_page[page_key] = (strength, doc)

        augmented = list(search_results)
        for (file_name, page_num), (strength, doc) in best_by_page.items():
            if (file_name, page_num) in present_pages:
                continue
            score = anchor_score
            if strength >= 10:
                score = 0.0
            elif strength >= 7:
                score = min(score, 0.02)
            elif strength >= 4:
                score = min(score, 0.08)
            augmented.append((doc, score))

        augmented.sort(key=lambda item: item[1])
        return augmented

    def _doc_matches_title_block_phrase(self, doc: Document, phrase: str) -> bool:
        """Return True when a title-block phrase appears in metadata or enriched text."""
        phrase = (phrase or "").strip().lower()
        if not phrase:
            return False

        metadata_blob = self._metadata_search_blob(getattr(doc, "metadata", {}) or {})
        if phrase in metadata_blob:
            return True

        content = str(getattr(doc, "page_content", "") or "").lower()
        if phrase in content:
            return True
        if "[enriched metadata]" in content:
            enriched = content.split("[enriched metadata]", 1)[-1]
            if phrase in enriched:
                return True
        if "[sheet category]" in content:
            category = content.split("[sheet category]", 1)[-1]
            if phrase in category:
                return True
        if "[general notes text]" in content:
            notes = content.split("[general notes text]", 1)[-1]
            if phrase in notes:
                return True
        for marker in (
            "[topology project]",
            "[topology sheet]",
            "[topology detail]",
            "[topology cross references]",
            "[search classification]",
            "[deep vision topology]",
        ):
            if marker in content:
                section = content.split(marker, 1)[-1]
                if phrase in section:
                    return True
        return False

    def _doc_search_blob(self, doc: Document) -> str:
        """Combine chunk text and metadata for phrase matching."""
        content = str(getattr(doc, "page_content", "") or "").lower()
        metadata_blob = self._metadata_search_blob(getattr(doc, "metadata", {}) or {})
        return f"{content} {metadata_blob}".strip()

    GENERAL_NOTES_QUERY_GROUPS: List[List[str]] = [
        ["caltrans"],
        ["standard", "plan"],
        ["general", "note"],
        ["general", "notes"],
        ["seismic", "design"],
        ["plan", "symbol"],
        ["abbreviation"],
        ["ars", "curve"],
        ["aashto"],
        ["lrfd"],
    ]

    GENERAL_NOTES_PHRASES: List[str] = [
        "ars curve",
        "design ars curve",
        "curve design ars",
        "damping design ars curve",
        "caltrans seismic design criteria",
        "standard plan",
        "standard plans",
        "standard plan sheet",
        "plan symbols",
        "general notes",
        "general note",
        "index to plans",
    ]

    def _query_token_set(self, query: str) -> Set[str]:
        return set(self._tokenize_for_keyword_search(self._normalize_query_text(query)))

    def _is_general_notes_reference_query(self, query: str) -> bool:
        """Detect queries that target content typically found on general-notes sheets."""
        query_tokens = self._query_token_set(query)
        if not query_tokens:
            return False

        for group in self.GENERAL_NOTES_QUERY_GROUPS:
            group_tokens = set()
            for term in group:
                group_tokens.update(self._tokenize_for_keyword_search(term))
            if group_tokens and group_tokens.issubset(query_tokens):
                return True
        return False

    def _is_general_notes_page(self, doc: Document) -> bool:
        metadata = getattr(doc, "metadata", {}) or {}
        if str(metadata.get("plan_sheet_type") or "") == "general_note":
            return True

        blob = self._doc_search_blob(doc)
        if "[general notes text]" in blob:
            return True
        return any(
            marker in blob
            for marker in (
                "sheet_type:general_note",
                "general notes",
                "index to plans",
            )
        )

    def _general_notes_match_strength(self, doc: Document, query: str) -> int:
        """Score how strongly a chunk matches a general-notes reference query."""
        blob = self._doc_search_blob(doc)
        if not blob:
            return 0

        strength = 0
        query_lower = (query or "").lower()
        for phrase in self.GENERAL_NOTES_PHRASES:
            if phrase in query_lower and phrase in blob:
                strength += 3

        if "ars" in query_lower and "curve" in query_lower:
            if re.search(r"ars.{0,30}curve|curve.{0,30}ars", blob):
                strength += 3

        query_tokens = [
            token
            for token in self._tokenize_for_keyword_search(query)
            if len(token) >= 4 or token in {"ars", "sdc"}
        ]
        token_hits = sum(1 for token in query_tokens if self._doc_contains_query_token(doc, token))
        strength += token_hits

        if self._is_general_notes_page(doc) and token_hits >= 2:
            strength += 2
        return strength

    def _is_project_overview_page(self, doc: Document) -> bool:
        """Detect synthesized project-profile chunks stored at page 0."""
        metadata = getattr(doc, "metadata", {}) or {}
        if str(metadata.get("index_level") or "") == "project_overview":
            return True
        if str(metadata.get("plan_sheet_type") or "") == "project_overview":
            return True
        page = metadata.get("page")
        if page is not None:
            try:
                if int(page) == 0:
                    return True
            except Exception:
                pass
        return "[project profile]" in self._doc_search_blob(doc)

    def _is_project_overview_query(self, query: str) -> bool:
        """Queries that should intentionally surface synthesized project profiles."""
        q = re.sub(r"\s+", " ", str(query or "").strip().lower())
        markers = (
            "project profile",
            "project overview",
            "design intent",
            "bridge layout",
            "describe ",
            "what type of bridge",
            "span count",
            "how many span",
        )
        return any(marker in q for marker in markers)

    def _filter_project_overview_pages(
        self, search_results: List[tuple], query: str
    ) -> List[tuple]:
        """Drop page-0 project profile chunks unless the query asks for project context."""
        if self._is_project_overview_query(query):
            return search_results
        return [
            (doc, score)
            for doc, score in search_results
            if not self._is_project_overview_page(doc)
        ]

    def _parse_sheet_intents(self, query: str) -> List[str]:
        """Detect sheet-level lookup intents layered on top of project filters."""
        q = re.sub(r"\s+", " ", str(query or "").strip().lower())
        intents: List[str] = []
        if re.search(r"\btypical\s+sections?\b", q):
            intents.append("typical_section")
        if re.search(r"\bgeneral\s+plans?\b", q):
            intents.append("general_plan")
        if re.search(r"\b(reinforcement|rebar)\s+details?\b", q) or (
            "reinforcement" in q and "detail" in q
        ):
            intents.append("reinforcement_detail")
        if re.search(r"\bgirder\s+details?\b", q):
            intents.append("girder_detail")
        if re.search(r"\babutment\s+details?\b", q):
            intents.append("abutment_detail")
        return intents

    def _parse_sheet_level_query(self, query: str) -> Optional[Dict]:
        """Parse sheet-level lookups without an explicit span filter."""
        sheet_intents = self._parse_sheet_intents(query)
        if not sheet_intents:
            return None

        q = re.sub(r"\s+", " ", str(query or "").strip().lower())
        return {
            "sheet_intents": sheet_intents,
            "wants_concrete": any(
                token in q for token in ("concrete", "cip", "box girder", "r/c", "rc")
            ),
            "wants_bridge": "bridge" in q,
        }

    def _apply_sheet_intent_ranking(
        self,
        search_results: List[tuple],
        descriptor: Dict,
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
        query: Optional[str] = None,
    ) -> List[tuple]:
        """Rank sheet-level lookups within matching bridge projects."""
        target_files = scoped_file_names or set(self.metadata_manager.get_all_projects().keys())
        if not target_files:
            return search_results

        profiles = self._build_project_descriptor_profiles(keyword_candidates, target_files)
        if descriptor.get("wants_concrete"):
            matching_files = {
                file_name for file_name, profile in profiles.items() if profile.get("concrete")
            }
        elif descriptor.get("wants_bridge"):
            matching_files = {
                file_name for file_name, profile in profiles.items() if profile.get("bridge")
            }
        else:
            matching_files = set(target_files)

        if not matching_files:
            matching_files = set(target_files)

        return self._apply_layered_project_sheet_ranking(
            search_results,
            descriptor,
            keyword_candidates,
            scoped_file_names,
            matching_files,
            profiles,
            query=query,
        )

    def _parse_project_descriptor_query(self, query: str) -> Optional[Dict]:
        """
        Detect layered bridge lookups like '3 span concrete bridge' or
        'CIP box girder typical section for 4 span bridge'.
        Parsed from the raw query so normalization does not strip key terms.
        """
        q = re.sub(r"\s+", " ", str(query or "").strip().lower())
        q = q.replace("cocnrete", "concrete")
        match = re.search(r"\b(\d+)\s*[-]?\s*span", q)
        if not match:
            return None

        span_count = int(match.group(1))
        wants_concrete = any(
            token in q for token in ("concrete", "cip", "box girder", "r/c", "rc")
        )
        wants_bridge = "bridge" in q
        sheet_intents = self._parse_sheet_intents(q)
        if not wants_bridge and not wants_concrete and not sheet_intents:
            return None

        return {
            "span_count": span_count,
            "wants_concrete": wants_concrete,
            "wants_bridge": wants_bridge,
            "sheet_intents": sheet_intents,
        }

    def _is_general_plan_sheet(self, doc: Document) -> bool:
        metadata = getattr(doc, "metadata", {}) or {}
        if str(metadata.get("plan_sheet_type") or "") == "general_plan":
            return True
        blob = self._doc_search_blob(doc)
        return any(
            marker in blob
            for marker in (
                "general plan no",
                "sheet category: general_plan",
                "plan type: general plan",
            )
        )

    def _saved_profile_to_descriptor(self, saved: Dict) -> Dict:
        structured = saved.get("structured") or {}
        source_pages = saved.get("source_pages") or []
        gp_page = source_pages[0] if source_pages else None
        gp_doc = None
        if gp_page is not None:
            chunk = self.vector_store.get_page_chunk(saved.get("file_name", ""), int(gp_page))
            if chunk:
                gp_doc = Document(page_content=chunk.get("content", ""), metadata=chunk.get("metadata", {}))

        return {
            "span_count": structured.get("span_count"),
            "concrete": structured.get("concrete_box_girder"),
            "bridge": structured.get("is_bridge"),
            "gp_page": gp_page,
            "gp_doc": gp_doc,
            "narrative": saved.get("narrative", ""),
            "saved_profile": True,
        }

    def _build_project_descriptor_profiles(
        self,
        keyword_candidates: List[Dict],
        target_files: Set[str],
    ) -> Dict[str, Dict]:
        """Use saved project profiles when available, else layout signals from GP."""
        profiles: Dict[str, Dict] = {}
        saved_profiles = ProjectProfileBuilder.load_all_profiles()

        for file_name in target_files:
            if file_name in saved_profiles:
                profiles[file_name] = self._saved_profile_to_descriptor(saved_profiles[file_name])
                continue

            layout = collect_layout_signals(file_name, self.vector_store, keyword_candidates)
            gp_doc = None
            if layout.get("gp_page") is not None:
                chunk = self.vector_store.get_page_chunk(file_name, int(layout["gp_page"]))
                if chunk:
                    gp_doc = Document(page_content=chunk.get("content", ""), metadata=chunk.get("metadata", {}))

            profiles[file_name] = {
                "span_count": layout.get("span_count"),
                "concrete": layout.get("concrete"),
                "bridge": layout.get("bridge"),
                "gp_page": layout.get("gp_page"),
                "gp_doc": gp_doc,
            }

        return profiles

    def _project_matches_descriptor(self, profile: Dict, descriptor: Dict) -> bool:
        span_count = profile.get("span_count")
        if span_count is None or span_count != descriptor.get("span_count"):
            return False
        if descriptor.get("wants_concrete") and not profile.get("concrete"):
            return False
        if descriptor.get("wants_bridge") and descriptor.get("wants_concrete"):
            return bool(profile.get("bridge"))
        return True

    def _is_typical_section_sheet(self, doc: Document) -> bool:
        metadata = getattr(doc, "metadata", {}) or {}
        title = str(
            metadata.get("topology_sheet_title")
            or metadata.get("plan_sheet_title")
            or ""
        ).lower()
        primary = str(metadata.get("plan_primary_type") or "").lower()
        blob = self._metadata_search_blob(metadata)
        if "typical section" in title or "typical section" in primary:
            return True
        if "typical section" in blob:
            return True
        return "typical section" in self._visible_sheet_text(doc)[:2500]

    def _sheet_intent_match_strength(self, doc: Document, descriptor: Dict) -> int:
        """Score how strongly a page matches layered sheet-level query intent."""
        intents = descriptor.get("sheet_intents") or []
        if not intents:
            return 0

        metadata = getattr(doc, "metadata", {}) or {}
        title = str(
            metadata.get("topology_sheet_title")
            or metadata.get("plan_sheet_title")
            or ""
        ).lower()
        primary = str(metadata.get("plan_primary_type") or "").lower()
        blob = self._metadata_search_blob(metadata)
        visible = self._visible_sheet_text(doc)
        strength = 0

        if "typical_section" in intents:
            if (
                self._is_general_plan_sheet(doc)
                or "index to plans" in title
                or primary == "index to plans"
            ):
                strength -= 20
            elif "typical section" in title:
                strength += 14
                if re.search(r"typical section\s+no\.?\s*\d", title):
                    strength += 3
                if "stage 1" in title or "stage 2" in title:
                    strength += 2
                if "partial" in title:
                    strength -= 4
            elif primary == "typical section" or "typical section" in primary:
                strength += 12
            elif "typical section" in blob or "typical section" in visible[:2500]:
                strength += 4
                if "partial typical section" in blob:
                    strength -= 3

            if "reinforcement" in title and "typical section" not in title:
                strength -= 10
            if "girder layout" in title:
                strength -= 10

            if descriptor.get("wants_concrete"):
                cip_markers = (
                    "cip",
                    "box girder",
                    "conc box girder",
                    "concrete box girder",
                    "cip/ps",
                )
                if any(marker in blob or marker in visible for marker in cip_markers):
                    strength += 3

        if "general_plan" in intents and self._is_general_plan_sheet(doc):
            strength += 12

        if "reinforcement_detail" in intents:
            if any(
                phrase in title
                for phrase in (
                    "reinforcement",
                    "reinf detail",
                    "rebar",
                    "girder details",
                )
            ):
                strength += 10
            if "reinforcement" in blob or "rebar" in blob:
                strength += 4

        if "girder_detail" in intents:
            if "girder detail" in title:
                strength += 10
            elif "girder details" in title:
                strength += 8

        if "abutment_detail" in intents:
            if "abutment detail" in title:
                strength += 10

        span_count = descriptor.get("span_count")
        if span_count is not None:
            span_range = re.search(r"spans?\s+(\d+)\s+thru(?:ugh)?\s+(\d+)", title)
            if span_range:
                upper = int(span_range.group(2))
                if upper > span_count + 1:
                    strength -= 6

        return strength

    def _preserve_engineer_best_pages_for_sheet_ranking(
        self,
        query: Optional[str],
        ranked_results: List[tuple],
        search_results: List[tuple],
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
    ) -> List[tuple]:
        """Keep exact-query Best labels even when sheet-intent strength is below threshold."""
        if not query or not self._is_detail_intent_query(query):
            return ranked_results

        exact_labels = self._load_exact_feedback_labels(query)
        best_pages = sorted(
            {
                (file_name, page_number)
                for (file_name, page_number), (label, _weight) in exact_labels.items()
                if label == "best"
            }
        )
        if not best_pages:
            return ranked_results

        present_pages = {
            (
                str(doc.metadata.get("file_name") or ""),
                int(doc.metadata.get("page") or 0),
            )
            for doc, _ in ranked_results
            if doc.metadata.get("file_name") is not None and doc.metadata.get("page") is not None
        }
        docs_by_page: Dict[tuple, Document] = {}
        for doc, _ in search_results:
            file_name = str(doc.metadata.get("file_name") or "")
            page_number = doc.metadata.get("page")
            if not file_name or page_number is None:
                continue
            docs_by_page[(file_name, int(page_number))] = doc

        augmented = list(ranked_results)
        injected = 0
        for file_name, page_number in best_pages:
            if scoped_file_names and file_name not in scoped_file_names:
                continue
            if (file_name, page_number) in present_pages:
                continue

            doc = docs_by_page.get((file_name, page_number))
            if not doc:
                doc = self._resolve_feedback_page_document(
                    file_name,
                    page_number,
                    keyword_candidates,
                    allow_placeholder=True,
                )
            if not doc:
                continue

            augmented.append((doc, -0.99))
            present_pages.add((file_name, page_number))
            injected += 1

        if injected:
            logger.info(
                "Preserved %s engineer Best page(s) through sheet-intent ranking for '%s'",
                injected,
                query,
            )

        augmented.sort(key=lambda item: item[1])
        return augmented

    def _apply_layered_project_sheet_ranking(
        self,
        search_results: List[tuple],
        descriptor: Dict,
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
        matching_files: Set[str],
        profiles: Dict[str, Dict],
        query: Optional[str] = None,
    ) -> List[tuple]:
        """
        Layer 1: matching projects by span/material.
        Layer 2: typical section / detail sheets inside those projects.
        """
        if not matching_files:
            return []

        page_docs: Dict[tuple, Document] = {}
        page_strength: Dict[tuple, int] = {}

        def consider_doc(doc: Document) -> None:
            metadata = getattr(doc, "metadata", {}) or {}
            file_name = str(metadata.get("file_name") or "")
            page_num = metadata.get("page")
            if file_name not in matching_files or page_num is None:
                return

            profile = profiles.get(file_name, {})
            if descriptor.get("wants_concrete") and not profile.get("concrete"):
                return

            if self._is_project_overview_page(doc):
                return

            strength = self._sheet_intent_match_strength(doc, descriptor)
            intents = descriptor.get("sheet_intents") or []
            if "typical_section" in intents and "reinforcement_detail" in intents:
                min_strength = 10
            elif "typical_section" in intents:
                min_strength = 12
            else:
                min_strength = 8
            if strength < min_strength:
                return

            page_key = (file_name, int(page_num))
            if strength > page_strength.get(page_key, 0):
                page_strength[page_key] = strength
                page_docs[page_key] = doc

        for doc, _ in search_results:
            consider_doc(doc)

        for item in keyword_candidates:
            metadata = item.get("metadata", {}) or {}
            doc = Document(page_content=item.get("content", ""), metadata=metadata)
            consider_doc(doc)

        if not page_strength:
            return self._preserve_engineer_best_pages_for_sheet_ranking(
                query,
                [],
                search_results,
                keyword_candidates,
                scoped_file_names,
            )

        sheets_by_file: Dict[str, List[tuple]] = defaultdict(list)
        for page_key, strength in page_strength.items():
            sheets_by_file[page_key[0]].append((strength, page_docs[page_key]))

        finalized: List[tuple] = []
        for file_name in sorted(matching_files):
            ranked = sorted(
                sheets_by_file.get(file_name, []),
                key=lambda item: (-item[0], item[1].metadata.get("page", 0)),
            )
            dedicated = [
                (strength, doc)
                for strength, doc in ranked
                if strength >= 12
                and "typical section"
                in str(
                    doc.metadata.get("plan_sheet_title")
                    or doc.metadata.get("topology_sheet_title")
                    or doc.metadata.get("plan_primary_type")
                    or ""
                ).lower()
            ] if "typical_section" in (descriptor.get("sheet_intents") or []) else ranked
            chosen = dedicated or ranked
            for strength, doc in chosen[:3]:
                finalized.append((doc, 0.0 - (strength / 100.0)))

        finalized.sort(key=lambda item: item[1])
        return self._preserve_engineer_best_pages_for_sheet_ranking(
            query,
            finalized,
            search_results,
            keyword_candidates,
            scoped_file_names,
        )

    def _apply_project_descriptor_ranking(
        self,
        search_results: List[tuple],
        descriptor: Dict,
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
        query: Optional[str] = None,
    ) -> List[tuple]:
        """
        Rank layered bridge lookups: project filter first, then sheet intent.
        Falls back to GP No. 1 when no sheet intent is present.
        """
        target_files = scoped_file_names or set(self.metadata_manager.get_all_projects().keys())
        if not target_files:
            return search_results

        profiles = self._build_project_descriptor_profiles(keyword_candidates, target_files)
        matching_files = {
            file_name
            for file_name, profile in profiles.items()
            if self._project_matches_descriptor(profile, descriptor)
        }

        if matching_files:
            logger.info(
                "Project descriptor match for span=%s: %s",
                descriptor.get("span_count"),
                ", ".join(sorted(matching_files)),
            )

        if descriptor.get("sheet_intents"):
            return self._apply_layered_project_sheet_ranking(
                search_results,
                descriptor,
                keyword_candidates,
                scoped_file_names,
                matching_files,
                profiles,
                query=query,
            )

        present_pages = {
            (
                str(doc.metadata.get("file_name") or ""),
                int(doc.metadata.get("page") or 0),
            )
            for doc, _ in search_results
            if doc.metadata.get("file_name") and doc.metadata.get("page") is not None
        }

        filtered: List[tuple] = []
        for doc, score in search_results:
            file_name = str(doc.metadata.get("file_name") or "")
            if file_name not in matching_files:
                continue
            if not self._is_general_plan_sheet(doc):
                continue
            filtered.append((doc, min(score, 0.0)))

        for file_name in matching_files:
            profile = profiles.get(file_name, {})
            gp_doc = profile.get("gp_doc")
            gp_page = profile.get("gp_page")
            if not gp_doc or gp_page is None:
                continue
            if (file_name, int(gp_page)) in present_pages:
                continue
            filtered.append((gp_doc, 0.0))

        best_gp_by_file: Dict[str, tuple] = {}
        for doc, score in filtered:
            file_name = str(doc.metadata.get("file_name") or "")
            page_number = int(doc.metadata.get("page") or 0)
            current = best_gp_by_file.get(file_name)
            if not current or page_number < current[0]:
                best_gp_by_file[file_name] = (page_number, doc, score)

        if not matching_files:
            return []

        finalized = [(doc, score) for _, doc, score in best_gp_by_file.values()]
        finalized.sort(key=lambda item: item[1])
        return finalized

    def _apply_topology_layer_boost(self, search_results: List[tuple], query: str) -> List[tuple]:
        """Boost chunks with layered topology evidence for structure/bridge queries."""
        query_lower = (query or "").lower()
        triggers = (
            "bridge",
            "structure type",
            "type of bridge",
            "box girder",
            "superstructure",
            "substructure",
            "foundation system",
            "cross reference",
            "span",
            "project profile",
            "design intent",
        )
        if not any(token in query_lower for token in triggers):
            return search_results

        boosted = []
        for doc, score in search_results:
            adjusted = score
            blob = self._doc_search_blob(doc)
            has_project_layer = "[topology project]" in blob
            has_detail_layer = "[topology detail]" in blob
            has_cross_refs = "[topology cross references]" in blob
            has_classification = "[search classification]" in blob
            has_project_profile = "[project profile]" in blob
            token_hits = sum(
                1
                for token in self._get_query_tokens(query)
                if self._doc_contains_query_token(doc, token)
            )

            if has_project_profile:
                if self._is_project_overview_query(query) and token_hits >= 1:
                    adjusted = min(adjusted, max(0.0, score - 0.92))
                elif not self._is_project_overview_query(query):
                    adjusted = max(adjusted, score + 2.0)
            elif has_classification and token_hits >= 1:
                adjusted = min(adjusted, max(0.0, score - 0.85))
            elif has_project_layer and token_hits >= 1:
                adjusted = min(adjusted, max(0.0, score - 0.8))
            elif has_detail_layer and token_hits >= 2:
                adjusted = min(adjusted, max(0.0, score - 0.7))
            elif has_cross_refs and "reference" in query_lower:
                adjusted = min(adjusted, max(0.0, score - 0.65))

            boosted.append((doc, adjusted))

        boosted.sort(key=lambda item: item[1])
        return boosted

    def _apply_general_notes_reference_boost(
        self, search_results: List[tuple], query: str
    ) -> List[tuple]:
        """
        Boost general-notes pages for standard-sheet reference queries so each
        project survives tight relevance filters.
        """
        if not self._is_general_notes_reference_query(query):
            return search_results

        boosted = []
        for doc, score in search_results:
            adjusted = score
            match_strength = self._general_notes_match_strength(doc, query)
            is_general_notes_page = self._is_general_notes_page(doc)

            if match_strength >= 5:
                adjusted = min(adjusted, max(0.0, score - 0.9))
            elif match_strength >= 3 and is_general_notes_page:
                adjusted = min(adjusted, max(0.0, score - 0.85))
            elif is_general_notes_page and match_strength >= 1:
                adjusted = min(adjusted, max(0.0, score - 0.75))

            boosted.append((doc, adjusted))

        boosted.sort(key=lambda item: item[1])
        return boosted

    def _ensure_per_project_general_notes_coverage(
        self,
        search_results: List[tuple],
        query: str,
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
    ) -> List[tuple]:
        """
        Ensure each indexed project contributes a general-notes hit for
        reference-style queries when matching content exists.
        """
        if not self._is_general_notes_reference_query(query):
            return search_results

        target_projects = scoped_file_names or set(self.metadata_manager.get_all_projects().keys())
        if not target_projects:
            return search_results

        present_projects = {
            str(doc.metadata.get("file_name") or "")
            for doc, _ in search_results
            if doc.metadata.get("file_name")
        }
        anchor_score = min(score for _, score in search_results) if search_results else 0.0

        best_by_project: Dict[str, tuple] = {}
        for item in keyword_candidates:
            metadata = item.get("metadata", {}) or {}
            file_name = str(metadata.get("file_name") or "")
            if file_name not in target_projects:
                continue

            doc = Document(page_content=item.get("content", ""), metadata=metadata)
            if not self._is_general_notes_page(doc):
                continue

            strength = self._general_notes_match_strength(doc, query)
            if strength <= 0:
                continue

            current = best_by_project.get(file_name)
            if not current or strength > current[0]:
                best_by_project[file_name] = (strength, doc)

        augmented = list(search_results)
        for file_name, (strength, doc) in best_by_project.items():
            if file_name in present_projects:
                continue
            score = anchor_score
            if strength >= 5:
                score = 0.0
            elif strength >= 3:
                score = min(score, 0.05)
            augmented.append((doc, score))

        augmented.sort(key=lambda item: item[1])
        return augmented

    def _apply_title_block_boost(self, search_results: List[tuple], query: str) -> List[tuple]:
        """
        Boost pages whose title-block metadata matches sheet-type queries.

        Distance scores are lower-is-better, so matching title blocks receive a
        strong score reduction to survive tight relevance filters.
        """
        query_lower = (query or "").lower()
        phrase_boosts = []
        if "general plan" in query_lower:
            phrase_boosts.append(("general plan", 0.9))
        if "general note" in query_lower:
            phrase_boosts.append(("general note", 0.75))
        if "foundation plan" in query_lower:
            phrase_boosts.append(("foundation plan", 0.8))

        if not phrase_boosts:
            return search_results

        boosted = []
        for doc, score in search_results:
            adjusted = score
            for phrase, boost in phrase_boosts:
                if self._doc_matches_title_block_phrase(doc, phrase):
                    adjusted = min(adjusted, max(0.0, score - boost))
                    break
            boosted.append((doc, adjusted))

        boosted.sort(key=lambda item: item[1])
        return boosted

    def _active_pdf_file_names(self) -> Set[str]:
        return self.metadata_manager.get_active_pdf_file_names()

    def _is_active_feedback_file(self, file_name: str) -> bool:
        return bool(file_name) and file_name in self._active_pdf_file_names()

    def _feedback_store_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "data", "feedback", "search_feedback.jsonl")

    def _feedback_model_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "data", "feedback", "search_feedback_model.json")

    def _load_feedback_model(self) -> Dict:
        path = self._feedback_model_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            return {}

    def _normalize_scope(self, value: Optional[str]) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    def _get_query_tokens(self, query: str) -> Set[str]:
        """Extract significant tokens from normalized query (length >= 3)."""
        normalized = self._normalize_query_text(query)
        # Remove common stop words and extract tokens >= 3 chars
        stop_words = {"and", "the", "for", "with", "all", "are", "one", "two", "pin"}
        tokens = set()
        for token in normalized.split():
            if len(token) >= 3 and token not in stop_words:
                tokens.add(token)
        return tokens

    def _feedback_label_delta(self) -> Dict[str, float]:
        return {
            "best": -0.95,
            "relevant": -0.75,
            "irrelevant": 0.85,
        }

    def _feedback_source_weights(self, query: str, project_scope: Optional[str]) -> Dict[str, float]:
        """Exact and generalized query weights for feedback lookup."""
        query_key = self._canonical_feedback_query(query)
        feedback_sources = {query_key: 1.0}

        model = self._load_feedback_model()
        related_lookup_keys = {query_key}
        if query_key.endswith(" detail"):
            related_lookup_keys.add(query_key.replace(" detail", " details"))
        for lookup_key in related_lookup_keys:
            for related_query, similarity in model.get("related_queries", {}).get(lookup_key, {}).items():
                canonical_related = self._canonical_feedback_query(related_query)
                feedback_sources[canonical_related] = max(
                    feedback_sources.get(canonical_related, 0.0),
                    0.65 * float(similarity),
                )

        similar_queries = self._find_similar_queries_in_feedback(query, project_scope)
        for sim_query, similarity_score in similar_queries.items():
            canonical_similar = self._canonical_feedback_query(sim_query)
            feedback_sources[canonical_similar] = max(
                feedback_sources.get(canonical_similar, 0.0),
                0.55 * similarity_score,
            )

        if len(feedback_sources) > 1:
            logger.info(
                f"Feedback generalization for '{query}': "
                f"using {len(feedback_sources) - 1} related query source(s)"
            )

        return feedback_sources

    def _find_similar_queries_in_feedback(self, query: str, project_scope: Optional[str]) -> Dict[str, float]:
        """
        Find queries in feedback store that are similar to the given query.
        Returns dict of {normalized_query: similarity_score (0.0-1.0)}.
        """
        path = self._feedback_store_path()
        if not os.path.exists(path):
            return {}

        query_key = self._canonical_feedback_query(query)
        query_tokens = self._get_query_tokens(query)
        if not query_tokens:
            return {}

        min_overlap = 1 if len(query_tokens) <= 3 else 2
        similar_queries: Dict[str, float] = {}
        seen_queries: Set[str] = set()

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    record_query = str(record.get("query", ""))
                    normalized_record_query = self._canonical_feedback_query(record_query)
                    if normalized_record_query == query_key:
                        continue
                    if normalized_record_query in seen_queries:
                        continue
                    seen_queries.add(normalized_record_query)

                    record_tokens = self._get_query_tokens(record_query)
                    if not record_tokens:
                        continue

                    overlap = len(query_tokens & record_tokens)
                    contains = query_key in normalized_record_query or normalized_record_query in query_key
                    similarity = 0.0
                    if contains and overlap >= 1:
                        similarity = 0.8
                    elif overlap >= min_overlap:
                        similarity = overlap / len(query_tokens | record_tokens)

                    if similarity > 0.0:
                        similar_queries[normalized_record_query] = max(
                            similar_queries.get(normalized_record_query, 0.0),
                            similarity,
                        )
        except Exception:
            return {}

        return similar_queries

    def _iter_latest_feedback_labels(
        self, feedback_sources: Dict[str, float]
    ) -> Dict[tuple, tuple]:
        """
        Latest label per (file, page) across matching feedback queries.
        Returns {(file, page): (label, weight)}.
        """
        path = self._feedback_store_path()
        if not os.path.exists(path):
            return {}

        latest: Dict[tuple, tuple] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    record_query = str(record.get("query", ""))
                    weight = feedback_sources.get(self._canonical_feedback_query(record_query))
                    if weight is None:
                        continue

                    file_name = str(record.get("pdf_file_name", "")).strip()
                    page_number = record.get("page_number")
                    label = str(record.get("feedback", "")).strip().lower()
                    if (
                        not file_name
                        or not self._is_active_feedback_file(file_name)
                        or page_number is None
                        or label not in self._feedback_label_delta()
                    ):
                        continue

                    try:
                        page_number = int(page_number)
                    except Exception:
                        continue

                    key = (file_name, page_number)
                    current = latest.get(key)
                    if current is not None and weight < current[1]:
                        continue
                    latest[key] = (label, weight)
        except Exception:
            return {}

        return latest

    def _load_feedback_adjustments(self, query: str, project_scope: Optional[str]) -> Dict[tuple, float]:
        """
        Load page-level score adjustments from recorded engineer feedback.
        Includes exact query matches and generalized feedback from similar queries.
        Feedback is GLOBAL across projects; project_scope is kept for compatibility.
        """
        feedback_sources = self._feedback_source_weights(query, project_scope)
        latest_labels = self._iter_latest_feedback_labels(feedback_sources)
        if not latest_labels:
            return {}

        label_delta = self._feedback_label_delta()
        adjustments: Dict[tuple, float] = {}
        for key, (label, weight) in latest_labels.items():
            adjustments[key] = max(-1.0, min(1.0, weight * label_delta[label]))

        return adjustments

    def _load_positive_feedback_pages(
        self, query: str, project_scope: Optional[str]
    ) -> Dict[tuple, str]:
        """Return best/relevant pages labeled on the exact query (not related queries)."""
        return {
            key: label
            for key, (label, _weight) in self._load_exact_feedback_labels(query).items()
            if label in {"best", "relevant"}
        }

    def _load_exact_feedback_labels(self, query: str) -> Dict[tuple, tuple]:
        """Return latest labels for the exact canonical query only."""
        query_key = self._canonical_feedback_query(query)
        if not query_key:
            return {}

        path = self._feedback_store_path()
        if not os.path.exists(path):
            return {}

        latest: Dict[tuple, tuple] = {}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    if self._canonical_feedback_query(str(record.get("query", ""))) != query_key:
                        continue

                    file_name = str(record.get("pdf_file_name", "")).strip()
                    page_number = record.get("page_number")
                    label = str(record.get("feedback", "")).strip().lower()
                    if (
                        not file_name
                        or not self._is_active_feedback_file(file_name)
                        or page_number is None
                        or label not in self._feedback_label_delta()
                    ):
                        continue

                    try:
                        page_number = int(page_number)
                    except Exception:
                        continue

                    latest[(file_name, page_number)] = (label, 1.0)
        except Exception:
            return {}

        return latest

    def _apply_exact_feedback_filters(
        self, search_results: List[tuple], query: str
    ) -> List[tuple]:
        """Drop pages the engineer marked irrelevant on this exact query."""
        exact_labels = self._load_exact_feedback_labels(query)
        if not exact_labels:
            return search_results

        irrelevant_pages = {
            key
            for key, (label, _weight) in exact_labels.items()
            if label == "irrelevant"
        }
        if not irrelevant_pages:
            return search_results

        return [
            (doc, score)
            for doc, score in search_results
            if (
                str(doc.metadata.get("file_name") or ""),
                int(doc.metadata.get("page") or 0),
            )
            not in irrelevant_pages
        ]

    def _prioritize_exact_feedback_best_pages(
        self, search_results: List[tuple], query: str
    ) -> List[tuple]:
        """
        When the engineer labeled Best pages for this exact query, keep only those
        pages per project instead of every incidental shear-key mention.
        """
        if not self._is_detail_intent_query(query):
            return search_results

        exact_labels = self._load_exact_feedback_labels(query)
        best_by_file: Dict[str, Set[int]] = defaultdict(set)
        for (file_name, page_number), (label, _weight) in exact_labels.items():
            if label == "best":
                best_by_file[file_name].add(page_number)

        if not best_by_file:
            return search_results

        prioritized: List[tuple] = []
        remainder: List[tuple] = []
        for doc, score in search_results:
            file_name = str(doc.metadata.get("file_name") or "")
            page_number = int(doc.metadata.get("page") or 0)
            if file_name in best_by_file:
                if page_number in best_by_file[file_name]:
                    prioritized.append((doc, score - 0.25))
                continue
            remainder.append((doc, score))

        if not prioritized:
            return search_results

        prioritized.sort(key=lambda item: item[1])
        remainder.sort(key=lambda item: item[1])
        return prioritized + remainder

    def _filter_weak_unlabeled_detail_results(
        self, search_results: List[tuple], query: str
    ) -> List[tuple]:
        """
        When engineers labeled Best pages for this query, drop unlabeled projects
        that only have incidental keyword mentions (e.g. abutment notes).
        """
        if not self._is_detail_intent_query(query):
            return search_results

        exact_labels = self._load_exact_feedback_labels(query)
        best_by_file: Dict[str, Set[int]] = defaultdict(set)
        for (file_name, page_number), (label, _weight) in exact_labels.items():
            if label == "best":
                best_by_file[file_name].add(page_number)

        if len(best_by_file) < 2:
            return search_results

        filtered: List[tuple] = []
        for doc, score in search_results:
            file_name = str(doc.metadata.get("file_name") or "")
            page_number = int(doc.metadata.get("page") or 0)
            if file_name in best_by_file:
                if page_number in best_by_file[file_name]:
                    filtered.append((doc, score))
                continue

            strength = self._detail_intent_match_strength(doc, query)
            if strength >= 9:
                filtered.append((doc, score))

        return filtered if filtered else search_results

    def _feedback_placeholder_document(self, file_name: str, page_number: int) -> Document:
        """Minimal chunk so engineer-labeled pages still appear when PDF/index is missing."""
        return Document(
            page_content=(
                f"[engineer feedback best page]\n"
                f"file: {file_name}\n"
                f"page: {page_number}"
            ),
            metadata={"file_name": file_name, "page": page_number},
        )

    def _load_page_document_from_pdf(
        self, file_name: str, page_number: int
    ) -> Optional[Document]:
        """Load page text directly from storage when the vector index has no chunk."""
        try:
            from storage_adapter import storage
            import fitz

            if not storage.pdf_exists(file_name):
                return None

            pdf_path = storage.get_pdf_temp_path(file_name)
            with fitz.open(pdf_path) as pdf:
                if page_number < 1 or page_number > pdf.page_count:
                    return None
                text = pdf[page_number - 1].get_text() or ""
                if not text.strip():
                    return None
                return Document(
                    page_content=text,
                    metadata={"file_name": file_name, "page": page_number},
                )
        except Exception:
            return None

    def _resolve_feedback_page_document(
        self,
        file_name: str,
        page_number: int,
        keyword_candidates: List[Dict],
        *,
        allow_placeholder: bool = False,
    ) -> Optional[Document]:
        for item in keyword_candidates:
            metadata = item.get("metadata", {}) or {}
            if str(metadata.get("file_name") or "") != file_name:
                continue
            page = metadata.get("page")
            try:
                page = int(page)
            except Exception:
                continue
            if page != page_number:
                continue
            return Document(page_content=item.get("content", ""), metadata=metadata)

        chunk = self.vector_store.get_page_chunk(file_name, page_number)
        if chunk:
            return Document(
                page_content=chunk.get("content", ""),
                metadata=chunk.get("metadata", {}),
            )

        pdf_doc = self._load_page_document_from_pdf(file_name, page_number)
        if pdf_doc:
            return pdf_doc

        if allow_placeholder:
            return self._feedback_placeholder_document(file_name, page_number)
        return None

    def _ensure_missing_feedback_best_pages(
        self,
        search_results: List[tuple],
        query: str,
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
    ) -> List[tuple]:
        """Backfill every engineer Best page for this query, even if retrieval missed it."""
        if not self._is_detail_intent_query(query):
            return search_results

        exact_labels = self._load_exact_feedback_labels(query)
        best_pages = sorted(
            {
                (file_name, page_number)
                for (file_name, page_number), (label, _weight) in exact_labels.items()
                if label == "best"
            }
        )
        if not best_pages:
            return search_results

        present_pages = {
            (
                str(doc.metadata.get("file_name") or ""),
                int(doc.metadata.get("page") or 0),
            )
            for doc, _ in search_results
            if doc.metadata.get("file_name") is not None and doc.metadata.get("page") is not None
        }

        augmented = list(search_results)
        injected = 0
        for file_name, page_number in best_pages:
            if scoped_file_names and file_name not in scoped_file_names:
                continue
            if (file_name, page_number) in present_pages:
                continue

            doc = self._resolve_feedback_page_document(
                file_name,
                page_number,
                keyword_candidates,
                allow_placeholder=True,
            )
            if not doc:
                continue

            augmented.append((doc, 0.0))
            present_pages.add((file_name, page_number))
            injected += 1

        if injected:
            logger.info(
                f"Backfilled {injected} missing engineer Best page(s) for '{query}'"
            )

        augmented.sort(key=lambda item: item[1])
        return augmented

    def _inject_feedback_labeled_pages(
        self,
        search_results: List[tuple],
        positive_pages: Dict[tuple, str],
        keyword_candidates: List[Dict],
        scoped_file_names: Set[str],
    ) -> List[tuple]:
        """Ensure engineer-labeled best/relevant pages appear in ranked results."""
        if not positive_pages:
            return search_results

        present_pages = {
            (
                str(doc.metadata.get("file_name") or ""),
                int(doc.metadata.get("page") or 0),
            )
            for doc, _ in search_results
            if doc.metadata.get("file_name") and doc.metadata.get("page") is not None
        }
        anchor_score = min(score for _, score in search_results) if search_results else 0.08

        augmented = list(search_results)
        injected = 0
        for (file_name, page_number), label in positive_pages.items():
            if scoped_file_names and file_name not in scoped_file_names:
                continue
            if (file_name, page_number) in present_pages:
                continue

            doc = self._resolve_feedback_page_document(
                file_name,
                page_number,
                keyword_candidates,
                allow_placeholder=(label == "best"),
            )
            if not doc:
                continue

            score = 0.0 if label == "best" else min(anchor_score, 0.03)
            augmented.append((doc, score))
            injected += 1

        if injected:
            logger.info(f"Injected {injected} feedback-labeled page(s) into search results")

        augmented.sort(key=lambda item: item[1])
        return augmented

    def _filter_strong_irrelevant_feedback(
        self, search_results: List[tuple], adjustments: Dict[tuple, float]
    ) -> List[tuple]:
        """Drop pages with strong irrelevant feedback before rescoring survivors."""
        filtered = []
        for doc, score in search_results:
            file_name = str(doc.metadata.get("file_name", "")).strip()
            page_number = doc.metadata.get("page")
            try:
                page_number = int(page_number)
            except Exception:
                page_number = None

            delta = adjustments.get((file_name, page_number), 0.0)
            if delta >= 0.55:
                continue
            filtered.append((doc, score))
        return filtered

    def _apply_feedback_adjustments(self, search_results: List[tuple], adjustments: Dict[tuple, float]) -> List[tuple]:
        """Apply page-level distance adjustments (lower is better)."""
        rescored = []
        for doc, score in search_results:
            file_name = str(doc.metadata.get("file_name", "")).strip()
            page_number = doc.metadata.get("page")
            try:
                page_number = int(page_number)
            except Exception:
                page_number = None

            delta = adjustments.get((file_name, page_number), 0.0)
            rescored.append((doc, score + delta))

        rescored.sort(key=lambda x: x[1])
        return rescored

    def _keyword_search_bm25(self, query: str, candidates: List[Dict], top_k: int) -> List[tuple]:
        """Rank candidates with a local BM25-style scorer (no external API calls)."""
        if not candidates:
            return []

        query_tokens = self._tokenize_for_keyword_search(query)
        if not query_tokens:
            return []

        # Keep short n-grams to strongly reward specific phrases like "box girder".
        query_phrases = set()
        for n in (2, 3):
            for i in range(len(query_tokens) - n + 1):
                query_phrases.add(" ".join(query_tokens[i : i + n]))

        query_lower = query.lower()
        if "concrete box girder" in query_lower:
            query_phrases.update({"r/c box girder", "rc box girder"})
        if "reinforcement" in query_lower:
            query_phrases.update({"rebar detail", "reinforcing steel"})
        if "general plan" in query_lower:
            query_phrases.update({"general plan", "general plans", "general plan no"})
        if "ars curve" in query_lower or ("ars" in query_lower and "curve" in query_lower):
            query_phrases.update(
                {
                    "ars curve",
                    "design ars curve",
                    "curve design ars",
                    "design ars",
                    "damping design ars",
                }
            )
        if "caltrans" in query_lower or "standard plan" in query_lower:
            query_phrases.update(
                {
                    "caltrans",
                    "standard plan",
                    "standard plans",
                    "caltrans seismic design criteria",
                    "standard plan sheet",
                }
            )
        if "abutment" in query_lower and "pile" in query_lower:
            query_phrases.update(
                {
                    "abutment pile",
                    "pile detail",
                    "steel pipe pile detail",
                    "steel pipe pile",
                    "abutment details",
                    "abutment details no",
                    "pile (type a) detail",
                }
            )

        tokenized_docs = []
        doc_freq = Counter()
        doc_lengths = []

        for item in candidates:
            tokens = self._tokenize_for_keyword_search(item.get("content", ""))
            tokenized_docs.append(tokens)
            doc_lengths.append(len(tokens) if tokens else 1)
            for token in set(tokens):
                doc_freq[token] += 1

        n_docs = len(tokenized_docs)
        if n_docs == 0:
            return []

        avg_doc_len = sum(doc_lengths) / n_docs
        k1 = 1.5
        b = 0.75

        scored = []
        for idx, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue

            term_freq = Counter(tokens)
            doc_len = doc_lengths[idx]
            score = 0.0

            for term in query_tokens:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue

                df = doc_freq.get(term, 0)
                idf = math.log(1 + ((n_docs - df + 0.5) / (df + 0.5)))
                tf_weight = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
                score += idf * tf_weight

            doc_text_raw = str(candidates[idx].get("content", ""))
            doc_text_lower = doc_text_raw.lower()

            # Penalize low-signal fallback text that often appears when vision extraction fails.
            if "unable to analyze the image directly" in doc_text_lower:
                score -= 5.0
            if "identify the type of drawing" in doc_text_lower and "drawing type" in doc_text_lower:
                score -= 2.0

            # Strongly boost phrase matches; this improves precision on specific sheet intents.
            phrase_hits = sum(1 for phrase in query_phrases if phrase in doc_text_lower)
            if phrase_hits > 0:
                score += 6.0 * phrase_hits

            detail_title_text = " | ".join(self._extract_detail_titles(doc_text_raw)).lower()
            if detail_title_text:
                detail_title_hits = sum(1 for phrase in query_phrases if phrase in detail_title_text)
                if detail_title_hits > 0:
                    score += 10.0 * detail_title_hits

            # Metadata-aware bonus improves identifier lookup (project numbers, sheet titles, engineer names).
            metadata_text = self._metadata_search_blob(candidates[idx].get("metadata", {}) or {})
            metadata_text += " " + " ".join(
                str(candidates[idx].get("metadata", {}).get(field, ""))
                for field in [
                    "enriched_project_number",
                    "enriched_engineer_on_record",
                ]
            ).lower()
            if metadata_text:
                metadata_hit_count = sum(1 for term in query_tokens if term in metadata_text)
                if metadata_hit_count > 0:
                    score += 3.0 * metadata_hit_count

                metadata_phrase_hits = sum(1 for phrase in query_phrases if phrase in metadata_text)
                if metadata_phrase_hits > 0:
                    score += 8.0 * metadata_phrase_hits

            if score > 0:
                doc = Document(
                    page_content=candidates[idx].get("content", ""),
                    metadata=candidates[idx].get("metadata", {}),
                )
                scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _hybrid_fuse_results(self, vector_results: List[tuple], keyword_results: List[tuple], k: int) -> List[tuple]:
        """Fuse vector and keyword rankings using reciprocal rank fusion."""
        if not vector_results and not keyword_results:
            return []
        if not keyword_results:
            return vector_results[:k]
        if not vector_results:
            keyword_only = keyword_results[:k]
            max_score = keyword_only[0][1] if keyword_only else 1.0
            min_score = keyword_only[-1][1] if keyword_only else 0.0
            denom = max_score - min_score if max_score != min_score else 1.0
            return [(doc, 1.0 - ((score - min_score) / denom)) for doc, score in keyword_only]

        rrf_k = max(1, config.HYBRID_RRF_K)

        vector_rank = {}
        keyword_rank = {}
        docs_by_key = {}

        for rank, (doc, _score) in enumerate(sorted(vector_results, key=lambda x: x[1]), start=1):
            key = self._doc_key(doc.metadata)
            vector_rank[key] = rank
            docs_by_key[key] = doc

        for rank, (doc, _score) in enumerate(sorted(keyword_results, key=lambda x: x[1], reverse=True), start=1):
            key = self._doc_key(doc.metadata)
            keyword_rank[key] = rank
            docs_by_key[key] = doc

        fused_scores = {}
        for key in docs_by_key:
            score = 0.0
            if key in vector_rank:
                score += 1.0 / (rrf_k + vector_rank[key])
            if key in keyword_rank:
                score += 1.0 / (rrf_k + keyword_rank[key])
            fused_scores[key] = score

        fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        if not fused_sorted:
            return []

        max_fused = fused_sorted[0][1]
        min_fused = fused_sorted[-1][1]
        denom = max_fused - min_fused if max_fused != min_fused else 1.0

        # Convert fused score to a distance-like score where lower is better.
        return [
            (
                docs_by_key[key],
                1.0 - ((score - min_fused) / denom),
            )
            for key, score in fused_sorted
        ]
    
    def _organize_results_by_project(
        self,
        search_results: List[tuple]
    ) -> Dict:
        """
        Organize search results by project file.
        
        Args:
            search_results: List of (document, score) tuples
            
        Returns:
            Dictionary organized by project file name
        """
        projects = defaultdict(lambda: {
            'pages': set(),
            'chunks': [],
            'best_score': float('inf'),
        })
        
        for doc, score in search_results:
            file_name = doc.metadata.get('file_name')
            page = doc.metadata.get('page')
            
            if file_name:
                projects[file_name]['pages'].add(page)
                projects[file_name]['chunks'].append({
                    'content': doc.page_content,
                    'page': page,
                    'score': score,
                    'plan_sheet_type': doc.metadata.get('plan_sheet_type', ''),
                    'chunk_id': doc.metadata.get('chunk_id', 0),
                    'drawing_label': (
                        doc.metadata.get('cad_drawing_label', '') or ''
                        if '[DRAWING REGION CHUNK]' in (doc.page_content or '')
                        else ''
                    ),
                    'drawing_view': (
                        doc.metadata.get('cad_drawing_view', '') or ''
                        if '[DRAWING REGION CHUNK]' in (doc.page_content or '')
                        else ''
                    ),
                })
                
                # Track best (lowest) score
                if score < projects[file_name]['best_score']:
                    projects[file_name]['best_score'] = score
        
        # Order pages by best (lowest) relevance score per page
        for file_name in projects:
            page_best_scores: Dict = {}
            for chunk in projects[file_name]['chunks']:
                page = chunk.get('page')
                score = chunk.get('score', float('inf'))
                if page is None:
                    continue
                if page not in page_best_scores or score < page_best_scores[page]:
                    page_best_scores[page] = score
            projects[file_name]['pages'] = sorted(
                page_best_scores.keys(),
                key=lambda page: page_best_scores[page],
            )
            projects[file_name]['chunks'].sort(
                key=lambda chunk: (chunk.get('score', float('inf')), chunk.get('page', 0))
            )
        
        return dict(projects)
    
    def _enrich_with_metadata(self, projects_data: Dict) -> List[Dict]:
        """
        Enrich project data with metadata.
        
        Args:
            projects_data: Organized search results by project
            
        Returns:
            List of enriched project results
        """
        enriched = []
        
        for file_name, data in projects_data.items():
            # Get metadata from metadata manager
            metadata = self.metadata_manager.get_project_metadata(file_name)
            
            if metadata:
                enriched.append({
                    'project_name': metadata.get('project_name', file_name),
                    'file_name': file_name,
                    'file_path': metadata.get('file_path', ''),
                    'phase': metadata.get('phase', 'Unknown'),
                    'engineer_of_record': metadata.get('engineer_of_record', 'Unknown'),
                    'date': metadata.get('date', 'Unknown'),
                    'categories': metadata.get('categories', []),
                    'relevant_pages': data['pages'],
                    'total_pages': metadata.get('total_pages', 0),
                    'relevance_score': data['best_score'],
                    'sample_content': data['chunks'][0]['content'][:300] + '...' if data['chunks'] else '',
                    'chunks': data['chunks'],  # Include all chunks with their page-specific scores
                })
            else:
                # Fallback if metadata not found
                enriched.append({
                    'project_name': file_name,
                    'file_name': file_name,
                    'file_path': '',
                    'phase': 'Unknown',
                    'engineer_of_record': 'Unknown',
                    'date': 'Unknown',
                    'categories': [],
                    'relevant_pages': data['pages'],
                    'total_pages': 0,
                    'relevance_score': data['best_score'],
                    'sample_content': data['chunks'][0]['content'][:300] + '...' if data['chunks'] else '',
                    'chunks': data['chunks'],  # Include all chunks with their page-specific scores
                })
        
        # Sort by relevance score (lower is better)
        enriched.sort(key=lambda x: x['relevance_score'])
        
        return enriched
    
    def _generate_search_summary(
        self,
        query: str,
        results: List[Dict]
    ) -> str:
        """
        Generate a natural language summary of search results using LLM.
        
        Args:
            query: Original search query
            results: Enriched search results
            
        Returns:
            Summary string
        """
        if not results:
            return "No relevant documents found for the query."
        
        # Prepare context for LLM
        context = f"Search Query: {query}\n\n"
        context += f"Found {len(results)} relevant project(s):\n\n"
        
        for i, result in enumerate(results[:5], 1):  # Top 5 results
            context += f"{i}. Project: {result['project_name']}\n"
            context += f"   Phase: {result['phase']}\n"
            context += f"   Engineer: {result['engineer_of_record']}\n"
            context += f"   Relevant Pages: {', '.join(map(str, result['relevant_pages'][:10]))}\n"
            context += f"   Sample: {result['sample_content'][:200]}\n\n"
        
        # Generate summary
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes search results for structural engineering projects. Provide a concise summary highlighting the most relevant findings."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize these search results:\n\n{context}"
                    }
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return f"Found {len(results)} relevant project(s). See results below."
    
    def advanced_search(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> Dict:
        """
        Perform advanced search with multiple filters.
        
        Args:
            query: Search query
            filters: Dictionary of filters (category, phase, engineer, date_range)
            
        Returns:
            Search results
        """
        filters = filters or {}
        
        # Start with semantic search
        k = filters.get('max_results', 30)
        results = self.search(
            query=query,
            k=k,
            category_filter=filters.get('category'),
            phase_filter=filters.get('phase')
        )
        
        # Apply additional filters
        filtered_results = results['results']
        
        if filters.get('engineer'):
            engineer = filters['engineer'].lower()
            filtered_results = [
                r for r in filtered_results
                if engineer in r['engineer_of_record'].lower()
            ]
        
        if filters.get('category'):
            category = filters['category']
            filtered_results = [
                r for r in filtered_results
                if category in r['categories']
            ]
        
        # Update results
        results['results'] = filtered_results
        results['total_projects'] = len(filtered_results)
        
        return results
    
    def get_project_details(self, file_name: str) -> Optional[Dict]:
        """Get detailed information about a specific project."""
        return self.metadata_manager.get_project_metadata(file_name)
    
    def list_all_projects(self) -> List[Dict]:
        """List all indexed projects."""
        projects = self.metadata_manager.get_all_projects()
        return list(projects.values())
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base."""
        metadata_stats = self.metadata_manager.get_stats()
        vector_info = self.vector_store.get_collection_info()
        
        return {
            'metadata': metadata_stats,
            'vector_store': vector_info,
        }

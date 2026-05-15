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
            project_scope: Optional project/file substring scope (e.g., "mar vista")
            
        Returns:
            Dictionary containing search results and metadata
        """
        print(f"\nSearching for: '{query}'")

        normalized_query = self._normalize_query_text(query)

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
                "general plan",
                "rebar detail",
            ]
        )
        avoid_expansion = looks_like_identifier_query or (has_specific_phrase and token_count <= 6) or (token_count == 1)
        
        # Expand query with engineering terminology synonyms
        expanded_query = normalized_query
        if use_query_expansion and not avoid_expansion:
            expanded_terms = expand_query(normalized_query, max_expansions=3)
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
            search_results = self._apply_feedback_adjustments(search_results, feedback_adjustments)
        
        # Organize results by project
        projects_data = self._organize_results_by_project(search_results)
        
        # Enrich with metadata
        enriched_results = self._enrich_with_metadata(projects_data)
        
        should_generate_summary = config.GENERATE_SEARCH_SUMMARY if generate_summary is None else generate_summary
        if should_generate_summary:
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

        metadata = getattr(doc, "metadata", {}) or {}
        metadata_blob = " ".join(
            str(metadata.get(field, ""))
            for field in [
                "file_name",
                "project_name",
                "enriched_project_name",
                "plan_sheet_title",
                "plan_sheet_type",
                "detail_intents",
                "detail_types",
                "structural_elements",
            ]
        ).lower()
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

    def _feedback_store_path(self) -> str:
        return os.path.join(os.path.dirname(__file__), "data", "feedback", "search_feedback.jsonl")

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

    def _find_similar_queries_in_feedback(self, query: str, project_scope: Optional[str]) -> Dict[str, float]:
        """
        Find queries in feedback store that are similar to the given query.
        Returns dict of {normalized_query: similarity_score (0.0-1.0)}.
        Similarity is based on token overlap (at least 2 significant tokens).
        """
        path = self._feedback_store_path()
        if not os.path.exists(path):
            return {}

        query_tokens = self._get_query_tokens(query)
        if not query_tokens:
            return {}

        similar_queries: Dict[str, float] = {}

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

                    # Feedback is GLOBAL - applies across all projects
                    record_query = str(record.get("query", ""))
                    normalized_record_query = self._normalize_query_text(record_query)

                    # Skip exact match query (already handled with full weight)
                    if normalized_record_query == self._normalize_query_text(query):
                        continue

                    record_tokens = self._get_query_tokens(record_query)
                    if not record_tokens:
                        continue

                    overlap = len(query_tokens & record_tokens)
                    if overlap >= 2:
                        similarity = overlap / len(query_tokens | record_tokens)
                        similar_queries[normalized_record_query] = max(
                            similar_queries.get(normalized_record_query, 0.0),
                            similarity,
                        )
        except Exception:
            return {}

        return similar_queries

    def _load_feedback_adjustments(self, query: str, project_scope: Optional[str]) -> Dict[tuple, float]:
        """
        Load page-level score adjustments from recorded engineer feedback.
        Includes exact query matches and generalized feedback from similar queries.
        Feedback is GLOBAL across projects; project_scope is kept for compatibility.
        """
        path = self._feedback_store_path()
        if not os.path.exists(path):
            return {}

        query_key = self._normalize_query_text(query)
        adjustments: Dict[tuple, float] = defaultdict(float)

        label_delta = {
            "best": -0.35,
            "relevant": -0.20,
            "irrelevant": 0.35,
        }

        feedback_sources = {query_key: 1.0}
        similar_queries = self._find_similar_queries_in_feedback(query, project_scope)
        for sim_query, similarity_score in similar_queries.items():
            feedback_sources[sim_query] = 0.5 * similarity_score

        if similar_queries:
            logger.info(
                f"Feedback generalization for '{query}': "
                f"found {len(similar_queries)} similar queries with token overlap"
            )

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

                    record_query = self._normalize_query_text(str(record.get("query", "")))
                    if record_query not in feedback_sources:
                        continue

                    file_name = str(record.get("pdf_file_name", "")).strip()
                    page_number = record.get("page_number")
                    label = str(record.get("feedback", "")).strip().lower()
                    if not file_name or page_number is None or label not in label_delta:
                        continue

                    try:
                        page_number = int(page_number)
                    except Exception:
                        continue

                    key = (file_name, page_number)
                    weight = feedback_sources[record_query]
                    adjustments[key] += weight * label_delta[label]
        except Exception:
            return {}

        for key in list(adjustments.keys()):
            adjustments[key] = max(-1.0, min(1.0, adjustments[key]))

        return dict(adjustments)

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
            rescored.append((doc, max(0.0, score + delta)))

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
            metadata_text = " ".join(
                str(candidates[idx].get("metadata", {}).get(field, ""))
                for field in [
                    "file_name",
                    "project_name",
                    "enriched_project_name",
                    "enriched_project_number",
                    "plan_sheet_title",
                    "plan_sheet_type",
                    "enriched_engineer_on_record",
                    "structural_elements",
                    "detail_intents",
                    "detail_types",
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
                })
                
                # Track best (lowest) score
                if score < projects[file_name]['best_score']:
                    projects[file_name]['best_score'] = score
        
        # Convert sets to sorted lists
        for file_name in projects:
            projects[file_name]['pages'] = sorted(list(projects[file_name]['pages']))
        
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

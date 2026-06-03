"""Semantic query understanding engine for smarter search analysis."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import re


class SemanticQueryAnalyzer:
    """Understands query intent and maps terminology across projects."""

    # Structural component synonyms - maps variations to canonical form
    COMPONENT_SYNONYMS = {
        # Pile types
        "cidh": {"drilled shaft", "cast-in-drilled-hole", "caisson", "bored pile", "shaft", "deep foundation"},
        "steel pile": {"h-pile", "hp pile", "h pile", "steel h-pile", "driven pile", "hpile"},
        "driven pile": {"steel pile", "h-pile", "hp pile", "sheet pile", "hammered pile"},
        "abutment": {"abutment wall", "end support", "bridge abutment", "approach wall"},
        "bent": {"pier", "support", "bent cap", "column bent", "bent frame", "support pier"},
        "girder": {"beam", "box girder", "steel girder", "plate girder", "i-girder"},
        "deck": {"slab", "deck slab", "wearing surface", "bridge deck", "concrete deck"},
        "wingwall": {"wing wall", "retaining wall", "bridge wingwall", "side wall"},
        
        # Connection types
        "pin connection": {"pinned", "hinge connection", "pin detail", "pin joint"},
        "shear key": {"shear lock", "shear device", "dowel", "friction lock"},
        "expansion joint": {"expansion", "joint detail", "movement joint", "expansion bearing"},
        
        # Detail types
        "reinforcement": {"rebar", "reinforcing", "steel reinforcement", "reinforcing steel", "rebar detail", "steel layout"},
        "detail": {"drawing detail", "construction detail", "sheet detail", "callout", "schedule"},
        "section": {"cross section", "section view", "slice", "profile"},
    }

    # Query intent patterns - what is the user really asking for?
    INTENT_PATTERNS = {
        "locate": re.compile(r"\b(location|where|find|identify|locate)\b", re.IGNORECASE),
        "dimension": re.compile(r"\b(size|length|width|height|depth|diameter|dimension|thickness|spacing)\b", re.IGNORECASE),
        "material": re.compile(r"\b(material|concrete|steel|type|strength|grade|fc|fy)\b", re.IGNORECASE),
        "detail": re.compile(r"\b(detail|drawing|section|plan|elevation|view|cross section)\b", re.IGNORECASE),
        "connection": re.compile(r"\b(connection|joint|interface|support|bearing|anchor)\b", re.IGNORECASE),
        "reinforcement": re.compile(r"\b(reinforcement|rebar|steel|reinforcing|layout|bars)\b", re.IGNORECASE),
    }


class AnalysisEngine:
    """Semantic analysis of search results for smarter query understanding."""

    def __init__(self):
        self.query_analyzer = SemanticQueryAnalyzer()

    def build_analysis(
        self,
        query: str,
        normalized_results: List[Dict],
        search_summary: Optional[str] = None,
    ) -> Dict:
        """Analyze query intent and provide semantic evidence for results."""
        
        inferred_intent = self._infer_intent(query)
        terminology_variants = self._find_terminology_variants(query)
        semantic_evidence = self._build_semantic_evidence(query, normalized_results)
        confidence = self._estimate_confidence(inferred_intent, semantic_evidence)
        answer = self._build_answer(query, inferred_intent, semantic_evidence)

        return {
            "answer": answer,
            "inferred_intent": inferred_intent,
            "intent_confidence": confidence,
            "semantic_evidence": semantic_evidence,
            "terminology_variants": terminology_variants,
            "search_summary": search_summary,
        }

    def _infer_intent(self, query: str) -> Dict[str, any]:
        """Extract what the user is really asking for."""
        query_lower = query.lower()
        intent_types = {}

        for intent_type, pattern in SemanticQueryAnalyzer.INTENT_PATTERNS.items():
            if pattern.search(query_lower):
                intent_types[intent_type] = True

        # Identify structural components
        components = []
        for canonical, synonyms in SemanticQueryAnalyzer.COMPONENT_SYNONYMS.items():
            if canonical.lower() in query_lower:
                components.append(canonical)
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    components.append(canonical)
                    break

        return {
            "query": query,
            "intent_types": intent_types,
            "structural_components": list(set(components)),
            "normalized": self._normalize_query_text(query),
        }

    def _normalize_query_text(self, text: str) -> str:
        """Normalize query for comparison."""
        return re.sub(r'\s+', ' ', text.lower().strip())

    def _find_terminology_variants(self, query: str) -> Dict[str, List[str]]:
        """Find related terminology that might match the query concept."""
        variants = {}
        query_lower = query.lower()

        for canonical, synonyms in SemanticQueryAnalyzer.COMPONENT_SYNONYMS.items():
            for term in [canonical] + list(synonyms):
                if term.lower() in query_lower:
                    if canonical not in variants:
                        variants[canonical] = []
                    # Add all related terms
                    related = list(synonyms) + [canonical]
                    variants[canonical] = sorted(set(related))
                    break

        return variants

    def _build_semantic_evidence(
        self, query: str, normalized_results: List[Dict]
    ) -> List[Dict]:
        """Build semantic evidence showing why each result matches the query."""
        query_lower = query.lower()
        evidence_list = []

        for item in normalized_results[:15]:  # Top 15 results
            snippet = str(item.get("content_sample", "")).strip()
            if not snippet:
                continue

            # Score how well this result matches the query intent
            semantic_score = self._compute_semantic_match(query_lower, snippet.lower())
            component_matches = self._find_component_matches(query_lower, snippet.lower())

            evidence_list.append({
                "pdf_file_name": item.get("pdf_file_name", ""),
                "project_name": item.get("project_name", ""),
                "page_number": int(item.get("page_number", 0) or 0),
                "relevance_score": float(item.get("relevance_score", 0.0) or 0.0),
                "semantic_match_score": semantic_score,
                "component_matches": component_matches,
                "snippet": snippet[:500],
                "why_matched": self._explain_match(query_lower, snippet.lower(), component_matches),
            })

        return sorted(
            evidence_list,
            key=lambda x: x["semantic_match_score"],
            reverse=True,
        )

    def _compute_semantic_match(self, query_lower: str, snippet_lower: str) -> float:
        """Score how well the snippet semantically matches the query."""
        # Check for exact phrase match
        exact_match = 1.0 if query_lower in snippet_lower else 0.0

        # Check for term overlap
        query_terms = set(query_lower.split())
        snippet_terms = set(snippet_lower.split())
        overlap = len(query_terms & snippet_terms)
        term_match = min(1.0, overlap / max(len(query_terms), 1))

        # Favor higher relevance score
        score = 0.5 * exact_match + 0.5 * term_match
        return round(score, 3)

    def _find_component_matches(self, query_lower: str, snippet_lower: str) -> List[str]:
        """Find which structural components are mentioned in both query and snippet."""
        matches = []
        for canonical, synonyms in SemanticQueryAnalyzer.COMPONENT_SYNONYMS.items():
            # Check if any form of this component is in the query
            query_has = False
            for term in [canonical] + list(synonyms):
                if term.lower() in query_lower:
                    query_has = True
                    break

            # Check if any form is in the snippet
            if query_has:
                snippet_has = False
                for term in [canonical] + list(synonyms):
                    if term.lower() in snippet_lower:
                        snippet_has = True
                        break
                if snippet_has:
                    matches.append(canonical)

        return list(set(matches))

    def _explain_match(
        self, query_lower: str, snippet_lower: str, component_matches: List[str]
    ) -> str:
        """Explain in plain text why this result matches the query."""
        reasons = []

        if query_lower in snippet_lower:
            reasons.append("Exact phrase match")

        if component_matches:
            reasons.append(
                f"Structural components match: {', '.join(component_matches)}"
            )

        # Check for intent pattern matches
        for intent_type, pattern in SemanticQueryAnalyzer.INTENT_PATTERNS.items():
            if pattern.search(query_lower) and pattern.search(snippet_lower):
                reasons.append(f"Both mention {intent_type}")

        if not reasons:
            reasons.append("Query terms found in snippet")

        return " | ".join(reasons) if reasons else "Keyword match"

    def _estimate_confidence(self, inferred_intent: Dict, semantic_evidence: List[Dict]) -> float:
        """Estimate confidence that we understood the query correctly."""
        if not semantic_evidence:
            return 0.1

        # Confidence based on:
        # 1. How clear the intent is (number of intent types detected)
        intent_clarity = min(
            1.0,
            len(inferred_intent.get("intent_types", {})) / 3.0,
        )

        # 2. How well the top results match
        top_match = semantic_evidence[0].get("semantic_match_score", 0.0)

        # 3. How many results have component matches
        with_components = sum(
            1 for e in semantic_evidence if e.get("component_matches")
        )
        component_factor = min(1.0, with_components / max(len(semantic_evidence), 1))

        confidence = 0.4 * intent_clarity + 0.3 * top_match + 0.3 * component_factor
        return round(min(0.95, max(0.1, confidence)), 3)

    def _build_answer(
        self, query: str, inferred_intent: Dict, semantic_evidence: List[Dict]
    ) -> str:
        """Build an explanation of what the query is asking for."""
        if not semantic_evidence:
            return (
                "I could not find relevant results for this query. "
                "Try using more specific structural terms or adjusting your search scope."
            )

        intent_types = inferred_intent.get("intent_types", {})
        components = inferred_intent.get("structural_components", [])
        top_result = semantic_evidence[0] if semantic_evidence else {}

        answer_parts = [f"Query: {query}"]

        # Explain what we think they're asking for
        if intent_types or components:
            asking_parts = []
            if intent_types:
                asking_parts.append(
                    f"You're asking for {', '.join(intent_types.keys())}"
                )
            if components:
                asking_parts.append(f"related to {', '.join(components)}")
            if asking_parts:
                answer_parts.append(". ".join(asking_parts) + ".")

        # Show where we found it
        if top_result:
            answer_parts.append(
                f"Best match: {top_result.get('pdf_file_name')} "
                f"(Project: {top_result.get('project_name')}, Page {top_result.get('page_number')})"
            )
            answer_parts.append(f"Why matched: {top_result.get('why_matched')}")

        # Count relevant results
        if semantic_evidence:
            answer_parts.append(
                f"Found {len(semantic_evidence)} result(s) across "
                f"{len(set(e.get('pdf_file_name') for e in semantic_evidence))} document(s)."
            )

        return " ".join(answer_parts)

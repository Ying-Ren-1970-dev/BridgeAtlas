"""
Engineering terminology and synonym mapping for structural engineering.
Based on training test results and common industry terminology.
"""

from typing import List, Dict
import json
import os
import re


class EngineeringTerminology:
    """Manages engineering terminology, synonyms, and query expansion."""

    LEARNED_TERMS_FILE = "learned_terminology.json"
    _learned_terms: Dict[str, List[str]] = {}
    _auto_discovery_done = False
    
    # Foundation Systems
    FOUNDATION_TERMS = {
        "CIDH pile": [
            "drilled shaft", "cast-in-drilled-hole pile", "cast in drilled hole",
            "type I shaft", "type II shaft", "CIDH shaft", "CIDH foundation",
            "24\" CIDH", "72\" CIDH", "30\" CIDH"
        ],
        "driven pile": [
            "pile foundation", "driven foundation"
        ],
        "pile cap": [
            "cap", "pile footing"
        ],
        "footing": [
            "foundation", "spread footing", "isolated footing"
        ],
    }
    
    # Structural Elements
    STRUCTURAL_ELEMENTS = {
        "abutment": [
            "end support", "abutment structure", "bridge abutment"
        ],
        "bent": [
            "pier", "support bent", "column bent", "intermediate support"
        ],
        "column": [
            "pier column", "support column", "vertical support"
        ],
        "girder": [
            "beam", "main beam", "bridge girder", "box girder"
        ],
        "R/C box girder": [
            "box girder", "reinforced concrete box girder", "concrete box beam"
        ],
        "cap beam": [
            "cap", "bent cap", "pier cap"
        ],
        "diaphragm": [
            "end diaphragm", "cross diaphragm"
        ],
    }
    
    # Connection & Detail Elements
    CONNECTION_TERMS = {
        "pipe pin": [
            "steel shear key", "shear key", "pin connection", "shear connector"
        ],
        "bearing pad": [
            "elastomeric bearing pad", "bearing support", "elastomeric bearing",
            "bearing", "neoprene bearing"
        ],
        "expansion joint": [
            "joint", "deck joint", "bridge joint"
        ],
        "shear key": [
            "steel shear key", "pipe pin", "lateral restraint"
        ],
    }
    
    # Materials & Reinforcement
    MATERIAL_TERMS = {
        "rebar": [
            "reinforcing bars", "reinforcement", "reinforcing steel",
            "steel reinforcement", "#4 bars", "#5 bars", "#6 bars",
            "bar", "bars"
        ],
        "concrete": [
            "reinforced concrete", "R/C", "cast-in-place concrete", "CIP concrete"
        ],
        "steel": [
            "structural steel", "steel plate", "steel section"
        ],
    }
    
    # Retaining & Earth Structures
    RETAINING_TERMS = {
        "retaining wall": [
            "wall", "gravity wall", "cantilever wall", "MSE wall"
        ],
        "ground anchor": [
            "soil anchor", "tie-back", "SHGA", "anchor"
        ],
        "embedment": [
            "embedded depth", "burial depth"
        ],
    }
    
    # Railings & Safety Features
    RAILING_TERMS = {
        "railing": [
            "guard rail", "barrier", "safety railing", "pedestrian railing"
        ],
        "chain link fence": [
            "fence", "security fence", "perimeter fence"
        ],
        "tamper proof screws": [
            "security screws", "tamper-proof self-tapping stainless steel security screws",
            "stainless steel security screws"
        ],
    }
    
    # Construction & Process Terms
    CONSTRUCTION_TERMS = {
        "cast-in-place": [
            "CIP", "poured in place", "site cast"
        ],
        "precast": [
            "pre-cast", "prefabricated"
        ],
    }
    
    # Built-in terminology dictionaries
    ALL_TERMS = {
        **FOUNDATION_TERMS,
        **STRUCTURAL_ELEMENTS,
        **CONNECTION_TERMS,
        **MATERIAL_TERMS,
        **RETAINING_TERMS,
        **RAILING_TERMS,
        **CONSTRUCTION_TERMS,
    }

    @classmethod
    def _learned_terms_path(cls) -> str:
        return os.path.join(os.path.dirname(__file__), cls.LEARNED_TERMS_FILE)

    @classmethod
    def _load_learned_terms(cls):
        path = cls._learned_terms_path()
        if not os.path.exists(path):
            cls._learned_terms = {}
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cls._learned_terms = {
                    str(k): [str(v) for v in values if str(v).strip()]
                    for k, values in data.items()
                    if isinstance(values, list)
                }
            else:
                cls._learned_terms = {}
        except Exception:
            cls._learned_terms = {}

    @classmethod
    def _save_learned_terms(cls):
        path = cls._learned_terms_path()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cls._learned_terms, f, indent=2)
        except Exception:
            # Learning persistence should never break search flow.
            pass

    @classmethod
    def _extract_strings(cls, payload) -> List[str]:
        strings: List[str] = []
        if isinstance(payload, str):
            strings.append(payload)
        elif isinstance(payload, dict):
            for value in payload.values():
                strings.extend(cls._extract_strings(value))
        elif isinstance(payload, list):
            for item in payload:
                strings.extend(cls._extract_strings(item))
        return strings

    @classmethod
    def _normalize_phrase(cls, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip()).lower()

    @classmethod
    def _learn_mapping(cls, primary_term: str, synonym: str):
        primary = primary_term.strip()
        syn = synonym.strip()
        if not primary or not syn:
            return
        if primary.lower() == syn.lower():
            return

        existing = cls._learned_terms.setdefault(primary, [])
        if syn.lower() not in {x.lower() for x in existing}:
            existing.append(syn)

    @classmethod
    def _auto_discover_from_data(cls):
        if cls._auto_discovery_done:
            return

        data_dir = os.path.join(os.path.dirname(__file__), "data")
        if not os.path.isdir(data_dir):
            cls._auto_discovery_done = True
            return

        files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().startswith("enriched_") and f.lower().endswith(".json")
        ]

        # Limit startup cost while still sampling across corpus.
        for file_path in files[:120]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                strings = cls._extract_strings(data)
            except Exception:
                continue

            for s in strings:
                text = s.strip()
                if not text or len(text) > 250:
                    continue

                # Pattern: Long Phrase (ACRONYM)
                for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9/&\-\s]{4,80})\s*\(([A-Z]{2,8})\)\b", text):
                    phrase = cls._normalize_phrase(m.group(1))
                    acronym = m.group(2).upper()
                    cls._learn_mapping(acronym, phrase)

                # Pattern: ACRONYM (Long Phrase)
                for m in re.finditer(r"\b([A-Z]{2,8})\s*\(([A-Za-z][A-Za-z0-9/&\-\s]{4,80})\)", text):
                    acronym = m.group(1).upper()
                    phrase = cls._normalize_phrase(m.group(2))
                    cls._learn_mapping(acronym, phrase)

        cls._auto_discovery_done = True
        cls._save_learned_terms()

    @classmethod
    def _ensure_ready(cls):
        if not cls._learned_terms:
            cls._load_learned_terms()
        if not cls._auto_discovery_done:
            cls._auto_discover_from_data()

    @classmethod
    def _combined_terms(cls) -> Dict[str, List[str]]:
        cls._ensure_ready()
        merged: Dict[str, List[str]] = {k: list(v) for k, v in cls.ALL_TERMS.items()}
        for primary, synonyms in cls._learned_terms.items():
            if primary not in merged:
                merged[primary] = []
            existing = {s.lower() for s in merged[primary]}
            for s in synonyms:
                if s.lower() not in existing:
                    merged[primary].append(s)
                    existing.add(s.lower())
        return merged
    
    @classmethod
    def expand_query(cls, query: str, max_expansions: int = 3) -> List[str]:
        """
        Expand a query with relevant synonyms.
        
        Args:
            query: Original search query
            max_expansions: Maximum number of synonym expansions per term
            
        Returns:
            List of expanded query terms including original
        """
        query_lower = query.lower().strip()
        expanded_terms = [query]  # Always include original
        
        all_terms = cls._combined_terms()

        # Check if query matches any primary term or synonym
        for primary_term, synonyms in all_terms.items():
            primary_lower = primary_term.lower()
            
            # If query matches primary term, add synonyms
            if primary_lower in query_lower or query_lower in primary_lower:
                expanded_terms.extend(synonyms[:max_expansions])
                break
            
            # If query matches a synonym, add primary and other synonyms
            for synonym in synonyms:
                if synonym.lower() in query_lower or query_lower in synonym.lower():
                    expanded_terms.append(primary_term)
                    # Add other synonyms (not the matched one)
                    other_synonyms = [s for s in synonyms if s != synonym]
                    expanded_terms.extend(other_synonyms[:max_expansions-1])
                    break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for term in expanded_terms:
            term_lower = term.lower()
            if term_lower not in seen:
                seen.add(term_lower)
                unique_expanded.append(term)
        
        return unique_expanded
    
    @classmethod
    def get_primary_term(cls, term: str) -> str:
        """
        Get the primary (canonical) term for a given term or synonym.
        
        Args:
            term: Term to look up
            
        Returns:
            Primary term if found, otherwise original term
        """
        term_lower = term.lower().strip()
        
        all_terms = cls._combined_terms()

        # Check if it's already a primary term
        for primary_term in all_terms.keys():
            if primary_term.lower() == term_lower:
                return primary_term
        
        # Check if it's a synonym
        for primary_term, synonyms in all_terms.items():
            for synonym in synonyms:
                if synonym.lower() == term_lower:
                    return primary_term
        
        # Not found, return original
        return term
    
    @classmethod
    def get_all_synonyms(cls, term: str) -> List[str]:
        """
        Get all synonyms (including primary term) for a given term.
        
        Args:
            term: Term to look up
            
        Returns:
            List of all related terms including primary
        """
        primary = cls.get_primary_term(term)
        
        all_terms = cls._combined_terms()
        if primary in all_terms:
            return [primary] + all_terms[primary]
        
        return [term]
    
    @classmethod
    def add_custom_terms(cls, custom_terms: Dict[str, List[str]]):
        """
        Add custom terminology (e.g., from Caltrans standards or project-specific terms).
        
        Args:
            custom_terms: Dictionary of {primary_term: [synonyms]}
        """
        cls._ensure_ready()
        for primary, synonyms in custom_terms.items():
            for synonym in synonyms:
                cls._learn_mapping(primary, synonym)
        cls._save_learned_terms()

    @classmethod
    def add_feedback_mapping(cls, acronym: str, expansion: str):
        """Persist a user-confirmed acronym expansion for future searches."""
        cls._ensure_ready()
        cleaned_acronym = acronym.strip().upper()
        cleaned_expansion = cls._normalize_phrase(expansion)
        cls._learn_mapping(cleaned_acronym, cleaned_expansion)
        cls._save_learned_terms()
    
    @classmethod
    def get_term_categories(cls) -> Dict[str, List[str]]:
        """Get all terms organized by category."""
        return {
            "Foundations": list(cls.FOUNDATION_TERMS.keys()),
            "Structural Elements": list(cls.STRUCTURAL_ELEMENTS.keys()),
            "Connections": list(cls.CONNECTION_TERMS.keys()),
            "Materials": list(cls.MATERIAL_TERMS.keys()),
            "Retaining Structures": list(cls.RETAINING_TERMS.keys()),
            "Railings & Safety": list(cls.RAILING_TERMS.keys()),
            "Construction": list(cls.CONSTRUCTION_TERMS.keys()),
        }


# Convenience functions
def expand_query(query: str, max_expansions: int = 3) -> List[str]:
    """Expand query with engineering synonyms."""
    return EngineeringTerminology.expand_query(query, max_expansions)


def get_primary_term(term: str) -> str:
    """Get canonical term for a synonym."""
    return EngineeringTerminology.get_primary_term(term)


def get_all_synonyms(term: str) -> List[str]:
    """Get all synonyms for a term."""
    return EngineeringTerminology.get_all_synonyms(term)

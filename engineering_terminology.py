"""
Engineering terminology and synonym mapping for structural engineering.
Based on training test results and common industry terminology.
"""

from typing import List, Set, Dict


class EngineeringTerminology:
    """Manages engineering terminology, synonyms, and query expansion."""
    
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
    
    # Combine all terminology dictionaries
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
        
        # Check if query matches any primary term or synonym
        for primary_term, synonyms in cls.ALL_TERMS.items():
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
        
        # Check if it's already a primary term
        for primary_term in cls.ALL_TERMS.keys():
            if primary_term.lower() == term_lower:
                return primary_term
        
        # Check if it's a synonym
        for primary_term, synonyms in cls.ALL_TERMS.items():
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
        
        if primary in cls.ALL_TERMS:
            return [primary] + cls.ALL_TERMS[primary]
        
        return [term]
    
    @classmethod
    def add_custom_terms(cls, custom_terms: Dict[str, List[str]]):
        """
        Add custom terminology (e.g., from Caltrans standards or project-specific terms).
        
        Args:
            custom_terms: Dictionary of {primary_term: [synonyms]}
        """
        cls.ALL_TERMS.update(custom_terms)
    
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

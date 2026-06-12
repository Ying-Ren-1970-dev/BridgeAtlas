"""Loader for enriched metadata from JSON files."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import config
from cad_drawing_knowledge import CadDrawingKnowledge


class EnrichedMetadataLoader:
    """Loads and manages enriched metadata from JSON files."""

    @staticmethod
    def _to_list(value) -> List[str]:
        """Normalize possibly-string metadata fields into a list of non-empty strings."""
        if not value:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            # Support comma-separated values occasionally emitted by upstream extraction.
            return [v.strip() for v in value.split(',') if v.strip()]
        return [str(value).strip()]

    @staticmethod
    def classify_sheet_type(
        sheet_title: str,
        detail_types: List[str],
        primary_type: str = "",
    ) -> str:
        """Classify sheet intent for coarse routing in retrieval and filtering."""
        text = f"{primary_type} {sheet_title} {' '.join(detail_types)}".lower()

        if "general plan" in text:
            return "general_plan"
        if any(tok in text for tok in ["general note", "general notes", "notes"]):
            return "general_note"
        if any(tok in text for tok in ["bar", "rebar", "reinforcement", "reinf"]):
            return "rebar"
        if "fence" in text:
            return "fence"
        if any(tok in text for tok in ["detail", "details", "section", "elevation"]):
            return "details"
        if any(tok in text for tok in ["plan", "layout", "overall"]):
            return "structure_plan"
        return "other"

    @staticmethod
    def derive_detail_intents(
        structural_elements: List[str],
        detail_types: List[str],
        sheet_title: str,
        primary_type: str = "",
    ) -> List[str]:
        """Generate intent-like labels to help map user queries to relevant detail sheets."""
        intents = set()
        title_lower = (sheet_title or "").lower()
        primary_lower = (primary_type or "").lower()

        normalized_elements = [e.lower() for e in structural_elements]
        normalized_details = [d.lower() for d in detail_types]

        # Sheet-level intent tags
        sheet_type = EnrichedMetadataLoader.classify_sheet_type(
            sheet_title, detail_types, primary_type=primary_type
        )
        intents.add(f"sheet_type:{sheet_type}")

        if "general plan" in primary_lower or "general plan" in title_lower:
            intents.add("general plan")

        if any("general" in d and "note" in d for d in normalized_details) or "general note" in title_lower:
            intents.add("general project information")

        combined_text = f"{title_lower} {primary_lower} {' '.join(normalized_details)}"
        if "ars curve" in combined_text or ("ars" in combined_text and "curve" in combined_text):
            intents.add("ars curve")

        # Detail-level intents based on element + drawing type.
        for element in normalized_elements:
            clean_element = element.strip()
            if not clean_element:
                continue
            intents.add(f"detail of {clean_element}")
            if any(k in " ".join(normalized_details) for k in ["rebar", "reinforcement", "bar"]):
                intents.add(f"reinforcement detail of {clean_element}")

        if any("layout" in d or "plan" in d for d in normalized_details):
            intents.add("layout/detail plan information")

        return sorted(intents)
    
    @staticmethod
    def load_enriched_metadata(file_name: str) -> Optional[Dict]:
        """
        Load enriched metadata for a given PDF file.
        
        Args:
            file_name: Name of the PDF file (e.g., "Mar Vista POC.pdf")
            
        Returns:
            Dictionary mapping page numbers to enriched metadata, or None if not found
        """
        # Construct the enriched JSON file path
        pdf_stem = Path(file_name).stem
        enriched_file = config.DATA_FOLDER / f"enriched_{pdf_stem}.json"
        
        if not enriched_file.exists():
            return None
        
        try:
            with open(enriched_file, 'r', encoding='utf-8') as f:
                enriched_data = json.load(f)
            
            # Convert list to dictionary keyed by page number for easy lookup
            metadata_by_page = {}
            for page_data in enriched_data:
                page_num = page_data.get('page_number')
                if page_num:
                    metadata_by_page[page_num] = page_data
            
            return metadata_by_page
        
        except Exception as e:
            print(f"Warning: Could not load enriched metadata for {file_name}: {str(e)}")
            return None
    
    @staticmethod
    def extract_searchable_text(page_metadata: Dict) -> str:
        """
        Extract searchable text from enriched page metadata.
        
        Args:
            page_metadata: Enriched metadata for a single page
            
        Returns:
            Formatted text string containing key metadata fields
        """
        searchable_parts = []
        
        # Page type
        page_type = page_metadata.get('page_type', '')
        if page_type:
            searchable_parts.append(f"Page Type: {page_type}")
        
        title_block = page_metadata.get('title_block', {})
        if not title_block:
            return ' '.join(searchable_parts)
        
        # Project information
        proj_info = title_block.get('project_info', {})
        if proj_info:
            if proj_info.get('project_name'):
                searchable_parts.append(f"Project: {proj_info['project_name']}")
            if proj_info.get('client'):
                searchable_parts.append(f"Client: {proj_info['client']}")
            if proj_info.get('location'):
                searchable_parts.append(f"Location: {proj_info['location']}")
            if proj_info.get('engineer_on_record'):
                searchable_parts.append(f"Engineer: {proj_info['engineer_on_record']}")
            if proj_info.get('engineering_firm'):
                searchable_parts.append(f"Firm: {proj_info['engineering_firm']}")
            if proj_info.get('project_year'):
                searchable_parts.append(f"Year: {proj_info['project_year']}")
        
        # Plan type
        plan_type = title_block.get('plan_type', {})
        sheet_title = ""
        primary_type = ""
        detail_types = []
        if plan_type:
            if plan_type.get('primary_type'):
                primary_type = str(plan_type['primary_type'])
                searchable_parts.append(f"Plan Type: {primary_type}")
            if plan_type.get('sheet_title'):
                sheet_title = str(plan_type['sheet_title'])
                searchable_parts.append(f"Sheet Title: {sheet_title}")
            if plan_type.get('sheet_number'):
                searchable_parts.append(f"Sheet Number: {plan_type['sheet_number']}")
            if plan_type.get('scale'):
                searchable_parts.append(f"Scale: {plan_type['scale']}")
        
        # Plan contents
        contents = title_block.get('plan_contents', {})
        if contents:
            if contents.get('structural_elements'):
                elements = ', '.join(contents['structural_elements'])
                searchable_parts.append(f"Structural Elements: {elements}")
            
            if contents.get('detail_types'):
                detail_types = EnrichedMetadataLoader._to_list(contents['detail_types'])
                details = ', '.join(detail_types)
                searchable_parts.append(f"Detail Types: {details}")
            
            if contents.get('grid_references'):
                grids = ', '.join(contents['grid_references'])
                searchable_parts.append(f"Grid References: {grids}")
            
            if contents.get('has_tables'):
                searchable_parts.append("Contains Tables")
            
            if contents.get('has_diagrams'):
                searchable_parts.append("Contains Diagrams")

            structural_elements = EnrichedMetadataLoader._to_list(contents.get('structural_elements'))
            intents = EnrichedMetadataLoader.derive_detail_intents(
                structural_elements, detail_types, sheet_title, primary_type=primary_type
            )
            if intents:
                searchable_parts.append(f"Detail Intents: {', '.join(intents)}")

            sheet_type = EnrichedMetadataLoader.classify_sheet_type(
                sheet_title, detail_types, primary_type=primary_type
            )
            searchable_parts.append(f"Sheet Type: {sheet_type}")

        cad_fields = CadDrawingKnowledge.build_page_context(page_metadata)
        if cad_fields.get("cad_sheet_views"):
            searchable_parts.append(f"CAD Sheet Views: {cad_fields['cad_sheet_views']}")
        if cad_fields.get("cad_sheet_group"):
            group_bits = [cad_fields["cad_sheet_group"]]
            if cad_fields.get("cad_sheet_group_role"):
                group_bits.append(cad_fields["cad_sheet_group_role"])
            if cad_fields.get("cad_sheet_group_index"):
                group_bits.append(f"sheet {cad_fields['cad_sheet_group_index']}")
            searchable_parts.append(f"CAD Sheet Group: {' '.join(group_bits)}")
        if cad_fields.get("cad_cross_reference_summary"):
            searchable_parts.append(cad_fields["cad_cross_reference_summary"])
        
        return ' | '.join(searchable_parts)
    
    @staticmethod
    def get_metadata_fields(page_metadata: Dict) -> Dict:
        """
        Extract metadata fields for vector store filtering.
        
        Args:
            page_metadata: Enriched metadata for a single page
            
        Returns:
            Dictionary of metadata fields
        """
        fields = {
            'page_type': page_metadata.get('page_type', ''),
            'confidence': page_metadata.get('confidence', 0.0),
        }
        
        title_block = page_metadata.get('title_block', {})
        if not title_block:
            return fields
        
        # Flatten project info
        proj_info = title_block.get('project_info', {})
        for key, value in proj_info.items():
            if value:
                fields[f'enriched_{key}'] = str(value)
        
        # Flatten plan type
        plan_type = title_block.get('plan_type', {})
        sheet_title = ""
        primary_type = ""
        for key, value in plan_type.items():
            if value:
                fields[f'plan_{key}'] = str(value)
                if key == 'sheet_title':
                    sheet_title = str(value)
                if key == 'primary_type':
                    primary_type = str(value)
        
        # Flatten plan contents
        contents = title_block.get('plan_contents', {})
        detail_types = []
        structural_elements = []
        if contents:
            if contents.get('structural_elements'):
                structural_elements = EnrichedMetadataLoader._to_list(contents['structural_elements'])
                fields['structural_elements'] = ', '.join(structural_elements)
            if contents.get('detail_types'):
                detail_types = EnrichedMetadataLoader._to_list(contents['detail_types'])
                fields['detail_types'] = ', '.join(detail_types)
            if contents.get('grid_references'):
                fields['grid_references'] = ', '.join(contents['grid_references'])
            fields['has_tables'] = str(contents.get('has_tables', False))
            fields['has_diagrams'] = str(contents.get('has_diagrams', False))

        detail_intents = EnrichedMetadataLoader.derive_detail_intents(
            structural_elements, detail_types, sheet_title, primary_type=primary_type
        )
        if detail_intents:
            fields['detail_intents'] = ', '.join(detail_intents)

        fields['plan_sheet_type'] = EnrichedMetadataLoader.classify_sheet_type(
            sheet_title, detail_types, primary_type=primary_type
        )

        fields.update(CadDrawingKnowledge.build_page_context(page_metadata))
        
        return fields

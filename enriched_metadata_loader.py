"""Loader for enriched metadata from JSON files."""
import json
from pathlib import Path
from typing import Dict, List, Optional
import config


class EnrichedMetadataLoader:
    """Loads and manages enriched metadata from JSON files."""
    
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
        if plan_type:
            if plan_type.get('primary_type'):
                searchable_parts.append(f"Plan Type: {plan_type['primary_type']}")
            if plan_type.get('sheet_title'):
                searchable_parts.append(f"Sheet Title: {plan_type['sheet_title']}")
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
                details = ', '.join(contents['detail_types'])
                searchable_parts.append(f"Detail Types: {details}")
            
            if contents.get('grid_references'):
                grids = ', '.join(contents['grid_references'])
                searchable_parts.append(f"Grid References: {grids}")
            
            if contents.get('has_tables'):
                searchable_parts.append("Contains Tables")
            
            if contents.get('has_diagrams'):
                searchable_parts.append("Contains Diagrams")
        
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
        for key, value in plan_type.items():
            if value:
                fields[f'plan_{key}'] = str(value)
        
        # Flatten plan contents
        contents = title_block.get('plan_contents', {})
        if contents:
            if contents.get('structural_elements'):
                fields['structural_elements'] = ', '.join(contents['structural_elements'])
            if contents.get('detail_types'):
                fields['detail_types'] = ', '.join(contents['detail_types'])
            if contents.get('grid_references'):
                fields['grid_references'] = ', '.join(contents['grid_references'])
            fields['has_tables'] = str(contents.get('has_tables', False))
            fields['has_diagrams'] = str(contents.get('has_diagrams', False))
        
        return fields

"""Metadata management module for storing and retrieving project information."""
import json
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

import config


class MetadataManager:
    """Manages project metadata storage and retrieval."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the metadata manager."""
        self.db_path = db_path or config.METADATA_DB_PATH
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                return {'projects': {}, 'last_updated': None}
        return {'projects': {}, 'last_updated': None}
    
    def _save_metadata(self):
        """Save metadata to JSON file."""
        try:
            self.metadata['last_updated'] = datetime.now().isoformat()
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metadata: {str(e)}")
    
    def add_project_metadata(
        self,
        file_name: str,
        file_path: str,
        metadata: Dict,
        categories: List[str],
        total_pages: int
    ):
        """
        Add or update project metadata.
        
        Args:
            file_name: Name of the PDF file
            file_path: Full path to the PDF file
            metadata: Extracted metadata dictionary
            categories: List of project categories
            total_pages: Total number of pages in the PDF
        """
        project_data = {
            'file_name': file_name,
            'file_path': file_path,
            'project_name': metadata.get('project_name', ''),
            'phase': metadata.get('phase', ''),
            'engineer_of_record': metadata.get('engineer_of_record', ''),
            'date': metadata.get('date', ''),
            'categories': categories,
            'total_pages': total_pages,
            'indexed_at': datetime.now().isoformat(),
        }
        
        self.metadata['projects'][file_name] = project_data
    
    def get_project_metadata(self, file_name: str) -> Optional[Dict]:
        """Get metadata for a specific project."""
        return self.metadata['projects'].get(file_name)
    
    def get_all_projects(self) -> Dict:
        """Get all project metadata."""
        return self.metadata['projects']

    def get_active_pdf_file_names(self) -> set:
        """PDF file names currently in the indexed project library."""
        return set(self.metadata['projects'].keys())
    
    def delete_project(self, file_name: str):
        """Delete metadata for a specific project."""
        if file_name in self.metadata['projects']:
            del self.metadata['projects'][file_name]
            print(f"Deleted metadata for: {file_name}")

    def delete_projects_by_path_filters(self, path_filters: List[str]) -> int:
        """Delete metadata entries whose file_path matches any excluded filter."""
        removed = 0
        to_remove = []
        for file_name, project in self.metadata['projects'].items():
            file_path = str(project.get('file_path', '')).replace('\\', '/').lower()
            if any(filter_text in file_path for filter_text in path_filters):
                to_remove.append(file_name)

        for file_name in to_remove:
            del self.metadata['projects'][file_name]
            removed += 1
            print(f"Deleted metadata for excluded project: {file_name}")

        return removed

    def search_projects_by_category(self, category: str) -> List[Dict]:
        """Search projects by category."""
        results = []
        for project in self.metadata['projects'].values():
            if category in project.get('categories', []):
                results.append(project)
        return results
    
    def search_projects_by_engineer(self, engineer: str) -> List[Dict]:
        """Search projects by engineer of record."""
        results = []
        engineer_lower = engineer.lower()
        for project in self.metadata['projects'].values():
            project_engineer = project.get('engineer_of_record', '').lower()
            if engineer_lower in project_engineer:
                results.append(project)
        return results
    
    def search_projects_by_phase(self, phase: str) -> List[Dict]:
        """Search projects by phase."""
        results = []
        phase_lower = phase.lower()
        for project in self.metadata['projects'].values():
            project_phase = project.get('phase', '').lower()
            if phase_lower in project_phase:
                results.append(project)
        return results
    
    def save(self):
        """Save current metadata to disk."""
        self._save_metadata()
    
    def get_stats(self) -> Dict:
        """Get statistics about the metadata database."""
        projects = self.metadata['projects']
        
        # Count by category
        category_counts = {}
        for project in projects.values():
            for category in project.get('categories', []):
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count by phase
        phase_counts = {}
        for project in projects.values():
            phase = project.get('phase') or 'Unknown'
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        return {
            'total_projects': len(projects),
            'category_distribution': category_counts,
            'phase_distribution': phase_counts,
            'last_updated': self.metadata.get('last_updated'),
        }

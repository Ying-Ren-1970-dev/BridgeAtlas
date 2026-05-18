"""Firestore adapter for metadata storage in production."""
import json
from datetime import datetime
from typing import Dict, List, Optional
import config


class FirestoreMetadataAdapter:
    """
    Metadata storage adapter that works with both Firestore and local JSON.
    Automatically uses Firestore in production, JSON file in development.
    """
    
    def __init__(self):
        """Initialize the metadata adapter based on environment configuration."""
        self.use_firestore = config.USE_FIRESTORE
        self.db = None
        self.collection_name = config.FIRESTORE_COLLECTION
        
        if self.use_firestore:
            try:
                from google.cloud import firestore
                self.db = firestore.Client(project=config.GCS_PROJECT_ID)
                print(f"✓ Firestore initialized: {self.collection_name}")
            except Exception as e:
                print(f"Warning: Could not initialize Firestore: {e}")
                print("Falling back to local JSON file")
                self.use_firestore = False
    
    def get_all_projects(self) -> Dict[str, Dict]:
        """
        Get metadata for all projects.
        
        Returns:
            Dictionary mapping project filenames to their metadata
        """
        if self.use_firestore:
            projects = {}
            docs = self.db.collection(self.collection_name).stream()
            for doc in docs:
                projects[doc.id] = doc.to_dict()
            return projects
        else:
            # Use local JSON file
            if not config.METADATA_DB_PATH.exists():
                return {}
            
            with open(config.METADATA_DB_PATH, 'r') as f:
                data = json.load(f)
            return data.get('projects', {})
    
    def get_project_metadata(self, filename: str) -> Optional[Dict]:
        """
        Get metadata for a specific project.
        
        Args:
            filename: PDF filename
            
        Returns:
            Project metadata dictionary or None if not found
        """
        if self.use_firestore:
            doc_ref = self.db.collection(self.collection_name).document(filename)
            doc = doc_ref.get()
            return doc.to_dict() if doc.exists else None
        else:
            # Use local JSON file
            if not config.METADATA_DB_PATH.exists():
                return None
            
            with open(config.METADATA_DB_PATH, 'r') as f:
                data = json.load(f)
            return data.get('projects', {}).get(filename)
    
    def save_project_metadata(self, filename: str, metadata: Dict):
        """
        Save metadata for a project.
        
        Args:
            filename: PDF filename
            metadata: Project metadata dictionary
        """
        if self.use_firestore:
            doc_ref = self.db.collection(self.collection_name).document(filename)
            doc_ref.set(metadata, merge=True)
        else:
            # Use local JSON file
            config.METADATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing data
            if config.METADATA_DB_PATH.exists():
                with open(config.METADATA_DB_PATH, 'r') as f:
                    data = json.load(f)
            else:
                data = {'projects': {}, 'last_updated': None}
            
            # Update project metadata
            data['projects'][filename] = metadata
            data['last_updated'] = datetime.now().isoformat()
            
            # Save back to file
            with open(config.METADATA_DB_PATH, 'w') as f:
                json.dump(data, f, indent=2)
    
    def delete_project_metadata(self, filename: str):
        """
        Delete metadata for a project.
        
        Args:
            filename: PDF filename
        """
        if self.use_firestore:
            doc_ref = self.db.collection(self.collection_name).document(filename)
            doc_ref.delete()
        else:
            # Use local JSON file
            if not config.METADATA_DB_PATH.exists():
                return
            
            with open(config.METADATA_DB_PATH, 'r') as f:
                data = json.load(f)
            
            if filename in data.get('projects', {}):
                del data['projects'][filename]
                data['last_updated'] = datetime.now().isoformat()
                
                with open(config.METADATA_DB_PATH, 'w') as f:
                    json.dump(data, f, indent=2)
    
    def project_exists(self, filename: str) -> bool:
        """
        Check if project metadata exists.
        
        Args:
            filename: PDF filename
            
        Returns:
            True if project exists, False otherwise
        """
        if self.use_firestore:
            doc_ref = self.db.collection(self.collection_name).document(filename)
            return doc_ref.get().exists
        else:
            if not config.METADATA_DB_PATH.exists():
                return False
            
            with open(config.METADATA_DB_PATH, 'r') as f:
                data = json.load(f)
            return filename in data.get('projects', {})
    
    def get_projects_count(self) -> int:
        """
        Get the total number of projects.
        
        Returns:
            Number of projects
        """
        if self.use_firestore:
            docs = self.db.collection(self.collection_name).stream()
            return sum(1 for _ in docs)
        else:
            if not config.METADATA_DB_PATH.exists():
                return 0
            
            with open(config.METADATA_DB_PATH, 'r') as f:
                data = json.load(f)
            return len(data.get('projects', {}))
    
    def search_projects(self, field: str, value: str) -> List[Dict]:
        """
        Search projects by a specific field.
        
        Args:
            field: Field name to search (e.g., 'project_name', 'categories')
            value: Value to search for
            
        Returns:
            List of matching project metadata
        """
        if self.use_firestore:
            query = self.db.collection(self.collection_name).where(field, '==', value)
            docs = query.stream()
            return [doc.to_dict() for doc in docs]
        else:
            projects = self.get_all_projects()
            results = []
            for project_data in projects.values():
                if field in project_data and project_data[field] == value:
                    results.append(project_data)
            return results
    
    def migrate_to_firestore(self):
        """
        Migrate local JSON metadata to Firestore.
        Use this to upload existing metadata when first deploying to production.
        """
        if not self.use_firestore:
            print("Firestore not enabled, skipping migration")
            return
        
        if not config.METADATA_DB_PATH.exists():
            print("No local metadata file found")
            return
        
        with open(config.METADATA_DB_PATH, 'r') as f:
            data = json.load(f)
        
        projects = data.get('projects', {})
        print(f"Migrating {len(projects)} projects to Firestore...")
        
        batch = self.db.batch()
        count = 0
        
        for filename, metadata in projects.items():
            doc_ref = self.db.collection(self.collection_name).document(filename)
            batch.set(doc_ref, metadata)
            count += 1
            
            # Commit in batches of 500 (Firestore limit)
            if count % 500 == 0:
                batch.commit()
                batch = self.db.batch()
                print(f"  Migrated {count} projects...")
        
        # Commit remaining
        if count % 500 != 0:
            batch.commit()
        
        print(f"✓ Successfully migrated {count} projects to Firestore")


# Global instance
firestore_metadata = FirestoreMetadataAdapter()

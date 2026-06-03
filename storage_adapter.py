"""Cloud Storage adapter for handling PDF files in GCS or local filesystem."""
import os
from pathlib import Path
from typing import BinaryIO, List, Optional
import tempfile
import config


class StorageAdapter:
    """
    Unified storage adapter that works with both local filesystem and Google Cloud Storage.
    Automatically uses GCS in production, local files in development.
    """
    
    def __init__(self):
        """Initialize the storage adapter based on environment configuration."""
        self.use_gcs = config.USE_CLOUD_STORAGE
        self.gcs_client = None
        self.bucket_pdfs = None
        self.bucket_vectors = None
        
        if self.use_gcs:
            try:
                from google.cloud import storage
                self.gcs_client = storage.Client(project=config.GCS_PROJECT_ID)
                self.bucket_pdfs = self.gcs_client.bucket(config.GCS_BUCKET_PDFS)
                self.bucket_vectors = self.gcs_client.bucket(config.GCS_BUCKET_VECTORS)
                print(f"✓ Cloud Storage initialized: {config.GCS_BUCKET_PDFS}")
            except Exception as e:
                print(f"Warning: Could not initialize Cloud Storage: {e}")
                print("Falling back to local filesystem")
                self.use_gcs = False
    
    def list_pdfs(self) -> List[str]:
        """
        List all PDF files in storage.
        
        Returns:
            List of PDF filenames
        """
        if self.use_gcs:
            blobs = self.bucket_pdfs.list_blobs()
            return [blob.name for blob in blobs if blob.name.endswith('.pdf')]
        else:
            pdf_folder = config.PROJECTS_FOLDER
            if not pdf_folder.exists():
                return []
            return [f.name for f in pdf_folder.glob('*.pdf')]
    
    def get_pdf_path(self, filename: str) -> str:
        """
        Get the path/URL for a PDF file.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Local path or GCS path
        """
        if self.use_gcs:
            return f"gs://{config.GCS_BUCKET_PDFS}/{filename}"
        else:
            return str(config.PROJECTS_FOLDER / filename)
    
    def read_pdf(self, filename: str) -> bytes:
        """
        Read a PDF file from storage.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            PDF file content as bytes
        """
        if self.use_gcs:
            blob = self.bucket_pdfs.blob(filename)
            return blob.download_as_bytes()
        else:
            file_path = config.PROJECTS_FOLDER / filename
            with open(file_path, 'rb') as f:
                return f.read()
    
    def pdf_exists(self, filename: str) -> bool:
        """
        Check if a PDF file exists in storage.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            True if file exists, False otherwise
        """
        if self.use_gcs:
            blob = self.bucket_pdfs.blob(filename)
            return blob.exists()
        else:
            file_path = config.PROJECTS_FOLDER / filename
            return file_path.exists()
    
    def get_pdf_temp_path(self, filename: str) -> str:
        """
        Get a local temporary file path for a PDF.
        Downloads from GCS if needed, returns local path if already local.
        
        Args:
            filename: Name of the PDF file
            
        Returns:
            Local file path (temporary if downloaded from GCS)
        """
        if self.use_gcs:
            # Download to temporary file
            blob = self.bucket_pdfs.blob(filename)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"librarian_{filename}")
            
            # Download if not already cached
            if not os.path.exists(temp_path):
                blob.download_to_filename(temp_path)
            
            return temp_path
        else:
            return str(config.PROJECTS_FOLDER / filename)
    
    def upload_pdf(self, filename: str, content: bytes):
        """
        Upload a PDF file to storage.
        
        Args:
            filename: Name of the PDF file
            content: PDF file content as bytes
        """
        if self.use_gcs:
            blob = self.bucket_pdfs.blob(filename)
            blob.upload_from_string(content, content_type='application/pdf')
        else:
            file_path = config.PROJECTS_FOLDER / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(content)
    
    # Vector database storage methods
    
    def sync_vector_db_to_cloud(self):
        """
        Sync local vector database to cloud storage.
        Used after building/updating the vector database locally.
        """
        if not self.use_gcs:
            return
        
        vector_db_path = config.VECTOR_DB_PATH
        if not vector_db_path.exists():
            print("No vector database found to sync")
            return
        
        print("Syncing vector database to Cloud Storage...")
        for file_path in vector_db_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(vector_db_path)
                blob = self.bucket_vectors.blob(str(relative_path))
                blob.upload_from_filename(str(file_path))
        
        print("✓ Vector database synced to cloud")
    
    def sync_vector_db_from_cloud(self):
        """
        Sync vector database from cloud storage to local filesystem.
        Used on startup in production to load the vector database.
        """
        if not self.use_gcs:
            return
        
        vector_db_path = config.VECTOR_DB_PATH
        vector_db_path.mkdir(parents=True, exist_ok=True)
        
        print("Syncing vector database from Cloud Storage...")
        blobs = self.bucket_vectors.list_blobs()
        
        for blob in blobs:
            # Skip metadata_db.json in vector bucket - handle separately
            if blob.name == "metadata_db.json":
                # Download to correct location
                blob.download_to_filename(str(config.METADATA_DB_PATH))
                print("✓ Metadata database synced")
                continue
                
            local_path = vector_db_path / blob.name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))
        
        print("✓ Vector database synced from cloud")


# Global instance
storage = StorageAdapter()

"""Cloud Storage adapter for handling PDF files in GCS or local filesystem."""
import hashlib
import os
import shutil
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
            pdf_paths = list(pdf_folder.rglob('*.pdf'))
            return [f.name for f in pdf_paths]

    def _resolve_from_metadata(self, filename: str) -> Optional[Path]:
        """Resolve a PDF path from indexed metadata when folder search fails."""
        if not config.METADATA_DB_PATH.exists():
            return None

        try:
            import json
            with open(config.METADATA_DB_PATH, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            project = metadata.get('projects', {}).get(filename)
            if not project:
                for entry in metadata.get('projects', {}).values():
                    if entry.get('file_name') == filename:
                        project = entry
                        break

            if not project:
                return None

            file_path = Path(project.get('file_path', ''))
            if not file_path:
                return None
            if not file_path.is_absolute():
                file_path = config.BASE_DIR / file_path
            if file_path.exists():
                return file_path
        except Exception as e:
            print(f"Warning: Failed to resolve '{filename}' from metadata: {e}")

        return None

    def _resolve_local_pdf_path(self, filename: str) -> Optional[Path]:
        """Resolve a local PDF filename to a full path, including nested folders."""
        search_roots = []
        if config.PROJECTS_FOLDER.exists():
            search_roots.append(config.PROJECTS_FOLDER)
        if config.BASE_DIR not in search_roots:
            search_roots.append(config.BASE_DIR)

        for root in search_roots:
            candidate = root / filename
            if candidate.exists():
                return candidate

            matches = list(root.rglob(filename))
            if matches:
                if len(matches) > 1:
                    print(f"Warning: multiple local PDFs found for '{filename}', using first match: {matches[0]}")
                return matches[0]

        return self._resolve_from_metadata(filename)
    
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
            file_path = self._resolve_local_pdf_path(filename)
            if not file_path:
                raise FileNotFoundError(filename)
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
            return self._resolve_local_pdf_path(filename) is not None
    
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
            file_path = self._resolve_local_pdf_path(filename)
            if not file_path:
                raise FileNotFoundError(filename)
            return str(file_path)
    
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

    @staticmethod
    def _upload_file_safely(blob, file_path: Path) -> Optional[str]:
        """
        Upload a file to GCS.

        SQLite databases are copied to a temp file first so uploads stay
        consistent even when the local API has the DB open.

        Returns:
            SHA256 of uploaded bytes for sqlite snapshots, otherwise None.
        """
        if file_path.suffix.lower() == ".sqlite3":
            import sqlite3

            with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite3") as tmp:
                temp_path = Path(tmp.name)
            try:
                source_conn = sqlite3.connect(f"file:{file_path}?mode=ro", uri=True)
                dest_conn = sqlite3.connect(str(temp_path))
                try:
                    source_conn.backup(dest_conn)
                finally:
                    dest_conn.close()
                    source_conn.close()
                uploaded_hash = StorageAdapter._sha256(temp_path)
                blob.upload_from_filename(str(temp_path))
                return uploaded_hash
            finally:
                temp_path.unlink(missing_ok=True)

        blob.upload_from_filename(str(file_path))
        return None

    @staticmethod
    def _sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    
    def sync_vector_db_to_cloud(self):
        """
        Sync local vector database and metadata to cloud storage.
        Used after building or updating the vector database locally.
        """
        if not self.use_gcs:
            print("Cloud sync is disabled because USE_CLOUD_STORAGE is false.")
            print("Set USE_CLOUD_STORAGE=true and configure GCS_BUCKET_PDFS, GCS_BUCKET_VECTORS, and GCP_PROJECT_ID to enable cloud sync.")
            return

        self.sync_pdfs_to_cloud()

        vector_db_path = config.VECTOR_DB_PATH
        if not vector_db_path.exists():
            print("No vector database found to sync")
            return

        print("Syncing vector database to Cloud Storage...")
        desired_blobs = set()
        uploaded_chroma_hash = None

        for file_path in vector_db_path.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(vector_db_path)
                blob_name = str(relative_path).replace('\\', '/')
                desired_blobs.add(blob_name)
                blob = self.bucket_vectors.blob(blob_name)
                uploaded_hash = self._upload_file_safely(blob, file_path)
                if file_path.name == "chroma.sqlite3" and uploaded_hash:
                    uploaded_chroma_hash = uploaded_hash
                    print(
                        f"Uploaded chroma.sqlite3 snapshot "
                        f"({file_path.stat().st_size} bytes, sha256={uploaded_hash[:12]}...)"
                    )

        if config.METADATA_DB_PATH.exists():
            metadata_blob_name = "metadata_db.json"
            desired_blobs.add(metadata_blob_name)
            metadata_blob = self.bucket_vectors.blob(metadata_blob_name)
            metadata_blob.upload_from_filename(str(config.METADATA_DB_PATH))
            print(f"Uploaded metadata database: {metadata_blob_name}")

        # Delete stale objects from the bucket that no longer exist locally.
        existing_blobs = {blob.name for blob in self.bucket_vectors.list_blobs()}
        stale_blobs = existing_blobs - desired_blobs
        for blob_name in stale_blobs:
            self.bucket_vectors.blob(blob_name).delete()
            print(f"Deleted stale cloud object: {blob_name}")

        if uploaded_chroma_hash:
            remote_blob = self.bucket_vectors.blob("chroma.sqlite3")
            remote_blob.reload()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite3") as tmp:
                remote_path = Path(tmp.name)
            try:
                remote_blob.download_to_filename(str(remote_path))
                remote_hash = self._sha256(remote_path)
            finally:
                remote_path.unlink(missing_ok=True)

            if uploaded_chroma_hash != remote_hash:
                raise RuntimeError(
                    "Cloud vector sync verification failed: chroma.sqlite3 hash mismatch "
                    f"(uploaded={uploaded_chroma_hash[:12]}..., cloud={remote_hash[:12]}...)"
                )
            print(f"Verified chroma.sqlite3 checksum in cloud ({uploaded_chroma_hash[:12]}...)")

        print("✓ Vector database and metadata synced to cloud")

    def sync_pdfs_to_cloud(self):
        """
        Sync local PDF files to the cloud PDF bucket.
        This ensures the deployed API can render thumbnails from GCS.
        """
        if not self.use_gcs:
            print("Cloud sync is disabled because USE_CLOUD_STORAGE is false.")
            print("Set USE_CLOUD_STORAGE=true and configure GCS_BUCKET_PDFS, GCS_BUCKET_VECTORS, and GCP_PROJECT_ID to enable cloud sync.")
            return

        # Prefer PDF paths referenced in metadata so cloud sync mirrors the actual indexed docs.
        pdf_paths = []
        if config.METADATA_DB_PATH.exists():
            try:
                import json
                with open(config.METADATA_DB_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                for project in metadata.get('projects', {}).values():
                    project_path = project.get('file_path')
                    if not project_path:
                        continue
                    path_obj = Path(project_path)
                    if not path_obj.is_absolute():
                        path_obj = config.BASE_DIR / path_obj
                    pdf_paths.append(path_obj)
            except Exception as e:
                print(f"Warning: Failed to load metadata for PDF sync: {e}")

        if not pdf_paths:
            pdf_folder = config.PROJECTS_FOLDER
            if not pdf_folder.exists():
                print("No local Projects folder found to sync PDFs")
                return
            pdf_paths = list(pdf_folder.rglob('*.pdf'))

        print("Syncing local PDFs to Cloud Storage...")
        desired_blobs = set()
        uploaded_files = set()
        for file_path in pdf_paths:
            if not file_path.is_file():
                continue
            blob_name = file_path.name
            if blob_name in uploaded_files:
                print(f"Warning: duplicate PDF name skipped: {blob_name} (from {file_path})")
                continue
            uploaded_files.add(blob_name)
            desired_blobs.add(blob_name)
            blob = self.bucket_pdfs.blob(blob_name)
            blob.upload_from_filename(str(file_path), content_type='application/pdf')
            print(f"Uploaded PDF: {blob_name}")

        existing_blobs = {blob.name for blob in self.bucket_pdfs.list_blobs()}
        stale_blobs = existing_blobs - desired_blobs
        for blob_name in stale_blobs:
            self.bucket_pdfs.blob(blob_name).delete()
            print(f"Deleted stale PDF object: {blob_name}")

        print("✓ PDFs synced to cloud")

    def sync_vector_db_from_cloud(self):
        """
        Sync vector database from cloud storage to local filesystem.
        Used on startup in production to load the vector database.
        """
        if not self.use_gcs:
            return
        
        vector_db_path = config.VECTOR_DB_PATH
        if vector_db_path.exists():
            shutil.rmtree(vector_db_path)
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
        
        chroma_path = vector_db_path / "chroma.sqlite3"
        if chroma_path.exists():
            print(
                f"✓ Vector database synced from cloud "
                f"(chroma.sqlite3 sha256={self._sha256(chroma_path)[:12]}...)"
            )
        else:
            print("✓ Vector database synced from cloud")


# Global instance
storage = StorageAdapter()

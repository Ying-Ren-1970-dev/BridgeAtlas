"""Configuration settings for the Librarian system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (.env should win over stale shell env in local dev)
load_dotenv(override=True)

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, production
IS_PRODUCTION = ENVIRONMENT == "production"

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()  # Strip whitespace/newlines from Secret Manager
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
GENERATE_SEARCH_SUMMARY = os.getenv("GENERATE_SEARCH_SUMMARY", "false").lower() == "true"
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
HYBRID_KEYWORD_CANDIDATE_LIMIT = int(os.getenv("HYBRID_KEYWORD_CANDIDATE_LIMIT", 4000))
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", 60))
HYBRID_DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("HYBRID_DEFAULT_RELEVANCE_THRESHOLD", 1.0))
VECTOR_DEFAULT_RELEVANCE_THRESHOLD = float(os.getenv("VECTOR_DEFAULT_RELEVANCE_THRESHOLD", 0.5))

# Cloud Storage Configuration (for GCP deployment)
USE_CLOUD_STORAGE = os.getenv("USE_CLOUD_STORAGE", str(IS_PRODUCTION)).lower() == "true"
GCS_BUCKET_PDFS = os.getenv("GCS_BUCKET_PDFS", "")
GCS_BUCKET_VECTORS = os.getenv("GCS_BUCKET_VECTORS", "")
GCS_PROJECT_ID = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", ""))

# Firestore Configuration (replaces metadata_db.json in production)
USE_FIRESTORE = os.getenv("USE_FIRESTORE", str(IS_PRODUCTION)).lower() == "true"
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "projects_metadata")

# Project Paths (local development)
BASE_DIR = Path(__file__).parent
PROJECTS_FOLDER = Path(os.getenv("PROJECTS_FOLDER", BASE_DIR))
PROJECTS_FOLDER_EXCLUDE = [
    pattern.strip().lower()
    for pattern in os.getenv("PROJECTS_FOLDER_EXCLUDE", "Public Projects").split(",")
    if pattern.strip()
]
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", BASE_DIR / "vector_db"))
METADATA_DB_PATH = Path(os.getenv("METADATA_DB_PATH", BASE_DIR / "metadata_db.json"))
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", BASE_DIR / "data"))

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_PAGES_PER_PDF = int(os.getenv("MAX_PAGES_PER_PDF", 500))

# Chunking Strategy
SECTION_AWARE_CHUNKING_PROJECT_PATTERNS = [
    p.strip().lower() for p in os.getenv("SECTION_AWARE_CHUNKING_PROJECT_PATTERNS", "all").split(",") if p.strip()
]

# Structural Engineering Categories
STRUCTURAL_CATEGORIES = [
    "Bridges",
    "Tunnels",
    "Stations",
    "Retaining Walls",
    "Pedestrian Structures",
    "Overhead Structures",
    "Foundations",
    "Seismic Design",
    "Structural Analysis",
    "General Structural",
]

# Plan Phases
PLAN_PHASES = [
    "Conceptual",
    "Preliminary",
    "30% Design",
    "60% Design",
    "90% Design",
    "100% Final",
    "IFB (Invitation for Bid)",
    "As-Built",
    "Markup",
]

# Ensure directories exist
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
METADATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

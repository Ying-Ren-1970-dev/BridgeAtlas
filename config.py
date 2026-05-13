"""Configuration settings for the Librarian system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")

# Project Paths
BASE_DIR = Path(__file__).parent
PROJECTS_FOLDER = Path(os.getenv("PROJECTS_FOLDER", BASE_DIR / "Projects"))
VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", BASE_DIR / "vector_db"))
METADATA_DB_PATH = Path(os.getenv("METADATA_DB_PATH", BASE_DIR / "metadata_db.json"))
DATA_FOLDER = Path(os.getenv("DATA_FOLDER", BASE_DIR / "data"))

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_PAGES_PER_PDF = int(os.getenv("MAX_PAGES_PER_PDF", 500))

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

"""
Configuration module for the Multi-Agent Literature Review and Research Gap Finder.
Handles environment variables, API endpoints, caching paths, and LLM configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load .env if present
load_dotenv(BASE_DIR / ".env")

# Directory configurations
DATA_DIR = BASE_DIR / "data"
PAPERS_DIR = DATA_DIR / "papers"
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"
SAMPLE_DIR = DATA_DIR / "sample"

# Ensure directories exist
for path in [DATA_DIR, PAPERS_DIR, REPORTS_DIR, CACHE_DIR, SAMPLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# API Configurations
OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "academic_researcher@example.edu")

ARXIV_BASE_URL = "http://export.arxiv.org/api/query"

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# HTTP Client Settings
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "15"))
MAX_RETRIES = 3
DEFAULT_SEARCH_LIMIT = 10

# LLM Configurations
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "heuristic").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME", "google/flan-t5-base")

# Text extraction limits
MAX_ABSTRACT_CHARS = 4000
MAX_FULLTEXT_CHARS = 25000

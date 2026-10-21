"""
Utility modules for literature retrieval, deduplication, and evidence verification.
"""

from .deduplication import deduplicate_papers, rank_papers, normalize_title, normalize_doi
from .evidence_verification import verify_evidence, extract_claims_audit

__all__ = [
    "deduplicate_papers",
    "rank_papers",
    "normalize_title",
    "normalize_doi",
    "verify_evidence",
    "extract_claims_audit",
]

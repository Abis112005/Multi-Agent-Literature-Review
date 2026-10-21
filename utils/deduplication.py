"""
Deduplication and ranking utilities for scholarly research papers.
Supports DOI normalization, arXiv ID matching, title similarity matching,
and relevance/recency scoring.
"""

import re
from typing import List, Tuple
from difflib import SequenceMatcher
from datetime import datetime, timezone
from models.schemas import PaperMetadata


def normalize_title(title: str) -> str:
    """
    Cleans and normalizes a paper title for deduplication.
    Strips punctuation, lowercases, and condenses spaces.
    """
    if not title:
        return ""
    cleaned = title.lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_doi(doi: str | None) -> str | None:
    """
    Normalizes a DOI string by removing common URL prefixes and lowercasing.
    """
    if not doi:
        return None
    cleaned = doi.strip().lower()
    cleaned = re.sub(r"^https?://(dx\.)?doi\.org/", "", cleaned)
    cleaned = re.sub(r"^doi:\s*", "", cleaned)
    return cleaned if cleaned else None


def normalize_arxiv_id(paper_id: str | None, url: str | None) -> str | None:
    """
    Extracts canonical arXiv ID (e.g., '2305.12345' from 'arXiv:2305.12345v2' or 'abs/2305.12345').
    """
    candidates = [paper_id or "", url or ""]
    for text in candidates:
        match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text)
        if match:
            return match.group(1)
    return None


def titles_are_similar(t1: str, t2: str, threshold: float = 0.88) -> bool:
    """
    Checks if two normalized titles are essentially identical.
    """
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    return SequenceMatcher(None, n1, n2).ratio() >= threshold


def merge_paper_metadata(base: PaperMetadata, duplicate: PaperMetadata) -> PaperMetadata:
    """
    Merges duplicate paper records, preserving the most complete fields.
    """
    # Prefer non-empty DOI
    if not base.doi and duplicate.doi:
        base.doi = duplicate.doi
    
    # Prefer longer abstract
    if len(duplicate.abstract or "") > len(base.abstract or ""):
        base.abstract = duplicate.abstract
    
    # Prefer working URL or arXiv PDF URL
    if not base.url and duplicate.url:
        base.url = duplicate.url
    
    # Combine or take max citations
    if duplicate.citation_count is not None:
        if base.citation_count is None or duplicate.citation_count > base.citation_count:
            base.citation_count = duplicate.citation_count
            
    # Merge authors to preserve the most complete list
    if duplicate.authors:
        if not base.authors:
            base.authors = duplicate.authors
        elif len(duplicate.authors) > len(base.authors):
            base.authors = duplicate.authors
        else:
            for a in duplicate.authors:
                if a not in base.authors:
                    base.authors.append(a)
        
    # Venue
    if (not base.venue or base.venue == "Not reported") and duplicate.venue:
        base.venue = duplicate.venue
        
    # Full text & pdf_path
    if not base.has_full_text and duplicate.has_full_text:
        base.has_full_text = duplicate.has_full_text
        base.pdf_path = duplicate.pdf_path
        
    return base


def deduplicate_papers(papers: List[PaperMetadata]) -> Tuple[List[PaperMetadata], int]:
    """
    Deduplicates a list of PaperMetadata records using:
    1. Normalized DOI
    2. Extracted canonical arXiv ID
    3. High-similarity normalized title
    
    Returns the deduplicated list and the count of duplicates removed.
    """
    unique_papers: List[PaperMetadata] = []
    doi_map: dict[str, int] = {}
    arxiv_map: dict[str, int] = {}
    duplicates_count = 0

    for paper in papers:
        is_dup = False
        doi = normalize_doi(paper.doi)
        arxiv_id = normalize_arxiv_id(paper.paper_id, paper.url)

        # 1. Match by DOI
        if doi and doi in doi_map:
            idx = doi_map[doi]
            unique_papers[idx] = merge_paper_metadata(unique_papers[idx], paper)
            duplicates_count += 1
            is_dup = True

        # 2. Match by arXiv ID
        elif arxiv_id and arxiv_id in arxiv_map:
            idx = arxiv_map[arxiv_id]
            unique_papers[idx] = merge_paper_metadata(unique_papers[idx], paper)
            duplicates_count += 1
            is_dup = True

        # 3. Match by Title Similarity
        else:
            for idx, existing in enumerate(unique_papers):
                if titles_are_similar(paper.title, existing.title):
                    unique_papers[idx] = merge_paper_metadata(existing, paper)
                    duplicates_count += 1
                    is_dup = True
                    break

        if not is_dup:
            cur_idx = len(unique_papers)
            unique_papers.append(paper)
            if doi:
                doi_map[doi] = cur_idx
            if arxiv_id:
                arxiv_map[arxiv_id] = cur_idx

    return unique_papers, duplicates_count


def calculate_relevance_score(paper: PaperMetadata, query: str) -> float:
    """
    Computes a hybrid relevance score (0.0 to 1.0) using:
    - Term frequency / coverage of query keywords in title & abstract
    - Recency boost for recent years
    - Citation count logarithmic scaling
    """
    if not query:
        return 0.5

    query_tokens = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    if not query_tokens:
        return 0.5

    title_text = paper.title.lower()
    abstract_text = (paper.abstract or "").lower()

    # Title matches are weighted significantly higher
    title_matches = sum(1 for token in query_tokens if token in title_text)
    abstract_matches = sum(1 for token in query_tokens if token in abstract_text)

    title_ratio = title_matches / len(query_tokens)
    abstract_ratio = abstract_matches / len(query_tokens)

    # Keyword relevance score (0 to 1)
    text_score = (title_ratio * 0.65) + (abstract_ratio * 0.35)

    # Recency factor
    current_year = datetime.now(timezone.utc).year
    year = paper.publication_year or (current_year - 5)
    age = max(0, current_year - year)
    recency_factor = max(0.5, 1.0 - (age * 0.05))

    # Citation bonus (up to 0.1 bonus for highly cited works)
    cites = paper.citation_count or 0
    citation_bonus = min(0.1, (cites ** 0.5) / 100.0) if cites > 0 else 0.0

    final_score = min(1.0, (text_score * 0.8) + (recency_factor * 0.1) + citation_bonus)
    return round(final_score, 3)


def rank_papers(papers: List[PaperMetadata], query: str) -> List[PaperMetadata]:
    """
    Scores and sorts papers in descending order of relevance and recency.
    """
    for paper in papers:
        paper.relevance_score = calculate_relevance_score(paper, query)

    # Sort descending by relevance score, with year as tie-breaker
    sorted_papers = sorted(
        papers,
        key=lambda p: (p.relevance_score or 0.0, p.publication_year or 0),
        reverse=True,
    )
    return sorted_papers

"""
Agent 1: Research Paper Search Agent.
Responsibilities:
- Accept research topic/question.
- Query OpenAlex, arXiv, and Semantic Scholar.
- Retrieve and normalize metadata (title, authors, year, abstract, DOI, URL, venue, citations).
- Deduplicate by DOI, arXiv ID, and title similarity.
- Score and rank by query relevance and recency.
- Save metadata locally and report retrieval metrics and API status.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import config
from models.schemas import PaperMetadata, SearchRequest
from services.openalex_service import OpenAlexService
from services.arxiv_service import ArxivService
from services.semantic_scholar_service import SemanticScholarService
from utils.deduplication import deduplicate_papers, rank_papers

logger = logging.getLogger(__name__)


class SearchAgent:
    """
    Coordinates multi-provider paper searches, deduplication, ranking, and local caching.
    """

    def __init__(self, cache_dir: Path = config.CACHE_DIR, sample_dir: Path = config.SAMPLE_DIR):
        self.cache_dir = Path(cache_dir)
        self.sample_dir = Path(sample_dir)
        self.openalex = OpenAlexService()
        self.arxiv = ArxivService()
        self.semantic_scholar = SemanticScholarService()

    def search(self, request: SearchRequest, use_sample_fallback: bool = False) -> Tuple[List[PaperMetadata], Dict[str, Any]]:
        """
        Executes unified search across selected sources, deduplicates, and ranks.
        Returns the ranked list of papers and execution metrics.
        """
        raw_papers: List[PaperMetadata] = []
        api_status: Dict[str, str] = {}
        queried_sources: List[str] = []

        # 1. Query OpenAlex
        if "openalex" in request.sources:
            queried_sources.append("OpenAlex")
            try:
                oa_papers = self.openalex.search_papers(
                    query=request.query,
                    limit=request.limit,
                    start_year=request.start_year,
                    end_year=request.end_year,
                )
                raw_papers.extend(oa_papers)
                api_status["OpenAlex"] = f"Success ({len(oa_papers)} papers retrieved)"
            except Exception as e:
                logger.error(f"OpenAlex search error: {e}")
                api_status["OpenAlex"] = f"Failed: {str(e)}"

        # 2. Query arXiv
        if "arxiv" in request.sources:
            queried_sources.append("arXiv")
            try:
                arxiv_papers = self.arxiv.search_papers(
                    query=request.query,
                    limit=request.limit,
                    start_year=request.start_year,
                    end_year=request.end_year,
                )
                raw_papers.extend(arxiv_papers)
                api_status["arXiv"] = f"Success ({len(arxiv_papers)} papers retrieved)"
            except Exception as e:
                logger.error(f"arXiv search error: {e}")
                api_status["arXiv"] = f"Failed: {str(e)}"

        # 3. Query Semantic Scholar
        if "semantic_scholar" in request.sources:
            queried_sources.append("Semantic Scholar")
            try:
                s2_papers = self.semantic_scholar.search_papers(
                    query=request.query,
                    limit=request.limit,
                    start_year=request.start_year,
                    end_year=request.end_year,
                )
                raw_papers.extend(s2_papers)
                api_status["Semantic Scholar"] = f"Success ({len(s2_papers)} papers retrieved)"
            except Exception as e:
                logger.error(f"Semantic Scholar search error: {e}")
                api_status["Semantic Scholar"] = f"Failed: {str(e)}"

        # If zero papers retrieved (e.g. offline machine or network restriction) and fallback requested:
        if not raw_papers and use_sample_fallback:
            logger.info("Using curated sample dataset as fallback.")
            sample_papers = self.load_sample_papers()
            raw_papers.extend(sample_papers)
            api_status["Fallback Mode"] = "Offline Sample Dataset Loaded"

        total_retrieved = len(raw_papers)

        # 4. Deduplicate
        deduped_papers, duplicates_removed = deduplicate_papers(raw_papers)

        # 5. Rank by relevance and recency
        ranked_papers = rank_papers(deduped_papers, request.query)

        # Slice to requested limit
        final_papers = ranked_papers[:request.limit]

        # 6. Save locally
        self.save_cache(final_papers, request.query)

        metrics = {
            "query": request.query,
            "sources_queried": queried_sources,
            "api_status": api_status,
            "total_retrieved": total_retrieved,
            "duplicates_removed": duplicates_removed,
            "unique_papers_count": len(deduped_papers),
            "final_selected_count": len(final_papers),
        }

        return final_papers, metrics

    def load_sample_papers(self) -> List[PaperMetadata]:
        """Loads curated demonstration papers from sample_papers.json."""
        sample_path = self.sample_dir / "sample_papers.json"
        if not sample_path.exists():
            return []
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [PaperMetadata(**item) for item in data]
        except Exception as e:
            logger.error(f"Error loading sample papers: {e}")
            return []

    def save_cache(self, papers: List[PaperMetadata], query: str):
        """Saves search results to local JSON cache for reproducibility."""
        cache_file = self.cache_dir / "last_search.json"
        try:
            payload = {
                "query": query,
                "count": len(papers),
                "papers": [p.model_dump() for p in papers],
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")

    def load_cached_search(self) -> Tuple[List[PaperMetadata], str]:
        """Loads papers from the most recent search cache."""
        cache_file = self.cache_dir / "last_search.json"
        if not cache_file.exists():
            return [], ""
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            papers = [PaperMetadata(**item) for item in data.get("papers", [])]
            return papers, data.get("query", "")
        except Exception as e:
            logger.error(f"Failed to load cached search: {e}")
            return [], ""

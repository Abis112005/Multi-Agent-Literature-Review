"""
Semantic Scholar API Service for academic literature retrieval.
Supports optional API keys and gracefully degrades on rate-limiting.
"""

import logging
import requests
from typing import List, Optional, Dict, Any
import config
from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)


class SemanticScholarService:
    """
    Interface for querying the Semantic Scholar Academic Graph API.
    """

    def __init__(
        self,
        api_key: str = config.SEMANTIC_SCHOLAR_API_KEY,
        timeout: int = config.REQUEST_TIMEOUT
    ):
        self.base_url = config.SEMANTIC_SCHOLAR_BASE_URL
        self.api_key = api_key
        self.timeout = timeout

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> List[PaperMetadata]:
        """
        Executes a paper search query on Semantic Scholar.
        """
        if not query or not query.strip():
            return []

        endpoint = f"{self.base_url}/paper/search"
        headers: Dict[str, str] = {
            "User-Agent": "MultiAgentLitReview/1.0"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key

        fields = "title,authors,year,abstract,url,venue,citationCount,externalIds,openAccessPdf"
        params: Dict[str, Any] = {
            "query": query.strip(),
            "limit": min(limit, 30),
            "fields": fields,
        }

        # Year filter parameter in Semantic Scholar format: "start_year-end_year"
        if start_year and end_year:
            params["year"] = f"{start_year}-{end_year}"
        elif start_year:
            params["year"] = f"{start_year}-"
        elif end_year:
            params["year"] = f"-{end_year}"

        papers: List[PaperMetadata] = []

        try:
            logger.info(f"Querying Semantic Scholar for: '{query}'")
            response = requests.get(endpoint, params=params, headers=headers, timeout=self.timeout)

            if response.status_code == 429:
                logger.warning(
                    "Semantic Scholar API rate limit reached (HTTP 429). "
                    "Skipping Semantic Scholar and continuing with OpenAlex and arXiv."
                )
                return []
            elif response.status_code != 200:
                logger.warning(
                    f"Semantic Scholar returned status {response.status_code}: {response.text[:150]}"
                )
                return []

            data = response.json()
            items = data.get("data", [])

            for item in items:
                try:
                    title = item.get("title")
                    if not title:
                        continue

                    authors = [
                        a.get("name") for a in item.get("authors", []) if a.get("name")
                    ]
                    pub_year = item.get("year")
                    abstract = item.get("abstract") or ""

                    ext_ids = item.get("externalIds") or {}
                    doi = ext_ids.get("DOI")
                    arxiv_id = ext_ids.get("ArXiv")

                    # URL
                    oa_pdf = item.get("openAccessPdf") or {}
                    url = oa_pdf.get("url") or item.get("url") or (f"https://doi.org/{doi}" if doi else None)

                    paper_id = doi or arxiv_id or item.get("paperId") or f"s2_{title[:25]}"

                    papers.append(
                        PaperMetadata(
                            paper_id=str(paper_id),
                            title=title.strip(),
                            authors=authors,
                            publication_year=pub_year,
                            abstract=abstract.strip(),
                            doi=doi,
                            url=url,
                            venue=item.get("venue") or "Not reported",
                            citation_count=item.get("citationCount"),
                            source="semantic_scholar",
                        )
                    )
                except Exception as ex:
                    logger.debug(f"Failed to parse Semantic Scholar item: {ex}")
                    continue

        except requests.exceptions.Timeout:
            logger.error("Semantic Scholar request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Semantic Scholar request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Semantic Scholar retrieval: {e}")

        return papers

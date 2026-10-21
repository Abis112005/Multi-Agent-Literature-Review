"""
OpenAlex API Service for retrieving scholarly paper metadata and abstracts.
Provides free, keyless access to millions of academic publications.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
import config
from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)


def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """
    Reconstructs continuous text from OpenAlex's abstract_inverted_index format.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    
    words_by_pos = {}
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                words_by_pos[pos] = word
                
    if not words_by_pos:
        return ""
        
    sorted_positions = sorted(words_by_pos.keys())
    return " ".join(words_by_pos[pos] for pos in sorted_positions)


class OpenAlexService:
    """
    Interface for querying the OpenAlex academic works endpoint.
    """

    def __init__(self, mailto: str = config.OPENALEX_MAILTO, timeout: int = config.REQUEST_TIMEOUT):
        self.base_url = config.OPENALEX_BASE_URL
        self.mailto = mailto
        self.timeout = timeout
        self.headers = {
            "User-Agent": f"MultiAgentLitReview/1.0 (mailto:{self.mailto})"
        }

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None
    ) -> List[PaperMetadata]:
        """
        Executes a search query against OpenAlex works.
        """
        if not query or not query.strip():
            return []

        url = f"{self.base_url}/works"
        params: Dict[str, Any] = {
            "search": query.strip(),
            "per-page": min(limit, 50),
            "mailto": self.mailto,
        }

        # Apply year filter if requested
        filter_parts = []
        if start_year:
            filter_parts.append(f"from_publication_date:{start_year}-01-01")
        if end_year:
            filter_parts.append(f"to_publication_date:{end_year}-12-31")
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        papers: List[PaperMetadata] = []

        try:
            logger.info(f"Querying OpenAlex for: '{query}'")
            response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)

            if response.status_code != 200:
                logger.warning(f"OpenAlex returned status {response.status_code}: {response.text[:200]}")
                return []

            data = response.json()
            results = data.get("results", [])

            for item in results:
                try:
                    title = item.get("title")
                    if not title:
                        continue

                    # Authors
                    authorships = item.get("authorships", [])
                    authors = [
                        a.get("author", {}).get("display_name")
                        for a in authorships
                        if a.get("author", {}).get("display_name")
                    ]

                    # Abstract
                    abstract_index = item.get("abstract_inverted_index")
                    abstract = reconstruct_abstract(abstract_index)

                    # Year
                    pub_year = item.get("publication_year")

                    # DOI & ID
                    doi = item.get("doi")
                    paper_id = doi or item.get("id") or f"openalex_{item.get('id', '')}"

                    # URL
                    primary_loc = item.get("primary_location") or {}
                    url_candidate = primary_loc.get("landing_page_url") or doi or item.get("id")

                    # Venue
                    source_info = primary_loc.get("source") or {}
                    venue = source_info.get("display_name") or "Not reported"

                    # Citation count
                    citations = item.get("cited_by_count")

                    papers.append(
                        PaperMetadata(
                            paper_id=str(paper_id),
                            title=title.strip(),
                            authors=authors,
                            publication_year=pub_year,
                            abstract=abstract.strip(),
                            doi=doi,
                            url=url_candidate,
                            venue=venue,
                            citation_count=citations,
                            source="openalex",
                        )
                    )
                except Exception as ex:
                    logger.debug(f"Failed to parse OpenAlex item: {ex}")
                    continue

        except requests.exceptions.Timeout:
            logger.error("OpenAlex request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAlex request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error querying OpenAlex: {e}")

        return papers

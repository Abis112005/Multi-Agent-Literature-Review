"""
arXiv API Service for retrieving scientific preprints and papers.
Queries the open export.arxiv.org API and parses Atom XML entries.
"""

import re
import logging
import xml.etree.ElementTree as ET
import requests
from typing import List, Optional
import config
from models.schemas import PaperMetadata

logger = logging.getLogger(__name__)

# Atom XML namespaces used by arXiv
ATOM_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def clean_xml_text(text: Optional[str]) -> str:
    """
    Cleans up newlines and excessive whitespace common in arXiv XML feeds.
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class ArxivService:
    """
    Interface for querying arXiv preprints via its open public API.
    """

    def __init__(self, timeout: int = config.REQUEST_TIMEOUT):
        self.base_url = config.ARXIV_BASE_URL
        self.timeout = timeout

    def search_papers(
        self,
        query: str,
        limit: int = 10,
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
    ) -> List[PaperMetadata]:
        """
        Executes a search query against arXiv.
        """
        if not query or not query.strip():
            return []

        # Format arXiv search query cleanly
        sanitized_query = re.sub(r"[^\w\s\-]", " ", query).strip()
        search_query = f"all:{sanitized_query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(limit, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        papers: List[PaperMetadata] = []

        try:
            logger.info(f"Querying arXiv for: '{query}'")
            response = requests.get(
                self.base_url,
                params=params,
                headers={"User-Agent": "MultiAgentLitReview/1.0"},
                timeout=self.timeout
            )

            if response.status_code != 200:
                logger.warning(f"arXiv returned status {response.status_code}")
                return []

            root = ET.fromstring(response.content)

            for entry in root.findall("atom:entry", ATOM_NS):
                try:
                    title_elem = entry.find("atom:title", ATOM_NS)
                    title = clean_xml_text(title_elem.text if title_elem is not None else "")
                    if not title or title.lower() == "error":
                        continue

                    # Summary / Abstract
                    summary_elem = entry.find("atom:summary", ATOM_NS)
                    abstract = clean_xml_text(summary_elem.text if summary_elem is not None else "")

                    # Authors
                    authors = []
                    for author in entry.findall("atom:author", ATOM_NS):
                        name_elem = author.find("atom:name", ATOM_NS)
                        if name_elem is not None and name_elem.text:
                            authors.append(clean_xml_text(name_elem.text))

                    # Published date / year
                    published_elem = entry.find("atom:published", ATOM_NS)
                    pub_year = None
                    if published_elem is not None and published_elem.text:
                        match = re.search(r"(\d{4})", published_elem.text)
                        if match:
                            pub_year = int(match.group(1))

                    # Apply year filters if provided
                    if start_year and pub_year and pub_year < start_year:
                        continue
                    if end_year and pub_year and pub_year > end_year:
                        continue

                    # ID and URLs
                    id_elem = entry.find("atom:id", ATOM_NS)
                    arxiv_url = id_elem.text.strip() if id_elem is not None else ""
                    
                    # DOI if present in arXiv metadata
                    doi_elem = entry.find("arxiv:doi", ATOM_NS)
                    doi = doi_elem.text.strip() if doi_elem is not None else None

                    # Prefer PDF link if available
                    pdf_link = None
                    for link in entry.findall("atom:link", ATOM_NS):
                        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                            pdf_link = link.attrib.get("href")
                            break

                    paper_id = arxiv_url or f"arxiv_{title[:30]}"

                    # Journal / Venue
                    journal_elem = entry.find("arxiv:journal_ref", ATOM_NS)
                    venue = journal_elem.text.strip() if journal_elem is not None else "arXiv Preprint"

                    papers.append(
                        PaperMetadata(
                            paper_id=paper_id,
                            title=title,
                            authors=authors,
                            publication_year=pub_year,
                            abstract=abstract,
                            doi=doi,
                            url=pdf_link or arxiv_url,
                            venue=venue,
                            citation_count=None,
                            source="arxiv",
                        )
                    )
                except Exception as ex:
                    logger.debug(f"Failed to parse arXiv entry: {ex}")
                    continue

        except requests.exceptions.Timeout:
            logger.error("arXiv request timed out.")
        except requests.exceptions.RequestException as e:
            logger.error(f"arXiv request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error querying arXiv: {e}")

        return papers

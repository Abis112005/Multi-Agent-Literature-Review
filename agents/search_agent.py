import requests
import time


def search_papers(topic, max_results=10):
    """
    Search for research papers.

    Tries OpenAlex first.
    If OpenAlex fails, tries Semantic Scholar.
    """

    # Try OpenAlex
    try:
        papers = search_openalex(topic, max_results)

        if papers:
            return papers

    except Exception as e:
        print(f"OpenAlex failed: {e}")

    # Try Semantic Scholar
    try:
        papers = search_semantic_scholar(topic, max_results)

        if papers:
            return papers

    except Exception as e:
        print(f"Semantic Scholar failed: {e}")

    # Both sources failed
    raise Exception(
        "Unable to retrieve research papers at the moment. "
        "Both OpenAlex and Semantic Scholar are unavailable. "
        "Please try again after a few minutes."
    )



def search_openalex(topic, max_results):
    """Search OpenAlex."""

    url = "https://api.openalex.org/works"

    params = {
        "search": topic,
        "per-page": max_results
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "Multi-Agent-Literature-Review/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for work in data.get("results", []):
        papers.append({
            "title": work.get("title"),
            "year": work.get("publication_year"),
            "doi": work.get("doi"),
            "abstract": extract_openalex_abstract(work),
            "url": work.get("primary_location", {}).get("landing_page_url")
        })

    return papers


def search_semantic_scholar(topic, max_results):
    """Search Semantic Scholar as a fallback."""

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": topic,
        "limit": max_results,
        "fields": "title,year,abstract,url,externalIds"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "Multi-Agent-Literature-Review/1.0"
        }
    )

    response.raise_for_status()

    data = response.json()

    papers = []

    for paper in data.get("data", []):
        external_ids = paper.get("externalIds") or {}

        papers.append({
            "title": paper.get("title"),
            "year": paper.get("year"),
            "doi": external_ids.get("DOI"),
            "abstract": paper.get("abstract") or "Abstract not available",
            "url": paper.get("url")
        })

    return papers


def extract_openalex_abstract(work):
    """Reconstruct abstract from OpenAlex inverted index."""

    inverted_index = work.get("abstract_inverted_index")

    if not inverted_index:
        return "Abstract not available"

    words = []

    for word, positions in inverted_index.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(word for position, word in words)


if __name__ == "__main__":
    topic = "LiDAR 3D object detection in snowy weather"

    papers = search_papers(topic, max_results=5)

    print(f"\nFound {len(papers)} papers:\n")

    for i, paper in enumerate(papers, start=1):
        print(f"{i}. {paper['title']}")
        print(f"   Year: {paper['year']}")
        print(f"   DOI: {paper['doi']}")
        print()


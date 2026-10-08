import requests
import time


def search_papers(topic, max_results=10):
    """
    Search for research papers.

    First tries OpenAlex.
    If OpenAlex is rate-limited, automatically uses Semantic Scholar.
    """

    try:
        return search_openalex(topic, max_results)

    except Exception as e:
        print(f"OpenAlex search failed: {e}")
        print("Trying Semantic Scholar...")

        return search_semantic_scholar(topic, max_results)


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


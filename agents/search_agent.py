import requests


def search_papers(topic, max_results=10):
    """
    Search for research papers using multiple sources.

    Order:
    1. OpenAlex
    2. Semantic Scholar
    3. Crossref
    """

    # Try OpenAlex
    try:
        papers = search_openalex(topic, max_results)
        if papers:
            print("Papers retrieved from OpenAlex.")
            return papers
    except Exception as e:
        print(f"OpenAlex failed: {e}")

    # Try Semantic Scholar
    try:
        papers = search_semantic_scholar(topic, max_results)
        if papers:
            print("Papers retrieved from Semantic Scholar.")
            return papers
    except Exception as e:
        print(f"Semantic Scholar failed: {e}")

    # Try Crossref
    try:
        papers = search_crossref(topic, max_results)
        if papers:
            print("Papers retrieved from Crossref.")
            return papers
    except Exception as e:
        print(f"Crossref failed: {e}")

    raise Exception(
        "Unable to retrieve research papers at the moment. "
        "All paper search sources are unavailable."
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
    """Search Semantic Scholar."""

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


def search_crossref(topic, max_results):
    """Search Crossref as the third fallback."""

    url = "https://api.crossref.org/works"

    params = {
        "query.bibliographic": topic,
        "rows": max_results,
        "mailto": "azamabis829@gmail.com"
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

    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title", [])
        title = title_list[0] if title_list else "Title not available"

        authors = item.get("author", [])

        abstract = item.get("abstract")

        if abstract:
            # Remove simple HTML/XML tags from Crossref abstracts
            import re
            abstract = re.sub("<[^>]+>", "", abstract)
        else:
            abstract = "Abstract not available"

        papers.append({
            "title": title,
            "year": get_crossref_year(item),
            "doi": item.get("DOI"),
            "abstract": abstract,
            "url": item.get("URL")
        })

    return papers


def get_crossref_year(item):
    """Get publication year from Crossref metadata."""

    for field in ["published-print", "published-online", "issued", "created"]:
        date_info = item.get(field)

        if date_info:
            date_parts = date_info.get("date-parts", [])

            if date_parts and date_parts[0]:
                return date_parts[0][0]

    return None


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




import requests
import time


def search_papers(topic, max_results=10):
    url = "https://api.openalex.org/works"

    params = {
        "search": topic,
        "per-page": max_results
    }

    max_retries = 3

    for attempt in range(max_retries):
        response = requests.get(
            url,
            params=params,
            timeout=30,
            headers={
                "User-Agent": "Multi-Agent-Literature-Review/1.0"
            }
        )

        if response.status_code == 429:
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue

            raise Exception(
                "OpenAlex is temporarily rate-limiting requests. "
                "Please wait a few minutes and try again."
            )

        response.raise_for_status()

        data = response.json()
        break

    papers = []

    for work in data.get("results", []):
        paper = {
            "title": work.get("title"),
            "year": work.get("publication_year"),
            "doi": work.get("doi"),
            "abstract": extract_abstract(work),
            "url": work.get("primary_location", {}).get("landing_page_url")
        }

        papers.append(paper)

    return papers


def extract_abstract(work):
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


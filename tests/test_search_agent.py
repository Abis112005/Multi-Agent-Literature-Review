"""
Unit tests for SearchAgent, deduplication, and ranking.
"""

import pytest
from models.schemas import SearchRequest, PaperMetadata
from agents.search_agent import SearchAgent
from utils.deduplication import deduplicate_papers, rank_papers, normalize_title, normalize_doi


def test_normalize_title():
    t1 = "MetaGPT: Meta Programming for A Multi-Agent Framework!"
    t2 = "metagpt meta programming for a multi agent framework"
    assert normalize_title(t1) == t2


def test_normalize_doi():
    doi1 = "https://doi.org/10.1016/j.artint.2023.103980"
    doi2 = "doi: 10.1016/j.artint.2023.103980"
    assert normalize_doi(doi1) == "10.1016/j.artint.2023.103980"
    assert normalize_doi(doi2) == "10.1016/j.artint.2023.103980"


def test_deduplication_by_doi():
    p1 = PaperMetadata(
        paper_id="id1",
        title="Agent Collaboration Survey",
        authors=["Alice"],
        publication_year=2023,
        abstract="Short abstract",
        doi="10.1234/test.doi",
        source="openalex"
    )
    p2 = PaperMetadata(
        paper_id="id2",
        title="Agent Collaboration Survey (Extended)",
        authors=["Alice", "Bob"],
        publication_year=2023,
        abstract="Much longer abstract with detailed methodology",
        doi="https://doi.org/10.1234/test.doi",
        source="arxiv"
    )

    deduped, removed = deduplicate_papers([p1, p2])
    assert len(deduped) == 1
    assert removed == 1
    # Check that it preserved the longer abstract and merged authors
    assert len(deduped[0].abstract) > len("Short abstract")
    assert "Bob" in deduped[0].authors


def test_deduplication_by_title_similarity():
    p1 = PaperMetadata(
        paper_id="p1",
        title="Voyager: An Open-Ended Embodied Agent with Large Language Models",
        authors=["Author A"],
        abstract="Minecraft agent using skills",
        source="arxiv"
    )
    p2 = PaperMetadata(
        paper_id="p2",
        title="Voyager: An Open Ended Embodied Agent with Large Language Models.",
        authors=["Author A"],
        abstract="Minecraft agent using skills",
        source="openalex"
    )

    deduped, removed = deduplicate_papers([p1, p2])
    assert len(deduped) == 1
    assert removed == 1


def test_sample_dataset_loading():
    agent = SearchAgent()
    sample_papers = agent.load_sample_papers()
    assert len(sample_papers) >= 3
    for p in sample_papers:
        assert p.title
        assert p.abstract
        assert p.paper_id


def test_ranking_papers():
    p1 = PaperMetadata(
        paper_id="p1",
        title="General Biology in Plants",
        abstract="Cellular biology of trees and leaves.",
        publication_year=2020,
    )
    p2 = PaperMetadata(
        paper_id="p2",
        title="Multi-Agent Systems for Collaborative Problem Solving",
        abstract="Autonomous agents communicating to solve complex tasks using LLMs.",
        publication_year=2024,
    )

    ranked = rank_papers([p1, p2], query="multi-agent collaborative systems")
    assert ranked[0].paper_id == "p2"
    assert ranked[0].relevance_score > ranked[1].relevance_score

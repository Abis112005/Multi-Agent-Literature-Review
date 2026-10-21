"""
Unit tests for AnalysisAgent and HeuristicNLPProvider.
"""

import pytest
from models.schemas import PaperMetadata
from agents.analysis_agent import AnalysisAgent
from services.llm_service import NOT_REPORTED


def test_analysis_agent_extraction_accuracy():
    agent = AnalysisAgent(llm_provider="heuristic")

    paper = PaperMetadata(
        paper_id="test_001",
        title="MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework",
        authors=["Sirui Hong et al."],
        publication_year=2023,
        abstract=(
            "We address the problem of cascading hallucination in multi-agent collaboration. "
            "Our goal is to coordinate role-based agents through Standardized Operating Procedures. "
            "We propose MetaGPT, a meta-programming framework incorporating SOPs. "
            "The system uses Transformer and LLM models. "
            "We evaluate on the HumanEval and SoftwareDev benchmarks using Pass@1 accuracy. "
            "Results show MetaGPT outperforms baseline models by 15%. "
            "However, the limitation is high token cost and latency in real-time execution. "
            "Future research directions include self-healing error recovery."
        ),
        url="https://arxiv.org/abs/2308.00352"
    )

    analysis = agent.analyze_paper(paper)

    assert analysis.paper_id == "test_001"
    assert "cascading hallucination" in analysis.research_problem.lower()
    assert analysis.proposed_methodology != NOT_REPORTED
    assert any("Transformer" in m or "LLM" in m for m in analysis.algorithms_and_models)
    assert any("HumanEval" in d or "SoftwareDev" in d for d in analysis.datasets_used)
    assert any("ACCURACY" in m or "PASS@1" in m for m in analysis.evaluation_metrics)
    assert any("token" in lim.lower() or "latency" in lim.lower() for lim in analysis.limitations)
    assert any("self-healing" in fut.lower() or "error recovery" in fut.lower() for fut in analysis.future_research_directions)
    assert analysis.confidence_score > 0.6


def test_analysis_agent_missing_data_reporting():
    """Verifies that when information is absent, it NEVER invents it and emits the exact required notice."""
    agent = AnalysisAgent(llm_provider="heuristic")

    sparse_paper = PaperMetadata(
        paper_id="sparse_01",
        title="A Brief Position Paper on Modern AI",
        authors=["Researcher"],
        abstract="This is a brief opinion note discussing philosophical implications of artificial intelligence."
    )

    analysis = agent.analyze_paper(sparse_paper)

    # Neither datasets nor metrics were mentioned
    assert analysis.datasets_used == [NOT_REPORTED]
    assert analysis.evaluation_metrics == [NOT_REPORTED]
    assert analysis.limitations == [NOT_REPORTED]

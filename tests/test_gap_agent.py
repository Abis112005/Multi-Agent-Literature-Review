"""
Unit tests for GapAgent and Research Gap Synthesis.
"""

import pytest
from models.schemas import PaperAnalysis, GapCategory, ConfidenceLevel
from agents.gap_agent import GapAgent


def test_gap_agent_identifies_thematic_gaps():
    agent = GapAgent(llm_provider="heuristic")

    analyses = [
        PaperAnalysis(
            paper_id="p1",
            paper_title="Multi-Agent Coding System",
            source_url="https://arxiv.org/abs/2301.0001",
            research_problem="Code generation in teams",
            limitations=["The framework incurs severe token overhead and inference latency during long dialogues."],
            datasets_used=["HumanEval"],
        ),
        PaperAnalysis(
            paper_id="p2",
            paper_title="Autonomous Agent Society",
            source_url="https://arxiv.org/abs/2301.0002",
            research_problem="Social simulation with agents",
            limitations=["High computational cost and quadratic token scaling prevent real-time scaling."],
            datasets_used=["Custom synthetic dataset"],
        ),
    ]

    gaps = agent.identify_gaps(analyses, topic="Multi-Agent Scaling")

    assert len(gaps) >= 1
    # Check that scalability gap was triggered
    categories = [g.category for g in gaps]
    assert GapCategory.SCALABILITY in categories or GapCategory.DATASET in categories

    # Check evidence grounding
    scalability_gap = next(g for g in gaps if g.category == GapCategory.SCALABILITY)
    assert len(scalability_gap.supporting_papers) >= 1
    assert scalability_gap.supporting_papers[0].paper_id in ["p1", "p2"]
    assert scalability_gap.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
    assert len(scalability_gap.suggested_research_questions) > 0
    assert len(scalability_gap.recommended_experiments) > 0

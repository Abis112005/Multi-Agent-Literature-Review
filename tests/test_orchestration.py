"""
Unit and integration tests for WorkflowEngine orchestration.
"""

import pytest
from models.schemas import SearchRequest
from orchestration.workflow import WorkflowEngine


def test_full_pipeline_orchestration_with_sample_dataset():
    engine = WorkflowEngine(llm_provider="heuristic")

    request = SearchRequest(
        query="LLM Multi-Agent Collaboration",
        limit=3,
        sources=["openalex", "arxiv"],
    )

    # Run pipeline with sample fallback so it is reliable even offline in testing
    report = engine.run_full_pipeline(request, use_sample_fallback=True)

    assert report is not None
    assert report.topic == "LLM Multi-Agent Collaboration"
    assert len(report.papers) > 0
    assert len(report.research_gaps) > 0
    assert len(report.comparison_matrix) > 0
    assert len(report.references) > 0

    # Verify report service exports
    md = engine.report_service.generate_markdown_report(report)
    assert "# Literature Review & Research Gap Analysis" in md
    assert "Search Methodology" in md
    assert len(md) > 500

    pdf_path = engine.report_service.generate_pdf_report(report, "test_report.pdf")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000

"""
Unit tests for evidence verification and audit trails.
"""

import pytest
from models.schemas import (
    PaperMetadata,
    PaperAnalysis,
    ResearchGap,
    SupportingEvidence,
    GapCategory,
    GapType,
    ConfidenceLevel,
)
from utils.evidence_verification import verify_evidence, extract_claims_audit


def test_evidence_verification_flags_and_audits():
    p1 = PaperMetadata(
        paper_id="paper_123",
        title="Foundations of Autonomous Agents",
        abstract="We investigate communication stagnation in LLM agents.",
        url="https://doi.org/10.1234/foundations",
    )

    a1 = PaperAnalysis(
        paper_id="paper_123",
        paper_title="Foundations of Autonomous Agents",
        source_url="https://doi.org/10.1234/foundations",
        limitations=["Communication stagnation during long task execution."],
    )

    gap = ResearchGap(
        gap_id="GAP-001",
        gap_title="Dialogue Stagnation in Agent Loops",
        category=GapCategory.ALGORITHM,
        gap_type=GapType.EXPLICIT_LIMITATION,
        description="Agents stop making forward progress in conversation chains.",
        why_it_matters="Causes task failure and infinite looping.",
        supporting_papers=[
            SupportingEvidence(
                paper_id="paper_123",
                paper_title="Foundations of Autonomous Agents",
                evidence_text="Communication stagnation during long task execution.",
            )
        ],
        confidence_level=ConfidenceLevel.HIGH,
        confidence_justification="Direct author limitation in paper_123",
    )

    verified_gaps, audits = verify_evidence([gap], [p1], [a1])

    assert len(verified_gaps) == 1
    assert "Verified" in verified_gaps[0].verification_status
    assert len(audits) == 1
    assert len(audits[0]["verified_papers"]) == 1
    assert audits[0]["verified_papers"][0]["paper_id"] == "paper_123"
    assert audits[0]["verified_papers"][0]["url"] == "https://doi.org/10.1234/foundations"

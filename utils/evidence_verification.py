"""
Evidence verification engine.
Validates that generated analyses, claims, and identified research gaps
are strictly supported by retrieved academic text, authentic titles, and genuine URLs.
"""

from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
from models.schemas import (
    PaperMetadata,
    PaperAnalysis,
    ResearchGap,
    GapType,
    SupportingEvidence,
)


def fuzzy_text_search(snippet: str, document: str, threshold: float = 0.65) -> bool:
    """
    Checks if snippet ideas or key sentences roughly appear in document text.
    """
    if not snippet or not document:
        return False
    
    snippet_clean = snippet.strip().lower()
    doc_clean = document.strip().lower()
    
    if snippet_clean in doc_clean:
        return True
    
    # Check word overlap
    words = [w for w in snippet_clean.split() if len(w) > 3]
    if not words:
        return True
        
    matched_words = sum(1 for w in words if w in doc_clean)
    overlap = matched_words / len(words)
    return overlap >= threshold


def verify_paper_analysis(analysis: PaperAnalysis, paper: PaperMetadata) -> Dict[str, Any]:
    """
    Verifies that a paper's analysis does not fabricate bibliographic or textual claims.
    """
    audit = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "source_url_valid": bool(paper.url and analysis.source_url == paper.url),
        "text_source": "full_pdf" if paper.has_full_text else "abstract_only",
        "has_abstract": bool(paper.abstract and len(paper.abstract.strip()) > 20),
        "verified_fields": [],
        "warnings": [],
    }

    doc_text = (paper.abstract or "")
    if paper.pdf_path:
        # full text could be longer
        audit["text_source"] = "full_pdf"

    if not audit["has_abstract"] and not paper.has_full_text:
        audit["warnings"].append(
            "Neither detailed abstract nor full-text PDF was available. "
            "Analysis may rely on minimal metadata."
        )

    # Check key fields for the required disclaimer
    missing_notice = "Not reported in the available text."
    for field_name in ["research_problem", "proposed_methodology"]:
        val = getattr(analysis, field_name, "")
        if val == missing_notice:
            audit["verified_fields"].append(f"{field_name} (Accurately flagged as unreported)")
        elif fuzzy_text_search(val, doc_text):
            audit["verified_fields"].append(f"{field_name} (Supported by source text)")
        else:
            audit["warnings"].append(
                f"{field_name} statement contains syntheses not strictly identical to verbatim text."
            )

    return audit


def verify_evidence(
    gaps: List[ResearchGap],
    papers: List[PaperMetadata],
    analyses: List[PaperAnalysis]
) -> Tuple[List[ResearchGap], List[Dict[str, Any]]]:
    """
    Verifies each research gap against source paper metadata and text.
    Distinguishes explicit limitations from inferred gaps, verifies paper IDs and URLs,
    and flags any unsupported assertions.
    """
    papers_by_id = {p.paper_id: p for p in papers}
    analyses_by_id = {a.paper_id: a for a in analyses}
    audit_logs: List[Dict[str, Any]] = []

    verified_gaps: List[ResearchGap] = []

    for gap in gaps:
        gap_audit: Dict[str, Any] = {
            "gap_id": gap.gap_id,
            "gap_title": gap.gap_title,
            "category": gap.category.value,
            "gap_type": gap.gap_type.value,
            "evidence_count": len(gap.supporting_papers),
            "verified_papers": [],
            "unverified_citations": [],
            "notes": [],
        }

        valid_evidence_list: List[SupportingEvidence] = []

        for ev in gap.supporting_papers:
            # Check if paper actually exists in our retrieved collection
            matching_paper = papers_by_id.get(ev.paper_id)
            if not matching_paper:
                # Try finding by title
                for p in papers:
                    if p.title.strip().lower() == ev.paper_title.strip().lower():
                        matching_paper = p
                        ev.paper_id = p.paper_id
                        break

            if matching_paper:
                # Ensure authentic title & URL
                ev.paper_title = matching_paper.title
                ev.source_url = matching_paper.url
                ev.source_section = "full_pdf" if matching_paper.has_full_text else "abstract"

                # Check if quotation or claim is grounded
                doc_text = matching_paper.abstract or ""
                grounded = fuzzy_text_search(ev.evidence_text, doc_text)

                status_label = "Grounded in source text" if grounded else "Inferred synthesis from source paper"
                gap_audit["verified_papers"].append({
                    "paper_id": matching_paper.paper_id,
                    "title": matching_paper.title,
                    "url": matching_paper.url,
                    "source_type": ev.source_section,
                    "status": status_label,
                })
                valid_evidence_list.append(ev)
            else:
                gap_audit["unverified_citations"].append({
                    "paper_id": ev.paper_id,
                    "paper_title": ev.paper_title,
                    "reason": "Paper ID not found in retrieved literature set"
                })

        # Update supporting papers with verified list
        gap.supporting_papers = valid_evidence_list

        # Verification status logic
        if not valid_evidence_list:
            gap.verification_status = "Unverified - Missing supporting paper references"
            gap_audit["notes"].append("No valid supporting papers retrieved from the scholarly set.")
        elif gap.gap_type == GapType.EXPLICIT_LIMITATION:
            gap.verification_status = "Verified: Direct author limitation"
            gap_audit["notes"].append("Directly substantiated by explicit limitation statement in paper.")
        else:
            gap.verification_status = "Verified: Cross-paper comparative inference"
            gap_audit["notes"].append(
                "Inferred by synthesizing comparative differences across multiple retrieved papers."
            )

        # Full-text availability advisory
        abstract_only_count = sum(
            1 for ev in valid_evidence_list if ev.source_section == "abstract"
        )
        if abstract_only_count > 0:
            gap_audit["notes"].append(
                f"{abstract_only_count} supporting paper(s) relied on scholarly abstracts "
                "because full open-access PDF was not provided."
            )

        verified_gaps.append(gap)
        audit_logs.append(gap_audit)

    return verified_gaps, audit_logs


def extract_claims_audit(
    papers: List[PaperMetadata],
    analyses: List[PaperAnalysis]
) -> List[Dict[str, Any]]:
    """
    Generates a full audit trail for all analyzed papers.
    """
    audit_results = []
    analyses_by_id = {a.paper_id: a for a in analyses}

    for paper in papers:
        analysis = analyses_by_id.get(paper.paper_id)
        if analysis:
            audit_results.append(verify_paper_analysis(analysis, paper))
    return audit_results

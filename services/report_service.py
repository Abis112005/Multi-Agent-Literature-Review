"""
Report generation service.
Produces academic literature review reports in both Markdown and publication-ready PDF format
with clickable citations and complete evidence verification logs.
"""

from pathlib import Path
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

import config
from models.schemas import LiteratureReviewReport


class ReportService:
    """
    Builds Markdown and PDF reports from synthesized literature reviews.
    """

    def __init__(self, reports_dir: Path = config.REPORTS_DIR):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown_report(self, report: LiteratureReviewReport) -> str:
        """
        Formats report into comprehensive Markdown with academic structure.
        """
        md_lines = []
        md_lines.append(f"# Literature Review & Research Gap Analysis: {report.topic}")
        md_lines.append(f"\n*Generated on: {report.generated_at}*")
        md_lines.append(f"\n---\n")

        # 1. Executive Summary & Search Methodology
        md_lines.append("## 1. Search Methodology & Retrieval Summary")
        md_lines.append(f"{report.search_methodology}\n")

        sel = report.paper_selection_summary
        md_lines.append("### Paper Selection Metrics")
        md_lines.append(f"- **Total Sources Queried**: {', '.join(sel.get('sources_queried', []))}")
        md_lines.append(f"- **Total Retrieved Records**: {sel.get('total_retrieved', 0)}")
        md_lines.append(f"- **Duplicates Removed**: {sel.get('duplicates_removed', 0)}")
        md_lines.append(f"- **Analyzed Papers**: {sel.get('analyzed_count', len(report.papers))}")
        md_lines.append(f"- **Full-Text PDFs Analyzed**: {sel.get('full_text_count', 0)}")
        md_lines.append(f"- **Analysis Engine**: {sel.get('analysis_mode', 'Heuristic / Local LLM')}\n")

        # 2. Individual Paper Analyses
        md_lines.append("## 2. Structured Paper Analyses")
        for idx, paper in enumerate(report.papers, 1):
            md_lines.append(f"### 2.{idx} {paper.paper_title}")
            if paper.source_url:
                md_lines.append(f"**Source URL / DOI**: [{paper.source_url}]({paper.source_url})  ")
            md_lines.append(f"**Analysis Method**: `{paper.analysis_mode}` | **Confidence**: {int(paper.confidence_score * 100)}%\n")

            md_lines.append(f"- **Research Problem**: {paper.research_problem}")
            md_lines.append(f"- **Proposed Methodology**: {paper.proposed_methodology}")
            md_lines.append(f"- **Algorithms / Models**: {', '.join(paper.algorithms_and_models)}")
            md_lines.append(f"- **Datasets**: {', '.join(paper.datasets_used)}")
            md_lines.append(f"- **Evaluation Metrics**: {', '.join(paper.evaluation_metrics)}")
            
            md_lines.append(f"- **Main Findings**:")
            for f in paper.main_findings:
                md_lines.append(f"  - {f}")
                
            md_lines.append(f"- **Author-Stated Limitations**:")
            for lim in paper.limitations:
                md_lines.append(f"  - {lim}")

            md_lines.append(f"- **Future Directions**:")
            for fd in paper.future_research_directions:
                md_lines.append(f"  - {fd}")
            md_lines.append("")

        # 3. Comparative Matrix
        md_lines.append("## 3. Literature Comparison Matrix")
        md_lines.append("| Paper Title | Models/Algorithms | Datasets | Key Limitations |")
        md_lines.append("| :--- | :--- | :--- | :--- |")
        for row in report.comparison_matrix:
            title_trunc = row.get("Paper", "")[:35] + ("..." if len(row.get("Paper", "")) > 35 else "")
            models_str = row.get("Model", "N/A")[:30]
            datasets_str = row.get("Dataset", "N/A")[:25]
            lim_str = row.get("Limitations", "N/A")[:40] + ("..." if len(row.get("Limitations", "")) > 40 else "")
            md_lines.append(f"| {title_trunc} | {models_str} | {datasets_str} | {lim_str} |")
        md_lines.append("\n")

        # 4. Research Gaps
        md_lines.append("## 4. Potential Research Gaps & Evidence Grounding")
        md_lines.append(
            "> *Notice: Findings are identified as potential research gaps based on cross-paper comparative "
            "analysis and explicit author limitations. Empirical verification is recommended.*\n"
        )

        for idx, gap in enumerate(report.research_gaps, 1):
            md_lines.append(f"### Gap {idx}: {gap.gap_title}")
            md_lines.append(f"- **Category**: `{gap.category.value}`")
            md_lines.append(f"- **Type**: `{gap.gap_type.value}`")
            md_lines.append(f"- **Confidence Level**: **{gap.confidence_level.value}** ({gap.confidence_justification})")
            md_lines.append(f"- **Verification Status**: `{gap.verification_status}`\n")
            md_lines.append(f"**Description**:\n{gap.description}\n")
            md_lines.append(f"**Significance (Why It Matters)**:\n{gap.why_it_matters}\n")

            md_lines.append("**Supporting Evidence from Literature**:")
            for ev in gap.supporting_papers:
                link_text = f"[{ev.paper_title}]({ev.source_url})" if ev.source_url else ev.paper_title
                md_lines.append(f"- *Paper*: {link_text}")
                md_lines.append(f"  - *Evidence quote/statement*: \"{ev.evidence_text}\" *(Source: {ev.source_section})*")
            md_lines.append("")

            if gap.suggested_research_questions:
                md_lines.append("**Suggested Research Questions**:")
                for q in gap.suggested_research_questions:
                    md_lines.append(f"  1. {q}")
                md_lines.append("")

            if gap.recommended_experiments:
                md_lines.append("**Recommended Future Experiments**:")
                for exp in gap.recommended_experiments:
                    md_lines.append(f"  - {exp}")
                md_lines.append("")

        # 5. Synthesis & Future Directions
        md_lines.append("## 5. Synthesis & Strategic Future Research Directions")
        md_lines.append(f"{report.synthesis_and_future_directions}\n")

        # 6. References
        md_lines.append("## 6. Bibliographic References")
        for idx, ref in enumerate(report.references, 1):
            title = ref.get("title", "Untitled")
            authors = ref.get("authors", "Unknown Authors")
            year = ref.get("year", "n.d.")
            venue = ref.get("venue", "Scholarly Archive")
            url = ref.get("url")
            doi = ref.get("doi")

            ref_str = f"{idx}. {authors} ({year}). **{title}**. *{venue}*."
            if doi:
                ref_str += f" DOI: [{doi}](https://doi.org/{doi})."
            elif url:
                ref_str += f" Available at: [{url}]({url})."
            md_lines.append(ref_str)

        return "\n".join(md_lines)

    def generate_pdf_report(self, report: LiteratureReviewReport, output_filename: str = "literature_review.pdf") -> Path:
        """
        Compiles the literature review into a formatted PDF using ReportLab.
        """
        pdf_path = self.reports_dir / output_filename
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=54,
            leftMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()

        # Custom academic styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=10,
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=15,
        )
        h1_style = ParagraphStyle(
            "Heading1_Custom",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=12,
            spaceAfter=6,
        )
        h2_style = ParagraphStyle(
            "Heading2_Custom",
            parent=styles["Heading3"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=8,
            spaceAfter=4,
        )
        body_style = ParagraphStyle(
            "Body_Custom",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
        bullet_style = ParagraphStyle(
            "Bullet_Custom",
            parent=body_style,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=3,
        )

        story = []

        # Title block
        story.append(Paragraph(f"Literature Review & Research Gap Analysis", title_style))
        story.append(Paragraph(f"Topic: <b>{report.topic}</b> | Generated: {report.generated_at}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=15))

        # 1. Methodology
        story.append(Paragraph("1. Search Methodology & Retrieval Overview", h1_style))
        story.append(Paragraph(report.search_methodology, body_style))
        story.append(Spacer(1, 8))

        # 2. Papers Analyzed
        story.append(Paragraph("2. Analyzed Academic Papers", h1_style))
        for idx, paper in enumerate(report.papers, 1):
            story.append(Paragraph(f"<b>{idx}. {paper.paper_title}</b>", h2_style))
            story.append(Paragraph(f"<b>Problem:</b> {paper.research_problem}", body_style))
            story.append(Paragraph(f"<b>Methodology:</b> {paper.proposed_methodology}", body_style))
            models_txt = ", ".join(paper.algorithms_and_models)
            story.append(Paragraph(f"<b>Models/Algorithms:</b> {models_txt}", body_style))
            datasets_txt = ", ".join(paper.datasets_used)
            story.append(Paragraph(f"<b>Datasets:</b> {datasets_txt}", body_style))
            story.append(Spacer(1, 6))

        # 3. Gaps
        story.append(Paragraph("3. Potential Research Gaps & Grounding Evidence", h1_style))
        for idx, gap in enumerate(report.research_gaps, 1):
            story.append(Paragraph(f"<b>Gap {idx}: {gap.gap_title}</b> (Confidence: {gap.confidence_level.value})", h2_style))
            story.append(Paragraph(f"<b>Category:</b> {gap.category.value} | <b>Type:</b> {gap.gap_type.value}", body_style))
            story.append(Paragraph(f"<b>Description:</b> {gap.description}", body_style))
            story.append(Paragraph(f"<b>Significance:</b> {gap.why_it_matters}", body_style))
            story.append(Paragraph("<b>Supporting Evidence:</b>", body_style))
            for ev in gap.supporting_papers:
                ev_text = f"• <i>{ev.paper_title}</i>: \"{ev.evidence_text}\""
                story.append(Paragraph(ev_text, bullet_style))
            story.append(Spacer(1, 8))

        # 4. References
        story.append(Paragraph("4. Bibliographic References", h1_style))
        for idx, ref in enumerate(report.references, 1):
            title = ref.get("title", "")
            authors = ref.get("authors", "")
            year = ref.get("year", "")
            venue = ref.get("venue", "")
            ref_str = f"{idx}. {authors} ({year}). <i>{title}</i>. {venue}."
            story.append(Paragraph(ref_str, bullet_style))

        doc.build(story)
        return pdf_path

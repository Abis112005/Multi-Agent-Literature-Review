"""
Agent 2: Research Paper Analysis Agent.
Responsibilities:
- Analyze each retrieved or uploaded paper individually.
- Extract the 9 academic dimensions:
    1. Research problem
    2. Research objectives
    3. Proposed methodology
    4. Algorithms and models used
    5. Datasets used
    6. Evaluation metrics
    7. Main findings and results
    8. Limitations
    9. Future research directions
- Extract text from PDFs using PyMuPDF when available, otherwise analyze the abstract.
- Strictly ground claims in available text; emit "Not reported in the available text." for missing info.
- Never invent citations, metrics, or datasets.
- Store structured JSON output per paper.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import config
from models.schemas import PaperMetadata, PaperAnalysis
from services.llm_service import LLMService, NOT_REPORTED
from services.pdf_service import PDFService

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Independently inspects and extracts structured scholarly dimensions from papers.
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        cache_dir: Path = config.CACHE_DIR
    ):
        self.cache_dir = Path(cache_dir)
        self.llm_service = LLMService(provider=llm_provider)
        self.pdf_service = PDFService()

    def set_llm_provider(self, provider_name: str):
        """Switches the active LLM / heuristic provider."""
        self.llm_service = LLMService(provider=provider_name)

    def analyze_paper(self, paper: PaperMetadata) -> PaperAnalysis:
        """
        Analyzes a single paper using its full-text PDF if present, or its abstract.
        """
        full_text_used = False
        text_to_analyze = paper.abstract or ""

        # Check if full PDF is locally available
        if paper.pdf_path and Path(paper.pdf_path).exists():
            pdf_result = self.pdf_service.extract_text_from_pdf(paper.pdf_path)
            if pdf_result.get("success") and pdf_result.get("text"):
                text_to_analyze = pdf_result["text"]
                full_text_used = True
                paper.has_full_text = True

        if not text_to_analyze.strip():
            logger.warning(f"Paper '{paper.title}' has no abstract or full text. Defaulting to unreported.")
            return PaperAnalysis(
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source_url=paper.url,
                analysis_mode="unreported_fallback",
                confidence_score=0.1,
                is_full_text_used=False,
            )

        # Call extraction through LLMService
        raw_analysis = self.llm_service.analyze_paper_text(
            title=paper.title,
            text=text_to_analyze,
            full_pdf_used=full_text_used,
        )

        # Capture raw evidence excerpts for audit trail
        evidence_excerpts = {
            "title": paper.title,
            "abstract_snippet": (paper.abstract or "")[:400],
            "text_source": "full_pdf" if full_text_used else "abstract",
            "source_url": paper.url or "",
        }

        # Build validated Pydantic model
        analysis = PaperAnalysis(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            source_url=paper.url,
            research_problem=raw_analysis.get("research_problem", NOT_REPORTED),
            research_objectives=raw_analysis.get("research_objectives", [NOT_REPORTED]),
            proposed_methodology=raw_analysis.get("proposed_methodology", NOT_REPORTED),
            algorithms_and_models=raw_analysis.get("algorithms_and_models", [NOT_REPORTED]),
            datasets_used=raw_analysis.get("datasets_used", [NOT_REPORTED]),
            evaluation_metrics=raw_analysis.get("evaluation_metrics", [NOT_REPORTED]),
            main_findings=raw_analysis.get("main_findings", [NOT_REPORTED]),
            limitations=raw_analysis.get("limitations", [NOT_REPORTED]),
            future_research_directions=raw_analysis.get("future_research_directions", [NOT_REPORTED]),
            analysis_mode=raw_analysis.get("analysis_mode", "heuristic_nlp"),
            confidence_score=raw_analysis.get("confidence_score", 0.7),
            is_full_text_used=full_text_used,
            raw_evidence_excerpts=evidence_excerpts,
        )

        return analysis

    def analyze_batch(
        self,
        papers: List[PaperMetadata],
        progress_callback=None
    ) -> List[PaperAnalysis]:
        """
        Analyzes a batch of papers sequentially, saving progress.
        """
        analyses: List[PaperAnalysis] = []
        total = len(papers)

        for idx, paper in enumerate(papers):
            logger.info(f"Analyzing paper {idx+1}/{total}: '{paper.title[:50]}'")
            analysis = self.analyze_paper(paper)
            analyses.append(analysis)

            if progress_callback:
                progress_callback(idx + 1, total, paper.title)

        # Cache analyses locally
        self.save_analyses_cache(analyses)
        return analyses

    def save_analyses_cache(self, analyses: List[PaperAnalysis]):
        """Persists analyzed outputs to JSON."""
        cache_path = self.cache_dir / "last_analyses.json"
        try:
            data = [a.model_dump() for a in analyses]
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save analyses cache: {e}")

    def load_cached_analyses(self) -> List[PaperAnalysis]:
        """Loads cached analyses if present."""
        cache_path = self.cache_dir / "last_analyses.json"
        if not cache_path.exists():
            return []
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [PaperAnalysis(**item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load cached analyses: {e}")
            return []

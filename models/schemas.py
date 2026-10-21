"""
Pydantic data models for literature metadata, paper analyses, research gaps,
and final synthesized reports.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class GapCategory(str, Enum):
    DATASET = "dataset"
    ALGORITHM = "algorithm"
    GENERALIZATION = "generalization"
    EVALUATION = "evaluation"
    SCALABILITY = "scalability"
    REAL_WORLD_DEPLOYMENT = "real_world_deployment"
    METHODOLOGY = "methodology"


class GapType(str, Enum):
    EXPLICIT_LIMITATION = "explicit_author_limitation"
    INFERRED_CROSS_PAPER = "inferred_cross_paper"


class ConfidenceLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Research topic or scientific question")
    limit: int = Field(default=10, ge=1, le=50, description="Number of papers to retrieve")
    start_year: Optional[int] = Field(default=None, description="Earliest publication year")
    end_year: Optional[int] = Field(default=None, description="Latest publication year")
    sources: List[str] = Field(
        default=["openalex", "arxiv"],
        description="Scholarly data sources to query (openalex, arxiv, semantic_scholar)"
    )


class PaperMetadata(BaseModel):
    paper_id: str = Field(..., description="Unique paper identifier (DOI, arXiv ID, or hash)")
    title: str = Field(..., description="Article title")
    authors: List[str] = Field(default_factory=list, description="Author names")
    publication_year: Optional[int] = Field(default=None, description="Year of publication")
    abstract: str = Field(default="", description="Paper abstract")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    url: Optional[str] = Field(default=None, description="Landing page or PDF URL")
    venue: Optional[str] = Field(default="Not reported", description="Journal, conference, or repository name")
    citation_count: Optional[int] = Field(default=None, description="Citation count if reported")
    source: str = Field(default="unknown", description="Source provider (openalex, arxiv, semantic_scholar, upload)")
    relevance_score: Optional[float] = Field(default=0.0, description="Normalized relevance score (0-1)")
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of retrieval")
    pdf_path: Optional[str] = Field(default=None, description="Path to locally cached or uploaded PDF")
    has_full_text: bool = Field(default=False, description="Whether full PDF text was extracted")


class PaperAnalysis(BaseModel):
    paper_id: str = Field(..., description="Matching paper identifier")
    paper_title: str = Field(..., description="Title of the analyzed paper")
    source_url: Optional[str] = Field(default=None, description="Reference URL for evidence tracking")
    research_problem: str = Field(default="Not reported in the available text.")
    research_objectives: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    proposed_methodology: str = Field(default="Not reported in the available text.")
    algorithms_and_models: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    datasets_used: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    evaluation_metrics: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    main_findings: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    limitations: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    future_research_directions: List[str] = Field(default_factory=lambda: ["Not reported in the available text."])
    analysis_mode: str = Field(default="heuristic_nlp", description="Method used: llm_ollama, llm_openai, heuristic_nlp")
    confidence_score: float = Field(default=0.7, ge=0.0, le=1.0, description="Confidence in extraction accuracy")
    is_full_text_used: bool = Field(default=False, description="True if full PDF was analyzed, False if abstract only")
    raw_evidence_excerpts: Dict[str, str] = Field(default_factory=dict, description="Excerpts used for verification")


class SupportingEvidence(BaseModel):
    paper_id: str
    paper_title: str
    source_url: Optional[str] = None
    evidence_text: str
    source_section: str = Field(default="abstract", description="abstract or full_text")


class ResearchGap(BaseModel):
    gap_id: str = Field(..., description="Unique gap identifier")
    gap_title: str = Field(..., description="Concise summary title of the research gap")
    category: GapCategory = Field(default=GapCategory.METHODOLOGY, description="Gap classification")
    gap_type: GapType = Field(default=GapType.INFERRED_CROSS_PAPER, description="Explicit limitation or inferred gap")
    description: str = Field(..., description="Detailed explanation of the potential research gap")
    why_it_matters: str = Field(..., description="Scientific or practical significance of this gap")
    supporting_papers: List[SupportingEvidence] = Field(default_factory=list, description="Academic papers evidencing this gap")
    confidence_level: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence assessment")
    confidence_justification: str = Field(..., description="Reasoning behind the confidence level")
    suggested_research_questions: List[str] = Field(default_factory=list, description="Targeted questions for future researchers")
    recommended_experiments: List[str] = Field(default_factory=list, description="Actionable experiments to address the gap")
    verification_status: str = Field(
        default="Verified with source citations",
        description="Audit status of the gap's grounding"
    )


class LiteratureReviewReport(BaseModel):
    topic: str
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    search_methodology: str
    paper_selection_summary: Dict[str, Any]
    papers: List[PaperAnalysis]
    comparison_matrix: List[Dict[str, Any]]
    research_gaps: List[ResearchGap]
    synthesis_and_future_directions: str
    references: List[Dict[str, str]]


class WorkflowState(BaseModel):
    session_id: str
    current_step: str = "initialized"
    logs: List[str] = Field(default_factory=list)
    raw_papers: List[PaperMetadata] = Field(default_factory=list)
    selected_papers: List[PaperMetadata] = Field(default_factory=list)
    analyses: List[PaperAnalysis] = Field(default_factory=list)
    gaps: List[ResearchGap] = Field(default_factory=list)
    report: Optional[LiteratureReviewReport] = None
    errors: List[str] = Field(default_factory=list)

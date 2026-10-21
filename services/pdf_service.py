"""
PDF processing service using PyMuPDF.
Extracts full text, detects standard academic sections (Abstract, Methodology, Limitations, etc.),
and safely processes user-uploaded research PDFs.
"""

import re
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pymupdf
import config

logger = logging.getLogger(__name__)

# Common section headers in scientific papers
SECTION_PATTERNS = {
    "abstract": re.compile(r"^(abstract|summary)", re.IGNORECASE),
    "introduction": re.compile(r"^(1\.?\s*)?introduction", re.IGNORECASE),
    "related_work": re.compile(r"^(2\.?\s*)?(related\s+work|literature\s+review|background)", re.IGNORECASE),
    "methodology": re.compile(r"^(\d\.?\s*)?(methodology|method|methods|proposed\s+approach|architecture|system\s+design)", re.IGNORECASE),
    "experiments": re.compile(r"^(\d\.?\s*)?(experiments|experimental\s+setup|evaluation|results)", re.IGNORECASE),
    "discussion": re.compile(r"^(\d\.?\s*)?(discussion|analysis)", re.IGNORECASE),
    "limitations": re.compile(r"^(\d\.?\s*)?(limitations|threats\s+to\s+validity)", re.IGNORECASE),
    "conclusion": re.compile(r"^(\d\.?\s*)?(conclusion|concluding\s+remarks|future\s+work)", re.IGNORECASE),
}


class PDFService:
    """
    Handles PDF ingestion, text extraction, and section segmenting.
    """

    def __init__(self, papers_dir: Path = config.PAPERS_DIR):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: str | Path, max_pages: int = 25) -> Dict[str, Any]:
        """
        Extracts clean textual content and detects academic sections from a local PDF.
        """
        path = Path(pdf_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"PDF file not found at {path}",
                "text": "",
                "sections": {},
                "page_count": 0,
            }

        try:
            doc = pymupdf.open(str(path))
            total_pages = len(doc)
            page_limit = min(total_pages, max_pages)

            full_text_parts = []
            for page_idx in range(page_limit):
                page = doc[page_idx]
                text = page.get_text("text")
                if text:
                    full_text_parts.append(text)

            doc.close()
            combined_text = "\n\n".join(full_text_parts)

            # Extract sections
            sections = self.detect_sections(combined_text)

            return {
                "success": True,
                "text": combined_text[:config.MAX_FULLTEXT_CHARS],
                "sections": sections,
                "page_count": total_pages,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Failed to process PDF {pdf_path}: {e}")
            return {
                "success": False,
                "error": f"Error parsing PDF: {str(e)}",
                "text": "",
                "sections": {},
                "page_count": 0,
            }

    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Segments full document text into detected scientific sections.
        """
        lines = text.splitlines()
        sections: Dict[str, list] = {}
        current_section = "preamble"
        sections[current_section] = []

        for line in lines:
            line_str = line.strip()
            # Check if this line resembles a section header (short line matching header pattern)
            matched_header = None
            if len(line_str) < 60 and line_str:
                for sec_name, pattern in SECTION_PATTERNS.items():
                    if pattern.search(line_str):
                        matched_header = sec_name
                        break

            if matched_header:
                current_section = matched_header
                if current_section not in sections:
                    sections[current_section] = []
            else:
                sections[current_section].append(line)

        # Collapse sections to strings
        result: Dict[str, str] = {}
        for sec, parts in sections.items():
            content = "\n".join(parts).strip()
            if content:
                result[sec] = content[:4000]

        return result

    def save_uploaded_pdf(self, file_bytes: bytes, filename: str) -> Path:
        """
        Saves user-uploaded PDF bytes safely into data/papers.
        """
        # Clean filename to avoid path traversal
        safe_name = re.sub(r"[^\w\-_.]", "_", filename)
        target_path = self.papers_dir / safe_name
        with open(target_path, "wb") as f:
            f.write(file_bytes)
        return target_path

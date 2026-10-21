# Multi-Agent Literature Review and Research Gap Finder Using LLMs

An academic-grade, end-to-end multi-agent AI framework designed to automate scientific literature retrieval, extract structured methodological dimensions, perform cross-paper comparative synthesis, and discover evidence-grounded research gaps.

Built with **Python**, **Streamlit**, **PyMuPDF**, **Pydantic**, **Pandas**, and **Plotly**. The system operates with **zero paid API dependencies** out of the box, supporting local LLMs (**Ollama**), external OpenAI-compatible endpoints, and a high-precision deterministic **Heuristic NLP Engine**.

---

## 1. System Architecture

The application adopts a decoupled, multi-agent architecture with a custom orchestrator coordinating sequential information flow and Pydantic-validated state transitions.

```
                    ┌───────────────────────────────┐
                    │  User Research Topic / Query  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    Agent 1: Search Agent      │
                    │   (OpenAlex, arXiv, S2 APIs)  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Deduplication & Ranking      │
                    │   (DOI, arXiv ID, Title Sim)  │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   Agent 2: Analysis Agent     │
                    │ (9 Dimensions, PDF Full-Text) │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Cross-Paper Comparison       │
                    │    (Comparative Matrix)       │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │    Agent 3: Gap Agent         │
                    │ (Dataset/Algo/Scaling Gaps)   │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  Evidence Verification Audit  │
                    │ (Anti-Hallucination Grounding)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Final Review Report & Export  │
                    │      (Markdown & PDF)         │
                    └───────────────────────────────┘
```

---

## 2. Multi-Agent Responsibilities

### **Agent 1: Research Paper Search Agent** (`agents/search_agent.py`)
- Queries public scholarly indexes:
  - **OpenAlex API**: Over 250M cataloged works; abstracts dynamically reconstructed from inverted indexes.
  - **arXiv API**: Latest computer science and AI preprints parsed from Atom XML feeds.
  - **Semantic Scholar API**: Graph API with citation counts and open-access PDF links.
- **Deduplication Engine**: Resolves duplicate records using normalized DOIs (`10.xxxx/...`), canonical arXiv IDs (`2308.xxxxx`), and sequence-matcher title similarity.
- **Hybrid Ranking**: Calculates a relevance score combining query keyword density in titles/abstracts, publication recency weighting, and citation impact bonuses.
- **Fault-Tolerant Caching**: Local JSON caching ensures zero duplicated network requests and reproducible evaluations.

### **Agent 2: Research Paper Analysis Agent** (`agents/analysis_agent.py`)
- Analyzes each paper individually using its abstract or full-text PDF.
- Extracts **9 core scientific dimensions**:
  1. *Research problem*
  2. *Research objectives*
  3. *Proposed methodology*
  4. *Algorithms and models used*
  5. *Datasets used*
  6. *Evaluation metrics*
  7. *Main findings and results*
  8. *Explicit limitations (author-reported)*
  9. *Future research directions*
- **Strict Evidence Standard**: If an attribute is missing from the available text, it emits `"Not reported in the available text."` rather than hallucinating facts.
- **PDF Ingestion**: Uses **PyMuPDF** (`pymupdf`) to extract text and segment academic sections (*Abstract, Methodology, Limitations, Results, Conclusion*).

### **Agent 3: Research Gap Identification Agent** (`agents/gap_agent.py`)
- Compares structured analyses across all retrieved papers.
- Discovers and categorizes potential gaps across 6 dimensions:
  - **Dataset**: Lack of diverse benchmarks, synthetic dataset reliance.
  - **Algorithm**: Cascading hallucination, dialogue stagnation, brittle prompting.
  - **Generalization**: Overfitting to curated benchmark environments.
  - **Evaluation**: Absence of standardized, reproducible multi-agent metrics.
  - **Scalability**: Token overhead, quadratic message growth, high inference latency.
  - **Real-World Deployment**: Runtime error recovery failures, sandbox security deficits.
- **Grounding & Transparency**: Every identified gap cites supporting papers with authentic titles, source URLs, and verbatim evidence excerpts.
- **Confidence Scoring**: Assigns **High**, **Medium**, or **Low** confidence based on the quantity and consistency of supporting publications.
- **Future Research Proposals**: Generates targeted research questions and actionable experiment designs.

---

## 3. Evidence Verification & Anti-Hallucination Standards

To maintain academic rigor and project integrity:
- **No Hallucinated Citations**: Papers must exist in the retrieved metadata set with verifiable DOIs or URLs.
- **Explicit vs Inferred Classification**: Distinguishes direct statements made by paper authors from cross-paper comparative inferences.
- **Traceability Audit**: The system maintains an evidence audit trail noting whether evidence was drawn from a full-text PDF or an abstract.
- **Cautions on Novelty**: Avoids claiming a gap is novel without proof; labels findings as *potential research gaps requiring empirical validation*.

---

## 4. LLM & Local Model Support

The architecture is model-agnostic and does not mandate paid APIs or the Gemini API:

| Provider | Description | Setup |
| :--- | :--- | :--- |
| **Heuristic NLP Engine** *(Default)* | High-precision deterministic scientific extractor using rule-based pattern matching and linguistic analysis. | **No configuration needed.** Zero cost, works completely offline. |
| **Local Ollama** | Communicates with a local Ollama instance (`llama3`, `mistral`, `gemma2`). | Set `OLLAMA_BASE_URL` (default `http://localhost:11434`) and choose your model. |
| **OpenAI-Compatible** | Connects to external OpenAI-compatible endpoints (Groq, DeepSeek, LocalAI, vLLM). | Provide `OPENAI_API_BASE` and `OPENAI_API_KEY` in `.env` or Streamlit sidebar. |

---

## 5. Directory Structure

```
multi_agent_literature_review/
│
├── app.py                     # Streamlit multi-tab academic dashboard
├── config.py                  # Global settings, paths, and timeouts
├── requirements.txt           # Project dependencies
├── .env.example               # Template environment configuration
├── conftest.py                # Pytest path configuration
├── README.md                  # Comprehensive documentation
│
├── agents/                    # Multi-agent implementations
│   ├── __init__.py
│   ├── search_agent.py        # Agent 1: Retrieval, Dedup, Ranking
│   ├── analysis_agent.py      # Agent 2: 9-Dimension Paper Analyzer
│   └── gap_agent.py           # Agent 3: Cross-Paper Gap Synthesizer
│
├── orchestration/             # Workflow coordination
│   ├── __init__.py
│   └── workflow.py            # WorkflowEngine coordinating stages
│
├── services/                  # External services and utility engines
│   ├── __init__.py
│   ├── openalex_service.py    # OpenAlex API client
│   ├── arxiv_service.py       # arXiv Atom XML client
│   ├── semantic_scholar_service.py # Semantic Scholar API client
│   ├── pdf_service.py         # PyMuPDF parser & section detector
│   ├── llm_service.py         # Ollama, OpenAI-compatible & Heuristic NLP
│   └── report_service.py      # Markdown & ReportLab PDF generator
│
├── models/                    # Pydantic schemas & state models
│   ├── __init__.py
│   └── schemas.py             # PaperMetadata, PaperAnalysis, ResearchGap
│
├── utils/                     # Verification and math utilities
│   ├── __init__.py
│   ├── deduplication.py       # DOI/arXiv/Title deduplication & ranking
│   └── evidence_verification.py # Claim auditing & citation verification
│
├── data/                      # Persistent storage and sample datasets
│   ├── papers/                # User-uploaded research PDFs
│   ├── reports/               # Compiled PDF and Markdown review reports
│   ├── cache/                 # Local query & analysis JSON caches
│   └── sample/
│       └── sample_papers.json # Curated sample dataset for offline demo
│
└── tests/                     # Unit and integration test suite
    ├── test_search_agent.py
    ├── test_analysis_agent.py
    ├── test_gap_agent.py
    ├── test_evidence_verification.py
    └── test_orchestration.py
```

---

## 6. Installation & Execution Guide (Windows)

### **Prerequisites**
- Python 3.10, 3.11, or 3.12 installed on Windows.
- PowerShell or Windows Command Prompt.

### **Step 1: Clone or Navigate to the Project Directory**
```powershell
cd "C:\Users\Abis\.gemini\antigravity\scratch\multi_agent_literature_review"
```

### **Step 2: Create and Activate a Virtual Environment**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(Note: If PowerShell displays a script execution policy restriction, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### **Step 3: Install Dependencies**
```powershell
pip install -r requirements.txt
```

### **Step 4: Configure Environment Variables (Optional)**
Copy the template configuration file:
```powershell
copy .env.example .env
```
*(Optional: edit `.env` to configure your email for OpenAlex polite pool or local Ollama URL).*

### **Step 5: Run the Test Suite**
Verify all 11 unit and integration tests pass:
```powershell
pytest -v
```

### **Step 6: Launch the Streamlit Dashboard**
```powershell
streamlit run app.py
```
The application will launch automatically in your web browser at:
`http://localhost:8501`

---

## 7. Streamlit Dashboard Features

1. **🔎 1. Search & Ingestion Tab**:
   - Query academic topics across OpenAlex, arXiv, and Semantic Scholar.
   - Filter by year range and max retrieved paper count.
   - Deduplicate results and view relevance scores.
   - Upload custom scientific PDFs for PyMuPDF text extraction.
   - "Load Pre-Curated Sample Dataset" button for offline evaluation.
2. **📄 2. Paper Analyses Tab**:
   - Structured 9-dimension view of each paper.
   - Clear indicators showing whether analysis used a full-text PDF or an abstract.
   - Analysis mode badges (`heuristic_nlp`, `llm_ollama`, `llm_external`).
3. **📊 3. Comparison Matrix Tab**:
   - Interactive, filterable comparative matrix contrasting models, datasets, evaluation metrics, and limitations across papers.
   - Downloadable as CSV.
4. **🔬 4. Research Gaps Tab**:
   - Categorized research gaps with interactive Plotly visual distribution charts.
   - Evidence grounding badges, author limitation statements, and confidence justifications.
   - Suggested research questions and actionable experiment designs.
5. **📑 5. Review Report Tab**:
   - Formatted academic literature review preview.
   - Download full report as clean GitHub-Flavored **Markdown (.md)**.
   - Download publication-ready formatted **PDF (.pdf)** generated using ReportLab.
6. **🛡️ 6. Verification & Logs Tab**:
   - Complete audit trail verifying claim grounding in source texts.
   - Live execution stream showing multi-agent orchestration logs.

---

## 8. Academic Evaluation Notes

- **Data Provenance**: All paper records originate from genuine scholarly APIs (OpenAlex and arXiv) or user-uploaded research PDFs. Zero synthetic or fake citations are introduced.
- **Traceability**: Every research gap explicitly points to its supporting paper title, paper ID, and source URL.
- **Reproducibility**: Search results and analyses are cached locally in `data/cache/` to ensure reproducible results across evaluator demonstrations.

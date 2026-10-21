"""
Streamlit Application: Multi-Agent Literature Review and Research Gap Finder Using LLMs.
An academic AI dashboard coordinating:
- Agent 1: Research Paper Search Agent
- Agent 2: Research Paper Analysis Agent
- Agent 3: Research Gap Identification Agent
with Evidence Verification, Cross-Paper Comparison, and Report Export.
"""

import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import config
from models.schemas import SearchRequest, PaperMetadata, PaperAnalysis
from orchestration.workflow import WorkflowEngine
from services.pdf_service import PDFService

# Page Setup
st.set_page_config(
    page_title="Multi-Agent Literature Review & Research Gap Finder",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.8rem;
    }
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-high { background-color: #dcfce7; color: #166534; }
    .badge-medium { background-color: #fef9c3; color: #854d0e; }
    .badge-low { background-color: #fee2e2; color: #991b1b; }
    .badge-mode { background-color: #e0f2fe; color: #0369a1; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Session State Initialization
if "engine" not in st.session_state:
    st.session_state.engine = WorkflowEngine(llm_provider=config.DEFAULT_LLM_PROVIDER)
if "selected_papers" not in st.session_state:
    st.session_state.selected_papers = []
if "analyses" not in st.session_state:
    st.session_state.analyses = []
if "gaps" not in st.session_state:
    st.session_state.gaps = []
if "report" not in st.session_state:
    st.session_state.report = None
if "current_topic" not in st.session_state:
    st.session_state.current_topic = "LLM Multi-Agent Collaboration"
if "verification_audit" not in st.session_state:
    st.session_state.verification_audit = []
if "search_metrics" not in st.session_state:
    st.session_state.search_metrics = {}

pdf_service = PDFService()

# ==============================================================================
# SIDEBAR: Configuration & Model Provider Settings
# ==============================================================================
with st.sidebar:
    st.title("⚙️ System Control")
    st.markdown("### Model & Agent Settings")

    llm_choice = st.selectbox(
        "Analysis & Gap LLM Provider",
        options=["Heuristic NLP (Offline / No API Key)", "Local Ollama", "OpenAI-Compatible API"],
        index=0 if config.DEFAULT_LLM_PROVIDER == "heuristic" else (1 if config.DEFAULT_LLM_PROVIDER == "ollama" else 2),
        help="Select between local Ollama LLM, external API, or the built-in deterministic Heuristic NLP engine."
    )

    provider_internal_name = "heuristic"
    if "Ollama" in llm_choice:
        provider_internal_name = "ollama"
        ollama_url = st.text_input("Ollama Base URL", value=config.OLLAMA_BASE_URL)
        ollama_model = st.text_input("Ollama Model Name", value=config.OLLAMA_MODEL)
        config.OLLAMA_BASE_URL = ollama_url
        config.OLLAMA_MODEL = ollama_model
        # Check health
        info = st.session_state.engine.analysis_agent.llm_service.get_active_provider_info()
        if info["ollama_available"]:
            st.success(f"🟢 Ollama Online. Models: {', '.join(info['ollama_models']) or 'None loaded'}")
        else:
            st.warning("⚠️ Ollama not detected at this URL. Will fallback gracefully to Heuristic NLP.")

    elif "OpenAI" in llm_choice:
        provider_internal_name = "openai_compatible"
        api_base = st.text_input("API Base URL", value=config.OPENAI_API_BASE or "https://api.openai.com/v1")
        api_key = st.text_input("API Key", value=config.OPENAI_API_KEY, type="password")
        model_name = st.text_input("Model Name", value=config.OPENAI_MODEL_NAME)
        config.OPENAI_API_BASE = api_base
        config.OPENAI_API_KEY = api_key
        config.OPENAI_MODEL_NAME = model_name
    else:
        st.info("ℹ️ Using Heuristic NLP Engine. Operates 100% locally with high precision and zero API costs.")

    st.session_state.engine.set_llm_provider(provider_internal_name)

    st.markdown("---")
    st.markdown("### Demonstration & Data")
    if st.button("📂 Load Pre-Curated Sample Dataset", use_container_width=True):
        sample_papers = st.session_state.engine.search_agent.load_sample_papers()
        st.session_state.selected_papers = sample_papers
        st.session_state.current_topic = "Large Language Model Multi-Agent Systems"
        st.success(f"Loaded {len(sample_papers)} verified multi-agent papers!")
        st.rerun()

    st.markdown("---")
    st.caption("Developed for Academic Literature Synthesis & Research Gap Auditing.")


# ==============================================================================
# MAIN PAGE HEADER
# ==============================================================================
st.markdown("<div class='main-header'>Multi-Agent Literature Review and Research Gap Finder</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Powered by an agentic pipeline: <b>Search Agent</b> → <b>Deduplication & Ranking</b> → "
    "<b>Analysis Agent</b> → <b>Research Gap Agent</b> → <b>Evidence Verification</b> → <b>Synthesis Report</b>.</div>",
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔎 1. Search & Ingestion",
    "📄 2. Paper Analyses",
    "📊 3. Comparison Matrix",
    "🔬 4. Research Gaps",
    "📑 5. Review Report",
    "🛡️ 6. Verification & Logs"
])

# ==============================================================================
# TAB 1: Search & Retrieval
# ==============================================================================
with tab1:
    st.subheader("1. Scholarly Literature Search")
    st.markdown("Query open scholarly repositories to retrieve verified papers and metadata.")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query_input = st.text_input(
            "Research Topic or Research Question",
            value=st.session_state.current_topic,
            placeholder="e.g. LLM Multi-Agent Collaboration and Dialogue Reasoning"
        )
    with col2:
        paper_limit = st.number_input("Max Papers", min_value=2, max_value=25, value=5, step=1)
    with col3:
        year_range = st.slider("Publication Years", min_value=2015, max_value=2026, value=(2020, 2026))

    col_src1, col_src2 = st.columns([3, 2])
    with col_src1:
        source_options = st.multiselect(
            "Scholarly API Sources",
            options=["openalex", "arxiv", "semantic_scholar"],
            default=["openalex", "arxiv"],
            format_func=lambda s: {"openalex": "OpenAlex (Open)", "arxiv": "arXiv (Preprints)", "semantic_scholar": "Semantic Scholar"}.get(s, s)
        )
    with col_src2:
        use_sample_fallback = st.checkbox("Allow Sample Fallback if APIs are offline", value=True)

    col_btn1, col_btn2 = st.columns([2, 3])
    with col_btn1:
        start_search = st.button("🚀 Retrieve Scholarly Papers", type="primary", use_container_width=True)

    if start_search:
        if not query_input.strip():
            st.error("Please enter a valid research topic.")
        else:
            st.session_state.current_topic = query_input
            with st.spinner("Searching scholarly APIs, deduplicating, and ranking..."):
                search_req = SearchRequest(
                    query=query_input,
                    limit=paper_limit,
                    start_year=year_range[0],
                    end_year=year_range[1],
                    sources=source_options or ["openalex", "arxiv"]
                )
                try:
                    papers, metrics = st.session_state.engine.run_search_stage(
                        search_req,
                        use_sample_fallback=use_sample_fallback
                    )
                    st.session_state.selected_papers = papers
                    st.session_state.search_metrics = metrics
                    st.success(f"Retrieved {len(papers)} ranked papers! Ready for individual analysis.")
                except Exception as e:
                    st.error(f"Search retrieval error: {e}")

    # Display Search Results
    if st.session_state.selected_papers:
        st.markdown("---")
        st.markdown(f"### Retrieved Papers ({len(st.session_state.selected_papers)})")
        
        # Summary metrics
        if st.session_state.search_metrics:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Retrieved Records", st.session_state.search_metrics.get("total_retrieved", 0))
            m2.metric("Duplicates Removed", st.session_state.search_metrics.get("duplicates_removed", 0))
            m3.metric("Final Ranked Papers", len(st.session_state.selected_papers))
            m4.metric("Sources Used", ", ".join(st.session_state.search_metrics.get("sources_queried", [])))

        for idx, paper in enumerate(st.session_state.selected_papers, 1):
            with st.expander(f"**{idx}. {paper.title}** ({paper.publication_year or 'n.d.'}) — Relevance: {paper.relevance_score:.2f}", expanded=(idx <= 2)):
                st.markdown(f"**Authors**: {', '.join(paper.authors) if paper.authors else 'Not reported'}")
                st.markdown(f"**Venue**: {paper.venue} | **Source**: `{paper.source}` | **Citations**: {paper.citation_count or 'N/A'}")
                if paper.url:
                    st.markdown(f"**Link**: [{paper.url}]({paper.url})")
                if paper.doi:
                    st.markdown(f"**DOI**: [{paper.doi}](https://doi.org/{paper.doi})")
                st.markdown(f"**Abstract**:\n{paper.abstract or 'No abstract provided in metadata.'}")

        # PDF Upload Section
        st.markdown("---")
        st.markdown("### 📤 Upload Custom Research PDF")
        uploaded_pdf = st.file_uploader("Upload an academic PDF to analyze along with retrieved papers", type=["pdf"])
        if uploaded_pdf is not None:
            if st.button("Process Uploaded PDF"):
                with st.spinner("Extracting text and scientific sections using PyMuPDF..."):
                    saved_path = pdf_service.save_uploaded_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
                    extracted = pdf_service.extract_text_from_pdf(saved_path)
                    if extracted["success"]:
                        custom_paper = PaperMetadata(
                            paper_id=f"upload_{uploaded_pdf.name}",
                            title=uploaded_pdf.name.replace(".pdf", ""),
                            authors=["User Contributed"],
                            publication_year=2024,
                            abstract=extracted.get("sections", {}).get("abstract", extracted["text"][:1000]),
                            url=None,
                            source="user_upload",
                            pdf_path=str(saved_path),
                            has_full_text=True,
                            relevance_score=1.0,
                        )
                        st.session_state.selected_papers.append(custom_paper)
                        st.success(f"Extracted {extracted['page_count']} pages! Added '{uploaded_pdf.name}' to the analysis pool.")
                        st.rerun()
                    else:
                        st.error(f"Failed to process PDF: {extracted.get('error')}")

        st.markdown("---")
        if st.button("⏩ Proceed to Stage 2: Analyze Retrieved Papers", type="primary"):
            st.session_state.analyses = []
            with st.spinner("Agent 2 is analyzing each paper..."):
                analyses = st.session_state.engine.run_analysis_stage(st.session_state.selected_papers)
                st.session_state.analyses = analyses
                st.success(f"Successfully analyzed {len(analyses)} papers!")
                st.rerun()

# ==============================================================================
# TAB 2: Individual Paper Analyses
# ==============================================================================
with tab2:
    st.subheader("2. Structured Paper Analyses (Agent 2)")
    st.markdown("Detailed 9-dimension academic breakdown generated by the Analysis Agent.")

    if not st.session_state.analyses:
        st.info("No papers have been analyzed yet. Run search in Tab 1 or click below to analyze.")
        if st.session_state.selected_papers:
            if st.button("Start Paper Analysis", type="primary"):
                with st.spinner("Analyzing papers..."):
                    analyses = st.session_state.engine.run_analysis_stage(st.session_state.selected_papers)
                    st.session_state.analyses = analyses
                    st.rerun()
    else:
        for idx, analysis in enumerate(st.session_state.analyses, 1):
            with st.expander(f"**{idx}. {analysis.paper_title}**", expanded=(idx == 1)):
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.markdown(f"**Mode**: `<span class='metric-badge badge-mode'>{analysis.analysis_mode}</span>`", unsafe_allow_html=True)
                col_m2.markdown(f"**Confidence**: {int(analysis.confidence_score * 100)}%")
                col_m3.markdown(f"**Full PDF Used**: {'Yes' if analysis.is_full_text_used else 'Abstract Only'}")

                if analysis.source_url:
                    st.markdown(f"🔗 **Source**: [{analysis.source_url}]({analysis.source_url})")

                st.markdown(f"**1. Research Problem**:\n> {analysis.research_problem}")
                st.markdown(f"**2. Research Objectives**:")
                for obj in analysis.research_objectives:
                    st.markdown(f"- {obj}")

                st.markdown(f"**3. Proposed Methodology**:\n{analysis.proposed_methodology}")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**4. Algorithms & Models**: {', '.join(analysis.algorithms_and_models)}")
                    st.markdown(f"**5. Datasets Used**: {', '.join(analysis.datasets_used)}")
                with col_b:
                    st.markdown(f"**6. Evaluation Metrics**: {', '.join(analysis.evaluation_metrics)}")

                st.markdown(f"**7. Main Findings & Results**:")
                for finding in analysis.main_findings:
                    st.markdown(f"- {finding}")

                st.markdown(f"**8. Explicit Limitations (Author-Reported)**:")
                for lim in analysis.limitations:
                    st.markdown(f"- ⚠️ {lim}")

                st.markdown(f"**9. Future Research Directions**:")
                for fut in analysis.future_research_directions:
                    st.markdown(f"- 🔮 {fut}")

        st.markdown("---")
        if st.button("⏩ Proceed to Stage 3 & 4: Cross-Paper Gap Analysis", type="primary"):
            with st.spinner("Agent 3 synthesizing cross-paper gaps..."):
                gaps = st.session_state.engine.run_gap_stage(st.session_state.current_topic, st.session_state.analyses)
                verified_gaps, audits = st.session_state.engine.run_verification_stage(
                    gaps, st.session_state.selected_papers, st.session_state.analyses
                )
                st.session_state.gaps = verified_gaps
                st.session_state.verification_audit = audits
                st.success(f"Synthesized and verified {len(verified_gaps)} potential research gaps!")
                st.rerun()

# ==============================================================================
# TAB 3: Comparison Matrix
# ==============================================================================
with tab3:
    st.subheader("3. Cross-Paper Literature Comparison Matrix")
    st.markdown("Comparative matrix displaying methodology, model selection, dataset utilization, and limitations.")

    if not st.session_state.analyses:
        st.info("Please analyze papers in Tab 1 or Tab 2 first.")
    else:
        matrix_data = st.session_state.engine.build_comparison_matrix(st.session_state.analyses)
        df_matrix = pd.DataFrame(matrix_data)
        
        # Display styled table
        st.dataframe(
            df_matrix[["Paper", "Model", "Dataset", "Evaluation Metrics", "Limitations", "Future Work"]],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            label="📥 Download Comparison Matrix (CSV)",
            data=df_matrix.to_csv(index=False),
            file_name="literature_comparison_matrix.csv",
            mime="text/csv",
        )

# ==============================================================================
# TAB 4: Research Gap Dashboard
# ==============================================================================
with tab4:
    st.subheader("4. Grounded Research Gap Identification (Agent 3)")
    st.markdown(
        "Potential research gaps synthesized through cross-paper comparison and explicit author limitations. "
        "Every claim is grounded in retrieved paper evidence."
    )

    if not st.session_state.gaps:
        st.info("No research gaps synthesized yet.")
        if st.session_state.analyses:
            if st.button("Synthesize Research Gaps Now", type="primary"):
                with st.spinner("Synthesizing gaps and verifying evidence..."):
                    gaps = st.session_state.engine.run_gap_stage(st.session_state.current_topic, st.session_state.analyses)
                    verified_gaps, audits = st.session_state.engine.run_verification_stage(
                        gaps, st.session_state.selected_papers, st.session_state.analyses
                    )
                    st.session_state.gaps = verified_gaps
                    st.session_state.verification_audit = audits
                    st.rerun()
    else:
        # Plotly Gap Distribution Visualizations
        gap_cats = [g.category.value for g in st.session_state.gaps]
        gap_confs = [g.confidence_level.value for g in st.session_state.gaps]

        vcol1, vcol2 = st.columns(2)
        with vcol1:
            fig_cat = px.bar(
                x=list(pd.Series(gap_cats).value_counts().index),
                y=list(pd.Series(gap_cats).value_counts().values),
                labels={"x": "Gap Category", "y": "Count"},
                title="Gaps by Academic Dimension",
                color_discrete_sequence=["#3b82f6"]
            )
            fig_cat.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_cat, use_container_width=True)

        with vcol2:
            fig_conf = px.pie(
                names=gap_confs,
                title="Evidence Grounding Confidence",
                color=gap_confs,
                color_discrete_map={"High": "#22c55e", "Medium": "#eab308", "Low": "#ef4444"}
            )
            fig_conf.update_layout(height=280, margin=dict(l=20, r=20, t=35, b=20))
            st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown("---")

        for idx, gap in enumerate(st.session_state.gaps, 1):
            badge_class = f"badge-{gap.confidence_level.value.lower()}"
            st.markdown(f"""
            <div class='card'>
                <h4>Gap {idx}: {gap.gap_title}</h4>
                <p>
                    <span class='metric-badge {badge_class}'>Confidence: {gap.confidence_level.value}</span>
                    <span class='metric-badge badge-mode'>Category: {gap.category.value}</span>
                    <span class='metric-badge' style='background:#f1f5f9; color:#475569;'>Type: {gap.gap_type.value}</span>
                    <span class='metric-badge' style='background:#f0fdf4; color:#15803d;'>Status: {gap.verification_status}</span>
                </p>
                <p><b>Description</b>: {gap.description}</p>
                <p><b>Why It Matters</b>: {gap.why_it_matters}</p>
                <p><b>Confidence Justification</b>: <i>{gap.confidence_justification}</i></p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"🔍 Grounding Evidence & Experiment Proposals for Gap {idx}"):
                st.markdown("**Supporting Evidence from Literature**:")
                for ev in gap.supporting_papers:
                    link = f"[{ev.paper_title}]({ev.source_url})" if ev.source_url else ev.paper_title
                    st.markdown(f"- 📄 **{link}** (*Source: {ev.source_section}*)")
                    st.markdown(f"  > \"{ev.evidence_text}\"")

                if gap.suggested_research_questions:
                    st.markdown("**Suggested Research Questions**:")
                    for q in gap.suggested_research_questions:
                        st.markdown(f"- ❓ {q}")

                if gap.recommended_experiments:
                    st.markdown("**Recommended Future Experiments**:")
                    for exp in gap.recommended_experiments:
                        st.markdown(f"- 🧪 {exp}")

# ==============================================================================
# TAB 5: Review Report & Export
# ==============================================================================
with tab5:
    st.subheader("5. Synthesized Literature Review Report")
    st.markdown("Compile and export a publication-ready literature review with verified references.")

    if not st.session_state.report:
        if st.session_state.selected_papers and st.session_state.analyses:
            if st.button("Generate Final Literature Review Report", type="primary"):
                with st.spinner("Synthesizing comprehensive review report..."):
                    report = st.session_state.engine.run_report_stage(
                        topic=st.session_state.current_topic,
                        papers=st.session_state.selected_papers,
                        analyses=st.session_state.analyses,
                        gaps=st.session_state.gaps
                    )
                    st.session_state.report = report
                    st.rerun()
        else:
            st.info("Run literature search and analysis first to compile the final report.")
    else:
        report = st.session_state.report
        markdown_content = st.session_state.engine.report_service.generate_markdown_report(report)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                label="📥 Download Report as Markdown (.md)",
                data=markdown_content,
                file_name=f"literature_review_{st.session_state.current_topic.replace(' ', '_')[:25]}.md",
                mime="text/markdown",
                use_container_width=True
            )
        with col_d2:
            try:
                pdf_filename = f"literature_review_{st.session_state.current_topic.replace(' ', '_')[:20]}.pdf"
                pdf_file_path = st.session_state.engine.report_service.generate_pdf_report(report, pdf_filename)
                with open(pdf_file_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    label="📄 Download Report as Formatted PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=pdf_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"PDF compilation notice: {e}")

        st.markdown("---")
        st.markdown("### Report Preview")
        st.markdown(markdown_content)

# ==============================================================================
# TAB 6: Evidence Verification & System Logs
# ==============================================================================
with tab6:
    st.subheader("6. Evidence Verification Audit & Traceability")
    st.markdown(
        "Audit log validating that extracted claims, gap assertions, and author citations originate "
        "from verified publications rather than hallucinated facts."
    )

    if st.session_state.verification_audit:
        for idx, audit in enumerate(st.session_state.verification_audit, 1):
            with st.expander(f"Audit Trail for Gap {idx}: {audit.get('gap_title')}"):
                st.markdown(f"**Category**: `{audit.get('category')}` | **Type**: `{audit.get('gap_type')}`")
                st.markdown(f"**Verified Supporting Papers**: {len(audit.get('verified_papers', []))}")
                for vp in audit.get("verified_papers", []):
                    st.markdown(f"- ✅ **{vp['title']}** (Status: `{vp['status']}` | Source: `{vp['source_type']}`)")
                    if vp.get("url"):
                        st.markdown(f"  URL: [{vp['url']}]({vp['url']})")

                if audit.get("notes"):
                    st.markdown("**Verification Notes**:")
                    for note in audit.get("notes", []):
                        st.markdown(f"- ℹ️ {note}")
    else:
        st.info("Run the research gap stage to generate verification audit logs.")

    st.markdown("---")
    st.markdown("### Multi-Agent Orchestration Logs")
    log_text = "\n".join(st.session_state.engine.state.logs)
    st.text_area("Execution Log Stream", value=log_text or "No execution logs yet.", height=240, disabled=True)

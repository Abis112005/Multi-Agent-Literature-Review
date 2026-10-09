
import streamlit as st
from agents.search_agent import search_papers
from agents.analysis_agent import analyze_paper
from agents.gap_agent import find_research_gaps


st.title("Multi-Agent Literature Review & Research Gap Finder")

st.write(
    "Enter a research topic to search, analyze, and identify research gaps."
)


topic = st.text_input(
    "Enter your research topic",
    placeholder="Example: LiDAR 3D object detection in snowy weather"
)


if st.button("Start Literature Review"):

    if topic:

        # =====================================
        # AGENT 1: SEARCH AGENT
        # =====================================

        with st.spinner("Search Agent is finding relevant papers..."):

            papers = search_papers(topic, max_results=5)

        st.success(f"Search Agent found {len(papers)} papers!")

        # =====================================
        # AGENT 2: ANALYSIS AGENT
        # =====================================

        st.subheader("Literature Analysis")

        paper_analyses = []

        for i, paper in enumerate(papers, start=1):

            st.markdown(f"## {i}. {paper['title']}")

            st.write(f"**Year:** {paper['year']}")

            with st.spinner(
                f"Analysis Agent is analyzing paper {i}..."
            ):

                analysis = analyze_paper(paper)

            paper_analyses.append(
                f"""
Paper Title: {paper['title']}
Year: {paper['year']}

{analysis}
"""
            )

            st.markdown("### Analysis")

            st.write(analysis)

            st.divider()

        # =====================================
        # AGENT 3: GAP AGENT
        # =====================================

        st.subheader("Research Gap Identification")

        with st.spinner(
            "Gap Agent is comparing the papers and identifying research gaps..."
        ):

            research_gaps = find_research_gaps(paper_analyses)

        st.success("Gap Agent completed the research gap analysis!")

        st.markdown("### Identified Research Gaps")

        st.write(research_gaps)

    else:

        st.warning("Please enter a research topic.")

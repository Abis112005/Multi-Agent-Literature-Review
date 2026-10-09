
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

    if not topic.strip():
        st.warning("Please enter a research topic.")

    else:
        try:
            # AGENT 1: SEARCH
            with st.spinner("Search Agent is finding relevant papers..."):
                papers = search_papers(topic.strip(), max_results=2)

            if not papers:
                st.warning("No papers found. Try another research topic.")
                st.stop()

            st.success(f"Search Agent found {len(papers)} papers!")

            # AGENT 2: ANALYSIS
            st.subheader("Literature Analysis")
            paper_analyses = []

            for i, paper in enumerate(papers, start=1):
                title = paper.get("title", "Title not available")
                year = paper.get("year", "Year not available")

                st.markdown(f"## {i}. {title}")
                st.write(f"**Year:** {year}")

                try:
                    with st.spinner(f"Analyzing paper {i} of {len(papers)}..."):
                        analysis = analyze_paper(paper)

                    paper_analyses.append(
                        f"Paper Title: {title}\nYear: {year}\n{analysis}"
                    )

                    st.markdown("### Analysis")
                    st.write(analysis)

                except Exception as e:
                    st.error(f"Could not analyze this paper: {e}")
                    st.info("Please try again later if the AI service is temporarily unavailable.")

                st.divider()

            # AGENT 3: GAP IDENTIFICATION
            if paper_analyses:
                st.subheader("Research Gap Identification")

                try:
                    with st.spinner("Gap Agent is identifying research gaps..."):
                        research_gaps = find_research_gaps(paper_analyses)

                    st.success("Gap Agent completed the research gap analysis!")
                    st.markdown("### Identified Research Gaps")
                    st.write(research_gaps)

                except Exception as e:
                    st.error(f"Research gap identification failed: {e}")
            else:
                st.warning("No papers could be analyzed, so research gaps could not be identified.")

        except Exception as e:
            st.error(f"The literature review could not be completed: {e}")
            st.info("Check the app logs for details and try again.")


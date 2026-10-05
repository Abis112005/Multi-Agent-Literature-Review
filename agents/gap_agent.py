import os
from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def find_research_gaps(paper_analyses):
    """
    Compare multiple paper analyses and identify
    evidence-grounded research gaps.
    """

    combined_analysis = ""

    for i, analysis in enumerate(paper_analyses, start=1):
        combined_analysis += f"""
        
===== PAPER {i} =====

{analysis}

"""

    prompt = f"""
You are a research gap identification agent.

You are given structured analyses of multiple research papers.

Your task is to compare the papers and identify
potential research gaps that are supported by the evidence.

{combined_analysis}

For each candidate research gap, provide:

1. Research Gap
2. Evidence from Papers
3. Supporting Papers
4. Contradictory Evidence
5. Evidence Status
6. Possible Research Direction

Important rules:

- Do not invent facts.
- Do not claim something is a research gap unless the provided
  papers give evidence for it.
- If evidence is insufficient, clearly state:
  "Insufficient evidence."
- If papers disagree, mention the disagreement.
- Focus on gaps related to datasets, methods, evaluation,
  environmental conditions, model comparisons, or unexplored areas.

Use clear headings and concise explanations.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text

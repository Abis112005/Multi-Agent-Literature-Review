import os
from dotenv import load_dotenv
from google import genai


# Load environment variables from .env
load_dotenv()

# Create Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_paper(paper):
    """
    Use Gemini to analyze a research paper abstract
    and extract structured research information.
    """

    title = paper.get("title", "Title not available")
    abstract = paper.get("abstract", "Abstract not available")

    prompt = f"""
You are a research literature analysis agent.

Analyze the following research paper.

Paper Title:
{title}

Abstract:
{abstract}

Extract the following information:

1. Research Problem
2. Dataset
3. Method or Model
4. Evaluation Metrics
5. Limitations
6. Future Work

Return the answer in a clear format with these six headings.

If information is not available in the abstract, write:
"Not mentioned in the abstract."

Do not invent information.
"""

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text


if __name__ == "__main__":

    test_paper = {
        "title": "LiDAR Snowfall Simulation for Robust 3D Object Detection",
        "abstract": """
        Autonomous driving systems using LiDAR sensors can be affected by
        snowfall. This paper presents a method for simulating snowfall on
        real LiDAR point clouds and evaluates its effect on 3D object
        detection under adverse weather conditions.
        """
    }

    result = analyze_paper(test_paper)

    print("\n===== ANALYSIS RESULT =====\n")
    print(result)

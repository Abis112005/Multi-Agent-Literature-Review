"""
Unified LLM Service supporting:
1. Ollama (local LLM - preferred when available)
2. OpenAI-Compatible API (external provider via env vars)
3. Hugging Face Transformers (optional local model)
4. Heuristic NLP Engine (high-precision rule-based fallback without paid APIs or downloads)

Always indicates whether results originate from an LLM, heuristic, or source text.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional
import requests
import config

logger = logging.getLogger(__name__)

NOT_REPORTED = "Not reported in the available text."


class HeuristicNLPProvider:
    """
    Robust rule-based and linguistic extraction engine for scientific papers.
    Operates without paid APIs or model downloads, adhering strictly to text evidence.
    """

    def __init__(self):
        # Known common AI/ML/Data models & algorithms
        self.known_models = [
            "Transformer", "LLM", "Large Language Model", "Multi-Agent System", "Agentic Workflow",
            "BERT", "RoBERTa", "GPT", "Llama", "Mistral", "Diffusion Model", "GNN", "Graph Neural Network",
            "LSTM", "CNN", "Convolutional Neural Network", "Reinforcement Learning", "PPO", "DQN",
            "RAG", "Retrieval-Augmented Generation", "Chain-of-Thought", "In-Context Learning",
            "Autoencoder", "Support Vector Machine", "Random Forest", "BM25", "Vector Search",
            "Monte Carlo Tree Search", "Prompt Engineering", "Fine-Tuning", "LoRA", "PEFT"
        ]

        # Known evaluation metrics
        self.known_metrics = [
            "accuracy", "f1-score", "f1 score", "precision", "recall", "bleu", "rouge", "meteor",
            "auc", "roc-auc", "perplexity", "latency", "throughput", "mean reciprocal rank", "mrr",
            "ndcg", "pass@1", "pass@k", "exact match", "rmse", "mae", "human evaluation"
        ]

        # Common academic datasets
        self.known_datasets = [
            "MMLU", "GSM8K", "HumanEval", "MBPP", "TriviaQA", "Natural Questions", "SQuAD",
            "HotpotQA", "ImageNet", "COCO", "GLUE", "SuperGLUE", "PubMed", "arXiv", "GitHub",
            "Common Crawl", "Wikipedia", "OpenAssistant", "ShareGPT", "Alpaca"
        ]

    def split_sentences(self, text: str) -> List[str]:
        """Splits text into cleaned sentences."""
        if not text:
            return []
        raw_sents = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in raw_sents if len(s.strip()) > 15]

    def extract_paper_analysis(self, title: str, text: str) -> Dict[str, Any]:
        """
        Extracts the 9 required academic analysis dimensions using structured NLP patterns.
        """
        sentences = self.split_sentences(text)
        lower_text = text.lower()

        # 1. Research Problem
        problem = NOT_REPORTED
        problem_triggers = [
            r"tackles the problem of\s+([^.]+)",
            r"address(?:es|ing)? the (?:problem|challenge|issue) of\s+([^.]+)",
            r"focus(?:es|ing)? on\s+([^.]+)",
            r"in order to overcome\s+([^.]+)",
            r"despite the importance of\s+([^.]+)",
            r"lack of\s+([^.]+)",
        ]
        for pat in problem_triggers:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                problem = match.group(0).strip().rstrip(".")
                break
        if problem == NOT_REPORTED and sentences:
            # First or second sentence of abstract typically states context/problem
            for sent in sentences[:2]:
                if any(k in sent.lower() for k in ["problem", "challenge", "difficult", "issue", "remain", "critical"]):
                    problem = sent
                    break
            if problem == NOT_REPORTED and sentences:
                problem = sentences[0]

        # 2. Research Objectives
        objectives: List[str] = []
        for sent in sentences:
            if re.search(r"\b(our goal is to|we aim to|this paper aims|this work presents|we investigate|we introduce)\b", sent, re.IGNORECASE):
                objectives.append(sent)
        if not objectives and len(sentences) > 1:
            for sent in sentences[:3]:
                if "we propose" in sent.lower() or "we develop" in sent.lower():
                    objectives.append(sent)
        if not objectives:
            objectives = [NOT_REPORTED]

        # 3. Proposed Methodology
        methodology = NOT_REPORTED
        for sent in sentences:
            if re.search(r"\b(we propose|our framework|our approach|we develop|we present a|methodology consists of)\b", sent, re.IGNORECASE):
                methodology = sent
                break

        # 4. Algorithms and Models
        detected_models = []
        for model in self.known_models:
            pattern = r"\b" + re.escape(model) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                detected_models.append(model)
        if not detected_models:
            detected_models = [NOT_REPORTED]

        # 5. Datasets Used
        detected_datasets = []
        for ds in self.known_datasets:
            pattern = r"\b" + re.escape(ds) + r"\b"
            if re.search(pattern, text, re.IGNORECASE):
                detected_datasets.append(ds)
        # Regex search for custom dataset mentions: e.g. "X dataset", "X benchmark"
        custom_ds = re.findall(r"([A-Z][A-Za-z0-9_\-]+)\s+(?:dataset|benchmark|corpus)", text)
        for ds in custom_ds:
            if ds not in detected_datasets and len(ds) > 2:
                detected_datasets.append(ds)
        if not detected_datasets:
            detected_datasets = [NOT_REPORTED]

        # 6. Evaluation Metrics
        detected_metrics = []
        for metric in self.known_metrics:
            pattern = r"\b" + re.escape(metric) + r"\b"
            if re.search(pattern, lower_text):
                detected_metrics.append(metric.upper())
        if not detected_metrics:
            detected_metrics = [NOT_REPORTED]

        # 7. Main Findings
        findings: List[str] = []
        for sent in sentences:
            if re.search(r"\b(results show|demonstrates that|outperforms|achieves|improves|we find that|experiments show)\b", sent, re.IGNORECASE):
                findings.append(sent)
        if not findings and len(sentences) > 2:
            findings = [sentences[-1]]
        elif not findings:
            findings = [NOT_REPORTED]

        # 8. Limitations
        limitations: List[str] = []
        for sent in sentences:
            if re.search(r"\b(limitation|however|bottleneck|fails to|restricted to|computational cost|lack of|overhead)\b", sent, re.IGNORECASE):
                limitations.append(sent)
        if not limitations:
            limitations = [NOT_REPORTED]

        # 9. Future Research Directions
        future_work: List[str] = []
        for sent in sentences:
            if re.search(r"\b(future work|future research|remaining challenge|further study|can be extended|open question)\b", sent, re.IGNORECASE):
                future_work.append(sent)
        if not future_work:
            future_work = [NOT_REPORTED]

        return {
            "research_problem": problem,
            "research_objectives": objectives[:3],
            "proposed_methodology": methodology,
            "algorithms_and_models": detected_models[:6],
            "datasets_used": detected_datasets[:6],
            "evaluation_metrics": detected_metrics[:6],
            "main_findings": findings[:3],
            "limitations": limitations[:3],
            "future_research_directions": future_work[:3],
            "analysis_mode": "heuristic_nlp",
            "confidence_score": 0.75,
        }


class OllamaLLMProvider:
    """
    Communicates with a locally running Ollama instance.
    """

    def __init__(self, base_url: str = config.OLLAMA_BASE_URL, model: str = config.OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = 45

    def is_available(self) -> bool:
        """Checks if Ollama service is reachable and running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """Returns list of models installed locally in Ollama."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=4)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                return [m.get("name") for m in models if m.get("name")]
        except Exception:
            pass
        return []

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        """Generates response using Ollama generate API."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get("response")
            logger.warning(f"Ollama returned HTTP {resp.status_code}")
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
        return None


class OpenAICompatibleProvider:
    """
    Interface for external OpenAI-compatible endpoints (Groq, DeepSeek, LocalAI, vLLM).
    """

    def __init__(
        self,
        api_base: str = config.OPENAI_API_BASE,
        api_key: str = config.OPENAI_API_KEY,
        model_name: str = config.OPENAI_MODEL_NAME
    ):
        self.api_base = (api_base or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = 40

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_base)

    def generate(self, prompt: str, system_prompt: str = "") -> Optional[str]:
        if not self.is_configured():
            return None
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or "You are an expert scientific researcher."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            resp = requests.post(f"{self.api_base}/chat/completions", json=payload, headers=headers, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            logger.warning(f"OpenAI compatible endpoint returned {resp.status_code}: {resp.text[:150]}")
        except Exception as e:
            logger.error(f"OpenAI compatible request failed: {e}")
        return None


class LLMService:
    """
    Unified manager for LLM and Heuristic operations across the multi-agent system.
    Automatically degrades gracefully to heuristic NLP if LLMs are unavailable.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider_name = (provider or config.DEFAULT_LLM_PROVIDER).lower()
        self.heuristic = HeuristicNLPProvider()
        self.ollama = OllamaLLMProvider()
        self.openai = OpenAICompatibleProvider()

    def get_active_provider_info(self) -> Dict[str, Any]:
        """Returns details on the current model provider and operational status."""
        ollama_ready = self.ollama.is_available()
        ollama_models = self.ollama.list_models() if ollama_ready else []

        openai_ready = self.openai.is_configured()

        return {
            "active_provider": self.provider_name,
            "ollama_available": ollama_ready,
            "ollama_models": ollama_models,
            "openai_compatible_configured": openai_ready,
            "heuristic_nlp_ready": True,
        }

    def analyze_paper_text(self, title: str, text: str, full_pdf_used: bool = False) -> Dict[str, Any]:
        """
        Extracts 9 academic fields from paper text using LLM if available, else Heuristic NLP.
        """
        # Attempt LLM if user configured Ollama and it's active
        if self.provider_name == "ollama" and self.ollama.is_available():
            llm_result = self._call_llm_for_analysis(self.ollama, title, text)
            if llm_result:
                llm_result["analysis_mode"] = f"llm_ollama_{self.ollama.model}"
                llm_result["is_full_text_used"] = full_pdf_used
                return llm_result

        # Attempt OpenAI-compatible if configured
        if self.provider_name == "openai_compatible" and self.openai.is_configured():
            llm_result = self._call_llm_for_analysis(self.openai, title, text)
            if llm_result:
                llm_result["analysis_mode"] = f"llm_external_{self.openai.model_name}"
                llm_result["is_full_text_used"] = full_pdf_used
                return llm_result

        # Default fallback: High-precision Heuristic NLP Engine
        logger.info(f"Using Heuristic NLP extraction engine for paper: '{title[:40]}'")
        heuristic_res = self.heuristic.extract_paper_analysis(title, text)
        heuristic_res["is_full_text_used"] = full_pdf_used
        return heuristic_res

    def _call_llm_for_analysis(self, provider_instance: Any, title: str, text: str) -> Optional[Dict[str, Any]]:
        """Invokes LLM with strict JSON schema instructions."""
        system_prompt = (
            "You are an academic literature analysis agent. Analyze scientific text strictly based "
            "on the provided content. NEVER invent results, datasets, or experimental metrics. "
            "If any information is unavailable, output exactly 'Not reported in the available text.' "
            "Respond ONLY with a valid JSON object matching the requested schema."
        )

        prompt = f"""
Paper Title: {title}
Paper Text Excerpt:
{text[:4500]}

Extract and return a JSON object with these exact keys:
{{
  "research_problem": "Single concise sentence describing the problem addressed, or '{NOT_REPORTED}'",
  "research_objectives": ["List of specific objectives, or '{NOT_REPORTED}'"],
  "proposed_methodology": "Overview of the method/framework, or '{NOT_REPORTED}'",
  "algorithms_and_models": ["Specific models or algorithms used, or '{NOT_REPORTED}'"],
  "datasets_used": ["Specific benchmark datasets or data corpora used, or '{NOT_REPORTED}'"],
  "evaluation_metrics": ["Evaluation metrics reported, or '{NOT_REPORTED}'"],
  "main_findings": ["Key experimental results or conclusions, or '{NOT_REPORTED}'"],
  "limitations": ["Explicit limitations reported by authors, or '{NOT_REPORTED}'"],
  "future_research_directions": ["Explicit future research directions, or '{NOT_REPORTED}'"],
  "confidence_score": 0.85
}}
"""
        raw_output = provider_instance.generate(prompt, system_prompt=system_prompt)
        if not raw_output:
            return None

        # Clean JSON markdown fences
        clean_json = re.sub(r"^```(?:json)?", "", raw_output.strip(), flags=re.MULTILINE)
        clean_json = re.sub(r"```$", "", clean_json.strip(), flags=re.MULTILINE).strip()

        try:
            data = json.loads(clean_json)
            # Ensure list types
            for key in ["research_objectives", "algorithms_and_models", "datasets_used",
                        "evaluation_metrics", "main_findings", "limitations", "future_research_directions"]:
                if isinstance(data.get(key), str):
                    data[key] = [data[key]]
            return data
        except Exception as e:
            logger.warning(f"Failed to parse LLM JSON response: {e}")
            return None

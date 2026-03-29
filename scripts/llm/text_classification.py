"""
LLM Text Classification
=======================
Classify short texts (e.g. support tickets, log summaries, emails) using the
OpenAI Chat Completions API with a few-shot or zero-shot prompt strategy.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_zero_shot_prompt(text: str, labels: List[str]) -> str:
    label_str = ", ".join(f'"{l}"' for l in labels)
    return (
        f"Classify the following text into exactly one of these categories: {label_str}.\n"
        "Reply with ONLY the category name and nothing else.\n\n"
        f"Text: {text}"
    )


def _build_few_shot_prompt(
    text: str,
    labels: List[str],
    examples: List[dict],
) -> str:
    """
    Build a few-shot prompt.

    Parameters
    ----------
    examples : list of dicts with keys ``text`` and ``label``.
    """
    label_str = ", ".join(f'"{l}"' for l in labels)
    header = (
        f"Classify the following text into exactly one of these categories: {label_str}.\n"
        "Reply with ONLY the category name.\n\n"
        "Examples:\n"
    )
    shots = "".join(
        f'Text: {ex["text"]}\nCategory: {ex["label"]}\n\n' for ex in examples
    )
    query = f"Text: {text}\nCategory:"
    return header + shots + query


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class TextClassifier:
    """
    Zero-shot or few-shot text classifier backed by an OpenAI LLM.

    Parameters
    ----------
    labels : list of valid class labels
    model : OpenAI model name (default: gpt-3.5-turbo)
    examples : optional list of ``{text, label}`` dicts for few-shot prompting
    temperature : sampling temperature (use 0 for deterministic output)
    """

    def __init__(
        self,
        labels: List[str],
        model: str = "gpt-3.5-turbo",
        examples: Optional[List[dict]] = None,
        temperature: float = 0.0,
    ):
        self.labels = labels
        self.model = model
        self.examples = examples or []
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError(
                        "OPENAI_API_KEY environment variable is not set."
                    )
                self._client = OpenAI(api_key=api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                ) from exc
        return self._client

    def classify(self, text: str) -> str:
        """
        Classify a single text string.

        Returns
        -------
        predicted label (str)
        """
        if self.examples:
            prompt = _build_few_shot_prompt(text, self.labels, self.examples)
        else:
            prompt = _build_zero_shot_prompt(text, self.labels)

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip()
        # Normalise: pick the closest label if the model added extra text
        for label in self.labels:
            if label.lower() in raw.lower():
                return label
        logger.warning("Unexpected model output '%s'. Returning as-is.", raw)
        return raw

    def classify_batch(self, texts: List[str]) -> List[str]:
        """Classify a list of texts sequentially."""
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    LABELS = ["Billing", "Technical Support", "General Inquiry", "Complaint"]
    EXAMPLES = [
        {"text": "I was charged twice for my subscription.", "label": "Billing"},
        {"text": "My app crashes every time I open it.", "label": "Technical Support"},
        {"text": "What are your opening hours?", "label": "General Inquiry"},
    ]
    TEXTS = [
        "The invoice shows the wrong amount.",
        "I cannot log in to my account after the update.",
        "Where can I find your refund policy?",
        "This service is absolutely terrible!",
    ]

    classifier = TextClassifier(labels=LABELS, examples=EXAMPLES)
    for t in TEXTS:
        label = classifier.classify(t)
        print(f"  [{label}] {t}")

"""
LLM Text Summarization
======================
Summarise long-form text (documents, articles, incident reports) using the
OpenAI Chat Completions API.  Supports single-pass summarisation and a
map-reduce strategy for very long inputs.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Conservative chunk size (characters) for map-reduce splitting
_DEFAULT_CHUNK_SIZE = 4000


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _summarise_prompt(text: str, style: str, max_words: int) -> str:
    return (
        f"Summarise the following text in a {style} style, "
        f"using at most {max_words} words. "
        "Focus on the most important points.\n\n"
        f"{text}"
    )


def _merge_prompt(summaries: List[str], max_words: int) -> str:
    combined = "\n\n---\n\n".join(summaries)
    return (
        f"Below are summaries of different sections of a document. "
        f"Merge them into a single coherent summary of at most {max_words} words.\n\n"
        f"{combined}"
    )


# ---------------------------------------------------------------------------
# Summariser class
# ---------------------------------------------------------------------------

class TextSummarizer:
    """
    Summarise text using an OpenAI LLM.

    Parameters
    ----------
    model : OpenAI model name
    style : summary style hint ('concise', 'bullet-point', 'technical', …)
    max_words : target maximum word count for the final summary
    chunk_size : character limit per chunk when using map-reduce
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        style: str = "concise",
        max_words: int = 150,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ):
        self.model = model
        self.style = style
        self.max_words = max_words
        self.chunk_size = chunk_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise EnvironmentError("OPENAI_API_KEY is not set.")
                self._client = OpenAI(api_key=api_key)
            except ImportError as exc:
                raise ImportError("Run: pip install openai") from exc
        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def summarise(self, text: str) -> str:
        """
        Summarise *text*.

        If the text is longer than ``chunk_size`` characters the map-reduce
        strategy is used automatically.
        """
        if len(text) <= self.chunk_size:
            prompt = _summarise_prompt(text, self.style, self.max_words)
            return self._call_llm(prompt)
        return self._map_reduce(text)

    def _map_reduce(self, text: str) -> str:
        """Split text into chunks, summarise each, then merge."""
        chunks = [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size)
        ]
        logger.info("Map-reduce over %d chunks.", len(chunks))
        partial_summaries = [
            self._call_llm(_summarise_prompt(chunk, self.style, self.max_words))
            for chunk in chunks
        ]
        merge_prompt = _merge_prompt(partial_summaries, self.max_words)
        return self._call_llm(merge_prompt, max_tokens=500)

    def summarise_batch(self, texts: List[str]) -> List[str]:
        """Summarise a list of texts sequentially."""
        return [self.summarise(t) for t in texts]


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ARTICLE = (
        "Artificial intelligence (AI) has rapidly evolved over the past decade, "
        "transforming industries ranging from healthcare and finance to transportation "
        "and entertainment.  Machine learning models can now diagnose diseases from "
        "medical images, predict stock market movements, power autonomous vehicles, "
        "and generate realistic synthetic media.  However, these advances also raise "
        "significant concerns around algorithmic bias, data privacy, and the displacement "
        "of human labour.  Researchers and policymakers are increasingly calling for "
        "ethical frameworks, transparency requirements, and regulation to ensure that AI "
        "systems are developed and deployed responsibly."
    )

    summarizer = TextSummarizer(style="bullet-point", max_words=60)
    summary = summarizer.summarise(ARTICLE)
    print("Summary:\n", summary)

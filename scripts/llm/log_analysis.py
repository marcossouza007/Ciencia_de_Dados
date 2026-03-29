"""
LLM Log Analysis & Incident Summarization
==========================================
Use an OpenAI LLM to:
1. Triage log batches and identify the most relevant anomalies.
2. Generate human-readable incident summaries from raw log lines.
3. Suggest probable root causes and remediation steps.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _triage_prompt(logs: List[str], top_n: int) -> str:
    joined = "\n".join(f"  [{i+1}] {line}" for i, line in enumerate(logs))
    return (
        f"You are a Site-Reliability Engineer.  Review the following log entries "
        f"and identify the {top_n} most critical issues that require immediate attention.\n\n"
        f"Log entries:\n{joined}\n\n"
        "For each critical issue, output:\n"
        "  Issue #: <number from the list>\n"
        "  Severity: <HIGH | MEDIUM | LOW>\n"
        "  Description: <one-sentence explanation>\n"
    )


def _incident_summary_prompt(logs: List[str]) -> str:
    joined = "\n".join(logs)
    return (
        "You are an experienced incident responder.  Based on the following log snippet, "
        "write a concise incident summary (3-5 sentences) describing:\n"
        "  1. What happened\n"
        "  2. The likely impact\n"
        "  3. Probable root cause\n"
        "  4. Suggested remediation steps\n\n"
        f"Logs:\n{joined}"
    )


def _root_cause_prompt(incident_description: str) -> str:
    return (
        "Given the following incident description, propose the top 3 probable root causes "
        "and for each suggest a concrete remediation action.\n\n"
        f"Incident:\n{incident_description}"
    )


# ---------------------------------------------------------------------------
# LogAnalyzer class
# ---------------------------------------------------------------------------

class LogAnalyzer:
    """
    LLM-powered log analyser and incident summariser.

    Parameters
    ----------
    model : OpenAI model name
    temperature : sampling temperature (lower = more deterministic)
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.2,
    ):
        self.model = model
        self.temperature = temperature
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

    def _call_llm(self, prompt: str, max_tokens: int = 600) -> str:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    def triage(self, logs: List[str], top_n: int = 5) -> str:
        """
        Triage a list of log entries and return a formatted triage report.

        Parameters
        ----------
        logs  : raw log lines (strings)
        top_n : number of critical issues to identify
        """
        # Truncate to avoid exceeding context window
        truncated = logs[:200]
        prompt = _triage_prompt(truncated, top_n)
        return self._call_llm(prompt, max_tokens=800)

    def summarise_incident(self, logs: List[str]) -> str:
        """
        Generate a structured incident summary from a set of log lines.
        """
        truncated = logs[:100]
        prompt = _incident_summary_prompt(truncated)
        return self._call_llm(prompt, max_tokens=500)

    def analyse_root_cause(self, incident_description: str) -> str:
        """
        Given a textual incident description, suggest root causes and
        remediation actions.
        """
        prompt = _root_cause_prompt(incident_description)
        return self._call_llm(prompt, max_tokens=600)

    def full_analysis(self, logs: List[str]) -> dict:
        """
        Run the complete analysis pipeline:
        1. Triage
        2. Incident summary
        3. Root-cause analysis

        Returns a dict with keys ``triage``, ``summary``, and ``root_cause``.
        """
        triage_report = self.triage(logs)
        incident_summary = self.summarise_incident(logs)
        root_cause = self.analyse_root_cause(incident_summary)
        return {
            "triage": triage_report,
            "summary": incident_summary,
            "root_cause": root_cause,
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    SAMPLE_LOGS = [
        "Jan 15 03:14:07 webserver01 nginx[4512]: ERROR 502 Bad Gateway upstream timed out (110: Connection timed out) while reading response header from upstream",
        "Jan 15 03:14:09 webserver01 nginx[4512]: ERROR 502 Bad Gateway upstream timed out (110: Connection timed out) while reading response header from upstream",
        "Jan 15 03:14:10 dbserver01 postgres[7890]: FATAL terminating connection due to administrator command",
        "Jan 15 03:14:11 dbserver01 postgres[7890]: FATAL the database system is shutting down",
        "Jan 15 03:14:15 webserver01 app[3301]: ERROR Database connection pool exhausted – all 100 connections in use",
        "Jan 15 03:14:20 monitor01 alertmanager: CRITICAL service=api latency_p99=4500ms threshold=2000ms",
        "Jan 15 03:14:22 webserver01 app[3301]: ERROR Failed to process request: timeout after 5000ms",
    ]

    analyzer = LogAnalyzer()
    results = analyzer.full_analysis(SAMPLE_LOGS)
    print("=== TRIAGE ===")
    print(results["triage"])
    print("\n=== INCIDENT SUMMARY ===")
    print(results["summary"])
    print("\n=== ROOT CAUSE ANALYSIS ===")
    print(results["root_cause"])

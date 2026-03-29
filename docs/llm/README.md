# LLM Applications Documentation

## Overview

The `scripts/llm/` package integrates the OpenAI Chat Completions API for NLP tasks:
text classification, summarisation, and log analysis.

> **Authentication**: All classes read the API key from the `OPENAI_API_KEY`
> environment variable.  Never hard-code credentials in source code.

---

## `text_classification.py`

### Purpose
Classify short text snippets (support tickets, emails, log summaries) into
predefined categories using zero-shot or few-shot LLM prompting.

### `TextClassifier`

```python
from scripts.llm.text_classification import TextClassifier

classifier = TextClassifier(
    labels=['Billing', 'Technical Support', 'Complaint'],
    examples=[                             # optional few-shot examples
        {'text': 'I was charged twice.', 'label': 'Billing'},
    ],
    model='gpt-3.5-turbo',
    temperature=0.0,
)

label = classifier.classify('My app crashes when I log in.')
labels = classifier.classify_batch(['text1', 'text2', 'text3'])
```

---

## `text_summarization.py`

### Purpose
Produce concise summaries of long documents, articles, or reports.

### `TextSummarizer`

```python
from scripts.llm.text_summarization import TextSummarizer

summarizer = TextSummarizer(
    model='gpt-3.5-turbo',
    style='bullet-point',   # 'concise', 'technical', etc.
    max_words=150,
    chunk_size=4000,        # map-reduce chunk size (characters)
)

summary = summarizer.summarise(long_text)
summaries = summarizer.summarise_batch([text1, text2])
```

For texts longer than `chunk_size` characters, the map-reduce strategy is
used automatically: each chunk is summarised independently and the partial
summaries are merged in a final LLM call.

---

## `log_analysis.py`

### Purpose
Triage large log batches, generate human-readable incident summaries, and
suggest root causes using LLM reasoning.

### `LogAnalyzer`

```python
from scripts.llm.log_analysis import LogAnalyzer

analyzer = LogAnalyzer(model='gpt-3.5-turbo', temperature=0.2)

# Triage: identify the N most critical issues
triage = analyzer.triage(log_lines, top_n=5)

# Incident summary
summary = analyzer.summarise_incident(log_lines)

# Root-cause analysis
rca = analyzer.analyse_root_cause(summary)

# Full pipeline (triage + summary + RCA)
results = analyzer.full_analysis(log_lines)
# → {'triage': ..., 'summary': ..., 'root_cause': ...}
```

---

## Cost & token management

- Use `gpt-3.5-turbo` for high-throughput, low-cost tasks.
- Use `gpt-4` for complex reasoning tasks (root-cause analysis, detailed summarisation).
- The map-reduce strategy in `TextSummarizer` keeps each API call within context limits.
- Inputs are automatically truncated to the first 200 log lines in `LogAnalyzer.triage`
  and 100 lines in `LogAnalyzer.summarise_incident`.

---

## Real-world applications

- **Customer support automation**: Auto-route tickets to the right team.
- **Compliance monitoring**: Summarise audit logs for reviewers.
- **Incident response**: On-call engineers receive a structured summary instead of raw logs.
- **Knowledge management**: Summarise long technical documents for quick review.

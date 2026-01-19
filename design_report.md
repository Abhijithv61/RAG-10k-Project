# Concept Note: RAG Design for Financial & Legal Question Answering

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system to answer financial and legal questions using Apple’s 2024 Form 10-K and Tesla’s 2023 Form 10-K. The design prioritizes accuracy, traceability, and hallucination avoidance, which are critical in regulated domains. This note summarizes the key decisions around chunking, LLM selection, and out-of-scope handling.

---

## 1. Chunking Strategy

10-K filings are long and structurally dense documents containing narrative text and financial disclosures. To support accurate retrieval and citation, documents are parsed into text with explicit page markers and chunked using a recursive character-based splitter.

Chunks are approximately 1000 characters with overlap to preserve semantic continuity across boundaries. Chunk boundaries respect paragraph and sentence structure wherever possible. Each chunk preserves metadata including document name, section (e.g., Item 7 or Item 8), page number, and a unique chunk identifier.

This metadata enables transparent, auditable citations in generated answers (e.g., `["Apple 10-K", "Item 8", "p. 282"]`). While this approach performs well for narrative sections, financial tables are flattened during parsing, which may introduce minor numeric inaccuracies. This limitation is acknowledged and addressed as future work via layout-aware parsing.

---

## 2. LLM Choice

The system uses **llama-3.3-70b-versatile** via Groq as the generation model. This model was selected for its  open-access availability and reasoning capability when paired with strong retrieval.

The LLM is configured with a temperature of 0.0 to ensure deterministic outputs and reduce hallucinations. Importantly, the model is treated strictly as a generator: all factual content must come from retrieved document chunks, and the model is not relied upon for external knowledge.

This choice balances performance, cost efficiency, and reproducibility, making it suitable for both interactive use and automated evaluation.

---

## 3. Out-of-Scope and Unanswerable Handling

Large language models tend to speculate when information is missing or a question falls outside the document scope. To prevent this, the system enforces explicit guardrails.

At the prompt level, the model is instructed to return fixed, predefined responses for missing or out-of-scope information, such as:
- “Not specified in the document.”
- “This question cannot be answered based on the provided documents.”

No additional explanation is permitted after these responses. At the evaluation level, rule-based checks ensure these guardrails are followed exactly, preventing partial or speculative answers from being scored as correct.

---

## Conclusion

The system design emphasizes grounded generation, transparent citations, and safe failure modes. Through careful chunking, an efficient LLM choice, and strict out-of-scope handling, the RAG pipeline achieves reliable performance for financial and legal question answering while remaining simple, reproducible, and extensible.

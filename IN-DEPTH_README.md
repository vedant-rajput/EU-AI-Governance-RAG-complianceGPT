# 🧑‍💼 RAG Regulators — Tech Overview

Welcome! This document provides a comprehensive technical overview of the **RAG Regulators** project. I have designed this project to help recruiters, engineering managers, and technical leads understand the architecture, design choices, and engineering depth behind this Enterprise AI Governance Assistant.

## 🎯 What is it?

**RAG Regulators** is a production-grade **Multi-Agent Retrieval-Augmented Generation (RAG)** application. Its primary purpose is to help legal, compliance, and enterprise teams navigate and query complex AI and data governance frameworks, specifically the **EU AI Act**, **GDPR**, and **NIST AI RMF**.

Instead of simply passing a document to an LLM, this system utilizes a robust, multi-step pipeline combining hybrid search, deterministic scoring, listwise LLM re-ranking, and external multi-agent fact-checking to ensure responses are highly accurate, grounded, and verified.

---

## 🏗️ Core Architecture & Tech Stack

- **Frontend / UI:** Gradio (Python) — highly customized with CSS glassmorphism, responsive components, and an inline PDF viewer.
- **Backend Model:** Google Gemini (`First model used: 'chatgpt gpt-4o-mini',gemini-2.5-flash` for generation and reasoning, `gemini-embedding-001` for vector embeddings).
- **Vector Database:** Qdrant (running in an isolated Docker container).
- **Keyword Search:** In-memory BM25 index.
- **Integration & Tooling:** FastMCP (Model Context Protocol) for connecting agents to external tools (DuckDuckGo search, file system operations).
- **Deployment:** Fully Dockerized with `docker-compose`.

---

## ⚙️ How the Pipeline Works (Step-by-Step)

When a user asks a question, the system does not simply generate an answer immediately. It executes a complex, rigorously designed workflow:

### 1. Data Ingestion & Indexing (Offline/Startup):
*   **Extraction:** Raw compliance PDFs are processed using `PyMuPDF`. Text is cleaned, normalized, and stripped of repeated headers/footers to improve embedding quality.
*   **Chunking:** The text is split into semantic chunks (e.g., 500 characters with 100 character overlaps).
*   **Embedding & Storage:** Chunks are embedded using Gemini's embedding model and pushed to the **Qdrant** vector database. Simultaneously, an in-memory **BM25** index is built for keyword matching.

### 2. Hybrid Retrieval (The Search Phase):

*   **Vector Search:** The user's query is embedded, and Qdrant performs a fast cosine-similarity search to find semantically related chunks.
*   **Keyword Search:** BM25 runs an exact-keyword match search against the corpus.
*   **Reciprocal Rank Fusion (RRF):** The results from Vector and Keyword searches are mathematically fused using RRF. This prevents edge cases where a user searches for a specific article number (better for keyword) or a vague concept (better for semantic).

### 3. LLM Listwise Re-Ranking:
*   The top candidate chunks from the hybrid search are bundled and sent to a Gemini LLM. The LLM acts as a strict judge, re-ordering the chunks from most to least relevant based on the user's specific context. This dramatically improves context precision.

### 4. Multi-Agent Synthesis (The Generation Phase):

The system employs three asynchronous agents working in tandem:
*   **Agent 1 (Internal Researcher):** Drafts a highly detailed response based *strictly* on the retrieved chunks from the internal PDFs.
*   **Agent 2 (External Fact-Checker):** If agentic mode is enabled, this agent generates a web search query, connects to a **FastMCP** server, executes a DuckDuckGo search, and analyzes the live web results for recent updates or contradictions.
*   **Agent 3 (Synthesizer):** Merges the internal draft and the external web findings into a final, cohesive compliance report, clearly delineating internal vs. external facts.

### 5. Verification & Highlighting:

*   **Visual Grounding:** To prevent hallucinations, the system identifies the exact PDF pages where the retrieved information originated. Utilizing a custom `SequenceMatcher` algorithm, it dynamically overlays yellow highlights on the source PDF words, extracts only those specific pages, and displays them directly in the Gradio UI next to the chatbot.
*   **Human-in-the-Loop (HITL):** A human operator can review the generated answer and, if satisfied, hit "Approve" to automatically compile it into a markdown report via the MCP server.

### 6. Dynamic Live Indexing:
*   Users can upload new compliance updates (e.g., a PDF amendment) directly through the UI. The MCP server extracts the text, chunks it, and queues it. The user can then hit "Re-index", which instantly computes new embeddings and injects them into the live Qdrant container *without requiring a system restart*.

---

## 🌟 Standout Engineering Highlights for Recruiters:

*   **Production-Ready Containerization:** The application correctly separates state. The stateless Python server (Gradio) runs alongside a dedicated, stateful Qdrant container, communicating over HTTP.
*   **Advanced RAG Techniques:** Implements **Hybrid Search + RRF + LLM Re-ranking**. This is an industry standard for maximizing retrieval accuracy but is rarely seen in standard portfolio projects.
*   **Model Context Protocol (MCP):** Demonstrates cutting-edge agentic workflows by using FastMCP to separate AI logic from tool execution (web searching and file manipulation).
*   **Dynamic Visual Source Highlighting:** Instead of just outputting standard text citations, the app manipulates the original PDFs in real-time, matching tokens to draw visual boxes around the evidence. This proves a deep understanding of data provenance and user trust.
*   **Evaluation Pipeline Strategy:** Includes dedicated evaluation scripts (`run_eval.py`) that test the RAG system against a golden dataset, measuring both *Retrieval Hit Rate* and *Generation Accuracy* using an LLM-as-a-judge approach.
*   **Multi-Modal Interaction:** Beyond the web UI, it includes a `signal_bot.py` script that hooks into `signal-cli`, allowing users to query the RAG pipeline natively from their smartphones via the Signal messaging app.

---

## 🚀 Pipeline Creation & Evaluation Methodology:

Building a reliable RAG system requires a systematic approach to data extraction, processing, and ongoing rigorous evaluation. Here is how the pipeline was technically constructed from the ground up:

### 1. Robust Data Extraction & Normalization:

The foundation of the pipeline was built via `scripts/extract.py`. Rather than blindly passing text to an embedder, a custom text cleanup pipeline was engineered for complex legal PDFs:
- **PyMuPDF Engine:** Extracts raw text and precise bounding boxes.
- **Deduplication:** A custom algorithm mathematically detects and removes repetitive headers, footers, and page numbers that pollute vector distance measurements.
- **Syntactic Normalization:** Fixes mid-sentence hyphenations and collapses unicode irregularities ensuring clean, contiguous sentences before chunking.
The output is a highly structured `corpus.json` representing continuous cleaned text.

### 2. High-Fidelity Vectorization:

The `scripts/index_data.py` script then takes over to map the corpus to vector space:
- **Strategic Chunking:** The corpus is split into 500-character chunks with a 100-character overlap, maintaining context limits while keeping facts self-contained.
- **Batch Processing:** Uses `google-genai` to compute embeddings via `gemini-embedding-001`, specifically handling rate limits natively with batching and exponential backoff logic.
- **Qdrant Initialization:** Injects the embeddings into the Qdrant container with payloads containing exact `source` and `page` metadata for provenance tracking.

### 3. Quantitative Evaluation Pipeline:

A major differentiator of this project is the inclusion of an automated, LLM-as-a-Judge evaluation test suite (`scripts/run_eval.py` & `scripts/eval_runner.py`). A true RAG system must be tested before going to production:
- **Golden Dataset:** A controlled set of questions (`data/questions.json`) mapping to known exact facts across categories: Factual, Cross-reference, Ambiguous, and Out-of-scope.
- **Evaluation Metrics:** The runner executes fully programmatic tests computing two crucial metrics:
  - **Retrieval Hit Rate:** Confirms the exact expected source and page were successfully surfaced by the hybrid search algorithm within the top-K chunks.
  - **Generation Accuracy:** Uses strict LLM parsing to measure keyword overlaps and appropriate refusal logic (e.g., ensuring "I don't have enough information" is safely triggered for out-of-scope prompts).

By automating this testing framework, every iteration of the chunking strategy, embedding model, or prompt template can be empirically proven to improve or degrade system performance.

---

## 🔍 Dependencies Deep Dive: Tracking PyMuPDF:

To demonstrate the depth of tool integration across this project, here is an exact map of how and where the `PyMuPDF` dependency (used for high-fidelity PDF extraction and rendering) is utilized throughout the codebase:

- **Dependency Declaration:**
  - `requirements.txt`: Line 15 (`PyMuPDF==1.26.5`)
  - `Dockerfile`: Line 5 (System-level dependencies comment for PyMuPDF)
- **Extraction Pipeline (`scripts/extract.py`):**
  - Line 16 (`import pymupdf`)
  - Line 113 (`doc = pymupdf.open(str(pdf_path))`)
- **Visual Highlighting Engine (`scripts/pdf_highlighter.py`):**
  - Line 31 (`import pymupdf`)
  - Lines 102, 141, 262 (`pymupdf.open()`)
  - Line 164 (`def _highlight_words(page: pymupdf.Page, chunk_text: str) -> bool:`)
  - Line 200 (`annot = page.add_highlight_annot(pymupdf.Rect(*merged))`)
  - Lines 229, 233 (`def _draw_page_marker(page: pymupdf.Page) -> None:`)
  - Line 265 (`mat = pymupdf.Matrix(dpi / 72, dpi / 72)`)
- **Multi-Agent Server (`app/mcp_server.py`):**
  - Line 48 (Docstring explaining PyMuPDF converts files dynamically)
- **Frontend App (`app/app.py`):**
  - Line 81 (Comment noting PyMuPDF load time handling)
- **Documentation:**
  - `scripts/EXTRACTION_REPORT.md`: Lines 56, 74 (Discussing extraction choices)
  - `RECRUITER_README.md`: Lines 29, 73 (Documenting architectural usage)

---

## 🔗 Model Context Protocol (MCP) Integration:

A key highlight of this architecture is the decoupling of agentic tools using the **Model Context Protocol (MCP)** via the `fastmcp` framework. Instead of hardcoding API calls directly into the LLM synthesis logic, the system relies on a dedicated background server (`app/mcp_server.py`).

### How it links:
1. **Background Spawning:** When the native python app (`app/app.py` or `signal_bot.py`) boots up, it automatically spins up the `fastmcp run` server as a completely standalone background subprocess bound to a local HTTP port (8001).
2. **Client Connectivity:** When the External Fact-Checker agent determines a need to search the live web, it opens an asynchronous `fastmcp.Client` connection to that local port.
3. **Execution & Isolation:** The agent requests a unified action (like `search_web` or `create_markdown_report`), and the MCP Server executes the physical python code (using dependencies like `duckduckgo_search`) in its own isolated memory space. It securely streams the text or JSON results back to the agent over HTTP.

### Why this matters to engineering managers:

This implementation proves an understanding of modern **Microservice Tool Calling**. By exposing tools through an MCP Server, the web-search capability and file-system write privileges are isolated. We could hypothetically move the MCP server to a secure, locked-down VM, or share these exact same tools across multiple different AI workflows across a whole company, without altering the core LLM's architecture.

---

Bugs Fixed
1. MCP server crash on startup (app/mcp_server.py)

Root cause: _PROJECT_ROOT and sys.path.insert were defined on line 14 — after the from logger import get_logger import on line 9. When fastmcp spawned the server in a subprocess, logger.py wasn't on sys.path yet → ModuleNotFoundError → crash every time
Fix: Moved _PROJECT_ROOT/_APP_DIR calculation and sys.path.insert to lines 9–14, before all other imports
2. API quota hang (scripts/query_rag.py, app/agents.py)

Root cause: Free tier of gemini-2.5-flash is 20 requests/day. When quota is exhausted the retry loop would attempt 6 retries × ~35s each = 3.5 minutes of hanging before returning the generic error
Fix: Added early exit in execute_with_retry / execute_with_retry_async when error contains PerDay (daily quota). generate_answer now shows a clear quota message with a link to the rate-limit dashboard instead of the generic error
3. Duplicate broken function body (app/signal_bot.py)

Root cause: check_for_messages() had a second unreachable block of code (lines 168–201) that re-ran the subprocess poll and called process_query(message_text.strip()) synchronously — a sync call to an async function, which would crash
Fix: Removed the entire duplicate block
4. Unused import (app/app.py)

Removed import numpy as np which was imported but never used

---

## 🛠️ To Run Locally:

1. Set your `GEMINI_API_KEY` in a `.env` file.
2. Run `docker compose up --build -d`.
3. Native access at `http://localhost:7860`
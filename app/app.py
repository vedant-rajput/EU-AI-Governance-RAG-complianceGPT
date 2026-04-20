"""Gradio frontend for RAG Regulators — backed by the Qdrant + Google Gemini pipeline."""

import atexit
import json
import os
import socket
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import uuid

import gradio as gr

# Since this file is named app.py, running it makes Python treat "app" as this very file.
# To avoid the ModuleNotFoundError, we import directly from 'agents' since we are already in the app folder.
from agents import agent_2_external_fact_checker, agent_3_synthesizer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.logger import get_logger
from scripts.pdf_highlighter import create_highlighted_pdfs, render_highlighted_pages_html
from scripts.query_rag import build_bm25, generate_answer, get_embeddings, load_index_and_chunks, retrieve

logger = get_logger(__name__)

try:
    from fastmcp import Client
except ImportError:
    Client = None

MCP_PORT = 8001
# fastmcp serves at /mcp (no trailing slash needed by the client)
MCP_URL = f"http://localhost:{MCP_PORT}/mcp"
_mcp_proc = None


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if something is already listening on the given port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _get_fastmcp_bin() -> str:
    """Return the fastmcp binary path from the active venv, falling back to PATH."""
    # Look beside the current Python interpreter (inside the venv)
    candidate = os.path.join(os.path.dirname(sys.executable), "fastmcp")
    if os.path.isfile(candidate):
        return candidate
    return "fastmcp"  # fall back to whatever is on PATH


def start_mcp_server():
    """Launch mcp_server.py via `fastmcp run` in a background subprocess."""
    global _mcp_proc
    if _is_port_open(MCP_PORT):
        logger.info(f"✅ MCP server already running on port {MCP_PORT}.")
        return
    mcp_script = os.path.join(PROJECT_ROOT, "app", "mcp_server.py")
    fastmcp_bin = _get_fastmcp_bin()
    cmd = [
        fastmcp_bin, "run",
        mcp_script + ":mcp",
        "--transport", "http",
        "--host", "127.0.0.1",
        "--port", str(MCP_PORT),
    ]
    logger.info(f"🚀 Starting MCP server — binary: {fastmcp_bin}")
    _mcp_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Wait up to 40 seconds — heavier imports (PyMuPDF) need more boot time
    for i in range(80):
        time.sleep(0.5)
        # Check if the process crashed before the port opened
        if _mcp_proc.poll() is not None:
            out = _mcp_proc.stdout.read() if _mcp_proc.stdout else ""
            logger.error(f"❌ MCP server process exited early (code {_mcp_proc.returncode}):\n{out}")
            return
        if _is_port_open(MCP_PORT):
            logger.info(f"✅ MCP server is up on port {MCP_PORT} (after {(i+1)*0.5:.1f}s).")
            return
    logger.warning("⚠️  MCP server did not start in time — agentic mode may fail.")


def _stop_mcp_server():
    if _mcp_proc and _mcp_proc.poll() is None:
        _mcp_proc.terminate()
        logger.info("MCP server stopped.")


atexit.register(_stop_mcp_server)
start_mcp_server()

APP_TITLE = "⚖️ RAG Regulators"
EXAMPLE_QUESTIONS = [
    "What are the fines under GDPR?",
    "What makes an AI system high-risk under the EU AI Act?",
    "What are the four functions of the NIST AI RMF?",
    "How do the fines compare between the EU AI Act and GDPR?",
]

# Load index and chunks once at startup (no API call needed)
logger.info("Loading vectors and corpus...")
INDEX, CHUNKS = load_index_and_chunks()
BM25 = build_bm25(CHUNKS)
logger.info(f"✅ Loaded {len(CHUNKS)} records.")

LIVE_FACTS_PATH = os.path.join(PROJECT_ROOT, "data", "live_facts.json")
_indexed_fact_count = 0  # track how many live facts are already embedded


def reindex_live_facts() -> str:
    """Read any new rows from live_facts.json and merge them into the live FAISS + BM25 index."""
    global INDEX, CHUNKS, BM25, _indexed_fact_count

    if not os.path.exists(LIVE_FACTS_PATH):
        return "⚠️ No live_facts.json found yet. Add facts first via 'Update Vector DB queue & Save'."

    # Read all lines and only process the ones we haven't indexed yet
    with open(LIVE_FACTS_PATH, "r", encoding="utf-8") as f:
        all_lines = [l.strip() for l in f if l.strip()]

    new_lines = all_lines[_indexed_fact_count:]
    if not new_lines:
        return "✅ Nothing new to index — all live facts are already in the search index."

    new_chunks = []
    for line in new_lines:
        try:
            fact = json.loads(line)
            new_chunks.append({
                "text": fact.get("text", ""),
                "source": fact.get("source", "Live Web Search"),
                "page": "live",
                "score": 0.0
            })
        except json.JSONDecodeError:
            continue

    if not new_chunks:
        return "⚠️ Found new lines but couldn't parse any valid facts."

    # Embed the new chunks
    texts = [c["text"] for c in new_chunks]
    try:
        embeddings = get_embeddings(texts)
    except Exception as e:
        return f"❌ Embedding failed: {e}"

    # Add to Qdrant
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qclient = QdrantClient(url=qdrant_url)

    start_chunk_id = len(CHUNKS)
    points = []
    for i, (chunk, vector) in enumerate(zip(new_chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "page": chunk["page"],
                    "source": chunk["source"],
                    "chunk_id": start_chunk_id + i
                }
            )
        )
    qclient.upload_points("rag_regulators", points)

    # Append to the live chunks list so retrieval can look them up
    CHUNKS.extend(new_chunks)

    # Rebuild BM25 to include the new chunks
    BM25 = build_bm25(CHUNKS)

    _indexed_fact_count += len(new_chunks)
    return f"✅ Re-indexed {len(new_chunks)} new fact(s) — total corpus now has {len(CHUNKS)} chunks. Your next query will use the updated knowledge!"


def format_sources(results: list) -> str:
    """Return a markdown-formatted source block."""
    if not results:
        return ""
    lines = ["\n\n---\n📚 **Sources**"]
    for r in results:
        lines.append(
            f"- **{r['source']}** — page {r['page']} "
            f"*(relevance: {r['score']:.2f})*"
        )
    return "\n".join(lines)


def _open_pdfs_in_viewer(highlighted: list) -> None:
    """Open each highlighted PDF in Preview (macOS).
    The PDF contains only the retrieved pages, so Preview opens at page 1
    which is already the highlighted content — no page navigation needed.
    """
    if sys.platform != "darwin":
        return
    for pdf_path, _src, _first_page in highlighted:
        try:
            r = subprocess.run(["open", "-a", "Preview", pdf_path], capture_output=True)
            if r.returncode != 0:
                logger.error(f"[pdf] open failed (code {r.returncode}): {r.stderr.decode()}")
            else:
                logger.info(f"[pdf] Opened highlighted extract for '{_src}' in Preview")
        except Exception as e:
            logger.error(f"[pdf] Could not open file: {e}")


def _pdf_source_panel_html(retrieved: list, highlighted: list) -> str:
    """Build an HTML summary of which sources were retrieved and highlighted."""
    if not retrieved:
        return ""

    # Collect unique (source, page) pairs in order
    seen: set[tuple] = set()
    rows: list[str] = []
    for chunk in retrieved:
        src = chunk.get("source", "?")
        pg = chunk.get("page", "?")
        key = (src, pg)
        if key not in seen:
            seen.add(key)
            score = chunk.get("score", 0.0)
            rows.append(
                f"<tr><td style='padding:6px 12px; border-bottom:1px solid rgba(255,255,255,0.05); color:#e2e8f0;'>{src}</td>"
                f"<td style='padding:6px 12px; text-align:center; border-bottom:1px solid rgba(255,255,255,0.05); color:#94a3b8;'>p.{pg}</td>"
                f"<td style='padding:6px 12px; text-align:center; border-bottom:1px solid rgba(255,255,255,0.05); color:#38bdf8; font-weight:600;'>{score:.2f}</td></tr>"
            )

    highlight_note = ""
    if highlighted:
        opened = ", ".join(s for _, s, _ in highlighted)
        highlight_note = (
            f"<div style='margin-top:12px; padding:8px 12px; border-radius:6px; background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.2); color:#34d399; font-weight:500; font-size:0.85rem;'>"
            f"<span>✨ Highlighted PDF opened: </span><strong>{opened}</strong></div>"
        )

    return f"""
<div style='border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px;
            background:rgba(15, 23, 42, 0.6); backdrop-filter:blur(8px); font-size:0.9rem; box-shadow: 0 4px 20px rgba(0,0,0,0.2);'>
  <div style='display:flex; align-items:center; gap:8px; margin-bottom:12px; color:#f8fafc; font-weight:600;'>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>
      Retrieved Passages
  </div>
  <table style='border-collapse:collapse; width:100%; font-size:0.85rem;'>
    <thead>
      <tr style='background:rgba(30, 41, 59, 0.5); text-transform:uppercase; letter-spacing:0.5px; font-size:0.75rem; color:#94a3b8;'>
        <th style='padding:8px 12px; text-align:left; border-radius:6px 0 0 6px;'>Source</th>
        <th style='padding:8px 12px;'>Page</th>
        <th style='padding:8px 12px; border-radius:0 6px 6px 0;'>Score</th>
      </tr>
    </thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
  {highlight_note}
</div>"""


def _status_bar_html(stage: str) -> str:
    """
    Returns an HTML progress indicator for embedding → searching → generating.
    stage: 'embedding' | 'searching' | 'generating' | '' (hidden)
    """
    if not stage:
        return ""

    steps = [("embedding", "Embedding"), ("searching", "Searching"), ("generating", "Generating")]
    order = [s[0] for s in steps]
    active_idx = order.index(stage) if stage in order else -1

    parts: list[str] = []
    for i, (key, label) in enumerate(steps):
        if i < active_idx:
            # Completed — pulsing cyan checkmark
            parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;'
                f'color:#0ea5e9;font-weight:600;">'
                f'<svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0;'
                f'filter:drop-shadow(0 0 4px rgba(14,165,233,0.5));">'
                f'<circle cx="7" cy="7" r="6" fill="#0ea5e9"/>'
                f'<path d="M4 7.2l2 2 4-4" stroke="#0f172a" stroke-width="1.5" '
                f'stroke-linecap="round" stroke-linejoin="round"/></svg>{label}</span>'
            )
        elif i == active_idx:
            # Active — glowing spinning ring
            parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:8px;'
                f'color:#38bdf8;font-weight:700; text-shadow:0 0 8px rgba(56,189,248,0.4);">'
                f'<span style="width:14px;height:14px;border:2px solid rgba(56,189,248,0.2);'
                f'border-top-color:#38bdf8;border-radius:50%;'
                f'animation:rag-spin 0.75s linear infinite;'
                f'display:inline-block;flex-shrink:0;'
                f'box-shadow:0 0 8px rgba(56,189,248,0.4);"></span>{label}</span>'
            )
        else:
            # Pending — dimmed ring
            parts.append(
                f'<span style="display:inline-flex;align-items:center;gap:6px;color:#64748b;">'
                f'<span style="width:14px;height:14px;border:2px solid #334155;border-radius:50%;'
                f'display:inline-block;flex-shrink:0;"></span>{label}</span>'
            )
        if i < len(steps) - 1:
            parts.append('<span style="color:#475569;margin:0 8px;font-size:0.7rem;">▶</span>')

    inner = "".join(parts)
    return (
        '<style>@keyframes rag-spin{to{transform:rotate(360deg)}}</style>'
        '<div style="display:flex;align-items:center;gap:4px;padding:12px 18px;'
        'background:rgba(15, 23, 42, 0.6); backdrop-filter:blur(8px);'
        'border:1px solid rgba(255,255,255,0.05);border-radius:12px;font-size:0.9rem;'
        "font-family:'Outfit',sans-serif;"
        'box-shadow:0 4px 20px rgba(0,0,0,0.3);margin-bottom:8px; transition:all 0.3s ease;">'
        + inner
        + '</div>'
    )


def _normalize_history(history: list) -> list:
    """Convert Gradio 6 ChatMessage objects to plain dicts for safe manipulation."""
    normalized = []
    for msg in history:
        if isinstance(msg, dict):
            normalized.append({"role": msg["role"], "content": msg["content"]})
        else:
            # Gradio 6 ChatMessage objects use attribute access
            normalized.append({"role": msg.role, "content": msg.content})
    return normalized


async def rag_chat(message: str, history: list, use_agentic: bool, auto_open_pdf: bool, k: int = 8):
    """Run the real RAG pipeline. history is Gradio 6 messages format."""
    _no_pdf = ("", gr.update(visible=False))
    _empty_pages = ""   # placeholder until highlighted pages are ready

    if not message.strip():
        yield history, "", "", "", gr.update(), *_no_pdf, "", _empty_pages
        return

    # Normalize history — Gradio 6 may pass ChatMessage objects (not dicts)
    history = _normalize_history(history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": "⏳ Processing your question…"})

    logger.info(f"[rag_chat] query={message!r}  k={int(k)}")

    # ── Stage 1: Embedding ────────────────────────────────────────────────────
    yield history, "", "", "", gr.update(visible=False), *_no_pdf, _status_bar_html("embedding"), _empty_pages
    try:
        query_vec = get_embeddings([message])[0]
    except Exception as e:
        history[-1]["content"] = f"❌ Embedding error: {e}"
        yield history, "", "", "", gr.update(visible=False), *_no_pdf, "", _empty_pages
        return

    # ── Stage 2: Searching ────────────────────────────────────────────────────
    history[-1]["content"] = "🔍 Searching internal corpus…"
    yield history, "", "", "", gr.update(visible=False), *_no_pdf, _status_bar_html("searching"), _empty_pages
    try:
        retrieved, highlight_candidates = retrieve(
            message, INDEX, CHUNKS, bm25=BM25, k=int(k),
            rerank=True, query_vec=query_vec, return_candidates=True
        )
    except Exception as e:
        history[-1]["content"] = f"❌ Retrieval error: {e}"
        yield history, "", "", "", gr.update(visible=False), *_no_pdf, "", _empty_pages
        return

    # --- PDF highlighting (after search, before generate) ---
    # Use the broader pre-rerank candidate pool so more unique pages are included;
    # the reranked `retrieved` is used only for the answer.
    highlighted = create_highlighted_pdfs(highlight_candidates)
    logger.info(f"[pdf] highlighted={[(s, pg) for _, s, pg in highlighted]}, auto_open={auto_open_pdf}")
    if auto_open_pdf and highlighted:
        _open_pdfs_in_viewer(highlighted)
    panel_html = _pdf_source_panel_html(retrieved, highlighted if auto_open_pdf else [])
    _pdf_outputs = (panel_html, gr.update(visible=bool(panel_html)))
    pages_html = render_highlighted_pages_html(highlighted)

    # ── Stage 3: Generating ───────────────────────────────────────────────────
    history[-1]["content"] = "✏️ Generating answer…"
    yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, _status_bar_html("generating"), pages_html
    try:
        internal_ans = generate_answer(message, retrieved)
    except Exception as e:
        history[-1]["content"] = f"❌ Generation error: {e}"
        yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html
        return

    if not use_agentic or Client is None:
        reply = internal_ans + format_sources(retrieved)
        history[-1]["content"] = reply
        yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html
        return

    # ── Agentic mode ─────────────────────────────────────────────────────────
    history[-1]["content"] = internal_ans + "\n\n---\n*🌐 Connecting to web agent for fact-checking...*"
    yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            history[-1]["content"] = internal_ans + "\n\n---\n*🌐 Web Agent is researching...*"
            yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

            external_answer = await agent_2_external_fact_checker(message, internal_ans, mcp_client)

            history[-1]["content"] = internal_ans + f"\n\n---\n*🌐 Web findings summary:*\n{external_answer}\n\n*🤖 Synthesizing final response...*"
            yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

            final_response = await agent_3_synthesizer(message, internal_ans, external_answer)
            reply = final_response + format_sources(retrieved)
            history[-1]["content"] = reply
            yield history, "", final_response, external_answer, gr.update(visible=True), *_pdf_outputs, "", pages_html
    except Exception as e:
        reply = internal_ans + format_sources(retrieved) + f"\n\n*(Error connecting to Web Agent: {e}. Is the fastmcp server running?)*"
        history[-1]["content"] = reply
        yield history, "", "", "", gr.update(visible=False), *_pdf_outputs, "", pages_html

async def on_approve(filename: str, final_response: str):
    if not Client: return "FastMCP Client missing"
    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            res = await mcp_client.call_tool("create_markdown_report", {"filename": filename, "content": final_response})
        return res
    except Exception as e:
        return f"Error: {e}"

async def on_update_db(filename: str, final_response: str, external_answer: str):
    if not Client: return "FastMCP Client missing"
    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            res1 = await mcp_client.call_tool("add_to_database", {"text": external_answer, "source": "Live Web Search MCP"})
            res2 = await mcp_client.call_tool("create_markdown_report", {"filename": filename, "content": final_response})
        return f"{res1} | {res2}"
    except Exception as e:
        return f"Error: {e}"

def on_reject():
    return "Draft rejected. Output not saved.", gr.update(visible=False)


async def on_upload_pdf(file) -> str:
    """Called when user uploads a PDF. Sends it to the MCP load_pdf_to_database tool."""
    if file is None:
        return "⚠️ No file selected."
    if not Client:
        return "❌ FastMCP Client not available."
    pdf_path = file.name if hasattr(file, "name") else str(file)
    try:
        mcp_client = Client(MCP_URL)
        async with mcp_client:
            result = await mcp_client.call_tool("load_pdf_to_database", {"pdf_path": pdf_path})
        return str(result)
    except Exception as e:
        return f"❌ Error loading PDF via MCP: {e}"


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

body, .gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0d1117 !important;
    background-image: radial-gradient(circle at 15% 50%, rgba(14, 165, 233, 0.05), transparent 40%),
                      radial-gradient(circle at 85% 30%, rgba(16, 185, 129, 0.03), transparent 40%);
    color: #c9d1d9 !important;
}

/* Glassmorphism panels */
.gradio-container .gr-box, .gradio-container .gr-panel, .gradio-container .gr-form {
    background: rgba(22, 27, 34, 0.6) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}

/* Enhancing header animation */
@keyframes title-glow {
    0% { text-shadow: 0 0 10px rgba(56,189,248,0.3); }
    50% { text-shadow: 0 0 20px rgba(56,189,248,0.6), 0 0 30px rgba(14,165,233,0.4); }
    100% { text-shadow: 0 0 10px rgba(56,189,248,0.3); }
}
.glow-title {
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: title-glow 3s infinite ease-in-out;
}

/* Buttons */
.gr-button-primary {
    background: linear-gradient(135deg, #0ea5e9, #3b82f6) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 14px 0 rgba(14, 165, 233, 0.2) !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px 0 rgba(14, 165, 233, 0.4) !important;
    background: linear-gradient(135deg, #38bdf8, #60a5fa) !important;
}

.gr-button-secondary {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}
.gr-button-secondary:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.3) !important;
}

/* Chatbot Area */
.gr-chatbot {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
}

/* Custom Webkit Scrollbars for sleekness */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148, 163, 184, 0.2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.4); }

/* Input fields focus states */
textarea:focus, input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
    outline: none !important;
}
"""

theme = gr.themes.Default(
    primary_hue="sky",
    secondary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "sans-serif"],
).set(
    body_background_fill="#0d1117",
    body_text_color="#c9d1d9",
    background_fill_primary="#0d1117",
    background_fill_secondary="#161b22",
    border_color_primary="rgba(255, 255, 255, 0.08)",
    border_color_accent="#38bdf8",
    color_accent_soft="#1e293b",
    block_background_fill="rgba(22, 27, 34, 0.6)",
    block_border_width="1px",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_radius="16px",
    button_large_radius="8px",
    button_small_radius="6px",
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=APP_TITLE, theme=theme, css=custom_css) as demo:
        # ── Full-width header ─────────────────────────────────────────────────
        gr.HTML("""
        <div style="
            display:flex; align-items:center; justify-content: space-between;
            padding:24px 32px; border-bottom:1px solid rgba(255,255,255,0.06); 
            margin-bottom:24px; background: rgba(22, 27, 34, 0.4); border-radius: 16px;
            backdrop-filter: blur(12px); box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        ">
            <div style="display:flex; align-items:center; gap:18px;">
                <div style="font-size:2.8rem; line-height:1; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.3));">⚖️</div>
                <div>
                    <h1 class="glow-title" style="margin:0; font-size:2rem; font-weight:700; letter-spacing:-0.5px;">
                        RAG Regulators
                    </h1>
                    <p style="margin:6px 0 0; font-size:0.95rem; color:#94a3b8; font-weight: 400; letter-spacing: 0.2px;">
                        Enterprise Intelligence · GDPR · EU AI Act · NIST AI RMF
                    </p>
                </div>
            </div>
            <div style="padding: 8px 16px; background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.2); border-radius: 20px; color: #38bdf8; font-size: 0.85rem; font-weight: 600;">
                <span style="display:inline-block; width:8px; height:8px; background:#10b981; border-radius:50%; margin-right:6px; box-shadow: 0 0 8px #10b981;"></span>
                System Online
            </div>
        </div>
        """)

        with gr.Row(equal_height=False):
            # ── Left column: chat interface ───────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="",
                    show_label=False,
                    height=480,
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/376/balance-scale_2696-fe0f.png",
                    ),
                )

                status_bar = gr.HTML(value="", elem_id="rag-status-bar")

                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask a question about GDPR, the EU AI Act, or the NIST AI RMF…",
                        show_label=False,
                        scale=8,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                with gr.Row():
                    use_agents_checkbox = gr.Checkbox(label="Enable Agentic Workflow (Web Fact-Checking)", value=False)
                    auto_open_pdf_checkbox = gr.Checkbox(label="Also open PDF in Preview (macOS)", value=False)
                    k_slider = gr.Slider(
                        minimum=1, maximum=12, step=1, value=8,
                        label="k — chunks retrieved",
                        info="How many passages to retrieve and pass to the LLM",
                    )

                current_final_response = gr.State(value="")
                current_external_answer = gr.State(value="")

                with gr.Group(visible=False) as hitl_panel:
                    gr.Markdown("### 🛠️ Human-in-the-Loop Actions")
                    with gr.Row():
                        filename_input = gr.Textbox(label="Save Filename", value="data/final_report.md", scale=2)
                        hitl_status = gr.Textbox(label="Action Status", interactive=False, scale=3)
                    with gr.Row():
                        approve_btn = gr.Button("Approve & Save Markdown", variant="primary")
                        update_db_btn = gr.Button("Update Vector DB queue & Save", variant="secondary")
                        reject_btn = gr.Button("Reject Draft", variant="stop")

                with gr.Group(visible=False) as pdf_sources_panel:
                    gr.Markdown("### 🔍 Source Documents")
                    pdf_sources_html = gr.HTML(value="")

                with gr.Accordion("📄 Load a New PDF into the Database (via MCP)", open=False):
                    gr.Markdown(
                        "Upload a PDF file. The MCP server will extract its text, chunk it, "
                        "and queue it in `data/live_facts.json`. "
                        "Then click **🔄 Re-index** below to instantly make it searchable."
                    )
                    with gr.Row():
                        pdf_upload = gr.File(
                            label="Upload PDF",
                            file_types=[".pdf"],
                            scale=3,
                        )
                        pdf_status = gr.Textbox(label="PDF Load Status", interactive=False, scale=4)
                    pdf_upload_btn = gr.Button("📥 Extract & Queue PDF into Database", variant="primary")

                with gr.Row():
                    reindex_btn = gr.Button("🔄 Re-index Live Facts into RAG", variant="secondary", size="sm")
                    reindex_status = gr.Textbox(label="Re-index Status", interactive=False, scale=4)

                with gr.Accordion("💡 Example questions", open=False):
                    gr.Examples(
                        examples=EXAMPLE_QUESTIONS,
                        inputs=msg_box,
                        label="",
                    )

                gr.HTML("""
                <p style="text-align:center; font-size:0.8rem; color:#475569; margin-top:16px; font-weight: 500; letter-spacing: 0.5px;">
                    Powered by <span style='color:#ff69b4;'>Qdrant</span> + <span style='color:#38bdf8;'>Google Gemini</span> · <span style='color:#e2e8f0;'>RAG Regulators Team</span>
                </p>
                """)

            # ── Right column: inline PDF viewer ──────────────────────────────
            with gr.Column(scale=2, min_width=320):
                gr.HTML("""
                <div style="font-size:1rem;font-weight:600;color:#f8fafc;
                            padding:12px 16px;border-bottom:1px solid rgba(255,255,255,0.06);
                            margin-bottom:12px; background: rgba(30, 41, 59, 0.4);
                            border-radius: 12px 12px 0 0;
                            display: flex; align-items: center; gap: 8px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                    Source Document Viewer
                </div>
                """)
                pdf_pages_viewer = gr.HTML(
                    value=(
                        "<div style='"
                        "height:calc(100vh - 220px);min-height:300px;"
                        "overflow-y:auto;overflow-x:hidden;"
                        "padding:60px 20px;box-sizing:border-box;"
                        "color:#64748b;font-size:0.9rem;text-align:center;"
                        "background: rgba(15, 23, 42, 0.3); border-radius: 0 0 12px 12px;"
                        "border: 1px dashed rgba(255,255,255,0.1);'>"
                        "<div style='margin-bottom:16px;'><svg width='48' height='48' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1' style='opacity:0.5; margin:0 auto;'><path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><polyline points='14 2 14 8 20 8'/><line x1='16' y1='13' x2='8' y2='13'/><line x1='16' y1='17' x2='8' y2='17'/><polyline points='10 9 9 9 8 9'/></svg></div>"
                        "Retrieval pipeline inactive.<br/>Highlighted source pages will appear here after a query."
                        "</div>"
                    ),
                )

        # ── Wire up events ────────────────────────────────────────────────────
        _rag_inputs  = [msg_box, chatbot, use_agents_checkbox, auto_open_pdf_checkbox, k_slider]
        _rag_outputs = [chatbot, msg_box, current_final_response, current_external_answer,
                        hitl_panel, pdf_sources_html, pdf_sources_panel, status_bar,
                        pdf_pages_viewer]

        send_btn.click(rag_chat, _rag_inputs, _rag_outputs)
        msg_box.submit(rag_chat, _rag_inputs, _rag_outputs)

        approve_btn.click(on_approve, [filename_input, current_final_response], [hitl_status])
        update_db_btn.click(on_update_db, [filename_input, current_final_response, current_external_answer], [hitl_status])
        reject_btn.click(on_reject, [], [hitl_status, hitl_panel])
        pdf_upload_btn.click(on_upload_pdf, [pdf_upload], [pdf_status])
        reindex_btn.click(reindex_live_facts, [], [reindex_status])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))

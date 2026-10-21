import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.logger import log_query
from src.timer import timed_stage
from src.web_search import choose_route, search_web
from src.index_builder import build_index
from src.retriever import Retriever
from src.reranker import rerank_results
from src.qa_engine import (
    build_evidence_extraction_prompt,
    build_final_answer_prompt,
    build_fast_answer_prompt,
)
from src.ollama_client import stream_ollama

import fitz  # PyMuPDF
from src.diagram_builder import extract_diagram_data, build_mermaid
from src.html_diagram_builder import build_html_diagram
from streamlit_searchbox import st_searchbox
from difflib import SequenceMatcher

import json

from src.query_planner import classify_question, build_retrieval_queries

from src.adaptive_retrieval import adaptive_retrieve

DATA_DIR = PROJECT_ROOT / "data"
PAPERS_DIR = DATA_DIR / "papers"
INDEX_DIR = DATA_DIR / "indexes"
METADATA_DIR = DATA_DIR / "metadata"

PAPERS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_RETRIEVAL_K = 30

DOCUMENT_METADATA_FILE = METADATA_DIR / "document_display_metadata.json"

def load_document_metadata_cache() -> dict:
    if DOCUMENT_METADATA_FILE.exists():
        try:
            with open(DOCUMENT_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    return {}

def save_document_metadata_cache(cache: dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(DOCUMENT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def get_available_documents():
    documents = set()

    for index_file in INDEX_DIR.glob("*.faiss"):
        documents.add(index_file.stem)

    return sorted(documents)


def retrieval_confidence(results):
    if not results:
        return 0

    top_score = results[0].get("rerank_score", results[0].get("hybrid_score", 0))
    return int(min(max(top_score / 1.25, 0), 1) * 100)


def inject_css():
    """Presentation-only styling; retrieval and generation logic is unchanged."""
    st.markdown("""
    <style>
    :root { --ink:#17243a; --muted:#69788b; --accent:#3b65c4; --line:#e4eaf1; }
    .stApp, [data-testid="stAppViewContainer"] { background:#f7f9fc; color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(247,249,252,.88); }
    [data-testid="stMainBlockContainer"] { max-width:1080px; padding-top:2.1rem; padding-bottom:5rem; }
    [data-testid="stSidebar"] { width:265px !important; min-width:265px !important; background:#ffffff; border-right:1px solid var(--line); }
    [data-testid="stSidebarContent"] { width:265px !important; padding-top:1.4rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; font-weight:700; }
    h1 { font-size:2.15rem !important; line-height:1.18; }
    h2 { font-size:1.48rem !important; }
    h3 { font-size:1.17rem !important; }
    p, li { line-height:1.65; }
    .app-eyebrow { font-size:.72rem; letter-spacing:.15em; text-transform:uppercase; font-weight:750; color:var(--accent); margin-bottom:12px; }
    .app-hero { padding:25px 32px 27px; background:#ffffff; border:1px solid var(--line); border-radius:18px; margin-bottom:24px; box-shadow:0 6px 25px rgba(24,45,80,.035); }
    .app-hero h1 { margin:0 0 10px !important; }
    .app-hero p { margin:0; max-width:700px; color:var(--muted); font-size:1rem; }
    .section-kicker { color:var(--accent); font-size:.72rem; letter-spacing:.13em; text-transform:uppercase; font-weight:750; margin:8px 0 3px; }
    .sidebar-brand { font-weight:750; font-size:1.1rem; letter-spacing:-.03em; color:var(--ink); margin:0 0 4px; }
    .sidebar-description { color:var(--muted); font-size:.81rem; line-height:1.5; margin-bottom:22px; }
    .sidebar-title { font-size:.75rem; text-transform:uppercase; letter-spacing:.12em; font-weight:750; color:var(--muted); margin:18px 0 12px; }
    .date-heading { color:var(--muted); font-size:.75rem; font-weight:650; margin:15px 0 7px; }
    div[data-testid="stButton"] > button { border-radius:10px; border-color:var(--line); transition:background .15s,border-color .15s; }
    div[data-testid="stButton"] > button:hover { border-color:#9db4e8; background:#f1f5ff; color:var(--ink); }
    div[data-testid="stButton"] > button[kind="primary"] { background:var(--accent); border-color:var(--accent); color:white; min-height:2.9rem; font-weight:650; }
    div[data-testid="stButton"] > button[kind="primary"]:hover { background:#2e53a6; color:white; }
    [data-testid="stFileUploader"], [data-testid="stExpander"], [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] > div > div { border-radius:12px !important; }
    [data-testid="stFileUploader"] { background:#ffffff; border:1px solid var(--line); padding:9px 14px; border-radius:14px; }
    [data-testid="stExpander"] { background:#ffffff; border:1px solid var(--line); border-radius:12px; overflow:hidden; }
    [data-testid="stTextArea"] textarea { background:#ffffff; border-color:#dce4ee; font-size:.98rem; }
    [data-testid="stTextArea"] textarea:focus { border-color:var(--accent); box-shadow:0 0 0 1px var(--accent); }
    [data-testid="stAlert"] { border-radius:12px; }
    hr { border-color:var(--line) !important; }
    @media (max-width:800px) { [data-testid="stMainBlockContainer"] { padding-top:1rem; } .app-hero { padding:20px; } }
    </style>
    """, unsafe_allow_html=True)


def render_progress_panel(title, steps, completed_count, current_status):
    step_html = ""

    for index, step in enumerate(steps):
        if index < completed_count:
            dot_class = "dot done"
            dot_text = "✓"
        elif index == completed_count:
            dot_class = "dot active"
            dot_text = ""
        else:
            dot_class = "dot"
            dot_text = ""

        step_html += f"""
        <div class="step">
            <div class="{dot_class}">{dot_text}</div>
            <div>{step}</div>
        </div>
        """

    completed_lines = ""
    for step in steps[:completed_count]:
        completed_lines += f"<div class='status-done'>✓ {step}</div>"

    current_line = ""
    if current_status:
        current_line = f"<div class='status-current'>→ {current_status}</div>"

    html = f"""
    <style>
    body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    .loading-card {{
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 26px 30px;
        background: #ffffff;
        box-sizing: border-box;
        animation: fadeIn 0.35s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .loading-top {{
        display: flex;
        align-items: center;
        gap: 26px;
    }}

    .paper-icon {{
        width: 44px;
        height: 44px;
        flex-shrink: 0;
    }}

    .paper-line {{
        animation: scanLine 1.4s infinite ease-in-out;
    }}

    .paper-line.two {{ animation-delay: 0.18s; }}
    .paper-line.three {{ animation-delay: 0.36s; }}

    @keyframes scanLine {{
        0% {{ opacity: 0.25; transform: translateX(0); }}
        50% {{ opacity: 1; transform: translateX(3px); }}
        100% {{ opacity: 0.25; transform: translateX(0); }}
    }}

    .loading-main {{
        flex: 1;
    }}

    .loading-title {{
        font-size: 18px;
        font-weight: 700;
        color: #111;
        margin-bottom: 6px;
    }}

    .loading-subtitle {{
        font-size: 14px;
        color: #555;
        margin-bottom: 24px;
    }}

    .steps {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        position: relative;
        margin-bottom: 26px;
    }}

    .steps::before {{
        content: "";
        position: absolute;
        top: 10px;
        left: 10%;
        right: 10%;
        height: 2px;
        background: #d1d5db;
        z-index: 0;
    }}

    .step {{
        text-align: center;
        font-size: 11.5px;
        color: #111;
        position: relative;
        z-index: 1;
        line-height: 1.3;
    }}

    .dot {{
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 2px solid #d1d5db;
        background: #fff;
        margin: 0 auto 8px auto;
        box-sizing: border-box;
    }}

    .dot.done {{
        background: #000;
        border-color: #000;
        color: white;
        font-size: 12px;
        line-height: 15px;
        font-weight: 700;
    }}

    .dot.active {{
        background: #000;
        border-color: #000;
        animation: pulseDot 1.2s infinite ease-in-out;
    }}

    @keyframes pulseDot {{
        0% {{ transform: scale(1); opacity: 0.75; }}
        50% {{ transform: scale(1.12); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 0.75; }}
    }}

    .progress-track {{
        height: 7px;
        width: 100%;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
    }}

    .progress-fill {{
        height: 100%;
        background: #000;
        border-radius: 999px;
        animation: load 2.2s infinite ease-in-out;
    }}

    @keyframes load {{
        0% {{ width: 24%; }}
        50% {{ width: 74%; }}
        100% {{ width: 24%; }}
    }}

    .status-box {{
        margin-top: 18px;
        padding: 12px 14px;
        border-radius: 10px;
        background: #fafafa;
        font-size: 13px;
        color: #333;
        line-height: 1.7;
    }}

    .status-done {{
        font-weight: 600;
        color: #111;
    }}

    .status-current {{
        color: #444;
    }}
    </style>

    <div class="loading-card">
        <div class="loading-top">
            <svg class="paper-icon" viewBox="0 0 64 64" fill="none">
                <path d="M18 8H39L50 19V56H18V8Z" stroke="black" stroke-width="3" fill="white"/>
                <path d="M39 8V20H50" stroke="black" stroke-width="3"/>
                <path class="paper-line" d="M25 30H43" stroke="black" stroke-width="3" stroke-linecap="round"/>
                <path class="paper-line two" d="M25 38H43" stroke="black" stroke-width="3" stroke-linecap="round"/>
                <path class="paper-line three" d="M25 46H36" stroke="black" stroke-width="3" stroke-linecap="round"/>
            </svg>

            <div class="loading-main">
                <div class="loading-title">{title}</div>
                <div class="loading-subtitle">Local model running. The response will appear progressively.</div>

                <div class="steps">
                    {step_html}
                </div>

                <div class="progress-track">
                    <div class="progress-fill"></div>
                </div>
            </div>
        </div>

        <div class="status-box">
            {completed_lines}
            {current_line}
        </div>
    </div>
    """

    components.html(html, height=330)


def render_web_progress_panel(completed_count, current_status):
    steps = [
        "Expanding query",
        "Searching web",
        "Collecting results",
        "Filtering & reranking",
        "Extracting evidence",
        "Building answer",
        "Generating response",
    ]

    step_html = ""

    for index, step in enumerate(steps):
        if index < completed_count:
            dot_class = "web-dot done"
            dot_text = "✓"
        elif index == completed_count:
            dot_class = "web-dot active"
            dot_text = ""
        else:
            dot_class = "web-dot"
            dot_text = ""

        step_html += f"""
        <div class="web-step">
            <div class="{dot_class}">{dot_text}</div>
            <div>{step}</div>
        </div>
        """

    html = f"""
    <style>
    .web-card {{
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px 32px;
        background: #ffffff;
        box-shadow: 0 8px 24px rgba(0,0,0,0.04);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        animation: fadeIn 0.35s ease-in-out;
    }}

    .web-header {{
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 34px;
    }}

    .web-icon {{
        width: 54px;
        height: 54px;
        border-radius: 14px;
        background: #f5f5f5;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }}

    .web-title {{
        font-size: 22px;
        font-weight: 750;
        color: #111827;
        margin-bottom: 6px;
    }}

    .web-subtitle {{
        font-size: 15px;
        color: #4b5563;
    }}

    .web-steps {{
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        position: relative;
        margin-bottom: 30px;
    }}

    .web-steps::before {{
        content: "";
        position: absolute;
        top: 15px;
        left: 6%;
        right: 6%;
        height: 3px;
        background: #d1d5db;
        z-index: 0;
    }}

    .web-step {{
        text-align: center;
        font-size: 12px;
        color: #111827;
        line-height: 1.35;
        position: relative;
        z-index: 1;
    }}

    .web-dot {{
        width: 26px;
        height: 26px;
        border-radius: 50%;
        border: 3px solid #d1d5db;
        background: white;
        margin: 0 auto 10px auto;
        box-sizing: border-box;
    }}

    .web-dot.done {{
        background: #ff4b4b;
        border-color: #ff4b4b;
        color: white;
        font-size: 15px;
        line-height: 20px;
        font-weight: 700;
    }}

    .web-dot.active {{
        border-color: #ff4b4b;
        animation: pulseWebDot 1.2s infinite ease-in-out;
    }}

    @keyframes pulseWebDot {{
        0% {{ transform: scale(1); opacity: 0.75; }}
        50% {{ transform: scale(1.14); opacity: 1; }}
        100% {{ transform: scale(1); opacity: 0.75; }}
    }}

    .web-progress-track {{
        height: 7px;
        width: 100%;
        background: #e5e7eb;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 28px;
    }}

    .web-progress-fill {{
        height: 100%;
        background: #ff4b4b;
        border-radius: 999px;
        animation: webLoad 2.2s infinite ease-in-out;
    }}

    @keyframes webLoad {{
        0% {{ width: 24%; }}
        50% {{ width: 58%; }}
        100% {{ width: 24%; }}
    }}

    .web-status {{
        display: flex;
        align-items: center;
        gap: 12px;
        background: #fafafa;
        border-radius: 14px;
        padding: 18px 20px;
        font-size: 15px;
        color: #374151;
    }}
    </style>

    <div class="web-card">
        <div class="web-header">
            <div class="web-icon">🌐</div>
            <div>
                <div class="web-title">Searching the web</div>
                <div class="web-subtitle">Gathering and analysing information from the web. This may take a few moments.</div>
            </div>
        </div>

        <div class="web-steps">
            {step_html}
        </div>

        <div class="web-progress-track">
            <div class="web-progress-fill"></div>
        </div>

        <div class="web-status">
            🔎 {current_status}
        </div>
    </div>
    """

    components.html(html, height=330)



def find_pdf_path(document_name: str) -> Path | None:
    possible_paths = [
        PAPERS_DIR / f"{document_name}.pdf",
        PAPERS_DIR / document_name,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    for path in PAPERS_DIR.glob("*.pdf"):
        if path.stem == document_name:
            return path

    return None


@st.cache_data(show_spinner=False)
def render_first_page_preview(pdf_path_str: str) -> bytes:
    doc = fitz.open(pdf_path_str)
    page = doc.load_page(0)

    pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4), alpha=False)
    img_bytes = pix.tobytes("png")

    doc.close()
    return img_bytes


def get_pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    page_count = doc.page_count
    doc.close()
    return page_count


def document_search_score(query: str, document_name: str) -> float:
    if not query:
        return 1.0

    q = query.lower().strip()
    display = get_display_title(document_name).lower()
    raw = document_name.lower()

    searchable = f"{display} {raw}".replace("_", " ").replace("-", " ")

    if q in searchable:
        return 10.0

    tokens = searchable.split()

    token_score = max(
        SequenceMatcher(None, q, token).ratio()
        for token in tokens
    ) if tokens else 0.0

    full_score = SequenceMatcher(None, q, searchable).ratio()

    return max(token_score, full_score)


def search_documents(query: str):
    documents = st.session_state.get("documents", [])

    if not query:
        return documents

    scored = [
        (doc, document_search_score(query, doc))
        for doc in documents
    ]

    return [
        doc for doc, score in sorted(
            scored,
            key=lambda item: item[1],
            reverse=True,
        )
        if score >= 0.35
    ]


def extract_pdf_title_author(pdf_path: Path) -> dict:
    title = ""
    author = ""

    try:
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata or {}

        meta_title = (metadata.get("title") or "").strip()
        meta_author = (metadata.get("author") or "").strip()

        if meta_title and len(meta_title) > 3:
            title = meta_title

        if meta_author and len(meta_author) > 3:
            author = meta_author

        first_page = doc.load_page(0)
        text = first_page.get_text("text")
        doc.close()

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        junk_words = [
            "journal",
            "copyright",
            "doi",
            "http",
            "www",
            "cambridge",
            "springer",
            "elsevier",
            "university press",
            "volume",
            "vol.",
            "issue",
            "issn",
        ]

        if not title:
            for line in lines[:30]:
                lower = line.lower()

                if any(word in lower for word in junk_words):
                    continue

                if lower in ["abstract", "introduction", "references"]:
                    continue

                if 5 <= len(line) <= 140:
                    title = line
                    break

        if not author and title:
            try:
                title_index = lines.index(title)

                for line in lines[title_index + 1:title_index + 6]:
                    lower = line.lower()

                    if any(word in lower for word in junk_words):
                        continue

                    if lower in ["abstract", "introduction"]:
                        break

                    if 3 <= len(line) <= 100:
                        author = line
                        break

            except ValueError:
                pass

        if not title:
            title = pdf_path.stem

        if not author:
            author = "Unknown author"

        return {
            "title": title,
            "author": author,
        }

    except Exception:
        return {
            "title": pdf_path.stem,
            "author": "Unknown author",
        }


import json
import re


@st.cache_data(show_spinner=False)
def extract_title_and_author(pdf_path: Path) -> dict:
    try:
        doc = fitz.open(str(pdf_path))
        page = doc.load_page(0)
        first_page_text = page.get_text("text")
        doc.close()

        if not first_page_text.strip():
            return {
                "title": pdf_path.stem,
                "author": "Unknown author",
            }

        prompt = f"""
You are extracting metadata from the first page of a PDF.

Return ONLY valid JSON.

Use this exact format:

{{
  "title": "...",
  "author": "..."
}}

Rules:
- Extract the actual paper/report title.
- Extract the actual author name(s).
- Ignore journal names.
- Ignore website names.
- Ignore DOI text.
- Ignore copyright text.
- Ignore editor notes.
- Ignore issue/volume/page information.
- Ignore publisher names.
- If no author is clearly stated, use "Unknown author".
- Do not explain anything.
- Do not include markdown.

FIRST PAGE TEXT:

{first_page_text[:6000]}
""".strip()

        raw = collect_llm_output(prompt).strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            return {
                "title": pdf_path.stem,
                "author": "Unknown author",
            }

        data = json.loads(match.group(0))

        title = str(data.get("title", "")).strip()
        author = str(data.get("author", "")).strip()

        if not title:
            title = pdf_path.stem

        if not author:
            author = "Unknown author"

        return {
            "title": title,
            "author": author,
        }

    except Exception:
        return {
            "title": pdf_path.stem,
            "author": "Unknown author",
        }

import json

DOCUMENT_METADATA_FILE = DATA_DIR / "metadata" / "document_display_metadata.json"


def load_document_metadata_cache():
    if DOCUMENT_METADATA_FILE.exists():
        try:
            with open(DOCUMENT_METADATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    return {}







def get_display_title(document_name: str) -> str:
    cache = load_document_metadata_cache()
    metadata = cache.get(document_name, {})

    title = metadata.get("title", document_name)

    if title != document_name:
        return f"{title} — {document_name}"

    return document_name



def render_document_preview(document_name: str):
    pdf_path = find_pdf_path(document_name)

    if pdf_path is None:
        st.warning("Could not find the PDF file for this document.")
        return

    try:
        preview_bytes = render_first_page_preview(str(pdf_path))
        page_count = get_pdf_page_count(pdf_path)

        metadata = extract_title_and_author(pdf_path)

        cache = load_document_metadata_cache()
        cache[document_name] = metadata
        save_document_metadata_cache(cache)

        st.markdown(f"**{metadata['title']} — first page preview**")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(preview_bytes, use_container_width=True)

        with col2:
            st.markdown(f"**Paper:** {metadata['title']}")
            st.markdown(f"**Pages:** {page_count}")
            st.markdown(f"**Author:** {metadata['author']}")

    except Exception as e:
        st.error(f"Could not render preview: {e}")




def stream_answer(prompt):
    answer_placeholder = st.empty()
    full_answer = ""

    for chunk in stream_ollama(prompt):
        full_answer += chunk
        answer_placeholder.markdown(full_answer + "▌")

    clean_answer, _ = extract_diagram_data(full_answer)
    answer_placeholder.markdown(clean_answer)

    return full_answer


def collect_llm_output(prompt):
    full_output = ""

    for chunk in stream_ollama(prompt):
        full_output += chunk

    return full_output.strip()
def extract_mermaid_diagram(answer: str):
    """Extract an optional Mermaid diagram from an LLM answer."""
    import re

    pattern = (
        r"(?:^|\n)#{1,3}\s*Explanatory diagram\s*\n"
        r"\s*```mermaid\s*\n"
        r"(.*?)"
        r"\n\s*```"
    )

    match = re.search(
        pattern, answer, flags=re.IGNORECASE | re.DOTALL
    )

    if not match:
        return answer, None

    diagram_code = match.group(1).strip()

    clean_answer = (
        answer[:match.start()] + answer[match.end():]
    ).strip()

    return clean_answer, diagram_code


def render_mermaid_diagram(diagram_code: str):
    """Render Mermaid inside an isolated Streamlit component."""
    import json
    import re

    if not diagram_code:
        return

    if not re.match(
        r"^\s*(flowchart|graph)\s+(TD|TB|LR|RL|BT)\b",
        diagram_code,
        flags=re.IGNORECASE,
    ):
        st.caption("Unsupported diagram format.")
        st.code(diagram_code, language="text")
        return

    if len(diagram_code) > 5000:
        st.caption("Diagram is too large to display.")
        return

    safe_diagram = json.dumps(diagram_code).replace("<", "\\u003c")

    html = """
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
        <style>
            body {
                margin: 0;
                padding: 16px;
                font-family: -apple-system, BlinkMacSystemFont,
                             "Segoe UI", sans-serif;
                background: #ffffff;
            }
            .diagram-container {
                border: 1px solid #e4eaf1;
                border-radius: 14px;
                padding: 20px;
                overflow-x: auto;
                text-align: center;
            }
            svg {
                max-width: 100%;
                height: auto;
            }
            #error {
                color: #69788b;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="diagram-container">
            <div id="diagram"></div>
            <div id="error"></div>
        </div>

        <script>
            const diagramCode = DIAGRAM_PLACEHOLDER;

            async function drawDiagram() {
                try {
                    mermaid.initialize({
                        startOnLoad: false,
                        securityLevel: "strict",
                        theme: "neutral",
                        flowchart: {
                            useMaxWidth: true,
                            htmlLabels: false
                        }
                    });

                    const result = await mermaid.render(
                        "researchDiagram",
                        diagramCode
                    );

                    document.getElementById("diagram").innerHTML =
                        result.svg;

                } catch (error) {
                    document.getElementById("error").textContent =
                        "Could not render this diagram.";
                }
            }

            drawDiagram();
        </script>
    </body>
    </html>
    """

    html = html.replace("DIAGRAM_PLACEHOLDER", safe_diagram)

    components.html(html, height=400, scrolling=True)

def show_sources_and_details(result):

    with st.expander("Sources", expanded=False):
        for i, chunk in enumerate(result["results"], start=1):
            page = chunk.get("page", "unknown")
            text = chunk.get("text", "")
            document = chunk.get("document", "Unknown document")
            author = chunk.get("author", "Unknown author")

            if i == 1:
                label = f"⭐ Most Relevant — Source {i} — {author} — Page {page}"
            else:
                label = f"Source {i} — {author} — Page {page}"

            with st.expander(label, expanded=False):
                st.caption(f"Document: {document}")
                st.write(text)

                hybrid = chunk.get("hybrid_score", 0)
                semantic = chunk.get("semantic_score", 0)
                bm25 = chunk.get("bm25_score", 0)
                rerank = chunk.get("rerank_score", None)

                st.markdown("#### Retrieval Scores")

                if rerank is not None:
                    st.markdown(
                        f"""
| Score Type | Value |
|---|---:|
| Rerank Score | {rerank:.3f} |
| Hybrid Score | {hybrid:.3f} |
| Semantic Score | {semantic:.3f} |
| BM25 Score | {bm25:.3f} |
"""
                    )
                else:
                    st.markdown(
                        f"""
| Score Type | Value |
|---|---:|
| Hybrid Score | {hybrid:.3f} |
| Semantic Score | {semantic:.3f} |
| BM25 Score | {bm25:.3f} |
"""
                    )

    # ← Notice this is OUTSIDE the Sources expander
    with st.expander("Retrieval Details", expanded=False):
        timings = result.get("timings", {})

        st.markdown(
            f"""
| Metric | Value |
|---|---|
| Mode | {result["mode"]} |
| Raw Retrieved Sources | {result["raw_sources"]} |
| Reranked Sources Used | {len(result["results"])} |
| Documents Used | {", ".join(result["documents"])} |
| Relevant Pages | {", ".join(map(str, result["pages"]))} |
| Retrieval Confidence | {result["confidence"]}% |
| Retrieval | {timings.get("retrieval", result.get("retrieval_time", 0)):.2f}s |
| Evidence Extraction | {timings.get("evidence_extraction", 0):.2f}s |
| Answer Generation | {timings.get("answer_generation", result.get("llm_time", 0)):.2f}s |
| Total | {result["total_time"]:.2f}s |
"""
        )


def show_result(result):
    st.subheader("Answer")

    clean_answer, diagram_data = extract_diagram_data(
        result["answer"]
    )

    st.markdown(clean_answer)

    if diagram_data:
        diagram_code = build_mermaid(diagram_data)

        if diagram_code:
            st.subheader("Explanatory diagram")
            render_mermaid_diagram(diagram_code)

    show_sources_and_details(result)


st.set_page_config(
    page_title="LLM Research Paper Assistant",
    page_icon="📄",
    layout="wide",
)

inject_css()

if "question_history" not in st.session_state:
    st.session_state["question_history"] = []

if "selected_history_index" not in st.session_state:
    st.session_state["selected_history_index"] = None

if "documents" not in st.session_state:
    st.session_state["documents"] = get_available_documents()


from datetime import datetime


def truncate_question(text, max_len=55):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."


def group_questions_by_date(history):
    """
    If timestamps don't exist yet, everything is grouped under Today.
    Once timestamps are added, this function will automatically group
    them properly.
    """
    groups = {}

    for item in history:
        if "timestamp" not in item:
            label = "Today"
        else:
            dt = datetime.fromisoformat(item["timestamp"])
            today = datetime.now().date()

            if dt.date() == today:
                label = "Today"
            elif (today - dt.date()).days == 1:
                label = "Yesterday"
            else:
                label = dt.strftime("%d %b %Y")

        groups.setdefault(label, []).append(item)

    return groups


with st.sidebar:
    st.markdown('<div class="sidebar-brand">◈ Research Assistant</div><div class="sidebar-description">Your local document intelligence workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Question history</div>', unsafe_allow_html=True)

    history = st.session_state["question_history"]

    if not history:
        st.caption("No questions asked yet.")

    else:

        grouped = group_questions_by_date(history)

        button_id = 0

        for group_name, questions in grouped.items():

            st.markdown(
                f'<div class="date-heading">{group_name}</div>',
                unsafe_allow_html=True,
            )

            for item in questions:

                label = truncate_question(item["question"])

                if st.button(label, key=f"history_{button_id}"):

                    idx = history.index(item)

                    st.session_state["selected_history_index"] = idx

                button_id += 1

    st.divider()

    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state["question_history"] = []
        st.session_state["selected_history_index"] = None
        st.rerun()

st.markdown("""
<div class="app-hero">
  <div class="app-eyebrow">RESEARCH WORKSPACE &nbsp; / &nbsp; LOCAL AI</div>
  <h1>Research Paper Assistant.</h1>
  <p>Explore your papers with hybrid retrieval, intelligent reranking, and grounded answers powered by a local language model.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-kicker">01 / YOUR LIBRARY</div>', unsafe_allow_html=True)
st.subheader("Add a research paper")
st.caption("Upload a PDF to build a searchable document index.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    accept_multiple_files=False,
    key="main_pdf_uploader",
)

if uploaded_file is not None:
    pdf_path = PAPERS_DIR / uploaded_file.name

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    document_name = pdf_path.stem

    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner(f"Building index for {uploaded_file.name}..."):
        build_index(pdf_path)

    if document_name not in st.session_state["documents"]:
        st.session_state["documents"].append(document_name)
        st.session_state["documents"] = sorted(st.session_state["documents"])

    st.success(f"Index built for {uploaded_file.name}")

st.divider()

st.markdown('<div class="section-kicker">02 / EXPLORE</div>', unsafe_allow_html=True)
st.header("Ask your research a question")

if not st.session_state["documents"]:
    st.warning("Upload a PDF and build the document index before asking questions.")
else:
    if "preview_document" not in st.session_state:
        st.session_state["preview_document"] = None

    with st.expander(
            f"📚 Available Documents ({len(st.session_state['documents'])})",
            expanded=False,
    ):
        st.caption("Click a document to preview its first page.")

        for doc in st.session_state["documents"]:
            if st.button(
                    f"📄 {doc}",
                    key=f"preview_doc_{doc}",
                    use_container_width=True,
            ):
                if st.session_state["preview_document"] == doc:
                    st.session_state["preview_document"] = None
                else:
                    st.session_state["preview_document"] = doc

        if st.session_state["preview_document"]:
            st.divider()
            render_document_preview(st.session_state["preview_document"])

document_options = st.session_state["documents"]

selected_document = st_searchbox(
    search_function=search_documents,
    label="Choose a paper",
    placeholder="Search by title, author, or file name...",
    key="document_searchbox",
)

if selected_document is None:
    selected_document = st.session_state["documents"][0] if st.session_state["documents"] else None

question = st.text_area(
    "Question",
    placeholder="Ask a focused question, e.g. What is the author's central argument?",
    key="question_input",
    height=95,
)

fast_mode = st.checkbox("Fast mode (single LLM call)", value=True, help="Faster experimental path; compare citation quality with the original two-stage mode.")

num_chunks = st.slider(
    "Number of source chunks",
    min_value=6,
    max_value=16,
    value=10,
    key="source_chunk_slider",
)

if st.button(
    "Generate answer →",
    key="search_button",
    type="primary",
    use_container_width=True,
):
    st.session_state["selected_history_index"] = None

    if not question.strip():
        st.warning("Please enter a question.")

    elif not selected_document:
        st.warning("Please choose a paper.")


    else:

        progress_placeholder = st.empty()

        route = choose_route(question)

        if route == "WEB":

            with progress_placeholder.container():

                render_web_progress_panel(

                    completed_count=0,

                    current_status="Expanding your question into related search queries..."

                )


        else:

            steps = [

                "Scanning paper",

                "Reranking passages",

                "Extracting evidence",

                "Building answer",

                "Generating response",

            ]

            with progress_placeholder.container():

                render_progress_panel(

                    title="Analysing paper",

                    steps=steps,

                    completed_count=0,

                    current_status="Scanning the selected paper",

                )

        timings = {}

        with timed_stage("retrieval", timings):

            route = choose_route(question)

            if route == "CHROMA":

                retrieval_data = adaptive_retrieve(
                    question=question,
                    document_name=selected_document,
                    top_k=num_chunks,
                    raw_k=RAW_RETRIEVAL_K,
                )

                results = retrieval_data["results"]
                raw_results = retrieval_data["raw_results"]
                question_type = retrieval_data["question_type"]
                coverage_score = retrieval_data["coverage_score"]

            elif route == "WEB":

                results = search_web(question, max_results=num_chunks)
                raw_results = results
                question_type = "web_search"
                coverage_score = 80

            elif route == "BOTH":

                retrieval_data = adaptive_retrieve(
                    question=question,
                    document_name=selected_document,
                    top_k=num_chunks,
                    raw_k=RAW_RETRIEVAL_K,
                )

                pdf_results = retrieval_data["results"]
                web_results = search_web(question, max_results=5)

                results = pdf_results + web_results
                raw_results = retrieval_data["raw_results"] + web_results
                question_type = retrieval_data["question_type"]
                coverage_score = retrieval_data["coverage_score"]

        retrieval_time = timings["retrieval"]

        with progress_placeholder.container():

            if route == "WEB":

                render_web_progress_panel(
                    completed_count=4,
                    current_status="Extracting evidence from the highest-quality sources..."
                )

            else:

                render_progress_panel(
                    title="Analysing paper",
                    steps=steps,
                    completed_count=2,
                    current_status="Extracting evidence from the strongest passages",
                )

        if fast_mode:
            timings["evidence_extraction"] = 0.0
            final_prompt = build_fast_answer_prompt(question, results)
        else:
            with timed_stage("evidence_extraction", timings):
                evidence_prompt = build_evidence_extraction_prompt(question, results)
                extracted_evidence = collect_llm_output(evidence_prompt)
            final_prompt = build_final_answer_prompt(
                question=question,
                extracted_evidence=extracted_evidence,
            )

        with progress_placeholder.container():

            if route == "WEB":

                render_web_progress_panel(
                    completed_count=5,
                    current_status="Building a grounded answer..."
                )

            else:

                render_progress_panel(
                    title="Analysing paper",
                    steps=steps,
                    completed_count=3,
                    current_status="Building a grounded answer from extracted evidence",
                )


        with progress_placeholder.container():

            if route == "WEB":

                render_web_progress_panel(
                    completed_count=6,
                    current_status="Generating final response..."
                )

            else:

                render_progress_panel(
                    title="Analysing paper",
                    steps=steps,
                    completed_count=4,
                    current_status="Generating response with the local model",
                )

        if not fast_mode:
            st.subheader("Answer")

        with timed_stage("answer_generation", timings):
            answer = stream_answer(final_prompt)

        llm_time = timings["answer_generation"]
        total_time = sum(timings.values())

        clean_answer, diagram_data = extract_diagram_data(answer)

        if diagram_data:
            with st.expander("Debug: Structured diagram data"):
                st.json(diagram_data)

            diagram_html = build_html_diagram(diagram_data)

            if diagram_html:
                st.subheader("Explanatory diagram")
                st.html(diagram_html)

        progress_placeholder.empty()


        pages = sorted(set(chunk.get("page", "unknown") for chunk in results))
        confidence = coverage_score

        result_data = {
            "mode": f"{route} / {'FAST' if fast_mode else 'TWO-STAGE'}",
            "timings": timings,
            "question": question,
            "answer": answer,
            "results": results,
            "raw_sources": len(raw_results),
            "pages": pages,
            "documents": [selected_document],
            "confidence": coverage_score,
            "coverage_score": coverage_score,
            "question_type": question_type,
            "retrieval_time": retrieval_time,
            "llm_time": llm_time,
            "total_time": total_time,
        }

        log_query(result_data)

        st.session_state["question_history"].insert(0, result_data)

        show_sources_and_details(result_data)

selected_idx = st.session_state.get("selected_history_index")

if selected_idx is not None and st.session_state["question_history"]:
    selected_result = st.session_state["question_history"][selected_idx]

    st.divider()
    st.caption("Showing previous answer")
    st.markdown(f"### Question: {selected_result['question']}")

    show_result(selected_result)




import os
import time
import streamlit as st

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), "gcp-key.json"
)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from rag.ingest import load_documents
from rag.embed_store import VectorStore, _load_reranker
from rag.generate import rewrite_search_query, llm_answer_stream

import base64
import urllib.request

DATA_FOLDER = "data/sample_docs"

# Porsche-inspired Color Palette (Titanium Slate & Guards Red)
BG = "#0e1116"          # Deep Titanium Slate Gray
CARD = "#171c24"        # Dark metallic card background
BORDER = "#2a3240"      # Slate gray border outline
TEXT_PRIMARY = "#f1f3f5"  # High-readability warm white
TEXT_MUTED = "#98a2b3"    # Medium contrast body/label gray
ACCENT_RED = "#ff2a3b"    # Bright Guards Red for active states
R_GRADIENT = "linear-gradient(135deg, #ff2a3b 0%, #c10816 100%)"
R_GLOW = "0 0 16px rgba(255, 42, 59, 0.3)"

def get_porsche_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return ""

st.set_page_config(page_title="Porsche Intelligence RAG", page_icon="🏎️", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syncopate:wght@700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ---- GLOBAL RESET & FONTS ---- */
/* Guarding internal Streamlit/Material icon components from global overrides */
body, .stApp, p, li, a, label, input, textarea, button, select, div, 
span:not(.notranslate):not([class*="material"]):not([data-testid="stIconMaterial"]) {{
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; 
}}
.stApp {{ 
    background: {BG}; 
    background-image: radial-gradient(circle at 50% 0%, #1e2430 0%, {BG} 75%);
}}

/* Eliminate white bar at the top */
.stApp > header, header[data-testid="stHeader"], [data-testid="stHeader"] {{ 
    background-color: transparent !important; 
    background: transparent !important;
}}
.stApp > header:before {{ display: none !important; }}
#root > div:first-child {{ background: transparent !important; }}
.block-container {{ padding-top: 25px !important; padding-bottom: 25px !important; max-width: 950px !important; }}

/* ---- TYPOGRAPHY ---- */
h1, h2, h3, .syncopate {{
    font-family: 'Syncopate', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
}}

/* ---- HEADER ---- */
.hdr {{ 
    display: flex; 
    align-items: center; 
    gap: 16px; 
    padding: 14px 20px; 
    margin-bottom: 24px; 
    background: linear-gradient(135deg, #171c24 0%, #0c0e12 100%) !important;
    border: 1px solid {ACCENT_RED} !important;
    border-radius: 12px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
}}
.hdr .ico {{ 
    display: flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    padding: 0;
}}
.hdr-logo-img {{
    width: 46px;
    height: auto;
    filter: drop-shadow(0 0 10px rgba(255, 42, 59, 0.25));
}}
.hdr-text {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0px !important;
    margin-left: 4px;
}}
.hdr h1 {{ 
    font-size: 20px !important; 
    margin: 0 !important; 
    padding: 0 !important;
    color: #ffffff !important; 
    letter-spacing: 3px;
    font-weight: 700;
    line-height: 1.1 !important;
}}
.hdr .sub {{ 
    font-size: 10px !important; 
    color: {ACCENT_RED} !important; 
    text-transform: uppercase; 
    letter-spacing: 1.5px; 
    margin: 0 !important; 
    padding: 0 !important;
    font-weight: 700;
    line-height: 1.1 !important;
}}

/* ---- CHAT BUBBLES ---- */
.msg {{ 
    display: flex; 
    gap: 16px; 
    margin: 24px 0; 
    align-items: flex-start;
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}}
.msg.user {{ 
    flex-direction: row-reverse; 
}}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.av {{ 
    width: 38px; 
    height: 38px; 
    border-radius: 8px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    flex-shrink: 0; 
    transition: transform 0.2s ease;
}}
.av.u {{ 
    background: {R_GRADIENT}; 
    color: #fff; 
    box-shadow: {R_GLOW};
}}
.av.a {{ 
    background: #1f2530; 
    color: #a2abb7; 
    border: 1px solid {BORDER}; 
}}
.av-svg {{
    width: 18px;
    height: 18px;
}}

.bubble {{ 
    max-width: 78%; 
    padding: 16px 20px; 
    border-radius: 14px; 
    font-size: 15px; 
    line-height: 1.6; 
    color: {TEXT_PRIMARY}; 
    word-wrap: break-word; 
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
    border: 1px solid {BORDER};
}}
.bubble.u {{ 
    background: rgba(255, 42, 59, 0.08); 
    border-color: rgba(255, 42, 59, 0.35);
    border-bottom-right-radius: 4px; 
}}
.bubble.a {{ 
    background: rgba(23, 28, 36, 0.85); 
    backdrop-filter: blur(10px);
    border-color: {BORDER};
    border-bottom-left-radius: 4px; 
}}
.bubble p {{ margin: 0; }}
.bubble p + p {{ margin-top: 12px; }}

/* ---- SOURCES & TELEMETRY ---- */
.src {{ 
    margin: 8px 0 0 54px; 
    max-width: 78%;
}}
.src details {{
    background: rgba(23, 28, 36, 0.5);
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
.src details summary {{ 
    cursor: pointer; 
    color: {TEXT_MUTED}; 
    font-size: 12px; 
    font-weight: 600;
    letter-spacing: 0.8px; 
    padding: 12px 16px; 
    user-select: none;
    text-transform: uppercase;
    transition: all 0.2s ease;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.src details summary::-webkit-details-marker {{
    display: none !important;
}}
.src details summary::after {{
    content: "";
    width: 6px;
    height: 6px;
    border-right: 2px solid {TEXT_MUTED};
    border-bottom: 2px solid {TEXT_MUTED};
    transform: rotate(45deg);
    transition: transform 0.2s ease;
    margin-right: 4px;
}}
.src details[open] summary::after {{
    transform: rotate(-135deg);
}}
.src summary:hover {{ 
    color: #fff; 
    background: rgba(255, 255, 255, 0.03);
}}
.src .chunk-card {{
    padding: 14px 16px;
    border-top: 1px solid {BORDER};
    background: rgba(14, 17, 22, 0.8);
}}
.src .chunk-header {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 700;
    color: #a2abb7;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
.src .chunk-body {{
    color: #cbd5e1;
    font-size: 13.5px;
    line-height: 1.55;
    white-space: pre-wrap;
    font-family: 'Space Grotesk', monospace !important;
}}
.telemetry-row {{
    display: flex;
    gap: 8px;
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 700;
    margin-top: 8px;
    margin-left: 54px;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-family: 'Space Grotesk', sans-serif !important;
}}
.telemetry-tag {{
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid {BORDER};
    padding: 3px 8px;
    border-radius: 4px;
}}

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] > div:first-child {{ 
    background: #0b0d10 !important; 
    padding: 24px 20px !important; 
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-emotion-cache-1gulkj5 {{ 
    color: {TEXT_PRIMARY} !important; 
    font-weight: 600;
    font-size: 13px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}}
[data-testid="stSidebar"] hr {{ 
    border-color: {BORDER} !important; 
}}
[data-testid="stSidebar"] .streamlit-expanderHeader {{ 
    color: {TEXT_MUTED} !important; 
    font-size: 12px !important; 
    background: rgba(23, 28, 36, 0.5) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px;
    margin-bottom: 6px;
}}
.sb-title {{ 
    font-size: 22px; 
    font-weight: 700; 
    color: #fff; 
    letter-spacing: 4px;
    margin-bottom: 2px;
}}
.sb-sub {{ 
    font-size: 10px; 
    color: {TEXT_MUTED}; 
    text-transform: uppercase; 
    letter-spacing: 2.5px; 
    margin-bottom: 24px; 
    font-weight: 600;
}}
.sb-card {{ 
    background: {CARD}; 
    border: 1px solid {BORDER}; 
    border-radius: 12px; 
    padding: 16px; 
    margin: 20px 0; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}}
.sb-card .row {{ 
    display: flex; 
    text-align: center; 
}}
.sb-card .row > div {{ 
    flex: 1; 
}}
.sb-card .row > div:first-child {{ 
    border-right: 1px solid {BORDER}; 
}}
.sb-card .num {{ 
    font-size: 24px; 
    font-weight: 700; 
    color: #fff; 
    font-family: 'Space Grotesk', sans-serif !important;
}}
.sb-card .lbl {{ 
    font-size: 9px; 
    color: {TEXT_MUTED}; 
    text-transform: uppercase; 
    letter-spacing: 1px; 
    margin-top: 4px; 
    font-weight: 700;
}}
.sb-caption {{
    color: #a2abb7 !important;
    font-size: 11px !important;
    font-weight: 500;
    text-align: center;
    margin-top: 16px;
    letter-spacing: 0.5px;
}}

/* ---- INPUT CONTAINER ---- */
[data-testid="stBottom"], [data-testid="stBottom"] > div, div:has(> [data-testid="stChatInput"]), [data-testid="stChatInputContainer"], div[class*="stChatInputContainer"] {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] div, [data-testid="stChatInput"] div div {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
div:has(> [data-testid="stChatInput"]) {{ 
    max-width: 950px !important; 
    margin: 0 auto !important; 
    padding: 0 16px 24px 16px !important; 
}}
[data-testid="stChatInput"] {{ 
    border: none !important; 
    background: transparent !important;
}}
[data-testid="stChatInput"] textarea {{ 
    background: rgba(23, 28, 36, 0.95) !important; 
    color: #fff !important; 
    border: 1px solid {BORDER} !important; 
    border-radius: 12px !important; 
    padding: 16px 20px !important; 
    font-size: 15px !important; 
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.3s ease !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: #8e99a8 !important;
    opacity: 0.85 !important;
}}
[data-testid="stChatInput"] textarea:focus {{ 
    border-color: {ACCENT_RED} !important; 
    box-shadow: 0 0 0 2px rgba(255, 42, 59, 0.2), 0 10px 40px rgba(0, 0, 0, 0.4) !important; 
}}
[data-testid="stChatInput"] button {{ 
    background: {R_GRADIENT} !important; 
    border-radius: 8px !important; 
    color: white !important;
    box-shadow: {R_GLOW};
    transition: transform 0.2s ease, opacity 0.2s ease !important;
}}
[data-testid="stChatInput"] button:hover {{
    transform: scale(1.05);
    opacity: 0.95;
}}

/* ---- SPINNER & SLIDER ---- */
.stSpinner > div {{ 
    border-color: {ACCENT_RED} transparent transparent transparent !important; 
}}
.stSpinner p {{ 
    color: {TEXT_MUTED} !important; 
    font-size: 13px;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background-color: {ACCENT_RED} !important;
}}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #4a5668; }}
</style>
""", unsafe_allow_html=True)

# SVG Icons
USER_SVG = """
<svg class="av-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
  <circle cx="12" cy="7" r="4"></circle>
</svg>
"""

ASSISTANT_SVG = """
<svg class="av-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 2L3 7v9c0 5 9 7 9 7s9-2 9-7V7l-9-5z"/>
  <path d="M12 22V12" stroke-width="1.5"/>
  <path d="M8 8h8" stroke-width="1.5"/>
  <path d="M8 12h8" stroke-width="1.5"/>
  <path d="M8 16h8" stroke-width="1.5"/>
</svg>
"""

def render_sources(sources, rerank):
    if not sources:
        return ""
    parts = ['<div class="src"><details><summary>System Reference Logs</summary>']
    for chunk, score in sources:
        label = f"{chunk.doc_title} · Relevance: {score:.2f}" if rerank else f"{chunk.doc_title} · Chroma Score: {score:.2f}"
        parts.append(
            f'<div class="chunk-card">'
            f'<div class="chunk-header"><span>{label}</span></div>'
            f'<div class="chunk-body">{chunk.text}</div>'
            f'</div>'
        )
    parts.append("</details></div>")
    return "".join(parts)

@st.cache_resource(show_spinner=False)
def get_vector_store():
    return VectorStore()

@st.cache_resource(show_spinner=False)
def get_cached_docs():
    return load_documents(DATA_FOLDER)

# Initialize and track setup stage using session state to prevent flashing on message reruns
if "initialized" not in st.session_state:
    loading_placeholder = st.empty()

    def update_loading_status(step, title, details):
        loading_placeholder.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syncopate:wght@700&family=Space+Grotesk:wght@400;500;700&display=swap');
.brand-wrapper {{
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-bottom: 28px;
    animation: bouncePulse 2.5s infinite ease-in-out;
}}
.porsche-shield {{
    width: 65px;
    height: 75px;
    filter: drop-shadow(0 0 15px rgba(197, 160, 89, 0.4));
}}
.porsche-wordmark {{
    font-family: 'Syncopate', sans-serif !important;
    font-size: 13px;
    font-weight: 700;
    color: #fff;
    letter-spacing: 7px;
    margin-top: 16px;
    text-align: center;
    background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
@keyframes bouncePulse {{
    0%, 100% {{ transform: scale(1) translateY(0); }}
    50% {{ transform: scale(1.04) translateY(-6px); }}
}}
.loading-container {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 120px 20px;
    text-align: center;
    background: #08090b;
    border: 1px solid #232830;
    border-radius: 20px;
    margin-top: 80px;
    background-image: radial-gradient(circle at 50% 50%, #151a24 0%, #08090b 100%);
    box-shadow: 0 20px 50px rgba(0,0,0,0.5);
    animation: fadeIn 0.5s ease;
}}
.loading-title {{
    font-family: 'Syncopate', sans-serif !important;
    font-size: 16px;
    font-weight: 700;
    color: #8a939f;
    letter-spacing: 3px;
    margin-bottom: 16px;
    text-transform: uppercase;
}}
.loading-step {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 11px;
    color: #e30613;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
    text-shadow: 0 0 8px rgba(227, 6, 19, 0.3);
}}
.loading-sub {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px;
    color: #cbd5e1;
    margin-bottom: 4px;
    letter-spacing: 0.5px;
    font-weight: 600;
}}
.loading-details {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 11px;
    color: #8a939f;
    margin-bottom: 32px;
}}
.loading-bar-bg {{
    width: 240px;
    height: 4px;
    background: #232830;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}}
.loading-bar-fill {{
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, #e30613 0%, #ad000a 100%);
    border-radius: 10px;
    position: absolute;
    left: -100%;
    animation: loading 1.8s infinite ease-in-out;
}}
@keyframes loading {{
    0% {{ left: -100%; }}
    50% {{ left: 0; }}
    100% {{ left: 100%; }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: scale(0.98); }}
    to {{ opacity: 1; transform: scale(1); }}
}}
</style>
<div class="loading-container">
    <div class="brand-wrapper">
        <svg class="porsche-shield" viewBox="0 0 60 70" xmlns="http://www.w3.org/2000/svg">
            <path d="M30 2 L52 12 V38 C52 53 42 63 30 68 C18 63 8 53 8 38 V12 Z" fill="#0c0d10" stroke="#c5a059" stroke-width="2"/>
            <path d="M30 2 V35 H51" stroke="#c5a059" stroke-width="1"/>
            <path d="M30 35 H9" stroke="#c5a059" stroke-width="1"/>
            <path d="M12 18 H26 M12 26 H26" stroke="#e30613" stroke-width="3"/>
            <path d="M34 44 H48 M34 52 H48" stroke="#e30613" stroke-width="3"/>
            <path d="M35 18 C38 18 42 22 45 18 M35 25 C38 25 42 29 45 25" stroke="#c5a059" stroke-width="1.5" stroke-linecap="round" fill="none"/>
            <path d="M25 44 C22 44 18 48 15 44 M25 51 C22 51 18 55 15 51" stroke="#c5a059" stroke-width="1.5" stroke-linecap="round" fill="none"/>
            <path d="M24 25 H36 V43 C36 48 30 52 30 52 C30 52 24 48 24 43 Z" fill="#c5a059" stroke="#000" stroke-width="1"/>
            <path d="M29 33 C29 33 27 34 27 36 C27 38 31 37 30 41" stroke="#000" stroke-width="2" stroke-linecap="round" fill="none"/>
        </svg>
        <div class="porsche-wordmark">PORSCHE</div>
    </div>
    <div class="loading-title">INTELLIGENCE SYSTEM</div>
    <div class="loading-step">SYSTEM STARTUP: STAGE {step} OF 4</div>
    <div class="loading-sub">{title}</div>
    <div class="loading-details">{details}</div>
    <div class="loading-bar-bg">
        <div class="loading-bar-fill"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    try:
        # Connect to database (Cached resource)
        update_loading_status(1, "Connecting to Database", "Establishing connection to ChromaDB index...")
        store = get_vector_store()
        
        # Sync files
        update_loading_status(2, "Syncing Document Corpus", "Scanning local document directory and updating collection registry...")
        store.sync_with_folder(DATA_FOLDER)
        
        # Get docs (Cached resource)
        update_loading_status(3, "Loading Document Metadata", "Rebuilding memory mapping and loading titles...")
        docs = get_cached_docs()
        chunks = store.collection.count()
        
        # Load cross encoder models (Cached resource)
        update_loading_status(4, "Loading Cross-Encoder Reranker", "Importing sentence-transformers and loading MS-MARCO MiniLM...")
        _load_reranker()
        
        loading_placeholder.empty()
        st.session_state.initialized = True
    except Exception as e:
        loading_placeholder.empty()
        st.error(f"Failed to load: {e}")
        if "JWT" in str(e) or "auth" in str(e).lower():
            st.error(
                "GCP auth error — your `gcp-key.json` may be expired, the system clock may be wrong, "
                "or the service account lacks Vertex AI permissions. "
                "Try re-downloading the key from GCP Console."
            )
        st.info("The app needs a valid GCP service account key at `gcp-key.json` to run.")
        st.stop()
else:
    store = get_vector_store()
    docs = get_cached_docs()
    chunks = store.collection.count()

with st.sidebar:
    st.markdown(f'<div class="sb-title">PORSCHE</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sb-sub">Knowledge System</div>', unsafe_allow_html=True)

    top_k = st.slider("Chunks retrieved", min_value=1, max_value=10, value=3)
    rerank = st.toggle("Rerank results", value=True)
    hybrid = st.toggle("Hybrid search (BM25 + vector)", value=True)

    st.markdown(f"""
    <div class="sb-card">
        <div class="row">
            <div><div class="num">{len(docs)}</div><div class="lbl">Documents</div></div>
            <div><div class="num">{chunks}</div><div class="lbl">Chunks</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Document Repository"):
        for d in docs:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:8px; font-size:13px; color:#cbd5e1; padding: 6px 0;'>"
                f"<svg viewBox='0 0 24 24' fill='none' stroke='#ff2a3b' stroke-width='2' style='width:14px; height:14px; flex-shrink:0;'><path d='M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9C2 11.2 2 11.6 2 12v3c0 .6.4 1 1 1h2m10 0h4m-12 0a2.5 2.5 0 1 0 5 0 2.5 2.5 0 1 0-5 0zm10 0a2.5 2.5 0 1 0 5 0 2.5 2.5 0 1 0-5 0z'></path></svg>"
                f"<span>{d['title']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.markdown(f'<div class="sb-caption">Gemini 3.5 Flash · ChromaDB · Cross-encoder</div>', unsafe_allow_html=True)

# Header
logo_b64 = get_porsche_logo_base64()
logo_src = f"data:image/png;base64,{logo_b64}" if logo_b64 else "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Porsche_Crest.svg/200px-Porsche_Crest.svg.png"

st.markdown(f"""
<div class="hdr">
    <div class="ico">
        <img class="hdr-logo-img" src="{logo_src}" alt="Porsche Crest">
    </div>
    <div class="hdr-text">
        <h1>PORSCHE INTELLIGENCE</h1>
        <div class="sub">RAG Engine: Models, Engineering, Heritage & Motorsport</div>
    </div>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat messages
for msg in st.session_state.messages:
    role = msg["role"]
    bubble_class = "u" if role == "user" else "a"
    avatar_html = USER_SVG if role == "user" else ASSISTANT_SVG
    st.markdown(
        f'<div class="msg {role}">'
        f'<div class="av {bubble_class}">{avatar_html}</div>'
        f'<div class="bubble {bubble_class}">{msg["content"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if "sources" in msg and msg["sources"]:
        st.markdown(render_sources(msg["sources"], rerank), unsafe_allow_html=True)
        if "timings" in msg and msg["timings"]:
            st.markdown(msg["timings"], unsafe_allow_html=True)

# User input
if prompt := st.chat_input("Ask about model specs, lap times, engineering..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f'<div class="msg user">'
        f'<div class="av u">{USER_SVG}</div>'
        f'<div class="bubble u">{prompt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]

    t0 = time.perf_counter()
    with st.spinner("Analyzing queries..."):
        t1 = time.perf_counter()
        if hist:
            search_query = rewrite_search_query(prompt, hist)
        else:
            search_query = prompt
        t2 = time.perf_counter()
        retrieved = store.query(search_query, top_k=top_k, rerank=rerank, hybrid=hybrid)
        t3 = time.perf_counter()

    placeholder = st.empty()
    full = ""
    with st.spinner("Synthesizing answer..."):
        stream = llm_answer_stream(prompt, retrieved, history=hist, doc_dates=store.doc_dates)
        for token in stream:
            full += token
            placeholder.markdown(
                f'<div class="msg assistant">'
                f'<div class="av a">{ASSISTANT_SVG}</div>'
                f'<div class="bubble a">{full}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    t4 = time.perf_counter()

    sources_html = render_sources(retrieved, rerank)
    timings_content = f"rewrite: {t2-t1:.2f}s · retrieve: {t3-t2:.2f}s · generate: {t4-t3:.2f}s · total: {t4-t0:.2f}s"
    timings_html = (
        f'<div class="telemetry-row">'
        f'<span class="telemetry-tag">Telemetry</span>'
        f'<span class="telemetry-tag">{timings_content}</span>'
        f'</div>'
    )
    
    placeholder.markdown(
        f'<div class="msg assistant">'
        f'<div class="av a">{ASSISTANT_SVG}</div>'
        f'<div class="bubble a">{full}</div>'
        f'</div>'
        f'{sources_html}'
        f'{timings_html}',
        unsafe_allow_html=True,
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full,
        "sources": retrieved,
        "timings": timings_html,
    })
    st.rerun()

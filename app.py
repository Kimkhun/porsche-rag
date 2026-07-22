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

DATA_FOLDER = "data/sample_docs"

# Porsche-inspired Color Palette
R_GRADIENT = "linear-gradient(135deg, #e30613 0%, #ad000a 100%)"
R_GLOW = "0 0 16px rgba(227, 6, 19, 0.4)"
BG = "#08090b"
CARD = "#121418"
BORDER = "#232830"
TEXT_PRIMARY = "#f5f6f7"
TEXT_MUTED = "#8a939f"
ACCENT_SILVER = "#8f99a5"

st.set_page_config(page_title="Porsche Intelligence RAG", page_icon="🏎️", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syncopate:wght@700&family=Space+Grotesk:wght@400;500;700&display=swap');

/* ---- GLOBAL RESET & FONTS ---- */
* {{ 
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important; 
}}
.stApp {{ 
    background: {BG}; 
    background-image: radial-gradient(circle at 50% 0%, #151a24 0%, {BG} 70%);
}}
.stApp > header {{ background: transparent !important; }}
.stApp > header:before {{ display: none !important; }}
#root > div:first-child {{ background: transparent !important; }}
.block-container {{ padding-top: 25px !important; padding-bottom: 25px !important; max-width: 950px !important; }}

/* ---- TYPOGRAPHY ---- */
h1, h2, h3, .syncopate {{
    font-family: 'Syncopate', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
}}

/* ---- HEADER ---- */
.hdr {{ 
    display: flex; 
    align-items: center; 
    gap: 20px; 
    padding: 16px; 
    margin-bottom: 28px; 
    background: rgba(18, 20, 24, 0.6);
    border: 1px solid {BORDER};
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}}
.hdr .ico {{ 
    font-size: 32px; 
    line-height: 1; 
    display: flex;
    align-items: center;
    justify-content: center;
    background: {R_GRADIENT};
    padding: 12px;
    border-radius: 12px;
    box-shadow: {R_GLOW};
}}
.hdr h1 {{ 
    font-size: 22px; 
    margin: 0; 
    color: #fff; 
    letter-spacing: 2px;
    background: linear-gradient(180deg, #fff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}
.hdr .sub {{ 
    font-size: 11px; 
    color: {TEXT_MUTED}; 
    text-transform: uppercase; 
    letter-spacing: 1.5px; 
    margin-top: 4px; 
    font-weight: 500;
}}

/* ---- CHAT BUBBLES ---- */
.msg {{ 
    display: flex; 
    gap: 16px; 
    margin: 20px 0; 
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
    width: 40px; 
    height: 40px; 
    border-radius: 10px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    flex-shrink: 0; 
    transition: transform 0.2s ease;
}}
.av:hover {{
    transform: scale(1.05);
}}
.av.u {{ 
    background: {R_GRADIENT}; 
    color: #fff; 
    box-shadow: {R_GLOW};
}}
.av.a {{ 
    background: #1e222b; 
    color: {ACCENT_SILVER}; 
    border: 1px solid {BORDER}; 
}}
.av-svg {{
    width: 20px;
    height: 20px;
}}

.bubble {{ 
    max-width: 78%; 
    padding: 16px 22px; 
    border-radius: 16px; 
    font-size: 15px; 
    line-height: 1.6; 
    color: {TEXT_PRIMARY}; 
    word-wrap: break-word; 
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    border: 1px solid {BORDER};
}}
.bubble.u {{ 
    background: rgba(227, 6, 19, 0.08); 
    border-color: rgba(227, 6, 19, 0.3);
    border-bottom-right-radius: 4px; 
}}
.bubble.a {{ 
    background: rgba(18, 20, 24, 0.85); 
    backdrop-filter: blur(8px);
    border-color: {BORDER};
    border-bottom-left-radius: 4px; 
}}
.bubble p {{ margin: 0; }}
.bubble p + p {{ margin-top: 12px; }}

/* ---- SOURCES & TELEMETRY ---- */
.src {{ 
    margin: 8px 0 0 56px; 
    max-width: 78%;
}}
.src details {{
    background: rgba(18, 20, 24, 0.4);
    border: 1px solid {BORDER};
    border-radius: 8px;
    overflow: hidden;
}}
.src summary {{ 
    cursor: pointer; 
    color: {TEXT_MUTED}; 
    font-size: 12px; 
    font-weight: 600;
    letter-spacing: 0.5px; 
    padding: 10px 14px; 
    user-select: none;
    text-transform: uppercase;
    transition: all 0.2s ease;
}}
.src summary:hover {{ 
    color: #fff; 
    background: rgba(255, 255, 255, 0.02);
}}
.src .chunk-card {{
    padding: 12px 14px;
    border-top: 1px solid {BORDER};
    background: rgba(10, 11, 14, 0.7);
}}
.src .chunk-header {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    font-weight: 700;
    color: {ACCENT_SILVER};
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.src .chunk-body {{
    color: #b3bcc7;
    font-size: 13px;
    line-height: 1.5;
    white-space: pre-wrap;
    font-family: 'Space Grotesk', monospace !important;
}}
.telemetry-row {{
    display: flex;
    gap: 8px;
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 600;
    margin-top: 8px;
    margin-left: 56px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-family: 'Space Grotesk', sans-serif !important;
}}
.telemetry-tag {{
    background: rgba(255, 255, 255, 0.04);
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
    font-weight: 500;
}}
[data-testid="stSidebar"] hr {{ 
    border-color: {BORDER} !important; 
}}
[data-testid="stSidebar"] .streamlit-expanderHeader {{ 
    color: {TEXT_MUTED} !important; 
    font-size: 12px !important; 
    background: transparent !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px;
    margin-bottom: 6px;
}}
[data-testid="stSidebar"] .stCaption {{ 
    color: {TEXT_MUTED} !important; 
    font-size: 11px !important; 
}}
.sb-title {{ 
    font-size: 20px; 
    font-weight: 700; 
    color: #fff; 
    letter-spacing: 3px;
    margin-bottom: 2px;
}}
.sb-sub {{ 
    font-size: 10px; 
    color: {TEXT_MUTED}; 
    text-transform: uppercase; 
    letter-spacing: 2px; 
    margin-bottom: 24px; 
    font-weight: 600;
}}
.sb-card {{ 
    background: {CARD}; 
    border: 1px solid {BORDER}; 
    border-radius: 12px; 
    padding: 16px; 
    margin: 20px 0; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
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
    font-weight: 600;
}}

/* ---- INPUT CONTAINER ---- */
div:has(> [data-testid="stChatInput"]) {{ 
    max-width: 950px !important; 
    margin: 0 auto !important; 
    padding: 0 16px !important; 
}}
[data-testid="stChatInput"] {{ 
    border: none !important; 
    background: transparent !important;
}}
[data-testid="stChatInput"] textarea {{ 
    background: rgba(18, 20, 24, 0.9) !important; 
    color: #fff !important; 
    border: 1px solid {BORDER} !important; 
    border-radius: 14px !important; 
    padding: 16px 20px !important; 
    font-size: 15px !important; 
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.3s ease !important;
}}
[data-testid="stChatInput"] textarea:focus {{ 
    border-color: #e30613 !important; 
    box-shadow: 0 0 0 2px rgba(227, 6, 19, 0.2), 0 10px 30px rgba(0, 0, 0, 0.3) !important; 
}}
[data-testid="stChatInput"] button {{ 
    background: {R_GRADIENT} !important; 
    border-radius: 10px !important; 
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
    border-color: #e30613 transparent transparent transparent !important; 
}}
.stSpinner p {{ 
    color: {TEXT_MUTED} !important; 
    font-size: 13px;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background-color: #e30613 !important;
}}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_SILVER}; }}
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

@st.cache_resource(show_spinner="Loading and indexing documents...")
def load_store():
    store = VectorStore()
    store.sync_with_folder(DATA_FOLDER)
    docs = load_documents(DATA_FOLDER)
    chunks = store.collection.count()
    _load_reranker()
    return store, docs, chunks


try:
    store, docs, chunks = load_store()
except Exception as e:
    st.error(f"Failed to load: {e}")
    if "JWT" in str(e) or "auth" in str(e).lower():
        st.error(
            "GCP auth error — your `gcp-key.json` may be expired, the system clock may be wrong, "
            "or the service account lacks Vertex AI permissions. "
            "Try re-downloading the key from GCP Console."
        )
    st.info("The app needs a valid GCP service account key at `gcp-key.json` to run.")
    st.stop()

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
            st.markdown(f"<div style='font-size:13px; color:#cbd5e1; padding: 4px 0;'>🏎️ {d['title']}</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("Gemini 3.5 Flash · ChromaDB · Cross-encoder")

# Header
st.markdown(f"""
<div class="hdr">
    <div class="ico">🏁</div>
    <div>
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
        search_query = rewrite_search_query(prompt, hist)
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


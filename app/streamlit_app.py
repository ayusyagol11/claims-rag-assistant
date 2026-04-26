"""
Claims Document Intelligence — Streamlit UI
Commercial LLM-style layout: centered landing → chat view on first message.
Brand: zinc dark + amber accent, matching aayushyagol.com.
"""

import logging

import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def ensure_index() -> None:
    """Build the ChromaDB index on first startup if it doesn't exist yet."""
    import chromadb
    from src.config import CHROMADB_DIR, COLLECTION_NAME
    from src.ingest import build_index

    try:
        client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() > 0:
            logger.info("Index already exists (%d chunks). Skipping ingest.", collection.count())
            return
    except Exception:
        pass  # collection doesn't exist yet — fall through to build

    logger.info("No index found. Building from data/raw/ …")
    build_index()


def ask(query: str) -> dict:
    from src.rag_pipeline import ask as _ask
    return _ask(query)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Claims Document Intelligence",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Ensure vector index exists (runs once per server process) ─────────────────
with st.spinner("Loading document index… (first launch takes ~2 minutes)"):
    ensure_index()

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

is_empty = len(st.session_state.messages) == 0

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Google+Sans+Mono:wght@400;500&display=swap');

:root {
    --font-sans:    'Google Sans', 'DM Sans', sans-serif;
    --font-mono:    'Google Sans Mono', 'JetBrains Mono', monospace;
    --bg:           #09090B;
    --bg-card:      #18181B;
    --bg-raised:    #27272A;
    --border:       #3F3F46;
    --text:         #FAFAFA;
    --text-muted:   #A1A1AA;
    --amber:        #BA7517;
    --amber-mid:    #EF9F27;
    --amber-light:  #FAEEDA;
    --amber-dark:   #633806;
    --r-sm: 6px; --r-md: 10px; --r-lg: 14px;
}

/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
}
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

/* ── Main content area ───────────────────────────────────────────────────── */
.main .block-container {
    padding: 0 2.5rem 5rem 2.5rem !important;
    max-width: 780px !important;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    font-family: var(--font-sans) !important;
}

/* ── New Chat button ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type button {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    color: var(--text) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.15s, background-color 0.15s !important;
}
[data-testid="stSidebar"] [data-testid="stButton"]:first-of-type button:hover {
    border-color: var(--amber) !important;
    background-color: #1f1f22 !important;
    color: var(--text) !important;
}

/* ── Chat messages ───────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
}
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--amber) !important;
    background-color: var(--bg-raised) !important;
}
[data-testid="chatAvatarIcon-user"] {
    background-color: var(--amber) !important;
    color: var(--amber-dark) !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
    line-height: 1.7 !important;
    font-size: 0.925rem !important;
}
[data-testid="stChatMessage"] strong { color: var(--amber-mid) !important; }
[data-testid="stChatMessage"] code {
    font-family: var(--font-mono) !important;
    background: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    padding: 0.1em 0.4em !important;
    font-size: 0.85em !important;
    color: var(--amber-light) !important;
}

/* ── Chat input ──────────────────────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(186, 117, 23, 0.12) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: var(--font-sans) !important;
    color: var(--text) !important;
    background: transparent !important;
    font-size: 0.925rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: var(--text-muted) !important; }
[data-testid="stChatInput"] button {
    background-color: var(--amber) !important;
    border-radius: var(--r-sm) !important;
    color: var(--amber-dark) !important;
}
[data-testid="stChatInput"] button:hover { background-color: var(--amber-mid) !important; }

/* ── Example question buttons ────────────────────────────────────────────── */
.example-btn button {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    color: var(--text-muted) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.82rem !important;
    text-align: left !important;
    padding: 0.75rem 1rem !important;
    line-height: 1.45 !important;
    transition: border-color 0.15s, color 0.15s, background-color 0.15s !important;
    min-height: 3.75rem !important;
    width: 100% !important;
}
.example-btn button:hover {
    border-color: var(--amber) !important;
    color: var(--text) !important;
    background-color: var(--bg-raised) !important;
}

/* ── Expanders ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background-color: var(--bg-raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    margin-top: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}
[data-testid="stExpander"] summary:hover { color: var(--amber-mid) !important; }
[data-testid="stExpander"] summary svg { color: var(--amber) !important; }
[data-testid="stExpander"] > div > div {
    padding: 0.75rem 1rem 1rem !important;
    border-top: 1px solid var(--border) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--amber); }
</style>
""", unsafe_allow_html=True)


# ── Helper: render source citations ──────────────────────────────────────────
def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} sources retrieved"):
        for s in sources:
            st.markdown(f"""
<div style="padding:0.6rem 0;border-bottom:1px solid #27272A;margin-bottom:0.2rem;">
  <div style="font-family:'Google Sans',sans-serif;font-size:0.82rem;font-weight:500;
              color:#FAFAFA;margin-bottom:0.2rem;">[{s['index']}] {s['file']}</div>
  <div style="font-family:'Google Sans Mono',monospace;font-size:0.68rem;color:#BA7517;
              margin-bottom:0.35rem;letter-spacing:0.04em;">relevance · {s['score']:.3f}</div>
  <div style="font-family:'Google Sans',sans-serif;font-size:0.78rem;color:#A1A1AA;
              line-height:1.5;border-left:2px solid #3F3F46;padding-left:0.75rem;">
    {s['preview']}…</div>
</div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand mark
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;
                padding:0.25rem 0 1.25rem 0;border-bottom:1px solid #3F3F46;
                margin-bottom:1.25rem;">
        <span style="font-family:'Google Sans Mono',monospace;font-size:1.35rem;
                     font-weight:700;color:#FAFAFA;letter-spacing:-0.02em;">
            AY<span style="color:#BA7517;">.</span>
        </span>
        <div>
            <div style="font-family:'Google Sans',sans-serif;font-size:0.78rem;
                        font-weight:500;color:#FAFAFA;">Claims Intelligence</div>
            <div style="font-family:'Google Sans Mono',monospace;font-size:0.63rem;
                        color:#A1A1AA;letter-spacing:0.06em;text-transform:uppercase;">
                RAG · v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Chat button
    if st.button("✦  New chat", use_container_width=True, key="new_chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # Corpus list
    st.markdown("""
    <div style="font-family:'Google Sans Mono',monospace;font-size:0.63rem;
                color:#A1A1AA;text-transform:uppercase;letter-spacing:0.09em;
                margin-bottom:0.6rem;">Document corpus</div>
    """, unsafe_allow_html=True)

    corpus = [
        ("📄", "Workers Compensation Act 1951"),
        ("📄", "Workers Compensation Regulation 2002"),
        ("📋", "National RTW Strategy 2020–2030"),
        ("📋", "Comcare RTW Employee Overview"),
        ("📋", "SIRA Standards of Practice 2025"),
        ("📊", "APRA GPS 320 — Actuarial Matters"),
        ("🗂️", "Synthetic Claim Narratives ×10"),
    ]
    for icon, name in corpus:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:0.38rem 0;
                    border-bottom:1px solid #27272A;">
            <span style="font-size:0.73rem;">{icon}</span>
            <span style="font-family:'Google Sans',sans-serif;font-size:0.77rem;
                         color:#A1A1AA;line-height:1.3;">{name}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    with st.expander("⚠️ Disclaimer"):
        st.markdown("""
        <div style="font-family:'Google Sans',sans-serif;font-size:0.77rem;
                    color:#A1A1AA;line-height:1.6;">
            <strong style="color:#FAFAFA;">Decision-support only.</strong>
            Answers are grounded in the indexed corpus with citations.
            Not legal, medical, or actuarial advice.<br><br>
            Verify critical decisions against source documents directly.
            Synthetic claim narratives are fictional.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:1.25rem;font-family:'Google Sans Mono',monospace;
                font-size:0.6rem;color:#3F3F46;text-transform:uppercase;
                letter-spacing:0.06em;">aayushyagol.com · 2026</div>
    """, unsafe_allow_html=True)


# ── LANDING STATE ─────────────────────────────────────────────────────────────
if is_empty:
    st.markdown("<div style='height:16vh;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-bottom:2.75rem;">
        <div style="font-family:'Google Sans Mono',monospace;font-size:0.68rem;
                    color:#BA7517;text-transform:uppercase;letter-spacing:0.14em;
                    margin-bottom:1rem;">
            ACT Workers Compensation · Insurance RAG
        </div>
        <h1 style="font-family:'Google Sans',sans-serif;font-size:2.15rem;
                   font-weight:700;color:#FAFAFA;margin:0 0 0.8rem 0;
                   line-height:1.15;letter-spacing:-0.025em;">
            What can I help you<br>find today?
        </h1>
        <p style="font-family:'Google Sans',sans-serif;font-size:0.92rem;
                  color:#A1A1AA;margin:0 auto;max-width:500px;line-height:1.65;">
            Ask about workers compensation legislation, return-to-work policy,
            or claim procedures. Every answer cites its source document.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 2×2 example question grid
    example_questions = [
        "What weekly compensation is a worker entitled to during total incapacity?",
        "What is the difference between total and partial incapacity?",
        "What are an employer's return-to-work obligations after a workplace injury?",
        "What is a biopsychosocial approach to rehabilitation?",
    ]

    col1, col2 = st.columns(2, gap="small")
    for i, q in enumerate(example_questions):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown('<div class="example-btn">', unsafe_allow_html=True)
            if st.button(q, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ── ACTIVE CHAT STATE ─────────────────────────────────────────────────────────
else:
    # Slim header
    st.markdown("""
    <div style="padding:1.5rem 0 1rem 0;border-bottom:1px solid #27272A;
                margin-bottom:1.5rem;">
        <div style="font-family:'Google Sans Mono',monospace;font-size:0.63rem;
                    color:#BA7517;text-transform:uppercase;letter-spacing:0.12em;
                    margin-bottom:0.3rem;">
            Claims Intelligence · ACT Workers Compensation
        </div>
        <div style="font-family:'Google Sans',sans-serif;font-size:1.05rem;
                    font-weight:600;color:#FAFAFA;">
            Claims Document Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])


# ── CHAT INPUT (always at bottom) ─────────────────────────────────────────────
typed_prompt = st.chat_input(
    "Ask about workers compensation, legislation, or claim procedures…"
)
prompt = typed_prompt or st.session_state.pop("pending_question", None)

if prompt:
    # First message from landing: switch layout then re-deliver the prompt
    if is_empty:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.pending_question = prompt
        st.rerun()

    # On the rerun, pop again and avoid double-appending the user message
    prompt = st.session_state.pop("pending_question", None) or prompt

    last = st.session_state.messages[-1] if st.session_state.messages else {}
    if last.get("role") != "user" or last.get("content") != prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and generating answer…"):
            try:
                result = ask(prompt)
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as exc:
                logger.error("Pipeline error for query '%.80s': %s", prompt, exc, exc_info=True)
                st.error(
                    "Something went wrong while processing your question. "
                    "Please check your API key and that the index has been built, then try again."
                )
                st.stop()

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })

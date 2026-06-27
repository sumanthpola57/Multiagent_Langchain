import streamlit as st
import time
import re
from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #edf3ff;
}

.stApp {
    background: #07111f;
    background-image:
        radial-gradient(circle at top left, rgba(0,191,255,0.14), transparent 32%),
        radial-gradient(circle at bottom right, rgba(124,58,237,0.12), transparent 30%),
        linear-gradient(180deg, #07111f 0%, #0a1729 100%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1rem;
    opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f8fbff;
    margin: 0 0 1rem;
}
.hero h1 span {
    background: linear-gradient(135deg, #38bdf8, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent !important;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #b5c3d9;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.65;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.35), transparent);
    margin: 2rem 0;
}

.input-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 22px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 40px rgba(0,0,0,0.28);
}

.stTextInput > div > div > input,
.stTextInput > div > div > input:focus,
[data-baseweb="input"] input,
[data-baseweb="base-input"] input {
    background: #0d1f35 !important;
    border: 1px solid rgba(56,189,248,0.25) !important;
    border-radius: 12px !important;
    color: #f8fbff !important;
    -webkit-text-fill-color: #f8fbff !important;
    caret-color: #38bdf8 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.8rem 1rem !important;
    transition: all 0.2s ease !important;
}
[data-baseweb="input"],
[data-baseweb="base-input"] {
    background: #0d1f35 !important;
    border-radius: 12px !important;
}
.stTextInput > div > div > input::placeholder {
    color: #4e6278 !important;
    -webkit-text-fill-color: #4e6278 !important;
}
.stTextInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 4px rgba(56,189,248,0.14) !important;
}
.stTextInput > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #38bdf8 !important;
    font-weight: 500 !important;
    -webkit-text-fill-color: #38bdf8 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em {
    color: #e2eafc !important;
    -webkit-text-fill-color: #e2eafc !important;
}

.stButton > button {
    background: linear-gradient(135deg, #38bdf8 0%, #8b5cf6 100%) !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 2.2rem !important;
    cursor: pointer !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 8px 30px rgba(56,189,248,0.22) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.01) !important;
    box-shadow: 0 12px 35px rgba(56,189,248,0.32) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.step-card {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease;
    backdrop-filter: blur(10px);
}
.step-card:hover { transform: translateY(-2px); border-color: rgba(56,189,248,0.18); }
.step-card.active { border-color: rgba(56,189,248,0.45); background: rgba(56,189,248,0.06); }
.step-card.done   { border-color: rgba(34,197,94,0.28);  background: rgba(34,197,94,0.05);  }
.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    border-radius: 18px 0 0 18px;
    background: rgba(255,255,255,0.06);
    transition: background 0.3s;
}
.step-card.active::before { background: #38bdf8; }
.step-card.done::before   { background: #22c55e; }

.step-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.3rem; }
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #38bdf8;
    opacity: 0.85;
}
.step-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #f8fbff; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; }
.status-waiting { color: #64748b; }
.status-running { color: #38bdf8; }
.status-done    { color: #22c55e; }

.result-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
}
.result-panel-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1rem;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(56,189,248,0.15);
}

.report-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-top: 1rem;
    backdrop-filter: blur(14px);
}
.feedback-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-top: 1rem;
    backdrop-filter: blur(14px);
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
}
.panel-label.orange {
    color: #38bdf8;
    -webkit-text-fill-color: #38bdf8;
    border-bottom: 1px solid rgba(56,189,248,0.15);
}
.panel-label.green {
    color: #22c55e;
    -webkit-text-fill-color: #22c55e;
    border-bottom: 1px solid rgba(34,197,94,0.15);
}

.stSpinner > div { color: #38bdf8 !important; }

details {
    background: rgba(255,255,255,0.02);
    border-radius: 14px;
    padding: 0.3rem 0.8rem;
    border: 1px solid rgba(255,255,255,0.06);
}
details summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #b5c3d9 !important;
    -webkit-text-fill-color: #b5c3d9 !important;
    letter-spacing: 0.1em !important;
    cursor: pointer;
}

.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fbff;
    margin: 2rem 0 1rem;
}

.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #7f93ad;
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

# Domains that cannot be scraped (JS-heavy, auth-walled, or block bots)
UNSCRAPPABLE_DOMAINS = {
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "instagram.com", "facebook.com",
    "linkedin.com", "tiktok.com",
    "reddit.com",                   # Usually works but returns mobile redirect; skip
    "researchgate.net",             # Auth wall
    "academia.edu",                 # Auth wall
}

def is_scrappable(url: str) -> bool:
    """Return True only for URLs worth handing to the scraper."""
    url_lower = url.lower()
    # Skip non-HTTP
    if not url_lower.startswith("http"):
        return False
    # Skip PDFs (often need special handling)
    if url_lower.endswith(".pdf"):
        return False
    # Skip known bad domains
    for domain in UNSCRAPPABLE_DOMAINS:
        if domain in url_lower:
            return False
    return True

def safe_html(text: str) -> str:
    """Escape < and > so raw AI output doesn't break innerHTML panels."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def extract_scrappable_urls(text: str, limit: int = 5) -> list[str]:
    """Pull URLs from text and filter to only scrappable ones."""
    all_urls = re.findall(r'https?://[^\s\n\)\]\"\']+', text)
    seen = set()
    good = []
    for u in all_urls:
        u = u.rstrip(".,;)")   # strip trailing punctuation
        if u not in seen and is_scrappable(u):
            seen.add(u)
            good.append(u)
        if len(good) >= limit:
            break
    return good


def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING",    "status-waiting"),
        "running": ("● RUNNING",  "status-running"),
        "done":    ("✓ DONE",     "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.82rem;color:#94a3b8;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ("results", "running", "done"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else False


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Researcher<span>Agent</span></h1>
    <p class="hero-sub">
        Four specialized AI agents collaborate — searching, scraping, writing,
        and critiquing — to deliver a polished research report on any topic.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Roadmap for AGI development in next 5 years",
        key="topic_input",
        label_visibility="visible",
    )

    run_btn = st.button("⚡ Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;align-items:center;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#7f93ad;letter-spacing:0.1em;">TRY →</span>
    """, unsafe_allow_html=True)

    for ex in [
        "Future of LLM in Tech Industry",
        "All Latest AI Agents in 2026",
        "Roadmap for AGI development in next 5 years",
    ]:
        st.markdown(f"""
        <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
            border-radius:8px;padding:0.35rem 0.8rem;font-size:0.75rem;color:#d8e2f0;
            font-family:'DM Sans',sans-serif;cursor:default;">{ex}</span>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)

    slot_search = st.empty()
    slot_reader = st.empty()
    slot_writer = st.empty()
    slot_critic = st.empty()

    def render_pipeline(current_step: str | None = None):
        """
        current_step: one of "search","reader","writer","critic", or None.
        Steps before current_step → done. current_step → running. Rest → waiting.
        """
        order  = ["search", "reader", "writer", "critic"]
        labels = {
            "search": ("01", "Search Agent",  "Gathers recent web information"),
            "reader": ("02", "Reader Agent",  "Scrapes & extracts deep content"),
            "writer": ("03", "Writer Chain",  "Drafts the full research report"),
            "critic": ("04", "Critic Chain",  "Reviews & scores the report"),
        }
        slots  = {
            "search": slot_search,
            "reader": slot_reader,
            "writer": slot_writer,
            "critic": slot_critic,
        }

        r = st.session_state.results
        done_steps = set(r.keys())

        for step in order:
            if step in done_steps:
                state = "done"
            elif step == current_step:
                state = "running"
            else:
                state = "waiting"

            num, title, desc = labels[step]
            with slots[step]:
                step_card(num, title, state, desc)

    render_pipeline()


# ── Trigger run ───────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results  = {}
        st.session_state.running  = True
        st.session_state.done     = False
        st.rerun()


# ── Pipeline execution ────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.done:

    results    = {}
    topic_val  = st.session_state.topic_input

    # ── Step 1: Search ──────────────────────────────────────────────────────
    render_pipeline(current_step="search")
    with st.spinner("🔍 Search Agent is working…"):
        try:
            search_agent = build_search_agent()
            sr = search_agent.invoke({
                "messages": [("user",
                    f"Find recent, reliable and detailed information about: {topic_val}")]
            })
            results["search"] = sr["messages"][-1].content
        except Exception as e:
            results["search"] = f"Search agent error: {str(e)}"

    st.session_state.results = dict(results)

    # ── Step 2: Reader ──────────────────────────────────────────────────────
    render_pipeline(current_step="reader")
    with st.spinner("📄 Reader Agent is scraping top resources…"):
        try:
            # Filter URLs before sending to Reader so it never tries YouTube etc.
            scrappable = extract_scrappable_urls(results["search"], limit=4)
            urls_for_reader = "\n".join(scrappable) if scrappable else "(no scrappable URLs found)"

            reader_agent = build_reader_agent()
            rr = reader_agent.invoke({
                "messages": [("user",
                    f"Scrape and extract deep content from these URLs about '{topic_val}'.\n"
                    f"Only use URLs from this list (skip any not listed here):\n{urls_for_reader}\n\n"
                    f"Context from search:\n{results['search'][:1500]}"
                )]
            })
            results["reader"] = rr["messages"][-1].content
        except Exception as e:
            results["reader"] = f"Reader agent error: {str(e)}"

    st.session_state.results = dict(results)

    # ── Step 3: Writer ──────────────────────────────────────────────────────
    render_pipeline(current_step="writer")
    with st.spinner("✍️ Writer is drafting the report…"):
        try:
            scrappable_urls = extract_scrappable_urls(results["search"], limit=10)
            urls_block = "\n".join(
                [f"{i+1}. {url}" for i, url in enumerate(scrappable_urls)]
            )

            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results['reader']}\n\n"
                f"EXTRACTED URLS FOR SOURCES SECTION (use these, skip YouTube/social):\n{urls_block}"
            )
            results["writer"] = writer_chain.invoke({
                "topic":    topic_val,
                "research": research_combined,
            })
        except Exception as e:
            results["writer"] = f"Writer error: {str(e)}"

    st.session_state.results = dict(results)

    # ── Step 4: Critic ──────────────────────────────────────────────────────
    render_pipeline(current_step="critic")
    with st.spinner("🧐 Critic is reviewing the report…"):
        try:
            results["critic"] = critic_chain.invoke({"report": results["writer"]})
        except Exception as e:
            results["critic"] = f"Critic error: {str(e)}"

    st.session_state.results  = dict(results)
    st.session_state.running  = False
    st.session_state.done     = True
    st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    if "search" in r:
        with st.expander("🔍 Search Results (raw)", expanded=False):
            st.markdown(
                f'<div class="result-panel">'
                f'<div class="result-panel-title">Search Agent Output</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            # Use st.text for raw AI output — no HTML injection risk
            st.text(r["search"])

    if "reader" in r:
        with st.expander("📄 Scraped Content (raw)", expanded=False):
            st.markdown(
                f'<div class="result-panel">'
                f'<div class="result-panel-title">Reader Agent Output</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.text(r["reader"])

    if "writer" in r:
        st.markdown("""
        <div class="report-panel">
            <div class="panel-label orange">📝 Final Research Report</div>
        </div>
        """, unsafe_allow_html=True)
        # st.markdown renders the report's own markdown safely
        st.markdown(r["writer"])

        st.download_button(
            label="⬇ Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown("""
        <div class="feedback-panel">
            <div class="panel-label green">🧐 Critic Feedback</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(r["critic"])

    # Re-render pipeline as all-done after results load
    render_pipeline()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchAgent · Powered by LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)
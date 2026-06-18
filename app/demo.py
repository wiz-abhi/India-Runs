"""
Streamlit demo — Intelligent Candidate Ranking System
India Runs by Redrob AI · Track 1: Data & AI Challenge

Run with:
    streamlit run app/demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root on import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.embeddings import EmbeddingEngine
from src.jd_parser import JDParser
from src.profile_parser import ProfileParser
from src.ranker import CandidateRanker, RankedCandidate
from src.utils import OUTPUTS_DIR, ensure_dir

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndiaRanks · Intelligent Candidate Ranking",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* Header */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #a0a0c0;
        font-size: 1.05rem;
        margin-top: -0.5rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Candidate row */
    .candidate-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
    }
    .candidate-card:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(102, 126, 234, 0.3);
    }

    /* Rank badges */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.95rem;
        margin-right: 0.8rem;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #1a1a2e; }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A8A8A8); color: #1a1a2e; }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #B87333); color: #1a1a2e; }
    .rank-n { background: rgba(102, 126, 234, 0.2); color: #667eea; }

    /* Progress bar */
    .match-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .match-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.6s ease;
    }

    /* Reason tag */
    .reason-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #a0b0ff;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 4px;
    }

    /* Flag */
    .flag-note {
        color: #ff6b6b;
        font-size: 0.8rem;
        font-style: italic;
    }

    /* Signal bars */
    .signal-row {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .signal-label {
        width: 130px;
        font-size: 0.75rem;
        color: #8888aa;
    }
    .signal-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .signal-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .signal-val {
        width: 40px;
        text-align: right;
        font-size: 0.75rem;
        color: #ccc;
        margin-left: 8px;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* How it works */
    .how-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    .how-section h3 { color: #667eea; }
    .how-section p { color: #a0a0c0; font-size: 0.9rem; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ─────────────────────────────────────────────────────
def _rank_badge(rank: int) -> str:
    """Return HTML for a rank badge."""
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    cls = f"rank-{rank}" if rank <= 3 else "rank-n"
    label = medals.get(rank, str(rank))
    return f'<span class="rank-badge {cls}">{label}</span>'


def _match_bar(pct: int) -> str:
    """Return HTML for a match percentage progress bar."""
    return f"""
    <div style="display:flex;align-items:center;gap:8px;">
        <div class="match-bar-bg"><div class="match-bar-fill" style="width:{pct}%"></div></div>
        <span style="color:#fff;font-weight:600;font-size:0.9rem;">{pct}%</span>
    </div>
    """


def _signal_bar(label: str, value: float, color: str = "#667eea") -> str:
    """Return HTML for a single signal bar."""
    pct = int(value * 100)
    return f"""
    <div class="signal-row">
        <span class="signal-label">{label}</span>
        <div class="signal-bar-bg">
            <div class="signal-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <span class="signal-val">{value:.2f}</span>
    </div>
    """


SIGNAL_COLORS = {
    "semantic_similarity": "#667eea",
    "skill_match": "#764ba2",
    "skill_recency": "#f093fb",
    "experience_fit": "#43e97b",
    "career_velocity": "#38f9d7",
    "domain_alignment": "#fa709a",
    "profile_freshness": "#fee140",
    "culture_fit": "#30cfd0",
    "education_tier_bonus": "#a18cd1",
}

SIGNAL_LABELS = {
    "semantic_similarity": "Semantic Match",
    "skill_match": "Skill Match",
    "skill_recency": "Skill Recency",
    "experience_fit": "Experience Fit",
    "career_velocity": "Career Velocity",
    "domain_alignment": "Domain Align",
    "profile_freshness": "Freshness",
    "culture_fit": "Culture Fit",
    "education_tier_bonus": "Education Bonus",
}


# ── Cache shared resources ───────────────────────────────────────────────
@st.cache_resource
def get_engine() -> EmbeddingEngine:
    return EmbeddingEngine()


@st.cache_resource
def get_ranker() -> CandidateRanker:
    return CandidateRanker(embedding_engine=get_engine())


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇮🇳 IndiaRanks")
    st.markdown("---")

    jd_mode = st.radio(
        "Job Description Input",
        ["📝 Paste text", "📂 Upload JSON", "📄 Use sample JD"],
        index=2,
        key="jd_mode",
    )

    jd_text = ""
    if jd_mode == "📝 Paste text":
        jd_text = st.text_area(
            "Paste JD here",
            height=200,
            placeholder="Senior Machine Learning Engineer at…",
        )
    elif jd_mode == "📂 Upload JSON":
        uploaded_jd = st.file_uploader("Upload JD (JSON)", type=["json"])
        if uploaded_jd:
            jd_text = json.loads(uploaded_jd.read().decode("utf-8"))
    else:
        sample_path = Path(__file__).parent.parent / "data" / "sample" / "jd.json"
        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                jd_text = json.load(f)
            st.success("Using sample JD ✓")
        else:
            st.warning("Sample JD not found — paste or upload.")

    st.markdown("---")

    profile_mode = st.radio(
        "Candidate Profiles",
        ["📂 Upload CSV", "📄 Use sample profiles"],
        index=1,
        key="prof_mode",
    )

    profiles_df = None
    if profile_mode == "📂 Upload CSV":
        uploaded = st.file_uploader("Upload Profiles (CSV)", type=["csv", "xlsx", "json"])
        if uploaded:
            suffix = uploaded.name.split(".")[-1].lower()
            if suffix == "csv":
                profiles_df = pd.read_csv(uploaded)
            elif suffix in ("xlsx", "xls"):
                profiles_df = pd.read_excel(uploaded)
            elif suffix == "json":
                profiles_df = pd.DataFrame(json.loads(uploaded.read().decode("utf-8")))
    else:
        sample_profiles = Path(__file__).parent.parent / "data" / "sample" / "profiles.csv"
        if sample_profiles.exists():
            profiles_df = pd.read_csv(sample_profiles)
            st.success(f"Loaded {len(profiles_df)} sample profiles ✓")

    st.markdown("---")

    num_show = st.slider("Candidates to show", 5, 50, 10, key="num_show")
    show_signals = st.toggle("Show signal breakdown", value=True, key="show_signals")

    run_btn = st.button("🚀 Rank Candidates", use_container_width=True, type="primary")


# ── MAIN PANEL ───────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">IndiaRanks</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Intelligent Candidate Ranking · Powered by Semantic AI + India-Native Signals</p>',
    unsafe_allow_html=True,
)

if run_btn and jd_text and profiles_df is not None:
    with st.spinner("⚡ Running ranking pipeline…"):
        jd_parser = JDParser()
        jd = jd_parser.parse(jd_text)

        profile_parser = ProfileParser()
        raw_records = profiles_df.to_dict(orient="records")
        profiles = profile_parser.parse_many(raw_records)

        ranker = get_ranker()
        ranked = ranker.rank(profiles, jd, refresh_embeddings=True)
        ranked = ranked[:num_show]

    # ── JD Summary Card ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Parsed Job Description")
    cols = st.columns(4)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{jd.seniority_level.title()}</div>'
            f'<div class="metric-label">Seniority</div></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{jd.min_experience_years:.0f}+ yrs</div>'
            f'<div class="metric-label">Min Experience</div></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{jd.domain.replace("_", " ").title()}</div>'
            f'<div class="metric-label">Domain</div></div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{jd.location_preference or "Any"}</div>'
            f'<div class="metric-label">Location</div></div>',
            unsafe_allow_html=True,
        )

    # Skills
    col_req, col_pref = st.columns(2)
    with col_req:
        st.markdown("**Required Skills**")
        if jd.required_skills:
            st.markdown(" · ".join([f"`{s}`" for s in jd.required_skills]))
    with col_pref:
        st.markdown("**Preferred Skills**")
        if jd.preferred_skills:
            st.markdown(" · ".join([f"`{s}`" for s in jd.preferred_skills]))

    if jd.culture_signals:
        st.markdown(
            "**Culture Signals:** "
            + " · ".join([f"`{s}`" for s in jd.culture_signals])
        )

    # ── Ranked Candidates ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🏆 Top {len(ranked)} Ranked Candidates")

    for c in ranked:
        with st.container():
            badge = _rank_badge(c.rank)
            bar = _match_bar(c.match_percentage)

            # Build HTML card
            reasons_html = "".join(
                [f'<span class="reason-tag">{r[:80]}</span>' for r in c.top_3_reasons[:3]]
            )
            flags_html = (
                f'<div class="flag-note">⚠ {c.flag_notes}</div>' if c.flag_notes else ""
            )

            st.markdown(
                f"""
                <div class="candidate-card">
                    <div style="display:flex;align-items:center;margin-bottom:8px;">
                        {badge}
                        <div style="flex:1;">
                            <div style="color:#fff;font-weight:600;font-size:1.05rem;">{c.name}</div>
                            <div style="color:#8888aa;font-size:0.8rem;">ID: {c.candidate_id} · {c.experience_years:.1f} yrs exp</div>
                        </div>
                        <div style="width:180px;">{bar}</div>
                    </div>
                    <div>{reasons_html}</div>
                    {flags_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            if show_signals:
                with st.expander(f"📊 Signal breakdown — {c.name}", expanded=False):
                    signals = c.signal_scores
                    signal_html = ""
                    for key in [
                        "semantic_similarity", "skill_match", "skill_recency",
                        "experience_fit", "career_velocity", "domain_alignment",
                        "profile_freshness", "culture_fit", "education_tier_bonus",
                    ]:
                        val = signals.get(key, 0)
                        color = SIGNAL_COLORS.get(key, "#667eea")
                        label = SIGNAL_LABELS.get(key, key)
                        signal_html += _signal_bar(label, val, color)

                    st.markdown(signal_html, unsafe_allow_html=True)

                    # Plotly radar chart
                    categories = list(SIGNAL_LABELS.values())
                    values = [signals.get(k, 0) for k in SIGNAL_LABELS.keys()]
                    values.append(values[0])  # close the polygon
                    categories.append(categories[0])

                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill="toself",
                        fillcolor="rgba(102, 126, 234, 0.15)",
                        line=dict(color="#667eea", width=2),
                        marker=dict(size=5, color="#764ba2"),
                    ))
                    fig.update_layout(
                        polar=dict(
                            bgcolor="rgba(0,0,0,0)",
                            radialaxis=dict(
                                visible=True,
                                range=[0, 1],
                                tickfont=dict(size=9, color="#666"),
                                gridcolor="rgba(255,255,255,0.08)",
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=10, color="#a0a0c0"),
                                gridcolor="rgba(255,255,255,0.08)",
                            ),
                        ),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=50, r=50, t=20, b=20),
                        height=300,
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Download button ──────────────────────────────────────────────
    st.markdown("---")
    csv_path = ensure_dir(OUTPUTS_DIR) / "ranked_output.csv"
    CandidateRanker.export_csv(ranked, csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_data = f.read()
    st.download_button(
        label="📥 Download ranked_output.csv",
        data=csv_data,
        file_name="ranked_output.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── How this works ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="how-section">
            <h3>🔬 How This Works</h3>
            <p>
            This system goes beyond keyword matching by combining <b>semantic embeddings</b>
            (sentence-transformers) with <b>9 distinct ranking signals</b> tailored for the
            Indian job market. Each candidate is scored on semantic relevance to the JD,
            hard + soft skill match using an India-specific synonym dictionary, experience
            fit, career velocity, education tier (IIT/NIT/BITS awareness), profile freshness,
            domain alignment, and cultural fit (startup vs enterprise company background).
            </p>
            <p>
            All signals are combined into a weighted composite score with configurable weights.
            Every ranking decision is <b>explainable</b> — the top 3 reasons are generated
            using template-based NLG, and flag notes highlight edge cases like over-qualification
            or stale profiles. The system processes 1,000+ profiles in seconds, ready for
            India-scale deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif not run_btn:
    # Landing state
    st.markdown("---")
    st.info(
        "👈 **Configure your inputs** in the sidebar and click **🚀 Rank Candidates** to begin.",
        icon="💡",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">9</div>'
            '<div class="metric-label">Ranking Signals</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">🧠</div>'
            '<div class="metric-label">Semantic AI</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">🇮🇳</div>'
            '<div class="metric-label">India-Native</div></div>',
            unsafe_allow_html=True,
        )

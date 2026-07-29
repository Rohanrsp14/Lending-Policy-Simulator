"""
Lending Policy Simulator — Streamlit Dashboard

The clickable, visual frontend for the pipeline built in src/. Non-technical
viewers can explore champion vs. challenger, the vintage curve, the
approval/return frontier, RAROC sensitivity, and the (explicitly
illustrative) fair-lending screen without reading any code.

Theme: forced via CSS injection targeting Streamlit's actual component
selectors (data-testid attributes), NOT reliant on .streamlit/config.toml
being detected -- that alone proved unreliable across environments (some
browsers/systems override it with an auto dark-mode preference). This is
the belt-and-suspenders version: config.toml sets the default, CSS forces
it regardless.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fair_lending import assign_geography_proxy, run_parity_screen
from src.features import parse_issue_date, prepare_features, time_based_split
from src.frontier import best_point, compute_frontier
from src.models import (
    DEFAULT_TARGET_APPROVAL_RATE,
    calibrate_fico_cutoff,
    calibrate_pd_threshold,
    champion_decision,
    champion_decision_volume_matched,
    challenger_decision,
    challenger_decision_raroc_optimized,
    challenger_decision_volume_matched,
    compute_raroc,
    evaluate_challenger,
    train_challenger,
)
from src.sensitivity import run_sensitivity, summarize_robustness
from src.vintage import compute_vintage_curve, vintage_summary

SPLIT_DATE = "2015-01-01"

# ---------------------------------------------------------------------------
# Palette -- a clean, muted "fintech product" look, not a terminal.
# ---------------------------------------------------------------------------
BG = "#FAFAFA"
CARD_BG = "#FFFFFF"
CARD_BORDER = "#E5E7EB"
TEXT_PRIMARY = "#111827"
TEXT_MUTED = "#6B7280"
CHAMPION = "#B45309"     # warm amber -- strong contrast on white
CHAMPION_BG = "#FEF3E2"
CHALLENGER = "#0F766E"   # deep teal
CHALLENGER_BG = "#E6F5F3"
FLAG = "#B91C1C"
FLAG_BG = "#FEF2F2"

st.set_page_config(page_title="Lending Policy Simulator", layout="wide", page_icon="📊")

st.markdown(f"""
<style>
    /* Force light theme regardless of system/browser dark-mode detection */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    .main, section.main {{
        background-color: {BG} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stSidebar"] {{ background-color: {CARD_BG} !important; }}
    h1, h2, h3, h4, h5, p, span, div, label {{ color: {TEXT_PRIMARY}; }}

    /* Larger base font sizes across the app -- default Streamlit sizing
       reads small, especially for body text and captions. */
    html, body, [class*="css"] {{ font-size: 17px; }}
    .stMarkdown p, .stMarkdown li {{ font-size: 17px !important; line-height: 1.6; }}
    [data-testid="stCaptionContainer"], .stCaption {{ font-size: 15px !important; }}
    [data-testid="stMetricValue"] {{ font-size: 30px !important; }}
    [data-testid="stMetricLabel"] {{ font-size: 15px !important; }}
    [data-testid="stDataFrame"] * {{ font-size: 15px !important; }}
    .stTable table {{ font-size: 16px !important; }}
    [data-baseweb="tab"] p {{ font-size: 16px !important; }}

    /* Tabs */
    [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {CARD_BORDER}; }}
    [data-baseweb="tab"] {{
        color: {TEXT_MUTED} !important; font-weight: 500; padding: 10px 16px;
    }}
    [aria-selected="true"] {{ color: {CHAMPION} !important; font-weight: 700 !important; }}

    /* Native st.metric -- force readable colors as a fallback where used */
    [data-testid="stMetricValue"] {{ color: {TEXT_PRIMARY} !important; font-weight: 700; }}
    [data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

    /* Dataframes / tables */
    [data-testid="stDataFrame"], [data-testid="stTable"] {{
        background-color: {CARD_BG} !important;
    }}

    /* Custom hero card */
    .hero-card {{
        background-color: {CARD_BG}; border: 1px solid {CARD_BORDER};
        border-radius: 12px; padding: 24px 28px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .hero-label {{
        font-size: 14px; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; padding: 4px 10px; border-radius: 6px;
        display: inline-block; margin-bottom: 12px;
    }}
    .hero-value {{ font-size: 42px; font-weight: 800; line-height: 1.1; }}
    .hero-sub {{ font-size: 15px; color: {TEXT_MUTED}; margin-top: 4px; }}
    .info-callout {{
        background-color: #EFF6FF; border-left: 4px solid #2563EB; border-radius: 6px;
        padding: 14px 18px; font-size: 16px; color: #1E3A8A; margin-bottom: 16px;
    }}
    .warn-callout {{
        background-color: {FLAG_BG}; border-left: 4px solid {FLAG}; border-radius: 6px;
        padding: 14px 18px; font-size: 16px; color: {FLAG}; margin-bottom: 16px;
    }}
    .finding-banner {{
        background-color: {CARD_BG}; border: 1px solid {CARD_BORDER};
        border-left: 4px solid {CHAMPION}; border-radius: 8px;
        padding: 20px 24px; margin: 16px 0 24px; font-size: 17px;
    }}
</style>
""", unsafe_allow_html=True)


def hero_card(label: str, value: str, sub: str, color: str, bg: str):
    st.markdown(f"""
    <div class="hero-card">
        <span class="hero-label" style="background-color:{bg}; color:{color};">{label}</span>
        <div class="hero-value" style="color:{color};">{value}</div>
        <div class="hero-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def info_callout(text: str):
    st.markdown(f'<div class="info-callout">ℹ️ {text}</div>', unsafe_allow_html=True)


def warn_callout(text: str):
    st.markdown(f'<div class="warn-callout">⚠️ {text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------
FULL_DATA_PATH = "data/processed/loans_clean.parquet"
DEMO_SAMPLE_PATH = "data/processed/loans_demo_sample.parquet"


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    """
    Loads the full dataset if present (local dev, with the real Lending
    Club file downloaded). Falls back to the committed ~100K-loan demo
    sample if not (the public deployment -- the full dataset is too large
    and its redistribution terms are unclear, so it isn't committed).
    Returns (df, is_sample: bool) so the UI can show an honest banner.
    """
    if Path(FULL_DATA_PATH).exists():
        df = pd.read_parquet(FULL_DATA_PATH)
        is_sample = False
    elif Path(DEMO_SAMPLE_PATH).exists():
        df = pd.read_parquet(DEMO_SAMPLE_PATH)
        is_sample = True
    else:
        raise FileNotFoundError(
            f"Neither {FULL_DATA_PATH} nor {DEMO_SAMPLE_PATH} found. See README.md setup."
        )
    df = parse_issue_date(df)
    df = prepare_features(df)
    return df, is_sample


@st.cache_data(show_spinner="Splitting train/test by issue date...")
def get_split(df):
    return time_based_split(df, cutoff_date=SPLIT_DATE)


@st.cache_resource(show_spinner="Training challenger (logistic regression)...")
def get_model(train_df):
    return train_challenger(train_df)


@st.cache_data(show_spinner="Computing the approval/return frontier...")
def get_frontier(_model, train_df, test_df):
    return compute_frontier(_model, train_df, test_df)


@st.cache_data(show_spinner="Computing the vintage loss curve...")
def get_vintage(df):
    return compute_vintage_curve(df), vintage_summary(df)


@st.cache_data(show_spinner="Running the RAROC sensitivity sweep (takes a bit longer)...")
def get_sensitivity(_model, train_df, test_df):
    quantiles = np.linspace(0.50, 0.99, 15)
    return run_sensitivity(_model, train_df, test_df, quantiles=quantiles)


@st.cache_data(show_spinner="Running the fair-lending parity screen...")
def get_parity(_model, df, train_df, test_df):
    df_with_proxy = assign_geography_proxy(test_df)
    champ_mask = champion_decision_volume_matched(test_df, train_df, DEFAULT_TARGET_APPROVAL_RATE)
    vm_mask = challenger_decision_volume_matched(_model, test_df, train_df, DEFAULT_TARGET_APPROVAL_RATE)
    raroc_mask = challenger_decision_raroc_optimized(_model, test_df, train_df)
    masks = {
        "Champion": champ_mask,
        "Challenger (volume-matched)": vm_mask,
        "Challenger (RAROC-optimized)": raroc_mask,
    }
    return run_parity_screen(df_with_proxy, masks)


def plotly_layout(fig, height=450, xaxis_title="", yaxis_title=""):
    fig.update_layout(
        template="plotly_white", height=height,
        paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_PRIMARY),
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
df, is_sample = load_data()
train_df, test_df = get_split(df)
model = get_model(train_df)
frontier_df = get_frontier(model, train_df, test_df)
vintage_curve, vintage_summ = get_vintage(df)

champ_best = best_point(frontier_df, "champion")
chall_best = best_point(frontier_df, "challenger")
auc = evaluate_challenger(model, test_df)["auc"]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Lending Policy Simulator")
st.caption(
    "A champion/challenger consumer-lending policy simulator, built on real, public "
    "Lending Club loan-level data. **Portfolio/analytical project — not a production "
    "underwriting system.**"
)

if is_sample:
    st.markdown(f"""
    <div class="warn-callout">
    📎 This public demo runs on a random <b>{len(df):,}-loan sample</b> of the full
    685,806-loan dataset analyzed in <b>docs/MODEL_VALIDATION.md</b>. Numbers here will
    differ slightly from the full analysis — the full-dataset results are what every
    written finding in this project is actually based on.
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div class="finding-banner">
<b style="color:{CHAMPION};">🔑 The headline finding</b><br/>
<b>Champion</b> (a simple credit-score cutoff) beats <b>Challenger</b> (a trained
statistical model, AUC {auc:.2f}) on risk-adjusted return, even though the challenger is
better at predicting who will default. A smarter model doesn't automatically mean a more
profitable decision — that's the core lesson of this whole project. Full detail in the
<b>Model Validation</b> tab.
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "1. Overview", "2. Champion vs. Challenger", "3. Vintage Curve",
    "4. Approval/Return Frontier", "5. RAROC Sensitivity",
    "6. Fair-Lending Screen", "7. Model Validation",
])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("What is this?")
    st.markdown(
        "This tool compares two different ways a lender could decide who to approve for "
        "a loan: **Champion** — today's simple rule (\"approve if credit score is above "
        "X\") — versus **Challenger** — a statistical model that weighs many factors at "
        "once. The question this whole project answers: *does the smarter model actually "
        "make the business more money, per dollar of risk taken?*"
    )

    st.subheader("The data")
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in zip(
        [c1, c2, c3, c4],
        ["Loans analyzed", "Training period", "Test period", "Credit tier"],
        [f"{len(df):,}", "before 2015", "2015 onward", "Near-prime / subprime"],
    ):
        col.markdown(f"""<div class="hero-card" style="padding:16px;">
            <div class="hero-sub" style="margin:0;">{label}</div>
            <div style="font-size:22px; font-weight:700; margin-top:4px;">{value}</div>
        </div>""", unsafe_allow_html=True)

    st.caption(
        "Loans issued 2012 or later, with a known final outcome (paid off or "
        "charged off). See the repo's `CLAUDE.md` for full data rules."
    )

    st.subheader("Each policy's best result")
    st.caption(
        "\"Best\" means: the approval threshold that gives that policy its own highest "
        "risk-adjusted return (RAROC), found by testing many thresholds."
    )
    c1, c2 = st.columns(2)
    with c1:
        hero_card(
            "🏆 Champion", f"{champ_best['champion_raroc']*100:.1f}% RAROC",
            f"Credit score ≥ {champ_best['champion_fico_cutoff']:.0f} · approves {champ_best['champion_approval_rate']*100:.1f}% of applicants",
            CHAMPION, CHAMPION_BG,
        )
    with c2:
        hero_card(
            "🤖 Challenger", f"{chall_best['challenger_raroc']*100:.1f}% RAROC",
            f"Predicted risk ≤ {chall_best['challenger_pd_threshold']*100:.1f}% · approves {chall_best['challenger_approval_rate']*100:.1f}% of applicants",
            CHALLENGER, CHALLENGER_BG,
        )

# ---------------------------------------------------------------------------
# Tab 2: Champion vs. Challenger (interactive)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Try it yourself")
    info_callout(
        "Move the slider to set how many applicants get approved. Both policies are "
        "recalculated live to hit that same approval rate — so you're always comparing "
        "them fairly, at the same volume."
    )
    target_rate = st.slider("Target approval rate", 0.50, 0.99, DEFAULT_TARGET_APPROVAL_RATE, 0.01)

    fico_cutoff = calibrate_fico_cutoff(train_df, target_rate)
    pd_threshold = calibrate_pd_threshold(model, train_df, target_rate)
    champ_mask = champion_decision(test_df, cutoff=fico_cutoff)
    chall_mask = challenger_decision(model, test_df, pd_threshold)
    champ_m = compute_raroc(test_df, champ_mask)
    chall_m = compute_raroc(test_df, chall_mask)

    c1, c2 = st.columns(2)
    with c1:
        hero_card("🏆 Champion", f"{champ_m['raroc']*100:.1f}% RAROC",
                   f"Score cutoff {fico_cutoff:.0f} · loss rate {champ_m['loss_rate']*100:.2f}%",
                   CHAMPION, CHAMPION_BG)
    with c2:
        hero_card("🤖 Challenger", f"{chall_m['raroc']*100:.1f}% RAROC",
                   f"Risk threshold {pd_threshold*100:.1f}% · loss rate {chall_m['loss_rate']*100:.2f}%",
                   CHALLENGER, CHALLENGER_BG)

    st.divider()
    st.subheader("Each policy's individually-optimal result, for comparison")
    st.table(pd.DataFrame({
        "Policy": ["🏆 Champion", "🤖 Challenger"],
        "Approval rate": [f"{champ_best['champion_approval_rate']*100:.1f}%", f"{chall_best['challenger_approval_rate']*100:.1f}%"],
        "Best possible RAROC": [f"{champ_best['champion_raroc']*100:.1f}%", f"{chall_best['challenger_raroc']*100:.1f}%"],
    }))

# ---------------------------------------------------------------------------
# Tab 3: Vintage Curve
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Are recent loans actually safer, or just too young to have failed yet?")
    info_callout(
        "Each line is one year of loans. The line shows what percentage had defaulted "
        "by a given number of months after origination. A newer line that stops early "
        "and looks low isn't necessarily safer — it just hasn't had time to mature."
    )

    fig = go.Figure()
    palette = ["#B45309", "#0F766E", "#7C3AED", "#B91C1C", "#0369A1", "#15803D", "#C2410C"]
    for i, year in enumerate(sorted(vintage_curve["issue_year"].unique())):
        yd = vintage_curve[vintage_curve["issue_year"] == year]
        fig.add_trace(go.Scatter(
            x=yd["month"], y=yd["cumulative_default_rate"] * 100,
            mode="lines", name=str(year), line=dict(color=palette[i % len(palette)], width=2.5),
        ))
    fig = plotly_layout(fig, height=450, xaxis_title="Months since the loan was issued",
                         yaxis_title="% of that year's loans that have defaulted")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Summary by year")
    st.dataframe(vintage_summ, width='stretch')

# ---------------------------------------------------------------------------
# Tab 4: Approval/Return Frontier
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("What happens at every possible approval rate?")
    info_callout(
        "Each point on a line is one possible policy setting (how strict or loose). "
        "The star marks each policy's own best spot. Champion's line has a clear peak. "
        "Challenger's line keeps getting worse the stricter it gets — its 'best' point "
        "is really just the least-bad option in the range tested, not a true peak."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["champion_approval_rate"] * 100, y=frontier_df["champion_raroc"] * 100,
        mode="lines+markers", name="Champion", line=dict(color=CHAMPION, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=frontier_df["challenger_approval_rate"] * 100, y=frontier_df["challenger_raroc"] * 100,
        mode="lines+markers", name="Challenger", line=dict(color=CHALLENGER, width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=[champ_best["champion_approval_rate"] * 100], y=[champ_best["champion_raroc"] * 100],
        mode="markers", name="Champion's best", marker=dict(size=16, color=CHAMPION, symbol="star"),
    ))
    fig.add_trace(go.Scatter(
        x=[chall_best["challenger_approval_rate"] * 100], y=[chall_best["challenger_raroc"] * 100],
        mode="markers", name="Challenger's best", marker=dict(size=16, color=CHALLENGER, symbol="star"),
    ))
    fig = plotly_layout(fig, height=500, xaxis_title="% of applicants approved",
                         yaxis_title="Risk-adjusted return (RAROC, %)")
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Tab 5: RAROC Sensitivity
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Does Champion's win depend on guessing the right cost assumptions?")
    info_callout(
        "Every RAROC number here depends on 3 assumptions this project had to estimate: "
        "loss severity, servicing cost, and cost of capital. This tests whether Champion "
        "still wins even if those guesses are wrong, across a wide realistic range."
    )
    sensitivity_df = get_sensitivity(model, train_df, test_df)
    robustness = summarize_robustness(sensitivity_df)
    st.dataframe(robustness, width='stretch')

    param_labels = {"lgd": "Loss severity (LGD)", "opex_rate": "Servicing cost", "capital_rate": "Cost of capital"}
    param = st.selectbox("See the detail behind one assumption:", list(param_labels.keys()), format_func=lambda x: param_labels[x])
    subset = sensitivity_df[sensitivity_df["swept_parameter"] == param]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["parameter_value"], y=subset["champion_best_raroc"] * 100,
                              mode="lines+markers", name="Champion", line=dict(color=CHAMPION, width=2.5)))
    fig.add_trace(go.Scatter(x=subset["parameter_value"], y=subset["challenger_best_raroc"] * 100,
                              mode="lines+markers", name="Challenger", line=dict(color=CHALLENGER, width=2.5)))
    fig = plotly_layout(fig, height=400, xaxis_title=param_labels[param], yaxis_title="Best possible RAROC (%)")
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------------
# Tab 6: Fair-Lending Screen
# ---------------------------------------------------------------------------
with tabs[5]:
    warn_callout(
        "This uses MADE-UP group data, not real demographic information — Lending Club's "
        "public data doesn't include race or ethnicity. This tab shows HOW a fair-lending "
        "check works, not a real finding about any actual group of people."
    )

    parity_df = get_parity(model, df, train_df, test_df)
    st.dataframe(parity_df, width='stretch')

    st.subheader("Result")
    for _, row in parity_df.iterrows():
        if row["flagged"]:
            st.error(f"**{row['policy']}**: ratio = {row['four_fifths_ratio']:.3f} — flagged (below 0.80)")
        else:
            st.success(f"**{row['policy']}**: ratio = {row['four_fifths_ratio']:.3f} — passes")

# ---------------------------------------------------------------------------
# Tab 7: Model Validation (embedded doc)
# ---------------------------------------------------------------------------
with tabs[6]:
    st.caption("The full independent-style review of this project — read this before quoting any number from it elsewhere.")
    try:
        validation_text = Path("docs/MODEL_VALIDATION.md").read_text()
        st.markdown(validation_text)
    except FileNotFoundError:
        st.warning("docs/MODEL_VALIDATION.md not found — run this app from the repo root.")

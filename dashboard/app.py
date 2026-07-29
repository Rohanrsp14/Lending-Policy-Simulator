"""
Lending Policy Simulator — Streamlit Dashboard

The clickable, visual frontend for the pipeline built in src/. Non-technical
viewers can explore champion vs. challenger, the vintage curve, the
approval/return frontier, RAROC sensitivity, and the (explicitly
illustrative) fair-lending screen without reading any code.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow running as `streamlit run dashboard/app.py` from the repo root
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
# Theme -- dark, serious risk-desk aesthetic, consistent with the project's
# earlier standalone demo artifact.
# ---------------------------------------------------------------------------
COLOR_BG = "#0B1220"
COLOR_PANEL = "#111B2E"
COLOR_LINE = "#22304A"
COLOR_TEXT = "#E8ECF1"
COLOR_MUTED = "#8B96A8"
COLOR_GOLD = "#C9A961"    # champion / money
COLOR_TEAL = "#4FD1C5"    # challenger / approve
COLOR_ROSE = "#E0667A"    # risk / decline / flag

st.set_page_config(page_title="Lending Policy Simulator", layout="wide", page_icon="📊")

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    [data-testid="stSidebar"] {{ background-color: {COLOR_PANEL}; }}
    .disclaimer {{
        border-left: 3px solid {COLOR_GOLD}; padding: 10px 16px; font-size: 13px;
        color: {COLOR_MUTED}; background: rgba(201,169,97,0.06); border-radius: 4px;
        margin-bottom: 12px;
    }}
    .flag-warning {{
        border-left: 3px solid {COLOR_ROSE}; padding: 12px 16px; font-size: 14px;
        color: {COLOR_ROSE}; background: rgba(224,102,122,0.08); border-radius: 4px;
        margin-bottom: 16px;
    }}
    .metric-card {{
        background: {COLOR_PANEL}; border: 1px solid {COLOR_LINE}; border-radius: 8px;
        padding: 16px; text-align: center;
    }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data / model loading -- expensive steps run once per session.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading cleaned dataset...")
def load_data():
    df = pd.read_parquet("data/processed/loans_clean.parquet")
    df = parse_issue_date(df)
    df = prepare_features(df)
    return df


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


@st.cache_data(show_spinner="Running the RAROC sensitivity sweep (this takes a bit)...")
def get_sensitivity(_model, train_df, test_df):
    # Smaller grid than the full CLI script, for a responsive dashboard --
    # still a real sweep, just fewer points for interactive use.
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


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
df = load_data()
train_df, test_df = get_split(df)
model = get_model(train_df)
frontier_df = get_frontier(model, train_df, test_df)
vintage_curve, vintage_summ = get_vintage(df)

champ_best = best_point(frontier_df, "champion")
chall_best = best_point(frontier_df, "challenger")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Lending Policy Simulator")
st.markdown(
    "A champion/challenger consumer-lending policy simulator, built on real, public "
    "**Lending Club** loan-level data. Portfolio/analytical project — not a production "
    "underwriting system."
)
st.markdown(f"""
<div class="disclaimer">
<b>Key finding:</b> the champion (a simple FICO cutoff) has a genuine RAROC-maximizing
point; the challenger (a logistic regression, AUC {evaluate_challenger(model, test_df)['auc']:.2f})
does not clearly beat it once revenue mix and realistic cost assumptions are accounted for —
real predictive power did not translate into real economic value here. See the "Model
Validation" tab for the full review.
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "Overview", "Champion vs. Challenger", "Vintage Curve", "Approval/Return Frontier",
    "RAROC Sensitivity", "Fair-Lending Screen", "Model Validation",
])

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Loans loaded", f"{len(df):,}")
    c2.metric("Train (before 2015)", f"{len(train_df):,}")
    c3.metric("Test (2015+)", f"{len(test_df):,}")
    c4.metric("Grades", "C – F")
    st.markdown(
        "Scoped to near-prime/subprime grades (C–F), matured/known-outcome loans only, "
        "issued 2012 or later. See `CLAUDE.md` in the repo for full data-provenance rules."
    )

    st.subheader("Champion's best point vs. Challenger's best point")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="metric-card">
        <h4 style="color:{COLOR_GOLD}">Champion</h4>
        <p>FICO cutoff ≥ {champ_best['champion_fico_cutoff']:.0f}</p>
        <p>Approval rate: {champ_best['champion_approval_rate']*100:.1f}%</p>
        <p style="font-size:24px; color:{COLOR_GOLD}"><b>RAROC: {champ_best['champion_raroc']*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
        <h4 style="color:{COLOR_TEAL}">Challenger</h4>
        <p>PD threshold ≤ {chall_best['challenger_pd_threshold']:.4f}</p>
        <p>Approval rate: {chall_best['challenger_approval_rate']*100:.1f}%</p>
        <p style="font-size:24px; color:{COLOR_TEAL}"><b>RAROC: {chall_best['challenger_raroc']*100:.1f}%</b></p>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: Champion vs. Challenger (interactive)
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Pick a target approval rate")
    target_rate = st.slider("Target approval rate", 0.50, 0.99, DEFAULT_TARGET_APPROVAL_RATE, 0.01)

    fico_cutoff = calibrate_fico_cutoff(train_df, target_rate)
    pd_threshold = calibrate_pd_threshold(model, train_df, target_rate)
    champ_mask = champion_decision(test_df, cutoff=fico_cutoff)
    chall_mask = challenger_decision(model, test_df, pd_threshold)
    champ_m = compute_raroc(test_df, champ_mask)
    chall_m = compute_raroc(test_df, chall_mask)

    c1, c2, c3 = st.columns(3)
    c1.metric("Champion FICO cutoff", f"{fico_cutoff:.0f}")
    c2.metric("Champion RAROC", f"{champ_m['raroc']*100:.1f}%")
    c3.metric("Champion loss rate", f"{champ_m['loss_rate']*100:.2f}%")

    c1, c2, c3 = st.columns(3)
    c1.metric("Challenger PD threshold", f"{pd_threshold:.4f}")
    c2.metric("Challenger RAROC", f"{chall_m['raroc']*100:.1f}%")
    c3.metric("Challenger loss rate", f"{chall_m['loss_rate']*100:.2f}%")

    st.subheader("At their own individually-optimal points (RAROC-maximized)")
    st.table(pd.DataFrame({
        "Policy": ["Champion", "Challenger"],
        "Approval rate": [f"{champ_best['champion_approval_rate']*100:.1f}%", f"{chall_best['challenger_approval_rate']*100:.1f}%"],
        "RAROC": [f"{champ_best['champion_raroc']*100:.1f}%", f"{chall_best['challenger_raroc']*100:.1f}%"],
    }))

# ---------------------------------------------------------------------------
# Tab 3: Vintage Curve
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Vintage loss emergence, by issue year")
    st.markdown(
        "Cumulative default rate by months-on-book, per origination cohort. Recent "
        "cohorts (2017-2018) show artificially low default rates purely because they "
        "haven't had time to mature — not because they're genuinely safer."
    )

    fig = go.Figure()
    for year in sorted(vintage_curve["issue_year"].unique()):
        yd = vintage_curve[vintage_curve["issue_year"] == year]
        fig.add_trace(go.Scatter(
            x=yd["month"], y=yd["cumulative_default_rate"] * 100,
            mode="lines", name=str(year),
        ))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor=COLOR_PANEL, paper_bgcolor=COLOR_BG,
        xaxis_title="Months on book", yaxis_title="Cumulative default rate (%)",
        legend_title="Issue year", height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(vintage_summ, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Approval/Return Frontier
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Approval/return frontier — both policies swept symmetrically")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["champion_approval_rate"] * 100, y=frontier_df["champion_raroc"] * 100,
        mode="lines+markers", name="Champion", line=dict(color=COLOR_GOLD),
    ))
    fig.add_trace(go.Scatter(
        x=frontier_df["challenger_approval_rate"] * 100, y=frontier_df["challenger_raroc"] * 100,
        mode="lines+markers", name="Challenger", line=dict(color=COLOR_TEAL),
    ))
    fig.add_trace(go.Scatter(
        x=[champ_best["champion_approval_rate"] * 100], y=[champ_best["champion_raroc"] * 100],
        mode="markers", name="Champion best", marker=dict(size=14, color=COLOR_GOLD, symbol="star"),
    ))
    fig.add_trace(go.Scatter(
        x=[chall_best["challenger_approval_rate"] * 100], y=[chall_best["challenger_raroc"] * 100],
        mode="markers", name="Challenger best", marker=dict(size=14, color=COLOR_TEAL, symbol="star"),
    ))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor=COLOR_PANEL, paper_bgcolor=COLOR_BG,
        xaxis_title="Approval rate (%)", yaxis_title="RAROC (%)", height=500,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "Champion has a genuine interior peak. Challenger's RAROC degrades as it gets "
        "more selective across the whole tested range — its best point sits at the edge, "
        "not a real peak. See the Model Validation tab for the full analysis."
    )

# ---------------------------------------------------------------------------
# Tab 5: RAROC Sensitivity
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Does champion's advantage hold across a plausible range of assumptions?")
    st.markdown(
        "Sweeps LGD, opex rate, and capital rate one at a time (holding the others at "
        "base case), re-optimizing EACH policy's own best point under every tested value."
    )
    sensitivity_df = get_sensitivity(model, train_df, test_df)
    robustness = summarize_robustness(sensitivity_df)
    st.dataframe(robustness, use_container_width=True)

    param = st.selectbox("Show detail for:", ["lgd", "opex_rate", "capital_rate"])
    subset = sensitivity_df[sensitivity_df["swept_parameter"] == param]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subset["parameter_value"], y=subset["champion_best_raroc"] * 100,
                              mode="lines+markers", name="Champion best RAROC", line=dict(color=COLOR_GOLD)))
    fig.add_trace(go.Scatter(x=subset["parameter_value"], y=subset["challenger_best_raroc"] * 100,
                              mode="lines+markers", name="Challenger best RAROC", line=dict(color=COLOR_TEAL)))
    fig.update_layout(
        template="plotly_dark", plot_bgcolor=COLOR_PANEL, paper_bgcolor=COLOR_BG,
        xaxis_title=param, yaxis_title="Best RAROC (%)", height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 6: Fair-Lending Screen
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown(f"""
    <div class="flag-warning">
    <b>ILLUSTRATIVE ONLY, NOT EVIDENTIARY.</b> This uses a FABRICATED state-level
    probability table, not real Census demographic data. It demonstrates the four-fifths
    parity MECHANISM, not a real disparate-impact finding on this or any real population.
    </div>
    """, unsafe_allow_html=True)

    parity_df = get_parity(model, df, train_df, test_df)
    st.dataframe(parity_df, use_container_width=True)

    for _, row in parity_df.iterrows():
        if row["flagged"]:
            st.markdown(f"🔴 **{row['policy']}**: ratio = {row['four_fifths_ratio']:.3f} — flagged")
        else:
            st.markdown(f"🟢 **{row['policy']}**: ratio = {row['four_fifths_ratio']:.3f} — passes")

# ---------------------------------------------------------------------------
# Tab 7: Model Validation (embedded doc)
# ---------------------------------------------------------------------------
with tabs[6]:
    try:
        validation_text = Path("docs/MODEL_VALIDATION.md").read_text()
        st.markdown(validation_text)
    except FileNotFoundError:
        st.warning("docs/MODEL_VALIDATION.md not found — run this app from the repo root.")

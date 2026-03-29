"""
=============================================================
  Securitization Investment Decision Model
  Streamlit App — Full Deployment
=============================================================
  Pages:
    1. Home
    2. Single Loan Scorer
    3. Pool / Deal Analyser
    4. Deal Comparison
=============================================================
"""

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Securitization Investment Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve

# ─────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
C = {
    "invest":  "#1D9E75",
    "avoid":   "#E24B4A",
    "senior":  "#185FA5",
    "mezz":    "#BA7517",
    "equity":  "#993C1D",
    "neutral": "#888780",
    "prime":   "#1D9E75",
    "nearprime":"#BA7517",
    "subprime":"#D85A30",
    "highrisk":"#E24B4A",
}

TIER_COLORS = {
    "Prime":      C["prime"],
    "Near-prime": C["nearprime"],
    "Subprime":   C["subprime"],
    "High-risk":  C["highrisk"],
}

# ─────────────────────────────────────────────────────────────────
# MODEL TRAINING (cached so it only runs once)
# ─────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training model on synthetic data…")
def load_model():
    rng = np.random.default_rng(42)
    n   = 15000

    credit_score   = rng.integers(300, 851, n)
    annual_income  = np.clip(rng.lognormal(10.9, 0.55, n), 18000, 600000)
    age            = rng.integers(21, 72, n)
    employ_years   = np.clip(rng.exponential(7, n), 0, 40)
    num_dependants = rng.integers(0, 5, n).astype(float)
    property_value = np.clip(rng.lognormal(12.4, 0.45, n), 75000, 2_500_000)
    loan_amount    = property_value * rng.uniform(0.35, 0.98, n)
    interest_rate  = rng.uniform(2.0, 12.0, n)
    loan_term_yrs  = rng.choice([10,15,20,25,30], n, p=[0.05,0.10,0.15,0.20,0.50]).astype(float)
    fixed_rate     = rng.choice([1,0], n, p=[0.68,0.32]).astype(float)
    primary_res    = rng.choice([1,0], n, p=[0.72,0.28]).astype(float)
    own_property   = rng.choice([1,0], n, p=[0.60,0.40]).astype(float)
    prior_defaults = rng.choice([0,1,2], n, p=[0.78,0.17,0.05]).astype(float)
    existing_debt  = np.clip(rng.lognormal(6.4, 0.7, n), 0, 10000)
    location_risk  = rng.uniform(0, 1, n)
    macro_stress   = rng.uniform(0, 1, n)

    mr  = interest_rate / 100 / 12
    np_ = loan_term_yrs * 12
    mp  = loan_amount * (mr*(1+mr)**np_) / ((1+mr)**np_ - 1)
    dti = (mp + existing_debt) / (annual_income / 12)
    ltv = loan_amount / property_value

    logit = (
        -7.5
        + (700 - credit_score) * 0.014
        + (dti  - 0.36) * 7.5
        + (ltv  - 0.75) * 4.5
        + interest_rate * 0.20
        - employ_years  * 0.07
        - own_property  * 0.40
        + (1 - fixed_rate) * 0.70
        + prior_defaults * 1.80
        + location_risk * 1.20
        + macro_stress  * 0.80
        - np.log(np.clip(annual_income/50000, 0.1, 10)) * 0.45
    )
    prob    = 1 / (1 + np.exp(-logit))
    defaulted = rng.binomial(1, prob.clip(0.01, 0.99))

    df = pd.DataFrame({
        "credit_score":    credit_score.astype(float),
        "annual_income":   annual_income,
        "age":             age.astype(float),
        "employ_years":    employ_years,
        "num_dependants":  num_dependants,
        "interest_rate":   interest_rate,
        "loan_term_yrs":   loan_term_yrs,
        "fixed_rate":      fixed_rate,
        "primary_res":     primary_res,
        "own_property":    own_property,
        "prior_defaults":  prior_defaults,
        "existing_debt":   existing_debt,
        "location_risk":   location_risk,
        "macro_stress":    macro_stress,
        "dti_ratio":       dti,
        "ltv_ratio":       ltv,
        "income_per_dep":  annual_income / (num_dependants + 1),
        "dti_x_ltv":       dti * ltv,
        "credit_income_rat": credit_score / (annual_income / 10000),
        "rate_spread":     interest_rate - 4.5,
        "age_employ_ratio":employ_years / np.clip(age, 1, 100),
        "high_dti_flag":   (dti > 0.43).astype(float),
        "high_ltv_flag":   (ltv > 0.90).astype(float),
        "subprime_flag":   (credit_score < 620).astype(float),
        "defaulted":       defaulted.astype(float),
    })

    FEATURES = [c for c in df.columns if c != "defaulted"]
    X = df[FEATURES]; y = df["defaulted"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    Xtr_p   = scaler.fit_transform(imputer.fit_transform(X_tr))
    Xte_p   = scaler.transform(imputer.transform(X_te))

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.08, subsample=0.8,
        random_state=42
    )
    model.fit(Xtr_p, y_tr)

    y_prob = model.predict_proba(Xte_p)[:, 1]
    auc    = roc_auc_score(y_te, y_prob)

    return model, imputer, scaler, FEATURES, auc


# ─────────────────────────────────────────────────────────────────
# SCORING HELPERS
# ─────────────────────────────────────────────────────────────────

def score_loan_dict(loan: dict, model, imputer, scaler, features) -> dict:
    row = pd.DataFrame([{f: loan.get(f, np.nan) for f in features}])
    inp = scaler.transform(imputer.transform(row))
    pd_p = model.predict_proba(inp)[0][1]
    qs   = round((1 - pd_p) * 100, 1)
    tier = ("Prime"      if qs >= 80 else
            "Near-prime" if qs >= 65 else
            "Subprime"   if qs >= 45 else "High-risk")
    return {"quality_score": qs, "pd_pct": round(pd_p*100,2), "tier": tier}


def score_pool(pool_df: pd.DataFrame, model, imputer, scaler, features):
    cols = [f for f in features if f in pool_df.columns]
    X    = pool_df[cols].reindex(columns=features)
    Xp   = scaler.transform(imputer.transform(X))
    pds  = model.predict_proba(Xp)[:, 1]
    qs   = (1 - pds) * 100
    tiers= pd.cut(qs, [0,45,65,80,100],
                  labels=["High-risk","Subprime","Near-prime","Prime"])
    return pds, qs, tiers


def make_decision(avg_pd, avg_qs, tier_dist, pool_df):
    subprime_sh = (tier_dist.get("Subprime",0) + tier_dist.get("High-risk",0)) / 100
    high_risk_sh= tier_dist.get("High-risk",0) / 100
    flags = []
    if avg_pd        > 0.12:  flags.append(f"High avg default probability ({avg_pd:.1%})")
    if subprime_sh   > 0.25:  flags.append(f"Subprime concentration exceeds 25% ({subprime_sh:.1%})")
    if high_risk_sh  > 0.05:  flags.append(f"High-risk loans exceed 5% ({high_risk_sh:.1%})")
    if "dti_ratio" in pool_df and pool_df["dti_ratio"].mean() > 0.43:
        flags.append("Pool avg DTI exceeds regulatory threshold (43%)")
    if "ltv_ratio" in pool_df and pool_df["ltv_ratio"].mean() > 0.85:
        flags.append(f"Pool avg LTV is elevated ({pool_df['ltv_ratio'].mean():.0%})")

    hard_block = avg_pd > 0.12 or subprime_sh > 0.25
    pool_score = avg_qs / 100
    decision   = "AVOID" if hard_block else ("INVEST" if pool_score >= 0.55 else "AVOID")
    confidence = round(pool_score * 100, 1) if decision == "INVEST" else 0.0

    prime_pct     = tier_dist.get("Prime", 0)
    nearprime_pct = tier_dist.get("Near-prime", 0)
    senior_sz  = min(85, prime_pct*0.90 + nearprime_pct*0.55)
    mezz_sz    = max(5, min(15, 100 - senior_sz - 5))
    equity_sz  = max(5, 100 - senior_sz - mezz_sz)

    if decision == "INVEST":
        if confidence >= 72:
            tranche = "Senior tranche (AAA)"
            why     = "Strong pool quality supports senior positioning"
        elif confidence >= 60:
            tranche = "Mezzanine tranche (A–BBB)"
            why     = "Good risk-return balance; credit enhancement recommended"
        else:
            tranche = "Equity / Junior tranche"
            why     = "Higher yield compensates elevated pool risk"
    else:
        tranche = "None — do not invest"
        why     = "Pool risk exceeds acceptable thresholds"

    return {
        "decision":    decision,
        "confidence":  confidence,
        "flags":       flags,
        "tranche_rec": tranche,
        "tranche_why": why,
        "senior_sz":   round(senior_sz,1),
        "mezz_sz":     round(mezz_sz,1),
        "equity_sz":   round(equity_sz,1),
    }


def build_loan_from_inputs(inputs: dict) -> dict:
    """Compute derived features from raw user inputs."""
    mr  = inputs["interest_rate"] / 100 / 12
    n_  = inputs["loan_term_yrs"] * 12
    mp  = inputs["loan_amount"] * (mr*(1+mr)**n_) / ((1+mr)**n_ - 1)
    dti = (mp + inputs["existing_debt"]) / (inputs["annual_income"] / 12)
    ltv = inputs["loan_amount"] / inputs["property_value"]

    d = dict(inputs)
    d["monthly_payment"]   = mp
    d["dti_ratio"]         = dti
    d["ltv_ratio"]         = ltv
    d["income_per_dep"]    = inputs["annual_income"] / (inputs["num_dependants"] + 1)
    d["dti_x_ltv"]         = dti * ltv
    d["credit_income_rat"] = inputs["credit_score"] / (inputs["annual_income"] / 10000)
    d["rate_spread"]       = inputs["interest_rate"] - 4.5
    d["age_employ_ratio"]  = inputs["employ_years"] / max(inputs["age"], 1)
    d["high_dti_flag"]     = float(dti > 0.43)
    d["high_ltv_flag"]     = float(ltv > 0.90)
    d["subprime_flag"]     = float(inputs["credit_score"] < 620)
    return d


# ─────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────

def score_donut(score, tier, pd_pct, label=""):
    color = TIER_COLORS.get(tier, C["neutral"])
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.pie([score, 100-score], startangle=90,
           colors=[color, "#EEEEEE"],
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0,  0.12, f"{score:.0f}", ha="center", va="center",
            fontsize=32, fontweight="bold", color=color)
    ax.text(0, -0.28, tier, ha="center", va="center",
            fontsize=11, color=color, fontweight="500")
    ax.text(0, -0.52, f"PD: {pd_pct:.1f}%",
            ha="center", va="center", fontsize=9, color="gray")
    if label:
        ax.text(0, -0.75, label, ha="center", va="center",
                fontsize=8, color="gray")
    ax.axis("off")
    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0.2)
    return fig


def factor_bars(factors: dict):
    names = list(factors.keys())
    vals  = list(factors.values())
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bc = [C["invest"] if v>=70 else (C["mezz"] if v>=40 else C["avoid"]) for v in vals]
    bars = ax.barh(names, vals, color=bc, edgecolor="none", height=0.55)
    ax.set_xlim(0, 115)
    ax.axvline(70, color="gray", ls="--", lw=1, alpha=0.5, label="Good (≥70)")
    for bar, val in zip(bars, vals):
        ax.text(val+1.5, bar.get_y()+bar.get_height()/2,
                str(val), va="center", fontsize=9)
    ax.set_xlabel("Sub-score (0–100)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def pool_tier_pie(tier_dist: dict):
    tiers  = ["Prime","Near-prime","Subprime","High-risk"]
    vals   = [tier_dist.get(t,0) for t in tiers]
    clrs   = [TIER_COLORS[t] for t in tiers]
    nonzero= [(t,v,c) for t,v,c in zip(tiers,vals,clrs) if v>0]
    if not nonzero: return None
    fig, ax = plt.subplots(figsize=(4,4))
    ax.pie([v for _,v,_ in nonzero],
           labels=[t for t,_,_ in nonzero],
           colors=[c for _,_,c in nonzero],
           autopct="%1.1f%%", startangle=90,
           textprops={"fontsize":9},
           wedgeprops=dict(edgecolor="white", linewidth=1.5))
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def tranche_bar(senior, mezz, equity, highlight=None):
    labels = ["Senior\n(AAA)", "Mezzanine\n(A-BBB)", "Equity\n(Junior)"]
    sizes  = [senior, mezz, equity]
    clrs   = [C["senior"], C["mezz"], C["equity"]]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars = ax.bar(labels, sizes, color=clrs, edgecolor="none", width=0.5)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.8,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="500")
    if highlight:
        for i, (bar, lbl) in enumerate(zip(bars, ["Senior","Mezzanine","Equity"])):
            if lbl.lower() in highlight.lower():
                bar.set_edgecolor("black"); bar.set_linewidth(2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Pool", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


def score_hist(q_scores, color, avg):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(q_scores, bins=30, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(avg, color="black", ls="--", lw=1.5, label=f"Mean: {avg:.1f}")
    ax.axvline(55, color="gray", ls=":", lw=1, label="Invest threshold")
    ax.set_xlabel("Quality Score (0–100)", fontsize=9)
    ax.set_ylabel("Loans", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.patch.set_alpha(0)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  .invest-badge {
    background:#E1F5EE; color:#085041;
    padding:12px 24px; border-radius:10px;
    font-size:22px; font-weight:700;
    border-left:5px solid #1D9E75;
    display:inline-block; margin:8px 0;
  }
  .avoid-badge {
    background:#FCEBEB; color:#791F1F;
    padding:12px 24px; border-radius:10px;
    font-size:22px; font-weight:700;
    border-left:5px solid #E24B4A;
    display:inline-block; margin:8px 0;
  }
  .metric-box {
    background:#F8F8F6; border-radius:10px;
    padding:14px 18px; text-align:center;
    border:1px solid #E0DED8;
  }
  .metric-val { font-size:26px; font-weight:600; margin:4px 0 0; }
  .metric-lbl { font-size:11px; color:#888; }
  .flag-box {
    background:#FFF8EE; border-left:4px solid #EF9F27;
    padding:8px 14px; border-radius:4px;
    font-size:13px; margin:4px 0;
  }
  .ok-box {
    background:#F0FBF6; border-left:4px solid #1D9E75;
    padding:8px 14px; border-radius:4px; font-size:13px;
  }
  .section-title {
    font-size:16px; font-weight:600;
    border-bottom:2px solid #E0DED8;
    padding-bottom:6px; margin:18px 0 12px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
st.sidebar.title("Securitization\nInvestment Advisor")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Single Loan Scorer",
     "📊 Pool / Deal Analyser", "⚖️ Deal Comparison"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info**")

model, imputer, scaler, FEATURES, auc = load_model()

st.sidebar.metric("ROC-AUC", f"{auc:.3f}")
st.sidebar.metric("Training loans", "15,000")
st.sidebar.metric("Algorithm", "Gradient Boosting")
st.sidebar.markdown("---")
st.sidebar.caption("MSc FinTech · University of Birmingham\nRisk Management Module")



# ─────────────────────────────────────────────────────────────────
# Sub-score helpers (used by Single Loan page)
# ─────────────────────────────────────────────────────────────────
def _cs(c):
    if c<=580: return 5
    if c<=620: return 25
    if c<=660: return 45
    if c<=700: return 65
    if c<=740: return 80
    if c<=780: return 92
    return 100

def _dti(d):
    if d>=0.60: return 5
    if d>=0.50: return 20
    if d>=0.43: return 40
    if d>=0.36: return 65
    if d>=0.28: return 85
    return 100

def _ltv(l):
    if pd.isna(l) or l>=0.97: return 5
    if l>=0.90: return 25
    if l>=0.80: return 55
    if l>=0.70: return 78
    if l>=0.60: return 92
    return 100

def _inc(i):
    if i<25000:  return 15
    if i<40000:  return 40
    if i<60000:  return 65
    if i<90000:  return 85
    return 100

# ═════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.title("🏦 Securitization Investment Decision Model")
    st.markdown(
        "This tool helps investors decide **whether to invest** in a securitization deal "
        "by scoring the underlying loan pool using a trained Machine Learning model."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-box"><div class="metric-lbl">Model accuracy</div>'
                    f'<div class="metric-val">{auc:.1%}</div><div class="metric-lbl">ROC-AUC</div></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><div class="metric-lbl">Algorithm</div>'
                    '<div class="metric-val" style="font-size:17px">Gradient<br>Boosting</div></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><div class="metric-lbl">Features used</div>'
                    f'<div class="metric-val">{len(FEATURES)}</div></div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-box"><div class="metric-lbl">Training data</div>'
                    '<div class="metric-val">15K</div><div class="metric-lbl">synthetic loans</div></div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### How it works")
        st.markdown("""
1. **Input** — describe a loan or upload a pool CSV
2. **Score** — the ML model predicts the probability of default for each loan
3. **Aggregate** — loans are grouped into risk tiers (Prime → High-risk)
4. **Decide** — the engine outputs **INVEST** or **AVOID** with a confidence score
5. **Recommend** — tells you which tranche to buy (Senior / Mezzanine / Equity)
        """)

    with col_r:
        st.markdown("### Risk tiers")
        tier_data = {
            "Tier": ["🟢 Prime", "🟡 Near-prime", "🟠 Subprime", "🔴 High-risk"],
            "Score": ["80–100", "65–79", "45–64", "0–44"],
            "Tranche fit": ["Senior AAA", "Mezzanine", "Junior only", "Reject"],
        }
        st.dataframe(pd.DataFrame(tier_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown("### Navigate using the sidebar ←")
    st.info("Start with **Single Loan Scorer** to understand how one loan is assessed, "
            "then move to **Pool / Deal Analyser** for a full securitization deal.")


# ═════════════════════════════════════════════════════════════════
# PAGE 2: SINGLE LOAN SCORER
# ═════════════════════════════════════════════════════════════════

elif page == "🔍 Single Loan Scorer":
    st.title("🔍 Single Loan Scorer")
    st.markdown("Enter a borrower's details to get a quality score and investor recommendation.")

    with st.form("loan_form"):
        st.markdown('<div class="section-title">Borrower profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        credit_score  = c1.slider("Credit score", 300, 850, 720, 5)
        annual_income = c2.number_input("Annual income ($)", 10000, 500000, 85000, 1000)
        age           = c3.slider("Borrower age", 21, 75, 40)

        c4, c5, c6 = st.columns(3)
        employ_years   = c4.slider("Years employed", 0, 40, 8)
        num_dependants = c5.slider("Number of dependants", 0, 6, 2)
        prior_defaults = c6.selectbox("Prior defaults", [0, 1, 2], index=0)

        st.markdown('<div class="section-title">Loan details</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        loan_amount    = d1.number_input("Loan amount ($)", 20000, 2000000, 280000, 5000)
        property_value = d2.number_input("Property value ($)", 50000, 3000000, 370000, 5000)
        interest_rate  = d3.slider("Interest rate (%)", 1.0, 15.0, 5.5, 0.1)

        d4, d5, d6 = st.columns(3)
        loan_term_yrs  = d4.selectbox("Loan term (years)", [10,15,20,25,30], index=4)
        existing_debt  = d5.number_input("Existing monthly debt ($)", 0, 10000, 600, 50)
        fixed_rate     = d6.selectbox("Mortgage type", ["Fixed rate", "Adjustable (ARM)"])

        e1, e2, e3 = st.columns(3)
        primary_res    = e1.selectbox("Property use", ["Primary residence", "Investment/other"])
        own_property   = e2.selectbox("Owns property?", ["Yes", "No"])
        location_risk  = e3.slider("Market risk (0=safe, 1=volatile)", 0.0, 1.0, 0.3, 0.05)

        macro_stress = st.slider("Macro stress (0=boom, 1=recession)", 0.0, 1.0, 0.35, 0.05)

        submitted = st.form_submit_button("Score this loan →", use_container_width=True)

    if submitted:
        inputs = {
            "credit_score":    float(credit_score),
            "annual_income":   float(annual_income),
            "age":             float(age),
            "employ_years":    float(employ_years),
            "num_dependants":  float(num_dependants),
            "loan_amount":     float(loan_amount),
            "property_value":  float(property_value),
            "interest_rate":   float(interest_rate),
            "loan_term_yrs":   float(loan_term_yrs),
            "fixed_rate":      1.0 if fixed_rate == "Fixed rate" else 0.0,
            "primary_res":     1.0 if primary_res == "Primary residence" else 0.0,
            "own_property":    1.0 if own_property == "Yes" else 0.0,
            "prior_defaults":  float(prior_defaults),
            "existing_debt":   float(existing_debt),
            "location_risk":   float(location_risk),
            "macro_stress":    float(macro_stress),
        }
        loan = build_loan_from_inputs(inputs)
        res  = score_loan_dict(loan, model, imputer, scaler, FEATURES)

        st.markdown("---")
        col_score, col_factors = st.columns([1, 1.8])

        with col_score:
            st.markdown("#### Overall score")
            st.pyplot(score_donut(res["quality_score"], res["tier"], res["pd_pct"]),
                      use_container_width=True)

            st.markdown("#### Key metrics")
            m1, m2 = st.columns(2)
            m1.metric("LTV", f"{loan['ltv_ratio']*100:.1f}%",
                      delta="⚠️ High" if loan["ltv_ratio"]>0.9 else "✓ OK",
                      delta_color="inverse")
            m2.metric("DTI", f"{loan['dti_ratio']*100:.1f}%",
                      delta="⚠️ High" if loan["dti_ratio"]>0.43 else "✓ OK",
                      delta_color="inverse")
            m1.metric("Monthly payment", f"${loan['monthly_payment']:,.0f}")
            m2.metric("Default prob.", f"{res['pd_pct']:.1f}%")

        with col_factors:
            st.markdown("#### Factor breakdown")
            factors = {
                "Credit score":         _cs(credit_score),
                "Debt-to-income":       _dti(loan["dti_ratio"]),
                "Loan-to-value":        _ltv(loan["ltv_ratio"]),
                "Income level":         _inc(annual_income),
                "Employment stability": min(100, int(employ_years * 5)),
                "Fixed rate mortgage":  100 if fixed_rate=="Fixed rate" else 35,
            }
            st.pyplot(factor_bars(factors), use_container_width=True)

        # Verdict
        st.markdown("---")
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("#### Investment verdict")
            if res["quality_score"] >= 65:
                st.markdown('<div class="invest-badge">✅ SUITABLE FOR INVESTMENT</div>',
                            unsafe_allow_html=True)
                st.success(f"This is a **{res['tier']}** loan with a {res['pd_pct']:.1f}% "
                           f"probability of default. Suitable for securitization.")
            elif res["quality_score"] >= 45:
                st.markdown('<div class="avoid-badge">⚠️ CAUTION</div>', unsafe_allow_html=True)
                st.warning(f"This is a **{res['tier']}** loan. High-risk features detected. "
                           f"Only suitable for junior tranches with credit enhancement.")
            else:
                st.markdown('<div class="avoid-badge">❌ DO NOT INVEST</div>',
                            unsafe_allow_html=True)
                st.error(f"This is a **{res['tier']}** loan with {res['pd_pct']:.1f}% default "
                         f"probability. This profile contributed to the 2008 crisis at scale.")

        with v2:
            st.markdown("#### Tranche recommendation")
            tier_tranche = {
                "Prime":      ("Senior tranche (AAA)", C["senior"]),
                "Near-prime": ("Mezzanine tranche (A–BBB)", C["mezz"]),
                "Subprime":   ("Equity / Junior tranche", C["equity"]),
                "High-risk":  ("Do not securitise", C["avoid"]),
            }
            tranche, t_color = tier_tranche[res["tier"]]
            st.markdown(f"**Recommended position:** {tranche}")
            st.pyplot(tranche_bar(
                80 if res["tier"]=="Prime" else 40,
                12 if res["tier"] in ["Prime","Near-prime"] else 20,
                8  if res["tier"]=="Prime" else 40,
                highlight=tranche
            ), use_container_width=True)

        # CSV export
        export_df = pd.DataFrame([{
            "Credit score": credit_score,
            "Annual income": annual_income,
            "LTV (%)": round(loan["ltv_ratio"]*100,1),
            "DTI (%)": round(loan["dti_ratio"]*100,1),
            "Quality score": res["quality_score"],
            "Default prob (%)": res["pd_pct"],
            "Tier": res["tier"],
            "Recommendation": tranche,
        }])
        st.download_button(
            "📥 Download result as CSV",
            export_df.to_csv(index=False),
            file_name="loan_score.csv",
            mime="text/csv",
        )



# ═════════════════════════════════════════════════════════════════
# PAGE 3: POOL / DEAL ANALYSER
# ═════════════════════════════════════════════════════════════════

elif page == "📊 Pool / Deal Analyser":
    st.title("📊 Pool / Deal Analyser")
    st.markdown("Analyse a full securitization deal pool and get an **INVEST / AVOID** decision.")

    tab1, tab2 = st.tabs(["🎛️ Manual deal builder", "📁 Upload CSV"])

    # ── Tab 1: Manual ─────────────────────────────────────────────
    with tab1:
        st.markdown("Describe your typical borrower and loan characteristics. "
                    "The model will simulate a pool and score it.")

        with st.form("pool_form"):
            st.markdown('<div class="section-title">Pool overview</div>', unsafe_allow_html=True)
            p1, p2 = st.columns(2)
            deal_name = p1.text_input("Deal name", "My Securitization Deal")
            n_loans   = p2.slider("Number of loans in pool", 50, 2000, 500, 50)

            st.markdown('<div class="section-title">Average borrower profile</div>',
                        unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            avg_credit  = r1.slider("Avg credit score", 300, 850, 700, 5)
            avg_income  = r2.number_input("Avg annual income ($)", 20000, 300000, 75000, 1000)
            avg_age     = r3.slider("Avg borrower age", 25, 65, 42)

            r4, r5, r6 = st.columns(3)
            avg_employ  = r4.slider("Avg employment years", 0, 30, 9)
            avg_deps    = r5.slider("Avg dependants", 0, 5, 2)
            pct_prior   = r6.slider("% with prior defaults", 0, 40, 8)

            st.markdown('<div class="section-title">Average loan characteristics</div>',
                        unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            avg_loan    = s1.number_input("Avg loan amount ($)", 50000, 1000000, 280000, 5000)
            avg_propval = s2.number_input("Avg property value ($)", 80000, 2000000, 370000, 5000)
            avg_rate    = s3.slider("Avg interest rate (%)", 1.0, 12.0, 5.2, 0.1)

            s4, s5, s6 = st.columns(3)
            avg_term    = s4.selectbox("Loan term (years)", [15,20,25,30], index=3)
            pct_fixed   = s5.slider("% fixed rate mortgages", 0, 100, 70)
            pct_primary = s6.slider("% primary residences", 0, 100, 75)

            st.markdown('<div class="section-title">Market conditions</div>',
                        unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            avg_debt    = m1.number_input("Avg existing monthly debt ($)", 0, 5000, 600, 50)
            loc_risk    = m2.slider("Location market risk (0=safe, 1=volatile)", 0.0, 1.0, 0.30, 0.05)
            macro       = m3.slider("Macro stress (0=boom, 1=recession)", 0.0, 1.0, 0.35, 0.05)

            run_btn = st.form_submit_button("Analyse deal →", use_container_width=True)

        if run_btn:
            with st.spinner("Scoring pool loans…"):
                rng2 = np.random.default_rng(77)
                N    = n_loans

                pool = pd.DataFrame({
                    "credit_score":   np.clip(rng2.normal(avg_credit, 45, N), 300, 850),
                    "annual_income":  np.clip(rng2.normal(avg_income, 15000, N), 18000, 500000),
                    "age":            np.clip(rng2.normal(avg_age, 7, N), 21, 72),
                    "employ_years":   np.clip(rng2.normal(avg_employ, 3, N), 0, 40),
                    "num_dependants": rng2.integers(0, 5, N).astype(float),
                    "loan_amount":    np.clip(rng2.normal(avg_loan, 40000, N), 30000, 1500000),
                    "property_value": np.clip(rng2.normal(avg_propval, 50000, N), 60000, 2500000),
                    "interest_rate":  np.clip(rng2.normal(avg_rate, 0.8, N), 2.0, 12.0),
                    "loan_term_yrs":  np.full(N, float(avg_term)),
                    "fixed_rate":     rng2.choice([1,0], N, p=[pct_fixed/100, 1-pct_fixed/100]).astype(float),
                    "primary_res":    rng2.choice([1,0], N, p=[pct_primary/100, 1-pct_primary/100]).astype(float),
                    "own_property":   rng2.choice([1,0], N, p=[0.6,0.4]).astype(float),
                    "prior_defaults": rng2.choice([0,1,2], N,
                                                  p=[max(0,(100-pct_prior)/100),
                                                     min(pct_prior/100*0.8, 1.0),
                                                     min(pct_prior/100*0.2, 1.0)]).astype(float),
                    "existing_debt":  np.clip(rng2.normal(avg_debt, 200, N), 0, 5000),
                    "location_risk":  np.clip(rng2.normal(loc_risk, 0.12, N), 0, 1),
                    "macro_stress":   np.full(N, macro),
                })
                mr2 = pool["interest_rate"] / 100 / 12
                np2 = pool["loan_term_yrs"] * 12
                mp2 = pool["loan_amount"] * (mr2*(1+mr2)**np2) / ((1+mr2)**np2 - 1)
                pool["monthly_payment"]   = mp2
                pool["dti_ratio"]         = (mp2 + pool["existing_debt"]) / (pool["annual_income"]/12)
                pool["ltv_ratio"]         = pool["loan_amount"] / pool["property_value"]
                pool["income_per_dep"]    = pool["annual_income"] / (pool["num_dependants"]+1)
                pool["dti_x_ltv"]         = pool["dti_ratio"] * pool["ltv_ratio"]
                pool["credit_income_rat"] = pool["credit_score"] / (pool["annual_income"]/10000)
                pool["rate_spread"]       = pool["interest_rate"] - 4.5
                pool["age_employ_ratio"]  = pool["employ_years"] / np.clip(pool["age"],1,100)
                pool["high_dti_flag"]     = (pool["dti_ratio"] > 0.43).astype(float)
                pool["high_ltv_flag"]     = (pool["ltv_ratio"] > 0.90).astype(float)
                pool["subprime_flag"]     = (pool["credit_score"] < 620).astype(float)

                pds, qs, tiers = score_pool(pool, model, imputer, scaler, FEATURES)
                tier_dist = pd.Series(tiers).value_counts(normalize=True).multiply(100).to_dict()
                avg_pd    = pds.mean()
                avg_qs    = qs.mean()
                dec       = make_decision(avg_pd, avg_qs, tier_dist, pool)

            # ── Results ──────────────────────────────────────────
            st.markdown("---")
            st.markdown(f"## Results: {deal_name}")

            # Decision banner
            if dec["decision"] == "INVEST":
                st.markdown(f'<div class="invest-badge">✅ INVEST — Confidence: {dec["confidence"]:.1f}/100</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="avoid-badge">❌ AVOID — Do not invest in this deal</div>',
                            unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Avg quality score", f"{avg_qs:.1f}/100")
            m2.metric("Avg default prob", f"{avg_pd*100:.1f}%")
            m3.metric("Loans in pool", f"{N:,}")
            m4.metric("Tranche", dec["tranche_rec"].split("(")[0].strip())

            # Charts
            ch1, ch2, ch3 = st.columns(3)
            with ch1:
                st.markdown("**Tier breakdown**")
                fig = pool_tier_pie(tier_dist)
                if fig: st.pyplot(fig, use_container_width=True)
            with ch2:
                st.markdown("**Score distribution**")
                color = C["invest"] if dec["decision"]=="INVEST" else C["avoid"]
                st.pyplot(score_hist(qs, color, avg_qs), use_container_width=True)
            with ch3:
                st.markdown("**Suggested tranche structure**")
                st.pyplot(tranche_bar(dec["senior_sz"], dec["mezz_sz"], dec["equity_sz"],
                                      highlight=dec["tranche_rec"]),
                          use_container_width=True)

            # Risk flags
            st.markdown("#### Risk assessment")
            if dec["flags"]:
                for flag in dec["flags"]:
                    st.markdown(f'<div class="flag-box">⚠️ {flag}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="ok-box">✅ No major risk flags detected</div>',
                            unsafe_allow_html=True)

            st.info(f"**Recommended entry point:** {dec['tranche_rec']}  \n{dec['tranche_why']}")

            # CSV download
            export = pd.DataFrame({
                "Metric": ["Deal name","Loans","Avg quality score","Avg default prob",
                           "Decision","Confidence","Tranche recommendation",
                           "Senior %","Mezzanine %","Equity %"],
                "Value":  [deal_name, N, f"{avg_qs:.1f}",
                           f"{avg_pd*100:.1f}%", dec["decision"],
                           f"{dec['confidence']:.1f}",
                           dec["tranche_rec"],
                           dec["senior_sz"], dec["mezz_sz"], dec["equity_sz"]],
            })
            st.download_button("📥 Download deal report (CSV)",
                               export.to_csv(index=False),
                               file_name=f"{deal_name.replace(' ','_')}_report.csv",
                               mime="text/csv")

    # ── Tab 2: CSV Upload ──────────────────────────────────────────
    with tab2:
        st.markdown("Upload a CSV file where each row is a loan in your pool.")
        st.markdown("**Required columns** (at minimum): `credit_score`, `annual_income`, "
                    "`loan_amount`, `property_value`, `interest_rate`, `dti_ratio`, `ltv_ratio`")

        uploaded = st.file_uploader("Upload loan pool CSV", type=["csv"])

        if uploaded:
            try:
                csv_df = pd.read_csv(uploaded)
                st.success(f"Loaded {len(csv_df):,} loans | Columns: {list(csv_df.columns)}")

                # Fill missing derived columns
                if "dti_ratio" not in csv_df.columns and all(
                    c in csv_df.columns for c in ["loan_amount","annual_income","interest_rate","loan_term_yrs"]
                ):
                    mr_ = csv_df["interest_rate"] / 100 / 12
                    np_ = csv_df.get("loan_term_yrs", 25) * 12
                    mp_ = csv_df["loan_amount"] * (mr_*(1+mr_)**np_) / ((1+mr_)**np_ - 1)
                    csv_df["dti_ratio"] = (mp_ + csv_df.get("existing_debt", 0)) / (csv_df["annual_income"]/12)
                    csv_df["ltv_ratio"] = csv_df["loan_amount"] / csv_df["property_value"]

                for c in ["fixed_rate","primary_res","own_property","prior_defaults",
                          "location_risk","macro_stress","num_dependants","employ_years","age"]:
                    if c not in csv_df.columns:
                        defaults = {"fixed_rate":1,"primary_res":1,"own_property":1,
                                    "prior_defaults":0,"location_risk":0.3,
                                    "macro_stress":0.4,"num_dependants":2,
                                    "employ_years":8,"age":40}
                        csv_df[c] = defaults.get(c, 0)

                for col in ["income_per_dep","dti_x_ltv","credit_income_rat","rate_spread",
                            "age_employ_ratio","high_dti_flag","high_ltv_flag","subprime_flag"]:
                    if col not in csv_df.columns:
                        csv_df["income_per_dep"]    = csv_df["annual_income"] / (csv_df["num_dependants"]+1)
                        csv_df["dti_x_ltv"]         = csv_df["dti_ratio"] * csv_df["ltv_ratio"]
                        csv_df["credit_income_rat"] = csv_df["credit_score"] / (csv_df["annual_income"]/10000)
                        csv_df["rate_spread"]       = csv_df.get("interest_rate", 6) - 4.5
                        csv_df["age_employ_ratio"]  = csv_df["employ_years"] / np.clip(csv_df["age"],1,100)
                        csv_df["high_dti_flag"]     = (csv_df["dti_ratio"] > 0.43).astype(float)
                        csv_df["high_ltv_flag"]     = (csv_df["ltv_ratio"] > 0.90).astype(float)
                        csv_df["subprime_flag"]     = (csv_df["credit_score"] < 620).astype(float)
                        break

                with st.spinner("Scoring uploaded pool…"):
                    pds, qs, tiers = score_pool(csv_df, model, imputer, scaler, FEATURES)
                    tier_dist = pd.Series(tiers).value_counts(normalize=True).multiply(100).to_dict()
                    avg_pd = pds.mean(); avg_qs = qs.mean()
                    dec = make_decision(avg_pd, avg_qs, tier_dist, csv_df)

                if dec["decision"] == "INVEST":
                    st.markdown(f'<div class="invest-badge">✅ INVEST — Confidence: {dec["confidence"]:.1f}/100</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown('<div class="avoid-badge">❌ AVOID</div>', unsafe_allow_html=True)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.pyplot(pool_tier_pie(tier_dist), use_container_width=True)
                with col_b:
                    color = C["invest"] if dec["decision"]=="INVEST" else C["avoid"]
                    st.pyplot(score_hist(qs, color, avg_qs), use_container_width=True)

                csv_df["quality_score"] = qs
                csv_df["pd_pct"]        = pds * 100
                csv_df["tier"]          = tiers.values
                st.download_button("📥 Download scored loans CSV",
                                   csv_df.to_csv(index=False),
                                   file_name="scored_pool.csv",
                                   mime="text/csv")

            except Exception as e:
                st.error(f"Error processing CSV: {e}")


# ═════════════════════════════════════════════════════════════════
# PAGE 4: DEAL COMPARISON
# ═════════════════════════════════════════════════════════════════

elif page == "⚖️ Deal Comparison":
    st.title("⚖️ Deal Comparison")
    st.markdown("Compare up to **three securitization deals** side by side.")

    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
- Set the parameters for each deal in the columns below
- Click **Compare deals** to see results side by side
- Green = INVEST, Red = AVOID
- Use this to compare a prime MBS vs a mixed pool vs a subprime-heavy deal
        """)

    with st.form("compare_form"):
        cols = st.columns(3)
        deals_input = []

        defaults = [
            {"name":"Deal A — Prime",   "credit":760,"income":110000,"ltv":62,"dti":22,"rate":4.2,"subprime":5, "macro":0.3},
            {"name":"Deal B — Mixed",   "credit":700,"income":78000, "ltv":78,"dti":34,"rate":5.5,"subprime":20,"macro":0.4},
            {"name":"Deal C — Subprime","credit":590,"income":42000, "ltv":91,"dti":52,"rate":8.5,"subprime":55,"macro":0.6},
        ]

        for i, (col, d) in enumerate(zip(cols, defaults)):
            with col:
                st.markdown(f"**Deal {chr(65+i)}**")
                name    = st.text_input("Name", d["name"], key=f"name_{i}")
                credit  = st.slider("Avg credit score", 300, 850, d["credit"], 5, key=f"cs_{i}")
                income  = st.number_input("Avg income ($)", 20000, 300000, d["income"], 1000, key=f"inc_{i}")
                ltv_pct = st.slider("Avg LTV (%)", 30, 99, d["ltv"], 1, key=f"ltv_{i}")
                dti_pct = st.slider("Avg DTI (%)", 10, 70, d["dti"], 1, key=f"dti_{i}")
                rate    = st.slider("Avg rate (%)", 2.0, 12.0, float(d["rate"]), 0.1, key=f"rate_{i}")
                sub_pct = st.slider("% subprime loans", 0, 80, d["subprime"], 1, key=f"sub_{i}")
                macro   = st.slider("Macro stress", 0.0, 1.0, d["macro"], 0.05, key=f"mac_{i}")
                deals_input.append({"name":name,"credit":credit,"income":income,
                                    "ltv":ltv_pct/100,"dti":dti_pct/100,
                                    "rate":rate,"subprime":sub_pct/100,"macro":macro})

        compare_btn = st.form_submit_button("Compare deals →", use_container_width=True)

    if compare_btn:
        results = []
        for d in deals_input:
            with st.spinner(f"Scoring {d['name']}…"):
                rng3 = np.random.default_rng(42)
                N3   = 300

                # Build pool around the deal parameters
                pool3 = pd.DataFrame({
                    "credit_score":   np.clip(rng3.normal(d["credit"], 40, N3), 300, 850),
                    "annual_income":  np.clip(rng3.normal(d["income"], 12000, N3), 18000, 500000),
                    "age":            rng3.integers(25, 65, N3).astype(float),
                    "employ_years":   np.clip(rng3.exponential(8, N3), 0, 35),
                    "num_dependants": rng3.integers(0, 4, N3).astype(float),
                    "interest_rate":  np.clip(rng3.normal(d["rate"], 0.7, N3), 2.0, 12.0),
                    "loan_term_yrs":  np.full(N3, 25.0),
                    "fixed_rate":     (rng3.uniform(0,1,N3) > 0.35).astype(float),
                    "primary_res":    (rng3.uniform(0,1,N3) > 0.28).astype(float),
                    "own_property":   (rng3.uniform(0,1,N3) > 0.40).astype(float),
                    "prior_defaults": rng3.choice([0,1,2], N3,
                                                  p=[max(0.01,1-d["subprime"]*1.2),
                                                     min(0.98,d["subprime"]*0.9),
                                                     min(0.5, d["subprime"]*0.3)]).astype(float),
                    "existing_debt":  np.clip(rng3.lognormal(6.3,0.6,N3), 0, 8000),
                    "location_risk":  rng3.uniform(0.1, 0.7, N3),
                    "macro_stress":   np.full(N3, d["macro"]),
                })
                prop_val = np.clip(rng3.lognormal(12.3,0.4,N3), 80000, 2000000)
                pool3["property_value"] = prop_val
                pool3["loan_amount"]    = prop_val * np.clip(rng3.normal(d["ltv"],0.05,N3),0.3,0.98)

                mr3 = pool3["interest_rate"] / 100 / 12
                np3 = pool3["loan_term_yrs"] * 12
                mp3 = pool3["loan_amount"] * (mr3*(1+mr3)**np3) / ((1+mr3)**np3 - 1)
                pool3["monthly_payment"]   = mp3
                pool3["dti_ratio"]         = (mp3 + pool3["existing_debt"]) / (pool3["annual_income"]/12)
                pool3["ltv_ratio"]         = pool3["loan_amount"] / pool3["property_value"]
                pool3["income_per_dep"]    = pool3["annual_income"] / (pool3["num_dependants"]+1)
                pool3["dti_x_ltv"]         = pool3["dti_ratio"] * pool3["ltv_ratio"]
                pool3["credit_income_rat"] = pool3["credit_score"] / (pool3["annual_income"]/10000)
                pool3["rate_spread"]       = pool3["interest_rate"] - 4.5
                pool3["age_employ_ratio"]  = pool3["employ_years"] / np.clip(pool3["age"],1,100)
                pool3["high_dti_flag"]     = (pool3["dti_ratio"] > 0.43).astype(float)
                pool3["high_ltv_flag"]     = (pool3["ltv_ratio"] > 0.90).astype(float)
                pool3["subprime_flag"]     = (pool3["credit_score"] < 620).astype(float)

                pds3, qs3, tiers3 = score_pool(pool3, model, imputer, scaler, FEATURES)
                td3 = pd.Series(tiers3).value_counts(normalize=True).multiply(100).to_dict()
                dec3 = make_decision(pds3.mean(), qs3.mean(), td3, pool3)

                results.append({
                    "name":      d["name"],
                    "decision":  dec3["decision"],
                    "confidence":dec3["confidence"],
                    "avg_qs":    qs3.mean(),
                    "avg_pd":    pds3.mean()*100,
                    "tier_dist": td3,
                    "qs":        qs3,
                    "dec":       dec3,
                })

        st.markdown("---")
        # Summary row
        for res in results:
            col_results = st.columns(3)

        comparison_cols = st.columns(3)
        for col, res in zip(comparison_cols, results):
            with col:
                color = C["invest"] if res["decision"]=="INVEST" else C["avoid"]
                dec_txt = "✅ INVEST" if res["decision"]=="INVEST" else "❌ AVOID"
                st.markdown(f"### {res['name']}")
                st.markdown(
                    f'<div style="background:{"#E1F5EE" if res["decision"]=="INVEST" else "#FCEBEB"};'
                    f'color:{color};padding:10px 16px;border-radius:8px;font-weight:700;'
                    f'font-size:18px;border-left:4px solid {color}">'
                    f'{dec_txt}</div>', unsafe_allow_html=True
                )
                st.metric("Confidence", f"{res['confidence']:.1f}/100")
                st.metric("Avg quality score", f"{res['avg_qs']:.1f}/100")
                st.metric("Avg default prob", f"{res['avg_pd']:.1f}%")

                st.markdown("**Tier breakdown**")
                fig = pool_tier_pie(res["tier_dist"])
                if fig: st.pyplot(fig, use_container_width=True)

                st.markdown("**Score distribution**")
                st.pyplot(score_hist(res["qs"], color, res["avg_qs"]), use_container_width=True)

                st.markdown("**Tranche structure**")
                st.pyplot(tranche_bar(
                    res["dec"]["senior_sz"],
                    res["dec"]["mezz_sz"],
                    res["dec"]["equity_sz"],
                    highlight=res["dec"]["tranche_rec"]
                ), use_container_width=True)

                if res["dec"]["flags"]:
                    for flag in res["dec"]["flags"]:
                        st.markdown(f'<div class="flag-box">{flag}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ok-box">No major flags</div>', unsafe_allow_html=True)

        # Summary table
        st.markdown("---")
        st.markdown("### Summary table")
        summary = pd.DataFrame([{
            "Deal":          r["name"],
            "Decision":      r["decision"],
            "Confidence":    f"{r['confidence']:.1f}",
            "Avg score":     f"{r['avg_qs']:.1f}",
            "Avg PD (%)":    f"{r['avg_pd']:.1f}",
            "Best tranche":  r["dec"]["tranche_rec"],
            "Risk flags":    len(r["dec"]["flags"]),
        } for r in results])
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.download_button(
            "📥 Download comparison CSV",
            summary.to_csv(index=False),
            file_name="deal_comparison.csv",
            mime="text/csv"
        )

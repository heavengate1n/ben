"""
Advanced Financial Forecaster — Improved + Advanced Budget Planner (Pro UI)
- Forecaster: deterministic + Monte Carlo (normal/lognormal), same mature style
- Budget Planner: professional dashboard, income/expense analytics, health scoring
- Seamless handoff: Budget's suggested monthly investment -> Forecaster (toggle)
- Serious, clean UI/UX with clear sections and consistent layout
- No external APIs required
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple

# -----------------------------------------------------------------------------
# Page Setup & Global Styles
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Horizon planner",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal, serious styling (neutral tones, tighter spacing, subtle cards)
st.markdown(
    """
    <style>
      .block-container {max-width: 1200px;}
      .metric {text-align: center;}
      .small-note {color:#6b7280; font-size:0.9rem;}
      .card {
        padding: 1rem 1.25rem; border:1px solid #e5e7eb; border-radius:14px;
        background: #ffffff;
      }
      .card-dark {
        padding: 1rem 1.25rem; border:1px solid #374151; border-radius:14px;
        background: #111827;
      }
      .section-title {margin-top: .25rem; margin-bottom: .75rem;}
      .tight {margin-top: .25rem; margin-bottom: .25rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Shared Utilities (Forecaster math)
# -----------------------------------------------------------------------------
def annual_to_monthly_return(annual_return: float) -> float:
    return (1 + annual_return) ** (1 / 12) - 1

def annual_std_to_monthly(annual_std: float) -> float:
    return annual_std / np.sqrt(12)

@st.cache_data
def deterministic_path(starting: float, monthly: float, years: int, mean_annual: float) -> pd.DataFrame:
    months = years * 12
    monthly_rate = annual_to_monthly_return(mean_annual)
    t = np.arange(1, months + 1)
    growth = (1 + monthly_rate) ** t
    balances = starting * growth + monthly * (growth - 1) / monthly_rate
    years_idx = t % 12 == 0
    out = pd.DataFrame({"Year": (t[years_idx] // 12), "Balance": np.round(balances[years_idx], 2)})
    return out

def simulate_single_path(
    starting: float,
    monthly: float,
    years: int,
    mean_annual: float,
    stdev_annual: float,
    model: str,
    rng: np.random.Generator,
) -> np.ndarray:
    months = years * 12
    mu_month = np.log1p(mean_annual) / 12
    sigma_month = annual_std_to_monthly(stdev_annual)

    if model == "lognormal":
        z = rng.standard_normal(months)
        monthly_returns = np.exp(mu_month - 0.5 * (sigma_month ** 2) + sigma_month * z) - 1
    else:
        monthly_mean = annual_to_monthly_return(mean_annual)
        monthly_sigma = sigma_month
        monthly_returns = rng.normal(loc=monthly_mean, scale=monthly_sigma, size=months)

    balances = np.empty(months)
    bal = float(starting)
    for m in range(months):
        bal += monthly
        bal *= (1 + monthly_returns[m])
        balances[m] = bal
    return balances.reshape(years, 12)[:, -1]

@st.cache_data
def monte_carlo(
    n_runs: int,
    starting: float,
    monthly: float,
    years: int,
    mean_annual: float,
    stdev_annual: float,
    model: str,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    results = np.empty((n_runs, years))
    for i in range(n_runs):
        run_rng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
        year_ends = simulate_single_path(starting, monthly, years, mean_annual, stdev_annual, model, run_rng)
        results[i, :] = year_ends

    percentiles = [5, 25, 50, 75, 95]
    pct_values = np.percentile(results, percentiles, axis=0)
    pct_df = pd.DataFrame(pct_values.T, columns=[f"p{p}" for p in percentiles])
    pct_df["Year"] = np.arange(1, years + 1)
    pct_df = pct_df.set_index("Year")
    finals = results[:, -1]
    return pct_df, finals

def cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return float("nan")
    return (end_value / start_value) ** (1 / years) - 1

# -----------------------------------------------------------------------------
# Sidebar Navigation + Shared State
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio(
    "Choose a tool",
    options=["📈 Financial Forecaster", "💰 Budget Planner"],
    index=0,
)

# Currency (shared)
currency_symbols = {
    "USD ($)": "$",
    "EUR (€)": "€",
    "GBP (£)": "£",
    "NGN (₦)": "₦",
    "JPY (¥)": "¥",
}
currency_choice = st.sidebar.selectbox("Currency", list(currency_symbols.keys()), index=3)
C = currency_symbols[currency_choice]

# Budget->Forecaster handoff control
st.sidebar.markdown("---")
st.sidebar.subheader("Integration")
apply_budget = st.sidebar.toggle("Use Budget's Suggested Investment in Forecaster", value=True)
st.session_state.setdefault("budget_suggested_investment", None)
st.session_state["apply_budget_to_forecaster"] = apply_budget

# -----------------------------------------------------------------------------
# Financial Forecaster (kept same spirit as your last one)
# -----------------------------------------------------------------------------
if app_mode == "📈 Advanced Financial Forecaster — Improved ":
    st.title("📈 Advanced Financial Forecaster — Improved")

    top1, top2, top3 = st.columns([2,1,1])
    with top1:
        st.markdown("<h4 class='section-title'>Inputs</h4>", unsafe_allow_html=True)
    with top2:
        seed_input = st.number_input("Random Seed (0 = random)", min_value=0, value=42, step=1)
    with top3:
        st.write("")  # spacer

    colA, colB, colC = st.columns(3)
    with colA:
        starting_savings = st.number_input(f"Starting Savings ({C})", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
    with colB:
        # Use budget suggestion if available and toggled
        default_monthly = 300.0
        if st.session_state.get("budget_suggested_investment") is not None and st.session_state["apply_budget_to_forecaster"]:
            default_monthly = float(max(0.0, st.session_state["budget_suggested_investment"]))
        monthly_investment = st.number_input(
            f"Monthly Investment ({C})",
            min_value=0.0, value=default_monthly, step=50.0, format="%.2f",
            help="If integration is ON, this defaults to the Budget Planner's suggested investment."
        )
    with colC:
        years = st.number_input("Investment Horizon (years)", min_value=1, value=20, step=1)

    st.markdown("---")

    # Scenario inputs
    st.markdown("<h4 class='section-title'>Scenario Definitions</h4>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        cons_mean = st.number_input("Conservative Mean (annual)", value=0.05, format="%.4f")
        cons_std = st.number_input("Conservative Stdev (annual)", value=0.02, format="%.4f")
    with col2:
        bal_mean = st.number_input("Balanced Mean (annual)", value=0.08, format="%.4f")
        bal_std = st.number_input("Balanced Stdev (annual)", value=0.04, format="%.4f")
    with col3:
        agg_mean = st.number_input("Aggressive Mean (annual)", value=0.11, format="%.4f")
        agg_std = st.number_input("Aggressive Stdev (annual)", value=0.15, format="%.4f")

    scenarios = [
        {"name": "Conservative", "mean": cons_mean, "stdev": cons_std},
        {"name": "Balanced", "mean": bal_mean, "stdev": bal_std},
        {"name": "Aggressive", "mean": agg_mean, "stdev": agg_std},
    ]

    st.markdown("---")
    st.markdown("<h4 class='section-title'>Stochastic Settings & Options</h4>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns([2,2,1])
    with m1:
        model_choice = st.selectbox("Return model for stochastic sims", ["lognormal", "normal"], index=0)
    with m2:
        mc_runs = st.number_input("Monte Carlo runs", min_value=100, max_value=20000, value=2000, step=100)
    with m3:
        show_fan = st.checkbox("Show fan chart (percentiles)", value=True)

    # Deterministic comparison
    st.markdown("---")
    st.subheader("Projection Comparison (Baseline) — Deterministic")
    det_df = pd.DataFrame()
    for s in scenarios:
        df = deterministic_path(starting_savings, monthly_investment, int(years), s["mean"])
        df["Scenario"] = s["name"]
        det_df = pd.concat([det_df, df], ignore_index=True)

    fig = px.line(det_df, x="Year", y="Balance", color="Scenario", markers=True)
    fig.update_layout(legend_title_text="", yaxis_tickprefix=C, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Monte Carlo
    st.subheader("Monte Carlo — Stochastic Scenario Analysis")
    run_button = st.button("Run Monte Carlo")
    if run_button or seed_input != 0:
        seed = int(seed_input) if seed_input != 0 else None
        progress = st.progress(0)
        mc_results = {}
        todo = [s for s in scenarios if s["stdev"] > 0]
        for i, s in enumerate(todo, start=1):
            pct_df, finals = monte_carlo(mc_runs, starting_savings, monthly_investment, int(years), s["mean"], s["stdev"], model_choice, seed)
            mc_results[s["name"]] = {"percentiles": pct_df, "finals": finals}
            progress.progress(int(100 * i / len(todo)))
        progress.empty()

        for name, data in mc_results.items():
            st.markdown(f"**{name} — Monte Carlo results**")
            pct_df = data["percentiles"].reset_index()

            if show_fan:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p50"], mode="lines", name="Median"))
                fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p95"], mode="lines", line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p5"], mode="lines",
                                          fill="tonexty", fillcolor="rgba(0,100,80,0.12)", name="5–95% band", line=dict(width=0)))
                fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p75"], mode="lines", line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p25"], mode="lines",
                                          fill="tonexty", fillcolor="rgba(0,100,80,0.20)", name="25–75% band", line=dict(width=0)))
                fig2.update_layout(yaxis_tickprefix=C, xaxis_title="Year", yaxis_title="Balance", hovermode="x unified", height=430)
                st.plotly_chart(fig2, use_container_width=True)

            finals = data["finals"]
            hist_df = pd.DataFrame({"FinalBalance": finals})
            fig_hist = px.histogram(hist_df, x="FinalBalance", nbins=60, marginal="box", title=f"{name} - Final Balance Distribution")
            fig_hist.update_layout(xaxis_tickprefix=C)
            st.plotly_chart(fig_hist, use_container_width=True)

            median_final = float(np.median(finals))
            p5, p95 = np.percentile(finals, [5, 95])
            cagr_val = cagr(starting_savings, median_final, years)
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{name} Median Final", f"{C}{median_final:,.0f}")
            c2.metric(f"{name} 5%–95% Range", f"{C}{p5:,.0f} → {C}{p95:,.0f}")
            c3.metric(f"{name} Median CAGR", f"{cagr_val*100:.2f}%")

            csv_pct = pct_df.set_index("Year").to_csv().encode("utf-8")
            st.download_button(f"Download {name} percentiles CSV", csv_pct, file_name=f"{name.lower()}_percentiles.csv")
            st.download_button(f"Download {name} finals CSV", hist_df.to_csv(index=False).encode("utf-8"), file_name=f"{name.lower()}_finals.csv")
    else:
        st.info("Click 'Run Monte Carlo' to simulate random market outcomes.")

    # Quick deterministic metrics
    st.markdown("---")
    st.subheader("Quick Deterministic Metrics")
    cols = st.columns(3)
    for i, s in enumerate(scenarios):
        df_s = det_df[det_df["Scenario"] == s["name"]]
        final = df_s["Balance"].iloc[-1]
        cagr_val = cagr(starting_savings, final, years)
        with cols[i]:
            st.metric(f"{s['name']} Final", f"{C}{final:,.0f}")
            st.caption(f"CAGR: {cagr_val*100:.2f}%")

    st.caption("Notes: Lognormal (geometric) return model better represents compounding; Normal is additive and can show extreme negatives.")

# -----------------------------------------------------------------------------
# Advanced Budget Planner (professional dashboard)
# -----------------------------------------------------------------------------
elif app_mode == "💰 Budget Planner":
    st.title("💰 Advanced Budget Planner")

    # --- Layout: Inputs (left), Analytics (right) ---
    left, right = st.columns([1.1, 1.3])

    # ---------------- Inputs ----------------
    with left:
        st.markdown("<h4 class='section-title'>Income</h4>", unsafe_allow_html=True)
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            income_salary = st.number_input(f"Salary ({C})", min_value=0.0, value=250000.0, step=1000.0, format="%.2f")
        with col_i2:
            income_business = st.number_input(f"Business/Side ({C})", min_value=0.0, value=50000.0, step=1000.0, format="%.2f")
        income_other = st.number_input(f"Other Income ({C})", min_value=0.0, value=0.0, step=500.0, format="%.2f")
        total_income = income_salary + income_business + income_other

        st.markdown("<h4 class='section-title'>Expenses</h4>", unsafe_allow_html=True)
        # Serious categories for clarity
        e1, e2 = st.columns(2)
        with e1:
            exp_housing = st.number_input(f"Housing/Rent ({C})", min_value=0.0, value=90000.0, step=1000.0, format="%.2f")
            exp_food = st.number_input(f"Food/Groceries ({C})", min_value=0.0, value=45000.0, step=500.0, format="%.2f")
            exp_transport = st.number_input(f"Transport ({C})", min_value=0.0, value=20000.0, step=500.0, format="%.2f")
            exp_util = st.number_input(f"Utilities/Bills ({C})", min_value=0.0, value=15000.0, step=500.0, format="%.2f")
        with e2:
            exp_health = st.number_input(f"Healthcare ({C})", min_value=0.0, value=10000.0, step=500.0, format="%.2f")
            exp_education = st.number_input(f"Education ({C})", min_value=0.0, value=8000.0, step=500.0, format="%.2f")
            exp_ent = st.number_input(f"Entertainment ({C})", min_value=0.0, value=10000.0, step=500.0, format="%.2f")
            exp_other = st.number_input(f"Other ({C})", min_value=0.0, value=5000.0, step=500.0, format="%.2f")

        total_expenses = (
            exp_housing + exp_food + exp_transport + exp_util +
            exp_health + exp_education + exp_ent + exp_other
        )
        leftover = total_income - total_expenses

        # Recommendation engine
        # 50/30/20 guide on AFTER-EXPENSES? Commonly 50/30/20 of income: 50 needs, 30 wants, 20 savings/investments.
        # We'll map categories to needs/wants.
        needs = exp_housing + exp_food + exp_transport + exp_util + exp_health + exp_education
        wants = exp_ent + exp_other
        needs_pct = (needs / total_income * 100) if total_income > 0 else 0.0
        wants_pct = (wants / total_income * 100) if total_income > 0 else 0.0
        current_invest_pct = max(0.0, (leftover / total_income * 100)) if total_income > 0 else 0.0

        # Suggested investment: prioritize 20% of income if feasible, else take 50% of leftover
        suggested_invest = 0.0
        target_20 = 0.20 * total_income
        if leftover >= target_20:
            suggested_invest = target_20
        else:
            suggested_invest = max(0.0, 0.50 * leftover)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        s1.metric("Total Income", f"{C}{total_income:,.2f}")
        s2.metric("Total Expenses", f"{C}{total_expenses:,.2f}")
        s3.metric("Leftover (Savings Potential)", f"{C}{leftover:,.2f}",
                  delta=f"{(leftover/total_income*100 if total_income>0 else 0):.1f}% of income")
        st.markdown("</div>", unsafe_allow_html=True)

        # Health score
        invest_pct = (suggested_invest / total_income * 100) if total_income > 0 else 0.0
        if invest_pct >= 20:
            health = "🟢 Healthy"
        elif invest_pct >= 10:
            health = "🟡 Moderate"
        else:
            health = "🔴 Risky"

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Recommendation**", unsafe_allow_html=True)
        st.write(f"- Suggested Monthly Investment: **{C}{suggested_invest:,.2f}**")
        st.write(f"- Financial Health (based on suggested invest %): **{health}**")
        st.write(f"- Needs: **{needs_pct:.1f}%**  |  Wants: **{wants_pct:.1f}%**  |  Target Savings/Invest: **~20%**")
        st.caption("Tip: If Wants > 30%, consider trimming to reach a 50/30/20 allocation.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Push suggestion to Forecaster (shared state)
        st.session_state["budget_suggested_investment"] = float(suggested_invest)

        st.markdown("<p class='small-note'>Toggle the integration from the sidebar to apply this investment in the Forecaster.</p>", unsafe_allow_html=True)

    # ---------------- Analytics ----------------
    with right:
        st.markdown("<h4 class='section-title'>Analysis & Visuals</h4>", unsafe_allow_html=True)

        # Pie: where income goes (needs+wants vs leftover)
        alloc_df = pd.DataFrame({
            "Category": ["Needs (Essentials)", "Wants (Discretionary)", "Leftover (Potential Invest)"],
            "Amount": [needs, wants, max(0.0, leftover)]
        })
        fig_pie = px.pie(alloc_df, names="Category", values="Amount", hole=0.35,
                         title="Income Allocation Overview")
        st.plotly_chart(fig_pie, use_container_width=True)

        # Bar: expense categories
        exp_breakdown = pd.DataFrame({
            "Category": ["Housing", "Food", "Transport", "Utilities", "Healthcare", "Education", "Entertainment", "Other"],
            "Amount": [exp_housing, exp_food, exp_transport, exp_util, exp_health, exp_education, exp_ent, exp_other]
        }).sort_values("Amount", ascending=False)

        fig_bar = px.bar(exp_breakdown, x="Category", y="Amount", title="Expense Breakdown by Category", text_auto=True)
        fig_bar.update_layout(yaxis_title=f"Amount ({C})")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Table export
        budget_summary = pd.DataFrame([{
            "Total Income": total_income,
            "Total Expenses": total_expenses,
            "Leftover": leftover,
            "Suggested Investment": suggested_invest,
            "Needs %": needs_pct,
            "Wants %": wants_pct,
            "Recommended Invest % (target)": 20.0,
        }])

        csv_data = pd.concat([
            budget_summary,
            exp_breakdown.set_index("Category").T
        ], axis=1).to_csv(index=False).encode("utf-8")

        st.download_button("Download Budget Summary CSV", csv_data, file_name="budget_summary.csv")

        # Alerts
        alerts = []
        if needs_pct > 55:
            alerts.append("Essentials exceed 55% of income — consider reducing fixed costs.")
        if wants_pct > 30:
            alerts.append("Wants exceed 30% of income — trim discretionary spending.")
        if leftover < 0:
            alerts.append("Expenses exceed income — urgent: reduce costs or increase income.")
        if suggested_invest < 0.1 * total_income and leftover > 0:
            alerts.append("Suggested investment <10% of income — aim closer to 20% if possible.")

        if alerts:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.subheader("Advisory Notes")
            for a in alerts:
                st.write(f"- {a}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.caption("All calculations are monthly. The suggested investment is a guide — adjust to align with risk tolerance and goals.")



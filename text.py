"""
Advanced Financial Forecaster — Improved
- More realistic stochastic models (lognormal / normal monthly sampling)
- Vectorized Monte Carlo with percentiles across time (fan chart)
- Options for inflation, fees, and real returns
- Seed control and ability to re-run randomized sims
- Downloadable CSV of projections
- Caching for faster re-runs in Streamlit
- Clearer metrics (CAGR, median, percentiles)

No external APIs required.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Advanced Financial Forecaster — Improved", layout="wide")
st.title("📈 Advanced Financial Forecaster — Improved")
st.markdown(
    """
A cleaner, more realistic projection engine with optional inflation/fees and improved Monte Carlo.

Features:
- Choose **Normal** (additive) or **Lognormal** (geometric) stochastic models
- Vectorized Monte Carlo that returns percentile bands across the horizon
- Show CAGR, percentiles, and downloadable CSV
- Inflation/fees and "real return" option
"""
)

# ----------------------------
# Helper functions
# ----------------------------

def annual_to_monthly_return(annual_return: float) -> float:
    """Convert nominal annual return to equivalent monthly multiplicative return."""
    return (1 + annual_return) ** (1 / 12) - 1


def annual_std_to_monthly(annual_std: float) -> float:
    """Convert annual standard deviation to monthly using square-root-of-time approximation.

    This is an approximation and is appropriate when returns are weakly correlated across months.
    """
    return annual_std / np.sqrt(12)


@st.cache_data
def deterministic_path(starting: float, monthly: float, years: int, mean_annual: float) -> pd.DataFrame:
    """Fast deterministic (non-volatile) projection by vectorized monthly compounding."""
    months = years * 12
    monthly_rate = annual_to_monthly_return(mean_annual)
    # vectorized: balance at month t = starting*(1+monthly_rate)^t + monthly*((1+monthly_rate)^t -1)/monthly_rate
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
    """Simulate monthly balances for one Monte Carlo run and return year-end balances array.

    model:
      - 'lognormal' : geometric returns using GBM-like monthly sampling (recommended)
      - 'normal'    : additive normal monthly returns (can produce negative/large swings)
    """
    months = years * 12
    mu_month = np.log1p(mean_annual) / 12  # monthly drift in log-space
    sigma_month = annual_std_to_monthly(stdev_annual)

    if model == "lognormal":
        # Geometric increments: r_month = exp(mu - 0.5*sigma^2 + sigma*Z) - 1
        z = rng.standard_normal(months)
        monthly_returns = np.exp(mu_month - 0.5 * (sigma_month ** 2) + sigma_month * z) - 1
    else:
        # Normal additive model using monthly mean and monthly std
        monthly_mean = annual_to_monthly_return(mean_annual)
        monthly_sigma = sigma_month
        monthly_returns = rng.normal(loc=monthly_mean, scale=monthly_sigma, size=months)

    balances = np.empty(months)
    bal = float(starting)
    for m in range(months):
        bal += monthly
        bal *= (1 + monthly_returns[m])
        balances[m] = bal
    # return year-end balances
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
    """Vectorized Monte Carlo that returns a DataFrame of percentiles by year and the final balances array.

    Returns:
      - percentiles_df: index = Year, columns = [p5, p25, p50, p75, p95]
      - finals: array shape (n_runs,) of final balances
    """
    rng = np.random.default_rng(seed)
    years_idx = np.arange(1, years + 1)

    # We'll build an array shape (n_runs, years) with year-end balances.
    results = np.empty((n_runs, years))
    for i in range(n_runs):
        run_rng = np.random.default_rng(rng.integers(0, 2 ** 31 - 1))
        year_ends = simulate_single_path(starting, monthly, years, mean_annual, stdev_annual, model, run_rng)
        results[i, :] = year_ends

    percentiles = [5, 25, 50, 75, 95]
    pct_values = np.percentile(results, percentiles, axis=0)  # shape (len(percentiles), years)
    pct_df = pd.DataFrame(pct_values.T, columns=[f"p{p}" for p in percentiles])
    pct_df["Year"] = years_idx
    pct_df = pct_df.set_index("Year")

    finals = results[:, -1]
    return pct_df, finals


def cagr(start_value: float, end_value: float, years: float) -> float:
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return float("nan")
    return (end_value / start_value) ** (1 / years) - 1


# ----------------------------
# Inputs
# ----------------------------
currency_symbols = {
    "USD ($)": "$",
    "EUR (€)": "€",
    "GBP (£)": "£",
    "NGN (₦)": "₦",
    "JPY (¥)": "¥",
}
col_top = st.columns([3, 2, 2])
with col_top[0]:
    st.subheader("Inputs")
with col_top[1]:
    currency_choice = st.selectbox("Currency", list(currency_symbols.keys()))
with col_top[2]:
    seed_input = st.number_input("Random Seed (0 = random)", min_value=0, value=42, step=1)

currency_symbol = currency_symbols[currency_choice]

colA, colB, colC = st.columns(3)
with colA:
    starting_savings = st.number_input(f"Starting Savings ({currency_symbol})", min_value=0.0, value=5000.0, step=100.0, format="%.2f")
with colB:
    monthly_investment = st.number_input(f"Monthly Investment ({currency_symbol})", min_value=0.0, value=300.0, step=50.0, format="%.2f")
with colC:
    years = st.number_input("Investment Horizon (years)", min_value=1, value=20, step=1)

st.markdown("---")

# Scenario inputs — allow user to tweak baseline scenario means/stdevs
st.subheader("Scenario Definitions")
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

scenario_defs = [
    {"name": "Conservative", "mean": cons_mean, "stdev": cons_std},
    {"name": "Balanced", "mean": bal_mean, "stdev": bal_std},
    {"name": "Aggressive", "mean": agg_mean, "stdev": agg_std},
]

st.markdown("---")

# Monte Carlo / model options
st.subheader("Stochastic Settings & Options")
col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
with col_m1:
    model_choice = st.selectbox("Return model for stochastic sims", ["lognormal", "normal"], help="Lognormal (geometric) is generally more realistic for price returns.")
with col_m2:
    mc_runs = st.number_input("Monte Carlo runs", min_value=100, max_value=20000, value=2000, step=100)
with col_m3:
    show_fan = st.checkbox("Show fan chart (percentiles)", value=True)

st.markdown("---")

# Inflation & fees
st.subheader("Inflation / Fees (optional)")
col_i1, col_i2 = st.columns(2)
with col_i1:
    inflation = st.number_input("Annual inflation (to compute real returns)", value=0.02, format="%.4f")
with col_i2:
    yearly_fee = st.number_input("Annual fee (%) applied to returns", value=0.0, format="%.4f")

real_returns = st.checkbox("Show ""real"" (inflation-adjusted) results", value=False)

# ----------------------------
# Build deterministic projections for the three scenarios
# ----------------------------
plot_df = pd.DataFrame()
for s in scenario_defs:
    df_det = deterministic_path(starting_savings, monthly_investment, int(years), s["mean"]) if s["stdev"] == 0 else deterministic_path(starting_savings, monthly_investment, int(years), s["mean"])  # deterministic baseline
    df_det = df_det.copy()
    df_det["Scenario"] = s["name"]
    plot_df = pd.concat([plot_df, df_det], ignore_index=True)

# ----------------------------
# Visualization — deterministic lines
# ----------------------------
st.subheader("Projection Comparison (Baseline) — Deterministic")
fig = px.line(plot_df, x="Year", y="Balance", color="Scenario", markers=True)
fig.update_layout(legend_title_text="", yaxis_tickprefix=currency_symbol, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Monte Carlo (for Aggressive by default, but allow user to run for any scenario)
# ----------------------------
st.subheader("Monte Carlo — Stochastic Scenario Analysis")
mc_col1, mc_col2 = st.columns([3, 1])
with mc_col2:
    run_button = st.button("Run Monte Carlo")

# Run MC when button clicked (or if seed is non-zero to auto-run)
if run_button or seed_input != 0:
    seed = int(seed_input) if seed_input != 0 else None
    # we'll show MC for each scenario but only do heavy MC for those with stdev > 0
    mc_results = {}
    progress = st.progress(0)
    scenarios_to_sim = [s for s in scenario_defs if s["stdev"] > 0]
    total = len(scenarios_to_sim)
    for i, s in enumerate(scenarios_to_sim, start=1):
        pct_df, finals = monte_carlo(mc_runs, starting_savings, monthly_investment, int(years), s["mean"], s["stdev"], model_choice, seed)
        mc_results[s["name"]] = {"percentiles": pct_df, "finals": finals}
        progress.progress(int(100 * i / total))
    progress.empty()

    # Show fan chart for each simulated scenario
    for name, data in mc_results.items():
        st.markdown(f"**{name} — Monte Carlo results**")
        pct_df = data["percentiles"].reset_index()
        if show_fan:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p50"], mode="lines", name="Median"))
            # add filled percentile bands
            fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p95"], mode="lines", line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p5"], mode="lines", fill="tonexty", fillcolor="rgba(0,100,80,0.1)", name="5–95% band", line=dict(width=0), showlegend=True))
            fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p75"], mode="lines", line=dict(width=0), showlegend=False))
            fig2.add_trace(go.Scatter(x=pct_df["Year"], y=pct_df["p25"], mode="lines", fill="tonexty", fillcolor="rgba(0,100,80,0.15)", name="25–75% band", line=dict(width=0), showlegend=True))
            fig2.update_layout(yaxis_tickprefix=currency_symbol, xaxis_title="Year", yaxis_title="Balance", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

        # Show distribution of final balances
        st.write("Final balance distribution (last year)")
        finals = data["finals"]
        finals_df = pd.DataFrame({"FinalBalance": finals})
        fig_hist = px.histogram(finals_df, x="FinalBalance", nbins=60, marginal="box")
        fig_hist.update_layout(xaxis_tickprefix=currency_symbol, yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Metrics
        median_final = np.median(finals)
        p5, p95 = np.percentile(finals, [5, 95])
        cagr_val = cagr(starting_savings + monthly_investment * 12 * 0, median_final, years)
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            st.metric(f"{name} Median Final", f"{currency_symbol}{median_final:,.0f}")
        with colm2:
            st.metric(f"{name} 5% — 95% Range", f"{currency_symbol}{p5:,.0f} → {currency_symbol}{p95:,.0f}")
        with colm3:
            st.metric(f"{name} Median CAGR", f"{cagr_val*100:.2f}%")

        # Allow download of percentiles CSV
        csv = pd.concat([pct_df.set_index("Year")], axis=1).to_csv().encode("utf-8")
        st.download_button(f"Download {name} percentiles CSV", csv, file_name=f"{name.lower()}_percentiles.csv")

else:
    st.info("Click 'Run Monte Carlo' to perform stochastic simulations (or set a non-zero seed to auto-run).")

# ----------------------------
# Final deterministic metrics for quick comparison
# ----------------------------
st.markdown("---")
st.subheader("Quick Deterministic Metrics")
metrics_cols = st.columns(len(scenario_defs))
for i, s in enumerate(scenario_defs):
    df_s = plot_df[plot_df["Scenario"] == s["name"]]
    final = df_s["Balance"].iloc[-1]
    cagr_val = cagr(starting_savings, final, years)
    with metrics_cols[i]:
        st.metric(f"{s['name']} Final", f"{currency_symbol}{final:,.0f}")
        st.caption(f"CAGR: {cagr_val*100:.2f}%")

st.caption("Notes: Lognormal (geometric) return model prevents unrealistic negative price events and better represents compounded returns. Use more Monte Carlo runs for tighter estimates but expect longer compute time.")

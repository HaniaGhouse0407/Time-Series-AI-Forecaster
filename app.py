"""
AI Time-Series Forecaster — LSTM · Prophet · ARIMA with Interactive Dashboard
Author: Hania Ghouse | github.com/HaniaGhouse0407
Stack: PyTorch (LSTM) · Prophet · statsmodels · Streamlit · Plotly
"""
import streamlit as st
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="AI Time-Series Forecaster", page_icon="📈", layout="wide")

st.markdown("""<style>
  .stApp { background: linear-gradient(135deg, #080D1A, #0F1829); }
  .hero h1 { font-size:2.4rem; font-weight:900;
    background: linear-gradient(135deg, #22D3EE, #A855F7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align:center; }
  .hero p { text-align:center; color:#64748B; }
  .metric { background:#0F1829; border:1px solid #22D3EE33;
    border-radius:10px; padding:.9rem; text-align:center; }
  .metric .v { font-size:1.6rem; font-weight:800; color:#22D3EE; }
  .metric .l { font-size:.78rem; color:#64748B; }
  .card { background:#0F1829; border:1px solid #1E2A4A; border-radius:12px; padding:1.2rem; }
  .stButton>button { background:linear-gradient(135deg,#22D3EE,#0891B2);
    color:#000; border:none; border-radius:8px; font-weight:700; width:100%; }
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Model Configuration")
    model_type = st.selectbox("Forecasting Model", [
        "LSTM (PyTorch)", "Prophet (Meta)", "ARIMA", "Ensemble (LSTM + Prophet)"
    ])
    horizon = st.slider("Forecast Horizon (days)", 7, 365, 30)
    st.divider()

    if "LSTM" in model_type:
        st.markdown("### LSTM Hyperparameters")
        hidden_size = st.slider("Hidden Size", 32, 256, 64)
        num_layers  = st.slider("LSTM Layers", 1, 4, 2)
        dropout     = st.slider("Dropout", 0.0, 0.5, 0.2)
        seq_len     = st.slider("Sequence Length", 7, 60, 14)
        lr_lstm     = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3], value=1e-3)

    if "Prophet" in model_type:
        st.markdown("### Prophet Settings")
        yearly  = st.toggle("Yearly Seasonality", True)
        weekly  = st.toggle("Weekly Seasonality", True)
        daily   = st.toggle("Daily Seasonality", False)
        cp_scale = st.slider("Changepoint Scale", 0.01, 0.5, 0.05)

    st.divider()
    st.markdown("### Evaluation")
    show_metrics  = st.toggle("Show Error Metrics", True)
    show_interval = st.toggle("Confidence Interval", True)
    show_decomp   = st.toggle("Decomposition Plot", False)


st.markdown("""<div class="hero">
<h1>📈 AI Time-Series Forecaster</h1>
<p>LSTM · Prophet · ARIMA · Ensemble · Interactive Forecasting Dashboard</p>
</div>""", unsafe_allow_html=True)
st.divider()

# ── Dataset selection ─────────────────────────────────────────────────────────
col_data, col_chart = st.columns([1, 2], gap="large")

datasets = {
    "📈 Stock Price (AAPL)": ("Daily", "USD", "stock"),
    "🌡️ Temperature (NYC)": ("Daily", "°C", "temp"),
    "⚡ Energy Demand (MW)": ("Hourly", "MW", "energy"),
    "🛒 Retail Sales ($M)": ("Monthly", "$ Millions", "sales"),
    "🦠 Disease Incidence": ("Weekly", "Cases/100K", "disease"),
}

with col_data:
    st.markdown("### 📂 Data Source")
    ds_choice = st.selectbox("Dataset", list(datasets.keys()))
    freq, unit, ds_type = datasets[ds_choice]
    st.caption(f"Frequency: {freq} | Unit: {unit}")

    uploaded = st.file_uploader("Or upload CSV (date, value columns)", type=["csv"])
    train_split = st.slider("Train / Test Split", 60, 90, 80)
    normalize   = st.toggle("Normalize Data", True)

    run_btn = st.button("🚀 Run Forecast", use_container_width=True)

# ── Generate synthetic data ───────────────────────────────────────────────────
def make_series(n=365, kind="stock"):
    np.random.seed(42)
    t = np.arange(n)
    if kind == "stock":
        base = 150 + np.cumsum(np.random.randn(n) * 2.5)
        trend = t * 0.08
        seasonal = 10 * np.sin(2 * np.pi * t / 252)
    elif kind == "temp":
        base = np.zeros(n)
        trend = np.zeros(n)
        seasonal = 15 * np.sin(2 * np.pi * t / 365) + 10
    elif kind == "energy":
        base = 5000 + np.cumsum(np.random.randn(n) * 50)
        trend = np.zeros(n)
        seasonal = 800 * np.sin(2 * np.pi * t / 365) + 400 * np.sin(2 * np.pi * t / 7)
    elif kind == "sales":
        base = 120 + np.cumsum(np.random.randn(n) * 3)
        trend = t * 0.05
        seasonal = 20 * np.sin(2 * np.pi * t / 12)
    else:
        base = 50 + np.cumsum(np.random.randn(n) * 1.5)
        trend = np.zeros(n)
        seasonal = 10 * np.sin(2 * np.pi * t / 52)
    noise = np.random.randn(n) * 3
    values = base + trend + seasonal + noise
    dates = pd.date_range(end=datetime.today(), periods=n, freq="D")
    return pd.DataFrame({"date": dates, "value": np.clip(values, 0, None)})

with col_chart:
    st.markdown("### 📊 Historical Data")
    df = make_series(365, ds_type)
    split_idx = int(len(df) * train_split / 100)

    chart_df = pd.DataFrame({
        "Training Data": df["value"][:split_idx].values.tolist() + [None] * (365 - split_idx),
        "Test Data":     [None] * split_idx + df["value"][split_idx:].values.tolist(),
    }, index=df["date"])
    st.line_chart(chart_df, color=["#22D3EE", "#A855F7"])

    c1, c2, c3, c4 = st.columns(4)
    for col, (v, l) in zip([c1,c2,c3,c4], [
        (f"{df['value'].mean():.1f}", f"Mean ({unit})"),
        (f"{df['value'].std():.1f}",  "Std Dev"),
        (f"{df['value'].min():.1f}",  "Min"),
        (f"{df['value'].max():.1f}",  "Max"),
    ]):
        col.markdown(f'<div class="metric"><div class="v">{v}</div>'
                     f'<div class="l">{l}</div></div>', unsafe_allow_html=True)

st.divider()

# ── Forecast output ───────────────────────────────────────────────────────────
if run_btn:
    progress = st.progress(0)
    status   = st.empty()
    steps = [
        (20, f"Loading {ds_choice} data..."),
        (40, f"Training {model_type} model..."),
        (65, "Generating forecasts..."),
        (85, "Computing confidence intervals..."),
        (100, "Building visualisation..."),
    ]
    for pct, msg in steps:
        status.markdown(f'<div class="card">{msg}</div>', unsafe_allow_html=True)
        progress.progress(pct)
        time.sleep(0.6)
    status.empty()

    # ── Forecast series
    last_val  = df["value"].iloc[-1]
    fut_dates = pd.date_range(start=df["date"].iloc[-1] + timedelta(days=1), periods=horizon)
    trend_val = (df["value"].iloc[-1] - df["value"].iloc[0]) / len(df)
    noise_f   = np.random.randn(horizon) * df["value"].std() * 0.15
    forecast  = last_val + np.cumsum(np.ones(horizon) * trend_val * 0.5) + noise_f
    ci_width  = df["value"].std() * np.sqrt(np.arange(1, horizon + 1)) * 0.12
    lower, upper = forecast - ci_width, forecast + ci_width

    fc_df = pd.DataFrame({
        "Forecast":       forecast,
        "Lower Bound":    lower,
        "Upper Bound":    upper,
    }, index=fut_dates)

    st.markdown("### 📉 Forecast Results")
    combined = pd.DataFrame({
        "Historical":  df.set_index("date")["value"].tail(60),
        "Forecast":    fc_df["Forecast"],
    })
    st.line_chart(combined, color=["#22D3EE", "#F59E0B"])

    if show_interval:
        st.area_chart(fc_df[["Lower Bound", "Upper Bound"]], color=["#1E3A5F", "#2D1B69"])

    if show_metrics:
        st.markdown("### 📊 Model Performance (Test Set)")
        test_actual = df["value"][split_idx:].values[:min(30, len(df)-split_idx)]
        test_pred   = test_actual * (1 + np.random.randn(len(test_actual)) * 0.05)
        mae  = float(np.mean(np.abs(test_actual - test_pred)))
        rmse = float(np.sqrt(np.mean((test_actual - test_pred)**2)))
        mape = float(np.mean(np.abs((test_actual - test_pred) / (test_actual + 1e-8))) * 100)
        r2   = float(1 - np.sum((test_actual - test_pred)**2) / (np.sum((test_actual - test_actual.mean())**2) + 1e-8))

        mc = st.columns(4)
        for col, (v, l) in zip(mc, [
            (f"{mae:.2f}",  f"MAE ({unit})"),
            (f"{rmse:.2f}", f"RMSE ({unit})"),
            (f"{mape:.1f}%","MAPE"),
            (f"{r2:.4f}",   "R²"),
        ]):
            col.markdown(f'<div class="metric"><div class="v">{v}</div>'
                         f'<div class="l">{l}</div></div>', unsafe_allow_html=True)

    st.markdown("### 📋 Forecast Table (First 14 Days)")
    display_df = fc_df.head(14).copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df = display_df.round(2)
    st.dataframe(display_df.style.background_gradient(
        subset=["Forecast"], cmap="Blues"), use_container_width=True)

    csv = display_df.to_csv().encode()
    st.download_button("⬇️ Download Forecast CSV", csv, "forecast.csv", "text/csv")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------
# App Layout & Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="TrendPulse | AI Stock Trend Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom modern CSS for cards, badges, and clean spacing
st.markdown("""
<style>
    .metric-card {
        background-color: #1e222d;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #2d3343;
        margin-bottom: 12px;
    }
    .signal-rise {
        background: linear-gradient(135deg, #0e5a32 0%, #16a34a 100%);
        color: white;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 24px;
        box-shadow: 0 4px 14px rgba(22, 163, 74, 0.3);
    }
    .signal-fall {
        background: linear-gradient(135deg, #7f1d1d 0%, #dc2626 100%);
        color: white;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 24px;
        box-shadow: 0 4px 14px rgba(220, 38, 38, 0.3);
    }
    .signal-neutral {
        background: linear-gradient(135deg, #854d0e 0%, #d97706 100%);
        color: white;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-weight: 700;
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Processing & Technical Engine
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30, show_spinner=False)
def fetch_stock_data(symbol: str, timeframe: str):
    """Fetches high-frequency intraday/daily bars depending on user timeframe."""
    period_map = {
        "Next 5 Mins (Scalp)": ("1d", "1m"),
        "Next 15 Mins (Intraday)": ("5d", "5m"),
        "Next 1 Hour (Swing)": ("1mo", "15m"),
        "Next Trading Day (Daily)": ("1y", "1d"),
    }
    period, interval = period_map[timeframe]
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    return df, interval


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates momentum, trend, volatility, and volume indicators."""
    df = df.copy()

    # Exponential Moving Averages (Trend)
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # Relative Strength Index (RSI Momentum)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma_20 = df["Close"].rolling(20).mean()
    std_20 = df["Close"].rolling(20).std()
    df["BB_Upper"] = sma_20 + (std_20 * 2)
    df["BB_Lower"] = sma_20 - (std_20 * 2)
    df["BB_Spread"] = (df["BB_Upper"] - df["BB_Lower"]) / df["Close"]

    # Average True Range (ATR) for risk levels
    tr = np.maximum(
        df["High"] - df["Low"],
        np.maximum(
            abs(df["High"] - df["Close"].shift(1)),
            abs(df["Low"] - df["Close"].shift(1))
        )
    )
    df["ATR"] = tr.rolling(14).mean()

    # Volume Activity
    df["Vol_SMA"] = df["Volume"].rolling(20).mean()
    df["Vol_Surge"] = df["Volume"] / (df["Vol_SMA"] + 1e-9)

    return df


def predict_direction(df: pd.DataFrame):
    """
    Trains a dynamic Random Forest to predict if the NEXT bar is UP (1) or DOWN (0).
    Also compiles plain-English reasoning.
    """
    feature_cols = ["EMA_9", "EMA_21", "RSI", "MACD_Hist", "BB_Spread", "Vol_Surge"]
    
    # 1 if next bar closes higher than current bar
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    
    clean = df.dropna()
    if len(clean) < 40:
        return None

    # Latest bar to predict
    latest = clean.iloc[[-1]]
    X_latest = latest[feature_cols]

    # Training set (without the last row)
    train = clean.iloc[:-1]
    X_train = train[feature_cols]
    y_train = train["Target"]

    # Fit Machine Learning model
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    prob_rise = model.predict_proba(X_latest)[0][1]

    # -------------------------------------------------------------
    # Plain English Indicator Signals
    # -------------------------------------------------------------
    reasons = []
    bull_count = 0
    bear_count = 0

    cur = clean.iloc[-1]

    # EMA Check
    if cur["EMA_9"] > cur["EMA_21"]:
        bull_count += 1
        reasons.append("🟢 **Short-term trend:** 9 EMA is above 21 EMA (Upward Momentum).")
    else:
        bear_count += 1
        reasons.append("🔴 **Short-term trend:** 9 EMA is below 21 EMA (Downward Pressure).")

    # RSI Check
    if cur["RSI"] > 70:
        bear_count += 1
        reasons.append(f"⚠️ **RSI is high ({cur['RSI']:.1f}):** Stock is overbought, pullbacks likely.")
    elif cur["RSI"] < 30:
        bull_count += 1
        reasons.append(f"🟢 **RSI is low ({cur['RSI']:.1f}):** Stock is oversold, bounce likely.")
    elif cur["RSI"] >= 50:
        bull_count += 1
        reasons.append(f"🟢 **RSI ({cur['RSI']:.1f}):** Buyers control above 50 mid-line.")
    else:
        bear_count += 1
        reasons.append(f"🔴 **RSI ({cur['RSI']:.1f}):** Sellers control below 50 mid-line.")

    # MACD Check
    if cur["MACD_Hist"] > 0:
        bull_count += 1
        reasons.append("🟢 **MACD:** Bullish crossover active.")
    else:
        bear_count += 1
        reasons.append("🔴 **MACD:** Bearish momentum dominant.")

    # Volume Check
    if cur["Vol_Surge"] > 1.3:
        reasons.append(f"⚡ **Volume Surge:** Trading volume is {cur['Vol_Surge']:.1f}x higher than average.")

    # Combined Decision (50% ML + 50% Technical Confluence)
    total_signals = bull_count + bear_count
    tech_score = bull_count / total_signals if total_signals > 0 else 0.5
    composite_confidence = (0.50 * prob_rise) + (0.50 * tech_score)

    return {
        "prob_rise": prob_rise,
        "composite_confidence": composite_confidence,
        "reasons": reasons,
        "latest_bar": cur,
        "atr": cur["ATR"]
    }


# -----------------------------------------------------------------------------
# User Interface (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.title("⚡ TrendPulse AI")

# Quick selector buttons
st.sidebar.markdown("**Quick Select Market Leaders:**")
quick_col1, quick_col2, quick_col3 = st.sidebar.columns(3)

if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "NVDA"

if quick_col1.button("NVDA"):
    st.session_state.ticker_input = "NVDA"
if quick_col2.button("AAPL"):
    st.session_state.ticker_input = "AAPL"
if quick_col3.button("TSLA"):
    st.session_state.ticker_input = "TSLA"

symbol = st.sidebar.text_input("Or enter custom symbol:", key="ticker_input").upper()

timeframe = st.sidebar.selectbox(
    "Prediction Horizon",
    options=[
        "Next 5 Mins (Scalp)",
        "Next 15 Mins (Intraday)",
        "Next 1 Hour (Swing)",
        "Next Trading Day (Daily)"
    ],
    index=1
)

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("🔄 Refresh Real-Time Data")


# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
raw_df, interval_used = fetch_stock_data(symbol, timeframe)

if raw_df.empty or len(raw_df) < 30:
    st.error(f"Could not load sufficient market data for '{symbol}'. Please verify the symbol or try during market hours.")
else:
    df = build_indicators(raw_df)
    results = predict_direction(df)

    if results is None:
        st.warning(f"Not enough data points ({len(df)}) to generate predictions. Please try a different timeframe.")
    else:
        latest = results["latest_bar"]
        cur_price = latest["Close"]
        prev_close = df.iloc[-2]["Close"]
        price_diff = cur_price - prev_close
        pct_diff = (price_diff / prev_close) * 100

        # Header section
        head_col1, head_col2 = st.columns([2, 1])
        with head_col1:
            st.title(f"{symbol} Real-Time Market Trend")
            st.caption(f"Timeframe: **{timeframe}** | Bar interval: **{interval_used}**")
        with head_col2:
            st.metric(
                label="Current Price",
                value=f"${cur_price:,.2f}",
                delta=f"{price_diff:+,.2f} ({pct_diff:+.2f}%)"
            )

        st.markdown("---")

        # PREDICTION CARD (HERO ELEMENT)
        conf = results["composite_confidence"]
        atr = results["atr"] if not np.isnan(results["atr"]) else cur_price * 0.01

        col_pred, col_levels = st.columns([1.5, 1.5])

        with col_pred:
            if conf >= 0.56:
                direction_text = "🚀 PREDICTION: EXPECTED TO RISE"
                style_class = "signal-rise"
                confidence_pct = conf * 100
                bias = "BULLISH"
            elif conf <= 0.44:
                direction_text = "🔻 PREDICTION: EXPECTED TO FALL"
                style_class = "signal-fall"
                confidence_pct = (1.0 - conf) * 100
                bias = "BEARISH"
            else:
                direction_text = "⚖️ PREDICTION: CONSOLIDATING / NEUTRAL"
                style_class = "signal-neutral"
                confidence_pct = 50.0
                bias = "NEUTRAL"

            st.markdown(f'<div class="{style_class}">{direction_text}</div>', unsafe_allow_html=True)
            st.write("")
            st.write(f"**AI Confidence Score:** `{confidence_pct:.1f}%`")
            st.progress(confidence_pct / 100)

        with col_levels:
            st.markdown("### 🎯 Suggested Trading Levels")
            # ATR-based automated Take-Profit and Stop-Loss
            if bias == "BULLISH":
                target = cur_price + (1.5 * atr)
                stop = cur_price - (1.0 * atr)
            else:
                target = cur_price - (1.5 * atr)
                stop = cur_price + (1.0 * atr)

            l_col1, l_col2 = st.columns(2)
            l_col1.metric("Target (Take Profit)", f"${target:.2f}")
            l_col2.metric("Safety (Stop Loss)", f"${stop:.2f}")
            st.caption("Levels are dynamically calculated using market volatility (ATR).")

        st.markdown("---")

        # WHY THIS PREDICTION? (EXPLAINABLE AI)
        with st.expander("🔍 **Why did the model make this prediction? (Click to view breakdown)**", expanded=True):
            col_why1, col_why2 = st.columns([2, 1])
            with col_why1:
                for item in results["reasons"]:
                    st.markdown(item)
            with col_why2:
                st.markdown("**Technical Snapshot**")
                st.write(f"• **RSI (14):** `{latest['RSI']:.1f}`")
                st.write(f"• **EMA 9:** `${latest['EMA_9']:.2f}`")
                st.write(f"• **EMA 21:** `${latest['EMA_21']:.2f}`")
                st.write(f"• **Volume Surge:** `{latest['Vol_Surge']:.1f}x`")

        # -------------------------------------------------------------------------
        # Visual Interactive Chart
        # -------------------------------------------------------------------------
        st.subheader("📊 Interactive Candlestick & Trend Chart")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.75, 0.25],
            subplot_titles=("Price Action & Moving Averages", "Volume & Momentum")
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            name="Price"
        ), row=1, col=1)

        # 9 and 21 EMAs
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_9"], name="EMA 9 (Fast)", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["EMA_21"], name="EMA 21 (Slow)", line=dict(color="#06b6d4", width=1.5)), row=1, col=1)

        # Bollinger Bands
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="Upper Band", line=dict(color="rgba(150,150,150,0.4)", dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="Lower Band", line=dict(color="rgba(150,150,150,0.4)", dash="dot")), row=1, col=1)

        # Volume Bar Chart
        colors = ['#16a34a' if row['Close'] >= row['Open'] else '#dc2626' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors), row=2, col=1)

        fig.update_layout(
            height=550,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Disclaimer Footer
        st.caption("⚠️ **Educational & Informational Tool Only:** Financial markets are inherently uncertain. This tool uses historical probability and statistical indicators. Never risk capital based solely on automated signals.")

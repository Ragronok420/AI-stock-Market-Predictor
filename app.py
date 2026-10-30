import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import re
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

# A calmer visual hierarchy for the dashboard and its controls.
st.markdown("""
<style>
    :root {
        --tp-bg: #0b1020;
        --tp-border: rgba(148, 163, 184, .16);
        --tp-muted: #91a0b8;
        --tp-accent: #67e8d2;
    }
    .stApp {
        background: radial-gradient(ellipse at 15% 0%, rgba(20, 72, 86, .24), transparent 38%), var(--tp-bg);
    }
    [data-testid="stSidebar"] {
        background: #0e1627;
        border-right: 1px solid var(--tp-border);
    }
    .block-container { padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1440px; }
    h1, h2, h3 { letter-spacing: -.035em; }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(20, 31, 51, .9), rgba(15, 24, 40, .9));
        border: 1px solid var(--tp-border);
        border-radius: 14px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: var(--tp-muted); }
    [data-testid="stMetricValue"] { font-weight: 650; }
    .eyebrow {
        color: var(--tp-accent);
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .signal-rise, .signal-fall, .signal-neutral {
        color: #f8fafc;
        padding: 22px 24px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,.12);
        text-align: left;
        font-weight: 700;
        font-size: clamp(1.1rem, 2vw, 1.5rem);
        line-height: 1.35;
    }
    .signal-rise {
        background: linear-gradient(120deg, rgba(5, 82, 66, .96), rgba(15, 118, 89, .74));
        box-shadow: 0 12px 32px rgba(16, 185, 129, .12);
    }
    .signal-fall {
        background: linear-gradient(120deg, rgba(127, 29, 46, .96), rgba(190, 45, 65, .74));
        box-shadow: 0 12px 32px rgba(239, 68, 68, .12);
    }
    .signal-neutral { background: linear-gradient(120deg, rgba(115, 76, 20, .96), rgba(180, 113, 25, .74)); }
    .signal-label {
        display: block;
        font-size: .72rem;
        letter-spacing: .12em;
        opacity: .78;
        margin-bottom: .4rem;
    }
    .signal-rise small, .signal-fall small, .signal-neutral small {
        display: block;
        font-size: .86rem;
        font-weight: 450;
        opacity: .88;
        margin-top: .45rem;
    }
    .section-intro {
        color: var(--tp-muted);
        margin-top: -.45rem;
        margin-bottom: 1.15rem;
        font-size: .93rem;
    }
    .sidebar-note { color: var(--tp-muted); font-size: .82rem; line-height: 1.5; }
    .brand-mark {
        color: var(--tp-accent);
        font-size: .77rem;
        font-weight: 750;
        letter-spacing: .15em;
        text-transform: uppercase;
    }
    .stButton > button { border-radius: 10px; min-height: 2.7rem; }
    div[data-testid="stExpander"] {
        border: 1px solid var(--tp-border);
        border-radius: 14px;
        background: rgba(17, 26, 45, .58);
    }
    @media (max-width: 700px) {
        .block-container { padding-top: 1.25rem; }
        [data-testid="stMetric"] { padding: 12px; }
        .signal-rise, .signal-fall, .signal-neutral { padding: 18px; }
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Processing & Technical Engine
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30, max_entries=32, show_spinner=False)
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
    df = ticker.history(period=period, interval=interval, timeout=10)
    return df, interval


def build_demo_data(symbol: str, timeframe: str):
    """Build synthetic bars for a clearly labeled, offline UI preview."""
    interval_map = {
        "Next 5 Mins (Scalp)": "1m",
        "Next 15 Mins (Intraday)": "5m",
        "Next 1 Hour (Swing)": "15m",
        "Next Trading Day (Daily)": "1d",
    }
    interval = interval_map[timeframe]
    seed = sum((index + 1) * ord(char) for index, char in enumerate(f"{symbol}:{timeframe}"))
    rng = np.random.default_rng(seed)
    count = 180
    if interval == "1d":
        index = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=count)
    else:
        minutes = int(interval[:-1])
        index = pd.date_range(
            end=pd.Timestamp.now(tz="UTC").floor("min"),
            periods=count,
            freq=f"{minutes}min",
        )

    close = 150 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, count)))
    open_price = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0015, count))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0.0005, 0.004, count))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0.0005, 0.004, count))
    volume = rng.integers(100_000, 2_000_000, count)
    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    ), interval


def is_valid_symbol(symbol: str) -> bool:
    """Allow common Yahoo Finance symbols while rejecting markup and URL syntax."""
    return (
        isinstance(symbol, str)
        and len(symbol) <= 32
        and re.fullmatch(r"\^?[A-Z0-9][A-Z0-9.-]*(?:=[A-Z])?", symbol) is not None
    )


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
st.sidebar.markdown('<div class="brand-mark">⚡ TrendPulse</div>', unsafe_allow_html=True)
st.sidebar.page_link("pages/Trade.py", label="Open Alpaca trade desk")
st.sidebar.markdown("### Market setup")
st.sidebar.markdown(
    '<div class="sidebar-note">Choose a ticker and a forecast horizon to build your market snapshot.</div>',
    unsafe_allow_html=True,
)
st.sidebar.write("")

TIMEFRAME_OPTIONS = [
    "Next 5 Mins (Scalp)",
    "Next 15 Mins (Intraday)",
    "Next 1 Hour (Swing)",
    "Next Trading Day (Daily)",
]
st.session_state.setdefault("ticker_input", "NVDA")
st.session_state.setdefault("timeframe_input", TIMEFRAME_OPTIONS[1])
st.session_state.setdefault("active_symbol", "NVDA")
st.session_state.setdefault("active_timeframe", TIMEFRAME_OPTIONS[1])
if "use_demo_data" not in st.session_state:
    st.session_state.use_demo_data = st.query_params.get("demo") == "1"

st.sidebar.markdown("**Popular tickers**")
quick_col1, quick_col2, quick_col3 = st.sidebar.columns(3)

if quick_col1.button("NVDA", width="stretch"):
    st.session_state.ticker_input = "NVDA"
    st.session_state.active_symbol = "NVDA"
if quick_col2.button("AAPL", width="stretch"):
    st.session_state.ticker_input = "AAPL"
    st.session_state.active_symbol = "AAPL"
if quick_col3.button("TSLA", width="stretch"):
    st.session_state.ticker_input = "TSLA"
    st.session_state.active_symbol = "TSLA"

st.sidebar.toggle(
    "Use synthetic demo data",
    key="use_demo_data",
    help="Preview the interface with generated prices. This does not use live market data.",
)

with st.sidebar.form("market_setup"):
    st.text_input(
        "Ticker symbol",
        key="ticker_input",
        max_chars=32,
        placeholder="e.g. NVDA or BTC-USD",
        help="Enter a stock, ETF, or a Yahoo Finance symbol such as BTC-USD.",
    )
    st.selectbox(
        "Forecast horizon",
        options=TIMEFRAME_OPTIONS,
        key="timeframe_input",
        help="The forecast uses the next available bar for the selected horizon.",
    )
    apply_setup = st.form_submit_button("Update market view", width="stretch")

if apply_setup:
    requested_symbol = st.session_state.ticker_input.strip().upper()
    if is_valid_symbol(requested_symbol):
        st.session_state.active_symbol = requested_symbol
        st.session_state.active_timeframe = st.session_state.timeframe_input
    else:
        st.sidebar.error("Enter a valid ticker using letters, numbers, dots, hyphens, or a leading ^ / ending =X.")
        st.stop()

symbol = st.session_state.active_symbol
timeframe = st.session_state.active_timeframe

st.sidebar.markdown("---")
refresh_btn = st.sidebar.button("↻ Refresh market data", type="primary", width="stretch")
st.sidebar.markdown(
    '<div class="sidebar-note">Quotes are provided by Yahoo Finance and may be delayed. Refresh to request the latest available bars.</div>',
    unsafe_allow_html=True,
)
if refresh_btn:
    fetch_stock_data.clear()


# -----------------------------------------------------------------------------
# Main Application Dashboard
# -----------------------------------------------------------------------------
if not symbol:
    st.info("Enter a ticker symbol in the sidebar to start your market snapshot.")
    st.stop()
if not is_valid_symbol(symbol):
    st.error("The ticker format is invalid. Use letters, numbers, dots, hyphens, or a leading ^ / ending =X.")
    st.stop()

try:
    if st.session_state.use_demo_data:
        st.info("DEMO PREVIEW · Synthetic prices for interface testing; not live market data.")
        with st.spinner("Building a sample market snapshot…"):
            raw_df, interval_used = build_demo_data(symbol, timeframe)
    else:
        with st.spinner(f"Loading market data for {symbol}…"):
            raw_df, interval_used = fetch_stock_data(symbol, timeframe)
except Exception as exc:
    st.error(f"We couldn't load market data for **{symbol}**.")
    st.caption("Check the ticker spelling and your internet connection, then try refreshing the data.")
    with st.expander("Data provider details"):
        st.code(str(exc))
    st.stop()

if raw_df.empty or len(raw_df) < 30:
    st.warning(f"There isn't enough recent market data for **{symbol}** to create a reliable snapshot.")
    st.caption("Check the ticker symbol, try a longer forecast horizon, or try again during market hours.")
else:
    df = build_indicators(raw_df)
    results = predict_direction(df)

    if results is None:
        st.warning(f"Only {len(df)} usable bars are available for **{symbol}** right now.")
        st.caption("Try a longer forecast horizon or check again when more market data is available.")
    else:
        latest = results["latest_bar"]
        cur_price = latest["Close"]
        prev_close = df.iloc[-2]["Close"]
        price_diff = cur_price - prev_close
        pct_diff = (price_diff / prev_close) * 100

        # Market overview
        head_col1, head_col2 = st.columns([1.7, 1])
        with head_col1:
            st.markdown('<div class="eyebrow">Market overview</div>', unsafe_allow_html=True)
            st.title(symbol)
            st.markdown(
                f'<div class="section-intro">{timeframe} forecast · {interval_used} bars · '
                f'Latest bar: {df.index[-1].strftime("%b %d, %Y · %I:%M %p %Z")}</div>',
                unsafe_allow_html=True,
            )
        with head_col2:
            st.metric(
                label="Latest bar close",
                value=f"${cur_price:,.2f}",
                delta=f"{price_diff:+,.2f} ({pct_diff:+.2f}%) vs prior bar",
                help="Change from the previous available bar, not the previous day's close.",
            )

        # PREDICTION CARD (HERO ELEMENT)
        conf = results["composite_confidence"]
        atr = results["atr"] if not np.isnan(results["atr"]) else cur_price * 0.01

        col_pred, col_levels = st.columns([1.15, 1])

        with col_pred:
            if conf >= 0.56:
                direction_text = "Expected to rise"
                style_class = "signal-rise"
                confidence_pct = conf * 100
                bias = "BULLISH"
            elif conf <= 0.44:
                direction_text = "Expected to fall"
                style_class = "signal-fall"
                confidence_pct = (1.0 - conf) * 100
                bias = "BEARISH"
            else:
                direction_text = "Neutral outlook"
                style_class = "signal-neutral"
                confidence_pct = 50.0
                bias = "NEUTRAL"

            st.markdown(
                f'<div class="{style_class}"><span class="signal-label">NEXT BAR MODEL OUTLOOK</span>'
                f'{direction_text}<small>Statistical estimate based on recent price and volume patterns.</small></div>',
                unsafe_allow_html=True,
            )
            st.metric(
                "Model confidence",
                f"{confidence_pct:.1f}%",
                help="A blend of the model probability and technical indicator agreement. It is not a guarantee.",
            )
            st.progress(confidence_pct / 100)

        with col_levels:
            st.markdown("### ATR-based reference levels")
            st.markdown(
                '<div class="section-intro">Volatility estimates around the latest bar close.</div>',
                unsafe_allow_html=True,
            )
            if bias == "BULLISH":
                target = cur_price + (1.5 * atr)
                stop = cur_price - (1.0 * atr)
            elif bias == "BEARISH":
                target = cur_price - (1.5 * atr)
                stop = cur_price + (1.0 * atr)
            if bias == "NEUTRAL":
                st.info("The outlook is neutral, so no directional target or stop estimate is shown.")
            else:
                l_col1, l_col2 = st.columns(2)
                l_col1.metric(
                    "Target estimate",
                    f"${target:,.2f}",
                    delta=f"{target - cur_price:+,.2f}",
                    help="A volatility-based reference at 1.5 × ATR in the model's indicated direction.",
                )
                l_col2.metric(
                    "Stop estimate",
                    f"${stop:,.2f}",
                    delta=f"{stop - cur_price:+,.2f}",
                    delta_color="inverse",
                    help="A volatility-based reference at 1 × ATR opposite the indicated direction.",
                )
            st.caption("Illustrative estimates based on ATR; they are not trading recommendations.")

        st.markdown("#### Technical snapshot")
        snap1, snap2, snap3, snap4 = st.columns(4)
        snap1.metric("RSI (14)", f"{latest['RSI']:.1f}", help="Momentum indicator; readings above 70 or below 30 are often considered extreme.")
        snap2.metric("Short-term trend", "Upward" if latest["EMA_9"] > latest["EMA_21"] else "Downward", help="Compares the 9-period EMA with the 21-period EMA.")
        snap3.metric("MACD momentum", "Positive" if latest["MACD_Hist"] > 0 else "Negative")
        snap4.metric("Volume vs average", f"{latest['Vol_Surge']:.1f}×", help="Latest volume divided by its 20-bar average.")

        # WHY THIS PREDICTION? (EXPLAINABLE AI)
        with st.expander("How was this outlook formed?", expanded=False):
            col_why1, col_why2 = st.columns([2, 1])
            with col_why1:
                for item in results["reasons"]:
                    st.markdown(item)
            with col_why2:
                st.markdown("**Model inputs**")
                st.write(f"EMA 9: `${latest['EMA_9']:.2f}`")
                st.write(f"EMA 21: `${latest['EMA_21']:.2f}`")
                st.write(f"MACD histogram: `{latest['MACD_Hist']:.4f}`")

        # -------------------------------------------------------------------------
        # Interactive chart
        # -------------------------------------------------------------------------
        st.markdown("### Price chart")
        st.markdown(
            '<div class="section-intro">Explore recent candles, moving averages, Bollinger Bands, and volume. '
            'Hover over the chart for exact values; click legend items to show or hide indicators.</div>',
            unsafe_allow_html=True,
        )

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
            name="Price",
            increasing_line_color="#34d399",
            decreasing_line_color="#fb7185",
            increasing_fillcolor="#34d399",
            decreasing_fillcolor="#fb7185",
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
            height=570,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=45, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(11,16,32,.35)",
            font=dict(color="#dbe4f0", family="sans-serif"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            hovermode="x unified",
        )
        fig.update_xaxes(showgrid=False, zeroline=False)
        fig.update_yaxes(gridcolor="rgba(148,163,184,.12)", zeroline=False)
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False, "scrollZoom": True})

        # Disclaimer Footer
        st.caption("For education and research only. Market predictions are uncertain and are not financial advice. Review your own risk before making investment decisions.")

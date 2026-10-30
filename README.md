# ⚡ TrendPulse | AI Stock Trend Predictor

TrendPulse is a real-time market trend prediction and technical analysis dashboard built with Python, Streamlit, Plotly, and Scikit-Learn. It combines technical indicators (EMA, RSI, MACD, Bollinger Bands, ATR) with dynamic Machine Learning (Random Forest) to forecast next-bar momentum for US equities and crypto.

---

## ✨ Features

- **Real-Time Data Feed:** Fetches market data using `yfinance` across multiple horizons (5-min Scalp, 15-min Intraday, 1-hour Swing, and Daily).
- **Hybrid AI Engine:** Blends Random Forest machine learning classification with a technical confluence rule engine.
- **Explainable AI (XAI):** Plain-English breakdown explaining the exact reasoning behind every bullish, bearish, or neutral rating.
- **Dynamic Trade Planning:** Automated volatility-based Take-Profit and Stop-Loss levels computed via Average True Range (ATR).
- **Interactive Technical Charts:** Dual-pane Plotly candlestick charts with EMAs, Bollinger Bands, and volume momentum indicators.
- **Modern Dark UI:** Responsive card layout tailored for traders and market analysts.
- **Alpaca Trading Desk:** Review and submit manual stock/ETF market or limit orders, inspect account balances and positions, and review open orders. TrendPulse predictions never submit orders automatically.

---

## 🛠️ Tech Stack

- **Frontend / Dashboard:** [Streamlit](https://streamlit.io/)
- **Data Source:** [yfinance](https://github.com/ranaroussi/yfinance)
- **Machine Learning:** [scikit-learn](https://scikit-learn.org/) (RandomForestClassifier)
- **Technical Indicators:** [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/)
- **Visualizations:** [Plotly](https://plotly.com/python/)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/trendpulse.git
cd trendpulse
```

### 2. Set Up a Virtual Environment
```bash
# Using Python 3.11+
python -m venv .venv

# Activate on Windows:
.venv\Scripts\activate

# Activate on macOS / Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Alpaca

Open the local `.streamlit/secrets.toml` file and enter your paper API credentials under `[alpaca.paper]`. If it is missing, copy `.streamlit/secrets.toml.example` to create it. This credentials file is ignored by Git; the example template is safe to track.

To enable live trading, add your live API credentials in the `[alpaca.live]` section, then explicitly set `paper = false` and `enable_live_trading = true` in `[alpaca]`. The Trade page also requires an order preview, a confirmation checkbox, and typing `LIVE` before it will submit a live order. Never paste API keys into source code or commit them.

### 5. Run the Streamlit App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

Use the **Trade** page in Streamlit's page navigation to view your Alpaca account and place manual orders. Paper and live accounts use separate credentials. This integration is for a personal, locally run app; do not expose personal API keys through a public deployment.

---

## 📊 Technical Indicators Overview

| Indicator | Default Parameter | Role |
| :--- | :--- | :--- |
| **EMA 9 & EMA 21** | 9-period & 21-period span | Directional trend & moving average cross |
| **RSI** | 14-period | Overbought (>70) and oversold (<30) momentum |
| **MACD** | 12, 26, 9 span | Bullish / Bearish histogram momentum divergence |
| **Bollinger Bands** | 20-period SMA ± 2 STD | Volatility range and channel breakouts |
| **ATR** | 14-period | Volatility measurement for risk management |

---

## ⚠️ Disclaimer

*TrendPulse is intended for informational and educational purposes only. Financial markets carry significant risk. Automated predictions and statistical indicators should never be treated as financial advice or sole signals for executing trades. Always practice strict risk management.*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

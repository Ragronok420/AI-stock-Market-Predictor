"""Offline UI smoke tests using deterministic market and Alpaca fixtures."""

import sys
import types
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


def synthetic_history(*_args, **_kwargs):
    rng = np.random.default_rng(17)
    count = 180
    close = 120 + np.cumsum(rng.normal(0.02, 0.6, count))
    open_price = np.r_[close[0], close[:-1]]
    high = np.maximum(open_price, close) + rng.uniform(0.05, 0.7, count)
    low = np.minimum(open_price, close) - rng.uniform(0.05, 0.7, count)
    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(100_000, 1_000_000, count),
        },
        index=pd.date_range("2026-09-01", periods=count, freq="h", tz="UTC"),
    )


class FakeTicker:
    def __init__(self, _symbol):
        pass

    def history(self, **kwargs):
        return synthetic_history(**kwargs)


class FakeRandomForest:
    def __init__(self, **_kwargs):
        pass

    def fit(self, _features, _target):
        return self

    def predict_proba(self, _features):
        return np.array([[0.42, 0.58]])


class FakeOrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class FakeQueryOrderStatus(Enum):
    OPEN = "open"


class FakeTimeInForce(Enum):
    DAY = "day"


class FakeRequest:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTradingClient:
    submitted_orders = 0
    connected_keys = []

    def __init__(self, key, _secret, paper=True):
        self.paper = paper
        type(self).connected_keys.append(key)

    def get_account(self):
        return SimpleNamespace(
            status="ACTIVE",
            equity="100000",
            cash="50000",
            buying_power="50000",
            trading_blocked=False,
            account_blocked=False,
            trade_suspended_by_user=False,
        )

    def get_asset(self, symbol):
        return SimpleNamespace(
            symbol=symbol,
            asset_class="us_equity",
            status="active",
            tradable=True,
            fractionable=True,
        )

    def get_order_by_client_id(self, _client_order_id):
        error = RuntimeError("not found")
        error.status_code = 404
        raise error

    def submit_order(self, order_data):
        type(self).submitted_orders += 1
        return SimpleNamespace(status="accepted", request=order_data)

    def get_all_positions(self):
        return []

    def get_orders(self, **_kwargs):
        return []


def install_fake_alpaca():
    alpaca = types.ModuleType("alpaca")
    alpaca.__path__ = []
    trading = types.ModuleType("alpaca.trading")
    trading.__path__ = []
    client_module = types.ModuleType("alpaca.trading.client")
    client_module.TradingClient = FakeTradingClient
    enums_module = types.ModuleType("alpaca.trading.enums")
    enums_module.OrderSide = FakeOrderSide
    enums_module.QueryOrderStatus = FakeQueryOrderStatus
    enums_module.TimeInForce = FakeTimeInForce
    requests_module = types.ModuleType("alpaca.trading.requests")
    requests_module.GetOrdersRequest = FakeRequest
    requests_module.LimitOrderRequest = FakeRequest
    requests_module.MarketOrderRequest = FakeRequest
    return {
        "alpaca": alpaca,
        "alpaca.trading": trading,
        "alpaca.trading.client": client_module,
        "alpaca.trading.enums": enums_module,
        "alpaca.trading.requests": requests_module,
    }


class TrendPulseUiSmokeTests(unittest.TestCase):
    def test_trade_page_explains_missing_alpaca_package(self):
        missing_alpaca = {
            name: None
            for name in (
                "alpaca",
                "alpaca.trading",
                "alpaca.trading.client",
                "alpaca.trading.enums",
                "alpaca.trading.requests",
            )
        }
        with patch.dict(sys.modules, missing_alpaca):
            app = AppTest.from_file(str(ROOT / "pages" / "Trade.py")).run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any("needs the Alpaca Python package" in item.value for item in app.warning)
        )

    def test_dashboard_ticker_buttons_update_the_market_view(self):
        fake_yfinance = types.ModuleType("yfinance")
        fake_yfinance.Ticker = FakeTicker
        fake_sklearn = types.ModuleType("sklearn")
        fake_sklearn.__path__ = []
        fake_ensemble = types.ModuleType("sklearn.ensemble")
        fake_ensemble.RandomForestClassifier = FakeRandomForest
        with patch.dict(
            sys.modules,
            {
                "yfinance": fake_yfinance,
                "sklearn": fake_sklearn,
                "sklearn.ensemble": fake_ensemble,
            },
        ):
            app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any(item.value == "NVDA" for item in app.title))
            self.assertTrue(any("Price chart" in item.value for item in app.markdown))

            aapl_button = next(button for button in app.sidebar.button if button.label == "AAPL")
            aapl_button.click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any(item.value == "AAPL" for item in app.title))

    def test_paper_order_requires_review_and_explicit_confirmation(self):
        FakeTradingClient.submitted_orders = 0
        FakeTradingClient.connected_keys = []
        with patch.dict(
            sys.modules,
            install_fake_alpaca(),
        ):
            app = AppTest.from_file(str(ROOT / "pages" / "Trade.py")).run()
            self.assertFalse(app.exception)
            self.assertTrue(any(item.value == "Alpaca trade desk" for item in app.title))

            api_key_input = next(item for item in app.text_input if item.label == "Paper API key")
            secret_input = next(item for item in app.text_input if item.label == "Paper API secret")
            api_key_input.set_value("first-session-paper-key")
            secret_input.set_value("first-session-paper-secret")
            connect = next(item for item in app.button if item.label == "Connect paper account")
            connect.click().run()
            self.assertFalse(app.exception)
            self.assertIn("first-session-paper-key", FakeTradingClient.connected_keys)

            second_session = AppTest.from_file(str(ROOT / "pages" / "Trade.py")).run()
            second_session.text_input[0].set_value("second-session-paper-key")
            second_session.text_input[1].set_value("second-session-paper-secret")
            next(
                item for item in second_session.button if item.label == "Connect paper account"
            ).click().run()
            self.assertFalse(second_session.exception)
            self.assertEqual(
                FakeTradingClient.connected_keys[-2:],
                ["first-session-paper-key", "second-session-paper-key"],
            )

            limit_price = next(
                item for item in app.text_input if item.label.startswith("Limit price")
            )
            limit_price.set_value("150.00")
            review = next(item for item in app.button if item.label == "Review order")
            review.click().run()
            self.assertFalse(app.exception)
            self.assertEqual(FakeTradingClient.submitted_orders, 0)
            self.assertTrue(
                any("Review before submitting" in item.value for item in app.markdown),
                msg=f"Review missing. errors={[item.value for item in app.error]}, "
                    f"markdown={[item.value for item in app.markdown]}, "
                    f"buttons={[item.label for item in app.button]}",
            )

            confirm = next(item for item in app.checkbox if item.label.startswith("I confirm this exact buy order"))
            confirm.check().run()
            submit = next(item for item in app.button if item.label == "Place paper order")
            self.assertFalse(submit.disabled)
            submit.click().run()

        self.assertFalse(app.exception)
        self.assertEqual(FakeTradingClient.submitted_orders, 1)
        self.assertTrue(any("Alpaca accepted the buy order" in item.value for item in app.success))


if __name__ == "__main__":
    unittest.main()

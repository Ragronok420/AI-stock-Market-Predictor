import os
import re
import uuid
from decimal import Decimal, InvalidOperation

import pandas as pd
import streamlit as st

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, LimitOrderRequest, MarketOrderRequest
    ALPACA_IMPORT_ERROR = None
except ImportError as exc:
    TradingClient = None
    ALPACA_IMPORT_ERROR = exc

st.set_page_config(page_title="TrendPulse | Alpaca Trade Desk", page_icon="⚡", layout="wide")
st.markdown("""
<style>
    .stApp { background: radial-gradient(ellipse at 15% 0%, rgba(20,72,86,.2), transparent 38%), #0b1020; }
    .block-container { max-width: 1380px; padding-top: 2rem; }
    [data-testid="stMetric"] { background: rgba(20,31,51,.8); border: 1px solid rgba(148,163,184,.16); border-radius: 14px; padding: 14px 18px; }
    .trade-eyebrow { color: #67e8d2; font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

if ALPACA_IMPORT_ERROR is not None:
    st.warning("The Alpaca trade desk needs the Alpaca Python package before it can connect.")
    st.code("python -m pip install -r requirements.txt", language="powershell")
    st.caption("You can still use the market dashboard while the broker package is unavailable.")
    st.stop()


def as_bool(value, default):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def alpaca_settings():
    try:
        root = st.secrets.get("alpaca", {})
    except Exception:
        root = {}
    if not hasattr(root, "get"):
        root = {}

    paper = as_bool(os.getenv("ALPACA_PAPER", root.get("paper", True)), True)
    live_enabled = as_bool(
        os.getenv("ALPACA_ENABLE_LIVE_TRADING", root.get("enable_live_trading", False)),
        False,
    )
    mode = "paper" if paper else "live"
    credentials = st.session_state.get(f"alpaca_{mode}_credentials", {})
    if not isinstance(credentials, dict):
        credentials = {}
    api_key = credentials.get("api_key")
    secret_key = credentials.get("secret_key")
    return paper, live_enabled, api_key, secret_key


def disconnect_alpaca_session(mode):
    st.session_state.pop(f"alpaca_{mode}_credentials", None)
    version_key = f"alpaca_{mode}_form_version"
    version = st.session_state.get(version_key, 0)
    st.session_state.pop(f"alpaca_{mode}_api_key_{version}", None)
    st.session_state.pop(f"alpaca_{mode}_secret_key_{version}", None)
    st.session_state[version_key] = version + 1


def enum_text(value):
    value = getattr(value, "value", value)
    return str(value).split(".")[-1]


def money(value):
    try:
        return f"${Decimal(str(value)):,.2f}"
    except (InvalidOperation, TypeError, ValueError):
        return "—"


def account_can_trade(account):
    status = enum_text(account.status).upper()
    blocked = any(
        bool(getattr(account, flag, False))
        for flag in ("trading_blocked", "account_blocked", "trade_suspended_by_user")
    )
    return status in {"ACTIVE", "APPROVED"} and not blocked


def valid_equity_symbol(symbol):
    return re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", symbol) is not None


def review_order(client, paper, symbol, side, order_type, quantity_value, limit_price_value):
    symbol = symbol.strip().upper()
    if not valid_equity_symbol(symbol):
        raise ValueError("Enter a valid US stock or ETF symbol, such as AAPL or BRK.B.")

    try:
        quantity = Decimal(str(quantity_value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("Enter a valid share quantity.") from None
    if not quantity.is_finite() or quantity <= 0:
        raise ValueError("Share quantity must be greater than zero.")
    if quantity.as_tuple().exponent < -9:
        raise ValueError("Share quantity supports up to 9 decimal places.")

    limit_price = None
    if order_type == "Limit":
        try:
            limit_price = Decimal(str(limit_price_value).strip())
        except (InvalidOperation, TypeError, ValueError):
            raise ValueError("Enter a valid limit price.") from None
        if not limit_price.is_finite() or limit_price <= 0:
            raise ValueError("Limit price must be greater than zero.")
        minimum_increment = Decimal("0.01") if limit_price >= 1 else Decimal("0.0001")
        if limit_price % minimum_increment != 0:
            raise ValueError("Use up to 2 decimal places for prices of $1 or more, or 4 below $1.")
        if quantity != quantity.to_integral_value():
            raise ValueError("Limit orders require whole shares in this trade ticket.")

    asset = client.get_asset(symbol)
    asset_class = enum_text(asset.asset_class).lower()
    asset_status = enum_text(asset.status).lower()
    if asset_class != "us_equity":
        raise ValueError("This ticket supports US stocks and ETFs only.")
    if asset_status != "active" or not bool(asset.tradable):
        raise ValueError("Alpaca reports that this asset is not currently tradable.")
    if order_type == "Market" and quantity != quantity.to_integral_value() and not bool(asset.fractionable):
        raise ValueError("This asset does not support fractional market orders.")

    if side == "Sell":
        try:
            position = client.get_open_position(symbol)
        except Exception as exc:
            if getattr(exc, "status_code", None) == 404:
                raise ValueError("No open long position was found. Short selling is disabled in this ticket.") from None
            raise
        held_quantity = Decimal(str(position.qty))
        if held_quantity <= 0 or quantity > held_quantity:
            raise ValueError(f"Sell quantity exceeds the long position available ({held_quantity} shares).")

    return {
        "symbol": symbol,
        "side": side,
        "order_type": order_type,
        "quantity": str(quantity.normalize()),
        "limit_price": str(limit_price.normalize()) if limit_price is not None else None,
        "paper": paper,
        "client_order_id": f"trendpulse-{uuid.uuid4().hex}",
    }


def submit_order(client, order):
    # Reuse the same client ID to avoid duplicate orders after an uncertain response.
    try:
        existing = client.get_order_by_client_id(order["client_order_id"])
    except Exception as exc:
        if getattr(exc, "status_code", None) == 404:
            existing = None
        else:
            raise RuntimeError("Could not check for an existing order; nothing was submitted.") from None
    if existing is not None:
        return existing, True

    common = {
        "symbol": order["symbol"],
        "qty": float(Decimal(order["quantity"])),
        "side": OrderSide.BUY if order["side"] == "Buy" else OrderSide.SELL,
        "time_in_force": TimeInForce.DAY,
        "client_order_id": order["client_order_id"],
    }
    if order["order_type"] == "Market":
        request = MarketOrderRequest(**common)
    else:
        request = LimitOrderRequest(**common, limit_price=float(Decimal(order["limit_price"])))
    return client.submit_order(order_data=request), False


st.markdown('<div class="trade-eyebrow">TrendPulse · Broker connection</div>', unsafe_allow_html=True)
st.title("Alpaca trade desk")
st.caption("Manual US stock and ETF orders. TrendPulse signals never place trades automatically.")

paper, live_enabled, api_key, secret_key = alpaca_settings()
if not paper and not live_enabled:
    st.error("Live order entry is disabled.")
    st.caption("Set `paper = false` and `enable_live_trading = true` under `[alpaca]` in your local secrets file to enable it.")
    st.stop()

mode = "paper" if paper else "live"
mode_title = "Paper" if paper else "Live"
connected_for_session = bool(api_key and secret_key)
with st.expander(
    f"{mode_title} account connection" if not connected_for_session else f"Manage {mode_title.lower()} connection",
    expanded=not connected_for_session,
):
    st.caption(
        f"Enter your own Alpaca {mode_title.lower()} API key pair. The fields are masked, and credentials stay in this "
        "Streamlit browser session's server-side memory; they are not written to disk or shared with other sessions."
    )
    form_version_key = f"alpaca_{mode}_form_version"
    form_version = st.session_state.get(form_version_key, 0)
    with st.form(f"alpaca_{mode}_credentials_form_{form_version}"):
        user_api_key = st.text_input(
            f"{mode_title} API key",
            type="password",
            key=f"alpaca_{mode}_api_key_{form_version}",
            placeholder="Paste your personal API key",
        )
        user_secret_key = st.text_input(
            f"{mode_title} API secret",
            type="password",
            key=f"alpaca_{mode}_secret_key_{form_version}",
            placeholder="Paste your personal API secret",
        )
        connect_clicked = st.form_submit_button(
            f"Connect {mode_title.lower()} account",
            type="primary",
            width="stretch",
        )

    if connect_clicked:
        if not user_api_key.strip() or not user_secret_key.strip():
            st.error("Enter both your API key and API secret to connect.")
        else:
            st.session_state[f"alpaca_{mode}_credentials"] = {
                "api_key": user_api_key.strip(),
                "secret_key": user_secret_key.strip(),
            }
            st.session_state[form_version_key] = form_version + 1
            st.rerun()

    if connected_for_session:
        st.caption("Connected credentials belong to this browser session.")
        st.button(
            "Disconnect this session",
            key=f"disconnect_alpaca_{mode}",
            on_click=disconnect_alpaca_session,
            args=(mode,),
        )

if not api_key or not secret_key:
    st.info(f"Connect your own Alpaca {mode_title.lower()} account above to view balances and the order ticket.")
    st.stop()

client = TradingClient(api_key, secret_key, paper=paper)
try:
    account = client.get_account()
except Exception as exc:
    st.error("Alpaca could not authenticate this connection. Check that the selected mode and its API keys match.")
    st.caption(f"Provider error type: {type(exc).__name__}")
    st.stop()

account_status = enum_text(account.status).upper()
can_trade = account_can_trade(account)
if paper:
    st.info("PAPER MODE · Orders use Alpaca's simulated account and do not use real funds.")
else:
    st.error("LIVE MODE · Orders submitted here use real funds in your Alpaca brokerage account.")
if not can_trade:
    st.warning(f"Trading is disabled because the Alpaca account is restricted or not active (status: {account_status}).")

acct1, acct2, acct3, acct4 = st.columns(4)
acct1.metric("Account status", account_status)
acct2.metric("Equity", money(account.equity))
acct3.metric("Cash", money(account.cash))
acct4.metric("Buying power", money(account.buying_power), help="Alpaca's available buying power; open orders may already reserve part of it.")

st.markdown("### Order ticket")
st.caption("Orders are DAY duration. Eligible market orders can use fractional quantities; limit orders use whole shares here and may not fill. Sell orders cannot open short positions.")

default_symbol = str(st.session_state.get("active_symbol", "NVDA")).upper()
with st.form("alpaca_order_ticket"):
    order_symbol = st.text_input("US stock or ETF symbol", value=default_symbol, max_chars=15, placeholder="AAPL")
    side_col, type_col, qty_col = st.columns([1, 1, 1])
    with side_col:
        order_side = st.radio("Side", ["Buy", "Sell"], horizontal=True)
    with type_col:
        order_type = st.selectbox("Order type", ["Limit", "Market"], index=0)
    with qty_col:
        quantity_value = st.number_input("Shares", min_value=0.001, value=1.0, step=0.001, format="%.3f")
    limit_price_value = st.text_input("Limit price (USD; used for limit orders)", placeholder="For example: 150.00")
    review_clicked = st.form_submit_button("Review order", type="primary", width="stretch")

if review_clicked:
    st.session_state.pop("alpaca_pending_order", None)
    st.session_state.pop("alpaca_last_order_message", None)
    try:
        st.session_state.alpaca_pending_order = review_order(
            client, paper, order_symbol, order_side, order_type, quantity_value, limit_price_value
        )
    except ValueError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Alpaca could not review this order. Nothing was submitted.")
        st.caption(f"Provider error type: {type(exc).__name__}")

last_order_message = st.session_state.get("alpaca_last_order_message")
if last_order_message:
    st.success(last_order_message)

pending = st.session_state.get("alpaca_pending_order")
if pending:
    if pending["paper"] != paper:
        st.warning("The configured account mode changed after this review. Review the order again before submitting.")
        st.session_state.pop("alpaca_pending_order", None)
        st.stop()

    estimated_value = Decimal(pending["quantity"]) * Decimal(pending["limit_price"]) if pending["limit_price"] else None
    st.markdown("#### Review before submitting")
    review1, review2, review3, review4 = st.columns(4)
    review1.metric("Side", pending["side"])
    review2.metric("Symbol", pending["symbol"])
    review3.metric("Quantity", f"{pending['quantity']} shares")
    review4.metric("Order", pending["order_type"])
    if pending["order_type"] == "Limit":
        st.write(f"Limit price: **${Decimal(pending['limit_price']):,.4f}** · estimated order value: **{money(estimated_value)}** · duration: **DAY**")
    else:
        st.write("A market order has no guaranteed execution price and may fill at a different price from the latest displayed data.")

    if paper:
        confirmation_label = f"I confirm this exact {pending['side'].lower()} order for {pending['quantity']} shares of {pending['symbol']} in PAPER mode."
        live_word = "LIVE"
    else:
        confirmation_label = f"I authorize this exact {pending['side'].lower()} order for {pending['quantity']} shares of {pending['symbol']} using REAL MONEY."
        live_word = st.text_input("Type LIVE to confirm this real-money order", key=f"live_{pending['client_order_id']}").strip()
    confirmed = st.checkbox(confirmation_label, key=f"confirm_{pending['client_order_id']}")
    submit_label = "Place paper order" if paper else "Place live order"
    ready_to_submit = can_trade and confirmed and (paper or live_word == "LIVE")

    if not paper and confirmed and live_word != "LIVE":
        st.caption("Type LIVE exactly to enable the real-money submit button.")
    if st.button(submit_label, type="primary", disabled=not ready_to_submit, width="stretch"):
        try:
            # Refresh account permissions and sellable quantity immediately before submission.
            current_account = client.get_account()
            if not account_can_trade(current_account):
                raise RuntimeError("The Alpaca account is no longer enabled for trading.")
            if pending["side"] == "Sell":
                current_position = client.get_open_position(pending["symbol"])
                if Decimal(pending["quantity"]) > Decimal(str(current_position.qty)):
                    raise RuntimeError("The held position changed; review the order again.")
            with st.spinner("Checking for a duplicate and sending the order to Alpaca…"):
                submitted_order, already_existed = submit_order(client, pending)
            status = enum_text(submitted_order.status)
            if already_existed:
                message = f"Alpaca already has this order (status: {status}); no duplicate was sent."
            else:
                message = f"Alpaca accepted the {pending['side'].lower()} order for {pending['symbol']} ({pending['quantity']} shares). Status: {status}."
            st.session_state.alpaca_last_order_message = message
            st.session_state.pop("alpaca_pending_order", None)
            st.rerun()
        except Exception as exc:
            st.error("Alpaca did not confirm this order. Check Open orders before taking further action.")
            st.caption(f"Provider error type: {type(exc).__name__}. The reviewed client order ID is retained to prevent a duplicate on retry.")

st.markdown("### Portfolio and open orders")
portfolio_tab, orders_tab = st.tabs(["Positions", "Open orders"])

with portfolio_tab:
    try:
        positions = client.get_all_positions()
        if positions:
            position_rows = [
                {
                    "Symbol": position.symbol,
                    "Quantity": position.qty,
                    "Average entry": money(position.avg_entry_price),
                    "Current price": money(position.current_price),
                    "Market value": money(position.market_value),
                    "Unrealized P/L": money(position.unrealized_pl),
                }
                for position in positions
            ]
            st.dataframe(pd.DataFrame(position_rows), hide_index=True, width="stretch")
        else:
            st.caption("No open positions.")
    except Exception as exc:
        st.warning("Positions could not be loaded from Alpaca.")
        st.caption(f"Provider error type: {type(exc).__name__}")

with orders_tab:
    open_orders = []
    try:
        open_orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=20))
        if open_orders:
            order_rows = [
                {
                    "Symbol": order.symbol,
                    "Side": enum_text(order.side).title(),
                    "Type": enum_text(order.type).replace("_", " ").title(),
                    "Quantity": order.qty,
                    "Filled": order.filled_qty,
                    "Limit price": money(order.limit_price) if order.limit_price else "—",
                    "Status": enum_text(order.status).replace("_", " ").title(),
                    "Submitted": order.submitted_at,
                }
                for order in open_orders
            ]
            st.dataframe(pd.DataFrame(order_rows), hide_index=True, width="stretch")
        else:
            st.caption("No open orders.")
    except Exception as exc:
        st.warning("Open orders could not be loaded from Alpaca.")
        st.caption(f"Provider error type: {type(exc).__name__}")

    if open_orders:
        st.markdown("**Manage an open order**")
        for order in open_orders:
            cancel_col1, cancel_col2 = st.columns([4, 1])
            with cancel_col1:
                st.write(
                    f"{enum_text(order.side).title()} {order.qty} {order.symbol} · "
                    f"{enum_text(order.type).replace('_', ' ').title()} · "
                    f"{enum_text(order.status).replace('_', ' ').title()}"
                )
            with cancel_col2:
                if st.button("Review cancel", key=f"review_cancel_{order.id}"):
                    st.session_state[f"confirm_cancel_{order.id}"] = False
                    st.session_state.alpaca_pending_cancel = {
                        "id": str(order.id),
                        "symbol": order.symbol,
                        "side": enum_text(order.side).title(),
                        "quantity": str(order.qty),
                        "paper": paper,
                    }

    pending_cancel = st.session_state.get("alpaca_pending_cancel")
    if pending_cancel:
        if pending_cancel["paper"] != paper:
            st.session_state.pop("alpaca_pending_cancel", None)
            st.warning("The account mode changed. Review the cancellation again.")
        else:
            st.warning(
                f"Cancel {pending_cancel['side']} {pending_cancel['quantity']} "
                f"{pending_cancel['symbol']}? An order may fill before Alpaca processes a cancellation request."
            )
            cancel_confirmed = st.checkbox(
                "I confirm I want to request cancellation of this specific order.",
                key=f"confirm_cancel_{pending_cancel['id']}",
            )
            if st.button("Request cancellation", disabled=not cancel_confirmed, key="submit_cancel_request"):
                try:
                    client.cancel_order_by_id(pending_cancel["id"])
                    st.session_state.alpaca_last_order_message = (
                        f"Cancellation requested for {pending_cancel['symbol']}. Check Open orders for the final status."
                    )
                    st.session_state.pop("alpaca_pending_cancel", None)
                    st.rerun()
                except Exception as exc:
                    st.error("Alpaca could not confirm the cancellation request. Refresh Open orders to check the current status.")
                    st.caption(f"Provider error type: {type(exc).__name__}")

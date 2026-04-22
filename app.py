import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(
    page_title="Stock Insights Dashboard", page_icon="logo.png", layout="wide"
)

col1, col2 = st.columns([1, 14])

with col1:
    st.markdown(
        "<div style='margin-top: 10px; margin-left: 10px;'>", unsafe_allow_html=True
    )
    st.image("logo.png", width=80)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("## Stock Insights Dashboard")


# ********************Helper Functions********************
def get_last(series):
    if series is None or len(series) == 0:
        return None
    return series.iloc[-1]


def pct_change(firstval, lastval):
    if firstval is None or lastval is None:
        return None
    if firstval == 0:
        return None
    return ((lastval - firstval) / abs(firstval)) * 100


def get_mean(series):
    if series is None or len(series) == 0:
        return None
    return series.mean()


@st.cache_data
def resolve_ticker(user_input):
    try:
        search_results = yf.Search(user_input, max_results=1)
        quotes = search_results.quotes
        if quotes and len(quotes) > 0:
            ticker = quotes[0]["symbol"]
            name = quotes[0].get("longname", quotes[0]["symbol"])
            return ticker, name
    except Exception:
        pass

    if len(user_input) <= 5 and user_input.isalpha():
        return user_input.upper(), user_input.upper()

    return user_input.upper(), user_input.upper()


# ********************Data Fetch Functions********************
@st.cache_data
def get_stock_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start, end=end)

        if stock_data.empty:
            return None

        stock_data.reset_index(inplace=True)
        stock_data = stock_data[["Date", "Close"]]
        stock_data.rename(columns={"Close": "Stock Price"}, inplace=True)
        stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.tz_localize(None)

        return stock_data

    except Exception as e:
        st.error(f"Failed to fetch stock data for {ticker}: {e}")
        return None


@st.cache_data
def get_revenue_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.quarterly_financials

        if financials is None or financials.empty:
            return None

        if "Total Revenue" not in financials.index:
            return None

        revenue = financials.loc["Total Revenue"].reset_index()
        revenue.columns = ["Date", "Revenue"]
        revenue["Revenue"] = pd.to_numeric(revenue["Revenue"], errors="coerce")
        revenue["Revenue"] = revenue["Revenue"] / 1_000_000

        revenue["Date"] = pd.to_datetime(revenue["Date"], errors="coerce")
        if revenue["Date"].dt.tz is not None:
            revenue["Date"] = revenue["Date"].dt.tz_localize(None)

        revenue.dropna(subset=["Date", "Revenue"], inplace=True)
        revenue = revenue.sort_values("Date").reset_index(drop=True)
        revenue["Quarter"] = revenue["Date"].dt.to_period("Q").astype(str)

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        filtered = revenue[(revenue["Date"] >= start_dt) & (revenue["Date"] <= end_dt)]

        if filtered.empty:
            return revenue
        return filtered

    except Exception as e:
        st.warning(f"Could not load revenue data for {ticker}: {e}")
        return None


@st.cache_data
def get_netprofit_data(ticker, start, end):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials

        if financials is None or financials.empty:
            return None

        if (
            "Net Income" not in financials.index
            or "Total Revenue" not in financials.index
        ):
            return None

        net_income = financials.loc["Net Income"]
        revenue = financials.loc["Total Revenue"]
        margin = (net_income / revenue) * 100

        margin_df = margin.reset_index()
        margin_df.columns = ["Date", "Net Profit Margin %"]
        margin_df["Date"] = pd.to_datetime(margin_df["Date"], errors="coerce")
        if margin_df["Date"].dt.tz is not None:
            margin_df["Date"] = margin_df["Date"].dt.tz_localize(None)
        margin_df = margin_df.sort_values("Date").reset_index(drop=True)
        margin_df.dropna(inplace=True)

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        filtered = margin_df[
            (margin_df["Date"] >= start_dt) & (margin_df["Date"] <= end_dt)
        ]

        if len(filtered) < 2:
            return margin_df
        return filtered

    except Exception as e:
        st.warning(f"Could not load profit margin data for {ticker}: {e}")
        return None


# ********************Analysis Functions********************


def volatility(stock_data):
    if stock_data is None or len(stock_data) < 2:
        return None
    daily_returns = stock_data["Stock Price"].pct_change().dropna()
    annualized_vol = daily_returns.std() * (252**0.5) * 100
    return round(annualized_vol, 2)


def correlation(stock_data, revenue_data):
    if stock_data is None or revenue_data is None:
        return None
    if len(stock_data) < 4 or len(revenue_data) < 4:
        return None

    stock_indexed = stock_data.set_index("Date")

    try:
        quarterly_stock = (
            stock_indexed["Stock Price"].resample("QE").mean().reset_index()
        )
    except ValueError:
        quarterly_stock = (
            stock_indexed["Stock Price"].resample("Q").mean().reset_index()
        )

    quarterly_stock.columns = ["Date", "Stock Price"]

    rev_copy = revenue_data.copy()
    rev_copy["Quarter"] = pd.to_datetime(rev_copy["Date"]).dt.to_period("Q")
    quarterly_stock["Quarter"] = pd.to_datetime(quarterly_stock["Date"]).dt.to_period(
        "Q"
    )

    merged = pd.merge(quarterly_stock, rev_copy, on="Quarter", how="inner")

    if len(merged) < 4:
        return None

    corr = merged["Stock Price"].corr(merged["Revenue"])
    return round(corr, 2)


def revenue_growth(revenue_data):
    if revenue_data is None or len(revenue_data) < 8:
        return None
    recent_avg = revenue_data["Revenue"].iloc[-4:].mean()
    older_avg = revenue_data["Revenue"].iloc[-8:-4].mean()
    return pct_change(older_avg, recent_avg)


# ********************Chart Functions********************


def plot_stock_ma(stock_data, ticker):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stock_data["Date"],
            y=stock_data["Stock Price"],
            name="Stock Price",
            mode="lines",
            line=dict(color="royalblue"),
        )
    )

    if len(stock_data) >= 50:
        ma_50 = stock_data["Stock Price"].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=ma_50,
                name="50-Day MA",
                mode="lines",
                line=dict(color="orange", dash="dash"),
            )
        )

    if len(stock_data) >= 200:
        ma_200 = stock_data["Stock Price"].rolling(window=200).mean()
        fig.add_trace(
            go.Scatter(
                x=stock_data["Date"],
                y=ma_200,
                name="200-Day MA",
                mode="lines",
                line=dict(color="red", dash="dash"),
            )
        )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


def plot_revenue(revenue_data, ticker):
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=revenue_data["Quarter"],
            y=revenue_data["Revenue"],
            name="Quarterly Revenue (M)",
            marker_color="teal",
        )
    )

    fig.update_layout(
        xaxis_title="Quarter",
        yaxis_title="Revenue (Millions USD)",
    )
    return fig


def plot_netprofit(netprofit_data, ticker):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=netprofit_data["Date"],
            y=netprofit_data["Net Profit Margin %"],
            mode="lines+markers",
            name="Net Profit Margin %",
            line=dict(color="green"),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Net Profit Margin (%)",
    )
    return fig


def plot_comparison(stock_data_1, stock_data_2, ticker1, ticker2):
    fig = go.Figure()

    first_price_1 = stock_data_1["Stock Price"].iloc[0]
    normalized_1 = (stock_data_1["Stock Price"] / first_price_1) * 100

    first_price_2 = stock_data_2["Stock Price"].iloc[0]
    normalized_2 = (stock_data_2["Stock Price"] / first_price_2) * 100

    fig.add_trace(
        go.Scatter(
            x=stock_data_1["Date"], y=normalized_1, name=ticker1.upper(), mode="lines"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stock_data_2["Date"], y=normalized_2, name=ticker2.upper(), mode="lines"
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Price (Start = 100)",
        legend=dict(x=0.01, y=0.99),
    )
    return fig


# ********************UI********************

st.markdown("Analyze stock performance, revenue trends, and key financial metrics.")
st.divider()

if "recent_tickers" not in st.session_state:
    st.session_state.recent_tickers = []
if "report_data" not in st.session_state:
    st.session_state.report_data = None

input_col1, input_col2, input_col3 = st.columns([3, 3, 1])

with input_col1:
    user_input = st.text_input("Company Name or Ticker (e.g. Tesla or TSLA)")

with input_col2:
    use_custom_dates = st.toggle("Use custom date range")

with input_col3:
    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("Generate Report", use_container_width=True)

if use_custom_dates:
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        start = st.date_input("Start Date", datetime(2020, 1, 1))
    with date_col2:
        end = st.date_input("End Date", datetime.today())
else:
    end = datetime.today()
    start = end - pd.DateOffset(years=1)

if st.session_state.recent_tickers:
    st.markdown(
        "**Recently Viewed:**  " + "  |  ".join(st.session_state.recent_tickers)
    )

st.divider()

# ********************Reports********************
if generate:

    if not user_input:
        st.warning("Please enter a company name or stock ticker.")
        st.stop()

    with st.spinner("Resolving ticker..."):
        ticker, company_name = resolve_ticker(user_input.strip())

    try:
        test = yf.Ticker(ticker).history(period="5d")
        if test.empty:
            st.error(
                f"Could not find a valid stock for '{user_input}'. Please try the ticker directly."
            )
            st.stop()
    except Exception:
        st.error("Could not validate ticker. Please check your input.")
        st.stop()

    if ticker not in st.session_state.recent_tickers:
        st.session_state.recent_tickers.insert(0, ticker)
        st.session_state.recent_tickers = st.session_state.recent_tickers[:5]

    with st.spinner(f"Fetching data for {ticker}..."):
        stock_data = get_stock_data(ticker, start, end)
        revenue_data = get_revenue_data(ticker, start, end)
        netprofit_data = get_netprofit_data(ticker, start, end)

    if stock_data is None or stock_data.empty:
        st.error("No stock data found. Try a different ticker or date range.")
        st.stop()
    st.session_state.report_data = {
        "ticker": ticker,
        "company_name": company_name,
        "stock_data": stock_data,
        "revenue_data": revenue_data,
        "netprofit_data": netprofit_data,
    }

if st.session_state.report_data is not None:
    ticker = st.session_state.report_data["ticker"]
    company_name = st.session_state.report_data["company_name"]
    stock_data = st.session_state.report_data["stock_data"]
    revenue_data = st.session_state.report_data["revenue_data"]
    netprofit_data = st.session_state.report_data["netprofit_data"]

    # ********************KPI********************
    st.subheader("Snapshot")
    st.caption(f"Data from {start.strftime('%b %Y')} to {end.strftime('%b %Y')}")

    latest_price = get_last(stock_data["Stock Price"])
    prev_price = (
        stock_data["Stock Price"].iloc[-2] if len(stock_data) >= 2 else latest_price
    )
    price_delta = latest_price - prev_price if latest_price and prev_price else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Latest Stock Price", f"${latest_price:,.2f}", f"{price_delta:+.2f}")
    st.caption(f"Data from {start.strftime('%b %Y')} to {end.strftime('%b %Y')}")

    if revenue_data is not None and not revenue_data.empty:
        latest_rev = get_last(revenue_data["Revenue"])
        prev_rev = (
            revenue_data["Revenue"].iloc[-2] if len(revenue_data) >= 2 else latest_rev
        )
        rev_delta = latest_rev - prev_rev if latest_rev and prev_rev else 0
        col2.metric(
            "Latest Quarterly Revenue", f"${latest_rev:,.0f}M", f"{rev_delta:+.0f}M"
        )
    else:
        col2.metric("Latest Quarterly Revenue", "N/A")

    if netprofit_data is not None and not netprofit_data.empty:
        latest_margin = get_last(netprofit_data["Net Profit Margin %"])
        col3.metric("Net Profit Margin", f"{latest_margin:.1f}%")
    else:
        col3.metric("Net Profit Margin", "N/A")

    st.divider()

    # ********************Charts********************
    st.subheader(f"{company_name} Stock Price & Moving Averages")
    st.plotly_chart(plot_stock_ma(stock_data, ticker), use_container_width=True)

    if revenue_data is not None and not revenue_data.empty:
        st.subheader(f"{company_name} Quarterly Revenue")
        st.plotly_chart(plot_revenue(revenue_data, ticker), use_container_width=True)
    else:
        st.warning("Revenue data not available for this ticker.")

    if netprofit_data is not None and not netprofit_data.empty:
        st.subheader(f"{company_name} Net Profit Margin")
        st.plotly_chart(
            plot_netprofit(netprofit_data, ticker), use_container_width=True
        )
    else:
        st.warning("Net Profit Margin data not available for this ticker.")

    st.divider()

    # ********************Key Insights********************
    st.subheader("Key Insights")
    insights = []

    cutoff = stock_data["Date"].iloc[-1] - pd.DateOffset(years=3)
    recent_stock = stock_data[stock_data["Date"] >= cutoff]

    if len(recent_stock) >= 2:
        first_price = recent_stock["Stock Price"].iloc[0]
        latest_price_val = recent_stock["Stock Price"].iloc[-1]
        avg_price = get_mean(recent_stock["Stock Price"])
        price_change_pct = pct_change(first_price, latest_price_val)
        price_vs_avg = pct_change(avg_price, latest_price_val)
        direction = "appreciated" if price_change_pct > 0 else "declined"

        insights.append(
            f"**Stock Price (last 3 years):** {company_name} has {direction} by "
            f"**{abs(price_change_pct):.1f}%**. "
            f"The current price (${latest_price_val:,.2f}) is "
            f"{'above' if price_vs_avg > 0 else 'below'} its 3-year average "
            f"(${avg_price:,.2f}) by {abs(price_vs_avg):.1f}%."
        )

    vol_val = volatility(stock_data)
    if vol_val is not None:
        if vol_val > 50:
            vol_label = "highly volatile"
        elif vol_val > 25:
            vol_label = "moderately volatile"
        else:
            vol_label = "relatively stable"
        insights.append(
            f"**Volatility:** {company_name} has an annualized volatility of "
            f"**{vol_val:.1f}%**, suggesting it is {vol_label} over the selected period."
        )

    if revenue_data is not None and len(revenue_data) >= 8:
        rev_growth = revenue_growth(revenue_data)
        if rev_growth is not None:
            if rev_growth > 50:
                rev_label = "significant growth"
            elif rev_growth > 20:
                rev_label = "steady growth"
            elif rev_growth > 0:
                rev_label = "modest growth"
            elif rev_growth > -20:
                rev_label = "a slight decline"
            else:
                rev_label = "a notable decline"
            insights.append(
                f"**Revenue Growth:** {company_name} has demonstrated {rev_label} in quarterly "
                f"revenue when comparing the last 4 quarters to the prior 4 quarters."
            )
    elif revenue_data is not None and len(revenue_data) >= 2:
        insights.append(
            "**Revenue:** Insufficient data for full trend comparison (need 8+ quarters)."
        )

    if netprofit_data is not None and len(netprofit_data) >= 2:
        first_margin = netprofit_data["Net Profit Margin %"].iloc[0]
        latest_margin_val = get_last(netprofit_data["Net Profit Margin %"])
        avg_margin = get_mean(netprofit_data["Net Profit Margin %"])
        margin_direction = (
            "improved" if latest_margin_val > first_margin else "compressed"
        )

        if latest_margin_val > 15:
            health = "strong profitability"
        elif latest_margin_val > 8:
            health = "moderate profitability"
        elif latest_margin_val > 0:
            health = "thin margins"
        else:
            health = "a net loss position"

        insights.append(
            f"**Profitability:** Net profit margin has {margin_direction} from "
            f"**{first_margin:.1f}%** to **{latest_margin_val:.1f}%** "
            f"(period avg: {avg_margin:.1f}%), indicating {health}."
        )
    elif netprofit_data is None:
        insights.append("**Profitability:** Margin data unavailable for this ticker.")

    corr_val = correlation(stock_data, revenue_data)
    if corr_val is not None:
        if corr_val > 0.7:
            corr_label = "strong positive correlation"
            corr_meaning = "stock price has generally moved in line with revenue growth"
        elif corr_val > 0.3:
            corr_label = "moderate positive correlation"
            corr_meaning = "stock price has partially tracked revenue performance"
        elif corr_val < -0.3:
            corr_label = "negative correlation"
            corr_meaning = "stock price and revenue have moved in opposite directions, possibly reflecting sentiment-driven pricing"
        else:
            corr_label = "weak correlation"
            corr_meaning = (
                "stock price movement has not closely tracked revenue performance"
            )

        insights.append(
            f"**Price vs Revenue:** There is a {corr_label} ({corr_val:.2f}) between "
            f"quarterly stock price and revenue for {company_name}, suggesting {corr_meaning}."
        )

    for insight in insights:
        st.markdown(f"- {insight}")

    # ********************Summary********************
    st.divider()
    latest_price_summary = get_last(stock_data["Stock Price"])
    volatility_summary = volatility(stock_data)

    if volatility_summary is not None:
        if volatility_summary > 50:
            risk_label = "high-risk"
        elif volatility_summary > 25:
            risk_label = "moderate-risk"
        else:
            risk_label = "low-risk"

        st.markdown(
            f"**Summary:** Based on the selected period, {company_name} appears to be a "
            f"**{risk_label}** stock currently trading at **${latest_price_summary:,.2f}**. "
            f"Investors should consider both the revenue trajectory and margin trends "
            f"before drawing conclusions."
        )

    # ********************Comparison********************
    st.divider()
    st.subheader("Compare with Another Stock")

    compare_input = st.text_input(
        "Enter company or ticker to compare (e.g. Apple or AAPL)"
    )
    use_custom_compare = st.toggle("Use custom date range for comparison")

    if use_custom_compare:
        cmp_col1, cmp_col2 = st.columns(2)
        with cmp_col1:
            comp_start = st.date_input("Comparison Start Date", datetime(2020, 1, 1))
        with cmp_col2:
            comp_end = st.date_input("Comparison End Date", datetime.today())
    else:
        comp_end = datetime.today()
        comp_start = comp_end - pd.DateOffset(years=1)

    compare_button = st.button("Compare")

    if compare_button and compare_input.strip():
        ticker2, company_name_2 = resolve_ticker(compare_input.strip())

        try:
            test2 = yf.Ticker(ticker2).history(period="5d")
            if test2.empty:
                st.warning(f"Could not find a valid stock for '{compare_input}'.")
                ticker2 = None
        except Exception:
            st.warning(f"Could not validate '{compare_input}'.")
            ticker2 = None

        if ticker2:
            with st.spinner(f"Fetching data for {ticker2}..."):
                stock_data_2 = get_stock_data(ticker2, comp_start, comp_end)
                stock_data_1_comp = get_stock_data(ticker, comp_start, comp_end)

            if stock_data_2 is not None and not stock_data_2.empty:
                if stock_data_1_comp is not None and not stock_data_1_comp.empty:
                    st.plotly_chart(
                        plot_comparison(
                            stock_data_1_comp, stock_data_2, ticker, ticker2
                        ),
                        use_container_width=True,
                    )

                return_1 = (
                    pct_change(
                        stock_data_1_comp["Stock Price"].iloc[0],
                        stock_data_1_comp["Stock Price"].iloc[-1],
                    )
                    if stock_data_1_comp is not None and not stock_data_1_comp.empty
                    else None
                )
                return_2 = pct_change(
                    stock_data_2["Stock Price"].iloc[0],
                    stock_data_2["Stock Price"].iloc[-1],
                )
                vol_1 = volatility(stock_data_1_comp)
                vol_2 = volatility(stock_data_2)

                cmp_metrics_col1, cmp_metrics_col2 = st.columns(2)
                with cmp_metrics_col1:
                    st.markdown(f"**{company_name}**")
                    st.markdown(
                        f"- Return: {return_1:.1f}%"
                        if return_1 is not None
                        else "- Return: N/A"
                    )
                    st.markdown(
                        f"- Volatility: {vol_1:.1f}%"
                        if vol_1 is not None
                        else "- Volatility: N/A"
                    )
                with cmp_metrics_col2:
                    st.markdown(f"**{company_name_2}**")
                    st.markdown(
                        f"- Return: {return_2:.1f}%"
                        if return_2 is not None
                        else "- Return: N/A"
                    )
                    st.markdown(
                        f"- Volatility: {vol_2:.1f}%"
                        if vol_2 is not None
                        else "- Volatility: N/A"
                    )
            else:
                st.warning(f"Could not load data for {ticker2}.")

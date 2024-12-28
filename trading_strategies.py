import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="Trading Strategies", layout="wide")

# Application title
st.title("Trading Strategies")

# Select Strategy
st.sidebar.header("Strategy Selection")
strategy = st.sidebar.selectbox(
    "Choose a Strategy:",
    ["Moving Averages", "RSI (Relative Strength Index)", "Bollinger Bands"]
)

# Asset Parameters
st.sidebar.header("Asset Parameters")
ticker = st.sidebar.selectbox(
    "Select an asset ticker:",
    ["AAPL", "TSLA", "GOOGL", "MSFT", "BTC-USD", "ETH-USD", "AMZN"]
)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Load asset data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data["Return"] = data["Close"].pct_change()
    return data

data = load_data(ticker, start_date, end_date)

if data.empty:
    st.error("No data available for the specified asset. Please check the ticker or dates.")
else:
    # Strategy: Moving Averages
    if strategy == "Moving Averages":
        # Strategy Parameters
        st.sidebar.header("Moving Averages Parameters")
        short_window = st.sidebar.slider("Short Moving Average Period (days):", 5, 50, 20)
        long_window = st.sidebar.slider("Long Moving Average Period (days):", 50, 200, 100)

        # Calculate moving averages
        data["Short_MA"] = data["Close"].rolling(window=short_window).mean()
        data["Long_MA"] = data["Close"].rolling(window=long_window).mean()

        # Generate buy/sell signals
        data["Signal"] = 0
        data.loc[data["Short_MA"] > data["Long_MA"], "Signal"] = 1
        data.loc[data["Short_MA"] <= data["Long_MA"], "Signal"] = -1

        # Strategy returns
        data["Strategy_Return"] = data["Signal"].shift(1) * data["Return"]

        # Plot moving averages
        st.subheader(f"Asset Price ({ticker}) and Moving Averages")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data["Close"], label="Closing Price", color="blue")
        ax.plot(data.index, data["Short_MA"], label=f"Short MA ({short_window} days)", color="green")
        ax.plot(data.index, data["Long_MA"], label=f"Long MA ({long_window} days)", color="red")
        ax.legend(loc="upper left")
        ax.set_title(f"Price and Moving Averages for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    # Strategy: RSI
    elif strategy == "RSI (Relative Strength Index)":
        # RSI Calculation
        st.sidebar.header("RSI Parameters")
        rsi_period = st.sidebar.slider("RSI Period (days):", 5, 50, 14)

        delta = data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        # Generate buy/sell signals
        data["Signal"] = 0
        data.loc[data["RSI"] < 30, "Signal"] = 1  # Buy when RSI < 30
        data.loc[data["RSI"] > 70, "Signal"] = -1  # Sell when RSI > 70

        # Strategy returns
        data["Strategy_Return"] = data["Signal"].shift(1) * data["Return"]

        # Plot RSI
        st.subheader(f"RSI Strategy for {ticker}")
        fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Price plot
        ax[0].plot(data.index, data["Close"], label="Closing Price", color="blue")
        ax[0].set_title(f"Price of {ticker}")
        ax[0].set_ylabel("Price")
        ax[0].legend(loc="upper left")

        # RSI plot
        ax[1].plot(data.index, data["RSI"], label="RSI", color="purple")
        ax[1].axhline(70, color="red", linestyle="--", label="Overbought (70)")
        ax[1].axhline(30, color="green", linestyle="--", label="Oversold (30)")
        ax[1].set_title(f"RSI for {ticker}")
        ax[1].set_xlabel("Date")
        ax[1].set_ylabel("RSI")
        ax[1].legend(loc="upper left")
        st.pyplot(fig)

    # Strategy: Bollinger Bands
    elif strategy == "Bollinger Bands":
        # Bollinger Bands Calculation
        st.sidebar.header("Bollinger Bands Parameters")
        bb_period = st.sidebar.slider("Bollinger Bands Period (days):", 10, 50, 20)

        data["20_MA"] = data["Close"].rolling(window=bb_period).mean()
        data["Upper_Band"] = data["20_MA"] + (data["Close"].rolling(window=bb_period).std() * 2)
        data["Lower_Band"] = data["20_MA"] - (data["Close"].rolling(window=bb_period).std() * 2)

        # Generate buy/sell signals
        data["Signal"] = 0
        data.loc[data["Close"] < data["Lower_Band"], "Signal"] = 1  # Buy
        data.loc[data["Close"] > data["Upper_Band"], "Signal"] = -1  # Sell

        # Strategy returns
        data["Strategy_Return"] = data["Signal"].shift(1) * data["Return"]

        # Plot Bollinger Bands
        st.subheader(f"Bollinger Bands Strategy for {ticker}")
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data["Close"], label="Closing Price", color="blue")
        ax.plot(data.index, data["Upper_Band"], label="Upper Band", color="red", linestyle="--")
        ax.plot(data.index, data["Lower_Band"], label="Lower Band", color="green", linestyle="--")
        ax.fill_between(data.index, data["Lower_Band"], data["Upper_Band"], color="gray", alpha=0.2)
        ax.legend(loc="upper left")
        ax.set_title(f"Bollinger Bands for {ticker}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    # Calculate cumulative returns
    data["Cumulative_Market_Return"] = (1 + data["Return"]).cumprod()
    data["Cumulative_Strategy_Return"] = (1 + data["Strategy_Return"]).cumprod()

    # Plot cumulative performance
    st.subheader(f"Cumulative Performance: {strategy} vs Market")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data["Cumulative_Market_Return"], label="Market (Buy & Hold)", color="blue")
    ax.plot(data.index, data["Cumulative_Strategy_Return"], label=f"{strategy} Strategy", color="orange")
    ax.legend(loc="upper left")
    ax.set_title(f"Cumulative Performance for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    st.pyplot(fig)

    # Performance Summary
    st.sidebar.subheader("Performance Summary")
    total_return = data["Return"].sum()
    strategy_return = data["Strategy_Return"].sum()
    st.sidebar.metric("Total Market Return (%)", f"{total_return * 100:.2f}")
    st.sidebar.metric("Total Strategy Return (%)", f"{strategy_return * 100:.2f}")
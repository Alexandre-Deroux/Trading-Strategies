import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Streamlit page configuration
st.set_page_config(page_title="📈 Trading Strategies", layout="wide")

# Application title
st.title("📈 Trading Strategies")

# Select Strategy
st.sidebar.header("📊 Strategy Selection")
strategy = st.sidebar.selectbox(
    "Choose a Strategy:",
    ["Moving Averages", "RSI (Relative Strength Index)", "MACD (Moving Average Convergence Divergence)"]
)

# Asset Parameters
st.sidebar.header("📌 Asset Parameters")
index = st.sidebar.selectbox("Select an Index:", ["S&P 500", "NASDAQ-100", "Dow Jones", "CAC 40"])
tickers_index_average = {
    "S&P 500": "^GSPC",
    "NASDAQ-100": "^NDX",
    "Dow Jones": "^DJI",
    "CAC 40": "^FCHI"
}
tickers = {
    "S&P 500": pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist(),
    "NASDAQ-100": pd.read_html("https://en.wikipedia.org/wiki/NASDAQ-100")[4]["Symbol"].tolist(),
    "Dow Jones": pd.read_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")[2]["Symbol"].tolist(),
    "CAC 40": pd.read_html("https://en.wikipedia.org/wiki/CAC_40")[4]["Ticker"].tolist()
}
ticker = st.sidebar.selectbox(f'Select an Asset from {index} or "Index Average":', ["Index Average"] + tickers[index])
if ticker == "Index Average":
    ticker = tickers_index_average[index]
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
    st.error("❌ No data available for the specified asset. Please check the ticker or dates.")
else:
    # Strategy: Moving Averages
    if strategy == "Moving Averages":
        # Strategy Parameters
        st.sidebar.header("⚙️ Moving Averages Parameters")
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
        st.subheader(f"📊 Asset Price ({ticker}) and Moving Averages")
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
        st.sidebar.header("⚙️ RSI Parameters")
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
        st.subheader(f"📊 RSI Strategy for {ticker}")
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

    # Strategy: MACD Strategy
    elif strategy == "MACD (Moving Average Convergence Divergence)":
        # MACD Strategy Parameters
        st.sidebar.header("⚙️ MACD Parameters")
        short_ema = st.sidebar.slider("Short EMA Period (days):", 5, 50, 12)
        long_ema = st.sidebar.slider("Long EMA Period (days):", 20, 200, 26)
        signal_period = st.sidebar.slider("Signal Line EMA Period (days):", 5, 20, 9)
        
        # Calculate MACD
        data["Short_EMA"] = data["Close"].ewm(span=short_ema, adjust=False).mean()
        data["Long_EMA"] = data["Close"].ewm(span=long_ema, adjust=False).mean()
        data["MACD"] = data["Short_EMA"] - data["Long_EMA"]
        data["Signal_Line"] = data["MACD"].ewm(span=signal_period, adjust=False).mean()

        # Generate buy/sell signals based on MACD crossovers
        data["Signal"] = 0
        data.loc[data["MACD"] > data["Signal_Line"], "Signal"] = 1  # Buy
        data.loc[data["MACD"] < data["Signal_Line"], "Signal"] = -1  # Sell

        # Strategy returns
        data["Strategy_Return"] = data["Signal"].shift(1) * data["Return"]

        # Plot MACD with signals
        st.subheader(f"📊 MACD Strategy for {ticker}")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data["Close"], label="Closing Price", color="blue")
        ax1.set_title(f"{ticker} Price and MACD")
        ax1.set_ylabel("Price")
        ax1.legend(loc="upper left")

        # MACD chart
        ax2.plot(data.index, data["MACD"], label="MACD", color="purple")
        ax2.plot(data.index, data["Signal_Line"], label="Signal Line", color="orange", linestyle="--")
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("MACD")
        ax2.set_xlabel("Date")
        ax2.legend(loc="upper left")
        st.pyplot(fig)

    # Calculate cumulative returns
    data["Cumulative_Market_Return"] = (1 + data["Return"]).cumprod()
    data["Cumulative_Strategy_Return"] = (1 + data["Strategy_Return"]).cumprod()

    # Plot cumulative performance
    st.subheader(f"📊 Cumulative Performance: {strategy} vs Market")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index, data["Cumulative_Market_Return"], label="Market (Buy & Hold)", color="blue")
    ax.plot(data.index, data["Cumulative_Strategy_Return"], label=f"{strategy} Strategy", color="orange")
    ax.legend(loc="upper left")
    ax.set_title(f"Cumulative Performance for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    st.pyplot(fig)

    # Performance Summary
    st.sidebar.subheader("📈 Performance Summary")
    total_return = data["Return"].sum()
    strategy_return = data["Strategy_Return"].sum()
    st.sidebar.metric("Total Market Return (%)", f"{total_return * 100:.2f}")
    st.sidebar.metric("Total Strategy Return (%)", f"{strategy_return * 100:.2f}")

# Author
st.markdown("""Made by [Alexandre Deroux](https://www.linkedin.com/in/alexandre-deroux).""", unsafe_allow_html=True)
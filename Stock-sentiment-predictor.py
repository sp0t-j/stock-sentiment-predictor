import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from tqdm import tqdm
from transformers import pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Define a list of stock tickers to analyze
tickers = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "DIS", "MA", "HD", "BAC", "ADBE", "NFLX", "CRM", "PYPL",
    "INTC", "CMCSA", "PFE", "KO", "XOM", "CSCO", "T", "PEP", "ABBV", "COST"
]

# --- RSI computation ---
def compute_rsi(series, period=14):
    delta = series.diff()  # Calculate price difference between days
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # Avg gain
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # Avg loss
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))  # RSI formula
    return rsi

# --- Fetch stock data from Yahoo Finance ---
def fetch_stock_data(ticker, start="2020-01-01", end="2025-07-12"):
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No stock data returned for {ticker}.")

        # Handle multi-indexed columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[1] if col[0] == ticker else col[0] for col in df.columns]

        df.reset_index(inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df["ticker"] = ticker
        return df
    except Exception as e:
        print(f"Error in fetch_stock_data for {ticker}: {e}")
        return pd.DataFrame()

# --- Fetch news headlines from FinViz ---
def fetch_news_finviz(ticker):
    all_news = []
    base_url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        resp = requests.get(base_url, headers=headers)
        soup = BeautifulSoup(resp.text, 'html.parser')

        news_table = soup.find('table', class_='fullview-news-outer')
        if not news_table:
            print(f"No news table found for {ticker}")
            return pd.DataFrame()

        for row in news_table.find_all('tr'):
            cols = row.find_all('td')
            if len(cols) == 2:
                datetime_str = cols[0].text.strip()
                headline = cols[1].text.strip()

                news_date = None
                if "Today" in datetime_str:
                    news_date = datetime.now().date()
                elif "Yesterday" in datetime_str:
                    news_date = (datetime.now() - timedelta(days=1)).date()
                else:
                    try:
                        news_date = datetime.strptime(datetime_str.split()[0], "%b-%d-%y").date()
                    except ValueError:
                        continue

                all_news.append({'date': news_date, 'title': headline, 'ticker': ticker})

        return pd.DataFrame(all_news)

    except Exception as e:
        print(f"Error fetching FinViz news for {ticker}: {e}")
        return pd.DataFrame()

# --- Create target variable: price movement ---
def create_movement(df):
    if 'Close' not in df.columns:
        raise ValueError("The 'Close' column is missing from the DataFrame.")

    df = df.sort_values(by=['ticker', 'Date']).copy()
    df['prev_close'] = df.groupby('ticker')['Close'].shift(1)  # Previous day's close

    print("Columns in the DataFrame:", df.columns.tolist())  # Debug print
    print("DataFrame head:\n", df.head())

    df = df.dropna(subset=['prev_close'])
    df['price_movement'] = (df['Close'] > df['prev_close']).astype(int)  # 1 if price increased

    return df

# --- Load sentiment analysis model ---
sentiment_pipeline = pipeline("sentiment-analysis")
tqdm.pandas()  # Enable tqdm progress for pandas

# --- Main Data Processing Loop ---
all_merged = []
successful_tickers = 0

for ticker in tickers:
    print(f"Processing {ticker}")

    stock_df = fetch_stock_data(ticker, start="2018-01-01", end="2025-07-15")
    if stock_df.empty:
        continue

    stock_df = create_movement(stock_df)

    # Compute features
    stock_df['MA5'] = stock_df['Close'].rolling(window=5).mean()
    stock_df['MA20'] = stock_df['Close'].rolling(window=20).mean()
    stock_df['RSI'] = compute_rsi(stock_df['Close'])
    stock_df['Return'] = stock_df['Close'].pct_change()
    stock_df['Next_Close'] = stock_df.groupby('ticker')['Close'].shift(-1)  # Next day's close

    stock_df = stock_df.dropna(subset=['MA5', 'MA20', 'RSI', 'Return', 'Next_Close'])

    news_df = fetch_news_finviz(ticker)
    if news_df.empty:
        print(f"No news found for {ticker}, using neutral sentiment.")
        latest_stock_date = stock_df["Date"].max().date()
        news_df = pd.DataFrame([{
            "date": latest_stock_date,
            "title": "No news",
            "ticker": ticker
        }])

    news_df["date"] = pd.to_datetime(news_df["date"])
    news_df = news_df.sort_values(by="date")
    news_df["sentiment_result"] = news_df["title"].progress_apply(sentiment_pipeline)  # Predict sentiment
    news_df["sentiment_label"] = news_df["sentiment_result"].apply(lambda x: x[0]["label"])
    news_df["sentiment_score"] = news_df["sentiment_result"].apply(lambda x: x[0]["score"])
    news_df["target_date"] = news_df["date"] + pd.Timedelta(days=1)

    # Aggregate daily sentiment
    daily_sentiment = (
        news_df.groupby(["target_date", "ticker"])
        .agg({
            "sentiment_score": "mean",
            "sentiment_label": lambda x: x.mode()[0] if not x.mode().empty else "NEUTRAL"
        })
        .reset_index()
    )

    stock_df["Date"] = pd.to_datetime(stock_df["Date"])

    # Merge stock data with sentiment
    merged = pd.merge(
        stock_df,
        daily_sentiment,
        left_on=["Date", "ticker"],
        right_on=["target_date", "ticker"],
        how="left"
    )

    merged["sentiment_score"] = merged["sentiment_score"].fillna(0)  # Fill missing scores with 0

    threshold = 0.002  # Define target: price increases by at least 0.2%
    merged["pct_change"] = (merged["Next_Close"] - merged["Close"]) / merged["Close"]
    merged["movement"] = (merged["pct_change"] > threshold).astype(int)

    print(f"{ticker} -> Rows added: {merged.shape[0]}")
    all_merged.append(merged)
    successful_tickers += 1

    time.sleep(1)  # Pause between requests to avoid rate limiting

if successful_tickers == 0:
    raise ValueError("No data collected from any ticker. Check data or news sources.")

combined_df = pd.concat(all_merged, ignore_index=True)  # Combine all ticker data

# --- Prepare training data ---
feature_cols = ["sentiment_score", "MA5", "MA20", "RSI", "Return"]
df_model = combined_df.dropna(subset=feature_cols + ["movement"])

print(f"Shape of df_model after dropping NaNs: {df_model.shape}")
if df_model.empty:
    raise ValueError("df_model is empty. Check your previous data processing steps.")

print(f"Date range in df_model: {df_model['Date'].min()} to {df_model['Date'].max()}")

split_date = pd.to_datetime("2025-04-01")  # Date to split training/testing
train_mask = df_model["Date"] < split_date
test_mask = df_model["Date"] >= split_date

X_train = df_model.loc[train_mask, feature_cols]
y_train = df_model.loc[train_mask, "movement"]
X_test = df_model.loc[test_mask, feature_cols]
y_test = df_model.loc[test_mask, "movement"]

print(f"Rows in training set: {train_mask.sum()}")
print(f"Rows in test set: {test_mask.sum()}")
print("Original y_train class distribution:")
print(y_train.value_counts())

if y_train.empty:
    raise ValueError("Training target y_train is empty. Cannot apply SMOTE.")

# --- Balance dataset with SMOTE or ROS ---
from imblearn.over_sampling import RandomOverSampler

print("Training class distribution before resampling:")
print(y_train.value_counts())

if y_train.value_counts().min() < 6:
    print("Too few samples in a class for SMOTE — using RandomOverSampler instead.")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
else:
    print("Using SMOTE to balance classes.")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# --- Train classifier ---
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)

# --- Evaluate classifier ---
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
print("After SMOTE:")
print(pd.Series(y_train_res).value_counts())

# --- Backtest strategy ---
print("\n Running Backtest...")

results_df = df_model.loc[test_mask, ["Date", "Close", "Next_Close", "ticker"]].copy()
results_df["prediction"] = y_pred
results_df["actual_movement"] = y_test.values

results_df["trade_return"] = np.where(
    results_df["prediction"] == 1,
    (results_df["Next_Close"] - results_df["Close"]) / results_df["Close"],
    0.0
)

results_df["cumulative_return"] = (1 + results_df["trade_return"]).cumprod()

total_trades = (results_df["prediction"] == 1).sum()
winning_trades = ((results_df["prediction"] == 1) & (results_df["trade_return"] > 0)).sum()
losing_trades = total_trades - winning_trades

total_return = results_df["cumulative_return"].iloc[-1] - 1

daily_returns = results_df.loc[results_df["prediction"] == 1, "trade_return"]
sharpe_ratio = (
    daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    if daily_returns.std() != 0 else 0
)

print(f"Backtest Summary")
print(f"Total trades: {total_trades}")
print(f"Wins: {winning_trades}, Losses: {losing_trades}")
print(f"Win rate: {winning_trades / total_trades:.2%}" if total_trades > 0 else "Win rate: N/A")
print(f"Total return: {total_return:.2%}")
print(f"Sharpe ratio: {sharpe_ratio:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(results_df["Date"], results_df["cumulative_return"], label="Cumulative Return", color='blue')
plt.axhline(1, color="gray", linestyle="--", linewidth=0.8)
plt.title("Strategy Backtest: Cumulative Return")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Paper Trading Predictions ---
print("\nRunning today's predictions for paper trading...")

today = datetime.now().date()
yesterday = today - timedelta(days=1)

today_predictions = []

for ticker in tickers:
    print(f"Analyzing {ticker}...")

    df = fetch_stock_data(ticker, start="2024-12-01", end=datetime.now().strftime('%Y-%m-%d'))
    if df.empty or len(df) < 21:
        print(f"Not enough data for {ticker}. Skipping.")
        continue

    df = df.sort_values("Date").copy()
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['Return'] = df['Close'].pct_change()
    df = df.dropna(subset=['MA5', 'MA20', 'RSI', 'Return'])
    if df.empty:
        print(f"No valid feature row for {ticker}. Skipping.")
        continue

    latest_row = df.iloc[-1]

    news_df = fetch_news_finviz(ticker)
    if not news_df.empty:
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
        yesterday_news = news_df[news_df["date"] == yesterday]

        if not yesterday_news.empty:
            sentiments = sentiment_pipeline(list(yesterday_news["title"]))
            scores = [s["score"] if s["label"] == "POSITIVE" else -s["score"] for s in sentiments]
            avg_score = np.mean(scores)
        else:
            avg_score = 0  # No news = neutral
    else:
        avg_score = 0

    features = [
        avg_score,
        latest_row['MA5'],
        latest_row['MA20'],
        latest_row['RSI'],
        latest_row['Return']
    ]

    try:
        features_df = pd.DataFrame([features], columns=feature_cols)
        pred = clf.predict(features_df)[0]
        prob = clf.predict_proba(features_df)[0][1]
    except Exception as e:
        print(f"Prediction failed for {ticker}: {e}")
        continue

    today_predictions.append({
        "ticker": ticker,
        "prediction": pred,
        "confidence": prob,
        "date": latest_row["Date"].date(),
        "close_price": latest_row["Close"]
    })

pred_df = pd.DataFrame(today_predictions)
buy_signals = pred_df[pred_df["prediction"] == 1].sort_values(by="confidence", ascending=False)

print("\n=== Today's Buy Recommendations ===")
print(buy_signals[["date", "ticker", "close_price", "confidence"]])

buy_signals.to_csv("today_buy_signals.csv", index=False)

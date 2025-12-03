import yfinance as yf
import pandas as pd
from prophet import Prophet
import datetime
import streamlit as st

@st.cache_data
def get_all_bist_tickers():
    """
    Fetches all BIST tickers from a public GitHub repository.
    Returns a list of tickers with '.IS' appended.
    """
    url = "https://raw.githubusercontent.com/ahmeterenodaci/Istanbul-Stock-Exchange--BIST--including-symbols-and-logos/main/without_logo.csv"
    try:
        df = pd.read_csv(url)
        tickers = df['symbol'].tolist()
        # Append .IS to each ticker
        tickers = [f"{ticker}.IS" for ticker in tickers]
        return tickers
    except Exception as e:
        st.error(f"Error fetching tickers: {e}")
        # Fallback to a small list if fetching fails
        return ["XU100.IS", "THYAO.IS", "EREGL.IS", "TUPRS.IS", "GARAN.IS"]


def load_data(tickers, period="5y"):
    """
    Fetches historical data for the given tickers.
    """
    data = yf.download(tickers, period=period, group_by='ticker', progress=False)
    return data

def calculate_metrics(data, ticker):
    """
    Calculates 50-day and 200-day Moving Averages.
    """
    df = data[ticker].copy()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    return df

def calculate_correlation(data, ticker1, ticker2):
    """
    Calculates correlation between two tickers based on Close price.
    """
    # yfinance download structure might be multi-index if multiple tickers
    # Structure: (Price, Ticker) or (Ticker, Price) depending on group_by
    
    # If group_by='ticker', data[ticker] gives a DataFrame with Open, High, Low, Close, Volume
    s1 = data[ticker1]['Close']
    s2 = data[ticker2]['Close']
    
    # Align dates
    df = pd.concat([s1, s2], axis=1).dropna()
    df.columns = [ticker1, ticker2]
    
    return df.corr().iloc[0, 1]

def forecast_prophet(data, ticker, days=30):
    """
    Forecasts the next 'days' using Prophet.
    """
    df = data[ticker].reset_index()
    # Prophet expects columns 'ds' and 'y'
    # yfinance Date index is usually named 'Date'
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    
    # Remove timezone info if present, Prophet can be picky
    if df['ds'].dt.tz is not None:
        df['ds'] = df['ds'].dt.tz_localize(None)
        
    m = Prophet()
    m.fit(df)
    
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    
    return forecast

# BIST30 List for Pattern Matching (Reference)
BIST30_TICKERS = [
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", 
    "BRSAN.IS", "CANTE.IS", "DOAS.IS", "EKGYO.IS", "ENKAI.IS", 
    "EREGL.IS", "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HEKTS.IS", 
    "ISCTR.IS", "KCHOL.IS", "KONTR.IS", "KOZAL.IS", "KRDMD.IS", 
    "ODAS.IS", "OYAKC.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", 
    "SASA.IS", "SISE.IS", "TCELL.IS", "THYAO.IS", "TOASO.IS", 
    "TUPRS.IS", "YKBNK.IS"
]

import numpy as np

def normalize_series(series):
    """
    MinMax normalization to 0-1 range.
    """
    return (series - series.min()) / (series.max() - series.min())

@st.cache_data
def find_similar_patterns(current_ticker, lookback_days=60, future_days=10):
    """
    Finds top 3 similar historical patterns from BIST30 stocks.
    """
    # 1. Fetch data for current ticker to get the pattern
    # We need the last 'lookback_days'
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=lookback_days * 2) # Fetch more to be safe
    
    current_data = yf.download(current_ticker, start=start_date, progress=False)
    if current_data.empty:
        return []
        
    # Handle multi-index if necessary (yfinance structure changes)
    if isinstance(current_data.columns, pd.MultiIndex):
        current_series = current_data['Close'][current_ticker]
    else:
        current_series = current_data['Close']
        
    current_pattern = current_series.tail(lookback_days)
    if len(current_pattern) < lookback_days:
        return []
        
    norm_current = normalize_series(current_pattern).values
    
    # 2. Fetch reference data (BIST30) - 5 years
    # To optimize, we download all at once
    ref_start = end_date - datetime.timedelta(days=365 * 5)
    ref_data = yf.download(BIST30_TICKERS, start=ref_start, group_by='ticker', progress=False)
    
    matches = []
    
    # 3. Sliding Window Search
    for ticker in BIST30_TICKERS:
        if ticker not in ref_data.columns.levels[0]:
            continue
            
        series = ref_data[ticker]['Close'].dropna()
        if len(series) < lookback_days + future_days:
            continue
            
        # Vectorized sliding window is hard with different scales, so we loop
        # Optimization: stride of 5 days to speed up? No, let's do stride 1 but maybe limit history if slow.
        # Let's check every 5th day to speed up for now as requested "start fast"
        for i in range(0, len(series) - lookback_days - future_days, 2): 
            window = series.iloc[i : i + lookback_days]
            future = series.iloc[i + lookback_days : i + lookback_days + future_days]
            
            # Skip flat windows (no variance)
            if window.max() == window.min():
                continue
                
            norm_window = normalize_series(window).values
            
            # Euclidean Distance
            dist = np.linalg.norm(norm_current - norm_window)
            
            # Similarity Score (inverse of distance)
            # Max possible distance for two 0-1 vectors of len N is sqrt(N)
            max_dist = np.sqrt(lookback_days)
            score = 1 - (dist / max_dist)
            
            matches.append({
                'ticker': ticker,
                'date': window.index[-1],
                'score': score,
                'future_returns': (future.values - future.values[0]) / future.values[0], # Percentage change from end of window
                'window_data': window.values,
                'future_data': future.values
            })
            
    # Sort by score descending
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    return matches[:3]

@st.cache_data
def run_monte_carlo(ticker, simulations=1000, days=30):
    """
    Runs Monte Carlo Simulation using Geometric Brownian Motion (GBM).
    Returns simulation paths and metrics.
    """
    # Fetch historical data (last 1 year is enough for volatility)
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365)
    
    data = yf.download(ticker, start=start_date, progress=False)
    if data.empty:
        return None
        
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'][ticker]
    else:
        prices = data['Close']
        
    # Calculate returns
    log_returns = np.log(1 + prices.pct_change())
    
    # Drift and Volatility
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    
    # Simulation
    # Z is random variable
    Z = np.random.normal(0, 1, (days, simulations))
    
    daily_returns = np.exp(drift + stdev * Z)
    
    # Price paths
    price_paths = np.zeros((days, simulations))
    last_price = prices.iloc[-1]
    price_paths[0] = last_price
    
    for t in range(1, days):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    # Metrics
    final_prices = price_paths[-1]
    expected_price = np.mean(final_prices)
    # VaR 95% (5th percentile)
    var_95 = np.percentile(final_prices, 5)
    # Upside 95% (95th percentile)
    upside_95 = np.percentile(final_prices, 95)
    
    return {
        'paths': price_paths,
        'metrics': {
            'expected_price': expected_price,
            'var_95': var_95,
            'upside_95': upside_95,
            'last_price': last_price
        }
    }

@st.cache_data
def get_fundamental_metrics(ticker):
    """
    Fetches fundamental ratios and calculates health scores.
    Returns a dictionary with raw values and normalized scores (0-100).
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Helper to safely get value or 0
        def get_val(key):
            return info.get(key, 0) or 0
            
        # 1. Fetch Raw Data
        pe = get_val('trailingPE')
        pb = get_val('priceToBook')
        roe = get_val('returnOnEquity')
        de = get_val('debtToEquity')
        pm = get_val('profitMargins')
        
        # 2. Normalize & Score (0-100)
        # These are heuristic scorings for demonstration
        
        # P/E: Lower is better. Target < 15. > 50 is 0.
        if pe <= 0: score_pe = 0 # Negative PE usually bad
        else: score_pe = max(0, min(100, 100 - (pe - 15) * 2.5)) if pe > 15 else 100
        
        # P/B: Lower is better. Target < 1.5. > 10 is 0.
        score_pb = max(0, min(100, 100 - (pb - 1.5) * 10)) if pb > 1.5 else 100
        
        # ROE: Higher is better. Target > 20%. < 0 is 0.
        score_roe = max(0, min(100, roe * 500)) # ROE is usually 0.15 for 15%
        
        # Debt/Equity: Lower is better. Target < 50. > 200 is 0.
        # yfinance returns D/E as percentage (e.g., 45.5) or ratio? Usually percentage.
        # Let's assume percentage.
        score_de = max(0, min(100, 100 - (de - 50) * 0.6)) if de > 50 else 100
        
        # Profit Margin: Higher is better. Target > 20%. < 0 is 0.
        score_pm = max(0, min(100, pm * 500))
        
        return {
            'raw': {
                'P/E': pe,
                'P/B': pb,
                'ROE': roe,
                'Debt/Equity': de,
                'Profit Margin': pm
            },
            'scores': {
                'P/E': score_pe,
                'P/B': score_pb,
                'ROE': score_roe,
                'Debt/Equity': score_de,
                'Profit Margin': score_pm
            }
        }
    except Exception as e:
        st.error(f"Error fetching fundamentals: {e}")
        return None




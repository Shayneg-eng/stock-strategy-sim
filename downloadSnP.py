import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

# Download S&P 500 data since 2000
print("Downloading S&P 500 data...")

# Try multiple ticker symbols
tickers_to_try = ["SPY", "^GSPC", "VOO"]

sp500 = pd.DataFrame()

for ticker in tickers_to_try:
    print(f"\nTrying ticker: {ticker}")
    try:
        # Create ticker object first
        stock = yf.Ticker(ticker)
        
        # Download historical data
        sp500 = stock.history(start="2000-01-01", end=datetime.now().strftime("%Y-%m-%d"))
        
        if not sp500.empty:
            print(f"Success with {ticker}!")
            break
    except Exception as e:
        print(f"Failed with {ticker}: {e}")
        continue

# Check if download was successful
if sp500.empty:
    print("\n" + "="*50)
    print("ERROR: Could not download data with any ticker.")
    print("This might be a temporary Yahoo Finance API issue.")
    print("\nAlternative: Download data manually from:")
    print("https://finance.yahoo.com/quote/SPY/history")
    print("Save as 'sp500_data.csv' in the same folder")
    print("="*50)
else:
    # Display basic info
    print(f"\nData downloaded successfully!")
    print(f"Date range: {sp500.index[0]} to {sp500.index[-1]}")
    print(f"Total trading days: {len(sp500)}")
    print(f"\nFirst few rows:")
    print(sp500.head())
    print(f"\nLast few rows:")
    print(sp500.tail())
    print(f"\nBasic statistics:")
    print(sp500['Close'].describe())
    
    # Save to CSV for future use
    sp500.to_csv('sp500_data.csv')
    print("\nData saved to 'sp500_data.csv'")
    
    # Add some useful calculated columns
    sp500['Daily_Return'] = sp500['Close'].pct_change()
    sp500['Cumulative_Return'] = (1 + sp500['Daily_Return']).cumprod()
    
    print("\nData is ready for backtesting strategies!")
    print(f"Columns available: {list(sp500.columns)}")
    
    

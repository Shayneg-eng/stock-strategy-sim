import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
print("Loading S&P 500 data...")
sp500 = pd.read_csv('sp500_data.csv', index_col=0, parse_dates=True)
sp500['Daily_Return'] = sp500['Close'].pct_change()

# Starting capital
STARTING_CAPITAL = 10000

# ============================================================================
# STRATEGY DEFINITIONS
# ============================================================================

def buy_and_hold(data, capital=STARTING_CAPITAL):
    """Buy on day 1 and hold forever"""
    shares = capital / data['Close'].iloc[0]
    portfolio_value = shares * data['Close']
    return portfolio_value

def moving_average_crossover(data, short_window=50, long_window=200, capital=STARTING_CAPITAL):
    """Buy when short MA crosses above long MA, sell when it crosses below"""
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']
    signals['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    signals['Long_MA'] = data['Close'].rolling(window=long_window).mean()
    
    # Generate signals (1 = buy, 0 = sell)
    signals['Signal'] = 0
    signals.loc[signals['Short_MA'] > signals['Long_MA'], 'Signal'] = 1
    
    # Calculate positions
    signals['Position'] = signals['Signal'].diff()
    
    # Backtest
    cash = capital
    shares = 0
    portfolio_values = []
    
    for i in range(len(signals)):
        if i < long_window:
            portfolio_values.append(capital)
            continue
            
        # Buy signal
        if signals['Position'].iloc[i] == 1 and cash > 0:
            shares = cash / signals['Close'].iloc[i]
            cash = 0
        
        # Sell signal
        elif signals['Position'].iloc[i] == -1 and shares > 0:
            cash = shares * signals['Close'].iloc[i]
            shares = 0
        
        # Calculate portfolio value
        portfolio_value = cash + (shares * signals['Close'].iloc[i])
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=data.index)

def momentum_strategy(data, lookback=90, capital=STARTING_CAPITAL):
    """Buy if price is up over lookback period, otherwise stay in cash"""
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']
    signals['Returns'] = data['Close'].pct_change(lookback)
    
    # Buy if positive momentum, sell otherwise
    signals['Signal'] = (signals['Returns'] > 0).astype(int)
    signals['Position'] = signals['Signal'].diff()
    
    cash = capital
    shares = 0
    portfolio_values = []
    
    for i in range(len(signals)):
        if i < lookback:
            portfolio_values.append(capital)
            continue
        
        # Buy signal
        if signals['Position'].iloc[i] == 1:
            shares = cash / signals['Close'].iloc[i]
            cash = 0
        
        # Sell signal
        elif signals['Position'].iloc[i] == -1:
            cash = shares * signals['Close'].iloc[i]
            shares = 0
        
        portfolio_value = cash + (shares * signals['Close'].iloc[i])
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=data.index)

def rsi_strategy(data, period=14, oversold=30, overbought=70, capital=STARTING_CAPITAL):
    """Buy when RSI < oversold, sell when RSI > overbought"""
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']
    signals['RSI'] = rsi
    
    cash = capital
    shares = 0
    portfolio_values = []
    
    for i in range(len(signals)):
        if i < period:
            portfolio_values.append(capital)
            continue
        
        # Buy when oversold
        if signals['RSI'].iloc[i] < oversold and cash > 0:
            shares = cash / signals['Close'].iloc[i]
            cash = 0
        
        # Sell when overbought
        elif signals['RSI'].iloc[i] > overbought and shares > 0:
            cash = shares * signals['Close'].iloc[i]
            shares = 0
        
        portfolio_value = cash + (shares * signals['Close'].iloc[i])
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=data.index)

def dollar_cost_averaging(data, monthly_investment=500, capital=STARTING_CAPITAL):
    """Invest fixed amount every month"""
    cash = capital
    shares = 0
    portfolio_values = []
    last_month = None
    
    for i in range(len(data)):
        current_month = data.index[i].month
        
        # Invest on first trading day of each month
        if last_month is None or current_month != last_month:
            shares += monthly_investment / data['Close'].iloc[i]
        
        portfolio_value = cash + (shares * data['Close'].iloc[i])
        portfolio_values.append(portfolio_value)
        last_month = current_month
    
    return pd.Series(portfolio_values, index=data.index)

def buy_the_dip(data, dip_threshold=-0.05, capital=STARTING_CAPITAL):
    """Buy when price drops by threshold% from recent high"""
    signals = pd.DataFrame(index=data.index)
    signals['Close'] = data['Close']
    signals['Rolling_Max'] = data['Close'].rolling(window=252).max()  # 1 year
    signals['Drawdown'] = (signals['Close'] - signals['Rolling_Max']) / signals['Rolling_Max']
    
    cash = capital
    shares = 0
    portfolio_values = []
    
    for i in range(len(signals)):
        if i < 252:
            portfolio_values.append(capital)
            continue
        
        # Buy the dip
        if signals['Drawdown'].iloc[i] < dip_threshold and cash > 0:
            shares = cash / signals['Close'].iloc[i]
            cash = 0
        
        # Sell when back at high
        elif signals['Drawdown'].iloc[i] >= -0.01 and shares > 0:
            cash = shares * signals['Close'].iloc[i]
            shares = 0
        
        portfolio_value = cash + (shares * signals['Close'].iloc[i])
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=data.index)

# ============================================================================
# RUN ALL STRATEGIES
# ============================================================================

print("\nRunning backtests...")

strategies = {
    'Buy and Hold': buy_and_hold(sp500),
    'MA Crossover (50/200)': moving_average_crossover(sp500, 50, 200),
    'MA Crossover (20/50)': moving_average_crossover(sp500, 20, 50),
    'Momentum (90 days)': momentum_strategy(sp500, 90),
    'Momentum (180 days)': momentum_strategy(sp500, 180),
    'RSI (14, 30/70)': rsi_strategy(sp500, 14, 30, 70),
    'RSI (14, 20/80)': rsi_strategy(sp500, 14, 20, 80),
    'Dollar Cost Avg ($500/mo)': dollar_cost_averaging(sp500, 500),
    'Buy the Dip (-5%)': buy_the_dip(sp500, -0.05),
    'Buy the Dip (-10%)': buy_the_dip(sp500, -0.10),
}

# ============================================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================================

results = pd.DataFrame()

for name, portfolio in strategies.items():
    final_value = portfolio.iloc[-1]
    total_return = (final_value - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    
    # Calculate daily returns
    daily_returns = portfolio.pct_change().dropna()
    
    # Annualized return
    years = len(portfolio) / 252
    annualized_return = ((final_value / STARTING_CAPITAL) ** (1/years) - 1) * 100
    
    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (assuming 2% risk-free rate)
    risk_free_rate = 0.02
    sharpe = (annualized_return - risk_free_rate*100) / volatility if volatility > 0 else 0
    
    # Max Drawdown
    cumulative = portfolio / portfolio.cummax()
    max_drawdown = ((cumulative.min() - 1) * 100)
    
    results[name] = {
        'Final Value': f'${final_value:,.2f}',
        'Total Return': f'{total_return:.2f}%',
        'Annualized Return': f'{annualized_return:.2f}%',
        'Volatility': f'{volatility:.2f}%',
        'Sharpe Ratio': f'{sharpe:.2f}',
        'Max Drawdown': f'{max_drawdown:.2f}%'
    }

results_df = pd.DataFrame(results).T

print("\n" + "="*80)
print("STRATEGY PERFORMANCE COMPARISON")
print("="*80)
print(results_df.to_string())
print("="*80)

# ============================================================================
# PLOT RESULTS
# ============================================================================

plt.figure(figsize=(15, 8))

for name, portfolio in strategies.items():
    plt.plot(portfolio.index, portfolio, label=name, linewidth=2)

plt.title('Strategy Performance Comparison (Starting Capital: $10,000)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see all strategies better
plt.tight_layout()
plt.savefig('strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nChart saved as 'strategy_comparison.png'")
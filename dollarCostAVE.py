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

def dollar_cost_averaging_fixed(data, monthly_investment=500, capital=STARTING_CAPITAL):
    """DCA with ONLY the starting capital, no additional money"""
    cash = capital
    shares = 0
    portfolio_values = []
    last_month = None
    
    for i in range(len(data)):
        current_month = data.index[i].month
        
        # Invest on first trading day of each month if we have cash
        if (last_month is None or current_month != last_month) and cash >= monthly_investment:
            shares += monthly_investment / data['Close'].iloc[i]
            cash -= monthly_investment
        
        portfolio_value = cash + (shares * data['Close'].iloc[i])
        portfolio_values.append(portfolio_value)
        last_month = current_month
    
    return pd.Series(portfolio_values, index=data.index), capital

def dollar_cost_averaging_continuous(data, monthly_investment=500, capital=STARTING_CAPITAL):
    """DCA with continuous monthly contributions (like a 401k)"""
    cash = capital
    shares = 0
    portfolio_values = []
    last_month = None
    total_invested = capital
    
    for i in range(len(data)):
        current_month = data.index[i].month
        
        # Add monthly investment (this is NEW money each month)
        if last_month is None or current_month != last_month:
            cash += monthly_investment
            total_invested += monthly_investment
        
        # Invest all available cash
        if cash > 0:
            shares += cash / data['Close'].iloc[i]
            cash = 0
        
        portfolio_value = shares * data['Close'].iloc[i]
        portfolio_values.append(portfolio_value)
        last_month = current_month
    
    return pd.Series(portfolio_values, index=data.index), total_invested

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
# ANALYSIS FUNCTION
# ============================================================================

def analyze_strategy(portfolio_values, name, total_invested):
    """Calculate comprehensive metrics for a strategy"""
    portfolio = pd.Series(portfolio_values)
    
    # Basic metrics
    final_value = portfolio.iloc[-1]
    total_return = (final_value - total_invested) / total_invested * 100
    
    # Annualized return
    years = len(portfolio) / 252
    annualized_return = ((final_value / total_invested) ** (1/years) - 1) * 100
    
    # Daily returns
    daily_returns = portfolio.pct_change().dropna()
    
    # Volatility (annualized)
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Sharpe Ratio (2% risk-free rate)
    sharpe = (annualized_return - 2) / volatility if volatility > 0 else 0
    
    # Max Drawdown
    running_max = portfolio.expanding().max()
    drawdown = (portfolio - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'Final Value': final_value,
        'Total Invested': total_invested,
        'Total Return %': total_return,
        'Annual Return %': annualized_return,
        'Volatility %': volatility,
        'Sharpe Ratio': sharpe,
        'Max Drawdown %': max_drawdown
    }

# ============================================================================
# RUN ALL STRATEGIES
# ============================================================================

print("\nRunning strategies...")

strategies = {}
total_investments = {}

# Buy and Hold
strategies['Buy and Hold'] = buy_and_hold(sp500)
total_investments['Buy and Hold'] = STARTING_CAPITAL

# MA Crossovers
strategies['MA Cross (50/200)'] = moving_average_crossover(sp500, 50, 200)
total_investments['MA Cross (50/200)'] = STARTING_CAPITAL

strategies['MA Cross (20/50)'] = moving_average_crossover(sp500, 20, 50)
total_investments['MA Cross (20/50)'] = STARTING_CAPITAL

# Momentum
strategies['Momentum (90d)'] = momentum_strategy(sp500, 90)
total_investments['Momentum (90d)'] = STARTING_CAPITAL

strategies['Momentum (180d)'] = momentum_strategy(sp500, 180)
total_investments['Momentum (180d)'] = STARTING_CAPITAL

# RSI
strategies['RSI (30/70)'] = rsi_strategy(sp500, 14, 30, 70)
total_investments['RSI (30/70)'] = STARTING_CAPITAL

strategies['RSI (20/80)'] = rsi_strategy(sp500, 14, 20, 80)
total_investments['RSI (20/80)'] = STARTING_CAPITAL

# Buy the Dip
strategies['Buy Dip (-5%)'] = buy_the_dip(sp500, -0.05)
total_investments['Buy Dip (-5%)'] = STARTING_CAPITAL

strategies['Buy Dip (-10%)'] = buy_the_dip(sp500, -0.10)
total_investments['Buy Dip (-10%)'] = STARTING_CAPITAL

# DCA - Fixed capital
dca_fixed, invested_fixed = dollar_cost_averaging_fixed(sp500, 500)
strategies['DCA (Fixed $10k)'] = dca_fixed
total_investments['DCA (Fixed $10k)'] = invested_fixed

# DCA - Continuous contributions
dca_continuous, invested_continuous = dollar_cost_averaging_continuous(sp500, 500)
strategies['DCA (Continuous $500/mo)'] = dca_continuous
total_investments['DCA (Continuous $500/mo)'] = invested_continuous

# ============================================================================
# RESULTS TABLE
# ============================================================================

results = []
for name in strategies.keys():
    metrics = analyze_strategy(strategies[name], name, total_investments[name])
    metrics['Strategy'] = name
    results.append(metrics)

results_df = pd.DataFrame(results)
results_df = results_df.set_index('Strategy')

# Sort by final value
results_df = results_df.sort_values('Final Value', ascending=False)

print("\n" + "="*120)
print("STRATEGY PERFORMANCE COMPARISON")
print("="*120)
print(results_df.to_string(float_format=lambda x: f'{x:,.2f}'))
print("="*120)

# Show the DCA investment breakdown
print("\n" + "="*100)
print("DOLLAR COST AVERAGING BREAKDOWN")
print("="*100)
years = len(sp500) / 252
months = years * 12
print(f"Time period: {years:.1f} years ({months:.0f} months)")
print(f"\nContinuous DCA:")
print(f"  Starting capital: ${STARTING_CAPITAL:,.2f}")
print(f"  Monthly investment: $500")
print(f"  Total months: {months:.0f}")
print(f"  Total invested: ${invested_continuous:,.2f}")
print(f"  Final value: ${strategies['DCA (Continuous $500/mo)'].iloc[-1]:,.2f}")
print(f"  Profit: ${strategies['DCA (Continuous $500/mo)'].iloc[-1] - invested_continuous:,.2f}")
print(f"  True ROI: {((strategies['DCA (Continuous $500/mo)'].iloc[-1] - invested_continuous) / invested_continuous * 100):.2f}%")
print("\nFixed DCA:")
print(f"  Starting capital: ${STARTING_CAPITAL:,.2f}")
print(f"  Monthly investment: $500")
print(f"  Total invested: ${invested_fixed:,.2f}")
print(f"  Final value: ${strategies['DCA (Fixed $10k)'].iloc[-1]:,.2f}")
print(f"  Profit: ${strategies['DCA (Fixed $10k)'].iloc[-1] - invested_fixed:,.2f}")
print(f"  True ROI: {((strategies['DCA (Fixed $10k)'].iloc[-1] - invested_fixed) / invested_fixed * 100):.2f}%")
print("="*100)

# ============================================================================
# COMPARISON PLOTS
# ============================================================================

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Plot 1: All strategies (absolute values)
for name, portfolio in strategies.items():
    if 'Continuous' not in name:  # Exclude continuous DCA from main plot
        ax1.plot(portfolio.index, portfolio, label=name, linewidth=2)

ax1.set_title('Strategy Performance - $10,000 Starting Capital Only', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Return on Investment %
for name, portfolio in strategies.items():
    if 'Continuous' not in name:
        roi = (portfolio / STARTING_CAPITAL - 1) * 100
        ax2.plot(portfolio.index, roi, label=name, linewidth=2)

ax2.set_title('Return on Investment % Over Time (Based on $10k Initial)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('ROI (%)', fontsize=12)
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('strategy_comparison_detailed.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nChart saved as 'strategy_comparison_detailed.png'")

# ============================================================================
# ADDITIONAL PLOT: Best strategies comparison
# ============================================================================

# Get top 5 strategies by final value (excluding continuous DCA)
top_strategies = results_df[~results_df.index.str.contains('Continuous')].nlargest(5, 'Final Value')

plt.figure(figsize=(15, 8))
for name in top_strategies.index:
    plt.plot(strategies[name].index, strategies[name], label=name, linewidth=2.5)

plt.title('Top 5 Strategies Comparison ($10,000 Starting Capital)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('top_5_strategies.png', dpi=300, bbox_inches='tight')
plt.show()

print("Top 5 strategies chart saved as 'top_5_strategies.png'")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*100)
print("QUICK SUMMARY (Strategies with $10k starting capital only)")
print("="*100)

summary = results_df[~results_df.index.str.contains('Continuous')].copy()
summary = summary.sort_values('Total Return %', ascending=False)

print("\nTop 3 by Total Return:")
for i, (name, row) in enumerate(summary.head(3).iterrows(), 1):
    print(f"{i}. {name}: ${row['Final Value']:,.2f} ({row['Total Return %']:.2f}% return)")

print("\nTop 3 by Sharpe Ratio (risk-adjusted return):")
summary_sharpe = summary.sort_values('Sharpe Ratio', ascending=False)
for i, (name, row) in enumerate(summary_sharpe.head(3).iterrows(), 1):
    print(f"{i}. {name}: Sharpe {row['Sharpe Ratio']:.2f}, Return {row['Total Return %']:.2f}%")

print("\nLowest Drawdown (most stable):")
summary_dd = summary.sort_values('Max Drawdown %', ascending=False)
for i, (name, row) in enumerate(summary_dd.head(3).iterrows(), 1):
    print(f"{i}. {name}: {row['Max Drawdown %']:.2f}% drawdown, ${row['Final Value']:,.2f} final")

print("="*100)
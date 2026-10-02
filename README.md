# Stock Strategy Simulator

Backtests and compares simple S&P 500 trading strategies (buy-and-hold, dollar-cost averaging,
and others), producing comparison charts.

## Files
| File | Role |
|---|---|
| `downloadSnP.py` | Download S&P 500 price history → `sp500_data.csv` |
| `dollarCostAVE.py` | Dollar-cost-averaging simulation |
| `simpleTradingSim.py` | Strategy backtest + comparison |

Generated charts: `strategy_comparison*.png`, `top_5_strategies.png`.

## Run it
```bash
python -m pip install pandas numpy matplotlib yfinance
python downloadSnP.py
python simpleTradingSim.py
```

> For education/backtesting only — not investment advice.

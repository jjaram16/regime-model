# Market Regime Classification

Machine learning project that predicts high volatility market periods and uses those predictions to adjust a simple stock and bond portfolio.

## How to run
pip install yfinance scikit-learn matplotlib pandas numpy requests
python regime_model.py

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html and replace the placeholder in regime_model.py.

## Data sources
- Yahoo Finance (SPY, IEF, VIX)
- FRED API (Treasury yields, unemployment rate)

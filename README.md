MEXC Scalp Trading Dashboard
This Streamlit dashboard provides real-time analysis of MEXC USDT trading pairs for scalp trading opportunities. It fetches coin data from the MEXC API, calculates custom metrics, and visualizes the top candidates for short-term trading.

Features
API Integration: Fetches live coin data from multiple MEXC endpoints with robust error handling.
Custom Scoring: Calculates scalp scores based on volatility, volume, spread, momentum, and volume trend.
Configurable Analysis: Adjust budget, minimum volume, number of coins, and scoring weights via sidebar controls.
Visualization: Interactive charts and tables for top coins, score distribution, and performance analytics.
Export: Download results as CSV, JSON, or bot configuration files.
Auto-Refresh: Optionally refreshes analysis at set intervals.
Usage
Install dependencies:


pip install streamlit pandas plotly numpy requests
Run the dashboard:


streamlit run mexc_dashboard.py
Configure analysis:

Set trading budget, minimum volume, and number of coins to analyze.
Adjust scoring weights for volatility, volume, spread, momentum, and volume trend.
Enable auto-refresh if desired.
Start analysis:

Click "🚀 Start Analysis" to fetch and analyze coins.
View results, charts, and export data as needed.
File Overview
mexc_dashboard.py: Main Streamlit dashboard file containing all logic, UI, and coin analysis classes.
Notes
If MEXC API endpoints fail, the dashboard falls back to hardcoded popular USDT pairs and mock data.
This dashboard is for informational and research purposes only. Trading cryptocurrencies involves significant risk.
Screenshots
(Add screenshots of the dashboard UI and charts here)

License
MIT License

Disclaimer: This tool does not execute trades and is not financial advice. Always do your own research before trading.

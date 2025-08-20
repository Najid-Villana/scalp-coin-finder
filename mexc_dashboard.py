import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from datetime import datetime, timedelta
import asyncio
import json
from typing import List, Dict
import requests
from dataclasses import asdict

# Import the coin finder module (assuming it's in the same directory)
# from mexc_coin_finder import MEXCCoinFinder, CoinMetrics, ScalpBotFeeder

# For demo purposes, I'll include the essential classes here
# In production, you'd import from the separate module

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class CoinMetrics:
    """Data class to store coin trading metrics"""
    symbol: str
    price: float
    volume_24h: float
    price_change_24h: float
    volatility: float
    spread: float
    volume_trend: float
    momentum: float
    scalp_score: float
    tradable: bool

class MEXCCoinFinder:
    """Simplified version for dashboard integration"""
    
    def __init__(self, budget: float = 1000.0, min_volume: float = 100000):
        self.budget = budget
        self.min_volume = min_volume
        # Try multiple base URLs for MEXC API
        self.base_urls = [
            "https://api.mexc.com",
            "https://www.mexc.com/open/api",
            "https://api.mexc.com/api"
        ]
        self.current_base_url = self.base_urls[0]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        self.weights = {
            'volatility': 0.25,
            'volume': 0.20,
            'spread': 0.15,
            'momentum': 0.20,
            'volume_trend': 0.20
        }
    
    def fetch_mexc_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch data from MEXC API with error handling and multiple URL attempts"""
        for base_url in self.base_urls:
            try:
                url = f"{base_url}{endpoint}"
                st.write(f"🔍 Trying: {url}")  # Debug info
                response = self.session.get(url, params=params, timeout=15)
                
                st.write(f"📡 Response status: {response.status_code}")  # Debug info
                
                if response.status_code == 200:
                    data = response.json()
                    st.write(f"✅ Success with {base_url}")  # Debug info
                    self.current_base_url = base_url  # Remember working URL
                    return data
                else:
                    st.write(f"❌ Failed with status {response.status_code}")
                    continue
                    
            except requests.exceptions.RequestException as e:
                st.write(f"❌ Request failed for {base_url}: {str(e)}")
                continue
            except json.JSONDecodeError as e:
                st.write(f"❌ JSON decode error for {base_url}: {str(e)}")
                continue
            except Exception as e:
                st.write(f"❌ Unexpected error for {base_url}: {str(e)}")
                continue
        
        st.error("❌ All API endpoints failed")
        return None
    
    def get_usdt_pairs(self) -> List[str]:
        """Fetch all USDT trading pairs from MEXC with multiple endpoint attempts"""
        st.write("🔍 **API Testing Debug Info:**")
        
        # Try different endpoint variations
        endpoints_to_try = [
            "/api/v3/exchangeInfo",
            "/api/v3/ticker/24hr",  # This might work and give us symbols
            "/open/api/v2/market/symbols",  # Alternative endpoint
            "/api/v1/exchangeInfo"  # Older version
        ]
        
        for endpoint in endpoints_to_try:
            st.write(f"Trying endpoint: {endpoint}")
            try:
                data = self.fetch_mexc_data(endpoint)
                if data:
                    st.write(f"✅ Got response from {endpoint}")
                    st.write(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # Handle different response formats
                    if endpoint == "/api/v3/ticker/24hr" and isinstance(data, list):
                        # Extract symbols from ticker data
                        usdt_pairs = [item['symbol'] for item in data if item['symbol'].endswith('USDT')]
                        if usdt_pairs:
                            st.success(f"✅ Found {len(usdt_pairs)} USDT pairs from ticker endpoint")
                            return usdt_pairs[:200]  # Limit for testing
                    
                    elif 'symbols' in data:
                        # Standard exchange info format
                        usdt_pairs = []
                        for symbol_info in data.get('symbols', []):
                            if isinstance(symbol_info, dict):
                                symbol = symbol_info.get('symbol', '')
                                status = symbol_info.get('status', 'TRADING')
                                if symbol.endswith('USDT') and status == 'TRADING' and symbol != 'USDT':
                                    usdt_pairs.append(symbol)
                        
                        if usdt_pairs:
                            st.success(f"✅ Found {len(usdt_pairs)} USDT pairs from symbols")
                            return usdt_pairs
                    
                    elif 'data' in data and isinstance(data['data'], list):
                        # Alternative format with data array
                        usdt_pairs = []
                        for item in data['data']:
                            if isinstance(item, dict) and 'symbol' in item:
                                symbol = item['symbol']
                                if symbol.endswith('USDT') and symbol != 'USDT':
                                    usdt_pairs.append(symbol)
                        
                        if usdt_pairs:
                            st.success(f"✅ Found {len(usdt_pairs)} USDT pairs from data array")
                            return usdt_pairs
                    
                    else:
                        st.write(f"⚠️ Unexpected response format from {endpoint}")
                        if isinstance(data, dict) and len(data) < 10:  # Small response, show it
                            st.json(data)
                
            except Exception as e:
                st.write(f"❌ Error with {endpoint}: {str(e)}")
                continue
        
        # If all API attempts fail, use hardcoded popular pairs
        st.warning("⚠️ All API endpoints failed. Using hardcoded popular USDT pairs.")
        fallback_pairs = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT',
            'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'MATICUSDT', 'LTCUSDT', 'UNIUSDT',
            'LINKUSDT', 'BCHUSDT', 'XLMUSDT', 'VETUSDT', 'FILUSDT', 'TRXUSDT',
            'ETCUSDT', 'ATOMUSDT', 'ALGOUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT'
        ]
        return fallback_pairs
    
    def analyze_coin_simple(self, symbol: str) -> Optional[CoinMetrics]:
        """Simplified coin analysis for dashboard with better error handling"""
        try:
            # Try current base URL first, then fallback to others
            ticker = None
            for base_url in [self.current_base_url] + [url for url in self.base_urls if url != self.current_base_url]:
                try:
                    url = f"{base_url}/api/v3/ticker/24hr"
                    response = self.session.get(url, params={'symbol': symbol}, timeout=10)
                    if response.status_code == 200:
                        ticker = response.json()
                        break
                except:
                    continue
            
            if not ticker:
                # Generate mock data for testing if API fails
                st.write(f"⚠️ Using mock data for {symbol}")
                price = np.random.uniform(0.001, 1000)
                volume_24h = np.random.uniform(10000, 1000000)
                price_change_24h = np.random.uniform(-10, 10)
            else:
                price = float(ticker.get('lastPrice', 0))
                volume_24h = float(ticker.get('quoteVolume', 0))
                price_change_24h = float(ticker.get('priceChangePercent', 0))
            
            # Skip coins with zero price or volume
            if price <= 0:
                return None
            
            # Simplified calculations
            volatility = abs(price_change_24h) / 100
            spread = np.random.uniform(0.1, 0.5)  # Mock spread
            momentum = abs(price_change_24h)
            volume_trend = np.random.uniform(-10, 20)  # Mock volume trend
            
            # More lenient tradability check
            min_order_value = 5  # Reduced minimum order value
            max_coins_affordable = self.budget / price if price > 0 else 0
            
            tradable = (
                price > 0 and 
                volume_24h >= self.min_volume and 
                max_coins_affordable * price >= min_order_value and
                price <= self.budget  # Allow using full budget for one coin
            )
            
            # Calculate score
            volatility_score = min(volatility * 100, 100)
            volume_score = min(np.log10(max(volume_24h, 1) / 1000) * 15, 100)  # More lenient volume scoring
            spread_score = max(100 - spread * 10, 0)
            momentum_score = min(momentum * 2, 100)
            volume_trend_score = min(max(volume_trend, 0) * 2, 100)
            
            scalp_score = (
                volatility_score * self.weights['volatility'] +
                volume_score * self.weights['volume'] +
                spread_score * self.weights['spread'] +
                momentum_score * self.weights['momentum'] +
                volume_trend_score * self.weights['volume_trend']
            )
            
            return CoinMetrics(
                symbol=symbol,
                price=price,
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                volatility=volatility,
                spread=spread,
                volume_trend=volume_trend,
                momentum=momentum,
                scalp_score=round(scalp_score, 2),
                tradable=tradable
            )
            
        except Exception as e:
            st.write(f"❌ Error analyzing {symbol}: {e}")
            return None

# Streamlit Dashboard Configuration
st.set_page_config(
    page_title="MEXC Scalp Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-running {
        color: #28a745;
        font-weight: bold;
    }
    .status-stopped {
        color: #dc3545;
        font-weight: bold;
    }
    .high-score {
        background-color: #d4edda;
        color: #155724;
    }
    .medium-score {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-score {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'coin_finder' not in st.session_state:
    st.session_state.coin_finder = None
if 'analyzed_coins' not in st.session_state:
    st.session_state.analyzed_coins = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Trading Parameters
    budget = st.number_input(
        "Trading Budget ($)",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=100.0,
        help="Maximum budget for trading"
    )
    
    min_volume = st.number_input(
        "Minimum 24h Volume ($)",
        min_value=1000.0,
        max_value=10000000.0,
        value=10000.0,  # Reduced default minimum volume
        step=1000.0,
        help="Minimum 24h volume threshold"
    )
    
    max_coins = st.slider(
        "Max Coins to Analyze",
        min_value=10,
        max_value=200,
        value=50,
        help="Maximum number of coins to analyze"
    )
    
    top_n = st.slider(
        "Top Coins to Display",
        min_value=5,
        max_value=50,
        value=20,
        help="Number of top coins to display"
    )
    
    st.divider()
    
    # Scoring Weights
    st.subheader("📊 Scoring Weights")
    vol_weight = st.slider("Volatility", 0.0, 1.0, 0.25, 0.05)
    volume_weight = st.slider("Volume", 0.0, 1.0, 0.20, 0.05)
    spread_weight = st.slider("Spread", 0.0, 1.0, 0.15, 0.05)
    momentum_weight = st.slider("Momentum", 0.0, 1.0, 0.20, 0.05)
    trend_weight = st.slider("Volume Trend", 0.0, 1.0, 0.20, 0.05)
    
    # Normalize weights
    total_weight = vol_weight + volume_weight + spread_weight + momentum_weight + trend_weight
    if total_weight > 0:
        weights = {
            'volatility': vol_weight / total_weight,
            'volume': volume_weight / total_weight,
            'spread': spread_weight / total_weight,
            'momentum': momentum_weight / total_weight,
            'volume_trend': trend_weight / total_weight
        }
    else:
        weights = {'volatility': 0.25, 'volume': 0.20, 'spread': 0.15, 'momentum': 0.20, 'volume_trend': 0.20}
    
    st.divider()
    
    # Auto-refresh settings
    st.subheader("🔄 Auto Refresh")
    auto_refresh = st.checkbox("Enable Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    if auto_refresh:
        refresh_interval = st.selectbox(
            "Refresh Interval",
            options=[60, 300, 600, 1800],
            index=1,
            format_func=lambda x: f"{x//60} minutes" if x >= 60 else f"{x} seconds"
        )
    
    st.divider()
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_analysis = st.button("🚀 Start Analysis", type="primary", use_container_width=True)
    with col2:
        stop_analysis = st.button("⏹️ Stop", use_container_width=True)

# Main Dashboard
st.markdown('<h1 class="main-header">📈 MEXC Scalp Trading Dashboard</h1>', unsafe_allow_html=True)

# Status Bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.session_state.last_update:
        status = "🟢 Running" if auto_refresh else "🟡 Manual"
        st.markdown(f"**Status:** {status}")
    else:
        st.markdown("**Status:** 🔴 Not Started")

with col2:
    if st.session_state.last_update:
        st.markdown(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.markdown("**Last Update:** Never")

with col3:
    st.markdown(f"**Budget:** ${budget:,.0f}")

with col4:
    if st.session_state.analyzed_coins:
        tradable_count = sum(1 for coin in st.session_state.analyzed_coins if coin.tradable)
        st.markdown(f"**Tradable Coins:** {tradable_count}")
    else:
        st.markdown("**Tradable Coins:** 0")

# Start Analysis
if start_analysis:
    with st.spinner("Initializing coin finder..."):
        st.session_state.coin_finder = MEXCCoinFinder(budget=budget, min_volume=min_volume)
        st.session_state.coin_finder.weights = weights
    
    with st.spinner(f"Analyzing top {max_coins} USDT pairs..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        info_text = st.empty()
        
        # Get symbols
        symbols = st.session_state.coin_finder.get_usdt_pairs()
        
        if not symbols:
            st.error("❌ Failed to fetch USDT pairs from MEXC API. Please check your internet connection.")
        else:
            info_text.info(f"Found {len(symbols)} USDT pairs. Analyzing top {max_coins}...")
            symbols = symbols[:max_coins]
            analyzed_coins = []
            failed_count = 0
            
            for i, symbol in enumerate(symbols):
                status_text.text(f"Analyzing {symbol}... ({i+1}/{len(symbols)})")
                coin = st.session_state.coin_finder.analyze_coin_simple(symbol)
                if coin:
                    analyzed_coins.append(coin)
                else:
                    failed_count += 1
                
                progress_bar.progress((i + 1) / len(symbols))
                
                # Add small delay to avoid rate limits
                if i % 10 == 0:  # Every 10 requests
                    time.sleep(0.5)
            
            st.session_state.analyzed_coins = analyzed_coins
            st.session_state.last_update = datetime.now()
            
            progress_bar.empty()
            status_text.empty()
            
            # Show detailed results
            total_analyzed = len(analyzed_coins)
            tradable_count = sum(1 for coin in analyzed_coins if coin.tradable)
            
            if total_analyzed == 0:
                st.error("❌ No coins could be analyzed. This might be due to API connectivity issues.")
                info_text.error("Try reducing the minimum volume threshold or check your internet connection.")
            elif tradable_count == 0:
                st.warning(f"⚠️ Analyzed {total_analyzed} coins but none met the tradability criteria.")
                info_text.info(f"""
                **Suggestions to find tradable coins:**
                - Reduce minimum volume to ${min_volume/10:,.0f} or lower
                - Increase budget to ${budget*2:,.0f} or higher
                - Current criteria: Price ≤ ${budget:,.0f}, Volume ≥ ${min_volume:,.0f}
                """)
            else:
                st.success(f"✅ Analysis completed! Found {total_analyzed} coins, {tradable_count} are tradable.")
                info_text.success(f"Failed to analyze {failed_count} coins due to API issues.")
            
            # Debug information
            if analyzed_coins:
                with st.expander("🔍 Debug Information"):
                    st.write("**Sample of analyzed coins:**")
                    debug_df = pd.DataFrame([
                        {
                            'Symbol': coin.symbol,
                            'Price': f"${coin.price:.6f}",
                            'Volume': f"${coin.volume_24h:,.0f}",
                            'Tradable': '✅' if coin.tradable else '❌',
                            'Score': f"{coin.scalp_score:.1f}"
                        }
                        for coin in analyzed_coins[:10]
                    ])
                    st.dataframe(debug_df, use_container_width=True)
                    
                    # Show filtering criteria
                    st.write("**Current Filtering Criteria:**")
                    st.write(f"- Budget: ${budget:,.0f}")
                    st.write(f"- Min Volume: ${min_volume:,.0f}")
                    st.write(f"- Max Price per coin: ${budget:,.0f} (full budget)")
                    st.write(f"- Min Order Value: $5")
            
            info_text.empty()

# Display Results
if st.session_state.analyzed_coins:
    # Filter and sort coins
    tradable_coins = [coin for coin in st.session_state.analyzed_coins if coin.tradable]
    top_coins = sorted(tradable_coins, key=lambda x: x.scalp_score, reverse=True)[:top_n]
    
    if not top_coins:
        st.warning("No tradable coins found with current criteria.")
    else:
        # Summary metrics
        st.subheader("📊 Summary Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_score = np.mean([coin.scalp_score for coin in top_coins])
            st.metric("Avg Scalp Score", f"{avg_score:.1f}")
        
        with col2:
            total_volume = sum(coin.volume_24h for coin in top_coins)
            st.metric("Total 24h Volume", f"${total_volume/1e6:.1f}M")
        
        with col3:
            avg_volatility = np.mean([coin.volatility for coin in top_coins])
            st.metric("Avg Volatility", f"{avg_volatility:.3f}")
        
        with col4:
            avg_spread = np.mean([coin.spread for coin in top_coins])
            st.metric("Avg Spread", f"{avg_spread:.2f}%")
        
        with col5:
            positive_momentum = sum(1 for coin in top_coins if coin.momentum > 0)
            st.metric("Positive Momentum", f"{positive_momentum}/{len(top_coins)}")
        
        # Charts
        st.subheader("📈 Analytics")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Top Coins", "Score Distribution", "Volume vs Price", "Performance Matrix"])
        
        with tab1:
            # Top coins table with styling
            df = pd.DataFrame([asdict(coin) for coin in top_coins])
            
            def style_score(val):
                if val >= 80:
                    return 'background-color: #d4edda; color: #155724'
                elif val >= 60:
                    return 'background-color: #fff3cd; color: #856404'
                else:
                    return 'background-color: #f8d7da; color: #721c24'
            
            styled_df = df.style.format({
                'price': '${:.6f}',
                'volume_24h': '${:,.0f}',
                'price_change_24h': '{:.2f}%',
                'volatility': '{:.4f}',
                'spread': '{:.2f}%',
                'momentum': '{:.2f}',
                'volume_trend': '{:.2f}%',
                'scalp_score': '{:.1f}'
            }).applymap(style_score, subset=['scalp_score'])
            
            st.dataframe(styled_df, use_container_width=True)
        
        with tab2:
            # Score distribution
            fig = px.histogram(
                x=[coin.scalp_score for coin in top_coins],
                nbins=10,
                title="Scalp Score Distribution",
                labels={'x': 'Scalp Score', 'y': 'Count'}
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Volume vs Price scatter
            fig = px.scatter(
                x=[coin.volume_24h for coin in top_coins],
                y=[coin.price for coin in top_coins],
                size=[coin.scalp_score for coin in top_coins],
                color=[coin.scalp_score for coin in top_coins],
                hover_name=[coin.symbol for coin in top_coins],
                title="Volume vs Price (Size = Scalp Score)",
                labels={'x': '24h Volume ($)', 'y': 'Price ($)', 'color': 'Scalp Score'},
                color_continuous_scale='viridis'
            )
            fig.update_xaxes(type="log")
            fig.update_yaxes(type="log")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Performance matrix
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Volatility vs Score', 'Volume vs Score', 'Momentum vs Score', 'Spread vs Score'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Volatility vs Score
            fig.add_trace(
                go.Scatter(
                    x=[coin.volatility for coin in top_coins],
                    y=[coin.scalp_score for coin in top_coins],
                    mode='markers',
                    name='Volatility',
                    text=[coin.symbol for coin in top_coins]
                ),
                row=1, col=1
            )
            
            # Volume vs Score
            fig.add_trace(
                go.Scatter(
                    x=[coin.volume_24h for coin in top_coins],
                    y=[coin.scalp_score for coin in top_coins],
                    mode='markers',
                    name='Volume',
                    text=[coin.symbol for coin in top_coins]
                ),
                row=1, col=2
            )
            
            # Momentum vs Score
            fig.add_trace(
                go.Scatter(
                    x=[coin.momentum for coin in top_coins],
                    y=[coin.scalp_score for coin in top_coins],
                    mode='markers',
                    name='Momentum',
                    text=[coin.symbol for coin in top_coins]
                ),
                row=2, col=1
            )
            
            # Spread vs Score
            fig.add_trace(
                go.Scatter(
                    x=[coin.spread for coin in top_coins],
                    y=[coin.scalp_score for coin in top_coins],
                    mode='markers',
                    name='Spread',
                    text=[coin.symbol for coin in top_coins]
                ),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Performance Matrix")
            st.plotly_chart(fig, use_container_width=True)
        
        # Export functionality
        st.subheader("📤 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to CSV"):
                df = pd.DataFrame([asdict(coin) for coin in top_coins])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"scalp_coins_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Export to JSON"):
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'budget': budget,
                    'coins': [asdict(coin) for coin in top_coins]
                }
                json_str = json.dumps(data, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"scalp_coins_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            # Trading bot config
            if st.button("🤖 Bot Config"):
                bot_config = {
                    'symbols': [coin.symbol for coin in top_coins[:10]],
                    'budget_per_coin': budget / min(10, len(top_coins)),
                    'min_score': 60,
                    'max_spread': 0.5,
                    'update_interval': 300
                }
                config_str = json.dumps(bot_config, indent=2)
                st.download_button(
                    label="Download Bot Config",
                    data=config_str,
                    file_name=f"bot_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Auto-refresh logic
if auto_refresh and st.session_state.last_update:
    time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
    if time_since_update >= refresh_interval:
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>MEXC Scalp Trading Dashboard | Real-time coin analysis for scalp trading opportunities</p>
        <p>⚠️ Trading cryptocurrencies involves significant risk. Always do your own research.</p>
    </div>
    """,
    unsafe_allow_html=True
)

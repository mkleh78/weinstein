import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import warnings
import traceback
import logging
import os

# Set up logging
log_filename = f"weinstein_analyzer_{datetime.datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('WeinsteinAnalyzer')
warnings.filterwarnings('ignore')

class WeinsteinTickerAnalyzer:
    def __init__(self):
        """Initializes the Weinstein Ticker Analyzer"""
        self.data = None
        self.ticker_symbol = None
        self.period = "1y"
        self.interval = "1wk"
        self.indicators = {}
        self.phase = 0
        self.phase_desc = ""
        self.recommendation = ""
        self.detailed_analysis = ""
        self.last_price = None
        self.market_context = None
        self.sector_data = None
        self.support_resistance_levels = []
        self.errors = []
        self.warnings = []
        self.ticker_info = {}
        
    def load_data(self, ticker, period="1y", interval="1wk"):
        """Load data for a specific ticker with enhanced error handling"""
        logger.info(f"Loading data for {ticker} with period={period}, interval={interval}")
        self.ticker_symbol = ticker
        self.period = period
        self.interval = interval
        self.errors = []
        self.warnings = []
        
        # Clear previous data
        self.data = None
        self.phase = 0
        self.phase_desc = ""
        self.recommendation = ""
        self.detailed_analysis = ""
        self.last_price = None
        self.market_context = None
        self.sector_data = None
        self.support_resistance_levels = []
        
        try:
            # Normalize ticker
            normalized_ticker = self._normalize_ticker(ticker)
            
            # Load data with retry mechanism and adaptable periods
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        normalized_ticker, 
                        period=period, 
                        interval=interval, 
                        progress=False
                    )
                    
                    if len(data) > 0:
                        break
                    elif attempt < max_retries - 1:
                        # Try with a shorter period if no data is returned
                        if period == "5y":
                            period = "2y"
                        elif period == "2y":
                            period = "1y"
                        elif period == "1y":
                            period = "6mo"
                        elif period == "6mo":
                            period = "3mo"
                        elif period == "3mo":
                            period = "1mo"
                        else:
                            # If we're already at the shortest period, try a different interval
                            if interval == "1wk":
                                interval = "1d"
                                period = "1mo"  # Reset period for daily data
                            
                        logger.warning(f"Attempt {attempt+1}: No data returned for {ticker}. Trying with period={period}, interval={interval}")
                    else:
                        logger.error(f"No data found for {ticker} after {max_retries} attempts")
                        self.errors.append(f"No data found for {ticker}. The ticker may not exist or may not have data for the requested period.")
                        return False
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying...")
                    else:
                        raise e
            
            # Update the instance variables with actual values used
            self.period = period
            self.interval = interval
            
            # Ensure index is not a MultiIndex
            if isinstance(data.index, pd.MultiIndex):
                data = data.reset_index(level=0, drop=True)
            
            # Check if we have enough data
            if len(data) < 5:  # Need at least 5 data points for minimal analysis
                logger.warning(f"Insufficient data for {ticker}: only {len(data)} data points")
                self.warnings.append(f"Limited data available for {ticker}: only {len(data)} data points. Analysis may be less reliable.")
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.warning(f"Missing columns: {', '.join(missing_columns)} in data for {ticker}")
                self.warnings.append(f"Missing data: {', '.join(missing_columns)}. Analysis may be limited.")
                
                # Create missing columns with default values to allow analysis to proceed
                for col in missing_columns:
                    if col == 'Volume':  # Default volume to 0
                        data[col] = 0
                    elif col in ['Open', 'High', 'Low']:  # Use Close for missing price columns
                        if 'Close' in data.columns:
                            data[col] = data['Close']
                        else:
                            # If even Close is missing, we can't proceed
                            logger.error(f"Critical data missing for {ticker}: no price data available")
                            self.errors.append("No price data available for analysis.")
                            return False
            
            # Try to get ticker info for better context
            try:
                ticker_obj = yf.Ticker(normalized_ticker)
                info = ticker_obj.info
                if info:
                    # Extract useful fields if they exist
                    useful_fields = ['shortName', 'longName', 'sector', 'industry', 
                                    'exchange', 'currency', 'country', 'market']
                    self.ticker_info = {k: info[k] for k in useful_fields if k in info}
                    logger.info(f"Successfully retrieved info for {ticker}")
            except Exception as e:
                logger.warning(f"Could not retrieve ticker info: {str(e)}")
                # Not critical, we can continue without it
            
            # Bind the data
            self.data = data
            
            # Safe extraction of last price
            try:
                if isinstance(self.data['Close'].iloc[-1], (pd.Series, pd.DataFrame)):
                    self.last_price = float(self.data['Close'].iloc[-1].iloc[0])
                else:
                    self.last_price = float(self.data['Close'].iloc[-1])
            except Exception as e:
                logger.error(f"Error extracting last price: {str(e)}")
                self.warnings.append("Could not determine the last price.")
                self.last_price = None
                
            # Calculate all indicators - adaptive to available data
            self.calculate_indicators()
            
            # Find support and resistance levels
            self.identify_support_resistance()
            
            # Load market context - skip for indices to avoid self-comparison
            if not self._is_index(normalized_ticker):
                self.load_market_context()
            
            # Identify the phase
            self.identify_phase()
            
            # Generate recommendation
            self.generate_recommendation()
            
            # Generate detailed analysis text
            self.generate_detailed_analysis()
            
            logger.info(f"Successfully loaded and analyzed data for {ticker}")
            return True
                
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            traceback.print_exc()
            self.errors.append(f"Error analyzing {ticker}: {str(e)}")
            return False
    
    def _normalize_ticker(self, ticker):
        """Normalize ticker symbols to be compatible with yfinance"""
        # Remove any whitespace
        ticker = ticker.strip()
        
        # Convert to uppercase
        ticker = ticker.upper()
        
        # Handle special cases for different exchanges
        # For European tickers that might use '.' instead of ',' for decimal
        ticker = ticker.replace(',', '.')
        
        # For indices, ensure proper prefix
        if ticker.startswith('^'):
            return ticker
        
        # For common indices, add the ^ prefix if missing
        common_indices = {
            'SPX': '^GSPC',  # S&P 500
            'DJI': '^DJI',   # Dow Jones
            'IXIC': '^IXIC', # NASDAQ
            'RUT': '^RUT',   # Russell 2000
            'GSPC': '^GSPC', # S&P 500
            'NDX': '^NDX',   # NASDAQ-100
            'VIX': '^VIX'    # Volatility Index
        }
        
        if ticker in common_indices:
            return common_indices[ticker]
        
        return ticker
    
    def _is_index(self, ticker):
        """Check if the ticker is an index to avoid self-comparison in market context"""
        # Common indices usually start with ^
        if ticker.startswith('^'):
            return True
        
        # Check if it's one of the known indices
        known_indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^NDX', '^VIX']
        if ticker in known_indices:
            return True
            
        # If we have ticker info, check if it's an index
        if 'market' in self.ticker_info and 'index' in self.ticker_info['market'].lower():
            return True
            
        return False
    
    def load_market_context(self):
        """Load market context data (S&P 500 index) for relative analysis"""
        try:
            market_data = yf.download(
                "^GSPC",  # S&P 500 index
                period=self.period,
                interval=self.interval,
                progress=False
            )
            
            if len(market_data) > 0:
                # Calculate market indicators
                market_data['MA30'] = market_data['Close'].rolling(window=30).mean()
                market_data['MA30_Slope'] = market_data['MA30'].diff()
                
                # Determine market phase - Use scalar values to avoid Series comparison issues
                current = market_data.iloc[-1]
                
                # Ensure we have scalar values, not Series
                try:
                    current_close = float(current['Close']) if not pd.isna(current['Close']).any() else 0
                    current_ma30 = float(current['MA30']) if not pd.isna(current['MA30']).any() else 0
                    current_ma30_slope = float(current['MA30_Slope']) if not pd.isna(current['MA30_Slope']).any() else 0
                except Exception:
                    # Handle potential Series objects
                    if isinstance(current['Close'], pd.Series):
                        current_close = float(current['Close'].iloc[0]) if not current['Close'].empty and not pd.isna(current['Close'].iloc[0]) else 0
                    else:
                        current_close = float(current['Close']) if not pd.isna(current['Close']) else 0
                        
                    if isinstance(current['MA30'], pd.Series):
                        current_ma30 = float(current['MA30'].iloc[0]) if not current['MA30'].empty and not pd.isna(current['MA30'].iloc[0]) else 0
                    else:
                        current_ma30 = float(current['MA30']) if not pd.isna(current['MA30']) else 0
                        
                    if isinstance(current['MA30_Slope'], pd.Series):
                        current_ma30_slope = float(current['MA30_Slope'].iloc[0]) if not current['MA30_Slope'].empty and not pd.isna(current['MA30_Slope'].iloc[0]) else 0
                    else:
                        current_ma30_slope = float(current['MA30_Slope']) if not pd.isna(current['MA30_Slope']) else 0
                
                price_above_ma = current_close > current_ma30
                ma_slope_positive = current_ma30_slope > 0
                
                if price_above_ma and ma_slope_positive:
                    market_phase = 2  # Uptrend
                elif price_above_ma and not ma_slope_positive:
                    market_phase = 3  # Top formation
                elif not price_above_ma and not ma_slope_positive:
                    market_phase = 4  # Downtrend
                else:
                    market_phase = 1  # Base formation
                
                # Get market performance metrics
                if len(market_data) >= 4:  # At least 4 weeks of data
                    market_1month_perf = (float(market_data['Close'].iloc[-1]) / float(market_data['Close'].iloc[-4]) - 1) * 100
                else:
                    market_1month_perf = 0
                
                # Store market context
                self.market_context = {
                    'phase': market_phase,
                    'last_close': float(market_data['Close'].iloc[-1]),
                    'performance_1month': market_1month_perf
                }
                
                logger.info(f"Market context loaded: Phase {market_phase}")
                
                # Try to load sector data if available
                try:
                    # Get ticker info to identify sector
                    ticker_info = yf.Ticker(self.ticker_symbol).info
                    if 'sector' in ticker_info:
                        sector = ticker_info['sector']
                        # Map sectors to sector ETFs
                        sector_etfs = {
                            'Technology': 'XLK',
                            'Financial Services': 'XLF',
                            'Healthcare': 'XLV',
                            'Consumer Cyclical': 'XLY',
                            'Industrials': 'XLI',
                            'Communication Services': 'XLC',
                            'Consumer Defensive': 'XLP',
                            'Energy': 'XLE',
                            'Basic Materials': 'XLB',
                            'Real Estate': 'XLRE',
                            'Utilities': 'XLU'
                        }
                        
                        if sector in sector_etfs:
                            sector_etf = sector_etfs[sector]
                            sector_data = yf.download(
                                sector_etf,
                                period=self.period,
                                interval=self.interval,
                                progress=False
                            )
                            
                            if len(sector_data) > 0:
                                # Calculate sector indicators
                                sector_data['MA30'] = sector_data['Close'].rolling(window=30).mean()
                                sector_data['MA30_Slope'] = sector_data['MA30'].diff()
                                
                                # Determine sector phase
                                current_sector = sector_data.iloc[-1]
                                sector_price_above_ma = current_sector['Close'] > current_sector['MA30']
                                sector_ma_slope_positive = current_sector['MA30_Slope'] > 0
                                
                                if sector_price_above_ma and sector_ma_slope_positive:
                                    sector_phase = 2  # Uptrend
                                elif sector_price_above_ma and not sector_ma_slope_positive:
                                    sector_phase = 3  # Top formation
                                elif not sector_price_above_ma and not sector_ma_slope_positive:
                                    sector_phase = 4  # Downtrend
                                else:
                                    sector_phase = 1  # Base formation
                                
                                # Get sector performance metrics
                                if len(sector_data) >= 4:
                                    sector_1month_perf = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[-4] - 1) * 100
                                else:
                                    sector_1month_perf = 0
                                
                                # Calculate relative strength vs market
                                if len(market_data) == len(sector_data):
                                    relative_strength = (sector_data['Close'].iloc[-1] / sector_data['Close'].iloc[0]) / \
                                                       (market_data['Close'].iloc[-1] / market_data['Close'].iloc[0]) * 100 - 100
                                else:
                                    relative_strength = 0
                                
                                # Store sector context
                                self.sector_data = {
                                    'name': sector,
                                    'etf': sector_etf,
                                    'phase': sector_phase,
                                    'last_close': float(sector_data['Close'].iloc[-1]),
                                    'performance_1month': sector_1month_perf,
                                    'relative_strength': relative_strength
                                }
                                
                                logger.info(f"Sector data loaded: {sector} (Phase {sector_phase})")
                
                except Exception as e:
                    logger.warning(f"Could not load sector data: {str(e)}")
                    self.sector_data = None
            else:
                logger.warning("No market context data available")
                self.market_context = None
        
        except Exception as e:
            logger.warning(f"Error loading market context: {str(e)}")
            self.market_context = None

    def identify_support_resistance(self):
        """Identify key support and resistance levels using local minima/maxima and volume analysis"""
        if self.data is None or len(self.data) < 30:
            self.support_resistance_levels = []
            return
        
        try:
            df = self.data.copy()
            
            # Extract high and low series safely
            def get_safe_series(df, column):
                if column not in df.columns:
                    return pd.Series(index=df.index)
                
                series = df[column]
                if isinstance(series, pd.DataFrame):
                    try:
                        series = series.iloc[:, 0]
                    except:
                        return pd.Series(index=df.index)
                
                return pd.to_numeric(series, errors='coerce')
            
            high_series = get_safe_series(df, 'High')
            low_series = get_safe_series(df, 'Low')
            close_series = get_safe_series(df, 'Close')
            volume_series = get_safe_series(df, 'Volume')
            
            # Find local maxima and minima (rolling window of 5 periods)
            window = 5
            resistance_levels = []
            support_levels = []
            
            # Find resistance levels (local highs)
            for i in range(window, len(df) - window):
                if high_series.iloc[i] == high_series.iloc[i-window:i+window+1].max():
                    # Check if volume was significant
                    avg_vol = volume_series.iloc[i-window:i+window+1].mean()
                    if volume_series.iloc[i] > avg_vol * 1.2:  # 20% above average
                        resistance_levels.append({
                            'price': float(high_series.iloc[i]),
                            'date': df.index[i],
                            'strength': 'strong' if volume_series.iloc[i] > avg_vol * 1.5 else 'medium'
                        })
                    else:
                        resistance_levels.append({
                            'price': float(high_series.iloc[i]),
                            'date': df.index[i],
                            'strength': 'weak'
                        })
            
            # Find support levels (local lows)
            for i in range(window, len(df) - window):
                if low_series.iloc[i] == low_series.iloc[i-window:i+window+1].min():
                    # Check if volume was significant
                    avg_vol = volume_series.iloc[i-window:i+window+1].mean()
                    if volume_series.iloc[i] > avg_vol * 1.2:  # 20% above average
                        support_levels.append({
                            'price': float(low_series.iloc[i]),
                            'date': df.index[i],
                            'strength': 'strong' if volume_series.iloc[i] > avg_vol * 1.5 else 'medium'
                        })
                    else:
                        support_levels.append({
                            'price': float(low_series.iloc[i]),
                            'date': df.index[i],
                            'strength': 'weak'
                        })
            
            # Group nearby levels (within 3% of each other)
            def group_levels(levels):
                if not levels:
                    return []
                
                # Sort by price
                sorted_levels = sorted(levels, key=lambda x: x['price'])
                
                # Group nearby levels
                grouped = []
                current_group = [sorted_levels[0]]
                
                for i in range(1, len(sorted_levels)):
                    current_level = sorted_levels[i]
                    prev_level = current_group[-1]
                    
                    # If current level is within 3% of previous level, add to current group
                    if (current_level['price'] - prev_level['price']) / prev_level['price'] < 0.03:
                        current_group.append(current_level)
                    else:
                        # Find average price weighted by strength
                        strength_weights = {'weak': 1, 'medium': 2, 'strong': 3}
                        total_weight = sum(strength_weights[level['strength']] for level in current_group)
                        avg_price = sum(level['price'] * strength_weights[level['strength']] for level in current_group) / total_weight
                        
                        # Determine overall strength
                        max_strength = max(level['strength'] for level in current_group)
                        
                        grouped.append({
                            'price': avg_price,
                            'date': max(level['date'] for level in current_group),
                            'strength': max_strength
                        })
                        
                        # Start new group
                        current_group = [current_level]
                
                # Add last group
                if current_group:
                    strength_weights = {'weak': 1, 'medium': 2, 'strong': 3}
                    total_weight = sum(strength_weights[level['strength']] for level in current_group)
                    avg_price = sum(level['price'] * strength_weights[level['strength']] for level in current_group) / total_weight
                    max_strength = max(level['strength'] for level in current_group)
                    
                    grouped.append({
                        'price': avg_price,
                        'date': max(level['date'] for level in current_group),
                        'strength': max_strength
                    })
                
                return grouped
            
            # Group and combine levels
            resistance_levels = group_levels(resistance_levels)
            support_levels = group_levels(support_levels)
            
            # Add level type
            for level in resistance_levels:
                level['type'] = 'resistance'
            
            for level in support_levels:
                level['type'] = 'support'
            
            # Combine and sort by price
            self.support_resistance_levels = sorted(resistance_levels + support_levels, key=lambda x: x['price'])
            
            logger.info(f"Identified {len(resistance_levels)} resistance and {len(support_levels)} support levels")
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance levels: {str(e)}")
            traceback.print_exc()
            self.support_resistance_levels = []
            
    def calculate_indicators(self):
        """Calculate technical indicators with enhanced methods and adaptability for limited data"""
        if self.data is None or len(self.data) == 0:
            return
            
        try:
            df = self.data.copy()
            
            # Ensure we're working with Series and not DataFrames
            def get_safe_series(df, column):
                if column not in df.columns:
                    return pd.Series(index=df.index)
                
                series = df[column]
                if isinstance(series, pd.DataFrame):
                    try:
                        series = series.iloc[:, 0]
                    except:
                        return pd.Series(index=df.index)
                
                return pd.to_numeric(series, errors='coerce')
            
            close_series = get_safe_series(df, 'Close')
            open_series = get_safe_series(df, 'Open')
            high_series = get_safe_series(df, 'High')
            low_series = get_safe_series(df, 'Low')
            volume_series = get_safe_series(df, 'Volume')
            
            # Get lengths for adaptive windows
            data_length = len(df)
            
            # Moving Averages with adaptive windows based on available data
            ma_windows = {
                'MA10': min(10, max(3, data_length // 5)),  # At least 3 periods
                'MA30': min(30, max(5, data_length // 3)),  # At least 5 periods
                'MA50': min(50, max(10, data_length // 2)), # At least 10 periods
                'MA200': min(200, max(20, data_length))     # At least 20 periods
            }
            
            # Calculate MAs with adaptive windows
            df['MA10'] = close_series.rolling(window=ma_windows['MA10']).mean()
            df['MA30'] = close_series.rolling(window=ma_windows['MA30']).mean()
            
            # Only calculate longer MAs if enough data is available
            if data_length >= ma_windows['MA50'] * 1.2:  # Need 20% more data than window size
                df['MA50'] = close_series.rolling(window=ma_windows['MA50']).mean()
            
            if data_length >= ma_windows['MA200'] * 1.1:  # Need 10% more data than window size
                df['MA200'] = close_series.rolling(window=ma_windows['MA200']).mean()
            
            # Store the actual window sizes used for reference
            self.indicators['ma_windows'] = ma_windows
            
            # MA Slopes for trend direction and strength
            df['MA10_Slope'] = df['MA10'].diff()
            df['MA30_Slope'] = df['MA30'].diff()
            
            # Adaptive slope calculation based on available data
            slope_periods = min(4, max(1, data_length // 10))
            df['MA30_Slope_4Wk'] = df['MA30'].diff(slope_periods)
            
            # Distance from MAs as percentage (only calculate if MA exists)
            df['Pct_From_MA30'] = (close_series / df['MA30'] - 1) * 100 if 'MA30' in df else pd.Series(index=df.index)
            
            if 'MA200' in df:
                df['Pct_From_MA200'] = (close_series / df['MA200'] - 1) * 100
            
            # RSI (Relative Strength Index) with adaptive window
            rsi_window = min(14, max(5, data_length // 4))
            delta = close_series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=rsi_window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=rsi_window).mean()
            
            # Prevent division by zero
            loss = loss.replace(0, np.nan)
            rs = gain / loss
            rs = rs.fillna(0)
            
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands (2 standard deviations) with adaptive window
            bb_window = min(30, max(5, data_length // 3))
            df['BBand_Mid'] = df['MA30'] if 'MA30' in df else close_series.rolling(window=bb_window).mean()
            df['BBand_Std'] = close_series.rolling(window=bb_window).std()
            df['BBand_Upper'] = df['BBand_Mid'] + (df['BBand_Std'] * 2)
            df['BBand_Lower'] = df['BBand_Mid'] - (df['BBand_Std'] * 2)
            
            # Bollinger Band Width (volatility indicator)
            df['BB_Width'] = (df['BBand_Upper'] - df['BBand_Lower']) / df['BBand_Mid'] * 100
            
            # Volume analysis - only if volume data is available and not all zeros
            if not volume_series.isna().all() and (volume_series > 0).any():
                vol_window = min(30, max(5, data_length // 3))
                df['VolMA30'] = volume_series.rolling(window=vol_window).mean()
                
                # Handle zero values in volume MA
                vol_ma_mean = df['VolMA30'].mean()
                df['VolMA30'] = df['VolMA30'].replace(0, vol_ma_mean if vol_ma_mean > 0 else 1)
                
                df['Vol_Ratio'] = volume_series / df['VolMA30']
                df['Vol_Ratio'] = df['Vol_Ratio'].fillna(0)
                
                # On-Balance Volume (OBV) - only if we have enough data
                if data_length >= 5:
                    obv = pd.Series(0, index=df.index)
                    for i in range(1, len(df)):
                        if close_series.iloc[i] > close_series.iloc[i-1]:
                            obv.iloc[i] = obv.iloc[i-1] + volume_series.iloc[i]
                        elif close_series.iloc[i] < close_series.iloc[i-1]:
                            obv.iloc[i] = obv.iloc[i-1] - volume_series.iloc[i]
                        else:
                            obv.iloc[i] = obv.iloc[i-1]
                    
                    df['OBV'] = obv
                    
                    if data_length >= 20:
                        df['OBV_MA20'] = df['OBV'].rolling(window=min(20, data_length // 2)).mean()
            else:
                # Create placeholder volume indicators with zeros if volume data not available
                df['VolMA30'] = 0
                df['Vol_Ratio'] = 0
                df['OBV'] = 0
                
                self.warnings.append("No volume data available. Volume-based indicators will be limited.")
            
            # Price percent from high/low with adaptive window
            if data_length >= 10:
                lookback = min(52, max(data_length // 2, 5))  # At least 5 periods, up to 52
                rolling_high = close_series.rolling(window=lookback).max()
                rolling_low = close_series.rolling(window=lookback).min()
                df['Pct_From_52wk_High'] = (close_series / rolling_high - 1) * 100
                df['Pct_From_52wk_Low'] = (close_series / rolling_low - 1) * 100
            else:
                # For very limited data, use the available range
                max_price = close_series.max()
                min_price = close_series.min()
                if max_price > min_price:  # Avoid division by zero
                    df['Pct_From_52wk_High'] = (close_series / max_price - 1) * 100
                    df['Pct_From_52wk_Low'] = (close_series / min_price - 1) * 100
                else:
                    df['Pct_From_52wk_High'] = 0
                    df['Pct_From_52wk_Low'] = 0
            
            # Breakout detection - adaptive to data length
            if data_length >= 5:
                # Use adaptive window based on available data
                breakout_window = min(12, max(3, data_length // 3))
                
                # Use n-period high as breakout reference
                rolling_high = high_series.rolling(window=breakout_window).max()
                rolling_low = low_series.rolling(window=breakout_window).min()
                
                # Detect price breakouts
                df['Price_Breakout'] = close_series > rolling_high.shift(1)
                
                # Check for volume confirmation if volume data is available
                if 'Vol_Ratio' in df and (df['Vol_Ratio'] > 0).any():
                    df['Volume_Confirmed'] = df['Vol_Ratio'] > 1.2
                    df['New_Breakout'] = df['Price_Breakout'] & df['Volume_Confirmed']
                else:
                    # If no volume data, use price only
                    df['New_Breakout'] = df['Price_Breakout']
                
                # Detect breakdown (bearish breakout)
                df['Price_Breakdown'] = close_series < rolling_low.shift(1)
                
                if 'Vol_Ratio' in df and (df['Vol_Ratio'] > 0).any():
                    df['Breakdown'] = df['Price_Breakdown'] & df['Volume_Confirmed']
                else:
                    df['Breakdown'] = df['Price_Breakdown']
            else:
                # For very limited data, just use placeholder values
                df['New_Breakout'] = False
                df['Breakdown'] = False
            
            # Consolidation pattern detection - adaptive to data length
            if data_length >= 5:
                # Adaptive window for consolidation detection
                window_size = min(8, max(3, data_length // 2))
                
                # Check if price is moving in a narrow range
                price_range = high_series.rolling(window=window_size).max() / low_series.rolling(window=window_size).min() - 1
                narrow_range = price_range < 0.05
                
                # Check for low volume if volume data is available
                if 'Vol_Ratio' in df and (df['Vol_Ratio'] > 0).any():
                    low_volume = df['Vol_Ratio'] < 0.8
                    df['Is_Consolidating'] = narrow_range & low_volume
                else:
                    # If no volume data, use price range only
                    df['Is_Consolidating'] = narrow_range
            else:
                df['Is_Consolidating'] = False
            
            # Price Range as % (High-Low)/Close
            df['Price_Range_Pct'] = (high_series - low_series) / close_series * 100
            
            # Store the calculated data
            self.data = df
            
            logger.info(f"Calculated indicators for {self.ticker_symbol} with adaptive windows")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            traceback.print_exc()
            self.errors.append(f"Error calculating indicators: {str(e)}")
            # Create a minimum viable dataframe to continue analysis
            if self.data is not None and 'Close' in self.data.columns:
                self.data['MA30'] = self.data['Close'].rolling(window=min(30, len(self.data))).mean()
                self.data['RSI'] = 50  # Neutral RSI
                self.warnings.append("Limited indicators calculated due to errors.")
            else:
                logger.error("Critical error: cannot calculate minimum viable indicators")
                self.errors.append("Critical error in indicator calculation.")
                return
    
    def identify_phase(self):
        """Identify the market phase according to Weinstein's method with enhanced accuracy"""
        if self.data is None or len(self.data) < 4:  # Need at least 4 data points for basic phase analysis
            self.phase = 0
            self.phase_desc = "Insufficient data"
            return
            
        try:
            current = self.data.iloc[-1]
            
            # Define get_safe_series function
            def get_safe_series(df, column):
                if column not in df.columns:
                    return pd.Series(index=df.index)
                
                series = df[column]
                if isinstance(series, pd.DataFrame):
                    try:
                        series = series.iloc[:, 0]
                    except:
                        return pd.Series(index=df.index)
                
                return pd.to_numeric(series, errors='coerce')
            
            # Extract values safely and ensure we're working with scalar values
            def safe_get_value(row, column, default=0):
                if column not in row:
                    return default
                
                value = row[column]
                
                # Handle different data types
                if isinstance(value, pd.Series):
                    if value.empty:
                        return default
                    if pd.isna(value.iloc[0]):
                        return default
                    try:
                        return float(value.iloc[0])
                    except:
                        return default
                elif isinstance(value, pd.DataFrame):
                    if value.empty:
                        return default
                    if pd.isna(value.iloc[0,0]):
                        return default
                    try:
                        return float(value.iloc[0,0])
                    except:
                        return default
                elif isinstance(value, bool):
                    return value
                elif pd.isna(value):
                    return default
                else:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return default
            
            # Safe extraction of key indicators
            close_value = safe_get_value(current, 'Close')
            
            # Check if we have MA30 (or closest available)
            if 'MA30' in current:
                ma30_value = safe_get_value(current, 'MA30')
            else:
                # If no MA30, try MA10 or just use a placeholder
                if 'MA10' in current:
                    ma30_value = safe_get_value(current, 'MA10')
                    self.warnings.append("Using MA10 as a substitute for MA30 due to limited data.")
                else:
                    # If no MAs available, can't do proper phase analysis
                    self.phase = 0
                    self.phase_desc = "Insufficient indicator data"
                    self.warnings.append("Cannot perform accurate phase analysis without moving averages.")
                    return
            
            # Get other indicators if available
            ma10_value = safe_get_value(current, 'MA10') if 'MA10' in current else 0
            ma30_slope_value = safe_get_value(current, 'MA30_Slope') if 'MA30_Slope' in current else 0
            ma30_slope_4wk_value = safe_get_value(current, 'MA30_Slope_4Wk') if 'MA30_Slope_4Wk' in current else 0
            rsi_value = safe_get_value(current, 'RSI') if 'RSI' in current else 50  # Default to neutral RSI
            
            # Extract boolean indicators (with defaults for missing data)
            is_consolidating = False
            if 'Is_Consolidating' in current:
                is_consolidating = bool(safe_get_value(current, 'Is_Consolidating', False))
            
            new_breakout = False
            if 'New_Breakout' in current:
                new_breakout = bool(safe_get_value(current, 'New_Breakout', False))
            
            breakdown = False
            if 'Breakdown' in current:
                breakdown = bool(safe_get_value(current, 'Breakdown', False))
            
            # Calculate trend conditions based on Weinstein's criteria
            price_above_ma30 = close_value > ma30_value
            ma10_above_ma30 = ma10_value > ma30_value
            ma30_slope_positive = ma30_slope_value > 0
            ma30_slope_improving = ma30_slope_value > ma30_slope_4wk_value / 4 if ma30_slope_4wk_value != 0 else False
            rsi_bullish = rsi_value > 50
            
            # Check recent price action - adaptive to available data
            min_periods = min(5, len(self.data))
            if min_periods >= 3:  # Need at least 3 periods to check for trend
                recent_df = self.data.iloc[-min_periods:]
                
                # Calculate higher highs and higher lows
                higher_highs = True
                higher_lows = True
                
                highs = get_safe_series(recent_df, 'High')
                lows = get_safe_series(recent_df, 'Low')
                
                if len(highs) >= 3 and not highs.isna().all():
                    for i in range(1, len(highs)):
                        if highs.iloc[i] <= highs.iloc[i-1]:
                            higher_highs = False
                            break
                else:
                    higher_highs = False
                
                if len(lows) >= 3 and not lows.isna().all():
                    for i in range(1, len(lows)):
                        if lows.iloc[i] <= lows.iloc[i-1]:
                            higher_lows = False
                            break
                else:
                    higher_lows = False
            else:
                higher_highs = False
                higher_lows = False
            
            # Phase identification with enhanced logic
            
            # Stage 2 (Uptrend) - Enhanced with strength indication and breakout detection
            if price_above_ma30 and ma30_slope_positive:
                if new_breakout:
                    self.phase = 2
                    self.phase_desc = "Uptrend - New Breakout"
                elif higher_highs and higher_lows and ma30_slope_improving:
                    self.phase = 2
                    self.phase_desc = "Strong Uptrend"
                elif ma10_above_ma30 and rsi_bullish:
                    self.phase = 2
                    self.phase_desc = "Confirmed Uptrend"
                else:
                    self.phase = 2
                    self.phase_desc = "Uptrend"
            
            # Stage 4 (Downtrend) - Enhanced with breakdown detection
            elif not price_above_ma30 and not ma30_slope_positive:
                if breakdown:
                    self.phase = 4
                    self.phase_desc = "Strong Downtrend - New Breakdown"
                elif rsi_value < 30:
                    self.phase = 4
                    self.phase_desc = "Downtrend - Oversold"
                else:
                    self.phase = 4
                    self.phase_desc = "Downtrend"
            
            # Stage 3 (Top Formation)
            elif price_above_ma30 and not ma30_slope_positive:
                if rsi_value > 70:
                    self.phase = 3
                    self.phase_desc = "Top Formation - Overbought"
                else:
                    self.phase = 3
                    self.phase_desc = "Top Formation"
            
            # Stage 1 (Base Formation) - Enhanced with consolidation detection
            elif not price_above_ma30:
                if ma30_slope_positive or rsi_bullish:
                    if is_consolidating and rsi_value > 45:
                        self.phase = 1
                        self.phase_desc = "Base Formation - Late Stage"
                    else:
                        self.phase = 1
                        self.phase_desc = "Base Formation"
                else:
                    self.phase = 1
                    self.phase_desc = "Base Formation - Early Stage"
            
            # Handle edge cases
            else:
                self.phase = 0
                self.phase_desc = "Transition Phase"
                
            logger.info(f"Phase identified for {self.ticker_symbol}: Phase {self.phase} - {self.phase_desc}")
                
        except Exception as e:
            logger.error(f"Error in phase identification: {str(e)}")
            traceback.print_exc()
            self.phase = 0
            self.phase_desc = f"Error: {str(e)}"
            
    def generate_recommendation(self):
        """Generate a trading recommendation based on Weinstein's method with enhanced detail"""
        if self.phase == 0 or self.data is None or len(self.data) == 0:
            self.recommendation = "NO RECOMMENDATION - insufficient data"
            return
            
        try:
            current = self.data.iloc[-1]
            
            # Safe extraction of values from potentially complex structures
            def safe_get_value(row, column, default=0):
                if column not in row:
                    return default
                    
                value = row[column]
                # Handle different data types
                if isinstance(value, pd.Series):
                    if value.empty:
                        return default
                    if pd.isna(value.iloc[0]):
                        return default
                    try:
                        return float(value.iloc[0])
                    except:
                        return default
                elif isinstance(value, pd.DataFrame):
                    if value.empty:
                        return default
                    if pd.isna(value.iloc[0,0]):
                        return default
                    try:
                        return float(value.iloc[0,0])
                    except:
                        return default
                elif isinstance(value, bool):
                    return value
                elif pd.isna(value):
                    return default
                else:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return default
            
            # Extract key indicators
            rsi = safe_get_value(current, 'RSI')
            vol_ratio = safe_get_value(current, 'Vol_Ratio')
            bb_width = safe_get_value(current, 'BB_Width')
            pct_from_52wk_high = safe_get_value(current, 'Pct_From_52wk_High')
            pct_from_ma30 = safe_get_value(current, 'Pct_From_MA30')
            
            # Extract boolean indicators
            new_breakout = False
            if 'New_Breakout' in current:
                new_breakout = bool(safe_get_value(current, 'New_Breakout', False))
            
            is_consolidating = False
            if 'Is_Consolidating' in current:
                is_consolidating = bool(safe_get_value(current, 'Is_Consolidating', False))
            
            # Consider market and sector context
            market_bullish = False
            sector_bullish = False
            
            if self.market_context is not None and self.market_context['phase'] in [1, 2]:
                market_bullish = True
            
            if self.sector_data is not None and self.sector_data['phase'] in [1, 2]:
                sector_bullish = True
            
            # Generate recommendation based on phase and indicators
            if self.phase == 2:  # Uptrend
                # Check for overbought conditions
                if rsi > 75 and pct_from_ma30 > 15:
                    self.recommendation = "REDUCE POSITION / TIGHTEN STOPS - Overbought"
                # Check for strong uptrend with volume confirmation
                elif new_breakout and vol_ratio > 1.5 and market_bullish and sector_bullish:
                    self.recommendation = "STRONG BUY - CONFIRMED BREAKOUT"
                elif new_breakout and vol_ratio > 1.2:
                    self.recommendation = "BUY - BREAKOUT"
                # Check for healthy uptrend
                elif vol_ratio > 1.2 and 40 < rsi < 70 and market_bullish and sector_bullish:
                    self.recommendation = "STRONG BUY - HEALTHY UPTREND"
                elif vol_ratio > 1.0 and 40 < rsi < 70:
                    self.recommendation = "BUY"
                # Pullback opportunity
                elif -10 < pct_from_ma30 < -2 and rsi > 40:
                    self.recommendation = "BUY - PULLBACK OPPORTUNITY"
                else:
                    self.recommendation = "HOLD - UPTREND"
                    
            elif self.phase == 1:  # Base Formation
                # Check for potential breakout setup
                if is_consolidating and rsi > 50 and vol_ratio > 0.8 and bb_width < 10:
                    if market_bullish and sector_bullish:
                        self.recommendation = "ACCUMULATE - POTENTIAL BREAKOUT SOON"
                    else:
                        self.recommendation = "WATCH CLOSELY - POTENTIAL BREAKOUT"
                # Check for early accumulation
                elif rsi > 45 and vol_ratio > 1.0:
                    self.recommendation = "LIGHT ACCUMULATION - BASE BUILDING"
                # Early base
                elif is_consolidating:
                    self.recommendation = "WATCH - BASE FORMING"
                else:
                    self.recommendation = "MONITOR - WAIT FOR BASE COMPLETION"
                
            elif self.phase == 3:  # Top Formation
                # Check for distribution signs
                if rsi > 70:
                    self.recommendation = "SELL / TAKE PROFITS - OVERBOUGHT"
                elif vol_ratio > 1.2 and pct_from_ma30 < 0:
                    self.recommendation = "REDUCE POSITION - DISTRIBUTION SIGNS"
                else:
                    self.recommendation = "TIGHTEN STOPS - TOP FORMING"
                
            elif self.phase == 4:  # Downtrend
                # Check for oversold bounce potential
                if rsi < 30 and vol_ratio > 1.5:
                    self.recommendation = "AVOID / POTENTIAL OVERSOLD BOUNCE"
                # Strong downtrend
                elif vol_ratio > 1.2 and pct_from_52wk_high < -20:
                    self.recommendation = "AVOID / SHORT OPPORTUNITY"
                else:
                    self.recommendation = "AVOID - DOWNTREND"
            
            else:
                self.recommendation = "NEUTRAL - UNCLEAR PATTERN"
                
            logger.info(f"Recommendation for {self.ticker_symbol}: {self.recommendation}")
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            traceback.print_exc()
            self.recommendation = "ERROR - No recommendation possible"
    
    def generate_detailed_analysis(self):
        """Generate a detailed analysis text based on the identified phase and indicators"""
        if self.phase == 0 or self.data is None or len(self.data) == 0:
            self.detailed_analysis = "Insufficient data for analysis."
            return
        
        try:
            # Access the latest data point
            current = self.data.iloc[-1]
            
            # Safe extraction of values
            def safe_get_value(row, column, default=None):
                if column not in row:
                    return default
                
                value = row[column]
                # Handle different data types
                if isinstance(value, pd.Series):
                    if value.empty:
                        return default
                    if pd.isna(value.iloc[0]):
                        return default
                    try:
                        return float(value.iloc[0])
                    except:
                        return default
                elif isinstance(value, pd.DataFrame):
                    if value.empty:
                        return default
                    if pd.isna(value.iloc[0,0]):
                        return default
                    try:
                        return float(value.iloc[0,0])
                    except:
                        return default
                elif isinstance(value, bool):
                    return value
                elif pd.isna(value):
                    return default
                else:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return default
            
            # Extract key indicators
            rsi = safe_get_value(current, 'RSI')
            vol_ratio = safe_get_value(current, 'Vol_Ratio')
            ma30_slope = safe_get_value(current, 'MA30_Slope')
            pct_from_ma30 = safe_get_value(current, 'Pct_From_MA30')
            pct_from_52wk_high = safe_get_value(current, 'Pct_From_52wk_High')
            
            # Market and sector context
            market_context_text = ""
            if self.market_context:
                market_phase_desc = {
                    1: "Base Formation", 
                    2: "Uptrend", 
                    3: "Top Formation", 
                    4: "Downtrend"
                }.get(self.market_context['phase'], "Unknown")
                
                market_context_text = f"The overall market (S&P 500) is in a {market_phase_desc} phase. "
                if self.market_context['phase'] == 2:
                    market_context_text += "The market uptrend provides a favorable backdrop for bullish positions. "
                elif self.market_context['phase'] == 4:
                    market_context_text += "The market downtrend suggests caution for long positions. "
            
            sector_context_text = ""
            if self.sector_data:
                sector_phase_desc = {
                    1: "Base Formation", 
                    2: "Uptrend", 
                    3: "Top Formation", 
                    4: "Downtrend"
                }.get(self.sector_data['phase'], "Unknown")
                
                sector_context_text = f"The {self.sector_data['name']} sector is in a {sector_phase_desc} phase. "
                if self.sector_data['relative_strength'] > 5:
                    sector_context_text += f"This sector is showing strong relative strength (+{self.sector_data['relative_strength']:.1f}%) compared to the broader market. "
                elif self.sector_data['relative_strength'] < -5:
                    sector_context_text += f"This sector is underperforming the broader market ({self.sector_data['relative_strength']:.1f}% relative strength). "
            
            # Find closest support and resistance levels
            levels_text = ""
            if self.support_resistance_levels:
                resistance_levels = [level for level in self.support_resistance_levels if level['type'] == 'resistance']
                support_levels = [level for level in self.support_resistance_levels if level['type'] == 'support']
                
                if resistance_levels and self.last_price:
                    nearest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - self.last_price))
                    resistance_pct = (nearest_resistance['price'] - self.last_price) / self.last_price * 100
                    levels_text += f"The nearest resistance is at ${nearest_resistance['price']:.2f} (+{resistance_pct:.1f}%). "
                
                if support_levels and self.last_price:
                    nearest_support = min(support_levels, key=lambda x: abs(x['price'] - self.last_price))
                    support_pct = (self.last_price - nearest_support['price']) / self.last_price * 100
                    levels_text += f"The nearest support is at ${nearest_support['price']:.2f} (-{support_pct:.1f}%)."
            
            # Phase-specific analysis
            phase_analysis = ""
            if self.phase == 1:  # Base Formation
                phase_analysis = (
                    f"{self.ticker_symbol} is in a Stage 1 Base Formation. "
                    f"This is a consolidation phase following a downtrend where supply and demand are reaching equilibrium. "
                )
                
                if rsi and rsi > 50:
                    phase_analysis += "RSI is above 50, indicating improving momentum within the base. "
                
                if ma30_slope and ma30_slope > 0:
                    phase_analysis += "The 30-week moving average has begun to flatten and turn upward, a positive sign. "
                
                if vol_ratio and vol_ratio > 1.1:
                    phase_analysis += "Volume is showing signs of accumulation, suggesting smart money may be taking positions. "
                
                phase_analysis += "According to Weinstein's method, the ideal time to buy is when the stock breaks out from this base into Stage 2 with increased volume. "
                
            elif self.phase == 2:  # Uptrend
                phase_analysis = (
                    f"{self.ticker_symbol} is in a Stage 2 Uptrend. "
                    f"This is the most profitable stage where price is trending higher with higher highs and higher lows. "
                )
                
                if pct_from_ma30 is not None:
                    if pct_from_ma30 < -5:
                        phase_analysis += f"Price has pulled back {abs(pct_from_ma30):.1f}% from the 30-week MA, offering a potential buying opportunity. "
                    elif pct_from_ma30 > 10:
                        phase_analysis += f"Price is extended {pct_from_ma30:.1f}% above its 30-week MA, suggesting caution and tighter stops. "
                
                if vol_ratio and vol_ratio > 1.2:
                    phase_analysis += "Volume is confirming the uptrend with above-average participation. "
                
                if rsi:
                    if rsi > 70:
                        phase_analysis += f"RSI is overbought at {rsi:.1f}, suggesting potential for a short-term pullback. "
                    elif 40 < rsi < 70:
                        phase_analysis += f"RSI at {rsi:.1f} shows healthy momentum without being overbought. "
                
            elif self.phase == 3:  # Top Formation
                phase_analysis = (
                    f"{self.ticker_symbol} is in a Stage 3 Top Formation. "
                    f"This distribution phase typically occurs after a strong uptrend and precedes a downtrend. "
                )
                
                if ma30_slope and ma30_slope < 0:
                    phase_analysis += "The 30-week moving average has started to roll over, a warning sign. "
                
                if vol_ratio and vol_ratio > 1.2:
                    phase_analysis += "Higher volume on down days suggests distribution (selling) by institutions. "
                
                if rsi and rsi < 50:
                    phase_analysis += "Declining RSI indicates weakening momentum. "
                
                phase_analysis += "According to Weinstein's method, this is typically a time to take profits or tighten stops rather than establishing new positions. "
                
            elif self.phase == 4:  # Downtrend
                phase_analysis = (
                    f"{self.ticker_symbol} is in a Stage 4 Downtrend. "
                    f"This bearish phase is characterized by lower highs and lower lows with price below a declining 30-week MA. "
                )
                
                if pct_from_52wk_high is not None:
                    phase_analysis += f"Price is {abs(pct_from_52wk_high):.1f}% below its 52-week high. "
                
                if rsi and rsi < 30:
                    phase_analysis += f"RSI is oversold at {rsi:.1f}, which may lead to short-term bounces, but the primary trend remains down. "
                
                phase_analysis += "According to Weinstein's method, Stage 4 stocks should be avoided for long positions. Wait for a Stage 1 base to form before considering entry. "
                
            # Combine all analysis components
            self.detailed_analysis = (
                f"{phase_analysis}\n\n"
                f"{market_context_text}{sector_context_text}\n\n"
                f"{levels_text}\n\n"
                f"Recommendation: {self.recommendation}"
            )
            
            logger.info(f"Generated detailed analysis for {self.ticker_symbol}")
            
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {str(e)}")
            traceback.print_exc()
            self.detailed_analysis = f"Error generating analysis: {str(e)}"
    
    def create_interactive_chart(self):
        """Create an interactive chart with Plotly and enhanced visualization"""
        if self.data is None or len(self.data) == 0:
            # Create an empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=800)
            return fig
            
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=3, 
                cols=1, 
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('Price', 'Volume', 'RSI'),
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Extract safe series for plotting
            def get_safe_series(df, column):
                """Extract a safe series for plotting"""
                if column not in df.columns:
                    return pd.Series(index=df.index)
                    
                series = df[column]
                
                # Convert to a standard series if it's a DataFrame
                if isinstance(series, pd.DataFrame):
                    try:
                        series = series.iloc[:, 0]
                    except:
                        return pd.Series(index=df.index)
                
                # Ensure all values are numeric
                return pd.to_numeric(series, errors='coerce')
            
            # Safe extractions
            close = get_safe_series(self.data, 'Close')
            open_vals = get_safe_series(self.data, 'Open')
            high = get_safe_series(self.data, 'High')
            low = get_safe_series(self.data, 'Low')
            volume = get_safe_series(self.data, 'Volume')
            ma10 = get_safe_series(self.data, 'MA10')
            ma30 = get_safe_series(self.data, 'MA30')
            ma50 = get_safe_series(self.data, 'MA50')
            ma200 = get_safe_series(self.data, 'MA200')
            volma30 = get_safe_series(self.data, 'VolMA30')
            rsi = get_safe_series(self.data, 'RSI')
            bband_upper = get_safe_series(self.data, 'BBand_Upper')
            bband_lower = get_safe_series(self.data, 'BBand_Lower')
            
            # Add price candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=self.data.index,
                    open=open_vals,
                    high=high,
                    low=low,
                    close=close,
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Add moving averages
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=ma10,
                    line=dict(color='blue', width=1.5),
                    name="10-Week MA"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=ma30,
                    line=dict(color='red', width=2),
                    name="30-Week MA"
                ),
                row=1, col=1
            )
            
            if not ma50.isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=ma50,
                        line=dict(color='green', width=1.5, dash='dot'),
                        name="50-Week MA"
                    ),
                    row=1, col=1
                )
            
            if not ma200.isna().all():
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=ma200,
                        line=dict(color='purple', width=1.5, dash='dash'),
                        name="200-Week MA"
                    ),
                    row=1, col=1
                )
            
            # Add Bollinger Bands
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=bband_upper,
                    line=dict(color='rgba(0,0,0,0.3)', width=1),
                    name="Upper BB",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=bband_lower,
                    line=dict(color='rgba(0,0,0,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(0,0,0,0.05)',
                    name="Lower BB",
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add support and resistance levels
            for level in self.support_resistance_levels:
                color = 'green' if level['type'] == 'support' else 'red'
                width = 2 if level['strength'] == 'strong' else 1
                dash = 'solid' if level['strength'] == 'strong' else 'dash'
                
                fig.add_shape(
                    type="line",
                    x0=self.data.index[0],
                    y0=level['price'],
                    x1=self.data.index[-1],
                    y1=level['price'],
                    line=dict(
                        color=color,
                        width=width,
                        dash=dash
                    ),
                    row=1, col=1
                )
                
                # Add annotation only for strong levels
                if level['strength'] == 'strong':
                    fig.add_annotation(
                        x=self.data.index[-1],
                        y=level['price'],
                        text=f"{level['type'].capitalize()}: ${level['price']:.2f}",
                        showarrow=False,
                        xanchor="right",
                        yanchor="bottom" if level['type'] == 'resistance' else "top",
                        xshift=10,
                        font=dict(size=10, color=color)
                    )
            
            # Add volume bar chart with color-coding
            colors = []
            for i in range(len(self.data)):
                try:
                    o = float(open_vals.iloc[i])
                    c = float(close.iloc[i])
                    colors.append('red' if o > c else 'green')
                except:
                    colors.append('blue')  # Fallback color
            
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=volume,
                    marker=dict(color=colors),
                    name="Volume",
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add volume moving average
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=volma30,
                    line=dict(color='orange', width=1.5),
                    name="30-Week Vol MA",
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Add RSI
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=rsi,
                    line=dict(color='purple', width=1.5),
                    name="RSI"
                ),
                row=3, col=1
            )
            
            # Add RSI reference lines
            fig.add_trace(
                go.Scatter(
                    x=[self.data.index[0], self.data.index[-1]],
                    y=[70, 70],
                    line=dict(color='red', width=1, dash='dash'),
                    name="Overbought",
                    showlegend=False
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[self.data.index[0], self.data.index[-1]],
                    y=[30, 30],
                    line=dict(color='green', width=1, dash='dash'),
                    name="Oversold",
                    showlegend=False
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[self.data.index[0], self.data.index[-1]],
                    y=[50, 50],
                    line=dict(color='gray', width=1, dash='dot'),
                    name="Neutral",
                    showlegend=False
                ),
                row=3, col=1
            )
            
            # Add background color based on the phase
            phase_colors = {
                1: 'rgba(128,128,128,0.15)',  # Base - gray
                2: 'rgba(0,128,0,0.1)',       # Uptrend - green
                3: 'rgba(255,165,0,0.1)',     # Top - orange
                4: 'rgba(255,0,0,0.1)'        # Downtrend - red
            }
            
            if self.phase in phase_colors:
                fig.add_shape(
                    type="rect",
                    xref="paper", yref="paper",
                    x0=0, y0=0,
                    x1=1, y1=1,
                    fillcolor=phase_colors[self.phase],
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                )
            
            # Add Weinstein phase annotation
            phase_names = {
                1: "Stage 1: Base Formation",
                2: "Stage 2: Uptrend",
                3: "Stage 3: Top Formation",
                4: "Stage 4: Downtrend"
            }
            
            phase_name = phase_names.get(self.phase, "Unknown Phase")
            
            # Add breakout/breakdown annotations if detected
            if 'New_Breakout' in self.data.columns:
                breakout_points = self.data[self.data['New_Breakout'] == True]
                if not breakout_points.empty:
                    for idx, point in breakout_points.iterrows():
                        try:
                            point_high = float(high.loc[idx])
                            fig.add_annotation(
                                x=idx,
                                y=point_high,
                                text="Breakout",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="green",
                                arrowsize=1,
                                arrowwidth=2,
                                ax=0,
                                ay=-40
                            )
                        except:
                            pass
            
            if 'Breakdown' in self.data.columns:
                breakdown_points = self.data[self.data['Breakdown'] == True]
                if not breakdown_points.empty:
                    for idx, point in breakdown_points.iterrows():
                        try:
                            point_low = float(low.loc[idx])
                            fig.add_annotation(
                                x=idx,
                                y=point_low,
                                text="Breakdown",
                                showarrow=True,
                                arrowhead=2,
                                arrowcolor="red",
                                arrowsize=1,
                                arrowwidth=2,
                                ax=0,
                                ay=40
                            )
                        except:
                            pass
            
            # Update layout with better styling
            fig.update_layout(
                title=f"{self.ticker_symbol}: {phase_name} - {self.phase_desc}<br><sub>{self.recommendation}</sub>",
                xaxis_title="Date",
                yaxis_title="Price",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                height=800,
                dragmode='zoom',
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis_rangeslider_visible=False,
                margin=dict(l=50, r=50, t=100, b=50),
                title_font=dict(size=16)
            )
            
            # Update RSI y-axis
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
            
            # Update Volume y-axis
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating interactive chart: {str(e)}")
            traceback.print_exc()
            
            # Create an empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating chart: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12)
            )
            fig.update_layout(height=800)
            return fig
        
    def create_volume_profile(self, lookback_period=None):
        """Create a volume profile for the specified timeframe"""
        if self.data is None or len(self.data) == 0:
            # Create an empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(height=600)
            return fig
            
        try:
            # Filter data based on lookback period if provided
            if lookback_period is not None and lookback_period < len(self.data):
                df = self.data.iloc[-lookback_period:].copy()
            else:
                df = self.data.copy()
                
            # Get series with proper handling of potential multi-level data
            def get_safe_series(df, column):
                """Extract a safe series for plotting"""
                if column not in df.columns:
                    return pd.Series(index=df.index)
                    
                series = df[column]
                
                # Convert to a standard series if it's a DataFrame
                if isinstance(series, pd.DataFrame):
                    try:
                        series = series.iloc[:, 0]
                    except:
                        return pd.Series(index=df.index)
                
                # Ensure all values are numeric
                return pd.to_numeric(series, errors='coerce')
            
            # Extract safe series for Close and Volume
            close_series = get_safe_series(df, 'Close')
            high_series = get_safe_series(df, 'High')
            low_series = get_safe_series(df, 'Low')
            volume_series = get_safe_series(df, 'Volume')
            
            # Check for empty data after cleaning
            if close_series.isna().all() or volume_series.isna().all():
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid data for volume profile",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(height=600)
                return fig
                
            # Remove NaN values
            valid_data = pd.DataFrame({
                'Close': close_series,
                'High': high_series,
                'Low': low_series,
                'Volume': volume_series
            }).dropna()
            
            if len(valid_data) == 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="No valid data after removing NaN values",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(height=600)
                return fig
                
            # Use high-low range for better profile coverage
            min_price = float(valid_data['Low'].min())
            max_price = float(valid_data['High'].max())
            price_range = max_price - min_price
            
            if price_range <= 0:
                fig = go.Figure()
                fig.add_annotation(
                    text="Insufficient price data for volume profile",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(height=600)
                return fig
                
            # Determine optimal number of bins based on data size
            num_bins = min(max(20, len(valid_data) // 5), 50)
            if num_bins < 5:
                num_bins = 5  # Minimum 5 bins
                
            bin_size = price_range / num_bins
            
            # Calculate volume for each price bin with enhanced method
            volume_by_price = {}
            
            # Distribute volume across the high-low range for each bar
            for i, row in valid_data.iterrows():
                bar_low = row['Low']
                bar_high = row['High']
                bar_volume = row['Volume']
                
                # Skip if invalid data
                if pd.isna(bar_low) or pd.isna(bar_high) or pd.isna(bar_volume) or bar_high <= bar_low:
                    continue
                
                # Calculate which bins this bar spans
                low_bin = max(0, int((bar_low - min_price) / bin_size))
                high_bin = min(num_bins - 1, int((bar_high - min_price) / bin_size))
                
                # Evenly distribute volume across bins (could be weighted by time spent at each price)
                bins_spanned = max(1, high_bin - low_bin + 1)
                volume_per_bin = bar_volume / bins_spanned
                
                for bin_idx in range(low_bin, high_bin + 1):
                    if bin_idx in volume_by_price:
                        volume_by_price[bin_idx] += volume_per_bin
                    else:
                        volume_by_price[bin_idx] = volume_per_bin
            
            # Ensure all bins are represented
            all_bins = {i: volume_by_price.get(i, 0) for i in range(num_bins)}
            
            # Calculate bin midpoints for y-values
            y_values = [min_price + ((i + 0.5) * bin_size) for i in range(num_bins)]
            
            # Create volume profile chart
            fig = go.Figure()
            
            # Add volume bars (horizontal)
            fig.add_trace(
                go.Bar(
                    x=list(all_bins.values()),
                    y=y_values,
                    orientation='h',
                    marker=dict(
                        color=[
                            'rgba(0,128,0,0.7)' if y > self.last_price else 'rgba(255,0,0,0.7)' 
                            for y in y_values
                        ]
                    ),
                    name="Volume Profile"
                )
            )
            
            # Add price line for current price
            if self.last_price is not None:
                max_vol = max(all_bins.values()) if all_bins and max(all_bins.values()) > 0 else 1
                
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=self.last_price,
                    x1=max_vol * 1.1,
                    y1=self.last_price,
                    line=dict(color="black", width=2, dash="dash"),
                )
                
                fig.add_annotation(
                    x=max_vol * 1.05,
                    y=self.last_price,
                    text=f"Current: ${self.last_price:.2f}",
                    showarrow=False,
                    xanchor="right",
                    font=dict(size=12)
                )
            
            # Add point of control (price level with highest volume)
            if all_bins:
                poc_bin = max(all_bins, key=all_bins.get)
                poc_price = y_values[poc_bin]
                poc_volume = all_bins[poc_bin]
                
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=poc_price,
                    x1=poc_volume,
                    y1=poc_price,
                    line=dict(color="blue", width=2),
                )
                
                fig.add_annotation(
                    x=0,
                    y=poc_price,
                    text=f"POC: ${poc_price:.2f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(size=10, color="blue")
                )
            
            # Update layout
            lookback_text = f"past {lookback_period} periods" if lookback_period else "all data"
            
            fig.update_layout(
                title=f"Volume Profile for {self.ticker_symbol} ({lookback_text})",
                xaxis_title="Volume",
                yaxis_title="Price",
                height=600,
                plot_bgcolor='white',
                paper_bgcolor='white',
                margin=dict(l=50, r=50, t=80, b=50),
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating volume profile: {str(e)}")
            traceback.print_exc()
            
            # Create an empty chart with error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating volume profile: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=12)
            )
            fig.update_layout(height=600)
            return fig

# Streamlit app
def main():
    # Set page config
    st.set_page_config(
        page_title="Weinstein Ticker Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    # Initialize the analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = WeinsteinTickerAnalyzer()
    
    # App title
    st.title("Weinstein Ticker Analyzer")
    st.subheader("Based on Stan Weinstein's Stage Analysis Method")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker Symbol", value="AAPL")
        
        # Use columns for a more compact layout
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox(
                "Period",
                options=["3mo", "6mo", "1y", "2y", "5y", "max"],
                index=2  # Default to 1y
            )
        with col2:
            interval = st.selectbox(
                "Interval",
                options=["1d", "1wk", "1mo"],
                index=1  # Default to 1wk
            )
        
        analyze_button = st.button(
            "Analyze",
            type="primary",
            use_container_width=True
        )
        
        # Info section in sidebar
        st.markdown("---")
        st.markdown("### Weinstein's Stages")
        st.markdown("""
        1. **Stage 1**: Base Formation - Accumulation phase
        2. **Stage 2**: Uptrend - Best time to buy
        3. **Stage 3**: Top Formation - Distribution phase 
        4. **Stage 4**: Downtrend - Avoid or consider shorting
        """)

    # Main content area
    if analyze_button:
        # Show a spinner while analyzing
        with st.spinner(f'Analyzing {ticker}...'):
            success = st.session_state.analyzer.load_data(ticker, period, interval)
        
        if success:
            # Create tabs for different analysis views
            tabs = st.tabs(["Overview", "Chart", "Support & Resistance", "Volume Profile", "Detailed Analysis"])
            
            # Display overview in first tab
            with tabs[0]:
                # Create status display with key information
                st.subheader("Analysis Summary")
                
                # Use columns for the status display
                col1, col2, col3 = st.columns(3)
                
                # Safe extraction of ticker name
                analyzer = st.session_state.analyzer
                ticker_name = analyzer.ticker_info.get('longName', analyzer.ticker_info.get('shortName', analyzer.ticker_symbol))
                
                # Column 1: Ticker Info
                with col1:
                    st.markdown("##### Ticker Info")
                    st.markdown(f"**Symbol:** {analyzer.ticker_symbol}")
                    if ticker_name != analyzer.ticker_symbol:
                        st.markdown(f"**Name:** {ticker_name}")
                    st.markdown(f"**Last Price:** ${analyzer.last_price:.2f}" if analyzer.last_price is not None else "**Last Price:** N/A")
                
                # Column 2: Weinstein Analysis
                phase_colors = {
                    1: "#f0f0f0",  # Base - light gray
                    2: "#e6ffe6",  # Uptrend - light green
                    3: "#fff4e6",  # Top - light orange
                    4: "#ffe6e6"   # Downtrend - light red
                }
                phase_color = phase_colors.get(analyzer.phase, "#f9f9f9")
                
                with col2:
                    st.markdown("##### Weinstein Analysis")
                    st.markdown(f"**Phase:** {analyzer.phase} - {analyzer.phase_desc}")
                    st.markdown(f"**Recommendation:** {analyzer.recommendation}")
                
                # Column 3: Technical Indicators
                with col3:
                    st.markdown("##### Technical Indicators")
                    
                    # Safe extraction of RSI
                    rsi_value = "N/A"
                    if analyzer.data is not None and len(analyzer.data) > 0:
                        last_row = analyzer.data.iloc[-1]
                        if 'RSI' in last_row:
                            rsi_series = last_row['RSI']
                            if isinstance(rsi_series, pd.Series):
                                if not rsi_series.empty and not pd.isna(rsi_series.iloc[0]):
                                    rsi_value = f"{float(rsi_series.iloc[0]):.1f}"
                            elif not pd.isna(rsi_series):
                                rsi_value = f"{float(rsi_series):.1f}"
                    
                    # Safe extraction of Vol_Ratio
                    vol_ratio_value = "N/A"
                    if analyzer.data is not None and len(analyzer.data) > 0:
                        last_row = analyzer.data.iloc[-1]
                        if 'Vol_Ratio' in last_row:
                            vol_ratio_series = last_row['Vol_Ratio']
                            if isinstance(vol_ratio_series, pd.Series):
                                if not vol_ratio_series.empty and not pd.isna(vol_ratio_series.iloc[0]):
                                    vol_ratio_value = f"{float(vol_ratio_series.iloc[0]):.2f}x"
                            elif not pd.isna(vol_ratio_series):
                                vol_ratio_value = f"{float(vol_ratio_series):.2f}x"
                    
                    st.markdown(f"**RSI:** {rsi_value}")
                    st.markdown(f"**Volume Ratio:** {vol_ratio_value}")
                    
                # Market & Sector Context
                st.subheader("Market & Sector Context")
                
                # Use columns for market and sector info
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### Market Context")
                    if analyzer.market_context:
                        market_phase = {1: "Base", 2: "Uptrend", 3: "Top", 4: "Downtrend"}.get(
                            analyzer.market_context['phase'], "Unknown")
                        st.markdown(f"**Phase:** Stage {analyzer.market_context['phase']} ({market_phase})")
                        st.markdown(f"**S&P 500:** ${analyzer.market_context['last_close']:.2f}")
                        st.markdown(f"**1-Month Performance:** {analyzer.market_context['performance_1month']:.2f}%")
                    else:
                        st.markdown("No market context available")
                
                with col2:
                    st.markdown("##### Sector Context")
                    if analyzer.sector_data:
                        sector_phase = {1: "Base", 2: "Uptrend", 3: "Top", 4: "Downtrend"}.get(
                            analyzer.sector_data['phase'], "Unknown")
                        st.markdown(f"**Sector:** {analyzer.sector_data['name']} ({analyzer.sector_data['etf']})")
                        st.markdown(f"**Phase:** Stage {analyzer.sector_data['phase']} ({sector_phase})")
                        st.markdown(f"**Relative Strength:** {analyzer.sector_data['relative_strength']:.2f}%")
                    else:
                        st.markdown("No sector data available")
                
                # Display warnings if any
                if analyzer.warnings:
                    st.warning("Analysis Notes: " + " ".join(analyzer.warnings))
            
            # Display chart in second tab
            with tabs[1]:
                st.subheader("Price Chart with Weinstein Analysis")
                price_chart = analyzer.create_interactive_chart()
                st.plotly_chart(price_chart, use_container_width=True)
            
            # Display support and resistance levels in third tab
            with tabs[2]:
                st.subheader("Support & Resistance Levels")
                
                if analyzer.support_resistance_levels:
                    # Format levels for display
                    formatted_levels = []
                    for level in analyzer.support_resistance_levels:
                        # Calculate percentage distance from current price
                        if analyzer.last_price:
                            if level['type'] == 'resistance':
                                distance_pct = (level['price'] - analyzer.last_price) / analyzer.last_price * 100
                            else:  # support
                                distance_pct = (analyzer.last_price - level['price']) / analyzer.last_price * 100
                        else:
                            distance_pct = 0
                        
                        formatted_levels.append({
                            'Type': level['type'].capitalize(),
                            'Price': f"${level['price']:.2f}",
                            'Distance': f"{distance_pct:.1f}%",
                            'Strength': level['strength'].capitalize()
                        })
                    
                    # Create a dataframe for display
                    levels_df = pd.DataFrame(formatted_levels)
                    
                    # Apply conditional formatting
                    def highlight_level_type(row):
                        if row['Type'] == 'Support':
                            return ['background-color: rgba(0, 128, 0, 0.1); color: green' if col == 'Type' else '' for col in row.index]
                        elif row['Type'] == 'Resistance':
                            return ['background-color: rgba(255, 0, 0, 0.1); color: red' if col == 'Type' else '' for col in row.index]
                        return ['' for _ in row.index]
                    
                    def highlight_strength(row):
                        if row['Strength'] == 'Strong':
                            return ['font-weight: bold' if col == 'Strength' else '' for col in row.index]
                        return ['' for _ in row.index]
                    
                    # Display the table with styling
                    st.dataframe(
                        levels_df.style
                            .apply(highlight_level_type, axis=1)
                            .apply(highlight_strength, axis=1),
                        use_container_width=True
                    )
                else:
                    st.info("No support/resistance levels identified. This may be due to insufficient data or low volatility.")
            
            # Display volume profile in fourth tab
            with tabs[3]:
                st.subheader("Volume Profile Analysis")
                
                # Add a slider for lookback period
                lookback = st.slider(
                    "Lookback Period (number of periods)",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=10
                )
                
                volume_profile = analyzer.create_volume_profile(lookback_period=lookback)
                st.plotly_chart(volume_profile, use_container_width=True)
                
                st.markdown("""
                **About Volume Profile:**
                - Green bars represent volume at price levels above current price
                - Red bars represent volume at price levels below current price
                - The "Point of Control" (POC) is the price level with the highest volume
                - Significant volume nodes often act as support/resistance levels
                """)
            
            # Display detailed analysis in fifth tab
            with tabs[4]:
                st.subheader("Detailed Weinstein Analysis")
                
                # Format the detailed analysis with better styling
                detailed_text = analyzer.detailed_analysis
                if detailed_text:
                    # Split the analysis into sections
                    sections = detailed_text.split('\n\n')
                    
                    # Display each section with appropriate formatting
                    for i, section in enumerate(sections):
                        if i == 0:  # Phase analysis (first section)
                            st.markdown(f"### Phase Analysis\n{section}")
                        elif i == 1:  # Market and sector context
                            st.markdown(f"### Market & Sector Context\n{section}")
                        elif i == 2:  # Support and resistance levels
                            st.markdown(f"### Key Levels\n{section}")
                        elif i == 3:  # Recommendation
                            st.markdown(f"### {section}")
                else:
                    st.info("No detailed analysis available.")
        else:
            # Show error message if analysis failed
            if st.session_state.analyzer.errors:
                st.error(f"Error analyzing {ticker}: {st.session_state.analyzer.errors[0]}")
            else:
                st.error(f"Failed to analyze {ticker}. Please check the ticker symbol and try again.")

if __name__ == "__main__":
    main()

from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
import os
import time
from gnews import GNews

app = Flask(__name__)
CORS(app)

def detect_patterns(data):
    """Detect various stock patterns in the given data."""
    patterns = []
    
    if len(data) < 10:
        return patterns
    
    # Convert to numpy arrays for easier manipulation
    highs = data['High'].values
    lows = data['Low'].values
    closes = data['Close'].values
    timestamps = data.index.tolist()
    
    # Pattern detection functions
    def detect_double_bottom(highs, lows, closes, timestamps):
        """Detect Double Bottom (W) pattern"""
        patterns = []
        for i in range(2, len(lows) - 2):
            # Look for two lows with a peak in between
            if (lows[i-2] > lows[i-1] and lows[i-1] < lows[i] and 
                lows[i] > lows[i+1] and lows[i+1] < lows[i+2]):
                
                # Check if the two lows are similar (within 2%)
                first_low = lows[i-1]
                second_low = lows[i+1]
                if abs(first_low - second_low) / first_low <= 0.02:
                    patterns.append({
                        'type': 'Double Bottom',
                        'description': 'Bullish reversal pattern showing two distinct lows at similar price levels, indicating strong support.',
                        'points': [
                            {'x': timestamps[i-1], 'y': first_low, 'label': 'First Low'},
                            {'x': timestamps[i], 'y': highs[i], 'label': 'Peak'},
                            {'x': timestamps[i+1], 'y': second_low, 'label': 'Second Low'}
                        ],
                        'sentiment': 'bullish'
                    })
        return patterns
    
    def detect_double_top(highs, lows, closes, timestamps):
        """Detect Double Top (M) pattern"""
        patterns = []
        for i in range(2, len(highs) - 2):
            # Look for two highs with a low in between
            if (highs[i-2] < highs[i-1] and highs[i-1] > highs[i] and 
                highs[i] < highs[i+1] and highs[i+1] > highs[i+2]):
                
                # Check if the two highs are similar (within 2%)
                first_high = highs[i-1]
                second_high = highs[i+1]
                if abs(first_high - second_high) / first_high <= 0.02:
                    patterns.append({
                        'type': 'Double Top',
                        'description': 'Bearish reversal pattern with two peaks at similar price levels, indicating strong resistance.',
                        'points': [
                            {'x': timestamps[i-1], 'y': first_high, 'label': 'First Peak'},
                            {'x': timestamps[i], 'y': lows[i], 'label': 'Valley'},
                            {'x': timestamps[i+1], 'y': second_high, 'label': 'Second Peak'}
                        ],
                        'sentiment': 'bearish'
                    })
        return patterns
    
    def detect_bull_flag(highs, lows, closes, timestamps):
        """Detect Bull Flag pattern"""
        patterns = []
        for i in range(5, len(closes) - 3):
            # Look for rapid rise (flagpole)
            flagpole_start = i - 5
            flagpole_end = i
            price_change = (closes[flagpole_end] - closes[flagpole_start]) / closes[flagpole_start]
            
            if price_change > 0.03:  # 3% rise
                # Check for consolidation (flag)
                flag_highs = highs[flagpole_end:flagpole_end+3]
                flag_lows = lows[flagpole_end:flagpole_end+3]
                
                if len(flag_highs) >= 3 and np.std(flag_highs) < closes[flagpole_end] * 0.01:
                    patterns.append({
                        'type': 'Bull Flag',
                        'description': 'Bullish continuation pattern with a strong upward move followed by a small consolidation period.',
                        'points': [
                            {'x': timestamps[flagpole_start], 'y': closes[flagpole_start], 'label': 'Flagpole Start'},
                            {'x': timestamps[flagpole_end], 'y': closes[flagpole_end], 'label': 'Flagpole End'},
                            {'x': timestamps[flagpole_end+2], 'y': closes[flagpole_end+2], 'label': 'Flag End'}
                        ],
                        'sentiment': 'bullish'
                    })
        return patterns
    
    def detect_ascending_triangle(highs, lows, closes, timestamps):
        """Detect Ascending Triangle pattern"""
        patterns = []
        window = 7
        for i in range(window, len(highs) - 3):
            recent_highs = highs[i-window:i]
            recent_lows = lows[i-window:i]
            recent_times = timestamps[i-window:i]
            
            # Check for flat resistance (similar highs)
            high_std = np.std(recent_highs)
            if high_std < np.mean(recent_highs) * 0.015:  # Low variance in highs
                # Check for rising support (upward trending lows)
                low_slope = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
                if low_slope > 0:  # Positive slope
                    patterns.append({
                        'type': 'Ascending Triangle',
                        'description': 'Bullish pattern with flat resistance and rising support levels, often leads to upward breakout.',
                        'points': [
                            {'x': recent_times[0], 'y': recent_highs[0], 'label': 'Resistance Start'},
                            {'x': recent_times[-1], 'y': recent_highs[-1], 'label': 'Resistance End'},
                            {'x': recent_times[0], 'y': recent_lows[0], 'label': 'Support Start'},
                            {'x': recent_times[-1], 'y': recent_lows[-1], 'label': 'Support End'}
                        ],
                        'sentiment': 'bullish'
                    })
        return patterns
    
    def detect_head_and_shoulders(highs, lows, closes, timestamps):
        """Detect Head and Shoulders pattern"""
        patterns = []
        for i in range(4, len(highs) - 4):
            # Look for three peaks: shoulder, head, shoulder
            left_shoulder = highs[i-2]
            head = highs[i]
            right_shoulder = highs[i+2]
            
            # Head should be higher than both shoulders
            if (head > left_shoulder and head > right_shoulder and 
                abs(left_shoulder - right_shoulder) / left_shoulder <= 0.03):
                
                # Find neckline (lows between shoulders and head)
                left_valley = lows[i-1]
                right_valley = lows[i+1]
                neckline = (left_valley + right_valley) / 2
                
                patterns.append({
                    'type': 'Head and Shoulders',
                    'description': 'Bearish reversal pattern with three peaks where the middle peak (head) is highest.',
                    'points': [
                        {'x': timestamps[i-2], 'y': left_shoulder, 'label': 'Left Shoulder'},
                        {'x': timestamps[i], 'y': head, 'label': 'Head'},
                        {'x': timestamps[i+2], 'y': right_shoulder, 'label': 'Right Shoulder'},
                        {'x': timestamps[i-1], 'y': neckline, 'label': 'Neckline'}
                    ],
                    'sentiment': 'bearish'
                })
        return patterns
    
    # Detect all patterns
    all_patterns = []
    all_patterns.extend(detect_double_bottom(highs, lows, closes, timestamps))
    all_patterns.extend(detect_double_top(highs, lows, closes, timestamps))
    all_patterns.extend(detect_bull_flag(highs, lows, closes, timestamps))
    all_patterns.extend(detect_ascending_triangle(highs, lows, closes, timestamps))
    all_patterns.extend(detect_head_and_shoulders(highs, lows, closes, timestamps))
    
    return all_patterns

def get_pattern_analysis(ticker_symbol):
    """Get 15-minute data for today and detect patterns"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Get 15-minute data for today
        today = datetime.now().date()
        data_15m = ticker.history(period="1d", interval="15m")
        
        if data_15m.empty:
            return {
                'chartData': [],
                'patterns': [],
                'error': 'No 15-minute data available for today'
            }
        
        # Convert data for frontend
        chart_data = []
        for timestamp, row in data_15m.iterrows():
            chart_data.append({
                'timestamp': timestamp.isoformat(),
                'open': round(float(row['Open']), 2),
                'high': round(float(row['High']), 2),
                'low': round(float(row['Low']), 2),
                'close': round(float(row['Close']), 2),
                'volume': int(row['Volume'])
            })
        
        # Detect patterns
        patterns = detect_patterns(data_15m)
        
        # Convert timestamps in patterns to ISO format
        for pattern in patterns:
            for point in pattern['points']:
                if hasattr(point['x'], 'isoformat'):
                    point['x'] = point['x'].isoformat()
        
        return {
            'chartData': chart_data,
            'patterns': patterns,
            'error': None
        }
        
    except Exception as e:
        print(f"Error in pattern analysis: {str(e)}")
        return {
            'chartData': [],
            'patterns': [],
            'error': str(e)
        }

def get_sentiment(text):
    """Calculate sentiment score using TextBlob."""
    analysis = TextBlob(text)
    score = analysis.sentiment.polarity
    
    if score > 0.1:
        return 'positive', score
    elif score < -0.1:
        return 'negative', score
    else:
        return 'neutral', score

def calculate_pivot_predictions(ticker_symbol):
    """Calculate pivot price predictions for different timeframes using real historical data."""
    print(f"\n========== STARTING PIVOT CALCULATION FOR {ticker_symbol} ==========")
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        predictions = []
        
        # Daily timeframe calculation
        print(f"[PIVOT] Fetching daily data for {ticker_symbol}...")
        try:
            data_daily = ticker.history(period="30d", interval="1d")
            print(f"[PIVOT] Daily data retrieved: {len(data_daily)} rows")
            
            if not data_daily.empty and len(data_daily) >= 2:
                # Use last complete trading day
                last_day = data_daily.iloc[-2] if len(data_daily) > 1 else data_daily.iloc[-1]
                high_daily = float(last_day['High'])
                low_daily = float(last_day['Low'])
                close_daily = float(last_day['Close'])
                pivot_daily = (high_daily + low_daily + close_daily) / 3
                
                daily_prediction = {
                    'period': 'Daily',
                    'high': round(high_daily, 2),
                    'low': round(low_daily, 2),
                    'close': round(close_daily, 2),
                    'pivot': round(pivot_daily, 2),
                    'resistance1': round((2 * pivot_daily) - low_daily, 2),
                    'support1': round((2 * pivot_daily) - high_daily, 2)
                }
                predictions.append(daily_prediction)
                print(f"[PIVOT] Daily prediction added: {daily_prediction}")
            else:
                print(f"[PIVOT] Daily data insufficient: {len(data_daily)} rows")
        except Exception as daily_error:
            print(f"[PIVOT] Daily data error: {str(daily_error)}")
        
        # Hourly timeframe calculation
        print(f"[PIVOT] Fetching hourly data for {ticker_symbol}...")
        try:
            data_hourly = ticker.history(period="7d", interval="1h")
            print(f"[PIVOT] Hourly data retrieved: {len(data_hourly)} rows")
            
            if not data_hourly.empty and len(data_hourly) >= 2:
                last_hour = data_hourly.iloc[-2] if len(data_hourly) > 1 else data_hourly.iloc[-1]
                high_hourly = float(last_hour['High'])
                low_hourly = float(last_hour['Low'])
                close_hourly = float(last_hour['Close'])
                pivot_hourly = (high_hourly + low_hourly + close_hourly) / 3
                
                hourly_prediction = {
                    'period': '1-hour',
                    'high': round(high_hourly, 2),
                    'low': round(low_hourly, 2),
                    'close': round(close_hourly, 2),
                    'pivot': round(pivot_hourly, 2),
                    'resistance1': round((2 * pivot_hourly) - low_hourly, 2),
                    'support1': round((2 * pivot_hourly) - high_hourly, 2)
                }
                predictions.append(hourly_prediction)
                print(f"[PIVOT] Hourly prediction added: {hourly_prediction}")
            else:
                print(f"[PIVOT] Hourly data insufficient: {len(data_hourly)} rows")
        except Exception as hourly_error:
            print(f"[PIVOT] Hourly data error: {str(hourly_error)}")
        
        # 15-minute timeframe calculation
        print(f"[PIVOT] Fetching 15-min data for {ticker_symbol}...")
        try:
            data_15m = ticker.history(period="3d", interval="15m")
            print(f"[PIVOT] 15-min data retrieved: {len(data_15m)} rows")
            
            if not data_15m.empty and len(data_15m) >= 2:
                last_15m = data_15m.iloc[-2] if len(data_15m) > 1 else data_15m.iloc[-1]
                high_15m = float(last_15m['High'])
                low_15m = float(last_15m['Low'])
                close_15m = float(last_15m['Close'])
                pivot_15m = (high_15m + low_15m + close_15m) / 3
                
                min15_prediction = {
                    'period': '15-min',
                    'high': round(high_15m, 2),
                    'low': round(low_15m, 2),
                    'close': round(close_15m, 2),
                    'pivot': round(pivot_15m, 2),
                    'resistance1': round((2 * pivot_15m) - low_15m, 2),
                    'support1': round((2 * pivot_15m) - high_15m, 2)
                }
                predictions.append(min15_prediction)
                print(f"[PIVOT] 15-min prediction added: {min15_prediction}")
            else:
                print(f"[PIVOT] 15-min data insufficient: {len(data_15m)} rows")
        except Exception as min15_error:
            print(f"[PIVOT] 15-min data error: {str(min15_error)}")
        
        # Fallback calculation if no predictions were generated
        if not predictions:
            print(f"[PIVOT] No predictions generated, attempting fallback for {ticker_symbol}")
            try:
                fallback_data = ticker.history(period="5d", interval="1d")
                if not fallback_data.empty:
                    last_data = fallback_data.iloc[-1]
                    high_price = float(last_data['High'])
                    low_price = float(last_data['Low'])
                    close_price = float(last_data['Close'])
                    pivot = (high_price + low_price + close_price) / 3
                    
                    fallback_prediction = {
                        'period': 'Fallback Daily',
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close_price, 2),
                        'pivot': round(pivot, 2),
                        'resistance1': round((2 * pivot) - low_price, 2),
                        'support1': round((2 * pivot) - high_price, 2)
                    }
                    predictions.append(fallback_prediction)
                    print(f"[PIVOT] Fallback prediction added: {fallback_prediction}")
            except Exception as fallback_error:
                print(f"[PIVOT] Fallback calculation failed: {str(fallback_error)}")
        
        print(f"[PIVOT] Final predictions count for {ticker_symbol}: {len(predictions)}")
        print(f"========== COMPLETED PIVOT CALCULATION FOR {ticker_symbol} ==========\n")
        return predictions
        
    except Exception as main_error:
        print(f"[PIVOT] MAJOR ERROR in pivot calculation for {ticker_symbol}: {str(main_error)}")
        print(f"========== FAILED PIVOT CALCULATION FOR {ticker_symbol} ==========\n")
        return []

def get_technical_indicators(ticker_data):
    """Calculate technical indicators from stock data."""
    df = ticker_data.copy()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate SMA and EMA
    sma = df['Close'].rolling(window=20).mean()
    ema = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Calculate Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    upper_band = sma20 + (std20 * 2)
    lower_band = sma20 - (std20 * 2)
    
    # Create a price prediction using Bollinger Bands
    # If price is near lower band, there's potential for upward movement
    # If price is near upper band, there's potential for downward movement
    last_close = df['Close'].iloc[-1]
    last_upper = upper_band.iloc[-1]
    last_lower = lower_band.iloc[-1]
    last_middle = sma20.iloc[-1]
    band_width = last_upper - last_lower
    
    # Calculate where current price is within the bands
    band_position = (last_close - last_lower) / (band_width if band_width > 0 else 1)
    
    # Predict based on position in bands
    if band_position <= 0.2:  # Near lower band
        prediction = "Potential upward movement, price near lower band"
        prediction_sentiment = 0.5
    elif band_position >= 0.8:  # Near upper band
        prediction = "Potential downward movement, price near upper band"
        prediction_sentiment = -0.5
    elif band_position < 0.4:  # Below middle band
        prediction = "Moderately bullish, price below center of bands"
        prediction_sentiment = 0.2
    elif band_position > 0.6:  # Above middle band
        prediction = "Moderately bearish, price above center of bands"
        prediction_sentiment = -0.2
    else:  # Near middle band
        prediction = "Neutral, price near middle of bands"
        prediction_sentiment = 0
    
    # Get the latest values
    last_rsi = rsi.iloc[-1] if not rsi.empty else 50
    last_sma = sma.iloc[-1] if not sma.empty else df['Close'].iloc[-1]
    last_ema = ema.iloc[-1] if not ema.empty else df['Close'].iloc[-1]
    
    last_macd = {
        'macd': macd_line.iloc[-1] if not macd_line.empty else 0,
        'signal': signal_line.iloc[-1] if not signal_line.empty else 0, 
        'histogram': histogram.iloc[-1] if not histogram.empty else 0
    }
    
    last_bb = {
        'upper': upper_band.iloc[-1] if not upper_band.empty else df['Close'].iloc[-1] * 1.05,
        'middle': sma20.iloc[-1] if not sma20.empty else df['Close'].iloc[-1],
        'lower': lower_band.iloc[-1] if not lower_band.empty else df['Close'].iloc[-1] * 0.95,
        'prediction': prediction,
        'predictionSentiment': prediction_sentiment
    }
    
    # Determine RSI sentiment more precisely
    rsi_sentiment = 0
    if last_rsi > 70:
        rsi_sentiment = -0.7  # Strongly overbought - negative
    elif last_rsi > 60:
        rsi_sentiment = -0.3  # Mildly overbought - slightly negative
    elif last_rsi < 30:
        rsi_sentiment = 0.7   # Strongly oversold - positive
    elif last_rsi < 40:
        rsi_sentiment = 0.3   # Mildly oversold - slightly positive
    
    # MACD sentiment calculation
    macd_sentiment = 0
    if last_macd['histogram'] > 1:
        macd_sentiment = 0.6  # Strong bullish signal
    elif last_macd['histogram'] > 0:
        macd_sentiment = 0.3  # Mild bullish signal
    elif last_macd['histogram'] < -1:
        macd_sentiment = -0.6  # Strong bearish signal
    elif last_macd['histogram'] < 0:
        macd_sentiment = -0.3  # Mild bearish signal
    
    return {
        'rsi': round(float(last_rsi), 2),
        'rsiSentiment': round(float(rsi_sentiment), 2),
        'sma': round(float(last_sma), 2),
        'ema': round(float(last_ema), 2),
        'macd': {
            'macd': round(float(last_macd['macd']), 2),
            'signal': round(float(last_macd['signal']), 2),
            'histogram': round(float(last_macd['histogram']), 2),
            'sentiment': round(float(macd_sentiment), 2)
        },
        'bollingerBands': {
            'upper': round(float(last_bb['upper']), 2),
            'middle': round(float(last_bb['middle']), 2),
            'lower': round(float(last_bb['lower']), 2),
            'prediction': last_bb['prediction'],
            'predictionSentiment': round(float(last_bb['predictionSentiment']), 2)
        }
    }

def calculate_overall_sentiment(indicators, news_sentiment=None):
    """Calculate overall sentiment based on technical indicators and news."""
    # Technical signals - more sophisticated approach
    tech_signals = []
    
    # RSI signal
    if indicators['rsiSentiment'] is not None:
        tech_signals.append(indicators['rsiSentiment'])
    
    # MACD signal
    if indicators['macd']['sentiment'] is not None:
        tech_signals.append(indicators['macd']['sentiment'])
    
    # Bollinger Bands prediction
    if indicators['bollingerBands']['predictionSentiment'] is not None:
        tech_signals.append(indicators['bollingerBands']['predictionSentiment'])
    
    # Combined sentiment from technical indicators
    tech_score = sum(tech_signals) / len(tech_signals) if tech_signals else 0
    
    # If we have news sentiment, include it in the overall calculation
    if news_sentiment is not None:
        overall_score = (tech_score * 0.7) + (news_sentiment * 0.3)  # 70% technical, 30% news
    else:
        overall_score = tech_score
    
    sentiment_type = 'neutral'
    if overall_score > 0.15:
        sentiment_type = 'positive'
    elif overall_score < -0.15:
        sentiment_type = 'negative'
        
    return sentiment_type, round(overall_score, 2)

def get_news_for_stock(company_name, symbol=None, is_indian=False):
    """Get news articles for a specific stock."""
    try:
        # Initialize GNews with language and period
        news_client = GNews(language='en', period='7d', max_results=8)
        
        # Create search terms - try multiple approaches for Indian stocks
        search_terms = []
        
        if is_indian:
            # For Indian stocks, try multiple variations
            if symbol:
                search_terms = [
                    symbol,                     # Just the symbol
                    f"{symbol} stock",          # Symbol + stock
                    f"{symbol} share price",    # Symbol + share price
                    company_name.split('Limited')[0].strip()  # Company name without "Limited"
                ]
            else:
                search_terms = [company_name]
        else:
            # For foreign stocks
            search_terms = [company_name, symbol] if symbol else [company_name]
        
        # Try each search term until we get results
        news_articles = []
        for term in search_terms:
            print(f"Searching news for term: {term}")
            results = news_client.get_news(term)
            if results:
                news_articles = results
                print(f"Found {len(news_articles)} articles with term: {term}")
                break
                
        if not news_articles:
            print("No news articles found for any search term")
            return [], 0
        
        # Process and analyze sentiment for each article
        processed_articles = []
        total_sentiment_score = 0
        
        for article in news_articles:
            # Extract relevant info
            title = article.get('title', '')
            
            # Skip articles that are not relevant (basic filtering)
            if is_indian and symbol and symbol.upper() not in title.upper() and "stock" not in title.lower() and "market" not in title.lower():
                # Try to check if the company name without "Limited" is in the title
                company_simple = company_name.split('Limited')[0].strip()
                if company_simple not in title:
                    continue
            
            # Get sentiment of the title
            sentiment_type, sentiment_score = get_sentiment(title)
            
            # Add sentiment info to article
            article_data = {
                'title': title,
                'url': article.get('url', ''),
                'published': article.get('published date', ''),
                'source': article.get('publisher', {}).get('title', 'Unknown'),
                'sentiment': sentiment_type,
                'sentimentScore': sentiment_score
            }
            
            processed_articles.append(article_data)
            total_sentiment_score += sentiment_score
            
            # Limit to 5 most relevant articles
            if len(processed_articles) >= 5:
                break
        
        # Calculate average sentiment
        avg_sentiment = total_sentiment_score / len(processed_articles) if processed_articles else 0
        
        return processed_articles, avg_sentiment
        
    except Exception as e:
        print(f"Error getting news: {str(e)}")
        return [], 0

@app.route('/api/stock', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol', '')
    market_type = request.args.get('marketType', 'Foreign')
    
    print(f"\n🚀 API REQUEST STARTED for symbol: {symbol}, market: {market_type}")
    
    if not symbol:
        print("❌ ERROR: No symbol provided")
        return jsonify({'error': 'Symbol is required'}), 400
    
    try:
        # Convert symbol based on market type
        if market_type == 'Indian':
            yfinance_symbol = f"{symbol}.BO"
            tradingview_symbol = f"BSE:{symbol}"
            display_symbol = symbol
            currency_symbol = "₹"
            is_indian = True
            print(f"📊 Indian stock: {symbol} -> {yfinance_symbol}")
        else:
            yfinance_symbol = symbol
            tradingview_symbol = symbol
            display_symbol = symbol
            currency_symbol = "$"
            is_indian = False
            print(f"🌍 Foreign stock: {symbol}")
        
        # Get stock data from yfinance
        print(f"📈 Fetching stock data for {yfinance_symbol}...")
        ticker = yf.Ticker(yfinance_symbol)
        info = ticker.info
        
        # Get historical data for indicators
        hist = ticker.history(period="6mo")
        print(f"📊 Historical data retrieved: {len(hist)} rows")
        
        if hist.empty:
            print(f"❌ ERROR: No historical data found for {yfinance_symbol}")
            return jsonify({'error': 'No data found for the symbol'}), 404
        
        # Get current price and previous close
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        
        # Calculate change and change percent
        change = current_price - previous_close
        change_percent = (change / previous_close) * 100
        
        print(f"💰 Current price: {current_price}, Change: {change} ({change_percent:.2f}%)")
        
        # Get full name - for Indian stocks, clean up the name if needed
        full_name = info.get('longName', f"{display_symbol} Inc.")
        print(f"🏢 Company name: {full_name}")
        
        # Calculate technical indicators with enhanced sentiment
        print("🔢 Calculating technical indicators...")
        indicators = get_technical_indicators(hist)
        print(f"✅ Technical indicators calculated: RSI={indicators['rsi']}")
        
        # Calculate pivot predictions using real data with enhanced debugging
        print(f"🎯 Starting pivot predictions calculation...")
        pivot_predictions = calculate_pivot_predictions(yfinance_symbol)
        print(f"🎯 PIVOT PREDICTIONS RESULT: {len(pivot_predictions)} predictions generated")
        for i, pred in enumerate(pivot_predictions):
            print(f"   Prediction {i+1}: {pred['period']} - Pivot: {pred['pivot']}")
        
        # Get pattern analysis
        print("🔍 Analyzing patterns...")
        pattern_analysis = get_pattern_analysis(yfinance_symbol)
        print(f"🔍 PATTERN ANALYSIS: Found {len(pattern_analysis['patterns'])} patterns")
        
        # Get news and calculate news sentiment
        print("📰 Fetching news...")
        news_articles, news_sentiment = get_news_for_stock(full_name, symbol, is_indian)
        print(f"📰 News fetched: {len(news_articles)} articles, sentiment: {news_sentiment}")
        
        # Calculate overall sentiment with improved weighting
        sentiment, sentiment_score = calculate_overall_sentiment(indicators, news_sentiment)
        print(f"🎭 Overall sentiment: {sentiment} (score: {sentiment_score})")
        
        response = {
            'symbol': symbol,
            'displaySymbol': display_symbol,
            'fullName': full_name,
            'currentPrice': round(current_price, 2),
            'previousClose': round(previous_close, 2),
            'change': round(change, 2),
            'changePercent': round(change_percent, 2),
            'marketType': market_type,
            'currencySymbol': currency_symbol,
            'yfinanceSymbol': yfinance_symbol,
            'tradingViewSymbol': tradingview_symbol,
            'sentiment': sentiment,
            'sentimentScore': sentiment_score,
            'indicators': indicators,
            'pivotPredictions': pivot_predictions,
            'patternAnalysis': pattern_analysis,
            'news': {
                'articles': news_articles,
                'sentiment': news_sentiment
            },
            'isLoading': False,
            'error': None
        }
        
        print(f"✅ API RESPONSE READY with {len(pivot_predictions)} pivot predictions and {len(pattern_analysis['patterns'])} patterns")
        print(f"🏁 API REQUEST COMPLETED SUCCESSFULLY for {symbol}\n")
        return jsonify(response)
        
    except Exception as e:
        print(f"💥 CRITICAL ERROR in API: {str(e)}")
        print(f"🏁 API REQUEST FAILED for {symbol}\n")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'status': 'Stock Sentiment API is running'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)

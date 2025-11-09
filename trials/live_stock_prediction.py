import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
import joblib
import yfinance as yf
import tweepy
from textblob import TextBlob
from datetime import datetime, timedelta
import hashlib
import re
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import warnings
import time
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Live Stock Prediction & Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_cache' not in st.session_state:
    st.session_state.model_cache = {}
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'scaler_cache' not in st.session_state:
    st.session_state.scaler_cache = {}

class ModelConfig:
    """Configuration class to track model parameters"""
    def __init__(self, stock_ticker, start_date, end_date, prediction_days=60, lstm_units=64, dropout_rate=0.2):
        self.stock_ticker = stock_ticker
        self.start_date = start_date.strftime('%Y-%m-%d')
        self.end_date = end_date.strftime('%Y-%m-%d')
        self.prediction_days = prediction_days
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
    
    def get_hash(self):
        """Generate hash for model configuration"""
        config_str = f"{self.stock_ticker}_{self.start_date}_{self.end_date}_{self.prediction_days}_{self.lstm_units}_{self.dropout_rate}"
        return hashlib.md5(config_str.encode()).hexdigest()

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_data(ticker, start_date, end_date):
    """Fetch real-time stock data from yfinance"""
    try:
        with st.spinner(f"Fetching stock data for {ticker}..."):
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for ticker {ticker}")
                return None
            
            # Clean column names
            data.columns = [col.replace(' ', '_') for col in data.columns]
            data.index = pd.to_datetime(data.index.date)
            data = data[~data.index.duplicated(keep='first')]
            
            st.success(f"✅ Fetched {len(data)} records for {ticker}")
            return data
    except Exception as e:
        st.error(f"Error fetching stock data: {str(e)}")
        return None

def setup_twitter_api():
    """Setup Twitter API with credentials"""
    # You need to add your Twitter API credentials here
    # For demo purposes, we'll simulate the API
    try:
        # Uncomment and add your credentials when available:
        # consumer_key = "your_consumer_key"
        # consumer_secret = "your_consumer_secret"
        # access_token = "your_access_token"
        # access_token_secret = "your_access_token_secret"
        # 
        # auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        # auth.set_access_token(access_token, access_token_secret)
        # api = tweepy.API(auth, wait_on_rate_limit=True)
        # return api
        
        # For now, return None to use simulated sentiment
        return None
    except Exception as e:
        st.warning(f"Twitter API setup failed: {str(e)}")
        return None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def fetch_twitter_sentiment(ticker, api=None, count=100):
    """Fetch real-time Twitter sentiment or simulate if API unavailable"""
    try:
        if api is not None:
            # Real Twitter API implementation
            tweets = []
            query = f"${ticker} OR {ticker.replace('.NS', '').replace('.BO', '')} -RT"
            
            for tweet in tweepy.Cursor(api.search_tweets, 
                                     q=query, 
                                     lang="en", 
                                     result_type="recent").items(count):
                tweets.append(tweet.text)
            
            # Analyze sentiment
            sentiments = []
            for tweet in tweets:
                # Clean tweet text
                cleaned_tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
                cleaned_tweet = re.sub(r'\@\w+|\#','', cleaned_tweet)
                
                # Get sentiment
                blob = TextBlob(cleaned_tweet)
                sentiments.append(blob.sentiment.polarity)
            
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            
        else:
            # Simulated sentiment based on market hours and ticker patterns
            current_hour = datetime.now().hour
            ticker_hash = hash(ticker) % 100
            
            # Market hours sentiment simulation
            if 9 <= current_hour <= 16:
                base_sentiment = 0.1  # Slightly positive during market hours
            else:
                base_sentiment = 0.0  # Neutral off-hours
            
            # Add some ticker-specific variation
            ticker_variation = (ticker_hash - 50) / 500  # Small variation based on ticker
            
            # Add some randomness
            random_factor = np.random.uniform(-0.1, 0.1)
            
            avg_sentiment = np.clip(base_sentiment + ticker_variation + random_factor, -1, 1)
        
        return avg_sentiment
        
    except Exception as e:
        st.warning(f"Error fetching sentiment: {str(e)}")
        return 0.0

@st.cache_data(ttl=1800)  # Cache for 30 minutes  
def generate_sentiment_timeseries(stock_data, ticker, api=None):
    """Generate sentiment time series for stock data"""
    try:
        sentiments = []
        
        # Get overall sentiment for the ticker
        base_sentiment = fetch_twitter_sentiment(ticker, api)
        
        # Generate daily sentiment variations
        for i, date in enumerate(stock_data.index):
            # Add some daily variation around the base sentiment
            daily_variation = np.random.uniform(-0.2, 0.2)
            
            # Consider price movement for sentiment direction
            if i > 0:
                price_change = (stock_data['Close'].iloc[i] - stock_data['Close'].iloc[i-1]) / stock_data['Close'].iloc[i-1]
                sentiment_influence = price_change * 0.5  # Price movement influences sentiment
            else:
                sentiment_influence = 0
            
            daily_sentiment = np.clip(base_sentiment + daily_variation + sentiment_influence, -1, 1)
            sentiments.append(daily_sentiment)
        
        stock_data = stock_data.copy()
        stock_data['Sentiment'] = sentiments
        
        return stock_data
        
    except Exception as e:
        st.error(f"Error generating sentiment timeseries: {str(e)}")
        stock_data = stock_data.copy()
        stock_data['Sentiment'] = np.random.uniform(-0.1, 0.1, len(stock_data))
        return stock_data

@st.cache_data
def load_demo_data():
    """Load demo data files for professor demonstration"""
    demo_data = {}
    
    # Load demo CSV
    if os.path.exists("sss.csv"):
        try:
            csv_data = pd.read_csv("sss.csv")
            csv_data['Date'] = pd.to_datetime(csv_data['Date'])
            csv_data.set_index('Date', inplace=True)
            demo_data['stock_data'] = csv_data
        except Exception as e:
            st.warning(f"Could not load demo CSV: {str(e)}")
    
    # Load demo JSON sentiment files
    json_files = ["simulated_July_news_2025.json", "simulated_August_news_2025.json"]
    all_sentiment = {}
    
    for file_path in json_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_sentiment.update(data)
            except Exception as e:
                st.warning(f"Could not load {file_path}: {str(e)}")
    
    if all_sentiment:
        # Process sentiment data
        sentiment_scores = []
        dates = []
        
        for date_key, articles in all_sentiment.items():
            try:
                date_obj = pd.to_datetime(date_key)
                dates.append(date_obj)
                
                daily_sentiments = []
                for article in articles:
                    if 'ticker_sentiment' in article:
                        for ticker_data in article['ticker_sentiment']:
                            if 'ticker_sentiment_score' in ticker_data:
                                try:
                                    score = float(ticker_data['ticker_sentiment_score'])
                                    daily_sentiments.append(score)
                                except (ValueError, TypeError):
                                    continue
                
                avg_sentiment = np.mean(daily_sentiments) if daily_sentiments else 0.0
                sentiment_scores.append(avg_sentiment)
            except Exception:
                continue
        
        if dates and sentiment_scores:
            sentiment_df = pd.DataFrame({
                'Date': dates,
                'Sentiment': sentiment_scores
            })
            sentiment_df.set_index('Date', inplace=True)
            demo_data['sentiment_data'] = sentiment_df
    
    return demo_data

@st.cache_resource
def create_lstm_model(input_shape, config):
    """Create and cache LSTM model"""
    input_timesteps, input_features = input_shape
    
    inputs = Input(shape=(input_timesteps, input_features))
    
    # LSTM Encoder
    lstm_out, state_h, state_c = LSTM(
        config.lstm_units, return_state=True, return_sequences=True
    )(inputs)
    lstm_out = Dropout(config.dropout_rate)(lstm_out)
    
    # Self-Attention Block
    attention = Attention(use_scale=True)([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attention])
    normed = LayerNormalization()(concat)
    
    # LSTM Decoder
    lstm_out2 = LSTM(config.lstm_units, return_sequences=False)(normed)
    lstm_out2 = Dropout(config.dropout_rate)(lstm_out2)
    
    # Output
    dense_out = Dense(max(16, input_features), activation="relu")(lstm_out2)
    output = Dense(1)(dense_out)
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

def save_model_artifacts(model, close_scaler, sentiment_scaler, config, model_hash):
    """Save model and associated artifacts"""
    try:
        model_dir = f"models/{model_hash}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model.save(f"{model_dir}/model.keras")
        
        # Save scalers
        joblib.dump(close_scaler, f"{model_dir}/close_scaler.pkl")
        joblib.dump(sentiment_scaler, f"{model_dir}/sentiment_scaler.pkl")
        
        # Save config
        with open(f"{model_dir}/config.pkl", 'wb') as f:
            pickle.dump(config, f)
        
        st.success(f"✅ Model saved: {model_hash[:8]}...")
        
    except Exception as e:
        st.warning(f"Could not save model: {str(e)}")

def load_model_artifacts(model_hash):
    """Load model and associated artifacts"""
    try:
        model_dir = f"models/{model_hash}"
        
        if not os.path.exists(model_dir):
            return None, None, None, None
        
        # Load model
        model = load_model(f"{model_dir}/model.keras")
        
        # Load scalers
        close_scaler = joblib.load(f"{model_dir}/close_scaler.pkl")
        sentiment_scaler = joblib.load(f"{model_dir}/sentiment_scaler.pkl")
        
        # Load config
        with open(f"{model_dir}/config.pkl", 'rb') as f:
            config = pickle.load(f)
        
        return model, close_scaler, sentiment_scaler, config
        
    except Exception as e:
        st.warning(f"Could not load saved model: {str(e)}")
        return None, None, None, None

def prepare_data_for_training(data, prediction_days):
    """Prepare data for model training"""
    close_scaler = MinMaxScaler()
    sentiment_scaler = MinMaxScaler()
    
    scaled_close = close_scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    scaled_sentiment = sentiment_scaler.fit_transform(data['Sentiment'].values.reshape(-1, 1))
    
    scaled_data = np.hstack([scaled_close, scaled_sentiment])
    
    # Create sequences
    x_full, y_full = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_full.append(scaled_data[i-prediction_days:i])
        y_full.append(scaled_data[i, 0])
    
    return np.array(x_full), np.array(y_full), close_scaler, sentiment_scaler, scaled_data

def train_or_load_model(data, config):
    """Train new model or load existing one"""
    model_hash = config.get_hash()
    
    # Try to load existing model
    model, close_scaler, sentiment_scaler, loaded_config = load_model_artifacts(model_hash)
    
    if model is not None:
        st.success(f"✅ Loaded cached model: {model_hash[:8]}...")
        # Prepare data with loaded scalers
        scaled_close = close_scaler.transform(data['Close'].values.reshape(-1, 1))
        scaled_sentiment = sentiment_scaler.transform(data['Sentiment'].values.reshape(-1, 1))
        scaled_data = np.hstack([scaled_close, scaled_sentiment])
        return model, close_scaler, sentiment_scaler, scaled_data
    
    # Train new model
    st.info("🔄 Training new model...")
    
    x_full, y_full, close_scaler, sentiment_scaler, scaled_data = prepare_data_for_training(
        data, config.prediction_days
    )
    
    if len(x_full) == 0:
        st.error("❌ Insufficient data for training!")
        return None, None, None, None
    
    # Create and train model
    model = create_lstm_model((x_full.shape[1], x_full.shape[2]), config)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
    )
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Custom callback to update progress
    class ProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress = (epoch + 1) / 50
            progress_bar.progress(progress)
            status_text.text(f"Training... Epoch {epoch+1}/50 - Loss: {logs.get('loss', 0):.4f}")
    
    with st.spinner("Training model..."):
        history = model.fit(
            x_full, y_full,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[reduce_lr, ProgressCallback()]
        )
    
    progress_bar.empty()
    status_text.empty()
    
    # Save model artifacts
    save_model_artifacts(model, close_scaler, sentiment_scaler, config, model_hash)
    
    return model, close_scaler, sentiment_scaler, scaled_data

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    data = data.copy()
    
    # RSI
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['MA20'] + (rolling_std * 2)
    data['Lower_Band'] = data['MA20'] - (rolling_std * 2)
    
    # MACD
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def generate_predictions(model, scaled_data, close_scaler, prediction_days):
    """Generate predictions for the dataset"""
    predictions = []
    
    for i in range(prediction_days, len(scaled_data)):
        sequence = scaled_data[i-prediction_days:i]
        pred = model.predict(sequence.reshape(1, prediction_days, 2), verbose=0)
        predictions.append(pred[0, 0])
    
    # Inverse transform predictions
    pred_copies = np.repeat(np.array(predictions).reshape(-1, 1), 2, axis=-1)
    pred_copies[:, 1] = scaled_data[prediction_days:, 1]
    predictions_scaled = close_scaler.inverse_transform(pred_copies)[:, 0]
    
    return predictions_scaled

def simulate_trades(data, initial_capital=100000):
    """Simulate trading strategy"""
    trades = []
    capital = initial_capital
    position = None
    quantity = 100
    
    for i in range(1, len(data)):
        current_signal = data['Signal'].iloc[i]
        
        if current_signal == "Buy" and position is None:
            position = {
                "Buy Date": data.index[i],
                "Buy Price": data['Close'].iloc[i]
            }
        elif current_signal == "Sell" and position is not None:
            sell_price = data['Close'].iloc[i]
            profit = (sell_price - position["Buy Price"]) * quantity
            capital += profit
            trades.append({
                "Buy Date": position["Buy Date"],
                "Sell Date": data.index[i],
                "Buy Price": position["Buy Price"],
                "Sell Price": sell_price,
                "Quantity": quantity,
                "Profit": profit
            })
            position = None
    
    trade_log = pd.DataFrame(trades)
    total_profit = trade_log['Profit'].sum() if not trade_log.empty else 0
    avg_profit = trade_log['Profit'].mean() if not trade_log.empty else 0
    win_rate = (trade_log['Profit'] > 0).mean() * 100 if not trade_log.empty else 0
    
    return trade_log, total_profit, avg_profit, win_rate, capital

def main():
    st.title("📈 Live Stock Prediction & Sentiment Analysis")
    st.markdown("**Real-time data from yfinance + Twitter sentiment analysis**")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Data source selection
    data_mode = st.sidebar.selectbox(
        "Data Source",
        ["🔴 Live API Data (Recommended)", "📁 Demo Files (For Professor)"]
    )
    
    if data_mode == "🔴 Live API Data (Recommended)":
        # Stock selection
        col1, col2 = st.sidebar.columns(2)
        with col1:
            stock_ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS", help="e.g., AAPL, RELIANCE.NS, TSLA")
        
        # Date selection
        st.sidebar.subheader("📅 Date Range")
        
        # Get max available date
        try:
            test_stock = yf.Ticker(stock_ticker)
            test_data = test_stock.history(period="5d")
            if not test_data.empty:
                max_date = test_data.index.max().date()
                min_date = datetime.today().date() - timedelta(days=5*365)  # 5 years back
            else:
                max_date = datetime.today().date()
                min_date = datetime.today().date() - timedelta(days=5*365)
        except:
            max_date = datetime.today().date()
            min_date = datetime.today().date() - timedelta(days=5*365)
        
        start_date = st.sidebar.date_input(
            "Start Date", 
            value=min_date + timedelta(days=365),  # Default to 4 years back
            min_value=min_date, 
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date", 
            value=max_date,
            min_value=start_date, 
            max_value=max_date
        )
        
        # Model parameters
        st.sidebar.subheader("🤖 Model Parameters")
        prediction_days = st.sidebar.slider("Prediction Days", 30, 120, 60)
        lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 64)
        dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2)
        
        # API Setup
        st.sidebar.subheader("📡 API Configuration")
        twitter_api = setup_twitter_api()
        if twitter_api is None:
            st.sidebar.warning("⚠️ Twitter API not configured - using simulated sentiment")
        else:
            st.sidebar.success("✅ Twitter API connected")
        
        # Fetch and process data
        if st.sidebar.button("🚀 Fetch Live Data & Analyze", type="primary"):
            # Fetch stock data
            stock_data = fetch_stock_data(stock_ticker, start_date, end_date)
            
            if stock_data is not None:
                # Generate sentiment data
                with st.spinner("Analyzing market sentiment..."):
                    stock_data_with_sentiment = generate_sentiment_timeseries(stock_data, stock_ticker, twitter_api)
                
                # Display data overview
                st.subheader("📊 Live Data Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Records", len(stock_data_with_sentiment))
                with col2:
                    st.metric("📈 Current Price", f"₹{stock_data_with_sentiment['Close'].iloc[-1]:.2f}")
                with col3:
                    st.metric("💭 Avg Sentiment", f"{stock_data_with_sentiment['Sentiment'].mean():.3f}")
                with col4:
                    latest_sentiment = stock_data_with_sentiment['Sentiment'].iloc[-1]
                    sentiment_label = "😊 Positive" if latest_sentiment > 0.1 else "😐 Neutral" if latest_sentiment > -0.1 else "😟 Negative"
                    st.metric("🎭 Current Mood", sentiment_label)
                
                # Show recent data
                st.subheader("🔍 Recent Data Sample")
                display_data = stock_data_with_sentiment.tail(10).round(3)
                st.dataframe(display_data)
                
                # Model training/loading
                config = ModelConfig(stock_ticker, start_date, end_date, prediction_days, lstm_units, dropout_rate)
                
                model, close_scaler, sentiment_scaler, scaled_data = train_or_load_model(
                    stock_data_with_sentiment, config
                )
                
                if model is not None:
                    # Generate predictions
                    with st.spinner("Generating predictions..."):
                        predictions = generate_predictions(
                            model, scaled_data, close_scaler, config.prediction_days
                        )
                        
                        # Add predictions to data
                        full_predictions = np.full(len(stock_data_with_sentiment), np.nan)
                        full_predictions[config.prediction_days:] = predictions
                        stock_data_with_sentiment['Predicted'] = full_predictions
                    
                    # Calculate technical indicators
                    stock_data_with_sentiment = calculate_technical_indicators(stock_data_with_sentiment)
                    
                    # Generate trading signals
                    stock_data_with_sentiment['Signal'] = "Hold"
                    mask_buy = (
                        (stock_data_with_sentiment['Predicted'] > stock_data_with_sentiment['Close']) & 
                        (stock_data_with_sentiment['Sentiment'] > 0)
                    )
                    mask_sell = (
                        (stock_data_with_sentiment['Predicted'] < stock_data_with_sentiment['Close']) & 
                        (stock_data_with_sentiment['Sentiment'] < 0)
                    )
                    stock_data_with_sentiment.loc[mask_buy, 'Signal'] = "Buy"
                    stock_data_with_sentiment.loc[mask_sell, 'Signal'] = "Sell"
                    
                    # Results display
                    st.subheader("📈 Prediction Results")
                    
                    # Current prediction metrics
                    latest_actual = stock_data_with_sentiment['Close'].iloc[-1]
                    latest_predicted = stock_data_with_sentiment['Predicted'].iloc[-1]
                    
                    if not np.isnan(latest_predicted):
                        percentage_diff = ((latest_predicted - latest_actual) / latest_actual) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("💰 Current Price", f"₹{latest_actual:.2f}")
                        with col2:
                            delta_color = "normal" if abs(percentage_diff) < 2 else ("inverse" if percentage_diff < 0 else "normal")
                            st.metric("🔮 Predicted Price", f"₹{latest_predicted:.2f}", f"{percentage_diff:.1f}%")
                        with col3:
                            latest_rsi = stock_data_with_sentiment['RSI'].iloc[-1]
                            rsi_status = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Normal"
                            st.metric("📊 RSI", f"{latest_rsi:.1f}", rsi_status)
                        with col4:
                            signal_counts = stock_data_with_sentiment['Signal'].value_counts()
                            st.metric("🎯 Total Signals", f"{signal_counts.get('Buy', 0)} Buy / {signal_counts.get('Sell', 0)} Sell")
                    
                    # Main price chart
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    # Price lines
                    ax.plot(stock_data_with_sentiment.index, stock_data_with_sentiment['Close'], 
                           label="Actual Price", color="blue", linewidth=2)
                    ax.plot(stock_data_with_sentiment.index, stock_data_with_sentiment['Predicted'], 
                           label="Predicted Price", color="orange", linestyle="--", linewidth=2)
                    
                    # Bollinger Bands
                    ax.fill_between(stock_data_with_sentiment.index, 
                                   stock_data_with_sentiment['Lower_Band'], 
                                   stock_data_with_sentiment['Upper_Band'],
                                   alpha=0.1, color='gray', label='Bollinger Bands')
                    
                    # Trading signals
                    buy_signals = stock_data_with_sentiment[stock_data_with_sentiment['Signal'] == "Buy"]
                    sell_signals = stock_data_with_sentiment[stock_data_with_sentiment['Signal'] == "Sell"]
                    
                    if not buy_signals.empty:
                        ax.scatter(buy_signals.index, buy_signals['Close'], 
                                  color='green', marker='^', s=100, label='Buy Signal', zorder=5)
                    if not sell_signals.empty:
                        ax.scatter(sell_signals.index, sell_signals['Close'], 
                                  color='red', marker='v', s=100, label='Sell Signal', zorder=5)
                    
                    ax.set_title(f"📈 {stock_ticker} - Live Prediction Analysis", fontsize=16, fontweight='bold')
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Price (₹)")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Technical analysis
                    st.subheader("🔍 Technical Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # RSI Chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(stock_data_with_sentiment.index, stock_data_with_sentiment['RSI'], 
                               color='purple', linewidth=2)
                        ax.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
                        ax.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
                        ax.fill_between(stock_data_with_sentiment.index, 30, 70, alpha=0.1, color='yellow')
                        ax.set_title("RSI Analysis")
                        ax.set_ylabel("RSI")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    with col2:
                        # MACD Chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(stock_data_with_sentiment.index, stock_data_with_sentiment['MACD'], 
                               label='MACD', color='blue', linewidth=2)
                        ax.plot(stock_data_with_sentiment.index, stock_data_with_sentiment['Signal_Line'], 
                               label='Signal Line', color='orange', linestyle='--')
                        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
                        ax.set_title("MACD Analysis")
                        ax.set_ylabel("MACD")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Trading simulation
                    st.subheader("💼 Trading Simulation")
                    trade_log, total_profit, avg_profit, win_rate, final_capital = simulate_trades(stock_data_with_sentiment)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💰 Total Profit", f"₹{total_profit:,.0f}")
                    with col2:
                        st.metric("📊 Avg Profit/Trade", f"₹{avg_profit:,.0f}")
                    with col3:
                        st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
                    with col4:
                        roi = ((final_capital - 100000) / 100000) * 100
                        st.metric("📈 ROI", f"{roi:.1f}%")
                    
                    if not trade_log.empty:
                        st.subheader("📋 Trade History")
                        st.dataframe(trade_log.style.format({
                            'Buy Price': '₹{:.2f}',
                            'Sell Price': '₹{:.2f}',
                            'Profit': '₹{:.2f}'
                        }))
                    
                    # Future predictions
                    st.subheader("🔮 Future Predictions (Next 5 Trading Days)")
                    
                    # Generate future predictions
                    last_sequence = scaled_data[-config.prediction_days:]
                    future_predictions = []
                    future_dates = []
                    current_date = stock_data_with_sentiment.index[-1]
                    
                    for i in range(5):
                        current_date += timedelta(days=1)
                        while current_date.weekday() >= 5:  # Skip weekends
                            current_date += timedelta(days=1)
                        
                        pred = model.predict(last_sequence.reshape(1, config.prediction_days, 2), verbose=0)
                        
                        # Inverse transform
                        pred_full = np.zeros((1, 2))
                        pred_full[0, 0] = pred[0, 0]
                        pred_full[0, 1] = stock_data_with_sentiment['Sentiment'].iloc[-1]
                        
                        predicted_price = close_scaler.inverse_transform(pred_full)[0, 0]
                        future_predictions.append(predicted_price)
                        future_dates.append(current_date)
                        
                        # Update sequence
                        new_row = np.array([[pred[0, 0], stock_data_with_sentiment['Sentiment'].iloc[-1]]])
                        last_sequence = np.vstack([last_sequence[1:], new_row])
                    
                    # Display future predictions
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': [f"₹{p:.2f}" for p in future_predictions],
                        'Change from Current': [f"{((p-latest_actual)/latest_actual)*100:+.1f}%" for p in future_predictions]
                    })
                    st.dataframe(future_df, use_container_width=True)
    
    else:  # Demo mode
        st.subheader("📁 Demo Data (For Professor Demonstration)")
        st.info("This shows sample data files that demonstrate the application's capabilities")
        
        demo_data = load_demo_data()
        
        if demo_data:
            if 'stock_data' in demo_data:
                st.write("**Sample Stock Data (sss.csv):**")
                st.dataframe(demo_data['stock_data'].head(10))
                
            if 'sentiment_data' in demo_data:
                st.write("**Sample Sentiment Data (JSON files):**")
                st.dataframe(demo_data['sentiment_data'].head(10))
                
            st.success("✅ Demo files loaded successfully!")
            st.info("💡 Switch to 'Live API Data' mode for real-time analysis")
        else:
            st.warning("⚠️ Demo files not found. Please ensure sss.csv and JSON files are in the directory.")
    
    # Footer
    st.markdown("---")
    st.markdown("**📡 Data Sources:** yfinance (Stock Data) + Twitter API (Sentiment)")
    st.markdown("**🤖 Model:** LSTM with Attention + Technical Analysis")

if __name__ == "__main__":
    main()
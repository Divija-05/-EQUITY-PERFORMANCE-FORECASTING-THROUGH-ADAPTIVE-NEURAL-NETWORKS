import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pickle
import joblib
from datetime import datetime, timedelta
import hashlib
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
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Prediction & Sentiment Analysis",
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
    def __init__(self, prediction_days=60, lstm_units=64, dropout_rate=0.2):
        self.prediction_days = prediction_days
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
    
    def get_hash(self):
        """Generate hash for model configuration"""
        config_str = f"{self.prediction_days}_{self.lstm_units}_{self.dropout_rate}"
        return hashlib.md5(config_str.encode()).hexdigest()

@st.cache_data
def load_csv_data(file_path):
    """Load and cache CSV stock data"""
    try:
        if os.path.exists(file_path):
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            return data
        else:
            st.error(f"CSV file not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading CSV data: {str(e)}")
        return None

@st.cache_data
def load_json_sentiment_data(file_paths):
    """Load and process JSON sentiment data"""
    try:
        all_sentiment = {}
        for file_path in file_paths:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_sentiment.update(data)
        
        # Process sentiment data
        sentiment_scores = []
        dates = []
        
        for date_key, articles in all_sentiment.items():
            date_obj = pd.to_datetime(date_key)
            dates.append(date_obj)
            
            # Calculate average sentiment for the day
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
            
            # Average sentiment for the day, default to 0 if no data
            avg_sentiment = np.mean(daily_sentiments) if daily_sentiments else 0.0
            sentiment_scores.append(avg_sentiment)
        
        sentiment_df = pd.DataFrame({
            'Date': dates,
            'Sentiment': sentiment_scores
        })
        sentiment_df.set_index('Date', inplace=True)
        
        return sentiment_df
    except Exception as e:
        st.error(f"Error loading JSON sentiment data: {str(e)}")
        return None

def merge_stock_sentiment_data(stock_data, sentiment_data):
    """Merge stock and sentiment data"""
    try:
        # Align dates and merge
        merged_data = stock_data.copy()
        
        if sentiment_data is not None:
            # Merge on index (Date)
            merged_data = merged_data.join(sentiment_data, how='left')
            # Fill missing sentiment values with 0
            merged_data['Sentiment'].fillna(0, inplace=True)
        else:
            # Generate random sentiment if no data available
            merged_data['Sentiment'] = np.random.uniform(-0.1, 0.1, size=len(merged_data))
        
        return merged_data
    except Exception as e:
        st.error(f"Error merging data: {str(e)}")
        return stock_data

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
        
        st.success(f"Model artifacts saved to {model_dir}")
        
    except Exception as e:
        st.warning(f"Could not save model artifacts: {str(e)}")

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
        st.warning(f"Could not load model artifacts: {str(e)}")
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
        st.success("✅ Loaded existing trained model!")
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
        st.error("Insufficient data for training!")
        return None, None, None, None
    
    # Create and train model
    model = create_lstm_model((x_full.shape[1], x_full.shape[2]), config)
    
    reduce_lr = ReduceLROnPlateau(
        monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
    )
    
    with st.spinner("Training model..."):
        history = model.fit(
            x_full, y_full,
            epochs=50,
            batch_size=32,
            verbose=0,
            callbacks=[reduce_lr]
        )
    
    # Save model artifacts
    save_model_artifacts(model, close_scaler, sentiment_scaler, config, model_hash)
    
    return model, close_scaler, sentiment_scaler, scaled_data

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    # RSI
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['MA20'] = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['Upper Band'] = data['MA20'] + (rolling_std * 2)
    data['Lower Band'] = data['MA20'] - (rolling_std * 2)
    
    # MACD
    short_ema = data['Close'].ewm(span=12, adjust=False).mean()
    long_ema = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = short_ema - long_ema
    data['Signal Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    return data

def generate_predictions(model, scaled_data, close_scaler, sentiment_scaler, prediction_days):
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
        prev_signal = data['Signal'].iloc[i-1]
        
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
    st.title("📈 Advanced Stock Prediction & Sentiment Analysis")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Select Data Source",
        ["Local CSV Files", "Manual File Upload"]
    )
    
    # Model parameters
    prediction_days = st.sidebar.slider("Prediction Days", 30, 120, 60)
    lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 64)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.1, 0.5, 0.2)
    
    config = ModelConfig(prediction_days, lstm_units, dropout_rate)
    
    # Data loading section
    stock_data = None
    sentiment_data = None
    
    if data_source == "Local CSV Files":
        # Default file paths
        csv_file = "sss.csv"
        json_files = ["simulated_July_news_2025.json", "simulated_August_news_2025.json"]
        
        if st.sidebar.button("Load Local Data"):
            with st.spinner("Loading data..."):
                stock_data = load_csv_data(csv_file)
                sentiment_data = load_json_sentiment_data(json_files)
                
                if stock_data is not None:
                    st.success(f"✅ Stock data loaded: {len(stock_data)} records")
                if sentiment_data is not None:
                    st.success(f"✅ Sentiment data loaded: {len(sentiment_data)} records")
    
    else:  # Manual file upload
        uploaded_csv = st.sidebar.file_uploader("Upload Stock CSV", type=['csv'])
        uploaded_json = st.sidebar.file_uploader("Upload Sentiment JSON", type=['json'], accept_multiple_files=True)
        
        if uploaded_csv is not None:
            stock_data = pd.read_csv(uploaded_csv)
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data.set_index('Date', inplace=True)
            
        if uploaded_json:
            # Process uploaded JSON files
            all_sentiment = {}
            for json_file in uploaded_json:
                data = json.load(json_file)
                all_sentiment.update(data)
            
            # Convert to sentiment DataFrame (similar to load_json_sentiment_data)
            sentiment_scores = []
            dates = []
            
            for date_key, articles in all_sentiment.items():
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
            
            sentiment_data = pd.DataFrame({
                'Date': dates,
                'Sentiment': sentiment_scores
            })
            sentiment_data.set_index('Date', inplace=True)
    
    # Process data and train/load model
    if stock_data is not None:
        # Merge stock and sentiment data
        merged_data = merge_stock_sentiment_data(stock_data, sentiment_data)
        
        # Data overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Total Records", len(merged_data))
        with col2:
            st.metric("📅 Date Range", f"{merged_data.index.min().strftime('%Y-%m-%d')} to {merged_data.index.max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("💭 Avg Sentiment", f"{merged_data['Sentiment'].mean():.3f}")
        
        # Display data sample
        st.subheader("📋 Data Preview")
        st.dataframe(merged_data.tail(10))
        
        # Train/Load model
        if st.button("🚀 Start Analysis", type="primary"):
            model, close_scaler, sentiment_scaler, scaled_data = train_or_load_model(merged_data, config)
            
            if model is not None:
                # Generate predictions
                predictions = generate_predictions(
                    model, scaled_data, close_scaler, sentiment_scaler, config.prediction_days
                )
                
                # Add predictions to data
                full_predictions = np.full(len(merged_data), np.nan)
                full_predictions[config.prediction_days:] = predictions
                merged_data['Predicted'] = full_predictions
                
                # Calculate technical indicators
                merged_data = calculate_technical_indicators(merged_data)
                
                # Generate trading signals
                merged_data['Signal'] = "Hold"
                merged_data.loc[
                    (merged_data['Predicted'] > merged_data['Close']) & 
                    (merged_data['Sentiment'] > 0), 'Signal'
                ] = "Buy"
                merged_data.loc[
                    (merged_data['Predicted'] < merged_data['Close']) & 
                    (merged_data['Sentiment'] < 0), 'Signal'
                ] = "Sell"
                
                # Display results
                st.subheader("📈 Price Prediction Results")
                
                # Current metrics
                latest_actual = merged_data['Close'].iloc[-1]
                latest_predicted = merged_data['Predicted'].iloc[-1]
                if not np.isnan(latest_predicted):
                    percentage_diff = ((latest_predicted - latest_actual) / latest_actual) * 100
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💰 Current Price", f"₹{latest_actual:.2f}")
                    with col2:
                        st.metric("🔮 Predicted Price", f"₹{latest_predicted:.2f}")
                    with col3:
                        st.metric("📊 Difference", f"{percentage_diff:.2f}%")
                    with col4:
                        st.metric("😊 Sentiment", f"{merged_data['Sentiment'].iloc[-1]:.3f}")
                
                # Price chart
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(merged_data.index, merged_data['Close'], label="Actual Prices", color="blue", alpha=0.7)
                ax.plot(merged_data.index, merged_data['Predicted'], label="Predicted Prices", color="orange", linestyle="--", alpha=0.8)
                
                # Add buy/sell signals
                buy_signals = merged_data[merged_data['Signal'] == "Buy"]
                sell_signals = merged_data[merged_data['Signal'] == "Sell"]
                
                if not buy_signals.empty:
                    ax.scatter(buy_signals.index, buy_signals['Close'], 
                              label="Buy Signal", color="green", marker="^", s=50)
                if not sell_signals.empty:
                    ax.scatter(sell_signals.index, sell_signals['Close'], 
                              label="Sell Signal", color="red", marker="v", s=50)
                
                ax.set_title("Stock Price Prediction with Trading Signals")
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (₹)")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Technical indicators
                st.subheader("📊 Technical Analysis")
                
                # RSI Chart
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
                
                # Price with Bollinger Bands
                ax1.plot(merged_data.index, merged_data['Close'], label="Close Price", color="black")
                ax1.plot(merged_data.index, merged_data['MA20'], label="MA20", color="blue")
                ax1.fill_between(merged_data.index, merged_data['Lower Band'], merged_data['Upper Band'], 
                                alpha=0.2, color="gray", label="Bollinger Bands")
                ax1.set_title("Price with Bollinger Bands")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # RSI
                ax2.plot(merged_data.index, merged_data['RSI'], label="RSI", color="purple")
                ax2.axhline(70, color="red", linestyle="--", alpha=0.7, label="Overbought")
                ax2.axhline(30, color="green", linestyle="--", alpha=0.7, label="Oversold")
                ax2.set_title("RSI (Relative Strength Index)")
                ax2.set_ylabel("RSI")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Trading simulation
                st.subheader("💼 Trading Simulation")
                trade_log, total_profit, avg_profit, win_rate, final_capital = simulate_trades(merged_data)
                
                if not trade_log.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💰 Total Profit", f"₹{total_profit:,.2f}")
                    with col2:
                        st.metric("📈 Avg Profit/Trade", f"₹{avg_profit:,.2f}")
                    with col3:
                        st.metric("🎯 Win Rate", f"{win_rate:.1f}%")
                    with col4:
                        st.metric("💵 Final Capital", f"₹{final_capital:,.2f}")
                    
                    st.dataframe(trade_log)
                
                # Future predictions
                st.subheader("🔮 Future Price Predictions")
                
                # Predict next 5 trading days
                last_sequence = scaled_data[-config.prediction_days:]
                future_predictions = []
                future_dates = []
                current_date = merged_data.index[-1]
                
                for i in range(5):
                    current_date += timedelta(days=1)
                    while current_date.weekday() >= 5:  # Skip weekends
                        current_date += timedelta(days=1)
                    
                    pred = model.predict(last_sequence.reshape(1, config.prediction_days, 2), verbose=0)
                    
                    # Inverse transform
                    pred_full = np.zeros((1, 2))
                    pred_full[0, 0] = pred[0, 0]
                    pred_full[0, 1] = merged_data['Sentiment'].iloc[-1]
                    
                    predicted_price = close_scaler.inverse_transform(pred_full)[0, 0]
                    future_predictions.append(predicted_price)
                    future_dates.append(current_date)
                    
                    # Update sequence
                    new_row = np.array([[pred[0, 0], merged_data['Sentiment'].iloc[-1]]])
                    last_sequence = np.vstack([last_sequence[1:], new_row])
                
                # Display future predictions
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                st.dataframe(future_df.style.format({'Predicted Price': '₹{:.2f}'}))
    
    else:
        st.info("👆 Please load data using the sidebar to begin analysis.")

if __name__ == "__main__":
    main()
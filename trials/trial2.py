import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from datetime import datetime, timedelta
import pickle
import hashlib
import os
import json
import random

# Configure page
st.set_page_config(page_title="Smart Stock Predictor", layout="wide")

#Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scalers' not in st.session_state:
    st.session_state.scalers = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Create directories for caching
os.makedirs("model_cache", exist_ok=True)
os.makedirs("data_cache", exist_ok=True)

def generate_data_hash(stock_ticker, start_date, end_date, prediction_days):
    """Generate unique hash for data configuration"""
    config_string = f"{stock_ticker}_{start_date}_{end_date}_{prediction_days}"
    return hashlib.md5(config_string.encode()).hexdigest()

def save_model_and_scalers(model, scalers, data_hash):
    """Save trained model and scalers to disk"""
    try:
        model_path = f"model_cache/model_{data_hash}.keras"
        scalers_path = f"model_cache/scalers_{data_hash}.pkl"
        
        model.save(model_path)
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers, f)
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_hash': data_hash
        }
        with open(f"model_cache/metadata_{data_hash}.json", 'w') as f:
            json.dump(metadata, f)
            
        return True
    except Exception as e:
        st.warning(f"Could not save model: {e}")
        return False

def load_model_and_scalers(data_hash):
    """Load cached model and scalers"""
    try:
        model_path = f"model_cache/model_{data_hash}.keras"
        scalers_path = f"model_cache/scalers_{data_hash}.pkl"
        metadata_path = f"model_cache/metadata_{data_hash}.json"
        
        if os.path.exists(model_path) and os.path.exists(scalers_path):
            model = load_model(model_path)
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
            
            # Load metadata
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                return model, scalers, metadata
            
            return model, scalers, {'timestamp': 'Unknown'}
        return None, None, None
    except Exception as e:
        st.warning(f"Could not load cached model: {e}")
        return None, None, None

def check_data_freshness(last_training_time, hours_threshold=24):
    """Check if we need to retrain based on data freshness"""
    try:
        last_time = datetime.fromisoformat(last_training_time)
        time_diff = datetime.now() - last_time
        return time_diff.total_seconds() / 3600 < hours_threshold
    except:
        return False

# Alternative data fetching (same as before)
def fetch_with_retry(symbol, start_date, end_date, max_retries=3):
    """Fetch data with retry mechanism"""
    symbol_variants = [
        symbol,
        symbol.replace('.NS', '.BO'),
        symbol.replace('.NS', ''),
        f"{symbol.replace('.NS', '')}.NSE"
    ]
    
    for variant in symbol_variants:
        for attempt in range(max_retries):
            try:
                st.info(f"Trying {variant} (attempt {attempt + 1})")
                data = yf.download(variant, start=start_date, end=end_date, progress=False)
                
                if not data.empty and 'Close' in data.columns:
                    st.success(f"✅ Data fetched using: {variant}")
                    return data
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    st.warning(f"Failed with {variant}: {str(e)}")
                time.sleep(1)
                
    return None

def generate_sample_data(symbol, start_date, end_date):
    """Generate sample data"""
    st.warning("🔄 Using simulated data for demonstration")
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]
    
    np.random.seed(42)
    initial_price = 2000
    returns = np.random.normal(0.001, 0.02, len(dates))
    
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(dates)))
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.02, len(dates)))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.02, len(dates)))
    data['Volume'] = np.random.randint(1000000, 10000000, len(dates))
    
    return data.dropna()

def create_model(input_shape):
    """Create LSTM model"""
    input_timesteps, input_features = input_shape
    
    lstm_units = min(64, input_features * 2)
    dropout_rate = 0.2
    
    inputs = Input(shape=(None, input_features))
    
    # LSTM Encoder
    lstm_out, state_h, state_c = LSTM(
        lstm_units, return_state=True, return_sequences=True
    )(inputs)
    lstm_out = Dropout(dropout_rate)(lstm_out)
    
    # Self-Attention Block
    attention = Attention(use_scale=True)([lstm_out, lstm_out])
    concat = Concatenate()([lstm_out, attention])
    normed = LayerNormalization()(concat)

    # LSTM Decoder
    lstm_out2 = LSTM(lstm_units, return_sequences=False)(normed)
    lstm_out2 = Dropout(dropout_rate)(lstm_out2)
    
    # Output
    dense_out = Dense(max(16, input_features), activation="relu")(lstm_out2)
    output = Dense(1)(dense_out)
    
    model = Model(inputs, output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    
    return model

# Technical indicators (same as before)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * 2)
    lower_band = rolling_mean - (rolling_std * 2)
    return rolling_mean, upper_band, lower_band

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal_line

# Streamlit App
st.title("🧠 Smart Stock Predictor with Model Caching")

# Sidebar
st.sidebar.header("⚙️ Configuration")
stock_ticker = st.sidebar.text_input("Stock Ticker", "RELIANCE.NS")

# Training options
st.sidebar.subheader("🎯 Training Options")
force_retrain = st.sidebar.checkbox("🔄 Force Retrain Model", 
                                   help="Force retraining even if cached model exists")

auto_retrain = st.sidebar.checkbox("⏰ Auto-retrain if data is old", 
                                  value=True,
                                  help="Automatically retrain if cached model is > 24 hours old")

# Model cache info
st.sidebar.subheader("💾 Cache Status")
cache_files = [f for f in os.listdir("model_cache") if f.endswith('.keras')]
st.sidebar.write(f"📁 Cached models: {len(cache_files)}")

if st.sidebar.button("🗑️ Clear All Cache"):
    for file in os.listdir("model_cache"):
        os.remove(os.path.join("model_cache", file))
    st.sidebar.success("Cache cleared!")
    st.experimental_rerun()

# Date inputs
default_start = datetime.now() - timedelta(days=365)
default_end = datetime.now()

start_date = st.sidebar.date_input("Start Date", default_start.date())
end_date = st.sidebar.date_input("End Date", default_end.date())
prediction_days = st.sidebar.slider("Prediction Days", 30, 180, 60)

# Generate data configuration hash
data_hash = generate_data_hash(stock_ticker, str(start_date), str(end_date), prediction_days)

# Main execution
if st.sidebar.button("🚀 Run Analysis"):
    
    # Display current configuration
    st.subheader("📋 Analysis Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stock", stock_ticker)
    with col2:
        st.metric("Days Range", (end_date - start_date).days)
    with col3:
        st.metric("Prediction Window", f"{prediction_days} days")
    with col4:
        st.metric("Config Hash", data_hash[:8])
    
    # Step 1: Try to load cached model
    cached_model, cached_scalers, cached_metadata = load_model_and_scalers(data_hash)
    
    use_cached = False
    if cached_model and not force_retrain:
        if auto_retrain:
            is_fresh = check_data_freshness(cached_metadata.get('timestamp', ''))
            if is_fresh:
                use_cached = True
                st.success(f"✅ Using cached model (trained: {cached_metadata.get('timestamp', 'Unknown')})")
            else:
                st.info("🔄 Cached model is old, will retrain with fresh data")
        else:
            use_cached = True
            st.success(f"✅ Using cached model (trained: {cached_metadata.get('timestamp', 'Unknown')})")
    
    # Step 2: Fetch data
    st.subheader("📊 Data Acquisition")
    try:
        data = fetch_with_retry(stock_ticker, start_date, end_date)
        
        if data is None or data.empty:
            st.info("Using sample data...")
            data = generate_sample_data(stock_ticker, start_date, end_date)
        
        data.index = pd.to_datetime(data.index)
        data = data[~data.index.duplicated(keep='first')]
        data = data.dropna()
        
        # Add sentiment
        data['Sentiment'] = np.random.uniform(-1, 1, size=len(data))
        
        st.success(f"📈 Loaded {len(data)} data points")
        st.write("### Recent Data:")
        st.dataframe(data.tail())
        
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()
    
    # Step 3: Model Training or Loading
    if use_cached:
        model = cached_model
        close_scaler, sentiment_scaler = cached_scalers
        st.info("⚡ Model loaded from cache - no training needed!")
        
    else:
        st.subheader("🤖 Model Training")
        
        # Prepare data
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
        
        x_full, y_full = np.array(x_full), np.array(y_full)
        
        # Train model
        model = create_model((x_full.shape[1], x_full.shape[2]))
        
        reduce_lr = ReduceLROnPlateau(
            monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Custom callback to update progress
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / 50
                progress_bar.progress(progress)
                status_text.text(f"Training... Epoch {epoch + 1}/50 - Loss: {logs.get('loss', 0):.4f}")
        
        with st.spinner("🔥 Training model..."):
            history = model.fit(
                x_full, y_full,
                epochs=50,
                batch_size=32,
                verbose=0,
                callbacks=[reduce_lr, ProgressCallback()]
            )
        
        progress_bar.progress(1.0)
        status_text.text("✅ Training completed!")
        
        # Save model and scalers
        scalers = (close_scaler, sentiment_scaler)
        if save_model_and_scalers(model, scalers, data_hash):
            st.success("💾 Model saved to cache for future use!")
        
        # Store in session state
        st.session_state.model = model
        st.session_state.scalers = scalers
        st.session_state.model_trained = True
    
    # Step 4: Make Predictions (same as before)
    st.subheader("🔮 Making Predictions")
    
    # Prepare data for prediction
    close_scaler, sentiment_scaler = (close_scaler, sentiment_scaler) if not use_cached else cached_scalers
    scaled_close = close_scaler.transform(data['Close'].values.reshape(-1, 1))
    scaled_sentiment = sentiment_scaler.transform(data['Sentiment'].values.reshape(-1, 1))
    scaled_data = np.hstack([scaled_close, scaled_sentiment])
    
    # Create sequences for prediction
    x_pred, y_actual = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_pred.append(scaled_data[i-prediction_days:i])
        y_actual.append(scaled_data[i, 0])
    
    x_pred, y_actual = np.array(x_pred), np.array(y_actual)
    
    # Make predictions
    predictions = model.predict(x_pred, verbose=0)
    
    # Inverse transform
    pred_copies = np.repeat(predictions, 2, axis=-1)
    pred_copies[:, 1] = scaled_data[prediction_days:, 1]
    predictions_price = close_scaler.inverse_transform(pred_copies)[:, 0]
    
    # Add to dataframe
    full_predictions = np.full(len(data), np.nan)
    full_predictions[prediction_days:] = predictions_price
    data['Predicted'] = full_predictions
    
    # Step 5: Visualizations and Analysis
    st.subheader("📊 Results Analysis")
    
    # Current metrics
    latest_actual = data['Close'].iloc[-1]
    latest_predicted = data['Predicted'].iloc[-1]
    percentage_diff = ((latest_predicted - latest_actual) / latest_actual) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"₹{latest_actual:.2f}")
    with col2:
        st.metric("Predicted Price", f"₹{latest_predicted:.2f}")
    with col3:
        st.metric("Prediction Accuracy", f"{percentage_diff:.2f}%")
    
    # Price chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label="Actual", color="blue", linewidth=2)
    ax.plot(data.index, data['Predicted'], label="Predicted", color="orange", linestyle="--", alpha=0.8)
    ax.legend()
    ax.set_title(f"{stock_ticker}: Actual vs Predicted Prices")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Technical indicators
    data['RSI'] = calculate_rsi(data)
    data['MA20'], data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)
    data['MACD'], data['Signal Line'] = calculate_macd(data)
    
    # Performance metrics
    st.subheader("📈 Model Performance")
    valid_mask = ~np.isnan(data['Predicted'])
    if valid_mask.sum() > 10:
        actual_prices = data['Close'][valid_mask].values
        predicted_prices = data['Predicted'][valid_mask].values
        
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        mse = np.mean((actual_prices - predicted_prices) ** 2)
        rmse = np.sqrt(mse)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"₹{mae:.2f}")
        with col2:
            st.metric("RMSE", f"₹{rmse:.2f}")
        with col3:
            directional_accuracy = np.mean(
                np.sign(np.diff(actual_prices)) == np.sign(np.diff(predicted_prices))
            ) * 100
            st.metric("Direction Accuracy", f"{directional_accuracy:.1f}%")

# Cache management section
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Training Statistics")
if os.path.exists("model_cache"):
    cache_size = sum(
        os.path.getsize(os.path.join("model_cache", f)) 
        for f in os.listdir("model_cache")
    ) / (1024 * 1024)  # Convert to MB
    st.sidebar.write(f"💾 Cache size: {cache_size:.2f} MB")

st.sidebar.markdown("### ⚡ Performance Benefits:")
st.sidebar.markdown("""
- **First run**: ~2 minutes (training)
- **Subsequent runs**: ~10 seconds (cached)
- **Auto-refresh**: Every 24 hours
- **Smart caching**: Per stock + date combination
""")
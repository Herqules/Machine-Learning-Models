import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def fetch_stock_data(symbol, api_key):
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2020-01-01/2023-12-31?adjusted=true&sort=asc&limit=120&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            df = pd.DataFrame(data['results'], columns=['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n'])
            df.rename(columns={'v': 'volume', 'vw': 'volume_weighted', 'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 't': 'timestamp'}, inplace=True)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['adjusted_close'] = df['close']  # Assuming 'close' is the adjusted close price
            return df
        else:
            print("Error: No data returned for the stock.")
    else:
        print(f"Error fetching data: HTTP Status Code {response.status_code}")
    return pd.DataFrame()

def calculate_returns(data):
    data['1_day_return'] = data['adjusted_close'].pct_change(1)  # Calculating daily returns
    data.fillna(0, inplace=True)  # Handling NaN values
    return data

# Your API key
api_key = 'bbh72g9fhYUwsiEVXd8DnQeI8Yehl3XO'

stock_symbol = input("Please enter the stock symbol you are interested in: ")
stock_data = fetch_stock_data(stock_symbol, api_key)

if stock_data.empty:
    print(f"No data fetched for symbol: {stock_symbol}")
else:
    stock_data_with_returns = calculate_returns(stock_data)
    
    features = stock_data_with_returns[['volume', '1_day_return']]
    X = features.values[:-1]  # Exclude the last day for features
    y = features['1_day_return'].values[1:]  # Predicting the next day's return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')  # Output layer for predicting return
    ])

    model.compile(optimizer=Adam(), loss='mean_squared_error')
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

    next_day_features = np.array([features.values[-1]])
    next_day_features_scaled = scaler.transform(next_day_features)
    predicted_return = model.predict(next_day_features_scaled)[0][0]

    if predicted_return > 0:
        print("Recommendation: Buy")
    else:
        print("Recommendation: Don't Buy")

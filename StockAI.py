import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import requests

# Prompt the user for a stock symbol
stock_symbol = input("Please enter the stock symbol you are interested in: ")
api_key = 'bbh72g9fhYUwsiEVXd8DnQeI8Yehl3XO'

def fetch_stock_data(symbol):
    url = f'https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2020-01-01/2023-12-31?adjusted=true&sort=asc&limit=120&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    # Convert the Polygon.io data format to a DataFrame
    df = pd.DataFrame(data['results'], columns=['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n'])
    df.rename(columns={'v': 'volume', 'o': 'open', 'c': 'close', 'h': 'high', 'l': 'low', 't': 'timestamp', 'vw': 'volume_weighted'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df['adjusted_close'] = df['close']  # Adjusted close
    return df

def calculate_returns(data):
    # Calculate returns over different periods using the correct column name
    data['1_month_return'] = data['adjusted_close'].pct_change(periods=21)  # Approx. 1 month
    data['3_month_return'] = data['adjusted_close'].pct_change(periods=63)  # Approx. 3 months
    data['ytd_return'] = data['adjusted_close'].pct_change(periods=252)  # Approx. 1 year
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data



# Fetch the stock data
print(f"Fetching stock data for {stock_symbol}...")
stock_data = fetch_stock_data(stock_symbol)
print(stock_data.head())  # Check the first few rows


# Calculate returns and check if 1-month, 3-month, and YTD returns are all positive
stock_data_with_returns = calculate_returns(stock_data)
positive_trend = stock_data_with_returns[['1_month_return', '3_month_return', 'ytd_return']].tail(1) > 0

if positive_trend.all(axis=1).iloc[0]:
    print("All 1-month, 3-month, and year-to-date returns are positive.")
else:
    print("Not all 1-month, 3-month, and year-to-date returns are positive.")

# Select features for modeling (adjusted close, volume, and calculated returns)
features = stock_data_with_returns[['adjusted_close', 'volume', '1_month_return', '3_month_return', 'ytd_return']]
features = features.dropna(subset=['adjusted_close', 'volume'])  # Only drop rows where 'adjusted_close' or 'volume' are NaN
X = features.values[:-1]
y = features['adjusted_close'].values[1:]
# Check the shape or size of X and y
print(f"Features shape: {X.shape}, Target shape: {y.shape}")


if X.size == 0 or y.size == 0:
    raise ValueError("The feature matrix X or target array y is empty. Cannot proceed with model training.")

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the model
model = Sequential([
    Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer=Adam(), loss='mean_squared_error')
print("Training model...")
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
print("Evaluating model...")
test_loss = model.evaluate(X_test_scaled, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")


import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os



# Load model
model = load_model(".\model\stock_prediction.h5")


# Web App Header and Sidebar
st.title('Advanced Stock Price Prediction App 📈')
st.sidebar.header("Customize Settings")

# Stock Ticker Input
stock = st.sidebar.text_input('Enter Stock Ticker (e.g., GOOG, AAPL, TSLA)', 'GOOG')

# Date Range Selectors (Defaults to the stock data's available range)
start = st.sidebar.date_input('Start Date', pd.to_datetime('2010-01-01'))
end = st.sidebar.date_input('End Date', pd.to_datetime('today').date())

# Fetch Stock Data
data = yf.download(stock, start=start, end=end)
st.subheader(f'Raw Data for {stock}')
st.write(data)

# Date range info and available range display
st.sidebar.subheader("Available Data Date Range")
st.sidebar.write(f"From: {data.index[0].date()} To: {data.index[-1].date()}")

# Training and Testing Data Split
train_ratio = st.sidebar.slider("Training Data Ratio", 0.5, 0.9, 0.8)
data_train = pd.DataFrame(data['Close'][0:int(len(data) * train_ratio)])
data_test = pd.DataFrame(data['Close'][int(len(data) * train_ratio):])

# Display descriptive statistics for analysis
st.subheader("Descriptive Statistics")
st.write(data.describe())

# Moving Averages Settings
st.sidebar.subheader("Plot Moving Averages")
show_ma50 = st.sidebar.checkbox("Show 50-Day MA", True)
show_ma100 = st.sidebar.checkbox("Show 100-Day MA", True)
show_ma200 = st.sidebar.checkbox("Show 200-Day MA", True)

# Plot Moving Averages
st.subheader("Closing Price with Moving Averages")
fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(data['Close'], label='Close Price', color='blue')
if show_ma50:
    ax.plot(data['Close'].rolling(50).mean(), 'r', label='50-Day MA')
if show_ma100:
    ax.plot(data['Close'].rolling(100).mean(), 'g', label='100-Day MA')
if show_ma200:
    ax.plot(data['Close'].rolling(200).mean(), 'orange', label='200-Day MA')
plt.legend()
st.pyplot(fig)

# Scaling and Data Preparation for Predictions
scaler = MinMaxScaler(feature_range=(0, 1))
data_test = pd.concat([data_train.tail(100), data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

X_test = []
y_test = []
for i in range(100, data_test_scaled.shape[0]):
    X_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Predictions and Rescaling
predicted_prices = model.predict(X_test)
scale_factor = 1 / scaler.scale_[0]
predicted_prices *= scale_factor
y_test *= scale_factor

# Interactive Comparison Plot
st.subheader(f'Original vs. Predicted Prices for {stock}')
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.plot(y_test, color='green', label='Original Price')
ax2.plot(predicted_prices, color='red', linestyle='--', label='Predicted Price')
ax2.set_xlabel('Days')
ax2.set_ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Dataframe for Comparison with Extra Metrics
comparison_df = pd.DataFrame({
    'Date': data.index[-len(y_test):],
    'Original Price': y_test.flatten(),
    'Predicted Price': predicted_prices.flatten(),
})
comparison_df['Absolute Error'] = abs(comparison_df['Original Price'] - comparison_df['Predicted Price'])
comparison_df['Percentage Error'] = (comparison_df['Absolute Error'] / comparison_df['Original Price']) * 100
st.write("Comparison Table with Errors")
st.write(comparison_df)










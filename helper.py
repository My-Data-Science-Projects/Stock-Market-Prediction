import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import datetime as dt
import plotly.graph_objects as go
from pymongo import MongoClient
import bcrypt
from bson.objectid import ObjectId
import datetime as dt
import re

# MongoDB Configuration
client = MongoClient("mongodb://localhost:27017/")
db = client['stockData']
users_collection = db['users']
hist_collection = db['historical_data']
future_collection = db['future_data']
    
# Utility functions
def create_user(name, email, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    user = {"name": name, "email": email, "password": hashed_password}

    # Insert the user document into the collection and capture the _id
    result = users_collection.insert_one(user)
    
    # Update the user document to include user_id field with the value of _id
    user_id_str = str(result.inserted_id)
    users_collection.update_one({"_id": result.inserted_id}, {"$set": {"user_id": user_id_str}})

def authenticate_user(email, password):
    user = users_collection.find_one({"email": email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return user
    return None

def extract_percentage_value(html_string):
    # Regular expression to find the percentage value including the sign
    match = re.search(r'(\+?\-?\d+\.\d+%)', html_string)
    if match:
        # Return the matched percentage value as a string
        return match.group(1)
    return None

def add_historical_data(ticker, start_date, end_date, start_end_variance, user_id, username):
    current_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    hist_data = {
        "ticker": ticker,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "start_end_variance": extract_percentage_value(start_end_variance),
        "timestamp": current_timestamp,
        "user_id": user_id,
        "username": username
    }

    # Insert the user document into the collection and capture the _id
    result = hist_collection.insert_one(hist_data)
    hist_id_str = str(result.inserted_id)
    # Update the user document to include user_id field with the value of _id
    hist_collection.update_one({"_id": result.inserted_id}, {"$set": {"hist_id": hist_id_str}})

def add_future_data(ticker, no_of_days, start_date, variance, user_id, username):
    current_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    start_date_str = start_date.strftime('%Y-%m-%d')

    future_data = {
        "ticker": ticker,
        "no_of_days": no_of_days,
        "start_date": start_date_str,
        "variance": extract_percentage_value(variance),
        "timestamp": current_timestamp,
        "user_id": user_id,
        "username": username
    }

    # Insert the user document into the collection and capture the _id
    result = future_collection.insert_one(future_data)
    future_id_str = str(result.inserted_id)
    # Update the user document to include user_id field with the value of _id
    future_collection.update_one({"_id": result.inserted_id}, {"$set": {"future_id": future_id_str}})

# Function to fetch stock data from Yahoo Finance for a specified date range
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to fetch stock data from Yahoo Finance for the last 90 days
def fetch_stock_data_last_90_days(ticker):
    end_date = dt.datetime.today().date()
    start_date = end_date - dt.timedelta(days=90)
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to prepare data for LSTM model
def prepare_data_for_lstm(data, time_step):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close_prices = scaler.fit_transform(close_prices)
    
    X, Y = [], []
    for i in range(len(scaled_close_prices) - time_step - 1):
        X.append(scaled_close_prices[i:(i + time_step), 0])
        Y.append(scaled_close_prices[i + time_step, 0])
    
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y, scaler

# Function to define the LSTM model
def define_model(time_step):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the model
def train_model(model, X, Y, epochs=100, batch_size=32):
    model.fit(X, Y, epochs=epochs, batch_size=batch_size)
    return model

# Function to predict the next n days
def predict_next_days(model, data, scaler, time_step, n_days):
    future_dates = [data.index[-1] + dt.timedelta(days=i) for i in range(1, n_days + 1)]
    last_30_days = data['Close'].values[-time_step:].reshape(-1, 1)
    input_data = scaler.transform(last_30_days).reshape(1, time_step, 1)

    predicted_prices_scaled = []
    for i in range(n_days):
        prediction = model.predict(input_data)
        predicted_prices_scaled.append(prediction[0, 0])
        input_data = np.roll(input_data, -1)
        input_data[0, -1, 0] = prediction[0, 0]

    predicted_prices = scaler.inverse_transform(np.array(predicted_prices_scaled).reshape(-1, 1))
    return future_dates, predicted_prices


# Function to create candlestick chart with slider and hover data
def plot_candlestick_chart(stock_data):
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                         open=stock_data['Open'],
                                         high=stock_data['High'],
                                         low=stock_data['Low'],
                                         close=stock_data['Close'])])

    fig.update_layout(
        title='Candlestick Chart for Historical Data',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,  # Show the range slider
        hovermode='x unified'
    )

    # fig.show()
    return fig

# Function to show the candlestick chart with predicted values
def show_chart(stock_90_data, predicted_dates, predicted_prices):
    combined_dates = np.concatenate((stock_90_data.index, predicted_dates))
    combined_prices = np.concatenate((stock_90_data['Close'].values, predicted_prices.flatten()))

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Candlestick(x=stock_90_data.index,
                                 open=stock_90_data['Open'],
                                 high=stock_90_data['High'],
                                 low=stock_90_data['Low'],
                                 close=stock_90_data['Close'],
                                 name='Historical Data'))

    # Predicted data
    fig.add_trace(go.Scatter(x=predicted_dates,
                             y=predicted_prices.flatten(),
                             mode='lines',
                             name='Predicted Prices'))

    # Update layout
    fig.update_layout(
        title='Candlestick Chart with Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,  # Show the range slider
        hovermode='x unified'
    )

    # fig.show()
    return fig
        
# Function to calculate percentage gain/loss
# def calculate_percentage_change(start_close, end_close):
#     percentage_change = ((end_close - start_close) / start_close) * 100
#     sign = "+" if percentage_change > 0 else "-"
#     return f"{sign}{abs(percentage_change):.2f}%"

def calculate_percentage_change(start_close, end_close):
    percentage_change = ((end_close - start_close) / start_close) * 100
    sign = "+" if percentage_change > 0 else "-"
    percentage_formatted = f"{sign}{abs(percentage_change):.2f}%"

    # Determine color based on sign
    color = "green" if sign == "+" else "red"
    
    # Example font sizes
    font_size_positive = "24px"
    font_size_negative = "24px"
    
    # Determine font size based on sign
    font_size = font_size_positive if sign == "+" else font_size_negative
    
    # Create the message with inline style for color and font size
    styled_message = f'<span style="color:{color}; font-size:{font_size};">{percentage_formatted}</span>'
    
    return styled_message
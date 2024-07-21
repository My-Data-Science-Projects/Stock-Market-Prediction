import datetime as dt
import pandas as pd
import streamlit as st
from streamlit_cookies_manager import CookieManager
import nifty50
import helper

# Initialize Cookies Manager
cookies = CookieManager()

def main():
    # State management
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ""

    # Check cookies for persistent login
    if not st.session_state.logged_in:
        if cookies["logged_in"] == 'True' and cookies["username"]:
            st.session_state.logged_in = True
            st.session_state.username = cookies["username"]
            st.session_state.user_id = cookies["user_id"]

    # Pages
    def sign_up():
        st.title("Sign Up")
        name = st.text_input("Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Create"):
            if not name or not email or not password:
                st.error("All fields are required!")
            else:
                helper.create_user(name, email, password)
                st.success("Account created successfully! Please log in.")
                st.session_state.logged_in = False

    def login():
        st.title("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = helper.authenticate_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = user['name']
                st.session_state.user_id = user['user_id']
                # Set cookies
                cookies["logged_in"] = 'True'
                cookies["username"] = user['name']
                cookies["user_id"] = user['user_id']
                st.experimental_rerun()
            else:
                st.error("Invalid email or password")

    def home():
        st.sidebar.title(f"Welcome, {st.session_state.username}")

        st.subheader(f"Historical & Future Data of Nifty50")

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.user_id = ""
            # Clear cookies
            cookies["logged_in"] = 'False'
            del cookies["username"]
            del cookies["user_id"]
            st.experimental_rerun()

        st.image("nifty50.png", width=400)

        # Fetch all tickers and company names from NSE
        ticker_data = nifty50.nse_dict

        ticker_options = [f"{ticker} - {name}" for ticker, name in ticker_data.items()]
        selected_ticker = st.sidebar.selectbox("Select Ticker", options=ticker_options)

        start_date = st.sidebar.date_input("Select a start date", dt.date.today())
        end_date = st.sidebar.date_input("Select a end date", dt.date.today(), max_value=dt.date.today())
        n_days = st.sidebar.selectbox("Select number of days to be predicted", list(range(1, 16)), index=0)

        submit_button = st.sidebar.button("Submit")
        
        if submit_button:
            ticker = selected_ticker.split(' - ')[0]

            end_ext_date = pd.to_datetime(end_date)
            end_ext_date = end_ext_date + pd.Timedelta(days=1)
            end_ext_date = end_ext_date.strftime('%Y-%m-%d')

            stock_data = helper.fetch_stock_data(ticker, start_date, end_ext_date)

            if not stock_data.empty:
                stock_startdt_close = stock_data.iloc[0]['Close']
                stock_enddt_close = stock_data.iloc[-1]['Close']

                st.plotly_chart(helper.plot_candlestick_chart(stock_data))

                hist_var = helper.calculate_percentage_change(stock_startdt_close, stock_enddt_close)
                st.write("#### Variance of historical data percentage between start and end date : ", hist_var, unsafe_allow_html=True)

                stock_90_data = helper.fetch_stock_data_last_90_days(ticker)

                X, Y, scaler = helper.prepare_data_for_lstm(stock_90_data, time_step=30)

                model = helper.define_model(time_step=30)
                model = helper.train_model(model, X, Y, epochs=100, batch_size=32)

                predicted_dates, predicted_prices = helper.predict_next_days(model, stock_90_data, scaler, time_step=30, n_days=n_days)

                predicted_prices_close = predicted_prices[-1][0]

                future_dates = [stock_90_data.index[-1] + dt.timedelta(days=i) for i in range(1, n_days + 1)]

                for date, price in zip(future_dates, predicted_prices):
                    print(f"{date.date()}: Predicted Close Price = {price[0]:.2f}")

                st.plotly_chart(helper.show_chart(stock_90_data, predicted_dates, predicted_prices))

                pred_var = helper.calculate_percentage_change(stock_startdt_close, predicted_prices_close)
                st.write("#### Variance of data percentage between historical start and predicted end date : ", pred_var, unsafe_allow_html=True)

                # Add data to MongoDB
                helper.add_historical_data(ticker, start_date, end_date, hist_var, st.session_state.user_id, st.session_state.username)
                helper.add_future_data(ticker, n_days, start_date, pred_var, st.session_state.user_id, st.session_state.username)
            else:
                st.error("No data available for the selected ticker")

    # Page routing
    if st.session_state.logged_in:
        home()
    else:
        page = st.sidebar.selectbox("Choose a page", ["Login", "Sign Up"])
        if page == "Login":
            login()
        elif page == "Sign Up":
            sign_up()

if __name__ == "__main__":
    # Ensure cookies manager works correctly
    if cookies.ready():
        main()
    else:
        st.stop()

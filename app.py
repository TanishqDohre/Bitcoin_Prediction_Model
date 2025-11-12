# import numpy as np
# import pandas as pd
# import yfinance as yf
# from keras.models import load_model
# from sklearn.preprocessing import MinMaxScaler
# import streamlit as st
# #load Model 
# model = load_model('C:/Users/TANISHQ/OneDrive/Desktop/BitCoin/Bitcoin_price_Prediction_Model.keras')


# st.header('Bitcoin Price Prediction Model')
# st.subheader('Bitcoin Price Data')
# data = pd.DataFrame(yf.download('BTC-USD','2015-01-01','2023-11-30'))
# data = data.reset_index()
# st.write(data)

# # st.subheader('Bitcoin Line Chart')
# # data.drop(columns = ['Date','Close','High','Low','Open','Volume'], inplace=True)
# # st.line_chart(data)

# st.header('Bitcoin Line Chart')
# data['Date'] = pd.to_datetime(data['Date'])
# data.set_index('Date', inplace=True)
# st.line_chart(data['Close'])

# train_data = data[:-100]
# test_data = data[-200:]

# scaler = MinMaxScaler(feature_range=(0,1))
# train_data_scale = scaler.fit_transform(train_data)
# test_data_scale = scaler.transform(test_data)
# base_days = 100
# x = []
# y = []
# for i in range(base_days, test_data_scale.shape[0]):
#     x.append(test_data_scale[i-base_days:i])
#     y.append(test_data_scale[i,0])

# x, y = np.array(x), np.array(y)
# x = np.reshape(x, (x.shape[0],x.shape[1],1))

# st.subheader('Predicted vs Original Prices ')
# pred = model.predict(x)
# pred = scaler.inverse_transform(pred)
# preds = pred.reshape(-1,1)
# ys = scaler.inverse_transform(y.reshape(-1,1))
# preds = pd.DataFrame(preds, columns=['Predicted Price'])
# ys = pd.DataFrame(ys, columns=['Original Price'])
# chart_data = pd.concat((preds, ys), axis=1)
# st.write(chart_data)
# st.subheader('Predicted vs Original Prices Chart ')
# st.line_chart(chart_data)

# m = y
# z= []
# future_days = 5
# for i in range(base_days, len(m)+future_days):
#     m = m.reshape(-1,1)
#     inter = [m[-base_days:,0]]
#     inter = np.array(inter)
#     inter = np.reshape(inter, (inter.shape[0], inter.shape[1],1))
#     pred = model.predict(inter)
#     m = np.append(m ,pred)
#     z = np.append(z, pred)
# st.subheader('Predicted Future Days Bitcoin Price')
# z = np.array(z)
# z = scaler.inverse_transform(z.reshape(-1,1))
# st.line_chart(z)







# //------------------------------------------------------



# app.py
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense

# ------- USER SETTINGS -------
TICKER = "BTC-USD"
START = "2015-01-01"
END = "2023-11-30"
MODEL_PATH = r"C:/Users/TANISHQ/OneDrive/Desktop/BitCoin/Bitcoin_price_Prediction_Model.keras"
BASE_DAYS = 100      # lookback
FUTURE_DAYS = 5
# -----------------------------

st.header("Bitcoin Price Prediction Model")

# 1) Download data and show
data = yf.download(TICKER, START, END)
if data.empty:
    st.error("No data returned for ticker.")
else:
    data = data.reset_index()
    st.subheader("Raw data (first 5 rows)")
    st.write(data.head())

# 2) Chart the Close price
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
if 'Close' not in data.columns:
    st.error(f"'Close' column not found. Columns: {list(data.columns)}")
else:
    st.subheader("Bitcoin Close Price")
    st.line_chart(data['Close'])

# 3) Prepare series (only Close)
close_values = data[['Close']].values.astype(float)  # shape (N,1)

# 4) Train/test split (use last 200 points as test for example)
test_window = 200
if len(close_values) < (BASE_DAYS + 10):
    st.error("Not enough data for the chosen BASE_DAYS.")
else:
    train = close_values[:-test_window]
    test = close_values[-test_window:]

    # 5) Scale - fit only on train
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)         # shape (n_train,1)
    test_scaled = scaler.transform(test)               # shape (test_window,1)

    # 6) helper to create sequences
    def create_sequences(series_scaled, base_days):
        x, y = [], []
        for i in range(base_days, len(series_scaled)):
            x.append(series_scaled[i-base_days:i, 0])  # last base_days values
            y.append(series_scaled[i, 0])             # next value
        x = np.array(x)                                 # shape (samples, base_days)
        y = np.array(y)                                 # shape (samples,)
        # expand last dim -> features=1
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))  # (samples, timesteps, features)
        return x, y

    # Create sequences from test (you might want to create from combined series for walk-forward)
    x_test, y_test = create_sequences(test_scaled, BASE_DAYS)
    st.write(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    # 7) Load model (ensure path and extension are correct)
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Could not load model: {e}")
        model = None

    # 8) Predict if model loaded
    if model is not None and x_test.shape[0] > 0:
        preds_scaled = model.predict(x_test)               # shape (samples, 1) expected
        # If model outputs (samples, ) reshape:
        preds_scaled = np.reshape(preds_scaled, (-1, 1))

        preds = scaler.inverse_transform(preds_scaled)     # back to price scale
        ys = scaler.inverse_transform(y_test.reshape(-1, 1))

        chart_df = pd.DataFrame({
            "Predicted Price": preds.flatten(),
            "Original Price": ys.flatten()
        })
        st.subheader("Predicted vs Original (table)")
        st.write(chart_df.head(20))

        st.subheader("Predicted vs Original (chart)")
        st.line_chart(chart_df)

        # 9) Future prediction (iterative)
        # start with the last available scaled values (from combined last test sequence)
        last_sequence = test_scaled[-BASE_DAYS:, 0].tolist()  # list length BASE_DAYS
        future_preds = []
        seq = last_sequence.copy()
        for _ in range(FUTURE_DAYS):
            arr = np.array(seq[-BASE_DAYS:])      # ensure length = BASE_DAYS
            arr = arr.reshape(1, BASE_DAYS, 1)
            p_scaled = model.predict(arr)        # shape (1,1)
            p_val = p_scaled.flatten()[0]
            future_preds.append(p_val)
            seq.append(p_val)                    # append scaled prediction for next step

        future_preds = np.array(future_preds).reshape(-1, 1)
        future_preds_price = scaler.inverse_transform(future_preds)

        st.subheader(f"Predicted next {FUTURE_DAYS} days (prices)")
        st.write(pd.DataFrame(future_preds_price, columns=["Predicted Price"]))
        st.line_chart(future_preds_price)

    else:
        st.info("No predictions because model not loaded or test sequences empty.")



        


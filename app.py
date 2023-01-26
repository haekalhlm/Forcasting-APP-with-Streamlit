import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

st.title("ARIMA Forecasting App")

# Load the data
data = pd.read_csv("data.csv")

# Show data
st.write(data)

# Preprocessing function
def preprocessing(df): 
    df = df[(df['store'] == store) & (df['item'] == item)]
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d') 
    temp_df = df.set_index('date')
    train_df = temp_df.loc[:'2017-08-30'].reset_index(drop=False)
    test_df = temp_df.loc['2017-09-01':].reset_index(drop=False)
    return train_df, test_df

# Load the model
with open("sarima_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit components

store_options = data['store'].unique()
store = st.selectbox("Select store:", store_options)
item_options = data['item'].unique()
item = st.selectbox("Select item:", item_options)

# Run ARIMA model and display forecast
if st.button("Forecast"):
    store_data = data[(data['store'] == store) & (data['item'] == item)]
    train_df, test_df = preprocessing(store_data)
    st.line_chart(train_df, x='date', y='sales')
    forecast = model.forecast(steps=122)
    st.write(forecast)
    forecast = pd.Series(forecast)
    # convert the index of test_data to time-series index
    test_data = test_df[store_data.columns[-1]]
    test_data.index = pd.date_range(start='2017-09-01', periods=len(test_data), freq='D')
    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(test_data, forecast))
    mae = mean_absolute_error(test_data, forecast)
    mape = mean_absolute_percentage_error(test_data, forecast)
    # Create a plot of the forecast and actual values
    plt.figure(figsize=(14,7))
    plt.plot(test_data.index, test_data, label='actual sales')
    plt.plot(forecast.index, forecast, label='forecast')
    plt.legend(loc='best')
    plt.xlabel('date')
    plt.ylabel('sales')
    plt.title('Seasonal ARIMA (SARIMA) forecasts with actual sales')
    st.pyplot(plt.gcf())
    # Display evaluation metrics
    st.write("RMSE: ", rmse)
    st.write("MAE: ", mae)
    st.write("MAPE: ", mape)
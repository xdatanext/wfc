import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yfinance as yf # https://pypi.org/project/yfinance/
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

import datetime
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go

##############
# Stock data #
##############

# Download data
@st.cache_data
def load_data(option, start_date, end_date):
    data = yf.download(option,start= start_date,end= end_date, progress=False)
    data.reset_index(inplace=True)
    return data

def bollinger_bands(df):
    # Bollinger Bands
    indicator_bb = BollingerBands(df['Close'])
    bb = df
    bb['bb_h'] = indicator_bb.bollinger_hband()
    bb['bb_l'] = indicator_bb.bollinger_lband()
    bb = bb[['Close','bb_h','bb_l']]
    return bb

def macd(df):
    # Moving Average Convergence Divergence
    macd = MACD(df['Close']).macd()
    return macd

def rsi(df):
    # Resistence Strength Indicator
    rsi = RSIIndicator(df['Close']).rsi()
    return rsi

###########
# sidebar #
###########
# globals from select

def sidebar():
    option = st.sidebar.selectbox('Select one symbol', \
                    ( 'AAPL', 'MSFT','NVDA', 'TSLA', 'GOOG', 'AMZN'))
    st.session_state['ticker'] = option
    today = datetime.date.today()
    before = today - datetime.timedelta(days=100)
    start_date = st.sidebar.date_input('Start date', before)
    end_date = st.sidebar.date_input('End date', today)
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    else:
        st.sidebar.error('Error: End date must fall after start date.')
    
    data_load_state = st.sidebar.text('Loading data...')
    data = load_data(option, start_date, end_date)
    #data_load_state.text('Loading data... done!')
    

    n_months = st.sidebar.slider('Months of prediction:', 1, 6)
    st.session_state['period'] = n_months * 30
    future = today + datetime.timedelta(days=st.session_state['period'])
    predict_date =  st.sidebar.date_input('Predict date', future)
    if predict_date > end_date:
        st.sidebar.success('End date: `%s`\n\nPredict date:`%s`' % (end_date, predict_date))
    else:
        st.sidebar.error('Error: Predict date must fall after end date.')
    return data 

def raw_data(data):
    st.subheader('Raw data')
    st.write(data.tail())

def recent_data(days, data):
    # Data of recent days
    st.write('Recent data ')
    st.dataframe(data.tail(days))

# Plot raw data
def plot_raw_data(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="stock_close"))
    fig.layout.update(title_text='', xaxis_rangeslider_visible=True)
    #fig.layout.update(title_text='Time Series with RangeSlider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def forecast(data):
    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=st.session_state['period'])
    forecast = m.predict(future)
    # Show and plot forecast
    st.subheader('Forecast data')
    #st.write(forecast.tail())
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    # Forecast
    st.write(f'Forecast plot for {st.session_state["period"]} days')
    fig1 = plot_plotly(m,forecast)
    st.plotly_chart(fig1)
    fig2 = plot_components_plotly(m, forecast)
    st.plotly_chart(fig2)

def show_bb(data):
    # Plot the prices and the bolinger bands
    ubb = bollinger_bands(data)
    st.write('Stock Bollinger Bands (BB)')
    st.line_chart(ubb)

def show_rsi(data):
    # Plot RSI
    ursi = rsi(data)
    st.write('Stock Resistence Strength Indicator (RSI)')
    st.line_chart(ursi)

def show_macd(data):
    UMACD = macd(data)
    # Plot MACD
    st.write('Stock Moving Average Convergence Divergence (MACD)')
    st.area_chart(UMACD)

def main():
    data = sidebar()
    progress_bar = st.progress(0)
    plot_raw_data(data)
    raw_data(data)
    show_macd(data)
    show_rsi(data)
    show_bb(data)
    forecast(data)
    recent_data(5, data)

if __name__ == "__main__":
    main()

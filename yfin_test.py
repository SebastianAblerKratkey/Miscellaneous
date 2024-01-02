import numpy as np
import numpy_financial as npf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.stats import norm
import os
import datetime
import statsmodels.api as sm

# this module is utilized to prevent the annotations in the plot from overlapping
from adjustText import adjust_text

# Get Yahoo Finance Data
import yfinance as yf

# Library for Website creation
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

def convert_date_index(df):
    # Convert the index to datetime
    df.index = pd.to_datetime(df.index)
    # Extract the month and year from the datetime
    df.index = df.index.strftime("%b %Y")
    return df

tickers = []
Name_dict = {}
for t in tickers:
    Name_dict[t] = yf.Ticker(t).info["longName"]


st.header("YFinance Download Test")

input_tickers = st.text_input("Enter the [Yahoo Finace](https://finance.yahoo.com) tickers of the assets you are interested in (seperated by comma). Make sure to select at least two.")
# [link](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)
tickers  = input_tickers.split(",")
tickers = [x.strip() for x in tickers]
price_df = yf.download(tickers, period='max', interval='1mo')["Adj Close"]
price_df.sort_index(ascending=False, inplace=True)   
montly_adjusted_closing_prices = convert_date_index(price_df).dropna()

download_sucess = False
if input_tickers:
    if len(montly_adjusted_closing_prices) < 1:
        st.error("(Some) assets could not be found.")
    elif len(tickers) == 1:
        st.error("Make sure to select at least two assets.")
    else:
        st.success("Assets added!")
        download_sucess = True

st.write("Disclaimer: This is not financial advice.")


if download_sucess:
    monthly_log_retruns = np.log(montly_adjusted_closing_prices / montly_adjusted_closing_prices.shift(-1))

    annualized_mean_retruns = monthly_log_retruns.mean() * 12
    annualized_std_retruns = monthly_log_retruns.std() * 12**0.5
    annualized_cov_retruns = monthly_log_retruns.cov() * 12
    corr_retruns = monthly_log_retruns.corr()

    summary = pd.DataFrame()

    summary["mean retrun"] = annualized_mean_retruns
    summary["standard deviation"] = annualized_std_retruns
    summary["weight"] = 1/len(summary)
    
    


    st.write("Monthly adjusted closing prices:")
    st.dataframe(montly_adjusted_closing_prices)
    st.write("Monthly log retruns:")
    st.dataframe(monthly_log_retruns)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import joblib

model = joblib.load('bitcoin_prediction.pkl')

st.title("Bitcoin Price Prediction Dashboard")

df = pd.read_csv('dataset/BTC-Daily.csv')

st.sidebar.header("Model Input")

open_close_diff = st.sidebar.slider("Open-Close Difference", float(df['open'].min() - df['close'].min()), float(df['open'].max() - df['close'].max()))
low_high_diff = st.sidebar.slider("Low-High Difference", float(df['low'].min() - df['high'].min()), float(df['low'].max() - df['high'].max()))
volume_btc = st.sidebar.slider("Volume BTC", float(df['Volume BTC'].min()), float(df['Volume BTC'].max()))
volume_usd = st.sidebar.slider("Volume USD", float(df['Volume USD'].min()), float(df['Volume USD'].max()))

user_input = np.array([[open_close_diff, low_high_diff, volume_btc]])
prediction = model.predict(user_input)[0]

st.subheader("Model Prediction:")
if prediction == 1:
    st.success("The model predicts an increase in Bitcoin price.")
else:
    st.error("The model predicts a decrease or no change in Bitcoin price.")

st.subheader("Data Visualization")

st.bar_chart(df['open'])

st.line_chart(df['close'])

st.scatter_chart(df[['Volume BTC', 'Volume USD']])

st.subheader("Original Dataset")
st.write(df)

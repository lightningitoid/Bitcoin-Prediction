# streamlit_app.py
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

# Load the saved model
model = joblib.load('bitcoin_prediction.pkl')

st.title("Bitcoin Price Prediction Dashboard")

# Load the dataset
df = pd.read_csv('dataset/BTC-Daily.csv')

# Sidebar for interactive input
st.sidebar.header("Model Input")

# Add sliders for user input (replace with your features)
open_close_diff = st.sidebar.slider("Open-Close Difference", float(df['open'].min() - df['close'].min()), float(df['open'].max() - df['close'].max()))
low_high_diff = st.sidebar.slider("Low-High Difference", float(df['low'].min() - df['high'].min()), float(df['low'].max() - df['high'].max()))
volume_btc = st.sidebar.slider("Volume BTC", float(df['Volume BTC'].min()), float(df['Volume BTC'].max()))
volume_usd = st.sidebar.slider("Volume USD", float(df['Volume USD'].min()), float(df['Volume USD'].max()))

# Make predictions
user_input = np.array([[open_close_diff, low_high_diff, volume_btc]])
prediction = model.predict(user_input)[0]


# Display prediction
st.subheader("Model Prediction:")
if prediction == 1:
    st.success("The model predicts an increase in Bitcoin price.")
else:
    st.error("The model predicts a decrease or no change in Bitcoin price.")

# Visualization (replace with your visualizations)
st.subheader("Data Visualization")

# Example: Plotting a histogram for 'Open' feature
st.bar_chart(df['open'])

# Example: Displaying a line chart for 'Close' feature
st.line_chart(df['close'])

# Example: Displaying a scatter plot for 'Volume BTC' and 'Volume USD' features
st.scatter_chart(df[['Volume BTC', 'Volume USD']])

# Example: Displaying the original dataframe (replace with your data)
st.subheader("Original Dataset")
st.write(df)

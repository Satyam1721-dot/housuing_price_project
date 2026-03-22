import os
import pickle
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression

MODEL_PATH = "model/model.pkl"

def train_model():
    data = pd.read_csv("data/housing.csv")
    
    X = data[['area', 'bedrooms', 'bathrooms']]
    y = data['price']
    
    model = LinearRegression()
    model.fit(X, y)
    
    os.makedirs("model", exist_ok=True)
    pickle.dump(model, open(MODEL_PATH, "wb"))
    
    return model

#  MAIN LOGIC
if os.path.exists(MODEL_PATH):
    model = pickle.load(open(MODEL_PATH, "rb"))
else:
    model = train_model()

# UI
st.title("🏠 House Price Predictor")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

if st.button("Predict"):
    prediction = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Price: {prediction[0]}")
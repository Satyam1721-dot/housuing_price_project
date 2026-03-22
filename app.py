import streamlit as st
import pickle

model = pickle.load(open("model/model.pkl", "rb"))

st.title("🏠 House Price Predictor")

area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")

if st.button("Predict"):
    result = model.predict([[area, bedrooms, bathrooms]])
    st.success(f"Price: ₹ {int(result[0])}")
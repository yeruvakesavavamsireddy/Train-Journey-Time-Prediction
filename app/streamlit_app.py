import os
import streamlit as st
import joblib

model_path = os.path.join(os.path.dirname(__file__), "..", "models", "linear_regression_model.pkl")

model = joblib.load(model_path)

st.title("Train Journey Time Prediction")

distance = st.number_input("Distance (km)")
stops = st.number_input("Stops")

if st.button("Predict"):
    prediction = model.predict([[distance, stops]])
    st.success(f"Predicted Journey Time: {prediction[0]:.2f} hours")
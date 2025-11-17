import streamlit as st
import numpy as np
import pickle
from tensorflow import keras

# Load model
model = keras.models.load_model("house_model.h5")

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏡 House Price Prediction App")

st.write("Enter the house features below to get the predicted price.")

# Example inputs — change these to match your dataset features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

if st.button("Predict Price"):
    try:
        # Convert to array
        arr = np.array([[feature1, feature2, feature3, feature4]])

        # Scale
        arr_scaled = scaler.transform(arr)

        # Predict
        pred = model.predict(arr_scaled)[0][0]

        st.success(f"Predicted House Price: {pred:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

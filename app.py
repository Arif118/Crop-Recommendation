import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Streamlit Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌾",
    layout="centered"
)

# --------------------------------------------------
# Title and Description
# --------------------------------------------------
st.title("🌾 Crop Recommendation System")
st.write(
    "Enter soil and weather conditions to get the most suitable crop recommendation using a Machine Learning model."
)

# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------
try:
    model = joblib.load("Crop_Recommendation_System.pkl")
except FileNotFoundError:
    st.error("Model file 'Crop_Recommendation_System.pkl' not found.")
    st.stop()

# --------------------------------------------------
# Crop Labels
# --------------------------------------------------
# These labels should match the original dataset order
labels = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut',
    'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans',
    'lentil', 'maize', 'mango', 'mothbeans', 'mungbean',
    'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate',
    'rice', 'watermelon'
]

# Create Label Encoder
le = LabelEncoder()
le.fit(labels)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Input Parameters")
st.sidebar.write("Adjust the values below:")

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
N = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
K = st.sidebar.number_input("Potassium (K)", min_value=0.0, value=43.0)

temperature = st.sidebar.number_input("Temperature (°C)", value=20.8)
humidity = st.sidebar.number_input("Humidity (%)", value=82.0)
ph = st.sidebar.number_input("pH Value", value=6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=202.9)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------
if st.button("🌱 Recommend Crop"):

    # Prepare input data
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # Make prediction
    prediction = model.predict(input_data)

    # Decode prediction label
    crop_name = le.inverse_transform(prediction)[0]

    # Display Result
    st.success(f"Recommended Crop: {crop_name.upper()}")

    # Extra Information
    st.subheader("📊 Entered Values")
    st.write(f"Nitrogen (N): {N}")
    st.write(f"Phosphorus (P): {P}")
    st.write(f"Potassium (K): {K}")
    st.write(f"Temperature: {temperature} °C")
    st.write(f"Humidity: {humidity} %")
    st.write(f"pH Value: {ph}")
    st.write(f"Rainfall: {rainfall} mm")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Built with Streamlit and Random Forest Classifier 🚀")

import streamlit as st
import joblib
import numpy as np

# Load the trained scikit-learn model
model = joblib.load('weather_model.pkl')

# CSS for background and styling
page_bg = """
<style>
body {
  background-image: url("https://img.freepik.com/free-photo/wall-wallpaper-concrete-colored-painted-textured-concept_53876-31799.jpg?w=900&t=st=1693223602~exp=1693224202~hmac=e3ac3b9301407c739173c5648953b371a64459211832a8f3000f9f8964e98d52");
  background-size: cover;
}
h1 {
  color: #ff6f61;
  text-align: center;
  font-family: 'Arial', sans-serif;
}
.stSlider {
  background-color: rgba(255,255,255,0.7);
  border-radius: 10px;
  padding: 10px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App title and description
st.title('🌤️ Weather Forecasting App 🌧️')
st.write("This app predicts the weather based on input parameters. Provide the necessary data below:")

# Input fields for user to provide data
st.subheader("Input Weather Conditions")
temperature = st.slider("🌡️ Temperature (Celsius)", -10, 40, 25)
humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
wind_speed = st.slider("🌬️ Wind Speed (km/h)", 0, 50, 10)
visibility = st.slider("👀 Visibility (km)", 0, 20, 10)

# Prepare the input features as a NumPy array
input_features = np.array([[temperature, humidity, wind_speed, visibility]])

# Predict the weather using the trained model
prediction = model.predict(input_features)

# The prediction is assumed to be a string like 'Sunny', 'Cloudy', etc.
predicted_weather = prediction[0]  # Use the prediction directly

# Display the predicted weather visually
st.subheader("Predicted Weather:")
st.write(f"The predicted weather is: **{predicted_weather}**")

# Note to the user
st.write("Please note that this is a simplified demo app and the predictions may not be accurate.")

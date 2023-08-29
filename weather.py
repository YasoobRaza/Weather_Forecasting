import streamlit as st
import joblib
import numpy as np
from PIL import Image

# Load the trained scikit-learn model
model = joblib.load('weather_model.pkl')

# Streamlit app title and description with background
page_bg = """
<style>
body {
  background-image: url("https://img.freepik.com/free-photo/wall-wallpaper-concrete-colored-painted-textured-concept_53876-31799.jpg?w=900&t=st=1693223602~exp=1693224202~hmac=e3ac3b9301407c739173c5648953b371a64459211832a8f3000f9f8964e98d52");
  background-size: cover;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title('Weather Forecasting App')
st.write("This app predicts the weather based on input parameters.")

# Input fields for user to provide data
temperature = st.slider("Temperature (Celsius)", -10, 40, 25)
humidity = st.slider("Humidity (%)", 0, 100, 50)
wind_speed = st.slider("Wind Speed (km/h)", 0, 50, 10)
visibility = st.slider("Visibility (km)", 0, 20, 10)

# Prepare the input features as a NumPy array
input_features = np.array([[temperature, humidity, wind_speed, visibility]])

# Predict the weather using the trained model
prediction = model.predict(input_features)

# Define weather classes and corresponding images
weather_classes = ['Sunny', 'Cloudy', 'Rainy']
weather_images = ['sunny.jpg', 'cloudy.jpg', 'rainy.jpg']

# Display the predicted weather image
# weather_index = prediction[0]
# weather_image = Image.open(weather_images[weather_index])

st.subheader("Predicted Weather:")
# st.image(weather_image, caption=weather_classes[weather_index], use_column_width=True)

# Note to the user
st.write("Please note that this is a simplified demo app and the predictions may not be accurate.")

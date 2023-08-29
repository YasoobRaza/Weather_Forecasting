import streamlit as st
import joblib
import numpy as np

# Load the trained scikit-learn model
model = joblib.load('weather_model.pkl')

# Streamlit app title and description
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

# Define weather classes
weather_classes = ['Sunny', 'Cloudy', 'Rainy']

# Display the predicted weather
st.subheader("Predicted Weather:")
st.write(f"The predicted weather is: {prediction}")

# Note to the user
st.write("Please note that this is a simplified demo app and the predictions may not be accurate.")

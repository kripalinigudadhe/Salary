
import streamlit as st
import pickle
import numpy as np

# --- Load the Model ---
model_filename = 'random_forest_regressor_model.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    st.success(f"Model '{model_filename}' loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file '{model_filename}' not found. Make sure it's in the same directory as the app.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Streamlit App UI ---
st.title('Salary Prediction App')
st.write('Enter the years of experience to predict the salary.')

# Input field for YearsExperience
years_experience = st.number_input('Years of Experience', min_value=0.0, max_value=50.0, value=5.0, step=0.1)

# Make prediction when button is clicked
if st.button('Predict Salary'):
    # The model expects a 2D array, even for a single feature input
    input_data = np.array([[years_experience]])
    
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
--- 
This app uses a Random Forest Regressor model to predict salary based on years of experience.
""")

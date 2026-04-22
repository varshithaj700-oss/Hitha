import streamlit as st
import pickle
import numpy as np

# Load the trained model and feature names
try:
    with open('student_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('features.pkl', 'rb') as features_file:
        features = pickle.load(features_file)
except FileNotFoundError as e:
    st.error(f"Error loading model or features: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error loading files: {e}")
    st.stop()

# Title of the app
st.title("Student Performance Predictor")

# Input fields
st.header("Enter Student Details")

hours_studied = st.number_input("Hours Studied", min_value=0.0, step=0.1, format="%.1f")
previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
extracurricular_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, step=0.1, format="%.1f")
sample_question_papers = st.number_input("Sample Question Papers Practiced", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    try:
        # Convert Extracurricular Activities to numerical value
        extracurricular_num = 1 if extracurricular_activities == "Yes" else 0

        # Arrange inputs in the same order as features.pkl
        # Assuming features is a list of feature names in order
        input_data = [
            hours_studied,
            previous_scores,
            extracurricular_num,
            sleep_hours,
            sample_question_papers
        ]

        # Convert to 2D array for prediction
        input_array = np.array([input_data])

        # Make prediction
        prediction = model.predict(input_array)

        # Display result
        st.success(f"Predicted Performance: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
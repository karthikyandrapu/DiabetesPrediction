import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('random_forest_model.pkl')

# Define a function for making predictions
def predict_diabetes(input_data):
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    # Streamlit app title and description
    st.title('Diabetes Prediction')
    st.write('Enter the details below to predict the likelihood of having diabetes.')

    # Input fields for user data
    Pregnancies = st.slider("Number of Pregnancies", 0, 20, value=0)
    Glucose = st.slider("Glucose Level", 0, 200, value=120)
    BloodPressure = st.slider("Blood Pressure", 0, 200, value=80)
    SkinThickness = st.slider("Skin Thickness", 0, 100, value=20)
    Insulin = st.slider("Insulin Level", 0, 900, value=100)
    BMI = st.slider("BMI", 0.0, 70.0, value=25.0)
    DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 3.0, value=0.5)
    Age = st.slider("Age", 0, 100, value=30)

    # Button to make prediction
    if st.button('Predict'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = predict_diabetes(input_data)
        
        if prediction == 1:
            st.write('The Person has Diabetes')
        else:
            st.write('The Person does not have Diabetes')

if __name__ == '__main__':
    main()